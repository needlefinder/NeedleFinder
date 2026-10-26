#!/usr/bin/env python3
"""
Train & Export Production NeedleSliceMPNN + NeedleBundleGraphTransformer on all 54 GYN Cases:
1. Vectorized 3D Spatio-Temporal Graph Builder (~45 ms/case instead of 1.8 s/case).
2. 4-Layer Triplet Collinearity Message-Passing Graph Network (NeedleSliceMPNN) for 2D-to-3D
   disjoint path trajectory reconstruction and sub-voxel centerline refinement.
3. Trajectory-Level GATv2 + Set Transformer (NeedleBundleGraphTransformer) trained jointly with:
   - y_rec (any GT match <= 3.5 mm, for high-recall mode)
   - y_1to1 (unique 1-to-1 Hungarian GT winner <= 3.5 mm, for high-precision/F1 mode)
   - y_long_1to1 (unique 1-to-1 Hungarian CY long needle winner <= 3.5 mm)
4. Exports production weights to `needle_graphnet.npz` and `needle_graph_transformer.onnx`.
"""

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path

import glob
import math
import os
import pickle
import sys
import time
import types
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

for mod in ["vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"]:
    sys.modules[mod] = types.ModuleType(mod)
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder"))
from NeedleFinderLib.NeuralNeedleNet import extract_nn_labelmap_tubes

CACHE_DIR = resolve_path("/tmp/needle_unet3d_cache")
OOF_DIR = resolve_path("/home/guillaume/github/needlefinder/nn_models_eval")
META_PATH = resolve_path("/tmp/cases_meta_54.pkl")
BASE_TRAJS_PATH = resolve_path("/tmp/base_trajs_54cases.pkl")


def get_gts(g_list):
    out = []
    for g in g_list:
        pts = g["pts"] if isinstance(g, dict) else (g[1] if isinstance(g, tuple) else np.asarray(g))
        pts = np.asarray(pts, dtype=np.float64)
        pts = pts[np.argsort(pts[:, 2])]
        if len(pts) >= 2:
            out.append(pts)
    return out


def interp_gt_xy(gt_pts, z_val):
    if z_val < gt_pts[0, 2] - 2.5 or z_val > gt_pts[-1, 2] + 2.5:
        return None
    x = float(np.interp(z_val, gt_pts[:, 2], gt_pts[:, 0]))
    y = float(np.interp(z_val, gt_pts[:, 2], gt_pts[:, 1]))
    return x, y


def traj_gt_dist(p_ras, g_ras):
    z0 = max(float(p_ras[0, 2]), float(g_ras[0, 2]))
    z1 = min(float(p_ras[-1, 2]), float(g_ras[-1, 2]))
    if z1 >= z0 + 2.0:
        zs = np.linspace(z0, z1, 12)
    else:
        zs = np.linspace(float(g_ras[0, 2]), float(g_ras[-1, 2]), 12)
    px = np.interp(zs, p_ras[:, 2], p_ras[:, 0])
    py = np.interp(zs, p_ras[:, 2], p_ras[:, 1])
    gx = np.interp(zs, g_ras[:, 2], g_ras[:, 0])
    gy = np.interp(zs, g_ras[:, 2], g_ras[:, 1])
    return float(np.mean(np.hypot(px - gx, py - gy)))


def build_slice_graph_fast(X, p_mask, p_peak, vx_map, vy_map, p_obt, k0, i0, j0, sp_ras, ijk_to_ras, gts_all=None, gts_long=None):
    """
    Fully vectorized 3D Spatio-Temporal Graph construction (~40 ms per case on CPU).
    """
    rot = ijk_to_ras[:3, :3]
    trans = ijk_to_ras[:3, 3]
    sp_i, sp_j, sp_k = [float(x) for x in sp_ras]

    K_sub, H, W = p_mask.shape
    score_vol = 0.48 * p_peak + 0.30 * p_mask + 0.14 * p_obt + 0.12 * np.clip(X[1], 0.0, 1.2)
    nms_r = max(1, int(round(1.35 / sp_i)))
    loc_max = score_vol == maximum_filter(score_vol, size=(1, 2 * nms_r + 1, 2 * nms_r + 1))

    center_ras = rot @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0)]) + trans
    z_base_ras = float(center_ras[2])
    yy_w, xx_w = np.mgrid[-1:2, -1:2]

    pm_max3_vol = maximum_filter(p_mask, size=(1, 3, 3))
    pp_max3_vol = maximum_filter(p_peak, size=(1, 3, 3))

    nodes_kr, nodes_ic, nodes_jc, nodes_sub_i, nodes_sub_j = [], [], [], [], []
    for kr in range(K_sub):
        mask_k = loc_max[kr] & (score_vol[kr] >= 0.16) & ((p_mask[kr] >= 0.18) | (p_peak[kr] >= 0.16) | (p_obt[kr] >= 0.22))
        mask_k[0, :] = False; mask_k[-1, :] = False; mask_k[:, 0] = False; mask_k[:, -1] = False
        jj, ii = np.where(mask_k)
        if len(ii) == 0:
            continue
        if len(ii) > 38:
            order = np.argsort(-score_vol[kr, jj, ii])[:38]
            jj, ii = jj[order], ii[order]
        for jc, ic in zip(jj, ii):
            w_patch = np.maximum(0.0, score_vol[kr, jc - 1:jc + 2, ic - 1:ic + 2])
            w_sum = float(np.sum(w_patch))
            if w_sum > 1e-6:
                si = float(ic) + float(np.sum(xx_w * w_patch)) / w_sum
                sj = float(jc) + float(np.sum(yy_w * w_patch)) / w_sum
            else:
                si, sj = float(ic), float(jc)
            nodes_kr.append(kr)
            nodes_ic.append(ic)
            nodes_jc.append(jc)
            nodes_sub_i.append(si)
            nodes_sub_j.append(sj)

    N = len(nodes_kr)
    if N == 0:
        return None

    nodes_kr = np.asarray(nodes_kr, dtype=np.int32)
    nodes_ic = np.asarray(nodes_ic, dtype=np.int32)
    nodes_jc = np.asarray(nodes_jc, dtype=np.int32)
    nodes_sub_i = np.asarray(nodes_sub_i, dtype=np.float32)
    nodes_sub_j = np.asarray(nodes_sub_j, dtype=np.float32)

    ijk_pts = np.column_stack([nodes_sub_i + i0, nodes_sub_j + j0, nodes_kr + k0]).astype(np.float64)
    nodes_ras = (ijk_pts @ rot.T + trans).astype(np.float32)
    nodes_ij = ijk_pts[:, :2].astype(np.float32)

    pm_v = p_mask[nodes_kr, nodes_jc, nodes_ic]
    pp_v = p_peak[nodes_kr, nodes_jc, nodes_ic]
    po_v = p_obt[nodes_kr, nodes_jc, nodes_ic]
    vx_v = vx_map[nodes_kr, nodes_jc, nodes_ic]
    vy_v = vy_map[nodes_kr, nodes_jc, nodes_ic]
    pm_max3 = pm_max3_vol[nodes_kr, nodes_jc, nodes_ic]
    pp_max3 = pp_max3_vol[nodes_kr, nodes_jc, nodes_ic]
    pm_mean3 = 0.5 * (pm_v + pm_max3)

    nodes_feat = np.column_stack([
        (nodes_ras[:, 0] - center_ras[0]) / 45.0,
        (nodes_ras[:, 1] - center_ras[1]) / 45.0,
        (nodes_ras[:, 2] - z_base_ras) / 80.0,
        pm_v, pp_v, po_v, vx_v, vy_v,
        pm_max3, pp_max3, pm_mean3,
        X[0, nodes_kr, nodes_jc, nodes_ic],
        X[1, nodes_kr, nodes_jc, nodes_ic],
        X[2, nodes_kr, nodes_jc, nodes_ic],
        X[3, nodes_kr, nodes_jc, nodes_ic],
        X[5, nodes_kr, nodes_jc, nodes_ic],
    ]).astype(np.float32)

    # Vectorized inter-slice directed edge construction
    max_dk = max(4, int(round(24.0 / sp_k)))
    dk_mat = nodes_kr[None, :] - nodes_kr[:, None]  # [N, N], dk = kr_v - kr_u
    cand_u, cand_v = np.where((dk_mat >= 1) & (dk_mat <= max_dk))

    if len(cand_u) > 0:
        ru = nodes_ras[cand_u]
        rv = nodes_ras[cand_v]
        dz = rv[:, 2] - ru[:, 2]
        valid_z = dz > 0.2
        cand_u, cand_v, ru, rv, dz = cand_u[valid_z], cand_v[valid_z], ru[valid_z], rv[valid_z], dz[valid_z]

        sx = (rv[:, 0] - ru[:, 0]) / dz
        sy = (rv[:, 1] - ru[:, 1]) / dz
        s_norm = np.hypot(sx, sy)
        vx_u, vy_u = vx_v[cand_u], vy_v[cand_u]
        vx_w, vy_w = vx_v[cand_v], vy_v[cand_v]
        pred_x = ru[:, 0] + 0.5 * (vx_u + vx_w) * dz
        pred_y = ru[:, 1] + 0.5 * (vy_u + vy_w) * dz
        err_tan = np.hypot(rv[:, 0] - pred_x, rv[:, 1] - pred_y)
        raw_dxy = np.hypot(rv[:, 0] - ru[:, 0], rv[:, 1] - ru[:, 1])

        keep = (s_norm <= 0.56) & ((err_tan <= 2.8 + 0.14 * dz) | (raw_dxy <= 3.5 + 0.22 * dz))
        edge_src = cand_u[keep]
        edge_dst = cand_v[keep]
        sx, sy, s_norm, dz = sx[keep], sy[keep], s_norm[keep], dz[keep]
        err_tan = err_tan[keep]
        vx_u, vy_u = vx_u[keep], vy_u[keep]
        vx_w, vy_w = vx_w[keep], vy_w[keep]

        err_u = np.hypot(sx - vx_u, sy - vy_u)
        err_v = np.hypot(sx - vx_w, sy - vy_w)
        dv_uv = np.hypot(vx_u - vx_w, vy_u - vy_w)
        dk_e = (nodes_kr[edge_dst] - nodes_kr[edge_src]).astype(np.float32)

        km = np.clip(np.round(0.5 * (nodes_kr[edge_src] + nodes_kr[edge_dst])).astype(np.int32), 0, K_sub - 1)
        im = np.clip(np.round(0.5 * (nodes_sub_i[edge_src] + nodes_sub_i[edge_dst])).astype(np.int32), 0, W - 1)
        jm = np.clip(np.round(0.5 * (nodes_sub_j[edge_src] + nodes_sub_j[edge_dst])).astype(np.int32), 0, H - 1)
        pm_mid = p_mask[km, jm, im]
        pp_mid = p_peak[km, jm, im]
        min_pp = np.minimum(pp_v[edge_src], pp_v[edge_dst])

        edge_feat = np.column_stack([
            sx, sy, s_norm, dz / 25.0, dk_e / float(max_dk),
            err_tan / 5.0, err_u, err_v, dv_uv, pm_mid, pp_mid, min_pp
        ]).astype(np.float32)
    else:
        edge_src = np.zeros(0, dtype=np.int64)
        edge_dst = np.zeros(0, dtype=np.int64)
        edge_feat = np.zeros((0, 12), dtype=np.float32)

    # Vectorized in-plane spatial neighbor edges (same slice, dist_xy <= 22 mm)
    sp_u, sp_v = np.where((dk_mat == 0) & (np.arange(N)[:, None] != np.arange(N)[None, :]))
    if len(sp_u) > 0:
        dx_sp = nodes_ras[sp_v, 0] - nodes_ras[sp_u, 0]
        dy_sp = nodes_ras[sp_v, 1] - nodes_ras[sp_u, 1]
        dist_sp = np.hypot(dx_sp, dy_sp)
        keep_sp = dist_sp <= 22.0
        spatial_src = sp_u[keep_sp]
        spatial_dst = sp_v[keep_sp]
        spatial_feat = np.column_stack([
            dx_sp[keep_sp] / 22.0,
            dy_sp[keep_sp] / 22.0,
            dist_sp[keep_sp] / 22.0,
            vx_v[spatial_dst] - vx_v[spatial_src],
            vy_v[spatial_dst] - vy_v[spatial_src],
        ]).astype(np.float32)
    else:
        spatial_src = np.zeros(0, dtype=np.int64)
        spatial_dst = np.zeros(0, dtype=np.int64)
        spatial_feat = np.zeros((0, 5), dtype=np.float32)

    out = {
        "nodes_ras": nodes_ras,
        "nodes_kr": nodes_kr,
        "nodes_ij": nodes_ij,
        "nodes_feat": torch.tensor(nodes_feat, dtype=torch.float32),
        "edge_src": torch.tensor(edge_src, dtype=torch.long),
        "edge_dst": torch.tensor(edge_dst, dtype=torch.long),
        "edge_feat": torch.tensor(edge_feat, dtype=torch.float32),
        "spatial_src": torch.tensor(spatial_src, dtype=torch.long),
        "spatial_dst": torch.tensor(spatial_dst, dtype=torch.long),
        "spatial_feat": torch.tensor(spatial_feat, dtype=torch.float32),
        "k0": k0,
        "sp_ras": (sp_i, sp_j, sp_k),
        "ijk_to_ras": ijk_to_ras,
        "z_base_ras": z_base_ras,
        "center_ras": center_ras,
    }

    if gts_all is not None:
        nodes_gt_id = np.full(N, -1, dtype=np.int32)
        nodes_gt_long = np.full(N, -1, dtype=np.int32)
        nodes_offset = np.zeros((N, 2), dtype=np.float32)
        for u in range(N):
            zu = float(nodes_ras[u, 2])
            xu, yu = float(nodes_ras[u, 0]), float(nodes_ras[u, 1])
            best_d = 1.95
            for gid, g_pts in enumerate(gts_all):
                g_xy = interp_gt_xy(g_pts, zu)
                if g_xy is not None:
                    d = math.hypot(g_xy[0] - xu, g_xy[1] - yu)
                    if d < best_d:
                        best_d = d
                        nodes_gt_id[u] = gid
                        nodes_offset[u, 0] = g_xy[0] - xu
                        nodes_offset[u, 1] = g_xy[1] - yu
            if gts_long is not None:
                best_ld = 1.95
                for lid, l_pts in enumerate(gts_long):
                    l_xy = interp_gt_xy(l_pts, zu)
                    if l_xy is not None:
                        d = math.hypot(l_xy[0] - xu, l_xy[1] - yu)
                        if d < best_ld:
                            best_ld = d
                            nodes_gt_long[u] = lid
        edge_y = ((nodes_gt_id[edge_src] >= 0) & (nodes_gt_id[edge_src] == nodes_gt_id[edge_dst])).astype(np.float32) if len(edge_src) > 0 else np.zeros(0, dtype=np.float32)
        out["nodes_y_all"] = torch.tensor((nodes_gt_id >= 0).astype(np.float32))
        out["nodes_y_long"] = torch.tensor((nodes_gt_long >= 0).astype(np.float32))
        out["nodes_offset"] = torch.tensor(nodes_offset, dtype=torch.float32)
        out["edge_y"] = torch.tensor(edge_y, dtype=torch.float32)
        out["gts_all"] = gts_all
        out["gts_long"] = gts_long

    return out


from train_graphnet3d import (
    NeedleSliceMPNN,
    decode_mpnn_trajectories,
    evaluate_needle_set,
)


class DenseGraphTransformer(nn.Module):
    """
    ONNX-friendly Trajectory-Level Graph Attention + Set Transformer.
    Takes:
      - x: [M, 18] trajectory node features
      - adj_feat: [M, M, 6] pairwise trajectory edge features
      - adj_mask: [M, M] boolean/float mask (1.0 where distance <= 35 mm, 0.0 otherwise)
    Uses pure dense matrix operations (zero scatter_reduce ops!) so it exports cleanly to
    standard ONNX and runs in < 2 ms in ONNX Runtime or NumPy!
    Outputs 3 logits per trajectory:
      - out[:, 0]: logit_rec (any GT match <= 3.5 mm)
      - out[:, 1]: logit_1to1 (unique 1-to-1 Hungarian GT winner <= 3.5 mm)
      - out[:, 2]: logit_long_1to1 (unique 1-to-1 Hungarian CY long needle winner)
    """
    def __init__(self, in_dim=18, edge_dim=6, d_model=64):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, d_model),
            nn.SiLU(),
        )
        # Layer 1: Dense Edge-Biased Graph Attention
        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.v1 = nn.Linear(d_model, d_model)
        self.e_att1 = nn.Linear(d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)

        # Layer 2: Dense Edge-Biased Graph Attention
        self.q2 = nn.Linear(d_model, d_model)
        self.k2 = nn.Linear(d_model, d_model)
        self.v2 = nn.Linear(d_model, d_model)
        self.e_att2 = nn.Linear(d_model, 1)
        self.norm2 = nn.LayerNorm(d_model)

        # Layer 3 & 4: Global Bundle Self-Attention + FFN
        self.q3 = nn.Linear(d_model, d_model)
        self.k3 = nn.Linear(d_model, d_model)
        self.v3 = nn.Linear(d_model, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm4 = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),  # [logit_rec, logit_1to1, logit_long_1to1]
        )

    def forward(self, x, adj_feat, adj_mask):
        # x: [M, 18], adj_feat: [M, M, 6], adj_mask: [M, M]
        h = self.node_enc(x)  # [M, D]
        ef = self.edge_enc(adj_feat)  # [M, M, D]
        scale = 1.0 / math.sqrt(float(h.shape[-1]))

        # GAT Layer 1
        scores1 = torch.matmul(self.q1(h), self.k1(h).transpose(0, 1)) * scale + self.e_att1(ef).squeeze(-1)
        scores1 = scores1.masked_fill(adj_mask <= 0.0, -1e4)
        attn1 = torch.softmax(scores1, dim=-1) * adj_mask
        attn1 = attn1 / attn1.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        msg1 = torch.matmul(attn1, self.v1(h)) + (attn1.unsqueeze(-1) * ef).sum(dim=1)
        h = self.norm1(h + F.silu(msg1))

        # GAT Layer 2
        scores2 = torch.matmul(self.q2(h), self.k2(h).transpose(0, 1)) * scale + self.e_att2(ef).squeeze(-1)
        scores2 = scores2.masked_fill(adj_mask <= 0.0, -1e4)
        attn2 = torch.softmax(scores2, dim=-1) * adj_mask
        attn2 = attn2 / attn2.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        msg2 = torch.matmul(attn2, self.v2(h)) + (attn2.unsqueeze(-1) * ef).sum(dim=1)
        h = self.norm2(h + F.silu(msg2))

        # Global Set-Transformer Self-Attention Layer
        scores3 = torch.matmul(self.q3(h), self.k3(h).transpose(0, 1)) * scale
        attn3 = torch.softmax(scores3, dim=-1)
        h = self.norm3(h + torch.matmul(attn3, self.v3(h)))
        h = self.norm4(h + self.ffn(h))

        return self.head(h)


def build_bundle_graph_item(pool, g, oof_npz, gts_all=None, gts_long=None):
    pm_arr = oof_npz["p_mask"].astype(np.float32)
    pp_arr = oof_npz["p_peak"].astype(np.float32)
    po_arr = oof_npz["p_obt"].astype(np.float32)
    ras_to_ijk = np.linalg.inv(g["ijk_to_ras"])
    k0_c, i0_c, j0_c = int(oof_npz["k0"]), int(oof_npz["i0"]), int(oof_npz["j0"])
    K_s, H_s, W_s = pm_arr.shape

    M = len(pool)
    bases = np.array([p["base_xy"] for p in pool]) if M else np.zeros((0, 2))
    slopes = np.array([p["slope"] for p in pool]) if M else np.zeros((0, 2))
    med_base = np.median(bases, axis=0) if M else np.zeros(2)
    med_slope = np.median(slopes, axis=0) if M else np.zeros(2)

    x_feats = []
    for p in pool:
        ras = p["ras"]
        zs = np.linspace(ras[0, 2], ras[-1, 2], 15)
        xs = np.interp(zs, ras[:, 2], ras[:, 0])
        ys = np.interp(zs, ras[:, 2], ras[:, 1])
        ijks = (ras_to_ijk @ np.column_stack([xs, ys, zs, np.ones_like(zs)]).T).T
        pm_l, pp_l, po_l = [], [], []
        for row in ijks:
            ic, jc, kc = int(round(row[0] - i0_c)), int(round(row[1] - j0_c)), int(round(row[2] - k0_c))
            if 1 <= ic < W_s - 1 and 1 <= jc < H_s - 1 and 0 <= kc < K_s:
                pm_l.append(float(np.max(pm_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
                pp_l.append(float(np.max(pp_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
                po_l.append(float(np.max(po_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
        mean_pm = float(np.mean(pm_l)) if pm_l else 0.0
        p75_pm = float(np.percentile(pm_l, 75)) if pm_l else 0.0
        mean_pp = float(np.mean(pp_l)) if pp_l else 0.0
        mean_po = float(np.mean(po_l)) if po_l else 0.0

        d_base = float(np.hypot(p["base_xy"][0] - med_base[0], p["base_xy"][1] - med_base[1])) / 40.0
        d_slope = float(np.hypot(p["slope"][0] - med_slope[0], p["slope"][1] - med_slope[1])) / 0.25
        z_low = float(ras[0, 2] - g["z_base_ras"]) / 50.0
        z_high = float(ras[-1, 2] - g["z_base_ras"]) / 100.0

        x_feats.append([
            p["conf_heur"],
            p["from_mpnn"],
            p["mpnn_pa"],
            p["mpnn_pl"],
            p["mpnn_pe"],
            mean_pm,
            p75_pm,
            mean_pp,
            mean_po,
            max(mean_pm, mean_po),
            p["z_span"] / 60.0,
            p["rmse"],
            d_base,
            d_slope,
            z_low,
            z_high,
            float(p["slope"][0]),
            float(p["slope"][1]),
        ])

    adj_feat = np.zeros((M, M, 6), dtype=np.float32)
    adj_mask = np.zeros((M, M), dtype=np.float32)
    for u in range(M):
        for v in range(M):
            if u == v:
                continue
            db = float(np.hypot(pool[u]["base_xy"][0] - pool[v]["base_xy"][0], pool[u]["base_xy"][1] - pool[v]["base_xy"][1]))
            if db <= 35.0:
                ds = float(np.hypot(pool[u]["slope"][0] - pool[v]["slope"][0], pool[u]["slope"][1] - pool[v]["slope"][1]))
                sh_d = traj_gt_dist(pool[u]["ras"], pool[v]["ras"])
                adj_mask[u, v] = 1.0
                adj_feat[u, v] = [
                    db / 35.0,
                    sh_d / 35.0,
                    ds / 0.30,
                    1.0 if sh_d < 2.5 else 0.0,
                    float(x_feats[v][5] - x_feats[u][5]),
                    float(x_feats[v][0] - x_feats[u][0]),
                ]

    item = {
        "pool": pool,
        "x": torch.tensor(x_feats, dtype=torch.float32),
        "adj_feat": torch.tensor(adj_feat, dtype=torch.float32),
        "adj_mask": torch.tensor(adj_mask, dtype=torch.float32),
    }

    if gts_all is not None:
        y_rec = np.zeros(M, dtype=np.float32)
        y_1to1 = np.zeros(M, dtype=np.float32)
        y_long_1to1 = np.zeros(M, dtype=np.float32)
        if len(gts_all) > 0 and M > 0:
            cost_a = np.zeros((len(gts_all), M))
            for gi, gt in enumerate(gts_all):
                for pj, p in enumerate(pool):
                    cost_a[gi, pj] = traj_gt_dist(p["ras"], gt)
            y_rec = (np.min(cost_a, axis=0) <= 3.3).astype(np.float32)
            ri, ci = linear_sum_assignment(cost_a)
            for r, c in zip(ri, ci):
                if cost_a[r, c] <= 3.5:
                    y_1to1[c] = 1.0
        if gts_long is not None and len(gts_long) > 0 and M > 0:
            cost_l = np.zeros((len(gts_long), M))
            for gi, gt in enumerate(gts_long):
                for pj, p in enumerate(pool):
                    cost_l[gi, pj] = traj_gt_dist(p["ras"], gt)
            ri, ci = linear_sum_assignment(cost_l)
            for r, c in zip(ri, ci):
                if cost_l[r, c] <= 3.5:
                    y_long_1to1[c] = 1.0
        item["y_rec"] = torch.tensor(y_rec, dtype=torch.float32)
        item["y_1to1"] = torch.tensor(y_1to1, dtype=torch.float32)
        item["y_long_1to1"] = torch.tensor(y_long_1to1, dtype=torch.float32)
        item["gts_all"] = gts_all
        item["gts_long"] = gts_long

    return item


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Prod Export] Using device: {device} ({torch.cuda.get_device_name(device)})")

    t0 = time.perf_counter()
    meta_list = pickle.load(open(META_PATH, "rb"))
    base_trajs_data = pickle.load(open(BASE_TRAJS_PATH, "rb"))
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]

    graphs = []
    for i in range(len(meta_list)):
        d_pt = torch.load(f"{CACHE_DIR}/case_{i:02d}.pt", weights_only=False)
        oof_npz = np.load(f"{OOF_DIR}/oof_case_{i:02d}.npz")
        X = d_pt["X"].numpy()
        g = build_slice_graph_fast(
            X,
            oof_npz["p_mask"].astype(np.float32),
            oof_npz["p_peak"].astype(np.float32),
            oof_npz["vx"].astype(np.float32),
            oof_npz["vy"].astype(np.float32),
            oof_npz["p_obt"].astype(np.float32),
            int(d_pt["k0"]), int(d_pt["i0"]), int(d_pt["j0"]),
            meta_list[i]["sp_ras"], np.asarray(meta_list[i]["ijk_to_ras"], dtype=np.float64),
            gts_all=gts_all_list[i], gts_long=gts_long_list[i],
        )
        graphs.append(g)
    dt_build = time.perf_counter() - t0
    print(f"[Vectorized Graph Build] 54 cases built in {dt_build:.2f}s ({dt_build/54*1000:.1f} ms/case)!")

    # Step 1: 5-Fold CV + Full-Dataset Training of NeedleSliceMPNN
    oof_mpnn_trajs = [None] * len(graphs)
    for fold in range(5):
        train_gs = [g for i, g in enumerate(graphs) if i % 5 != fold]
        model = NeedleSliceMPNN(in_node_dim=16, in_edge_dim=12, in_sp_dim=5, hidden_dim=64, num_layers=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-4)
        for ep in range(40):
            model.train()
            for bi in np.random.permutation(len(train_gs)):
                g = train_gs[bi]
                opt.zero_grad(set_to_none=True)
                no, el = model(g, device)
                y_na, y_nl, y_off, y_e = g["nodes_y_all"].to(device), g["nodes_y_long"].to(device), g["nodes_offset"].to(device), g["edge_y"].to(device)
                pw_n = torch.tensor([3.0], device=device)
                fg_n = (y_na > 0.5).float().unsqueeze(1)
                loss = F.binary_cross_entropy_with_logits(no[:, 0], y_na, pos_weight=pw_n) + \
                       0.5 * F.binary_cross_entropy_with_logits(no[:, 1], y_nl, pos_weight=pw_n) + \
                       0.5 * (F.smooth_l1_loss(no[:, 2:4], y_off, reduction="none") * fg_n).sum() / fg_n.sum().clamp(min=1.0)
                if el.numel() > 0:
                    loss = loss + 2.0 * F.binary_cross_entropy_with_logits(el, y_e, pos_weight=torch.tensor([6.0], device=device))
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            for i, g in enumerate(graphs):
                if i % 5 == fold:
                    no, el = model(g, device)
                    oof_mpnn_trajs[i] = decode_mpnn_trajectories(g, no, el)

    # Train full 54-case production NeedleSliceMPNN
    prod_mpnn = NeedleSliceMPNN(in_node_dim=16, in_edge_dim=12, in_sp_dim=5, hidden_dim=64, num_layers=4).to(device)
    opt_m = torch.optim.AdamW(prod_mpnn.parameters(), lr=2.0e-3, weight_decay=1e-4)
    sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=45, eta_min=1e-4)
    for ep in range(45):
        prod_mpnn.train()
        for bi in np.random.permutation(len(graphs)):
            g = graphs[bi]
            opt_m.zero_grad(set_to_none=True)
            no, el = prod_mpnn(g, device)
            y_na, y_nl, y_off, y_e = g["nodes_y_all"].to(device), g["nodes_y_long"].to(device), g["nodes_offset"].to(device), g["edge_y"].to(device)
            pw_n = torch.tensor([3.0], device=device)
            fg_n = (y_na > 0.5).float().unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(no[:, 0], y_na, pos_weight=pw_n) + \
                   0.5 * F.binary_cross_entropy_with_logits(no[:, 1], y_nl, pos_weight=pw_n) + \
                   0.5 * (F.smooth_l1_loss(no[:, 2:4], y_off, reduction="none") * fg_n).sum() / fg_n.sum().clamp(min=1.0)
            if el.numel() > 0:
                loss = loss + 2.0 * F.binary_cross_entropy_with_logits(el, y_e, pos_weight=torch.tensor([6.0], device=device))
            loss.backward()
            opt_m.step()
        sched_m.step()
    torch.save(prod_mpnn.state_dict(), f"{OOF_DIR}/mpnn_prod.pt")

    # Step 2: Build Unified Candidate Pool (Base Tracker + Stage 2 Tangent Tubes + NeedleSliceMPNN)
    bundle_items = []
    for i, g in enumerate(graphs):
        d_pt = torch.load(f"{CACHE_DIR}/case_{i:02d}.pt", weights_only=False)
        X = d_pt["X"].numpy()
        oof_npz = np.load(f"{OOF_DIR}/oof_case_{i:02d}.npz")
        b_item = base_trajs_data[i]
        b_trajs = b_item["trajs"] if isinstance(b_item, dict) and "trajs" in b_item else b_item

        b_clean = []
        for t in b_trajs:
            tc = dict(t)
            tc["ras"] = np.asarray(tc["ras"] if "ras" in tc else tc["points"], dtype=np.float64)
            tc["points"] = tc["ras"]
            b_clean.append(tc)

        k0, i0, j0 = int(d_pt["k0"]), int(d_pt["i0"]), int(d_pt["j0"])
        sp_i, sp_j, sp_k = [float(x) for x in meta_list[i]["sp_ras"]]
        ijk2ras = g["ijk_to_ras"]
        ras2ijk = np.linalg.inv(ijk2ras)
        resp_approx = np.expm1(X[1] * 4.0)
        nn_d = {k: oof_npz[k].astype(np.float32) for k in ["p_mask", "p_peak", "vx", "vy", "p_obt"]}
        tubes = extract_nn_labelmap_tubes(nn_d, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, b_clean, include_obturator=True)

        pool = []
        for t in b_clean:
            ras = t["ras"][np.argsort(t["ras"][:, 2])]
            pool.append({
                "ras": ras,
                "base_xy": np.asarray(t.get("base_xy", ras[0, :2]), dtype=np.float64),
                "slope": np.asarray(t.get("slope", (ras[-1, :2] - ras[0, :2]) / max(1.0, ras[-1, 2] - ras[0, 2])), dtype=np.float64),
                "z_span": float(t.get("z_span_tracked", ras[-1, 2] - ras[0, 2])),
                "rmse": float(t.get("rmse", 0.35)),
                "conf_heur": float(t.get("confidence", 65.0)) / 100.0,
                "from_mpnn": 0.0, "mpnn_pa": 0.0, "mpnn_pl": 0.0, "mpnn_pe": 0.0,
            })
        for nt in tubes:
            ras = np.asarray(nt["points"], dtype=np.float64)
            ras = ras[np.argsort(ras[:, 2])]
            pool.append({
                "ras": ras,
                "base_xy": np.asarray(nt.get("base_xy", ras[0, :2]), dtype=np.float64),
                "slope": np.asarray(nt.get("slope", (ras[-1, :2] - ras[0, :2]) / max(1.0, ras[-1, 2] - ras[0, 2])), dtype=np.float64),
                "z_span": float(nt.get("z_span_tracked", ras[-1, 2] - ras[0, 2])),
                "rmse": float(nt.get("rmse", 0.35)),
                "conf_heur": float(nt.get("confidence", 65.0)) / 100.0,
                "from_mpnn": 0.5, "mpnn_pa": float(nt.get("confidence", 65.0)) / 100.0, "mpnn_pl": 0.5, "mpnn_pe": 0.6,
            })
        for mt in oof_mpnn_trajs[i]:
            ras = mt["ras"]
            matched_idx = -1
            best_d = 2.05
            for idx_p, p in enumerate(pool):
                d = traj_gt_dist(ras, p["ras"])
                if d < best_d:
                    best_d = d
                    matched_idx = idx_p
            if matched_idx >= 0:
                pool[matched_idx]["from_mpnn"] = 1.0
                pool[matched_idx]["mpnn_pa"] = mt["mean_pa"]
                pool[matched_idx]["mpnn_pl"] = mt["mean_pl"]
                pool[matched_idx]["mpnn_pe"] = mt["mean_pe"]
            else:
                pool.append({
                    "ras": ras,
                    "base_xy": mt["base_xy"],
                    "slope": mt["slope"],
                    "z_span": mt["z_span"],
                    "rmse": mt["rmse"],
                    "conf_heur": mt["mean_pa"] * mt["mean_pe"],
                    "from_mpnn": 1.0,
                    "mpnn_pa": mt["mean_pa"],
                    "mpnn_pl": mt["mean_pl"],
                    "mpnn_pe": mt["mean_pe"],
                })

        bundle_items.append(build_bundle_graph_item(pool, g, oof_npz, gts_all=gts_all_list[i], gts_long=gts_long_list[i]))

    # Step 3: 5-Fold CV + Full-Dataset Training of DenseGraphTransformer
    oof_scores = [None] * len(bundle_items)
    for fold in range(5):
        tr_b = [b for i, b in enumerate(bundle_items) if i % 5 != fold]
        gt_model = DenseGraphTransformer(in_dim=18, edge_dim=6, d_model=64).to(device)
        opt = torch.optim.AdamW(gt_model.parameters(), lr=1.8e-3, weight_decay=1e-4)
        for ep in range(75):
            gt_model.train()
            for bi in np.random.permutation(len(tr_b)):
                bg = tr_b[bi]
                opt.zero_grad(set_to_none=True)
                out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                l_rec = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"].to(device), pos_weight=torch.tensor([3.0], device=device))
                l_1to1 = F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device))
                l_long = F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device))
                loss = l_rec + 1.5 * l_1to1 + l_long
                loss.backward()
                opt.step()
        gt_model.eval()
        with torch.no_grad():
            for i, bg in enumerate(bundle_items):
                if i % 5 == fold:
                    out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                    oof_scores[i] = torch.sigmoid(out).cpu().numpy()

    print("\n=== UNIFIED PIPELINE 5-FOLD OOF CONFIDENCE SWEEP (1,063 Needles) ===")
    for min_conf in [25.0, 35.0, 45.0, 50.0, 60.0]:
        preds_all = []
        for i, bg in enumerate(bundle_items):
            sc = oof_scores[i]
            # Calibrated confidence [0..100]: blends high-recall head (sc[:, 0]) and 1-to-1 precision head (sc[:, 1])
            conf_arr = 100.0 * (0.45 * sc[:, 0] + 0.55 * sc[:, 1])
            sel = []
            for idx_c in np.argsort(-conf_arr):
                if conf_arr[idx_c] < min_conf:
                    continue
                cand = bg["pool"][idx_c]
                if any(traj_gt_dist(cand["ras"], s["ras"]) < 1.95 for s in sel):
                    continue
                sel.append(cand)
            preds_all.append(sel)
        r = evaluate_needle_set(preds_all, gts_all_list, thresh=3.5)
        print(
            f"  minConfidence={min_conf:4.1f}% -> Pred={r['tot_pred']:4d} | "
            f"Closest={r['closest_matched']:4d}/1063 ({r['closest_rec']:.2f}%) | "
            f"1-to-1={r['hung_matched']:4d}/1063 ({r['hung_rec']:.2f}%) | "
            f"Prec={r['hung_prec']:.2f}% | F1={r['hung_f1']:.2f}% | Err={r['mean_err']:.2f}mm"
        )

    # Train full 54-case production DenseGraphTransformer & export to ONNX + NPZ
    prod_gt = DenseGraphTransformer(in_dim=18, edge_dim=6, d_model=64).to(device)
    opt_g = torch.optim.AdamW(prod_gt.parameters(), lr=1.8e-3, weight_decay=1e-4)
    for ep in range(80):
        prod_gt.train()
        for bi in np.random.permutation(len(bundle_items)):
            bg = bundle_items[bi]
            opt_g.zero_grad(set_to_none=True)
            out = prod_gt(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
            loss = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"].to(device), pos_weight=torch.tensor([3.0], device=device)) + \
                   1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device)) + \
                   F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device))
            loss.backward()
            opt_g.step()

    prod_gt.eval().cpu()
    torch.save(prod_gt.state_dict(), f"{OOF_DIR}/graph_transformer_prod.pt")

    # Export DenseGraphTransformer to ONNX
    dummy_x = torch.randn(20, 18)
    dummy_af = torch.randn(20, 20, 6)
    dummy_am = torch.ones(20, 20)
    onnx_gt_path = f"{OOF_DIR}/needle_graph_transformer.onnx"
    torch.onnx.export(
        prod_gt,
        (dummy_x, dummy_af, dummy_am),
        onnx_gt_path,
        input_names=["x", "adj_feat", "adj_mask"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "M"},
            "adj_feat": {0: "M", 1: "M"},
            "adj_mask": {0: "M", 1: "M"},
            "logits": {0: "M"},
        },
        opset_version=17,
    )

    # Also save all weights of both prod_mpnn and prod_gt into a single compact NPZ
    weights_dict = {}
    for k, v in prod_mpnn.cpu().state_dict().items():
        weights_dict[f"mpnn.{k}"] = v.numpy().astype(np.float32)
    for k, v in prod_gt.cpu().state_dict().items():
        weights_dict[f"gt.{k}"] = v.numpy().astype(np.float32)
    npz_path = f"{OOF_DIR}/needle_graphnet.npz"
    np.savez_compressed(npz_path, **weights_dict)
    print(f"[Exported] {onnx_gt_path} ({os.path.getsize(onnx_gt_path)/1024:.1f} KB) and {npz_path} ({os.path.getsize(npz_path)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
