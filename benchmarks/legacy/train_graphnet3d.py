#!/usr/bin/env python3
"""
Experiment 2 (GPU 1 - RTX 4070 Ti SUPER):
1. NeedleSliceMPNN: 3D Spatio-Temporal Edge-Node Message-Passing Graph Neural Network (MPNN)
   for constructing 3D needle trajectories from 2D axial slice detections & Stage 1 labelmaps.
   - Nodes: 2D candidate cross-sections across slices k = 0..K-1 (16 features)
   - Directed Edges: Inter-slice candidate links k_i -> k_j (up to dz <= 28 mm) (12 features)
   - Message Passing: 4-layer Triplet/Collinearity MPNN + In-Plane Bundle Attention
   - Outputs: Node probability p(v_i), Edge link probability p(e_ij), Sub-voxel offset (dx_i, dy_i)
   - Graph Decoding: Max-weight degree-constrained disjoint path solver -> 3D needle splines.

2. NeedleBundleGraphTransformer: Trajectory-level GATv2 + Set Transformer operating on the
   candidate 3D trajectory graph (combining tracker, MPNN, and Stage 2 tubes) to jointly classify
   true needles vs. false positives and resolve competing/duplicate trajectories.

Evaluated with 5-fold cross-validation across all 54 GYN cases (759 CY long needles + 304 HT obturator needles = 1,063 total).
"""

import glob
import math
import os
import pickle
import time
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_DIR = "/tmp/needle_unet3d_cache"
OOF_DIR = "/home/guillaume/github/needlefinder/nn_models"
META_PATH = "/tmp/cases_meta_54.pkl"
BASE_TRAJS_PATH = "/tmp/base_trajs_54cases.pkl"


def interp_gt_xy(gt_pts, z_val):
    if z_val < gt_pts[0, 2] - 2.5 or z_val > gt_pts[-1, 2] + 2.5:
        return None
    x = float(np.interp(z_val, gt_pts[:, 2], gt_pts[:, 0]))
    y = float(np.interp(z_val, gt_pts[:, 2], gt_pts[:, 1]))
    return x, y


def build_slice_graph_for_case(case_idx, d_pt, oof_npz, meta):
    """
    Build a 3D Spatio-Temporal Graph (nodes V, directed edges E, node/edge ground-truth labels)
    from the Stage 1 OOF predictions and MRI/symmetry channels for a single clinical case.
    """
    X = d_pt["X"].numpy()  # [6, K_sub, H, W]
    k0 = int(d_pt["k0"])
    i0 = int(d_pt["i0"])
    j0 = int(d_pt["j0"])
    ijk_to_ras = np.asarray(meta["ijk_to_ras"], dtype=np.float64)
    rot = ijk_to_ras[:3, :3]
    trans = ijk_to_ras[:3, 3]
    sp_i, sp_j, sp_k = [float(x) for x in meta["sp_ras"]]

    p_mask = oof_npz["p_mask"].astype(np.float32)
    p_peak = oof_npz["p_peak"].astype(np.float32)
    vx_map = oof_npz["vx"].astype(np.float32)
    vy_map = oof_npz["vy"].astype(np.float32)
    p_obt = oof_npz["p_obt"].astype(np.float32)

    K_sub, H, W = p_mask.shape
    score_vol = 0.48 * p_peak + 0.30 * p_mask + 0.14 * p_obt + 0.12 * np.clip(X[1], 0.0, 1.2)
    nms_r = max(1, int(round(1.35 / sp_i)))
    loc_max = score_vol == maximum_filter(score_vol, size=(1, 2 * nms_r + 1, 2 * nms_r + 1))

    gts_all = []
    for g in meta["gts_all"]:
        pts = g["pts"] if isinstance(g, dict) else (g[1] if isinstance(g, tuple) else np.asarray(g))
        pts = np.asarray(pts, dtype=np.float64)
        pts = pts[np.argsort(pts[:, 2])]
        if len(pts) >= 2:
            gts_all.append(pts)

    gts_long = []
    for g in meta["gts_long"]:
        pts = g["pts"] if isinstance(g, dict) else (g[1] if isinstance(g, tuple) else np.asarray(g))
        pts = np.asarray(pts, dtype=np.float64)
        pts = pts[np.argsort(pts[:, 2])]
        if len(pts) >= 2:
            gts_long.append(pts)

    # Extract 2D candidate nodes on each slice k
    nodes_ras = []
    nodes_kr = []
    nodes_ij = []
    nodes_feat = []
    nodes_gt_id = []
    nodes_gt_long = []
    nodes_offset = []

    yy_w, xx_w = np.mgrid[-1:2, -1:2]
    center_ras = rot @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0)]) + trans
    z_base_ras = float(center_ras[2])

    for kr in range(K_sub):
        jj, ii = np.where(loc_max[kr] & (score_vol[kr] >= 0.16) & ((p_mask[kr] >= 0.18) | (p_peak[kr] >= 0.16) | (p_obt[kr] >= 0.22)))
        if len(ii) == 0:
            continue
        order = np.argsort(-score_vol[kr, jj, ii])[:38]
        for idx_p in order:
            ic, jc = int(ii[idx_p]), int(jj[idx_p])
            if not (1 <= ic < W - 1 and 1 <= jc < H - 1):
                continue
            w_patch = np.maximum(0.0, score_vol[kr, jc - 1:jc + 2, ic - 1:ic + 2])
            w_sum = float(np.sum(w_patch))
            if w_sum > 1e-6:
                sub_i = float(ic) + float(np.sum(xx_w * w_patch)) / w_sum
                sub_j = float(jc) + float(np.sum(yy_w * w_patch)) / w_sum
            else:
                sub_i, sub_j = float(ic), float(jc)

            pt_ras = rot @ np.array([sub_i + i0, sub_j + j0, float(kr + k0)]) + trans
            pm_v = float(p_mask[kr, jc, ic])
            pp_v = float(p_peak[kr, jc, ic])
            po_v = float(p_obt[kr, jc, ic])
            vx_v = float(vx_map[kr, jc, ic])
            vy_v = float(vy_map[kr, jc, ic])
            pm_max3 = float(np.max(p_mask[kr, jc - 1:jc + 2, ic - 1:ic + 2]))
            pp_max3 = float(np.max(p_peak[kr, jc - 1:jc + 2, ic - 1:ic + 2]))
            pm_mean3 = float(np.mean(p_mask[kr, jc - 1:jc + 2, ic - 1:ic + 2]))

            feat = [
                float(pt_ras[0] - center_ras[0]) / 45.0,
                float(pt_ras[1] - center_ras[1]) / 45.0,
                float(pt_ras[2] - z_base_ras) / 80.0,
                pm_v,
                pp_v,
                po_v,
                vx_v,
                vy_v,
                pm_max3,
                pp_max3,
                pm_mean3,
                float(X[0, kr, jc, ic]),
                float(X[1, kr, jc, ic]),
                float(X[2, kr, jc, ic]),
                float(X[3, kr, jc, ic]),
                float(X[5, kr, jc, ic]),
            ]

            # Match node to closest ground-truth needle at z = pt_ras[2]
            best_gid = -1
            best_d = 1.95
            best_dx, best_dy = 0.0, 0.0
            for gid, g_pts in enumerate(gts_all):
                g_xy = interp_gt_xy(g_pts, float(pt_ras[2]))
                if g_xy is not None:
                    d = math.hypot(g_xy[0] - float(pt_ras[0]), g_xy[1] - float(pt_ras[1]))
                    if d < best_d:
                        best_d = d
                        best_gid = gid
                        best_dx = g_xy[0] - float(pt_ras[0])
                        best_dy = g_xy[1] - float(pt_ras[1])

            best_long_id = -1
            best_ld = 1.95
            for lid, l_pts in enumerate(gts_long):
                l_xy = interp_gt_xy(l_pts, float(pt_ras[2]))
                if l_xy is not None:
                    d = math.hypot(l_xy[0] - float(pt_ras[0]), l_xy[1] - float(pt_ras[1]))
                    if d < best_ld:
                        best_ld = d
                        best_long_id = lid

            nodes_ras.append(pt_ras)
            nodes_kr.append(kr)
            nodes_ij.append((sub_i + i0, sub_j + j0))
            nodes_feat.append(feat)
            nodes_gt_id.append(best_gid)
            nodes_gt_long.append(best_long_id)
            nodes_offset.append([best_dx, best_dy])

    N = len(nodes_ras)
    if N == 0:
        return None

    nodes_ras = np.asarray(nodes_ras, dtype=np.float32)
    nodes_kr = np.asarray(nodes_kr, dtype=np.int32)
    nodes_ij = np.asarray(nodes_ij, dtype=np.float32)
    nodes_feat = np.asarray(nodes_feat, dtype=np.float32)
    nodes_gt_id = np.asarray(nodes_gt_id, dtype=np.int32)
    nodes_gt_long = np.asarray(nodes_gt_long, dtype=np.int32)
    nodes_offset = np.asarray(nodes_offset, dtype=np.float32)

    # Build directed inter-slice edges (i -> j) with 1 <= kr_j - kr_i <= max_dk
    max_dk = max(4, int(round(24.0 / sp_k)))
    edge_src = []
    edge_dst = []
    edge_feat = []
    edge_label = []

    by_slice = {}
    for idx_n, kr in enumerate(nodes_kr):
        by_slice.setdefault(int(kr), []).append(idx_n)

    for u in range(N):
        ku = int(nodes_kr[u])
        ru = nodes_ras[u]
        vx_u, vy_u = float(nodes_feat[u, 6]), float(nodes_feat[u, 7])
        for dk in range(1, max_dk + 1):
            kv = ku + dk
            if kv not in by_slice:
                continue
            for v in by_slice[kv]:
                rv = nodes_ras[v]
                dz = float(rv[2] - ru[2])
                if dz <= 0.2:
                    continue
                sx = float(rv[0] - ru[0]) / dz
                sy = float(rv[1] - ru[1]) / dz
                s_norm = math.hypot(sx, sy)
                if s_norm > 0.56:
                    continue
                vx_v, vy_v = float(nodes_feat[v, 6]), float(nodes_feat[v, 7])
                pred_x = float(ru[0]) + 0.5 * (vx_u + vx_v) * dz
                pred_y = float(ru[1]) + 0.5 * (vy_u + vy_v) * dz
                err_tan = math.hypot(float(rv[0]) - pred_x, float(rv[1]) - pred_y)
                max_err = 2.8 + 0.14 * dz
                if err_tan > max_err and math.hypot(float(rv[0] - ru[0]), float(rv[1] - ru[1])) > 3.5 + 0.22 * dz:
                    continue

                err_u = math.hypot(sx - vx_u, sy - vy_u)
                err_v = math.hypot(sx - vx_v, sy - vy_v)
                dv_uv = math.hypot(vx_u - vx_v, vy_u - vy_v)

                # Midpoint probability check in volume
                km = int(round(0.5 * (ku + kv)))
                im = int(round(0.5 * (nodes_ij[u, 0] + nodes_ij[v, 0]) - i0))
                jm = int(round(0.5 * (nodes_ij[u, 1] + nodes_ij[v, 1]) - j0))
                if 0 <= km < K_sub and 0 <= jm < H and 0 <= im < W:
                    pm_mid = float(p_mask[km, jm, im])
                    pp_mid = float(p_peak[km, jm, im])
                else:
                    pm_mid, pp_mid = 0.0, 0.0

                ef = [
                    sx,
                    sy,
                    s_norm,
                    dz / 25.0,
                    float(dk) / float(max_dk),
                    err_tan / 5.0,
                    err_u,
                    err_v,
                    dv_uv,
                    pm_mid,
                    pp_mid,
                    min(float(nodes_feat[u, 4]), float(nodes_feat[v, 4])),
                ]
                is_same_gt = (nodes_gt_id[u] >= 0) and (nodes_gt_id[u] == nodes_gt_id[v])
                edge_src.append(u)
                edge_dst.append(v)
                edge_feat.append(ef)
                edge_label.append(1.0 if is_same_gt else 0.0)

    # Also build in-plane spatial neighbor edges on the same slice (within 18 mm) for bundle context
    spatial_src, spatial_dst, spatial_feat = [], [], []
    for kr, idxs in by_slice.items():
        for i_a in range(len(idxs)):
            u = idxs[i_a]
            for i_b in range(len(idxs)):
                if i_a == i_b:
                    continue
                v = idxs[i_b]
                dx = float(nodes_ras[v, 0] - nodes_ras[u, 0])
                dy = float(nodes_ras[v, 1] - nodes_ras[u, 1])
                dist_xy = math.hypot(dx, dy)
                if dist_xy <= 22.0:
                    spatial_src.append(u)
                    spatial_dst.append(v)
                    spatial_feat.append([
                        dx / 22.0,
                        dy / 22.0,
                        dist_xy / 22.0,
                        float(nodes_feat[v, 6] - nodes_feat[u, 6]),
                        float(nodes_feat[v, 7] - nodes_feat[u, 7]),
                    ])

    return {
        "case_idx": case_idx,
        "case_id": meta["case_id"],
        "nodes_ras": nodes_ras,
        "nodes_kr": nodes_kr,
        "nodes_ij": nodes_ij,
        "nodes_feat": torch.tensor(nodes_feat, dtype=torch.float32),
        "nodes_y_all": torch.tensor((nodes_gt_id >= 0).astype(np.float32), dtype=torch.float32),
        "nodes_y_long": torch.tensor((nodes_gt_long >= 0).astype(np.float32), dtype=torch.float32),
        "nodes_offset": torch.tensor(nodes_offset, dtype=torch.float32),
        "edge_src": torch.tensor(edge_src, dtype=torch.long),
        "edge_dst": torch.tensor(edge_dst, dtype=torch.long),
        "edge_feat": torch.tensor(edge_feat, dtype=torch.float32) if edge_feat else torch.zeros((0, 12), dtype=torch.float32),
        "edge_y": torch.tensor(edge_label, dtype=torch.float32) if edge_label else torch.zeros((0,), dtype=torch.float32),
        "spatial_src": torch.tensor(spatial_src, dtype=torch.long) if spatial_src else torch.zeros((0,), dtype=torch.long),
        "spatial_dst": torch.tensor(spatial_dst, dtype=torch.long) if spatial_dst else torch.zeros((0,), dtype=torch.long),
        "spatial_feat": torch.tensor(spatial_feat, dtype=torch.float32) if spatial_feat else torch.zeros((0, 5), dtype=torch.float32),
        "k0": k0,
        "sp_ras": (sp_i, sp_j, sp_k),
        "ijk_to_ras": ijk_to_ras,
        "z_base_ras": z_base_ras,
        "center_ras": center_ras,
        "gts_long": gts_long,
        "gts_all": gts_all,
    }


def scatter_sum(src, index, dim_size):
    out = torch.zeros((dim_size, src.shape[1]), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def scatter_softmax(scores, index, dim_size):
    max_per_node = torch.full((dim_size, 1), -1e9, device=scores.device, dtype=scores.dtype)
    max_per_node.scatter_reduce_(0, index.unsqueeze(1), scores, reduce="amax", include_self=True)
    exp_s = torch.exp((scores - max_per_node[index]).clamp(-20.0, 20.0))
    denom = scatter_sum(exp_s, index, dim_size).clamp(min=1e-6)
    return exp_s / denom[index]


class TripletCollinearityMPNNLayer(nn.Module):
    """
    Message-Passing Graph Neural Network Layer with:
    1. Directed Incoming (h -> i) and Outgoing (i -> j) Attention Aggregation
    2. In-Plane Spatial Bundle Attention across same-slice catheters
    3. Triplet Collinearity Edge Update: e_ij <- MLP([e_ij, h_i, h_j, m_i^in, m_j^out, m_i^out, m_j^in])
       Because m_i^in summarizes incoming edges (h -> i) and m_j^out summarizes outgoing edges (j -> l),
       the edge update directly computes multi-slice collinearity and bending energy!
    """
    def __init__(self, node_dim=64, edge_dim=64, spatial_dim=32):
        super().__init__()
        self.in_msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
        )
        self.in_attn = nn.Linear(node_dim + edge_dim, 1)

        self.out_msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
        )
        self.out_attn = nn.Linear(node_dim + edge_dim, 1)

        self.sp_msg_mlp = nn.Sequential(
            nn.Linear(node_dim + spatial_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
        )
        self.sp_attn = nn.Linear(node_dim + spatial_dim, 1)

        self.node_upd = nn.Sequential(
            nn.Linear(node_dim * 4, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
            nn.LayerNorm(node_dim),
        )

        self.edge_upd = nn.Sequential(
            nn.Linear(edge_dim + node_dim * 4, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
        )

    def forward(self, h, e, src, dst, sp_src, sp_dst, sp_e):
        N = h.shape[0]
        E = e.shape[0]
        if E > 0:
            # Incoming messages to dst: h_src + e -> dst
            in_cat = torch.cat([h[src], e], dim=-1)
            w_in = scatter_softmax(self.in_attn(in_cat), dst, N)
            m_in = scatter_sum(w_in * self.in_msg_mlp(in_cat), dst, N)

            # Outgoing messages to src: h_dst + e -> src
            out_cat = torch.cat([h[dst], e], dim=-1)
            w_out = scatter_softmax(self.out_attn(out_cat), src, N)
            m_out = scatter_sum(w_out * self.out_msg_mlp(out_cat), src, N)
        else:
            m_in = torch.zeros_like(h)
            m_out = torch.zeros_like(h)

        if sp_src.numel() > 0:
            sp_cat = torch.cat([h[sp_src], sp_e], dim=-1)
            w_sp = scatter_softmax(self.sp_attn(sp_cat), sp_dst, N)
            m_sp = scatter_sum(w_sp * self.sp_msg_mlp(sp_cat), sp_dst, N)
        else:
            m_sp = torch.zeros_like(h)

        h_new = F.silu(h + self.node_upd(torch.cat([h, m_in, m_out, m_sp], dim=-1)))

        if E > 0:
            # Triplet collinearity update: combines h_src, h_dst, m_in[src] (incoming to src), m_out[dst] (outgoing from dst)
            e_cat = torch.cat([e, h_new[src], h_new[dst], m_in[src], m_out[dst]], dim=-1)
            e_new = F.silu(e + self.edge_upd(e_cat))
        else:
            e_new = e

        return h_new, e_new


class NeedleSliceMPNN(nn.Module):
    """4-Layer Spatio-Temporal Triplet MPNN for 3D Needle Graph Linking & Node Refinement."""
    def __init__(self, in_node_dim=16, in_edge_dim=12, in_sp_dim=5, hidden_dim=64, num_layers=4):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(in_edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.sp_enc = nn.Sequential(
            nn.Linear(in_sp_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            TripletCollinearityMPNNLayer(hidden_dim, hidden_dim, 32)
            for _ in range(num_layers)
        ])

        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 4),  # [logit_all, logit_long, dx_mm, dy_mm]
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),  # [logit_edge]
        )

    def forward(self, g, device):
        h = self.node_enc(g["nodes_feat"].to(device))
        e = self.edge_enc(g["edge_feat"].to(device))
        sp_e = self.sp_enc(g["spatial_feat"].to(device)) if g["spatial_feat"].shape[0] > 0 else torch.zeros((0, 32), device=device)
        src = g["edge_src"].to(device)
        dst = g["edge_dst"].to(device)
        sp_src = g["spatial_src"].to(device)
        sp_dst = g["spatial_dst"].to(device)

        for layer in self.layers:
            h, e = layer(h, e, src, dst, sp_src, sp_dst, sp_e)

        node_out = self.node_head(h)
        edge_logit = self.edge_head(e).squeeze(-1) if e.shape[0] > 0 else torch.zeros((0,), device=device)
        return node_out, edge_logit


def decode_mpnn_trajectories(g, node_out, edge_logit, edge_thresh=0.42, node_thresh=0.38):
    """
    Decode 3D needle trajectories from MPNN node & edge predictions via degree-constrained
    maximum-weight disjoint path linking (at most 1 incoming and 1 outgoing edge per node)
    followed by collinear path stitching and regularized polynomial fitting.
    """
    p_node_all = torch.sigmoid(node_out[:, 0]).cpu().numpy()
    p_node_long = torch.sigmoid(node_out[:, 1]).cpu().numpy()
    offsets = np.clip(node_out[:, 2:4].cpu().numpy(), -1.5, 1.5)
    p_edge = torch.sigmoid(edge_logit).cpu().numpy() if edge_logit.numel() > 0 else np.zeros(0)

    nodes_ras = g["nodes_ras"].copy()
    nodes_ras[:, 0] += offsets[:, 0]
    nodes_ras[:, 1] += offsets[:, 1]
    nodes_kr = g["nodes_kr"]
    src = g["edge_src"].numpy()
    dst = g["edge_dst"].numpy()

    # Greedy max-weight degree-constrained bipartite matching across directed edges
    valid_e = np.where(
        (p_edge >= edge_thresh)
        & (p_node_all[src] >= node_thresh * 0.8)
        & (p_node_all[dst] >= node_thresh * 0.8)
    )[0]
    edge_scores = p_edge[valid_e] * np.sqrt(p_node_all[src[valid_e]] * p_node_all[dst[valid_e]])
    order_e = valid_e[np.argsort(-edge_scores)]

    N = len(nodes_ras)
    next_node = np.full(N, -1, dtype=np.int32)
    prev_node = np.full(N, -1, dtype=np.int32)
    edge_prob_map = {}

    for e_idx in order_e:
        u = int(src[e_idx])
        v = int(dst[e_idx])
        if next_node[u] == -1 and prev_node[v] == -1:
            # Check local angle compatibility if u already has a predecessor h -> u
            if prev_node[u] != -1:
                h = prev_node[u]
                dz1 = max(0.5, float(nodes_ras[u, 2] - nodes_ras[h, 2]))
                dz2 = max(0.5, float(nodes_ras[v, 2] - nodes_ras[u, 2]))
                s1x, s1y = (nodes_ras[u, 0] - nodes_ras[h, 0]) / dz1, (nodes_ras[u, 1] - nodes_ras[h, 1]) / dz1
                s2x, s2y = (nodes_ras[v, 0] - nodes_ras[u, 0]) / dz2, (nodes_ras[v, 1] - nodes_ras[u, 1]) / dz2
                if math.hypot(s1x - s2x, s1y - s2y) > 0.22:
                    continue
            next_node[u] = v
            prev_node[v] = u
            edge_prob_map[(u, v)] = float(p_edge[e_idx])

    # Extract disjoint paths
    visited = np.zeros(N, dtype=bool)
    raw_paths = []
    for u in np.argsort(nodes_kr):
        if visited[u] or prev_node[u] != -1 or next_node[u] == -1:
            continue
        path = [int(u)]
        visited[u] = True
        cur = int(u)
        while next_node[cur] != -1 and not visited[next_node[cur]]:
            nxt = int(next_node[cur])
            path.append(nxt)
            visited[nxt] = True
            cur = nxt
        if len(path) >= 3:
            raw_paths.append(path)

    # Package paths into smooth 3D needle trajectories
    z_base_ras = g["z_base_ras"]
    sp_k = g["sp_ras"][2]
    trajs = []
    for path in raw_paths:
        pts = nodes_ras[path].astype(np.float64)
        krs = nodes_kr[path]
        z_span = float(pts[-1, 2] - pts[0, 2])
        mean_pa = float(np.mean(p_node_all[path]))
        mean_pl = float(np.mean(p_node_long[path]))
        e_probs = [edge_prob_map.get((path[i], path[i + 1]), 0.5) for i in range(len(path) - 1)]
        mean_pe = float(np.mean(e_probs)) if e_probs else 0.5

        if z_span < 7.5 or (len(path) < 4 and z_span < 10.5):
            continue
        if mean_pa < node_thresh:
            continue

        w = np.sqrt(np.maximum(0.1, p_node_all[path]))
        deg = 2 if (len(path) >= 8 and z_span >= 28.0 and sp_k < 2.8) else 1
        try:
            px = np.polyfit(pts[:, 2], pts[:, 0], deg, w=w)
            py = np.polyfit(pts[:, 2], pts[:, 1], deg, w=w)
            if deg == 2 and (abs(px[0]) > 0.0011 or abs(py[0]) > 0.0011):
                px = np.polyfit(pts[:, 2], pts[:, 0], 1, w=w)
                py = np.polyfit(pts[:, 2], pts[:, 1], 1, w=w)
        except Exception:
            continue

        fx = np.polyval(px, pts[:, 2])
        fy = np.polyval(py, pts[:, 2])
        rmse = float(np.sqrt(np.mean((pts[:, 0] - fx) ** 2 + (pts[:, 1] - fy) ** 2)))
        if rmse > 1.45:
            continue

        px1 = np.polyfit(pts[:, 2], pts[:, 0], 1, w=w)
        py1 = np.polyfit(pts[:, 2], pts[:, 1], 1, w=w)
        slope = np.array([float(px1[0]), float(py1[0])])
        if np.linalg.norm(slope) > 0.52:
            continue

        z_start = float(pts[0, 2])
        base_gap = max(0.0, z_start - z_base_ras)
        if base_gap <= 22.0 and int(krs[0]) <= 6:
            bx = float(px1[0] * z_base_ras + px1[1])
            by = float(py1[0] * z_base_ras + py1[1])
            smooth_pts = np.vstack([[bx, by, z_base_ras], np.column_stack([fx, fy, pts[:, 2]])])
        else:
            bx, by = float(fx[0]), float(fy[0])
            smooth_pts = np.column_stack([fx, fy, pts[:, 2]])

        trajs.append({
            "ras": smooth_pts,
            "base_xy": np.array([bx, by]),
            "slope": slope,
            "z_span": z_span,
            "k_min_rel": int(krs[0]),
            "k_max_rel": int(krs[-1]),
            "n_pts": len(path),
            "rmse": rmse,
            "mean_pa": mean_pa,
            "mean_pl": mean_pl,
            "mean_pe": mean_pe,
            "graph_score": mean_pa * mean_pe * math.log1p(len(path)),
        })

    # Deduplicate overlapping MPNN paths
    trajs.sort(key=lambda x: -x["graph_score"])
    dedup = []
    for t in trajs:
        is_dup = False
        for p in dedup:
            z0 = max(t["ras"][0, 2], p["ras"][0, 2])
            z1 = min(t["ras"][-1, 2], p["ras"][-1, 2])
            if z1 >= z0 + 3.0:
                zs = np.linspace(z0, z1, 9)
                d = float(np.mean(np.hypot(
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 0]) - np.interp(zs, p["ras"][:, 2], p["ras"][:, 0]),
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 1]) - np.interp(zs, p["ras"][:, 2], p["ras"][:, 1]),
                )))
                if d < 2.25:
                    is_dup = True
                    break
        if not is_dup:
            dedup.append(t)
    return dedup


class NeedleBundleGraphTransformer(nn.Module):
    """
    Trajectory-Level Graph Attention Network (GATv2) + Multi-Head Self-Attention Set Transformer.
    Takes all candidate 3D trajectories in a scan (from multi-pass tracking + MPNN graph decoding),
    models pairwise spatial/angular competition via GATv2 message passing, and models global
    bundle structure via Transformer self-attention.
    """
    def __init__(self, in_dim=18, edge_dim=6, d_model=64, n_heads=4):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        # 2 GATv2 relational layers for local trajectory competition/reinforcement
        self.gat1_msg = nn.Linear(d_model * 3, d_model)
        self.gat1_att = nn.Linear(d_model * 3, 1)
        self.gat1_norm = nn.LayerNorm(d_model)

        self.gat2_msg = nn.Linear(d_model * 3, d_model)
        self.gat2_att = nn.Linear(d_model * 3, 1)
        self.gat2_norm = nn.LayerNorm(d_model)

        # 2 Global Set-Transformer Self-Attention layers across the entire needle bundle
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.05, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2),  # [logit_all (CY+HT), logit_long (CY)]
        )

    def forward(self, x, e_src, e_dst, e_feat):
        # x: [M, in_dim]
        M = x.shape[0]
        h = self.node_proj(x)
        if e_src.numel() > 0:
            ef = self.edge_proj(e_feat)
            cat1 = torch.cat([h[e_src], h[e_dst], ef], dim=-1)
            w1 = scatter_softmax(self.gat1_att(cat1), e_dst, M)
            m1 = scatter_sum(w1 * F.silu(self.gat1_msg(cat1)), e_dst, M)
            h = self.gat1_norm(h + m1)

            cat2 = torch.cat([h[e_src], h[e_dst], ef], dim=-1)
            w2 = scatter_softmax(self.gat2_att(cat2), e_dst, M)
            m2 = scatter_sum(w2 * F.silu(self.gat2_msg(cat2)), e_dst, M)
            h = self.gat2_norm(h + m2)

        h = self.transformer(h.unsqueeze(0)).squeeze(0)
        return self.head(h)


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


def evaluate_needle_set(pred_list_per_case, gts_per_case, thresh=3.5):
    tot_gt = 0
    tot_pred = 0
    tot_closest = 0
    tot_hung = 0
    errs = []
    for preds, gts in zip(pred_list_per_case, gts_per_case):
        tot_gt += len(gts)
        tot_pred += len(preds)
        if not gts or not preds:
            continue
        cost = np.zeros((len(gts), len(preds)), dtype=np.float64)
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                cost[i, j] = traj_gt_dist(p["ras"], g)
        # Closest match
        min_d = np.min(cost, axis=1)
        tot_closest += int(np.sum(min_d <= thresh))
        # Hungarian 1-to-1 match
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] <= thresh:
                tot_hung += 1
                errs.append(float(cost[r, c]))
    rec_c = 100.0 * tot_closest / max(1, tot_gt)
    rec_h = 100.0 * tot_hung / max(1, tot_gt)
    prec_h = 100.0 * tot_hung / max(1, tot_pred)
    f1_h = 2 * rec_h * prec_h / max(1e-6, rec_h + prec_h)
    return {
        "tot_gt": tot_gt,
        "tot_pred": tot_pred,
        "closest_matched": tot_closest,
        "closest_rec": rec_c,
        "hung_matched": tot_hung,
        "hung_rec": rec_h,
        "hung_prec": prec_h,
        "hung_f1": f1_h,
        "mean_err": float(np.mean(errs)) if errs else 0.0,
        "med_err": float(np.median(errs)) if errs else 0.0,
    }


def main():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[NeedleGraphNet] Using device: {device} ({torch.cuda.get_device_name(device)})")

    t0 = time.time()
    meta_list = pickle.load(open(META_PATH, "rb"))
    graphs = []
    for i in range(len(meta_list)):
        d_pt = torch.load(f"{CACHE_DIR}/case_{i:02d}.pt", weights_only=False)
        oof_npz = np.load(f"{OOF_DIR}/oof_case_{i:02d}.npz")
        g = build_slice_graph_for_case(i, d_pt, oof_npz, meta_list[i])
        graphs.append(g)

    tot_nodes = sum(len(g["nodes_ras"]) for g in graphs)
    tot_edges = sum(len(g["edge_src"]) for g in graphs)
    pos_nodes = sum(int(g["nodes_y_all"].sum().item()) for g in graphs)
    pos_edges = sum(int(g["edge_y"].sum().item()) for g in graphs)
    print(
        f"[NeedleSliceMPNN] Built 54 3D Spatio-Temporal Graphs in {time.time()-t0:.1f}s | "
        f"Nodes: {tot_nodes:,} ({pos_nodes:,} positive, {100*pos_nodes/tot_nodes:.1f}%) | "
        f"Directed Edges: {tot_edges:,} ({pos_edges:,} positive, {100*pos_edges/tot_edges:.1f}%)"
    )

    # 5-Fold Cross-Validation of NeedleSliceMPNN
    oof_mpnn_trajs = [None] * len(graphs)
    edge_aucs = []
    node_aucs = []

    for fold in range(5):
        train_gs = [g for i, g in enumerate(graphs) if i % 5 != fold]
        val_gs = [g for i, g in enumerate(graphs) if i % 5 == fold]

        model = NeedleSliceMPNN(in_node_dim=16, in_edge_dim=12, in_sp_dim=5, hidden_dim=64, num_layers=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-4)
        epochs = 45
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        best_val_f1 = -1.0
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(train_gs))
            for bi in perm:
                g = train_gs[bi]
                opt.zero_grad(set_to_none=True)
                node_out, edge_logit = model(g, device)
                y_na = g["nodes_y_all"].to(device)
                y_nl = g["nodes_y_long"].to(device)
                y_off = g["nodes_offset"].to(device)
                y_e = g["edge_y"].to(device)

                pw_n = torch.tensor([3.0], device=device)
                loss_na = F.binary_cross_entropy_with_logits(node_out[:, 0], y_na, pos_weight=pw_n)
                loss_nl = F.binary_cross_entropy_with_logits(node_out[:, 1], y_nl, pos_weight=pw_n)
                fg_n = (y_na > 0.5).float().unsqueeze(1)
                loss_off = (F.smooth_l1_loss(node_out[:, 2:4], y_off, reduction="none") * fg_n).sum() / fg_n.sum().clamp(min=1.0)

                if edge_logit.numel() > 0:
                    pw_e = torch.tensor([6.0], device=device)
                    loss_e = F.binary_cross_entropy_with_logits(edge_logit, y_e, pos_weight=pw_e)
                else:
                    loss_e = 0.0

                loss = loss_na + 0.5 * loss_nl + 0.5 * loss_off + 2.0 * loss_e
                loss.backward()
                opt.step()
            sched.step()

            if ep % 15 == 0 or ep == epochs:
                model.eval()
                e_tp, e_fp, e_fn = 0, 0, 0
                n_tp, n_fp, n_fn = 0, 0, 0
                with torch.no_grad():
                    for g in val_gs:
                        no, el = model(g, device)
                        pn = (torch.sigmoid(no[:, 0]).cpu() >= 0.5).numpy()
                        yn = (g["nodes_y_all"].numpy() >= 0.5)
                        n_tp += int(np.sum(pn & yn))
                        n_fp += int(np.sum(pn & (~yn)))
                        n_fn += int(np.sum((~pn) & yn))
                        if el.numel() > 0:
                            pe = (torch.sigmoid(el).cpu() >= 0.5).numpy()
                            ye = (g["edge_y"].numpy() >= 0.5)
                            e_tp += int(np.sum(pe & ye))
                            e_fp += int(np.sum(pe & (~ye)))
                            e_fn += int(np.sum((~pe) & ye))
                n_f1 = 2 * n_tp / max(1, 2 * n_tp + n_fp + n_fn)
                e_f1 = 2 * e_tp / max(1, 2 * e_tp + e_fp + e_fn)
                comb = 0.6 * e_f1 + 0.4 * n_f1
                if comb > best_val_f1:
                    best_val_f1 = comb
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for i, g in enumerate(graphs):
                if i % 5 == fold:
                    no, el = model(g, device)
                    oof_mpnn_trajs[i] = decode_mpnn_trajectories(g, no, el)
        print(f"[NeedleSliceMPNN Fold {fold}] Best Val Graph Link+Node F1 = {best_val_f1*100:.2f}%")

    # Evaluate pure NeedleSliceMPNN decoded trajectories vs Ground Truth
    gts_long_list = [g["gts_long"] for g in graphs]
    gts_all_list = [g["gts_all"] for g in graphs]
    res_mpnn_all = evaluate_needle_set(oof_mpnn_trajs, gts_all_list, thresh=3.5)
    print(
        f"[Pure NeedleSliceMPNN (Stage 2 Graph Decoder Only) on 1,063 CY+HT Needles] "
        f"Closest Recall: {res_mpnn_all['closest_matched']}/{res_mpnn_all['tot_gt']} ({res_mpnn_all['closest_rec']:.2f}%) | "
        f"1-to-1 Recall: {res_mpnn_all['hung_matched']}/{res_mpnn_all['tot_gt']} ({res_mpnn_all['hung_rec']:.2f}%) | "
        f"Mean Err: {res_mpnn_all['mean_err']:.2f} mm (Median: {res_mpnn_all['med_err']:.2f} mm)"
    )

    # Now build Trajectory-Level Bundle Graph combining Base Tracker + Stage 2 Tubes + NeedleSliceMPNN Paths
    # and train NeedleBundleGraphTransformer (GATv2 + Set Transformer) via 5-fold CV
    base_trajs_data = pickle.load(open(BASE_TRAJS_PATH, "rb"))
    bundle_graphs = []
    for i, g in enumerate(graphs):
        b_item = base_trajs_data[i]
        b_trajs = b_item["trajs"] if isinstance(b_item, dict) and "trajs" in b_item else b_item

        # Merge candidate pool: base tracker candidates + MPNN graph-decoded trajectories
        pool = []
        for t in b_trajs:
            ras = np.asarray(t["ras"] if "ras" in t else t["points"], dtype=np.float64)
            ras = ras[np.argsort(ras[:, 2])]
            pool.append({
                "ras": ras,
                "base_xy": np.asarray(t.get("base_xy", ras[0, :2]), dtype=np.float64),
                "slope": np.asarray(t.get("slope", (ras[-1, :2] - ras[0, :2]) / max(1.0, ras[-1, 2] - ras[0, 2])), dtype=np.float64),
                "z_span": float(t.get("z_span_tracked", ras[-1, 2] - ras[0, 2])),
                "rmse": float(t.get("rmse", 0.35)),
                "conf_heur": float(t.get("confidence", 65.0)) / 100.0,
                "from_mpnn": 0.0,
                "mpnn_pa": 0.0,
                "mpnn_pl": 0.0,
                "mpnn_pe": 0.0,
            })

        for mt in oof_mpnn_trajs[i]:
            ras = mt["ras"]
            # Check if already in pool within 2.0 mm; if so, enrich existing candidate with MPNN graph scores!
            matched_idx = -1
            best_d = 2.05
            for idx_p, p in enumerate(pool):
                z0 = max(ras[0, 2], p["ras"][0, 2])
                z1 = min(ras[-1, 2], p["ras"][-1, 2])
                if z1 >= z0 + 2.0:
                    zs = np.linspace(z0, z1, 8)
                    d = float(np.mean(np.hypot(
                        np.interp(zs, ras[:, 2], ras[:, 0]) - np.interp(zs, p["ras"][:, 2], p["ras"][:, 0]),
                        np.interp(zs, ras[:, 2], ras[:, 1]) - np.interp(zs, p["ras"][:, 2], p["ras"][:, 1]),
                    )))
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

        # Extract 18-dim feature vector per candidate trajectory + Stage 1 OOF probabilities along shaft
        oof_npz = np.load(f"{OOF_DIR}/oof_case_{i:02d}.npz")
        pm_arr = oof_npz["p_mask"].astype(np.float32)
        pp_arr = oof_npz["p_peak"].astype(np.float32)
        po_arr = oof_npz["p_obt"].astype(np.float32)
        ras_to_ijk = np.linalg.inv(g["ijk_to_ras"])
        k0_c, i0_c, j0_c = int(oof_npz["k0"]), int(oof_npz["i0"]), int(oof_npz["j0"])
        K_s, H_s, W_s = pm_arr.shape

        bases = np.array([p["base_xy"] for p in pool]) if pool else np.zeros((0, 2))
        slopes = np.array([p["slope"] for p in pool]) if pool else np.zeros((0, 2))
        med_base = np.median(bases, axis=0) if len(bases) else np.zeros(2)
        med_slope = np.median(slopes, axis=0) if len(slopes) else np.zeros(2)

        x_feats = []
        y_all_t = []
        y_long_t = []
        for p in pool:
            ras = p["ras"]
            zs = np.linspace(ras[0, 2], ras[-1, 2], 15)
            xs = np.interp(zs, ras[:, 2], ras[:, 0])
            ys = np.interp(zs, ras[:, 2], ras[:, 1])
            pts_h = np.column_stack([xs, ys, zs, np.ones_like(zs)])
            ijks = (ras_to_ijk @ pts_h.T).T
            pm_l, pp_l, po_l = [], [], []
            for row in ijks:
                ic = int(round(row[0] - i0_c))
                jc = int(round(row[1] - j0_c))
                kc = int(round(row[2] - k0_c))
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

            d_all = min([traj_gt_dist(ras, gt) for gt in g["gts_all"]], default=99.0)
            d_lng = min([traj_gt_dist(ras, gt) for gt in g["gts_long"]], default=99.0)
            y_all_t.append(1.0 if d_all <= 3.2 else 0.0)
            y_long_t.append(1.0 if d_lng <= 3.2 else 0.0)

        # Build trajectory graph edges
        M = len(pool)
        es, ed, ef = [], [], []
        for u in range(M):
            for v in range(M):
                if u == v:
                    continue
                db = float(np.hypot(pool[u]["base_xy"][0] - pool[v]["base_xy"][0], pool[u]["base_xy"][1] - pool[v]["base_xy"][1]))
                if db <= 35.0:
                    ds = float(np.hypot(pool[u]["slope"][0] - pool[v]["slope"][0], pool[u]["slope"][1] - pool[v]["slope"][1]))
                    sh_d = traj_gt_dist(pool[u]["ras"], pool[v]["ras"])
                    es.append(u)
                    ed.append(v)
                    ef.append([
                        db / 35.0,
                        sh_d / 35.0,
                        ds / 0.30,
                        1.0 if sh_d < 2.5 else 0.0,
                        float(x_feats[v][5] - x_feats[u][5]),
                        float(x_feats[v][0] - x_feats[u][0]),
                    ])

        bundle_graphs.append({
            "pool": pool,
            "x": torch.tensor(x_feats, dtype=torch.float32),
            "y_all": torch.tensor(y_all_t, dtype=torch.float32),
            "y_long": torch.tensor(y_long_t, dtype=torch.float32),
            "e_src": torch.tensor(es, dtype=torch.long),
            "e_dst": torch.tensor(ed, dtype=torch.long),
            "e_feat": torch.tensor(ef, dtype=torch.float32) if ef else torch.zeros((0, 6), dtype=torch.float32),
            "gts_all": g["gts_all"],
            "gts_long": g["gts_long"],
        })

    # 5-Fold Cross-Validation of NeedleBundleGraphTransformer
    oof_preds_all = [None] * len(bundle_graphs)
    oof_preds_long = [None] * len(bundle_graphs)

    for fold in range(5):
        tr_b = [b for i, b in enumerate(bundle_graphs) if i % 5 != fold]
        va_b = [b for i, b in enumerate(bundle_graphs) if i % 5 == fold]

        b_model = NeedleBundleGraphTransformer(in_dim=18, edge_dim=6, d_model=64, n_heads=4).to(device)
        opt = torch.optim.AdamW(b_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        for ep in range(1, 61):
            b_model.train()
            for b in np.random.permutation(len(tr_b)):
                bg = tr_b[b]
                opt.zero_grad(set_to_none=True)
                out = b_model(bg["x"].to(device), bg["e_src"].to(device), bg["e_dst"].to(device), bg["e_feat"].to(device))
                pw = torch.tensor([2.5], device=device)
                loss = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_all"].to(device), pos_weight=pw) + \
                       F.binary_cross_entropy_with_logits(out[:, 1], bg["y_long"].to(device), pos_weight=pw)
                loss.backward()
                opt.step()

        b_model.eval()
        with torch.no_grad():
            for i, bg in enumerate(bundle_graphs):
                if i % 5 == fold:
                    out = b_model(bg["x"].to(device), bg["e_src"].to(device), bg["e_dst"].to(device), bg["e_feat"].to(device))
                    pa = torch.sigmoid(out[:, 0]).cpu().numpy()
                    pl = torch.sigmoid(out[:, 1]).cpu().numpy()

                    # NMS deduplication sorted by GraphTransformer probability
                    sel_all = []
                    for idx_c in np.argsort(-pa):
                        if pa[idx_c] < 0.42:
                            continue
                        cand = bg["pool"][idx_c]
                        if any(traj_gt_dist(cand["ras"], s["ras"]) < 2.15 for s in sel_all):
                            continue
                        sel_all.append(cand)
                    oof_preds_all[i] = sel_all

                    sel_long = []
                    for idx_c in np.argsort(-pl):
                        if pl[idx_c] < 0.45:
                            continue
                        cand = bg["pool"][idx_c]
                        if any(traj_gt_dist(cand["ras"], s["ras"]) < 2.15 for s in sel_long):
                            continue
                        sel_long.append(cand)
                    oof_preds_long[i] = sel_long

    res_gt_all = evaluate_needle_set(oof_preds_all, gts_all_list, thresh=3.5)
    res_gt_long = evaluate_needle_set(oof_preds_long, gts_long_list, thresh=3.5)

    print("=" * 85)
    print("[NeedleGraphNet + BundleGraphTransformer 5-Fold OOF Results across 54 Clinical Cases]")
    print(
        f"1. Full 1,063 Needles (759 CY + 304 HT Obturator):\n"
        f"   - Closest-Match Recall: {res_gt_all['closest_matched']} / {res_gt_all['tot_gt']} ({res_gt_all['closest_rec']:.2f}%)\n"
        f"   - 1-to-1 Hungarian Recall: {res_gt_all['hung_matched']} / {res_gt_all['tot_gt']} ({res_gt_all['hung_rec']:.2f}%)\n"
        f"   - Precision: {res_gt_all['hung_prec']:.2f}% | F1: {res_gt_all['hung_f1']:.2f}%\n"
        f"   - Shaft Error: Mean = {res_gt_all['mean_err']:.2f} mm | Median = {res_gt_all['med_err']:.2f} mm"
    )
    print(
        f"2. Standard 759 CY Long Needles (760CYScenes.txt):\n"
        f"   - Closest-Match Recall: {res_gt_long['closest_matched']} / {res_gt_long['tot_gt']} ({res_gt_long['closest_rec']:.2f}%)\n"
        f"   - 1-to-1 Hungarian Recall: {res_gt_long['hung_matched']} / {res_gt_long['tot_gt']} ({res_gt_long['hung_rec']:.2f}%)\n"
        f"   - Precision: {res_gt_long['hung_prec']:.2f}% | F1: {res_gt_long['hung_f1']:.2f}%\n"
        f"   - Shaft Error: Mean = {res_gt_long['mean_err']:.2f} mm | Median = {res_gt_long['med_err']:.2f} mm"
    )
    print("=" * 85)


if __name__ == "__main__":
    main()
