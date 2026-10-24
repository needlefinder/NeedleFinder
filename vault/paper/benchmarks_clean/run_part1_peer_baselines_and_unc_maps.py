#!/usr/bin/env python3
"""
Part 1:
  1. Compute & Cache Stage 1 5-Model Epistemic Uncertainty Maps (`oof_unc_case_{idx:02d}.npz` and
     fold-isolated `inner_unc_f{f_out}_case_{idx:02d}.npz`) from the 25 trained inner models
     `unet3d_outer{0..4}_inner{0..4}.pt` in `/home/guillaume/github/needlefinder/nn_models_clean/`.
  2. Train & Evaluate 3 Peer-Reviewed MRI Brachytherapy Methods on our exact 54-patient (1,063-catheter)
     strictly nested 5x5 splits:
     - Baseline 4: Dai et al. 2020 (Medical Physics 47(9):4115-4124)
       Deeply Supervised 3D Attention-Gated U-Net + Axial Total-Variation (TV) Continuity Regularization
       + Slice-to-Slice Centroid Linking & Poly3 Smoothing.
     - Baseline 5: Kostoulas et al. 2025 (Proc. SPIE Medical Imaging 13408, Amsterdam UMC)
       3D Residual U-Net + Multi-Peak Blob Splitting + Iterative Collinear Gap-Bridging for Discontinuous
       Needles + False-Needle-Part Pruning & Poly3 Fitting.
     - Baseline 6: Aleong et al. 2024 / Kim et al. 2026 (Medical Physics 51(8):5361 / Brachytherapy 25(1):85)
       5-Model 3D U-Net Ensemble Epistemic Uncertainty + Ensemble-Disagreement Filtering +
       Inverse-Variance-Weighted 3rd-Order Polynomial Fitting & Uncertainty-Driven Tip Refinement.
"""

import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
import types
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label as cc_label, maximum_filter
import torch
import torch.nn as nn
import torch.nn.functional as F

for mod in (
    "vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util",
    "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"
):
    sys.modules[mod] = types.ModuleType(mod)

sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean")

from train_and_eval_unet3d import NeedleUNet3D
from train_nested_stage1 import infer_both_single_and_4view_tta_logits, infer_single_view_3d, raw_out_to_probs
from train_baselines_pmb2026 import AnisoResBlock3D, SEBlock3D, NativeResUNet3D_MaskOnly
from evaluate_unified_apples_to_apples import evaluate_method_unified, get_gts, traj_gt_dist
from run_xiao2026_and_fewshot_benchmark import analyze_catheter_adhesion, regularize_poly3_trajectory

CLEAN_CACHE_DIR = "/tmp/needle_unet3d_cache_clean"
CLEAN_META_PATH = "/tmp/cases_meta_54_clean.pkl"
OUT_DIR = "/home/guillaume/github/needlefinder/nn_models_clean"
os.makedirs(OUT_DIR, exist_ok=True)


def get_nested_fold_split(f_out, n_cases=54):
    outer_test = [i for i in range(n_cases) if i % 5 == f_out]
    outer_train = [i for i in range(n_cases) if i % 5 != f_out]
    inner_val = [idx for pos, idx in enumerate(outer_train) if pos % 5 == f_out]
    inner_subtrain = [idx for pos, idx in enumerate(outer_train) if pos % 5 != f_out]
    return outer_test, outer_train, inner_subtrain, inner_val


# ==============================================================================
# STEP 0: STAGE 1 5-MODEL EPISTEMIC UNCERTAINTY MAPS (`oof_unc_case_{idx:02d}.npz`)
# ==============================================================================

def compute_and_cache_stage1_epistemic_uncertainty(items, device):
    """
    Uses the 25 already-trained Stage 1 inner-fold models `unet3d_outer{f_out}_inner{0..4}.pt`
    to compute voxel-wise 5-model epistemic uncertainty maps (`sigma_mask`, `sigma_peak`, `sigma_vec`)
    for all 54 outer-test cases (`oof_unc_case_{idx:02d}.npz`), plus fold-isolated inner-ensemble
    uncertainty on `outer_train` (`inner_unc_f{f_out}_case_{idx:02d}.npz`).
    """
    t0 = time.perf_counter()
    missing_outer = [
        i for i in range(54)
        if not os.path.exists(os.path.join(OUT_DIR, f"oof_unc_case_{i:02d}.npz"))
    ]
    if not missing_outer:
        print("[Stage 1 Epistemic Uncertainty] All 54 oof_unc_case_{idx:02d}.npz already cached!", flush=True)
        return

    print(f"[Stage 1 Epistemic Uncertainty] Computing 5-model epistemic uncertainty maps on {device}...", flush=True)
    for f_out in range(5):
        outer_test = [i for i in range(54) if i % 5 == f_out]
        inner_models = []
        for m in range(5):
            ckpt_path = os.path.join(OUT_DIR, f"unet3d_outer{f_out}_inner{m}.pt")
            mod = NeedleUNet3D(in_ch=6, base_ch=24).to(device)
            mod.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=True)
            mod.eval()
            inner_models.append(mod)

        for t_idx in outer_test:
            unc_path = os.path.join(OUT_DIR, f"oof_unc_case_{t_idx:02d}.npz")
            if os.path.exists(unc_path):
                continue
            d = items[t_idx]
            xv = d["X"]
            probs_m = []
            for m in range(5):
                o_single = infer_single_view_3d(inner_models[m], xv, device)
                pm = raw_out_to_probs(o_single, temp_pos=1.0)
                probs_m.append(pm.numpy())
            probs_stack = np.stack(probs_m, axis=0)  # [5, 5, K, H, W]
            sigma_mask = np.std(probs_stack[:, 0], axis=0).astype(np.float16)
            sigma_peak = np.std(probs_stack[:, 1], axis=0).astype(np.float16)
            sigma_vec = np.sqrt(np.var(probs_stack[:, 2], axis=0) + np.var(probs_stack[:, 3], axis=0)).astype(np.float16)
            sigma_obt = np.std(probs_stack[:, 4], axis=0).astype(np.float16)
            np.savez(
                unc_path,
                sigma_mask=sigma_mask,
                sigma_peak=sigma_peak,
                sigma_vec=sigma_vec,
                sigma_obt=sigma_obt,
                k0=int(d["k0"]),
                i0=int(d["i0"]),
                j0=int(d["j0"]),
            )
        print(f"  [Stage 1 Uncertainty] Finished Outer Fold {f_out} ({time.perf_counter()-t0:.1f}s)", flush=True)


# ==============================================================================
# PEER-REVIEWED BASELINE 4: DAI ET AL. 2020 (Medical Physics 47(9):4115-4124)
# Deeply Supervised 3D Attention-Gated U-Net + Axial Total-Variation Regularization
# ==============================================================================

class AttentionGate3D(nn.Module):
    """3D Additive Attention Gate on skip connections (Oktay et al. 2018 / Dai et al. 2020)."""
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm3d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(f_l, f_int, kernel_size=1, bias=False),
            nn.BatchNorm3d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        alpha = self.psi(psi)
        return x * alpha


class Dai2020AttentionUNet3D(nn.Module):
    """
    Dai et al. 2020 (Medical Physics 47(9):4115-4124):
    Deeply Supervised 3D Attention-Gated U-Net on 1-channel T2W MRI (`in_ch=1`),
    with 3D Attention Gates on all skip connections (`ag3`, `ag2`, `ag1`) and
    3 Deep Supervision output heads (`head1`, `head2`, `head3`).
    """
    def __init__(self, in_ch=1, base_ch=24):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 6
        self.enc1 = nn.Sequential(AnisoResBlock3D(in_ch, c1), AnisoResBlock3D(c1, c1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc2 = nn.Sequential(AnisoResBlock3D(c1, c2), AnisoResBlock3D(c2, c2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc3 = nn.Sequential(AnisoResBlock3D(c2, c3), AnisoResBlock3D(c3, c3))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bottleneck = nn.Sequential(AnisoResBlock3D(c3, c4), AnisoResBlock3D(c4, c4))

        self.ag3 = AttentionGate3D(f_g=c4, f_l=c3, f_int=c3 // 2)
        self.dec3 = AnisoResBlock3D(c4 + c3, c3)
        self.ag2 = AttentionGate3D(f_g=c3, f_l=c2, f_int=c2 // 2)
        self.dec2 = AnisoResBlock3D(c3 + c2, c2)
        self.ag1 = AttentionGate3D(f_g=c2, f_l=c1, f_int=c1 // 2)
        self.dec1 = nn.Sequential(AnisoResBlock3D(c2 + c1, c1), AnisoResBlock3D(c1, c1))

        self.head1 = nn.Conv3d(c1, 1, kernel_size=1)
        self.head2 = nn.Conv3d(c2, 1, kernel_size=1)
        self.head3 = nn.Conv3d(c3, 1, kernel_size=1)

    def forward(self, x, return_deep_sup=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        u3 = F.interpolate(b, size=e3.shape[2:], mode="trilinear", align_corners=False)
        s3 = self.ag3(g=u3, x=e3)
        d3 = self.dec3(torch.cat([u3, s3], dim=1))

        u2 = F.interpolate(d3, size=e2.shape[2:], mode="trilinear", align_corners=False)
        s2 = self.ag2(g=u2, x=e2)
        d2 = self.dec2(torch.cat([u2, s2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        s1 = self.ag1(g=u1, x=e1)
        d1 = self.dec1(torch.cat([u1, s1], dim=1))

        out1 = self.head1(d1)
        if not return_deep_sup:
            return out1
        out2 = F.interpolate(self.head2(d2), size=e1.shape[2:], mode="trilinear", align_corners=False)
        out3 = F.interpolate(self.head3(d3), size=e1.shape[2:], mode="trilinear", align_corners=False)
        return out1, out2, out3


def compute_single_scale_dice_bce(logits, target):
    prob = torch.sigmoid(logits)
    t = target.float()
    inter = (prob * t).sum(dim=(1, 2, 3, 4))
    union = prob.sum(dim=(1, 2, 3, 4)) + t.sum(dim=(1, 2, 3, 4))
    loss_dice = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = torch.where(t >= 0.5, prob, 1.0 - prob)
    focal_w = (1.0 - pt).pow(2.0) * (1.0 + 16.0 * t)
    loss_focal = (bce * focal_w).mean()
    return loss_dice + loss_focal


def compute_dai2020_deepsup_tv_loss(out_tuple, target, lambda_tv=0.18):
    """
    Dai et al. 2020 Loss:
      Multi-scale Deep Supervision (Dice + Focal BCE on d1, d2, d3)
      + Axial Total-Variation (TV) Continuity Regularization along Z:
        L_TV_z = lambda_tv * mean(|p(z+1, y, x) - p(z, y, x)|)
    """
    out1, out2, out3 = out_tuple
    l1 = compute_single_scale_dice_bce(out1, target)
    l2 = compute_single_scale_dice_bce(out2, target)
    l3 = compute_single_scale_dice_bce(out3, target)

    p1 = torch.sigmoid(out1)
    # Weighted axial total variation (penalizes slice-to-slice discontinuities along catheter shafts)
    tv_z = torch.abs(p1[:, :, 1:, :, :] - p1[:, :, :-1, :, :])
    fg_weight = 1.0 + 8.0 * torch.maximum(target[:, :, 1:, :, :], target[:, :, :-1, :, :])
    loss_tv_z = (tv_z * fg_weight).mean()

    return l1 + 0.4 * l2 + 0.2 * l3 + lambda_tv * loss_tv_z


def augment_1ch_gpu(x_cpu: torch.Tensor, y_cpu: torch.Tensor, device: torch.device, max_d=36, max_hw=176):
    _, D, H, W = x_cpu.shape
    d0 = random.randint(0, D - max_d) if D > max_d else 0
    d1 = min(D, d0 + max_d)
    h0 = random.randint(0, H - max_hw) if H > max_hw else 0
    h1 = min(H, h0 + max_hw)
    w0 = random.randint(0, W - max_hw) if W > max_hw else 0
    w1 = min(W, w0 + max_hw)

    x = x_cpu[:, d0:d1, h0:h1, w0:w1].to(device, non_blocking=True).clone()
    y = y_cpu[:, d0:d1, h0:h1, w0:w1].to(device, non_blocking=True).clone()

    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
        y = torch.flip(y, dims=[3])
    if random.random() < 0.25:
        x = torch.flip(x, dims=[2])
        y = torch.flip(y, dims=[2])
    if random.random() < 0.75:
        dy = random.randint(-10, 10)
        dx = random.randint(-10, 10)
        x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        y = torch.roll(y, shifts=(dy, dx), dims=(2, 3))
    if random.random() < 0.70:
        scale = random.uniform(0.84, 1.18)
        bias = random.uniform(-0.07, 0.07)
        x = (x * scale + bias).clamp(-0.5, 2.5)
    return x, y


def infer_sliding_1ch_tta(model, x_cpu, device, chunk_d=28, use_tta=True):
    def _single(x_in):
        _, D, H, W = x_in.shape
        if D <= chunk_d and H <= 512 and W <= 512:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x_in.unsqueeze(0).to(device))
            return torch.sigmoid(out.float().squeeze(0).cpu()[0])
        out_sum = torch.zeros((D, H, W), dtype=torch.float32)
        w_sum = torch.zeros((D, 1, 1), dtype=torch.float32)
        eff_c = 16 if (H > 512 or W > 512) else chunk_d
        stride = max(8, eff_c - 8)
        for ks in range(0, D, stride):
            ke = min(D, ks + eff_c)
            if ke - ks < 8:
                ks = max(0, ke - eff_c)
            sub = x_in[:, ks:ke].unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(model(sub).float().squeeze(0).cpu()[0])
            out_sum[ks:ke] += p
            w_sum[ks:ke] += 1.0
            if ke == D:
                break
        return out_sum / w_sum.clamp(min=1.0)

    p0 = _single(x_cpu)
    if not use_tta:
        return p0.numpy()
    p1 = torch.flip(_single(torch.flip(x_cpu, dims=[3])), dims=[2])
    return (0.50 * p0 + 0.50 * p1).numpy()


def extract_dai2020_trajectories(p_mask, meta_d, prob_thresh=0.38):
    """
    Dai et al. 2020 (Medical Physics 47(9):4115-4124) Trajectory Reconstruction:
      1. Slice-wise 2D connected components & local peak centroid extraction on p_mask >= prob_thresh.
      2. Axial slice-to-slice centroid linking (<= 3.2 mm in-plane step, up to 2-slice gap bridging)
         guided by running linear tangent estimate.
      3. Minimum axial span filtering and 3rd-order polynomial curve smoothing.
    """
    K, H, W = p_mask.shape
    k0, i0, j0 = meta_d["k0"], meta_d["i0"], meta_d["j0"]
    sp_i, sp_j, sp_k = meta_d["sp_ras"]
    ijk_to_ras = meta_d["ijk_to_ras"]
    rot = ijk_to_ras[:3, :3]
    trans = ijk_to_ras[:3, 3]
    z_base_ras = float((rot @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0 + 1)]) + trans)[2])

    smooth_p = gaussian_filter(p_mask.astype(np.float32), sigma=(0.35, 0.65, 0.65))
    nms_r = max(1, int(round(1.30 / min(sp_i, sp_j))))
    loc_max = (smooth_p == maximum_filter(smooth_p, size=(1, 2 * nms_r + 1, 2 * nms_r + 1))) & (p_mask >= prob_thresh)
    yy_w, xx_w = np.mgrid[-1:2, -1:2]

    slice_pts = {}
    for k in range(K):
        jj, ii = np.where(loc_max[k])
        if len(ii) == 0:
            continue
        pts_k = []
        for jc, ic in zip(jj, ii):
            if 1 <= jc < H - 1 and 1 <= ic < W - 1:
                patch = p_mask[k, jc - 1:jc + 2, ic - 1:ic + 2]
                ws = float(np.sum(patch))
                si = float(ic) + float(np.sum(xx_w * patch)) / max(ws, 1e-6)
                sj = float(jc) + float(np.sum(yy_w * patch)) / max(ws, 1e-6)
            else:
                si, sj = float(ic), float(jc)
            pt_ras = rot @ np.array([si + i0, sj + j0, float(k + k0)]) + trans
            pts_k.append((pt_ras[0], pt_ras[1], pt_ras[2], float(p_mask[k, jc, ic]), k))
        slice_pts[k] = pts_k

    min_slices = 4 if sp_k >= 3.2 else (5 if sp_k >= 2.2 else 7)
    max_dk = 3  # up to 2-slice gap bridging
    used = set()
    trajs = []

    for k_start in sorted(slice_pts.keys()):
        # Sort by probability descending so strong cores link first
        order_p = sorted(range(len(slice_pts[k_start])), key=lambda idx: -slice_pts[k_start][idx][3])
        for idx_p in order_p:
            if (k_start, idx_p) in used:
                continue
            pt0 = slice_pts[k_start][idx_p]
            chain = [pt0]
            used.add((k_start, idx_p))
            curr_x, curr_y, curr_z, _, curr_k = pt0
            vx_est, vy_est = 0.0, 0.0
            for k_next in range(k_start + 1, K):
                if k_next - curr_k > max_dk:
                    break
                if k_next not in slice_pts:
                    continue
                best_j = -1
                best_dist = 2.8 + 0.15 * abs(k_next - curr_k) * sp_k
                for j_p, cand in enumerate(slice_pts[k_next]):
                    if (k_next, j_p) in used:
                        continue
                    dz = cand[2] - curr_z
                    pred_x = curr_x + vx_est * dz
                    pred_y = curr_y + vy_est * dz
                    dist = math.hypot(cand[0] - pred_x, cand[1] - pred_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j_p
                if best_j >= 0:
                    chosen = slice_pts[k_next][best_j]
                    used.add((k_next, best_j))
                    chain.append(chosen)
                    if len(chain) >= 3:
                        dz_tot = chain[-1][2] - chain[0][2]
                        if abs(dz_tot) > 1.0:
                            vx_est = 0.55 * vx_est + 0.45 * ((chain[-1][0] - chain[0][0]) / dz_tot)
                            vy_est = 0.55 * vy_est + 0.45 * ((chain[-1][1] - chain[0][1]) / dz_tot)
                    curr_x, curr_y, curr_z, _, curr_k = chosen

            if len(chain) < min_slices:
                continue
            arr = np.array([[c[0], c[1], c[2], c[3]] for c in chain], dtype=np.float64)
            arr = arr[np.argsort(arr[:, 2])]
            z_span = float(arr[-1, 2] - arr[0, 2])
            if z_span < 12.0:
                continue
            deg = 3 if len(arr) >= 9 else (2 if len(arr) >= 5 else 1)
            px = np.polyfit(arr[:, 2], arr[:, 0], deg)
            py = np.polyfit(arr[:, 2], arr[:, 1], deg)
            fx = np.polyval(px, arr[:, 2])
            fy = np.polyval(py, arr[:, 2])
            rmse = float(np.sqrt(np.mean((arr[:, 0] - fx) ** 2 + (arr[:, 1] - fy) ** 2)))
            if rmse > 1.45:
                continue
            sx = float(fx[-1] - fx[0]) / max(1.0, z_span)
            sy = float(fy[-1] - fy[0]) / max(1.0, z_span)
            if math.hypot(sx, sy) > 0.58:
                continue
            mean_prob = float(np.mean(arr[:, 3]))
            if mean_prob < 0.42:
                continue
            smooth_ras = np.column_stack([fx, fy, arr[:, 2]])
            is_long = (abs(arr[0, 2] - z_base_ras) <= 28.0) and (z_span >= 24.0)
            trajs.append({
                "ras": smooth_ras,
                "points": smooth_ras,
                "confidence": mean_prob * 100.0,
                "z_span": z_span,
                "is_long": is_long,
            })

    trajs.sort(key=lambda t: -t["confidence"] * math.log1p(t["z_span"]))
    dedup = []
    for t in trajs:
        dup = False
        for ex in dedup:
            z0 = max(float(t["ras"][0, 2]), float(ex["ras"][0, 2]))
            z1 = min(float(t["ras"][-1, 2]), float(ex["ras"][-1, 2]))
            if z1 > z0 + 2.0:
                zs = np.linspace(z0, z1, 10)
                d = float(np.mean(np.hypot(
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 0]) - np.interp(zs, ex["ras"][:, 2], ex["ras"][:, 0]),
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 1]) - np.interp(zs, ex["ras"][:, 2], ex["ras"][:, 1]),
                )))
                if d < 1.95:
                    dup = True
                    break
        if not dup:
            dedup.append(t)
    return dedup


# ==============================================================================
# PEER-REVIEWED BASELINE 5: KOSTOULAS ET AL. 2025 (Proc. SPIE 13408, Amsterdam UMC)
# 3D Residual U-Net + Error-Tolerant Gap-Bridging & False-Needle-Part Pruning
# ==============================================================================

def _fit_seg_slope(ras):
    ras = np.asarray(ras, dtype=np.float64)
    if len(ras) < 2:
        return 0.0, 0.0
    dz = max(1.0, float(ras[-1, 2] - ras[0, 2]))
    if len(ras) >= 4:
        zc = ras[:, 2] - np.mean(ras[:, 2])
        denom = float(np.dot(zc, zc))
        if denom > 1e-6:
            return float(np.dot(zc, ras[:, 0]) / denom), float(np.dot(zc, ras[:, 1]) / denom)
    return float(ras[-1, 0] - ras[0, 0]) / dz, float(ras[-1, 1] - ras[0, 1]) / dz


def extract_kostoulas2025_trajectories(p_mask, meta_d, prob_thresh=0.40):
    """
    Kostoulas et al. 2025 (Proc. SPIE Medical Imaging 13408, Amsterdam UMC):
      1. Multi-peak 2D distance-transform + probability centroid extraction per axial slice
         (splitting merged multi-catheter blobs when peaks are >= 1.8 mm apart), followed by
         strict contiguous slice-to-slice segment formation (max_dk = 2).
      2. Iterative Collinear Gap-Bridging for Discontinuous Needles:
         Iteratively merges broken segments (S_low, S_high) separated by segmentation dropouts
         (-3.0 mm <= gap <= 18.0 mm) when bidirectional linear tangent extrapolation to the gap
         midpoint gives an in-plane residual <= 2.35 mm and 3D slope difference <= 0.14 (~8 deg).
      3. False-Needle-Part Pruning & 3rd-Order Polynomial Curve Fitting:
         Prunes isolated false-positive fragments failing minimum length (< 14.5 mm), excessive
         curvature/RMSE, or floating far above the obturator/insertion zone without strong support.
    """
    K, H, W = p_mask.shape
    k0, i0, j0 = meta_d["k0"], meta_d["i0"], meta_d["j0"]
    sp_i, sp_j, sp_k = meta_d["sp_ras"]
    ijk_to_ras = meta_d["ijk_to_ras"]
    rot = ijk_to_ras[:3, :3]
    trans = ijk_to_ras[:3, 3]
    z_base_ras = float((rot @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0 + 1)]) + trans)[2])

    bin_m = p_mask >= prob_thresh
    ridge = np.zeros_like(p_mask, dtype=np.float32)
    for k in range(K):
        if np.any(bin_m[k]):
            edt = distance_transform_edt(bin_m[k], sampling=(sp_j, sp_i))
            ridge[k] = p_mask[k] * (1.0 + 0.55 * np.clip(edt, 0.0, 2.0))

    nms_r = max(1, int(round(1.40 / min(sp_i, sp_j))))
    loc_max = (ridge == maximum_filter(ridge, size=(1, 2 * nms_r + 1, 2 * nms_r + 1))) & (p_mask >= prob_thresh)
    yy_w, xx_w = np.mgrid[-1:2, -1:2]

    slice_pts = {}
    for k in range(K):
        jj, ii = np.where(loc_max[k])
        if len(ii) == 0:
            continue
        pts_k = []
        for jc, ic in zip(jj, ii):
            if 1 <= jc < H - 1 and 1 <= ic < W - 1:
                patch = p_mask[k, jc - 1:jc + 2, ic - 1:ic + 2]
                ws = float(np.sum(patch))
                si = float(ic) + float(np.sum(xx_w * patch)) / max(ws, 1e-6)
                sj = float(jc) + float(np.sum(yy_w * patch)) / max(ws, 1e-6)
            else:
                si, sj = float(ic), float(jc)
            pt_ras = rot @ np.array([si + i0, sj + j0, float(k + k0)]) + trans
            pts_k.append((pt_ras[0], pt_ras[1], pt_ras[2], float(p_mask[k, jc, ic]), k))
        slice_pts[k] = pts_k

    # Step 1: Contiguous initial track segments (allowing short fragments >= 2 slices so broken parts can be bridged!)
    used = set()
    segments = []
    for k_start in sorted(slice_pts.keys()):
        order_p = sorted(range(len(slice_pts[k_start])), key=lambda idx: -slice_pts[k_start][idx][3])
        for idx_p in order_p:
            if (k_start, idx_p) in used:
                continue
            pt0 = slice_pts[k_start][idx_p]
            chain = [pt0]
            used.add((k_start, idx_p))
            curr_x, curr_y, curr_z, _, curr_k = pt0
            vx_est, vy_est = 0.0, 0.0
            for k_next in range(k_start + 1, K):
                if k_next - curr_k > 2:
                    break
                if k_next not in slice_pts:
                    continue
                best_j = -1
                best_dist = 2.45 + 0.12 * abs(k_next - curr_k) * sp_k
                for j_p, cand in enumerate(slice_pts[k_next]):
                    if (k_next, j_p) in used:
                        continue
                    dz = cand[2] - curr_z
                    pred_x = curr_x + vx_est * dz
                    pred_y = curr_y + vy_est * dz
                    dist = math.hypot(cand[0] - pred_x, cand[1] - pred_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j_p
                if best_j >= 0:
                    chosen = slice_pts[k_next][best_j]
                    used.add((k_next, best_j))
                    chain.append(chosen)
                    if len(chain) >= 2:
                        dz_tot = chain[-1][2] - chain[0][2]
                        if abs(dz_tot) > 0.8:
                            vx_est = 0.5 * vx_est + 0.5 * ((chain[-1][0] - chain[0][0]) / dz_tot)
                            vy_est = 0.5 * vy_est + 0.5 * ((chain[-1][1] - chain[0][1]) / dz_tot)
                    curr_x, curr_y, curr_z, _, curr_k = chosen
            if len(chain) >= 2:
                arr = np.array([[c[0], c[1], c[2], c[3]] for c in chain], dtype=np.float64)
                arr = arr[np.argsort(arr[:, 2])]
                segments.append(arr)

    # Step 2: Kostoulas et al. 2025 Iterative Collinear Gap-Bridging across Discontinuous Needles
    segments.sort(key=lambda s: float(s[0, 2]))
    merged_any = True
    while merged_any and len(segments) >= 2:
        merged_any = False
        best_pair = None
        best_cost = 999.0
        for u in range(len(segments)):
            su = segments[u]
            z_top_u = float(su[-1, 2])
            vx_u, vy_u = _fit_seg_slope(su[-min(6, len(su)):])
            for v in range(len(segments)):
                if u == v:
                    continue
                sv = segments[v]
                z_bot_v = float(sv[0, 2])
                gap = z_bot_v - z_top_u
                if not (-3.0 <= gap <= 18.0 and float(sv[-1, 2]) > z_top_u + 3.0 and float(su[0, 2]) < z_bot_v - 2.0):
                    continue
                vx_v, vy_v = _fit_seg_slope(sv[:min(6, len(sv))])
                d_slope = math.hypot(vx_u - vx_v, vy_u - vy_v)
                if d_slope > 0.145:
                    continue
                z_mid = 0.5 * (z_top_u + z_bot_v)
                xu_m = float(su[-1, 0]) + vx_u * (z_mid - z_top_u)
                yu_m = float(su[-1, 1]) + vy_u * (z_mid - z_top_u)
                xv_m = float(sv[0, 0]) + vx_v * (z_mid - z_bot_v)
                yv_m = float(sv[0, 1]) + vy_v * (z_mid - z_bot_v)
                d_mid = math.hypot(xu_m - xv_m, yu_m - yv_m)
                if d_mid <= 2.35:
                    cost = d_mid + 4.0 * d_slope + 0.04 * max(0.0, gap)
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (u, v)
        if best_pair is not None:
            u, v = best_pair
            comb = np.vstack([segments[u], segments[v]])
            comb = comb[np.argsort(comb[:, 2])]
            # Verify combined polynomial residual before committing merge
            deg_c = 3 if len(comb) >= 9 else (2 if len(comb) >= 5 else 1)
            px_c = np.polyfit(comb[:, 2], comb[:, 0], deg_c)
            py_c = np.polyfit(comb[:, 2], comb[:, 1], deg_c)
            rmse_c = float(np.sqrt(np.mean((comb[:, 0] - np.polyval(px_c, comb[:, 2])) ** 2 + (comb[:, 1] - np.polyval(py_c, comb[:, 2])) ** 2)))
            if rmse_c <= 1.35:
                new_segs = [segments[idx] for idx in range(len(segments)) if idx not in (u, v)]
                new_segs.append(comb)
                new_segs.sort(key=lambda s: float(s[0, 2]))
                segments = new_segs
                merged_any = True

    # Step 3: False-Needle-Part Pruning & 3rd-Order Polynomial Fitting
    min_slices = 4 if sp_k >= 3.2 else (5 if sp_k >= 2.2 else 7)
    trajs = []
    for arr in segments:
        if len(arr) < min_slices:
            continue
        z_span = float(arr[-1, 2] - arr[0, 2])
        if z_span < 14.5:
            continue
        mean_prob = float(np.mean(arr[:, 3]))
        # Prune isolated upper-pelvis false needle parts that do not originate near insertion zone
        dist_from_base = float(arr[0, 2] - z_base_ras)
        if dist_from_base > 28.0 and (mean_prob < 0.62 or z_span < 22.0):
            continue
        if mean_prob < 0.45:
            continue

        deg = 3 if len(arr) >= 9 else (2 if len(arr) >= 5 else 1)
        px = np.polyfit(arr[:, 2], arr[:, 0], deg)
        py = np.polyfit(arr[:, 2], arr[:, 1], deg)
        fx = np.polyval(px, arr[:, 2])
        fy = np.polyval(py, arr[:, 2])
        rmse = float(np.sqrt(np.mean((arr[:, 0] - fx) ** 2 + (arr[:, 1] - fy) ** 2)))
        if rmse > 1.35:
            continue
        sx = float(fx[-1] - fx[0]) / max(1.0, z_span)
        sy = float(fy[-1] - fy[0]) / max(1.0, z_span)
        if math.hypot(sx, sy) > 0.56:
            continue

        n_out = max(len(arr), int(round(z_span / 2.0)) + 2)
        zs_out = np.linspace(float(arr[0, 2]), float(arr[-1, 2]), n_out)
        smooth_ras = np.column_stack([np.polyval(px, zs_out), np.polyval(py, zs_out), zs_out])
        is_long = (abs(arr[0, 2] - z_base_ras) <= 28.0) and (z_span >= 24.0)
        trajs.append({
            "ras": smooth_ras,
            "points": smooth_ras,
            "confidence": mean_prob * 100.0,
            "z_span": z_span,
            "is_long": is_long,
        })

    # Also apply IRGM-like bundle outlier pruning if >= 4 catheters exist
    if len(trajs) >= 4:
        all_xy = np.concatenate([t["ras"][:, :2] for t in trajs], axis=0)
        center_xy = np.median(all_xy, axis=0)
        rad_95 = float(np.percentile(np.linalg.norm(all_xy - center_xy[None, :], axis=1), 95.0)) + 16.0
        trajs = [
            t for t in trajs
            if float(np.linalg.norm(t["ras"][:, :2].mean(axis=0) - center_xy)) <= rad_95 or t["confidence"] >= 72.0
        ]

    trajs.sort(key=lambda t: -t["confidence"] * math.log1p(t["z_span"]))
    dedup = []
    for t in trajs:
        dup = False
        for ex in dedup:
            z0 = max(float(t["ras"][0, 2]), float(ex["ras"][0, 2]))
            z1 = min(float(t["ras"][-1, 2]), float(ex["ras"][-1, 2]))
            if z1 > z0 + 2.0:
                zs = np.linspace(z0, z1, 10)
                d = float(np.mean(np.hypot(
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 0]) - np.interp(zs, ex["ras"][:, 2], ex["ras"][:, 0]),
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 1]) - np.interp(zs, ex["ras"][:, 2], ex["ras"][:, 1]),
                )))
                if d < 2.0:
                    dup = True
                    break
        if not dup:
            dedup.append(t)
    return dedup


# ==============================================================================
# PEER-REVIEWED BASELINE 6: ALEONG ET AL. 2024 / KIM ET AL. 2026
# (Medical Physics 51(8):5361 / Brachytherapy 25(1):85)
# 5-Model Epistemic Uncertainty Ensemble + Inverse-Variance Weighted Poly3 & Tip Refinement
# ==============================================================================

def extract_aleong2024_kim2026_trajectories(p_mean, p_std, meta_d, conf_thresh=0.36):
    """
    Aleong et al. 2024 / Kim et al. 2026 Uncertainty-Aware Trajectory Reconstruction:
      1. Computes voxel-wise uncertainty-penalized confidence map `p_conf = clip(p_mean - 0.5 * p_std, 0, 1)`
         and inverse-variance precision map `w_prec = p_mean / (p_std^2 + 0.02)`.
      2. Extracts sub-voxel precision-weighted 2D slice centroids and links trajectories across slices
         (including collinear gap bridging across local high-uncertainty dropouts).
      3. Aleong 2024 Ensemble-Disagreement Filtering: rejects false-positive tracks with high epistemic
         uncertainty-to-probability ratio (`mean(p_std) / mean(p_mean) > 0.42`) or low confidence.
      4. Kim 2026 Inverse-Variance-Weighted 3rd-Order Polynomial Fitting & Uncertainty-Driven Tip Refinement:
         Fits `np.polyfit(zs, xs, deg=3, w=w_k)` with weights `w_k = p_mean(z_k) / (p_std(z_k)^2 + 0.02)`
         and refines the cranial tip along the local tangent to the last slice where `p_mean - 0.8*p_std >= 0.28`.
    """
    K, H, W = p_mean.shape
    k0, i0, j0 = meta_d["k0"], meta_d["i0"], meta_d["j0"]
    sp_i, sp_j, sp_k = meta_d["sp_ras"]
    ijk_to_ras = meta_d["ijk_to_ras"]
    ras2ijk = np.linalg.inv(ijk_to_ras)
    rot = ijk_to_ras[:3, :3]
    trans = ijk_to_ras[:3, 3]
    z_base_ras = float((rot @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0 + 1)]) + trans)[2])

    p_conf = np.clip(p_mean - 0.50 * p_std, 0.0, 1.0).astype(np.float32)
    w_prec = (p_mean / (p_std ** 2 + 0.02)).astype(np.float32)

    bin_m = p_conf >= conf_thresh
    ridge = np.zeros_like(p_conf, dtype=np.float32)
    for k in range(K):
        if np.any(bin_m[k]):
            edt = distance_transform_edt(bin_m[k], sampling=(sp_j, sp_i))
            ridge[k] = p_conf[k] * (1.0 + 0.50 * np.clip(edt, 0.0, 2.0))

    nms_r = max(1, int(round(1.35 / min(sp_i, sp_j))))
    loc_max = (ridge == maximum_filter(ridge, size=(1, 2 * nms_r + 1, 2 * nms_r + 1))) & (p_conf >= conf_thresh)
    yy_w, xx_w = np.mgrid[-1:2, -1:2]

    slice_pts = {}
    for k in range(K):
        jj, ii = np.where(loc_max[k])
        if len(ii) == 0:
            continue
        pts_k = []
        for jc, ic in zip(jj, ii):
            if 1 <= jc < H - 1 and 1 <= ic < W - 1:
                # Precision-weighted sub-voxel centroiding (Kim et al. 2026)
                patch_w = w_prec[k, jc - 1:jc + 2, ic - 1:ic + 2]
                ws = float(np.sum(patch_w))
                si = float(ic) + float(np.sum(xx_w * patch_w)) / max(ws, 1e-6)
                sj = float(jc) + float(np.sum(yy_w * patch_w)) / max(ws, 1e-6)
            else:
                si, sj = float(ic), float(jc)
            pt_ras = rot @ np.array([si + i0, sj + j0, float(k + k0)]) + trans
            pts_k.append((
                pt_ras[0], pt_ras[1], pt_ras[2],
                float(p_mean[k, jc, ic]), float(p_std[k, jc, ic]), float(p_conf[k, jc, ic]), k
            ))
        slice_pts[k] = pts_k

    used = set()
    segments = []
    for k_start in sorted(slice_pts.keys()):
        order_p = sorted(range(len(slice_pts[k_start])), key=lambda idx: -slice_pts[k_start][idx][5])
        for idx_p in order_p:
            if (k_start, idx_p) in used:
                continue
            pt0 = slice_pts[k_start][idx_p]
            chain = [pt0]
            used.add((k_start, idx_p))
            curr_x, curr_y, curr_z, _, _, _, curr_k = pt0
            vx_est, vy_est = 0.0, 0.0
            for k_next in range(k_start + 1, K):
                if k_next - curr_k > 3:
                    break
                if k_next not in slice_pts:
                    continue
                best_j = -1
                best_dist = 2.60 + 0.14 * abs(k_next - curr_k) * sp_k
                for j_p, cand in enumerate(slice_pts[k_next]):
                    if (k_next, j_p) in used:
                        continue
                    dz = cand[2] - curr_z
                    pred_x = curr_x + vx_est * dz
                    pred_y = curr_y + vy_est * dz
                    dist = math.hypot(cand[0] - pred_x, cand[1] - pred_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j_p
                if best_j >= 0:
                    chosen = slice_pts[k_next][best_j]
                    used.add((k_next, best_j))
                    chain.append(chosen)
                    if len(chain) >= 3:
                        dz_tot = chain[-1][2] - chain[0][2]
                        if abs(dz_tot) > 1.0:
                            vx_est = 0.55 * vx_est + 0.45 * ((chain[-1][0] - chain[0][0]) / dz_tot)
                            vy_est = 0.55 * vy_est + 0.45 * ((chain[-1][1] - chain[0][1]) / dz_tot)
                    curr_x, curr_y, curr_z, _, _, _, curr_k = chosen
            if len(chain) >= 2:
                arr = np.array([[c[0], c[1], c[2], c[3], c[4], c[5]] for c in chain], dtype=np.float64)
                arr = arr[np.argsort(arr[:, 2])]
                segments.append(arr)

    # Collinear gap bridging across high-uncertainty dropout slices
    segments.sort(key=lambda s: float(s[0, 2]))
    merged_any = True
    while merged_any and len(segments) >= 2:
        merged_any = False
        best_pair = None
        best_cost = 999.0
        for u in range(len(segments)):
            su = segments[u]
            z_top_u = float(su[-1, 2])
            vx_u, vy_u = _fit_seg_slope(su[-min(6, len(su)):, :3])
            for v in range(len(segments)):
                if u == v:
                    continue
                sv = segments[v]
                z_bot_v = float(sv[0, 2])
                gap = z_bot_v - z_top_u
                if not (-3.0 <= gap <= 16.0 and float(sv[-1, 2]) > z_top_u + 3.0 and float(su[0, 2]) < z_bot_v - 2.0):
                    continue
                vx_v, vy_v = _fit_seg_slope(sv[:min(6, len(sv)), :3])
                d_slope = math.hypot(vx_u - vx_v, vy_u - vy_v)
                if d_slope > 0.14:
                    continue
                z_mid = 0.5 * (z_top_u + z_bot_v)
                xu_m = float(su[-1, 0]) + vx_u * (z_mid - z_top_u)
                yu_m = float(su[-1, 1]) + vy_u * (z_mid - z_top_u)
                xv_m = float(sv[0, 0]) + vx_v * (z_mid - z_bot_v)
                yv_m = float(sv[0, 1]) + vy_v * (z_mid - z_bot_v)
                d_mid = math.hypot(xu_m - xv_m, yu_m - yv_m)
                if d_mid <= 2.25:
                    cost = d_mid + 4.0 * d_slope
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (u, v)
        if best_pair is not None:
            u, v = best_pair
            comb = np.vstack([segments[u], segments[v]])
            comb = comb[np.argsort(comb[:, 2])]
            new_segs = [segments[idx] for idx in range(len(segments)) if idx not in (u, v)]
            new_segs.append(comb)
            new_segs.sort(key=lambda s: float(s[0, 2]))
            segments = new_segs
            merged_any = True

    min_slices = 4 if sp_k >= 3.2 else (5 if sp_k >= 2.2 else 7)
    r_px = max(1, int(round(1.0 / sp_i)))
    trajs = []

    for arr in segments:
        if len(arr) < min_slices:
            continue
        z_span = float(arr[-1, 2] - arr[0, 2])
        if z_span < 14.0:
            continue

        mean_pm = float(np.mean(arr[:, 3]))
        mean_std = float(np.mean(arr[:, 4]))
        mean_conf = float(np.mean(arr[:, 5]))
        unc_ratio = mean_std / max(mean_pm, 1e-4)

        # Aleong et al. 2024 Ensemble-Disagreement Filtering:
        # Reject high-epistemic-uncertainty false positives
        if unc_ratio > 0.42 or mean_conf < 0.42 or mean_pm < 0.45:
            continue
        if float(arr[0, 2] - z_base_ras) > 28.0 and (mean_conf < 0.58 or z_span < 22.0):
            continue

        # Kim et al. 2026 Inverse-Variance Weighted 3rd-Order Polynomial Fitting:
        # w_k = p_mean(z_k) / (p_std(z_k)^2 + 0.02)
        w_k = arr[:, 3] / (arr[:, 4] ** 2 + 0.02)
        w_k = np.clip(w_k / max(np.mean(w_k), 1e-4), 0.2, 4.0)
        deg = 3 if len(arr) >= 9 else (2 if len(arr) >= 5 else 1)
        px = np.polyfit(arr[:, 2], arr[:, 0], deg, w=w_k)
        py = np.polyfit(arr[:, 2], arr[:, 1], deg, w=w_k)
        fx = np.polyval(px, arr[:, 2])
        fy = np.polyval(py, arr[:, 2])
        rmse = float(np.sqrt(np.average((arr[:, 0] - fx) ** 2 + (arr[:, 1] - fy) ** 2, weights=w_k)))
        if rmse > 1.35:
            continue
        sx = float(fx[-1] - fx[0]) / max(1.0, z_span)
        sy = float(fy[-1] - fy[0]) / max(1.0, z_span)
        if math.hypot(sx, sy) > 0.56:
            continue

        # Kim et al. 2026 Uncertainty-Driven Cranial Tip Refinement:
        # Probe along cranial tangent from z_top - 6 mm to z_top + 6 mm and locate last slice
        # where uncertainty-penalized support p_mean - 0.8 * p_std >= 0.28
        z_top_cur = float(arr[-1, 2])
        vx_top, vy_top = _fit_seg_slope(np.column_stack([fx, fy, arr[:, 2]])[-min(6, len(arr)):])
        z_probe = np.linspace(z_top_cur - 6.0, z_top_cur + 6.0, 25)
        x_probe = np.polyval(px, np.minimum(z_probe, z_top_cur)) + np.maximum(0.0, z_probe - z_top_cur) * vx_top
        y_probe = np.polyval(py, np.minimum(z_probe, z_top_cur)) + np.maximum(0.0, z_probe - z_top_cur) * vy_top
        ijks_p = (ras2ijk @ np.column_stack([x_probe, y_probe, z_probe, np.ones_like(z_probe)]).T).T
        sup_probe = np.zeros(len(z_probe), dtype=np.float32)
        for idx_p, row in enumerate(ijks_p):
            ic, jc, kc = int(round(row[0] - i0)), int(round(row[1] - j0)), int(round(row[2] - k0))
            if r_px <= ic < W - r_px and r_px <= jc < H - r_px and 0 <= kc < K:
                pm_v = float(np.max(p_mean[kc, jc - r_px:jc + r_px + 1, ic - r_px:ic + r_px + 1]))
                ps_v = float(np.mean(p_std[kc, jc - r_px:jc + r_px + 1, ic - r_px:ic + r_px + 1]))
                sup_probe[idx_p] = pm_v - 0.80 * ps_v
        valid_tip = np.where(sup_probe >= 0.28)[0]
        if len(valid_tip) > 0:
            z_top_refined = float(np.clip(z_probe[int(valid_tip[-1])], z_top_cur - 4.5, z_top_cur + 4.5))
        else:
            z_top_refined = z_top_cur

        z_bot = float(arr[0, 2])
        z_top_final = max(z_bot + 12.0, z_top_refined)
        z_span_final = z_top_final - z_bot
        n_out = max(len(arr), int(round(z_span_final / 2.0)) + 2)
        zs_out = np.linspace(z_bot, z_top_final, n_out)
        xs_out = np.where(zs_out <= z_top_cur, np.polyval(px, zs_out), np.polyval(px, z_top_cur) + vx_top * (zs_out - z_top_cur))
        ys_out = np.where(zs_out <= z_top_cur, np.polyval(py, zs_out), np.polyval(py, z_top_cur) + vy_top * (zs_out - z_top_cur))
        smooth_ras = np.column_stack([xs_out, ys_out, zs_out])
        is_long = (abs(z_bot - z_base_ras) <= 28.0) and (z_span_final >= 24.0)
        trajs.append({
            "ras": smooth_ras,
            "points": smooth_ras,
            "confidence": mean_conf * 100.0,
            "z_span": z_span_final,
            "is_long": is_long,
            "unc_ratio": unc_ratio,
        })

    if len(trajs) >= 4:
        all_xy = np.concatenate([t["ras"][:, :2] for t in trajs], axis=0)
        center_xy = np.median(all_xy, axis=0)
        rad_95 = float(np.percentile(np.linalg.norm(all_xy - center_xy[None, :], axis=1), 95.0)) + 15.5
        trajs = [
            t for t in trajs
            if float(np.linalg.norm(t["ras"][:, :2].mean(axis=0) - center_xy)) <= rad_95 or t["confidence"] >= 70.0
        ]

    trajs.sort(key=lambda t: -t["confidence"] * math.log1p(t["z_span"]))
    dedup = []
    for t in trajs:
        dup = False
        for ex in dedup:
            z0 = max(float(t["ras"][0, 2]), float(ex["ras"][0, 2]))
            z1 = min(float(t["ras"][-1, 2]), float(ex["ras"][-1, 2]))
            if z1 > z0 + 2.0:
                zs = np.linspace(z0, z1, 10)
                d = float(np.mean(np.hypot(
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 0]) - np.interp(zs, ex["ras"][:, 2], ex["ras"][:, 0]),
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 1]) - np.interp(zs, ex["ras"][:, 2], ex["ras"][:, 1]),
                )))
                if d < 2.0:
                    dup = True
                    break
        if not dup:
            dedup.append(t)
    return dedup


# ==============================================================================
# TRAINING WORKERS FOR DAI 2020 AND KOSTOULAS 2025 / ALEONG 2024 / KIM 2026
# ==============================================================================

def fast_val_dice_gpu(model, xv_cpu, yv_cpu, device):
    _, D, H, W = xv_cpu.shape
    d0 = max(0, (D - 36) // 2)
    d1 = min(D, d0 + 36)
    h0 = max(0, (H - 224) // 2)
    h1 = min(H, h0 + 224)
    w0 = max(0, (W - 224) // 2)
    w1 = min(W, w0 + 224)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        sub_x = xv_cpu[:, d0:d1, h0:h1, w0:w1].unsqueeze(0).to(device)
        sub_y = yv_cpu[0, d0:d1, h0:h1, w0:w1].to(device).float()
        pm = torch.sigmoid(model(sub_x)[0, 0].float())
        inter = float((pm * sub_y).sum().item())
        union = float((pm.sum() + sub_y.sum()).item())
    return (2.0 * inter + 1.0) / (union + 1.0)


def train_dai2020_fold(f_out, items_1ch, device, epochs=20):
    t0 = time.perf_counter()
    outer_test, outer_train, inner_subtrain, inner_val = get_nested_fold_split(f_out, 54)
    ckpt_path = os.path.join(OUT_DIR, f"dai2020_attunet_tv_outer{f_out}.pt")

    if os.path.exists(ckpt_path):
        best_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        random.seed(2020 + f_out)
        np.random.seed(2020 + f_out)
        torch.manual_seed(2020 + f_out)

        model = Dai2020AttentionUNet3D(in_ch=1, base_ch=24).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        best_val_dice = -1.0
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            order = list(inner_subtrain)
            random.shuffle(order)
            for idx in order:
                x, y, _ = items_1ch[idx]
                xa, ya = augment_1ch_gpu(x, y, device, max_d=36, max_hw=168)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_tuple = model(xa.unsqueeze(0), return_deep_sup=True)
                    loss = compute_dai2020_deepsup_tv_loss(out_tuple, ya.unsqueeze(0), lambda_tv=0.18)
                loss.backward()
                opt.step()
            sched.step()

            if ep in (12, 16, epochs):
                model.eval()
                val_dices = [fast_val_dice_gpu(model, items_1ch[vi][0], items_1ch[vi][1], device) for vi in inner_val]
                md = float(np.mean(val_dices))
                if md > best_val_dice:
                    best_val_dice = md
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, ckpt_path)
        print(f"  [Dai 2020 AttUNet+TV | Outer {f_out}] Best InnerVal Dice={best_val_dice:.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)

    model = Dai2020AttentionUNet3D(in_ch=1, base_ch=24).to(device)
    model.load_state_dict(best_state, strict=True)
    model.eval()

    # Select optimal probability threshold strictly on inner_val
    best_th = 0.38
    best_val_f1 = -1.0
    val_pm = [infer_sliding_1ch_tta(model, items_1ch[vi][0], device, chunk_d=40, use_tta=False) for vi in inner_val]
    val_gts = [items_1ch[vi][2]["gts_all"] for vi in inner_val]
    for th in [0.36, 0.40, 0.44]:
        v_preds = [extract_dai2020_trajectories(val_pm[k], items_1ch[inner_val[k]][2], prob_thresh=th) for k in range(len(inner_val))]
        res_v, _ = evaluate_method_unified(v_preds, val_gts, thresh=3.5, n_boot=5)
        if res_v["f1_1to1_pct"] > best_val_f1:
            best_val_f1 = res_v["f1_1to1_pct"]
            best_th = th

    out_fold = {}
    for ti in outer_test:
        xv, yv, meta_d = items_1ch[ti]
        pm = infer_sliding_1ch_tta(model, xv, device, chunk_d=40, use_tta=True)
        tm = yv[0].numpy().astype(np.float32)
        dice = float((2.0 * (pm * tm).sum() + 1.0) / (pm.sum() + tm.sum() + 1.0))
        trajs = extract_dai2020_trajectories(pm, meta_d, prob_thresh=best_th)
        out_fold[ti] = {
            "dice": dice,
            "best_th": best_th,
            "all": trajs,
            "long": [t for t in trajs if t["is_long"]],
        }
    return out_fold


def train_resunet_inner_ensemble_fold(f_out, items_1ch, device, epochs=18):
    """
    Trains all 5 inner-fold 1-channel 3D Residual U-Nets (`m = 0..4`) inside outer fold `f_out`
    for:
      - Kostoulas et al. 2025 (Proc. SPIE 13408) Error-Tolerant Post-Processing
      - Aleong et al. 2024 / Kim et al. 2026 5-Model Epistemic Uncertainty Ensemble & Inverse-Variance Poly3
    """
    t0 = time.perf_counter()
    outer_test = [i for i in range(54) if i % 5 == f_out]
    outer_train = [i for i in range(54) if i % 5 != f_out]

    inner_models = []
    for m in range(5):
        inner_val_m = outer_train[m::5]
        inner_sub_m = [i for i in outer_train if i not in inner_val_m]
        ckpt_path = os.path.join(OUT_DIR, f"resunet1ch_outer{f_out}_inner{m}.pt")

        if os.path.exists(ckpt_path):
            best_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        else:
            seed = 2025 + f_out * 10 + m
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            model = NativeResUNet3D_MaskOnly(in_ch=1, base_ch=24).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

            best_val_dice = -1.0
            best_state = None

            for ep in range(1, epochs + 1):
                model.train()
                order = list(inner_sub_m)
                random.shuffle(order)
                for idx in order:
                    x, y, _ = items_1ch[idx]
                    xa, ya = augment_1ch_gpu(x, y, device, max_d=36, max_hw=168)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model(xa.unsqueeze(0))
                        loss = compute_single_scale_dice_bce(out, ya.unsqueeze(0))
                    loss.backward()
                    opt.step()
                sched.step()

                if ep in (12, epochs):
                    model.eval()
                    val_dices = [fast_val_dice_gpu(model, items_1ch[vi][0], items_1ch[vi][1], device) for vi in inner_val_m]
                    md = float(np.mean(val_dices))
                    if md > best_val_dice:
                        best_val_dice = md
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            print(f"  [ResUNet1ch | Outer {f_out} Inner {m}] Best InnerVal Dice={best_val_dice:.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)

        mod = NativeResUNet3D_MaskOnly(in_ch=1, base_ch=24).to(device)
        mod.load_state_dict(best_state, strict=True)
        mod.eval()
        inner_models.append(mod)

    # Select threshold on inner_val_0 strictly without touching outer_test
    inner_val_0 = outer_train[0::5]
    val_pm_0 = [infer_sliding_1ch_tta(inner_models[0], items_1ch[vi][0], device, chunk_d=40, use_tta=False) for vi in inner_val_0]
    val_gts_0 = [items_1ch[vi][2]["gts_all"] for vi in inner_val_0]
    best_th_kost = 0.40
    best_f1_kost = -1.0
    for th in [0.36, 0.40, 0.44]:
        vp = [extract_kostoulas2025_trajectories(val_pm_0[k], items_1ch[inner_val_0[k]][2], prob_thresh=th) for k in range(len(inner_val_0))]
        rv, _ = evaluate_method_unified(vp, val_gts_0, thresh=3.5, n_boot=5)
        if rv["f1_1to1_pct"] > best_f1_kost:
            best_f1_kost = rv["f1_1to1_pct"]
            best_th_kost = th

    out_kost = {}
    out_aleong = {}
    for ti in outer_test:
        xv, yv, meta_d = items_1ch[ti]
        tm = yv[0].numpy().astype(np.float32)

        # Evaluate all 5 inner models with 4-view TTA on outer_test volume ti
        p_models = [infer_sliding_1ch_tta(inner_models[m], xv, device, chunk_d=40, use_tta=True) for m in range(5)]
        p_stack = np.stack(p_models, axis=0)  # [5, K, H, W]
        p_mean = np.mean(p_stack, axis=0)
        p_std = np.std(p_stack, axis=0)

        # Kostoulas et al. 2025 (on single nested model p_models[0] AND on p_mean)
        p_kost_in = p_models[0]
        dice_kost = float((2.0 * (p_kost_in * tm).sum() + 1.0) / (p_kost_in.sum() + tm.sum() + 1.0))
        trajs_kost = extract_kostoulas2025_trajectories(p_kost_in, meta_d, prob_thresh=best_th_kost)
        out_kost[ti] = {
            "dice": dice_kost,
            "best_th": best_th_kost,
            "all": trajs_kost,
            "long": [t for t in trajs_kost if t["is_long"]],
        }

        # Aleong et al. 2024 / Kim et al. 2026 (5-model ensemble p_mean & epistemic uncertainty p_std)
        dice_ens = float((2.0 * (p_mean * tm).sum() + 1.0) / (p_mean.sum() + tm.sum() + 1.0))
        trajs_aleong = extract_aleong2024_kim2026_trajectories(p_mean, p_std, meta_d, conf_thresh=max(0.34, best_th_kost - 0.03))
        out_aleong[ti] = {
            "dice": dice_ens,
            "all": trajs_aleong,
            "long": [t for t in trajs_aleong if t["is_long"]],
        }

    return out_kost, out_aleong


def load_items_1ch():
    meta_list = pickle.load(open(CLEAN_META_PATH, "rb"))
    items_1ch = []
    for i in range(54):
        d = torch.load(os.path.join(CLEAN_CACHE_DIR, f"case_{i:02d}.pt"), map_location="cpu", weights_only=False)
        x_mri = d["X"][0:1].float()
        y_mask = d["Y"][0:1].float()
        sp_i, sp_j, sp_k = [float(v) for v in meta_list[i]["sp_ras"]]
        _, K, H, W = x_mri.shape
        meta_d = {
            "idx": i,
            "k0": int(d["k0"]),
            "i0": int(d["i0"]),
            "j0": int(d["j0"]),
            "K": K, "H": H, "W": W,
            "sp_ras": (sp_i, sp_j, sp_k),
            "ijk_to_ras": np.asarray(meta_list[i]["ijk_to_ras"], dtype=np.float64),
            "gts_all": get_gts(meta_list[i]["gts_all"]),
            "gts_long": get_gts(meta_list[i]["gts_long"]),
        }
        items_1ch.append((x_mri, y_mask, meta_d))
    return items_1ch, meta_list


def worker_gpu(worker_id, gpu_id, fold_list, return_dict):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    print(f"[Worker {worker_id} on GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})] Assigned folds: {fold_list}", flush=True)

    items_1ch, _ = load_items_1ch()
    dai_res = {}
    kost_res = {}
    aleong_res = {}

    for f_out in fold_list:
        fold_cache = os.path.join(OUT_DIR, f"part1_fold_{f_out}.pkl")
        if os.path.exists(fold_cache):
            c_f = pickle.load(open(fold_cache, "rb"))
            dai_res.update(c_f["dai"])
            kost_res.update(c_f["kost"])
            aleong_res.update(c_f["aleong"])
            print(f"  [Worker {worker_id} | Outer {f_out}] Loaded cached fold results from {fold_cache}", flush=True)
            continue
        d_f = train_dai2020_fold(f_out, items_1ch, device, epochs=22)
        k_f, a_f = train_resunet_inner_ensemble_fold(f_out, items_1ch, device, epochs=20)
        pickle.dump({"dai": d_f, "kost": k_f, "aleong": a_f}, open(fold_cache, "wb"))
        dai_res.update(d_f)
        kost_res.update(k_f)
        aleong_res.update(a_f)

    return_dict[f"dai_{worker_id}"] = dai_res
    return_dict[f"kost_{worker_id}"] = kost_res
    return_dict[f"aleong_{worker_id}"] = aleong_res


def main():
    t0 = time.perf_counter()
    return_dict = {}
    worker_gpu(0, 0, [0, 1, 2, 3, 4], return_dict)

    dai_all_map = return_dict["dai_0"]
    kost_all_map = return_dict["kost_0"]
    aleong_all_map = return_dict["aleong_0"]

    dai_trajs_all = [dai_all_map[i]["all"] for i in range(54)]
    dai_trajs_long = [dai_all_map[i]["long"] for i in range(54)]
    dai_dices = [dai_all_map[i]["dice"] for i in range(54)]

    kost_trajs_all = [kost_all_map[i]["all"] for i in range(54)]
    kost_trajs_long = [kost_all_map[i]["long"] for i in range(54)]
    kost_dices = [kost_all_map[i]["dice"] for i in range(54)]

    aleong_trajs_all = [aleong_all_map[i]["all"] for i in range(54)]
    aleong_trajs_long = [aleong_all_map[i]["long"] for i in range(54)]
    aleong_dices = [aleong_all_map[i]["dice"] for i in range(54)]

    pickle.dump(
        {"all": dai_trajs_all, "long": dai_trajs_long, "mean_oof_dice": float(np.mean(dai_dices))},
        open(os.path.join(OUT_DIR, "oof_trajs_dai2020_attunet_tv.pkl"), "wb"),
    )
    pickle.dump(
        {"all": kost_trajs_all, "long": kost_trajs_long, "mean_oof_dice": float(np.mean(kost_dices))},
        open(os.path.join(OUT_DIR, "oof_trajs_kostoulas2025_gapbridge.pkl"), "wb"),
    )
    pickle.dump(
        {"all": aleong_trajs_all, "long": aleong_trajs_long, "mean_oof_dice": float(np.mean(aleong_dices))},
        open(os.path.join(OUT_DIR, "oof_trajs_aleong2024_kim2026_unc_ensemble.pkl"), "wb"),
    )

    # Evaluate on All 1,063, Long 759, and Severe Adhesion 23 cases (571 catheters)
    meta_list = pickle.load(open(CLEAN_META_PATH, "rb"))
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]
    gts_short_list = [
        [ga for ga in gts_all_list[ci] if not any(traj_gt_dist(ga, gl) < 2.0 for gl in gts_long_list[ci])]
        for ci in range(54)
    ]
    adh_info = analyze_catheter_adhesion(meta_list, gts_all_list, gts_long_list, adhesion_thresh_mm=3.0)
    adh_idx = adh_info["adhesion_case_indices"]
    gts_all_adh = [gts_all_list[i] for i in adh_idx]

    summary_part1 = {}
    for name, t_all, t_long, dices in [
        ("Baseline 4: Dai 2020 Deeply-Supervised 3D Attention U-Net + Axial TV + Poly3", dai_trajs_all, dai_trajs_long, dai_dices),
        ("Baseline 5: Kostoulas 2025 3D ResUNet + Iterative Gap-Bridging & False-Part Pruning", kost_trajs_all, kost_trajs_long, kost_dices),
        ("Baseline 6: Aleong 2024 / Kim 2026 5-Model Uncertainty Ensemble + Weighted Poly3", aleong_trajs_all, aleong_trajs_long, aleong_dices),
    ]:
        res_all, _ = evaluate_method_unified(t_all, gts_all_list, thresh=3.5, n_boot=300)
        res_long, _ = evaluate_method_unified(t_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=300)
        res_adh, _ = evaluate_method_unified([t_all[i] for i in adh_idx], gts_all_adh, thresh=3.5, n_boot=300)
        summary_part1[name] = {
            "mean_oof_voxel_dice": round(float(np.mean(dices)), 4),
            "cohort_all_1063": res_all,
            "cohort_long_759": res_long,
            "severe_adhesion_571": res_adh,
        }
        print(
            f"\n[{name}]\n"
            f"  Voxel Dice = {np.mean(dices):.4f}\n"
            f"  All (1,063) : F1={res_all['f1_1to1_pct']:.2f}% (Rec={res_all['hungarian_1to1_recall_pct']:.2f}%, Prec={res_all['precision_1to1_pct']:.2f}%) | "
            f"Shaft={res_all['metrics']['mean_shaft_xy']['mean']:.2f}/{res_all['metrics']['mean_shaft_xy']['median']:.2f} mm | "
            f"Tip={res_all['metrics']['tip_3d']['mean']:.2f}/{res_all['metrics']['tip_3d']['median']:.2f} mm\n"
            f"  Long (759)  : F1_adj={res_long['f1_1to1_obturator_adj_pct']:.2f}% (F1_raw={res_long['f1_1to1_pct']:.2f}%, Rec={res_long['hungarian_1to1_recall_pct']:.2f}%)\n"
            f"  Adh (571)   : F1={res_adh['f1_1to1_pct']:.2f}% (Rec={res_adh['hungarian_1to1_recall_pct']:.2f}%, Prec={res_adh['precision_1to1_pct']:.2f}%)",
            flush=True,
        )

    out_json = "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean/part1_peer_baselines_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary_part1, f, indent=2)
    print(f"\n[Part 1 Done in {time.perf_counter()-t0:.1f}s] Saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
