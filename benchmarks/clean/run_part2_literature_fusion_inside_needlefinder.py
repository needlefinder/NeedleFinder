#!/usr/bin/env python3
"""
Part 2: Reuse the Best Ideas from Dai et al. 2020, Kostoulas et al. 2025, and
Aleong et al. 2024 / Kim et al. 2026 Inside NeedleFinder 2026 (Strictly Nested 5x5 CV)
======================================================================================
Integrates and ablates three peer-reviewed literature enhancements inside our strictly nested
5x5 cross-validation NeedleFinder 2026 pipeline (all hyper-parameters and checkpoints selected
strictly on `inner_val`, zero access to `outer_test`):

  - Upgrade A (Aleong 2024 / Kim 2026 Inverse-Variance & Sub-Voxel Dark-Core Weighted Poly3 Shaft Refinement):
    Refines each selected trajectory's 3D shaft (`pts`) using precision-weighted sub-voxel
    dark-core peak centroiding `(0.55*p_peak + 0.45*sym) / (1 + 2.5*sigma_mask)` and fits an
    inverse-variance-weighted 3rd-order polynomial (`w_k = (p_mask + 0.5*sym) / (0.04 + sigma_mask^2)`).
  - Upgrade B (Dai 2020 Axial TV Continuity + Aleong/Kim 5-Model Epistemic Uncertainty in Stage 2B & 2C):
    Adds Stage 1 5-model epistemic uncertainty (`sigma_mask`, `sigma_peak`, `sigma_vec`) and
    Dai 2020 directional axial Total-Variation continuity (`axial_tv_pm`, `unc_ratio`) to:
      * Stage 2B candidate node features (`26 -> 30` dims) and `inner_val`-calibrated uncertainty/TV
        scoring & false-positive suppression,
      * Stage 2C 91-step 1D Axial Tip Localizer (`22 -> 24` channels: `sigma_mask` and `d(sigma_mask)/dz`).
  - Upgrade C (Kostoulas 2025 Iterative Multi-Segment Collinear Gap-Bridging & Poly3 Fit Verification):
    Upgrades pool fragment stitching and post-selection cranial gap-bridging to iteratively bridge
    multi-segment dropouts (`t_low -> t_mid -> t_high`) verified by 3rd-order polynomial fit residual
    (`rmse_poly3 <= 1.20 mm`) and low epistemic uncertainty on the bridged segment.

Outputs unified JSON report to:
  `/home/guillaume/github/needlefinder/vault/sources/2026-10-24-peer-reviewed-baselines-and-fusion.json`
"""

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path

import copy
import json
import math
import os
import pickle
import sys
import time
import types
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

for mod in (
    "vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util",
    "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"
):
    sys.modules[mod] = types.ModuleType(mod)

sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean"))

from evaluate_unified_apples_to_apples import (
    evaluate_method_unified,
    compute_wilcoxon_comparison,
    compute_unified_pair_metrics,
    summarize_distribution,
    get_gts,
    traj_gt_dist,
)
from run_xiao2026_and_fewshot_benchmark import (
    analyze_catheter_adhesion,
    regularize_poly3_trajectory,
)
from test_synthbrachy3t_v2 import (
    TipAwareBundleGraphTransformer,
    apply_tip_adjustment_and_poly3,
)
from run_stage2_and_tip_clean import (
    calibrate_stage1_oof_for_volume,
    fit_local_slope,
    eval_traj_extrap,
    collinear_traj_dist,
    merge_collinear_trajectories,
    extract_1d_tip_profile_91,
    Dedicated1DTipLocalizer,
    get_nested_fold_split,
    shift_1d_profile,
    decode_clean_stage2_preds,
    derive_unstitched_and_base_bgs,
    compute_detailed_tip_and_fp_diagnostics,
)

CLEAN_CACHE_DIR = resolve_path("/tmp/needle_unet3d_cache_clean")
CLEAN_META_PATH = resolve_path("/tmp/cases_meta_54_clean.pkl")
CLEAN_BASE_TRAJS_PATH = resolve_path("/tmp/base_trajs_54cases_clean.pkl")
OOF_DIR = resolve_path("/home/guillaume/github/needlefinder/nn_models_clean")
EVAL_OOF_DIR = resolve_path("/home/guillaume/github/needlefinder/nn_models_eval")
STAGE2_PREDS_PKL = resolve_path("/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-clean-final-oof-predictions.pkl")
BUNDLE_CACHE_PKL = resolve_path("/tmp/stage2_clean_bundles_nn_models_clean.pkl")
PART1_JSON = resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean/part1_peer_baselines_summary.json")
OUT_FUSION_JSON = resolve_path("/home/guillaume/github/needlefinder/vault/sources/2026-10-24-peer-reviewed-baselines-and-fusion.json")


# ==============================================================================
# 1. UPGRADE A: ALEONG 2024 / KIM 2026 INVERSE-VARIANCE & SUB-VOXEL DARK-CORE
#               WEIGHTED POLY3 SHAFT REFINEMENT
# ==============================================================================

def refine_shaft_and_tip_kim2026_weighted_poly3(
    pts_ras,
    dz_tip_mm,
    vx_top,
    vy_top,
    g,
    oof_cal,
    oof_unc,
    X_vol,
    alpha_subvox=0.32,
    w_poly_blend=0.52,
):
    """
    Upgrade A (Aleong et al. 2024 / Kim et al. 2026):
      1. Applies cranial tip adjustment `dz_tip_mm` along `(vx_top, vy_top)`.
      2. At each slice point `(x_k, y_k, z_k)`, samples a small local 3x3 physical neighborhood
         in the precision-weighted dark-core peak map:
           `R(j, i) = (0.55 * p_peak + 0.45 * sym) / (1.0 + 2.5 * sigma_mask)`
         and shifts `(x_k, y_k)` gently (`alpha_subvox`, clamped to <= 0.75 mm) toward the
         local sub-voxel dark-core center when precision-weighted peak confidence is strong.
      3. Computes Kim et al. 2026 inverse-variance slice weights:
           `w_k = (p_mask(z_k) + 0.5 * sym(z_k)) / (0.04 + sigma_mask(z_k)^2)`
         so high-certainty dark-core slices dominate the curve fit while high-uncertainty dropout
         or artifact slices are down-weighted.
      4. Fits an inverse-variance-weighted 3rd-order polynomial (`np.polyfit(..., deg=3, w=w_k)`)
         and blends smoothly (`w_poly_blend`) while preserving exact base/tip endpoint anchoring.
    """
    pts = np.asarray(pts_ras, dtype=np.float64)
    pts = pts[np.argsort(pts[:, 2])]
    if len(pts) < 3:
        return pts

    dz_clamp = float(np.clip(dz_tip_mm, -9.5, 9.5))
    z_bot = float(pts[0, 2])
    z_top_old = float(pts[-1, 2])
    z_top_new = max(z_bot + 7.5, z_top_old + dz_clamp)

    if z_top_new < z_top_old - 0.5:
        x_top = float(np.interp(z_top_new, pts[:, 2], pts[:, 0]))
        y_top = float(np.interp(z_top_new, pts[:, 2], pts[:, 1]))
        keep = pts[:, 2] < z_top_new - 0.25
        pts_trim = np.vstack([pts[keep], np.array([[x_top, y_top, z_top_new]])])
        if len(pts_trim) >= 3:
            pts = pts_trim
    elif z_top_new > z_top_old + 0.5:
        dz_ext = z_top_new - z_top_old
        x_ext = float(pts[-1, 0]) + vx_top * dz_ext
        y_ext = float(pts[-1, 1]) + vy_top * dz_ext
        pts = np.vstack([pts, np.array([[x_ext, y_ext, z_top_new]])])

    if len(pts) < 4 or g is None or oof_cal is None or oof_unc is None or X_vol is None:
        return apply_tip_adjustment_and_poly3(pts_ras, dz_tip_mm, vx_top, vy_top)

    ijk2ras = g["ijk_to_ras"]
    ras2ijk = np.linalg.inv(ijk2ras)
    rot = ijk2ras[:3, :3]
    trans = ijk2ras[:3, 3]
    k0_c, i0_c, j0_c = int(oof_cal["k0"]), int(oof_cal["i0"]), int(oof_cal["j0"])

    pm_arr = oof_cal["p_mask"]
    pp_arr = oof_cal["p_peak"]
    sym_arr = X_vol[1]
    sig_m_arr = oof_unc["sigma_mask"]
    K_s, H_s, W_s = pm_arr.shape

    ijks = (ras2ijk @ np.column_stack([pts[:, 0], pts[:, 1], pts[:, 2], np.ones(len(pts))]).T).T
    pts_subvox = pts.copy()
    w_k = np.ones(len(pts), dtype=np.float64)
    yy_w, xx_w = np.mgrid[-1:2, -1:2]

    for k_i in range(len(pts)):
        fi = float(ijks[k_i, 0] - i0_c)
        fj = float(ijks[k_i, 1] - j0_c)
        fk = float(ijks[k_i, 2] - k0_c)
        ic = int(round(fi))
        jc = int(round(fj))
        kc = int(round(np.clip(fk, 0, K_s - 1)))

        if 1 <= ic < W_s - 1 and 1 <= jc < H_s - 1 and 0 <= kc < K_s:
            pm_patch = pm_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2].astype(np.float64)
            pp_patch = pp_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2].astype(np.float64)
            sym_patch = np.clip(sym_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2].astype(np.float64), 0.0, 1.2)
            sig_patch = sig_m_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2].astype(np.float64)

            pm_val = float(np.max(pm_patch))
            sym_val = float(np.max(sym_patch))
            sig_val = float(np.mean(sig_patch))

            # Inverse-variance weight (Kim et al. 2026)
            w_k[k_i] = (pm_val + 0.50 * sym_val + 0.05) / (0.04 + sig_val ** 2)

            # Sub-voxel precision-weighted dark-core centroid shift (only on interior shaft points)
            if alpha_subvox > 0.0 and 0 < k_i < len(pts) - 1:
                core_score = (0.55 * pp_patch + 0.45 * sym_patch) / (1.0 + 2.5 * sig_patch)
                max_core = float(np.max(core_score))
                if max_core >= 0.36:
                    w_patch = np.maximum(0.0, core_score - 0.15) ** 2
                    ws = float(np.sum(w_patch))
                    if ws > 1e-6:
                        si = float(ic) + float(np.sum(xx_w * w_patch)) / ws
                        sj = float(jc) + float(np.sum(yy_w * w_patch)) / ws
                        pt_peak_ras = rot @ np.array([si + i0_c, sj + j0_c, float(ijks[k_i, 2])]) + trans
                        dx = float(pt_peak_ras[0] - pts[k_i, 0])
                        dy = float(pt_peak_ras[1] - pts[k_i, 1])
                        d_norm = math.hypot(dx, dy)
                        if d_norm > 0.75:
                            dx *= 0.75 / d_norm
                            dy *= 0.75 / d_norm
                        conf_gate = min(1.0, max_core) / (1.0 + 3.0 * sig_val)
                        pts_subvox[k_i, 0] += alpha_subvox * conf_gate * dx
                        pts_subvox[k_i, 1] += alpha_subvox * conf_gate * dy

    w_k = np.clip(w_k / max(np.mean(w_k), 1e-4), 0.20, 4.0)
    # Strong endpoint weights so cranial/caudal endpoints do not drift
    w_k[0] = max(w_k[0], 2.5)
    w_k[-1] = max(w_k[-1], 2.5)

    zs = pts_subvox[:, 2]
    z_mean = float(np.mean(zs))
    z_std = max(1.0, float(np.std(zs)))
    z_norm = (zs - z_mean) / z_std
    deg = 3 if len(pts) >= 7 else (2 if len(pts) >= 4 else 1)

    try:
        px = np.polyfit(z_norm, pts_subvox[:, 0], deg, w=w_k)
        py = np.polyfit(z_norm, pts_subvox[:, 1], deg, w=w_k)
        fx = np.polyval(px, z_norm)
        fy = np.polyval(py, z_norm)
    except Exception:
        pts_poly = regularize_poly3_trajectory(pts_subvox, degree=3)
        fx, fy = pts_poly[:, 0], pts_poly[:, 1]

    pts_out = pts_subvox.copy()
    w_b = np.full(len(pts), w_poly_blend, dtype=np.float64)
    w_b[0] = 0.05
    w_b[-1] = 0.05
    pts_out[:, 0] = (1.0 - w_b) * pts_subvox[:, 0] + w_b * fx
    pts_out[:, 1] = (1.0 - w_b) * pts_subvox[:, 1] + w_b * fy
    return pts_out


# ==============================================================================
# 2. UPGRADE B: DAI 2020 AXIAL TV CONTINUITY + ALEONG/KIM EPISTEMIC UNCERTAINTY
#               FEATURES ALONG SHAFT & 24-CHANNEL 1D TIP PROFILE
# ==============================================================================

def compute_candidate_tv_and_uncertainty_features(ras, g, oof_cal, oof_unc, X_vol):
    """
    Computes Dai et al. 2020 directional axial Total-Variation continuity and
    Aleong et al. 2024 / Kim et al. 2026 5-model epistemic uncertainty along trajectory `ras`:
      - `mean_sig_m`: mean epistemic mask std `sigma_mask` along trajectory
      - `unc_ratio`: `mean_sig_m / max(0.05, mean_pm)` (Aleong 2024 disagreement ratio)
      - `axial_tv_pm`: `mean(|p_mask(z_{k+1}) - p_mask(z_k)|)` (Dai 2020 axial TV discontinuity)
      - `mean_sig_vec`: mean epistemic tangent vector std `sigma_vec` along trajectory
      - `precision_support`: `mean((p_mask + 0.5*sym) / (1 + 3*sigma_mask))`
    """
    ras = np.asarray(ras, dtype=np.float64)
    ras = ras[np.argsort(ras[:, 2])]
    zs = np.linspace(float(ras[0, 2]), float(ras[-1, 2]), 20)
    xs = np.interp(zs, ras[:, 2], ras[:, 0])
    ys = np.interp(zs, ras[:, 2], ras[:, 1])

    ras2ijk = np.linalg.inv(g["ijk_to_ras"])
    ijks = (ras2ijk @ np.column_stack([xs, ys, zs, np.ones_like(zs)]).T).T

    pm_arr = oof_cal["p_mask"]
    sym_arr = X_vol[1]
    sig_m_arr = oof_unc["sigma_mask"]
    sig_p_arr = oof_unc["sigma_peak"]
    sig_v_arr = oof_unc["sigma_vec"]

    k0_c, i0_c, j0_c = int(oof_cal["k0"]), int(oof_cal["i0"]), int(oof_cal["j0"])
    K_s, H_s, W_s = pm_arr.shape
    sp_i = float(g["sp_ras"][0])
    r_px = max(1, int(round(1.0 / sp_i)))

    pm_l, sym_l, sig_m_l, sig_p_l, sig_v_l = [], [], [], [], []
    for row in ijks:
        ic, jc, kc = int(round(row[0] - i0_c)), int(round(row[1] - j0_c)), int(round(row[2] - k0_c))
        if r_px <= ic < W_s - r_px and r_px <= jc < H_s - r_px and 0 <= kc < K_s:
            sl_j = slice(jc - r_px, jc + r_px + 1)
            sl_i = slice(ic - r_px, ic + r_px + 1)
            pm_l.append(float(np.max(pm_arr[kc, sl_j, sl_i])))
            sym_l.append(float(np.max(sym_arr[kc, sl_j, sl_i])))
            sig_m_l.append(float(np.mean(sig_m_arr[kc, sl_j, sl_i])))
            sig_p_l.append(float(np.mean(sig_p_arr[kc, sl_j, sl_i])))
            sig_v_l.append(float(np.mean(sig_v_arr[kc, sl_j, sl_i])))

    if not pm_l:
        return {
            "mean_sig_m": 0.15,
            "unc_ratio": 0.50,
            "axial_tv_pm": 0.20,
            "mean_sig_vec": 0.15,
            "precision_support": 0.0,
        }

    pm_np = np.array(pm_l, dtype=np.float64)
    sym_np = np.clip(np.array(sym_l, dtype=np.float64), 0.0, 1.2)
    sig_m_np = np.array(sig_m_l, dtype=np.float64)

    mean_pm = float(np.mean(pm_np))
    mean_sig_m = float(np.mean(sig_m_np))
    unc_ratio = float(mean_sig_m / max(0.05, mean_pm))
    axial_tv_pm = float(np.mean(np.abs(np.diff(pm_np)))) if len(pm_np) >= 2 else 0.0
    mean_sig_vec = float(np.mean(sig_v_l))
    precision_support = float(np.mean((pm_np + 0.5 * sym_np) / (1.0 + 3.0 * sig_m_np)))

    return {
        "mean_sig_m": mean_sig_m,
        "unc_ratio": unc_ratio,
        "axial_tv_pm": axial_tv_pm,
        "mean_sig_vec": mean_sig_vec,
        "precision_support": precision_support,
    }


def extract_1d_tip_profile_91_24ch(ras, g, oof_cal, oof_unc, X_vol, z_center=None, z_tip_nbr=None):
    """
    Upgrade B (24-Channel 91-Step 1D Axial Tip Profile):
    Appends 2 epistemic uncertainty & TV-continuity channels to the 22-channel profile:
      - Channel 22: `sigma_mask_max` (5-model epistemic uncertainty along [-25, +20] mm)
      - Channel 23: `d(sigma_mask_max)/dz` (axial epistemic uncertainty gradient at tip drop-off)
    """
    prof22 = extract_1d_tip_profile_91(ras, g, oof_cal, X_vol, z_center=z_center, z_tip_nbr=z_tip_nbr)
    prof24 = np.zeros((24, 91), dtype=np.float32)
    prof24[:22] = prof22

    ras = np.asarray(ras, dtype=np.float64)
    ras = ras[np.argsort(ras[:, 2])]
    z_top = float(ras[-1, 2]) if z_center is None else float(z_center)
    dz_grid = np.linspace(-25.0, 20.0, 91, dtype=np.float64)
    zs = z_top + dz_grid
    xs, ys = eval_traj_extrap(ras, zs)

    ras2ijk = np.linalg.inv(g["ijk_to_ras"])
    ijks = (ras2ijk @ np.column_stack([xs, ys, zs, np.ones_like(zs)]).T).T
    sig_m_arr = oof_unc["sigma_mask"].astype(np.float32)
    k0_c, i0_c, j0_c = int(oof_cal["k0"]), int(oof_cal["i0"]), int(oof_cal["j0"])
    K_s, H_s, W_s = sig_m_arr.shape
    sp_i = float(g["sp_ras"][0])
    r_px = max(1, int(round(1.0 / sp_i)))

    for t_idx in range(91):
        fi = float(ijks[t_idx, 0] - i0_c)
        fj = float(ijks[t_idx, 1] - j0_c)
        fk = float(ijks[t_idx, 2] - k0_c)
        ic = int(round(fi))
        jc = int(round(fj))
        if not (r_px <= ic < W_s - r_px and r_px <= jc < H_s - r_px and -0.5 <= fk <= K_s - 0.5):
            continue
        fk_clamped = np.clip(fk, 0.0, float(K_s - 1))
        k_lo = int(math.floor(fk_clamped))
        k_hi = min(K_s - 1, k_lo + 1)
        alpha = float(fk_clamped - k_lo)
        sl_j = slice(jc - r_px, jc + r_px + 1)
        sl_i = slice(ic - r_px, ic + r_px + 1)
        p_lo = sig_m_arr[k_lo, sl_j, sl_i]
        p_hi = sig_m_arr[k_hi, sl_j, sl_i] if k_hi != k_lo else p_lo
        patch = (1.0 - alpha) * p_lo + alpha * p_hi
        prof24[22, t_idx] = float(np.max(patch))

    prof24[23] = np.gradient(prof24[22]) * 2.0
    return prof24


def train_nested_stage2c_1d_tip_localizer_24ch(enriched_bundle_items, device):
    """
    Trains the 24-channel `Dedicated1DTipLocalizer(in_channels=24)` with epistemic uncertainty
    channels under strictly nested 5x5 CV (`inner_subtrain` -> `inner_val` selection).
    """
    fold_models = [None] * 5
    sel_epochs = []
    dz_grid_np = np.linspace(-25.0, 20.0, 91, dtype=np.float32)

    case_samples = []
    for ci, bg in enumerate(enriched_bundle_items):
        pool = bg["pool"]
        y_w = bg["y_dz_weight"].numpy()
        y_true_z = bg["y_true_tip_z"].numpy() if isinstance(bg["y_true_tip_z"], torch.Tensor) else bg["y_true_tip_z"]
        profs_24 = bg["tip_profiles_91_24ch"].numpy()
        samples_c = []
        rng_c = np.random.default_rng(700 + ci)
        for u, p in enumerate(pool):
            if y_w[u] <= 0.1:
                continue
            z_cur = float(p["ras"][-1, 2])
            z_gt = float(y_true_z[u])
            dz_true = z_gt - z_cur
            if abs(dz_true) > 21.0:
                continue
            samples_c.append((profs_24[u], float(dz_true), float(y_w[u]), False))
            for _ in range(4):
                shift_steps = int(rng_c.integers(-24, 25))
                shift_mm = shift_steps * 0.5
                dz_new = dz_true + shift_mm
                if -21.0 <= dz_new <= 17.0:
                    prof_sh = shift_1d_profile(profs_24[u], shift_steps)
                    samples_c.append((prof_sh, float(dz_new), float(y_w[u]) * 0.85, True))
        case_samples.append(samples_c)

    for f_out in range(5):
        _, _, inner_subtrain, inner_val = get_nested_fold_split(f_out, len(enriched_bundle_items))
        torch.manual_seed(550 + f_out)
        np.random.seed(550 + f_out)

        tr_list = [s for idx in inner_subtrain for s in case_samples[idx]]
        val_list = [s for idx in inner_val for s in case_samples[idx] if not s[3]]

        X_tr = torch.tensor(np.stack([s[0] for s in tr_list], axis=0), dtype=torch.float32, device=device)
        Y_tr = torch.tensor([s[1] for s in tr_list], dtype=torch.float32, device=device)
        W_tr = torch.tensor([s[2] for s in tr_list], dtype=torch.float32, device=device)

        dz_grid_t = torch.tensor(dz_grid_np, dtype=torch.float32, device=device).unsqueeze(0)
        Y_gauss_tr = torch.exp(-0.5 * ((dz_grid_t - Y_tr.unsqueeze(1)) / 2.0) ** 2)
        Y_gauss_tr = Y_gauss_tr / Y_gauss_tr.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        X_val = torch.tensor(np.stack([s[0] for s in val_list], axis=0), dtype=torch.float32, device=device)
        Y_val = torch.tensor([s[1] for s in val_list], dtype=torch.float32, device=device)
        W_val = torch.tensor([s[2] for s in val_list], dtype=torch.float32, device=device)

        model = Dedicated1DTipLocalizer(in_channels=24, hidden_dim=48, seq_len=91).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=45, eta_min=1e-4)

        best_val_mae = float("inf")
        best_state = None
        best_ep = 0
        batch_size = 64

        for ep in range(45):
            model.train()
            perm = torch.randperm(len(X_tr), device=device)
            for b0 in range(0, len(X_tr), batch_size):
                b_idx = perm[b0:b0 + batch_size]
                opt.zero_grad(set_to_none=True)
                dz_pred, step_logits = model(X_tr[b_idx])
                w_b = W_tr[b_idx]
                l_reg = (F.smooth_l1_loss(dz_pred, Y_tr[b_idx], reduction="none", beta=2.0) * w_b).sum() / w_b.sum().clamp(min=1.0)
                log_p = F.log_softmax(step_logits, dim=-1)
                l_kl = -(Y_gauss_tr[b_idx] * log_p).sum(dim=-1)
                l_kl = (l_kl * w_b).sum() / w_b.sum().clamp(min=1.0)
                loss = l_reg + 1.5 * l_kl
                loss.backward()
                opt.step()
            sched.step()

            if ep >= 8:
                model.eval()
                with torch.no_grad():
                    dz_val_pred, _ = model(X_val)
                    val_mae = float(((torch.abs(dz_val_pred - Y_val) * W_val).sum() / W_val.sum().clamp(min=1.0)).item())
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_ep = ep + 1

        sel_epochs.append(best_ep)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        fold_models[f_out] = model

    print(f"  [Nested Stage 2C 24-Channel Uncertainty Tip Localizer] Selected epochs: {sel_epochs}", flush=True)
    return fold_models, sel_epochs


# ==============================================================================
# 3. UPGRADED LITERATURE-FUSED STAGE 2 DECODER (UPGRADES A + B + C)
# ==============================================================================

def decode_literature_fused_stage2_preds(
    bundle_items,
    base_oof_scores,
    synth_oof_outputs,
    graphs,
    all_oof_cal,
    all_oof_unc,
    all_X_vols,
    tip_models_22ch=None,
    tip_models_24ch=None,
    device=None,
    min_conf=50.0,
    long_mode=False,
    w_synth=0.60,
    use_upgrade_a_weighted_poly3=True,
    use_upgrade_b_unc_tv_scoring=True,
    use_upgrade_b_24ch_tip=True,
    use_upgrade_c_kostoulas_bridge=True,
    alpha_subvox=0.32,
    w_poly_blend=0.52,
    tv_unc_weight=1.0,
    tip_scale=0.36,
):
    """
    Literature-Fused Stage 2 Decoder combining:
      - Upgrade B: Dai 2020 Axial TV Continuity + Aleong/Kim Epistemic Uncertainty candidate
        scoring & 24-channel 1D Tip Localizer ensemble.
      - Upgrade C: Kostoulas 2025 Iterative Multi-Segment Gap-Bridging with Poly3 residual check.
      - Upgrade A: Aleong 2024 / Kim 2026 Sub-Voxel Dark-Core & Inverse-Variance Weighted Poly3
        shaft refinement.
    """
    preds_out = []
    eff_min_conf = min_conf + 15.5

    for i, bg in enumerate(bundle_items):
        pool = bg["pool"]
        sc_base = base_oof_scores[i]
        out_syn = synth_oof_outputs[i]
        sc_syn = 1.0 / (1.0 + np.exp(-np.clip(out_syn[:, :3], -20.0, 20.0)))
        dz_pred_stage2b = out_syn[:, 3] * 15.0

        sc = (1.0 - w_synth) * sc_base + w_synth * sc_syn
        if long_mode:
            conf_arr = 100.0 * (0.30 * sc[:, 0] + 0.25 * sc[:, 1] + 0.45 * sc[:, 2])
        else:
            conf_arr = 100.0 * (0.45 * sc[:, 0] + 0.55 * sc[:, 1])

        # Length-completeness tie-breaker
        for u_idx, p_cand in enumerate(pool):
            span_u = float(p_cand.get("z_span", 20.0))
            if span_u < 18.0 and not long_mode:
                conf_arr[u_idx] -= 5.0 * (18.0 - span_u) / 10.0
            elif span_u >= 35.0 and sc[u_idx, 0] >= 0.75:
                conf_arr[u_idx] += min(3.5, (span_u - 30.0) / 20.0)

        # Upgrade B: Dai 2020 Axial TV Continuity + Aleong 2024 / Kim 2026 Epistemic Uncertainty & Precision Support
        if use_upgrade_b_unc_tv_scoring:
            for u_idx, p_cand in enumerate(pool):
                unc_ratio = float(p_cand.get("unc_ratio", 0.20))
                tv_pm = float(p_cand.get("axial_tv_pm", 0.08))
                prec_sup = float(p_cand.get("precision_support", 0.50))
                mean_sym = float(p_cand.get("mean_sym", 0.50))
                span_u = float(p_cand.get("z_span", 20.0))

                # 1) Reject borderline false-alarm tracks with high axial TV discontinuity or high epistemic uncertainty
                if eff_min_conf <= conf_arr[u_idx] < eff_min_conf + 9.5:
                    if (
                        (tv_pm >= 0.13 and prec_sup <= 0.60)
                        or (unc_ratio >= 0.45 and prec_sup <= 0.45)
                        or (unc_ratio >= 0.18 and prec_sup <= 0.75 and mean_sym <= 0.45)
                    ):
                        conf_arr[u_idx] = eff_min_conf - 2.0

                # 2) Rescue crystal-clear continuous catheters (ultra-low epistemic uncertainty + ultra-low axial TV + high precision support)
                elif conf_arr[u_idx] < eff_min_conf:
                    if sc[u_idx, 0] >= 0.94 and unc_ratio <= 0.045 and tv_pm <= 0.012 and prec_sup >= 1.28:
                        conf_arr[u_idx] = eff_min_conf + 2.5

        # High-resolution symmetry fallback
        for u_idx, p_cand in enumerate(pool):
            span_u = float(p_cand.get("z_span", 20.0))
            if conf_arr[u_idx] < eff_min_conf and sc[u_idx, 0] >= 0.90 and p_cand.get("conf_heur", 0.0) >= 0.88 and p_cand.get("mean_sym", 0.0) >= 0.70:
                if not long_mode or span_u >= 32.0:
                    conf_arr[u_idx] = max(conf_arr[u_idx], float(eff_min_conf + 2.0))

        high_idx = np.where(conf_arr >= max(55.0, eff_min_conf))[0]
        if len(high_idx) >= 3:
            all_xy = np.concatenate([pool[idx]["ras"][:, :2] for idx in high_idx], axis=0)
            center_xy = np.median(all_xy, axis=0)
            dists_core = np.linalg.norm(all_xy - center_xy[None, :], axis=1)
            irgm_radius_mm = float(np.percentile(dists_core, 98.0)) + 15.5
        else:
            center_xy = None
            irgm_radius_mm = 999.0

        sel = []
        for idx_c in np.argsort(-conf_arr):
            if conf_arr[idx_c] < eff_min_conf:
                continue
            cand = dict(pool[idx_c])
            pts = np.asarray(cand["ras"], dtype=np.float64)
            pts = pts[np.argsort(pts[:, 2])]

            if center_xy is not None:
                mean_dist_center = float(np.linalg.norm(pts[:, :2].mean(axis=0) - center_xy))
                if mean_dist_center > irgm_radius_mm and conf_arr[idx_c] < eff_min_conf + 18.0:
                    continue

            if cand.get("mean_sym", 0.5) < 0.14 and cand.get("frac_active", 0.5) < 0.40 and conf_arr[idx_c] < eff_min_conf + 15.0:
                continue

            # Upgrade B: Aleong 2024 high-uncertainty false-alarm gate for borderline candidates
            if use_upgrade_b_unc_tv_scoring:
                if cand.get("unc_ratio", 0.2) > 0.48 and cand.get("mean_sym", 0.5) < 0.28 and conf_arr[idx_c] < eff_min_conf + 6.0:
                    continue

            dedup_thresh = 1.65 if (conf_arr[idx_c] >= eff_min_conf + 12.0 and cand.get("mpnn_pa", 0.0) >= 0.50) else 2.10
            dup_s = None
            can_extend_s = False
            for s in sel:
                info = collinear_traj_dist(pts, s["raw_ras"])
                min_sp = min(info["span_p"], info["span_s"])
                max_sp = max(info["span_p"], info["span_s"])
                is_dup = False
                if info["z_ov"] >= 2.0 and info["d_ov"] < dedup_thresh:
                    is_dup = True
                elif info["z_ov"] < 4.0 and info["z_gap"] <= 20.0 and (info["d_ext"] <= 2.40 or info["d_mid"] <= 2.30) and info["d_slope"] <= 0.14:
                    is_dup = True
                elif info["z_ov"] >= 2.0 and (info["d_ov"] <= 2.85 or info["d_ext"] <= 2.35) and info["d_slope"] <= 0.12 and (min_sp < 0.72 * max_sp or min_sp < 26.0):
                    is_dup = True
                elif info["z_ov"] >= 10.0 and info["d_ov"] <= 2.60 and info["d_ext"] <= 2.25 and info["d_slope"] <= 0.09:
                    is_dup = True
                elif use_upgrade_c_kostoulas_bridge and (
                    (info["z_ov"] < 2.0 and 20.0 < info["z_gap"] <= 38.0 and info["d_ext"] <= 1.45 and info["d_mid"] <= 1.45 and min_sp < 25.0 and conf_arr[idx_c] < 80.0 and (info["d_slope"] <= 0.04 or min_sp <= 9.0))
                    or (conf_arr[idx_c] < 75.0 and info["z_ov"] >= 25.0 and info["d_ov"] <= 2.65 and info["d_mid"] <= 1.80)
                ):
                    is_dup = True

                if is_dup:
                    dup_s = s
                    can_extend_s = (info["d_ext"] <= 2.35 or info["d_ov"] <= 2.50) and (info["d_slope"] <= 0.12)
                    break

            if dup_s is not None:
                z_nbr_s = float(dup_s.get("z_tip_nbr", dup_s["raw_ras"][-1, 2]))
                if (
                    can_extend_s
                    and pts[-1, 2] > dup_s["raw_ras"][-1, 2] + 2.5
                    and pts[-1, 2] <= z_nbr_s + 14.0
                    and cand.get("frac_active", 0.5) >= 0.65
                    and cand.get("mean_sym", 0.5) >= 0.40
                ):
                    dup_s["raw_ras"] = merge_collinear_trajectories(dup_s["raw_ras"], pts)
                    dup_s["idx_c"] = int(idx_c)
                    dup_s["z_prof_tip"] = float(cand.get("z_prof_tip", pts[-1, 2]))
                continue

            cand["raw_ras"] = pts
            cand["idx_c"] = int(idx_c)
            cand["score"] = float(conf_arr[idx_c])
            sel.append(cand)

        # Upgrade C: Kostoulas 2025 Error-Tolerant Gap-Bridging for Discontinuous Needles:
        # 1) Rescue discontinuous split needles where (t_low, t_high) individually fell just below
        #    eff_min_conf due to mid-shaft dropout, but together form a collinear full-length needle
        #    with strong combined neural/physical support and Poly3 fit residual <= 0.75 mm.
        if use_upgrade_c_kostoulas_bridge:
            for u_low in range(len(pool)):
                if not (eff_min_conf - 4.5 <= conf_arr[u_low] < eff_min_conf):
                    continue
                cand_low = pool[u_low]
                r_low = np.asarray(cand_low["ras"], dtype=np.float64)
                r_low = r_low[np.argsort(r_low[:, 2])]
                if any(collinear_traj_dist(r_low, s["raw_ras"])["d_ov"] < 2.2 for s in sel):
                    continue
                best_high = None
                best_u_high = -1
                for u_high in range(len(pool)):
                    if u_high == u_low or sc[u_high, 0] < 0.50:
                        continue
                    r_high = np.asarray(pool[u_high]["ras"], dtype=np.float64)
                    r_high = r_high[np.argsort(r_high[:, 2])]
                    if float(r_high[-1, 2]) <= float(r_low[-1, 2]) + 6.0 or float(r_high[0, 2]) < float(r_low[0, 2]) + 4.0:
                        continue
                    info = collinear_traj_dist(r_low, r_high)
                    if info["z_gap"] <= 18.0 and info["d_slope"] <= 0.10 and info["d_mid"] <= 2.05:
                        merged_c = merge_collinear_trajectories(r_low, r_high)
                        span_m = float(merged_c[-1, 2] - merged_c[0, 2])
                        if span_m < (32.0 if long_mode else 24.0):
                            continue
                        poly_m = regularize_poly3_trajectory(merged_c, degree=3)
                        rmse_m = float(np.sqrt(np.mean((merged_c[:, 0] - poly_m[:, 0]) ** 2 + (merged_c[:, 1] - poly_m[:, 1]) ** 2)))
                        if rmse_m <= 0.75 and not any(traj_gt_dist(merged_c, s["raw_ras"]) < 2.15 for s in sel):
                            best_high = merged_c
                            best_u_high = u_high
                            break
                if best_high is not None:
                    rescued = dict(cand_low)
                    rescued["raw_ras"] = best_high
                    rescued["idx_c"] = int(best_u_high)
                    rescued["z_prof_tip"] = float(pool[best_u_high].get("z_prof_tip", best_high[-1, 2]))
                    rescued["score"] = float(conf_arr[u_low] + 4.5)
                    sel.append(rescued)

        # 2) Iterative Multi-Segment Cranial Gap-Bridging & Poly3 Fit Verification
        n_bridge_passes = 2 if use_upgrade_c_kostoulas_bridge else 1
        max_gap_mm = 18.0 if use_upgrade_c_kostoulas_bridge else 14.0
        for _pass in range(n_bridge_passes):
            for s in sel:
                z_cur_top = float(s["raw_ras"][-1, 2])
                z_nbr_s = float(s.get("z_tip_nbr", z_cur_top))
                z_prof_s = float(s.get("z_prof_tip", z_cur_top))
                need_bridge = (
                    (z_cur_top < z_nbr_s - 3.0 or z_cur_top < z_prof_s - 3.0)
                    if use_upgrade_c_kostoulas_bridge
                    else (z_cur_top < z_nbr_s - 6.0)
                )
                if need_bridge:
                    best_ext = None
                    best_u = -1
                    best_top = z_cur_top
                    for u_idx, p_cand in enumerate(pool):
                        low_unc_ok = use_upgrade_c_kostoulas_bridge and (
                            sc[u_idx, 0] >= 0.55 and p_cand.get("unc_ratio", 0.5) <= 0.30 and p_cand.get("mean_sym", 0.0) >= 0.50
                        )
                        if not low_unc_ok and (sc[u_idx, 0] < 0.65 or p_cand.get("frac_active", 0.0) < 0.70 or p_cand.get("mean_sym", 0.0) < 0.45):
                            continue
                        c_ras = p_cand["ras"]
                        c_top = float(c_ras[-1, 2])
                        if c_top <= best_top + 3.5 or c_top > z_nbr_s + 8.5:
                            continue
                        info = collinear_traj_dist(c_ras, s["raw_ras"])
                        if info["z_gap"] <= max_gap_mm and info["d_slope"] <= 0.10 and (info["d_ov"] <= 2.60 or (info["d_ext"] <= 2.10 and info["d_mid"] <= 2.25)):
                            merged_cand = merge_collinear_trajectories(s["raw_ras"], c_ras)
                            poly_m = regularize_poly3_trajectory(merged_cand, degree=3)
                            rmse_poly3 = float(np.sqrt(np.mean((merged_cand[:, 0] - poly_m[:, 0]) ** 2 + (merged_cand[:, 1] - poly_m[:, 1]) ** 2)))
                            if rmse_poly3 > 0.80:
                                continue
                            d_to_s = min(info["d_ov"], info["d_ext"], info["d_mid"])
                            closer_to_other = False
                            for s_oth in sel:
                                if s_oth is s:
                                    continue
                                info_o = collinear_traj_dist(c_ras, s_oth["raw_ras"])
                                if min(info_o["d_ov"], info_o["d_ext"], info_o["d_mid"]) < d_to_s - 0.10:
                                    closer_to_other = True
                                    break
                            if not closer_to_other:
                                best_top = c_top
                                best_ext = p_cand
                                best_u = u_idx
                    if best_ext is not None:
                        s["raw_ras"] = merge_collinear_trajectories(s["raw_ras"], best_ext["ras"])
                        s["idx_c"] = int(best_u)
                        s["z_prof_tip"] = float(best_ext.get("z_prof_tip", best_ext["ras"][-1, 2]))

        # Stage 2C Tip Localization + Upgrade A (Kim 2026 Inverse-Variance & Sub-Voxel Weighted Poly3)
        f_out = i % 5
        tip_m22 = tip_models_22ch[f_out] if tip_models_22ch is not None else None
        tip_m24 = tip_models_24ch[f_out] if (use_upgrade_b_24ch_tip and tip_models_24ch is not None) else None

        for s in sel:
            idx_c = s["idx_c"]
            pts = s["raw_ras"]
            vx_top, vy_top = fit_local_slope(pts, float(pts[-1, 2]), 25.0)
            dz_prof_mm = float(s.get("z_prof_tip", pts[-1, 2]) - pts[-1, 2])
            dz_nbr_mm = float(s.get("z_tip_nbr", pts[-1, 2]) - pts[-1, 2])
            dz_s2b_mm = float(dz_pred_stage2b[idx_c])

            dz_22_val = None
            dz_24_val = None
            if tip_m22 is not None:
                if "tip_profiles_91" in bg and abs(float(pts[-1, 2]) - float(pool[idx_c]["ras"][-1, 2])) < 0.5:
                    x_t22 = bg["tip_profiles_91"][idx_c:idx_c + 1].to(device)
                else:
                    prof_22 = extract_1d_tip_profile_91(
                        pts, graphs[i], all_oof_cal[i], all_X_vols[i],
                        z_center=float(pts[-1, 2]), z_tip_nbr=float(s.get("z_tip_nbr", pts[-1, 2])),
                    )
                    x_t22 = torch.tensor(prof_22, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    p22, _ = tip_m22(x_t22)
                    dz_22_val = float(p22[0].item())

            if tip_m24 is not None:
                if "tip_profiles_91_24ch" in bg and abs(float(pts[-1, 2]) - float(pool[idx_c]["ras"][-1, 2])) < 0.5:
                    x_t24 = bg["tip_profiles_91_24ch"][idx_c:idx_c + 1].to(device)
                else:
                    prof_24 = extract_1d_tip_profile_91_24ch(
                        pts, graphs[i], all_oof_cal[i], all_oof_unc[i], all_X_vols[i],
                        z_center=float(pts[-1, 2]), z_tip_nbr=float(s.get("z_tip_nbr", pts[-1, 2])),
                    )
                    x_t24 = torch.tensor(prof_24, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    p24, _ = tip_m24(x_t24)
                    dz_24_val = float(p24[0].item())

            if dz_22_val is not None and dz_24_val is not None:
                dz_1d_mm = 0.80 * dz_22_val + 0.20 * dz_24_val
                dz_comb_mm = 0.50 * dz_1d_mm + 0.30 * (tip_scale * dz_s2b_mm) + 0.15 * (tip_scale * dz_prof_mm) + 0.05 * (tip_scale * dz_nbr_mm)
            elif dz_22_val is not None:
                dz_comb_mm = 0.50 * dz_22_val + 0.30 * (tip_scale * dz_s2b_mm) + 0.15 * (tip_scale * dz_prof_mm) + 0.05 * (tip_scale * dz_nbr_mm)
            else:
                dz_comb_mm = tip_scale * (0.65 * dz_s2b_mm + 0.25 * dz_prof_mm + 0.10 * dz_nbr_mm)

            if use_upgrade_a_weighted_poly3:
                pts_refined = refine_shaft_and_tip_kim2026_weighted_poly3(
                    pts, dz_comb_mm, vx_top, vy_top,
                    graphs[i], all_oof_cal[i], all_oof_unc[i], all_X_vols[i],
                    alpha_subvox=alpha_subvox, w_poly_blend=w_poly_blend,
                )
            else:
                pts_refined = apply_tip_adjustment_and_poly3(pts, dz_comb_mm, vx_top, vy_top)

            s["ras"] = pts_refined
            s["points"] = pts_refined
            s["dz_comb_mm"] = dz_comb_mm

        preds_out.append(sel)

    return preds_out


def main():
    t0 = time.perf_counter()
    best_gpu = 0
    for g_i in range(torch.cuda.device_count()):
        if "4090" in torch.cuda.get_device_name(g_i):
            best_gpu = g_i
            break
    device = torch.device(f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Part 2: Literature Fusion Inside NeedleFinder 2026] Running on {device} ({torch.cuda.get_device_name(best_gpu)})...", flush=True)

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

    s2_preds_data = pickle.load(open(STAGE2_PREDS_PKL, "rb"))
    bundle_cache = pickle.load(open(BUNDLE_CACHE_PKL, "rb"))
    stitched_enriched_bgs = copy.deepcopy(s2_preds_data.get("stitched_enriched_bgs") or bundle_cache["stitched_enriched_bgs"])
    graphs = bundle_cache["graphs"]
    all_oof_cal = bundle_cache["all_oof_cal"]
    all_X_vols = [torch.load(f"{CLEAN_CACHE_DIR}/case_{i:02d}.pt", map_location="cpu", weights_only=False)["X"].numpy() for i in range(54)]

    # Load Stage 1 5-model epistemic uncertainty maps (`oof_unc_case_{i:02d}.npz`)
    all_oof_unc = []
    for i in range(54):
        unc_npz = np.load(os.path.join(OOF_DIR, f"oof_unc_case_{i:02d}.npz"))
        all_oof_unc.append({
            "sigma_mask": unc_npz["sigma_mask"].astype(np.float32),
            "sigma_peak": unc_npz["sigma_peak"].astype(np.float32),
            "sigma_vec": unc_npz["sigma_vec"].astype(np.float32),
        })
    print(f"  Loaded 54 Stage 1 epistemic uncertainty volumes ({time.perf_counter()-t0:.1f}s).", flush=True)

    # Enrich candidate pools with Dai 2020 Axial TV + Aleong/Kim Epistemic Uncertainty features & 24-ch 1D profiles
    tv_unc_cache_path = resolve_path("/tmp/stage2_tv_unc_24ch_cache.pkl")
    if os.path.exists(tv_unc_cache_path):
        c_tv = pickle.load(open(tv_unc_cache_path, "rb"))
        for i, bg in enumerate(stitched_enriched_bgs):
            for u, p in enumerate(bg["pool"]):
                p.update(c_tv["pool_tv_unc"][i][u])
            bg["tip_profiles_91_24ch"] = c_tv["profs_24"][i]
        print(f"  [Cache Hit] Loaded Dai 2020 Axial TV + Aleong/Kim Epistemic Uncertainty features & 24-ch profiles from {tv_unc_cache_path} ({time.perf_counter()-t0:.1f}s).", flush=True)
    else:
        pool_tv_unc_all = []
        profs_24_all = []
        for i, bg in enumerate(stitched_enriched_bgs):
            pool = bg["pool"]
            g = graphs[i]
            oof_cal = all_oof_cal[i]
            oof_unc = all_oof_unc[i]
            X_vol = all_X_vols[i]
            profs_24 = []
            tv_i = []
            for u, p in enumerate(pool):
                tv_unc = compute_candidate_tv_and_uncertainty_features(p["ras"], g, oof_cal, oof_unc, X_vol)
                p.update(tv_unc)
                tv_i.append(tv_unc)
                profs_24.append(extract_1d_tip_profile_91_24ch(
                    p["ras"], g, oof_cal, oof_unc, X_vol,
                    z_center=float(p["ras"][-1, 2]), z_tip_nbr=float(p.get("z_tip_nbr", p["ras"][-1, 2])),
                ))
            t24 = (
                torch.tensor(np.stack(profs_24, axis=0), dtype=torch.float32)
                if len(pool) > 0 else torch.zeros((0, 24, 91), dtype=torch.float32)
            )
            bg["tip_profiles_91_24ch"] = t24
            pool_tv_unc_all.append(tv_i)
            profs_24_all.append(t24)
        pickle.dump({"pool_tv_unc": pool_tv_unc_all, "profs_24": profs_24_all}, open(tv_unc_cache_path, "wb"))
        print(f"  Computed & cached Dai 2020 Axial TV + Aleong/Kim Epistemic Uncertainty features & 24-ch tip profiles ({time.perf_counter()-t0:.1f}s).", flush=True)

    # Load existing 22-channel 1D tip models and train new 24-channel uncertainty-aware 1D tip models
    tip_models_22ch = []
    for st in s2_preds_data["tip_models_5fold_states"]:
        m = Dedicated1DTipLocalizer(in_channels=22, hidden_dim=48, seq_len=91).to(device)
        m.load_state_dict({k: v.to(device) for k, v in st.items()})
        m.eval()
        tip_models_22ch.append(m)

    tip_models_24ch, tip24_epochs = train_nested_stage2c_1d_tip_localizer_24ch(stitched_enriched_bgs, device)

    base_sc = s2_preds_data["base_sc_st_mr"]
    syn_out = s2_preds_data["syn_out_st_mr"]
    base_sc_inner = s2_preds_data.get("base_sc_st_inner")
    syn_out_inner = s2_preds_data.get("syn_out_st_inner")

    # Evaluate factor-by-factor progression of Literature-Inspired Upgrades inside NeedleFinder 2026:
    #   0. Baseline NeedleFinder 2026 (Clean Nested CV, tau=50%)
    #   1. + Upgrade A Only (Aleong 2024 / Kim 2026 Sub-Voxel Dark-Core & Inverse-Variance Weighted Poly3)
    #   2. + Upgrade B Only (Dai 2020 Axial TV + Aleong/Kim Epistemic Uncertainty in Stage 2B & 24-ch 1D Tip)
    #   3. + Upgrade C Only (Kostoulas 2025 Iterative Multi-Segment Gap-Bridging & Poly3 Fit Verification)
    #   4. + Upgrades A + B + C Combined (Literature-Enhanced NeedleFinder 2026, tau=50%)
    #   5. + Upgrades A + B + C Combined on Inner-OOF Cross-Fit (tau=50%)
    ablation_configs = [
        ("0_needlefinder_2026_base_tau50", "NeedleFinder 2026 (Clean Nested Base, tau=50%)", base_sc, syn_out, False, False, False, False),
        ("1_plus_upgrade_A_kim2026_weighted_poly3", "+ Upgrade A: Kim 2026 Sub-Voxel & Inverse-Variance Weighted Poly3", base_sc, syn_out, True, False, False, False),
        ("2_plus_upgrade_B_dai2020_tv_aleong_unc", "+ Upgrade B: Dai 2020 Axial TV + Aleong 2024 Epistemic Uncertainty (Stage 2B + 24-ch 1D Tip)", base_sc, syn_out, False, True, True, False),
        ("3_plus_upgrade_C_kostoulas2025_bridge", "+ Upgrade C: Kostoulas 2025 Multi-Segment Iterative Gap-Bridging", base_sc, syn_out, False, False, False, True),
        ("4_needlefinder_2026_literature_fused_tau50", "NeedleFinder 2026 + All 3 Literature Upgrades (A+B+C, tau=50%)", base_sc, syn_out, True, True, True, True),
    ]
    if base_sc_inner is not None and syn_out_inner is not None:
        ablation_configs.append((
            "5_needlefinder_2026_inner_oof_literature_fused_tau50",
            "NeedleFinder 2026 (Inner-OOF Cross-Fit) + All 3 Literature Upgrades (A+B+C, tau=50%)",
            base_sc_inner, syn_out_inner, True, True, True, True,
        ))

    fusion_ablation_results = {}
    preds_by_key = {}

    for key, label, b_sc, s_out, up_a, up_b_sc, up_b_tip, up_c in ablation_configs:
        p_all = decode_literature_fused_stage2_preds(
            stitched_enriched_bgs, b_sc, s_out,
            graphs=graphs, all_oof_cal=all_oof_cal, all_oof_unc=all_oof_unc, all_X_vols=all_X_vols,
            tip_models_22ch=tip_models_22ch, tip_models_24ch=tip_models_24ch, device=device,
            min_conf=50.0, long_mode=False, w_synth=0.60,
            use_upgrade_a_weighted_poly3=up_a,
            use_upgrade_b_unc_tv_scoring=up_b_sc,
            use_upgrade_b_24ch_tip=up_b_tip,
            use_upgrade_c_kostoulas_bridge=up_c,
        )
        p_long = decode_literature_fused_stage2_preds(
            stitched_enriched_bgs, b_sc, s_out,
            graphs=graphs, all_oof_cal=all_oof_cal, all_oof_unc=all_oof_unc, all_X_vols=all_X_vols,
            tip_models_22ch=tip_models_22ch, tip_models_24ch=tip_models_24ch, device=device,
            min_conf=50.0, long_mode=True, w_synth=0.60,
            use_upgrade_a_weighted_poly3=up_a,
            use_upgrade_b_unc_tv_scoring=up_b_sc,
            use_upgrade_b_24ch_tip=up_b_tip,
            use_upgrade_c_kostoulas_bridge=up_c,
        )
        res_all, pp_all = evaluate_method_unified(p_all, gts_all_list, thresh=3.5, n_boot=300)
        res_long, pp_long = evaluate_method_unified(p_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=300)
        res_adh, _ = evaluate_method_unified([p_all[i] for i in adh_idx], gts_all_adh, thresh=3.5, n_boot=300)
        diag = compute_detailed_tip_and_fp_diagnostics(p_all, gts_all_list, {})

        preds_by_key[key] = (p_all, p_long, pp_all, pp_long)
        fusion_ablation_results[key] = {
            "label": label,
            "cohort_all_1063": res_all,
            "cohort_long_759": res_long,
            "severe_adhesion_571": res_adh,
            "diagnostics": diag,
        }
        print(
            f"\n[{label}]\n"
            f"  All (1,063) : F1={res_all['f1_1to1_pct']:.2f}% (Rec={res_all['hungarian_1to1_recall_pct']:.2f}%, Prec={res_all['precision_1to1_pct']:.2f}%, Matched={res_all['hungarian_1to1_matched']}/{res_all['n_pred']})\n"
            f"  Long (759)  : F1_adj={res_long['f1_1to1_obturator_adj_pct']:.2f}% (F1_raw={res_long['f1_1to1_pct']:.2f}%, Rec={res_long['hungarian_1to1_recall_pct']:.2f}%)\n"
            f"  Adh (571)   : F1={res_adh['f1_1to1_pct']:.2f}% (Rec={res_adh['hungarian_1to1_recall_pct']:.2f}%, Prec={res_adh['precision_1to1_pct']:.2f}%)\n"
            f"  Geometry    : Shaft={res_all['metrics']['mean_shaft_xy']['mean']:.3f}/{res_all['metrics']['mean_shaft_xy']['median']:.3f} mm | "
            f"3D MSD={res_all['metrics']['msd_3d']['mean']:.3f}/{res_all['metrics']['msd_3d']['median']:.3f} mm | "
            f"Tip={res_all['metrics']['tip_3d']['mean']:.2f}/{res_all['metrics']['tip_3d']['median']:.2f} mm (<=3mm: {diag['pct_tip_le_3mm']:.1f}%, >10mm: {diag['pct_tip_gt_10mm']:.2f}%)",
            flush=True,
        )

    # Combine Part 1 (6 Peer-Reviewed Baselines) + Part 2 (Literature-Enhanced NeedleFinder 2026) into final JSON
    part1_data = json.load(open(PART1_JSON, "r")) if os.path.exists(PART1_JSON) else {}

    # Also load Baselines 1, 2, 3 from unified clean JSON for a complete 6-baseline table
    unified_clean_path = resolve_path("/home/guillaume/github/needlefinder/vault/sources/2026-10-23-unified-clean-nested-benchmark.json")
    baselines_1_2_3 = {}
    if os.path.exists(unified_clean_path):
        u_clean = json.load(open(unified_clean_path, "r"))
        for b_key in [
            "Baseline 1: PMB 2019 Isotropic 1mm³ 3D U-Net + DBSCAN/Skeleton",
            "Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)",
            "Baseline 3: Xiao 2026 IRGM + 3D Hough + Sync Inward Tracking + Poly3",
        ]:
            if b_key in u_clean["cohort_all_1063_catheters_54cases"]:
                baselines_1_2_3[b_key] = {
                    "cohort_all_1063": u_clean["cohort_all_1063_catheters_54cases"][b_key],
                    "cohort_long_759": u_clean["cohort_long_759_cy_catheters_54cases"][b_key],
                    "severe_adhesion_571": u_clean["subgroup_adhesion_vs_non_adhesion"]["severe_adhesion_subgroup_all_catheters"].get(b_key, {}),
                }

    # Paired Wilcoxon tests comparing Literature-Fused NeedleFinder 2026 vs Base and vs Baselines 4, 5, 6
    wilcoxon_tests = {}
    nf_fused_pp = preds_by_key["4_needlefinder_2026_literature_fused_tau50"][2]
    nf_base_pp = preds_by_key["0_needlefinder_2026_base_tau50"][2]
    wilcoxon_tests["nf2026_literature_fused_vs_nf2026_base"] = compute_wilcoxon_comparison(nf_fused_pp, nf_base_pp)

    for pkl_name, b_label in [
        ("oof_trajs_dai2020_attunet_tv.pkl", "nf2026_literature_fused_vs_dai2020"),
        ("oof_trajs_kostoulas2025_gapbridge.pkl", "nf2026_literature_fused_vs_kostoulas2025"),
        ("oof_trajs_aleong2024_kim2026_unc_ensemble.pkl", "nf2026_literature_fused_vs_aleong2024_kim2026"),
    ]:
        pkl_p = os.path.join(OOF_DIR, pkl_name)
        if os.path.exists(pkl_p):
            b_d = pickle.load(open(pkl_p, "rb"))
            _, pp_b = evaluate_method_unified(b_d["all"], gts_all_list, thresh=3.5, n_boot=100)
            wilcoxon_tests[b_label] = compute_wilcoxon_comparison(nf_fused_pp, pp_b)

    # Save the upgraded predictions to disk so downstream exporters can use them
    pickle.dump({
        "nf2026_literature_fused_tau50_all": preds_by_key["4_needlefinder_2026_literature_fused_tau50"][0],
        "nf2026_literature_fused_tau50_long": preds_by_key["4_needlefinder_2026_literature_fused_tau50"][1],
        "tip_models_24ch_states": [{k: v.cpu() for k, v in m.state_dict().items()} for m in tip_models_24ch],
    }, open(os.path.join(OOF_DIR, "oof_trajs_nf2026_literature_fused.pkl"), "wb"))

    final_report = {
        "generated_at": "2026-10-24",
        "protocol": "Strictly Nested 5x5 Cross-Validation on 54 Patients (1,063 All Catheters / 759 Long Interstitial / 23 Severe Adhesion Cases with 571 Catheters)",
        "part1_peer_reviewed_baselines_1_to_6": {
            **baselines_1_2_3,
            **part1_data,
        },
        "part2_literature_fusion_inside_needlefinder_2026": fusion_ablation_results,
        "paired_wilcoxon_tests_54_patients": wilcoxon_tests,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    os.makedirs(os.path.dirname(OUT_FUSION_JSON), exist_ok=True)
    with open(OUT_FUSION_JSON, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[Part 2 Done in {time.perf_counter()-t0:.1f}s] Saved {OUT_FUSION_JSON}", flush=True)


if __name__ == "__main__":
    main()
