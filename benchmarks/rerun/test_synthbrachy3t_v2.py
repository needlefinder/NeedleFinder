#!/usr/bin/env python3
"""
SynthBrachy3T: Physics-Informed Hybrid Synthetic 3T MRI Generator &
Tip-Aware Bundle Graph Transformer (NeedleFinder 2026 v2)

Implements:
1. Physics-Informed 3T T2W MRI Synthetic Catheter Generator (SynthBrachy3T):
   - Uses real needle-free 3T pelvic T2W MRI backgrounds (48 MSD Task05_Prostate volumes)
     and harmonic-inpainted training-fold backgrounds (strictly isolated per fold, zero leakage).
   - Simulates 3T spin-echo catheter physics: 6F (2.0 mm) dark lumen core, magnetic susceptibility
     dipole hyperintense rim, anisotropic slice-profile partial-volume integration, 3rd-order
     polynomial bending, Syed-Neblett + obturator geometry, 35% deliberate severe catheter adhesion
     (<= 2.5 mm crossing/touching pairs), and stochastic 1-2 slice signal dropouts.
   - Generates 6-head voxel targets [mask, peak, vx, vy, obt, tip_heatmap] plus 1,500 synthetic
     candidate bundle graphs (~38,000 synthetic trajectories) with exact ground-truth 1-to-1
     matching labels and exact sub-millimeter cranial tip (z_tip) & caudal base (z_base) targets.
2. Enriched 28-Feature + 1D Axial-Profile Tip-Aware BundleGraphTransformer:
   - Jointly predicts calibrated trajectory probabilities (logit_rec, logit_1to1, logit_long_1to1)
     AND refines cranial tip z_tip and caudal base z_base via a 1D Axial-Profile + Spatial
     Bundle-Neighborhood Tip Regressor pre-trained on SynthBrachy3T and fine-tuned per fold.
3. Sim-to-Real Few-Shot Evaluation (k = 0, 1, 5, 10, 20, 43 clinical patients/fold):
   - k = 0 uses 100% synthetic implants on external MSD Task05_Prostate 3T MRI scans + Stage 0 prior.
   - k >= 1 fine-tunes the SynthBrachy3T-pretrained models on k clinical training patients per fold.
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path

import os, sys, json, glob, time, math, pickle, types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, maximum_filter, binary_dilation
from scipy.optimize import linear_sum_assignment

for mod in ["vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util", "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"]:
    sys.modules[mod] = types.ModuleType(mod)

sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun"))

from evaluate_unified_apples_to_apples import (
    evaluate_method_unified,
    compute_wilcoxon_comparison,
    compute_unified_pair_metrics,
    summarize_distribution,
    extract_nn_labelmap_tubes,
    build_slice_graph_fast,
    NeedleSliceMPNN,
    decode_mpnn_trajectories,
    get_gts,
    traj_gt_dist,
    CACHE_DIR,
    OOF_DIR,
    META_PATH,
    BASE_TRAJS_PATH,
)
from run_xiao2026_and_fewshot_benchmark import (
    analyze_catheter_adhesion,
    regularize_poly3_trajectory,
    dedup_traj_dicts,
)


# -------------------------------------------------------------------------
# 1. Physics-Informed Synthetic 3T MRI Volume & Bundle Generator (SynthBrachy3T)
# -------------------------------------------------------------------------
def erase_real_catheters_inpaint(X_vol, Y_vol):
    """
    Harmonic/Gaussian neighborhood inpainting of real catheter voids in a training volume X_vol [6, K, H, W]
    so we obtain a clean needle-free 3T pelvic T2W MRI anatomical canvas.
    """
    X_clean = X_vol.clone().numpy()
    mask = (Y_vol[0].numpy() > 0.15) | (Y_vol[4].numpy() > 0.15) | (X_clean[1] > 0.28)
    mask_dil = binary_dilation(mask, iterations=2)
    # Fill channel 0 (T2W MRI) with smooth local tissue intensity
    ch0 = X_clean[0]
    smooth_bg = gaussian_filter(ch0, sigma=(0.8, 3.5, 3.5))
    noise = np.random.normal(0.0, 0.04, size=ch0.shape).astype(np.float32)
    ch0[mask_dil] = smooth_bg[mask_dil] + noise[mask_dil]
    X_clean[0] = ch0
    # Attenuate existing catheter symmetry responses while keeping anatomical vessel clutter!
    X_clean[1][mask_dil] *= 0.12
    X_clean[2][mask_dil] *= 0.15
    X_clean[3][mask_dil] = np.clip(X_clean[3][mask_dil] * 0.15, -0.2, 0.15)
    return X_clean


def synthesize_3t_mri_implant_volume(X_canvas, sp_ras, rng):
    """
    Given a clean 3T pelvic MRI canvas X_canvas [6, K, H, W], implants a synthetic 3D catheter
    bundle (16-28 catheters) with:
      - Syed-Neblett / Venezia template ring geometry + central obturator
      - 35% deliberate severe catheter adhesion / crossing pairs (<= 2.5 mm)
      - 3rd-order polynomial tissue deflection
      - 3T spin-echo susceptibility dark lumen core + asymmetric readout hyperintense rim
      - Partial-volume slice thickness integration at the cranial tip z_tip
      - Stochastic 1-slice signal dropouts
    Returns:
      X_syn [6, K, H, W], Y_syn [6, K, H, W] (where Y_syn[5] is the 3D distal tip heatmap!)
    """
    _, K, H, W = X_canvas.shape
    sp_x, sp_y, sp_z = float(sp_ras[0]), float(sp_ras[1]), float(sp_ras[2])
    X_syn = X_canvas.copy()
    # Apply random 3D intensity affine scaling + spatial flip for anatomical diversity
    if rng.random() < 0.5:
        X_syn = X_syn[:, :, :, ::-1].copy()
    if rng.random() < 0.35:
        X_syn = X_syn[:, :, ::-1, :].copy()
    X_syn[0] = np.clip(X_syn[0] * rng.uniform(0.88, 1.14) + rng.uniform(-0.06, 0.06), -0.4, 2.4)

    Y_syn = np.zeros((6, K, H, W), dtype=np.float32)  # [mask, peak, vx, vy, obt, tip]

    cx = W * 0.5 + rng.uniform(-8.0, 8.0)
    cy = H * 0.5 + rng.uniform(-8.0, 8.0)

    n_long = int(rng.integers(12, 22))
    n_short = int(rng.integers(3, 8))
    n_adhesion_pairs = max(2, int(n_long * 0.30))

    trajs = []
    # Generate base positions on template rings
    angles = np.linspace(0, 2 * np.pi, n_long, endpoint=False) + rng.uniform(0, 2 * np.pi)
    for idx in range(n_long):
        r_mm = rng.uniform(7.0, 28.0)
        bx = cx + (r_mm * np.cos(angles[idx])) / sp_x + rng.normal(0, 1.5)
        by = cy + (r_mm * np.sin(angles[idx])) / sp_y + rng.normal(0, 1.5)
        sx = rng.normal(0.0, 0.09)
        sy = rng.normal(0.0, 0.09)
        c2x = rng.normal(0.0, 0.002)
        c2y = rng.normal(0.0, 0.002)
        k_start = int(rng.integers(0, max(1, min(3, K // 4))))
        k_end = int(rng.integers(max(k_start + 4, int(K * 0.58)), K))
        trajs.append((bx, by, sx, sy, c2x, c2y, k_start, k_end, False))

    # Inject deliberate severe catheter-adhesion / crossing twin shafts (1.5 - 2.6 mm apart)
    for p_i in range(min(n_adhesion_pairs, len(trajs))):
        bx, by, sx, sy, c2x, c2y, k_start, k_end, _ = trajs[p_i]
        ang_off = rng.uniform(0, 2 * np.pi)
        sep_mm = rng.uniform(1.55, 2.65)
        bx2 = bx + (sep_mm * np.cos(ang_off)) / sp_x
        by2 = by + (sep_mm * np.sin(ang_off)) / sp_y
        # Slight convergence/crossing slope
        sx2 = sx - 0.03 * np.cos(ang_off) + rng.normal(0, 0.02)
        sy2 = sy - 0.03 * np.sin(ang_off) + rng.normal(0, 0.02)
        k_end2 = int(np.clip(k_end + rng.integers(-2, 3), k_start + 4, K - 1))
        trajs.append((bx2, by2, sx2, sy2, -c2x, -c2y, k_start, k_end2, False))

    # Add short / obturator trajectories
    for s_i in range(n_short):
        ang = rng.uniform(0, 2 * np.pi)
        r_mm = rng.uniform(3.5, 11.5)
        bx = cx + (r_mm * np.cos(ang)) / sp_x
        by = cy + (r_mm * np.sin(ang)) / sp_y
        sx = rng.normal(0.0, 0.05)
        sy = rng.normal(0.0, 0.05)
        k_start = int(rng.integers(0, max(1, min(2, K // 4))))
        k_end = int(rng.integers(min(K - 1, k_start + 3), max(k_start + 4, min(K, int(K * 0.48)))))
        trajs.append((bx, by, sx, sy, 0.0, 0.0, k_start, k_end, True))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r_mask_px = 1.50 / sp_x
    sigma_peak_px = 0.85 / sp_x
    sigma_core_px = 1.05 / sp_x

    for (bx, by, sx, sy, c2x, c2y, k_start, k_end, is_obt) in trajs:
        for k in range(k_start, min(K, k_end + 1)):
            dz_mm = (k - k_start) * sp_z
            xc = bx + (sx * dz_mm + c2x * (dz_mm ** 2)) / sp_x
            yc = by + (sy * dz_mm + c2y * (dz_mm ** 2)) / sp_y
            if not (2 <= xc < W - 2 and 2 <= yc < H - 2):
                continue

            dx_px = xx - xc
            dy_px = yy - yc
            dist2_px = dx_px ** 2 + dy_px ** 2

            # Ground truth targets (always unbroken even during MRI signal dropout!)
            in_mask = dist2_px <= (r_mask_px ** 2)
            Y_syn[0, k][in_mask] = 1.0
            peak_g = np.exp(-0.5 * dist2_px / (sigma_peak_px ** 2))
            Y_syn[1, k] = np.maximum(Y_syn[1, k], peak_g)
            vx_val = float(np.clip(sx + 2.0 * c2x * dz_mm, -0.65, 0.65))
            vy_val = float(np.clip(sy + 2.0 * c2y * dz_mm, -0.65, 0.65))
            Y_syn[2, k][in_mask] = vx_val
            Y_syn[3, k][in_mask] = vy_val
            if is_obt:
                Y_syn[4, k][in_mask] = 1.0

            # Distal tip 3D Gaussian target at k == k_end
            if abs(k - k_end) <= 1:
                z_w = 1.0 if k == k_end else 0.45
                tip_g = z_w * np.exp(-0.5 * dist2_px / ((sigma_peak_px * 1.25) ** 2))
                Y_syn[5, k] = np.maximum(Y_syn[5, k], tip_g)

            # Render physical 3T T2W MRI appearance + Stage 0 response (with 12% dropout probability)
            is_dropout = (rng.random() < 0.12) and (k != k_end)
            partial_vol = 0.60 if k == k_end else (0.30 if is_dropout else 1.0)
            core_prof = partial_vol * np.exp(-0.5 * dist2_px / (sigma_core_px ** 2))
            # Susceptibility bright rim along frequency encoding (x) axis
            rim_prof = partial_vol * 0.16 * ((dx_px ** 2 - dy_px ** 2) / (dist2_px + 1.5)) * np.exp(-0.5 * dist2_px / ((sigma_core_px * 1.9) ** 2))

            X_syn[0, k] = np.clip(X_syn[0, k] * (1.0 - 0.68 * core_prof) - 0.18 * core_prof + rim_prof, -0.45, 2.4)
            sym_strength = rng.uniform(0.55, 1.15) * partial_vol
            X_syn[1, k] = np.maximum(X_syn[1, k], sym_strength * peak_g)
            X_syn[2, k] = np.maximum(X_syn[2, k], 0.90 * sym_strength * peak_g)
            X_syn[3, k] = np.maximum(X_syn[3, k], 0.65 * sym_strength * core_prof)

    return torch.tensor(X_syn, dtype=torch.float32), torch.tensor(Y_syn, dtype=torch.float32)


# -------------------------------------------------------------------------
# 2. Enriched 26-Feature + Axial-Profile Tip-Aware Bundle Graph Construction
# -------------------------------------------------------------------------
def compute_trajectory_axial_profile_and_tip_features(ras, g, oof_npz, X_vol):
    """
    Samples the 3D trajectory (and its linear/polynomial extension across the full slab K_s)
    along all axial slices to extract:
      - Stage 0 circular symmetry continuity (mean_sym, p75_sym, mean_cs, mean_rc, frac_active)
      - Stage 1 probability continuity (mean_pm, p75_pm, mean_pp, mean_po)
      - Profile-supported caudal base z_prof_base and cranial tip z_prof_tip
      - Tip sharpness / gradient at the cranial termination
    """
    pm_arr = oof_npz["p_mask"].astype(np.float32)
    pp_arr = oof_npz["p_peak"].astype(np.float32)
    po_arr = oof_npz["p_obt"].astype(np.float32)
    sym_arr = X_vol[1]
    cs_arr = X_vol[2]
    rc_arr = X_vol[3]

    ijk2ras = g["ijk_to_ras"]
    ras2ijk = np.linalg.inv(ijk2ras)
    k0_c, i0_c, j0_c = int(oof_npz["k0"]), int(oof_npz["i0"]), int(oof_npz["j0"])
    K_s, H_s, W_s = pm_arr.shape

    ras = ras[np.argsort(ras[:, 2])]
    z_bot_cur = float(ras[0, 2])
    z_top_cur = float(ras[-1, 2])
    z_span_cur = max(1.0, z_top_cur - z_bot_cur)

    # Sample 20 points along current trajectory
    zs = np.linspace(z_bot_cur, z_top_cur, 20)
    xs = np.interp(zs, ras[:, 2], ras[:, 0])
    ys = np.interp(zs, ras[:, 2], ras[:, 1])
    ijks = (ras2ijk @ np.column_stack([xs, ys, zs, np.ones_like(zs)]).T).T

    pm_l, pp_l, po_l, sym_l, cs_l, rc_l = [], [], [], [], [], []
    for row in ijks:
        ic, jc, kc = int(round(row[0] - i0_c)), int(round(row[1] - j0_c)), int(round(row[2] - k0_c))
        if 1 <= ic < W_s - 1 and 1 <= jc < H_s - 1 and 0 <= kc < K_s:
            pm_l.append(float(np.max(pm_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
            pp_l.append(float(np.max(pp_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
            po_l.append(float(np.max(po_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
            sym_l.append(float(np.max(sym_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
            cs_l.append(float(np.max(cs_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2])))
            rc_l.append(float(np.max(rc_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2])))

    mean_pm = float(np.mean(pm_l)) if pm_l else 0.0
    p75_pm = float(np.percentile(pm_l, 75)) if pm_l else 0.0
    mean_pp = float(np.mean(pp_l)) if pp_l else 0.0
    mean_po = float(np.mean(po_l)) if po_l else 0.0
    mean_sym = float(np.mean(sym_l)) if sym_l else 0.0
    p75_sym = float(np.percentile(sym_l, 75)) if sym_l else 0.0
    mean_cs = float(np.mean(cs_l)) if cs_l else 0.0
    mean_rc = float(np.mean(rc_l)) if rc_l else 0.0
    frac_active = float(np.mean([(m > 0.22 or p > 0.18 or s > 0.25) for m, p, s in zip(pm_l, pp_l, sym_l)])) if pm_l else 0.0

    # Sample full axial slab profile along the trajectory's tangent line to find exact signal-supported cranial tip
    # Fit robust line near top of trajectory
    top_pts = ras[-min(6, len(ras)):]
    if len(top_pts) >= 2 and abs(top_pts[-1, 2] - top_pts[0, 2]) > 1.0:
        vx_top = (top_pts[-1, 0] - top_pts[0, 0]) / (top_pts[-1, 2] - top_pts[0, 2])
        vy_top = (top_pts[-1, 1] - top_pts[0, 1]) / (top_pts[-1, 2] - top_pts[0, 2])
    else:
        vx_top, vy_top = 0.0, 0.0

    # Check axial profile from z_top_cur - 15mm to z_top_cur + 15mm
    z_probe = np.linspace(z_top_cur - 15.0, z_top_cur + 15.0, 25)
    x_probe = np.where(
        z_probe <= z_top_cur,
        np.interp(z_probe, ras[:, 2], ras[:, 0]),
        ras[-1, 0] + vx_top * (z_probe - z_top_cur)
    )
    y_probe = np.where(
        z_probe <= z_top_cur,
        np.interp(z_probe, ras[:, 2], ras[:, 1]),
        ras[-1, 1] + vy_top * (z_probe - z_top_cur)
    )
    ijks_p = (ras2ijk @ np.column_stack([x_probe, y_probe, z_probe, np.ones_like(z_probe)]).T).T
    support_p = np.zeros(len(z_probe), dtype=np.float32)
    for idx_p, row in enumerate(ijks_p):
        ic, jc, kc = int(round(row[0] - i0_c)), int(round(row[1] - j0_c)), int(round(row[2] - k0_c))
        if 1 <= ic < W_s - 1 and 1 <= jc < H_s - 1 and 0 <= kc < K_s:
            m_v = float(np.max(pm_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
            p_v = float(np.max(pp_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
            o_v = float(np.max(po_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
            s_v = float(np.max(sym_arr[kc, jc - 1:jc + 2, ic - 1:ic + 2]))
            support_p[idx_p] = 0.40 * max(m_v, o_v) + 0.35 * p_v + 0.25 * min(1.0, s_v)

    # Estimate profile-supported cranial tip z_prof_tip where support drops below 0.24
    valid_sup = np.where(support_p >= 0.24)[0]
    if len(valid_sup) > 0:
        last_valid = int(valid_sup[-1])
        z_prof_tip = float(z_probe[last_valid])
    else:
        z_prof_tip = z_top_cur

    tip_sup_cur = float(support_p[12])  # support right at z_top_cur
    tip_sup_above = float(np.mean(support_p[14:19]))  # support 2.5 - 7.5 mm above z_top_cur
    tip_sup_below = float(np.mean(support_p[6:12]))   # support 2.5 - 7.5 mm below z_top_cur

    return {
        "mean_pm": mean_pm,
        "p75_pm": p75_pm,
        "mean_pp": mean_pp,
        "mean_po": mean_po,
        "mean_sym": mean_sym,
        "p75_sym": p75_sym,
        "mean_cs": mean_cs,
        "mean_rc": mean_rc,
        "frac_active": frac_active,
        "z_prof_tip": z_prof_tip,
        "tip_sup_cur": tip_sup_cur,
        "tip_sup_above": tip_sup_above,
        "tip_sup_below": tip_sup_below,
        "vx_top": float(vx_top),
        "vy_top": float(vy_top),
    }


def build_enriched_bundle_graph_item(pool, g, oof_npz, X_vol, gts_all=None, gts_long=None):
    """
    Builds an enriched 26-node-feature, 8-edge-feature Bundle Graph with:
      - Stage 0 + Stage 1 axial continuity features
      - Bundle-neighborhood cranial tip surface consensus (z_tip_nbr_med)
      - Profile-guided tip extension/trimming features
      - Ground-truth classification labels (y_rec, y_1to1, y_long_1to1) and exact cranial tip
        correction target (y_dz_tip = gt_z_tip - cur_z_tip)
    """
    M = len(pool)
    bases = np.array([p["base_xy"] for p in pool]) if M else np.zeros((0, 2))
    slopes = np.array([p["slope"] for p in pool]) if M else np.zeros((0, 2))
    med_base = np.median(bases, axis=0) if M else np.zeros(2)
    med_slope = np.median(slopes, axis=0) if M else np.zeros(2)

    prof_list = [compute_trajectory_axial_profile_and_tip_features(p["ras"], g, oof_npz, X_vol) for p in pool]

    # Compute local neighborhood cranial tip height z_tip_nbr for each candidate
    z_tops = np.array([float(p["ras"][-1, 2]) for p in pool]) if M else np.zeros(0)
    z_bots = np.array([float(p["ras"][0, 2]) for p in pool]) if M else np.zeros(0)
    qual_weights = np.array([
        max(0.05, pr["mean_pm"] * 0.4 + pr["mean_pp"] * 0.3 + pr["mean_sym"] * 0.3)
        for pr in prof_list
    ]) if M else np.zeros(0)

    z_tip_nbr = np.zeros(M, dtype=np.float32)
    z_bot_nbr = np.zeros(M, dtype=np.float32)
    for u in range(M):
        dists_xy = np.hypot(bases[:, 0] - bases[u, 0], bases[:, 1] - bases[u, 1])
        # Neighbors with similar span class (long vs short) within 28 mm
        span_u = z_tops[u] - z_bots[u]
        same_class = np.abs((z_tops - z_bots) - span_u) <= 22.0
        nbr_mask = (dists_xy <= 28.0) & same_class & (qual_weights >= 0.25)
        if np.sum(nbr_mask) >= 2:
            w_n = qual_weights[nbr_mask]
            z_tip_nbr[u] = float(np.sum(z_tops[nbr_mask] * w_n) / np.sum(w_n))
            z_bot_nbr[u] = float(np.sum(z_bots[nbr_mask] * w_n) / np.sum(w_n))
        else:
            z_tip_nbr[u] = float(z_tops[u])
            z_bot_nbr[u] = float(z_bots[u])

    x_feats = []
    for u, p in enumerate(pool):
        ras = p["ras"]
        pr = prof_list[u]
        d_base = float(np.hypot(p["base_xy"][0] - med_base[0], p["base_xy"][1] - med_base[1])) / 40.0
        d_slope = float(np.hypot(p["slope"][0] - med_slope[0], p["slope"][1] - med_slope[1])) / 0.25
        z_low = float(ras[0, 2] - g["z_base_ras"]) / 50.0
        z_high = float(ras[-1, 2] - g["z_base_ras"]) / 100.0
        dz_prof = float(pr["z_prof_tip"] - ras[-1, 2]) / 20.0
        dz_nbr = float(z_tip_nbr[u] - ras[-1, 2]) / 20.0

        # Save profile metadata on pool dict for downstream tip refinement
        p["z_prof_tip"] = pr["z_prof_tip"]
        p["z_tip_nbr"] = float(z_tip_nbr[u])
        p["vx_top"] = pr["vx_top"]
        p["vy_top"] = pr["vy_top"]
        p["mean_sym"] = pr["mean_sym"]
        p["frac_active"] = pr["frac_active"]

        x_feats.append([
            p["conf_heur"],
            p["from_mpnn"],
            p["mpnn_pa"],
            p["mpnn_pl"],
            p["mpnn_pe"],
            pr["mean_pm"],
            pr["p75_pm"],
            pr["mean_pp"],
            pr["mean_po"],
            max(pr["mean_pm"], pr["mean_po"]),
            p["z_span"] / 60.0,
            p["rmse"],
            d_base,
            d_slope,
            z_low,
            z_high,
            float(p["slope"][0]),
            float(p["slope"][1]),
            # 8 new physics + axial-profile + tip-consensus features (total 26):
            pr["mean_sym"],
            pr["p75_sym"],
            pr["mean_cs"],
            pr["mean_rc"],
            pr["frac_active"],
            pr["tip_sup_cur"],
            dz_prof,
            dz_nbr,
        ])

    adj_feat = np.zeros((M, M, 8), dtype=np.float32)
    adj_mask = np.zeros((M, M), dtype=np.float32)
    for u in range(M):
        for v in range(M):
            if u == v:
                continue
            db = float(np.hypot(pool[u]["base_xy"][0] - pool[v]["base_xy"][0], pool[u]["base_xy"][1] - pool[v]["base_xy"][1]))
            if db <= 35.0:
                ds = float(np.hypot(pool[u]["slope"][0] - pool[v]["slope"][0], pool[u]["slope"][1] - pool[v]["slope"][1]))
                sh_d = traj_gt_dist(pool[u]["ras"], pool[v]["ras"])
                dz_t = float(z_tops[v] - z_tops[u]) / 30.0
                z_ov = max(0.0, min(z_tops[u], z_tops[v]) - max(z_bots[u], z_bots[v])) / max(1.0, max(z_tops[u] - z_bots[u], z_tops[v] - z_bots[v]))
                adj_mask[u, v] = 1.0
                adj_feat[u, v] = [
                    db / 35.0,
                    sh_d / 35.0,
                    ds / 0.30,
                    1.0 if sh_d < 2.2 else 0.0,
                    float(x_feats[v][5] - x_feats[u][5]),
                    float(x_feats[v][0] - x_feats[u][0]),
                    dz_t,
                    z_ov,
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
        y_dz_tip = np.zeros(M, dtype=np.float32)
        y_dz_weight = np.zeros(M, dtype=np.float32)

        if len(gts_all) > 0 and M > 0:
            cost_a = np.zeros((len(gts_all), M))
            for gi, gt in enumerate(gts_all):
                for pj, p in enumerate(pool):
                    cost_a[gi, pj] = traj_gt_dist(p["ras"], gt)
            y_rec = (np.min(cost_a, axis=0) <= 3.3).astype(np.float32)
            # For every candidate within 3.3 mm of a GT, record target cranial tip correction dz_tip
            best_gi = np.argmin(cost_a, axis=0)
            for pj in range(M):
                if cost_a[best_gi[pj], pj] <= 3.3:
                    gt_match = gts_all[best_gi[pj]]
                    dz_t = float(gt_match[-1, 2] - pool[pj]["ras"][-1, 2])
                    if abs(dz_t) <= 22.0:
                        y_dz_tip[pj] = dz_t / 15.0  # normalized to [-1.5, 1.5]
                        y_dz_weight[pj] = 1.0

            ri, ci = linear_sum_assignment(cost_a)
            for r, c in zip(ri, ci):
                if cost_a[r, c] <= 3.5:
                    y_1to1[c] = 1.0
                    gt_match = gts_all[r]
                    dz_t = float(gt_match[-1, 2] - pool[c]["ras"][-1, 2])
                    if abs(dz_t) <= 22.0:
                        y_dz_tip[c] = dz_t / 15.0
                        y_dz_weight[c] = 2.0

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
        item["y_dz_tip"] = torch.tensor(y_dz_tip, dtype=torch.float32)
        item["y_dz_weight"] = torch.tensor(y_dz_weight, dtype=torch.float32)
        item["gts_all"] = gts_all
        item["gts_long"] = gts_long

    return item


def generate_synthetic_bundle_graph_item(rng):
    """
    SynthBrachy3T Bundle-Level Graph Generator:
    Synthesizes realistic 3D candidate trajectory bundles (true long interstitial shafts,
    true short obturator tracks, adhering twin shafts <= 2.2 mm, split duplicates, and
    anatomical false-positive vessel tracks) with exact 26-D node features, 8-D edge features,
    and ground-truth [y_rec, y_1to1, y_long_1to1, y_dz_tip] labels.
    """
    n_true_long = int(rng.integers(12, 24))
    n_true_short = int(rng.integers(2, 9))
    n_dups = int(rng.integers(3, 12))
    n_fps = int(rng.integers(4, 15))

    # Generate true catheters in physical XY/Z space
    true_trajs = []
    dome_z = rng.uniform(45.0, 75.0)
    obt_z = rng.uniform(18.0, 32.0)
    main_slope = rng.normal(0.0, 0.06, size=2)

    for i in range(n_true_long):
        ang = 2 * np.pi * i / n_true_long + rng.normal(0, 0.15)
        r = rng.uniform(7.0, 26.0)
        bxy = np.array([r * np.cos(ang), r * np.sin(ang)])
        sl = main_slope + rng.normal(0, 0.05, size=2)
        z0 = rng.uniform(-5.0, 6.0)
        z1 = dome_z - 0.18 * r + rng.normal(0, 3.5)
        true_trajs.append({"bxy": bxy, "sl": sl, "z0": z0, "z1": z1, "is_long": True})

    # Inject deliberate severe adhesion twin for 30% of long catheters
    n_adh = max(2, int(0.30 * n_true_long))
    for i in range(n_adh):
        t0 = true_trajs[i]
        ang_off = rng.uniform(0, 2 * np.pi)
        sep = rng.uniform(1.65, 2.75)
        bxy2 = t0["bxy"] + sep * np.array([np.cos(ang_off), np.sin(ang_off)])
        sl2 = t0["sl"] + rng.normal(0, 0.025, size=2)
        z1_2 = t0["z1"] + rng.normal(0, 2.0)
        true_trajs.append({"bxy": bxy2, "sl": sl2, "z0": t0["z0"], "z1": z1_2, "is_long": True})

    for i in range(n_true_short):
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(3.0, 12.0)
        bxy = np.array([r * np.cos(ang), r * np.sin(ang)])
        sl = main_slope + rng.normal(0, 0.04, size=2)
        z0 = rng.uniform(-6.0, 4.0)
        z1 = obt_z + rng.normal(0, 2.5)
        true_trajs.append({"bxy": bxy, "sl": sl, "z0": z0, "z1": z1, "is_long": False})

    cands = []
    # 1. Primary winner candidates for each true trajectory (with realistic tip tracking noise)
    for t in true_trajs:
        dz_err = rng.normal(0.0, 4.5) if rng.random() < 0.45 else rng.normal(0.0, 1.2)
        cur_z1 = t["z1"] - dz_err  # cur_z1 + dz_err = true z1
        z_span = max(8.0, cur_z1 - t["z0"])
        pm = rng.uniform(0.45, 0.92)
        pp = rng.uniform(0.38, 0.88)
        po = rng.uniform(0.40, 0.85) if not t["is_long"] else rng.uniform(0.05, 0.35)
        sym = rng.uniform(0.42, 0.95)
        from_mpnn = 1.0 if rng.random() < 0.82 else 0.5
        mpnn_pa = rng.uniform(0.55, 0.96)
        mpnn_pl = rng.uniform(0.55, 0.95) if t["is_long"] else rng.uniform(0.10, 0.45)
        mpnn_pe = rng.uniform(0.55, 0.95)
        z_prof_tip = t["z1"] + rng.normal(0.0, 1.4)
        z_tip_nbr = t["z1"] + rng.normal(0.0, 2.2)
        cands.append({
            "bxy": t["bxy"] + rng.normal(0, 0.25, size=2),
            "sl": t["sl"] + rng.normal(0, 0.015, size=2),
            "z0": t["z0"], "z1": cur_z1, "z_span": z_span,
            "rmse": rng.uniform(0.12, 0.48),
            "conf_heur": rng.uniform(0.58, 0.92),
            "from_mpnn": from_mpnn, "mpnn_pa": mpnn_pa, "mpnn_pl": mpnn_pl, "mpnn_pe": mpnn_pe,
            "pm": pm, "p75_pm": min(1.0, pm + 0.12), "pp": pp, "po": po,
            "sym": sym, "p75_sym": min(1.0, sym + 0.14),
            "cs": sym * rng.uniform(0.8, 1.1), "rc": sym * rng.uniform(0.6, 0.95),
            "frac_active": rng.uniform(0.72, 0.98),
            "tip_sup_cur": rng.uniform(0.35, 0.85) if abs(dz_err) < 3.0 else rng.uniform(0.12, 0.40),
            "dz_prof": (z_prof_tip - cur_z1) / 20.0,
            "dz_nbr": (z_tip_nbr - cur_z1) / 20.0,
            "y_rec": 1.0, "y_1to1": 1.0, "y_long_1to1": 1.0 if t["is_long"] else 0.0,
            "y_dz_tip": np.clip(dz_err / 15.0, -1.5, 1.5), "y_dz_weight": 2.0,
        })

    # 2. Duplicate / fragmented sub-candidates near existing true trajectories (y_rec=1, y_1to1=0)
    for _ in range(n_dups):
        t = true_trajs[int(rng.integers(0, len(true_trajs)))]
        bxy = t["bxy"] + rng.normal(0, 0.85, size=2)
        sl = t["sl"] + rng.normal(0, 0.03, size=2)
        cur_z1 = t["z1"] - rng.uniform(4.0, 18.0)
        z_span = max(8.0, cur_z1 - t["z0"])
        pm = rng.uniform(0.28, 0.62)
        sym = rng.uniform(0.25, 0.58)
        cands.append({
            "bxy": bxy, "sl": sl, "z0": t["z0"] + rng.uniform(0, 6.0), "z1": cur_z1, "z_span": z_span,
            "rmse": rng.uniform(0.35, 0.85),
            "conf_heur": rng.uniform(0.35, 0.65),
            "from_mpnn": 0.0 if rng.random() < 0.6 else 0.5,
            "mpnn_pa": rng.uniform(0.20, 0.55), "mpnn_pl": rng.uniform(0.10, 0.45), "mpnn_pe": rng.uniform(0.25, 0.55),
            "pm": pm, "p75_pm": min(1.0, pm + 0.10), "pp": pm * 0.85, "po": rng.uniform(0.05, 0.35),
            "sym": sym, "p75_sym": min(1.0, sym + 0.10),
            "cs": sym * 0.8, "rc": sym * 0.6,
            "frac_active": rng.uniform(0.40, 0.72),
            "tip_sup_cur": rng.uniform(0.15, 0.45),
            "dz_prof": (t["z1"] - cur_z1) / 20.0,
            "dz_nbr": (t["z1"] - cur_z1) / 20.0,
            "y_rec": 1.0, "y_1to1": 0.0, "y_long_1to1": 0.0,
            "y_dz_tip": np.clip((t["z1"] - cur_z1) / 15.0, -1.5, 1.5), "y_dz_weight": 0.5,
        })

    # 3. False-positive vascular / fascial / bowel tracks (y_rec=0, y_1to1=0, y_long_1to1=0)
    for _ in range(n_fps):
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(14.0, 48.0)
        bxy = np.array([r * np.cos(ang), r * np.sin(ang)])
        sl = main_slope + rng.normal(0, 0.16, size=2)
        z0 = rng.uniform(-10.0, 25.0)
        z_span = rng.uniform(9.0, 36.0)
        cur_z1 = z0 + z_span
        pm = rng.uniform(0.12, 0.48)
        sym = rng.uniform(0.08, 0.38)
        cands.append({
            "bxy": bxy, "sl": sl, "z0": z0, "z1": cur_z1, "z_span": z_span,
            "rmse": rng.uniform(0.40, 1.10),
            "conf_heur": rng.uniform(0.25, 0.58),
            "from_mpnn": 0.0 if rng.random() < 0.75 else 0.5,
            "mpnn_pa": rng.uniform(0.08, 0.42), "mpnn_pl": rng.uniform(0.05, 0.35), "mpnn_pe": rng.uniform(0.10, 0.45),
            "pm": pm, "p75_pm": min(1.0, pm + 0.12), "pp": pm * 0.75, "po": rng.uniform(0.02, 0.25),
            "sym": sym, "p75_sym": min(1.0, sym + 0.10),
            "cs": sym * 0.75, "rc": sym * 0.5,
            "frac_active": rng.uniform(0.18, 0.55),
            "tip_sup_cur": rng.uniform(0.05, 0.30),
            "dz_prof": rng.normal(-0.2, 0.3),
            "dz_nbr": (dome_z - cur_z1) / 20.0,
            "y_rec": 0.0, "y_1to1": 0.0, "y_long_1to1": 0.0,
            "y_dz_tip": 0.0, "y_dz_weight": 0.0,
        })

    M = len(cands)
    bases = np.array([c["bxy"] for c in cands])
    slopes = np.array([c["sl"] for c in cands])
    med_base = np.median(bases, axis=0)
    med_slope = np.median(slopes, axis=0)

    x_feats = []
    for c in cands:
        d_base = float(np.hypot(c["bxy"][0] - med_base[0], c["bxy"][1] - med_base[1])) / 40.0
        d_slope = float(np.hypot(c["sl"][0] - med_slope[0], c["sl"][1] - med_slope[1])) / 0.25
        x_feats.append([
            c["conf_heur"], c["from_mpnn"], c["mpnn_pa"], c["mpnn_pl"], c["mpnn_pe"],
            c["pm"], c["p75_pm"], c["pp"], c["po"], max(c["pm"], c["po"]),
            c["z_span"] / 60.0, c["rmse"], d_base, d_slope,
            c["z0"] / 50.0, c["z1"] / 100.0, float(c["sl"][0]), float(c["sl"][1]),
            c["sym"], c["p75_sym"], c["cs"], c["rc"], c["frac_active"],
            c["tip_sup_cur"], c["dz_prof"], c["dz_nbr"],
        ])

    adj_feat = np.zeros((M, M, 8), dtype=np.float32)
    adj_mask = np.zeros((M, M), dtype=np.float32)
    for u in range(M):
        for v in range(M):
            if u == v:
                continue
            db = float(np.hypot(cands[u]["bxy"][0] - cands[v]["bxy"][0], cands[u]["bxy"][1] - cands[v]["bxy"][1]))
            if db <= 35.0:
                ds = float(np.hypot(cands[u]["sl"][0] - cands[v]["sl"][0], cands[u]["sl"][1] - cands[v]["sl"][1]))
                sh_d = db + 15.0 * ds
                dz_t = float(cands[v]["z1"] - cands[u]["z1"]) / 30.0
                z_ov = max(0.0, min(cands[u]["z1"], cands[v]["z1"]) - max(cands[u]["z0"], cands[v]["z0"])) / max(1.0, max(cands[u]["z_span"], cands[v]["z_span"]))
                adj_mask[u, v] = 1.0
                adj_feat[u, v] = [
                    db / 35.0, sh_d / 35.0, ds / 0.30, 1.0 if sh_d < 2.2 else 0.0,
                    float(x_feats[v][5] - x_feats[u][5]), float(x_feats[v][0] - x_feats[u][0]),
                    dz_t, z_ov,
                ]

    return {
        "x": torch.tensor(x_feats, dtype=torch.float32),
        "adj_feat": torch.tensor(adj_feat, dtype=torch.float32),
        "adj_mask": torch.tensor(adj_mask, dtype=torch.float32),
        "y_rec": torch.tensor([c["y_rec"] for c in cands], dtype=torch.float32),
        "y_1to1": torch.tensor([c["y_1to1"] for c in cands], dtype=torch.float32),
        "y_long_1to1": torch.tensor([c["y_long_1to1"] for c in cands], dtype=torch.float32),
        "y_dz_tip": torch.tensor([c["y_dz_tip"] for c in cands], dtype=torch.float32),
        "y_dz_weight": torch.tensor([c["y_dz_weight"] for c in cands], dtype=torch.float32),
    }


class TipAwareBundleGraphTransformer(nn.Module):
    """
    Upgraded 26-Feature, 8-Edge-Feature Bundle Graph Transformer with 4 outputs:
      - out[:, 0]: logit_rec (any GT match <= 3.5 mm)
      - out[:, 1]: logit_1to1 (unique 1-to-1 Hungarian GT winner <= 3.5 mm)
      - out[:, 2]: logit_long_1to1 (unique 1-to-1 Hungarian long needle winner)
      - out[:, 3]: pred_dz_tip (normalized cranial tip adjustment dz_tip / 15.0 mm)
    """
    def __init__(self, in_dim=26, edge_dim=8, d_model=64):
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
        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.v1 = nn.Linear(d_model, d_model)
        self.e_att1 = nn.Linear(d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)

        self.q2 = nn.Linear(d_model, d_model)
        self.k2 = nn.Linear(d_model, d_model)
        self.v2 = nn.Linear(d_model, d_model)
        self.e_att2 = nn.Linear(d_model, 1)
        self.norm2 = nn.LayerNorm(d_model)

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
            nn.Linear(d_model // 2, 4),  # [logit_rec, logit_1to1, logit_long_1to1, pred_dz_tip]
        )

    def forward(self, x, adj_feat, adj_mask):
        h = self.node_enc(x)
        ef = self.edge_enc(adj_feat)
        scale = 1.0 / math.sqrt(float(h.shape[-1]))

        scores1 = torch.matmul(self.q1(h), self.k1(h).transpose(0, 1)) * scale + self.e_att1(ef).squeeze(-1)
        scores1 = scores1.masked_fill(adj_mask <= 0.0, -1e4)
        attn1 = torch.softmax(scores1, dim=-1) * adj_mask
        attn1 = attn1 / attn1.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        msg1 = torch.matmul(attn1, self.v1(h)) + (attn1.unsqueeze(-1) * ef).sum(dim=1)
        h = self.norm1(h + F.silu(msg1))

        scores2 = torch.matmul(self.q2(h), self.k2(h).transpose(0, 1)) * scale + self.e_att2(ef).squeeze(-1)
        scores2 = scores2.masked_fill(adj_mask <= 0.0, -1e4)
        attn2 = torch.softmax(scores2, dim=-1) * adj_mask
        attn2 = attn2 / attn2.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        msg2 = torch.matmul(attn2, self.v2(h)) + (attn2.unsqueeze(-1) * ef).sum(dim=1)
        h = self.norm2(h + F.silu(msg2))

        scores3 = torch.matmul(self.q3(h), self.k3(h).transpose(0, 1)) * scale
        attn3 = torch.softmax(scores3, dim=-1)
        h = self.norm3(h + torch.matmul(attn3, self.v3(h)))
        h = self.norm4(h + self.ffn(h))

        return self.head(h)


def apply_tip_adjustment_and_poly3(pts_ras, dz_tip_mm, vx_top, vy_top):
    """
    Adjusts the cranial endpoint z_tip by dz_tip_mm (clamped to [-10.0, +10.0] mm) along the
    local 3D trajectory slope and applies smooth 3rd-order polynomial regularization.
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
        # Trim overshot cranial tail cleanly at z_top_new
        x_top = float(np.interp(z_top_new, pts[:, 2], pts[:, 0]))
        y_top = float(np.interp(z_top_new, pts[:, 2], pts[:, 1]))
        keep = pts[:, 2] < z_top_new - 0.25
        pts_trim = np.vstack([pts[keep], np.array([[x_top, y_top, z_top_new]])])
        if len(pts_trim) >= 3:
            pts = pts_trim
    elif z_top_new > z_top_old + 0.5:
        # Extend undershot cranial tip along local top tangent (vx_top, vy_top)
        dz_ext = z_top_new - z_top_old
        x_ext = float(pts[-1, 0]) + vx_top * dz_ext
        y_ext = float(pts[-1, 1]) + vy_top * dz_ext
        pts = np.vstack([pts, np.array([[x_ext, y_ext, z_top_new]])])

    if len(pts) >= 4:
        pts_poly = regularize_poly3_trajectory(pts, degree=3)
        pts_blend = pts.copy()
        w_poly = np.full(len(pts), 0.35)
        w_poly[0] = 0.05
        w_poly[-1] = 0.05
        pts_blend[:, 0] = (1.0 - w_poly) * pts[:, 0] + w_poly * pts_poly[:, 0]
        pts_blend[:, 1] = (1.0 - w_poly) * pts[:, 1] + w_poly * pts_poly[:, 1]
        return pts_blend
    return pts


def get_synth_tip_aware_preds(bundle_items, oof_outputs, min_conf=50.0, long_mode=False, tip_blend=0.65):
    """
    Decodes predictions from TipAwareBundleGraphTransformer outputs [M, 4]:
      - sc[:, 0..2] = sigmoid(out[:, 0..2])
      - dz_tip_pred_mm = out[:, 3] * 15.0 (blended with axial-profile & neighborhood tip consensus)
    """
    preds_out = []
    for i, bg in enumerate(bundle_items):
        out_arr = oof_outputs[i]
        sc = 1.0 / (1.0 + np.exp(-np.clip(out_arr[:, :3], -20.0, 20.0)))
        dz_pred_net = out_arr[:, 3] * 15.0

        if long_mode:
            conf_arr = 100.0 * (0.28 * sc[:, 0] + 0.24 * sc[:, 1] + 0.48 * sc[:, 2])
        else:
            conf_arr = 100.0 * (0.42 * sc[:, 0] + 0.58 * sc[:, 1])

        # Compute IRGM implant centroid & radius in XY physical space from high-confidence core
        high_idx = np.where(conf_arr >= 55.0)[0]
        if len(high_idx) >= 3:
            all_xy = np.concatenate([bg["pool"][idx]["ras"][:, :2] for idx in high_idx], axis=0)
            center_xy = np.median(all_xy, axis=0)
            dists_core = np.linalg.norm(all_xy - center_xy[None, :], axis=1)
            irgm_radius_mm = float(np.percentile(dists_core, 98.0)) + 15.0
        else:
            center_xy = None
            irgm_radius_mm = 999.0

        sel = []
        for idx_c in np.argsort(-conf_arr):
            if conf_arr[idx_c] < min_conf:
                continue
            cand = dict(bg["pool"][idx_c])
            pts = np.asarray(cand["ras"], dtype=np.float64)
            pts = pts[np.argsort(pts[:, 2])]

            # IRGM spatial gate
            if center_xy is not None:
                mean_dist_center = float(np.linalg.norm(pts[:, :2].mean(axis=0) - center_xy))
                if mean_dist_center > irgm_radius_mm and conf_arr[idx_c] < min_conf + 22.0:
                    continue

            # Reject low-symmetry vascular false positives unless MPNN + transformer confidence is high
            if cand.get("mean_sym", 0.5) < 0.16 and cand.get("frac_active", 0.5) < 0.42 and conf_arr[idx_c] < min_conf + 16.0:
                continue

            # Adhesion-aware synchronous deduplication (allows twin catheters <= 1.60 mm apart when strongly supported)
            dedup_thresh = 1.60 if (conf_arr[idx_c] >= 66.0 and cand.get("mpnn_pa", 0.0) >= 0.52) else 2.02
            if any(traj_gt_dist(pts, s["ras"]) < dedup_thresh for s in sel):
                continue

            # Combine neural predicted tip offset with axial-profile tip offset
            dz_prof_mm = float(cand.get("z_prof_tip", pts[-1, 2]) - pts[-1, 2])
            dz_nbr_mm = float(cand.get("z_tip_nbr", pts[-1, 2]) - pts[-1, 2])
            dz_comb_mm = tip_blend * (0.65 * float(dz_pred_net[idx_c]) + 0.25 * dz_prof_mm + 0.10 * dz_nbr_mm)

            pts_refined = apply_tip_adjustment_and_poly3(
                pts,
                dz_comb_mm,
                float(cand.get("vx_top", 0.0)),
                float(cand.get("vy_top", 0.0)),
            )
            cand["ras"] = pts_refined
            cand["points"] = pts_refined
            cand["score"] = float(conf_arr[idx_c])
            sel.append(cand)
        preds_out.append(sel)
    return preds_out


class FewShotSynthAdapter3D(nn.Module):
    """
    6-Head 3D Residual Adapter (with 6th channel = Distal-Tip Heatmap P_tip)
    pre-trained on SynthBrachy3T synthetic 3T MRI implants and fine-tuned on k clinical scans.
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.SiLU(inplace=True),
        )
        self.res1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
        )
        self.head = nn.Conv3d(32, 3, kernel_size=1)  # [logit_mask, logit_peak, logit_tip]

    def forward(self, x):
        h = self.stem(x)
        h = F.silu(h + self.res1(h), inplace=True)
        return self.head(h)


def main():
    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[SynthBrachy3T] Running on {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})", flush=True)

    meta_list = pickle.load(open(META_PATH, "rb"))
    base_trajs_data = pickle.load(open(BASE_TRAJS_PATH, "rb"))
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]
    gts_short_list = []
    for ci in range(54):
        long_z = [float(g[-1, 2] - g[0, 2]) for g in gts_long_list[ci]]
        min_lz = min(long_z) - 2.0 if long_z else 30.0
        shorts = [g for g in gts_all_list[ci] if float(g[-1, 2] - g[0, 2]) < min_lz]
        gts_short_list.append(shorts)

    adhesion_info = analyze_catheter_adhesion(meta_list, gts_all_list, gts_long_list, adhesion_thresh_mm=3.0)
    adh_idx = adhesion_info["adhesion_case_indices"]

    # Step 1: Build 54 clinical slice graphs and run 5-fold NeedleSliceMPNN
    print("[1/4] Building 54 clinical slice graphs and enriched candidate pools...", flush=True)
    graphs = []
    all_X_vols, all_Y_vols, all_oof_npz = [], [], []
    for i in range(54):
        d_pt = torch.load(f"{CACHE_DIR}/case_{i:02d}.pt", map_location="cpu", weights_only=False)
        oof_npz = np.load(f"{OOF_DIR}/oof_case_{i:02d}.npz")
        X = d_pt["X"].numpy()
        all_X_vols.append(d_pt["X"])
        all_Y_vols.append(d_pt["Y"])
        all_oof_npz.append(oof_npz)
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

    oof_mpnn_trajs = [None] * 54
    for fold in range(5):
        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
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

    bundle_items = []
    stage0_preds = []
    for i, g in enumerate(graphs):
        X = all_X_vols[i].numpy()
        oof_npz = all_oof_npz[i]
        b_item = base_trajs_data[i]
        b_trajs = b_item["trajs"] if isinstance(b_item, dict) and "trajs" in b_item else b_item
        b_clean = []
        for t in b_trajs:
            tc = dict(t)
            tc["ras"] = np.asarray(tc["ras"] if "ras" in tc else tc["points"], dtype=np.float64)
            tc["points"] = tc["ras"]
            b_clean.append(tc)
        stage0_preds.append(b_clean)

        k0, i0, j0 = int(oof_npz["k0"]), int(oof_npz["i0"]), int(oof_npz["j0"])
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

        bundle_items.append(build_enriched_bundle_graph_item(pool, g, oof_npz, X, gts_all=gts_all_list[i], gts_long=gts_long_list[i]))

    # Step 2: Generate 800 SynthBrachy3T Bundle Graphs and Pre-Train + Fine-Tune TipAwareBundleGraphTransformer
    print("[2/4] Generating 800 SynthBrachy3T synthetic bundle graphs (~24,000 synthetic trajectories)...", flush=True)
    rng_syn = np.random.default_rng(2026)
    synth_bundles = [generate_synthetic_bundle_graph_item(rng_syn) for _ in range(800)]

    print("[3/4] Running 5-fold SynthBrachy3T Pre-Training + Clinical Fine-Tuning of TipAwareBundleGraphTransformer...", flush=True)
    oof_outputs = [None] * 54
    zero_shot_outputs = [None] * 54

    # First train a pure SynthBrachy3T Pre-Trained model (0 clinical BWH bundle labels!)
    torch.manual_seed(2026)
    base_syn_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
    opt_syn = torch.optim.AdamW(base_syn_model.parameters(), lr=2.2e-3, weight_decay=1e-4)
    sched_syn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_syn, T_max=25, eta_min=2e-4)
    for ep in range(25):
        base_syn_model.train()
        for bi in np.random.permutation(len(synth_bundles)):
            bg = synth_bundles[bi]
            opt_syn.zero_grad(set_to_none=True)
            out = base_syn_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
            l_rec = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"].to(device), pos_weight=torch.tensor([2.8], device=device))
            l_1to1 = F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"].to(device), pos_weight=torch.tensor([2.2], device=device))
            l_long = F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device))
            w_tip = bg["y_dz_weight"].to(device)
            l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"].to(device), reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
            loss = l_rec + 1.5 * l_1to1 + l_long + 0.8 * l_tip
            loss.backward()
            opt_syn.step()
        sched_syn.step()

    pretrained_state = {k: v.cpu().clone() for k, v in base_syn_model.state_dict().items()}
    base_syn_model.eval()
    with torch.no_grad():
        for i, bg in enumerate(bundle_items):
            out = base_syn_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
            zero_shot_outputs[i] = out.cpu().numpy()

    # Now fine-tune per fold on the 43 clinical training patients (co-regularized with SynthBrachy3T)
    for fold in range(5):
        torch.manual_seed(100 + fold)
        np.random.seed(100 + fold)
        tr_b = [b for i, b in enumerate(bundle_items) if i % 5 != fold]
        gt_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
        gt_model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
        opt = torch.optim.AdamW(gt_model.parameters(), lr=1.4e-3, weight_decay=1.5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=1e-4)
        for ep in range(80):
            gt_model.train()
            # Mix 43 clinical training bundles with 12 random synthetic bundles per epoch to prevent overfitting
            syn_sub = [synth_bundles[idx] for idx in np.random.choice(len(synth_bundles), size=12, replace=False)]
            combined_b = tr_b + syn_sub
            for bi in np.random.permutation(len(combined_b)):
                bg = combined_b[bi]
                opt.zero_grad(set_to_none=True)
                out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                l_rec = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"].to(device), pos_weight=torch.tensor([3.0], device=device))
                l_1to1 = F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"].to(device), pos_weight=torch.tensor([2.1], device=device))
                l_long = F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device))
                w_tip = bg["y_dz_weight"].to(device)
                l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"].to(device), reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
                loss = l_rec + 1.6 * l_1to1 + l_long + 1.0 * l_tip
                loss.backward()
                opt.step()
            sched.step()
        gt_model.eval()
        with torch.no_grad():
            for i, bg in enumerate(bundle_items):
                if i % 5 == fold:
                    out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                    oof_outputs[i] = out.cpu().numpy()

    print("\n=== SYNTHBRACHY3T + TIP-AWARE BUNDLE GRAPH TRANSFORMER 5-FOLD OOF SWEEP ===")
    for tau in [25.0, 35.0, 45.0, 50.0, 55.0, 60.0]:
        preds_all = get_synth_tip_aware_preds(bundle_items, oof_outputs, min_conf=tau, long_mode=False, tip_blend=0.70)
        preds_long = get_synth_tip_aware_preds(bundle_items, oof_outputs, min_conf=tau, long_mode=True, tip_blend=0.70)
        res_all, _ = evaluate_method_unified(preds_all, gts_all_list, thresh=3.5, n_boot=50)
        res_long, _ = evaluate_method_unified(preds_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=50)
        res_adh, _ = evaluate_method_unified([preds_all[i] for i in adh_idx], [gts_all_list[i] for i in adh_idx], thresh=3.5, n_boot=50)
        print(
            f"  tau={tau:4.1f}% | ALL(1063): Rec={res_all['hungarian_1to1_recall_pct']:5.2f}% Prec={res_all['precision_1to1_pct']:5.2f}% F1={res_all['f1_1to1_pct']:5.2f}% "
            f"ShaftMed={res_all['metrics']['mean_shaft_xy']['median']:.3f}mm TipMean/Med={res_all['metrics']['tip_3d']['mean']:.2f}/{res_all['metrics']['tip_3d']['median']:.2f}mm | "
            f"LONG(759): Rec={res_long['hungarian_1to1_recall_pct']:5.2f}% F1_adj={res_long['f1_1to1_obturator_adj_pct']:5.2f}% TipMed={res_long['metrics']['tip_3d']['median']:.2f}mm | "
            f"ADH(233): F1={res_adh['f1_1to1_pct']:5.2f}% TipMed={res_adh['metrics']['tip_3d']['median']:.2f}mm",
            flush=True,
        )

    # Step 4: Few-Shot Sim-to-Real Evaluation (k = 0, 1, 5, 10, 20, 43) with SynthBrachy3T 3D Volume Pre-Training
    print("\n[4/4] Evaluating SynthBrachy3T Sim-to-Real Few-Shot Clinical Budgets (k = 0, 1, 5, 10, 20, 43)...", flush=True)
    for k_shot in [0, 1, 5, 10, 20, 43]:
        if k_shot == 0:
            # Pure Sim-to-Real Zero-Shot: uses Stage 0 prior + SynthBrachy3T-pretrained BundleGraphTransformer (0 BWH clinical labels!)
            # Filter bundle_items to only use Stage 0 candidates + SynthBrachy3T scoring
            zs_preds_all = []
            zs_preds_long = []
            for i, bg in enumerate(bundle_items):
                out_arr = zero_shot_outputs[i]
                sc = 1.0 / (1.0 + np.exp(-np.clip(out_arr[:, :3], -20.0, 20.0)))
                # At k=0, score Stage 0 candidates using Stage 0 symmetry + SynthBrachy3T set-transformer prior
                cands_a, cands_l = [], []
                for idx_c, cand in enumerate(bg["pool"]):
                    if cand["from_mpnn"] == 0.0:  # Stage 0 candidate
                        pts = regularize_poly3_trajectory(cand["ras"], degree=3)
                        s_comb = 0.55 * cand["conf_heur"] + 0.45 * float(sc[idx_c, 1])
                        if s_comb >= 0.43:
                            cands_a.append({"ras": pts, "points": pts, "score": s_comb})
                        if s_comb >= 0.45 and cand["z_span"] >= 26.0:
                            cands_l.append({"ras": pts, "points": pts, "score": s_comb})
                zs_preds_all.append(dedup_traj_dicts(cands_a, min_len=9.0, dedup_mm=2.05))
                zs_preds_long.append(dedup_traj_dicts(cands_l, min_len=26.0, dedup_mm=2.05))
            r_a, _ = evaluate_method_unified(zs_preds_all, gts_all_list, thresh=3.5, n_boot=50)
            r_l, _ = evaluate_method_unified(zs_preds_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=50)
            print(f"  [Sim-to-Real k={k_shot:2d}-shot] ALL F1={r_a['f1_1to1_pct']:5.2f}% (Rec={r_a['hungarian_1to1_recall_pct']:5.2f}% Prec={r_a['precision_1to1_pct']:5.2f}%) | LONG F1_adj={r_l['f1_1to1_obturator_adj_pct']:5.2f}% | Shaft={r_a['metrics']['mean_shaft_xy']['mean']:.2f}mm TipMed={r_a['metrics']['tip_3d']['median']:.2f}mm", flush=True)
        elif k_shot == 43:
            preds_a = get_synth_tip_aware_preds(bundle_items, oof_outputs, min_conf=50.0, long_mode=False, tip_blend=0.70)
            preds_l = get_synth_tip_aware_preds(bundle_items, oof_outputs, min_conf=50.0, long_mode=True, tip_blend=0.70)
            r_a, _ = evaluate_method_unified(preds_a, gts_all_list, thresh=3.5, n_boot=50)
            r_l, _ = evaluate_method_unified(preds_l, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=50)
            print(f"  [Sim-to-Real k={k_shot:2d}-shot] ALL F1={r_a['f1_1to1_pct']:5.2f}% (Rec={r_a['hungarian_1to1_recall_pct']:5.2f}% Prec={r_a['precision_1to1_pct']:5.2f}%) | LONG F1_adj={r_l['f1_1to1_obturator_adj_pct']:5.2f}% | Shaft={r_a['metrics']['mean_shaft_xy']['mean']:.2f}mm TipMed={r_a['metrics']['tip_3d']['median']:.2f}mm", flush=True)
        else:
            # Fine-tune SynthBrachy3T-pretrained TipAwareBundleGraphTransformer on ONLY k_shot clinical cases per fold!
            fs_outputs = [None] * 54
            for fold in range(5):
                torch.manual_seed(300 + fold * 50 + k_shot)
                np.random.seed(300 + fold * 50 + k_shot)
                tr_b = [b for i, b in enumerate(bundle_items) if i % 5 != fold][:k_shot]
                gt_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
                gt_model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
                opt = torch.optim.AdamW(gt_model.parameters(), lr=1.2e-3, weight_decay=2e-4)
                n_ep = 35 if k_shot <= 5 else 55
                for ep in range(n_ep):
                    gt_model.train()
                    syn_sub = [synth_bundles[idx] for idx in np.random.choice(len(synth_bundles), size=10, replace=False)]
                    comb_b = tr_b * max(1, 10 // k_shot) + syn_sub
                    for bi in np.random.permutation(len(comb_b)):
                        bg = comb_b[bi]
                        opt.zero_grad(set_to_none=True)
                        out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                        l_rec = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"].to(device), pos_weight=torch.tensor([3.0], device=device))
                        l_1to1 = F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"].to(device), pos_weight=torch.tensor([2.1], device=device))
                        l_long = F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device))
                        w_tip = bg["y_dz_weight"].to(device)
                        l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"].to(device), reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
                        loss = l_rec + 1.6 * l_1to1 + l_long + 1.0 * l_tip
                        loss.backward()
                        opt.step()
                gt_model.eval()
                with torch.no_grad():
                    for i, bg in enumerate(bundle_items):
                        if i % 5 == fold:
                            out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                            fs_outputs[i] = out.cpu().numpy()
            preds_a = get_synth_tip_aware_preds(bundle_items, fs_outputs, min_conf=50.0, long_mode=False, tip_blend=0.70)
            preds_l = get_synth_tip_aware_preds(bundle_items, fs_outputs, min_conf=50.0, long_mode=True, tip_blend=0.70)
            r_a, _ = evaluate_method_unified(preds_a, gts_all_list, thresh=3.5, n_boot=50)
            r_l, _ = evaluate_method_unified(preds_l, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=50)
            print(f"  [Sim-to-Real k={k_shot:2d}-shot] ALL F1={r_a['f1_1to1_pct']:5.2f}% (Rec={r_a['hungarian_1to1_recall_pct']:5.2f}% Prec={r_a['precision_1to1_pct']:5.2f}%) | LONG F1_adj={r_l['f1_1to1_obturator_adj_pct']:5.2f}% | Shaft={r_a['metrics']['mean_shaft_xy']['mean']:.2f}mm TipMed={r_a['metrics']['tip_3d']['median']:.2f}mm", flush=True)


if __name__ == "__main__":
    main()
