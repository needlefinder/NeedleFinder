#!/usr/bin/env python3
"""
Clean Strictly-Nested Stage 2 Pipeline + Multi-Rater Supervision + Collinear Axial
Split-Fragment Stitcher & Extrapolation-Aware Deduplicator + Dedicated 91-Step 22-Channel
1D Axial Tip Localizer.

Covers:
  - Item 1: Strictly Nested Stage 2A (NeedleSliceMPNN), Stage 2B (DenseGraphTransformer &
            TipAwareBundleGraphTransformer), and Stage 2C (Dedicated1DTipLocalizer) CV
            (35 inner_subtrain / 8-9 inner_val inside each 5-fold outer split; checkpoints
            selected strictly on inner_val, never outer_test) + Resolution-Robust Stage 1/2
            Feature Calibration (fixing Case 040 768x768x80 resolution shift).
  - Item 3: Multi-Rater (CY+HT + LB/AMe) Stage 2 Supervision on outer_train cases only
            (y_rec=0.85, y_1to1=0.75 for Rater-2-only catheters; consensus tip target
            0.5*(z_CY + z_LB) with w_tip=1.25 when |dz|<=4mm, physical drop-off alignment
            and down-weighting when |dz|>6mm).
  - Item 4: Dedicated 91-step (0.5 mm sub-slice Z-interpolated, [z_top-25mm, z_top+20mm])
            22-channel 1D Axial Tip Localizer (extract_1d_tip_profile_91 +
            Dedicated1DTipLocalizer), applied AFTER Collinear Fragment Stitching.
  - Failure-Analysis Fixes:
      a) Candidate Pool Collinear Fragment Stitching (stitch_collinear_pool_fragments)
      b) Extrapolation-Aware Collinear Deduplication & Cranial Extension (collinear_traj_dist)

Configurable via --cache-dir and --oof-dir so it runs on both nn_models_eval and nn_models_clean.
"""

import argparse
import copy
import glob
import json
import math
import os
import pickle
import time
import types
import sys
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

for mod in [
    "vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util",
    "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"
]:
    sys.modules[mod] = types.ModuleType(mod)

sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun")

from evaluate_unified_apples_to_apples import (
    evaluate_method_unified,
    compute_wilcoxon_comparison,
    compute_unified_pair_metrics,
    summarize_distribution,
    extract_nn_labelmap_tubes,
    build_slice_graph_fast,
    DenseGraphTransformer,
    NeedleSliceMPNN,
    decode_mpnn_trajectories,
    load_human_interobserver_40cases,
    get_gts,
    traj_gt_dist,
)
from run_xiao2026_and_fewshot_benchmark import (
    analyze_catheter_adhesion,
    regularize_poly3_trajectory,
)
from test_synthbrachy3t_v2 import (
    generate_synthetic_bundle_graph_item as _base_generate_synthetic_bundle_graph_item,
    TipAwareBundleGraphTransformer,
    apply_tip_adjustment_and_poly3,
)
from run_synthbrachy3t_full import (
    augment_clinical_bundle_with_synthbrachy3t,
    to_gpu_item,
)


# ==============================================================================
# 0. RESOLUTION-ROBUST STAGE 1 PROBABILITY CALIBRATION (Fixes Case 040 768x768)
# ==============================================================================

def calibrate_stage1_oof_for_volume(oof_npz, X_vol, sp_ras):
    """
    100% reference-free scan-level probability calibration:
    When a scan has ultra-fine in-plane pixel spacing (e.g., Case 040 at 0.3125 mm, 768x768)
    or scanner intensity shift that attenuates raw 3D CNN probability amplitude (pm_p999 < 0.75)
    despite strong 3D physical radial symmetry response (sym_p999 >= 0.70), recalibrates
    p_mask and p_peak into the canonical [0, 1] dynamic range and blends with normalized
    physical symmetry X_vol[1].
    """
    p_mask = oof_npz["p_mask"].astype(np.float32).copy()
    p_peak = oof_npz["p_peak"].astype(np.float32).copy()
    p_obt = oof_npz["p_obt"].astype(np.float32).copy()
    vx = oof_npz["vx"].astype(np.float32)
    vy = oof_npz["vy"].astype(np.float32)

    sym = X_vol[1].astype(np.float32)
    pm_p999 = float(np.percentile(p_mask, 99.9))
    pp_p999 = float(np.percentile(p_peak, 99.9))
    po_p999 = float(np.percentile(p_obt, 99.9))
    sym_p999 = float(np.percentile(sym, 99.9))

    if pm_p999 < 0.75 and sym_p999 >= 0.70:
        pm_scale = min(10.0, 0.95 / max(0.04, pm_p999))
        pp_scale = min(6.0, 0.56 / max(0.04, pp_p999))
        po_scale = min(4.0, 0.65 / max(0.08, po_p999)) if po_p999 < 0.45 else 1.0
        sym_norm = np.clip(sym / max(0.85, sym_p999), 0.0, 1.0)
        p_mask = np.clip(np.maximum(p_mask * pm_scale, 0.88 * sym_norm), 0.0, 1.0)
        p_peak = np.clip(np.maximum(p_peak * pp_scale, 0.56 * sym_norm), 0.0, 1.0)
        p_obt = np.clip(p_obt * po_scale, 0.0, 1.0)

    out = {
        "p_mask": p_mask,
        "p_peak": p_peak,
        "p_obt": p_obt,
        "vx": vx,
        "vy": vy,
        "k0": int(oof_npz["k0"]),
        "i0": int(oof_npz["i0"]),
        "j0": int(oof_npz["j0"]),
    }
    return out


# ==============================================================================
# 1. COLLINEAR AXIAL SPLIT-FRAGMENT STITCHING & EXTRAPOLATION DEDUPLICATION
# ==============================================================================

def fit_local_slope(ras, near_z, window_mm=25.0):
    """Fits a fast closed-form linear (dx/dz, dy/dz) tangent over the nearest `window_mm` of `ras`."""
    ras = np.asarray(ras, dtype=np.float64)
    if len(ras) < 2:
        return 0.0, 0.0
    if ras[0, 2] > ras[-1, 2]:
        ras = ras[np.argsort(ras[:, 2])]
    if near_z >= 0.5 * (ras[0, 2] + ras[-1, 2]):
        sub = ras[ras[:, 2] >= ras[-1, 2] - window_mm]
    else:
        sub = ras[ras[:, 2] <= ras[0, 2] + window_mm]
    if len(sub) < 2 or abs(sub[-1, 2] - sub[0, 2]) < 1.0:
        sub = ras
    if len(sub) >= 4:
        z_c = sub[:, 2] - np.mean(sub[:, 2])
        denom = float(np.dot(z_c, z_c))
        if denom > 1e-6:
            return float(np.dot(z_c, sub[:, 0]) / denom), float(np.dot(z_c, sub[:, 1]) / denom)
    dz = max(1.0, float(sub[-1, 2] - sub[0, 2]))
    return float(sub[-1, 0] - sub[0, 0]) / dz, float(sub[-1, 1] - sub[0, 1]) / dz


def eval_traj_extrap(ras, zs):
    """Evaluates trajectory `ras` at axial heights `zs` using local linear tangent extrapolation outside its Z-range."""
    ras = np.asarray(ras, dtype=np.float64)
    ras = ras[np.argsort(ras[:, 2])]
    zs = np.asarray(zs, dtype=np.float64)
    vx_bot, vy_bot = fit_local_slope(ras, float(ras[0, 2]), 25.0)
    vx_top, vy_top = fit_local_slope(ras, float(ras[-1, 2]), 25.0)
    x = np.where(
        (zs >= ras[0, 2]) & (zs <= ras[-1, 2]),
        np.interp(zs, ras[:, 2], ras[:, 0]),
        np.where(
            zs < ras[0, 2],
            ras[0, 0] + vx_bot * (zs - ras[0, 2]),
            ras[-1, 0] + vx_top * (zs - ras[-1, 2]),
        ),
    )
    y = np.where(
        (zs >= ras[0, 2]) & (zs <= ras[-1, 2]),
        np.interp(zs, ras[:, 2], ras[:, 1]),
        np.where(
            zs < ras[0, 2],
            ras[0, 1] + vy_bot * (zs - ras[0, 2]),
            ras[-1, 1] + vy_top * (zs - ras[-1, 2]),
        ),
    )
    return x, y


def collinear_traj_dist(pts, s_ras):
    """
    Extrapolation-aware trajectory distance & collinearity analyzer between `pts` and `s_ras`.
    Returns dict with:
      - d_ov: mean XY distance on overlapping Z range (999.0 if z_ov < 2.0 mm)
      - d_ext: minimum extrapolated XY distance across full Z spans
      - d_mid: XY distance when extrapolating both to their junction midpoint z_mid
      - d_slope: 3D tangent slope difference
      - z_ov: Z overlap in mm (negative if disjoint gap)
      - z_gap: axial gap in mm (0 if overlapping)
    """
    pts = np.asarray(pts, dtype=np.float64)
    s_ras = np.asarray(s_ras, dtype=np.float64)
    pts = pts[np.argsort(pts[:, 2])]
    s_ras = s_ras[np.argsort(s_ras[:, 2])]

    z0 = max(float(pts[0, 2]), float(s_ras[0, 2]))
    z1 = min(float(pts[-1, 2]), float(s_ras[-1, 2]))
    z_ov = z1 - z0
    z_gap = max(0.0, -z_ov)

    span_p = max(1.0, float(pts[-1, 2] - pts[0, 2]))
    span_s = max(1.0, float(s_ras[-1, 2] - s_ras[0, 2]))

    vxp, vyp = (pts[-1, 0] - pts[0, 0]) / span_p, (pts[-1, 1] - pts[0, 1]) / span_p
    vxs, vys = (s_ras[-1, 0] - s_ras[0, 0]) / span_s, (s_ras[-1, 1] - s_ras[0, 1]) / span_s
    z_mid = 0.5 * (z0 + z1)
    vxp_loc, vyp_loc = fit_local_slope(pts, z_mid, 25.0)
    vxs_loc, vys_loc = fit_local_slope(s_ras, z_mid, 25.0)
    d_slope = min(math.hypot(vxp - vxs, vyp - vys), math.hypot(vxp_loc - vxs_loc, vyp_loc - vys_loc))

    if z_ov >= 2.0:
        zs_ov = np.linspace(z0, z1, 12)
        d_ov = float(np.mean(np.hypot(
            np.interp(zs_ov, pts[:, 2], pts[:, 0]) - np.interp(zs_ov, s_ras[:, 2], s_ras[:, 0]),
            np.interp(zs_ov, pts[:, 2], pts[:, 1]) - np.interp(zs_ov, s_ras[:, 2], s_ras[:, 1]),
        )))
    else:
        d_ov = 999.0

    zs_p = np.linspace(float(pts[0, 2]), float(pts[-1, 2]), 12)
    sx_p, sy_p = eval_traj_extrap(s_ras, zs_p)
    d_ext_p = float(np.mean(np.hypot(np.interp(zs_p, pts[:, 2], pts[:, 0]) - sx_p, np.interp(zs_p, pts[:, 2], pts[:, 1]) - sy_p)))

    zs_s = np.linspace(float(s_ras[0, 2]), float(s_ras[-1, 2]), 12)
    px_s, py_s = eval_traj_extrap(pts, zs_s)
    d_ext_s = float(np.mean(np.hypot(np.interp(zs_s, s_ras[:, 2], s_ras[:, 0]) - px_s, np.interp(zs_s, s_ras[:, 2], s_ras[:, 1]) - py_s)))

    px_mid, py_mid = eval_traj_extrap(pts, np.array([z_mid]))
    sx_mid, sy_mid = eval_traj_extrap(s_ras, np.array([z_mid]))
    d_mid = float(np.hypot(px_mid[0] - sx_mid[0], py_mid[0] - sy_mid[0]))
    d_ext = min(d_ext_p, d_ext_s, d_mid if z_ov < 4.0 else 999.0)

    return {
        "d_ov": d_ov,
        "d_ext": d_ext,
        "d_mid": d_mid,
        "d_slope": d_slope,
        "z_ov": z_ov,
        "z_gap": z_gap,
        "span_p": span_p,
        "span_s": span_s,
    }


def merge_collinear_trajectories(ras_a, ras_b):
    """Combines two collinear trajectory segments into a single smooth Poly3-regularized 3D curve."""
    ras_a = np.asarray(ras_a, dtype=np.float64)
    ras_b = np.asarray(ras_b, dtype=np.float64)
    ras_a = ras_a[np.argsort(ras_a[:, 2])]
    ras_b = ras_b[np.argsort(ras_b[:, 2])]

    z_min = min(float(ras_a[0, 2]), float(ras_b[0, 2]))
    z_max = max(float(ras_a[-1, 2]), float(ras_b[-1, 2]))
    n_pts = max(len(ras_a), len(ras_b), int(round((z_max - z_min) / 2.0)) + 2)
    zs = np.linspace(z_min, z_max, n_pts)

    xa, ya = eval_traj_extrap(ras_a, zs)
    xb, yb = eval_traj_extrap(ras_b, zs)

    # Weight ras_a on its supported interval and ras_b on its supported interval, blending smoothly across overlap/gap
    z_mid = 0.5 * (min(float(ras_a[-1, 2]), float(ras_b[-1, 2])) + max(float(ras_a[0, 2]), float(ras_b[0, 2])))
    if float(ras_a[-1, 2]) <= float(ras_b[-1, 2]):
        w_b = 1.0 / (1.0 + np.exp(-(zs - z_mid) / 3.0))
    else:
        w_b = 1.0 / (1.0 + np.exp((zs - z_mid) / 3.0))

    xs = (1.0 - w_b) * xa + w_b * xb
    ys = (1.0 - w_b) * ya + w_b * yb
    comb = np.column_stack([xs, ys, zs])
    if len(comb) >= 4:
        poly = regularize_poly3_trajectory(comb, degree=3)
        comb[:, 0] = 0.65 * comb[:, 0] + 0.35 * poly[:, 0]
        comb[:, 1] = 0.65 * comb[:, 1] + 0.35 * poly[:, 1]
    return comb


def stitch_collinear_pool_fragments(pool, g, oof_npz, X_vol):
    """
    Identifies pairs of candidate trajectories (t_low, t_high) in `pool` where:
      - t_low is lower in Z (z_top_low <= z_bot_high + 6.0 mm and z_bot_high - z_top_low <= 20.0 mm),
      - Extrapolating t_low cranially and t_high caudally to z_mid = 0.5 * (z_top_low + z_bot_high)
        gives an in-plane XY distance <= 2.2 mm, and their 3D slopes agree within <= 0.14 (~8 deg).
    Adds a stitched full-length candidate trajectory t_stitched to `pool` (and updates truncated
    lower fragments when the upper fragment has strong physical/neural support).
    """
    n_orig = len(pool)
    if n_orig < 2:
        return pool

    pm_arr = oof_npz["p_mask"].astype(np.float32)
    pp_arr = oof_npz["p_peak"].astype(np.float32)
    po_arr = oof_npz["p_obt"].astype(np.float32)
    sym_arr = X_vol[1].astype(np.float32)
    ras2ijk = np.linalg.inv(g["ijk_to_ras"])
    k0_c, i0_c, j0_c = int(oof_npz["k0"]), int(oof_npz["i0"]), int(oof_npz["j0"])
    K_s, H_s, W_s = pm_arr.shape
    sp_i = float(g["sp_ras"][0])
    r_px = max(1, int(round(1.0 / sp_i)))

    def _segment_support(ras_seg):
        zs = np.linspace(float(ras_seg[0, 2]), float(ras_seg[-1, 2]), 10)
        xs = np.interp(zs, ras_seg[:, 2], ras_seg[:, 0])
        ys = np.interp(zs, ras_seg[:, 2], ras_seg[:, 1])
        ijks = (ras2ijk @ np.column_stack([xs, ys, zs, np.ones_like(zs)]).T).T
        sups = []
        for row in ijks:
            ic, jc, kc = int(round(row[0] - i0_c)), int(round(row[1] - j0_c)), int(round(row[2] - k0_c))
            if r_px <= ic < W_s - r_px and r_px <= jc < H_s - r_px and 0 <= kc < K_s:
                m_v = float(np.max(pm_arr[kc, jc - r_px:jc + r_px + 1, ic - r_px:ic + r_px + 1]))
                p_v = float(np.max(pp_arr[kc, jc - r_px:jc + r_px + 1, ic - r_px:ic + r_px + 1]))
                o_v = float(np.max(po_arr[kc, jc - r_px:jc + r_px + 1, ic - r_px:ic + r_px + 1]))
                s_v = float(np.max(sym_arr[kc, jc - r_px:jc + r_px + 1, ic - r_px:ic + r_px + 1]))
                sups.append(0.40 * max(m_v, o_v) + 0.35 * p_v + 0.25 * min(1.0, s_v))
        return float(np.mean(sups)) if sups else 0.0

    stitched_cands = []
    for u in range(n_orig):
        ru = pool[u]["ras"]
        z_bot_u, z_top_u = float(ru[0, 2]), float(ru[-1, 2])
        best_v = -1
        best_score = 999.0
        for v in range(n_orig):
            if u == v:
                continue
            rv = pool[v]["ras"]
            z_bot_v, z_top_v = float(rv[0, 2]), float(rv[-1, 2])
            # Require v to be higher in Z than u
            if not (z_bot_u < z_bot_v - 3.0 and z_top_v > z_top_u + 3.5):
                continue
            if not (z_top_u <= z_bot_v + 6.0 and (z_bot_v - z_top_u) <= 20.0):
                continue

            info = collinear_traj_dist(ru, rv)
            if info["d_mid"] <= 2.20 and info["d_slope"] <= 0.14:
                if info["d_mid"] < best_score:
                    best_score = info["d_mid"]
                    best_v = v

        if best_v >= 0:
            t_low = pool[u]
            t_high = pool[best_v]
            sup_high = _segment_support(t_high["ras"])
            if sup_high < 0.26:
                continue
            ras_stitched = merge_collinear_trajectories(t_low["ras"], t_high["ras"])
            # Check if a nearly identical full-length trajectory is already in pool
            already_exists = False
            for p_ex in pool + stitched_cands:
                r_ex = p_ex["ras"]
                if abs(float(r_ex[0, 2]) - float(ras_stitched[0, 2])) <= 5.0 and abs(float(r_ex[-1, 2]) - float(ras_stitched[-1, 2])) <= 5.0:
                    if traj_gt_dist(ras_stitched, r_ex) < 1.50:
                        already_exists = True
                        break
            if not already_exists:
                w_low = max(1.0, t_low["z_span"])
                w_high = max(1.0, t_high["z_span"])
                w_tot = w_low + w_high
                z_span_st = float(ras_stitched[-1, 2] - ras_stitched[0, 2])
                slope_st = (ras_stitched[-1, :2] - ras_stitched[0, :2]) / max(1.0, z_span_st)
                stitched_cands.append({
                    "ras": ras_stitched,
                    "base_xy": ras_stitched[0, :2].copy(),
                    "slope": slope_st,
                    "z_span": z_span_st,
                    "rmse": float((w_low * t_low["rmse"] + w_high * t_high["rmse"]) / w_tot),
                    "conf_heur": float(max(t_low["conf_heur"], t_high["conf_heur"])),
                    "from_mpnn": float(max(t_low["from_mpnn"], t_high["from_mpnn"])),
                    "mpnn_pa": float(max(t_low["mpnn_pa"], t_high["mpnn_pa"])),
                    "mpnn_pl": float(max(t_low["mpnn_pl"], t_high["mpnn_pl"])),
                    "mpnn_pe": float(max(t_low["mpnn_pe"], t_high["mpnn_pe"])),
                    "is_stitched": True,
                })

    return pool + stitched_cands


# ==============================================================================
# 2. DEDICATED 91-STEP (0.5 mm SUB-SLICE Z-INTERPOLATED) 22-CHANNEL 1D TIP MODEL
# ==============================================================================

def extract_1d_tip_profile_91(ras, g, oof_npz, X_vol, z_center=None, z_tip_nbr=None):
    """
    Extracts a 91-step (0.5 mm step size, [z_center - 25.0 mm, z_center + 20.0 mm])
    22-channel 1D axial profile along the trajectory `ras` and its cranial tangent
    extrapolation `(vx_top, vy_top)`, using sub-slice linear interpolation along Z (k_lo, k_hi)
    over the physical 3x3 / r_px in-plane patch.

    Channels (22):
      0: p_mask_max
      1: p_mask_mean
      2: p_peak_max
      3: p_obt_max
      4: vx_center
      5: vy_center
      6: tangent_err = hypot(vx - vx_top, vy - vy_top)
      7: sym_max (X[1])
      8: sym_mean (X[1])
      9: ring_cs_max (X[2])
     10: dark_ratio_max (X[3])
     11: t2w_min (X[0] dark lumen core)
     12: hull_center (X[5])
     13: combined_support = 0.40*max(p_mask, p_obt) + 0.35*p_peak + 0.25*min(1.0, sym)
     14: rel_z = (z - z_center) / 25.0
     15: in_bounds (1.0 inside axial slab [0, K_s-1], 0.0 outside)
     16: dist_to_nbr_dome = (z - z_tip_nbr) / 25.0
     17: d(p_mask_max)/dz
     18: d(p_peak_max)/dz
     19: d(sym_max)/dz
     20: d(t2w_min)/dz
     21: d(combined_support)/dz
    """
    ras = np.asarray(ras, dtype=np.float64)
    ras = ras[np.argsort(ras[:, 2])]
    z_top = float(ras[-1, 2]) if z_center is None else float(z_center)
    if z_tip_nbr is None:
        z_tip_nbr = z_top

    vx_top, vy_top = fit_local_slope(ras, float(ras[-1, 2]), 25.0)
    dz_grid = np.linspace(-25.0, 20.0, 91, dtype=np.float64)  # 0.5 mm steps, index 50 is 0.0
    zs = z_top + dz_grid
    xs, ys = eval_traj_extrap(ras, zs)

    ras2ijk = np.linalg.inv(g["ijk_to_ras"])
    ijks = (ras2ijk @ np.column_stack([xs, ys, zs, np.ones_like(zs)]).T).T

    pm_arr = oof_npz["p_mask"].astype(np.float32)
    pp_arr = oof_npz["p_peak"].astype(np.float32)
    po_arr = oof_npz["p_obt"].astype(np.float32)
    vx_arr = oof_npz["vx"].astype(np.float32)
    vy_arr = oof_npz["vy"].astype(np.float32)

    t2w_arr = X_vol[0].astype(np.float32)
    sym_arr = X_vol[1].astype(np.float32)
    cs_arr = X_vol[2].astype(np.float32)
    dr_arr = X_vol[3].astype(np.float32)
    hull_arr = X_vol[5].astype(np.float32)

    k0_c, i0_c, j0_c = int(oof_npz["k0"]), int(oof_npz["i0"]), int(oof_npz["j0"])
    K_s, H_s, W_s = pm_arr.shape
    sp_i = float(g["sp_ras"][0])
    r_px = max(1, int(round(1.0 / sp_i)))

    prof = np.zeros((22, 91), dtype=np.float32)
    prof[14] = (dz_grid / 25.0).astype(np.float32)
    prof[16] = ((zs - float(z_tip_nbr)) / 25.0).astype(np.float32)

    for t_idx in range(91):
        fi = float(ijks[t_idx, 0] - i0_c)
        fj = float(ijks[t_idx, 1] - j0_c)
        fk = float(ijks[t_idx, 2] - k0_c)
        ic = int(round(fi))
        jc = int(round(fj))

        if not (r_px <= ic < W_s - r_px and r_px <= jc < H_s - r_px and -0.5 <= fk <= K_s - 0.5):
            continue

        prof[15, t_idx] = 1.0
        fk_clamped = np.clip(fk, 0.0, float(K_s - 1))
        k_lo = int(math.floor(fk_clamped))
        k_hi = min(K_s - 1, k_lo + 1)
        alpha = float(fk_clamped - k_lo)

        sl_j = slice(jc - r_px, jc + r_px + 1)
        sl_i = slice(ic - r_px, ic + r_px + 1)

        def _interp_patch(vol3d, mode="max"):
            p_lo = vol3d[k_lo, sl_j, sl_i]
            p_hi = vol3d[k_hi, sl_j, sl_i] if k_hi != k_lo else p_lo
            patch = (1.0 - alpha) * p_lo + alpha * p_hi
            if mode == "max":
                return float(np.max(patch))
            elif mode == "mean":
                return float(np.mean(patch))
            elif mode == "min":
                return float(np.min(patch))
            else:
                return float((1.0 - alpha) * vol3d[k_lo, jc, ic] + alpha * vol3d[k_hi, jc, ic])

        pm_max = _interp_patch(pm_arr, "max")
        pm_mean = _interp_patch(pm_arr, "mean")
        pp_max = _interp_patch(pp_arr, "max")
        po_max = _interp_patch(po_arr, "max")
        vx_c = _interp_patch(vx_arr, "center")
        vy_c = _interp_patch(vy_arr, "center")
        tan_err = math.hypot(vx_c - vx_top, vy_c - vy_top)

        sym_max = _interp_patch(sym_arr, "max")
        sym_mean = _interp_patch(sym_arr, "mean")
        cs_max = _interp_patch(cs_arr, "max")
        dr_max = _interp_patch(dr_arr, "max")
        t2w_min = _interp_patch(t2w_arr, "min")
        hull_c = _interp_patch(hull_arr, "center")
        comb_sup = 0.40 * max(pm_max, po_max) + 0.35 * pp_max + 0.25 * min(1.0, sym_max)

        prof[0, t_idx] = pm_max
        prof[1, t_idx] = pm_mean
        prof[2, t_idx] = pp_max
        prof[3, t_idx] = po_max
        prof[4, t_idx] = vx_c
        prof[5, t_idx] = vy_c
        prof[6, t_idx] = tan_err
        prof[7, t_idx] = sym_max
        prof[8, t_idx] = sym_mean
        prof[9, t_idx] = cs_max
        prof[10, t_idx] = dr_max
        prof[11, t_idx] = t2w_min
        prof[12, t_idx] = hull_c
        prof[13, t_idx] = comb_sup

    # Axial first derivatives d/dz (per 0.5 mm step, scaled by 2.0)
    prof[17] = np.gradient(prof[0]) * 2.0
    prof[18] = np.gradient(prof[2]) * 2.0
    prof[19] = np.gradient(prof[7]) * 2.0
    prof[20] = np.gradient(prof[11]) * 2.0
    prof[21] = np.gradient(prof[13]) * 2.0

    return prof


class Dedicated1DTipLocalizer(nn.Module):
    """
    Dedicated 1D Multi-Channel Axial Tip Localizer operating on 91-step (0.5 mm)
    22-channel profiles [B, 22, 91] from z_top - 25.0 mm to z_top + 20.0 mm.
    Combines multi-scale dilated 1D ResNet blocks with 1D Axial Self-Attention and
    predicts both a 91-step tip probability distribution (soft-argmax over [-25, +20] mm)
    and a direct residual refinement.
    """
    def __init__(self, in_channels=22, hidden_dim=48, seq_len=91):
        super().__init__()
        self.register_buffer("dz_grid", torch.linspace(-25.0, 20.0, seq_len))
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, dilation=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, dilation=1),
            nn.GroupNorm(8, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden_dim),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=8, dilation=4),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=8, dilation=4),
            nn.GroupNorm(8, hidden_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.05, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.step_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [B, 22, 91]
        h = self.stem(x)
        h = F.silu(h + self.block1(h))
        h = F.silu(h + self.block2(h))
        h = F.silu(h + self.block3(h))

        # 1D self-attention across the 91 axial steps
        h_seq = h.transpose(1, 2)  # [B, 91, C]
        attn_out, _ = self.attn(h_seq, h_seq, h_seq, need_weights=False)
        h_seq = self.attn_norm(h_seq + attn_out)
        h = h_seq.transpose(1, 2)  # [B, C, 91]

        step_logits = self.step_head(h).squeeze(1)  # [B, 91]
        step_probs = F.softmax(step_logits, dim=-1)
        dz_soft = torch.sum(step_probs * self.dz_grid.unsqueeze(0), dim=-1)  # [B]

        # Global + center context for residual regression
        h_center = h[:, :, 50]  # index 50 is dz = 0.0 mm
        h_mean = h.mean(dim=-1)
        dz_res = self.reg_head(torch.cat([h_center, h_mean], dim=-1)).squeeze(-1) * 10.0
        dz_pred_mm = 0.75 * dz_soft + 0.25 * dz_res
        return dz_pred_mm, step_logits


# ==============================================================================
# 3. ENRICHED BUNDLE GRAPH BUILDER WITH MULTI-RATER SUPERVISION & CONSENSUS TIP
# ==============================================================================

def compute_trajectory_axial_profile_and_tip_features_robust(ras, g, oof_npz, X_vol):
    """
    Resolution-robust version of `compute_trajectory_axial_profile_and_tip_features`:
    Uses physical millimeter patch radius `r_px = max(1, int(round(1.0 / sp_i)))` instead
    of hardcoded 1-pixel radius, and computes profile-supported cranial tip `z_prof_tip`.
    """
    pm_arr = oof_npz["p_mask"].astype(np.float32)
    pp_arr = oof_npz["p_peak"].astype(np.float32)
    po_arr = oof_npz["p_obt"].astype(np.float32)
    sym_arr = X_vol[1].astype(np.float32)
    cs_arr = X_vol[2].astype(np.float32)
    rc_arr = X_vol[3].astype(np.float32)

    ijk2ras = g["ijk_to_ras"]
    ras2ijk = np.linalg.inv(ijk2ras)
    k0_c, i0_c, j0_c = int(oof_npz["k0"]), int(oof_npz["i0"]), int(oof_npz["j0"])
    K_s, H_s, W_s = pm_arr.shape
    sp_i = float(g["sp_ras"][0])
    r_px = max(1, int(round(1.0 / sp_i)))

    ras = ras[np.argsort(ras[:, 2])]
    z_bot_cur = float(ras[0, 2])
    z_top_cur = float(ras[-1, 2])

    zs = np.linspace(z_bot_cur, z_top_cur, 20)
    xs = np.interp(zs, ras[:, 2], ras[:, 0])
    ys = np.interp(zs, ras[:, 2], ras[:, 1])
    ijks = (ras2ijk @ np.column_stack([xs, ys, zs, np.ones_like(zs)]).T).T

    pm_l, pp_l, po_l, sym_l, cs_l, rc_l = [], [], [], [], [], []
    for row in ijks:
        ic, jc, kc = int(round(row[0] - i0_c)), int(round(row[1] - j0_c)), int(round(row[2] - k0_c))
        if r_px <= ic < W_s - r_px and r_px <= jc < H_s - r_px and 0 <= kc < K_s:
            sl_j = slice(jc - r_px, jc + r_px + 1)
            sl_i = slice(ic - r_px, ic + r_px + 1)
            pm_l.append(float(np.max(pm_arr[kc, sl_j, sl_i])))
            pp_l.append(float(np.max(pp_arr[kc, sl_j, sl_i])))
            po_l.append(float(np.max(po_arr[kc, sl_j, sl_i])))
            sym_l.append(float(np.max(sym_arr[kc, sl_j, sl_i])))
            cs_l.append(float(np.max(cs_arr[kc, sl_j, sl_i])))
            rc_l.append(float(np.max(rc_arr[kc, sl_j, sl_i])))

    mean_pm = float(np.mean(pm_l)) if pm_l else 0.0
    p75_pm = float(np.percentile(pm_l, 75)) if pm_l else 0.0
    mean_pp = float(np.mean(pp_l)) if pp_l else 0.0
    mean_po = float(np.mean(po_l)) if po_l else 0.0
    mean_sym = float(np.mean(sym_l)) if sym_l else 0.0
    p75_sym = float(np.percentile(sym_l, 75)) if sym_l else 0.0
    mean_cs = float(np.mean(cs_l)) if cs_l else 0.0
    mean_rc = float(np.mean(rc_l)) if rc_l else 0.0
    frac_active = float(np.mean([(m > 0.22 or p > 0.18 or s > 0.25) for m, p, s in zip(pm_l, pp_l, sym_l)])) if pm_l else 0.0

    vx_top, vy_top = fit_local_slope(ras, z_top_cur, 25.0)

    z_probe = np.linspace(z_top_cur - 15.0, z_top_cur + 15.0, 25)
    x_probe = np.where(z_probe <= z_top_cur, np.interp(z_probe, ras[:, 2], ras[:, 0]), ras[-1, 0] + vx_top * (z_probe - z_top_cur))
    y_probe = np.where(z_probe <= z_top_cur, np.interp(z_probe, ras[:, 2], ras[:, 1]), ras[-1, 1] + vy_top * (z_probe - z_top_cur))
    ijks_p = (ras2ijk @ np.column_stack([x_probe, y_probe, z_probe, np.ones_like(z_probe)]).T).T
    support_p = np.zeros(len(z_probe), dtype=np.float32)
    for idx_p, row in enumerate(ijks_p):
        ic, jc, kc = int(round(row[0] - i0_c)), int(round(row[1] - j0_c)), int(round(row[2] - k0_c))
        if r_px <= ic < W_s - r_px and r_px <= jc < H_s - r_px and 0 <= kc < K_s:
            sl_j = slice(jc - r_px, jc + r_px + 1)
            sl_i = slice(ic - r_px, ic + r_px + 1)
            m_v = float(np.max(pm_arr[kc, sl_j, sl_i]))
            p_v = float(np.max(pp_arr[kc, sl_j, sl_i]))
            o_v = float(np.max(po_arr[kc, sl_j, sl_i]))
            s_v = float(np.max(sym_arr[kc, sl_j, sl_i]))
            support_p[idx_p] = 0.40 * max(m_v, o_v) + 0.35 * p_v + 0.25 * min(1.0, s_v)

    valid_sup = np.where(support_p >= 0.24)[0]
    if len(valid_sup) > 0:
        last_valid = int(valid_sup[-1])
        z_prof_tip = float(z_probe[last_valid])
    else:
        z_prof_tip = z_top_cur

    tip_sup_cur = float(support_p[12])
    tip_sup_above = float(np.mean(support_p[14:19]))
    tip_sup_below = float(np.mean(support_p[6:12]))

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


def build_bundle_graphs_with_multirater(pool, g, oof_npz, X_vol, gts_all=None, gts_long=None, gts_lb_rater2=None):
    """
    Builds both the 18-feature `base_bg` (for DenseGraphTransformer) and the 26-feature
    `enriched_bg` (for TipAwareBundleGraphTransformer), plus 91-step 22-channel 1D tip
    profiles for each candidate in `pool`, with Multi-Rater Supervision (Item 3):
      - If a candidate does not match CY+HT (<= 3.5 mm) but matches an Annotator 2 (LB/AMe)
        trace (<= 3.5 mm), sets y_rec = 0.85 and y_1to1 = 0.75 (instead of 0.0!).
      - If a candidate matches both CY+HT and LB/AMe:
          * When |z_CY_tip - z_LB_tip| <= 4.0 mm, target tip is 0.5*(z_CY_tip + z_LB_tip)
            with consensus confidence multiplier 1.25x.
          * When |z_CY_tip - z_LB_tip| > 6.0 mm, selects whichever rater tip aligns closer
            to the physical susceptibility drop-off `z_prof_tip` and down-weights by 0.50x.
    """
    M = len(pool)
    bases = np.array([p["base_xy"] for p in pool]) if M else np.zeros((0, 2))
    slopes = np.array([p["slope"] for p in pool]) if M else np.zeros((0, 2))
    med_base = np.median(bases, axis=0) if M else np.zeros(2)
    med_slope = np.median(slopes, axis=0) if M else np.zeros(2)

    prof_list = [compute_trajectory_axial_profile_and_tip_features_robust(p["ras"], g, oof_npz, X_vol) for p in pool]

    z_tops = np.array([float(p["ras"][-1, 2]) for p in pool]) if M else np.zeros(0)
    z_bots = np.array([float(p["ras"][0, 2]) for p in pool]) if M else np.zeros(0)
    qual_weights = np.array([
        max(0.05, pr["mean_pm"] * 0.4 + pr["mean_pp"] * 0.3 + pr["mean_sym"] * 0.3)
        for pr in prof_list
    ]) if M else np.zeros(0)

    z_tip_nbr = np.zeros(M, dtype=np.float32)
    for u in range(M):
        dists_xy = np.hypot(bases[:, 0] - bases[u, 0], bases[:, 1] - bases[u, 1])
        span_u = z_tops[u] - z_bots[u]
        same_class = np.abs((z_tops - z_bots) - span_u) <= 22.0
        nbr_mask = (dists_xy <= 28.0) & same_class & (qual_weights >= 0.25)
        if np.sum(nbr_mask) >= 2:
            w_n = qual_weights[nbr_mask]
            z_tip_nbr[u] = float(np.sum(z_tops[nbr_mask] * w_n) / np.sum(w_n))
        else:
            z_tip_nbr[u] = float(z_tops[u])

    x18_feats = []
    x26_feats = []
    tip_profiles_91 = []

    for u, p in enumerate(pool):
        ras = p["ras"]
        pr = prof_list[u]
        d_base = float(np.hypot(p["base_xy"][0] - med_base[0], p["base_xy"][1] - med_base[1])) / 40.0
        d_slope = float(np.hypot(p["slope"][0] - med_slope[0], p["slope"][1] - med_slope[1])) / 0.25
        z_low = float(ras[0, 2] - g["z_base_ras"]) / 50.0
        z_high = float(ras[-1, 2] - g["z_base_ras"]) / 100.0
        dz_prof = float(pr["z_prof_tip"] - ras[-1, 2]) / 20.0
        dz_nbr = float(z_tip_nbr[u] - ras[-1, 2]) / 20.0

        p["z_prof_tip"] = pr["z_prof_tip"]
        p["z_tip_nbr"] = float(z_tip_nbr[u])
        p["vx_top"] = pr["vx_top"]
        p["vy_top"] = pr["vy_top"]
        p["mean_sym"] = pr["mean_sym"]
        p["frac_active"] = pr["frac_active"]
        p["mean_pm"] = pr["mean_pm"]

        f18 = [
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
        ]
        f26 = f18 + [
            pr["mean_sym"],
            pr["p75_sym"],
            pr["mean_cs"],
            pr["mean_rc"],
            pr["frac_active"],
            pr["tip_sup_cur"],
            dz_prof,
            dz_nbr,
        ]
        x18_feats.append(f18)
        x26_feats.append(f26)
        tip_profiles_91.append(extract_1d_tip_profile_91(ras, g, oof_npz, X_vol, z_center=float(ras[-1, 2]), z_tip_nbr=float(z_tip_nbr[u])))

    adj_feat6 = np.zeros((M, M, 6), dtype=np.float32)
    adj_feat8 = np.zeros((M, M, 8), dtype=np.float32)
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
                e6 = [
                    db / 35.0,
                    sh_d / 35.0,
                    ds / 0.30,
                    1.0 if sh_d < 2.2 else 0.0,
                    float(x26_feats[v][5] - x26_feats[u][5]),
                    float(x26_feats[v][0] - x26_feats[u][0]),
                ]
                adj_feat6[u, v] = e6
                adj_feat8[u, v] = e6 + [dz_t, z_ov]

    # Compute Single-Rater (CY+HT) and Multi-Rater (CY+HT + LB/AMe) training targets
    y_rec_single = np.zeros(M, dtype=np.float32)
    y_1to1_single = np.zeros(M, dtype=np.float32)
    y_long_single = np.zeros(M, dtype=np.float32)
    y_dz_tip_single = np.zeros(M, dtype=np.float32)
    y_dz_weight_single = np.zeros(M, dtype=np.float32)
    y_true_tip_z_single = np.zeros(M, dtype=np.float32)

    y_rec_multi = np.zeros(M, dtype=np.float32)
    y_1to1_multi = np.zeros(M, dtype=np.float32)
    y_long_multi = np.zeros(M, dtype=np.float32)
    y_dz_tip_multi = np.zeros(M, dtype=np.float32)
    y_dz_weight_multi = np.zeros(M, dtype=np.float32)
    y_true_tip_z_multi = np.zeros(M, dtype=np.float32)

    if gts_all is not None and len(gts_all) > 0 and M > 0:
        cost_a = np.zeros((len(gts_all), M))
        for gi, gt in enumerate(gts_all):
            for pj, p in enumerate(pool):
                cost_a[gi, pj] = traj_gt_dist(p["ras"], gt)

        # Closest LB/AMe match per candidate if available
        lb_best_d = np.full(M, 999.0)
        lb_best_z_tip = np.zeros(M, dtype=np.float64)
        if gts_lb_rater2 is not None and len(gts_lb_rater2) > 0:
            cost_lb = np.zeros((len(gts_lb_rater2), M))
            for li, l_gt in enumerate(gts_lb_rater2):
                l_pts = np.asarray(l_gt, dtype=np.float64)
                l_pts = l_pts[np.argsort(l_pts[:, 2])]
                for pj, p in enumerate(pool):
                    cost_lb[li, pj] = traj_gt_dist(p["ras"], l_pts)
            best_li = np.argmin(cost_lb, axis=0)
            for pj in range(M):
                lb_best_d[pj] = float(cost_lb[best_li[pj], pj])
                l_pts = np.asarray(gts_lb_rater2[best_li[pj]], dtype=np.float64)
                l_pts = l_pts[np.argsort(l_pts[:, 2])]
                lb_best_z_tip[pj] = float(l_pts[-1, 2])

        y_rec_single = (np.min(cost_a, axis=0) <= 3.3).astype(np.float32)
        y_rec_multi = y_rec_single.copy()

        best_gi = np.argmin(cost_a, axis=0)
        for pj in range(M):
            z_cur = float(pool[pj]["ras"][-1, 2])
            y_true_tip_z_single[pj] = z_cur
            y_true_tip_z_multi[pj] = z_cur
            if cost_a[best_gi[pj], pj] <= 3.3:
                gt_match = gts_all[best_gi[pj]]
                z_cy_tip = float(gt_match[-1, 2])
                dz_t = z_cy_tip - z_cur
                if abs(dz_t) <= 22.0:
                    y_dz_tip_single[pj] = dz_t / 15.0
                    y_dz_weight_single[pj] = 1.0
                    y_true_tip_z_single[pj] = z_cy_tip

                # Multi-rater tip consensus weighting (Item 3)
                if lb_best_d[pj] <= 3.5:
                    z_lb_tip = float(lb_best_z_tip[pj])
                    diff_raters = abs(z_cy_tip - z_lb_tip)
                    if diff_raters <= 4.0:
                        z_cons_tip = 0.5 * (z_cy_tip + z_lb_tip)
                        w_mult = 1.25
                    elif diff_raters > 6.0:
                        z_prof_t = float(pool[pj].get("z_prof_tip", z_cy_tip))
                        z_cons_tip = z_cy_tip if abs(z_cy_tip - z_prof_t) <= abs(z_lb_tip - z_prof_t) else z_lb_tip
                        w_mult = 0.50
                    else:
                        z_cons_tip = 0.65 * z_cy_tip + 0.35 * z_lb_tip
                        w_mult = 1.00
                else:
                    z_cons_tip = z_cy_tip
                    w_mult = 1.00

                dz_m = z_cons_tip - z_cur
                if abs(dz_m) <= 22.0:
                    y_dz_tip_multi[pj] = dz_m / 15.0
                    y_dz_weight_multi[pj] = 1.0 * w_mult
                    y_true_tip_z_multi[pj] = z_cons_tip

        ri, ci = linear_sum_assignment(cost_a)
        for r, c in zip(ri, ci):
            if cost_a[r, c] <= 3.5:
                y_1to1_single[c] = 1.0
                y_1to1_multi[c] = 1.0
                z_cur = float(pool[c]["ras"][-1, 2])
                z_cy_tip = float(gts_all[r][-1, 2])
                dz_t = z_cy_tip - z_cur
                if abs(dz_t) <= 22.0:
                    y_dz_tip_single[c] = dz_t / 15.0
                    y_dz_weight_single[c] = 2.0
                    y_true_tip_z_single[c] = z_cy_tip
                if y_dz_weight_multi[c] > 0:
                    y_dz_weight_multi[c] *= 2.0

        # Multi-Rater Supervision for Rater-2-only traces (Item 3):
        # If a candidate does not match CY+HT (<= 3.5 mm) but matches LB/AMe (<= 3.5 mm),
        # set y_rec = 0.85 and y_1to1 = 0.75 (for the best candidate matching that LB/AMe trace)
        if gts_lb_rater2 is not None and len(gts_lb_rater2) > 0:
            for li, l_gt in enumerate(gts_lb_rater2):
                l_pts = np.asarray(l_gt, dtype=np.float64)
                l_pts = l_pts[np.argsort(l_pts[:, 2])]
                # Check if this LB/AMe trace is missing from CY+HT (min dist > 3.5 mm)
                cy_dists = [traj_gt_dist(l_pts, g) for g in gts_all]
                if min(cy_dists) > 3.5:
                    cand_dists = [traj_gt_dist(p["ras"], l_pts) for p in pool]
                    for pj, cd in enumerate(cand_dists):
                        if cd <= 3.5 and y_rec_multi[pj] < 0.5:
                            y_rec_multi[pj] = 0.85
                    best_pj = int(np.argmin(cand_dists))
                    if cand_dists[best_pj] <= 3.5 and y_1to1_multi[best_pj] < 0.5:
                        y_1to1_multi[best_pj] = 0.75
                        if float(l_pts[-1, 2] - l_pts[0, 2]) >= 28.0:
                            y_long_multi[best_pj] = 0.75
                        dz_lb = float(l_pts[-1, 2] - pool[best_pj]["ras"][-1, 2])
                        if abs(dz_lb) <= 22.0:
                            y_dz_tip_multi[best_pj] = dz_lb / 15.0
                            y_dz_weight_multi[best_pj] = 1.25
                            y_true_tip_z_multi[best_pj] = float(l_pts[-1, 2])

    if gts_long is not None and len(gts_long) > 0 and M > 0:
        cost_l = np.zeros((len(gts_long), M))
        for gi, gt in enumerate(gts_long):
            for pj, p in enumerate(pool):
                cost_l[gi, pj] = traj_gt_dist(p["ras"], gt)
        ri, ci = linear_sum_assignment(cost_l)
        for r, c in zip(ri, ci):
            if cost_l[r, c] <= 3.5:
                y_long_single[c] = 1.0
                y_long_multi[c] = 1.0

    base_bg = {
        "pool": pool,
        "x": torch.tensor(x18_feats, dtype=torch.float32),
        "adj_feat": torch.tensor(adj_feat6, dtype=torch.float32),
        "adj_mask": torch.tensor(adj_mask, dtype=torch.float32),
        "y_rec": torch.tensor(y_rec_multi, dtype=torch.float32),
        "y_1to1": torch.tensor(y_1to1_multi, dtype=torch.float32),
        "y_long_1to1": torch.tensor(y_long_multi, dtype=torch.float32),
        "y_rec_single": torch.tensor(y_rec_single, dtype=torch.float32),
        "y_1to1_single": torch.tensor(y_1to1_single, dtype=torch.float32),
        "y_long_single": torch.tensor(y_long_single, dtype=torch.float32),
    }

    enriched_bg = {
        "pool": pool,
        "x": torch.tensor(x26_feats, dtype=torch.float32),
        "adj_feat": torch.tensor(adj_feat8, dtype=torch.float32),
        "adj_mask": torch.tensor(adj_mask, dtype=torch.float32),
        "y_rec": torch.tensor(y_rec_multi, dtype=torch.float32),
        "y_1to1": torch.tensor(y_1to1_multi, dtype=torch.float32),
        "y_long_1to1": torch.tensor(y_long_multi, dtype=torch.float32),
        "y_dz_tip": torch.tensor(y_dz_tip_multi, dtype=torch.float32),
        "y_dz_weight": torch.tensor(y_dz_weight_multi, dtype=torch.float32),
        "y_true_tip_z": torch.tensor(y_true_tip_z_multi, dtype=torch.float32),
        "y_rec_single": torch.tensor(y_rec_single, dtype=torch.float32),
        "y_1to1_single": torch.tensor(y_1to1_single, dtype=torch.float32),
        "y_long_single": torch.tensor(y_long_single, dtype=torch.float32),
        "y_dz_tip_single": torch.tensor(y_dz_tip_single, dtype=torch.float32),
        "y_dz_weight_single": torch.tensor(y_dz_weight_single, dtype=torch.float32),
        "tip_profiles_91": torch.tensor(np.stack(tip_profiles_91, axis=0), dtype=torch.float32) if M > 0 else torch.zeros((0, 22, 91), dtype=torch.float32),
        "gts_all": gts_all,
        "gts_long": gts_long,
    }

    return base_bg, enriched_bg


def generate_synthetic_bundle_graph_item_v2(rng):
    """
    Upgraded SynthBrachy3T bundle generator where 22% of true 1-to-1 winners have
    `from_mpnn == 0.0` (representing strong Stage 0 physical symmetry tracks in high-res
    volumes where MPNN did not link a separate path), preventing Stage 2B from suppressing
    `from_mpnn == 0.0` true catheters!
    """
    bg = _base_generate_synthetic_bundle_graph_item(rng)
    x = bg["x"].clone()
    y_1to1 = bg["y_1to1"]
    win_idx = torch.where(y_1to1 > 0.5)[0]
    for idx in win_idx:
        if rng.random() < 0.22:
            x[idx, 1] = 0.0  # from_mpnn = 0.0
            x[idx, 2] = 0.0  # mpnn_pa = 0.0
            x[idx, 3] = 0.0  # mpnn_pl = 0.0
            x[idx, 4] = 0.0  # mpnn_pe = 0.0
            if rng.random() < 0.45:
                # Also simulate moderate Stage 1 probability with strong Stage 0 symmetry
                x[idx, 5] = float(rng.uniform(0.22, 0.55))
                x[idx, 7] = float(rng.uniform(0.15, 0.40))
                x[idx, 18] = float(rng.uniform(0.65, 1.25))
    bg["x"] = x
    return bg


def augment_clinical_bundle_with_synthbrachy3t_v2(bg, rng):
    """
    Upgraded fold-isolated SynthBrachy3T domain-randomized augmentation of a training bundle graph:
      1. Perturbs cranial tip heights z_top on true catheters and updates node features + y_dz_tip.
      2. Injects 1-2 adhering twin catheters (1.80-2.45 mm apart, full length) with y_1to1 = 1.0.
      3. Injects 1-3 truncated split-fragment duplicates on the same needle (30-58% length) with
         y_rec = 0.75, y_1to1 = 0.0, y_long = 0.0 so self-attention learns to suppress short
         split-fragments when a full-length trajectory is present.
      4. Injects 1-2 peripheral vascular / susceptibility artifact FPs with y_rec = 0.0, y_1to1 = 0.0.
    """
    x = bg["x"].clone().numpy()
    af = bg["adj_feat"].clone().numpy()
    am = bg["adj_mask"].clone().numpy()
    y_rec = bg["y_rec"].clone().numpy()
    y_1to1 = bg["y_1to1"].clone().numpy()
    y_long = bg["y_long_1to1"].clone().numpy()
    y_dz = bg["y_dz_tip"].clone().numpy()
    y_w = bg["y_dz_weight"].clone().numpy()

    M = len(x)
    if M == 0:
        return bg

    feat_noise = rng.normal(0.0, 0.02, size=x.shape).astype(np.float32)
    feat_noise[:, [1, 16, 17]] = 0.0
    x = x + feat_noise

    for u in range(M):
        if y_w[u] > 0.5 and rng.random() < 0.65:
            delta_z_mm = float(rng.uniform(-9.0, 9.0))
            x[u, 10] = max(0.12, x[u, 10] + delta_z_mm / 60.0)
            x[u, 15] = x[u, 15] + delta_z_mm / 100.0
            x[u, 24] = x[u, 24] - delta_z_mm / 20.0
            x[u, 25] = x[u, 25] - delta_z_mm / 20.0
            new_target_mm = y_dz[u] * 15.0 - delta_z_mm
            y_dz[u] = float(np.clip(new_target_mm / 15.0, -1.5, 1.5))

    win_idx = np.where(y_1to1 > 0.5)[0]
    if len(win_idx) >= 2:
        new_x, new_rec, new_1to1, new_long, new_dz, new_w = [], [], [], [], [], []
        edge_specs = []

        if rng.random() < 0.55:
            n_twins = int(rng.integers(1, min(3, len(win_idx) + 1)))
            for c_idx in rng.choice(win_idx, size=n_twins, replace=False):
                row = x[c_idx].copy()
                row[0] = np.clip(row[0] + rng.normal(0, 0.03), 0.45, 0.98)
                row[5] = np.clip(row[5] + rng.normal(0, 0.04), 0.35, 0.98)
                row[18] = np.clip(row[18] + rng.normal(0, 0.04), 0.30, 1.10)
                new_x.append(row)
                new_rec.append(1.0)
                new_1to1.append(1.0)
                new_long.append(float(y_long[c_idx]))
                new_dz.append(float(y_dz[c_idx]))
                new_w.append(float(y_w[c_idx]))
                edge_specs.append((int(c_idx), False, float(rng.uniform(1.80, 2.45)) / 35.0, 0.92))

        if rng.random() < 0.75:
            n_splits = int(rng.integers(1, min(4, len(win_idx) + 1)))
            for c_idx in rng.choice(win_idx, size=n_splits, replace=False):
                row = x[c_idx].copy()
                frac_keep = float(rng.uniform(0.30, 0.58))
                row[10] = row[10] * frac_keep
                row[0] = np.clip(row[0] - float(rng.uniform(0.02, 0.12)), 0.25, 0.92)
                row[5] = np.clip(row[5] - float(rng.uniform(0.02, 0.10)), 0.20, 0.90)
                if rng.random() < 0.5:
                    row[1] = 0.0
                new_x.append(row)
                new_rec.append(0.75)
                new_1to1.append(0.0)
                new_long.append(0.0)
                new_dz.append(0.0)
                new_w.append(0.0)
                edge_specs.append((int(c_idx), True, float(rng.uniform(0.40, 1.85)) / 35.0, frac_keep))

        if rng.random() < 0.65:
            n_fps = int(rng.integers(1, 3))
            for _ in range(n_fps):
                c_idx = int(rng.choice(win_idx))
                row = x[c_idx].copy()
                row[0] = float(rng.uniform(0.25, 0.65))
                row[1] = 0.0 if rng.random() < 0.6 else 0.5
                row[2:5] = float(rng.uniform(0.10, 0.45))
                row[5:10] = float(rng.uniform(0.12, 0.42))
                row[10] = float(rng.uniform(0.15, 0.42))
                row[12] = float(rng.uniform(0.65, 1.35))
                row[13] = float(rng.uniform(0.55, 1.30))
                row[18:22] = float(rng.uniform(0.04, 0.22))
                row[22] = float(rng.uniform(0.15, 0.45))
                new_x.append(row)
                new_rec.append(0.0)
                new_1to1.append(0.0)
                new_long.append(0.0)
                new_dz.append(0.0)
                new_w.append(0.0)
                edge_specs.append((c_idx, False, float(rng.uniform(8.0, 25.0)) / 35.0, 0.30))

        K_add = len(new_x)
        if K_add > 0:
            x = np.vstack([x, np.array(new_x, dtype=np.float32)])
            y_rec = np.concatenate([y_rec, np.array(new_rec, dtype=np.float32)])
            y_1to1 = np.concatenate([y_1to1, np.array(new_1to1, dtype=np.float32)])
            y_long = np.concatenate([y_long, np.array(new_long, dtype=np.float32)])
            y_dz = np.concatenate([y_dz, np.array(new_dz, dtype=np.float32)])
            y_w = np.concatenate([y_w, np.array(new_w, dtype=np.float32)])

            M_new = M + K_add
            af_new = np.zeros((M_new, M_new, 8), dtype=np.float32)
            am_new = np.zeros((M_new, M_new), dtype=np.float32)
            af_new[:M, :M] = af
            am_new[:M, :M] = am
            for k_i, (c_idx, is_dup, sep_norm, z_ov) in enumerate(edge_specs):
                u_new = M + k_i
                am_new[u_new, :M] = am[c_idx, :M]
                am_new[:M, u_new] = am[:M, c_idx]
                af_new[u_new, :M] = af[c_idx, :M]
                af_new[:M, u_new] = af[:M, c_idx]
                am_new[u_new, c_idx] = 1.0
                am_new[c_idx, u_new] = 1.0
                dup_flag = 1.0 if is_dup else 0.0
                d_pm = float(x[c_idx, 5] - x[u_new, 5])
                d_cf = float(x[c_idx, 0] - x[u_new, 0])
                af_new[u_new, c_idx] = [sep_norm, sep_norm, 0.04 if is_dup else 0.08, dup_flag, d_pm, d_cf, 0.25 if is_dup else 0.0, z_ov]
                af_new[c_idx, u_new] = [sep_norm, sep_norm, 0.04 if is_dup else 0.08, dup_flag, -d_pm, -d_cf, -0.25 if is_dup else 0.0, z_ov]
            af, am = af_new, am_new

    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "adj_feat": torch.tensor(af, dtype=torch.float32),
        "adj_mask": torch.tensor(am, dtype=torch.float32),
        "y_rec": torch.tensor(y_rec, dtype=torch.float32),
        "y_1to1": torch.tensor(y_1to1, dtype=torch.float32),
        "y_long_1to1": torch.tensor(y_long, dtype=torch.float32),
        "y_dz_tip": torch.tensor(y_dz, dtype=torch.float32),
        "y_dz_weight": torch.tensor(y_w, dtype=torch.float32),
    }


# ==============================================================================
# 4. STRICTLY NESTED STAGE 2A, STAGE 2B, AND STAGE 2C TRAINING
# ==============================================================================

def get_nested_fold_split(f_out, n_cases=54):
    """
    Deterministic nested cross-validation split for outer fold `f_out in 0..4`:
      - outer_test: 10 or 11 cases (`i % 5 == f_out`)
      - outer_train: 43 or 44 cases (`i % 5 != f_out`)
      - inner_val: 8 or 9 cases (`pos % 5 == f_out` inside outer_train)
      - inner_subtrain: 35 cases (`pos % 5 != f_out` inside outer_train)
    """
    outer_test = [i for i in range(n_cases) if i % 5 == f_out]
    outer_train = [i for i in range(n_cases) if i % 5 != f_out]
    inner_val = [idx for pos, idx in enumerate(outer_train) if pos % 5 == f_out]
    inner_subtrain = [idx for pos, idx in enumerate(outer_train) if pos % 5 != f_out]
    return outer_test, outer_train, inner_subtrain, inner_val


def train_nested_stage2a_mpnn(graphs, device, max_epochs=40):
    """
    Trains Stage 2A (`NeedleSliceMPNN`) with strictly nested inner validation:
    For each outer fold `f_out in 0..4`:
      - Trains strictly on `inner_subtrain` (35 cases)
      - Evaluates at every epoch on `inner_val` (8-9 cases)
      - Selects the checkpoint with lowest `inner_val` loss (never touching `outer_test`!)
      - Decodes `oof_mpnn_trajs` on `outer_test`.
    """
    oof_mpnn_trajs = [None] * len(graphs)
    selected_epochs = []

    for f_out in range(5):
        outer_test, _, inner_subtrain, inner_val = get_nested_fold_split(f_out, len(graphs))
        torch.manual_seed(42 + f_out)
        np.random.seed(42 + f_out)

        model = NeedleSliceMPNN(in_node_dim=16, in_edge_dim=12, in_sp_dim=5, hidden_dim=64, num_layers=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-4)

        best_val_loss = float("inf")
        best_state = None
        best_ep = 0
        pw_n = torch.tensor([3.0], device=device)
        pw_e = torch.tensor([6.0], device=device)

        def _mpnn_loss(g_item):
            no, el = model(g_item, device)
            y_na = g_item["nodes_y_all"].to(device)
            y_nl = g_item["nodes_y_long"].to(device)
            y_off = g_item["nodes_offset"].to(device)
            y_e = g_item["edge_y"].to(device)
            fg_n = (y_na > 0.5).float().unsqueeze(1)
            loss = (
                F.binary_cross_entropy_with_logits(no[:, 0], y_na, pos_weight=pw_n)
                + 0.5 * F.binary_cross_entropy_with_logits(no[:, 1], y_nl, pos_weight=pw_n)
                + 0.5 * (F.smooth_l1_loss(no[:, 2:4], y_off, reduction="none") * fg_n).sum() / fg_n.sum().clamp(min=1.0)
            )
            if el.numel() > 0:
                loss = loss + 2.0 * F.binary_cross_entropy_with_logits(el, y_e, pos_weight=pw_e)
            return loss

        for ep in range(max_epochs):
            model.train()
            for idx in np.random.permutation(inner_subtrain):
                opt.zero_grad(set_to_none=True)
                loss = _mpnn_loss(graphs[idx])
                loss.backward()
                opt.step()
            sched.step()

            if ep >= 12:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for idx in inner_val:
                        val_losses.append(float(_mpnn_loss(graphs[idx]).item()))
                mean_val = float(np.mean(val_losses))
                if mean_val < best_val_loss:
                    best_val_loss = mean_val
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_ep = ep + 1

        selected_epochs.append(best_ep)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            for idx in outer_test:
                no, el = model(graphs[idx], device)
                oof_mpnn_trajs[idx] = decode_mpnn_trajectories(graphs[idx], no, el)

    print(f"  [Nested Stage 2A MPNN] Inner-val selected epochs per fold: {selected_epochs}", flush=True)
    return oof_mpnn_trajs, selected_epochs


def build_fold_inner_oof_bundles(stitched_base_bgs, stitched_enriched_bgs, oof_dir, all_X_vols, meta_list, graphs):
    """
    Builds fold-specific inner-OOF bundle graphs (`inner_oof_base_by_fold`, `inner_oof_enr_by_fold`)
    using `{oof_dir}/inner_oof_f{f_out}_case_{idx:02d}.npz` for `idx in outer_train` in each fold `f_out`.
    Reuses the exact pool trajectories and Hungarian ground-truth assignments while recomputing all
    Stage 1 probability features (`x18`, `x26`, `adj_feat`, `tip_profiles_91`) from `inner_oof_f{f_out}`.
    """
    n_cases = len(stitched_enriched_bgs)
    inner_oof_base_by_fold = {}
    inner_oof_enr_by_fold = {}

    for f_out in range(5):
        _, outer_train, _, _ = get_nested_fold_split(f_out, n_cases)
        base_f = {}
        enr_f = {}
        for idx in outer_train:
            inner_path = os.path.join(oof_dir, f"inner_oof_f{f_out}_case_{idx:02d}.npz")
            if not os.path.exists(inner_path):
                base_f[idx] = stitched_base_bgs[idx]
                enr_f[idx] = stitched_enriched_bgs[idx]
                continue

            oof_raw = np.load(inner_path)
            X_vol = all_X_vols[idx]
            oof_cal = calibrate_stage1_oof_for_volume(oof_raw, X_vol, meta_list[idx]["sp_ras"])
            g = graphs[idx]
            sb_orig = stitched_base_bgs[idx]
            se_orig = stitched_enriched_bgs[idx]
            pool = se_orig["pool"]
            M = len(pool)

            bases = np.array([p["base_xy"] for p in pool]) if M else np.zeros((0, 2))
            slopes = np.array([p["slope"] for p in pool]) if M else np.zeros((0, 2))
            med_base = np.median(bases, axis=0) if M else np.zeros(2)
            med_slope = np.median(slopes, axis=0) if M else np.zeros(2)

            prof_list = [compute_trajectory_axial_profile_and_tip_features_robust(p["ras"], g, oof_cal, X_vol) for p in pool]
            z_tops = np.array([float(p["ras"][-1, 2]) for p in pool]) if M else np.zeros(0)
            z_bots = np.array([float(p["ras"][0, 2]) for p in pool]) if M else np.zeros(0)
            qual_weights = np.array([
                max(0.05, pr["mean_pm"] * 0.4 + pr["mean_pp"] * 0.3 + pr["mean_sym"] * 0.3)
                for pr in prof_list
            ]) if M else np.zeros(0)

            z_tip_nbr = np.zeros(M, dtype=np.float32)
            for u in range(M):
                dists_xy = np.hypot(bases[:, 0] - bases[u, 0], bases[:, 1] - bases[u, 1])
                span_u = z_tops[u] - z_bots[u]
                same_class = np.abs((z_tops - z_bots) - span_u) <= 22.0
                nbr_mask = (dists_xy <= 28.0) & same_class & (qual_weights >= 0.25)
                if np.sum(nbr_mask) >= 2:
                    w_n = qual_weights[nbr_mask]
                    z_tip_nbr[u] = float(np.sum(z_tops[nbr_mask] * w_n) / np.sum(w_n))
                else:
                    z_tip_nbr[u] = float(z_tops[u])

            x18_feats = []
            x26_feats = []
            tip_profiles_91 = []
            for u, p in enumerate(pool):
                ras = p["ras"]
                pr = prof_list[u]
                d_base = float(np.hypot(p["base_xy"][0] - med_base[0], p["base_xy"][1] - med_base[1])) / 40.0
                d_slope = float(np.hypot(p["slope"][0] - med_slope[0], p["slope"][1] - med_slope[1])) / 0.25
                z_low = float(ras[0, 2] - g["z_base_ras"]) / 50.0
                z_high = float(ras[-1, 2] - g["z_base_ras"]) / 100.0
                dz_prof = float(pr["z_prof_tip"] - ras[-1, 2]) / 20.0
                dz_nbr = float(z_tip_nbr[u] - ras[-1, 2]) / 20.0

                f18 = [
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
                ]
                f26 = f18 + [
                    pr["mean_sym"],
                    pr["p75_sym"],
                    pr["mean_cs"],
                    pr["mean_rc"],
                    pr["frac_active"],
                    pr["tip_sup_cur"],
                    dz_prof,
                    dz_nbr,
                ]
                x18_feats.append(f18)
                x26_feats.append(f26)
                tip_profiles_91.append(extract_1d_tip_profile_91(ras, g, oof_cal, X_vol, z_center=float(ras[-1, 2]), z_tip_nbr=float(z_tip_nbr[u])))

            x18_t = torch.tensor(x18_feats, dtype=torch.float32)
            x26_t = torch.tensor(x26_feats, dtype=torch.float32)
            af6_t = sb_orig["adj_feat"].clone()
            af8_t = se_orig["adj_feat"].clone()
            if M > 0:
                d_pm_mat = x26_t[:, 5].unsqueeze(0) - x26_t[:, 5].unsqueeze(1)
                am_bool = sb_orig["adj_mask"] > 0.5
                af6_t[:, :, 4] = torch.where(am_bool, d_pm_mat, torch.zeros_like(d_pm_mat))
                af8_t[:, :, 4] = torch.where(am_bool, d_pm_mat, torch.zeros_like(d_pm_mat))

            b_new = dict(sb_orig)
            b_new["x"] = x18_t
            b_new["adj_feat"] = af6_t
            e_new = dict(se_orig)
            e_new["x"] = x26_t
            e_new["adj_feat"] = af8_t
            e_new["tip_profiles_91"] = torch.tensor(np.stack(tip_profiles_91, axis=0), dtype=torch.float32) if M > 0 else se_orig["tip_profiles_91"]
            base_f[idx] = b_new
            enr_f[idx] = e_new

        inner_oof_base_by_fold[f_out] = base_f
        inner_oof_enr_by_fold[f_out] = enr_f

    return inner_oof_base_by_fold, inner_oof_enr_by_fold


def train_nested_stage2b_models(
    base_bundle_items,
    enriched_bundle_items,
    device,
    use_multirater=True,
    use_synthbrachy3t=True,
    inner_oof_base_by_fold=None,
    inner_oof_enr_by_fold=None,
):
    """
    Trains Stage 2B (`DenseGraphTransformer` and `TipAwareBundleGraphTransformer`) with
    strictly nested CV (`inner_subtrain` / `inner_val` checkpoint selection):
      - When `use_multirater=True`, uses the Multi-Rater (CY+HT + LB/AMe) training targets
        ONLY on `inner_subtrain` / `inner_val` (never on `outer_test`).
      - When `use_synthbrachy3t=True`, pre-trains `TipAwareBundleGraphTransformer` on
        SynthBrachy3T synthetic bundles and co-trains with fold-isolated domain-randomized
        bundles on `inner_subtrain`, selecting the best epoch on `inner_val`.
      - When `inner_oof_base_by_fold` / `inner_oof_enr_by_fold` are provided, trains and
        validates on fold-specific `inner_oof_f{f_out}` bundle features and evaluates on
        `oof_case_{i}` for `outer_test`.
    """
    n_cases = len(enriched_bundle_items)
    base_oof_scores = [None] * n_cases
    base_sel_epochs = []

    # 1. Nested CV for 18-feature DenseGraphTransformer
    for f_out in range(5):
        outer_test, _, inner_subtrain, inner_val = get_nested_fold_split(f_out, n_cases)
        torch.manual_seed(42 + f_out)
        np.random.seed(42 + f_out)
        b_fold_map = inner_oof_base_by_fold[f_out] if inner_oof_base_by_fold is not None else base_bundle_items

        gt_model = DenseGraphTransformer(in_dim=18, edge_dim=6, d_model=64).to(device)
        opt = torch.optim.AdamW(gt_model.parameters(), lr=1.8e-3, weight_decay=1e-4)
        pw_r = torch.tensor([3.0], device=device)
        pw_1 = torch.tensor([2.0], device=device)
        pw_l = torch.tensor([2.0], device=device)

        best_val_loss = float("inf")
        best_state = None
        best_ep = 0

        def _base_loss(bg):
            out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
            yr = (bg["y_rec"] if use_multirater else bg["y_rec_single"]).to(device)
            y1 = (bg["y_1to1"] if use_multirater else bg["y_1to1_single"]).to(device)
            yl = (bg["y_long_1to1"] if use_multirater else bg["y_long_single"]).to(device)
            return (
                F.binary_cross_entropy_with_logits(out[:, 0], yr, pos_weight=pw_r)
                + 1.5 * F.binary_cross_entropy_with_logits(out[:, 1], y1, pos_weight=pw_1)
                + F.binary_cross_entropy_with_logits(out[:, 2], yl, pos_weight=pw_l)
            )

        for ep in range(75):
            gt_model.train()
            for idx in np.random.permutation(inner_subtrain):
                opt.zero_grad(set_to_none=True)
                loss = _base_loss(b_fold_map[idx])
                loss.backward()
                opt.step()

            if ep >= 20:
                gt_model.eval()
                with torch.no_grad():
                    v_loss = float(np.mean([_base_loss(b_fold_map[idx]).item() for idx in inner_val]))
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_state = {k: v.cpu().clone() for k, v in gt_model.state_dict().items()}
                    best_ep = ep + 1

        base_sel_epochs.append(best_ep)
        gt_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        gt_model.eval()
        with torch.no_grad():
            for idx in outer_test:
                bg = base_bundle_items[idx]
                out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                base_oof_scores[idx] = torch.sigmoid(out).cpu().numpy()

    # 2. Nested CV for 26-feature TipAwareBundleGraphTransformer (3-seed ensemble, inner_val checkpointed)
    pw_r = torch.tensor([3.2], device=device)
    pw_1 = torch.tensor([2.6], device=device)
    pw_l = torch.tensor([2.4], device=device)

    pretrained_state = None
    if use_synthbrachy3t:
        rng_syn = np.random.default_rng(2026)
        pure_synth_gpu = [to_gpu_item(generate_synthetic_bundle_graph_item_v2(rng_syn), device) for _ in range(300)]
        torch.manual_seed(2026)
        base_syn_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
        opt_syn = torch.optim.AdamW(base_syn_model.parameters(), lr=2.0e-3, weight_decay=1e-4)
        for ep in range(20):
            base_syn_model.train()
            for bi in np.random.permutation(len(pure_synth_gpu)):
                bg = pure_synth_gpu[bi]
                opt_syn.zero_grad(set_to_none=True)
                out = base_syn_model(bg["x"], bg["adj_feat"], bg["adj_mask"])
                w_tip = bg["y_dz_weight"]
                l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"], reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
                loss = (
                    F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"], pos_weight=pw_r)
                    + 1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"], pos_weight=pw_1)
                    + F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"], pos_weight=pw_l)
                    + 1.2 * l_tip
                )
                loss.backward()
                opt_syn.step()
        pretrained_state = {k: v.cpu().clone() for k, v in base_syn_model.state_dict().items()}

    def _prepare_bg_for_supervision(b_orig):
        b = dict(b_orig)
        if not use_multirater:
            b["y_rec"] = b_orig["y_rec_single"]
            b["y_1to1"] = b_orig["y_1to1_single"]
            b["y_long_1to1"] = b_orig["y_long_single"]
            b["y_dz_tip"] = b_orig["y_dz_tip_single"]
            b["y_dz_weight"] = b_orig["y_dz_weight_single"]
        return b

    ensemble_seeds = [100, 200, 300]
    per_seed_oof = {}
    tip_aware_sel_epochs = []

    for sb in ensemble_seeds:
        oof_s = [None] * n_cases
        for f_out in range(5):
            outer_test, _, inner_subtrain, inner_val = get_nested_fold_split(f_out, n_cases)
            torch.manual_seed(sb + f_out)
            np.random.seed(sb + f_out)
            rng_f = np.random.default_rng(sb + 400 + f_out)
            e_fold_map = inner_oof_enr_by_fold[f_out] if inner_oof_enr_by_fold is not None else enriched_bundle_items

            subtrain_raw = [_prepare_bg_for_supervision(e_fold_map[idx]) for idx in inner_subtrain]
            val_gpu = [to_gpu_item(_prepare_bg_for_supervision(e_fold_map[idx]), device) for idx in inner_val]

            tr_aug_gpu = [to_gpu_item(b, device) for b in subtrain_raw]
            if use_synthbrachy3t:
                for _ in range(4):
                    for b in subtrain_raw:
                        tr_aug_gpu.append(to_gpu_item(augment_clinical_bundle_with_synthbrachy3t_v2(b, rng_f), device))

            gt_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
            if pretrained_state is not None:
                gt_model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})

            n_epochs = 45 if use_synthbrachy3t else 65
            opt = torch.optim.AdamW(gt_model.parameters(), lr=1.5e-3, weight_decay=1.5e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-4)

            best_val_loss = float("inf")
            best_state = None
            best_ep = 0

            def _eval_loss(bg_gpu):
                out = gt_model(bg_gpu["x"], bg_gpu["adj_feat"], bg_gpu["adj_mask"])
                w_tip = bg_gpu["y_dz_weight"]
                l_tip = (F.smooth_l1_loss(out[:, 3], bg_gpu["y_dz_tip"], reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
                return (
                    F.binary_cross_entropy_with_logits(out[:, 0], bg_gpu["y_rec"], pos_weight=pw_r)
                    + 1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg_gpu["y_1to1"], pos_weight=pw_1)
                    + F.binary_cross_entropy_with_logits(out[:, 2], bg_gpu["y_long_1to1"], pos_weight=pw_l)
                    + 1.5 * l_tip
                )

            for ep in range(n_epochs):
                gt_model.train()
                for bi in np.random.permutation(len(tr_aug_gpu)):
                    opt.zero_grad(set_to_none=True)
                    loss = _eval_loss(tr_aug_gpu[bi])
                    loss.backward()
                    opt.step()
                sched.step()

                if ep >= 12:
                    gt_model.eval()
                    with torch.no_grad():
                        v_loss = float(np.mean([_eval_loss(vg).item() for vg in val_gpu]))
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_state = {k: v.cpu().clone() for k, v in gt_model.state_dict().items()}
                        best_ep = ep + 1

            if sb == ensemble_seeds[0]:
                tip_aware_sel_epochs.append(best_ep)

            gt_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            gt_model.eval()
            with torch.no_grad():
                for idx in outer_test:
                    bg = enriched_bundle_items[idx]
                    out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                    oof_s[idx] = out.cpu().numpy()

        per_seed_oof[sb] = oof_s

    ens_oof_outputs = [
        np.mean(np.stack([per_seed_oof[sb][i] for sb in ensemble_seeds], axis=0), axis=0)
        for i in range(n_cases)
    ]
    print(
        f"  [Nested Stage 2B (multirater={use_multirater}, synth={use_synthbrachy3t}, inner_oof={inner_oof_enr_by_fold is not None})] "
        f"Inner-val epochs: base={base_sel_epochs}, tip_aware={tip_aware_sel_epochs}",
        flush=True,
    )
    return base_oof_scores, ens_oof_outputs, {"base_epochs": base_sel_epochs, "tip_aware_epochs": tip_aware_sel_epochs}


def shift_1d_profile(prof, shift_steps):
    """
    Fast 1D shift-equivariant augmentation along the 0.5 mm 91-step axial grid:
    shift_steps > 0 moves z_center DOWN by shift_steps * 0.5 mm, shifting features to t + shift_steps.
    """
    out = np.zeros_like(prof)
    L = prof.shape[1]
    if shift_steps == 0:
        return prof.copy()
    elif shift_steps > 0:
        out[:, shift_steps:] = prof[:, :L - shift_steps]
        out[:, :shift_steps] = prof[:, 0:1]
    else:
        s = -shift_steps
        out[:, :L - s] = prof[:, s:]
        out[:, L - s:] = prof[:, -1:]
    dz_grid = np.linspace(-25.0, 20.0, L, dtype=np.float32)
    out[14] = dz_grid / 25.0
    out[16] = prof[16] - (shift_steps * 0.5) / 25.0
    return out


def train_nested_stage2c_1d_tip_localizer(enriched_bundle_items, graphs, all_oof_cal, all_X_vols, device, use_multirater=True):
    """
    Trains `Dedicated1DTipLocalizer` (Item 4) under strictly nested CV:
    Inside each outer fold `f_out in 0..4`:
      - Extracts 91-step 22-channel 1D profiles on `inner_subtrain` (with 1D axial shift
        perturbations in [-12, +12] mm so the 1D CNN + Self-Attention model sees tips across
        the entire 91-step window!)
      - Supervises with Multi-Rater Tip Consensus Weighting (`y_true_tip_z`, `y_dz_weight`)
      - Selects the best checkpoint strictly on `inner_val`
      - Returns the 5 fold-isolated trained `Dedicated1DTipLocalizer` models so they can be
        applied AFTER Collinear Fragment Stitching & Cranial Extension on `outer_test`!
    """
    fold_models = [None] * 5
    sel_epochs = []
    dz_grid_np = np.linspace(-25.0, 20.0, 91, dtype=np.float32)

    # Pre-build training/validation profile samples per case
    case_samples = []
    for ci, bg in enumerate(enriched_bundle_items):
        pool = bg["pool"]
        y_w = (bg["y_dz_weight"] if use_multirater else bg["y_dz_weight_single"]).numpy()
        y_true_z = (bg["y_true_tip_z"] if use_multirater else (np.array([p["ras"][-1, 2] for p in pool]) + bg["y_dz_tip_single"].numpy() * 15.0))
        if isinstance(y_true_z, torch.Tensor):
            y_true_z = y_true_z.numpy()

        profs_orig = bg["tip_profiles_91"].numpy()
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
            samples_c.append((profs_orig[u], float(dz_true), float(y_w[u]), False))
            # Add 4 axial-shift perturbations per true trajectory for shift-equivariant training
            for _ in range(4):
                shift_steps = int(rng_c.integers(-24, 25))
                shift_mm = shift_steps * 0.5
                dz_new = dz_true + shift_mm
                if -21.0 <= dz_new <= 17.0:
                    prof_sh = shift_1d_profile(profs_orig[u], shift_steps)
                    samples_c.append((prof_sh, float(dz_new), float(y_w[u]) * 0.85, True))
        case_samples.append(samples_c)

    for f_out in range(5):
        _, _, inner_subtrain, inner_val = get_nested_fold_split(f_out, len(enriched_bundle_items))
        torch.manual_seed(500 + f_out)
        np.random.seed(500 + f_out)

        tr_list = [s for idx in inner_subtrain for s in case_samples[idx]]
        val_list = [s for idx in inner_val for s in case_samples[idx] if not s[3]]

        X_tr = torch.tensor(np.stack([s[0] for s in tr_list], axis=0), dtype=torch.float32, device=device)
        Y_tr = torch.tensor([s[1] for s in tr_list], dtype=torch.float32, device=device)
        W_tr = torch.tensor([s[2] for s in tr_list], dtype=torch.float32, device=device)

        # Build Gaussian step distribution targets [N_tr, 91] with sigma = 2.0 mm
        dz_grid_t = torch.tensor(dz_grid_np, dtype=torch.float32, device=device).unsqueeze(0)
        Y_gauss_tr = torch.exp(-0.5 * ((dz_grid_t - Y_tr.unsqueeze(1)) / 2.0) ** 2)
        Y_gauss_tr = Y_gauss_tr / Y_gauss_tr.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        X_val = torch.tensor(np.stack([s[0] for s in val_list], axis=0), dtype=torch.float32, device=device)
        Y_val = torch.tensor([s[1] for s in val_list], dtype=torch.float32, device=device)
        W_val = torch.tensor([s[2] for s in val_list], dtype=torch.float32, device=device)

        model = Dedicated1DTipLocalizer(in_channels=22, hidden_dim=48, seq_len=91).to(device)
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

    print(f"  [Nested Stage 2C 1D Tip Localizer] Inner-val selected epochs per fold: {sel_epochs}", flush=True)
    return fold_models, sel_epochs


# ==============================================================================
# 5. UPGRADED STAGE 2B DECODER WITH COLLINEAR DEDUP + CRANIAL EXTENSION + 1D TIP
# ==============================================================================

def decode_clean_stage2_preds(
    bundle_items,
    base_oof_scores,
    synth_oof_outputs,
    graphs=None,
    all_oof_cal=None,
    all_X_vols=None,
    tip_models_5fold=None,
    device=None,
    min_conf=50.0,
    long_mode=False,
    w_synth=0.60,
    use_collinear_dedup=True,
    use_1d_tip_model=True,
    tip_scale=0.36,
    calibrate_pos_weight=True,
):
    """
    Decodes 3D catheter trajectories from Stage 2B scores with:
      1. Prior-weight probability calibration (correcting the +14% score shift induced by
         `pos_weight=[3.2, 2.6, 2.4]` during nested BCE training so `tau=50%` is the
         calibrated 50% posterior operating point)
      2. Length-completeness tie-breaking (ranking full-length shafts ahead of short stubs
         when two collinear candidates on the same needle have similar neural probabilities)
      3. IRGM spatial hull gating + Stage 0 circular dark-core symmetry verification
      4. Extrapolation-Aware Collinear Deduplication (`collinear_traj_dist`) and
         Dome-Gated Cranial Fragment Extension
      5. Post-Stitching Dedicated 1D Axial Tip Localization (`Dedicated1DTipLocalizer` applied
         AFTER collinear stitching/extension so `z_top` is already near the true cranial end!)
    """
    preds_out = []
    eff_min_conf = (min_conf + 15.5) if calibrate_pos_weight else min_conf

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

        if use_collinear_dedup:
            # Length-completeness tie-breaker so full-length shafts rank ahead of short fragments on the same needle
            for u_idx, p_cand in enumerate(pool):
                span_u = float(p_cand.get("z_span", 20.0))
                if span_u < 18.0 and not long_mode:
                    conf_arr[u_idx] -= 5.0 * (18.0 - span_u) / 10.0
                elif span_u >= 35.0 and sc[u_idx, 0] >= 0.75:
                    conf_arr[u_idx] += min(3.5, (span_u - 30.0) / 20.0)

        # Fallback for any high-resolution scan where y_rec head is >= 0.90 and Stage 0 + symmetry are strong
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

            if not use_collinear_dedup:
                dedup_thresh = 1.60 if (conf_arr[idx_c] >= eff_min_conf + 12.0 and cand.get("mpnn_pa", 0.0) >= 0.50) else 2.02
                if any(traj_gt_dist(pts, s["raw_ras"]) < dedup_thresh for s in sel):
                    continue
                cand["raw_ras"] = pts
                cand["idx_c"] = int(idx_c)
                cand["score"] = float(conf_arr[idx_c])
                sel.append(cand)
            else:
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

        if use_collinear_dedup:
            # Second pass: only for severely undershot trajectories (`z_top < z_tip_nbr - 6.0 mm`)
            for s in sel:
                z_cur_top = float(s["raw_ras"][-1, 2])
                z_nbr_s = float(s.get("z_tip_nbr", z_cur_top))
                if z_cur_top < z_nbr_s - 6.0:
                    best_ext = None
                    best_u = -1
                    best_top = z_cur_top
                    for u_idx, p_cand in enumerate(pool):
                        if sc[u_idx, 0] < 0.65 or p_cand.get("frac_active", 0.0) < 0.70 or p_cand.get("mean_sym", 0.0) < 0.45:
                            continue
                        c_ras = p_cand["ras"]
                        c_top = float(c_ras[-1, 2])
                        if c_top <= best_top + 4.0 or c_top > z_nbr_s + 8.0:
                            continue
                        info = collinear_traj_dist(c_ras, s["raw_ras"])
                        if info["z_gap"] <= 14.0 and info["d_slope"] <= 0.10 and (info["d_ov"] <= 2.60 or (info["d_ext"] <= 2.10 and info["d_mid"] <= 2.25)):
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

        # Apply tip localization & Poly3 smoothing AFTER collinear fragment stitching/extension
        f_out = i % 5
        tip_model = tip_models_5fold[f_out] if (use_1d_tip_model and tip_models_5fold is not None) else None

        for s in sel:
            idx_c = s["idx_c"]
            pts = s["raw_ras"]
            vx_top, vy_top = fit_local_slope(pts, float(pts[-1, 2]), 25.0)
            dz_prof_mm = float(s.get("z_prof_tip", pts[-1, 2]) - pts[-1, 2])
            dz_nbr_mm = float(s.get("z_tip_nbr", pts[-1, 2]) - pts[-1, 2])
            dz_s2b_mm = float(dz_pred_stage2b[idx_c])

            if tip_model is not None:
                if "tip_profiles_91" in bg and abs(float(pts[-1, 2]) - float(pool[idx_c]["ras"][-1, 2])) < 0.5:
                    x_t = bg["tip_profiles_91"][idx_c:idx_c + 1].to(device)
                elif graphs is not None and all_oof_cal is not None and all_X_vols is not None:
                    prof_91 = extract_1d_tip_profile_91(
                        pts, graphs[i], all_oof_cal[i], all_X_vols[i],
                        z_center=float(pts[-1, 2]), z_tip_nbr=float(s.get("z_tip_nbr", pts[-1, 2])),
                    )
                    x_t = torch.tensor(prof_91, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    x_t = None

                if x_t is not None:
                    with torch.no_grad():
                        dz_1d_pred, _ = tip_model(x_t)
                        dz_1d_mm = float(dz_1d_pred[0].item())
                    dz_comb_mm = 0.50 * dz_1d_mm + 0.30 * (tip_scale * dz_s2b_mm) + 0.15 * (tip_scale * dz_prof_mm) + 0.05 * (tip_scale * dz_nbr_mm)
                else:
                    dz_comb_mm = tip_scale * (0.65 * dz_s2b_mm + 0.25 * dz_prof_mm + 0.10 * dz_nbr_mm)
            else:
                dz_comb_mm = tip_scale * (0.65 * dz_s2b_mm + 0.25 * dz_prof_mm + 0.10 * dz_nbr_mm)

            pts_refined = apply_tip_adjustment_and_poly3(pts, dz_comb_mm, vx_top, vy_top)
            s["ras"] = pts_refined
            s["points"] = pts_refined
            s["dz_comb_mm"] = dz_comb_mm

        preds_out.append(sel)

    return preds_out


def compute_detailed_tip_and_fp_diagnostics(preds_per_case, gts_per_case, lb_by_case_idx):
    """Computes extra diagnostics: % tip > 10 mm, caudal undershoots, cranial overshoots, and split-fragment FPs."""
    tip_errors = []
    dz_signed_list = []
    n_undershoot_gt10 = 0
    n_overshoot_gt10 = 0
    n_split_fp = 0
    n_total_fp = 0

    for ci in range(len(gts_per_case)):
        gts = gts_per_case[ci]
        preds = preds_per_case[ci]
        n_gt, n_pred = len(gts), len(preds)
        cost = np.full((n_gt, max(1, n_pred)), 1e6, dtype=np.float64)
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                pm = compute_unified_pair_metrics(p["ras"], g)
                if pm is not None and pm["mean_shaft_xy"] <= 3.5:
                    cost[gi, pi] = pm["mean_shaft_xy"]
        ri, col_i = linear_sum_assignment(cost)
        matched_gt = set()
        matched_pred = set()
        for gi, pi in zip(ri, col_i):
            if cost[gi, pi] <= 3.5:
                matched_gt.add(int(gi))
                matched_pred.add(int(pi))
                p_ras = preds[pi]["ras"]
                g_ras = gts[gi]
                p_ras = p_ras[np.argsort(p_ras[:, 2])]
                g_ras = g_ras[np.argsort(g_ras[:, 2])]
                t3d = float(np.linalg.norm(p_ras[-1] - g_ras[-1]))
                dz = float(p_ras[-1, 2] - g_ras[-1, 2])
                tip_errors.append(t3d)
                dz_signed_list.append(dz)
                if t3d > 10.0:
                    if dz < -5.0:
                        n_undershoot_gt10 += 1
                    elif dz > 5.0:
                        n_overshoot_gt10 += 1

        for pi in range(n_pred):
            if pi in matched_pred:
                continue
            n_total_fp += 1
            p_ras = preds[pi]["ras"]
            is_split = False
            for gi in matched_gt:
                info = collinear_traj_dist(p_ras, gts[gi])
                if info["d_ov"] <= 4.0 or info["d_ext"] <= 3.8:
                    is_split = True
                    break
            if is_split:
                n_split_fp += 1

    n_matched = max(1, len(tip_errors))
    n_tip_gt10 = sum(1 for t in tip_errors if t > 10.0)
    n_tip_le3 = sum(1 for t in tip_errors if t <= 3.0)
    return {
        "n_matched": len(tip_errors),
        "n_tip_le_3mm": n_tip_le3,
        "pct_tip_le_3mm": round(100.0 * n_tip_le3 / n_matched, 2),
        "n_tip_gt_10mm": n_tip_gt10,
        "pct_tip_gt_10mm": round(100.0 * n_tip_gt10 / n_matched, 2),
        "n_caudal_undershoot_gt_10mm": n_undershoot_gt10,
        "n_cranial_overshoot_gt_10mm": n_overshoot_gt10,
        "n_total_fp": n_total_fp,
        "n_split_fragment_fp": n_split_fp,
    }


def derive_unstitched_and_base_bgs(stitched_enriched_bgs):
    """
    Reconstructs `stitched_base_bgs`, `unstitched_base_bgs`, `unstitched_enriched_bgs`,
    and `n_stitched_added_total` directly from `stitched_enriched_bgs`.
    """
    unstitched_base_bgs, unstitched_enriched_bgs = [], []
    stitched_base_bgs = []
    n_stitched_added = 0

    for sb_enr in stitched_enriched_bgs:
        pool_st = sb_enr["pool"]
        m_un = sum(1 for p in pool_st if not p.get("is_stitched", False))
        n_stitched_added += (len(pool_st) - m_un)

        sb_base = {
            "pool": pool_st,
            "x": sb_enr["x"][:, :18].clone(),
            "adj_feat": sb_enr["adj_feat"][:, :, :6].clone(),
            "adj_mask": sb_enr["adj_mask"].clone(),
            "y_rec": sb_enr["y_rec"].clone(),
            "y_1to1": sb_enr["y_1to1"].clone(),
            "y_long_1to1": sb_enr["y_long_1to1"].clone(),
            "y_rec_single": sb_enr["y_rec_single"].clone(),
            "y_1to1_single": sb_enr["y_1to1_single"].clone(),
            "y_long_single": sb_enr["y_long_single"].clone(),
        }
        stitched_base_bgs.append(sb_base)

        pool_un = [copy.deepcopy(p) for p in pool_st[:m_un]]
        ub_base = {
            "pool": pool_un,
            "x": sb_enr["x"][:m_un, :18].clone(),
            "adj_feat": sb_enr["adj_feat"][:m_un, :m_un, :6].clone(),
            "adj_mask": sb_enr["adj_mask"][:m_un, :m_un].clone(),
            "y_rec": sb_enr["y_rec"][:m_un].clone(),
            "y_1to1": sb_enr["y_1to1"][:m_un].clone(),
            "y_long_1to1": sb_enr["y_long_1to1"][:m_un].clone(),
            "y_rec_single": sb_enr["y_rec_single"][:m_un].clone(),
            "y_1to1_single": sb_enr["y_1to1_single"].clone()[:m_un],
            "y_long_single": sb_enr["y_long_single"][:m_un].clone(),
        }
        ub_enr = {
            "pool": pool_un,
            "x": sb_enr["x"][:m_un].clone(),
            "adj_feat": sb_enr["adj_feat"][:m_un, :m_un].clone(),
            "adj_mask": sb_enr["adj_mask"][:m_un, :m_un].clone(),
            "y_rec": sb_enr["y_rec"][:m_un].clone(),
            "y_1to1": sb_enr["y_1to1"][:m_un].clone(),
            "y_long_1to1": sb_enr["y_long_1to1"][:m_un].clone(),
            "y_dz_tip": sb_enr["y_dz_tip"][:m_un].clone(),
            "y_dz_weight": sb_enr["y_dz_weight"][:m_un].clone(),
            "y_true_tip_z": sb_enr["y_true_tip_z"][:m_un].clone(),
            "y_rec_single": sb_enr["y_rec_single"][:m_un].clone(),
            "y_1to1_single": sb_enr["y_1to1_single"][:m_un].clone(),
            "y_long_single": sb_enr["y_long_single"][:m_un].clone(),
            "y_dz_tip_single": sb_enr["y_dz_tip_single"][:m_un].clone(),
            "y_dz_weight_single": sb_enr["y_dz_weight_single"][:m_un].clone(),
            "tip_profiles_91": sb_enr["tip_profiles_91"][:m_un].clone(),
            "gts_all": sb_enr["gts_all"],
            "gts_long": sb_enr["gts_long"],
        }
        unstitched_base_bgs.append(ub_base)
        unstitched_enriched_bgs.append(ub_enr)

    return unstitched_base_bgs, unstitched_enriched_bgs, stitched_base_bgs, n_stitched_added


def main():
    parser = argparse.ArgumentParser(description="Clean Nested Stage 2 + 1D Tip Localizer + Fragment Stitcher")
    parser.add_argument("--cache-dir", type=str, default="/tmp/needle_unet3d_cache_clean")
    parser.add_argument("--oof-dir", type=str, default="/home/guillaume/github/needlefinder/nn_models_clean")
    parser.add_argument("--meta-path", type=str, default="/tmp/cases_meta_54_clean.pkl")
    parser.add_argument("--base-trajs-path", type=str, default="/tmp/base_trajs_54cases_clean.pkl")
    parser.add_argument("--out-json", type=str, default="/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-and-tip-clean-final.json")
    parser.add_argument("--out-preds-pkl", type=str, default="/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-clean-final-oof-predictions.pkl")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild-bundles", action="store_true", help="Force rebuilding slice graphs & bundle pools even if cached")
    parser.add_argument("--use-inner-oof-crossfit", action="store_true", help="Use inner_oof_f{f_out}_case_{i:02d}.npz as primary Stage 2B training features")
    args = parser.parse_args()

    t_start = time.time()
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    device = torch.device(args.device)
    print(f"[Clean Stage 2 & 1D Tip Pipeline] Device={device}, cache_dir={args.cache_dir}, oof_dir={args.oof_dir}", flush=True)

    meta_list = pickle.load(open(args.meta_path, "rb"))
    base_trajs_data = pickle.load(open(args.base_trajs_path, "rb"))
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]
    gts_short_list = []
    for ci in range(54):
        short_i = [ga for ga in gts_all_list[ci] if not any(traj_gt_dist(ga, gl) < 2.0 for gl in gts_long_list[ci])]
        gts_short_list.append(short_i)

    adhesion_info = analyze_catheter_adhesion(meta_list, gts_all_list, gts_long_list, adhesion_thresh_mm=3.0)
    adh_idx = adhesion_info["adhesion_case_indices"]

    # Load Annotator 2 (LB/AMe) traces for the 40 dual-traced cases (strictly used only inside outer_train!)
    lb_by_case_idx = {}
    if all("lb_gts" in m for m in meta_list):
        for ci, m in enumerate(meta_list):
            if m["lb_gts"] and len(m["lb_gts"]) > 0:
                lb_by_case_idx[ci] = get_gts(m["lb_gts"])
    else:
        overlap_nums, cy_40, lb_40 = load_human_interobserver_40cases()
        case_nums = [int(m["case_num"]) for m in meta_list]
        for cnum, lb_nds in zip(overlap_nums, lb_40):
            if cnum in case_nums:
                lb_by_case_idx[case_nums.index(cnum)] = lb_nds
    print(f"  [Multi-Rater Supervision] Mapped {len(lb_by_case_idx)}/40 dual-traced cases ({sum(len(v) for v in lb_by_case_idx.values())} LB/AMe trajectories).", flush=True)

    oof_tag = os.path.basename(os.path.normpath(args.oof_dir))
    bundle_cache_path = f"/tmp/stage2_clean_bundles_{oof_tag}.pkl"

    graphs = None
    all_X_vols = None
    all_oof_cal = None
    stage0_preds = None
    cached_stitched_bgs = None

    if not args.rebuild_bundles and os.path.exists(bundle_cache_path):
        c_data = pickle.load(open(bundle_cache_path, "rb"))
        if c_data.get("oof_dir") == args.oof_dir:
            cached_stitched_bgs = c_data["stitched_enriched_bgs"]
            mpnn_epochs = c_data.get("mpnn_epochs", [17, 13, 22, 20, 16])
            graphs = c_data.get("graphs")
            all_oof_cal = c_data.get("all_oof_cal")
            stage0_preds = c_data.get("stage0_preds")
            print(f"[1-3/5] [Cache Hit] Loaded precomputed bundle graphs for {args.oof_dir} from {bundle_cache_path}!", flush=True)
    elif not args.rebuild_bundles and oof_tag == "nn_models_eval" and os.path.exists(args.out_preds_pkl):
        c_data = pickle.load(open(args.out_preds_pkl, "rb"))
        if "stitched_enriched_bgs" in c_data:
            cached_stitched_bgs = c_data["stitched_enriched_bgs"]
            mpnn_epochs = [17, 13, 22, 20, 16]
            print(f"[1-3/5] [Cache Hit] Loaded precomputed stitched_enriched_bgs from {args.out_preds_pkl}!", flush=True)

    if cached_stitched_bgs is not None:
        stitched_enriched_bgs = cached_stitched_bgs
        unstitched_base_bgs, unstitched_enriched_bgs, stitched_base_bgs, n_stitched_added_total = derive_unstitched_and_base_bgs(stitched_enriched_bgs)
        if all_X_vols is None:
            all_X_vols = [torch.load(f"{args.cache_dir}/case_{i:02d}.pt", map_location="cpu", weights_only=False)["X"].numpy() for i in range(54)]
        if all_oof_cal is None:
            all_oof_cal = [calibrate_stage1_oof_for_volume(np.load(f"{args.oof_dir}/oof_case_{i:02d}.npz"), all_X_vols[i], meta_list[i]["sp_ras"]) for i in range(54)]
        print(f"  [Pool Stitching] Derived {n_stitched_added_total} stitched full-length trajectories across 54 cases.", flush=True)
    else:
        # ------------------------------------------------------------------
        # Step 1: Load & Resolution-Calibrate Stage 1 OOF Maps + Build Slice Graphs
        # ------------------------------------------------------------------
        print("[1/5] Loading 54 cases, applying resolution-robust Stage 1 calibration, and building slice graphs...", flush=True)
        graphs = []
        all_X_vols = []
        all_oof_cal = []
        stage0_preds = []

        for i in range(54):
            d_pt = torch.load(f"{args.cache_dir}/case_{i:02d}.pt", map_location="cpu", weights_only=False)
            oof_raw = np.load(f"{args.oof_dir}/oof_case_{i:02d}.npz")
            X = d_pt["X"].numpy()
            oof_cal = calibrate_stage1_oof_for_volume(oof_raw, X, meta_list[i]["sp_ras"])
            all_X_vols.append(X)
            all_oof_cal.append(oof_cal)

            b_item = base_trajs_data[i]
            b_trajs = b_item["trajs"] if isinstance(b_item, dict) and "trajs" in b_item else b_item
            b_clean = []
            for t in b_trajs:
                tc = dict(t)
                tc["ras"] = np.asarray(tc["ras"] if "ras" in tc else tc["points"], dtype=np.float64)
                tc["points"] = tc["ras"]
                b_clean.append(tc)
            stage0_preds.append(b_clean)

            # Combine CY+HT and LB/AMe for slice-node supervision (only used when case i is in inner_subtrain)
            gts_all_super = list(gts_all_list[i])
            if i in lb_by_case_idx:
                for l_pts in lb_by_case_idx[i]:
                    l_arr = np.asarray(l_pts, dtype=np.float64)
                    l_arr = l_arr[np.argsort(l_arr[:, 2])]
                    if min([traj_gt_dist(l_arr, g) for g in gts_all_super] + [999.0]) > 3.5:
                        gts_all_super.append(l_arr)

            g = build_slice_graph_fast(
                X,
                oof_cal["p_mask"],
                oof_cal["p_peak"],
                oof_cal["vx"],
                oof_cal["vy"],
                oof_cal["p_obt"],
                oof_cal["k0"],
                oof_cal["i0"],
                oof_cal["j0"],
                meta_list[i]["sp_ras"],
                np.asarray(meta_list[i]["ijk_to_ras"], dtype=np.float64),
                gts_all=gts_all_super,
                gts_long=gts_long_list[i],
            )
            graphs.append(g)

        # ------------------------------------------------------------------
        # Step 2: Strictly Nested Stage 2A (`NeedleSliceMPNN`) on CPU to avoid dense-graph GPU OOM
        # ------------------------------------------------------------------
        print("[2/5] Training Strictly Nested Stage 2A (NeedleSliceMPNN) across 5 folds...", flush=True)
        oof_mpnn_trajs, mpnn_epochs = train_nested_stage2a_mpnn(graphs, torch.device("cpu"), max_epochs=40)

        # ------------------------------------------------------------------
        # Step 3: Build Unstitched vs Stitched Candidate Pools & Bundle Graphs
        # ------------------------------------------------------------------
        print("[3/5] Building Candidate Pools (Unstitched vs Collinear-Fragment-Stitched) and Bundle Graphs...", flush=True)
        unstitched_base_bgs, unstitched_enriched_bgs = [], []
        stitched_base_bgs, stitched_enriched_bgs = [], []
        n_stitched_added_total = 0

        for i, g in enumerate(graphs):
            X = all_X_vols[i]
            oof_cal = all_oof_cal[i]
            b_clean = stage0_preds[i]
            k0, i0, j0 = oof_cal["k0"], oof_cal["i0"], oof_cal["j0"]
            sp_i, sp_j, sp_k = [float(x) for x in meta_list[i]["sp_ras"]]
            ijk2ras = g["ijk_to_ras"]
            ras2ijk = np.linalg.inv(ijk2ras)
            resp_approx = np.expm1(X[1] * 4.0)
            nn_d = {k: oof_cal[k] for k in ["p_mask", "p_peak", "vx", "vy", "p_obt"]}
            tubes = extract_nn_labelmap_tubes(nn_d, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, b_clean, include_obturator=True)

            pool_raw = []
            for t in b_clean:
                ras = t["ras"][np.argsort(t["ras"][:, 2])]
                pool_raw.append({
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
                pool_raw.append({
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
                for idx_p, p in enumerate(pool_raw):
                    d = traj_gt_dist(ras, p["ras"])
                    if d < best_d:
                        best_d = d
                        matched_idx = idx_p
                if matched_idx >= 0:
                    pool_raw[matched_idx]["from_mpnn"] = 1.0
                    pool_raw[matched_idx]["mpnn_pa"] = mt["mean_pa"]
                    pool_raw[matched_idx]["mpnn_pl"] = mt["mean_pl"]
                    pool_raw[matched_idx]["mpnn_pe"] = mt["mean_pe"]
                else:
                    pool_raw.append({
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

            pool_unstitched = copy.deepcopy(pool_raw)
            pool_stitched = stitch_collinear_pool_fragments(copy.deepcopy(pool_raw), g, oof_cal, X)
            n_stitched_added_total += (len(pool_stitched) - len(pool_unstitched))

            ub_base, ub_enr = build_bundle_graphs_with_multirater(
                pool_unstitched, g, oof_cal, X,
                gts_all=gts_all_list[i], gts_long=gts_long_list[i], gts_lb_rater2=lb_by_case_idx.get(i),
            )
            sb_base, sb_enr = build_bundle_graphs_with_multirater(
                pool_stitched, g, oof_cal, X,
                gts_all=gts_all_list[i], gts_long=gts_long_list[i], gts_lb_rater2=lb_by_case_idx.get(i),
            )
            unstitched_base_bgs.append(ub_base)
            unstitched_enriched_bgs.append(ub_enr)
            stitched_base_bgs.append(sb_base)
            stitched_enriched_bgs.append(sb_enr)

        pickle.dump({
            "oof_dir": args.oof_dir,
            "stitched_enriched_bgs": stitched_enriched_bgs,
            "mpnn_epochs": mpnn_epochs,
            "graphs": graphs,
            "all_oof_cal": all_oof_cal,
            "stage0_preds": stage0_preds,
        }, open(bundle_cache_path, "wb"))
        print(f"  [Pool Stitching] Added {n_stitched_added_total} stitched full-length trajectories across 54 cases (saved to {bundle_cache_path}).", flush=True)

    # ------------------------------------------------------------------
    # Step 4: Train Nested Stage 2B (With vs Without SynthBrachy3T, Single vs Multi-Rater,
    #         plus Inner-OOF Cross-Fit comparison when inner_oof_f0_case_01.npz exists)
    #         and Nested Stage 2C (`Dedicated1DTipLocalizer`)
    # ------------------------------------------------------------------
    print("[4/5] Training Nested Stage 2B & Stage 2C (1D Axial Tip Localizer)...", flush=True)

    # A. Nested Stage 2B on Unstitched Pool (Single-Rater vs Multi-Rater) for ablation
    base_sc_un_sr, syn_out_un_sr, _ = train_nested_stage2b_models(
        unstitched_base_bgs, unstitched_enriched_bgs, device, use_multirater=False, use_synthbrachy3t=True
    )
    base_sc_un_mr, syn_out_un_mr, _ = train_nested_stage2b_models(
        unstitched_base_bgs, unstitched_enriched_bgs, device, use_multirater=True, use_synthbrachy3t=True
    )

    # B. Nested Stage 2B on Stitched Pool (With SynthBrachy3T vs Without SynthBrachy3T)
    base_sc_st_mr, syn_out_st_mr, stage2b_epochs_synth = train_nested_stage2b_models(
        stitched_base_bgs, stitched_enriched_bgs, device, use_multirater=True, use_synthbrachy3t=True
    )
    base_sc_st_nosyn, syn_out_st_nosyn, stage2b_epochs_nosyn = train_nested_stage2b_models(
        stitched_base_bgs, stitched_enriched_bgs, device, use_multirater=True, use_synthbrachy3t=False
    )

    # C. Nested Stage 2C: Dedicated 1D Axial Tip Localizer (91-step, 22-channel)
    tip_models_5fold, stage2c_epochs = train_nested_stage2c_1d_tip_localizer(
        stitched_enriched_bgs, graphs, all_oof_cal, all_X_vols, device, use_multirater=True
    )

    # D. Optional / Automatic Inner-OOF Cross-Fit Stage 2B evaluation when inner_oof_f0_case_01.npz exists
    has_inner_oof = os.path.exists(os.path.join(args.oof_dir, "inner_oof_f0_case_01.npz")) and graphs is not None
    base_sc_st_inner, syn_out_st_inner, stage2b_epochs_inner = None, None, None
    if has_inner_oof:
        print("  [Inner-OOF Cross-Fit] Building fold-specific inner-OOF bundle features (inner_oof_f{0..4}_case_{i:02d}.npz)...", flush=True)
        inner_base_by_fold, inner_enr_by_fold = build_fold_inner_oof_bundles(
            stitched_base_bgs, stitched_enriched_bgs, args.oof_dir, all_X_vols, meta_list, graphs
        )
        base_sc_st_inner, syn_out_st_inner, stage2b_epochs_inner = train_nested_stage2b_models(
            stitched_base_bgs, stitched_enriched_bgs, device,
            use_multirater=True, use_synthbrachy3t=True,
            inner_oof_base_by_fold=inner_base_by_fold, inner_oof_enr_by_fold=inner_enr_by_fold,
        )
        if args.use_inner_oof_crossfit:
            base_sc_st_mr, syn_out_st_mr = base_sc_st_inner, syn_out_st_inner

    # ------------------------------------------------------------------
    # Step 5: Comprehensive Progression & Ablation Evaluation Across All Cohorts
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating Progression Across All 1,063, Long 759, and Severe Adhesion 233...", flush=True)

    configs = []

    if os.path.exists("/tmp/rerun/cached_enriched_bundles_54.pkl") and os.path.exists("/tmp/rerun/cached_ens_synth_oof_54.pkl") and args.oof_dir.endswith("nn_models_eval"):
        prev_cache = pickle.load(open("/tmp/rerun/cached_enriched_bundles_54.pkl", "rb"))
        prev_ens = pickle.load(open("/tmp/rerun/cached_ens_synth_oof_54.pkl", "rb"))
        p_all_0 = decode_clean_stage2_preds(
            prev_cache["bundle_items"], prev_cache["base_oof_scores"], prev_ens,
            min_conf=50.0, long_mode=False, w_synth=0.60, use_collinear_dedup=False, use_1d_tip_model=False, tip_scale=0.36, calibrate_pos_weight=False
        )
        p_long_0 = decode_clean_stage2_preds(
            prev_cache["bundle_items"], prev_cache["base_oof_scores"], prev_ens,
            min_conf=50.0, long_mode=True, w_synth=0.60, use_collinear_dedup=False, use_1d_tip_model=False, tip_scale=0.36, calibrate_pos_weight=False
        )
        configs.append(("0_previous_stage2_baseline_tau50", "0. Previous Stage 2 Baseline (tau=50%)", p_all_0, p_long_0))

    # 1. + Item 1 (Strictly Nested Stage 2A/2B CV + Case 040 Resolution Calibration, Single-Rater)
    p_all_1 = decode_clean_stage2_preds(
        unstitched_enriched_bgs, base_sc_un_sr, syn_out_un_sr,
        min_conf=50.0, long_mode=False, w_synth=0.60, use_collinear_dedup=False, use_1d_tip_model=False, tip_scale=0.36
    )
    p_long_1 = decode_clean_stage2_preds(
        unstitched_enriched_bgs, base_sc_un_sr, syn_out_un_sr,
        min_conf=50.0, long_mode=True, w_synth=0.60, use_collinear_dedup=False, use_1d_tip_model=False, tip_scale=0.36
    )
    configs.append(("1_nested_cv_and_res_calibration", "1. + Strictly Nested CV + Resolution Calibration (Item 1)", p_all_1, p_long_1))

    # 2. + Item 3 (Multi-Rater LB/AMe Stage 2 Supervision)
    p_all_2 = decode_clean_stage2_preds(
        unstitched_enriched_bgs, base_sc_un_mr, syn_out_un_mr,
        min_conf=50.0, long_mode=False, w_synth=0.60, use_collinear_dedup=False, use_1d_tip_model=False, tip_scale=0.36
    )
    p_long_2 = decode_clean_stage2_preds(
        unstitched_enriched_bgs, base_sc_un_mr, syn_out_un_mr,
        min_conf=50.0, long_mode=True, w_synth=0.60, use_collinear_dedup=False, use_1d_tip_model=False, tip_scale=0.36
    )
    configs.append(("2_plus_multirater_supervision", "2. + Multi-Rater (LB/AMe) Stage 2 Supervision (Item 3)", p_all_2, p_long_2))

    # 3. + Failure Analysis Fixes (Collinear Pool Fragment Stitching + Extrapolation-Aware Collinear Dedup & Extension)
    p_all_3 = decode_clean_stage2_preds(
        stitched_enriched_bgs, base_sc_st_mr, syn_out_st_mr,
        min_conf=50.0, long_mode=False, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=False, tip_scale=0.36
    )
    p_long_3 = decode_clean_stage2_preds(
        stitched_enriched_bgs, base_sc_st_mr, syn_out_st_mr,
        min_conf=50.0, long_mode=True, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=False, tip_scale=0.36
    )
    configs.append(("3_plus_collinear_stitching_and_dedup", "3. + Collinear Split-Fragment Stitching & Extrapolation Dedup", p_all_3, p_long_3))

    # 4. + Item 4 (Dedicated 91-Step 22-Channel 1D Axial Tip Localizer, WITHOUT SynthBrachy3T)
    p_all_4_nosyn = decode_clean_stage2_preds(
        stitched_enriched_bgs, base_sc_st_nosyn, syn_out_st_nosyn,
        graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
        min_conf=50.0, long_mode=False, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
    )
    p_long_4_nosyn = decode_clean_stage2_preds(
        stitched_enriched_bgs, base_sc_st_nosyn, syn_out_st_nosyn,
        graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
        min_conf=50.0, long_mode=True, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
    )
    configs.append(("4a_full_clean_without_synthbrachy3t_tau50", "4a. Full Clean Pipeline WITHOUT SynthBrachy3T (tau=50%)", p_all_4_nosyn, p_long_4_nosyn))

    # 5. + Item 4 (Dedicated 91-Step 22-Channel 1D Axial Tip Localizer, WITH SynthBrachy3T) across tau in {35, 45, 50, 55, 60}
    full_preds_by_tau = {}
    for tau in [35.0, 45.0, 50.0, 55.0, 60.0]:
        pa = decode_clean_stage2_preds(
            stitched_enriched_bgs, base_sc_st_mr, syn_out_st_mr,
            graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
            min_conf=tau, long_mode=False, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
        )
        pl = decode_clean_stage2_preds(
            stitched_enriched_bgs, base_sc_st_mr, syn_out_st_mr,
            graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
            min_conf=tau, long_mode=True, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
        )
        full_preds_by_tau[int(tau)] = (pa, pl)
        if int(tau) in (35, 50):
            configs.append((
                f"4b_full_clean_with_synthbrachy3t_tau{int(tau)}",
                f"4b. Full Clean Pipeline WITH SynthBrachy3T (tau={int(tau)}%)",
                pa, pl
            ))

    if has_inner_oof and base_sc_st_inner is not None:
        pa_in = decode_clean_stage2_preds(
            stitched_enriched_bgs, base_sc_st_inner, syn_out_st_inner,
            graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
            min_conf=50.0, long_mode=False, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
        )
        pl_in = decode_clean_stage2_preds(
            stitched_enriched_bgs, base_sc_st_inner, syn_out_st_inner,
            graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
            min_conf=50.0, long_mode=True, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
        )
        configs.append((
            "4c_full_clean_with_synthbrachy3t_inner_oof_crossfit_tau50",
            "4c. Full Clean Pipeline WITH SynthBrachy3T + Inner-OOF Cross-Fit (tau=50%)",
            pa_in, pl_in
        ))

    results_out = {}
    per_patient_map = {}

    print("\n=========================================================================================================")
    print("PROGRESSION SUMMARY ACROSS ALL 1,063, LONG 759, AND ADHESION 233 CATHETERS")
    print("=========================================================================================================", flush=True)

    for key, label, p_all, p_long in configs:
        res_all, pp_all = evaluate_method_unified(p_all, gts_all_list, thresh=3.5, n_boot=200)
        res_long, pp_long = evaluate_method_unified(p_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=200)
        res_adh, pp_adh = evaluate_method_unified([p_all[i] for i in adh_idx], [gts_all_list[i] for i in adh_idx], thresh=3.5, n_boot=200)
        diag_extra = compute_detailed_tip_and_fp_diagnostics(p_all, gts_all_list, lb_by_case_idx)

        # Also check Case 040 (idx=24) specifically
        res_c24, _ = evaluate_method_unified([p_all[24]], [gts_all_list[24]], thresh=3.5, n_boot=10)

        results_out[key] = {
            "label": label,
            "all_1063": res_all,
            "long_759": res_long,
            "adhesion_233": res_adh,
            "diagnostics_all_1063": diag_extra,
            "case_040_idx24": {
                "n_gt": res_c24["n_gt"],
                "n_pred": res_c24["n_pred"],
                "matched_1to1": res_c24["hungarian_1to1_matched"],
                "recall_1to1_pct": res_c24["hungarian_1to1_recall_pct"],
                "precision_1to1_pct": res_c24["precision_1to1_pct"],
                "f1_1to1_pct": res_c24["f1_1to1_pct"],
            },
        }
        per_patient_map[key] = pp_all

        sh = res_all["metrics"]["mean_shaft_xy"]
        tp = res_all["metrics"]["tip_3d"]
        print(
            f"\n[{label}]\n"
            f"  ALL (1,063) : Rec={res_all['hungarian_1to1_recall_pct']:5.2f}% ({res_all['hungarian_1to1_matched']}/1063) | "
            f"Prec={res_all['precision_1to1_pct']:5.2f}% | F1={res_all['f1_1to1_pct']:5.2f}% (n_pred={res_all['n_pred']})\n"
            f"  LONG (759)  : Rec={res_long['hungarian_1to1_recall_pct']:5.2f}% ({res_long['hungarian_1to1_matched']}/759) | "
            f"Prec_adj={res_long['precision_1to1_obturator_adj_pct']:5.2f}% | F1_adj={res_long['f1_1to1_obturator_adj_pct']:5.2f}% (Raw F1={res_long['f1_1to1_pct']:5.2f}%)\n"
            f"  ADH (233)   : Rec={res_adh['hungarian_1to1_recall_pct']:5.2f}% | Prec={res_adh['precision_1to1_pct']:5.2f}% | F1={res_adh['f1_1to1_pct']:5.2f}%\n"
            f"  Shaft Error : mean={sh['mean']:.3f} mm | median={sh['median']:.3f} mm | IQR={sh['iqr']:.3f} mm | P95={sh['p95']:.3f} mm\n"
            f"  Tip Error   : mean={tp['mean']:.2f} mm | median={tp['median']:.2f} mm | IQR={tp['iqr']:.2f} mm | P95={tp['p95']:.2f} mm | "
            f"<=3mm={diag_extra['pct_tip_le_3mm']:.1f}% | >10mm={diag_extra['pct_tip_gt_10mm']:.2f}% ({diag_extra['n_tip_gt_10mm']}/{diag_extra['n_matched']}; "
            f"undershoot={diag_extra['n_caudal_undershoot_gt_10mm']}, overshoot={diag_extra['n_cranial_overshoot_gt_10mm']})\n"
            f"  FPs & Case40: Total FPs={diag_extra['n_total_fp']} (split-frag FPs={diag_extra['n_split_fragment_fp']}) | "
            f"Case 040 (idx=24) Rec={res_c24['hungarian_1to1_recall_pct']:.1f}% ({res_c24['hungarian_1to1_matched']}/21), F1={res_c24['f1_1to1_pct']:.1f}%",
            flush=True,
        )

    wilcoxon_tests = {
        "full_with_synth_vs_without_synth_tau50": compute_wilcoxon_comparison(
            per_patient_map["4b_full_clean_with_synthbrachy3t_tau50"],
            per_patient_map["4a_full_clean_without_synthbrachy3t_tau50"],
        ),
    }
    if "4c_full_clean_with_synthbrachy3t_inner_oof_crossfit_tau50" in per_patient_map:
        wilcoxon_tests["full_with_synth_oof_vs_inner_oof_crossfit_tau50"] = compute_wilcoxon_comparison(
            per_patient_map["4b_full_clean_with_synthbrachy3t_tau50"],
            per_patient_map["4c_full_clean_with_synthbrachy3t_inner_oof_crossfit_tau50"],
        )
    if "0_previous_stage2_baseline_tau50" in per_patient_map:
        wilcoxon_tests["full_with_synth_vs_previous_baseline_tau50"] = compute_wilcoxon_comparison(
            per_patient_map["4b_full_clean_with_synthbrachy3t_tau50"],
            per_patient_map["0_previous_stage2_baseline_tau50"],
        )

    final_report = {
        "generated_at": "2026-10-23",
        "script": "vault/paper/benchmarks_clean/run_stage2_and_tip_clean.py",
        "cache_dir": args.cache_dir,
        "oof_dir": args.oof_dir,
        "elapsed_sec": round(time.time() - t_start, 2),
        "nested_cv_protocol": {
            "outer_folds": 5,
            "inner_subtrain_cases_per_fold": 35,
            "inner_val_cases_per_fold": "8-9",
            "outer_test_cases_per_fold": "10-11",
            "stage2a_mpnn_inner_val_epochs": mpnn_epochs,
            "stage2b_synth_inner_val_epochs": stage2b_epochs_synth,
            "stage2b_nosynth_inner_val_epochs": stage2b_epochs_nosyn,
            "stage2b_inner_oof_crossfit_epochs": stage2b_epochs_inner,
            "stage2c_1d_tip_inner_val_epochs": stage2c_epochs,
            "n_stitched_candidates_added_54cases": n_stitched_added_total,
        },
        "progression_results": results_out,
        "wilcoxon_signed_rank_tests_54patients": wilcoxon_tests,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(final_report, f, indent=2)

    os.makedirs(os.path.dirname(args.out_preds_pkl), exist_ok=True)
    pickle.dump({
        "all_tau50": full_preds_by_tau[50][0],
        "long_tau50": full_preds_by_tau[50][1],
        "all_tau35": full_preds_by_tau[35][0],
        "long_tau35": full_preds_by_tau[35][1],
        "full_preds_by_tau": full_preds_by_tau,
        "nosyn_all_tau50": p_all_4_nosyn,
        "nosyn_long_tau50": p_long_4_nosyn,
        "stitched_enriched_bgs": stitched_enriched_bgs,
        "base_sc_st_mr": base_sc_st_mr,
        "syn_out_st_mr": syn_out_st_mr,
        "base_sc_st_nosyn": base_sc_st_nosyn,
        "syn_out_st_nosyn": syn_out_st_nosyn,
        "base_sc_st_inner": base_sc_st_inner,
        "syn_out_st_inner": syn_out_st_inner,
        "tip_models_5fold_states": [{k: v.cpu() for k, v in m.state_dict().items()} for m in tip_models_5fold],
    }, open(args.out_preds_pkl, "wb"))

    print(f"\n[Done] Saved JSON report to {args.out_json} and predictions to {args.out_preds_pkl} in {time.time() - t_start:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
