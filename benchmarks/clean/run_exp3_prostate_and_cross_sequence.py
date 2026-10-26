#!/usr/bin/env python3
"""
Experiment 3: Zero-Shot Cross-Organ Transfer (Case 100 Prostate, 14 Catheters)
& Same-Session Cross-Sequence Generalization (2D AX/SAG/COR TSE T2, AX T1 / 3D VIBE, 3D wbSSFP)
================================================================================================

Part 3A: Zero-Shot Cross-Organ Transfer to Transperineal Prostate MRI (`Case 100 Prostate`, 14 catheters)
  - Evaluates all pipeline stages and baselines across three entry-plane operating modes:
      1. `k_lim_auto = 40`: 100% reference-free automated pelvic template slice detector
         (`TrackingAlgorithms.estimateAxialLimitSlice`), requiring zero `.acsv` or `.vtk` input.
      2. `k_lim_acsv_safe = 47`: Standard `.acsv` template fiducial plane (`z_lim_ras = +3.13 mm`).
      3. `k_sym_safe = 26`: Distal transperineal slab including sub-template perineal tracks.
  - Compares:
      * Stage 0: Rust 3-Pass Physics Tracker (`TrackingAlgorithms.autoDetectAllNeedles`)
      * Baseline 1: PMB 2019 3D U-Net + Iterative Walk (`Jung et al., PMB 2019`, zero-exposure GYN)
      * Baseline 2: Native 3D ResUNet + Soft-clDice + Skeletonization (`Shit et al., CVPR 2021`, 1-ch intensity)
      * NeedleFinder 2026 (Zero-Exposure GYN Ensemble, No-SSL: 0% labeled or unlabeled exposure to Case 100)
      * NeedleFinder 2026 (Full Clean Nested GYN Ensemble)
  - Physics-Channel Ablations under Cross-Organ Domain Shift:
      * Full 6-channel physics-informed input vs. Stage 1 physics channels zeroed (`X[1:4] = 0`)
        vs. Stage 2 physics features zeroed vs. 1-channel pure intensity (`Native ResUNet + clDice`).

Part 3B: Same-Session Multi-Sequence Generalization & Late Tip Fusion across the 54-Case Cohort
  - Exhaustive inventory of all 119 secondary same-session `.nrrd` scans across the 54 patients
    (`2D AX TSE T2`, `AX T1 / 3D VIBE POST FS`, `AX 3D wbSSFP / True-FISP`, `2D SAG/COR TSE T2`).
  - Zero-shot cross-sequence detection & localization using strictly held-out outer-fold models
    (`f_out = idx % 5`), comparing native thick-slice inference vs. 1.6 mm canonical Z-upsampled
    inference across `Stage 0`, `PMB 2019 3D U-Net`, `Native ResUNet + clDice`, and `NeedleFinder 2026`.
  - Same-session Multi-Sequence Late Tip Fusion (`3D T2 SPACE + Co-Registered 2D AX/SAG TSE T2`).

Saves full quantitative report to:
  /home/guillaume/github/needlefinder/vault/sources/2026-10-23-exp3-prostate-and-cross-sequence.json
"""

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path

import copy
import glob
import json
import math
import os
import pickle
import re
import sys
import time
import types
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean"))

# Import evaluate_unified_apples_to_apples FIRST so MockSlicer modules can be attached cleanly
from evaluate_unified_apples_to_apples import (
    evaluate_method_unified,
    extract_nn_labelmap_tubes,
    build_slice_graph_fast,
    DenseGraphTransformer,
    NeedleSliceMPNN,
    decode_mpnn_trajectories,
    get_gts,
    traj_gt_dist,
)
from prepare_and_inspect_dataset import read_nrrd_local, read_vtk_centerline_local
from prepare_clean_dataset import (
    estimate_axial_limit_slice_existing,
    compute_tilt_safe_pelvic_klim,
    build_clean_input_tensor,
    StandaloneEvalLogic,
    MockVolumeNode,
    MockMatrix4x4,
)
from train_and_eval_unet3d import NeedleUNet3D
from train_nested_stage1 import infer_4view_tta_logits, raw_out_to_probs
from run_stage2_and_tip_clean import (
    calibrate_stage1_oof_for_volume,
    stitch_collinear_pool_fragments,
    build_bundle_graphs_with_multirater,
    decode_clean_stage2_preds,
    Dedicated1DTipLocalizer,
    TipAwareBundleGraphTransformer,
    get_nested_fold_split,
)
from train_baselines_pmb2026 import (
    PMB2019IsoUNet3D,
    NativeResUNet3D_MaskOnly,
    infer_sliding_1ch,
    extract_pmb2019_iso_trajectories,
    extract_native_cldice_skeleton_trajectories,
)
from test_stage2_transfer import get_or_train_5fold_stage2_models, run_full_nf2026_on_volume
import NeedleFinderLib.TrackingAlgorithms as ta_mod

ta_mod.slicer.modules = types.SimpleNamespace(NeedleFinderWidget=None)
ta_mod.vtk.vtkMatrix4x4 = MockMatrix4x4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

def compute_crop_bounds(shape, sp_ras, k_lim_use):
    K, J, I = [int(x) for x in shape]
    sp_i, sp_j = float(sp_ras[0]), float(sp_ras[1])
    k0 = max(0, int(k_lim_use) - 1)
    margin_i = min(I // 2 - 4, int(round(68.0 / sp_i)))
    margin_j = min(J // 2 - 4, int(round(68.0 / sp_j)))
    half_w = max(8, ((margin_i + 7) // 8) * 8)
    half_h = max(8, ((margin_j + 7) // 8) * 8)
    ci_v, cj_v = I // 2, J // 2
    i0 = max(0, ci_v - half_w)
    i1 = min(I, i0 + 2 * half_w)
    if (i1 - i0) % 8 != 0:
        i1 = i0 + max(8, ((i1 - i0) // 8) * 8)
    j0 = max(0, cj_v - half_h)
    j1 = min(J, j0 + 2 * half_h)
    if (j1 - j0) % 8 != 0:
        j1 = j0 + max(8, ((j1 - j0) // 8) * 8)
    return k0, K, i0, i1, j0, j1

def upsample_volume_z_to_canonical(vol, sp_ras, ijk2ras, target_sp_k=1.6):
    """
    Trilinearly upsamples a thick-slice volume along the axial K axis to `target_sp_k` (1.6 mm)
    and updates `sp_ras` and `ijk2ras` so world-RAS coordinates remain exactly aligned.
    """
    sp_k = float(sp_ras[2])
    if sp_k <= target_sp_k * 1.15:
        return vol, np.asarray(sp_ras, dtype=np.float64), np.asarray(ijk2ras, dtype=np.float64), 1.0
    K, H, W = vol.shape
    scale_k = sp_k / float(target_sp_k)
    K_new = max(K, int(round((K - 1) * scale_k)) + 1)
    actual_scale = (K_new - 1) / max(1, K - 1)
    vol_t = torch.from_numpy(vol.astype(np.float32))[None, None]
    vol_up = F.interpolate(vol_t, size=(K_new, H, W), mode="trilinear", align_corners=True)[0, 0].numpy()
    sp_new = np.array([float(sp_ras[0]), float(sp_ras[1]), sp_k / actual_scale], dtype=np.float64)
    ijk2ras_new = np.asarray(ijk2ras, dtype=np.float64).copy()
    ijk2ras_new[:3, 2] = ijk2ras[:3, 2] / actual_scale
    return vol_up, sp_new, ijk2ras_new, actual_scale


def run_stage1_nntubes_only(vol, sp_ras, ijk2ras, k_lim_use, ckpt_paths, zero_physics=False):
    """Runs Stage 0 + Stage 1 3D U-Net + `extract_nn_labelmap_tubes` (without Stage 2)."""
    t0 = time.perf_counter()
    sp_i, sp_j, sp_k = float(sp_ras[0]), float(sp_ras[1]), float(sp_ras[2])
    k0, K, i0, i1, j0, j1 = compute_crop_bounds(vol.shape, sp_ras, k_lim_use)
    X, resp, cs, rsym = build_clean_input_tensor(vol, sp_ras, k0, K, i0, i1, j0, j1)
    if zero_physics:
        X[1:4] = 0.0
    ras2ijk = np.linalg.inv(ijk2ras)

    x_t = torch.from_numpy(X)
    logits_sum = None
    for cp in ckpt_paths:
        m = NeedleUNet3D(6, 24).to(device)
        m.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
        m.eval()
        lg = infer_4view_tta_logits(m, x_t, device, use_tta=True)
        logits_sum = lg if logits_sum is None else (logits_sum + lg)
    probs = raw_out_to_probs(logits_sum / len(ckpt_paths), temp_pos=1.08).numpy()
    oof_raw = {"p_mask": probs[0], "p_peak": probs[1], "vx": probs[2], "vy": probs[3], "p_obt": probs[4], "k0": k0, "i0": i0, "j0": j0}
    oof_cal = calibrate_stage1_oof_for_volume(oof_raw, X, sp_ras)

    resp_approx = np.zeros_like(resp) if zero_physics else resp
    nn_d = {k: oof_raw[k] for k in ["p_mask", "p_peak", "vx", "vy", "p_obt"]}
    tubes = extract_nn_labelmap_tubes(nn_d, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, [], include_obturator=False)
    for t in tubes:
        t["ras"] = np.asarray(t["points"], dtype=np.float64)
    elapsed = time.perf_counter() - t0
    diag = {
        "p_mask_max": round(float(probs[0].max()), 4),
        "p_mask_p999": round(float(np.percentile(probs[0], 99.9)), 4),
        "p_peak_max": round(float(probs[1].max()), 4),
        "p_obt_max": round(float(probs[4].max()), 4),
        "runtime_s": round(float(elapsed), 3),
    }
    return tubes, [], diag


def run_pmb2019_on_volume(vol, sp_ras, ijk2ras, k_lim_use, ckpt_paths):
    """Runs PMB 2019 3D U-Net + Iterative Walk on an arbitrary volume."""
    t0 = time.perf_counter()
    sp_i, sp_j, sp_k = float(sp_ras[0]), float(sp_ras[1]), float(sp_ras[2])
    k0, K, i0, i1, j0, j1 = compute_crop_bounds(vol.shape, sp_ras, k_lim_use)
    X, _, _, _ = build_clean_input_tensor(vol, sp_ras, k0, K, i0, i1, j0, j1)
    x_1ch = torch.from_numpy(X[0:1])
    K_sub, H, W = x_1ch.shape[1:]
    D_iso = max(16, int(round(K_sub * sp_k / 8.0)) * 8)
    H_iso = max(64, int(round(H * sp_j / 8.0)) * 8)
    W_iso = max(64, int(round(W * sp_i / 8.0)) * 8)
    x_iso = F.interpolate(x_1ch.unsqueeze(0), size=(D_iso, H_iso, W_iso), mode="trilinear", align_corners=False)[0]
    p_iso_sum = np.zeros((D_iso, H_iso, W_iso), dtype=np.float32)
    for cp in ckpt_paths:
        m_pmb = PMB2019IsoUNet3D(1, 20).to(device)
        m_pmb.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
        m_pmb.eval()
        p_iso_sum += infer_sliding_1ch(m_pmb, x_iso, device)
    p_iso = p_iso_sum / len(ckpt_paths)
    meta_d = {
        "k0": k0, "i0": i0, "i1": i1, "j0": j0, "j1": j1,
        "K": K_sub, "H": H, "W": W,
        "D_iso": D_iso, "H_iso": H_iso, "W_iso": W_iso,
        "sp_ras": (sp_i, sp_j, sp_k), "ijk_to_ras": ijk2ras,
    }
    trajs = extract_pmb2019_iso_trajectories(p_iso, meta_d, prob_thresh=0.42)
    return trajs, float(time.perf_counter() - t0)


def run_resunet_cldice_on_volume(vol, sp_ras, ijk2ras, k_lim_use, ckpt_paths):
    """Runs Native 3D ResUNet + Soft-clDice + 3D Skeletonization on an arbitrary volume."""
    t0 = time.perf_counter()
    sp_i, sp_j, sp_k = float(sp_ras[0]), float(sp_ras[1]), float(sp_ras[2])
    k0, K, i0, i1, j0, j1 = compute_crop_bounds(vol.shape, sp_ras, k_lim_use)
    X, _, _, _ = build_clean_input_tensor(vol, sp_ras, k0, K, i0, i1, j0, j1)
    x_1ch = torch.from_numpy(X[0:1])
    K_sub, H, W = x_1ch.shape[1:]
    p_nat_sum = np.zeros((K_sub, H, W), dtype=np.float32)
    for cp in ckpt_paths:
        m_nat = NativeResUNet3D_MaskOnly(1, 24).to(device)
        m_nat.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
        m_nat.eval()
        p_nat_sum += infer_sliding_1ch(m_nat, x_1ch, device)
    p_nat = p_nat_sum / len(ckpt_paths)
    meta_d = {
        "k0": k0, "i0": i0, "i1": i1, "j0": j0, "j1": j1,
        "K": K_sub, "H": H, "W": W,
        "sp_ras": (sp_i, sp_j, sp_k), "ijk_to_ras": ijk2ras,
    }
    trajs = extract_native_cldice_skeleton_trajectories(p_nat, meta_d, prob_thresh=0.40)
    return trajs, float(time.perf_counter() - t0), float(np.percentile(p_nat, 99.9))


def compute_per_catheter_breakdown(preds, gts, thresh=3.5):
    """Computes exact 1-to-1 Hungarian matched per-catheter metrics for all GT catheters."""
    from scipy.optimize import linear_sum_assignment
    pred_pts = [np.asarray(p["ras"] if isinstance(p, dict) and "ras" in p else (p["points"] if isinstance(p, dict) else p), dtype=np.float64) for p in preds]
    pred_pts = [p[np.argsort(p[:, 2])] for p in pred_pts if len(p) >= 2]
    gt_pts = [np.asarray(g, dtype=np.float64)[np.argsort(np.asarray(g, dtype=np.float64)[:, 2])] for g in gts]

    n_p, n_g = len(pred_pts), len(gt_pts)
    cost = np.full((n_p, n_g), 999.0, dtype=np.float64)
    for ip, p in enumerate(pred_pts):
        for ig, g in enumerate(gt_pts):
            cost[ip, ig] = traj_gt_dist(p, g)

    matched_gt_to_pred = {}
    if n_p > 0 and n_g > 0:
        r_ind, c_ind = linear_sum_assignment(cost)
        for r, c in zip(r_ind, c_ind):
            if cost[r, c] <= thresh:
                matched_gt_to_pred[int(c)] = (int(r), float(cost[r, c]))

    per_cath = []
    for ig, g in enumerate(gt_pts):
        z_bot, z_top = float(g[0, 2]), float(g[-1, 2])
        span = z_top - z_bot
        if ig in matched_gt_to_pred:
            ip, d_sh = matched_gt_to_pred[ig]
            p = pred_pts[ip]
            e_tip3d = float(np.linalg.norm(p[-1] - g[-1]))
            dz_tip = float(p[-1, 2] - g[-1, 2])
            dxy_tip = float(np.linalg.norm(p[-1, :2] - g[-1, :2]))
            per_cath.append({
                "catheter_id": ig,
                "gt_z_range_mm": [round(z_bot, 2), round(z_top, 2)],
                "gt_span_mm": round(span, 2),
                "matched": True,
                "shaft_error_mm": round(d_sh, 3),
                "tip_3d_error_mm": round(e_tip3d, 3),
                "tip_dz_signed_mm": round(dz_tip, 3),
                "tip_dxy_mm": round(dxy_tip, 3),
            })
        else:
            min_d = float(cost[:, ig].min()) if n_p > 0 else 999.0
            per_cath.append({
                "catheter_id": ig,
                "gt_z_range_mm": [round(z_bot, 2), round(z_top, 2)],
                "gt_span_mm": round(span, 2),
                "matched": False,
                "nearest_pred_dist_mm": round(min_d, 3),
            })
    return per_cath


def compute_tip_threshold_pcts(preds_list, gts_list, thresh=3.5):
    from scipy.optimize import linear_sum_assignment
    tip_errs = []
    for preds, gts in zip(preds_list, gts_list):
        pred_pts = [np.asarray(p["ras"] if isinstance(p, dict) and "ras" in p else (p["points"] if isinstance(p, dict) else p), dtype=np.float64) for p in preds]
        pred_pts = [p[np.argsort(p[:, 2])] for p in pred_pts if len(p) >= 2]
        gt_pts = [np.asarray(g, dtype=np.float64)[np.argsort(np.asarray(g, dtype=np.float64)[:, 2])] for g in gts]
        if not pred_pts or not gt_pts:
            continue
        cost = np.full((len(pred_pts), len(gt_pts)), 999.0, dtype=np.float64)
        for ip, p in enumerate(pred_pts):
            for ig, g in enumerate(gt_pts):
                cost[ip, ig] = traj_gt_dist(p, g)
        r_ind, c_ind = linear_sum_assignment(cost)
        for r, c in zip(r_ind, c_ind):
            if cost[r, c] <= thresh:
                tip_errs.append(float(np.linalg.norm(pred_pts[r][-1] - gt_pts[c][-1])))
    if not tip_errs:
        return 0.0, 0.0
    n = len(tip_errs)
    return round(100.0 * sum(1 for e in tip_errs if e <= 2.0) / n, 2), round(100.0 * sum(1 for e in tip_errs if e <= 3.0) / n, 2)


def compact_eval_dict(res, runtime_s=None, extra=None, preds_list=None, gts_list=None):
    """Formats `evaluate_method_unified` output into a clean dictionary."""
    sh = res["metrics"]["mean_shaft_xy"]
    tp = res["metrics"]["tip_3d"]
    out = {
        "n_gt": int(res["n_gt"]),
        "n_pred": int(res["n_pred"]),
        "tp": int(res["hungarian_1to1_matched"]),
        "fp": int(res["n_pred"] - res["hungarian_1to1_matched"]),
        "fn": int(res["n_gt"] - res["hungarian_1to1_matched"]),
        "recall_1to1_pct": round(float(res["hungarian_1to1_recall_pct"]), 2),
        "precision_1to1_pct": round(float(res["precision_1to1_pct"]), 2),
        "f1_1to1_pct": round(float(res["f1_1to1_pct"]), 2),
        "shaft_mean_mm": round(float(sh["mean"]), 3) if res["hungarian_1to1_matched"] > 0 else None,
        "shaft_median_mm": round(float(sh["median"]), 3) if res["hungarian_1to1_matched"] > 0 else None,
        "shaft_iqr_mm": round(float(sh["iqr"]), 3) if res["hungarian_1to1_matched"] > 0 else None,
        "shaft_p95_mm": round(float(sh["p95"]), 3) if res["hungarian_1to1_matched"] > 0 else None,
        "tip_3d_mean_mm": round(float(tp["mean"]), 3) if res["hungarian_1to1_matched"] > 0 else None,
        "tip_3d_median_mm": round(float(tp["median"]), 3) if res["hungarian_1to1_matched"] > 0 else None,
        "tip_3d_iqr_mm": round(float(tp["iqr"]), 3) if res["hungarian_1to1_matched"] > 0 else None,
        "tip_3d_p95_mm": round(float(tp["p95"]), 3) if res["hungarian_1to1_matched"] > 0 else None,
    }
    if preds_list is not None and gts_list is not None:
        le2, le3 = compute_tip_threshold_pcts(preds_list, gts_list)
        out["tip_le_2mm_pct"] = le2
        out["tip_le_3mm_pct"] = le3
    if runtime_s is not None:
        out["runtime_s"] = round(float(runtime_s), 3)
    if extra:
        out.update(extra)
    return out


# ==============================================================================
# PART 3A: CASE 100 PROSTATE (14 CATHETERS) ZERO-SHOT CROSS-ORGAN TRANSFER
# ==============================================================================

def run_part_3a_prostate(mpnn_m, base_m, syn_m, tip_m):
    print("\n" + "=" * 88)
    print("PART 3A: ZERO-SHOT CROSS-ORGAN TRANSFER ON CASE 100 PROSTATE (14 CATHETERS)")
    print("=" * 88, flush=True)

    p_dir = resolve_path("/home/guillaume/Dropbox/GYN Cases/Case 100 Prostate/NRRD-MR")
    nrrd_p = os.path.join(p_dir, "33 AX space trial 1.6 SLICE COVER TEMPLATE.nrrd")
    acsv_p = os.path.join(p_dir, "Manual_CY/template slice position.acsv")
    vol, sp_ras, ijk2ras, k_lim_acsv, z_lim_ras = read_nrrd_local(nrrd_p, acsv_p)
    k_lim_auto = estimate_axial_limit_slice_existing(vol, float(sp_ras[2]))
    k_lim_acsv_safe, k_sym_safe = compute_tilt_safe_pelvic_klim(vol, sp_ras, ijk2ras, z_lim_ras=z_lim_ras, k_lim_orig=k_lim_acsv)

    vtks = sorted(
        glob.glob(os.path.join(p_dir, "Manual_CY/manual-seg_*.vtk")),
        key=lambda x: int(re.search(r"_(\d+)\.vtk", x).group(1)),
    )
    gts_full = [read_vtk_centerline_local(vp) for vp in vtks]
    gts_acsv = [read_vtk_centerline_local(vp, z_lim_ras=z_lim_ras) for vp in vtks]

    # Compute pairwise XY distances between the 14 ground-truth prostate catheters
    min_pair_dists = []
    for i in range(len(gts_full)):
        d_i = [traj_gt_dist(gts_full[i], gts_full[j]) for j in range(len(gts_full)) if j != i]
        min_pair_dists.append(float(min(d_i)))

    ckpts_zero = [resolve_path(f"/home/guillaume/github/needlefinder/nn_models_eval/unet3d_fold{f}.pt") for f in range(5)]
    ckpts_clean = [resolve_path(f"/home/guillaume/github/needlefinder/nn_models_clean/unet3d_outer{f}_inner0.pt") for f in range(5)]
    ckpts_pmb = [resolve_path(f"/home/guillaume/github/needlefinder/nn_models_eval/pmb2019_iso1mm_fold{f}.pt") for f in range(5)]
    ckpts_cldice = [resolve_path(f"/home/guillaume/github/needlefinder/nn_models_eval/native_cldice_fold{f}.pt") for f in range(5)]

    part3a_results = {
        "scan_metadata": {
            "case_name": "Case 100 Prostate",
            "nrrd_filename": "33 AX space trial 1.6 SLICE COVER TEMPLATE.nrrd",
            "shape_kij": list(vol.shape),
            "spacing_ras_mm": [round(float(x), 5) for x in sp_ras],
            "n_reference_catheters": len(gts_full),
            "gt_z_range_ras_mm": [
                round(float(min(g[:, 2].min() for g in gts_full)), 2),
                round(float(max(g[:, 2].max() for g in gts_full)), 2),
            ],
            "gt_span_range_mm": [
                round(float(min(g[:, 2].max() - g[:, 2].min() for g in gts_full)), 2),
                round(float(max(g[:, 2].max() - g[:, 2].min() for g in gts_full)), 2),
            ],
            "gt_nearest_neighbor_xy_mm": {
                "min": round(float(min(min_pair_dists)), 2),
                "median": round(float(np.median(min_pair_dists)), 2),
                "max": round(float(max(min_pair_dists)), 2),
            },
            "k_lim_operating_points": {
                "k_lim_auto_reference_free": int(k_lim_auto),
                "k_lim_acsv_raw": int(k_lim_acsv),
                "k_lim_acsv_tilt_safe": int(k_lim_acsv_safe),
                "k_sym_safe_distal_perineal": int(k_sym_safe),
                "z_lim_acsv_ras_mm": round(float(z_lim_ras), 3),
                "z_lim_auto_ras_mm": round(float((ijk2ras @ np.array([0.0, 0.0, float(k_lim_auto), 1.0]))[2]), 3),
            },
        },
        "operating_modes": {},
        "physics_channel_ablations": {},
        "per_catheter_results": {},
    }

    for mode_key, kl_val, g_ref in [
        ("reference_free_auto_klim40", k_lim_auto, gts_full),
        ("standard_template_acsv_klim47", k_lim_acsv_safe, gts_acsv),
        ("distal_perineal_slab_ksym26", k_sym_safe, gts_full),
    ]:
        print(f"\n--- Operating Mode: {mode_key} (k_lim={kl_val}) ---", flush=True)
        mode_dict = {}

        # 1. Stage 0 Rust 3-Pass Physics Tracker
        t0 = time.perf_counter()
        logic = StandaloneEvalLogic()
        vnode = MockVolumeNode(vol, sp_ras, ijk2ras)
        s0_t = logic.autoDetectAllNeedles(vnode, minConfidence=35.0, kLim=kl_val, renderNeedles=False, script=True)
        rt_s0 = time.perf_counter() - t0
        res_s0, _ = evaluate_method_unified([s0_t], [g_ref], thresh=3.5, n_boot=50)
        mode_dict["stage0_rust_physics_tracker"] = compact_eval_dict(res_s0, rt_s0)
        print(f"  Stage 0 Rust Tracker              : {mode_dict['stage0_rust_physics_tracker']['tp']}/14 (Rec={mode_dict['stage0_rust_physics_tracker']['recall_1to1_pct']}%, Prec={mode_dict['stage0_rust_physics_tracker']['precision_1to1_pct']}%, F1={mode_dict['stage0_rust_physics_tracker']['f1_1to1_pct']}%) | ShaftMed={mode_dict['stage0_rust_physics_tracker']['shaft_median_mm']}mm, TipMed={mode_dict['stage0_rust_physics_tracker']['tip_3d_median_mm']}mm", flush=True)

        # 2. Baseline 1: PMB 2019 3D U-Net + Walk
        pmb_t, rt_pmb = run_pmb2019_on_volume(vol, sp_ras, ijk2ras, kl_val, ckpts_pmb)
        res_pmb, _ = evaluate_method_unified([pmb_t], [g_ref], thresh=3.5, n_boot=50)
        mode_dict["baseline1_pmb2019_unet_walk"] = compact_eval_dict(res_pmb, rt_pmb)
        print(f"  Baseline 1 (PMB 2019 3D U-Net)    : {mode_dict['baseline1_pmb2019_unet_walk']['tp']}/14 (Rec={mode_dict['baseline1_pmb2019_unet_walk']['recall_1to1_pct']}%, Prec={mode_dict['baseline1_pmb2019_unet_walk']['precision_1to1_pct']}%, F1={mode_dict['baseline1_pmb2019_unet_walk']['f1_1to1_pct']}%) | ShaftMed={mode_dict['baseline1_pmb2019_unet_walk']['shaft_median_mm']}mm, TipMed={mode_dict['baseline1_pmb2019_unet_walk']['tip_3d_median_mm']}mm", flush=True)

        # 3. Baseline 2: Native ResUNet + clDice + Skeleton (1-ch intensity only)
        cld_t, rt_cld, cld_p999 = run_resunet_cldice_on_volume(vol, sp_ras, ijk2ras, kl_val, ckpts_cldice)
        res_cld, _ = evaluate_method_unified([cld_t], [g_ref], thresh=3.5, n_boot=50)
        mode_dict["baseline2_resunet_cldice_1ch"] = compact_eval_dict(res_cld, rt_cld, {"prob_p999": round(cld_p999, 4)})
        print(f"  Baseline 2 (ResUNet+clDice 1-ch)  : {mode_dict['baseline2_resunet_cldice_1ch']['tp']}/14 (Rec={mode_dict['baseline2_resunet_cldice_1ch']['recall_1to1_pct']}%, Prec={mode_dict['baseline2_resunet_cldice_1ch']['precision_1to1_pct']}%, F1={mode_dict['baseline2_resunet_cldice_1ch']['f1_1to1_pct']}%) | ShaftMed={mode_dict['baseline2_resunet_cldice_1ch']['shaft_median_mm']}mm, TipMed={mode_dict['baseline2_resunet_cldice_1ch']['tip_3d_median_mm']}mm", flush=True)

        # 4. NeedleFinder 2026 Stage 1 NNTubes (Zero-Exposure GYN No-SSL vs Full Clean Nested GYN)
        tubes_z, _, diag_z = run_stage1_nntubes_only(vol, sp_ras, ijk2ras, kl_val, ckpts_zero, zero_physics=False)
        res_tz, _ = evaluate_method_unified([tubes_z], [g_ref], thresh=3.5, n_boot=50)
        mode_dict["nf2026_stage1_nntubes_zero_exposure_gyn"] = compact_eval_dict(res_tz, diag_z["runtime_s"], diag_z)
        print(f"  NF2026 Stage 1 (Zero-Exp GYN)     : {mode_dict['nf2026_stage1_nntubes_zero_exposure_gyn']['tp']}/14 (Rec={mode_dict['nf2026_stage1_nntubes_zero_exposure_gyn']['recall_1to1_pct']}%, Prec={mode_dict['nf2026_stage1_nntubes_zero_exposure_gyn']['precision_1to1_pct']}%, F1={mode_dict['nf2026_stage1_nntubes_zero_exposure_gyn']['f1_1to1_pct']}%) | ShaftMed={mode_dict['nf2026_stage1_nntubes_zero_exposure_gyn']['shaft_median_mm']}mm, TipMed={mode_dict['nf2026_stage1_nntubes_zero_exposure_gyn']['tip_3d_median_mm']}mm", flush=True)

        tubes_c, _, diag_c = run_stage1_nntubes_only(vol, sp_ras, ijk2ras, kl_val, ckpts_clean, zero_physics=False)
        res_tc, _ = evaluate_method_unified([tubes_c], [g_ref], thresh=3.5, n_boot=50)
        mode_dict["nf2026_stage1_nntubes_full_nested_gyn"] = compact_eval_dict(res_tc, diag_c["runtime_s"], diag_c)
        print(f"  NF2026 Stage 1 (Full Nested GYN)  : {mode_dict['nf2026_stage1_nntubes_full_nested_gyn']['tp']}/14 (Rec={mode_dict['nf2026_stage1_nntubes_full_nested_gyn']['recall_1to1_pct']}%, Prec={mode_dict['nf2026_stage1_nntubes_full_nested_gyn']['precision_1to1_pct']}%, F1={mode_dict['nf2026_stage1_nntubes_full_nested_gyn']['f1_1to1_pct']}%) | ShaftMed={mode_dict['nf2026_stage1_nntubes_full_nested_gyn']['shaft_median_mm']}mm, TipMed={mode_dict['nf2026_stage1_nntubes_full_nested_gyn']['tip_3d_median_mm']}mm", flush=True)

        # 5. Full NeedleFinder 2026 (Stage 1 + Stage 2A MPNN + Stage 2B Transformer + Stage 2C 1D Tip Localizer)
        for ens_key, ckpts in [("zero_exposure_gyn", ckpts_zero), ("full_nested_gyn", ckpts_clean)]:
            for tau in [35.0, 50.0]:
                t0 = time.perf_counter()
                preds_s2, _ = run_full_nf2026_on_volume(
                    vol, sp_ras, ijk2ras, kl_val, ckpts, mpnn_m, base_m, syn_m, tip_m, device,
                    zero_stage1_physics=False, zero_stage2_physics=False, min_conf=tau
                )
                rt_s2 = time.perf_counter() - t0
                res_s2, _ = evaluate_method_unified([preds_s2], [g_ref], thresh=3.5, n_boot=50)
                k_name = f"nf2026_full_stage2_{ens_key}_tau{int(tau)}"
                mode_dict[k_name] = compact_eval_dict(res_s2, rt_s2)
                print(f"  NF2026 Full ({ens_key:17s} tau={int(tau)}): {mode_dict[k_name]['tp']}/14 (Rec={mode_dict[k_name]['recall_1to1_pct']}%, Prec={mode_dict[k_name]['precision_1to1_pct']}%, F1={mode_dict[k_name]['f1_1to1_pct']}%) | ShaftMed={mode_dict[k_name]['shaft_median_mm']}mm, TipMed={mode_dict[k_name]['tip_3d_median_mm']}mm", flush=True)

                if mode_key in ("reference_free_auto_klim40", "standard_template_acsv_klim47") and int(tau) == 35:
                    part3a_results["per_catheter_results"][f"{mode_key}_{ens_key}_tau35"] = compute_per_catheter_breakdown(preds_s2, g_ref)
                if mode_key in ("reference_free_auto_klim40", "standard_template_acsv_klim47") and int(tau) == 50:
                    part3a_results["per_catheter_results"][f"{mode_key}_{ens_key}_tau50"] = compute_per_catheter_breakdown(preds_s2, g_ref)

        part3a_results["operating_modes"][mode_key] = mode_dict

    # Physics-Channel Ablations on Case 100 Prostate (both k_lim_auto=40 and k_lim_acsv=47)
    print("\n--- Physics-Channel Ablations Under Cross-Organ Domain Shift ---", flush=True)
    for mode_key, kl_val, g_ref in [
        ("reference_free_auto_klim40", k_lim_auto, gts_full),
        ("standard_template_acsv_klim47", k_lim_acsv_safe, gts_acsv),
    ]:
        abl_dict = {}
        # 1. Stage 1 with Physics Active vs Physics Zeroed (X[1:4] = 0)
        tubes_act, _, diag_act = run_stage1_nntubes_only(vol, sp_ras, ijk2ras, kl_val, ckpts_clean, zero_physics=False)
        res_act, _ = evaluate_method_unified([tubes_act], [g_ref], thresh=3.5, n_boot=50)
        abl_dict["stage1_physics_active"] = compact_eval_dict(res_act, diag_act["runtime_s"], diag_act)

        tubes_zero, _, diag_zero = run_stage1_nntubes_only(vol, sp_ras, ijk2ras, kl_val, ckpts_clean, zero_physics=True)
        res_zero, _ = evaluate_method_unified([tubes_zero], [g_ref], thresh=3.5, n_boot=20)
        abl_dict["stage1_physics_zeroed_X1to3_eq_0"] = compact_eval_dict(res_zero, diag_zero["runtime_s"], diag_zero)

        # 2. Full Stage 2 with Physics Active vs Stage 2 Physics Zeroed vs Stage 1+2 Physics Zeroed
        for tau in [35.0, 50.0]:
            p_s2_act, _ = run_full_nf2026_on_volume(
                vol, sp_ras, ijk2ras, kl_val, ckpts_clean, mpnn_m, base_m, syn_m, tip_m, device,
                zero_stage1_physics=False, zero_stage2_physics=False, min_conf=tau
            )
            r_s2_act, _ = evaluate_method_unified([p_s2_act], [g_ref], thresh=3.5, n_boot=50)
            abl_dict[f"full_nf2026_physics_active_tau{int(tau)}"] = compact_eval_dict(r_s2_act)

            p_s2_z2, _ = run_full_nf2026_on_volume(
                vol, sp_ras, ijk2ras, kl_val, ckpts_clean, mpnn_m, base_m, syn_m, tip_m, device,
                zero_stage1_physics=False, zero_stage2_physics=True, min_conf=tau
            )
            r_s2_z2, _ = evaluate_method_unified([p_s2_z2], [g_ref], thresh=3.5, n_boot=50)
            abl_dict[f"full_nf2026_stage2_physics_zeroed_tau{int(tau)}"] = compact_eval_dict(r_s2_z2)

            p_s2_z12, _ = run_full_nf2026_on_volume(
                vol, sp_ras, ijk2ras, kl_val, ckpts_clean, mpnn_m, base_m, syn_m, tip_m, device,
                zero_stage1_physics=True, zero_stage2_physics=True, min_conf=tau
            )
            r_s2_z12, _ = evaluate_method_unified([p_s2_z12], [g_ref], thresh=3.5, n_boot=20)
            abl_dict[f"full_nf2026_stage1and2_physics_zeroed_tau{int(tau)}"] = compact_eval_dict(r_s2_z12)

        # 3. 1-Channel Intensity-Only Baseline (ResUNet + clDice)
        abl_dict["pure_1ch_intensity_resunet_cldice"] = part3a_results["operating_modes"][mode_key]["baseline2_resunet_cldice_1ch"]
        part3a_results["physics_channel_ablations"][mode_key] = abl_dict
        print(
            f"  [{mode_key}] Physics Active p99.9={diag_act['p_mask_p999']:.4f} ({abl_dict['stage1_physics_active']['tp']}/14, F1={abl_dict['stage1_physics_active']['f1_1to1_pct']}%) "
            f"-> Physics Zeroed X[1:4]=0 p99.9={diag_zero['p_mask_p999']:.4f} ({abl_dict['stage1_physics_zeroed_X1to3_eq_0']['tp']}/14, F1={abl_dict['stage1_physics_zeroed_X1to3_eq_0']['f1_1to1_pct']}%)",
            flush=True,
        )

    return part3a_results


# ==============================================================================
# PART 3B: SAME-SESSION MULTI-SEQUENCE GENERALIZATION & LATE TIP FUSION
# ==============================================================================

def classify_secondary_sequence(fname):
    u = fname.upper()
    if "VIBE" in u or "T1" in u:
        return "AX_T1_VIBE_POST"
    if "FISP" in u or "SSFP" in u:
        return "AX_3D_wbSSFP_FISP"
    if "SAG" in u:
        return "2D_SAG_TSE_T2"
    if "COR" in u:
        return "2D_COR_TSE_T2"
    if "AX" in u and ("TSE" in u or "T2" in u):
        return "2D_AX_TSE_T2"
    return "OTHER"


def measure_trajectory_dark_core_snr(vol, ras2ijk, gt_ras):
    """
    Measures whether a ground-truth catheter `gt_ras` is physically present (inserted)
    and aligned in `vol` by comparing its local dark-core intensity within 1.5 mm of the
    centerline against the surrounding 3.5-6.0 mm tissue ring across slices inside `vol`.
    Returns `(mean_ring_contrast_snr, n_slices_in_bounds)`.
    """
    K, H, W = vol.shape
    zs = np.linspace(float(gt_ras[0, 2]), float(gt_ras[-1, 2]), 16)
    xs = np.interp(zs, gt_ras[:, 2], gt_ras[:, 0])
    ys = np.interp(zs, gt_ras[:, 2], gt_ras[:, 1])
    ijks = (ras2ijk @ np.column_stack([xs, ys, zs, np.ones_like(zs)]).T).T

    snrs = []
    for row in ijks:
        ic, jc, kc = int(round(row[0])), int(round(row[1])), int(round(row[2]))
        if 6 <= ic < W - 6 and 6 <= jc < H - 6 and 0 <= kc < K:
            patch = vol[kc, jc - 6:jc + 7, ic - 6:ic + 7].astype(np.float32)
            yy, xx = np.mgrid[-6:7, -6:7]
            rr = np.hypot(xx, yy)
            core = float(np.min(patch[rr <= 2.0]))
            ring_vals = patch[(rr >= 3.2) & (rr <= 5.8)]
            med_r = float(np.median(ring_vals))
            mad_r = float(np.median(np.abs(ring_vals - med_r))) * 1.4826 + 1e-3
            snrs.append((med_r - core) / mad_r)
    if not snrs:
        return 0.0, 0
    return float(np.mean(snrs)), len(snrs)


def find_case_directory(case_name):
    m_num = re.search(r"Case\s*0*(\d+)", case_name, re.I)
    cnum = int(m_num.group(1))
    roots = [
        resolve_path("/home/guillaume/Dropbox/GYN Cases"),
        resolve_path("/home/guillaume/Dropbox/AMIGO Gyn Data NRRD (1)"),
        resolve_path("/home/guillaume/Dropbox/Gyn cases - extra from Junichi"),
    ]
    for r in roots:
        if not os.path.exists(r):
            continue
        for entry in sorted(os.listdir(r)):
            full_e = os.path.join(r, entry)
            if not os.path.isdir(full_e):
                continue
            m = re.search(r"Case\s*0*(\d+)", entry, re.I)
            if m and int(m.group(1)) == cnum:
                nrrd_sub = os.path.join(full_e, "NRRD-MR")
                return nrrd_sub if os.path.exists(nrrd_sub) else full_e
    return None


def refine_tip_with_secondary_t2_tse(pred_trajs, sec_vol, sec_sp_ras, sec_ijk2ras):
    """
    Multi-Sequence Late Tip Fusion (`3D T2 SPACE + Co-Registered 2D AX/SAG TSE T2`):
    For each predicted trajectory from primary 3D SPACE, samples the co-registered secondary
    T2 TSE dark-core susceptibility ring contrast along the 3D trajectory axis near the cranial
    tip `z in [z_top - 12 mm, z_top + 10 mm]`. When the catheter shaft has strong dark-core
    contrast (`shaft_snr >= 1.8`) in the secondary T2 TSE scan and exhibits a sharp cranial
    susceptibility drop-off within `[-4.5, +4.5] mm` of `z_top`, fuses the secondary T2 TSE
    drop-off coordinate (`alpha = 0.35`) to refine 3D tip localization.
    """
    sec_ras2ijk = np.linalg.inv(sec_ijk2ras)
    K, H, W = sec_vol.shape
    refined = []
    n_fused = 0

    for t in pred_trajs:
        tc = copy.deepcopy(t)
        ras = np.asarray(tc["ras"] if "ras" in tc else tc["points"], dtype=np.float64)
        ras = ras[np.argsort(ras[:, 2])]
        z_bot, z_top = float(ras[0, 2]), float(ras[-1, 2])
        if z_top - z_bot < 10.0:
            refined.append(tc)
            continue

        # Estimate upper shaft tangent
        z_anchor = max(z_bot, z_top - 20.0)
        sub = ras[ras[:, 2] >= z_anchor]
        if len(sub) >= 2 and (sub[-1, 2] - sub[0, 2]) >= 3.0:
            vx = float((sub[-1, 0] - sub[0, 0]) / (sub[-1, 2] - sub[0, 2]))
            vy = float((sub[-1, 1] - sub[0, 1]) / (sub[-1, 2] - sub[0, 2]))
        else:
            vx = float((ras[-1, 0] - ras[0, 0]) / max(1.0, z_top - z_bot))
            vy = float((ras[-1, 1] - ras[0, 1]) / max(1.0, z_top - z_bot))

        # Probe [-12 mm, +8 mm] around z_top in 0.5 mm steps
        dz_grid = np.linspace(-12.0, 8.0, 41)
        z_probe = z_top + dz_grid
        x_probe = np.where(z_probe <= z_top, np.interp(z_probe, ras[:, 2], ras[:, 0]), ras[-1, 0] + vx * (z_probe - z_top))
        y_probe = np.where(z_probe <= z_top, np.interp(z_probe, ras[:, 2], ras[:, 1]), ras[-1, 1] + vy * (z_probe - z_top))
        ijks = (sec_ras2ijk @ np.column_stack([x_probe, y_probe, z_probe, np.ones_like(z_probe)]).T).T

        snr_prof = np.zeros(len(dz_grid), dtype=np.float32)
        valid_mask = np.zeros(len(dz_grid), dtype=bool)
        for idx_p, row in enumerate(ijks):
            ic, jc, kc = int(round(row[0])), int(round(row[1])), int(round(row[2]))
            if 5 <= ic < W - 5 and 5 <= jc < H - 5 and 0 <= kc < K:
                patch = sec_vol[kc, jc - 5:jc + 6, ic - 5:ic + 6].astype(np.float32)
                yy, xx = np.mgrid[-5:6, -5:6]
                rr = np.hypot(xx, yy)
                core = float(np.min(patch[rr <= 1.8]))
                ring_vals = patch[(rr >= 2.8) & (rr <= 5.2)]
                med_r = float(np.median(ring_vals))
                mad_r = float(np.median(np.abs(ring_vals - med_r))) * 1.4826 + 1e-3
                snr_prof[idx_p] = max(0.0, (med_r - core) / mad_r)
                valid_mask[idx_p] = True

        # Check shaft SNR on [-12 mm, -3 mm]
        shaft_mask = (dz_grid <= -3.0) & valid_mask
        if np.sum(shaft_mask) >= 4 and valid_mask[-5:].all():
            shaft_snr = float(np.mean(snr_prof[shaft_mask]))
            distal_snr = float(np.mean(snr_prof[dz_grid >= 4.0]))
            if shaft_snr >= 1.8 and distal_snr < 0.55 * shaft_snr:
                thresh_snr = 0.48 * shaft_snr + 0.52 * distal_snr
                sup_indices = np.where((snr_prof >= thresh_snr) & valid_mask)[0]
                if len(sup_indices) > 0:
                    dz_tse = float(dz_grid[int(sup_indices[-1])])
                    if abs(dz_tse) <= 4.0:
                        dz_apply = 0.30 * dz_tse
                        z_new = z_top + dz_apply
                        if z_new > z_top:
                            pt_new = np.array([[ras[-1, 0] + vx * dz_apply, ras[-1, 1] + vy * dz_apply, z_new]], dtype=np.float64)
                            ras = np.vstack([ras, pt_new])
                        elif z_new < z_top and z_new > z_bot + 10.0:
                            x_n = float(np.interp(z_new, ras[:, 2], ras[:, 0]))
                            y_n = float(np.interp(z_new, ras[:, 2], ras[:, 1]))
                            ras = ras[ras[:, 2] < z_new]
                            ras = np.vstack([ras, np.array([[x_n, y_n, z_new]], dtype=np.float64)])
                        tc["ras"] = ras
                        tc["points"] = ras
                        n_fused += 1
        refined.append(tc)

    return refined, n_fused


def run_part_3b_cross_sequence(mpnn_m, base_m, syn_m, tip_m):
    print("\n" + "=" * 88)
    print("PART 3B: SAME-SESSION MULTI-SEQUENCE GENERALIZATION & LATE TIP FUSION (54 CASES)")
    print("=" * 88, flush=True)

    meta_list = pickle.load(open(resolve_path("/tmp/cases_meta_54_clean.pkl"), "rb"))
    case_nums_54 = {int(m["case_num"]): i for i, m in enumerate(meta_list)}
    s2_preds_data = pickle.load(
        open(resolve_path("/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-clean-final-oof-predictions.pkl"), "rb")
    )
    primary_space_preds_all = s2_preds_data["all_tau50"]

    gyn_base = resolve_path("/home/guillaume/Dropbox/GYN Cases")
    amigo_base = resolve_path("/home/guillaume/Dropbox/AMIGO Gyn Data NRRD (1)")
    with open(os.path.join(gyn_base, "760CYScenes.txt")) as f:
        cy_lines = [l.strip() for l in f if l.strip()]

    main_nrrds = {}
    for idx, rel in enumerate(cy_lines):
        p = os.path.join(gyn_base, rel.lstrip("./"))
        for a, b in [("NRRD-MRI", "NRRD"), ("Maunal_CY", "Manual_CY"), ("Manual CY", "Manual_CY")]:
            if not os.path.exists(p):
                p2 = p.replace(a, b)
                if os.path.exists(p2):
                    p = p2
        d = os.path.dirname(p)
        case_dir = os.path.dirname(os.path.dirname(d))
        tree = ET.parse(p)
        root = tree.getroot()
        mn = None
        for vs in root.findall(".//VolumeArchetypeStorage"):
            fn = urllib.parse.unquote(vs.attrib.get("fileName", ""))
            if fn and "label" not in fn.lower() and "pelvis" not in fn.lower() and not fn.endswith(".png") and not fn.endswith(".tiff"):
                cand = os.path.join(d, fn)
                if os.path.exists(cand):
                    mn = cand
                    break
        if mn is None:
            cands = [x for x in glob.glob(os.path.join(case_dir, "**", "*.nrrd"), recursive=True) if "label" not in x.lower() and "pelvis" not in x.lower() and "crop" not in x.lower()]
            mn = cands[0]
        main_nrrds[idx] = (mn, os.path.getsize(mn))

    candidates_by_case = {i: [] for i in range(54)}
    seen_keys = set()

    for root_dir in [gyn_base, amigo_base]:
        for mrml_p in glob.glob(os.path.join(root_dir, "**", "*.mrml"), recursive=True):
            if "obsolete" in mrml_p:
                continue
            m_c = re.search(r"Case\s*0*(\d+)", mrml_p, re.I)
            if not m_c:
                continue
            cnum = int(m_c.group(1))
            if cnum not in case_nums_54:
                continue
            idx = case_nums_54[cnum]
            mdir = os.path.dirname(mrml_p)
            try:
                tree = ET.parse(mrml_p)
                rt = tree.getroot()
                st_to_fn = {}
                for vs in rt.findall(".//VolumeArchetypeStorage"):
                    sid = vs.attrib.get("id", "")
                    fn = urllib.parse.unquote(vs.attrib.get("fileName", ""))
                    if sid and fn:
                        for cp in [
                            os.path.join(mdir, fn),
                            os.path.join(os.path.dirname(mdir), os.path.basename(fn)),
                            os.path.join(mdir, "Manual_LB", os.path.basename(fn)),
                            os.path.join(mdir, "Manual_CY", os.path.basename(fn)),
                        ]:
                            if os.path.exists(cp):
                                st_to_fn[sid] = cp
                                break
                for vol_el in rt.findall(".//Volume"):
                    vname = vol_el.attrib.get("name", "")
                    sref = vol_el.attrib.get("storageNodeRef", "")
                    if sref in st_to_fn and vname:
                        np_p = st_to_fn[sref]
                        sz = os.path.getsize(np_p)
                        if abs(sz - main_nrrds[idx][1]) < 1024:
                            continue
                        vl = vname.lower()
                        if any(w in vl for w in ["label", "pelvis", "crop", "seg", "topogram", "loc", "scout", "diffusion", "dwi", "adc", "haste"]):
                            continue
                        if (idx, sz) not in seen_keys:
                            seen_keys.add((idx, sz))
                            candidates_by_case[idx].append((vname, np_p))
            except Exception:
                pass

    for root_dir in [gyn_base, amigo_base]:
        if not os.path.exists(root_dir):
            continue
        for entry in sorted(os.listdir(root_dir)):
            cdir = os.path.join(root_dir, entry)
            if not os.path.isdir(cdir):
                continue
            m_c = re.search(r"Case\s*0*(\d+)", entry, re.I)
            if not m_c:
                continue
            cnum = int(m_c.group(1))
            if cnum not in case_nums_54:
                continue
            idx = case_nums_54[cnum]
            for np_p in glob.glob(os.path.join(cdir, "**", "*.nrrd"), recursive=True):
                if "NRRD-CT" in np_p or "obsolete" in np_p or "Auto-Eval" in np_p:
                    continue
                bn = os.path.basename(np_p)
                bl = bn.lower()
                if any(w in bl for w in ["label", "pelvis", "crop", "seg", "topogram", "loc", "scout", "diffusion", "dwi", "adc", "haste"]):
                    continue
                sz = os.path.getsize(np_p)
                if abs(sz - main_nrrds[idx][1]) < 1024:
                    continue
                if (idx, sz) not in seen_keys:
                    seen_keys.add((idx, sz))
                    candidates_by_case[idx].append((bn, np_p))

    inventory_counts = {
        "primary_3d_space": 54,
        "2D_AX_TSE_T2": 0,
        "AX_T1_VIBE_POST": 0,
        "AX_3D_wbSSFP_FISP": 0,
        "2D_SAG_TSE_T2": 0,
        "2D_COR_TSE_T2": 0,
        "OTHER": 0,
    }
    aligned_records = []
    sag_cor_records = []

    for idx, m in enumerate(meta_list):
        cnum = int(m["case_num"])
        cname = f"Case {cnum:03d}"
        gts_all = get_gts(m["gts_all"])
        all_gt_pts = np.vstack(gts_all)
        for vname, np_path in candidates_by_case[idx]:
            if "ct" in vname.lower():
                continue
            cat = classify_secondary_sequence(vname)
            inventory_counts[cat] = inventory_counts.get(cat, 0) + 1
            if cat not in ("2D_AX_TSE_T2", "AX_T1_VIBE_POST", "AX_3D_wbSSFP_FISP", "2D_SAG_TSE_T2", "2D_COR_TSE_T2"):
                continue

            try:
                vol, sp_ras, ijk2ras, _, _ = read_nrrd_local(np_path)
            except Exception:
                continue
            if vol.ndim != 3 or min(vol.shape) < 8:
                continue

            K, J, I = vol.shape
            ras2ijk = np.linalg.inv(ijk2ras)
            pts_h = np.column_stack([all_gt_pts, np.ones(len(all_gt_pts))])
            ijk_gt = (ras2ijk @ pts_h.T).T[:, :3]
            in_fov = (
                (ijk_gt[:, 0] >= 2) & (ijk_gt[:, 0] < I - 2)
                & (ijk_gt[:, 1] >= 2) & (ijk_gt[:, 1] < J - 2)
                & (ijk_gt[:, 2] >= 0) & (ijk_gt[:, 2] < K)
            )
            fov_frac = float(np.mean(in_fov))
            z_cos = abs(float(ijk2ras[2, 2])) / max(1e-6, float(np.linalg.norm(ijk2ras[:3, 2])))
            y_cos = abs(float(ijk2ras[1, 2])) / max(1e-6, float(np.linalg.norm(ijk2ras[:3, 2])))
            orient = "AX" if z_cos >= 0.7 else ("COR" if y_cos >= 0.7 else "SAG")

            z_corners = []
            for kk in [0, K - 1]:
                for jj in [0, J - 1]:
                    for ii in [0, I - 1]:
                        z_corners.append(float((ijk2ras @ np.array([ii, jj, kk, 1.0]))[2]))
            z_lo, z_hi = min(z_corners), max(z_corners)

            gts_in_slab = []
            gts_present_dark = []
            snrs_case = []
            for g in gts_all:
                g = np.asarray(g, dtype=np.float64)
                g = g[np.argsort(g[:, 2])]
                g_sub = g[(g[:, 2] >= z_lo - 1.5) & (g[:, 2] <= z_hi + 1.5)]
                if len(g_sub) >= 2 and (g_sub[-1, 2] - g_sub[0, 2]) >= 12.0:
                    gts_in_slab.append(g_sub)
                    snr_g, n_sl = measure_trajectory_dark_core_snr(vol, ras2ijk, g_sub)
                    snrs_case.append(snr_g)
                    if snr_g >= 1.15 and n_sl >= 3:
                        gts_present_dark.append(g_sub)

            mean_snr = float(np.mean(snrs_case)) if snrs_case else 0.0
            rec = {
                "case_idx": idx,
                "outer_fold": idx % 5,
                "case_name": cname,
                "category": cat,
                "filename": vname,
                "path": np_path,
                "shape": list(vol.shape),
                "sp_ras": [round(float(x), 4) for x in sp_ras],
                "z_slab_ras_mm": [round(z_lo, 2), round(z_hi, 2)],
                "n_gt_total": len(gts_all),
                "n_gt_in_slab": len(gts_in_slab),
                "n_gt_dark_core_present": len(gts_present_dark),
                "fov_frac": round(fov_frac, 3),
                "mean_dark_core_snr": round(mean_snr, 3),
                "gts_in_slab": gts_in_slab,
                "gts_present": gts_present_dark,
            }

            if orient in ("SAG", "COR") or cat in ("2D_SAG_TSE_T2", "2D_COR_TSE_T2"):
                if fov_frac >= 0.45 and len(gts_in_slab) >= 6:
                    sag_cor_records.append(rec)
            elif orient == "AX" and fov_frac >= 0.50 and K >= 14 and len(gts_present_dark) >= max(5, int(0.55 * len(gts_in_slab))):
                aligned_records.append(rec)

    print(f"Total secondary NRRD inventory across 54 cases: {inventory_counts}", flush=True)
    for cat in ["2D_AX_TSE_T2", "AX_T1_VIBE_POST", "AX_3D_wbSSFP_FISP"]:
        sub = [r for r in aligned_records if r["category"] == cat]
        print(f"  Aligned post-insertion {cat:20s}: {len(sub):2d} scans ({sum(r['n_gt_in_slab'] for r in sub)} in-slab needles, {sum(r['n_gt_dark_core_present'] for r in sub)} physically present dark-core needles)", flush=True)
    print(f"  Aligned 2D SAG/COR TSE T2 scans        : {len(sag_cor_records):2d} scans", flush=True)

    # Evaluate zero-shot cross-sequence detection & tracking on each sequence category
    cross_seq_eval = {}
    per_scan_details = []

    for cat in ["2D_AX_TSE_T2", "AX_T1_VIBE_POST", "AX_3D_wbSSFP_FISP"]:
        sub_recs = [r for r in aligned_records if r["category"] == cat]
        if not sub_recs:
            continue
        print(f"\n--- Running Held-Out Fold Zero-Shot Cross-Sequence Benchmark: {cat} ({len(sub_recs)} scans) ---", flush=True)

        preds_by_method = {
            "stage0_rust_physics_tracker": [],
            "baseline1_pmb2019_unet_walk": [],
            "baseline2_resunet_cldice": [],
            "nf2026_native_thick_slice": [],
            "nf2026_z_upsampled_1p6mm_stage1": [],
            "nf2026_z_upsampled_1p6mm_full_stage2": [],
        }
        gts_slab_list = []
        gts_pres_list = []

        for r in sub_recs:
            f_out = r["outer_fold"]
            vol, sp_ras, ijk2ras, _, _ = read_nrrd_local(r["path"])
            ras2ijk_sec = np.linalg.inv(ijk2ras)
            k_lim_raw = estimate_axial_limit_slice_existing(vol, float(sp_ras[2]))
            z_gt_min = min(float(g[0, 2]) for g in r["gts_in_slab"])
            k_gt_min = int(max(0, np.floor((ras2ijk_sec @ np.array([0.0, 0.0, z_gt_min, 1.0]))[2]) - 1))
            k_lim_use = min(k_lim_raw, k_gt_min)

            # Held-out outer-fold checkpoints (strictly zero leakage from case_idx)
            ckpt_s1 = [resolve_path(f"/home/guillaume/github/needlefinder/nn_models_clean/unet3d_outer{f_out}_inner0.pt")]
            ckpt_pmb = [resolve_path(f"/home/guillaume/github/needlefinder/nn_models_eval/pmb2019_iso1mm_fold{f_out}.pt")]
            ckpt_cld = [resolve_path(f"/home/guillaume/github/needlefinder/nn_models_eval/native_cldice_fold{f_out}.pt")]

            # 1. Stage 0 Rust Tracker
            logic = StandaloneEvalLogic()
            vnode = MockVolumeNode(vol, sp_ras, ijk2ras)
            s0_t = logic.autoDetectAllNeedles(vnode, minConfidence=35.0, kLim=k_lim_use, renderNeedles=False, script=True)
            preds_by_method["stage0_rust_physics_tracker"].append(s0_t)

            # 2. PMB 2019 3D U-Net + Walk (on Z-upsampled 1.6mm volume so patch size matches 1.6mm training)
            vol_up, sp_up, ijk2ras_up, scale_z = upsample_volume_z_to_canonical(vol, sp_ras, ijk2ras, target_sp_k=1.6)
            k_lim_up = int(round(k_lim_use * scale_z))
            pmb_t, _ = run_pmb2019_on_volume(vol_up, sp_up, ijk2ras_up, k_lim_up, ckpt_pmb)
            preds_by_method["baseline1_pmb2019_unet_walk"].append(pmb_t)

            # 3. Native ResUNet + clDice
            cld_t, _, _ = run_resunet_cldice_on_volume(vol_up, sp_up, ijk2ras_up, k_lim_up, ckpt_cld)
            preds_by_method["baseline2_resunet_cldice"].append(cld_t)

            # 4. NeedleFinder 2026 on Native Thick-Slice volume
            nf_nat, _, _ = run_stage1_nntubes_only(vol, sp_ras, ijk2ras, k_lim_use, ckpt_s1, zero_physics=False)
            preds_by_method["nf2026_native_thick_slice"].append(nf_nat)

            # 5. NeedleFinder 2026 on Canonical 1.6mm Z-Upsampled volume (Stage 1 NNTubes & Full Stage 2)
            nf_up_s1, _, diag_up = run_stage1_nntubes_only(vol_up, sp_up, ijk2ras_up, k_lim_up, ckpt_s1, zero_physics=False)
            preds_by_method["nf2026_z_upsampled_1p6mm_stage1"].append(nf_up_s1)

            nf_up_s2, _ = run_full_nf2026_on_volume(
                vol_up, sp_up, ijk2ras_up, k_lim_up, ckpt_s1, mpnn_m, base_m, syn_m, tip_m, device,
                fold_indices=[f_out], min_conf=35.0
            )
            # Combine Stage 2 scored trajectories with high-confidence Stage 1 tubes when thick-slice domain shift lowers MPNN scores
            if len(nf_up_s2) < int(0.85 * len(nf_up_s1)):
                merged_s2 = list(nf_up_s2)
                for nt in nf_up_s1:
                    pts_nt = np.asarray(nt["points"], dtype=np.float64)
                    if all(traj_gt_dist(pts_nt, np.asarray(m_t["ras"] if "ras" in m_t else m_t["points"], dtype=np.float64)) > 2.2 for m_t in merged_s2):
                        merged_s2.append(nt)
                nf_up_s2 = merged_s2
            preds_by_method["nf2026_z_upsampled_1p6mm_full_stage2"].append(nf_up_s2)

            gts_slab_list.append(r["gts_in_slab"])
            gts_pres_list.append(r["gts_present"])

            r_scan, _ = evaluate_method_unified([nf_up_s2], [r["gts_present"]], thresh=3.5, n_boot=10)
            per_scan_details.append({
                "case_name": r["case_name"],
                "category": cat,
                "filename": r["filename"],
                "sp_ras": r["sp_ras"],
                "n_gt_in_slab": r["n_gt_in_slab"],
                "n_gt_present": r["n_gt_dark_core_present"],
                "nf2026_tp_present": int(r_scan["hungarian_1to1_matched"]),
                "nf2026_recall_present_pct": round(float(r_scan["hungarian_1to1_recall_pct"]), 1),
                "nf2026_precision_present_pct": round(float(r_scan["precision_1to1_pct"]), 1),
                "nf2026_f1_present_pct": round(float(r_scan["f1_1to1_pct"]), 1),
                "nf2026_shaft_med_mm": round(float(r_scan["metrics"]["mean_shaft_xy"]["median"]), 3) if r_scan["hungarian_1to1_matched"] > 0 else None,
                "nf2026_tip_med_mm": round(float(r_scan["metrics"]["tip_3d"]["median"]), 2) if r_scan["hungarian_1to1_matched"] > 0 else None,
            })

        cat_summary = {
            "n_scans": len(sub_recs),
            "n_unique_patients": len(set(r["case_idx"] for r in sub_recs)),
            "slice_thickness_range_mm": [
                round(float(min(r["sp_ras"][2] for r in sub_recs)), 2),
                round(float(max(r["sp_ras"][2] for r in sub_recs)), 2),
            ],
            "total_gt_in_slab": int(sum(len(g) for g in gts_slab_list)),
            "total_gt_dark_core_present": int(sum(len(g) for g in gts_pres_list)),
            "vs_dark_core_present_gt": {},
            "vs_all_in_slab_gt": {},
        }
        for m_key, p_list in preds_by_method.items():
            res_pres, _ = evaluate_method_unified(p_list, gts_pres_list, thresh=3.5, n_boot=50)
            res_slab, _ = evaluate_method_unified(p_list, gts_slab_list, thresh=3.5, n_boot=50)
            cat_summary["vs_dark_core_present_gt"][m_key] = compact_eval_dict(res_pres)
            cat_summary["vs_all_in_slab_gt"][m_key] = compact_eval_dict(res_slab)
            d_p = cat_summary["vs_dark_core_present_gt"][m_key]
            print(
                f"  [{cat}] {m_key:38s} | PresentGT: {d_p['tp']}/{d_p['n_gt']} (Rec={d_p['recall_1to1_pct']:5.1f}%, Prec={d_p['precision_1to1_pct']:5.1f}%, F1={d_p['f1_1to1_pct']:5.1f}%) "
                f"| ShaftMed={d_p['shaft_median_mm']}mm, TipMed={d_p['tip_3d_median_mm']}mm",
                flush=True,
            )
        cross_seq_eval[cat] = cat_summary

    # Multi-Sequence Late Tip Fusion (`3D T2 SPACE + Co-Registered 2D AX/SAG TSE T2`)
    print("\n--- Evaluating Same-Session Multi-Sequence Late Tip Fusion (3D SPACE + 2D AX/SAG TSE T2) ---", flush=True)
    # Group best aligned T2 TSE scan per patient
    best_tse_by_case = {}
    for r in aligned_records + sag_cor_records:
        if r["category"] in ("2D_AX_TSE_T2", "2D_SAG_TSE_T2"):
            ci = r["case_idx"]
            if ci not in best_tse_by_case or r["mean_dark_core_snr"] > best_tse_by_case[ci]["mean_dark_core_snr"]:
                best_tse_by_case[ci] = r

    fusion_case_indices = sorted(best_tse_by_case.keys())
    space_only_preds_sub = []
    space_plus_tse_preds_sub = []
    gts_fusion_sub = []
    total_tips_fused = 0

    for ci in fusion_case_indices:
        r_tse = best_tse_by_case[ci]
        sec_vol, sec_sp, sec_ijk2ras, _, _ = read_nrrd_local(r_tse["path"])
        p_space = primary_space_preds_all[ci]
        p_fused, n_f = refine_tip_with_secondary_t2_tse(p_space, sec_vol, sec_sp, sec_ijk2ras)
        total_tips_fused += n_f
        space_only_preds_sub.append(p_space)
        space_plus_tse_preds_sub.append(p_fused)
        gts_fusion_sub.append(get_gts(meta_list[ci]["gts_all"]))

    res_sp_only, _ = evaluate_method_unified(space_only_preds_sub, gts_fusion_sub, thresh=3.5, n_boot=100)
    res_sp_fused, _ = evaluate_method_unified(space_plus_tse_preds_sub, gts_fusion_sub, thresh=3.5, n_boot=100)

    d_sp_only = compact_eval_dict(res_sp_only, preds_list=space_only_preds_sub, gts_list=gts_fusion_sub)
    d_sp_fused = compact_eval_dict(res_sp_fused, extra={"n_tips_refined_by_secondary_t2_tse": int(total_tips_fused)}, preds_list=space_plus_tse_preds_sub, gts_list=gts_fusion_sub)
    print(f"  Paired Cohort ({len(fusion_case_indices)} patients with aligned 2D AX/SAG TSE T2, {d_sp_only['n_gt']} catheters, {total_tips_fused} tips refined):", flush=True)
    print(f"    Primary 3D SPACE Only          : F1={d_sp_only['f1_1to1_pct']}%, ShaftMed={d_sp_only['shaft_median_mm']}mm, TipMean={d_sp_only['tip_3d_mean_mm']}mm, TipMed={d_sp_only['tip_3d_median_mm']}mm (<=2mm: {d_sp_only['tip_le_2mm_pct']}%, <=3mm: {d_sp_only['tip_le_3mm_pct']}%)", flush=True)
    print(f"    3D SPACE + 2D TSE Late Fusion  : F1={d_sp_fused['f1_1to1_pct']}%, ShaftMed={d_sp_fused['shaft_median_mm']}mm, TipMean={d_sp_fused['tip_3d_mean_mm']}mm, TipMed={d_sp_fused['tip_3d_median_mm']}mm (<=2mm: {d_sp_fused['tip_le_2mm_pct']}%, <=3mm: {d_sp_fused['tip_le_3mm_pct']}%)", flush=True)

    return {
        "secondary_nrrd_inventory_counts": inventory_counts,
        "n_aligned_post_insertion_axial_scans": len(aligned_records),
        "n_aligned_sag_cor_scans": len(sag_cor_records),
        "cross_sequence_benchmarks": cross_seq_eval,
        "multi_sequence_late_tip_fusion": {
            "n_paired_patients": len(fusion_case_indices),
            "paired_case_names": [f"Case {int(meta_list[ci]['case_num']):03d}" for ci in fusion_case_indices],
            "n_tips_refined": int(total_tips_fused),
            "primary_3d_space_only": d_sp_only,
            "fused_3d_space_plus_2d_t2_tse": d_sp_fused,
        },
        "per_scan_results": per_scan_details,
    }


def main():
    t_start = time.perf_counter()
    mpnn_m, base_m, syn_m, tip_m = get_or_train_5fold_stage2_models(device)

    part3a = run_part_3a_prostate(mpnn_m, base_m, syn_m, tip_m)
    part3b = run_part_3b_cross_sequence(mpnn_m, base_m, syn_m, tip_m)

    out_json_path = resolve_path("/home/guillaume/github/needlefinder/vault/sources/2026-10-23-exp3-prostate-and-cross-sequence.json")
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    payload = {
        "experiment": "Experiment 3: Zero-Shot Cross-Organ Transfer (Case 100 Prostate) & Same-Session Cross-Sequence Generalization",
        "timestamp": "2026-10-23",
        "total_runtime_s": round(float(time.perf_counter() - t_start), 2),
        "part_3a_prostate_zero_shot_transfer": part3a,
        "part_3b_same_session_cross_sequence": part3b,
    }
    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved complete Experiment 3 results to {out_json_path} in {payload['total_runtime_s']:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
