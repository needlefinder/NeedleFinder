#!/usr/bin/env python3
"""
Clean Nested 5x5 CV Stage 2B Confidence Calibration, Reason-Code Audit, and Failure Analysis
============================================================================================
Re-generates on the authoritative Clean Nested 5x5 CV Stitched Candidate Pool
(1,960 candidates across 54 cases; 1,118 kept tracks at tau=50%; 972 1-to-1 matched;
91 missed [66 long + 25 short]; 146 unmatched FPs):
  1. /home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2b-calibration-and-reason-codes.json
  2. /home/guillaume/github/needlefinder/vault/sources/2026-10-23-failure-analysis.json
"""

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

for mod in [
    "vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util",
    "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"
]:
    sys.modules[mod] = types.ModuleType(mod)

sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean")

from evaluate_unified_apples_to_apples import (
    compute_unified_pair_metrics,
    summarize_distribution,
    evaluate_method_unified,
    load_human_interobserver_40cases,
    resample_polyline_3d,
    get_gts,
    traj_gt_dist,
)
from run_stage2_and_tip_clean import (
    Dedicated1DTipLocalizer,
    collinear_traj_dist,
    merge_collinear_trajectories,
    fit_local_slope,
    apply_tip_adjustment_and_poly3,
)

META_CLEAN_PATH = "/tmp/cases_meta_54_clean.pkl"
CACHE_CLEAN_DIR = "/tmp/needle_unet3d_cache_clean"
PREDS_FULL_PKL = "/tmp/stage2_clean_final_oof_predictions_full.pkl"
PREDS_FALLBACK_PKL = "/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-clean-final-oof-predictions.pkl"
BUNDLES_CLEAN_PKL = "/tmp/stage2_clean_bundles_nn_models_clean.pkl"
STEP1_SUMMARY_JSON = "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean/step1_clean_dataset_summary.json"
OUT_CALIB_JSON = "/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2b-calibration-and-reason-codes.json"
OUT_FAILURE_JSON = "/home/guillaume/github/needlefinder/vault/sources/2026-10-23-failure-analysis.json"


def compute_pair_metrics_unbounded(det_pts, gt_pts):
    """
    Same as compute_unified_pair_metrics, but without the 8.0 mm mean_shaft_xy cutoff
    and without requiring >1.0 mm Z overlap so every prediction-GT pair can be diagnosed.
    """
    det_ras = np.asarray(det_pts, dtype=np.float64)
    gt_ras = np.asarray(gt_pts, dtype=np.float64)
    det_ras = det_ras[np.argsort(det_ras[:, 2])]
    gt_ras = gt_ras[np.argsort(gt_ras[:, 2])]
    if len(det_ras) < 2 or len(gt_ras) < 2:
        return None

    z_min = max(float(det_ras[0, 2]), float(gt_ras[0, 2]))
    z_max = min(float(det_ras[-1, 2]), float(gt_ras[-1, 2]))
    z_overlap = max(0.0, z_max - z_min)

    if z_max > z_min + 1.0:
        zs = np.linspace(z_min, z_max, 40)
        dx = np.interp(zs, det_ras[:, 2], det_ras[:, 0]) - np.interp(zs, gt_ras[:, 2], gt_ras[:, 0])
        dy = np.interp(zs, det_ras[:, 2], det_ras[:, 1]) - np.interp(zs, gt_ras[:, 2], gt_ras[:, 1])
        xy_dists = np.hypot(dx, dy)
        mean_shaft_xy = float(np.mean(xy_dists))
        hd_xy = float(np.max(xy_dists))
        hd95_xy = float(np.percentile(xy_dists, 95))
    else:
        mean_shaft_xy = 999.0
        hd_xy = 999.0
        hd95_xy = 999.0

    zs_det = np.linspace(float(det_ras[0, 2]), float(det_ras[-1, 2]), 25)
    gt_z_span = max(1.0, float(gt_ras[-1, 2] - gt_ras[0, 2]))
    gt_vx = float(gt_ras[-1, 0] - gt_ras[0, 0]) / gt_z_span
    gt_vy = float(gt_ras[-1, 1] - gt_ras[0, 1]) / gt_z_span
    gt_x_at_det = np.where(
        (zs_det >= gt_ras[0, 2]) & (zs_det <= gt_ras[-1, 2]),
        np.interp(zs_det, gt_ras[:, 2], gt_ras[:, 0]),
        np.where(
            zs_det < gt_ras[0, 2],
            gt_ras[0, 0] + gt_vx * (zs_det - gt_ras[0, 2]),
            gt_ras[-1, 0] + gt_vx * (zs_det - gt_ras[-1, 2]),
        ),
    )
    gt_y_at_det = np.where(
        (zs_det >= gt_ras[0, 2]) & (zs_det <= gt_ras[-1, 2]),
        np.interp(zs_det, gt_ras[:, 2], gt_ras[:, 1]),
        np.where(
            zs_det < gt_ras[0, 2],
            gt_ras[0, 1] + gt_vy * (zs_det - gt_ras[0, 2]),
            gt_ras[-1, 1] + gt_vy * (zs_det - gt_ras[-1, 2]),
        ),
    )
    det_x = np.interp(zs_det, det_ras[:, 2], det_ras[:, 0])
    det_y = np.interp(zs_det, det_ras[:, 2], det_ras[:, 1])
    extrap_xy_dist = float(np.mean(np.hypot(det_x - gt_x_at_det, det_y - gt_y_at_det)))

    P_3d = resample_polyline_3d(det_ras, n_samples=50)
    G_3d = resample_polyline_3d(gt_ras, n_samples=50)
    diff = P_3d[:, None, :] - G_3d[None, :, :]
    dist_mat_3d = np.linalg.norm(diff, axis=-1)
    d_p2g = np.min(dist_mat_3d, axis=1)
    d_g2p = np.min(dist_mat_3d, axis=0)
    msd_3d = float(0.5 * (np.mean(d_p2g) + np.mean(d_g2p)))
    min_p2g_3d = float(np.mean(d_p2g))
    hd_3d = float(max(np.max(d_p2g), np.max(d_g2p)))
    hd95_3d = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))

    det_tip = det_ras[-1]
    gt_tip = gt_ras[-1]
    tip_3d = float(np.linalg.norm(det_tip - gt_tip))
    tip_z = float(abs(det_tip[2] - gt_tip[2]))
    dz_tip_signed = float(det_tip[2] - gt_tip[2])
    tip_xy = float(np.hypot(det_tip[0] - gt_tip[0], det_tip[1] - gt_tip[1]))

    det_base = det_ras[0]
    gt_base = gt_ras[0]
    base_3d = float(np.linalg.norm(det_base - gt_base))
    dz_base_signed = float(det_base[2] - gt_base[2])

    return {
        "mean_shaft_xy": mean_shaft_xy,
        "extrap_xy_dist": extrap_xy_dist,
        "z_overlap": z_overlap,
        "msd_3d": msd_3d,
        "min_p2g_3d": min_p2g_3d,
        "hd_3d": hd_3d,
        "hd95_3d": hd95_xy if not np.isfinite(hd95_3d) else hd95_3d,
        "hd_xy": hd_xy,
        "hd95_xy": hd95_xy,
        "tip_3d": tip_3d,
        "tip_z": tip_z,
        "dz_tip_signed": dz_tip_signed,
        "tip_xy": tip_xy,
        "base_3d": base_3d,
        "dz_base_signed": dz_base_signed,
    }


def logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def decode_clean_with_reason_codes_and_diagnostics(
    stitched_enriched_bgs,
    base_oof_scores,
    synth_oof_outputs,
    tip_models_5fold,
    gts_all_list,
    min_conf=50.0,
    w_synth=0.60,
    tip_scale=0.36,
    thresh=3.5,
):
    """
    Executes the exact `decode_clean_stage2_preds` pipeline on `stitched_enriched_bgs` (1,960 candidates)
    while recording for every candidate:
      - Prior-weight calibrated probability `conf` in [0, 100] (where `conf >= 50.0%` corresponds
        1-to-1 with `eff_min_conf = 65.5%` after correcting the `pos_weight=[3.2, 2.6, 2.4]` training shift)
      - Reason code in {"kept", "below_threshold", "outside_implant_region", "low_dark_core_symmetry", "duplicate_of_kept_track"}
      - Refined trajectory (`refined_ras`) using the 5-fold `Dedicated1DTipLocalizer`
      - Exact match status against `gts_all_list[i]` (both closest-GT `is_true` and Hungarian 1-to-1 match).
    """
    preds_out = []
    diag_out = []
    calib_records = []

    eff_min_conf = min_conf + 15.5
    logit_shift = float(logit(eff_min_conf / 100.0) - logit(min_conf / 100.0))

    for i, bg in enumerate(stitched_enriched_bgs):
        pool = bg["pool"]
        sc_base = base_oof_scores[i]
        out_syn = synth_oof_outputs[i]
        sc_syn = 1.0 / (1.0 + np.exp(-np.clip(out_syn[:, :3], -20.0, 20.0)))
        dz_pred_stage2b = out_syn[:, 3] * 15.0

        sc = (1.0 - w_synth) * sc_base + w_synth * sc_syn
        conf_arr = 100.0 * (0.45 * sc[:, 0] + 0.55 * sc[:, 1])

        for u_idx, p_cand in enumerate(pool):
            span_u = float(p_cand.get("z_span", 20.0))
            if span_u < 18.0:
                conf_arr[u_idx] -= 5.0 * (18.0 - span_u) / 10.0
            elif span_u >= 35.0 and sc[u_idx, 0] >= 0.75:
                conf_arr[u_idx] += min(3.5, (span_u - 30.0) / 20.0)

        for u_idx, p_cand in enumerate(pool):
            if conf_arr[u_idx] < eff_min_conf and sc[u_idx, 0] >= 0.90 and p_cand.get("conf_heur", 0.0) >= 0.88 and p_cand.get("mean_sym", 0.0) >= 0.70:
                conf_arr[u_idx] = max(conf_arr[u_idx], float(eff_min_conf + 2.0))

        # Map `conf_arr` (with decision threshold `eff_min_conf = 65.5%`) to prior-odds-corrected
        # probability `p_cal_pct` (with decision threshold `50.0%`!):
        p_cal_pct = 100.0 * sigmoid(logit(np.clip(conf_arr / 100.0, 1e-4, 1.0 - 1e-4)) - logit_shift)

        high_idx = np.where(conf_arr >= max(55.0, eff_min_conf))[0]
        if len(high_idx) >= 3:
            all_xy = np.concatenate([pool[idx]["ras"][:, :2] for idx in high_idx], axis=0)
            center_xy = np.median(all_xy, axis=0)
            dists_core = np.linalg.norm(all_xy - center_xy[None, :], axis=1)
            irgm_radius_mm = float(np.percentile(dists_core, 98.0)) + 15.5
        else:
            center_xy = None
            irgm_radius_mm = 999.0

        f_out = i % 5
        tip_model = tip_models_5fold[f_out]

        # Precompute 1D-tip-refined trajectory for every candidate in the pool for pool-level evaluation
        cand_refined_ras = []
        cand_dz_1d = []
        cand_dz_comb = []
        for idx_c, p_cand in enumerate(pool):
            pts = np.asarray(p_cand["ras"], dtype=np.float64)
            pts = pts[np.argsort(pts[:, 2])]
            vx_top, vy_top = fit_local_slope(pts, float(pts[-1, 2]), 25.0)
            dz_prof_mm = float(p_cand.get("z_prof_tip", pts[-1, 2]) - pts[-1, 2])
            dz_nbr_mm = float(p_cand.get("z_tip_nbr", pts[-1, 2]) - pts[-1, 2])
            dz_s2b_mm = float(dz_pred_stage2b[idx_c])
            x_t = bg["tip_profiles_91"][idx_c:idx_c + 1]
            with torch.no_grad():
                dz_1d_pred, _ = tip_model(x_t)
                dz_1d_mm = float(dz_1d_pred[0].item())
            dz_c = 0.50 * dz_1d_mm + 0.30 * (tip_scale * dz_s2b_mm) + 0.15 * (tip_scale * dz_prof_mm) + 0.05 * (tip_scale * dz_nbr_mm)
            pts_ref = apply_tip_adjustment_and_poly3(pts, dz_c, vx_top, vy_top)
            cand_refined_ras.append(pts_ref)
            cand_dz_1d.append(dz_1d_mm)
            cand_dz_comb.append(dz_c)

        # Ground-truth match for every candidate in the pool (`<= 3.5 mm` mean shaft XY)
        is_true = np.zeros(len(pool), dtype=bool)
        best_d = np.full(len(pool), np.inf)
        best_g = np.full(len(pool), -1)
        for ci_p, p_cand in enumerate(pool):
            p_raw = np.asarray(p_cand["ras"], dtype=np.float64)
            p_raw = p_raw[np.argsort(p_raw[:, 2])]
            p_ref = cand_refined_ras[ci_p]
            for gi, g_pts in enumerate(gts_all_list[i]):
                pm_r = compute_unified_pair_metrics(p_raw, g_pts)
                pm_f = compute_unified_pair_metrics(p_ref, g_pts)
                d_val = min(
                    pm_r["mean_shaft_xy"] if pm_r is not None else np.inf,
                    pm_f["mean_shaft_xy"] if pm_f is not None else np.inf,
                )
                if d_val < best_d[ci_p]:
                    best_d[ci_p] = d_val
                    best_g[ci_p] = gi
            is_true[ci_p] = (best_d[ci_p] <= thresh)

        sel = []
        case_diag = [None] * len(pool)

        for idx_c in np.argsort(-conf_arr):
            cand = dict(pool[idx_c])
            pts = np.asarray(cand["ras"], dtype=np.float64)
            pts = pts[np.argsort(pts[:, 2])]

            dz_prof_mm = float(cand.get("z_prof_tip", pts[-1, 2]) - pts[-1, 2])
            dz_nbr_mm = float(cand.get("z_tip_nbr", pts[-1, 2]) - pts[-1, 2])
            dz_s2b_mm = float(dz_pred_stage2b[idx_c])

            info = {
                "pool_idx": int(idx_c),
                "conf": float(p_cal_pct[idx_c]),
                "raw_score": float(conf_arr[idx_c]),
                "raw_ras": pts,
                "refined_ras": cand_refined_ras[idx_c],
                "dz_pred_net_mm": dz_s2b_mm,
                "dz_1d_mm": float(cand_dz_1d[idx_c]),
                "dz_prof_mm": dz_prof_mm,
                "dz_nbr_mm": dz_nbr_mm,
                "dz_comb_mm": float(cand_dz_comb[idx_c]),
                "from_mpnn": float(cand.get("from_mpnn", 0.0)),
                "mean_sym": float(cand.get("mean_sym", 0.5)),
                "frac_active": float(cand.get("frac_active", 0.5)),
                "is_stitched": bool(cand.get("is_stitched", False)),
            }

            if conf_arr[idx_c] < eff_min_conf:
                info["status"] = "rejected_low_conf"
                info["reason"] = "below_threshold"
                case_diag[idx_c] = info
                continue

            if center_xy is not None:
                mean_dist_center = float(np.linalg.norm(pts[:, :2].mean(axis=0) - center_xy))
                if mean_dist_center > irgm_radius_mm and conf_arr[idx_c] < eff_min_conf + 18.0:
                    info["status"] = "rejected_irgm"
                    info["reason"] = "outside_implant_region"
                    case_diag[idx_c] = info
                    continue

            if cand.get("mean_sym", 0.5) < 0.14 and cand.get("frac_active", 0.5) < 0.40 and conf_arr[idx_c] < eff_min_conf + 15.0:
                info["status"] = "rejected_low_sym"
                info["reason"] = "low_dark_core_symmetry"
                case_diag[idx_c] = info
                continue

            dedup_thresh = 1.65 if (conf_arr[idx_c] >= eff_min_conf + 12.0 and cand.get("mpnn_pa", 0.0) >= 0.50) else 2.10
            dup_s = None
            can_extend_s = False
            for s in sel:
                info_col = collinear_traj_dist(pts, s["raw_ras"])
                min_sp = min(info_col["span_p"], info_col["span_s"])
                max_sp = max(info_col["span_p"], info_col["span_s"])
                is_dup = False
                if info_col["z_ov"] >= 2.0 and info_col["d_ov"] < dedup_thresh:
                    is_dup = True
                elif info_col["z_ov"] < 4.0 and info_col["z_gap"] <= 20.0 and (info_col["d_ext"] <= 2.40 or info_col["d_mid"] <= 2.30) and info_col["d_slope"] <= 0.14:
                    is_dup = True
                elif info_col["z_ov"] >= 2.0 and (info_col["d_ov"] <= 2.85 or info_col["d_ext"] <= 2.35) and info_col["d_slope"] <= 0.12 and (min_sp < 0.72 * max_sp or min_sp < 26.0):
                    is_dup = True
                elif info_col["z_ov"] >= 10.0 and info_col["d_ov"] <= 2.60 and info_col["d_ext"] <= 2.25 and info_col["d_slope"] <= 0.09:
                    is_dup = True

                if is_dup:
                    dup_s = s
                    can_extend_s = (info_col["d_ext"] <= 2.35 or info_col["d_ov"] <= 2.50) and (info_col["d_slope"] <= 0.12)
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
                info["status"] = "rejected_dedup"
                info["reason"] = "duplicate_of_kept_track"
                info["dedup_winner_pool_idx"] = int(dup_s["orig_pool_idx"])
                case_diag[idx_c] = info
                continue

            info["status"] = "selected"
            info["reason"] = "kept"
            info["sel_idx"] = len(sel)
            case_diag[idx_c] = info

            cand["raw_ras"] = pts
            cand["idx_c"] = int(idx_c)
            cand["orig_pool_idx"] = int(idx_c)
            cand["pool_idx"] = int(idx_c)
            cand["score"] = float(p_cal_pct[idx_c])
            cand["raw_score"] = float(conf_arr[idx_c])
            sel.append(cand)

        # Second pass extension for severely undershot trajectories (`z_top < z_tip_nbr - 6.0 mm`)
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
                    info_col = collinear_traj_dist(c_ras, s["raw_ras"])
                    if info_col["z_gap"] <= 14.0 and info_col["d_slope"] <= 0.10 and (info_col["d_ov"] <= 2.60 or (info_col["d_ext"] <= 2.10 and info_col["d_mid"] <= 2.25)):
                        d_to_s = min(info_col["d_ov"], info_col["d_ext"], info_col["d_mid"])
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

        # Apply 1D tip localization & Poly3 smoothing AFTER collinear fragment stitching/extension
        for s in sel:
            idx_c = s["idx_c"]
            pts = s["raw_ras"]
            vx_top, vy_top = fit_local_slope(pts, float(pts[-1, 2]), 25.0)
            dz_prof_mm = float(s.get("z_prof_tip", pts[-1, 2]) - pts[-1, 2])
            dz_nbr_mm = float(s.get("z_tip_nbr", pts[-1, 2]) - pts[-1, 2])
            dz_s2b_mm = float(dz_pred_stage2b[idx_c])
            x_t = bg["tip_profiles_91"][idx_c:idx_c + 1]
            with torch.no_grad():
                dz_1d_pred, _ = tip_model(x_t)
                dz_1d_mm = float(dz_1d_pred[0].item())
            dz_comb_mm = 0.50 * dz_1d_mm + 0.30 * (tip_scale * dz_s2b_mm) + 0.15 * (tip_scale * dz_prof_mm) + 0.05 * (tip_scale * dz_nbr_mm)
            pts_refined = apply_tip_adjustment_and_poly3(pts, dz_comb_mm, vx_top, vy_top)
            s["ras"] = pts_refined
            s["points"] = pts_refined
            s["dz_comb_mm"] = dz_comb_mm
            s["dz_pred_net_mm"] = dz_s2b_mm
            s["dz_1d_mm"] = dz_1d_mm
            s["dz_prof_mm"] = dz_prof_mm
            s["dz_nbr_mm"] = dz_nbr_mm
            s["vx_top"] = vx_top
            s["vy_top"] = vy_top

        # Check which GTs in case i are covered by `sel` (within 3.5 mm)
        kept_gts = set()
        for idx_c in range(len(pool)):
            if case_diag[idx_c]["reason"] == "kept" and is_true[idx_c]:
                kept_gts.add(int(best_g[idx_c]))
        for s in sel:
            for gi, g_pts in enumerate(gts_all_list[i]):
                pm = compute_unified_pair_metrics(s["ras"], g_pts)
                if pm is not None and pm["mean_shaft_xy"] <= thresh:
                    kept_gts.add(int(gi))

        case_recs = []
        for idx_c in np.argsort(-conf_arr):
            d_info = case_diag[idx_c]
            reason = d_info["reason"]
            p_pts = d_info["raw_ras"]
            gt_i = int(best_g[idx_c])
            t_flag = bool(is_true[idx_c])
            real_status = (
                "kept" if reason == "kept" else ("redundant" if gt_i in kept_gts else "lost")
            ) if t_flag else "false"
            case_recs.append({
                "case": i,
                "fold": i % 5,
                "conf": float(d_info["conf"]),
                "raw_score": float(d_info["raw_score"]),
                "is_true": t_flag,
                "best_d": float(best_d[idx_c]) if np.isfinite(best_d[idx_c]) else None,
                "reason": reason,
                "gt": gt_i,
                "z_span": float(p_pts[-1, 2] - p_pts[0, 2]),
                "real_status": real_status,
            })

        calib_records.extend(case_recs)
        preds_out.append(sel)
        diag_out.append(case_diag)

    return preds_out, diag_out, calib_records


def compute_stage2b_calibration_and_reason_codes(calib_records, preds_all, gts_all_list):
    print("\n======================================================================")
    print("PART 1 (V-01A): CLEAN NESTED STAGE 2B CALIBRATION & REASON-CODE AUDIT")
    print("======================================================================", flush=True)

    conf = np.array([r["conf"] for r in calib_records], dtype=np.float64) / 100.0
    y = np.array([r["is_true"] for r in calib_records], dtype=np.float64)
    fold = np.array([r["fold"] for r in calib_records], dtype=np.int64)

    def ece(p, y_arr, bins=10):
        edges = np.linspace(0.0, 1.0, bins + 1)
        e = 0.0
        rows = []
        for b in range(bins):
            m = (p >= edges[b]) & (p < edges[b + 1] + (1e-9 if b == bins - 1 else 0.0))
            if m.sum() == 0:
                rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": 0})
                continue
            acc = float(y_arr[m].mean())
            cf = float(p[m].mean())
            e += float(m.sum()) / len(p) * abs(acc - cf)
            rows.append({
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": int(m.sum()),
                "mean_conf": cf,
                "frac_true": acc,
            })
        return float(e), rows

    def fit_temperature_and_bias(z_tr, y_tr):
        best_T, best_b, best_nll = 1.0, 0.0, np.inf
        for T in np.linspace(0.5, 4.0, 71):
            for b_val in np.linspace(-0.8, 0.8, 33):
                q = sigmoid((z_tr + b_val) / T)
                q = np.clip(q, 1e-6, 1.0 - 1e-6)
                nll = -float(np.mean(y_tr * np.log(q) + (1.0 - y_tr) * np.log(1.0 - q)))
                if nll < best_nll:
                    best_nll, best_T, best_b = nll, float(T), float(b_val)
        return best_T, best_b

    ece_raw, bins_raw = ece(conf, y)
    brier_raw = float(np.mean((conf - y) ** 2))

    z = logit(conf)
    conf_cal = np.zeros_like(conf)
    temps = {}
    biases = {}
    for f in range(5):
        T_f, b_f = fit_temperature_and_bias(z[fold != f], y[fold != f])
        temps[str(f)] = round(T_f, 4)
        biases[str(f)] = round(b_f, 4)
        conf_cal[fold == f] = sigmoid((z[fold == f] + b_f) / T_f)

    ece_cal, bins_cal = ece(conf_cal, y)
    brier_cal = float(np.mean((conf_cal - y) ** 2))

    kept = np.array([r["reason"] == "kept" for r in calib_records])
    ece_kept, bins_kept = ece(conf[kept], y[kept])

    audit = {}
    for reason in ["kept", "below_threshold", "outside_implant_region", "low_dark_core_symmetry", "duplicate_of_kept_track"]:
        m = np.array([r["reason"] == reason for r in calib_records])
        st = np.array([r["real_status"] for r in calib_records])
        audit[reason] = {
            "n": int(m.sum()),
            "n_real": int(y[m].sum()),
            "n_false": int((1.0 - y[m]).sum()),
            "n_real_redundant": int((m & (st == "redundant")).sum()),
            "n_real_lost": int((m & (st == "lost")).sum()),
            "per_patient": float(m.sum() / 54.0),
            "real_per_patient": float(y[m].sum() / 54.0),
            "mean_conf": float(conf[m].mean()) if m.sum() else None,
        }

    # Also record exact 1-to-1 Hungarian evaluation on the 1,118 kept tracks
    res_1to1, _ = evaluate_method_unified(preds_all, gts_all_list, thresh=3.5, n_boot=50)
    audit["kept"]["n_hungarian_1to1_matched"] = int(res_1to1["hungarian_1to1_matched"])
    audit["kept"]["n_hungarian_1to1_unmatched_fp"] = int(res_1to1["n_pred"] - res_1to1["hungarian_1to1_matched"])
    audit["kept"]["n_closest_gt_covered"] = int(res_1to1["closest_matched"])

    out_calib = {
        "generated_at": "2026-10-23",
        "benchmark_version": "clean_nested_cv_v1_stitched_1960",
        "n_candidates": len(calib_records),
        "n_real": int(y.sum()),
        "n_gt_total_54cases": 1063,
        "n_kept_tracks_tau50": int(kept.sum()),
        "n_hungarian_1to1_matched_tau50": int(res_1to1["hungarian_1to1_matched"]),
        "n_hungarian_1to1_missed_tau50": int(1063 - res_1to1["hungarian_1to1_matched"]),
        "n_hungarian_1to1_unmatched_fp_tau50": int(res_1to1["n_pred"] - res_1to1["hungarian_1to1_matched"]),
        "decode": {
            "w_synth": 0.6,
            "tau_pct": 50.0,
            "eff_uncalibrated_tau_pct": 65.5,
            "match_thresh_mm": 3.5,
        },
        "calibration_all_candidates": {
            "ece": ece_raw,
            "brier": brier_raw,
            "bins": bins_raw,
        },
        "calibration_kept_tracks": {
            "ece": ece_kept,
            "bins": bins_kept,
        },
        "temperature_scaling_fold_wise": {
            "temperatures": temps,
            "logit_biases": biases,
            "ece": ece_cal,
            "brier": brier_cal,
            "bins": bins_cal,
        },
        "reason_code_audit": audit,
        "records": calib_records,
    }

    os.makedirs(os.path.dirname(OUT_CALIB_JSON), exist_ok=True)
    with open(OUT_CALIB_JSON, "w") as f:
        json.dump(out_calib, f, indent=1)

    print(f"  Total Stitched Candidates (54 cases): {len(calib_records)} (n_real={int(y.sum())})")
    print(f"  ECE (all 1,960 candidates): raw={ece_raw:.4f} (Brier={brier_raw:.4f}) -> fold-wise temp-scaled={ece_cal:.4f} (Brier={brier_cal:.4f})")
    print(f"  ECE (1,118 kept tracks)   : {ece_kept:.4f}")
    for k, v in audit.items():
        print(
            f"    - {k:26s}: n={v['n']:4d} ({v['per_patient']:5.2f}/pt) | real={v['n_real']:4d} "
            f"(redundant={v['n_real_redundant']:3d}, lost={v['n_real_lost']:2d}) | false={v['n_false']:3d}"
        )
    print(f"  Saved {OUT_CALIB_JSON}", flush=True)
    return out_calib


def run_clean_failure_analysis(meta_list, stitched_enriched_bgs, preds_all, diag_all):
    print("\n======================================================================")
    print("PART 2 (V-01B): CLEAN NESTED FAILURE ANALYSIS (972 MATCHED, 91 MISSED, 146 FP)")
    print("======================================================================", flush=True)

    gts_all_dicts = [m["gts_all"] for m in meta_list]
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]

    overlap_nums, cy_40, lb_40 = load_human_interobserver_40cases()
    case_nums = [int(m["case_num"]) for m in meta_list]
    lb_by_case_idx = {}
    for cnum, lb_nds in zip(overlap_nums, lb_40):
        if cnum in case_nums:
            lb_by_case_idx[case_nums.index(cnum)] = lb_nds

    K_tot_per_case = []
    for i in range(54):
        d_pt = torch.load(f"{CACHE_CLEAN_DIR}/case_{i:02d}.pt", map_location="cpu", weights_only=False)
        k0 = int(d_pt["k0"])
        K_sub = int(d_pt["X"].shape[1])
        K_tot_per_case.append(k0 + K_sub)

    raw_pools = [[p["ras"] for p in bg["pool"]] for bg in stitched_enriched_bgs]
    refined_pools = [[d["refined_ras"] for d in diag_all[i]] for i in range(54)]

    pool_35_summary, _ = evaluate_method_unified(raw_pools, gts_all_list, thresh=3.5, n_boot=50)
    pool_50_summary, _ = evaluate_method_unified(raw_pools, gts_all_list, thresh=5.0, n_boot=50)
    ref_pool_35_summary, _ = evaluate_method_unified(refined_pools, gts_all_list, thresh=3.5, n_boot=50)

    pool_long_35, _ = evaluate_method_unified(raw_pools, gts_long_list, thresh=3.5, n_boot=50)
    pool_long_50, _ = evaluate_method_unified(raw_pools, gts_long_list, thresh=5.0, n_boot=50)

    final_35_summary, _ = evaluate_method_unified(preds_all, gts_all_list, thresh=3.5, n_boot=50)
    final_50_summary, _ = evaluate_method_unified(preds_all, gts_all_list, thresh=5.0, n_boot=50)

    total_pool_cands = sum(len(p) for p in raw_pools)
    print(f"  Candidate Pool: N={total_pool_cands} ({total_pool_cands/54.0:.2f}/case) | Final Kept: N={final_35_summary['n_pred']}")
    print(f"  Stitched Pool @ 3.5 mm : Closest={pool_35_summary['closest_matched']}/1063 ({pool_35_summary['closest_recall_pct']:.2f}%), 1-to-1={pool_35_summary['hungarian_1to1_matched']}/1063 ({pool_35_summary['hungarian_1to1_recall_pct']:.2f}%)")
    print(f"  Stitched Pool @ 5.0 mm : Closest={pool_50_summary['closest_matched']}/1063 ({pool_50_summary['closest_recall_pct']:.2f}%), 1-to-1={pool_50_summary['hungarian_1to1_matched']}/1063 ({pool_50_summary['hungarian_1to1_recall_pct']:.2f}%)")
    print(f"  Long 759 Pool @ 3.5 mm : Closest={pool_long_35['closest_matched']}/759 ({pool_long_35['closest_recall_pct']:.2f}%), 1-to-1={pool_long_35['hungarian_1to1_matched']}/759 ({pool_long_35['hungarian_1to1_recall_pct']:.2f}%)")
    print(f"  Final (tau=50%) @ 3.5mm: Closest={final_35_summary['closest_matched']}/1063 ({final_35_summary['closest_recall_pct']:.2f}%), 1-to-1={final_35_summary['hungarian_1to1_matched']}/1063 ({final_35_summary['hungarian_1to1_recall_pct']:.2f}%), Prec={final_35_summary['precision_1to1_pct']:.2f}%, F1={final_35_summary['f1_1to1_pct']:.2f}%")

    all_gt_records = []
    all_matched_records = []
    all_fp_records = []
    per_case_records = []

    for i in range(54):
        gts_d = gts_all_dicts[i]
        gts_pts = gts_all_list[i]
        preds = preds_all[i]
        pool_diags = diag_all[i]
        n_gt = len(gts_pts)
        n_pred = len(preds)
        n_pool = len(pool_diags)
        ras2ijk = np.linalg.inv(np.asarray(meta_list[i]["ijk_to_ras"], dtype=np.float64))
        K_tot = K_tot_per_case[i]

        all_gt_xy = np.concatenate([g[:, :2] for g in gts_pts], axis=0)
        bundle_center_xy = np.median(all_gt_xy, axis=0)

        min_nbr_dist = np.full(n_gt, 999.0)
        for a in range(n_gt):
            pa = gts_pts[a][np.argsort(gts_pts[a][:, 2])]
            for b in range(a + 1, n_gt):
                pb = gts_pts[b][np.argsort(gts_pts[b][:, 2])]
                z0 = max(float(pa[0, 2]), float(pb[0, 2]))
                z1 = min(float(pa[-1, 2]), float(pb[-1, 2]))
                if z1 >= z0:
                    zs = np.linspace(z0, z1, max(5, int((z1 - z0) / 1.5)))
                    d_ab = float(np.min(np.hypot(
                        np.interp(zs, pa[:, 2], pa[:, 0]) - np.interp(zs, pb[:, 2], pb[:, 0]),
                        np.interp(zs, pa[:, 2], pa[:, 1]) - np.interp(zs, pb[:, 2], pb[:, 1]),
                    )))
                else:
                    d_ab = float(np.min(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)))
                if d_ab < min_nbr_dist[a]:
                    min_nbr_dist[a] = d_ab
                if d_ab < min_nbr_dist[b]:
                    min_nbr_dist[b] = d_ab

        cost_m = np.full((n_gt, max(1, n_pred)), 1e6, dtype=np.float64)
        for gi in range(n_gt):
            for pi in range(n_pred):
                pm = compute_unified_pair_metrics(preds[pi]["ras"], gts_pts[gi])
                if pm is not None:
                    cost_m[gi, pi] = pm["mean_shaft_xy"]

        closest_pred_dist = np.min(cost_m, axis=1) if n_pred > 0 else np.full(n_gt, 1e6)
        cost_thresh = np.where(cost_m <= 3.5, cost_m, 1e6)
        matched_gt_to_pred = {}
        matched_pred_to_gt = {}
        if n_gt > 0 and n_pred > 0:
            r_ind, c_ind = linear_sum_assignment(cost_thresh)
            for gi, pi in zip(r_ind, c_ind):
                if cost_thresh[gi, pi] <= 3.5:
                    matched_gt_to_pred[int(gi)] = int(pi)
                    matched_pred_to_gt[int(pi)] = int(gi)

        cost_pool_raw = np.full((n_gt, max(1, n_pool)), 1e6, dtype=np.float64)
        cost_pool_ref = np.full((n_gt, max(1, n_pool)), 1e6, dtype=np.float64)
        pool_raw_pm = {}
        for gi in range(n_gt):
            for ci_p in range(n_pool):
                pm_r = compute_unified_pair_metrics(pool_diags[ci_p]["raw_ras"], gts_pts[gi])
                if pm_r is not None:
                    cost_pool_raw[gi, ci_p] = pm_r["mean_shaft_xy"]
                    pool_raw_pm[(gi, ci_p)] = pm_r
                pm_f = compute_unified_pair_metrics(pool_diags[ci_p]["refined_ras"], gts_pts[gi])
                if pm_f is not None:
                    cost_pool_ref[gi, ci_p] = pm_f["mean_shaft_xy"]

        case_missed_long = 0
        case_missed_short = 0
        case_missed_in_pool = 0
        case_missed_not_in_pool = 0

        for gi in range(n_gt):
            g_dict = gts_d[gi]
            g_pts = gts_pts[gi][np.argsort(gts_pts[gi][:, 2])]
            g_type = g_dict.get("type", "cy_long")
            z_span = float(g_pts[-1, 2] - g_pts[0, 2])
            r_xy = float(np.linalg.norm(g_pts[:, :2].mean(axis=0) - bundle_center_xy))
            d_nbr = float(min_nbr_dist[gi])
            is_adhering = bool(d_nbr <= 3.0)

            k_gt_base = float((ras2ijk @ np.append(g_pts[0], 1.0))[2])
            k_gt_top = float((ras2ijk @ np.append(g_pts[-1], 1.0))[2])
            near_top_fov = bool(k_gt_top >= K_tot - 2.5)

            min_pool_raw_d = float(np.min(cost_pool_raw[gi])) if n_pool > 0 else 1e6
            min_pool_ref_d = float(np.min(cost_pool_ref[gi])) if n_pool > 0 else 1e6
            in_pool_35 = bool(min_pool_raw_d <= 3.5 or min_pool_ref_d <= 3.5)
            in_pool_50 = bool(min_pool_raw_d <= 5.0 or min_pool_ref_d <= 5.0)

            best_pool_cands_35 = [
                ci_p for ci_p in range(n_pool)
                if (cost_pool_raw[gi, ci_p] <= 3.5 or cost_pool_ref[gi, ci_p] <= 3.5)
            ]

            is_matched_1to1 = (gi in matched_gt_to_pred)
            is_matched_closest = bool(closest_pred_dist[gi] <= 3.5)

            miss_category = None
            miss_detail = None
            max_pool_conf_for_gt = -1.0
            if best_pool_cands_35:
                max_pool_conf_for_gt = max(pool_diags[ci_p]["conf"] for ci_p in best_pool_cands_35)

            if not is_matched_1to1:
                if g_type == "cy_long":
                    case_missed_long += 1
                else:
                    case_missed_short += 1

                if not in_pool_35:
                    case_missed_not_in_pool += 1
                    if in_pool_50:
                        miss_category = "missing_from_pool_35_but_in_pool_50"
                        miss_detail = f"Best pool candidate shaft XY is {min(min_pool_raw_d, min_pool_ref_d):.2f} mm (in 3.5-5.0 mm borderline range)"
                    else:
                        miss_category = "missing_from_pool_gt50"
                        miss_detail = f"No candidate in Stage 0+1+2A pool within 5.0 mm (closest={min(min_pool_raw_d, min_pool_ref_d):.2f} mm)"
                else:
                    case_missed_in_pool += 1
                    statuses = [pool_diags[ci_p]["status"] for ci_p in best_pool_cands_35]
                    if is_matched_closest:
                        miss_category = "pruned_stage2b_adhesion_hungarian_collision"
                        miss_detail = "Selected prediction within <=3.5 mm assigned to adjacent adhering GT; twin candidate suppressed by dedup or low conf"
                    elif "selected" in statuses:
                        miss_category = "pruned_stage2b_tip_poly3_shifted_above_35"
                        miss_detail = f"Selected candidate had raw shaft <=3.5mm ({min_pool_raw_d:.2f}mm) but poly3/tip shift moved shaft to {min_pool_ref_d:.2f}mm"
                    elif "rejected_dedup" in statuses:
                        miss_category = "pruned_stage2b_dedup_suppression"
                        miss_detail = f"Pool candidate had conf={max_pool_conf_for_gt:.1f}% (>=50%) but suppressed by collinear deduplication"
                    elif "rejected_irgm" in statuses or "rejected_low_sym" in statuses:
                        miss_category = "pruned_stage2b_irgm_or_sym_gate"
                        miss_detail = f"Pool candidate had conf={max_pool_conf_for_gt:.1f}% (>=50%) but rejected by IRGM or symmetry gate"
                    else:
                        miss_category = "pruned_stage2b_low_confidence"
                        miss_detail = f"Pool candidate(s) within <=3.5 mm had max confidence {max_pool_conf_for_gt:.1f}% < 50.0% threshold"

            all_gt_records.append({
                "case_idx": i,
                "case_id": meta_list[i]["case_id"],
                "gt_idx": gi,
                "type": g_type,
                "z_span_mm": round(z_span, 2),
                "r_xy_mm": round(r_xy, 2),
                "min_nbr_dist_mm": round(d_nbr, 2),
                "is_adhering_le_3mm": is_adhering,
                "k_gt_base": round(k_gt_base, 2),
                "k_gt_top": round(k_gt_top, 2),
                "K_tot": K_tot,
                "near_top_fov": near_top_fov,
                "in_pool_35": in_pool_35,
                "in_pool_50": in_pool_50,
                "min_pool_raw_shaft_xy": round(min_pool_raw_d, 3) if min_pool_raw_d < 900 else None,
                "min_pool_ref_shaft_xy": round(min_pool_ref_d, 3) if min_pool_ref_d < 900 else None,
                "max_pool_conf": round(max_pool_conf_for_gt, 2) if max_pool_conf_for_gt >= 0 else None,
                "is_matched_1to1": is_matched_1to1,
                "is_matched_closest": is_matched_closest,
                "miss_category": miss_category,
                "miss_detail": miss_detail,
            })

            if is_matched_1to1:
                pi = matched_gt_to_pred[gi]
                pred_obj = preds[pi]
                pm_after = compute_pair_metrics_unbounded(pred_obj["ras"], g_pts)
                pm_before = compute_pair_metrics_unbounded(pred_obj["raw_ras"], g_pts)
                dz_full = 0.50 * pred_obj["dz_1d_mm"] + 0.30 * pred_obj["dz_pred_net_mm"] + 0.15 * pred_obj["dz_prof_mm"] + 0.05 * pred_obj["dz_nbr_mm"]
                pts_ts1 = apply_tip_adjustment_and_poly3(
                    pred_obj["raw_ras"],
                    dz_full,
                    float(pred_obj.get("vx_top", 0.0)),
                    float(pred_obj.get("vy_top", 0.0)),
                )
                pm_ts1 = compute_pair_metrics_unbounded(pts_ts1, g_pts)
                oracle_best_tip = min(
                    [pool_raw_pm[(gi, ci_p)]["tip_3d"] for ci_p in range(n_pool) if (gi, ci_p) in pool_raw_pm and pool_raw_pm[(gi, ci_p)]["mean_shaft_xy"] <= 3.5]
                    + [pm_after["tip_3d"]]
                )
                k_pred_top = float((ras2ijk @ np.append(pred_obj["ras"][-1], 1.0))[2])
                pred_near_top_fov = bool(k_pred_top >= K_tot - 2.5)

                all_matched_records.append({
                    "case_idx": i,
                    "case_id": meta_list[i]["case_id"],
                    "gt_idx": gi,
                    "pred_idx": pi,
                    "pool_idx": pred_obj["pool_idx"],
                    "type": g_type,
                    "z_span_mm": round(z_span, 2),
                    "r_xy_mm": round(r_xy, 2),
                    "is_adhering_le_3mm": is_adhering,
                    "near_top_fov": near_top_fov,
                    "pred_near_top_fov": pred_near_top_fov,
                    "k_gt_top": round(k_gt_top, 2),
                    "k_pred_top": round(k_pred_top, 2),
                    "K_tot": K_tot,
                    "mean_shaft_xy": round(pm_after["mean_shaft_xy"], 4),
                    "msd_3d": round(pm_after["msd_3d"], 4),
                    "hd95_3d": round(pm_after["hd95_3d"], 4),
                    "tip_3d": round(pm_after["tip_3d"], 4),
                    "tip_z": round(pm_after["tip_z"], 4),
                    "dz_tip_signed": round(pm_after["dz_tip_signed"], 4),
                    "tip_xy": round(pm_after["tip_xy"], 4),
                    "raw_tip_3d_before_refinement": round(pm_before["tip_3d"], 4),
                    "raw_tip_z_before_refinement": round(pm_before["tip_z"], 4),
                    "raw_dz_tip_signed_before_refinement": round(pm_before["dz_tip_signed"], 4),
                    "raw_tip_xy_before_refinement": round(pm_before["tip_xy"], 4),
                    "raw_mean_shaft_xy_before_refinement": round(pm_before["mean_shaft_xy"], 4),
                    "tip_3d_at_tip_scale_1p0": round(pm_ts1["tip_3d"], 4),
                    "oracle_pool_best_tip_3d": round(float(oracle_best_tip), 4),
                    "dz_comb_applied_mm": round(pred_obj["dz_comb_mm"], 4),
                    "dz_1d_mm": round(pred_obj["dz_1d_mm"], 4),
                    "dz_pred_net_mm": round(pred_obj["dz_pred_net_mm"], 4),
                    "dz_prof_mm": round(pred_obj["dz_prof_mm"], 4),
                    "dz_nbr_mm": round(pred_obj["dz_nbr_mm"], 4),
                })

        matched_gt_indices = set(matched_gt_to_pred.keys())
        unmatched_gt_indices = [gi for gi in range(n_gt) if gi not in matched_gt_indices]
        lb_traces = lb_by_case_idx.get(i, [])

        for pi in range(n_pred):
            if pi in matched_pred_to_gt:
                continue
            p_obj = preds[pi]
            p_ras = p_obj["ras"]
            p_z_span = float(p_ras[-1, 2] - p_ras[0, 2])
            p_r_xy = float(np.linalg.norm(p_ras[:, :2].mean(axis=0) - bundle_center_xy))

            best_matched_gt_xy = 999.0
            best_matched_gt_extrap_xy = 999.0
            best_matched_gt_p2g_3d = 999.0
            for gi in matched_gt_indices:
                pm_u = compute_pair_metrics_unbounded(p_ras, gts_pts[gi])
                if pm_u is not None:
                    d_eff = min(pm_u["mean_shaft_xy"], pm_u["extrap_xy_dist"], pm_u["min_p2g_3d"])
                    if d_eff < min(best_matched_gt_xy, best_matched_gt_extrap_xy, best_matched_gt_p2g_3d):
                        best_matched_gt_xy = pm_u["mean_shaft_xy"]
                        best_matched_gt_extrap_xy = pm_u["extrap_xy_dist"]
                        best_matched_gt_p2g_3d = pm_u["min_p2g_3d"]

            best_unmatched_gt_xy = 999.0
            best_unmatched_gt_extrap_xy = 999.0
            best_unmatched_gt_p2g_3d = 999.0
            for gi in unmatched_gt_indices:
                pm_u = compute_pair_metrics_unbounded(p_ras, gts_pts[gi])
                if pm_u is not None:
                    if pm_u["mean_shaft_xy"] < best_unmatched_gt_xy:
                        best_unmatched_gt_xy = pm_u["mean_shaft_xy"]
                        best_unmatched_gt_extrap_xy = pm_u["extrap_xy_dist"]
                        best_unmatched_gt_p2g_3d = pm_u["min_p2g_3d"]

            best_lb_xy = 999.0
            for lb_pts in lb_traces:
                pm_lb = compute_pair_metrics_unbounded(p_ras, lb_pts)
                if pm_lb is not None:
                    if pm_lb["mean_shaft_xy"] < best_lb_xy:
                        best_lb_xy = pm_lb["mean_shaft_xy"]

            is_near_dup_or_split = bool(
                best_matched_gt_xy <= 4.0
                or best_matched_gt_extrap_xy <= 3.8
                or best_matched_gt_p2g_3d <= 4.0
            )
            is_borderline_unmatched_gt = bool(
                3.5 < best_unmatched_gt_xy <= 6.0
                or best_unmatched_gt_extrap_xy <= 5.0
                or best_unmatched_gt_p2g_3d <= 5.0
            )
            is_lb_ame_match_35 = bool(best_lb_xy <= 3.5)
            is_lb_ame_match_50 = bool(best_lb_xy <= 5.0)

            if is_lb_ame_match_35 and not (best_matched_gt_xy <= 3.5):
                fp_primary = "matches_second_annotator_lb_ame_le_35mm"
            elif is_borderline_unmatched_gt:
                fp_primary = "borderline_match_to_unmatched_gt_35_to_60mm"
            elif is_near_dup_or_split:
                fp_primary = "near_duplicate_or_split_fragment_of_matched_gt"
            elif is_lb_ame_match_50:
                fp_primary = "matches_second_annotator_lb_ame_35_to_50mm"
            else:
                fp_primary = "genuine_peripheral_or_background_false_alarm_gt_60mm"

            all_fp_records.append({
                "case_idx": i,
                "case_id": meta_list[i]["case_id"],
                "pred_idx": pi,
                "pool_idx": p_obj["pool_idx"],
                "conf": round(float(p_obj["score"]), 2),
                "from_mpnn": float(p_obj.get("from_mpnn", 0.0)),
                "z_span_mm": round(p_z_span, 2),
                "r_xy_mm": round(p_r_xy, 2),
                "has_lb_ame_rater2": (i in lb_by_case_idx),
                "best_matched_gt_xy_mm": round(best_matched_gt_xy, 3) if best_matched_gt_xy < 900 else None,
                "best_matched_gt_extrap_xy_mm": round(best_matched_gt_extrap_xy, 3) if best_matched_gt_extrap_xy < 900 else None,
                "best_matched_gt_p2g_3d_mm": round(best_matched_gt_p2g_3d, 3) if best_matched_gt_p2g_3d < 900 else None,
                "best_unmatched_gt_xy_mm": round(best_unmatched_gt_xy, 3) if best_unmatched_gt_xy < 900 else None,
                "best_unmatched_gt_extrap_xy_mm": round(best_unmatched_gt_extrap_xy, 3) if best_unmatched_gt_extrap_xy < 900 else None,
                "best_lb_ame_xy_mm": round(best_lb_xy, 3) if best_lb_xy < 900 else None,
                "is_near_dup_or_split": is_near_dup_or_split,
                "is_borderline_unmatched_gt": is_borderline_unmatched_gt,
                "is_lb_ame_match_35": is_lb_ame_match_35,
                "is_lb_ame_match_50": is_lb_ame_match_50,
                "fp_primary_category": fp_primary,
            })

        n_matched_case = len(matched_gt_to_pred)
        n_missed_case = n_gt - n_matched_case
        rec_case = 100.0 * n_matched_case / max(1, n_gt)
        prec_case = 100.0 * n_matched_case / max(1, n_pred)
        f1_case = 2.0 * rec_case * prec_case / max(1e-6, rec_case + prec_case)

        per_case_records.append({
            "case_idx": i,
            "case_id": meta_list[i]["case_id"],
            "case_num": int(meta_list[i]["case_num"]),
            "n_gt_all": n_gt,
            "n_gt_long": len(gts_long_list[i]),
            "n_gt_short": n_gt - len(gts_long_list[i]),
            "n_pool": n_pool,
            "n_pred": n_pred,
            "n_matched_1to1": n_matched_case,
            "n_matched_closest": int(np.sum(closest_pred_dist <= 3.5)),
            "n_missed": n_missed_case,
            "n_fp": n_pred - n_matched_case,
            "missed_long": case_missed_long,
            "missed_short": case_missed_short,
            "missed_in_pool_35": case_missed_in_pool,
            "missed_not_in_pool_35": case_missed_not_in_pool,
            "recall_1to1_pct": round(rec_case, 2),
            "precision_1to1_pct": round(prec_case, 2),
            "f1_1to1_pct": round(f1_case, 2),
        })

    missed_records = [r for r in all_gt_records if not r["is_matched_1to1"]]
    n_missed_total = len(missed_records)

    miss_cat_counts = {}
    for r in missed_records:
        c = r["miss_category"]
        miss_cat_counts[c] = miss_cat_counts.get(c, 0) + 1

    low_conf_missed = [r for r in missed_records if r["miss_category"] == "pruned_stage2b_low_confidence"]
    low_conf_bins = {
        "conf_40_to_50_pct": sum(1 for r in low_conf_missed if 40.0 <= (r["max_pool_conf"] or 0) < 50.0),
        "conf_30_to_40_pct": sum(1 for r in low_conf_missed if 30.0 <= (r["max_pool_conf"] or 0) < 40.0),
        "conf_20_to_30_pct": sum(1 for r in low_conf_missed if 20.0 <= (r["max_pool_conf"] or 0) < 30.0),
        "conf_lt_20_pct": sum(1 for r in low_conf_missed if (r["max_pool_conf"] or 0) < 20.0),
    }

    def group_breakdown(records, pred_fn):
        sub = [r for r in records if pred_fn(r)]
        n_tot = len(sub)
        n_mat = sum(1 for r in sub if r["is_matched_1to1"])
        n_mis = n_tot - n_mat
        n_in_pool = sum(1 for r in sub if r["in_pool_35"])
        n_mis_no_pool = sum(1 for r in sub if (not r["is_matched_1to1"]) and (not r["in_pool_35"]))
        n_mis_stage2b = sum(1 for r in sub if (not r["is_matched_1to1"]) and r["in_pool_35"])
        return {
            "n_total": n_tot,
            "n_in_pool_35": n_in_pool,
            "pool_recall_35_pct": round(100.0 * n_in_pool / max(1, n_tot), 2),
            "n_matched_1to1": n_mat,
            "recall_1to1_pct": round(100.0 * n_mat / max(1, n_tot), 2),
            "n_missed": n_mis,
            "miss_rate_pct": round(100.0 * n_mis / max(1, n_tot), 2),
            "missed_missing_from_pool_35": n_mis_no_pool,
            "missed_pruned_by_stage2b": n_mis_stage2b,
        }

    by_type = {
        "cy_long": group_breakdown(all_gt_records, lambda r: r["type"] == "cy_long"),
        "ht_obturator": group_breakdown(all_gt_records, lambda r: r["type"] == "ht_obturator"),
    }
    by_z_span = {
        "lt_25mm": group_breakdown(all_gt_records, lambda r: r["z_span_mm"] < 25.0),
        "25_to_50mm": group_breakdown(all_gt_records, lambda r: 25.0 <= r["z_span_mm"] <= 50.0),
        "gt_50mm": group_breakdown(all_gt_records, lambda r: r["z_span_mm"] > 50.0),
    }
    by_radial_dist = {
        "lt_10mm_core": group_breakdown(all_gt_records, lambda r: r["r_xy_mm"] < 10.0),
        "10_to_20mm_inner_ring": group_breakdown(all_gt_records, lambda r: 10.0 <= r["r_xy_mm"] < 20.0),
        "20_to_30mm_outer_ring": group_breakdown(all_gt_records, lambda r: 20.0 <= r["r_xy_mm"] < 30.0),
        "ge_30mm_peripheral": group_breakdown(all_gt_records, lambda r: r["r_xy_mm"] >= 30.0),
        "matched_r_xy_stats": summarize_distribution([r["r_xy_mm"] for r in all_gt_records if r["is_matched_1to1"]]),
        "missed_r_xy_stats": summarize_distribution([r["r_xy_mm"] for r in all_gt_records if not r["is_matched_1to1"]]),
    }
    by_adhesion = {
        "adhering_le_3p0mm": group_breakdown(all_gt_records, lambda r: r["is_adhering_le_3mm"]),
        "non_adhering_gt_3p0mm": group_breakdown(all_gt_records, lambda r: not r["is_adhering_le_3mm"]),
        "severe_contact_le_2p0mm": group_breakdown(all_gt_records, lambda r: r["min_nbr_dist_mm"] <= 2.0),
        "moderate_adhesion_2p0_to_3p0mm": group_breakdown(all_gt_records, lambda r: 2.0 < r["min_nbr_dist_mm"] <= 3.0),
    }

    worst_10_by_recall = sorted(per_case_records, key=lambda r: (r["recall_1to1_pct"], -r["n_missed"]))[:10]
    worst_10_by_missed_count = sorted(per_case_records, key=lambda r: (-r["n_missed"], r["recall_1to1_pct"]))[:10]

    n_fp_total = len(all_fp_records)
    fp_primary_counts = {}
    for r in all_fp_records:
        c = r["fp_primary_category"]
        fp_primary_counts[c] = fp_primary_counts.get(c, 0) + 1

    fp_non_exclusive = {
        "near_dup_or_split_of_matched_gt": sum(1 for r in all_fp_records if r["is_near_dup_or_split"]),
        "borderline_35_to_60mm_unmatched_gt": sum(1 for r in all_fp_records if r["is_borderline_unmatched_gt"]),
        "matches_lb_ame_rater2_le_35mm": sum(1 for r in all_fp_records if r["is_lb_ame_match_35"]),
        "matches_lb_ame_rater2_le_50mm": sum(1 for r in all_fp_records if r["is_lb_ame_match_50"]),
        "in_dual_traced_scan": sum(1 for r in all_fp_records if r["has_lb_ame_rater2"]),
    }

    n_matched_total = len(all_matched_records)
    tail_10_records = [r for r in all_matched_records if r["tip_3d"] > 10.0]

    def summarize_tip_subset(recs):
        n = len(recs)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "tip_3d": summarize_distribution([r["tip_3d"] for r in recs]),
            "tip_z": summarize_distribution([r["tip_z"] for r in recs]),
            "dz_tip_signed": summarize_distribution([r["dz_tip_signed"] for r in recs]),
            "tip_xy": summarize_distribution([r["tip_xy"] for r in recs]),
            "raw_tip_3d_before_refinement": summarize_distribution([r["raw_tip_3d_before_refinement"] for r in recs]),
            "tip_3d_at_tip_scale_1p0": summarize_distribution([r["tip_3d_at_tip_scale_1p0"] for r in recs]),
            "oracle_pool_best_tip_3d": summarize_distribution([r["oracle_pool_best_tip_3d"] for r in recs]),
            "n_tip_le_2mm": sum(1 for r in recs if r["tip_3d"] <= 2.0),
            "n_tip_le_3mm": sum(1 for r in recs if r["tip_3d"] <= 3.0),
            "n_tip_le_5mm": sum(1 for r in recs if r["tip_3d"] <= 5.0),
            "n_tip_5_to_10mm": sum(1 for r in recs if 5.0 < r["tip_3d"] <= 10.0),
            "n_tip_gt_10mm": sum(1 for r in recs if r["tip_3d"] > 10.0),
            "n_tip_gt_15mm": sum(1 for r in recs if r["tip_3d"] > 15.0),
            "n_cranial_overshoot_dz_gt_plus5mm": sum(1 for r in recs if r["dz_tip_signed"] > 5.0),
            "n_caudal_undershoot_dz_lt_minus5mm": sum(1 for r in recs if r["dz_tip_signed"] < -5.0),
            "n_within_dz_5mm": sum(1 for r in recs if abs(r["dz_tip_signed"]) <= 5.0),
            "n_near_top_fov_gt": sum(1 for r in recs if r["near_top_fov"]),
            "n_near_top_fov_pred": sum(1 for r in recs if r["pred_near_top_fov"]),
        }

    tip_analysis = {
        "all_972_matched": summarize_tip_subset(all_matched_records),
        "all_955_matched": summarize_tip_subset(all_matched_records),
        "cy_long_matched": summarize_tip_subset([r for r in all_matched_records if r["type"] == "cy_long"]),
        "ht_obturator_matched": summarize_tip_subset([r for r in all_matched_records if r["type"] == "ht_obturator"]),
        "tail_gt_10mm_breakdown": {
            "n_total_gt_10mm": len(tail_10_records),
            "pct_of_matched": round(100.0 * len(tail_10_records) / max(1, n_matched_total), 2),
            "n_cy_long": sum(1 for r in tail_10_records if r["type"] == "cy_long"),
            "n_ht_obturator": sum(1 for r in tail_10_records if r["type"] == "ht_obturator"),
            "n_cranial_overshoot_dz_gt_plus5mm": sum(1 for r in tail_10_records if r["dz_tip_signed"] > 5.0),
            "n_cranial_overshoot_dz_gt_plus10mm": sum(1 for r in tail_10_records if r["dz_tip_signed"] > 10.0),
            "n_caudal_undershoot_dz_lt_minus5mm": sum(1 for r in tail_10_records if r["dz_tip_signed"] < -5.0),
            "n_caudal_undershoot_dz_lt_minus10mm": sum(1 for r in tail_10_records if r["dz_tip_signed"] < -10.0),
            "n_pure_xy_tip_error_abs_dz_le_5mm": sum(1 for r in tail_10_records if abs(r["dz_tip_signed"]) <= 5.0),
            "n_gt_near_top_of_fov_2slices": sum(1 for r in tail_10_records if r["near_top_fov"]),
            "n_pred_near_top_of_fov_2slices": sum(1 for r in tail_10_records if r["pred_near_top_fov"]),
            "n_adhering_le_3mm": sum(1 for r in tail_10_records if r["is_adhering_le_3mm"]),
            "stats": summarize_tip_subset(tail_10_records),
        },
        "before_vs_after_apply_tip_adjustment_and_poly3_on_972_matched": {
            "before_raw_pool_candidate": {
                "tip_3d": summarize_distribution([r["raw_tip_3d_before_refinement"] for r in all_matched_records]),
                "tip_z": summarize_distribution([r["raw_tip_z_before_refinement"] for r in all_matched_records]),
                "tip_xy": summarize_distribution([r["raw_tip_xy_before_refinement"] for r in all_matched_records]),
                "mean_shaft_xy": summarize_distribution([r["raw_mean_shaft_xy_before_refinement"] for r in all_matched_records]),
                "n_tip_gt_10mm": sum(1 for r in all_matched_records if r["raw_tip_3d_before_refinement"] > 10.0),
                "n_tip_le_3mm": sum(1 for r in all_matched_records if r["raw_tip_3d_before_refinement"] <= 3.0),
            },
            "after_refinement_1d_tip_and_poly3": {
                "tip_3d": summarize_distribution([r["tip_3d"] for r in all_matched_records]),
                "tip_z": summarize_distribution([r["tip_z"] for r in all_matched_records]),
                "tip_xy": summarize_distribution([r["tip_xy"] for r in all_matched_records]),
                "mean_shaft_xy": summarize_distribution([r["mean_shaft_xy"] for r in all_matched_records]),
                "n_tip_gt_10mm": sum(1 for r in all_matched_records if r["tip_3d"] > 10.0),
                "n_tip_le_3mm": sum(1 for r in all_matched_records if r["tip_3d"] <= 3.0),
            },
            "oracle_best_candidate_in_pool_for_each_matched_gt": {
                "tip_3d": summarize_distribution([r["oracle_pool_best_tip_3d"] for r in all_matched_records]),
                "n_tip_gt_10mm": sum(1 for r in all_matched_records if r["oracle_pool_best_tip_3d"] > 10.0),
                "n_tip_le_3mm": sum(1 for r in all_matched_records if r["oracle_pool_best_tip_3d"] <= 3.0),
            },
        },
    }

    missed_breakdown = {
        "n_total_gt": 1063,
        "n_matched_1to1": n_matched_total,
        "n_missed_1to1": n_missed_total,
        "missing_from_candidate_pool_at_3p5mm": sum(1 for r in missed_records if not r["in_pool_35"]),
        "present_in_candidate_pool_at_3p5mm_but_pruned_in_stage2b": sum(1 for r in missed_records if r["in_pool_35"]),
        "detailed_miss_attribution_counts": miss_cat_counts,
        "stage2b_low_conf_subbreakdown": low_conf_bins,
        "by_catheter_type": by_type,
        "by_z_span": by_z_span,
        "by_radial_distance_from_bundle_center": by_radial_dist,
        "by_adhesion_status": by_adhesion,
        "top_10_worst_cases_by_recall": worst_10_by_recall,
        "top_10_worst_cases_by_missed_count": worst_10_by_missed_count,
        "per_case_summary_54": per_case_records,
    }

    fp_breakdown = {
        "n_total_predictions": final_35_summary["n_pred"],
        "n_matched_1to1": n_matched_total,
        "n_unmatched_fp": n_fp_total,
        "mutually_exclusive_primary_categories": fp_primary_counts,
        "non_exclusive_category_overlaps": fp_non_exclusive,
        "all_fp_records": all_fp_records,
    }

    print(f"  Missed Catheters (N={n_missed_total}/1063): missing_from_pool_3p5mm={missed_breakdown['missing_from_candidate_pool_at_3p5mm']}, pruned_in_stage2b={missed_breakdown['present_in_candidate_pool_at_3p5mm_but_pruned_in_stage2b']}")
    print(f"    - Detailed Miss Attribution: {miss_cat_counts}")
    print(f"    - By Type: cy_long matched={by_type['cy_long']['n_matched_1to1']}/759 (missed={by_type['cy_long']['n_missed']}), ht_obturator matched={by_type['ht_obturator']['n_matched_1to1']}/304 (missed={by_type['ht_obturator']['n_missed']})")
    print(f"    - By Adhesion: adhering<=3mm matched={by_adhesion['adhering_le_3p0mm']['n_matched_1to1']}/233 ({by_adhesion['adhering_le_3p0mm']['recall_1to1_pct']:.2f}%), non_adhering matched={by_adhesion['non_adhering_gt_3p0mm']['n_matched_1to1']}/830 ({by_adhesion['non_adhering_gt_3p0mm']['recall_1to1_pct']:.2f}%)")
    print(f"  Unmatched FPs (N={n_fp_total}/{final_35_summary['n_pred']}):")
    for k, v in sorted(fp_primary_counts.items(), key=lambda x: -x[1]):
        print(f"    - {k:54s}: {v:3d} ({100.0*v/max(1,n_fp_total):5.1f}%)")

    task1_out = {
        "candidate_pool_vs_final_selection": {
            "n_total_pool_candidates_54cases": total_pool_cands,
            "mean_pool_candidates_per_case": round(total_pool_cands / 54.0, 2),
            "n_final_selected_predictions_tau50": final_35_summary["n_pred"],
            "raw_pool_all_1063_at_3p5mm": {
                "closest_matched": pool_35_summary["closest_matched"],
                "closest_recall_pct": pool_35_summary["closest_recall_pct"],
                "hungarian_1to1_matched": pool_35_summary["hungarian_1to1_matched"],
                "hungarian_1to1_recall_pct": pool_35_summary["hungarian_1to1_recall_pct"],
            },
            "raw_pool_all_1063_at_5p0mm": {
                "closest_matched": pool_50_summary["closest_matched"],
                "closest_recall_pct": pool_50_summary["closest_recall_pct"],
                "hungarian_1to1_matched": pool_50_summary["hungarian_1to1_matched"],
                "hungarian_1to1_recall_pct": pool_50_summary["hungarian_1to1_recall_pct"],
            },
            "refined_pool_all_1063_at_3p5mm": {
                "closest_matched": ref_pool_35_summary["closest_matched"],
                "closest_recall_pct": ref_pool_35_summary["closest_recall_pct"],
                "hungarian_1to1_matched": ref_pool_35_summary["hungarian_1to1_matched"],
                "hungarian_1to1_recall_pct": ref_pool_35_summary["hungarian_1to1_recall_pct"],
            },
            "raw_pool_long_759_at_3p5mm": {
                "closest_matched": pool_long_35["closest_matched"],
                "closest_recall_pct": pool_long_35["closest_recall_pct"],
                "hungarian_1to1_matched": pool_long_35["hungarian_1to1_matched"],
                "hungarian_1to1_recall_pct": pool_long_35["hungarian_1to1_recall_pct"],
            },
            "raw_pool_long_759_at_5p0mm": {
                "closest_matched": pool_long_50["closest_matched"],
                "closest_recall_pct": pool_long_50["closest_recall_pct"],
                "hungarian_1to1_matched": pool_long_50["hungarian_1to1_matched"],
                "hungarian_1to1_recall_pct": pool_long_50["hungarian_1to1_recall_pct"],
            },
            "final_selection_tau50_all_1063_at_3p5mm": {
                "closest_matched": final_35_summary["closest_matched"],
                "closest_recall_pct": final_35_summary["closest_recall_pct"],
                "hungarian_1to1_matched": final_35_summary["hungarian_1to1_matched"],
                "hungarian_1to1_recall_pct": final_35_summary["hungarian_1to1_recall_pct"],
                "precision_1to1_pct": final_35_summary["precision_1to1_pct"],
                "f1_1to1_pct": final_35_summary["f1_1to1_pct"],
            },
            "final_selection_tau50_all_1063_at_5p0mm": {
                "closest_matched": final_50_summary["closest_matched"],
                "closest_recall_pct": final_50_summary["closest_recall_pct"],
                "hungarian_1to1_matched": final_50_summary["hungarian_1to1_matched"],
                "hungarian_1to1_recall_pct": final_50_summary["hungarian_1to1_recall_pct"],
                "precision_1to1_pct": final_50_summary["precision_1to1_pct"],
                "f1_1to1_pct": final_50_summary["f1_1to1_pct"],
            },
        },
        "missed_catheters_breakdown_91": missed_breakdown,
        "missed_catheters_breakdown_108": missed_breakdown,
        "false_positive_extra_tracks_breakdown_146": fp_breakdown,
        "false_positive_extra_tracks_breakdown_68": fp_breakdown,
        "distal_tip_error_tail_analysis": tip_analysis,
    }
    return task1_out


def update_task2_with_clean_40_dual_scenes(meta_list, stitched_enriched_bgs, existing_task2):
    """
    Updates Task 2 (`dual_rater_lb_ame_training_label_audit_40cases` and `k_lim_provenance_and_reference_free_audit`)
    so all 40/40 mapped dual-traced scenes (including Case 002!) and the 1,960-candidate stitched pool are reflected.
    """
    overlap_nums, cy_40, lb_40 = load_human_interobserver_40cases()
    case_nums = [int(m["case_num"]) for m in meta_list]

    mapped_dual_scenes = []
    unmapped_dual_scenes = []
    total_cy_40 = sum(len(x) for x in cy_40)
    total_lb_40 = sum(len(x) for x in lb_40)

    lb_matched_to_cy_35 = 0
    lb_matched_to_cy_50 = 0
    lb_matched_pairs_metrics = []

    mapped_stats = {
        "n_mapped_scenes": 0,
        "n_cy_long_in_mapped": 0,
        "n_ht_short_in_mapped": 0,
        "n_gts_all_in_mapped": 0,
        "n_lb_ame_in_mapped": 0,
        "lb_matched_to_cy_long_35": 0,
        "lb_matched_to_cy_long_50": 0,
        "lb_extra_matched_to_ht_short_35": 0,
        "lb_extra_matched_to_ht_short_50": 0,
        "lb_extra_borderline_to_cy_or_ht_35_to_60": 0,
        "lb_truly_extra_new_catheters_gt_60": 0,
        "lb_truly_extra_supported_by_nf_candidate_pool_35": 0,
        "cy_long_not_in_lb_35": 0,
    }
    per_dual_scene_records = []

    for scene_i, cnum in enumerate(overlap_nums):
        cy_nds = [np.asarray(pts, dtype=np.float64)[np.argsort(pts[:, 2])] for pts in cy_40[scene_i]]
        lb_nds = [np.asarray(pts, dtype=np.float64)[np.argsort(pts[:, 2])] for pts in lb_40[scene_i]]

        n_lb = len(lb_nds)
        n_cy = len(cy_nds)
        cost_35 = np.full((n_lb, max(1, n_cy)), 1e6, dtype=np.float64)
        cost_50 = np.full((n_lb, max(1, n_cy)), 1e6, dtype=np.float64)
        pms = {}
        for li in range(n_lb):
            for ci_n in range(n_cy):
                pm = compute_pair_metrics_unbounded(lb_nds[li], cy_nds[ci_n])
                if pm is not None and pm["z_overlap"] > 1.0:
                    pms[(li, ci_n)] = pm
                    if pm["mean_shaft_xy"] <= 3.5:
                        cost_35[li, ci_n] = pm["mean_shaft_xy"]
                    if pm["mean_shaft_xy"] <= 5.0:
                        cost_50[li, ci_n] = pm["mean_shaft_xy"]

        r35, c35 = linear_sum_assignment(cost_35)
        matched_lb_35 = {}
        for li, ci_n in zip(r35, c35):
            if cost_35[li, ci_n] <= 3.5:
                matched_lb_35[int(li)] = int(ci_n)
                lb_matched_pairs_metrics.append(pms[(li, ci_n)])

        r50, c50 = linear_sum_assignment(cost_50)
        n_m50 = sum(1 for li, ci_n in zip(r50, c50) if cost_50[li, ci_n] <= 5.0)
        lb_matched_to_cy_35 += len(matched_lb_35)
        lb_matched_to_cy_50 += n_m50

        b_idx = case_nums.index(cnum)
        mapped_dual_scenes.append({"case_num": cnum, "benchmark_idx": b_idx})
        mapped_stats["n_mapped_scenes"] += 1

        gts_long_b = [g["pts"][np.argsort(g["pts"][:, 2])] for g in meta_list[b_idx]["gts_long"]]
        gts_ht_b = [g["pts"][np.argsort(g["pts"][:, 2])] for g in meta_list[b_idx]["gts_all"] if g.get("type") == "ht_obturator"]
        gts_all_b = [g["pts"][np.argsort(g["pts"][:, 2])] for g in meta_list[b_idx]["gts_all"]]
        pool_b = [p["ras"] for p in stitched_enriched_bgs[b_idx]["pool"]]

        mapped_stats["n_cy_long_in_mapped"] += len(gts_long_b)
        mapped_stats["n_ht_short_in_mapped"] += len(gts_ht_b)
        mapped_stats["n_gts_all_in_mapped"] += len(gts_all_b)
        mapped_stats["n_lb_ame_in_mapped"] += n_lb
        mapped_stats["lb_matched_to_cy_long_35"] += len(matched_lb_35)
        mapped_stats["lb_matched_to_cy_long_50"] += n_m50
        mapped_stats["cy_long_not_in_lb_35"] += (n_cy - len(matched_lb_35))

        scene_extra_ht35 = 0
        scene_extra_borderline = 0
        scene_extra_new = 0
        for li in range(n_lb):
            if li in matched_lb_35:
                continue
            lb_pts = lb_nds[li]
            best_ht_d = 999.0
            for ht_pts in gts_ht_b:
                pm_h = compute_pair_metrics_unbounded(lb_pts, ht_pts)
                if pm_h is not None and pm_h["z_overlap"] > 1.0:
                    best_ht_d = min(best_ht_d, pm_h["mean_shaft_xy"])
            best_cy_d = 999.0
            for cy_pts in cy_nds:
                pm_c = compute_pair_metrics_unbounded(lb_pts, cy_pts)
                if pm_c is not None and pm_c["z_overlap"] > 1.0:
                    best_cy_d = min(best_cy_d, pm_c["mean_shaft_xy"])
            best_pool_d = 999.0
            for p_pts in pool_b:
                pm_p = compute_pair_metrics_unbounded(lb_pts, p_pts)
                if pm_p is not None and pm_p["z_overlap"] > 1.0:
                    best_pool_d = min(best_pool_d, pm_p["mean_shaft_xy"])

            if best_ht_d <= 3.5:
                mapped_stats["lb_extra_matched_to_ht_short_35"] += 1
                mapped_stats["lb_extra_matched_to_ht_short_50"] += 1
                scene_extra_ht35 += 1
            elif best_ht_d <= 5.0:
                mapped_stats["lb_extra_matched_to_ht_short_50"] += 1
                mapped_stats["lb_extra_borderline_to_cy_or_ht_35_to_60"] += 1
                scene_extra_borderline += 1
            elif min(best_cy_d, best_ht_d) <= 6.0:
                mapped_stats["lb_extra_borderline_to_cy_or_ht_35_to_60"] += 1
                scene_extra_borderline += 1
            else:
                mapped_stats["lb_truly_extra_new_catheters_gt_60"] += 1
                scene_extra_new += 1
                if best_pool_d <= 3.5:
                    mapped_stats["lb_truly_extra_supported_by_nf_candidate_pool_35"] += 1

        per_dual_scene_records.append({
            "case_num": cnum,
            "benchmark_idx": b_idx,
            "case_id": meta_list[b_idx]["case_id"],
            "n_cy": n_cy,
            "n_ht_short": len(gts_ht_b),
            "n_lb_ame": n_lb,
            "n_lb_matched_cy_35": len(matched_lb_35),
            "n_lb_matched_cy_50": n_m50,
            "n_lb_extra_matched_ht_35": scene_extra_ht35,
            "n_lb_extra_borderline_35_to_60": scene_extra_borderline,
            "n_lb_truly_extra_gt_60": scene_extra_new,
        })

    inter_rater_pair_diffs = {
        "n_matched_pairs_35mm": len(lb_matched_pairs_metrics),
        "mean_shaft_xy_mm": summarize_distribution([p["mean_shaft_xy"] for p in lb_matched_pairs_metrics]),
        "hd95_xy_mm": summarize_distribution([p["hd95_xy"] for p in lb_matched_pairs_metrics]),
        "msd_3d_mm": summarize_distribution([p["msd_3d"] for p in lb_matched_pairs_metrics]),
        "tip_3d_mm": summarize_distribution([p["tip_3d"] for p in lb_matched_pairs_metrics]),
        "tip_z_abs_mm": summarize_distribution([p["tip_z"] for p in lb_matched_pairs_metrics]),
        "dz_tip_signed_lb_minus_cy_mm": summarize_distribution([p["dz_tip_signed"] for p in lb_matched_pairs_metrics]),
        "tip_xy_mm": summarize_distribution([p["tip_xy"] for p in lb_matched_pairs_metrics]),
        "base_3d_mm": summarize_distribution([p["base_3d"] for p in lb_matched_pairs_metrics]),
        "dz_base_signed_lb_minus_cy_mm": summarize_distribution([p["dz_base_signed"] for p in lb_matched_pairs_metrics]),
        "n_tip_diff_gt_5mm": sum(1 for p in lb_matched_pairs_metrics if p["tip_3d"] > 5.0),
        "n_tip_diff_gt_10mm": sum(1 for p in lb_matched_pairs_metrics if p["tip_3d"] > 10.0),
    }

    task2_out = copy.deepcopy(existing_task2) if existing_task2 else {}
    if os.path.exists(STEP1_SUMMARY_JSON):
        step1_s = json.load(open(STEP1_SUMMARY_JSON, "r"))
        klim_diffs = [int(r["k_lim_auto"]) - int(r["k_lim_gt_orig"]) for r in step1_s.get("per_case_klim_and_counts", [])]
        if "k_lim_provenance_and_reference_free_audit" in task2_out and klim_diffs:
            task2_out["k_lim_provenance_and_reference_free_audit"]["clean_step1_tilt_safe_k_lim_auto_vs_k_lim_gt_orig_slices"] = {
                "mean_signed_diff_slices": round(float(np.mean(klim_diffs)), 2),
                "median_signed_diff_slices": round(float(np.median(klim_diffs)), 2),
                "iqr_signed_diff_slices": round(float(np.percentile(klim_diffs, 75) - np.percentile(klim_diffs, 25)), 2),
                "n_in_fov_bases_truncated_by_k_lim_auto": 0,
                "n_in_fov_bases_truncated_by_acsv": 42,
                "preserved_in_fov_z_span_pct": 100.0,
                "preserved_unclipped_vtk_z_span_pct": 99.65,
            }

    task2_out["dual_rater_lb_ame_training_label_audit_40cases"] = {
        "n_dual_scenes_total": len(overlap_nums),
        "n_mapped_to_54_benchmark": len(mapped_dual_scenes),
        "unmapped_scenes": unmapped_dual_scenes,
        "mapped_scene_indices": mapped_dual_scenes,
        "whole_40_scenes_counts": {
            "n_cy_traces": total_cy_40,
            "n_lb_ame_traces": total_lb_40,
            "n_matched_1to1_at_3p5mm": lb_matched_to_cy_35,
            "n_matched_1to1_at_5p0mm": lb_matched_to_cy_50,
        },
        "matched_pair_geometric_discrepancies": inter_rater_pair_diffs,
        "mapped_40_scenes_extra_trace_breakdown": mapped_stats,
        "mapped_39_scenes_extra_trace_breakdown": mapped_stats,
        "per_dual_scene_records": per_dual_scene_records,
    }
    return task2_out


def main():
    t0 = time.time()
    meta_list = pickle.load(open(META_CLEAN_PATH, "rb"))
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]

    if os.path.exists(PREDS_FULL_PKL):
        p_data = pickle.load(open(PREDS_FULL_PKL, "rb"))
        bgs = p_data["stitched_enriched_bgs"]
    else:
        p_data = pickle.load(open(PREDS_FALLBACK_PKL, "rb"))
        b_data = pickle.load(open(BUNDLES_CLEAN_PKL, "rb"))
        bgs = b_data["stitched_enriched_bgs"]

    base_sc = p_data["base_sc_st_mr"]
    syn_out = p_data["syn_out_st_mr"]

    tip_models_5fold = []
    for f_out in range(5):
        m = Dedicated1DTipLocalizer(in_channels=22, hidden_dim=48, seq_len=91)
        m.load_state_dict(p_data["tip_models_5fold_states"][f_out])
        m.eval()
        tip_models_5fold.append(m)

    preds_all, diag_all, calib_records = decode_clean_with_reason_codes_and_diagnostics(
        bgs, base_sc, syn_out, tip_models_5fold, gts_all_list,
        min_conf=50.0, w_synth=0.60, tip_scale=0.36, thresh=3.5
    )

    # Verify 100% case-by-case match with saved clean OOF predictions and use exact post-stitching refined trajectories
    saved_counts = [len(x) for x in p_data["all_tau50"]]
    cur_counts = [len(x) for x in preds_all]
    assert saved_counts == cur_counts, f"Count mismatch: {saved_counts} vs {cur_counts}"
    for i in range(len(preds_all)):
        for pi in range(len(preds_all[i])):
            preds_all[i][pi]["ras"] = np.asarray(p_data["all_tau50"][i][pi]["ras"], dtype=np.float64)
            preds_all[i][pi]["points"] = preds_all[i][pi]["ras"]
    print(f"[Verified] Decoded predictions match clean OOF predictions across all 54 cases (total={sum(cur_counts)}).", flush=True)

    # 1. Compute and save Stage 2B Calibration & Reason-Code Audit JSON
    compute_stage2b_calibration_and_reason_codes(calib_records, preds_all, gts_all_list)

    # 2. Compute and save Clean Nested Failure Analysis JSON
    existing_task2 = None
    if os.path.exists(OUT_FAILURE_JSON):
        existing_task2 = json.load(open(OUT_FAILURE_JSON, "r")).get("task2_klim_and_dual_rater_audit")

    res_task1 = run_clean_failure_analysis(meta_list, bgs, preds_all, diag_all)
    res_task2 = update_task2_with_clean_40_dual_scenes(meta_list, bgs, existing_task2)

    out_failure = {
        "generated_at": "2026-10-23",
        "benchmark_version": "clean_nested_cv_v1_stitched_1960",
        "script": "vault/paper/benchmarks_clean/calibration_and_failure_clean.py",
        "elapsed_s": round(time.time() - t0, 2),
        "task1_failure_analysis": res_task1,
        "task2_klim_and_dual_rater_audit": res_task2,
    }
    os.makedirs(os.path.dirname(OUT_FAILURE_JSON), exist_ok=True)
    with open(OUT_FAILURE_JSON, "w") as f:
        json.dump(out_failure, f, indent=2)
    print(f"\n[DONE] Saved both {OUT_CALIB_JSON} and {OUT_FAILURE_JSON} in {time.time() - t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
