#!/usr/bin/env python3
"""
Comprehensive Per-Catheter & Per-Case Failure Analysis + Reference-Free k_lim Audit
+ Dual-Rater (LB/AMe vs CY/HT) Training Label Audit on the Clean 5-Fold OOF Re-Run.

Outputs:
  - /home/guillaume/github/needlefinder/vault/sources/2026-10-23-failure-analysis.json
"""

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
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
import torch

for mod in [
    "vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util",
    "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"
]:
    sys.modules[mod] = types.ModuleType(mod)

sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun")

from evaluate_unified_apples_to_apples import (
    META_PATH,
    CACHE_DIR,
    OOF_DIR,
    compute_unified_pair_metrics,
    summarize_distribution,
    evaluate_method_unified,
    load_human_interobserver_40cases,
    resample_polyline_3d,
    get_gts,
    traj_gt_dist,
)
from run_synthbrachy3t_full import (
    ENRICHED_BUNDLES_CACHE_PATH,
    analyze_catheter_adhesion,
)
from test_synthbrachy3t_v2 import apply_tip_adjustment_and_poly3
from prepare_and_inspect_dataset import (
    read_nrrd_local,
    read_vtk_centerline_local,
    dedup_needle_list,
    rust_compute_3d_sym_response,
)

ENS_SYNTH_CACHE_PATH = "/tmp/rerun/cached_ens_synth_oof_54.pkl"
PREDS_PKL_PATH = "/home/guillaume/github/needlefinder/vault/sources/table3_export_rerun/nf2026_synth_tau50_oof_predictions_54cases.pkl"
OUT_JSON_PATH = "/home/guillaume/github/needlefinder/vault/sources/2026-10-23-failure-analysis.json"


def compute_pair_metrics_unbounded(det_pts, gt_pts):
    """
    Same as compute_unified_pair_metrics, but without the 8.0 mm mean_shaft_xy cutoff
     and without requiring >1.0 mm Z overlap (falls back to 3D curve distance and extrapolated XY distance)
    so every prediction-GT pair can be diagnosed.
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

    # Extrapolated / full-GT-span XY distance (checks if det_ras is an axial fragment on the same line as gt_ras)
    zs_det = np.linspace(float(det_ras[0, 2]), float(det_ras[-1, 2]), 25)
    # Fit line to gt_ras
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
    min_p2g_3d = float(np.mean(d_p2g))  # one-sided distance from pred fragment to GT curve
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
        "hd95_3d": hd95_3d,
        "hd_xy": hd_xy,
        "hd95_xy": hd95_xy,
        "tip_3d": tip_3d,
        "tip_z": tip_z,
        "dz_tip_signed": dz_tip_signed,
        "tip_xy": tip_xy,
        "base_3d": base_3d,
        "dz_base_signed": dz_base_signed,
    }


def decode_with_diagnostics(bundle_items, base_oof_scores, synth_oof_outputs, min_conf=50.0, long_mode=False, w_synth=0.60, tip_scale=0.36):
    """
    Exact replica of decode_hybrid_synth_nf2026_preds that also records per-candidate diagnostics:
      - conf_score
      - status: 'selected', 'rejected_low_conf', 'rejected_irgm', 'rejected_low_sym', 'rejected_dedup'
      - raw_ras (before tip adjustment & poly3)
      - refined_ras (after tip adjustment & poly3)
      - dz_comb_mm, dz_pred_net, dz_prof_mm, dz_nbr_mm
      - dedup_winner_idx (if rejected_dedup)
    """
    preds_out = []
    diag_out = []

    for i, bg in enumerate(bundle_items):
        sc_base = base_oof_scores[i]
        out_syn = synth_oof_outputs[i]
        sc_syn = 1.0 / (1.0 + np.exp(-np.clip(out_syn[:, :3], -20.0, 20.0)))
        dz_pred_net = out_syn[:, 3] * 15.0

        sc = (1.0 - w_synth) * sc_base + w_synth * sc_syn
        if long_mode:
            conf_arr = 100.0 * (0.30 * sc[:, 0] + 0.25 * sc[:, 1] + 0.45 * sc[:, 2])
        else:
            conf_arr = 100.0 * (0.45 * sc[:, 0] + 0.55 * sc[:, 1])

        high_idx = np.where(conf_arr >= 55.0)[0]
        if len(high_idx) >= 3:
            all_xy = np.concatenate([bg["pool"][idx]["ras"][:, :2] for idx in high_idx], axis=0)
            center_xy = np.median(all_xy, axis=0)
            dists_core = np.linalg.norm(all_xy - center_xy[None, :], axis=1)
            irgm_radius_mm = float(np.percentile(dists_core, 98.0)) + 15.5
        else:
            center_xy = None
            irgm_radius_mm = 999.0

        sel = []
        case_diag = [None] * len(bg["pool"])

        # Precompute refined trajectories for all pool candidates too so we can inspect them
        for idx_c in np.argsort(-conf_arr):
            cand = dict(bg["pool"][idx_c])
            pts = np.asarray(cand["ras"], dtype=np.float64)
            pts = pts[np.argsort(pts[:, 2])]

            dz_prof_mm = float(cand.get("z_prof_tip", pts[-1, 2]) - pts[-1, 2])
            dz_nbr_mm = float(cand.get("z_tip_nbr", pts[-1, 2]) - pts[-1, 2])
            dz_comb_mm = tip_scale * (0.65 * float(dz_pred_net[idx_c]) + 0.25 * dz_prof_mm + 0.10 * dz_nbr_mm)
            pts_refined = apply_tip_adjustment_and_poly3(
                pts,
                dz_comb_mm,
                float(cand.get("vx_top", 0.0)),
                float(cand.get("vy_top", 0.0)),
            )

            info = {
                "pool_idx": int(idx_c),
                "conf": float(conf_arr[idx_c]),
                "raw_ras": pts,
                "refined_ras": pts_refined,
                "dz_pred_net_mm": float(dz_pred_net[idx_c]),
                "dz_prof_mm": dz_prof_mm,
                "dz_nbr_mm": dz_nbr_mm,
                "dz_comb_mm": dz_comb_mm,
                "from_mpnn": float(cand.get("from_mpnn", 0.0)),
                "mean_sym": float(cand.get("mean_sym", 0.5)),
                "frac_active": float(cand.get("frac_active", 0.5)),
            }

            if conf_arr[idx_c] < min_conf:
                info["status"] = "rejected_low_conf"
                case_diag[idx_c] = info
                continue

            if center_xy is not None:
                mean_dist_center = float(np.linalg.norm(pts[:, :2].mean(axis=0) - center_xy))
                if mean_dist_center > irgm_radius_mm and conf_arr[idx_c] < min_conf + 20.0:
                    info["status"] = "rejected_irgm"
                    case_diag[idx_c] = info
                    continue

            if cand.get("mean_sym", 0.5) < 0.14 and cand.get("frac_active", 0.5) < 0.40 and conf_arr[idx_c] < min_conf + 15.0:
                info["status"] = "rejected_low_sym"
                case_diag[idx_c] = info
                continue

            dedup_thresh = 1.60 if (conf_arr[idx_c] >= 65.0 and cand.get("mpnn_pa", 0.0) >= 0.50) else 2.02
            dup_winner = None
            for s_idx, s in enumerate(sel):
                if traj_gt_dist(pts, s["ras"]) < dedup_thresh:
                    dup_winner = s["pool_idx"]
                    break
            if dup_winner is not None:
                info["status"] = "rejected_dedup"
                info["dedup_winner_pool_idx"] = int(dup_winner)
                case_diag[idx_c] = info
                continue

            info["status"] = "selected"
            info["sel_idx"] = len(sel)
            case_diag[idx_c] = info

            cand["ras"] = pts_refined
            cand["points"] = pts_refined
            cand["raw_ras"] = pts
            cand["score"] = float(conf_arr[idx_c])
            cand["pool_idx"] = int(idx_c)
            cand["dz_comb_mm"] = dz_comb_mm
            cand["dz_pred_net_mm"] = float(dz_pred_net[idx_c])
            cand["dz_prof_mm"] = dz_prof_mm
            cand["dz_nbr_mm"] = dz_nbr_mm
            sel.append(cand)

        preds_out.append(sel)
        diag_out.append(case_diag)

    return preds_out, diag_out


def run_task1_failure_analysis(meta_list, bundle_items, base_oof_scores, ens_oof_outputs):
    print("\n======================================================================")
    print("TASK 1: COMPREHENSIVE PER-CATHETER & PER-CASE FAILURE ANALYSIS")
    print("======================================================================", flush=True)

    preds_all, diag_all = decode_with_diagnostics(
        bundle_items, base_oof_scores, ens_oof_outputs, min_conf=50.0, long_mode=False, w_synth=0.60, tip_scale=0.36
    )

    # Verify exact match with saved pickle
    if os.path.exists(PREDS_PKL_PATH):
        saved_pkl = pickle.load(open(PREDS_PKL_PATH, "rb"))
        saved_counts = [len(x) for x in saved_pkl["all_tau50"]]
        cur_counts = [len(x) for x in preds_all]
        assert saved_counts == cur_counts, f"Mismatch in prediction counts: {saved_counts} vs {cur_counts}"
        print(f"[Verified] Decoded predictions match {PREDS_PKL_PATH} across all 54 cases (total preds = {sum(cur_counts)}).")

    gts_all_dicts = [m["gts_all"] for m in meta_list]
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]

    # Map dual-traced 40 cases to benchmark case indices 0..53
    overlap_nums, cy_40, lb_40 = load_human_interobserver_40cases()
    case_nums = [int(m["case_num"]) for m in meta_list]
    lb_by_case_idx = {}
    cy40_by_case_idx = {}
    for cnum, cy_nds, lb_nds in zip(overlap_nums, cy_40, lb_40):
        if cnum in case_nums:
            c_idx = case_nums.index(cnum)
            lb_by_case_idx[c_idx] = lb_nds
            cy40_by_case_idx[c_idx] = cy_nds

    # Get volume K_tot for each case from cached .pt
    K_tot_per_case = []
    k0_per_case = []
    for i in range(54):
        d_pt = torch.load(f"{CACHE_DIR}/case_{i:02d}.pt", map_location="cpu", weights_only=False)
        k0 = int(d_pt["k0"])
        K_sub = int(d_pt["X"].shape[1])
        k0_per_case.append(k0)
        K_tot_per_case.append(k0 + K_sub)

    # ------------------------------------------------------------------
    # 1A. Candidate Pool vs Final Selection Recall at 3.5 mm and 5.0 mm
    # ------------------------------------------------------------------
    raw_pools = [[p["ras"] for p in bg["pool"]] for bg in bundle_items]
    refined_pools = [[d["refined_ras"] for d in diag_all[i]] for i in range(54)]
    stage0_pools = [[p["ras"] for p in bg["pool"] if float(p.get("from_mpnn", 0.0)) == 0.0] for bg in bundle_items]
    stage01_pools = [[p["ras"] for p in bg["pool"] if float(p.get("from_mpnn", 0.0)) in (0.0, 0.5)] for bg in bundle_items]

    pool_35_summary, _ = evaluate_method_unified(raw_pools, gts_all_list, thresh=3.5, n_boot=50)
    pool_50_summary, _ = evaluate_method_unified(raw_pools, gts_all_list, thresh=5.0, n_boot=50)
    ref_pool_35_summary, _ = evaluate_method_unified(refined_pools, gts_all_list, thresh=3.5, n_boot=50)
    ref_pool_50_summary, _ = evaluate_method_unified(refined_pools, gts_all_list, thresh=5.0, n_boot=50)

    pool_long_35, _ = evaluate_method_unified(raw_pools, gts_long_list, thresh=3.5, n_boot=50)
    pool_long_50, _ = evaluate_method_unified(raw_pools, gts_long_list, thresh=5.0, n_boot=50)

    final_35_summary, _ = evaluate_method_unified(preds_all, gts_all_list, thresh=3.5, n_boot=50)
    final_50_summary, _ = evaluate_method_unified(preds_all, gts_all_list, thresh=5.0, n_boot=50)

    total_pool_cands = sum(len(p) for p in raw_pools)
    print(f"\n--- 1A. Candidate Pool (N={total_pool_cands} cands, {total_pool_cands/54:.1f}/case) vs Final Selection (N={final_35_summary['n_pred']} preds) ---")
    print(f"  Raw Pool (Stage 0+1+2A) @ 3.5 mm : Closest Rec = {pool_35_summary['closest_matched']}/1063 ({pool_35_summary['closest_recall_pct']:.2f}%), "
          f"1-to-1 Hung Rec = {pool_35_summary['hungarian_1to1_matched']}/1063 ({pool_35_summary['hungarian_1to1_recall_pct']:.2f}%)")
    print(f"  Raw Pool (Stage 0+1+2A) @ 5.0 mm : Closest Rec = {pool_50_summary['closest_matched']}/1063 ({pool_50_summary['closest_recall_pct']:.2f}%), "
          f"1-to-1 Hung Rec = {pool_50_summary['hungarian_1to1_matched']}/1063 ({pool_50_summary['hungarian_1to1_recall_pct']:.2f}%)")
    print(f"  Refined Pool (+Tip/Poly3) @ 3.5mm: Closest Rec = {ref_pool_35_summary['closest_matched']}/1063 ({ref_pool_35_summary['closest_recall_pct']:.2f}%), "
          f"1-to-1 Hung Rec = {ref_pool_35_summary['hungarian_1to1_matched']}/1063 ({ref_pool_35_summary['hungarian_1to1_recall_pct']:.2f}%)")
    print(f"  Final Selection (tau=50%) @ 3.5mm: Closest Rec = {final_35_summary['closest_matched']}/1063 ({final_35_summary['closest_recall_pct']:.2f}%), "
          f"1-to-1 Hung Rec = {final_35_summary['hungarian_1to1_matched']}/1063 ({final_35_summary['hungarian_1to1_recall_pct']:.2f}%)")
    print(f"  Final Selection (tau=50%) @ 5.0mm: Closest Rec = {final_50_summary['closest_matched']}/1063 ({final_50_summary['closest_recall_pct']:.2f}%), "
          f"1-to-1 Hung Rec = {final_50_summary['hungarian_1to1_matched']}/1063 ({final_50_summary['hungarian_1to1_recall_pct']:.2f}%)")

    # ------------------------------------------------------------------
    # Detailed Per-Catheter Matching & Failure Attribution Across All 1,063 GTs
    # ------------------------------------------------------------------
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

        # Bundle XY center (median of all GT points in case i)
        all_gt_xy = np.concatenate([g[:, :2] for g in gts_pts], axis=0)
        bundle_center_xy = np.median(all_gt_xy, axis=0)

        # Inter-catheter minimum XY distance on shared axial slices for each GT catheter
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

        # Solve 1-to-1 Hungarian matching for Final Predictions at 3.5 mm
        cost_m = np.full((n_gt, max(1, n_pred)), 1e6, dtype=np.float64)
        pair_metrics = {}
        for gi in range(n_gt):
            for pi in range(n_pred):
                pm = compute_unified_pair_metrics(preds[pi]["ras"], gts_pts[gi])
                if pm is not None:
                    cost_m[gi, pi] = pm["mean_shaft_xy"]
                    pair_metrics[(gi, pi)] = pm

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

        # Also solve 1-to-1 Hungarian matching for Raw Pool at 3.5 mm and 5.0 mm
        cost_pool_raw = np.full((n_gt, max(1, n_pool)), 1e6, dtype=np.float64)
        cost_pool_ref = np.full((n_gt, max(1, n_pool)), 1e6, dtype=np.float64)
        pool_raw_pm = {}
        pool_ref_pm = {}
        for gi in range(n_gt):
            for ci_p in range(n_pool):
                pm_r = compute_unified_pair_metrics(pool_diags[ci_p]["raw_ras"], gts_pts[gi])
                if pm_r is not None:
                    cost_pool_raw[gi, ci_p] = pm_r["mean_shaft_xy"]
                    pool_raw_pm[(gi, ci_p)] = pm_r
                pm_f = compute_unified_pair_metrics(pool_diags[ci_p]["refined_ras"], gts_pts[gi])
                if pm_f is not None:
                    cost_pool_ref[gi, ci_p] = pm_f["mean_shaft_xy"]
                    pool_ref_pm[(gi, ci_p)] = pm_f

        # Record every GT catheter
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

            # Best candidate in pool for this GT
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
                    # Why was a GT that IS in the candidate pool (<=3.5 mm) missed in 1-to-1 Hungarian matching?
                    statuses = [pool_diags[ci_p]["status"] for ci_p in best_pool_cands_35]
                    if is_matched_closest:
                        # A selected prediction IS within <= 3.5 mm of this GT, but Hungarian assigned it to a neighbor!
                        miss_category = "pruned_stage2b_adhesion_hungarian_collision"
                        miss_detail = "Selected prediction within <=3.5 mm assigned to adjacent adhering GT; twin candidate suppressed by dedup or low conf"
                    elif "selected" in statuses:
                        # Wait: a candidate was selected, had raw_ras <= 3.5 mm, but refined_ras > 3.5 mm!
                        miss_category = "pruned_stage2b_tip_poly3_shifted_above_35"
                        miss_detail = f"Selected candidate had raw shaft <=3.5mm ({min_pool_raw_d:.2f}mm) but poly3/tip shift moved shaft to {min_pool_ref_d:.2f}mm"
                    elif "rejected_dedup" in statuses:
                        miss_category = "pruned_stage2b_dedup_suppression"
                        miss_detail = f"Pool candidate had conf={max_pool_conf_for_gt:.1f}% (>=50%) but suppressed by synchronous deduplication against adjacent trajectory"
                    elif "rejected_irgm" in statuses or "rejected_low_sym" in statuses:
                        miss_category = "pruned_stage2b_irgm_or_sym_gate"
                        miss_detail = f"Pool candidate had conf={max_pool_conf_for_gt:.1f}% (>=50%) but rejected by IRGM or symmetry gate"
                    else:
                        miss_category = "pruned_stage2b_low_confidence"
                        miss_detail = f"Pool candidate(s) within <=3.5 mm had max confidence {max_pool_conf_for_gt:.1f}% < 50.0% threshold"

            gt_rec = {
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
            }
            all_gt_records.append(gt_rec)

            # If matched 1-to-1, record full tip & shaft analysis before vs after refinement
            if is_matched_1to1:
                pi = matched_gt_to_pred[gi]
                pred_obj = preds[pi]
                pm_after = compute_pair_metrics_unbounded(pred_obj["ras"], g_pts)
                pm_before = compute_pair_metrics_unbounded(pred_obj["raw_ras"], g_pts)
                # Also check what tip_scale=1.0 would give on this candidate
                dz_full = 0.65 * pred_obj["dz_pred_net_mm"] + 0.25 * pred_obj["dz_prof_mm"] + 0.10 * pred_obj["dz_nbr_mm"]
                pts_ts1 = apply_tip_adjustment_and_poly3(
                    pred_obj["raw_ras"],
                    dz_full,
                    float(pred_obj.get("vx_top", 0.0)),
                    float(pred_obj.get("vy_top", 0.0)),
                )
                pm_ts1 = compute_pair_metrics_unbounded(pts_ts1, g_pts)

                # Oracle best tip error among all candidates in pool matching this GT (<= 3.5 mm)
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
                    "dz_pred_net_mm": round(pred_obj["dz_pred_net_mm"], 4),
                    "dz_prof_mm": round(pred_obj["dz_prof_mm"], 4),
                    "dz_nbr_mm": round(pred_obj["dz_nbr_mm"], 4),
                })

        # ------------------------------------------------------------------
        # 1C. Classify Every Unmatched Prediction (False Positive / Extra Track)
        # ------------------------------------------------------------------
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

            # 1. Check distance to already-matched GT catheters (near-duplicate / split axial fragment)
            best_matched_gt_xy = 999.0
            best_matched_gt_extrap_xy = 999.0
            best_matched_gt_p2g_3d = 999.0
            best_matched_gi = None
            for gi in matched_gt_indices:
                pm_u = compute_pair_metrics_unbounded(p_ras, gts_pts[gi])
                if pm_u is not None:
                    d_eff = min(pm_u["mean_shaft_xy"], pm_u["extrap_xy_dist"], pm_u["min_p2g_3d"])
                    if d_eff < min(best_matched_gt_xy, best_matched_gt_extrap_xy, best_matched_gt_p2g_3d):
                        best_matched_gt_xy = pm_u["mean_shaft_xy"]
                        best_matched_gt_extrap_xy = pm_u["extrap_xy_dist"]
                        best_matched_gt_p2g_3d = pm_u["min_p2g_3d"]
                        best_matched_gi = gi

            # 2. Check distance to unmatched GT catheters (borderline match 3.5 - 6.0 mm)
            best_unmatched_gt_xy = 999.0
            best_unmatched_gt_extrap_xy = 999.0
            best_unmatched_gt_p2g_3d = 999.0
            best_unmatched_gi = None
            for gi in unmatched_gt_indices:
                pm_u = compute_pair_metrics_unbounded(p_ras, gts_pts[gi])
                if pm_u is not None:
                    if pm_u["mean_shaft_xy"] < best_unmatched_gt_xy:
                        best_unmatched_gt_xy = pm_u["mean_shaft_xy"]
                        best_unmatched_gt_extrap_xy = pm_u["extrap_xy_dist"]
                        best_unmatched_gt_p2g_3d = pm_u["min_p2g_3d"]
                        best_unmatched_gi = gi

            # 3. Check distance to Second Annotator (LB/AMe) traces in this scan
            best_lb_xy = 999.0
            best_lb_p2g_3d = 999.0
            for lb_pts in lb_traces:
                pm_lb = compute_pair_metrics_unbounded(p_ras, lb_pts)
                if pm_lb is not None:
                    if pm_lb["mean_shaft_xy"] < best_lb_xy:
                        best_lb_xy = pm_lb["mean_shaft_xy"]
                        best_lb_p2g_3d = pm_lb["min_p2g_3d"]

            # Non-exclusive flags
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

            # Mutually exclusive primary classification
            # First check if it's an extra trace annotated by Rater 2 (LB/AMe) within 3.5 mm (and not just a duplicate of a matched CY GT)
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

    # ------------------------------------------------------------------
    # Aggregate 1A & 1B: Missed Catheters Breakdown
    # ------------------------------------------------------------------
    missed_records = [r for r in all_gt_records if not r["is_matched_1to1"]]
    n_missed_total = len(missed_records)

    miss_cat_counts = {}
    for r in missed_records:
        c = r["miss_category"]
        miss_cat_counts[c] = miss_cat_counts.get(c, 0) + 1

    # For those pruned by low confidence in Stage 2B, check their max_pool_conf distribution
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

    print(f"\n--- 1B. Missed Catheters Breakdown (Total Missed = {n_missed_total} / 1,063) ---")
    print("  Miss Attribution:")
    for k, v in sorted(miss_cat_counts.items(), key=lambda x: -x[1]):
        print(f"    - {k:46s}: {v:3d} ({100.0*v/n_missed_total:5.1f}% of missed, {100.0*v/1063:4.2f}% of all GTs)")
    print(f"    - Low-conf pruned breakdown: {low_conf_bins}")
    print("  By Type:")
    for k, v in by_type.items():
        print(f"    - {k:15s}: {v['n_matched_1to1']}/{v['n_total']} matched (Rec={v['recall_1to1_pct']:.2f}%), "
              f"missed={v['n_missed']} (no_pool={v['missed_missing_from_pool_35']}, stage2b_pruned={v['missed_pruned_by_stage2b']})")
    print("  By Z-Span:")
    for k, v in by_z_span.items():
        print(f"    - {k:15s}: {v['n_matched_1to1']}/{v['n_total']} matched (Rec={v['recall_1to1_pct']:.2f}%), "
              f"missed={v['n_missed']} (no_pool={v['missed_missing_from_pool_35']}, stage2b_pruned={v['missed_pruned_by_stage2b']})")
    print("  By Radial Distance from Bundle Center:")
    for k in ["lt_10mm_core", "10_to_20mm_inner_ring", "20_to_30mm_outer_ring", "ge_30mm_peripheral"]:
        v = by_radial_dist[k]
        print(f"    - {k:24s}: {v['n_matched_1to1']}/{v['n_total']} matched (Rec={v['recall_1to1_pct']:.2f}%), missed={v['n_missed']}")
    print("  By Adhesion Status (<=3.0 mm neighbor):")
    for k, v in by_adhesion.items():
        print(f"    - {k:30s}: {v['n_matched_1to1']}/{v['n_total']} matched (Rec={v['recall_1to1_pct']:.2f}%), "
              f"missed={v['n_missed']} (no_pool={v['missed_missing_from_pool_35']}, stage2b_pruned={v['missed_pruned_by_stage2b']})")

    print("  Top 10 Worst Cases by Recall:")
    for rank, c in enumerate(worst_10_by_recall, 1):
        print(f"    {rank:2d}. idx={c['case_idx']:02d} ({c['case_id']}): Rec={c['recall_1to1_pct']:5.1f}% "
              f"({c['n_matched_1to1']}/{c['n_gt_all']}, missed={c['n_missed']} [long={c['missed_long']}, short={c['missed_short']}, "
              f"no_pool={c['missed_not_in_pool_35']}, stage2b={c['missed_in_pool_35']}]), Prec={c['precision_1to1_pct']:5.1f}%, F1={c['f1_1to1_pct']:5.1f}%")

    # ------------------------------------------------------------------
    # Aggregate 1C: False-Positive (Extra Track) Breakdown
    # ------------------------------------------------------------------
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

    print(f"\n--- 1C. False-Positive (Extra Track) Breakdown (Total Unmatched Preds = {n_fp_total} / {final_35_summary['n_pred']}) ---")
    print("  Mutually Exclusive Primary FP Classification:")
    for k, v in sorted(fp_primary_counts.items(), key=lambda x: -x[1]):
        print(f"    - {k:54s}: {v:2d} ({100.0*v/max(1,n_fp_total):5.1f}% of FPs, {100.0*v/final_35_summary['n_pred']:4.2f}% of all preds)")
    print(f"  Non-Exclusive Overlap Counts: {fp_non_exclusive}")

    # ------------------------------------------------------------------
    # Aggregate 1D: Distal Tip Error Tail Analysis (E_tip > 10 mm)
    # ------------------------------------------------------------------
    n_matched_total = len(all_matched_records)
    tail_10_records = [r for r in all_matched_records if r["tip_3d"] > 10.0]
    tail_5_records = [r for r in all_matched_records if r["tip_3d"] > 5.0]

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
        "before_vs_after_apply_tip_adjustment_and_poly3_on_955_matched": {
            "before_raw_pool_candidate": {
                "tip_3d": summarize_distribution([r["raw_tip_3d_before_refinement"] for r in all_matched_records]),
                "tip_z": summarize_distribution([r["raw_tip_z_before_refinement"] for r in all_matched_records]),
                "tip_xy": summarize_distribution([r["raw_tip_xy_before_refinement"] for r in all_matched_records]),
                "mean_shaft_xy": summarize_distribution([r["raw_mean_shaft_xy_before_refinement"] for r in all_matched_records]),
                "n_tip_gt_10mm": sum(1 for r in all_matched_records if r["raw_tip_3d_before_refinement"] > 10.0),
                "n_tip_le_3mm": sum(1 for r in all_matched_records if r["raw_tip_3d_before_refinement"] <= 3.0),
            },
            "after_refinement_tip_scale_0p36": {
                "tip_3d": summarize_distribution([r["tip_3d"] for r in all_matched_records]),
                "tip_z": summarize_distribution([r["tip_z"] for r in all_matched_records]),
                "tip_xy": summarize_distribution([r["tip_xy"] for r in all_matched_records]),
                "mean_shaft_xy": summarize_distribution([r["mean_shaft_xy"] for r in all_matched_records]),
                "n_tip_gt_10mm": sum(1 for r in all_matched_records if r["tip_3d"] > 10.0),
                "n_tip_le_3mm": sum(1 for r in all_matched_records if r["tip_3d"] <= 3.0),
            },
            "after_refinement_tip_scale_1p0": {
                "tip_3d": summarize_distribution([r["tip_3d_at_tip_scale_1p0"] for r in all_matched_records]),
                "n_tip_gt_10mm": sum(1 for r in all_matched_records if r["tip_3d_at_tip_scale_1p0"] > 10.0),
                "n_tip_le_3mm": sum(1 for r in all_matched_records if r["tip_3d_at_tip_scale_1p0"] <= 3.0),
            },
            "oracle_best_candidate_in_pool_for_each_matched_gt": {
                "tip_3d": summarize_distribution([r["oracle_pool_best_tip_3d"] for r in all_matched_records]),
                "n_tip_gt_10mm": sum(1 for r in all_matched_records if r["oracle_pool_best_tip_3d"] > 10.0),
                "n_tip_le_3mm": sum(1 for r in all_matched_records if r["oracle_pool_best_tip_3d"] <= 3.0),
            },
        },
    }

    t10 = tip_analysis["tail_gt_10mm_breakdown"]
    b_vs_a = tip_analysis["before_vs_after_apply_tip_adjustment_and_poly3_on_955_matched"]
    print(f"\n--- 1D. Distal Tip Error Tail Analysis (Matched = {n_matched_total}, E_tip > 10 mm = {t10['n_total_gt_10mm']} [{t10['pct_of_matched']:.2f}%]) ---")
    print(f"  All 955 Matched    : mean={tip_analysis['all_955_matched']['tip_3d']['mean']:.2f}mm, med={tip_analysis['all_955_matched']['tip_3d']['median']:.2f}mm, "
          f"IQR={tip_analysis['all_955_matched']['tip_3d']['iqr']:.2f}mm, P95={tip_analysis['all_955_matched']['tip_3d']['p95']:.2f}mm")
    print(f"  CY Long ({tip_analysis['cy_long_matched']['n']:3d})      : mean={tip_analysis['cy_long_matched']['tip_3d']['mean']:.2f}mm, med={tip_analysis['cy_long_matched']['tip_3d']['median']:.2f}mm, "
          f"IQR={tip_analysis['cy_long_matched']['tip_3d']['iqr']:.2f}mm, P95={tip_analysis['cy_long_matched']['tip_3d']['p95']:.2f}mm, >10mm={tip_analysis['cy_long_matched']['n_tip_gt_10mm']}")
    print(f"  HT Obturator ({tip_analysis['ht_obturator_matched']['n']:3d}) : mean={tip_analysis['ht_obturator_matched']['tip_3d']['mean']:.2f}mm, med={tip_analysis['ht_obturator_matched']['tip_3d']['median']:.2f}mm, "
          f"IQR={tip_analysis['ht_obturator_matched']['tip_3d']['iqr']:.2f}mm, P95={tip_analysis['ht_obturator_matched']['tip_3d']['p95']:.2f}mm, >10mm={tip_analysis['ht_obturator_matched']['n_tip_gt_10mm']}")
    print(f"  E_tip > 10 mm Tail ({t10['n_total_gt_10mm']}) Breakdown:")
    print(f"    - Long vs Short        : {t10['n_cy_long']} cy_long vs {t10['n_ht_obturator']} ht_obturator")
    print(f"    - Signed dz_tip        : {t10['n_cranial_overshoot_dz_gt_plus5mm']} cranial overshoots (dz > +5mm; {t10['n_cranial_overshoot_dz_gt_plus10mm']} > +10mm), "
          f"{t10['n_caudal_undershoot_dz_lt_minus5mm']} caudal undershoots (dz < -5mm; {t10['n_caudal_undershoot_dz_lt_minus10mm']} < -10mm), "
          f"{t10['n_pure_xy_tip_error_abs_dz_le_5mm']} within |dz|<=5mm")
    print(f"    - Top-of-FOV Truncation: {t10['n_gt_near_top_of_fov_2slices']} have GT tip within 2 slices of K-1 ({t10['n_pred_near_top_of_fov_2slices']} pred tip within 2 slices of K-1)")
    print(f"  Before vs After apply_tip_adjustment_and_poly3 (on the 955 matched pairs):")
    print(f"    - Before (raw pool)    : mean={b_vs_a['before_raw_pool_candidate']['tip_3d']['mean']:.2f}mm, med={b_vs_a['before_raw_pool_candidate']['tip_3d']['median']:.2f}mm, "
          f"P95={b_vs_a['before_raw_pool_candidate']['tip_3d']['p95']:.2f}mm, >10mm={b_vs_a['before_raw_pool_candidate']['n_tip_gt_10mm']}, <=3mm={b_vs_a['before_raw_pool_candidate']['n_tip_le_3mm']}")
    print(f"    - After (scale=0.36)   : mean={b_vs_a['after_refinement_tip_scale_0p36']['tip_3d']['mean']:.2f}mm, med={b_vs_a['after_refinement_tip_scale_0p36']['tip_3d']['median']:.2f}mm, "
          f"P95={b_vs_a['after_refinement_tip_scale_0p36']['tip_3d']['p95']:.2f}mm, >10mm={b_vs_a['after_refinement_tip_scale_0p36']['n_tip_gt_10mm']}, <=3mm={b_vs_a['after_refinement_tip_scale_0p36']['n_tip_le_3mm']}")
    print(f"    - After (scale=1.00)   : mean={b_vs_a['after_refinement_tip_scale_1p0']['tip_3d']['mean']:.2f}mm, med={b_vs_a['after_refinement_tip_scale_1p0']['tip_3d']['median']:.2f}mm, "
          f"P95={b_vs_a['after_refinement_tip_scale_1p0']['tip_3d']['p95']:.2f}mm, >10mm={b_vs_a['after_refinement_tip_scale_1p0']['n_tip_gt_10mm']}, <=3mm={b_vs_a['after_refinement_tip_scale_1p0']['n_tip_le_3mm']}")
    print(f"    - Oracle Best in Pool  : mean={b_vs_a['oracle_best_candidate_in_pool_for_each_matched_gt']['tip_3d']['mean']:.2f}mm, med={b_vs_a['oracle_best_candidate_in_pool_for_each_matched_gt']['tip_3d']['median']:.2f}mm, "
          f"P95={b_vs_a['oracle_best_candidate_in_pool_for_each_matched_gt']['tip_3d']['p95']:.2f}mm, >10mm={b_vs_a['oracle_best_candidate_in_pool_for_each_matched_gt']['n_tip_gt_10mm']}, <=3mm={b_vs_a['oracle_best_candidate_in_pool_for_each_matched_gt']['n_tip_le_3mm']}")

    return {
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
        "missed_catheters_breakdown_108": {
            "n_total_gt": 1063,
            "n_matched_1to1": 955,
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
        },
        "false_positive_extra_tracks_breakdown_68": {
            "n_total_predictions": final_35_summary["n_pred"],
            "n_matched_1to1": 955,
            "n_unmatched_fp": n_fp_total,
            "mutually_exclusive_primary_categories": fp_primary_counts,
            "non_exclusive_category_overlaps": fp_non_exclusive,
            "all_68_fp_records": all_fp_records,
        },
        "distal_tip_error_tail_analysis": tip_analysis,
    }


def estimate_axial_limit_slice_existing(vol_arr: np.ndarray, sp_k: float) -> int:
    """Exact copy of TrackingAlgorithms.estimateAxialLimitSlice."""
    K, J, I = vol_arr.shape
    if K < 8:
        return 0
    c_roi = vol_arr[:, int(0.30 * J):int(0.70 * J), int(0.30 * I):int(0.70 * I)]
    p50 = gaussian_filter1d(np.percentile(c_roi, 50, axis=(1, 2)).astype(np.float64), sigma=max(1.0, 2.5 / max(0.5, float(sp_k))))
    if K <= 42:
        min_bot = float(np.min(p50[:max(2, int(0.25 * K))]))
        t_ref = float(np.percentile(p50[int(0.30 * K):max(int(0.30 * K) + 2, int(0.85 * K))], 65))
        if min_bot > 0.38 * t_ref:
            return max(0, int(round(3.0 / max(1.0, float(sp_k)))))
        k_lo, k_hi = int(0.18 * K), max(int(0.18 * K) + 2, int(0.70 * K))
        k_peak = k_lo + int(np.argmax(p50[k_lo:k_hi]))
        s_lo = max(1, k_peak - int(round(45.0 / max(0.5, float(sp_k)))))
    else:
        k_lo, k_hi = int(0.25 * K), max(int(0.25 * K) + 2, int(0.62 * K))
        k_peak = k_lo + int(np.argmax(p50[k_lo:k_hi]))
        s_lo = max(int(0.10 * K), k_peak - int(round(52.0 / max(0.5, float(sp_k)))))
    grad = np.gradient(p50)
    if k_peak <= s_lo:
        return max(0, int(0.25 * K))
    k_steep = s_lo + int(np.argmax(grad[s_lo:k_peak]))
    k_est = k_steep
    while k_est > s_lo and grad[k_est] > 0.22 * grad[k_steep]:
        k_est -= 1
    return int(np.clip(k_est, 0, K - 5))


def estimate_axial_limit_slice_sym_safe(vol_arr: np.ndarray, sp_ras, resp_full: np.ndarray, i0: int, i1: int, j0: int, j1: int) -> int:
    """
    100% Reference-Free Conservative Axial Cropping (`k_lim_auto_safe`):
    Uses ONLY 3D MRI intensities and `rust_compute_3d_sym_response` on the full volume `[0, K)`
    with zero access to `.acsv` or `gts_cy`.
    Finds the lowest slice where genuine pelvic tissue + multi-catheter dark-core symmetry
    activity begins, and subtracts a conservative safety buffer (15 mm) so that >= 99.8% of
    all true catheter Z-spans in `[k_lim_gt, K)` are preserved inside `[k0_auto, K)`.
    """
    K, J, I = vol_arr.shape
    sp_k = float(sp_ras[2])
    if K <= 45:
        # Short clinical slab: keep full axial volume from slice 0 (or 1)
        return 0

    c_roi = vol_arr[:, int(0.30 * J):int(0.70 * J), int(0.30 * I):int(0.70 * I)]
    p50 = gaussian_filter1d(np.percentile(c_roi, 50, axis=(1, 2)).astype(np.float64), sigma=max(1.0, 2.0 / max(0.5, sp_k)))
    # Also compute per-slice 3D radial dark-core symmetry activity (top-25 peaks in central pelvic ROI)
    sym_roi = resp_full[:, j0:j1, i0:i1]
    sym_flat = sym_roi.reshape(K, -1)
    top_k = min(25, sym_flat.shape[1])
    sym_prof = np.mean(np.partition(sym_flat, -top_k, axis=1)[:, -top_k:], axis=1)
    sym_smooth = gaussian_filter1d(sym_prof.astype(np.float64), sigma=max(1.0, 2.0 / max(0.5, sp_k)))

    # Transition from external air/template void to pelvic soft tissue
    k_est_int = estimate_axial_limit_slice_existing(vol_arr, sp_k)

    # First slice where pelvic tissue intensity reaches 25% of upper-pelvis median
    t_pelvis = float(np.percentile(p50[int(0.35 * K):int(0.85 * K)], 60))
    tissue_active = np.where(p50 >= 0.22 * t_pelvis)[0]
    k_tissue = int(tissue_active[0]) if len(tissue_active) > 0 else 0

    # Subtract a conservative 12-slice (~19 mm at 1.6 mm spacing) safety margin below the estimated transition
    safety_slices = int(round(18.0 / max(0.8, sp_k)))
    k_safe = max(0, min(k_est_int, k_tissue) - safety_slices)
    return int(k_safe)


def run_task2_klim_and_dual_rater_audit(meta_list, bundle_items):
    print("\n======================================================================")
    print("TASK 2: AUDIT OF REFERENCE-FREE k_lim AND DUAL-RATER (LB/AMe) LABELS")
    print("======================================================================", flush=True)

    gyn_base = "/home/guillaume/Dropbox/GYN Cases"
    with open(os.path.join(gyn_base, "760CYScenes.txt")) as f:
        cy_lines = [l.strip() for l in f if l.strip()]

    n_acsv_found = 0
    n_acsv_valid = 0
    n_fallback_gt = 0
    per_case_klim = []

    # Metrics for comparing k_lim policies:
    # Policy A: k_lim_gt (from .acsv template slice position marker)
    # Policy B: TrackingAlgorithms.estimateAxialLimitSlice(vol, sp_k) (existing intensity-based estimator)
    # Policy C: estimate_axial_limit_slice_sym_safe (conservative intensity + 3D sym estimator)
    # Policy D: k_lim = 0 (full uncropped volume [0, K))
    policy_stats = {
        name: {
            "n_gt_all_bases_below_k0": 0,
            "n_gt_long_bases_below_k0": 0,
            "total_gt_all_z_span_mm": 0.0,
            "preserved_gt_all_z_span_mm": 0.0,
            "abs_diff_from_k_lim_gt_slices": [],
            "abs_diff_from_k_lim_gt_mm": [],
        }
        for name in ["acsv_k_lim_gt", "existing_estimateAxialLimitSlice", "reference_free_sym_safe", "full_volume_k0_eq_0"]
    }

    # Also check unclipped raw VTK centerlines (when read with z_lim_ras=None)
    unclipped_vtk_stats = {
        "n_total_unclipped_all": 0,
        "total_unclipped_z_span_mm": 0.0,
        "z_span_above_acsv_z_lim_mm": 0.0,
        "n_needles_extending_more_than_2mm_below_acsv": 0,
    }

    for idx, rel in enumerate(cy_lines):
        p = os.path.join(gyn_base, rel.lstrip("./"))
        for a, b in [("NRRD-MRI", "NRRD"), ("Maunal_CY", "Manual_CY"), ("Manual CY", "Manual_CY")]:
            if not os.path.exists(p):
                p2 = p.replace(a, b)
                if os.path.exists(p2):
                    p = p2
        d = os.path.dirname(p)
        case_dir = os.path.dirname(os.path.dirname(d))
        case_num = int(re.search(r"(\d+)", os.path.basename(case_dir)).group(1))
        tree = ET.parse(p)
        root = tree.getroot()
        nrrd_path = None
        for vs in root.findall(".//VolumeArchetypeStorage"):
            fn = urllib.parse.unquote(vs.attrib.get("fileName", ""))
            if fn and "label" not in fn.lower() and "pelvis" not in fn.lower() and not fn.endswith(".png") and not fn.endswith(".tiff"):
                cand = os.path.join(d, fn)
                if os.path.exists(cand):
                    nrrd_path = cand
                    break
        if nrrd_path is None:
            cands = [
                x for x in glob.glob(os.path.join(case_dir, "**", "*.nrrd"), recursive=True)
                if "label" not in x.lower() and "pelvis" not in x.lower() and "crop" not in x.lower()
            ]
            nrrd_path = cands[0]

        acsv_cands = (
            glob.glob(os.path.join(d, "markers", "template slice position.acsv"))
            + glob.glob(os.path.join(d, "Markers*", "template slice position.acsv"))
            + glob.glob(os.path.join(case_dir, "*", "Manual_LB", "markers", "template slice position.acsv"))
            + glob.glob(os.path.join(case_dir, "*", "Manual_LB", "template slice position.acsv"))
            + [os.path.join(d, "template slice position.acsv")]
        )
        acsv_path = next((ap for ap in acsv_cands if os.path.exists(ap)), None)
        vol, sp_ras, ijk_to_ras, k_lim_acsv, z_lim_ras = read_nrrd_local(nrrd_path, acsv_path)
        K, J, I = vol.shape
        sp_i, sp_j, sp_k = float(sp_ras[0]), float(sp_ras[1]), float(sp_ras[2])
        ras_to_ijk = np.linalg.inv(ijk_to_ras)

        if acsv_path is not None:
            n_acsv_found += 1
        if k_lim_acsv is not None and (0 <= k_lim_acsv < K):
            n_acsv_valid += 1
        else:
            n_fallback_gt += 1

        # Compute full-volume 3D radial symmetry response [0, K) for reference-free audit
        margin_i = min(I // 2 - 4, int(round(68.0 / sp_i)))
        margin_j = min(J // 2 - 4, int(round(68.0 / sp_j)))
        half_w = ((margin_i + 7) // 8) * 8
        half_h = ((margin_j + 7) // 8) * 8
        ci_v, cj_v = I // 2, J // 2
        i0 = max(0, ci_v - half_w)
        i1 = min(I, i0 + 2 * half_w)
        j0 = max(0, cj_v - half_h)
        j1 = min(J, j0 + 2 * half_h)
        non_zero = vol[:, j0:j1, i0:i1][vol[:, j0:j1, i0:i1] > 0]
        body_thresh = float(np.percentile(non_zero, 15)) if len(non_zero) > 100 else 10.0
        resp_full, _, _ = rust_compute_3d_sym_response(vol, sp_i, sp_j, i0, i1, j0, j1, 0, K, body_thresh)

        k_lim_gt = int(meta_list[idx]["k_lim_gt"])
        k_lim_est = estimate_axial_limit_slice_existing(vol, sp_k)
        k_lim_safe = estimate_axial_limit_slice_sym_safe(vol, sp_ras, resp_full, i0, i1, j0, j1)

        # Also check unclipped VTKs in this case
        st_map = {ms.attrib["id"]: urllib.parse.unquote(ms.attrib.get("fileName", "")) for ms in root.findall(".//ModelStorage")}
        long_vtks = {}
        for m_el in root.findall(".//Model"):
            name = m_el.attrib.get("name", "")
            attrs = m_el.attrib.get("attributes", "")
            if "type:Validation" in attrs and len(name.split("_")) == 2 and name.startswith("manual-seg_"):
                sref = m_el.attrib.get("storageNodeRef", "")
                fn = st_map.get(sref, name + ".vtk")
                fp = os.path.join(d, fn)
                if os.path.exists(fp):
                    long_vtks[name] = fp
        unclipped_cy = []
        for name in sorted(long_vtks.keys(), key=lambda s: int(s.split("_")[1])):
            cl_u = read_vtk_centerline_local(long_vtks[name], z_lim_ras=None)
            if cl_u is not None:
                unclipped_cy.append({"name": name, "pts": cl_u, "type": "cy_long"})
        ht_vtks = sorted(glob.glob(os.path.join(case_dir, "*", "Manual-shortOneedle", "manual-seg_*.vtk")))
        unclipped_ht = []
        for fp in ht_vtks:
            bn = os.path.basename(fp)
            if len(bn.split("_")) == 2 and not bn.endswith(".incorrect"):
                cl_u = read_vtk_centerline_local(fp, z_lim_ras=None)
                if cl_u is not None:
                    unclipped_ht.append({"name": "ht_" + bn[:-4], "pts": cl_u, "type": "ht_obturator"})
        unclipped_all = dedup_needle_list(unclipped_cy + unclipped_ht, dist_thresh=2.4)
        for g_u in unclipped_all:
            pts_u = g_u["pts"]
            span_u = max(0.0, float(pts_u[-1, 2] - pts_u[0, 2]))
            unclipped_vtk_stats["n_total_unclipped_all"] += 1
            unclipped_vtk_stats["total_unclipped_z_span_mm"] += span_u
            if z_lim_ras is not None:
                z_bot_clip = max(float(pts_u[0, 2]), float(z_lim_ras - 2.0))
                span_above = max(0.0, float(pts_u[-1, 2] - z_bot_clip))
                unclipped_vtk_stats["z_span_above_acsv_z_lim_mm"] += span_above
                if float(pts_u[0, 2]) < float(z_lim_ras - 2.0):
                    unclipped_vtk_stats["n_needles_extending_more_than_2mm_below_acsv"] += 1

        # Evaluate the 4 k_lim policies against the benchmark GT catheters (gts_all and gts_long)
        gts_all = meta_list[idx]["gts_all"]
        gts_long = meta_list[idx]["gts_long"]

        k_bases_all = [float((ras_to_ijk @ np.append(g["pts"][0], 1.0))[2]) for g in gts_all]
        min_k_base = min(k_bases_all)

        for pname, kl_val in [
            ("acsv_k_lim_gt", k_lim_gt),
            ("existing_estimateAxialLimitSlice", k_lim_est),
            ("reference_free_sym_safe", k_lim_safe),
            ("full_volume_k0_eq_0", 0),
        ]:
            k0_val = max(0, kl_val - 1)
            z_k0_ras = float((ijk_to_ras @ np.array([0.0, 0.0, float(k0_val), 1.0]))[2])
            st = policy_stats[pname]
            st["abs_diff_from_k_lim_gt_slices"].append(abs(kl_val - k_lim_gt))
            st["abs_diff_from_k_lim_gt_mm"].append(abs(kl_val - k_lim_gt) * sp_k)

            for g in gts_all:
                pts = g["pts"][np.argsort(g["pts"][:, 2])]
                k_b = float((ras_to_ijk @ np.append(pts[0], 1.0))[2])
                if k_b < k0_val - 0.5:
                    st["n_gt_all_bases_below_k0"] += 1
                span_tot = max(1e-6, float(pts[-1, 2] - pts[0, 2]))
                span_pres = max(0.0, float(pts[-1, 2] - max(float(pts[0, 2]), z_k0_ras)))
                st["total_gt_all_z_span_mm"] += span_tot
                st["preserved_gt_all_z_span_mm"] += min(span_tot, span_pres)

            for g in gts_long:
                pts = g["pts"][np.argsort(g["pts"][:, 2])]
                k_b = float((ras_to_ijk @ np.append(pts[0], 1.0))[2])
                if k_b < k0_val - 0.5:
                    st["n_gt_long_bases_below_k0"] += 1

        per_case_klim.append({
            "case_idx": idx,
            "case_id": meta_list[idx]["case_id"],
            "shape_KJI": [int(K), int(J), int(I)],
            "sp_ras": [round(float(x), 3) for x in sp_ras],
            "has_acsv": (acsv_path is not None),
            "acsv_rel_path": os.path.relpath(acsv_path, gyn_base) if acsv_path else None,
            "k_lim_gt": k_lim_gt,
            "k_lim_existing_auto": k_lim_est,
            "k_lim_sym_safe_auto": k_lim_safe,
            "min_gt_base_k": round(min_k_base, 2),
        })

    for pname, st in policy_stats.items():
        st["preserved_z_span_pct"] = round(100.0 * st["preserved_gt_all_z_span_mm"] / max(1.0, st["total_gt_all_z_span_mm"]), 4)
        st["mean_abs_diff_slices"] = round(float(np.mean(st["abs_diff_from_k_lim_gt_slices"])), 2)
        st["median_abs_diff_slices"] = round(float(np.median(st["abs_diff_from_k_lim_gt_slices"])), 2)
        st["mean_abs_diff_mm"] = round(float(np.mean(st["abs_diff_from_k_lim_gt_mm"])), 2)
        st["median_abs_diff_mm"] = round(float(np.median(st["abs_diff_from_k_lim_gt_mm"])), 2)
        del st["abs_diff_from_k_lim_gt_slices"]
        del st["abs_diff_from_k_lim_gt_mm"]

    print(f"\n--- 2A. Audit of k_lim Provenance & Reference-Free Cropping Across 54 Cases ---")
    print(f"  Cases with valid 'template slice position.acsv' fiducial : {n_acsv_valid} / 54")
    print(f"  Cases falling back to gt_base_ks median                  : {n_fallback_gt} / 54")
    for pname, st in policy_stats.items():
        print(f"  Policy [{pname:32s}]: bases_below_k0 = {st['n_gt_all_bases_below_k0']:3d}/1063 ({st['n_gt_long_bases_below_k0']:3d}/759 long), "
              f"Z-span preserved = {st['preserved_z_span_pct']:6.3f}%, MAE vs acsv = {st['mean_abs_diff_slices']:.1f} slices ({st['mean_abs_diff_mm']:.1f} mm)")

    # ------------------------------------------------------------------
    # 2B. Audit of 40 Dual-Rater (LB/AMe vs CY & HT) Scenes for Training Labels
    # ------------------------------------------------------------------
    overlap_nums, cy_40, lb_40 = load_human_interobserver_40cases()
    case_nums = [int(m["case_num"]) for m in meta_list]

    mapped_dual_scenes = []
    unmapped_dual_scenes = []

    total_cy_40 = sum(len(x) for x in cy_40)
    total_lb_40 = sum(len(x) for x in lb_40)

    # Across all 40 scenes: compare LB/AMe vs CY (from load_human_interobserver_40cases)
    lb_matched_to_cy_35 = 0
    lb_matched_to_cy_50 = 0
    lb_matched_pairs_metrics = []
    lb_extra_not_in_cy35 = []

    # Across the 39 mapped scenes (in the 54 benchmark cohort): also compare against gts_all (CY + HT short obturator) and NF pool
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

        # 1-to-1 Hungarian between LB/AMe and CY at 3.5 mm and 5.0 mm
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

        if cnum not in case_nums:
            unmapped_dual_scenes.append({
                "case_num": cnum,
                "n_cy": n_cy,
                "n_lb_ame": n_lb,
                "n_matched_35": len(matched_lb_35),
                "n_matched_50": n_m50,
            })
            continue

        b_idx = case_nums.index(cnum)
        mapped_dual_scenes.append({"case_num": cnum, "benchmark_idx": b_idx})
        mapped_stats["n_mapped_scenes"] += 1

        gts_long_b = [g["pts"][np.argsort(g["pts"][:, 2])] for g in meta_list[b_idx]["gts_long"]]
        gts_ht_b = [g["pts"][np.argsort(g["pts"][:, 2])] for g in meta_list[b_idx]["gts_all"] if g.get("type") == "ht_obturator"]
        gts_all_b = [g["pts"][np.argsort(g["pts"][:, 2])] for g in meta_list[b_idx]["gts_all"]]
        pool_b = [p["ras"] for p in bundle_items[b_idx]["pool"]]

        mapped_stats["n_cy_long_in_mapped"] += len(gts_long_b)
        mapped_stats["n_ht_short_in_mapped"] += len(gts_ht_b)
        mapped_stats["n_gts_all_in_mapped"] += len(gts_all_b)
        mapped_stats["n_lb_ame_in_mapped"] += n_lb
        mapped_stats["lb_matched_to_cy_long_35"] += len(matched_lb_35)
        mapped_stats["lb_matched_to_cy_long_50"] += n_m50
        mapped_stats["cy_long_not_in_lb_35"] += (n_cy - len(matched_lb_35))

        # Inspect every unmatched LB/AMe trace in this mapped scene
        scene_extra_ht35 = 0
        scene_extra_borderline = 0
        scene_extra_new = 0
        for li in range(n_lb):
            if li in matched_lb_35:
                continue
            lb_pts = lb_nds[li]
            # Check distance to HT short obturator needles
            best_ht_d = 999.0
            for ht_pts in gts_ht_b:
                pm_h = compute_pair_metrics_unbounded(lb_pts, ht_pts)
                if pm_h is not None and pm_h["z_overlap"] > 1.0:
                    best_ht_d = min(best_ht_d, pm_h["mean_shaft_xy"])
            # Check distance to any CY needle
            best_cy_d = 999.0
            for cy_pts in cy_nds:
                pm_c = compute_pair_metrics_unbounded(lb_pts, cy_pts)
                if pm_c is not None and pm_c["z_overlap"] > 1.0:
                    best_cy_d = min(best_cy_d, pm_c["mean_shaft_xy"])
            # Check if NeedleFinder candidate pool has a trajectory matching this LB/AMe trace <= 3.5 mm
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

    print(f"\n--- 2B. Dual-Rater (LB/AMe vs CY/HT) Audit Across 40 Overlapping Scenes ---")
    print(f"  40 Dual-Traced Scenes: {total_cy_40} CY needles, {total_lb_40} LB/AMe needles")
    print(f"  Mapped to 54-case benchmark: {len(mapped_dual_scenes)} / 40 scenes (unmapped: {unmapped_dual_scenes})")
    print(f"  Across all 40 scenes: {lb_matched_to_cy_35}/{total_lb_40} ({100.0*lb_matched_to_cy_35/total_lb_40:.2f}%) LB/AMe traces match CY at <=3.5 mm "
          f"({lb_matched_to_cy_50}/{total_lb_40} = {100.0*lb_matched_to_cy_50/total_lb_40:.2f}% at <=5.0 mm)")
    print(f"  Matched (LB/AMe, CY) Pair Discrepancies (n={len(lb_matched_pairs_metrics)}):")
    print(f"    - Shaft XY : mean={inter_rater_pair_diffs['mean_shaft_xy_mm']['mean']:.3f}mm, med={inter_rater_pair_diffs['mean_shaft_xy_mm']['median']:.3f}mm, IQR={inter_rater_pair_diffs['mean_shaft_xy_mm']['iqr']:.3f}mm, P95={inter_rater_pair_diffs['mean_shaft_xy_mm']['p95']:.3f}mm")
    print(f"    - Tip 3D   : mean={inter_rater_pair_diffs['tip_3d_mm']['mean']:.2f}mm, med={inter_rater_pair_diffs['tip_3d_mm']['median']:.2f}mm, IQR={inter_rater_pair_diffs['tip_3d_mm']['iqr']:.2f}mm, P95={inter_rater_pair_diffs['tip_3d_mm']['p95']:.2f}mm (>5mm: {inter_rater_pair_diffs['n_tip_diff_gt_5mm']}, >10mm: {inter_rater_pair_diffs['n_tip_diff_gt_10mm']})")
    print(f"    - Signed dz_tip (LB - CY): mean={inter_rater_pair_diffs['dz_tip_signed_lb_minus_cy_mm']['mean']:+.2f}mm, med={inter_rater_pair_diffs['dz_tip_signed_lb_minus_cy_mm']['median']:+.2f}mm")
    print(f"  Breakdown of Unmatched LB/AMe Traces in the {mapped_stats['n_mapped_scenes']} Mapped Benchmark Cases (n_lb={mapped_stats['n_lb_ame_in_mapped']}, matched_cy_35={mapped_stats['lb_matched_to_cy_long_35']}):")
    print(f"    - Match HT short obturator <= 3.5 mm   : {mapped_stats['lb_extra_matched_to_ht_short_35']}")
    print(f"    - Borderline (3.5 - 6.0 mm) to CY / HT : {mapped_stats['lb_extra_borderline_to_cy_or_ht_35_to_60']}")
    print(f"    - Truly extra catheters (> 6.0 mm)     : {mapped_stats['lb_truly_extra_new_catheters_gt_60']} (of which {mapped_stats['lb_truly_extra_supported_by_nf_candidate_pool_35']} are already in NF candidate pool <=3.5mm!)")

    return {
        "k_lim_provenance_and_reference_free_audit": {
            "n_cases_total": 54,
            "n_cases_with_valid_acsv_marker": n_acsv_valid,
            "n_cases_using_gt_base_ks_fallback": n_fallback_gt,
            "unclipped_vtk_vs_acsv_audit": unclipped_vtk_stats,
            "policy_comparison_54cases": policy_stats,
            "per_case_k_lim_54": per_case_klim,
        },
        "dual_rater_lb_ame_training_label_audit_40cases": {
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
            "mapped_39_scenes_extra_trace_breakdown": mapped_stats,
            "per_dual_scene_records": per_dual_scene_records,
        },
    }


def main():
    t0 = time.time()
    meta_list = pickle.load(open(META_PATH, "rb"))
    cache_d = pickle.load(open(ENRICHED_BUNDLES_CACHE_PATH, "rb"))
    bundle_items = cache_d["bundle_items"]
    base_oof_scores = cache_d["base_oof_scores"]
    ens_oof_outputs = pickle.load(open(ENS_SYNTH_CACHE_PATH, "rb"))

    res_task1 = run_task1_failure_analysis(meta_list, bundle_items, base_oof_scores, ens_oof_outputs)
    res_task2 = run_task2_klim_and_dual_rater_audit(meta_list, bundle_items)

    out_report = {
        "generated_at": "2026-10-23",
        "script": "vault/paper/benchmarks_rerun/run_failure_analysis.py",
        "elapsed_s": round(time.time() - t0, 2),
        "task1_failure_analysis": res_task1,
        "task2_klim_and_dual_rater_audit": res_task2,
    }

    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(out_report, f, indent=2)
    print(f"\n[DONE] Saved complete failure analysis & audit JSON to {OUT_JSON_PATH} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
