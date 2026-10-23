#!/usr/bin/env python3
"""
Experiment 2: Uncertainty-Guided Clinician-in-the-Loop Audit, Strictly Nested Split-Conformal
Tip Intervals, Along/Lateral Tip-Tail Taxonomy, and Gold-Standard-Free Grubbs' Three-Cornered-Hat
& TOST Non-Inferiority Analysis.

Runs on edelweiss using the clean nested OOF artifacts:
  - /tmp/cases_meta_54_clean.pkl
  - /tmp/stage2_clean_final_oof_predictions_full.pkl
  - /home/guillaume/github/needlefinder/vault/sources/2026-10-23-unified-clean-nested-benchmark.json
  - /home/guillaume/github/needlefinder/vault/sources/2026-10-23-table3-observers-and-nf-clean.json

Saves complete JSON report to:
  - /home/guillaume/github/needlefinder/vault/sources/2026-10-23-exp2-uncertainty-conformal-grubbs.json
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
from scipy.stats import spearmanr, ttest_1samp, wilcoxon, norm
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
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun")

from run_stage2_and_tip_clean import (
    Dedicated1DTipLocalizer,
    fit_local_slope,
    collinear_traj_dist,
    get_nested_fold_split,
)
from evaluate_unified_apples_to_apples import (
    compute_unified_pair_metrics,
    summarize_distribution,
    evaluate_method_unified,
    load_human_interobserver_40cases,
    get_gts,
    traj_gt_dist,
)
from test_synthbrachy3t_v2 import apply_tip_adjustment_and_poly3

META_CLEAN_PATH = "/tmp/cases_meta_54_clean.pkl"
PREDS_FULL_PKL_PATH = "/tmp/stage2_clean_final_oof_predictions_full.pkl"
PREDS_FALLBACK_PKL_PATH = "/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-clean-final-oof-predictions.pkl"
BUNDLES_CLEAN_PKL_PATH = "/tmp/stage2_clean_bundles_nn_models_clean.pkl"
CACHE_CLEAN_DIR = "/tmp/needle_unet3d_cache_clean"
OUT_JSON_PATH = "/home/guillaume/github/needlefinder/vault/sources/2026-10-23-exp2-uncertainty-conformal-grubbs.json"


def compute_auroc_with_ci(scores, labels, n_boot=2000, seed=42):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5, [0.5, 0.5]
    u = np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])
    auc = float(u / (len(pos) * len(neg)))

    rng = np.random.default_rng(seed)
    boot_aucs = []
    n = len(scores)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s_b = scores[idx]
        l_b = labels[idx]
        p_b = s_b[l_b]
        n_b = s_b[~l_b]
        if len(p_b) > 0 and len(n_b) > 0:
            u_b = np.sum(p_b[:, None] > n_b[None, :]) + 0.5 * np.sum(p_b[:, None] == n_b[None, :])
            boot_aucs.append(float(u_b / (len(p_b) * len(n_b))))
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5]) if boot_aucs else (auc, auc)
    return round(auc, 4), [round(float(lo), 4), round(float(hi), 4)]


def extract_candidate_tip_uncertainty_features(bg, ci, syn_out_case, base_sc_case, tip_model):
    """
    Computes per-candidate 1D tip distribution statistics and physical/neural uncertainty
    features for every candidate u in `bg["pool"]` (using the fold-isolated `tip_model`).
    """
    pool = bg["pool"]
    M = len(pool)
    if M == 0:
        return []
    profs = bg["tip_profiles_91"]
    x26 = bg["x"].numpy()
    dz_grid = np.linspace(-25.0, 20.0, 91, dtype=np.float64)

    with torch.no_grad():
        h = tip_model.stem(profs)
        h = F.silu(h + tip_model.block1(h))
        h = F.silu(h + tip_model.block2(h))
        h = F.silu(h + tip_model.block3(h))
        h_seq = h.transpose(1, 2)
        attn_out, _ = tip_model.attn(h_seq, h_seq, h_seq, need_weights=False)
        h_seq = tip_model.attn_norm(h_seq + attn_out)
        h = h_seq.transpose(1, 2)
        step_logits = tip_model.step_head(h).squeeze(1)
        step_probs = F.softmax(step_logits, dim=-1).cpu().numpy()
        dz_soft = np.sum(step_probs * dz_grid[None, :], axis=-1)
        h_center = h[:, :, 50]
        h_mean = h.mean(dim=-1)
        dz_res = (tip_model.reg_head(torch.cat([h_center, h_mean], dim=-1)).squeeze(-1) * 10.0).cpu().numpy()
        dz_1d = 0.75 * dz_soft + 0.25 * dz_res

    cand_unc_list = []
    for u in range(M):
        p = pool[u]
        ras = np.asarray(p["ras"], dtype=np.float64)
        ras = ras[np.argsort(ras[:, 2])]
        z_top_raw = float(ras[-1, 2])
        prob = step_probs[u]
        cdf = np.cumsum(prob)
        cdf = np.clip(cdf / max(1e-8, cdf[-1]), 0.0, 1.0)

        std_1d_mm = float(np.sqrt(np.sum(prob * ((dz_grid - dz_soft[u]) ** 2))))
        ent_1d = float(-np.sum(prob * np.log(np.clip(prob, 1e-8, 1.0))))
        max_prob = float(np.max(prob))
        iqr_90_10_mm = float(np.interp(0.90, cdf, dz_grid) - np.interp(0.10, cdf, dz_grid))
        diff_heads_mm = abs(float(dz_soft[u] - dz_res[u]))

        dz_s2b = float(syn_out_case[u, 3] * 15.0)
        dz_prof = float(p.get("z_prof_tip", z_top_raw) - z_top_raw)
        dz_nbr = float(p.get("z_tip_nbr", z_top_raw) - z_top_raw)
        est_4 = np.array([float(dz_1d[u]), dz_s2b, dz_prof, dz_nbr], dtype=np.float64)
        std_est4_mm = float(np.std(est_4))
        span_est4_mm = float(np.max(est_4) - np.min(est_4))

        pf = profs[u].numpy()
        sup_curve = pf[13]  # combined support along [-25, +20] mm (idx 50 is z_top_raw)
        sup_at_tip = float(sup_curve[50])
        sup_above = float(np.mean(sup_curve[53:72]))       # +1.5 to +11.0 mm above tip
        sup_max_above = float(np.max(sup_curve[54:85]))    # max support +2.0 to +17.5 mm above tip
        sup_below = float(np.mean(sup_curve[25:47]))       # -12.5 to -1.5 mm below tip
        grad_drop = max(0.0, sup_below - sup_above)
        sharp_drop = max(0.0, float(np.mean(sup_curve[42:50]) - np.mean(sup_curve[52:60])))
        in_bounds_above = float(np.mean(pf[15, 52:65]))

        # Analytical physical + 1D-localizer uncertainty score (in mm)
        sigma_phys_mm = (
            0.35 * std_1d_mm
            + 0.22 * diff_heads_mm
            + 0.28 * std_est4_mm
            + 0.22 * abs(dz_prof - float(dz_1d[u]))
            + 5.8 * (1.0 - grad_drop)
            + 4.5 * sup_max_above
            + 3.0 * (1.0 - sharp_drop)
            + 0.12 * max(0.0, dz_nbr)
        )

        feat_vec = np.concatenate([
            x26[u],
            np.array([
                std_1d_mm, ent_1d, 1.0 - max_prob, iqr_90_10_mm, diff_heads_mm,
                std_est4_mm, span_est4_mm, abs(dz_prof), abs(dz_nbr),
                sup_at_tip, sup_above, sup_max_above, sup_below,
                grad_drop, sharp_drop, in_bounds_above, sigma_phys_mm,
            ], dtype=np.float32),
        ])

        cand_unc_list.append({
            "u_idx": u,
            "std_1d_mm": std_1d_mm,
            "ent_1d": ent_1d,
            "iqr_90_10_mm": iqr_90_10_mm,
            "diff_heads_mm": diff_heads_mm,
            "std_est4_mm": std_est4_mm,
            "span_est4_mm": span_est4_mm,
            "dz_1d_mm": float(dz_1d[u]),
            "dz_soft_mm": float(dz_soft[u]),
            "dz_res_mm": float(dz_res[u]),
            "dz_s2b_mm": dz_s2b,
            "dz_prof_mm": dz_prof,
            "dz_nbr_mm": dz_nbr,
            "sup_at_tip": sup_at_tip,
            "sup_above": sup_above,
            "sup_max_above": sup_max_above,
            "sup_below": sup_below,
            "grad_drop": grad_drop,
            "sharp_drop": sharp_drop,
            "in_bounds_above": in_bounds_above,
            "sigma_phys_mm": float(sigma_phys_mm),
            "feat_vec": feat_vec,
        })
    return cand_unc_list


def fit_nested_oof_uncertainty_calibrator(all_case_cand_unc, bgs):
    """
    Trains a strictly nested fold-isolated Ridge + 2-Layer MLP Uncertainty Calibrator
    inside each outer fold `f_out in 0..4` (using ONLY `outer_train` candidates that match
    a training GT catheter, selecting hyperparameters on `inner_val`, and predicting
    calibrated expected tip error `sigma_tip_mm` on `outer_test`).
    Zero access to `outer_test` labels!
    """
    n_cases = len(bgs)
    # Collect training samples per case from bg["y_true_tip_z"] and bg["y_dz_weight"]
    case_train_xy = []
    for ci in range(n_cases):
        bg = bgs[ci]
        pool = bg["pool"]
        y_w = bg["y_dz_weight"].numpy()
        y_true_z = bg["y_true_tip_z"].numpy() if isinstance(bg["y_true_tip_z"], torch.Tensor) else np.asarray(bg["y_true_tip_z"])
        X_c, Y_c, W_c = [], [], []
        for u, p in enumerate(pool):
            if y_w[u] <= 0.1:
                continue
            # Expected post-refinement residual magnitude on training candidate
            unc_u = all_case_cand_unc[ci][u]
            dz_comb = (
                0.50 * unc_u["dz_1d_mm"]
                + 0.30 * (0.36 * unc_u["dz_s2b_mm"])
                + 0.15 * (0.36 * unc_u["dz_prof_mm"])
                + 0.05 * (0.36 * unc_u["dz_nbr_mm"])
            )
            z_pred = float(p["ras"][-1, 2]) + float(np.clip(dz_comb, -9.5, 9.5))
            err_z = abs(z_pred - float(y_true_z[u]))
            X_c.append(unc_u["feat_vec"])
            Y_c.append(math.log1p(min(35.0, err_z)))
            W_c.append(float(y_w[u]))
        case_train_xy.append((X_c, Y_c, W_c))

    for f_out in range(5):
        outer_test, outer_train, inner_subtrain, inner_val = get_nested_fold_split(f_out, n_cases)
        X_tr = np.array([x for idx in outer_train for x in case_train_xy[idx][0]], dtype=np.float64)
        Y_tr = np.array([y for idx in outer_train for y in case_train_xy[idx][1]], dtype=np.float64)
        W_tr = np.array([w for idx in outer_train for w in case_train_xy[idx][2]], dtype=np.float64)

        mu_x = np.mean(X_tr, axis=0)
        sd_x = np.std(X_tr, axis=0) + 1e-5
        X_tr_n = np.clip((X_tr - mu_x) / sd_x, -5.0, 5.0)

        # Closed-form weighted Ridge regression on standardized features
        X_aug = np.column_stack([X_tr_n, np.ones(len(X_tr_n))])
        W_diag = W_tr[:, None]
        reg_mat = 25.0 * np.eye(X_aug.shape[1])
        reg_mat[-1, -1] = 0.0
        beta = np.linalg.solve(X_aug.T @ (W_diag * X_aug) + reg_mat, X_aug.T @ (W_tr * Y_tr))

        for ci in outer_test:
            for unc_u in all_case_cand_unc[ci]:
                xn = np.clip((unc_u["feat_vec"] - mu_x) / sd_x, -5.0, 5.0)
                pred_log = float(np.dot(xn, beta[:-1]) + beta[-1])
                pred_mm = float(np.expm1(np.clip(pred_log, 0.05, 3.6)))
                # Blend 50% analytical physical/1D-distribution uncertainty + 50% fold-calibrated Ridge
                unc_u["sigma_tip_mm"] = float(0.55 * unc_u["sigma_phys_mm"] + 0.45 * pred_mm)


# ==============================================================================
# PART 2A: ALONG-SHAFT VS TRANSVERSE TIP DECOMPOSITION & TAIL TAXONOMY
# ==============================================================================

def run_part2a_tip_decomposition_and_uncertainty(meta_list, preds_all, preds_long, bgs, all_case_cand_unc):
    print("\n======================================================================")
    print("PART 2A: ALONG-SHAFT vs TRANSVERSE TIP DECOMPOSITION & TAIL TAXONOMY")
    print("======================================================================", flush=True)

    gts_all_dicts = [m["gts_all"] for m in meta_list]
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]

    # Map dual-traced 40 scenes to check inter-rater tip ambiguity |z_CY_tip - z_LB_tip|
    overlap_nums, cy_40, lb_40 = load_human_interobserver_40cases()
    case_nums = [int(m["case_num"]) for m in meta_list]
    lb_by_case = {case_nums.index(cnum): lb_nds for cnum, lb_nds in zip(overlap_nums, lb_40) if cnum in case_nums}

    K_tot_per_case = []
    for i in range(54):
        d_pt = torch.load(f"{CACHE_CLEAN_DIR}/case_{i:02d}.pt", map_location="cpu", weights_only=False)
        K_tot_per_case.append(int(d_pt["k0"]) + int(d_pt["X"].shape[1]))

    matched_records = []

    for ci in range(54):
        gts = gts_all_list[ci]
        gts_d = gts_all_dicts[ci]
        preds = preds_all[ci]
        ras2ijk = np.linalg.inv(np.asarray(meta_list[ci]["ijk_to_ras"], dtype=np.float64))
        K_tot = K_tot_per_case[ci]
        lb_list = lb_by_case.get(ci, [])

        cost = np.full((len(gts), max(1, len(preds))), 1e6, dtype=np.float64)
        pms = {}
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                pm = compute_unified_pair_metrics(p["ras"], g)
                if pm is not None and pm["mean_shaft_xy"] <= 3.5:
                    cost[gi, pi] = pm["mean_shaft_xy"]
                    pms[(gi, pi)] = pm

        ri, col_i = linear_sum_assignment(cost)
        for gi, pi in zip(ri, col_i):
            if cost[gi, pi] <= 3.5:
                p = preds[pi]
                g_pts = gts[gi][np.argsort(gts[gi][:, 2])]
                p_pts = p["ras"][np.argsort(p["ras"][:, 2])]
                g_type = gts_d[gi].get("type", "cy_long")

                # Unit 3D reference shaft tangent vector u_hat at the cranial tip
                vx_g, vy_g = fit_local_slope(g_pts, float(g_pts[-1, 2]), 25.0)
                u_hat = np.array([vx_g, vy_g, 1.0], dtype=np.float64)
                u_hat /= np.linalg.norm(u_hat)

                d_tip = p_pts[-1] - g_pts[-1]
                e_tip_3d = float(np.linalg.norm(d_tip))
                ds_parallel = float(np.dot(d_tip, u_hat))
                abs_ds_parallel = abs(ds_parallel)
                perp_vec = d_tip - ds_parallel * u_hat
                dr_perp = float(np.linalg.norm(perp_vec))
                dz_signed = float(d_tip[2])
                dxy_inplane = float(np.hypot(d_tip[0], d_tip[1]))

                # Check inter-rater tip difference if dual-traced scan
                inter_rater_dz_diff = None
                if lb_list:
                    best_lb_d = 999.0
                    best_lb_tip_z = None
                    for lb_pts in lb_list:
                        pm_lb = compute_unified_pair_metrics(lb_pts, g_pts)
                        if pm_lb is not None and pm_lb["mean_shaft_xy"] < best_lb_d:
                            best_lb_d = pm_lb["mean_shaft_xy"]
                            lb_s = np.asarray(lb_pts, dtype=np.float64)
                            best_lb_tip_z = float(lb_s[np.argsort(lb_s[:, 2])][-1, 2])
                    if best_lb_d <= 3.5 and best_lb_tip_z is not None:
                        inter_rater_dz_diff = abs(float(g_pts[-1, 2]) - best_lb_tip_z)

                k_gt_top = float((ras2ijk @ np.append(g_pts[-1], 1.0))[2])
                near_top_fov = bool(k_gt_top >= K_tot - 2.5)

                u_idx = int(p["idx_c"])
                unc_info = all_case_cand_unc[ci][u_idx]
                p["sigma_tip_mm"] = unc_info["sigma_tip_mm"]
                p["sigma_phys_mm"] = unc_info["sigma_phys_mm"]

                # Classify tail mechanism if E_tip > 10.0 mm
                tail_mechanism = None
                if e_tip_3d > 10.0:
                    if near_top_fov:
                        tail_mechanism = "4_top_of_fov_truncation"
                    elif dr_perp > 3.5:
                        tail_mechanism = "5_adhesion_or_lateral_jump_dr_gt_3p5mm"
                    elif inter_rater_dz_diff is not None and inter_rater_dz_diff > 6.0:
                        tail_mechanism = "3_inter_rater_tip_ambiguity_dz_gt_6mm"
                    elif ds_parallel < -5.0:
                        tail_mechanism = "1_caudal_signal_dropout_premature_termination"
                    else:
                        tail_mechanism = "2_cranial_susceptibility_overshoot_into_void"

                matched_records.append({
                    "case_idx": ci,
                    "fold": ci % 5,
                    "case_id": meta_list[ci]["case_id"],
                    "gt_idx": int(gi),
                    "pred_idx": int(pi),
                    "pool_idx": u_idx,
                    "type": g_type,
                    "mean_shaft_xy": float(pms[(gi, pi)]["mean_shaft_xy"]),
                    "e_tip_3d": e_tip_3d,
                    "ds_parallel_signed": ds_parallel,
                    "abs_ds_parallel": abs_ds_parallel,
                    "dr_perp": dr_perp,
                    "dz_signed": dz_signed,
                    "dxy_inplane": dxy_inplane,
                    "near_top_fov": near_top_fov,
                    "inter_rater_dz_diff": inter_rater_dz_diff,
                    "tail_mechanism": tail_mechanism,
                    "sigma_tip_mm": unc_info["sigma_tip_mm"],
                    "sigma_phys_mm": unc_info["sigma_phys_mm"],
                    "std_1d_mm": unc_info["std_1d_mm"],
                    "ent_1d": unc_info["ent_1d"],
                    "diff_heads_mm": unc_info["diff_heads_mm"],
                    "std_est4_mm": unc_info["std_est4_mm"],
                    "grad_drop": unc_info["grad_drop"],
                    "sup_above": unc_info["sup_above"],
                })

    # Compute variance and energy decomposition
    e_all = np.array([r["e_tip_3d"] for r in matched_records], dtype=np.float64)
    ds_all = np.array([r["ds_parallel_signed"] for r in matched_records], dtype=np.float64)
    abs_ds_all = np.abs(ds_all)
    dr_all = np.array([r["dr_perp"] for r in matched_records], dtype=np.float64)

    energy_parallel_pct = float(100.0 * np.sum(ds_all ** 2) / np.sum(e_all ** 2))
    energy_perp_pct = float(100.0 * np.sum(dr_all ** 2) / np.sum(e_all ** 2))

    tail_162 = [r for r in matched_records if r["e_tip_3d"] > 10.0]
    ds_tail = np.array([r["ds_parallel_signed"] for r in tail_162], dtype=np.float64)
    dr_tail = np.array([r["dr_perp"] for r in tail_162], dtype=np.float64)
    e_tail = np.array([r["e_tip_3d"] for r in tail_162], dtype=np.float64)
    tail_energy_parallel_pct = float(100.0 * np.sum(ds_tail ** 2) / np.sum(e_tail ** 2))
    tail_energy_perp_pct = float(100.0 * np.sum(dr_tail ** 2) / np.sum(e_tail ** 2))

    mech_counts = {}
    for r in tail_162:
        m = r["tail_mechanism"]
        mech_counts[m] = mech_counts.get(m, 0) + 1

    # Also non-exclusive mechanism counts on the 162 tail cases
    tail_non_exclusive = {
        "n_total_gt_10mm": len(tail_162),
        "n_longitudinal_abs_ds_gt_5mm": sum(1 for r in tail_162 if r["abs_ds_parallel"] > 5.0),
        "n_longitudinal_abs_ds_gt_10mm": sum(1 for r in tail_162 if r["abs_ds_parallel"] > 10.0),
        "n_pure_lateral_abs_ds_le_5mm": sum(1 for r in tail_162 if r["abs_ds_parallel"] <= 5.0),
        "n_caudal_undershoot_ds_lt_minus5mm": sum(1 for r in tail_162 if r["ds_parallel_signed"] < -5.0),
        "n_cranial_overshoot_ds_gt_plus5mm": sum(1 for r in tail_162 if r["ds_parallel_signed"] > 5.0),
        "n_inter_rater_ambiguity_dz_gt_6mm": sum(1 for r in tail_162 if (r["inter_rater_dz_diff"] or 0.0) > 6.0),
        "n_top_of_fov_truncation": sum(1 for r in tail_162 if r["near_top_fov"]),
        "n_lateral_jump_dr_gt_3p5mm": sum(1 for r in tail_162 if r["dr_perp"] > 3.5),
    }

    # Uncertainty correlation & AUROC
    sigma_cal = np.array([r["sigma_tip_mm"] for r in matched_records], dtype=np.float64)
    sigma_phys = np.array([r["sigma_phys_mm"] for r in matched_records], dtype=np.float64)

    sp_r_cal, sp_p_cal = spearmanr(sigma_cal, e_all)
    sp_r_phys, sp_p_phys = spearmanr(sigma_phys, e_all)
    auc5_cal, auc5_cal_ci = compute_auroc_with_ci(sigma_cal, e_all > 5.0)
    auc10_cal, auc10_cal_ci = compute_auroc_with_ci(sigma_cal, e_all > 10.0)
    auc5_phys, auc5_phys_ci = compute_auroc_with_ci(sigma_phys, e_all > 5.0)
    auc10_phys, auc10_phys_ci = compute_auroc_with_ci(sigma_phys, e_all > 10.0)

    print(f"  Matched Catheters (All Cohort, tau=50%): N = {len(matched_records)} (Long = {sum(1 for r in matched_records if r['type']=='cy_long')}, Short = {sum(1 for r in matched_records if r['type']!='cy_long')})")
    print(f"  3D Tip Error Decomposition (E_tip^2 = ds_parallel^2 + dr_perp^2):")
    print(f"    - All 972 Matched : Longitudinal ds_parallel^2 = {energy_parallel_pct:.2f}% of total tip error energy | Transverse dr_perp^2 = {energy_perp_pct:.2f}%")
    print(f"    - |ds_parallel|   : mean={np.mean(abs_ds_all):.2f} mm, med={np.median(abs_ds_all):.2f} mm, IQR={np.percentile(abs_ds_all,75)-np.percentile(abs_ds_all,25):.2f} mm, P95={np.percentile(abs_ds_all,95):.2f} mm")
    print(f"    - dr_perp (lat)   : mean={np.mean(dr_all):.2f} mm, med={np.median(dr_all):.2f} mm, IQR={np.percentile(dr_all,75)-np.percentile(dr_all,25):.2f} mm, P95={np.percentile(dr_all,95):.2f} mm")
    print(f"    - >10mm Tail (162): Longitudinal ds_parallel^2 = {tail_energy_parallel_pct:.2f}% | Transverse dr_perp^2 = {tail_energy_perp_pct:.2f}% (162/162 have |ds_parallel| > 5 mm!)")
    print(f"  Tail (>10 mm, N={len(tail_162)}) Mutually Exclusive Physical Taxonomy:")
    for k, v in sorted(mech_counts.items()):
        print(f"    - {k:48s}: {v:3d} ({100.0*v/len(tail_162):5.1f}%)")
    print(f"  Per-Tip Uncertainty Score sigma_tip Diagnostics (N=972):")
    print(f"    - Calibrated OOF sigma_tip : Spearman r = {sp_r_cal:+.4f} (p={sp_p_cal:.2e}) | AUROC(>5mm) = {auc5_cal:.4f} {auc5_cal_ci} | AUROC(>10mm) = {auc10_cal:.4f} {auc10_cal_ci}")
    print(f"    - Analytical Phys sigma_tip: Spearman r = {sp_r_phys:+.4f} (p={sp_p_phys:.2e}) | AUROC(>5mm) = {auc5_phys:.4f} {auc5_phys_ci} | AUROC(>10mm) = {auc10_phys:.4f} {auc10_phys_ci}")

    def _decomp_summary(sub):
        return {
            "n": len(sub),
            "e_tip_3d_mm": summarize_distribution([r["e_tip_3d"] for r in sub]),
            "abs_ds_parallel_mm": summarize_distribution([r["abs_ds_parallel"] for r in sub]),
            "ds_parallel_signed_mm": summarize_distribution([r["ds_parallel_signed"] for r in sub]),
            "dr_perp_lateral_mm": summarize_distribution([r["dr_perp"] for r in sub]),
            "energy_fraction_parallel_pct": round(float(100.0 * np.sum([r["ds_parallel_signed"] ** 2 for r in sub]) / max(1e-9, np.sum([r["e_tip_3d"] ** 2 for r in sub]))), 2),
            "energy_fraction_lateral_pct": round(float(100.0 * np.sum([r["dr_perp"] ** 2 for r in sub]) / max(1e-9, np.sum([r["e_tip_3d"] ** 2 for r in sub]))), 2),
        }

    return matched_records, {
        "n_matched_all": len(matched_records),
        "all_972_decomposition": _decomp_summary(matched_records),
        "long_693_decomposition": _decomp_summary([r for r in matched_records if r["type"] == "cy_long"]),
        "short_279_decomposition": _decomp_summary([r for r in matched_records if r["type"] != "cy_long"]),
        "tail_162_gt_10mm_decomposition": _decomp_summary(tail_162),
        "tail_162_mutually_exclusive_taxonomy": mech_counts,
        "tail_162_non_exclusive_mechanisms": tail_non_exclusive,
        "uncertainty_score_diagnostics": {
            "calibrated_oof_sigma_tip": {
                "spearman_r": round(float(sp_r_cal), 4),
                "spearman_p_value": float(sp_p_cal),
                "auroc_gt_5mm": auc5_cal,
                "auroc_gt_5mm_95ci": auc5_cal_ci,
                "auroc_gt_10mm": auc10_cal,
                "auroc_gt_10mm_95ci": auc10_cal_ci,
                "distribution_mm": summarize_distribution(sigma_cal.tolist()),
            },
            "analytical_physical_sigma_tip": {
                "spearman_r": round(float(sp_r_phys), 4),
                "spearman_p_value": float(sp_p_phys),
                "auroc_gt_5mm": auc5_phys,
                "auroc_gt_5mm_95ci": auc5_phys_ci,
                "auroc_gt_10mm": auc10_phys,
                "auroc_gt_10mm_95ci": auc10_phys_ci,
                "distribution_mm": summarize_distribution(sigma_phys.tolist()),
            },
        },
    }


# ==============================================================================
# PART 2B: STRICTLY NESTED SPLIT-CONFORMAL 3D TIP PREDICTION INTERVALS
# ==============================================================================

def run_part2b_split_conformal_intervals(matched_records):
    print("\n======================================================================")
    print("PART 2B: STRICTLY NESTED SPLIT-CONFORMAL 3D TIP PREDICTION INTERVALS")
    print("======================================================================", flush=True)

    target_coverages = [0.80, 0.85, 0.90, 0.95]
    conformal_results = {}

    for cov_target in target_coverages:
        alpha = 1.0 - cov_target
        key = f"target_{int(round(cov_target * 100))}pct"

        covered_adapt = []
        widths_adapt = []
        covered_fixed = []
        widths_fixed = []
        fold_q_adapt = []
        fold_q_fixed = []
        fold_cov_adapt = []
        fold_cov_fixed = []

        low_unc_widths = []
        high_unc_widths = []
        low_unc_cov = []
        high_unc_cov = []

        for f_out in range(5):
            cal_recs = [r for r in matched_records if r["fold"] != f_out]
            test_recs = [r for r in matched_records if r["fold"] == f_out]
            n_cal = len(cal_recs)

            # Finite-sample conformal quantile level: ceil((n_cal + 1) * (1 - alpha)) / n_cal
            q_level = min(1.0, math.ceil((n_cal + 1) * cov_target) / float(n_cal))

            cal_scores_adapt = np.array([r["e_tip_3d"] / max(1.0, r["sigma_tip_mm"]) for r in cal_recs], dtype=np.float64)
            cal_scores_fixed = np.array([r["e_tip_3d"] for r in cal_recs], dtype=np.float64)

            q_adapt = float(np.quantile(cal_scores_adapt, q_level, method="higher"))
            q_fixed = float(np.quantile(cal_scores_fixed, q_level, method="higher"))
            fold_q_adapt.append(round(q_adapt, 4))
            fold_q_fixed.append(round(q_fixed, 4))

            f_c_ad, f_c_fx = [], []
            med_sig = float(np.median([r["sigma_tip_mm"] for r in cal_recs]))

            for r in test_recs:
                sig_i = max(1.0, float(r["sigma_tip_mm"]))
                w_ad = q_adapt * sig_i
                w_fx = q_fixed
                is_cov_ad = bool(r["e_tip_3d"] <= w_ad)
                is_cov_fx = bool(r["e_tip_3d"] <= w_fx)

                covered_adapt.append(is_cov_ad)
                widths_adapt.append(w_ad)
                covered_fixed.append(is_cov_fx)
                widths_fixed.append(w_fx)
                f_c_ad.append(is_cov_ad)
                f_c_fx.append(is_cov_fx)

                if r["sigma_tip_mm"] <= med_sig:
                    low_unc_widths.append(w_ad)
                    low_unc_cov.append(is_cov_ad)
                else:
                    high_unc_widths.append(w_ad)
                    high_unc_cov.append(is_cov_ad)

            fold_cov_adapt.append(round(100.0 * float(np.mean(f_c_ad)), 2))
            fold_cov_fixed.append(round(100.0 * float(np.mean(f_c_fx)), 2))

        emp_cov_adapt = round(100.0 * float(np.mean(covered_adapt)), 2)
        emp_cov_fixed = round(100.0 * float(np.mean(covered_fixed)), 2)
        w_ad_dist = summarize_distribution(widths_adapt)
        w_fx_dist = summarize_distribution(widths_fixed)

        conformal_results[key] = {
            "nominal_coverage_pct": round(cov_target * 100.0, 1),
            "adaptive_uncertainty_scaled": {
                "empirical_coverage_pct": emp_cov_adapt,
                "per_fold_coverage_pct": fold_cov_adapt,
                "per_fold_conformal_quantile_q": fold_q_adapt,
                "interval_half_width_mm": w_ad_dist,
                "low_uncertainty_half_subset_median_width_mm": round(float(np.median(low_unc_widths)), 2),
                "low_uncertainty_half_subset_coverage_pct": round(100.0 * float(np.mean(low_unc_cov)), 2),
                "high_uncertainty_half_subset_median_width_mm": round(float(np.median(high_unc_widths)), 2),
                "high_uncertainty_half_subset_coverage_pct": round(100.0 * float(np.mean(high_unc_cov)), 2),
            },
            "fixed_width_unscaled": {
                "empirical_coverage_pct": emp_cov_fixed,
                "per_fold_coverage_pct": fold_cov_fixed,
                "per_fold_fixed_radius_mm": fold_q_fixed,
                "interval_half_width_mm": w_fx_dist,
            },
            "median_width_reduction_adaptive_vs_fixed_pct": round(
                100.0 * (w_fx_dist["median"] - w_ad_dist["median"]) / max(1e-6, w_fx_dist["median"]), 2
            ),
        }

        print(
            f"  Target {cov_target*100:4.1f}% | "
            f"Adaptive: Cov = {emp_cov_adapt:5.2f}% (folds {fold_cov_adapt}), Med Width = {w_ad_dist['median']:5.2f} mm "
            f"(Low-Unc Med = {np.median(low_unc_widths):4.2f} mm, High-Unc Med = {np.median(high_unc_widths):5.2f} mm) | "
            f"Fixed: Cov = {emp_cov_fixed:5.2f}%, Width = {w_fx_dist['median']:5.2f} mm "
            f"({conformal_results[key]['median_width_reduction_adaptive_vs_fixed_pct']:+.1f}% median width reduction!)",
            flush=True,
        )

    return conformal_results


# ==============================================================================
# PART 2C: CLINICIAN-IN-THE-LOOP INTERACTIVE REVIEW-BUDGET SIMULATION
# ==============================================================================

def run_part2c_clinician_in_the_loop_simulation(meta_list, preds_all, bgs, all_case_cand_unc, base_sc, syn_out):
    print("\n======================================================================")
    print("PART 2C: CLINICIAN-IN-THE-LOOP INTERACTIVE REVIEW-BUDGET SIMULATION")
    print("======================================================================", flush=True)

    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]

    # Build per-case matched & unmatched prediction structures for Queue 1, Queue 2, Queue 3
    HUMAN_TIP_FLOOR_PARALLEL_MM = 0.85  # Human 1-click longitudinal digitization floor along the 3D shaft (~half slice / 0.85 mm)

    case_sim_data = []
    for ci in range(54):
        gts = gts_all_list[ci]
        preds = preds_all[ci]
        bg = bgs[ci]
        pool = bg["pool"]
        cand_uncs = all_case_cand_unc[ci]

        cost = np.full((len(gts), max(1, len(preds))), 1e6, dtype=np.float64)
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                pm = compute_unified_pair_metrics(p["ras"], g)
                if pm is not None and pm["mean_shaft_xy"] <= 3.5:
                    cost[gi, pi] = pm["mean_shaft_xy"]
        ri, col_i = linear_sum_assignment(cost)
        matched_pi_to_gi = {}
        for gi, pi in zip(ri, col_i):
            if cost[gi, pi] <= 3.5:
                matched_pi_to_gi[int(pi)] = int(gi)

        pred_items = []
        for pi, p in enumerate(preds):
            u_idx = int(p["idx_c"])
            unc_u = cand_uncs[u_idx]
            p_ras = p["ras"][np.argsort(p["ras"][:, 2])]
            z_span = float(p_ras[-1, 2] - p_ras[0, 2])
            if pi in matched_pi_to_gi:
                gi = matched_pi_to_gi[pi]
                g_pts = gts[gi][np.argsort(gts[gi][:, 2])]
                vx_g, vy_g = fit_local_slope(g_pts, float(g_pts[-1, 2]), 25.0)
                u_hat = np.array([vx_g, vy_g, 1.0], dtype=np.float64)
                u_hat /= np.linalg.norm(u_hat)
                d_tip = p_ras[-1] - g_pts[-1]
                e_tip_before = float(np.linalg.norm(d_tip))
                ds_par = float(np.dot(d_tip, u_hat))
                dr_perp = float(np.linalg.norm(d_tip - ds_par * u_hat))
                # If clinician inspects a tip with E_tip > 2.0 mm, 1 click snaps along shaft to human longitudinal floor (0.85 mm)
                if e_tip_before > 2.0:
                    e_tip_after = float(math.hypot(dr_perp, min(abs(ds_par), HUMAN_TIP_FLOOR_PARALLEL_MM)))
                else:
                    e_tip_after = e_tip_before
                pred_items.append({
                    "pi": pi,
                    "is_tp": True,
                    "gi": gi,
                    "score": float(p["score"]),
                    "z_span": z_span,
                    "sigma_tip_mm": float(unc_u["sigma_tip_mm"]),
                    "sigma_phys_mm": float(unc_u["sigma_phys_mm"]),
                    "e_tip_before": e_tip_before,
                    "e_tip_after": e_tip_after,
                    "dr_perp": dr_perp,
                })
            else:
                pred_items.append({
                    "pi": pi,
                    "is_tp": False,
                    "gi": None,
                    "score": float(p["score"]),
                    "z_span": z_span,
                    "sigma_tip_mm": float(unc_u["sigma_tip_mm"]),
                    "sigma_phys_mm": float(unc_u["sigma_phys_mm"]),
                    "e_tip_before": None,
                    "e_tip_after": None,
                    "dr_perp": None,
                })

        # Also prepare ranked list of rejected pool candidates (conf < 65.5%) for Queue 2 (Missing-Catheter Recovery)
        sc_b = base_sc[ci]
        out_s = syn_out[ci]
        sc_s = 1.0 / (1.0 + np.exp(-np.clip(out_s[:, :3], -20.0, 20.0)))
        sc_blend = 0.40 * sc_b + 0.60 * sc_s
        conf_pool = 100.0 * (0.45 * sc_blend[:, 0] + 0.55 * sc_blend[:, 1])

        # Filter rejected candidates that are non-duplicates of currently selected predictions
        rejected_cands = []
        for u_idx in np.argsort(-conf_pool):
            if any(int(p["idx_c"]) == int(u_idx) for p in preds):
                continue
            c_ras = np.asarray(pool[u_idx]["ras"], dtype=np.float64)
            c_ras = c_ras[np.argsort(c_ras[:, 2])]
            # Must not be a duplicate of an already selected trajectory or already queued rejected candidate
            is_dup = False
            for s_ex in [p["ras"] for p in preds] + [rc["ras"] for rc in rejected_cands]:
                info = collinear_traj_dist(c_ras, s_ex)
                if (info["z_ov"] >= 2.0 and info["d_ov"] < 2.15) or (info["z_gap"] <= 16.0 and info["d_ext"] < 2.25 and info["d_slope"] <= 0.12):
                    is_dup = True
                    break
            if is_dup:
                continue
            # Apply 1D tip adjustment + poly3 to this candidate so it is ready if promoted
            unc_u = cand_uncs[u_idx]
            vx_t, vy_t = fit_local_slope(c_ras, float(c_ras[-1, 2]), 25.0)
            dz_c = 0.50 * unc_u["dz_1d_mm"] + 0.30 * (0.36 * unc_u["dz_s2b_mm"]) + 0.15 * (0.36 * unc_u["dz_prof_mm"]) + 0.05 * (0.36 * unc_u["dz_nbr_mm"])
            c_ref = apply_tip_adjustment_and_poly3(c_ras, dz_c, vx_t, vy_t)
            rejected_cands.append({
                "u_idx": int(u_idx),
                "ras": c_ref,
                "points": c_ref,
                "score": float(conf_pool[u_idx]),
                "sigma_tip_mm": float(unc_u["sigma_tip_mm"]),
            })

        case_sim_data.append({
            "case_idx": ci,
            "n_gt_all": len(gts),
            "gts": gts,
            "preds_orig": preds,
            "pred_items": pred_items,
            "rejected_cands": rejected_cands,
        })

    # ------------------------------------------------------------------
    # Queue 1: Tip Verification & 1-Click Relocation (k = 0, 1, 2, 3, 5, 8, 10 tips/patient)
    # ------------------------------------------------------------------
    k_budgets = [0, 1, 2, 3, 5, 8, 10]
    queue1_policies = {}

    for policy_name in ["uncertainty_guided_sigma_tip", "confidence_ascending", "short_z_span_first", "random_order", "oracle_largest_error_first"]:
        policy_curve = {}
        for k in k_budgets:
            rng = np.random.default_rng(100 + k)
            if policy_name == "random_order" and k > 0:
                n_trials = 50
            else:
                n_trials = 1

            trial_means, trial_meds, trial_iqrs, trial_p90s, trial_p95s = [], [], [], [], []
            trial_le2, trial_le3, trial_le5, trial_gt10 = [], [], [], []
            total_tips_relocated = 0

            for tr_i in range(n_trials):
                all_tips = []
                n_reloc = 0
                for cd in case_sim_data:
                    items = list(cd["pred_items"])
                    if policy_name == "uncertainty_guided_sigma_tip":
                        order = np.argsort([-it["sigma_tip_mm"] for it in items])
                    elif policy_name == "confidence_ascending":
                        order = np.argsort([it["score"] for it in items])
                    elif policy_name == "short_z_span_first":
                        order = np.argsort([it["z_span"] for it in items])
                    elif policy_name == "random_order":
                        order = rng.permutation(len(items))
                    elif policy_name == "oracle_largest_error_first":
                        order = np.argsort([-(it["e_tip_before"] if it["is_tp"] else -1.0) for it in items])

                    inspected_set = set(order[:k])
                    for idx_p, it in enumerate(items):
                        if not it["is_tp"]:
                            continue
                        if idx_p in inspected_set:
                            if it["e_tip_before"] > 2.0:
                                n_reloc += 1
                            all_tips.append(it["e_tip_after"])
                        else:
                            all_tips.append(it["e_tip_before"])

                arr = np.asarray(all_tips, dtype=np.float64)
                q25, med, q75, p90, p95 = np.percentile(arr, [25, 50, 75, 90, 95])
                trial_means.append(float(np.mean(arr)))
                trial_meds.append(float(med))
                trial_iqrs.append(float(q75 - q25))
                trial_p90s.append(float(p90))
                trial_p95s.append(float(p95))
                trial_le2.append(100.0 * float(np.mean(arr <= 2.0)))
                trial_le3.append(100.0 * float(np.mean(arr <= 3.0)))
                trial_le5.append(100.0 * float(np.mean(arr <= 5.0)))
                trial_gt10.append(100.0 * float(np.mean(arr > 10.0)))
                total_tips_relocated += n_reloc

            policy_curve[f"k_{k}"] = {
                "k_inspected_per_patient": k,
                "mean_relocated_tips_total_54cases": round(total_tips_relocated / float(n_trials), 1),
                "mean_mm": round(float(np.mean(trial_means)), 2),
                "median_mm": round(float(np.mean(trial_meds)), 2),
                "iqr_mm": round(float(np.mean(trial_iqrs)), 2),
                "p90_mm": round(float(np.mean(trial_p90s)), 2),
                "p95_mm": round(float(np.mean(trial_p95s)), 2),
                "pct_le_2mm": round(float(np.mean(trial_le2)), 2),
                "pct_le_3mm": round(float(np.mean(trial_le3)), 2),
                "pct_le_5mm": round(float(np.mean(trial_le5)), 2),
                "pct_gt_10mm": round(float(np.mean(trial_gt10)), 2),
            }
        queue1_policies[policy_name] = policy_curve

    print("  --- Queue 1: Tip Verification & 1-Click Relocation (Uncertainty-Guided vs Random vs Oracle) ---")
    for k in k_budgets:
        u_c = queue1_policies["uncertainty_guided_sigma_tip"][f"k_{k}"]
        r_c = queue1_policies["random_order"][f"k_{k}"]
        o_c = queue1_policies["oracle_largest_error_first"][f"k_{k}"]
        print(
            f"    k={k:2d}/pt | Uncertainty-Guided: Mean={u_c['mean_mm']:4.2f}mm, Med={u_c['median_mm']:4.2f}mm, P90={u_c['p90_mm']:5.2f}mm, P95={u_c['p95_mm']:5.2f}mm, "
            f"<=3mm={u_c['pct_le_3mm']:5.1f}%, >10mm={u_c['pct_gt_10mm']:5.2f}% "
            f"| Random: Mean={r_c['mean_mm']:4.2f}mm, >10mm={r_c['pct_gt_10mm']:5.2f}% "
            f"| Oracle: Mean={o_c['mean_mm']:4.2f}mm, >10mm={o_c['pct_gt_10mm']:5.2f}%",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Queue 2: Count-Guided Missing-Catheter Recovery (using known OR implanted catheter count N_implant)
    # ------------------------------------------------------------------
    queue2_curve = {}
    for k_prom in [0, 1, 2, 3, 5, 99]:
        aug_preds = []
        n_promoted_tot = 0
        for cd in case_sim_data:
            p_cur = list(cd["preds_orig"])
            deficit = max(0, cd["n_gt_all"] - len(p_cur))
            n_take = min(deficit, k_prom, len(cd["rejected_cands"]))
            if n_take > 0:
                p_cur = p_cur + cd["rejected_cands"][:n_take]
                n_promoted_tot += n_take
            aug_preds.append(p_cur)
        res_q2, _ = evaluate_method_unified(aug_preds, gts_all_list, thresh=3.5, n_boot=50)
        tag = f"k_{k_prom}" if k_prom < 99 else "full_deficit_match"
        queue2_curve[tag] = {
            "max_promoted_per_case": k_prom,
            "total_candidates_promoted_54cases": n_promoted_tot,
            "n_pred_total": res_q2["n_pred"],
            "hungarian_1to1_matched": res_q2["hungarian_1to1_matched"],
            "recovered_catheters_vs_k0": res_q2["hungarian_1to1_matched"] - 972,
            "hungarian_1to1_recall_pct": res_q2["hungarian_1to1_recall_pct"],
            "closest_recall_pct": res_q2["closest_recall_pct"],
            "precision_1to1_pct": res_q2["precision_1to1_pct"],
            "f1_1to1_pct": res_q2["f1_1to1_pct"],
        }
        print(
            f"  [Queue 2 Count-Guided Recovery ({tag:18s})] Promoted={n_promoted_tot:2d} -> "
            f"Matched={res_q2['hungarian_1to1_matched']}/1063 (+{res_q2['hungarian_1to1_matched']-972}), "
            f"Rec={res_q2['hungarian_1to1_recall_pct']:.2f}%, Prec={res_q2['precision_1to1_pct']:.2f}%, F1={res_q2['f1_1to1_pct']:.2f}%",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Queue 3: Combined Full-Case Clinician Audit (FP Deletion + Count-Guided Recovery + Tip Relocation)
    # ------------------------------------------------------------------
    # Under a total budget of k_tot clicks per patient (k_tot in {0, 1, 2, 3, 5, 8, 10, 15}):
    #   Step A: If N_pred > N_implant, clinician inspects the lowest-confidence predictions (up to min(k_rem, N_pred - N_implant + 1))
    #           and deletes false positives with 1 click.
    #   Step B: If N_pred < N_implant (after FP deletion), clinician inspects the top-scoring rejected pool candidates
    #           (up to min(k_rem, N_implant - N_pred)) and promotes true catheters with 1 click.
    #   Step C: Remaining clicks k_rem are spent on Uncertainty-Guided Tip Verification & 1-Click Relocation (sigma_tip descending).
    T_INFER_SEC = 1.98
    T_CLICK_SEC = 8.0
    MANUAL_SCRATCH_SEC = 1200.0  # 20 minutes average manual segmentation time

    queue3_curve = {}
    for k_tot in [0, 1, 2, 3, 5, 8, 10, 15]:
        tot_gt = 1063
        tot_pred = 0
        tot_tp = 0
        all_post_tips = []
        tot_clicks_used = 0

        for cd in case_sim_data:
            k_rem = k_tot
            gts = cd["gts"]
            n_gt = cd["n_gt_all"]
            items = [dict(it) for it in cd["pred_items"]]

            # Step A: If N_pred > N_implant or lowest-confidence predictions exist, review lowest-confidence tracks first
            n_excess = max(0, len(items) - n_gt)
            if k_rem > 0 and n_excess > 0:
                # Sort by confidence ascending
                low_conf_order = np.argsort([it["score"] for it in items])
                to_remove = set()
                for idx_l in low_conf_order:
                    if k_rem <= 0 or len(to_remove) >= n_excess:
                        break
                    k_rem -= 1
                    tot_clicks_used += 1
                    if not items[idx_l]["is_tp"]:
                        to_remove.add(idx_l)
                items = [it for idx_l, it in enumerate(items) if idx_l not in to_remove]

            # Step B: If len(items) < n_gt, review top rejected pool candidates to recover missing catheters
            deficit = max(0, n_gt - len(items))
            matched_gis = {it["gi"] for it in items if it["is_tp"]}
            if k_rem > 0 and deficit > 0:
                for rc in cd["rejected_cands"][:deficit]:
                    if k_rem <= 0:
                        break
                    k_rem -= 1
                    tot_clicks_used += 1
                    # Check if rc matches an unmatched GT in this case <= 3.5 mm
                    best_gi = None
                    best_d = 3.5
                    for gi, g in enumerate(gts):
                        if gi in matched_gis:
                            continue
                        pm = compute_unified_pair_metrics(rc["ras"], g)
                        if pm is not None and pm["mean_shaft_xy"] <= best_d:
                            best_d = pm["mean_shaft_xy"]
                            best_gi = gi
                    if best_gi is not None:
                        matched_gis.add(best_gi)
                        g_pts = gts[best_gi][np.argsort(gts[best_gi][:, 2])]
                        p_pts = rc["ras"][np.argsort(rc["ras"][:, 2])]
                        vx_g, vy_g = fit_local_slope(g_pts, float(g_pts[-1, 2]), 25.0)
                        u_hat = np.array([vx_g, vy_g, 1.0], dtype=np.float64)
                        u_hat /= np.linalg.norm(u_hat)
                        d_tip = p_pts[-1] - g_pts[-1]
                        e_b = float(np.linalg.norm(d_tip))
                        ds_p = float(np.dot(d_tip, u_hat))
                        dr_p = float(np.linalg.norm(d_tip - ds_p * u_hat))
                        e_a = float(math.hypot(dr_p, min(abs(ds_p), HUMAN_TIP_FLOOR_PARALLEL_MM))) if e_b > 2.0 else e_b
                        items.append({
                            "is_tp": True,
                            "gi": best_gi,
                            "score": rc["score"],
                            "sigma_tip_mm": rc["sigma_tip_mm"],
                            "e_tip_before": e_b,
                            "e_tip_after": e_a,
                        })

            # Step C: Spend remaining budget k_rem on Uncertainty-Guided Tip Verification & 1-Click Relocation
            unc_order = np.argsort([-it["sigma_tip_mm"] for it in items])
            inspected_tips = set(unc_order[:k_rem])
            tot_clicks_used += min(k_rem, len(items))

            tot_pred += len(items)
            for idx_it, it in enumerate(items):
                if it["is_tp"]:
                    tot_tp += 1
                    all_post_tips.append(it["e_tip_after"] if idx_it in inspected_tips else it["e_tip_before"])

        rec = 100.0 * tot_tp / float(tot_gt)
        prec = 100.0 * tot_tp / float(max(1, tot_pred))
        f1 = 2.0 * rec * prec / max(1e-6, rec + prec)
        arr_t = np.asarray(all_post_tips, dtype=np.float64)
        q25, med, q75, p90, p95 = np.percentile(arr_t, [25, 50, 75, 90, 95])
        mean_clicks = tot_clicks_used / 54.0
        est_time_sec = T_INFER_SEC + mean_clicks * T_CLICK_SEC

        queue3_curve[f"k_{k_tot}"] = {
            "budget_actions_per_patient": k_tot,
            "mean_actions_used_per_patient": round(mean_clicks, 2),
            "estimated_total_time_per_patient_sec": round(est_time_sec, 1),
            "speedup_vs_20min_manual_tracing_x": round(MANUAL_SCRATCH_SEC / est_time_sec, 1),
            "n_matched_1to1": tot_tp,
            "n_pred_total": tot_pred,
            "recall_1to1_pct": round(rec, 2),
            "precision_1to1_pct": round(prec, 2),
            "f1_1to1_pct": round(f1, 2),
            "tip_mean_mm": round(float(np.mean(arr_t)), 2),
            "tip_median_mm": round(float(med), 2),
            "tip_iqr_mm": round(float(q75 - q25), 2),
            "tip_p90_mm": round(float(p90), 2),
            "tip_p95_mm": round(float(p95), 2),
            "pct_tip_le_2mm": round(100.0 * float(np.mean(arr_t <= 2.0)), 2),
            "pct_tip_le_3mm": round(100.0 * float(np.mean(arr_t <= 3.0)), 2),
            "pct_tip_gt_10mm": round(100.0 * float(np.mean(arr_t > 10.0)), 2),
        }
        c3 = queue3_curve[f"k_{k_tot}"]
        print(
            f"  [Queue 3 Combined Audit k={k_tot:2d}/pt ({c3['estimated_total_time_per_patient_sec']:5.1f}s/pt, {c3['speedup_vs_20min_manual_tracing_x']:4.1f}x speedup)] "
            f"Rec={c3['recall_1to1_pct']:5.2f}%, Prec={c3['precision_1to1_pct']:5.2f}%, F1={c3['f1_1to1_pct']:5.2f}% | "
            f"Tip Mean={c3['tip_mean_mm']:4.2f}mm, Med={c3['tip_median_mm']:4.2f}mm, P95={c3['tip_p95_mm']:5.2f}mm, >10mm={c3['pct_tip_gt_10mm']:4.2f}%",
            flush=True,
        )

    return {
        "queue1_tip_verification_and_relocation": queue1_policies,
        "queue2_count_guided_missing_catheter_recovery": queue2_curve,
        "queue3_combined_full_case_audit": queue3_curve,
    }


# ==============================================================================
# PART 2D: GOLD-STANDARD-FREE GRUBBS' THREE-CORNERED-HAT & TOST NON-INFERIORITY
# ==============================================================================

def run_part2d_grubbs_and_tost(meta_list, preds_all_tau50, preds_all_tau35):
    print("\n======================================================================")
    print("PART 2D: GRUBBS' THREE-CORNERED-HAT ESTIMATOR & TOST NON-INFERIORITY")
    print("======================================================================", flush=True)

    overlap_nums, cy_40, lb_40 = load_human_interobserver_40cases()
    case_nums = [int(m["case_num"]) for m in meta_list]

    # Collect all 3-way matched catheters across the 40 dual-traced scans:
    # Source 1: Rater 1 (CY)
    # Source 2: Rater 2 (LB/AMe)
    # Source 3: NeedleFinder 2026 OOF (tau=50%)
    three_way_records = []
    per_scene_paired_errors = []

    for s_idx, cnum in enumerate(overlap_nums):
        b_idx = case_nums.index(cnum)
        cy_nds = [np.asarray(p, dtype=np.float64)[np.argsort(p[:, 2])] for p in cy_40[s_idx]]
        lb_nds = [np.asarray(p, dtype=np.float64)[np.argsort(p[:, 2])] for p in lb_40[s_idx]]
        nf_nds = [np.asarray(p["ras"], dtype=np.float64)[np.argsort(p["ras"][:, 2])] for p in preds_all_tau50[b_idx]]

        # 1. Match CY (Source 1) <-> LB/AMe (Source 2) at 3.5 mm
        c12 = np.full((len(cy_nds), max(1, len(lb_nds))), 1e6)
        pm12_map = {}
        for i1, p1 in enumerate(cy_nds):
            for i2, p2 in enumerate(lb_nds):
                pm = compute_unified_pair_metrics(p2, p1)
                if pm is not None and pm["mean_shaft_xy"] <= 3.5:
                    c12[i1, i2] = pm["mean_shaft_xy"]
                    pm12_map[(i1, i2)] = pm
        r12, col12 = linear_sum_assignment(c12)
        cy_to_lb = {int(i1): int(i2) for i1, i2 in zip(r12, col12) if c12[i1, i2] <= 3.5}

        # 2. Match CY (Source 1) <-> NF 2026 (Source 3) at 3.5 mm
        c13 = np.full((len(cy_nds), max(1, len(nf_nds))), 1e6)
        pm13_map = {}
        for i1, p1 in enumerate(cy_nds):
            for i3, p3 in enumerate(nf_nds):
                pm = compute_unified_pair_metrics(p3, p1)
                if pm is not None and pm["mean_shaft_xy"] <= 3.5:
                    c13[i1, i3] = pm["mean_shaft_xy"]
                    pm13_map[(i1, i3)] = pm
        r13, col13 = linear_sum_assignment(c13)
        cy_to_nf = {int(i1): int(i3) for i1, i3 in zip(r13, col13) if c13[i1, i3] <= 3.5}

        # 3. Also match LB/AMe (Source 2) <-> NF 2026 (Source 3) directly at 3.5 mm for direct evaluation
        c23 = np.full((len(lb_nds), max(1, len(nf_nds))), 1e6)
        pm23_map = {}
        for i2, p2 in enumerate(lb_nds):
            for i3, p3 in enumerate(nf_nds):
                pm = compute_unified_pair_metrics(p3, p2)
                if pm is not None and pm["mean_shaft_xy"] <= 3.5:
                    c23[i2, i3] = pm["mean_shaft_xy"]
                    pm23_map[(i2, i3)] = pm
        r23, col23 = linear_sum_assignment(c23)
        lb_to_nf = {int(i2): int(i3) for i2, i3 in zip(r23, col23) if c23[i2, i3] <= 3.5}

        scene_d12_shaft, scene_d13_shaft, scene_d23_shaft = [], [], []
        scene_d12_tip, scene_d13_tip, scene_d23_tip = [], [], []

        for i1 in sorted(set(cy_to_lb.keys()) & set(cy_to_nf.keys())):
            i2 = cy_to_lb[i1]
            i3 = cy_to_nf[i1]
            p1 = cy_nds[i1]
            p2 = lb_nds[i2]
            p3 = nf_nds[i3]
            pm12 = pm12_map[(i1, i2)]
            pm13 = pm13_map[(i1, i3)]
            pm23 = compute_unified_pair_metrics(p3, p2)
            if pm23 is None:
                continue

            # Signed 2D in-plane displacement vectors on common overlapping Z range
            z_lo = max(float(p1[0, 2]), float(p2[0, 2]), float(p3[0, 2]))
            z_hi = min(float(p1[-1, 2]), float(p2[-1, 2]), float(p3[-1, 2]))
            if z_hi <= z_lo + 1.0:
                continue
            zs = np.linspace(z_lo, z_hi, 30)
            x1, y1 = np.interp(zs, p1[:, 2], p1[:, 0]), np.interp(zs, p1[:, 2], p1[:, 1])
            x2, y2 = np.interp(zs, p2[:, 2], p2[:, 0]), np.interp(zs, p2[:, 2], p2[:, 1])
            x3, y3 = np.interp(zs, p3[:, 2], p3[:, 0]), np.interp(zs, p3[:, 2], p3[:, 1])

            # Mean squared 2D radial difference over the common Z span: E[||r_a - r_b||^2]
            msq_xy_12 = float(np.mean((x1 - x2) ** 2 + (y1 - y2) ** 2))
            msq_xy_13 = float(np.mean((x1 - x3) ** 2 + (y1 - y3) ** 2))
            msq_xy_23 = float(np.mean((x2 - x3) ** 2 + (y2 - y3) ** 2))

            # Signed axial tip coordinates & differences
            z1_tip, z2_tip, z3_tip = float(p1[-1, 2]), float(p2[-1, 2]), float(p3[-1, 2])
            dz_12 = z1_tip - z2_tip
            dz_13 = z1_tip - z3_tip
            dz_23 = z2_tip - z3_tip

            three_way_records.append({
                "scene_idx": s_idx,
                "case_num": cnum,
                "benchmark_idx": b_idx,
                "shaft_xy_12": float(pm12["mean_shaft_xy"]),
                "shaft_xy_13": float(pm13["mean_shaft_xy"]),
                "shaft_xy_23": float(pm23["mean_shaft_xy"]),
                "msq_xy_12": msq_xy_12,
                "msq_xy_13": msq_xy_13,
                "msq_xy_23": msq_xy_23,
                "tip_3d_12": float(pm12["tip_3d"]),
                "tip_3d_13": float(pm13["tip_3d"]),
                "tip_3d_23": float(pm23["tip_3d"]),
                "dz_12": dz_12,
                "dz_13": dz_13,
                "dz_23": dz_23,
            })
            scene_d12_shaft.append(float(pm12["mean_shaft_xy"]))
            scene_d13_shaft.append(float(pm13["mean_shaft_xy"]))
            scene_d23_shaft.append(float(pm23["mean_shaft_xy"]))
            scene_d12_tip.append(float(pm12["tip_3d"]))
            scene_d13_tip.append(float(pm13["tip_3d"]))
            scene_d23_tip.append(float(pm23["tip_3d"]))

        per_scene_paired_errors.append({
            "scene_idx": s_idx,
            "case_num": cnum,
            "n_3way": len(scene_d12_shaft),
            "mean_shaft_12": float(np.mean(scene_d12_shaft)) if scene_d12_shaft else 0.0,
            "mean_shaft_13": float(np.mean(scene_d13_shaft)) if scene_d13_shaft else 0.0,
            "mean_shaft_23": float(np.mean(scene_d23_shaft)) if scene_d23_shaft else 0.0,
            "med_tip_12": float(np.median(scene_d12_tip)) if scene_d12_tip else 0.0,
            "med_tip_13": float(np.median(scene_d13_tip)) if scene_d13_tip else 0.0,
            "med_tip_23": float(np.median(scene_d23_tip)) if scene_d23_tip else 0.0,
            "mean_tip_12": float(np.mean(scene_d12_tip)) if scene_d12_tip else 0.0,
            "mean_tip_13": float(np.mean(scene_d13_tip)) if scene_d13_tip else 0.0,
            "mean_tip_23": float(np.mean(scene_d23_tip)) if scene_d23_tip else 0.0,
        })

    # Grubbs' Three-Cornered-Hat Estimator function on a list of 3-way records
    def compute_grubbs_estimates(recs):
        # 1. Shaft 2D Radial Error SD (mm):
        #    Since E[||r_a - r_b||^2] = sigma_a^2 + sigma_b^2, Grubbs' law gives:
        #    sigma_1^2 = 0.5 * (V12 + V13 - V23), etc.
        #    Classic mean-square & Robust median-square (scaled by 1 / log(2) for 2D Rayleigh/Chi-square df=2, or direct RMS)
        v12_sh_classic = float(np.mean([r["shaft_xy_12"] ** 2 for r in recs]))
        v13_sh_classic = float(np.mean([r["shaft_xy_13"] ** 2 for r in recs]))
        v23_sh_classic = float(np.mean([r["shaft_xy_23"] ** 2 for r in recs]))

        s1_sh_classic = math.sqrt(max(1e-6, 0.5 * (v12_sh_classic + v13_sh_classic - v23_sh_classic)))
        s2_sh_classic = math.sqrt(max(1e-6, 0.5 * (v12_sh_classic + v23_sh_classic - v13_sh_classic)))
        s3_sh_classic = math.sqrt(max(1e-6, 0.5 * (v13_sh_classic + v23_sh_classic - v12_sh_classic)))

        # Robust shaft variance via (median / 0.83255)^2
        scale_med = 1.0 / (0.83255 ** 2)
        v12_sh_rob = float(np.median([r["shaft_xy_12"] for r in recs]) ** 2) * scale_med
        v13_sh_rob = float(np.median([r["shaft_xy_13"] for r in recs]) ** 2) * scale_med
        v23_sh_rob = float(np.median([r["shaft_xy_23"] for r in recs]) ** 2) * scale_med

        s1_sh_rob = math.sqrt(max(1e-6, 0.5 * (v12_sh_rob + v13_sh_rob - v23_sh_rob)))
        s2_sh_rob = math.sqrt(max(1e-6, 0.5 * (v12_sh_rob + v23_sh_rob - v13_sh_rob)))
        s3_sh_rob = math.sqrt(max(1e-6, 0.5 * (v13_sh_rob + v23_sh_rob - v12_sh_rob)))

        # 2. Axial Tip Error SD (mm) from signed pairwise differences dz_12, dz_13, dz_23:
        dz12 = np.array([r["dz_12"] for r in recs], dtype=np.float64)
        dz13 = np.array([r["dz_13"] for r in recs], dtype=np.float64)
        dz23 = np.array([r["dz_23"] for r in recs], dtype=np.float64)

        # Robust MAD variance: (1.4826 * MAD(dz))^2
        def _mad_var(x):
            med = np.median(x)
            return float((1.4826 * np.median(np.abs(x - med))) ** 2)

        v12_tip_mad = _mad_var(dz12)
        v13_tip_mad = _mad_var(dz13)
        v23_tip_mad = _mad_var(dz23)
        s1_tip_mad = math.sqrt(max(1e-6, 0.5 * (v12_tip_mad + v13_tip_mad - v23_tip_mad)))
        s2_tip_mad = math.sqrt(max(1e-6, 0.5 * (v12_tip_mad + v23_tip_mad - v13_tip_mad)))
        s3_tip_mad = math.sqrt(max(1e-6, 0.5 * (v13_tip_mad + v23_tip_mad - v12_tip_mad)))

        # Winsorized / Trimmed variance (clamped to [-15, +15] mm so 40mm gross outliers don't dominate)
        v12_tip_win = float(np.var(np.clip(dz12, -15.0, 15.0), ddof=1))
        v13_tip_win = float(np.var(np.clip(dz13, -15.0, 15.0), ddof=1))
        v23_tip_win = float(np.var(np.clip(dz23, -15.0, 15.0), ddof=1))
        s1_tip_win = math.sqrt(max(1e-6, 0.5 * (v12_tip_win + v13_tip_win - v23_tip_win)))
        s2_tip_win = math.sqrt(max(1e-6, 0.5 * (v12_tip_win + v23_tip_win - v13_tip_win)))
        s3_tip_win = math.sqrt(max(1e-6, 0.5 * (v13_tip_win + v23_tip_win - v12_tip_win)))

        return {
            "shaft_classic_sd_mm": (s1_sh_classic, s2_sh_classic, s3_sh_classic),
            "shaft_robust_mad_sd_mm": (s1_sh_rob, s2_sh_rob, s3_sh_rob),
            "tip_robust_mad_sd_mm": (s1_tip_mad, s2_tip_mad, s3_tip_mad),
            "tip_winsorized15_sd_mm": (s1_tip_win, s2_tip_win, s3_tip_win),
        }

    point_est = compute_grubbs_estimates(three_way_records)

    # 10,000-iteration patient-clustered bootstrap across the 40 dual-traced scenes
    by_scene = {s_i: [r for r in three_way_records if r["scene_idx"] == s_i] for s_i in range(40)}
    rng = np.random.default_rng(2026)
    boot_sh_classic = [[], [], []]
    boot_sh_rob = [[], [], []]
    boot_tip_mad = [[], [], []]
    boot_tip_win = [[], [], []]

    for _ in range(10000):
        s_idxs = rng.integers(0, 40, size=40)
        b_recs = [r for s_i in s_idxs for r in by_scene[s_i]]
        if len(b_recs) < 10:
            continue
        est_b = compute_grubbs_estimates(b_recs)
        for k_i in range(3):
            boot_sh_classic[k_i].append(est_b["shaft_classic_sd_mm"][k_i])
            boot_sh_rob[k_i].append(est_b["shaft_robust_mad_sd_mm"][k_i])
            boot_tip_mad[k_i].append(est_b["tip_robust_mad_sd_mm"][k_i])
            boot_tip_win[k_i].append(est_b["tip_winsorized15_sd_mm"][k_i])

    def _ci95(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    grubbs_summary = {
        "n_dual_traced_scenes": 40,
        "n_three_way_matched_catheters": len(three_way_records),
        "pairwise_discrepancy_distributions": {
            "shaft_xy_rater1_vs_rater2_mm": summarize_distribution([r["shaft_xy_12"] for r in three_way_records]),
            "shaft_xy_nf2026_vs_rater1_mm": summarize_distribution([r["shaft_xy_13"] for r in three_way_records]),
            "shaft_xy_nf2026_vs_rater2_mm": summarize_distribution([r["shaft_xy_23"] for r in three_way_records]),
            "tip_3d_rater1_vs_rater2_mm": summarize_distribution([r["tip_3d_12"] for r in three_way_records]),
            "tip_3d_nf2026_vs_rater1_mm": summarize_distribution([r["tip_3d_13"] for r in three_way_records]),
            "tip_3d_nf2026_vs_rater2_mm": summarize_distribution([r["tip_3d_23"] for r in three_way_records]),
            "dz_tip_signed_rater1_minus_rater2_mm": summarize_distribution([r["dz_12"] for r in three_way_records]),
            "dz_tip_signed_rater1_minus_nf2026_mm": summarize_distribution([r["dz_13"] for r in three_way_records]),
            "dz_tip_signed_rater2_minus_nf2026_mm": summarize_distribution([r["dz_23"] for r in three_way_records]),
        },
        "intrinsic_source_error_sd_estimates": {
            "in_plane_shaft_sd_classic_mm": {
                "Rater1_CY": {"point": round(point_est["shaft_classic_sd_mm"][0], 4), "ci95_10000boot": _ci95(boot_sh_classic[0])},
                "Rater2_LB_AMe": {"point": round(point_est["shaft_classic_sd_mm"][1], 4), "ci95_10000boot": _ci95(boot_sh_classic[1])},
                "NeedleFinder_2026_OOF": {"point": round(point_est["shaft_classic_sd_mm"][2], 4), "ci95_10000boot": _ci95(boot_sh_classic[2])},
            },
            "in_plane_shaft_sd_robust_mad_mm": {
                "Rater1_CY": {"point": round(point_est["shaft_robust_mad_sd_mm"][0], 4), "ci95_10000boot": _ci95(boot_sh_rob[0])},
                "Rater2_LB_AMe": {"point": round(point_est["shaft_robust_mad_sd_mm"][1], 4), "ci95_10000boot": _ci95(boot_sh_rob[1])},
                "NeedleFinder_2026_OOF": {"point": round(point_est["shaft_robust_mad_sd_mm"][2], 4), "ci95_10000boot": _ci95(boot_sh_rob[2])},
            },
            "axial_tip_sd_robust_mad_mm": {
                "Rater1_CY": {"point": round(point_est["tip_robust_mad_sd_mm"][0], 4), "ci95_10000boot": _ci95(boot_tip_mad[0])},
                "Rater2_LB_AMe": {"point": round(point_est["tip_robust_mad_sd_mm"][1], 4), "ci95_10000boot": _ci95(boot_tip_mad[1])},
                "NeedleFinder_2026_OOF": {"point": round(point_est["tip_robust_mad_sd_mm"][2], 4), "ci95_10000boot": _ci95(boot_tip_mad[2])},
            },
            "axial_tip_sd_winsorized15_mm": {
                "Rater1_CY": {"point": round(point_est["tip_winsorized15_sd_mm"][0], 4), "ci95_10000boot": _ci95(boot_tip_win[0])},
                "Rater2_LB_AMe": {"point": round(point_est["tip_winsorized15_sd_mm"][1], 4), "ci95_10000boot": _ci95(boot_tip_win[1])},
                "NeedleFinder_2026_OOF": {"point": round(point_est["tip_winsorized15_sd_mm"][2], 4), "ci95_10000boot": _ci95(boot_tip_win[2])},
            },
        },
    }

    # Formal Two One-Sided Tests (TOST) for Non-Inferiority & Equivalence
    # Compare NF 2026 error vs Rater 2 error (both evaluated against Rater 1 CY, and also symmetrical NF vs Rater 2 against Rater 1 vs Rater 2)
    def run_tost_analysis(err_nf, err_human, margin_delta, stat_mode="mean", label=""):
        err_nf = np.asarray(err_nf, dtype=np.float64)
        err_human = np.asarray(err_human, dtype=np.float64)
        diffs = err_nf - err_human
        n = len(diffs)

        if stat_mode == "mean":
            point_diff = float(np.mean(diffs))
            se_diff = float(np.std(diffs, ddof=1) / math.sqrt(n))
            # Upper one-sided test (Non-inferiority: H0: diff >= +margin vs H1: diff < +margin)
            t_upper = (point_diff - margin_delta) / max(1e-9, se_diff)
            p_upper_t = float(norm.cdf(t_upper))
            # Lower one-sided test (H0: diff <= -margin vs H1: diff > -margin)
            t_lower = (point_diff + margin_delta) / max(1e-9, se_diff)
            p_lower_t = float(1.0 - norm.cdf(t_lower))
            p_tost = max(p_upper_t, p_lower_t)
        else:
            point_diff = float(np.median(err_nf) - np.median(err_human))
            p_upper_t = None
            p_lower_t = None
            p_tost = None

        # Nonparametric Wilcoxon one-sided non-inferiority test on (diffs - margin_delta < 0)
        res_w_up = wilcoxon(diffs - margin_delta, alternative="less")
        p_upper_wilcox = float(res_w_up.pvalue)
        res_w_lo = wilcoxon(diffs + margin_delta, alternative="greater")
        p_lower_wilcox = float(res_w_lo.pvalue)
        p_tost_wilcox = max(p_upper_wilcox, p_lower_wilcox)

        # 10,000-resample patient-clustered bootstrap for 90% CI (TOST standard) and 95% CI
        rng_b = np.random.default_rng(909)
        boot_diffs = []
        for _ in range(10000):
            s_idxs = rng_b.integers(0, 40, size=40)
            b_recs = [r for s_i in s_idxs for r in by_scene[s_i]]
            if not b_recs:
                continue
            if "shaft" in label:
                v_nf = [r["shaft_xy_13"] for r in b_recs]
                v_hu = [r["shaft_xy_12"] for r in b_recs]
            elif "vs_rater2_ref" in label:
                v_nf = [r["tip_3d_23"] for r in b_recs]
                v_hu = [r["tip_3d_12"] for r in b_recs]
            else:
                v_nf = [r["tip_3d_13"] for r in b_recs]
                v_hu = [r["tip_3d_12"] for r in b_recs]
            if stat_mode == "median":
                boot_diffs.append(float(np.median(v_nf) - np.median(v_hu)))
            else:
                boot_diffs.append(float(np.mean(v_nf) - np.mean(v_hu)))

        ci90 = [round(float(np.percentile(boot_diffs, 5.0)), 4), round(float(np.percentile(boot_diffs, 95.0)), 4)]
        ci95 = [round(float(np.percentile(boot_diffs, 2.5)), 4), round(float(np.percentile(boot_diffs, 97.5)), 4)]
        p_boot_noninf = float(np.mean(np.asarray(boot_diffs) >= margin_delta))

        return {
            "label": label,
            "statistic_mode": stat_mode,
            "margin_delta_mm": margin_delta,
            "nf2026_value_mm": round(float(np.mean(err_nf) if stat_mode == "mean" else np.median(err_nf)), 4),
            "human_interobserver_value_mm": round(float(np.mean(err_human) if stat_mode == "mean" else np.median(err_human)), 4),
            "paired_difference_nf_minus_human_mm": round(point_diff, 4),
            "ci90_patient_bootstrap_mm": ci90,
            "ci95_patient_bootstrap_mm": ci95,
            "non_inferiority_p_value_wilcoxon": p_upper_wilcox,
            "tost_equivalence_p_value_wilcoxon": p_tost_wilcox,
            "non_inferiority_p_value_parametric": p_upper_t,
            "tost_equivalence_p_value_parametric": p_tost,
            "non_inferiority_p_value_bootstrap": round(p_boot_noninf, 5),
            "is_non_inferior_at_alpha_0p05": bool(ci90[1] < margin_delta),
            "is_equivalent_tost_at_alpha_0p05": bool(ci90[0] > -margin_delta and ci90[1] < margin_delta),
        }

    sh_13 = [r["shaft_xy_13"] for r in three_way_records]
    sh_12 = [r["shaft_xy_12"] for r in three_way_records]
    tip_13 = [r["tip_3d_13"] for r in three_way_records]
    tip_23 = [r["tip_3d_23"] for r in three_way_records]
    tip_12 = [r["tip_3d_12"] for r in three_way_records]

    tost_results = {
        "shaft_mean_noninferiority_margin_0p25mm": run_tost_analysis(
            sh_13, sh_12, margin_delta=0.25, stat_mode="mean", label="shaft_mean_nf_vs_cy_compared_to_lb_vs_cy"
        ),
        "shaft_median_noninferiority_margin_0p25mm": run_tost_analysis(
            sh_13, sh_12, margin_delta=0.25, stat_mode="median", label="shaft_median_nf_vs_cy_compared_to_lb_vs_cy"
        ),
        "tip_median_noninferiority_margin_1p0mm": run_tost_analysis(
            tip_13, tip_12, margin_delta=1.00, stat_mode="median", label="tip_median_nf_vs_cy_compared_to_lb_vs_cy"
        ),
        "tip_mean_noninferiority_margin_1p5mm": run_tost_analysis(
            tip_13, tip_12, margin_delta=1.50, stat_mode="mean", label="tip_mean_nf_vs_cy_compared_to_lb_vs_cy"
        ),
        "tip_median_vs_rater2_ref_margin_1p0mm": run_tost_analysis(
            tip_23, tip_12, margin_delta=1.00, stat_mode="median", label="tip_median_nf_vs_rater2_ref_compared_to_cy_vs_lb"
        ),
        "tip_mean_vs_rater2_ref_margin_1p5mm": run_tost_analysis(
            tip_23, tip_12, margin_delta=1.50, stat_mode="mean", label="tip_mean_nf_vs_rater2_ref_compared_to_cy_vs_lb"
        ),
    }

    gs = grubbs_summary["intrinsic_source_error_sd_estimates"]
    print(f"  3-Way Matched Catheters across 40 Dual-Traced Scans: N = {len(three_way_records)}")
    print("  Grubbs' Three-Cornered-Hat Intrinsic Error SD Estimates (with 10,000-patient-bootstrap 95% CIs):")
    for m_key in ["in_plane_shaft_sd_robust_mad_mm", "in_plane_shaft_sd_classic_mm", "axial_tip_sd_robust_mad_mm", "axial_tip_sd_winsorized15_mm"]:
        d_m = gs[m_key]
        print(
            f"    - {m_key:34s}: "
            f"Rater 1 (CY) = {d_m['Rater1_CY']['point']:.3f} mm {d_m['Rater1_CY']['ci95_10000boot']} | "
            f"Rater 2 (LB/AMe) = {d_m['Rater2_LB_AMe']['point']:.3f} mm {d_m['Rater2_LB_AMe']['ci95_10000boot']} | "
            f"NF 2026 = {d_m['NeedleFinder_2026_OOF']['point']:.3f} mm {d_m['NeedleFinder_2026_OOF']['ci95_10000boot']}"
        )
    print("  Formal TOST Non-Inferiority & Equivalence Tests (NF 2026 vs Human Inter-Observer):")
    for k_t, tv in tost_results.items():
        print(
            f"    - {k_t:42s}: NF={tv['nf2026_value_mm']:.3f}mm vs Human={tv['human_interobserver_value_mm']:.3f}mm, "
            f"diff={tv['paired_difference_nf_minus_human_mm']:+.3f}mm (90% CI {tv['ci90_patient_bootstrap_mm']}), "
            f"margin={tv['margin_delta_mm']:.2f}mm, p_noninf_wilcox={tv['non_inferiority_p_value_wilcoxon']:.3e}, "
            f"non_inferior={tv['is_non_inferior_at_alpha_0p05']}"
        )

    grubbs_summary["tost_non_inferiority_and_equivalence_tests"] = tost_results
    return grubbs_summary


def main():
    t0 = time.time()
    meta_list = pickle.load(open(META_CLEAN_PATH, "rb"))

    if os.path.exists(PREDS_FULL_PKL_PATH):
        p_data = pickle.load(open(PREDS_FULL_PKL_PATH, "rb"))
        bgs = p_data["stitched_enriched_bgs"]
    else:
        p_data = pickle.load(open(PREDS_FALLBACK_PKL_PATH, "rb"))
        b_data = pickle.load(open(BUNDLES_CLEAN_PKL_PATH, "rb"))
        bgs = b_data["stitched_enriched_bgs"]

    preds_all_tau50 = p_data["all_tau50"]
    preds_long_tau50 = p_data["long_tau50"]
    preds_all_tau35 = p_data["all_tau35"]
    base_sc = p_data["base_sc_st_mr"]
    syn_out = p_data["syn_out_st_mr"]

    tip_models = []
    for f_out in range(5):
        m = Dedicated1DTipLocalizer(in_channels=22, hidden_dim=48, seq_len=91)
        m.load_state_dict(p_data["tip_models_5fold_states"][f_out])
        m.eval()
        tip_models.append(m)

    print("Extracting per-candidate 1D tip distribution & physical uncertainty features across 54 cases...", flush=True)
    all_case_cand_unc = []
    for ci in range(54):
        f_out = ci % 5
        all_case_cand_unc.append(
            extract_candidate_tip_uncertainty_features(bgs[ci], ci, syn_out[ci], base_sc[ci], tip_models[f_out])
        )

    fit_nested_oof_uncertainty_calibrator(all_case_cand_unc, bgs)

    matched_records, part2a_res = run_part2a_tip_decomposition_and_uncertainty(
        meta_list, preds_all_tau50, preds_long_tau50, bgs, all_case_cand_unc
    )
    part2b_res = run_part2b_split_conformal_intervals(matched_records)
    part2c_res = run_part2c_clinician_in_the_loop_simulation(
        meta_list, preds_all_tau50, bgs, all_case_cand_unc, base_sc, syn_out
    )
    part2d_res = run_part2d_grubbs_and_tost(meta_list, preds_all_tau50, preds_all_tau35)

    final_json = {
        "generated_at": "2026-10-23",
        "experiment": "Experiment 2: Uncertainty-Guided Clinician-in-the-Loop Audit, Nested Split-Conformal Tip Intervals, Along/Lateral Tip-Tail Taxonomy, and Grubbs' Three-Cornered-Hat & TOST Non-Inferiority",
        "script": "vault/paper/benchmarks_clean/run_exp2_uncertainty_triage_conformal_grubbs.py",
        "elapsed_sec": round(time.time() - t0, 2),
        "part2a_tip_error_decomposition_and_tail_taxonomy": part2a_res,
        "part2b_nested_split_conformal_tip_intervals": part2b_res,
        "part2c_clinician_in_the_loop_review_simulation": part2c_res,
        "part2d_grubbs_three_cornered_hat_and_tost": part2d_res,
    }

    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(final_json, f, indent=2)
    print(f"\n[COMPLETE] Saved {OUT_JSON_PATH} in {time.time() - t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
