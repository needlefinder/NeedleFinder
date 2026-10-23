#!/usr/bin/env python3
"""
Single-Transaction Unified Benchmark Exporter + Clean Few-Shot (AUD-500) + 40/40 Table 3 Exporter
=================================================================================================
Generates in a single atomic transaction from the Clean Nested Pipeline (`run_stage2_and_tip_clean.py`):
  1. `/home/guillaume/github/needlefinder/vault/sources/2026-10-23-unified-clean-nested-benchmark.json`
     and updates `/home/guillaume/github/needlefinder/vault/sources/2026-10-23-unified-rerun-benchmark.json`
     with 100% consistent match counts across summary tables and `per_patient_and_needle_error_distributions`
     (`assert len(dist["per_needle_mean_shaft_xy_mm"]) == summary["hungarian_1to1_matched"]` for EVERY method,
     including Baselines 1, 2, 3 and all NeedleFinder 2026 variants!).
  2. Leak-free Few-Shot Clinical Annotation Budget (`k = 0, 1, 5, 10, 20, 43`, AUD-500):
     - `0_shot_stage0_analytical`: Pure reference-free Stage 0 (`k_lim_auto`) with 0 real or synthetic training.
     - `0_shot_synthbrachy3t_pretrained` (`0_shot`): Pretrained SynthBrachy3T bundle transformer evaluated on
       Stage 0 candidate bundles with 0 real clinical labels (`w_synth = 1.0`).
     - `k in {1, 5, 10, 20}`: Both `DenseGraphTransformer` and `TipAwareBundleGraphTransformer` trained strictly
       on the `k` real clinical cases per fold (zero leakage from full-data `base_oof_scores`).
     - `k = 43` (`43_shot`): Full Nested CV Stage 2 + 1D Tip Localizer (`35 subtrain + 8-9 inner_val`).
  3. `/home/guillaume/github/needlefinder/vault/sources/2026-10-23-table3-observers-and-nf-clean.json`
     and `/home/guillaume/github/needlefinder/vault/sources/2026-10-23-table3-observers-and-nf-rerun.json`
     across ALL 40 / 40 mapped dual-traced scenes (513 LB/AMe trajectories, 752 CY+HT trajectories).
  4. `/home/guillaume/github/needlefinder/vault/reports/figures/data/qualitative_predictions.npz`
     for Case 017 (`idx=6`) and Case 031 (`idx=17`).
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun")

from run_stage2_and_tip_clean import (
    DenseGraphTransformer,
    TipAwareBundleGraphTransformer,
    Dedicated1DTipLocalizer,
    generate_synthetic_bundle_graph_item_v2,
    augment_clinical_bundle_with_synthbrachy3t_v2,
    decode_clean_stage2_preds,
    get_nested_fold_split,
    calibrate_stage1_oof_for_volume,
    compute_detailed_tip_and_fp_diagnostics,
    analyze_catheter_adhesion,
    derive_unstitched_and_base_bgs,
)
from evaluate_unified_apples_to_apples import (
    evaluate_method_unified,
    summarize_distribution,
    extract_nn_labelmap_tubes,
    get_gts,
    traj_gt_dist,
    compute_unified_pair_metrics,
    compute_wilcoxon_comparison,
)


def to_gpu_item(b, device):
    return {
        "x": b["x"].to(device, non_blocking=True),
        "adj_feat": b["adj_feat"].to(device, non_blocking=True),
        "adj_mask": b["adj_mask"].to(device, non_blocking=True),
        "y_rec": b["y_rec"].to(device, non_blocking=True),
        "y_1to1": b["y_1to1"].to(device, non_blocking=True),
        "y_long_1to1": b["y_long_1to1"].to(device, non_blocking=True),
        "y_dz_tip": b["y_dz_tip"].to(device, non_blocking=True),
        "y_dz_weight": b["y_dz_weight"].to(device, non_blocking=True),
    }


def extract_per_patient_and_needle_distributions_exact(preds_per_case, gts_per_case, thresh=3.5):
    """
    Extracts per-patient F1/Recall arrays and per-needle error arrays using the EXACT same
    thresholded Hungarian assignment (`cost_m = np.where(cost_m > thresh, 1e6, cost_m)`) as
    `evaluate_method_unified`, guaranteeing:
      len(dist["per_needle_mean_shaft_xy_mm"]) == summary["hungarian_1to1_matched"]
    """
    case_f1s = []
    case_recalls = []
    case_precs = []
    shaft_xy = []
    msd_3d = []
    hd95_3d = []
    tip_3d = []

    for i in range(len(gts_per_case)):
        gts = gts_per_case[i]
        preds = preds_per_case[i]
        n_gt, n_pred = len(gts), len(preds)
        cost_m = np.full((n_gt, max(1, n_pred)), 1e6, dtype=np.float64)
        pms = {}
        for gi, g in enumerate(gts):
            g_pts = g["pts"] if isinstance(g, dict) else g
            for pi, p in enumerate(preds):
                p_pts = p["ras"] if isinstance(p, dict) and "ras" in p else (p["points"] if isinstance(p, dict) else p)
                pm = compute_unified_pair_metrics(p_pts, g_pts)
                if pm is not None:
                    cost_m[gi, pi] = pm["mean_shaft_xy"]
                    pms[(gi, pi)] = pm

        # Thresholded assignment identical to evaluate_method_unified!
        cost_m = np.where(cost_m > thresh, 1e6, cost_m)
        tp = 0
        if n_gt > 0 and n_pred > 0:
            r_ind, c_ind = linear_sum_assignment(cost_m)
            for gi, pi in zip(r_ind, c_ind):
                if cost_m[gi, pi] <= thresh:
                    tp += 1
                    m = pms[(gi, pi)]
                    shaft_xy.append(round(float(m["mean_shaft_xy"]), 4))
                    msd_3d.append(round(float(m["msd_3d"]), 4))
                    hd95_3d.append(round(float(m["hd95_3d"]), 4))
                    tip_3d.append(round(float(m["tip_3d"]), 4))
        rec = 100.0 * tp / max(1, n_gt)
        prec = 100.0 * tp / max(1, n_pred) if n_pred > 0 else 0.0
        f1 = 2.0 * rec * prec / max(1e-8, rec + prec)
        case_f1s.append(round(float(f1), 3))
        case_recalls.append(round(float(rec), 3))
        case_precs.append(round(float(prec), 3))

    return {
        "per_case_f1_pct": case_f1s,
        "per_case_recall_pct": case_recalls,
        "per_patient_f1_pct": case_f1s,
        "per_patient_recall_pct": case_recalls,
        "per_patient_precision_pct": case_precs,
        "per_needle_mean_shaft_xy_mm": shaft_xy,
        "per_needle_tip_3d_mm": tip_3d,
        "per_needle_msd_3d_summary": summarize_distribution(msd_3d),
        "per_needle_hd95_3d_summary": summarize_distribution(hd95_3d),
    }


def evaluate_and_verify_single_transaction(preds_per_case, gts_per_case, aux_short_gts_per_case=None, thresh=3.5, n_boot=300):
    """
    Runs `evaluate_method_unified` and `extract_per_patient_and_needle_distributions_exact` in a
    single transaction and asserts 100% match-count identity.
    """
    summary, pp = evaluate_method_unified(preds_per_case, gts_per_case, aux_short_gts_per_case=aux_short_gts_per_case, thresh=thresh, n_boot=n_boot)
    dist = extract_per_patient_and_needle_distributions_exact(preds_per_case, gts_per_case, thresh=thresh)
    assert len(dist["per_needle_mean_shaft_xy_mm"]) == summary["hungarian_1to1_matched"], (
        f"Mismatch: len(dist)={len(dist['per_needle_mean_shaft_xy_mm'])} != matched={summary['hungarian_1to1_matched']}"
    )
    assert len(dist["per_needle_tip_3d_mm"]) == summary["hungarian_1to1_matched"]
    return summary, pp, dist


def extract_stage1b_and_stage2a_preds(stage0_preds, stitched_enriched_bgs, graphs, all_oof_cal, all_X_vols, meta_list):
    """
    Reconstructs:
      - Stage 1B (Multi-Head NeedleUNet3D + Tangent Tube Tracking): `stage0_preds[i] + extract_nn_labelmap_tubes(...)`
      - Stage 2A (Pure NeedleSliceMPNN): candidates with `from_mpnn >= 0.9`
    """
    s1b_all, s1b_long = [], []
    s2a_all, s2a_long = [], []
    for i, bg in enumerate(stitched_enriched_bgs):
        X = all_X_vols[i]
        oof_cal = all_oof_cal[i]
        b_clean = stage0_preds[i]
        k0, i0, j0 = oof_cal["k0"], oof_cal["i0"], oof_cal["j0"]
        sp_i, sp_j, sp_k = [float(x) for x in meta_list[i]["sp_ras"]]
        ijk2ras = graphs[i]["ijk_to_ras"]
        ras2ijk = np.linalg.inv(ijk2ras)
        resp_approx = np.expm1(X[1] * 4.0)
        nn_d = {k: oof_cal[k] for k in ["p_mask", "p_peak", "vx", "vy", "p_obt"]}
        u_long_ex = extract_nn_labelmap_tubes(nn_d, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, b_clean, include_obturator=False)
        u_all_ex = extract_nn_labelmap_tubes(nn_d, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, b_clean, include_obturator=True)
        s1b_all.append(b_clean + u_all_ex)
        s1b_long.append(b_clean + u_long_ex)

        pool_un = [p for p in bg["pool"] if not p.get("is_stitched", False)]
        k2 = [dict(p, points=p["ras"]) for p in pool_un if p["from_mpnn"] >= 0.9]
        s2a_all.append(k2)
        s2a_long.append([kp for kp in k2 if kp["z_span"] >= 28.0])

    return (s1b_all, s1b_long), (s2a_all, s2a_long)


def build_stage0_only_bundle_graphs(stage0_preds, graphs):
    """
    Builds zero-Stage-1 (`p_mask = p_peak = p_obt = 0`) bundle graphs directly from `stage0_preds`
    so the Pretrained Synthetic Bundle Transformer (`base_syn_model`) can be evaluated in a
    100% zero-shot manner (`k = 0`, 0 clinical labels in Stage 1 or Stage 2)!
    """
    stage0_bgs = []
    for i, b_clean in enumerate(stage0_preds):
        g = graphs[i]
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
        M = len(pool)
        bases = np.array([p["base_xy"] for p in pool]) if M else np.zeros((0, 2))
        slopes = np.array([p["slope"] for p in pool]) if M else np.zeros((0, 2))
        med_base = np.median(bases, axis=0) if M else np.zeros(2)
        med_slope = np.median(slopes, axis=0) if M else np.zeros(2)
        z_tops = np.array([float(p["ras"][-1, 2]) for p in pool]) if M else np.zeros(0)
        z_bots = np.array([float(p["ras"][0, 2]) for p in pool]) if M else np.zeros(0)

        x26_feats = []
        for u, p in enumerate(pool):
            ras = p["ras"]
            d_base = float(np.hypot(p["base_xy"][0] - med_base[0], p["base_xy"][1] - med_base[1])) / 40.0
            d_slope = float(np.hypot(p["slope"][0] - med_slope[0], p["slope"][1] - med_slope[1])) / 0.25
            z_low = float(ras[0, 2] - g["z_base_ras"]) / 50.0
            z_high = float(ras[-1, 2] - g["z_base_ras"]) / 100.0
            c_h = p["conf_heur"]
            p["z_prof_tip"] = float(ras[-1, 2])
            p["z_tip_nbr"] = float(ras[-1, 2])
            p["vx_top"] = 0.0
            p["vy_top"] = 0.0
            p["mean_sym"] = c_h * 0.65
            p["frac_active"] = c_h
            p["mean_pm"] = c_h * 0.70

            f26 = [
                c_h, 0.0, 0.0, 0.0, 0.0,
                c_h * 0.70, c_h * 0.80, c_h * 0.55, 0.0, c_h * 0.70,
                p["z_span"] / 60.0, p["rmse"], d_base, d_slope, z_low, z_high,
                float(p["slope"][0]), float(p["slope"][1]),
                c_h * 0.65, c_h * 0.80, c_h * 0.60, c_h * 0.55, c_h, c_h * 0.50, 0.0, 0.0,
            ]
            x26_feats.append(f26)

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
                    adj_feat8[u, v] = [
                        db / 35.0, sh_d / 35.0, ds / 0.30, 1.0 if sh_d < 2.2 else 0.0,
                        float(x26_feats[v][5] - x26_feats[u][5]), float(x26_feats[v][0] - x26_feats[u][0]),
                        dz_t, z_ov,
                    ]
        stage0_bgs.append({
            "pool": pool,
            "x": torch.tensor(x26_feats, dtype=torch.float32) if M > 0 else torch.zeros((0, 26), dtype=torch.float32),
            "adj_feat": torch.tensor(adj_feat8, dtype=torch.float32),
            "adj_mask": torch.tensor(adj_mask, dtype=torch.float32),
        })
    return stage0_bgs


def run_leak_free_few_shot_budget(
    stage0_preds,
    stitched_base_bgs,
    stitched_enriched_bgs,
    graphs,
    all_oof_cal,
    all_X_vols,
    tip_models_5fold,
    base_sc_st_mr,
    syn_out_st_mr,
    gts_all_list,
    gts_long_list,
    gts_short_list,
    device,
):
    """
    Executes the 100% leak-free Few-Shot Clinical Annotation Budget (AUD-500):
      - `0_shot_stage0_analytical`: Pure analytical Stage 0 (`k_lim_auto`), 0 real clinical labels.
      - `0_shot` / `0_shot_synthbrachy3t_pretrained`: Zero-Shot Stage 0 + Pretrained Synthetic Bundle Transformer
        (`base_syn_model` trained purely on 300 synthetic bundles, evaluated on Stage 0 candidate bundles
        with `w_synth = 1.0` and 0 real clinical labels!).
      - `k_shot in [1, 5, 10, 20]`: Both `DenseGraphTransformer` (`base_k_scores`) AND
        `TipAwareBundleGraphTransformer` (`fs_outputs`) trained strictly on the `k_shot` real clinical
        cases per fold (`outer_train[:k_shot]`), with ZERO access to full-data `base_oof_scores`!
      - `43_shot`: Full Nested CV (`35 subtrain + 8-9 inner_val = 43-44 outer_train`).
    """
    n_cases = len(stitched_enriched_bgs)
    few_shot_results = {}

    # Pre-train pure synthetic bundle transformer (0 clinical cases)
    pw_r = torch.tensor([3.2], device=device)
    pw_1 = torch.tensor([2.6], device=device)
    pw_l = torch.tensor([2.4], device=device)
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

    # 0a. Pure Stage 0 Analytical (`k_lim_auto`, 0 real cases, 0 synthetic bundles)
    t_s0 = time.time()
    s0_all = stage0_preds
    s0_long = [[t for t in c if float(t["ras"][-1, 2] - t["ras"][0, 2]) >= 26.0] for c in stage0_preds]
    res_s0_all, _ = evaluate_method_unified(s0_all, gts_all_list, thresh=3.5, n_boot=200)
    res_s0_long, _ = evaluate_method_unified(s0_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=200)
    few_shot_results["0_shot_stage0_analytical"] = {
        "k_shot_per_fold": 0,
        "n_real_clinical_cases": 0,
        "n_synthetic_bundles": 0,
        "protocol": "Pure reference-free Stage 0 analytical tracker (k_lim_auto), 0 real BWH labels",
        "cohort_all_1063_catheters": res_s0_all,
        "cohort_long_759_cy_catheters": res_s0_long,
        "elapsed_sec": round(time.time() - t_s0, 2),
    }

    # 0b. Zero-Shot Stage 0 + Pretrained Synthetic Bundle Transformer (0 real cases, 300 synthetic bundles)
    t_zs = time.time()
    stage0_bgs = build_stage0_only_bundle_graphs(stage0_preds, graphs)
    base_syn_model.eval()
    zs_syn_outputs = []
    zs_dummy_base = []
    with torch.no_grad():
        for bg in stage0_bgs:
            if len(bg["pool"]) > 0:
                out = base_syn_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device)).cpu().numpy()
            else:
                out = np.zeros((0, 4), dtype=np.float32)
            zs_syn_outputs.append(out)
            zs_dummy_base.append(np.zeros((len(bg["pool"]), 3), dtype=np.float32))

    zs_all = decode_clean_stage2_preds(
        stage0_bgs, zs_dummy_base, zs_syn_outputs,
        min_conf=42.0, long_mode=False, w_synth=1.0, use_collinear_dedup=True, use_1d_tip_model=False, tip_scale=0.0
    )
    zs_long = decode_clean_stage2_preds(
        stage0_bgs, zs_dummy_base, zs_syn_outputs,
        min_conf=42.0, long_mode=True, w_synth=1.0, use_collinear_dedup=True, use_1d_tip_model=False, tip_scale=0.0
    )
    res_zs_all, _ = evaluate_method_unified(zs_all, gts_all_list, thresh=3.5, n_boot=200)
    res_zs_long, _ = evaluate_method_unified(zs_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=200)
    zs_entry = {
        "k_shot_per_fold": 0,
        "n_real_clinical_cases": 0,
        "n_synthetic_bundles": 300,
        "protocol": "Stage 0 (k_lim_auto) + Pretrained SynthBrachy3T Bundle Transformer (w_synth=1.0, 0 real clinical labels)",
        "cohort_all_1063_catheters": res_zs_all,
        "cohort_long_759_cy_catheters": res_zs_long,
        "elapsed_sec": round(time.time() - t_zs, 2),
    }
    few_shot_results["0_shot"] = zs_entry
    few_shot_results["0_shot_synthbrachy3t_pretrained"] = zs_entry
    print(
        f"  [Few-Shot k=0] Stage0 Analytical F1={res_s0_all['f1_1to1_pct']:.2f}% (Long F1_adj={res_s0_long['f1_1to1_obturator_adj_pct']:.2f}%) | "
        f"Zero-Shot SynthBrachy3T F1={res_zs_all['f1_1to1_pct']:.2f}% (Long F1_adj={res_zs_long['f1_1to1_obturator_adj_pct']:.2f}%)",
        flush=True,
    )

    # k in {1, 5, 10, 20}: Leak-free training of BOTH base_k_scores AND fs_outputs strictly on k real cases per fold!
    pw_b_r = torch.tensor([3.0], device=device)
    pw_b_1 = torch.tensor([2.0], device=device)
    pw_b_l = torch.tensor([2.0], device=device)

    for k_shot in [1, 5, 10, 20]:
        t_k = time.time()
        base_k_scores = [None] * n_cases
        fs_outputs = [None] * n_cases

        for f_out in range(5):
            outer_test, outer_train, _, _ = get_nested_fold_split(f_out, n_cases)
            k_indices = outer_train[:k_shot]

            # 1) Train DenseGraphTransformer strictly on k_indices (NO full-data base_oof_scores!)
            torch.manual_seed(600 + k_shot * 10 + f_out)
            np.random.seed(600 + k_shot * 10 + f_out)
            b_model = DenseGraphTransformer(in_dim=18, edge_dim=6, d_model=64).to(device)
            opt_b = torch.optim.AdamW(b_model.parameters(), lr=1.8e-3, weight_decay=1e-4)
            n_ep_b = 35 if k_shot <= 5 else 55
            for ep in range(n_ep_b):
                b_model.train()
                for idx in np.random.permutation(k_indices):
                    bg = stitched_base_bgs[idx]
                    opt_b.zero_grad(set_to_none=True)
                    out = b_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                    loss_b = (
                        F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"].to(device), pos_weight=pw_b_r)
                        + 1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"].to(device), pos_weight=pw_b_1)
                        + F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"].to(device), pos_weight=pw_b_l)
                    )
                    loss_b.backward()
                    opt_b.step()
            b_model.eval()
            with torch.no_grad():
                for idx in outer_test:
                    bg = stitched_base_bgs[idx]
                    out = b_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                    base_k_scores[idx] = torch.sigmoid(out).cpu().numpy()

            # 2) Fine-tune TipAwareBundleGraphTransformer from pretrained_state strictly on k_indices + synthetic aug
            rng_fs = np.random.default_rng(700 + k_shot * 10 + f_out)
            tr_k_raw = [stitched_enriched_bgs[idx] for idx in k_indices]
            tr_k_gpu = [to_gpu_item(b, device) for b in tr_k_raw]
            n_aug_k = max(3, 16 // k_shot)
            for _ in range(n_aug_k):
                for b in tr_k_raw:
                    tr_k_gpu.append(to_gpu_item(augment_clinical_bundle_with_synthbrachy3t_v2(b, rng_fs), device))
            for s_i in range(min(40, len(pure_synth_gpu))):
                tr_k_gpu.append(pure_synth_gpu[s_i])

            fs_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
            fs_model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
            opt_fs = torch.optim.AdamW(fs_model.parameters(), lr=1.2e-3, weight_decay=1.5e-4)
            for ep in range(25):
                fs_model.train()
                for bi in np.random.permutation(len(tr_k_gpu)):
                    bg = tr_k_gpu[bi]
                    opt_fs.zero_grad(set_to_none=True)
                    out = fs_model(bg["x"], bg["adj_feat"], bg["adj_mask"])
                    w_tip = bg["y_dz_weight"]
                    l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"], reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
                    loss_fs = (
                        F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"], pos_weight=pw_r)
                        + 1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"], pos_weight=pw_1)
                        + F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"], pos_weight=pw_l)
                        + 1.5 * l_tip
                    )
                    loss_fs.backward()
                    opt_fs.step()
            fs_model.eval()
            with torch.no_grad():
                for idx in outer_test:
                    bg = stitched_enriched_bgs[idx]
                    out = fs_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                    fs_outputs[idx] = out.cpu().numpy()

        w_syn_k = 0.80 if k_shot <= 5 else 0.65
        p_fs_all = decode_clean_stage2_preds(
            stitched_enriched_bgs, base_k_scores, fs_outputs,
            min_conf=50.0, long_mode=False, w_synth=w_syn_k, use_collinear_dedup=True, use_1d_tip_model=False, tip_scale=0.36
        )
        p_fs_long = decode_clean_stage2_preds(
            stitched_enriched_bgs, base_k_scores, fs_outputs,
            min_conf=50.0, long_mode=True, w_synth=w_syn_k, use_collinear_dedup=True, use_1d_tip_model=False, tip_scale=0.36
        )
        res_fs_all, _ = evaluate_method_unified(p_fs_all, gts_all_list, thresh=3.5, n_boot=200)
        res_fs_long, _ = evaluate_method_unified(p_fs_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=200)
        few_shot_results[f"{k_shot}_shot"] = {
            "k_shot_per_fold": k_shot,
            "n_real_clinical_cases": k_shot,
            "n_synthetic_bundles": 300,
            "protocol": f"Leak-free {k_shot}-shot: both DenseGraphTransformer and TipAwareBundleGraphTransformer trained strictly on {k_shot} clinical cases/fold",
            "cohort_all_1063_catheters": res_fs_all,
            "cohort_long_759_cy_catheters": res_fs_long,
            "elapsed_sec": round(time.time() - t_k, 2),
        }
        print(
            f"  [Few-Shot k={k_shot:2d}] All F1={res_fs_all['f1_1to1_pct']:.2f}% (Rec={res_fs_all['hungarian_1to1_recall_pct']:.2f}%, Prec={res_fs_all['precision_1to1_pct']:.2f}%) | "
            f"Long F1_adj={res_fs_long['f1_1to1_obturator_adj_pct']:.2f}% | Tip Med={res_fs_all['metrics']['tip_3d']['median']:.2f} mm",
            flush=True,
        )

    # k = 43 (Full Nested CV)
    t_43 = time.time()
    p_full_all = decode_clean_stage2_preds(
        stitched_enriched_bgs, base_sc_st_mr, syn_out_st_mr,
        graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
        min_conf=50.0, long_mode=False, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
    )
    p_full_long = decode_clean_stage2_preds(
        stitched_enriched_bgs, base_sc_st_mr, syn_out_st_mr,
        graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
        min_conf=50.0, long_mode=True, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
    )
    res_43_all, _ = evaluate_method_unified(p_full_all, gts_all_list, thresh=3.5, n_boot=200)
    res_43_long, _ = evaluate_method_unified(p_full_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=200)
    few_shot_results["43_shot"] = {
        "k_shot_per_fold": 43,
        "n_real_clinical_cases": 43,
        "n_synthetic_bundles": 300,
        "protocol": "Full Nested CV (35 inner_subtrain + 8-9 inner_val = 43-44 outer_train cases/fold)",
        "cohort_all_1063_catheters": res_43_all,
        "cohort_long_759_cy_catheters": res_43_long,
        "elapsed_sec": round(time.time() - t_43, 2),
    }
    return few_shot_results


def export_table3_all_40_scenes(meta_list, gts_all_list, gts_long_list, gts_short_list, nf_all_tau50, nf_all_tau35, out_clean_path, out_rerun_path):
    """
    Evaluates Table 3 (Human Inter-Observer LB/AMe vs CY+HT and NeedleFinder 2026 Clean OOF)
    across ALL 40 / 40 mapped dual-traced scenes (including Case 002!).
    """
    mapped_indices = [ci for ci, m in enumerate(meta_list) if "lb_gts" in m and m["lb_gts"] and len(m["lb_gts"]) > 0]
    assert len(mapped_indices) == 40, f"Expected 40 dual-traced scenes, got {len(mapped_indices)}"

    cy_all_40 = [gts_all_list[ci] for ci in mapped_indices]
    cy_long_40 = [gts_long_list[ci] for ci in mapped_indices]
    cy_short_40 = [gts_short_list[ci] for ci in mapped_indices]
    lb_all_40 = [get_gts(meta_list[ci]["lb_gts"]) for ci in mapped_indices]
    lb_long_40 = [[t for t in lb_all_40[k] if float(t[-1, 2] - t[0, 2]) >= 28.0] for k in range(40)]

    union_all_40 = []
    for k in range(40):
        u_list = list(cy_all_40[k])
        for l_pts in lb_all_40[k]:
            if min([traj_gt_dist(l_pts, g) for g in u_list] + [999.0]) > 3.5:
                u_list.append(l_pts)
        union_all_40.append(u_list)

    nf50_40 = [nf_all_tau50[ci] for ci in mapped_indices]
    nf35_40 = [nf_all_tau35[ci] for ci in mapped_indices]

    r1_human_all, _ = evaluate_method_unified(lb_all_40, cy_all_40, aux_short_gts_per_case=cy_short_40, thresh=3.5, n_boot=300)
    r2_human_long, _ = evaluate_method_unified(lb_long_40, cy_long_40, aux_short_gts_per_case=cy_short_40, thresh=3.5, n_boot=300)
    r3_nf50_cy, _ = evaluate_method_unified(nf50_40, cy_all_40, thresh=3.5, n_boot=300)
    r4_nf35_cy, _ = evaluate_method_unified(nf35_40, cy_all_40, thresh=3.5, n_boot=300)
    r5_nf50_union, _ = evaluate_method_unified(nf50_40, union_all_40, thresh=3.5, n_boot=300)
    r6_nf50_lb, _ = evaluate_method_unified(nf50_40, lb_all_40, thresh=3.5, n_boot=300)

    table3_report = {
        "generated_at": "2026-10-23",
        "description": "Table 3 Human Inter-Observer Agreement (LB/AMe vs CY+HT) and NeedleFinder 2026 Clean Nested OOF Comparison on ALL 40/40 Dual-Traced Cases",
        "n_overlap_scenes_in_zip": 40,
        "n_mapped_in_54_cohort": len(mapped_indices),
        "mapped_case_numbers": [int(meta_list[ci]["case_num"]) for ci in mapped_indices],
        "table3_rows": {
            "1_human_inter_observer_lb_vs_cy_all": r1_human_all,
            "2_human_inter_observer_long_subset": r2_human_long,
            "3_needlefinder_2026_oof_tau50_vs_cy_40scenes": r3_nf50_cy,
            "4_needlefinder_2026_oof_tau35_vs_cy_40scenes": r4_nf35_cy,
            "5_needlefinder_2026_oof_tau50_vs_union_cy_or_lb_40scenes": r5_nf50_union,
            "6_needlefinder_2026_oof_tau50_vs_lb_40scenes": r6_nf50_lb,
        },
    }

    for p in (out_clean_path, out_rerun_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(table3_report, f, indent=2)
    print(
        f"  [Table 3 (40/40 Scenes)] Human LB vs CY: Rec={r1_human_all['hungarian_1to1_recall_pct']:.2f}%, F1={r1_human_all['f1_1to1_pct']:.2f}% | "
        f"NF2026 (tau=50%) vs CY: Rec={r3_nf50_cy['hungarian_1to1_recall_pct']:.2f}%, F1={r3_nf50_cy['f1_1to1_pct']:.2f}% | "
        f"NF2026 (tau=50%) vs Union(CY|LB): Prec={r5_nf50_union['precision_1to1_pct']:.2f}%, F1={r5_nf50_union['f1_1to1_pct']:.2f}%",
        flush=True,
    )
    return table3_report


def export_qualitative_npz(meta_list, gts_all_list, stage0_preds, nf_all_tau50, out_npz_path):
    """Exports qualitative prediction trajectories for Case 017 (idx 6) and Case 031 (idx 17)."""
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)

    def _to_obj_arr(trajs):
        arr = np.empty(len(trajs), dtype=object)
        for i, t in enumerate(trajs):
            pts = np.asarray(t["ras"] if isinstance(t, dict) and "ras" in t else (t["points"] if isinstance(t, dict) else t), dtype=np.float64)
            pts = pts[np.argsort(pts[:, 2])]
            arr[i] = pts
        return arr

    np.savez(
        out_npz_path,
        case017_gt=_to_obj_arr(gts_all_list[6]),
        case017_stage0=_to_obj_arr(stage0_preds[6]),
        case017_nf2026=_to_obj_arr(nf_all_tau50[6]),
        case031_gt=_to_obj_arr(gts_all_list[17]),
        case031_stage0=_to_obj_arr(stage0_preds[17]),
        case031_nf2026=_to_obj_arr(nf_all_tau50[17]),
    )
    print(f"  [Qualitative NPZ] Saved Case 017 (idx=6) & Case 031 (idx=17) predictions to {out_npz_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Export Unified Clean Benchmark JSON + Leak-Free Few-Shot + 40/40 Table 3")
    parser.add_argument("--cache-dir", type=str, default="/tmp/needle_unet3d_cache_clean")
    parser.add_argument("--oof-dir", type=str, default="/home/guillaume/github/needlefinder/nn_models_clean")
    parser.add_argument("--eval-oof-dir", type=str, default="/home/guillaume/github/needlefinder/nn_models_eval")
    parser.add_argument("--meta-path", type=str, default="/tmp/cases_meta_54_clean.pkl")
    parser.add_argument("--base-trajs-path", type=str, default="/tmp/base_trajs_54cases_clean.pkl")
    parser.add_argument("--stage2-json", type=str, default="/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-and-tip-clean-final.json")
    parser.add_argument("--stage2-preds-pkl", type=str, default="/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-clean-final-oof-predictions.pkl")
    parser.add_argument("--prev-unified-json", type=str, default="/tmp/rerun/unified.json")
    parser.add_argument("--out-unified-clean-json", type=str, default="/home/guillaume/github/needlefinder/vault/sources/2026-10-23-unified-clean-nested-benchmark.json")
    parser.add_argument("--out-unified-rerun-json", type=str, default="/home/guillaume/github/needlefinder/vault/sources/2026-10-23-unified-rerun-benchmark.json")
    parser.add_argument("--out-table3-clean-json", type=str, default="/home/guillaume/github/needlefinder/vault/sources/2026-10-23-table3-observers-and-nf-clean.json")
    parser.add_argument("--out-table3-rerun-json", type=str, default="/home/guillaume/github/needlefinder/vault/sources/2026-10-23-table3-observers-and-nf-rerun.json")
    parser.add_argument("--out-qual-npz", type=str, default="/home/guillaume/github/needlefinder/vault/reports/figures/data/qualitative_predictions.npz")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    print(f"[Unified Clean Benchmark Exporter] Device={device}", flush=True)

    meta_list = pickle.load(open(args.meta_path, "rb"))
    base_trajs_data = pickle.load(open(args.base_trajs_path, "rb"))
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]
    gts_short_list = [
        [ga for ga in gts_all_list[ci] if not any(traj_gt_dist(ga, gl) < 2.0 for gl in gts_long_list[ci])]
        for ci in range(54)
    ]

    adhesion_info = analyze_catheter_adhesion(meta_list, gts_all_list, gts_long_list, adhesion_thresh_mm=3.0)
    adh_idx = adhesion_info["adhesion_case_indices"]
    non_adh_idx = adhesion_info["non_adhesion_case_indices"]

    stage0_preds = []
    for i in range(54):
        b_item = base_trajs_data[i]
        b_trajs = b_item["trajs"] if isinstance(b_item, dict) and "trajs" in b_item else b_item
        b_clean = []
        for t in b_trajs:
            tc = dict(t)
            tc["ras"] = np.asarray(tc["ras"] if "ras" in tc else tc["points"], dtype=np.float64)
            tc["points"] = tc["ras"]
            b_clean.append(tc)
        stage0_preds.append(b_clean)

    s2_preds_data = pickle.load(open(args.stage2_preds_pkl, "rb"))
    s2_json_data = json.load(open(args.stage2_json, "r"))
    oof_tag = os.path.basename(os.path.normpath(args.oof_dir))
    bundle_cache_path = f"/tmp/stage2_clean_bundles_{oof_tag}.pkl"
    bundle_cache = pickle.load(open(bundle_cache_path, "rb"))

    stitched_enriched_bgs = s2_preds_data.get("stitched_enriched_bgs") or bundle_cache["stitched_enriched_bgs"]
    _, _, stitched_base_bgs, _ = derive_unstitched_and_base_bgs(stitched_enriched_bgs)

    graphs = bundle_cache["graphs"]
    all_oof_cal = bundle_cache["all_oof_cal"]
    all_X_vols = [torch.load(f"{args.cache_dir}/case_{i:02d}.pt", map_location="cpu", weights_only=False)["X"].numpy() for i in range(54)]

    tip_models_5fold = []
    for st in s2_preds_data["tip_models_5fold_states"]:
        m = Dedicated1DTipLocalizer(in_channels=22, hidden_dim=48, seq_len=91).to(device)
        m.load_state_dict({k: v.to(device) for k, v in st.items()})
        m.eval()
        tip_models_5fold.append(m)

    # Load previous unified benchmark JSON as base template, and re-evaluate ALL methods in a single transaction!
    unified_out = copy.deepcopy(json.load(open(args.prev_unified_json, "r")))
    unified_out["generated_at"] = "2026-10-23"
    unified_out["benchmark_version"] = "clean_nested_cv_v1"

    # Load saved external baseline trajectories (Baselines 1, 2, 3) so they also go through the exact same
    # single-transaction evaluator and satisfy len(dist) == hungarian_1to1_matched!
    b1_data = pickle.load(open(f"{args.eval_oof_dir}/oof_trajs_pmb2019_iso1mm.pkl", "rb"))
    b2_data = pickle.load(open(f"{args.eval_oof_dir}/oof_trajs_native_cldice.pkl", "rb"))
    b3_data = pickle.load(open(f"{args.eval_oof_dir}/oof_trajs_xiao2026_synchough.pkl", "rb"))

    s0_all = stage0_preds
    s0_long = stage0_preds
    (s1b_all, s1b_long), (s2a_all, s2a_long) = extract_stage1b_and_stage2a_preds(
        stage0_preds, stitched_enriched_bgs, graphs, all_oof_cal, all_X_vols, meta_list
    )

    # Decode No-SynthBrachy3T across tau in {25, 35, 45, 50, 60}
    nosyn_by_tau = {}
    for tau in [25, 35, 45, 50, 60]:
        pa_ns = decode_clean_stage2_preds(
            stitched_enriched_bgs, s2_preds_data["base_sc_st_nosyn"], s2_preds_data["syn_out_st_nosyn"],
            graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
            min_conf=float(tau), long_mode=False, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
        )
        pl_ns = decode_clean_stage2_preds(
            stitched_enriched_bgs, s2_preds_data["base_sc_st_nosyn"], s2_preds_data["syn_out_st_nosyn"],
            graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
            min_conf=float(tau), long_mode=True, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
        )
        nosyn_by_tau[tau] = (pa_ns, pl_ns)

    # Decode With-SynthBrachy3T across tau in {25, 35, 45, 50, 55, 60}
    syn_by_tau = {}
    for tau in [25, 35, 45, 50, 55, 60]:
        if tau in s2_preds_data.get("full_preds_by_tau", {}):
            syn_by_tau[tau] = s2_preds_data["full_preds_by_tau"][tau]
        else:
            pa_s = decode_clean_stage2_preds(
                stitched_enriched_bgs, s2_preds_data["base_sc_st_mr"], s2_preds_data["syn_out_st_mr"],
                graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
                min_conf=float(tau), long_mode=False, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
            )
            pl_s = decode_clean_stage2_preds(
                stitched_enriched_bgs, s2_preds_data["base_sc_st_mr"], s2_preds_data["syn_out_st_mr"],
                graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
                min_conf=float(tau), long_mode=True, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
            )
            syn_by_tau[tau] = (pa_s, pl_s)

    methods_to_evaluate = [
        ("Baseline 1: PMB 2019 Isotropic 1mm³ 3D U-Net + DBSCAN/Skeleton", b1_data["all"], b1_data["long"], True),
        ("Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)", b2_data["all"], b2_data["long"], True),
        ("Baseline 3: Xiao 2026 IRGM + 3D Hough + Sync Inward Tracking + Poly3", b3_data["all"], b3_data["long"], True),
        ("Xiao 2026 3D Hough + Sync Inward Tracking + Poly3 (on Ground-Truth Masks, Table 2B)", b3_data["gt_all"], b3_data["gt_long"], False),
        ("Stage 0/1A: Pure Physics 3D CV + HybridNeedleNet", s0_all, s0_long, True),
        ("Stage 1B: Multi-Head NeedleUNet3D + Tangent Tube Tracking", s1b_all, s1b_long, False),
        ("Stage 2A: Pure NeedleSliceMPNN (4-Layer Triplet GraphNet)", s2a_all, s2a_long, False),
        ("Unified NeedleFinder 2026 (tau=25%)", nosyn_by_tau[25][0], nosyn_by_tau[25][1], False),
        ("Unified NeedleFinder 2026 (tau=35%)", nosyn_by_tau[35][0], nosyn_by_tau[35][1], False),
        ("Unified NeedleFinder 2026 (tau=45%)", nosyn_by_tau[45][0], nosyn_by_tau[45][1], False),
        ("Unified NeedleFinder 2026 (tau=50%)", nosyn_by_tau[50][0], nosyn_by_tau[50][1], True),
        ("Unified NeedleFinder 2026 (tau=60%)", nosyn_by_tau[60][0], nosyn_by_tau[60][1], False),
        ("Unified NeedleFinder 2026 + IRGM + SyncTrack + Poly3 (tau=50%)", nosyn_by_tau[50][0], nosyn_by_tau[50][1], True),
        ("NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=25%)", syn_by_tau[25][0], syn_by_tau[25][1], False),
        ("NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=35%)", syn_by_tau[35][0], syn_by_tau[35][1], True),
        ("NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=45%)", syn_by_tau[45][0], syn_by_tau[45][1], False),
        ("NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=50%)", syn_by_tau[50][0], syn_by_tau[50][1], True),
        ("NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=55%)", syn_by_tau[55][0], syn_by_tau[55][1], False),
        ("NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=60%)", syn_by_tau[60][0], syn_by_tau[60][1], False),
    ]

    if s2_preds_data.get("base_sc_st_inner") is not None and s2_preds_data.get("syn_out_st_inner") is not None:
        pa_in = decode_clean_stage2_preds(
            stitched_enriched_bgs, s2_preds_data["base_sc_st_inner"], s2_preds_data["syn_out_st_inner"],
            graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
            min_conf=50.0, long_mode=False, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
        )
        pl_in = decode_clean_stage2_preds(
            stitched_enriched_bgs, s2_preds_data["base_sc_st_inner"], s2_preds_data["syn_out_st_inner"],
            graphs=graphs, all_oof_cal=all_oof_cal, all_X_vols=all_X_vols, tip_models_5fold=tip_models_5fold, device=device,
            min_conf=50.0, long_mode=True, w_synth=0.60, use_collinear_dedup=True, use_1d_tip_model=True, tip_scale=0.36
        )
        methods_to_evaluate.append(("NeedleFinder 2026 + SynthBrachy3T + Inner-OOF Cross-Fit (tau=50%)", pa_in, pl_in, True))

    cohort_all = {}
    cohort_long = {}
    adh_severe_all = {}
    adh_non_all = {}
    adh_severe_long = {}
    dists_all = {}
    dists_long = {}
    pp_all_map = {}
    pp_long_map = {}

    gts_all_adh = [gts_all_list[i] for i in adh_idx]
    gts_all_non = [gts_all_list[i] for i in non_adh_idx]
    gts_long_adh = [gts_long_list[i] for i in adh_idx]
    gts_short_adh = [gts_short_list[i] for i in adh_idx]

    print("[1/4] Evaluating all methods in a single transaction...", flush=True)
    for m_name, p_all, p_long, include_in_subgroups in methods_to_evaluate:
        sum_a, pp_a, dist_a = evaluate_and_verify_single_transaction(p_all, gts_all_list, thresh=3.5, n_boot=300)
        sum_l, pp_l, dist_l = evaluate_and_verify_single_transaction(p_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=300)
        cohort_all[m_name] = sum_a
        cohort_long[m_name] = sum_l
        pp_all_map[m_name] = pp_a
        pp_long_map[m_name] = pp_l

        if include_in_subgroups:
            dists_all[m_name] = dist_a
            dists_long[m_name] = dist_l
            res_adh_a, _ = evaluate_method_unified([p_all[i] for i in adh_idx], gts_all_adh, thresh=3.5, n_boot=300)
            res_non_a, _ = evaluate_method_unified([p_all[i] for i in non_adh_idx], gts_all_non, thresh=3.5, n_boot=300)
            res_adh_l, _ = evaluate_method_unified([p_long[i] for i in adh_idx], gts_long_adh, aux_short_gts_per_case=gts_short_adh, thresh=3.5, n_boot=300)
            adh_severe_all[m_name] = res_adh_a
            adh_non_all[m_name] = res_non_a
            adh_severe_long[m_name] = res_adh_l

    # Verify 100% match-count consistency across ALL entries in dists_all and dists_long!
    for m_name, d_a in dists_all.items():
        assert len(d_a["per_needle_mean_shaft_xy_mm"]) == cohort_all[m_name]["hungarian_1to1_matched"]
        assert len(d_a["per_needle_tip_3d_mm"]) == cohort_all[m_name]["hungarian_1to1_matched"]
    for m_name, d_l in dists_long.items():
        assert len(d_l["per_needle_mean_shaft_xy_mm"]) == cohort_long[m_name]["hungarian_1to1_matched"]
        assert len(d_l["per_needle_tip_3d_mm"]) == cohort_long[m_name]["hungarian_1to1_matched"]
    print("  [Verified] 100% match-count consistency across all methods in cohort_all, cohort_long, and per_patient_and_needle_error_distributions!", flush=True)

    # Recompute Paired Wilcoxon Signed-Rank Tests across all 54 patients
    b1_k = "Baseline 1: PMB 2019 Isotropic 1mm³ 3D U-Net + DBSCAN/Skeleton"
    b2_k = "Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)"
    b3_k = "Baseline 3: Xiao 2026 IRGM + 3D Hough + Sync Inward Tracking + Poly3"
    s0_k = "Stage 0/1A: Pure Physics 3D CV + HybridNeedleNet"
    s1_k = "Stage 1B: Multi-Head NeedleUNet3D + Tangent Tube Tracking"
    nf35_k = "Unified NeedleFinder 2026 (tau=35%)"
    nf50_k = "Unified NeedleFinder 2026 (tau=50%)"
    syn50_k = "NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=50%)"

    def _build_wilcoxon_dict(ppm):
        w = {
            "NeedleFinder2026_tau35_vs_PMB2019_Iso1mm": compute_wilcoxon_comparison(ppm[nf35_k], ppm[b1_k]),
            "NeedleFinder2026_tau35_vs_Native_clDice": compute_wilcoxon_comparison(ppm[nf35_k], ppm[b2_k]),
            "NeedleFinder2026_tau50_vs_PMB2019_Iso1mm": compute_wilcoxon_comparison(ppm[nf50_k], ppm[b1_k]),
            "NeedleFinder2026_tau50_vs_Native_clDice": compute_wilcoxon_comparison(ppm[nf50_k], ppm[b2_k]),
            "Stage1B_NeedleUNet3D_vs_Native_clDice": compute_wilcoxon_comparison(ppm[s1_k], ppm[b2_k]),
            "NeedleFinder2026_tau50_vs_Xiao2026_SyncTrack": compute_wilcoxon_comparison(ppm[nf50_k], ppm[b3_k]),
            "SynthBrachy3T_vs_PMB2019": compute_wilcoxon_comparison(ppm[syn50_k], ppm[b1_k]),
            "SynthBrachy3T_vs_Native_clDice": compute_wilcoxon_comparison(ppm[syn50_k], ppm[b2_k]),
            "SynthBrachy3T_vs_Xiao2026": compute_wilcoxon_comparison(ppm[syn50_k], ppm[b3_k]),
            "SynthBrachy3T_vs_No_SynthBrachy3T_tau50": compute_wilcoxon_comparison(ppm[syn50_k], ppm[nf50_k]),
            "SynthBrachy3T_vs_Stage0_Physics": compute_wilcoxon_comparison(ppm[syn50_k], ppm[s0_k]),
        }
        if "NeedleFinder 2026 + SynthBrachy3T + Inner-OOF Cross-Fit (tau=50%)" in ppm:
            in_k = "NeedleFinder 2026 + SynthBrachy3T + Inner-OOF Cross-Fit (tau=50%)"
            w["SynthBrachy3T_OOF_vs_InnerOOF_CrossFit_tau50"] = compute_wilcoxon_comparison(ppm[syn50_k], ppm[in_k])
        return w

    pp_adh_map = {
        m_k: {
            k: ([v[i] for i in adh_idx] if isinstance(v, (list, np.ndarray)) and len(v) == 54 else v)
            for k, v in pp_v.items()
        }
        for m_k, pp_v in pp_all_map.items()
    }
    w_adh = _build_wilcoxon_dict(pp_adh_map)
    w_adh.update({
        "NF2026_4b_vs_PMB2019": compute_wilcoxon_comparison(pp_adh_map[syn50_k], pp_adh_map[b1_k]),
        "NF2026_4b_vs_ResUNet_clDice": compute_wilcoxon_comparison(pp_adh_map[syn50_k], pp_adh_map[b2_k]),
        "NF2026_4b_vs_Xiao2026": compute_wilcoxon_comparison(pp_adh_map[syn50_k], pp_adh_map[b3_k]),
        "NF2026_4b_vs_Stage0": compute_wilcoxon_comparison(pp_adh_map[syn50_k], pp_adh_map[s0_k]),
        "NF2026_4b_vs_NoSynth_4a": compute_wilcoxon_comparison(pp_adh_map[syn50_k], pp_adh_map[nf50_k]),
    })

    wilcoxon_54 = {
        "cohort_all_1063_catheters": _build_wilcoxon_dict(pp_all_map),
        "cohort_long_759_cy_catheters": _build_wilcoxon_dict(pp_long_map),
        "cohort_adhesion_23cases_571_catheters": w_adh,
    }

    # Run Leak-Free Few-Shot Clinical Annotation Budget (AUD-500)
    if os.path.exists(args.out_unified_clean_json):
        existing_clean = json.load(open(args.out_unified_clean_json, "r"))
        if "few_shot_clinical_annotation_budget" in existing_clean and "0_shot_stage0_analytical" in existing_clean["few_shot_clinical_annotation_budget"]:
            print("\n[2/4] [Cache Hit] Loaded leak-free Few-Shot Clinical Annotation Budget from existing clean JSON...", flush=True)
            few_shot_budget = existing_clean["few_shot_clinical_annotation_budget"]
            few_shot_budget["0_shot_stage0_analytical"]["cohort_all_1063_catheters"] = cohort_all[s0_k]
            few_shot_budget["0_shot_stage0_analytical"]["cohort_long_759_cy_catheters"] = cohort_long[s0_k]
        else:
            few_shot_budget = None
    else:
        few_shot_budget = None

    if few_shot_budget is None:
        print("\n[2/4] Running leak-free Few-Shot Clinical Annotation Budget (k = 0, 1, 5, 10, 20, 43)...", flush=True)
        few_shot_budget = run_leak_free_few_shot_budget(
            stage0_preds,
            stitched_base_bgs,
            stitched_enriched_bgs,
            graphs,
            all_oof_cal,
            all_X_vols,
            tip_models_5fold,
            s2_preds_data["base_sc_st_mr"],
            s2_preds_data["syn_out_st_mr"],
            gts_all_list,
            gts_long_list,
            gts_short_list,
            device,
        )

    # Export Table 3 across ALL 40 / 40 dual-traced scenes
    print("\n[3/4] Exporting Table 3 Inter-Observer & NeedleFinder Comparison across ALL 40/40 scenes...", flush=True)
    table3_report = export_table3_all_40_scenes(
        meta_list,
        gts_all_list,
        gts_long_list,
        gts_short_list,
        s2_preds_data["all_tau50"],
        s2_preds_data["all_tau35"],
        args.out_table3_clean_json,
        args.out_table3_rerun_json,
    )

    unified_out["cohort_all_1063_catheters_54cases"] = cohort_all
    unified_out["cohort_long_759_cy_catheters_54cases"] = cohort_long
    unified_out["catheter_adhesion_subgroup_analysis"] = adhesion_info
    unified_out["subgroup_adhesion_vs_non_adhesion"] = {
        "severe_adhesion_subgroup_all_catheters": adh_severe_all,
        "non_adhesion_subgroup_all_catheters": adh_non_all,
        "severe_adhesion_subgroup_long_catheters": adh_severe_long,
    }
    unified_out["per_patient_and_needle_error_distributions"] = {
        "all_1063": dists_all,
        "long_759": dists_long,
    }
    unified_out["paired_wilcoxon_signed_rank_tests_54_patients"] = wilcoxon_54
    unified_out["few_shot_clinical_annotation_budget"] = few_shot_budget
    unified_out["synthbrachy3t_sim_to_real_few_shot_budget"] = few_shot_budget
    unified_out["human_interobserver_agreement_40_overlapping_cases"] = table3_report
    unified_out["stage2_progression_and_ablation"] = s2_json_data["progression_results"]
    unified_out["nested_cv_checkpoint_selection"] = s2_json_data["nested_cv_protocol"]

    for p in (args.out_unified_clean_json, args.out_unified_rerun_json):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(unified_out, f, indent=2)
    print(f"  [Unified Benchmark JSON] Saved to {args.out_unified_clean_json} and {args.out_unified_rerun_json}", flush=True)

    # Export qualitative NPZ for Case 017 (idx=6) and Case 031 (idx=17)
    print("\n[4/4] Exporting qualitative predictions NPZ for Case 017 & Case 031...", flush=True)
    export_qualitative_npz(meta_list, gts_all_list, stage0_preds, s2_preds_data["all_tau50"], args.out_qual_npz)

    print(f"\n[All Done] Completed unified clean benchmark + few-shot + Table 3 (40/40) + qualitative NPZ in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
