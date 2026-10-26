#!/usr/bin/env python3
"""
Finalize Unified Benchmark JSON with NeedleFinder 2026 + SynthBrachy3T + Tip-Aware
(using cached /tmp/cached_enriched_bundles_54.pkl and /tmp/cached_ens_synth_oof_54.pkl).
Runs in ~25 seconds on edelweiss RTX 4090 (cuda:0).
"""
import os, sys, json, time, pickle
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/tmp")

from run_synthbrachy3t_full import (
    evaluate_method_unified,
    compute_wilcoxon_comparison,
    analyze_catheter_adhesion,
    get_gts,
    traj_gt_dist,
    decode_hybrid_synth_nf2026_preds,
    get_nf2026_synctrack_poly3_preds,
    generate_synthetic_bundle_graph_item,
    augment_clinical_bundle_with_synthbrachy3t,
    TipAwareBundleGraphTransformer,
    to_gpu_item,
    META_PATH,
    OOF_DIR,
    ENRICHED_BUNDLES_CACHE_PATH,
)
from run_xiao2026_and_fewshot_benchmark import (
    extract_per_patient_and_needle_distributions,
    regularize_poly3_trajectory,
    dedup_traj_dicts,
)


def main():
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Finalize SynthBrachy3T] Running on {device}...", flush=True)

    with open("/tmp/2026-10-22-unified-apples-to-apples-benchmark.json") as f:
        bench = json.load(f)

    meta_list = pickle.load(open(META_PATH, "rb"))
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]
    gts_short_list = [[ga for ga in gts_all_list[i] if not any(traj_gt_dist(ga, gl) < 2.0 for gl in gts_long_list[i])] for i in range(54)]

    adhesion_info = analyze_catheter_adhesion(meta_list, gts_all_list, gts_long_list, adhesion_thresh_mm=3.0)
    adh_idx = adhesion_info["adhesion_case_indices"]
    non_adh_idx = adhesion_info["non_adhesion_case_indices"]

    cache_d = pickle.load(open(ENRICHED_BUNDLES_CACHE_PATH, "rb"))
    bundle_items = cache_d["bundle_items"]
    base_oof_scores = cache_d["base_oof_scores"]
    stage0_preds = cache_d["stage0_preds"]
    ens_oof_outputs = pickle.load(open("/tmp/cached_ens_synth_oof_54.pkl", "rb"))

    # 1. Decode NeedleFinder 2026 + SynthBrachy3T + Tip-Aware across tau in {25, 35, 45, 50, 55, 60}%
    synth_preds_all = {}
    synth_preds_long = {}
    for tau in [25, 35, 45, 50, 55, 60]:
        synth_preds_all[tau] = decode_hybrid_synth_nf2026_preds(
            bundle_items, base_oof_scores, ens_oof_outputs, min_conf=float(tau), long_mode=False, w_synth=0.60, tip_scale=0.36
        )
        synth_preds_long[tau] = decode_hybrid_synth_nf2026_preds(
            bundle_items, base_oof_scores, ens_oof_outputs, min_conf=float(tau) * 0.85, long_mode=True, w_synth=0.60, tip_scale=0.36
        )

    # Evaluate main operating point (tau=50%) and high-precision/sensitivity sweep with 1,000 bootstrap iterations
    name_main = "NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=50%)"
    res_synth50_all, pp_synth50_all = evaluate_method_unified(synth_preds_all[50], gts_all_list, thresh=3.5, n_boot=1000)
    res_synth50_long, pp_synth50_long = evaluate_method_unified(synth_preds_long[50], gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=1000)
    bench["cohort_all_1063_catheters_54cases"][name_main] = res_synth50_all
    bench["cohort_long_759_cy_catheters_54cases"][name_main] = res_synth50_long

    print(
        f"[MAIN] {name_main}\n"
        f"  ALL-1063 : Rec={res_synth50_all['hungarian_1to1_recall_pct']:.2f}% Prec={res_synth50_all['precision_1to1_pct']:.2f}% F1={res_synth50_all['f1_1to1_pct']:.2f}% "
        f"Shaft={res_synth50_all['metrics']['mean_shaft_xy']['mean']:.3f}±{res_synth50_all['metrics']['mean_shaft_xy']['sd']:.3f}mm (med {res_synth50_all['metrics']['mean_shaft_xy']['median']:.3f}mm) "
        f"D_avg={res_synth50_all['metrics']['msd_3d']['median']:.2f}mm HD95={res_synth50_all['metrics']['hd95_3d']['median']:.2f}mm "
        f"Tip={res_synth50_all['metrics']['tip_3d']['mean']:.2f}mm (med {res_synth50_all['metrics']['tip_3d']['median']:.2f}mm)\n"
        f"  LONG-759 : Rec={res_synth50_long['hungarian_1to1_recall_pct']:.2f}% Prec_adj={res_synth50_long['precision_1to1_obturator_adj_pct']:.2f}% F1_adj={res_synth50_long['f1_1to1_obturator_adj_pct']:.2f}% "
        f"Shaft={res_synth50_long['metrics']['mean_shaft_xy']['mean']:.3f}mm (med {res_synth50_long['metrics']['mean_shaft_xy']['median']:.3f}mm) "
        f"Tip={res_synth50_long['metrics']['tip_3d']['mean']:.2f}mm (med {res_synth50_long['metrics']['tip_3d']['median']:.2f}mm)",
        flush=True,
    )

    for tau in [25, 35, 45, 55, 60]:
        nm = f"NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau={tau}%)"
        ra, _ = evaluate_method_unified(synth_preds_all[tau], gts_all_list, thresh=3.5, n_boot=300)
        rl, _ = evaluate_method_unified(synth_preds_long[tau], gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=300)
        bench["cohort_all_1063_catheters_54cases"][nm] = ra
        bench["cohort_long_759_cy_catheters_54cases"][nm] = rl

    # 2. Paired Wilcoxon signed-rank tests (54 patients) against all baselines and previous NF2026
    b1_data = pickle.load(open(f"{OOF_DIR}/oof_trajs_pmb2019_iso1mm.pkl", "rb"))
    b2_data = pickle.load(open(f"{OOF_DIR}/oof_trajs_native_cldice.pkl", "rb"))
    b3_data = pickle.load(open(f"{OOF_DIR}/oof_trajs_xiao2026_synchough.pkl", "rb"))
    nf2026_irgm_all = get_nf2026_synctrack_poly3_preds(bundle_items, base_oof_scores, meta_list, min_conf=50.0, long_mode=False)

    _, pp_b1 = evaluate_method_unified(b1_data["all"], gts_all_list, thresh=3.5, n_boot=20)
    _, pp_b2 = evaluate_method_unified(b2_data["all"], gts_all_list, thresh=3.5, n_boot=20)
    _, pp_b3 = evaluate_method_unified(b3_data["all"], gts_all_list, thresh=3.5, n_boot=20)
    _, pp_irgm = evaluate_method_unified(nf2026_irgm_all, gts_all_list, thresh=3.5, n_boot=20)

    w_tests = bench["paired_wilcoxon_signed_rank_tests_54_patients"]["cohort_all_1063_catheters"]
    w_tests["SynthBrachy3T_vs_PMB2019"] = compute_wilcoxon_comparison(pp_synth50_all, pp_b1)
    w_tests["SynthBrachy3T_vs_Native_clDice"] = compute_wilcoxon_comparison(pp_synth50_all, pp_b2)
    w_tests["SynthBrachy3T_vs_Xiao2026"] = compute_wilcoxon_comparison(pp_synth50_all, pp_b3)
    w_tests["SynthBrachy3T_vs_NF2026_IRGM_Poly3"] = compute_wilcoxon_comparison(pp_synth50_all, pp_irgm)

    for k_cmp in ["SynthBrachy3T_vs_PMB2019", "SynthBrachy3T_vs_Native_clDice", "SynthBrachy3T_vs_Xiao2026", "SynthBrachy3T_vs_NF2026_IRGM_Poly3"]:
        wt = w_tests[k_cmp]
        print(f"  Wilcoxon {k_cmp:36s}: F1 p={wt['f1_1to1_pct']['p_value']:.3e} | Tip3D p={wt['tip_3d']['p_value']:.3e}", flush=True)

    # 3. Severe Catheter-Adhesion Subgroup (23 patients, 233 catheters <= 3.0 mm)
    gts_all_adh = [gts_all_list[i] for i in adh_idx]
    gts_all_non = [gts_all_list[i] for i in non_adh_idx]
    gts_long_adh = [gts_long_list[i] for i in adh_idx]
    gts_short_adh = [gts_short_list[i] for i in adh_idx]

    res_adh_a, _ = evaluate_method_unified([synth_preds_all[50][i] for i in adh_idx], gts_all_adh, thresh=3.5, n_boot=500)
    res_non_a, _ = evaluate_method_unified([synth_preds_all[50][i] for i in non_adh_idx], gts_all_non, thresh=3.5, n_boot=500)
    res_adh_l, _ = evaluate_method_unified([synth_preds_long[50][i] for i in adh_idx], gts_long_adh, aux_short_gts_per_case=gts_short_adh, thresh=3.5, n_boot=500)

    bench["subgroup_adhesion_vs_non_adhesion"]["severe_adhesion_subgroup_all_catheters"][name_main] = res_adh_a
    bench["subgroup_adhesion_vs_non_adhesion"]["non_adhesion_subgroup_all_catheters"][name_main] = res_non_a
    bench["subgroup_adhesion_vs_non_adhesion"]["severe_adhesion_subgroup_long_catheters"][name_main] = res_adh_l
    bench["per_patient_and_needle_error_distributions"]["all_1063"][name_main] = extract_per_patient_and_needle_distributions(synth_preds_all[50], gts_all_list, thresh=3.5)
    bench["per_patient_and_needle_error_distributions"]["long_759"][name_main] = extract_per_patient_and_needle_distributions(synth_preds_long[50], gts_long_list, thresh=3.5)

    print(
        f"[ADHESION SUBGROUP (23 pts, 233 cath)] {name_main}\n"
        f"  ALL-233  : Rec={res_adh_a['hungarian_1to1_recall_pct']:.2f}% Prec={res_adh_a['precision_1to1_pct']:.2f}% F1={res_adh_a['f1_1to1_pct']:.2f}% "
        f"ShaftMed={res_adh_a['metrics']['mean_shaft_xy']['median']:.3f}mm TipMean/Med={res_adh_a['metrics']['tip_3d']['mean']:.2f}/{res_adh_a['metrics']['tip_3d']['median']:.2f}mm\n"
        f"  LONG-194 : Rec={res_adh_l['hungarian_1to1_recall_pct']:.2f}% F1_adj={res_adh_l['f1_1to1_obturator_adj_pct']:.2f}% TipMed={res_adh_l['metrics']['tip_3d']['median']:.2f}mm",
        flush=True,
    )

    # 4. Few-Shot Sim-to-Real Evaluation (k = 0, 1, 5, 10, 20, 43 clinical patients/fold)
    print("\n[4/4] Computing SynthBrachy3T Sim-to-Real Few-Shot Curve (k = 0, 1, 5, 10, 20, 43)...", flush=True)
    rng_syn = np.random.default_rng(2026)
    pure_synth_gpu = [to_gpu_item(generate_synthetic_bundle_graph_item(rng_syn), device) for _ in range(300)]
    torch.manual_seed(2026)
    base_syn_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
    opt_syn = torch.optim.AdamW(base_syn_model.parameters(), lr=2.0e-3, weight_decay=1e-4)
    pw_r = torch.tensor([3.2], device=device)
    pw_1 = torch.tensor([2.5], device=device)
    pw_l = torch.tensor([2.5], device=device)
    for ep in range(20):
        base_syn_model.train()
        for bi in np.random.permutation(len(pure_synth_gpu)):
            bg = pure_synth_gpu[bi]
            opt_syn.zero_grad(set_to_none=True)
            out = base_syn_model(bg["x"], bg["adj_feat"], bg["adj_mask"])
            w_tip = bg["y_dz_weight"]
            l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"], reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
            loss = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"], pos_weight=pw_r) + \
                   1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"], pos_weight=pw_1) + \
                   F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"], pos_weight=pw_l) + \
                   1.2 * l_tip
            loss.backward()
            opt_syn.step()
    pretrained_state = {k: v.cpu().clone() for k, v in base_syn_model.state_dict().items()}

    synth_fewshot = {}
    for k_shot in [0, 1, 5, 10, 20, 43]:
        if k_shot == 0:
            # Zero-Shot Sim-to-Real: Stage 0 3D Radial Dark-Core Prior + SynthBrachy3T Prior & Poly3 Tip Regularization (0 real BWH labels!)
            zs_all, zs_long = [], []
            for i, s0_list in enumerate(stage0_preds):
                c_all, c_long = [], []
                # IRGM hull from Stage 0 high-confidence trajectories
                high_pts = [t["ras"][:, :2] for t in s0_list if float(t.get("confidence", 65.0)) >= 62.0]
                if len(high_pts) >= 3:
                    all_xy = np.concatenate(high_pts, axis=0)
                    c_xy = np.median(all_xy, axis=0)
                    r_irgm = float(np.percentile(np.linalg.norm(all_xy - c_xy[None, :], axis=1), 98.0)) + 14.0
                else:
                    c_xy, r_irgm = None, 999.0
                for t in s0_list:
                    pts = np.asarray(t["ras"], dtype=np.float64)
                    pts = pts[np.argsort(pts[:, 2])]
                    conf = float(t.get("confidence", 65.0))
                    if conf < 52.0:
                        continue
                    if c_xy is not None and float(np.linalg.norm(pts[:, :2].mean(axis=0) - c_xy)) > r_irgm and conf < 72.0:
                        continue
                    pts_poly = regularize_poly3_trajectory(pts, degree=3)
                    pts_blend = 0.68 * pts + 0.32 * pts_poly
                    c_all.append({"ras": pts_blend, "points": pts_blend, "score": conf})
                    if (pts[-1, 2] - pts[0, 2]) >= 22.0:
                        c_long.append({"ras": pts_blend, "points": pts_blend, "score": conf})
                zs_all.append(dedup_traj_dicts(c_all, min_len=9.0, dedup_mm=2.05))
                zs_long.append(dedup_traj_dicts(c_long, min_len=22.0, dedup_mm=2.05))
            ra, _ = evaluate_method_unified(zs_all, gts_all_list, thresh=3.5, n_boot=200)
            rl, _ = evaluate_method_unified(zs_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=200)
        elif k_shot == 43:
            ra, rl = res_synth50_all, res_synth50_long
        else:
            fs_outputs = [None] * 54
            for fold in range(5):
                rng_f = np.random.default_rng(700 + fold * 50 + k_shot)
                tr_b_raw = [b for i, b in enumerate(bundle_items) if i % 5 != fold][:k_shot]
                tr_aug_gpu = [to_gpu_item(b, device) for b in tr_b_raw]
                n_aug = max(4, 24 // k_shot)
                for _ in range(n_aug):
                    for b in tr_b_raw:
                        tr_aug_gpu.append(to_gpu_item(augment_clinical_bundle_with_synthbrachy3t(b, rng_f), device))
                gt_model = TipAwareBundleGraphTransformer(in_dim=26, edge_dim=8, d_model=64).to(device)
                gt_model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
                opt = torch.optim.AdamW(gt_model.parameters(), lr=1.3e-3, weight_decay=1.5e-4)
                n_ep = 25 if k_shot <= 5 else 40
                for ep in range(n_ep):
                    gt_model.train()
                    for bi in np.random.permutation(len(tr_aug_gpu)):
                        bg = tr_aug_gpu[bi]
                        opt.zero_grad(set_to_none=True)
                        out = gt_model(bg["x"], bg["adj_feat"], bg["adj_mask"])
                        w_tip = bg["y_dz_weight"]
                        l_tip = (F.smooth_l1_loss(out[:, 3], bg["y_dz_tip"], reduction="none") * w_tip).sum() / w_tip.sum().clamp(min=1.0)
                        loss = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"], pos_weight=pw_r) + \
                               1.5 * F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"], pos_weight=pw_1) + \
                               F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"], pos_weight=pw_l) + \
                               1.5 * l_tip
                        loss.backward()
                        opt.step()
                gt_model.eval()
                with torch.no_grad():
                    for i, bg in enumerate(bundle_items):
                        if i % 5 == fold:
                            out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                            fs_outputs[i] = out.cpu().numpy()

            w_syn_k = float(np.clip(0.85 - 0.006 * k_shot, 0.58, 0.85))
            pa = decode_hybrid_synth_nf2026_preds(bundle_items, base_oof_scores, fs_outputs, min_conf=50.0, long_mode=False, w_synth=w_syn_k, tip_scale=0.36)
            pl = decode_hybrid_synth_nf2026_preds(bundle_items, base_oof_scores, fs_outputs, min_conf=42.5, long_mode=True, w_synth=w_syn_k, tip_scale=0.36)
            ra, _ = evaluate_method_unified(pa, gts_all_list, thresh=3.5, n_boot=200)
            rl, _ = evaluate_method_unified(pl, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=200)

        synth_fewshot[f"{k_shot}_shot"] = {
            "k_shot_per_fold": k_shot,
            "cohort_all_1063_catheters": ra,
            "cohort_long_759_cy_catheters": rl,
        }
        print(
            f"  [SynthBrachy3T Sim-to-Real k={k_shot:2d}] ALL F1={ra['f1_1to1_pct']:5.2f}% (Rec={ra['hungarian_1to1_recall_pct']:5.2f}% Prec={ra['precision_1to1_pct']:5.2f}%) "
            f"| LONG F1_adj={rl['f1_1to1_obturator_adj_pct']:5.2f}% | ShaftMed={ra['metrics']['mean_shaft_xy']['median']:.3f}mm TipMed={ra['metrics']['tip_3d']['median']:.2f}mm",
            flush=True,
        )

    bench["synthbrachy3t_sim_to_real_few_shot_budget"] = synth_fewshot

    out_path = "/tmp/2026-10-22-unified-apples-to-apples-benchmark.json"
    with open(out_path, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"\n[COMPLETE] Updated {out_path} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
