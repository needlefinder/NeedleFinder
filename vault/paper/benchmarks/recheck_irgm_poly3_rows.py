#!/usr/bin/env python3
"""Recompute every Table 2 cell of the 'NeedleFinder 2026 + IRGM + Poly3 (tau=50%)' row (whole cohort, long shafts,
severe-adhesion subgroup) from the current cached bundles, alongside the SynthBrachy3T row as a cross-check."""
import os, sys, json, pickle, time
import numpy as np
sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks")
sys.path.insert(0, "/tmp")
from evaluate_unified_apples_to_apples import META_PATH, get_gts, traj_gt_dist, evaluate_method_unified
from run_synthbrachy3t_full import ENRICHED_BUNDLES_CACHE_PATH, decode_hybrid_synth_nf2026_preds
from run_xiao2026_and_fewshot_benchmark import get_nf2026_synctrack_poly3_preds
from finalize_synthbrachy3t_benchmark import analyze_catheter_adhesion

meta_list = pickle.load(open(META_PATH, "rb"))
gts_all = [get_gts(m["gts_all"]) for m in meta_list]
gts_long = [get_gts(m["gts_long"]) for m in meta_list]
gts_short = [[ga for ga in gts_all[i] if not any(traj_gt_dist(ga, gl) < 2.0 for gl in gts_long[i])] for i in range(54)]
adh = analyze_catheter_adhesion(meta_list, gts_all, gts_long, adhesion_thresh_mm=3.0)["adhesion_case_indices"]
cache_d = pickle.load(open(ENRICHED_BUNDLES_CACHE_PATH, "rb"))
bundle_items, base_oof_scores = cache_d["bundle_items"], cache_d["base_oof_scores"]
ens = pickle.load(open("/tmp/cached_ens_synth_oof_54.pkl", "rb"))
rows = {
 "synth_all": decode_hybrid_synth_nf2026_preds(bundle_items, base_oof_scores, ens, min_conf=50.0, long_mode=False, w_synth=0.60, tip_scale=0.36),
 "synth_long": decode_hybrid_synth_nf2026_preds(bundle_items, base_oof_scores, ens, min_conf=42.5, long_mode=True, w_synth=0.60, tip_scale=0.36),
 "irgm_all": get_nf2026_synctrack_poly3_preds(bundle_items, base_oof_scores, meta_list, min_conf=50.0, long_mode=False),
 "irgm_long": get_nf2026_synctrack_poly3_preds(bundle_items, base_oof_scores, meta_list, min_conf=50.0, long_mode=True),
}
def summ(r):
    m = r["metrics"]
    return {"rec": r["hungarian_1to1_recall_pct"], "prec": r["precision_1to1_pct"], "f1": r["f1_1to1_pct"],
            "prec_adj": r.get("precision_1to1_obturator_adj_pct"), "f1_adj": r.get("f1_1to1_obturator_adj_pct"),
            "shaft_mean": m["mean_shaft_xy"]["mean"], "shaft_sd": m["mean_shaft_xy"]["sd"], "shaft_median": m["mean_shaft_xy"]["median"], "shaft_p95": m["mean_shaft_xy"]["p95"],
            "tip_mean": m["tip_3d"]["mean"], "tip_median": m["tip_3d"]["median"], "msd_median": m["msd_3d"]["median"], "hd95_median": m["hd95_3d"]["median"]}
out = {}
for key in ("synth", "irgm"):
    r, _ = evaluate_method_unified(rows[f"{key}_all"], gts_all, thresh=3.5, n_boot=50); out[f"{key}_whole_cohort"] = summ(r)
    r, _ = evaluate_method_unified(rows[f"{key}_long"], gts_long, aux_short_gts_per_case=gts_short, thresh=3.5, n_boot=50); out[f"{key}_long_shafts"] = summ(r)
    r, _ = evaluate_method_unified([rows[f"{key}_all"][i] for i in adh], [gts_all[i] for i in adh], thresh=3.5, n_boot=50); out[f"{key}_adhesion_subgroup"] = summ(r)
for k, v in out.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}, flush=True)
json.dump({"generated_at": time.strftime("%Y-%m-%d"), "adhesion_case_indices": adh, "rows": out},
          open("/home/guillaume/github/needlefinder/vault/sources/2026-10-22-irgm-poly3-rows-recheck.json", "w"), indent=2)
print("[done]")
