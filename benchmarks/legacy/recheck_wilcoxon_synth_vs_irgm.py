#!/usr/bin/env python3
"""Recompute per-patient F1 arrays and the paired Wilcoxon test for
NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=50%) versus NeedleFinder 2026 + IRGM + Poly3 (tau=50%),
because the per-patient array stored for the latter in the unified JSON duplicates the plain NF array."""
import os, sys, json, pickle, time
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks")
sys.path.insert(0, "/tmp")
from evaluate_unified_apples_to_apples import META_PATH, get_gts, traj_gt_dist, evaluate_method_unified
from run_synthbrachy3t_full import ENRICHED_BUNDLES_CACHE_PATH, decode_hybrid_synth_nf2026_preds
from run_xiao2026_and_fewshot_benchmark import get_nf2026_synctrack_poly3_preds

meta_list = pickle.load(open(META_PATH, "rb"))
gts_all = [get_gts(m["gts_all"]) for m in meta_list]
cache_d = pickle.load(open(ENRICHED_BUNDLES_CACHE_PATH, "rb"))
bundle_items, base_oof_scores = cache_d["bundle_items"], cache_d["base_oof_scores"]
ens = pickle.load(open("/tmp/cached_ens_synth_oof_54.pkl", "rb"))
synth = decode_hybrid_synth_nf2026_preds(bundle_items, base_oof_scores, ens, min_conf=50.0, long_mode=False, w_synth=0.60, tip_scale=0.36)
irgm = get_nf2026_synctrack_poly3_preds(bundle_items, base_oof_scores, meta_list, min_conf=50.0, long_mode=False)
out = {}
for name, preds in [("synth_tau50", synth), ("irgm_poly3_tau50", irgm)]:
    r, pp = evaluate_method_unified(preds, gts_all, thresh=3.5, n_boot=50)
    out[name] = {"pooled": {k: r[k] for k in ("hungarian_1to1_recall_pct", "precision_1to1_pct", "f1_1to1_pct")},
                 "per_patient": {k: [float(x) for x in v] for k, v in pp.items() if isinstance(v, (list, np.ndarray))}}
    print(name, out[name]["pooled"], "per-patient keys:", list(out[name]["per_patient"].keys())[:8], flush=True)
tests = {}
for metric in out["synth_tau50"]["per_patient"]:
    a = np.array(out["synth_tau50"]["per_patient"][metric]); b = np.array(out["irgm_poly3_tau50"]["per_patient"].get(metric, []))
    if len(a) == len(b) == 54 and np.any(a != b):
        tests[metric] = {"mean_synth": float(a.mean()), "mean_irgm": float(b.mean()), "p_value": float(wilcoxon(a, b).pvalue)}
        print(metric, tests[metric], flush=True)
json.dump({"generated_at": time.strftime("%Y-%m-%d"), "results": out, "wilcoxon_synth_vs_irgm_poly3": tests},
          open("/home/guillaume/github/needlefinder/vault/sources/2026-10-22-wilcoxon-recheck-synth-vs-irgm.json", "w"), indent=2)
print("[done]")
