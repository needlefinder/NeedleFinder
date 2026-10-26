#!/usr/bin/env python3
"""
Export the final NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=50%) out-of-fold
predictions for all 54 cases, and compute every row of the inter-observer table (Table 3)
from one loader, so that figure overlays and table cells share a measured source.

Runs on edelweiss with the cached /tmp pickles produced by run_synthbrachy3t_full.py.
Outputs (in OUT_DIR):
  nf2026_synth_tau50_oof_predictions_54cases.pkl   -- list[54] of list[dict(ras=(N,3))]
  qualitative_predictions.npz                        -- RAS polylines for figure cases (idx 6, 17)
  2026-10-22-table3-observers-and-nf.json            -- all Table 3 rows + environment record
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path

import os, sys, json, time, pickle, platform
import numpy as np
import torch

sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun"))

from evaluate_unified_apples_to_apples import (
    META_PATH, OOF_DIR, get_gts, traj_gt_dist, evaluate_method_unified, load_human_interobserver_40cases,
)
from run_synthbrachy3t_full import ENRICHED_BUNDLES_CACHE_PATH
from finalize_synthbrachy3t_benchmark import decode_hybrid_synth_nf2026_preds

OUT_DIR = os.path.expanduser("~/github/needlefinder/vault/sources/table3_export_rerun")
os.makedirs(OUT_DIR, exist_ok=True)
FIG_CASES = {"dense": 6, "adhesion": 17}   # meta_list indices of Case 017 and Case 031 in qualitative_cases.npz


def case_number(meta):
    for k in ("case_id", "case", "name", "id", "case_name"):
        v = meta.get(k)
        if v is not None:
            import re
            m = re.search(r"(\d+)", str(v))
            if m:
                return int(m.group(1))
    return None


def main():
    t0 = time.time()
    meta_list = pickle.load(open(META_PATH, "rb"))
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]
    gts_short_list = [[ga for ga in gts_all_list[i] if not any(traj_gt_dist(ga, gl) < 2.0 for gl in gts_long_list[i])] for i in range(54)]

    cache_d = pickle.load(open(ENRICHED_BUNDLES_CACHE_PATH, "rb"))
    bundle_items = cache_d["bundle_items"]
    base_oof_scores = cache_d["base_oof_scores"]
    ens_oof_outputs = pickle.load(open(resolve_path("/tmp/rerun/cached_ens_synth_oof_54.pkl"), "rb"))

    preds_all = decode_hybrid_synth_nf2026_preds(bundle_items, base_oof_scores, ens_oof_outputs, min_conf=50.0, long_mode=False, w_synth=0.60, tip_scale=0.36)
    preds_long = decode_hybrid_synth_nf2026_preds(bundle_items, base_oof_scores, ens_oof_outputs, min_conf=50.0 * 0.85, long_mode=True, w_synth=0.60, tip_scale=0.36)

    # Sanity: reproduce the headline whole-cohort numbers
    res_all, _ = evaluate_method_unified(preds_all, gts_all_list, thresh=3.5, n_boot=50)
    print(f"[check] whole cohort tau=50: Rec={res_all['hungarian_1to1_recall_pct']:.2f} Prec={res_all['precision_1to1_pct']:.2f} F1={res_all['f1_1to1_pct']:.2f} "
          f"shaft med={res_all['metrics']['mean_shaft_xy']['median']:.3f} tip med={res_all['metrics']['tip_3d']['median']:.2f}", flush=True)

    # 1. Dump predictions
    def clean(preds):
        out = []
        for case in preds:
            cl = []
            for t in case:
                ras = np.asarray(t["ras"], dtype=np.float64)
                cl.append({"ras": ras[np.argsort(ras[:, 2])], "conf": float(t.get("conf", t.get("confidence", np.nan)))})
            out.append(cl)
        return out
    preds_all_c, preds_long_c = clean(preds_all), clean(preds_long)
    case_nums = [case_number(m) for m in meta_list]
    pickle.dump({"case_numbers": case_nums, "all_tau50": preds_all_c, "long_tau50": preds_long_c}, open(os.path.join(OUT_DIR, "nf2026_synth_tau50_oof_predictions_54cases.pkl"), "wb"))

    qual = {}
    for prefix, idx in FIG_CASES.items():
        qual[f"{prefix}_idx"] = idx
        qual[f"{prefix}_case_number"] = case_nums[idx] if case_nums[idx] is not None else -1
        qual[f"{prefix}_pred_ras_obj"] = np.array([t["ras"] for t in preds_all_c[idx]], dtype=object)
        qual[f"{prefix}_pred_conf"] = np.array([t["conf"] for t in preds_all_c[idx]], dtype=float)
        qual[f"{prefix}_n_gt_all"] = len(gts_all_list[idx])
    np.savez(os.path.join(OUT_DIR, "qualitative_predictions.npz"), **qual)
    print("[saved] predictions for", {k: (int(v) if np.ndim(v) == 0 else len(v)) for k, v in qual.items()}, flush=True)

    # 2. Table 3: one loader for everything
    overlap_nums, cy_40, lb_40 = load_human_interobserver_40cases()
    print(f"[observers] {len(overlap_nums)} overlapping cases; CY n={sum(len(x) for x in cy_40)}, LB/AMe n={sum(len(x) for x in lb_40)}", flush=True)
    idx_map = {}
    for n in overlap_nums:
        hits = [i for i, c in enumerate(case_nums) if c == n]
        if hits:
            idx_map[n] = hits[0]
    missing = [n for n in overlap_nums if n not in idx_map]
    print(f"[map] {len(idx_map)} of {len(overlap_nums)} dual scenes mapped to benchmark cases; missing: {missing}", flush=True)
    sel = [n for n in overlap_nums if n in idx_map]
    cy_sel = [cy_40[overlap_nums.index(n)] for n in sel]
    lb_sel = [lb_40[overlap_nums.index(n)] for n in sel]
    nf_sel = [preds_all_c[idx_map[n]] for n in sel]
    gt_sel = [gts_all_list[idx_map[n]] for n in sel]

    rows = {}
    def row(name, preds, refs, thresh):
        r, _ = evaluate_method_unified(preds, refs, thresh=thresh, n_boot=200)
        m = r["metrics"]
        rows[name] = {
            "thresh_mm": thresh, "n_pred": r["n_pred"], "n_ref": r["n_gt"], "n_matched": r["hungarian_1to1_matched"],
            "rate_matched_over_ref_pct": r["hungarian_1to1_recall_pct"],
            "shaft_mm": m["mean_shaft_xy"], "hd_slice_xy_mm": m.get("hd_xy"), "hd95_xy_mm": m.get("hd95_xy"),
            "tip_3d_mm": m["tip_3d"], "tip_xy_mm": m.get("tip_xy"), "msd_3d_mm": m.get("msd_3d"),
            "bootstrap_95ci": r.get("bootstrap_95ci_1000_patients"),
        }
        print(f"  {name}: n_ref={r['n_gt']} matched={r['hungarian_1to1_matched']} rate={r['hungarian_1to1_recall_pct']:.1f}% "
              f"shaft mean={m['mean_shaft_xy']['mean']:.3f} sd={m['mean_shaft_xy']['sd']:.3f} med={m['mean_shaft_xy']['median']:.3f} iqr={m['mean_shaft_xy']['iqr']:.3f} p95={m['mean_shaft_xy']['p95']:.3f} "
              f"hd_xy med={m.get('hd_xy',{}).get('median',float('nan')):.3f} tip med={m['tip_3d']['median']:.2f}", flush=True)
    print("[table3]", flush=True)
    row("CY_vs_LBAMe_ref_3.5mm", cy_sel, lb_sel, 3.5)
    row("LBAMe_vs_CY_ref_3.5mm", lb_sel, cy_sel, 3.5)
    row("CY_vs_LBAMe_ref_5.0mm", cy_sel, lb_sel, 5.0)
    row("NF2026_vs_CY_ref_3.5mm", nf_sel, cy_sel, 3.5)
    row("NF2026_vs_LBAMe_ref_3.5mm", nf_sel, lb_sel, 3.5)
    row("NF2026_vs_benchmark_GT_same_scenes_3.5mm", nf_sel, gt_sel, 3.5)

    env = {
        "host": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": np.__version__, "onnx_producer_version_of_exported_models": "pytorch 2.4.1 (read from needle_unet3d.onnx / needle_graph_transformer.onnx headers)",
    }
    out = {"generated_at": time.strftime("%Y-%m-%d"), "script": "vault/paper/benchmarks/export_nf_predictions_and_table3.py",
           "method": "NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=50%), 5-fold OOF, decoded from cached bundles",
           "dual_scene_case_numbers": sel, "unmapped_dual_scenes": missing, "whole_cohort_check": {k: res_all[k] for k in ("hungarian_1to1_recall_pct", "precision_1to1_pct", "f1_1to1_pct")},
           "rows": rows, "environment": env, "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT_DIR, "2026-10-22-table3-observers-and-nf.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[done] {OUT_DIR} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
