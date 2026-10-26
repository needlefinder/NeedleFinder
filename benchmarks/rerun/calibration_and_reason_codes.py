#!/usr/bin/env python3
"""Stage 2B confidence calibration and reason-coded decisions (NeedleFinder 2026 + SynthBrachy3T, tau=50%).

For every candidate track of the 54-case out-of-fold pool:
  * confidence p_m exactly as decoded for Table 2 (blend of clinical and SynthBrachy3T transformer scores);
  * ground truth: the candidate is a real catheter if its mean in-plane distance to some reference trajectory
    over the shared slices is <= 3.5 mm (same pair metric as the evaluation);
  * the decision the decoder takes and WHY (typed reason code).
Outputs reliability bins, ECE, Brier, fold-wise temperature scaling (fit on the other four folds),
and a reason-code audit (how many candidates each rule removes and how many of those were real).
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path

import os, sys, json, pickle, time
import numpy as np
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun"))
from evaluate_unified_apples_to_apples import META_PATH, get_gts, traj_gt_dist, compute_unified_pair_metrics
from run_synthbrachy3t_full import ENRICHED_BUNDLES_CACHE_PATH

W_SYNTH, MIN_CONF, THRESH = 0.60, 50.0, 3.5
OUT = resolve_path("/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2b-calibration-and-reason-codes.json")

meta_list = pickle.load(open(META_PATH, "rb"))
gts_all = [get_gts(m["gts_all"]) for m in meta_list]
cache_d = pickle.load(open(ENRICHED_BUNDLES_CACHE_PATH, "rb"))
bundle_items, base_oof_scores = cache_d["bundle_items"], cache_d["base_oof_scores"]
ens = pickle.load(open(resolve_path("/tmp/rerun/cached_ens_synth_oof_54.pkl"), "rb"))

def pts_of(c):
    p = np.asarray(c["ras"], dtype=np.float64); return p[np.argsort(p[:, 2])]

records = []  # one per candidate
for i, bg in enumerate(bundle_items):
    sc_syn = 1.0 / (1.0 + np.exp(-np.clip(ens[i][:, :3], -20.0, 20.0)))
    sc = (1.0 - W_SYNTH) * base_oof_scores[i] + W_SYNTH * sc_syn
    conf = 100.0 * (0.45 * sc[:, 0] + 0.55 * sc[:, 1])
    pool = bg["pool"]
    # ground truth per candidate: distance to the closest reference trajectory (evaluation pair metric)
    is_true = np.zeros(len(pool), dtype=bool); best_d = np.full(len(pool), np.inf); best_g = np.full(len(pool), -1)
    for ci, c in enumerate(pool):
        p = pts_of(c)
        for gi, g in enumerate(gts_all[i]):
            g_pts = g["pts"] if isinstance(g, dict) else g
            pm = compute_unified_pair_metrics(p, g_pts)
            if pm is not None and pm["mean_shaft_xy"] < best_d[ci]:
                best_d[ci] = pm["mean_shaft_xy"]; best_g[ci] = gi
        is_true[ci] = best_d[ci] <= THRESH
    # implant-region gate as in the decoder
    high_idx = np.where(conf >= 55.0)[0]
    if len(high_idx) >= 3:
        all_xy = np.concatenate([pool[idx]["ras"][:, :2] for idx in high_idx], axis=0)
        center_xy = np.median(all_xy, axis=0)
        radius = float(np.percentile(np.linalg.norm(all_xy - center_xy[None, :], axis=1), 98.0)) + 15.5
    else:
        center_xy, radius = None, 999.0
    sel = []; kept_gts = set(); case_recs = []
    for idx_c in np.argsort(-conf):
        c = pool[idx_c]; p = pts_of(c)
        reason = "kept"
        if conf[idx_c] < MIN_CONF:
            reason = "below_threshold"
        elif center_xy is not None and float(np.linalg.norm(p[:, :2].mean(axis=0) - center_xy)) > radius and conf[idx_c] < MIN_CONF + 20.0:
            reason = "outside_implant_region"
        elif c.get("mean_sym", 0.5) < 0.14 and c.get("frac_active", 0.5) < 0.40 and conf[idx_c] < MIN_CONF + 15.0:
            reason = "low_dark_core_symmetry"
        else:
            dedup = 1.60 if (conf[idx_c] >= 65.0 and c.get("mpnn_pa", 0.0) >= 0.50) else 2.02
            if any(traj_gt_dist(p, s) < dedup for s in sel):
                reason = "duplicate_of_kept_track"
        if reason == "kept":
            sel.append(p)
            if is_true[idx_c]: kept_gts.add(int(best_g[idx_c]))
        case_recs.append({"case": i, "fold": i % 5, "conf": float(conf[idx_c]), "is_true": bool(is_true[idx_c]),
                        "best_d": float(best_d[idx_c]) if np.isfinite(best_d[idx_c]) else None, "reason": reason,
                        "gt": int(best_g[idx_c]), "z_span": float(p[-1, 2] - p[0, 2])})
    # a discarded real candidate is 'redundant' if its reference catheter is covered by a kept track, else 'lost'
    for r in case_recs:
        r["real_status"] = ("kept" if r["reason"] == "kept" else ("redundant" if r["gt"] in kept_gts else "lost")) if r["is_true"] else "false"
    records.extend(case_recs)
    print(f"case {i:02d}: {len(pool)} candidates, {int(is_true.sum())} real, {len(sel)} kept", flush=True)

conf = np.array([r["conf"] for r in records]) / 100.0
y = np.array([r["is_true"] for r in records], dtype=float)
fold = np.array([r["fold"] for r in records])

def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; rows = []
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] + (1e-9 if b == bins - 1 else 0))
        if m.sum() == 0: rows.append({"lo": edges[b], "hi": edges[b + 1], "n": 0}); continue
        acc, cf = y[m].mean(), p[m].mean(); e += m.sum() / len(p) * abs(acc - cf)
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()), "mean_conf": float(cf), "frac_true": float(acc)})
    return float(e), rows

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def fit_temperature(z, y):
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.3, 5.0, 471):
        q = 1 / (1 + np.exp(-z / T)); q = np.clip(q, 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best_nll: best_nll, best_T = nll, T
    return float(best_T)

ece_raw, bins_raw = ece(conf, y)
brier_raw = float(np.mean((conf - y) ** 2))
z = logit(conf); conf_cal = np.zeros_like(conf); temps = {}
for f in range(5):
    T = fit_temperature(z[fold != f], y[fold != f]); temps[f] = T
    conf_cal[fold == f] = 1 / (1 + np.exp(-z[fold == f] / T))
ece_cal, bins_cal = ece(conf_cal, y)
brier_cal = float(np.mean((conf_cal - y) ** 2))
kept = np.array([r["reason"] == "kept" for r in records])
ece_kept, bins_kept = ece(conf[kept], y[kept])

audit = {}
for reason in ["kept", "below_threshold", "outside_implant_region", "low_dark_core_symmetry", "duplicate_of_kept_track"]:
    m = np.array([r["reason"] == reason for r in records])
    st = np.array([r["real_status"] for r in records])
    audit[reason] = {"n": int(m.sum()), "n_real": int(y[m].sum()), "n_false": int((1 - y[m]).sum()),
                     "n_real_redundant": int((m & (st == "redundant")).sum()), "n_real_lost": int((m & (st == "lost")).sum()),
                     "per_patient": float(m.sum() / 54), "real_per_patient": float(y[m].sum() / 54),
                     "mean_conf": float(conf[m].mean()) if m.sum() else None}
res = {"generated_at": time.strftime("%Y-%m-%d"), "n_candidates": len(records), "n_real": int(y.sum()),
       "decode": {"w_synth": W_SYNTH, "tau_pct": MIN_CONF, "match_thresh_mm": THRESH},
       "calibration_all_candidates": {"ece": ece_raw, "brier": brier_raw, "bins": bins_raw},
       "calibration_kept_tracks": {"ece": ece_kept, "bins": bins_kept},
       "temperature_scaling_fold_wise": {"temperatures": temps, "ece": ece_cal, "brier": brier_cal, "bins": bins_cal},
       "reason_code_audit": audit,
       "records": records}
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nECE raw={ece_raw:.4f} Brier={brier_raw:.4f} | temperature-scaled (fold-wise) ECE={ece_cal:.4f} Brier={brier_cal:.4f} T={ {k: round(v,2) for k,v in temps.items()} }")
print(f"ECE on kept tracks only={ece_kept:.4f}")
for k, v in audit.items(): print(f"{k:26s} n={v['n']:5d} real={v['n_real']:4d} (redundant {v['n_real_redundant']}, lost {v['n_real_lost']}) false={v['n_false']:4d} per_patient={v['per_patient']:.2f}")
for b in bins_raw: print(b)
print("[done]", OUT)
