#!/usr/bin/env python3
"""
Unified Apples-to-Apples 5-Fold OOF Benchmark Evaluator Across All 8 Methods
for the Physics in Medicine & Biology (PMB) Paper:
1. Baseline 1: PMB 2019 Isotropic 1mm^3 3D U-Net + DBSCAN/Skeleton (PMB-2019-Iso1mm-UNet3D)
2. Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only) (Native-nnUNet3D-clDice)
3. Stage 0/1A: Pure Physics 3D CV + HybridNeedleNet
4. Stage 1B: Multi-Head NeedleUNet3D + Tangent Tube Tracking
5. Stage 1B: Hybrid AxialTransUNet3D + Tangent Tube Tracking
6. Stage 2A: Pure NeedleSliceMPNN (4-Layer Triplet GraphNet)
7. Unified NeedleFinder 2026 (NeedleUNet3D + NeedleSliceMPNN + DenseGraphTransformer) at tau in {25%, 35%, 45%, 50%, 60%}
8. Human Inter-Observer Agreement (CY vs. LB/AMe across 40 overlapping cases)
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
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
import torch.nn.functional as F

for mod in ["vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util", "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"]:
    sys.modules[mod] = types.ModuleType(mod)

sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/tmp")

from NeedleFinderLib.NeuralNeedleNet import extract_nn_labelmap_tubes
from evaluate_nn_pipeline import filter_and_augment_with_nn
from prepare_and_inspect_dataset import read_vtk_centerline_local, read_nrrd_local
from train_export_graphnet_prod import (
    build_slice_graph_fast,
    build_bundle_graph_item,
    DenseGraphTransformer,
    get_gts,
    traj_gt_dist,
)
from train_graphnet3d import NeedleSliceMPNN, decode_mpnn_trajectories

CACHE_DIR = "/tmp/needle_unet3d_cache"
OOF_DIR = "/home/guillaume/github/needlefinder/nn_models"
META_PATH = "/tmp/cases_meta_54.pkl"
BASE_TRAJS_PATH = "/tmp/base_trajs_54cases.pkl"
PROVENANCE_PATH = "/tmp/mbp2_pmb2019_provenance.json"


# ==============================================================================
# 1. UNIFIED GEOMETRIC METRIC SUITE FOR A MATCHED (PRED, GT) CATHETER PAIR
# ==============================================================================

def resample_polyline_3d(pts, n_samples=50):
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts[np.argsort(pts[:, 2])]
    zs = np.linspace(float(pts[0, 2]), float(pts[-1, 2]), n_samples)
    xs = np.interp(zs, pts[:, 2], pts[:, 0])
    ys = np.interp(zs, pts[:, 2], pts[:, 1])
    return np.column_stack([xs, ys, zs])


def compute_unified_pair_metrics(det_pts, gt_pts):
    det_ras = np.asarray(det_pts, dtype=np.float64)
    gt_ras = np.asarray(gt_pts, dtype=np.float64)
    det_ras = det_ras[np.argsort(det_ras[:, 2])]
    gt_ras = gt_ras[np.argsort(gt_ras[:, 2])]
    if len(det_ras) < 2 or len(gt_ras) < 2:
        return None

    # 1. Slice-wise In-Plane Shaft Error over overlapping Z interval
    z_min = max(float(det_ras[0, 2]), float(gt_ras[0, 2]))
    z_max = min(float(det_ras[-1, 2]), float(gt_ras[-1, 2]))
    if z_max <= z_min + 1.0:
        return None

    zs = np.linspace(z_min, z_max, 40)
    dx = np.interp(zs, det_ras[:, 2], det_ras[:, 0]) - np.interp(zs, gt_ras[:, 2], gt_ras[:, 0])
    dy = np.interp(zs, det_ras[:, 2], det_ras[:, 1]) - np.interp(zs, gt_ras[:, 2], gt_ras[:, 1])
    xy_dists = np.hypot(dx, dy)
    mean_shaft_xy = float(np.mean(xy_dists))
    if mean_shaft_xy > 8.0:
        return None
    hd_xy = float(np.max(xy_dists))
    hd95_xy = float(np.percentile(xy_dists, 95))

    # 2. PMB 2019 3D Symmetric Mean Surface/Curve Distance (D_avg / msd_3d) & 3D Hausdorff (HD, HD95)
    P_3d = resample_polyline_3d(det_ras, n_samples=50)
    G_3d = resample_polyline_3d(gt_ras, n_samples=50)
    diff = P_3d[:, None, :] - G_3d[None, :, :]
    dist_mat_3d = np.linalg.norm(diff, axis=-1)  # [50, 50]
    d_p2g = np.min(dist_mat_3d, axis=1)
    d_g2p = np.min(dist_mat_3d, axis=0)
    msd_3d = float(0.5 * (np.mean(d_p2g) + np.mean(d_g2p)))
    hd_3d = float(max(np.max(d_p2g), np.max(d_g2p)))
    hd95_3d = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))

    # 3. 3D Distal Tip Error (tip_3d, tip_z, tip_xy)
    det_tip = det_ras[-1]
    gt_tip = gt_ras[-1]
    tip_3d = float(np.linalg.norm(det_tip - gt_tip))
    tip_z = float(abs(det_tip[2] - gt_tip[2]))
    tip_xy = float(np.hypot(det_tip[0] - gt_tip[0], det_tip[1] - gt_tip[1]))

    return {
        "mean_shaft_xy": mean_shaft_xy,
        "msd_3d": msd_3d,
        "hd_3d": hd_3d,
        "hd95_3d": hd95_3d,
        "hd_xy": hd_xy,
        "hd95_xy": hd95_xy,
        "tip_3d": tip_3d,
        "tip_z": tip_z,
        "tip_xy": tip_xy,
    }


def summarize_distribution(vals):
    if not vals:
        return {"mean": 0.0, "sd": 0.0, "median": 0.0, "iqr": 0.0, "q25": 0.0, "q75": 0.0, "p95": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    q25, med, q75, p95 = np.percentile(arr, [25, 50, 75, 95])
    return {
        "mean": round(float(np.mean(arr)), 4),
        "sd": round(float(np.std(arr)), 4),
        "median": round(float(med), 4),
        "iqr": round(float(q75 - q25), 4),
        "q25": round(float(q25), 4),
        "q75": round(float(q75), 4),
        "p95": round(float(p95), 4),
    }


def evaluate_method_unified(preds_per_case, gts_per_case, aux_short_gts_per_case=None, thresh=3.5, n_boot=1000):
    """
    Evaluates a method across all cases (54 cases or 40 dual-observer cases) and computes:
    - Pooled detection metrics (Closest Recall, 1-to-1 Hungarian Recall, 1-to-1 Precision, 1-to-1 F1)
    - Complete distribution statistics for all geometric error metrics
    - Patient-level 95% bootstrap confidence intervals (1,000 resamples)
    - Per-patient arrays for paired Wilcoxon signed-rank testing
    """
    n_cases = len(gts_per_case)
    case_records = []
    all_matched_pairs = []

    for i in range(n_cases):
        gts = gts_per_case[i]
        preds = preds_per_case[i]
        n_gt = len(gts)
        n_pred = len(preds)

        cost_m = np.full((n_gt, max(1, n_pred)), 1e6, dtype=np.float64)
        pair_metrics = {}
        for gi, g in enumerate(gts):
            g_pts = g["pts"] if isinstance(g, dict) else g
            for pi, p in enumerate(preds):
                p_pts = p["ras"] if isinstance(p, dict) and "ras" in p else (p["points"] if isinstance(p, dict) else p)
                pm = compute_unified_pair_metrics(p_pts, g_pts)
                if pm is not None:
                    cost_m[gi, pi] = pm["mean_shaft_xy"]
                    pair_metrics[(gi, pi)] = pm

        closest_m = int(np.sum(np.min(cost_m, axis=1) <= thresh)) if (n_gt > 0 and n_pred > 0) else 0
        # Thresholded assignment: pairs above the matching threshold are forbidden before solving, so the
        # assignment maximises the number of valid (<= thresh) pairs and only then minimises their cost.
        cost_m = np.where(cost_m > thresh, 1e6, cost_m)
        hung_pairs = []
        matched_pred_indices = set()
        if n_gt > 0 and n_pred > 0:
            r_ind, c_ind = linear_sum_assignment(cost_m)
            for gi, pi in zip(r_ind, c_ind):
                if cost_m[gi, pi] <= thresh:
                    hung_pairs.append(pair_metrics[(gi, pi)])
                    matched_pred_indices.add(pi)

        # Check how many unmatched predictions match an auxiliary short obturator needle (for long-only cohort reporting)
        aux_short_matched = 0
        if aux_short_gts_per_case is not None and len(aux_short_gts_per_case[i]) > 0 and n_pred > 0:
            short_gts = aux_short_gts_per_case[i]
            unmatched_preds = [preds[pi] for pi in range(n_pred) if pi not in matched_pred_indices]
            if unmatched_preds:
                c_s = np.full((len(short_gts), len(unmatched_preds)), 1e6, dtype=np.float64)
                for sgi, sg in enumerate(short_gts):
                    sg_pts = sg["pts"] if isinstance(sg, dict) else sg
                    for upi, up in enumerate(unmatched_preds):
                        up_pts = up["ras"] if isinstance(up, dict) and "ras" in up else (up["points"] if isinstance(up, dict) else up)
                        pm_s = compute_unified_pair_metrics(up_pts, sg_pts)
                        if pm_s is not None:
                            c_s[sgi, upi] = pm_s["mean_shaft_xy"]
                c_s = np.where(c_s > thresh, 1e6, c_s)
                rs, cs = linear_sum_assignment(c_s)
                aux_short_matched = sum(1 for r, c in zip(rs, cs) if c_s[r, c] <= thresh)

        all_matched_pairs.extend(hung_pairs)
        case_records.append({
            "n_gt": n_gt,
            "n_pred": n_pred,
            "closest_matched": closest_m,
            "hung_matched": len(hung_pairs),
            "aux_short_matched": aux_short_matched,
            "pairs": hung_pairs,
        })

    tot_gt = sum(r["n_gt"] for r in case_records)
    tot_pred = sum(r["n_pred"] for r in case_records)
    tot_clos = sum(r["closest_matched"] for r in case_records)
    tot_hung = sum(r["hung_matched"] for r in case_records)
    tot_aux_short = sum(r["aux_short_matched"] for r in case_records)

    clos_rec = 100.0 * tot_clos / max(1, tot_gt)
    hung_rec = 100.0 * tot_hung / max(1, tot_gt)
    prec_raw = 100.0 * tot_hung / max(1, tot_pred)
    f1_raw = 2.0 * hung_rec * prec_raw / max(1e-6, hung_rec + prec_raw)

    eff_pred = max(tot_hung, tot_pred - tot_aux_short)
    prec_adj = 100.0 * tot_hung / max(1, eff_pred)
    f1_adj = 2.0 * hung_rec * prec_adj / max(1e-6, hung_rec + prec_adj)

    metric_keys = ["mean_shaft_xy", "msd_3d", "hd_3d", "hd95_3d", "hd_xy", "hd95_xy", "tip_3d", "tip_z", "tip_xy"]
    distributions = {
        k: summarize_distribution([p[k] for p in all_matched_pairs])
        for k in metric_keys
    }

    # Patient-level bootstrap 95% confidence intervals (1,000 resamples of the N patients)
    rng = np.random.default_rng(42)
    boot_clos_rec, boot_hung_rec, boot_prec_raw, boot_f1_raw, boot_prec_adj, boot_f1_adj = [], [], [], [], [], []
    boot_metric_means = {k: [] for k in metric_keys}
    boot_metric_medians = {k: [] for k in metric_keys}

    for _ in range(n_boot):
        idxs = rng.integers(0, n_cases, size=n_cases)
        b_gt = sum(case_records[ii]["n_gt"] for ii in idxs)
        b_pred = sum(case_records[ii]["n_pred"] for ii in idxs)
        b_clos = sum(case_records[ii]["closest_matched"] for ii in idxs)
        b_hung = sum(case_records[ii]["hung_matched"] for ii in idxs)
        b_aux = sum(case_records[ii]["aux_short_matched"] for ii in idxs)

        cr = 100.0 * b_clos / max(1, b_gt)
        hr = 100.0 * b_hung / max(1, b_gt)
        pr = 100.0 * b_hung / max(1, b_pred)
        f1r = 2.0 * hr * pr / max(1e-6, hr + pr)
        pa = 100.0 * b_hung / max(1, max(b_hung, b_pred - b_aux))
        f1a = 2.0 * hr * pa / max(1e-6, hr + pa)

        boot_clos_rec.append(cr)
        boot_hung_rec.append(hr)
        boot_prec_raw.append(pr)
        boot_f1_raw.append(f1r)
        boot_prec_adj.append(pa)
        boot_f1_adj.append(f1a)

        b_pairs = [p for ii in idxs for p in case_records[ii]["pairs"]]
        if b_pairs:
            for k in metric_keys:
                vals = [p[k] for p in b_pairs]
                boot_metric_means[k].append(float(np.mean(vals)))
                boot_metric_medians[k].append(float(np.median(vals)))

    def ci95(arr):
        if not arr:
            return [0.0, 0.0]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    bootstrap_ci95 = {
        "closest_recall_pct": ci95(boot_clos_rec),
        "hungarian_1to1_recall_pct": ci95(boot_hung_rec),
        "precision_1to1_pct": ci95(boot_prec_raw),
        "f1_1to1_pct": ci95(boot_f1_raw),
        "precision_1to1_obturator_adj_pct": ci95(boot_prec_adj),
        "f1_1to1_obturator_adj_pct": ci95(boot_f1_adj),
        "metrics_mean": {k: ci95(boot_metric_means[k]) for k in metric_keys},
        "metrics_median": {k: ci95(boot_metric_medians[k]) for k in metric_keys},
    }

    # Per-patient summary arrays for paired Wilcoxon signed-rank tests
    per_patient = {
        "hung_recall_pct": [100.0 * r["hung_matched"] / max(1, r["n_gt"]) for r in case_records],
        "closest_recall_pct": [100.0 * r["closest_matched"] / max(1, r["n_gt"]) for r in case_records],
        "f1_1to1_pct": [
            (200.0 * r["hung_matched"] / max(1, r["n_gt"] + r["n_pred"])) for r in case_records
        ],
        "f1_1to1_adj_pct": [
            (200.0 * r["hung_matched"] / max(1, r["n_gt"] + max(r["hung_matched"], r["n_pred"] - r["aux_short_matched"])))
            for r in case_records
        ],
        "mean_shaft_xy": [
            float(np.mean([p["mean_shaft_xy"] for p in r["pairs"]])) if r["pairs"] else 3.5
            for r in case_records
        ],
        "msd_3d": [
            float(np.mean([p["msd_3d"] for p in r["pairs"]])) if r["pairs"] else 10.0
            for r in case_records
        ],
        "hd95_3d": [
            float(np.mean([p["hd95_3d"] for p in r["pairs"]])) if r["pairs"] else 20.0
            for r in case_records
        ],
        "tip_3d": [
            float(np.mean([p["tip_3d"] for p in r["pairs"]])) if r["pairs"] else 20.0
            for r in case_records
        ],
    }

    summary = {
        "n_cases": n_cases,
        "n_gt": tot_gt,
        "n_pred": tot_pred,
        "closest_matched": tot_clos,
        "closest_recall_pct": round(clos_rec, 2),
        "hungarian_1to1_matched": tot_hung,
        "hungarian_1to1_recall_pct": round(hung_rec, 2),
        "precision_1to1_pct": round(prec_raw, 2),
        "f1_1to1_pct": round(f1_raw, 2),
        "aux_short_obturator_matched": tot_aux_short,
        "precision_1to1_obturator_adj_pct": round(prec_adj, 2),
        "f1_1to1_obturator_adj_pct": round(f1_adj, 2),
        "metrics": distributions,
        "bootstrap_95ci_1000_patients": bootstrap_ci95,
    }
    return summary, per_patient


def compute_wilcoxon_comparison(ref_per_patient, base_per_patient):
    out = {}
    for k in ["hung_recall_pct", "closest_recall_pct", "f1_1to1_pct", "f1_1to1_adj_pct", "mean_shaft_xy", "msd_3d", "hd95_3d", "tip_3d"]:
        a = np.asarray(ref_per_patient[k], dtype=np.float64)
        b = np.asarray(base_per_patient[k], dtype=np.float64)
        diff = a - b
        if np.all(np.abs(diff) < 1e-9):
            p_val = 1.0
            stat = 0.0
        else:
            res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            stat = float(res.statistic)
            p_val = float(res.pvalue)
        out[k] = {
            "ref_patient_mean": round(float(np.mean(a)), 4),
            "base_patient_mean": round(float(np.mean(b)), 4),
            "mean_paired_diff": round(float(np.mean(diff)), 4),
            "median_paired_diff": round(float(np.median(diff)), 4),
            "wilcoxon_statistic": round(stat, 2),
            "p_value": p_val,
            "p_value_formatted": f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.5f}",
        }
    return out


# ==============================================================================
# 2. LOAD HUMAN INTER-OBSERVER DATA (CY vs. LB/AMe ACROSS 40 OVERLAPPING CASES)
# ==============================================================================

def load_human_interobserver_40cases(gyn_base="/home/guillaume/Dropbox/GYN Cases"):
    with open(os.path.join(gyn_base, "760CYScenes.txt")) as f:
        cy_lines = [l.strip() for l in f if l.strip()]
    with open(os.path.join(gyn_base, "543LB_AMe_scenes.txt")) as f:
        lb_lines = [l.strip() for l in f if l.strip()]

    def parse_mrml_needles(rel_path):
        p = os.path.join(gyn_base, rel_path.lstrip("./"))
        for a, b in [("NRRD-MRI", "NRRD"), ("Maunal_CY", "Manual_CY"), ("Manual CY", "Manual_CY"), ("Auto-Eval-LB", "Manual_LB"), ("Manual Alireza", "Manual_AMe")]:
            if not os.path.exists(p):
                p2 = p.replace(a, b)
                if os.path.exists(p2):
                    p = p2
        d = os.path.dirname(p)
        case_dir = os.path.dirname(os.path.dirname(d))
        case_num = int(re.search(r"(\d+)", os.path.basename(case_dir)).group(1))
        tree = ET.parse(p)
        root = tree.getroot()
        st_map = {ms.attrib["id"]: urllib.parse.unquote(ms.attrib.get("fileName", "")) for ms in root.findall(".//ModelStorage")}
        vtks = {}
        for m in root.findall(".//Model"):
            name = m.attrib.get("name", "")
            attrs = m.attrib.get("attributes", "")
            if ("type:Validation" in attrs or "manual-seg" in name or "needle" in name.lower()) and not name.endswith("Pelvis"):
                sref = m.attrib.get("storageNodeRef", "")
                fn = st_map.get(sref, name + ".vtk")
                fp = os.path.join(d, fn)
                if os.path.exists(fp) and os.path.basename(fp).startswith("manual-seg_") and len(os.path.basename(fp).split("_")) == 2:
                    vtks[name] = fp
        if not vtks:
            for fp in sorted(glob.glob(os.path.join(d, "manual-seg_*.vtk"))):
                bn = os.path.basename(fp)
                if len(bn.split("_")) == 2 and not bn.endswith(".incorrect"):
                    vtks[bn[:-4]] = fp
        needles = []
        for k in sorted(vtks.keys()):
            cl = read_vtk_centerline_local(vtks[k], z_lim_ras=None)
            if cl is not None:
                needles.append(cl)
        return case_num, needles

    cy_map = {}
    for l in cy_lines:
        cnum, nds = parse_mrml_needles(l)
        cy_map[cnum] = nds

    lb_map = {}
    for l in lb_lines:
        cnum, nds = parse_mrml_needles(l)
        lb_map[cnum] = nds

    overlap_cases = sorted(set(cy_map.keys()) & set(lb_map.keys()))
    cy_list = [cy_map[c] for c in overlap_cases]
    lb_list = [lb_map[c] for c in overlap_cases]
    return overlap_cases, cy_list, lb_list


# ==============================================================================
# 3. MAIN UNIFIED EVALUATION PIPELINE
# ==============================================================================

def main():
    t0 = time.perf_counter()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    meta_list = pickle.load(open(META_PATH, "rb"))
    base_trajs_data = pickle.load(open(BASE_TRAJS_PATH, "rb"))
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]

    # Auxiliary short obturator GTs per case (in gts_all but not in gts_long)
    gts_short_list = []
    for i in range(54):
        short_i = []
        for ga in gts_all_list[i]:
            if not any(traj_gt_dist(ga, gl) < 2.0 for gl in gts_long_list[i]):
                short_i.append(ga)
        gts_short_list.append(short_i)

    print(f"Loaded 54 cases: {sum(len(x) for x in gts_long_list)} CY Long Needles, "
          f"{sum(len(x) for x in gts_short_list)} Short Obturator Needles, "
          f"{sum(len(x) for x in gts_all_list)} Total Needles.")

    # 1. Load Baseline 1 (PMB-2019-Iso1mm-UNet3D) & Baseline 2 (Native-nnUNet3D-clDice)
    b1_data = pickle.load(open(f"{OOF_DIR}/oof_trajs_pmb2019_iso1mm.pkl", "rb"))
    b2_data = pickle.load(open(f"{OOF_DIR}/oof_trajs_native_cldice.pkl", "rb"))

    # 2. Prepare Stage 0/1A (Pure Physics 3D CV + HybridNeedleNet)
    stage0_preds = []
    for i in range(54):
        b_item = base_trajs_data[i]
        b_trajs = b_item["trajs"] if isinstance(b_item, dict) and "trajs" in b_item else b_item
        clean = []
        for t in b_trajs:
            tc = dict(t)
            tc["ras"] = np.asarray(tc["ras"] if "ras" in tc else tc["points"], dtype=np.float64)
            tc["points"] = tc["ras"]
            clean.append(tc)
        stage0_preds.append(clean)

    # 3. Prepare Stage 1B: Multi-Head NeedleUNet3D + Tangent Tube Tracking
    #    and Stage 1B: Hybrid AxialTransUNet3D + Tangent Tube Tracking
    unet_tubes_long, unet_tubes_all = [], []
    transunet_tubes_long, transunet_tubes_all = [], []
    graphs = []

    for i in range(54):
        d_pt = torch.load(f"{CACHE_DIR}/case_{i:02d}.pt", weights_only=False)
        X = d_pt["X"].numpy()
        k0, i0, j0 = int(d_pt["k0"]), int(d_pt["i0"]), int(d_pt["j0"])
        sp_i, sp_j, sp_k = [float(x) for x in meta_list[i]["sp_ras"]]
        ijk2ras = np.asarray(meta_list[i]["ijk_to_ras"], dtype=np.float64)
        ras2ijk = np.linalg.inv(ijk2ras)
        resp_approx = np.expm1(X[1] * 4.0)

        oof_u = np.load(f"{OOF_DIR}/oof_case_{i:02d}.npz")
        nn_u = {k: oof_u[k].astype(np.float32) for k in ["p_mask", "p_peak", "vx", "vy", "p_obt"]}
        # Fused Stage 0 + Stage 1B NeedleUNet3D Tangent Tube Tracking
        u_long_ex = extract_nn_labelmap_tubes(nn_u, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, stage0_preds[i], include_obturator=False)
        u_all_ex = extract_nn_labelmap_tubes(nn_u, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, stage0_preds[i], include_obturator=True)
        unet_tubes_long.append(stage0_preds[i] + u_long_ex)
        unet_tubes_all.append(stage0_preds[i] + u_all_ex)

        oof_t = np.load(f"{OOF_DIR}/oof_transunet_case_{i:02d}.npz")
        nn_t = {k: oof_t[k].astype(np.float32) for k in ["p_mask", "p_peak", "vx", "vy", "p_obt"]}
        t_long_ex = extract_nn_labelmap_tubes(nn_t, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, stage0_preds[i], include_obturator=False)
        t_all_ex = extract_nn_labelmap_tubes(nn_t, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, stage0_preds[i], include_obturator=True)
        transunet_tubes_long.append(stage0_preds[i] + t_long_ex)
        transunet_tubes_all.append(stage0_preds[i] + t_all_ex)

        g = build_slice_graph_fast(
            X, nn_u["p_mask"], nn_u["p_peak"], nn_u["vx"], nn_u["vy"], nn_u["p_obt"],
            k0, i0, j0, meta_list[i]["sp_ras"], ijk2ras,
            gts_all=gts_all_list[i], gts_long=gts_long_list[i],
        )
        graphs.append(g)

    # 4. Run 5-Fold CV (`i % 5 == fold`) for Stage 2A: Pure NeedleSliceMPNN (4-Layer Triplet GraphNet)
    print("Running 5-Fold OOF evaluation for Stage 2A (NeedleSliceMPNN) and Stage 2B (DenseGraphTransformer)...", flush=True)
    oof_mpnn_trajs = [None] * 54
    oof_mpnn_trajs_long = [None] * 54
    for fold in range(5):
        train_gs = [g for i, g in enumerate(graphs) if i % 5 != fold]
        model = NeedleSliceMPNN(in_node_dim=16, in_edge_dim=12, in_sp_dim=5, hidden_dim=64, num_layers=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-4)
        for ep in range(40):
            model.train()
            for bi in np.random.permutation(len(train_gs)):
                g = train_gs[bi]
                opt.zero_grad(set_to_none=True)
                no, el = model(g, device)
                y_na, y_nl, y_off, y_e = g["nodes_y_all"].to(device), g["nodes_y_long"].to(device), g["nodes_offset"].to(device), g["edge_y"].to(device)
                pw_n = torch.tensor([3.0], device=device)
                fg_n = (y_na > 0.5).float().unsqueeze(1)
                loss = F.binary_cross_entropy_with_logits(no[:, 0], y_na, pos_weight=pw_n) + \
                       0.5 * F.binary_cross_entropy_with_logits(no[:, 1], y_nl, pos_weight=pw_n) + \
                       0.5 * (F.smooth_l1_loss(no[:, 2:4], y_off, reduction="none") * fg_n).sum() / fg_n.sum().clamp(min=1.0)
                if el.numel() > 0:
                    loss = loss + 2.0 * F.binary_cross_entropy_with_logits(el, y_e, pos_weight=torch.tensor([6.0], device=device))
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            for i, g in enumerate(graphs):
                if i % 5 == fold:
                    no, el = model(g, device)
                    mt_all = decode_mpnn_trajectories(g, no, el)
                    oof_mpnn_trajs[i] = mt_all
                    oof_mpnn_trajs_long[i] = [t for t in mt_all if t.get("mean_pl", 0.5) >= 0.38]

    # 5. Build Unified Candidate Pool & Run 5-Fold CV (`i % 5 == fold`) for DenseGraphTransformer
    bundle_items = []
    for i, g in enumerate(graphs):
        oof_npz = np.load(f"{OOF_DIR}/oof_case_{i:02d}.npz")
        pool = []
        for t in stage0_preds[i]:
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
        for nt in unet_tubes_all[i][len(stage0_preds[i]):]:
            ras = np.asarray(nt["points"], dtype=np.float64)
            ras = ras[np.argsort(ras[:, 2])]
            pool.append({
                "ras": ras,
                "base_xy": np.asarray(nt.get("base_xy", ras[0, :2]), dtype=np.float64),
                "slope": np.asarray(nt.get("slope", (ras[-1, :2] - ras[0, :2]) / max(1.0, ras[-1, 2] - ras[0, 2])), dtype=np.float64),
                "z_span": float(nt.get("z_span_tracked", ras[-1, 2] - ras[0, 2])),
                "rmse": float(nt.get("rmse", 0.35)),
                "conf_heur": float(nt.get("confidence", 65.0)) / 100.0,
                "from_mpnn": 0.5, "mpnn_pa": float(nt.get("confidence", 65.0)) / 100.0, "mpnn_pl": 0.5, "mpnn_pe": 0.6,
            })
        for mt in oof_mpnn_trajs[i]:
            ras = mt["ras"]
            matched_idx = -1
            best_d = 2.05
            for idx_p, p in enumerate(pool):
                d = traj_gt_dist(ras, p["ras"])
                if d < best_d:
                    best_d = d
                    matched_idx = idx_p
            if matched_idx >= 0:
                pool[matched_idx]["from_mpnn"] = 1.0
                pool[matched_idx]["mpnn_pa"] = mt["mean_pa"]
                pool[matched_idx]["mpnn_pl"] = mt["mean_pl"]
                pool[matched_idx]["mpnn_pe"] = mt["mean_pe"]
            else:
                pool.append({
                    "ras": ras,
                    "base_xy": mt["base_xy"],
                    "slope": mt["slope"],
                    "z_span": mt["z_span"],
                    "rmse": mt["rmse"],
                    "conf_heur": mt["mean_pa"] * mt["mean_pe"],
                    "from_mpnn": 1.0,
                    "mpnn_pa": mt["mean_pa"],
                    "mpnn_pl": mt["mean_pl"],
                    "mpnn_pe": mt["mean_pe"],
                })
        bundle_items.append(build_bundle_graph_item(pool, g, oof_npz, gts_all=gts_all_list[i], gts_long=gts_long_list[i]))

    oof_scores = [None] * 54
    for fold in range(5):
        tr_b = [b for i, b in enumerate(bundle_items) if i % 5 != fold]
        gt_model = DenseGraphTransformer(in_dim=18, edge_dim=6, d_model=64).to(device)
        opt = torch.optim.AdamW(gt_model.parameters(), lr=1.8e-3, weight_decay=1e-4)
        for ep in range(75):
            gt_model.train()
            for bi in np.random.permutation(len(tr_b)):
                bg = tr_b[bi]
                opt.zero_grad(set_to_none=True)
                out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                l_rec = F.binary_cross_entropy_with_logits(out[:, 0], bg["y_rec"].to(device), pos_weight=torch.tensor([3.0], device=device))
                l_1to1 = F.binary_cross_entropy_with_logits(out[:, 1], bg["y_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device))
                l_long = F.binary_cross_entropy_with_logits(out[:, 2], bg["y_long_1to1"].to(device), pos_weight=torch.tensor([2.0], device=device))
                loss = l_rec + 1.5 * l_1to1 + l_long
                loss.backward()
                opt.step()
        gt_model.eval()
        with torch.no_grad():
            for i, bg in enumerate(bundle_items):
                if i % 5 == fold:
                    out = gt_model(bg["x"].to(device), bg["adj_feat"].to(device), bg["adj_mask"].to(device))
                    oof_scores[i] = torch.sigmoid(out).cpu().numpy()

    def get_nf2026_preds(min_conf, long_mode=False):
        preds_out = []
        for i, bg in enumerate(bundle_items):
            sc = oof_scores[i]
            if long_mode:
                # Blend recall head (sc[:, 0]), 1-to-1 head (sc[:, 1]), and long-specific 1-to-1 head (sc[:, 2])
                conf_arr = 100.0 * (0.30 * sc[:, 0] + 0.25 * sc[:, 1] + 0.45 * sc[:, 2])
            else:
                conf_arr = 100.0 * (0.45 * sc[:, 0] + 0.55 * sc[:, 1])
            sel = []
            for idx_c in np.argsort(-conf_arr):
                if conf_arr[idx_c] < min_conf:
                    continue
                cand = bg["pool"][idx_c]
                if any(traj_gt_dist(cand["ras"], s["ras"]) < 1.95 for s in sel):
                    continue
                sel.append(cand)
            preds_out.append(sel)
        return preds_out

    # ==========================================================================
    # EVALUATE ALL METHODS ON BOTH COHORTS (1,063 ALL & 759 CY LONG)
    # ==========================================================================
    methods_spec = [
        ("Baseline 1: PMB 2019 Isotropic 1mm³ 3D U-Net + DBSCAN/Skeleton", b1_data["all"], b1_data["long"]),
        ("Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)", b2_data["all"], b2_data["long"]),
        ("Stage 0/1A: Pure Physics 3D CV + HybridNeedleNet", stage0_preds, stage0_preds),
        ("Stage 1B: Multi-Head NeedleUNet3D + Tangent Tube Tracking", unet_tubes_all, unet_tubes_long),
        ("Stage 1B: Hybrid AxialTransUNet3D + Tangent Tube Tracking", transunet_tubes_all, transunet_tubes_long),
        ("Stage 2A: Pure NeedleSliceMPNN (4-Layer Triplet GraphNet)", oof_mpnn_trajs, oof_mpnn_trajs_long),
    ]
    for tau in [25, 35, 45, 50, 60]:
        methods_spec.append((
            f"Unified NeedleFinder 2026 (tau={tau}%)",
            get_nf2026_preds(float(tau), long_mode=False),
            get_nf2026_preds(float(tau), long_mode=True),
        ))

    results_1063 = {}
    per_patient_1063 = {}
    results_759 = {}
    per_patient_759 = {}

    for name, p_all, p_long in methods_spec:
        print(f"Evaluating: {name} ...", flush=True)
        res_a, pp_a = evaluate_method_unified(p_all, gts_all_list, aux_short_gts_per_case=None, thresh=3.5, n_boot=1000)
        results_1063[name] = res_a
        per_patient_1063[name] = pp_a

        res_l, pp_l = evaluate_method_unified(p_long, gts_long_list, aux_short_gts_per_case=gts_short_list, thresh=3.5, n_boot=1000)
        results_759[name] = res_l
        per_patient_759[name] = pp_l

    # Evaluate Human Inter-Observer Agreement (CY vs. LB/AMe across 40 overlapping cases)
    overlap_case_nums, cy_40_list, lb_40_list = load_human_interobserver_40cases()
    human_vs_lb_summary, _ = evaluate_method_unified(cy_40_list, lb_40_list, thresh=3.5, n_boot=1000)
    human_vs_cy_summary, _ = evaluate_method_unified(lb_40_list, cy_40_list, thresh=3.5, n_boot=1000)

    # Compute Paired Wilcoxon Signed-Rank p-values (N=54 patients)
    wilcoxon_1063 = {
        "NeedleFinder2026_tau35_vs_PMB2019_Iso1mm": compute_wilcoxon_comparison(
            per_patient_1063["Unified NeedleFinder 2026 (tau=35%)"],
            per_patient_1063["Baseline 1: PMB 2019 Isotropic 1mm³ 3D U-Net + DBSCAN/Skeleton"],
        ),
        "NeedleFinder2026_tau35_vs_Native_clDice": compute_wilcoxon_comparison(
            per_patient_1063["Unified NeedleFinder 2026 (tau=35%)"],
            per_patient_1063["Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)"],
        ),
        "NeedleFinder2026_tau50_vs_PMB2019_Iso1mm": compute_wilcoxon_comparison(
            per_patient_1063["Unified NeedleFinder 2026 (tau=50%)"],
            per_patient_1063["Baseline 1: PMB 2019 Isotropic 1mm³ 3D U-Net + DBSCAN/Skeleton"],
        ),
        "NeedleFinder2026_tau50_vs_Native_clDice": compute_wilcoxon_comparison(
            per_patient_1063["Unified NeedleFinder 2026 (tau=50%)"],
            per_patient_1063["Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)"],
        ),
        "Stage1B_NeedleUNet3D_vs_Native_clDice": compute_wilcoxon_comparison(
            per_patient_1063["Stage 1B: Multi-Head NeedleUNet3D + Tangent Tube Tracking"],
            per_patient_1063["Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)"],
        ),
    }

    wilcoxon_759 = {
        "NeedleFinder2026_tau35_vs_PMB2019_Iso1mm": compute_wilcoxon_comparison(
            per_patient_759["Unified NeedleFinder 2026 (tau=35%)"],
            per_patient_759["Baseline 1: PMB 2019 Isotropic 1mm³ 3D U-Net + DBSCAN/Skeleton"],
        ),
        "NeedleFinder2026_tau35_vs_Native_clDice": compute_wilcoxon_comparison(
            per_patient_759["Unified NeedleFinder 2026 (tau=35%)"],
            per_patient_759["Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)"],
        ),
        "NeedleFinder2026_tau50_vs_PMB2019_Iso1mm": compute_wilcoxon_comparison(
            per_patient_759["Unified NeedleFinder 2026 (tau=50%)"],
            per_patient_759["Baseline 1: PMB 2019 Isotropic 1mm³ 3D U-Net + DBSCAN/Skeleton"],
        ),
        "NeedleFinder2026_tau50_vs_Native_clDice": compute_wilcoxon_comparison(
            per_patient_759["Unified NeedleFinder 2026 (tau=50%)"],
            per_patient_759["Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)"],
        ),
    }

    prov = json.load(open(PROVENANCE_PATH)) if os.path.exists(PROVENANCE_PATH) else {}
    prov_summary = {k: v for k, v in prov.items() if k != "per_case"}

    final_json = {
        "generated_at": "2026-10-22",
        "experiment": "PMB 2026 Experiment 1: Unified Apples-to-Apples 5-Fold Cross-Validation Benchmark (54 Clinical GYN Cases)",
        "hardware": {
            "compute_host": "edelweiss (guillaume@100.122.46.67)",
            "gpus": ["NVIDIA GeForce RTX 4090 (24 GB)", "NVIDIA GeForce RTX 4070 Ti SUPER (16 GB)"],
            "provenance_host": "mbp-2 (guillaume@100.115.197.17)",
        },
        "cv_protocol": {
            "n_cases": 54,
            "n_folds": 5,
            "split_rule": "patient_index i % 5 == fold (identical across all methods)",
            "matching_threshold_mm": 3.5,
            "n_long_cy_catheters": 759,
            "n_short_ht_obturator_catheters": 304,
            "n_total_catheters": 1063,
        },
        "pmb2019_provenance_reconciliation_mbp2": prov_summary,
        "voxel_segmentation_oof_dice": {
            "Baseline_1_PMB2019_Iso1mm_UNet3D": round(float(b1_data["mean_oof_dice"]), 4),
            "Baseline_2_Native_nnUNet3D_clDice": round(float(b2_data["mean_oof_dice"]), 4),
        },
        "cohort_all_1063_catheters_54cases": results_1063,
        "cohort_long_759_cy_catheters_54cases": results_759,
        "human_interobserver_agreement_40_overlapping_cases": {
            "overlapping_case_numbers": overlap_case_nums,
            "CY_evaluated_against_LB_AMe_reference_513_needles": human_vs_lb_summary,
            "LB_AMe_evaluated_against_CY_reference_505_needles": human_vs_cy_summary,
        },
        "paired_wilcoxon_signed_rank_tests_54_patients": {
            "cohort_all_1063_catheters": wilcoxon_1063,
            "cohort_long_759_cy_catheters": wilcoxon_759,
        },
        "total_eval_wall_time_s": round(time.perf_counter() - t0, 2),
    }

    out_path = "/tmp/2026-10-22-unified-apples-to-apples-benchmark.json"
    with open(out_path, "w") as f:
        json.dump(final_json, f, indent=2)
    print(f"\n[Saved] {out_path} in {time.perf_counter() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
