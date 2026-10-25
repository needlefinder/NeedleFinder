"""Paper matching and geometry metrics, independent of training and Slicer APIs."""
import numpy as np
from scipy.optimize import linear_sum_assignment

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


