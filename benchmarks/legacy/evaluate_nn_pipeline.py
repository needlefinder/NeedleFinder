#!/usr/bin/env python3
"""
Evaluate Out-of-Fold (5-Fold CV) 3D NeedleUNet + Stage 2 Trajectory Reconstruction
across all 54 GYN Cases (both 759 CY Long Needles and 1,063 CY+HT All Needles)
and on full-FOV case.nrrd (Case 016 NRRD 640x416x144).
"""

import glob
import math
import os
import sys
import time
import types
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment

for mod in ("vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util", "qt", "ctk", "SimpleITK", "sitkUtils"):
    sys.modules[mod] = types.ModuleType(mod)


class MockMatrix4x4:
    def __init__(self):
        self._m = np.eye(4, dtype=np.float64)

    def GetElement(self, r, c):
        return float(self._m[r, c])


sys.modules["vtk"].vtkMatrix4x4 = MockMatrix4x4
sys.modules["slicer"].modules = types.SimpleNamespace(NeedleFinderWidget=None)

sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")

from NeedleFinderLib.TrackingAlgorithms import TrackingAlgorithmsMixin
import NeedleFinderLib.TrackingAlgorithms as ta_mod
from prepare_and_inspect_dataset import load_all_dataset_cases, read_nrrd_local, read_vtk_centerline_local

ta_mod.get_volume_f64_buffer = lambda img: (img, img.shape[::-1])


class StandaloneEvalLogic(TrackingAlgorithmsMixin):
    def findAxialSegmentationLimitFromMarker(self, bForceFallback=False):
        return 0, 0.0


class MockVolumeNode:
    def __init__(self, vol, sp, ijk_to_ras):
        self._vol = vol
        self._sp = tuple(float(x) for x in sp)
        self._mat = np.array(ijk_to_ras, dtype=np.float64)

    def GetImageData(self):
        return self._vol

    def GetSpacing(self):
        return self._sp

    def GetIJKToRASMatrix(self, vtk_mat):
        vtk_mat._m = self._mat.copy()

    def GetID(self):
        return "MockVolumeNode_1"


def compute_pair_metrics(det_ras, gt_ras):
    z_min = max(float(det_ras[0, 2]), float(gt_ras[0, 2]))
    z_max = min(float(det_ras[-1, 2]), float(gt_ras[-1, 2]))
    if z_max <= z_min + 1.0:
        return None
    zs = np.linspace(z_min, z_max, 40)
    dx = np.interp(zs, det_ras[:, 2], det_ras[:, 0]) - np.interp(zs, gt_ras[:, 2], gt_ras[:, 0])
    dy = np.interp(zs, det_ras[:, 2], det_ras[:, 1]) - np.interp(zs, gt_ras[:, 2], gt_ras[:, 1])
    xy_dists = np.hypot(dx, dy)
    mean_shaft_xy = float(np.mean(xy_dists))
    max_hd_xy = float(np.max(xy_dists))
    p95_hd_xy = float(np.percentile(xy_dists, 95))

    det_tip = det_ras[-1]
    gt_tip = gt_ras[-1]
    tip_err_3d = float(np.linalg.norm(det_tip - gt_tip))
    tip_err_xy = float(np.hypot(det_tip[0] - gt_tip[0], det_tip[1] - gt_tip[1]))
    tip_err_z = float(abs(det_tip[2] - gt_tip[2]))

    gt_base = gt_ras[0]
    bx_at_gt = float(np.interp(gt_base[2], det_ras[:, 2], det_ras[:, 0]))
    by_at_gt = float(np.interp(gt_base[2], det_ras[:, 2], det_ras[:, 1]))
    base_err_xy = float(np.hypot(bx_at_gt - gt_base[0], by_at_gt - gt_base[1]))

    return {
        "mean_shaft_xy": mean_shaft_xy,
        "max_hd_xy": max_hd_xy,
        "p95_hd_xy": p95_hd_xy,
        "tip_err_3d": tip_err_3d,
        "tip_err_xy": tip_err_xy,
        "tip_err_z": tip_err_z,
        "base_err_xy": base_err_xy,
    }


def extract_nn_labelmap_tubes(
    p_mask, p_peak, vx_map, vy_map, p_obt,
    resp_roi, k0, i0, j0, sp_i, sp_j, sp_k,
    ijk_to_ras, ras_to_ijk, existing_trajs,
    include_obturator=False,
):
    """
    Stage 2 Direct Labelmap + Direction-Field Tube Extractor:
    1. Masks out existing_trajs from the 3D neural probability volume (p_mask, p_peak).
    2. Traces continuous 3D centerline tubes along local maxima of (p_peak * p_mask)
       guided by the predicted 3D tangent vector field (vx_map, vy_map).
    """
    K_sub, H, W = p_mask.shape
    rot_mat = ijk_to_ras[:3, :3]
    trans_vec = ijk_to_ras[:3, 3]
    z_base_ras = float((rot_mat @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0 + 1)]) + trans_vec)[2])

    work_peak = (p_peak.astype(np.float32) * (0.45 + 0.55 * p_mask.astype(np.float32))).copy()
    if include_obturator:
        work_peak = np.maximum(work_peak, p_peak.astype(np.float32) * p_obt.astype(np.float32) * 1.15)

    # Mask out already-detected trajectories (1.85 mm radius cylinder)
    ri_m = max(1, int(round(1.85 / sp_i)))
    rj_m = max(1, int(round(1.85 / sp_j)))
    yy_m, xx_m = np.mgrid[-rj_m:rj_m + 1, -ri_m:ri_m + 1]
    disk_m = ((xx_m * sp_i) ** 2 + (yy_m * sp_j) ** 2) <= (1.85 ** 2)

    for tr in existing_trajs:
        pts_r = tr["ras"] if "ras" in tr else tr["points"]
        pts_h = np.column_stack([pts_r, np.ones(len(pts_r))])
        pts_ijk = (ras_to_ijk @ pts_h.T).T[:, :3]
        order = np.argsort(pts_ijk[:, 2])
        pts_ijk = pts_ijk[order]
        k_lo = max(0, int(math.floor(pts_ijk[0, 2])) - k0)
        k_hi = min(K_sub - 1, int(math.ceil(pts_ijk[-1, 2])) - k0)
        for kr in range(k_lo, k_hi + 1):
            ic = int(round(float(np.interp(kr + k0, pts_ijk[:, 2], pts_ijk[:, 0])) - i0))
            jc = int(round(float(np.interp(kr + k0, pts_ijk[:, 2], pts_ijk[:, 1])) - j0))
            if 0 <= ic < W and 0 <= jc < H:
                j0_p, j1_p = max(0, jc - rj_m), min(H, jc + rj_m + 1)
                i0_p, i1_p = max(0, ic - ri_m), min(W, ic + ri_m + 1)
                d_sub = disk_m[(j0_p - (jc - rj_m)):(j1_p - (jc - rj_m)), (i0_p - (ic - ri_m)):(i1_p - (ic - ri_m))]
                work_peak[kr, j0_p:j1_p, i0_p:i1_p][d_sub] = 0.0

    # Smooth slightly in-plane and find 2D local maxima per slice
    nms_px = max(1, int(round(1.35 / min(sp_i, sp_j))))
    lm = work_peak == maximum_filter(work_peak, size=(1, 2 * nms_px + 1, 2 * nms_px + 1))
    seed_Mask = lm & (work_peak >= 0.42) & (p_mask >= 0.45)
    sk, sj, si = np.where(seed_Mask)
    if len(sk) == 0:
        return []

    scores = work_peak[sk, sj, si]
    order_seeds = np.argsort(-scores)
    visited = np.zeros_like(work_peak, dtype=bool)
    search_r = max(2, int(round(2.4 / min(sp_i, sp_j))))

    new_trajs = []
    min_slices = 4 if sp_k >= 3.5 else (5 if sp_k >= 2.2 else 7)

    for idx_s in order_seeds[:400]:
        ks0, js0, is0 = int(sk[idx_s]), int(sj[idx_s]), int(si[idx_s])
        if visited[ks0, js0, is0]:
            continue

        pts = {ks0: (float(is0), float(js0), float(work_peak[ks0, js0, is0]))}

        # Trace forward (+1) and backward (-1) using predicted 3D tangent field (vx, vy)
        for direction in (1, -1):
            curr_k = ks0
            curr_i, curr_j = float(is0), float(js0)
            missed = 0
            while True:
                next_k = curr_k + direction
                if not (0 <= next_k < K_sub):
                    break
                # Read neural tangent slope (dx/dz, dy/dz) and convert to (di/dk, dj/dk)
                ci_int = int(np.clip(round(curr_i), 0, W - 1))
                cj_int = int(np.clip(round(curr_j), 0, H - 1))
                vx_loc = float(vx_map[curr_k, cj_int, ci_int])
                vy_loc = float(vy_map[curr_k, cj_int, ci_int])
                # Convert physical (dx, dy) step over dz = direction * sp_k into (di, dj)
                pt_curr_ras = rot_mat @ np.array([curr_i + i0, curr_j + j0, float(curr_k + k0)]) + trans_vec
                dz_step = float((rot_mat @ np.array([curr_i + i0, curr_j + j0, float(next_k + k0)]) + trans_vec)[2] - pt_curr_ras[2])
                target_ras = pt_curr_ras + np.array([vx_loc * dz_step, vy_loc * dz_step, dz_step])
                target_ijk = ras_to_ijk @ np.append(target_ras, 1.0)
                pred_i = float(target_ijk[0] - i0)
                pred_j = float(target_ijk[1] - j0)

                pi_int, pj_int = int(round(pred_i)), int(round(pred_j))
                if not (1 <= pi_int < W - 1 and 1 <= pj_int < H - 1):
                    break

                j_lo, j_hi = max(0, pj_int - search_r), min(H, pj_int + search_r + 1)
                i_lo, i_hi = max(0, pi_int - search_r), min(W, pi_int + search_r + 1)
                patch = work_peak[next_k, j_lo:j_hi, i_lo:i_hi]
                yy_p, xx_p = np.mgrid[j_lo:j_hi, i_lo:i_hi]
                dist_mm = np.hypot((xx_p - pred_i) * sp_i, (yy_p - pred_j) * sp_j)
                weighted = patch * np.exp(-0.5 * (dist_mm / 1.65) ** 2) * (dist_mm <= 2.45)
                if weighted.size == 0 or float(np.max(weighted)) < 0.24:
                    missed += 1
                    if missed > 2:
                        break
                    curr_k = next_k
                    curr_i, curr_j = pred_i, pred_j
                    continue

                lj, li = np.unravel_index(int(np.argmax(weighted)), weighted.shape)
                best_j, best_i = j_lo + lj, i_lo + li
                raw_p = float(work_peak[next_k, best_j, best_i])
                if raw_p < 0.25 or float(p_mask[next_k, best_j, best_i]) < 0.28:
                    missed += 1
                    if missed > 2:
                        break
                    curr_k = next_k
                    curr_i, curr_j = pred_i, pred_j
                    continue

                # Sub-pixel centroid refinement on work_peak
                if 1 <= best_j < H - 1 and 1 <= best_i < W - 1:
                    sub3 = np.maximum(0.0, work_peak[next_k, best_j - 1:best_j + 2, best_i - 1:best_i + 2])
                    wsum = float(np.sum(sub3))
                    if wsum > 1e-4:
                        ym, xm = np.mgrid[-1:2, -1:2]
                        best_i_f = float(best_i) + float(np.sum(xm * sub3)) / wsum
                        best_j_f = float(best_j) + float(np.sum(ym * sub3)) / wsum
                    else:
                        best_i_f, best_j_f = float(best_i), float(best_j)
                else:
                    best_i_f, best_j_f = float(best_i), float(best_j)

                pts[next_k] = (best_i_f, best_j_f, raw_p)
                curr_k = next_k
                curr_i, curr_j = best_i_f, best_j_f
                missed = 0

        if len(pts) < min_slices:
            continue

        ks = sorted(pts.keys())
        k_min_rel = min(ks)
        if not include_obturator and k_min_rel > int(round(32.0 / sp_k)):
            continue

        arr_pts = np.array([[pts[k][0] + i0, pts[k][1] + j0, k + k0, pts[k][2]] for k in ks], dtype=np.float64)
        ras_pts = (rot_mat @ arr_pts[:, :3].T).T + trans_vec
        order_z = np.argsort(ras_pts[:, 2])
        ras_pts = ras_pts[order_z]
        arr_pts = arr_pts[order_z]
        ks = [ks[i] for i in order_z]

        z_vals = ras_pts[:, 2]
        z_span = float(z_vals[-1] - z_vals[0])
        min_span = 11.5 if include_obturator else (12.0 if K_sub <= 28 else 16.0)
        if z_span < min_span:
            continue

        # Fit line / degree-2 polynomial
        deg = 2 if len(z_vals) >= 8 else 1
        px = np.polyfit(z_vals, ras_pts[:, 0], deg)
        py = np.polyfit(z_vals, ras_pts[:, 1], deg)
        fit_x = np.polyval(px, z_vals)
        fit_y = np.polyval(py, z_vals)
        rmse = float(np.percentile(np.hypot(ras_pts[:, 0] - fit_x, ras_pts[:, 1] - fit_y), 85))
        if rmse > 1.35:
            continue

        dz = float(z_vals[-1] - z_vals[0])
        sx = float(fit_x[-1] - fit_x[0]) / dz
        sy = float(fit_y[-1] - fit_y[0]) / dz
        if math.hypot(sx, sy) > 0.65:
            continue

        # Check mean NN mask probability and resp support
        mean_pm = float(np.mean([p_mask[k, int(round(pts[k][1])), int(round(pts[k][0]))] for k in ks]))
        mean_po = float(np.mean([p_obt[k, int(round(pts[k][1])), int(round(pts[k][0]))] for k in ks]))
        mean_r = float(np.mean([resp_roi[k, int(round(pts[k][1])), int(round(pts[k][0]))] for k in ks]))
        if mean_pm < 0.45 and mean_po < 0.45:
            continue
        if mean_r <= 0.0:
            continue

        # Mark visited and mask out work_peak
        for k_v in ks:
            iv, jv = int(round(pts[k_v][0])), int(round(pts[k_v][1]))
            j0_p, j1_p = max(0, jv - rj_m), min(H, jv + rj_m + 1)
            i0_p, i1_p = max(0, iv - ri_m), min(W, iv + ri_m + 1)
            visited[k_v, j0_p:j1_p, i0_p:i1_p] = True
            work_peak[k_v, j0_p:j1_p, i0_p:i1_p] *= 0.05

        # Construct smooth output trajectory points
        if not include_obturator and z_vals[0] > z_base_ras + 0.5 * sp_k and abs(z_vals[0] - z_base_ras) <= 22.0:
            bx = float(fit_x[0] + sx * (z_base_ras - z_vals[0]))
            by = float(fit_y[0] + sy * (z_base_ras - z_vals[0]))
            smooth_ras = np.vstack([[bx, by, z_base_ras], np.column_stack([fit_x, fit_y, z_vals])])
        else:
            smooth_ras = np.column_stack([fit_x, fit_y, z_vals])

        new_trajs.append({
            "ras": smooth_ras,
            "points": smooth_ras,
            "base_xy": smooth_ras[0, :2],
            "tip_ras": smooth_ras[-1],
            "confidence": float(np.clip(mean_pm * 100.0, 50.0, 99.9)),
            "length_mm": float(smooth_ras[-1, 2] - smooth_ras[0, 2]),
            "num_points": len(ks),
            "pass": 4 if include_obturator else 3,
        })

    return new_trajs


def filter_and_augment_with_nn(
    base_trajs, oof_npz, vol, sp_ras, ijk_to_ras, include_obturator=False
):
    """
    Combines the classical/JEV trajectories with the 3D U-Net Out-of-Fold predictions:
    1. Scores each trajectory against the 3D U-Net probability map (p_mask, p_peak, p_obt)
       and 3D tangent vector field (vx, vy), rejecting false positives (vessels/air/artifact).
    2. Extracts any missed long needles and (when include_obturator=True) short obturator needles
       directly from the 3D U-Net labelmap + tangent field.
    """
    p_mask = oof_npz["p_mask"].astype(np.float32)
    p_peak = oof_npz["p_peak"].astype(np.float32)
    vx_map = oof_npz["vx"].astype(np.float32)
    vy_map = oof_npz["vy"].astype(np.float32)
    p_obt = oof_npz["p_obt"].astype(np.float32)
    k0 = int(oof_npz["k0"])
    i0, i1 = int(oof_npz["i0"]), int(oof_npz["i1"])
    j0, j1 = int(oof_npz["j0"]), int(oof_npz["j1"])
    K_sub, H, W = p_mask.shape

    ras_to_ijk = np.linalg.inv(ijk_to_ras)
    sp_i, sp_j, sp_k = float(sp_ras[0]), float(sp_ras[1]), float(sp_ras[2])

    # Evaluate each base trajectory on the 3D U-Net labelmap
    kept_trajs = []
    for tr in base_trajs:
        pts_r = tr["ras"] if "ras" in tr else tr["points"]
        pts_h = np.column_stack([pts_r, np.ones(len(pts_r))])
        pts_ijk = (ras_to_ijk @ pts_h.T).T[:, :3]
        order = np.argsort(pts_ijk[:, 2])
        pts_ijk = pts_ijk[order]
        pts_r_s = pts_r[order]

        k_lo = max(0, int(math.ceil(pts_ijk[0, 2])) - k0)
        k_hi = min(K_sub - 1, int(math.floor(pts_ijk[-1, 2])) - k0)
        if k_hi < k_lo:
            kept_trajs.append(tr)
            continue

        pm_vals = []
        pp_vals = []
        for kr in range(k_lo, k_hi + 1):
            ic = int(round(float(np.interp(kr + k0, pts_ijk[:, 2], pts_ijk[:, 0])) - i0))
            jc = int(round(float(np.interp(kr + k0, pts_ijk[:, 2], pts_ijk[:, 1])) - j0))
            if 1 <= ic < W - 1 and 1 <= jc < H - 1:
                pm_vals.append(float(np.max(p_mask[kr, jc - 1:jc + 2, ic - 1:ic + 2])))
                pp_vals.append(float(np.max(p_peak[kr, jc - 1:jc + 2, ic - 1:ic + 2])))

        if not pm_vals:
            kept_trajs.append(tr)
            continue

        mean_pm = float(np.mean(pm_vals))
        p75_pm = float(np.percentile(pm_vals, 75))
        mean_pp = float(np.mean(pp_vals))

        # Reject trajectories that the 3D U-Net clearly classifies as non-needle background
        if mean_pm < 0.16 and p75_pm < 0.24 and mean_pp < 0.14 and tr.get("confidence", 100.0) < 88.0:
            continue

        kept_trajs.append(tr)

    # Extract any missed needles from the 3D U-Net labelmap + tangent field
    sub_roi = vol[k0:k0 + K_sub, j0:j1, i0:i1]
    non_zero = sub_roi[sub_roi > 0]
    body_thresh = float(np.percentile(non_zero, 15)) if len(non_zero) > 100 else 10.0
    from NeedleFinderLib.RustCore import rust_compute_3d_sym_response
    resp_full, _, _ = rust_compute_3d_sym_response(vol, sp_i, sp_j, i0, i1, j0, j1, k0, k0 + K_sub, body_thresh)
    resp_roi = resp_full[k0:k0 + K_sub, j0:j1, i0:i1]

    nn_extra = extract_nn_labelmap_tubes(
        p_mask, p_peak, vx_map, vy_map, p_obt,
        resp_roi, k0, i0, j0, sp_i, sp_j, sp_k,
        ijk_to_ras, ras_to_ijk, kept_trajs,
        include_obturator=include_obturator,
    )
    return kept_trajs + nn_extra


def evaluate_all():
    cases = load_all_dataset_cases()
    logic = StandaloneEvalLogic()
    oof_dir = "/home/guillaume/github/needlefinder/nn_models"

    for mode_name, use_obt in [("LONG NEEDLES (759 CY GT)", False), ("ALL NEEDLES: LONG + OBTURATOR (1,063 CY+HT GT)", True)]:
        tot_gt = 0
        tot_det = 0
        tot_clos = 0
        tot_hun = 0
        shaft_errs = []
        tip_errs = []

        for idx, c in enumerate(cases):
            vnode = MockVolumeNode(c["vol"], c["sp_ras"], c["ijk_to_ras"])
            base_trajs = logic.autoDetectAllNeedles(
                vnode,
                minConfidence=35.0,
                kLim=c["k_lim_gt"],
                renderNeedles=False,
                script=True,
            )
            oof_path = os.path.join(oof_dir, f"oof_case_{idx:02d}.npz")
            if os.path.exists(oof_path):
                oof_npz = np.load(oof_path)
                trajs = filter_and_augment_with_nn(
                    base_trajs, oof_npz, c["vol"], c["sp_ras"], c["ijk_to_ras"],
                    include_obturator=use_obt,
                )
            else:
                trajs = base_trajs

            gts = c["gts_all"] if use_obt else c["gts_long"]
            n_gt = len(gts)
            n_det = len(trajs)
            tot_gt += n_gt
            tot_det += n_det

            cost_m = np.full((n_gt, max(1, n_det)), 1e6, dtype=np.float64)
            pair_info = {}
            for gi, g in enumerate(gts):
                for di, d in enumerate(trajs):
                    d_pts = d["ras"] if "ras" in d else d["points"]
                    pm = compute_pair_metrics(d_pts, g["pts"])
                    if pm is not None:
                        cost_m[gi, di] = pm["mean_shaft_xy"]
                        pair_info[(gi, di)] = pm

            clos_5 = int(np.sum(np.min(cost_m, axis=1) <= 5.0)) if n_det > 0 else 0
            tot_clos += clos_5

            if n_det > 0:
                r_ind, c_ind = linear_sum_assignment(cost_m)
                for gi, di in zip(r_ind, c_ind):
                    if cost_m[gi, di] <= 5.0:
                        tot_hun += 1
                        shaft_errs.append(pair_info[(gi, di)]["mean_shaft_xy"])
                        tip_errs.append(pair_info[(gi, di)]["tip_err_xy"])

        print(f"\n=======================================================")
        print(f"BENCHMARK: {mode_name}")
        print(f"=======================================================")
        print(f"  Total GT Needles:        {tot_gt}")
        print(f"  Total Detected:          {tot_det}")
        print(f"  Closest-Match Recall:    {tot_clos}/{tot_gt} ({tot_clos/tot_gt*100:.2f}%)")
        print(f"  1-to-1 Hungarian Recall: {tot_hun}/{tot_gt} ({tot_hun/tot_gt*100:.2f}%)")
        print(f"  Precision:               {tot_hun}/{tot_det} ({tot_hun/max(1,tot_det)*100:.2f}%)")
        print(f"  Mean Shaft XY Error:     {np.mean(shaft_errs):.2f} mm (median {np.median(shaft_errs):.2f} mm)")
        print(f"  Mean Tip XY Error:       {np.mean(tip_errs):.2f} mm")


if __name__ == "__main__":
    evaluate_all()
