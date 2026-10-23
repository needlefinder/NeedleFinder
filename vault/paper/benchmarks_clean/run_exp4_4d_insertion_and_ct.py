#!/usr/bin/env python3
"""
Experiment 4: 4D Intra-Procedural Serial Insertion & Deflection Tracking (Cases 68-71)
& Paired Post-Implant Pelvic CT (40 NRRD-CT Volumes) Cross-Modality Evaluation.

Saves full quantitative report to:
  /home/guillaume/github/needlefinder/vault/sources/2026-10-23-exp4-4d-insertion-and-ct.json
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
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr
import torch

for mod in (
    "vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util",
    "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"
):
    sys.modules[mod] = types.ModuleType(mod)


class MockMatrix4x4:
    def __init__(self):
        self._m = np.eye(4, dtype=np.float64)

    def GetElement(self, r, c):
        return float(self._m[r, c])


sys.modules["vtk"].vtkMatrix4x4 = MockMatrix4x4
sys.modules["slicer"].modules = types.SimpleNamespace(NeedleFinderWidget=None)

sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun")
sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")

from prepare_clean_dataset import (
    StandaloneEvalLogic,
    MockVolumeNode,
    read_nrrd_local,
    read_vtk_centerline_local,
    dedup_needle_list,
    estimate_axial_limit_slice_sym_safe,
)
from NeedleFinderLib.RustCore import rust_compute_3d_sym_response
from NeedleFinderLib.NeuralNeedleNet import extract_nn_labelmap_tubes
from train_and_eval_unet3d import NeedleUNet3D
from train_nested_stage1 import infer_single_view_3d, raw_out_to_probs
from evaluate_unified_apples_to_apples import (
    get_gts,
    compute_unified_pair_metrics,
    evaluate_method_unified,
    traj_gt_dist,
)
from run_xiao2026_and_fewshot_benchmark import regularize_poly3_trajectory

CLEAN_META_PATH = "/tmp/cases_meta_54_clean.pkl"
CLEAN_OOF_PREDS_PATH = "/tmp/stage2_clean_final_oof_predictions_full.pkl"
CLEAN_MODELS_DIR = "/home/guillaume/github/needlefinder/nn_models_clean"
HT_DIR = "/home/guillaume/Dropbox/GYN Cases/Hongtao Tracking Catheter Motion During Procedure"
GYN_DIR = "/home/guillaume/Dropbox/GYN Cases"
OUT_JSON_PATH = "/home/guillaume/github/needlefinder/vault/sources/2026-10-23-exp4-4d-insertion-and-ct.json"


# ==============================================================================
# HELPER FUNCTIONS: GEOMETRY, RIGID REGISTRATION, AND HELD-OUT OUTER-FOLD INFERENCE
# ==============================================================================

def get_traj_pts(t):
    if isinstance(t, dict):
        p = t.get("ras", t.get("points", t.get("pts")))
    else:
        p = t
    arr = np.asarray(p, dtype=np.float64)
    return arr[np.argsort(arr[:, 2])]


def read_acsv_point(path):
    for line in open(path):
        if line.startswith("point|"):
            parts = line.strip().split("|")
            return np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    return None


def load_acsv_grouped_trajectories(folder_path):
    """Loads .G-K.acsv fiducial files grouped by needle index G into sorted 3D polylines."""
    acsvs = sorted(glob.glob(os.path.join(folder_path, ".*-*.acsv")))
    groups = {}
    for ap in acsvs:
        bn = os.path.basename(ap)
        m = re.match(r"^\.(\d+)-(\d+)\.acsv$", bn)
        if not m:
            continue
        g_id, pt_id = int(m.group(1)), int(m.group(2))
        pt = read_acsv_point(ap)
        if pt is not None:
            groups.setdefault(g_id, []).append((pt_id, pt))
    trajs = []
    for g_id in sorted(groups.keys()):
        pts = np.array([x[1] for x in sorted(groups[g_id], key=lambda x: x[0])], dtype=np.float64)
        pts = pts[np.argsort(pts[:, 2])]
        if len(pts) >= 2 and (pts[-1, 2] - pts[0, 2]) >= 5.0:
            trajs.append(pts)
    return trajs


def load_vtk_trajectories_from_dir(folder_path):
    vtks = sorted(glob.glob(os.path.join(folder_path, "manual-seg_*.vtk")))
    raw = []
    for vp in vtks:
        bn = os.path.basename(vp)
        if bn.endswith(".incorrect"):
            continue
        pts = read_vtk_centerline_local(vp)
        if pts is not None and len(pts) >= 2:
            pts = pts[np.argsort(pts[:, 2])]
            if (pts[-1, 2] - pts[0, 2]) >= 5.0:
                raw.append({"name": bn[:-4], "pts": pts})
    deduped = dedup_needle_list(raw, dist_thresh=2.0)
    return [d["pts"] for d in deduped]


def load_outer_fold_stage1_ensemble(outer_fold: int, device: torch.device):
    models = []
    for m_in in range(5):
        ckpt = os.path.join(CLEAN_MODELS_DIR, f"unet3d_outer{outer_fold}_inner{m_in}.pt")
        if os.path.exists(ckpt):
            net = NeedleUNet3D(6, 24).to(device)
            net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            net.eval()
            models.append(net)
    return models


def run_heldout_outer_fold_inference_on_mri(vol, sp_ras, ijk_to_ras, outer_fold_models, logic, device, min_conf=50.0):
    """
    Runs NeedleFinder 2026 on a 3T T2W MRI volume using:
      - Stage 0 physical 3D circular symmetry response (`rust_compute_3d_sym_response`)
      - Held-out outer-fold Stage 1 `NeedleUNet3D` ensemble (`outer_fold_models`, never trained on this patient)
      - Stage 2 graph trajectory decoding & 3rd-order polynomial regularization
    """
    t0 = time.perf_counter()
    K, J, I = vol.shape
    sp_i, sp_j, sp_k = float(sp_ras[0]), float(sp_ras[1]), float(sp_ras[2])
    ras_to_ijk = np.linalg.inv(ijk_to_ras)

    k_lim = estimate_axial_limit_slice_sym_safe(vol, sp_ras) if K > 55 else 0
    vnode = MockVolumeNode(vol, sp_ras, ijk_to_ras)
    trajs_u = logic.autoDetectAllNeedles(vnode, minConfidence=25.0, kLim=k_lim, renderNeedles=False, script=False)

    # Also run held-out outer-fold Clean Stage 1 ensemble on the pelvic ROI to corroborate & refine tips/tubes
    margin_i = min(I // 2 - 4, int(round(68.0 / sp_i)))
    margin_j = min(J // 2 - 4, int(round(68.0 / sp_j)))
    half_w = ((margin_i + 7) // 8) * 8
    half_h = ((margin_j + 7) // 8) * 8
    ci_v, cj_v = I // 2, J // 2
    i0 = max(0, ci_v - half_w)
    i1 = min(I, i0 + 2 * half_w)
    j0 = max(0, cj_v - half_h)
    j1 = min(J, j0 + 2 * half_h)
    k0 = k_lim

    roi_raw = vol[k0:K, j0:j1, i0:i1]
    pos_raw = roi_raw[roi_raw > 5.0]
    med_tissue = float(np.percentile(pos_raw, 55)) if len(pos_raw) else 85.0
    if med_tissue > 450.0:
        scale_factor = 160.0 / med_tissue
        vol_work = np.ascontiguousarray(vol * scale_factor, dtype=np.float64)
        body_thresh = float(np.percentile(pos_raw * scale_factor, 30))
    else:
        vol_work = vol.copy()
        body_thresh = float(np.percentile(pos_raw, 30)) if len(pos_raw) else 25.0

    resp_full, cs_full, rsym_full = rust_compute_3d_sym_response(vol_work, sp_i, sp_j, i0, i1, j0, j1, k0, K, body_thresh)
    resp1 = resp_full[k0:K, j0:j1, i0:i1].copy()
    cs1 = cs_full[k0:K, j0:j1, i0:i1]
    rsym1 = rsym_full[k0:K, j0:j1, i0:i1]

    s1_clean_tubes = []
    if outer_fold_models:
        sub_roi_c = vol_work[k0:K, j0:j1, i0:i1]
        K_c, H_c, W_c = sub_roi_c.shape
        nz_c = sub_roi_c[sub_roi_c > 0]
        p10_c = float(np.percentile(nz_c, 10)) if len(nz_c) > 100 else 0.0
        p95_c = float(np.percentile(nz_c, 95)) if len(nz_c) > 100 else 100.0
        ch0_c = np.clip((sub_roi_c - p10_c) / max(1.0, p95_c - p10_c), -0.5, 2.5).astype(np.float32)
        pos_rc = resp1[resp1 > 0]
        r_sc = float(np.percentile(pos_rc, 99)) if len(pos_rc) > 20 else 50.0
        ch1_c = np.clip(np.log1p(np.maximum(0.0, resp1)) / max(1.0, math.log1p(r_sc)), 0.0, 1.5).astype(np.float32)
        pos_cc = cs1[cs1 > 0]
        c_sc = float(np.percentile(pos_cc, 99)) if len(pos_cc) > 20 else 25.0
        ch2_c = np.clip(cs1 / max(1.0, c_sc), 0.0, 1.5).astype(np.float32)
        ch3_c = np.clip((rsym1 - sub_roi_c) / (rsym1 + 15.0), -0.5, 1.0).astype(np.float32)
        ks_rel_c = np.arange(K_c, dtype=np.float32)[:, None, None]
        ch4_c = np.broadcast_to((ks_rel_c * sp_k) / 120.0, (K_c, H_c, W_c)).copy()
        yy_c, xx_c = np.mgrid[0:H_c, 0:W_c].astype(np.float32)
        r_mm_c = np.hypot((xx_c - W_c * 0.5) * sp_i, (yy_c - H_c * 0.5) * sp_j) / 50.0
        ch5_c = np.broadcast_to(r_mm_c[None, :, :], (K_c, H_c, W_c)).copy()
        x_c = np.stack([ch0_c, ch1_c, ch2_c, ch3_c, ch4_c, ch5_c], axis=0)
        pk = (2 - (K_c % 2)) % 2
        ph = (8 - (H_c % 8)) % 8
        pw = (8 - (W_c % 8)) % 8
        if pk > 0 or ph > 0 or pw > 0:
            x_c = np.pad(x_c, ((0, 0), (0, pk), (0, ph), (0, pw)), mode="edge")
        x_ssl = torch.from_numpy(x_c)
        outs = [infer_single_view_3d(m, x_ssl, device)[:, :K_c, :H_c, :W_c] for m in outer_fold_models[:3]]
        probs_c = raw_out_to_probs(torch.stack(outs, dim=0).mean(dim=0), temp_pos=1.0)
        nn_clean = {
            "p_mask": probs_c[0].numpy().astype(np.float32),
            "p_peak": probs_c[1].numpy().astype(np.float32),
            "vx": probs_c[2].numpy().astype(np.float32),
            "vy": probs_c[3].numpy().astype(np.float32),
            "p_obt": probs_c[4].numpy().astype(np.float32),
        }
        s1_clean_tubes = extract_nn_labelmap_tubes(
            nn_clean, resp1, k0, i0, j0, sp_i, sp_j, sp_k,
            ijk_to_ras, ras_to_ijk, trajs_u, include_obturator=True,
        )

    # Merge high-confidence trajectories and supplement with held-out Stage 1 clean tubes
    sel = []
    for t in sorted(trajs_u, key=lambda x: -float(x.get("confidence", 0.0))):
        conf = float(t.get("confidence", 0.0))
        if conf < min_conf:
            continue
        pts = regularize_poly3_trajectory(get_traj_pts(t), degree=3)
        if any(traj_gt_dist(pts, s["ras"]) < 1.95 for s in sel):
            continue
        sel.append({
            "ras": pts,
            "points": pts,
            "confidence": conf,
            "base_xy": pts[0, :2].copy(),
            "tip_xyz": pts[-1].copy(),
            "z_span": float(pts[-1, 2] - pts[0, 2]),
        })

    for nt in s1_clean_tubes:
        conf = float(nt.get("confidence", 55.0))
        if conf < min_conf:
            continue
        pts = regularize_poly3_trajectory(get_traj_pts(nt), degree=3)
        if (pts[-1, 2] - pts[0, 2]) < 12.0:
            continue
        if any(traj_gt_dist(pts, s["ras"]) < 2.10 for s in sel):
            continue
        sel.append({
            "ras": pts,
            "points": pts,
            "confidence": conf,
            "base_xy": pts[0, :2].copy(),
            "tip_xyz": pts[-1].copy(),
            "z_span": float(pts[-1, 2] - pts[0, 2]),
        })

    elapsed = time.perf_counter() - t0
    return sel, trajs_u, resp_full, ijk_to_ras, elapsed


# ==============================================================================
# PART 4A: CAUSAL 4D TEMPORAL KALMAN/HUNGARIAN TRACKER & BIOMECHANICS
# ==============================================================================

def estimate_interscan_rigid_xy_z_shift(prev_tracks, curr_dets, prev_z_fov=None, curr_z_fov=None):
    """
    Estimates the rigid whole-bundle shift (dx, dy, dz) between consecutive intra-procedural scans
    t_{k-1} and t_k so that table/patient translation or FOV Z-origin changes are separated from
    local inter-catheter tissue-push deformation.
    """
    if not prev_tracks or not curr_dets:
        return np.zeros(3, dtype=np.float64)

    # Check if the scanner table landmark Z-origin was reset between scans (only occurs in Case 68 Series 11 -> 12,
    # where cranial z_fov[1] jumps from +106.9 mm to +17.6 mm, a -89.3 mm landmark shift).
    dz_landmark = 0.0
    if prev_z_fov is not None and curr_z_fov is not None:
        dz_top = float(curr_z_fov[1]) - float(prev_z_fov[1])
        if abs(dz_top) > 60.0:
            dz_landmark = dz_top

    p_mids = np.array([t["ras"][len(t["ras"]) // 2] for t in prev_tracks])
    c_mids = np.array([t["ras"][len(t["ras"]) // 2] for t in curr_dets])

    # Search physiological inter-scan table/patient XY shift within [-6.0, +6.0] mm
    best_dx, best_dy = 0.0, 0.0
    best_cost = 1e9
    for gx in np.linspace(-6.0, 6.0, 13):
        for gy in np.linspace(-6.0, 6.0, 13):
            dmat = np.hypot((p_mids[:, 0:1] + gx) - c_mids[None, :, 0], (p_mids[:, 1:2] + gy) - c_mids[None, :, 1])
            ri, ci = linear_sum_assignment(dmat)
            vals = np.sort(dmat[ri, ci])
            k_keep = max(2, int(0.75 * len(vals)))
            c_val = float(np.mean(vals[:k_keep]))
            if c_val < best_cost:
                best_cost = c_val
                best_dx, best_dy = float(gx), float(gy)

    dmat = np.hypot((p_mids[:, 0:1] + best_dx) - c_mids[None, :, 0], (p_mids[:, 1:2] + best_dy) - c_mids[None, :, 1])
    ri, ci = linear_sum_assignment(dmat)
    inliers = [(r, c) for r, c in zip(ri, ci) if dmat[r, c] <= 4.5]
    if len(inliers) >= 3:
        dx_in = float(np.median([c_mids[c, 0] - p_mids[r, 0] for r, c in inliers]))
        dy_in = float(np.median([c_mids[c, 1] - p_mids[r, 1] for r, c in inliers]))
        return np.array([dx_in, dy_in, dz_landmark], dtype=np.float64)

    return np.array([best_dx if abs(best_dx) > 1.5 else 0.0, best_dy if abs(best_dy) > 1.5 else 0.0, dz_landmark], dtype=np.float64)


def run_causal_4d_kalman_tracker(timepoint_records):
    """
    Given sequential timepoint records for a patient, runs Causal 4D Temporal Kalman/Hungarian Tracking:
      - Propagates active catheter tracks T(t_{k-1}) into t_k using estimated rigid inter-scan shift
      - Associates current detections (including lower-threshold candidates >= 32% that match an existing
        template hole within 3.5 mm) via Hungarian matching
      - Applies Kalman filtering on shaft XY (alpha=0.75 current, 0.25 prior) and enforces smooth/monotonic
        cranial tip progression z_tip(t_k) >= z_tip(t_{k-1}) - 1.5 mm (within FOV bounds)
      - Retains persistent tracks across 1-frame signal dropouts if confidence was high
      - Identifies newly inserted catheters at step t_k and measures inter-catheter tissue-push displacement!
    """
    active_tracks = []
    next_track_id = 1

    step_analytics = []
    all_tissue_push_samples = []
    all_tip_advancements = []

    for step_idx, rec in enumerate(timepoint_records):
        curr_dets_50 = rec["indep_preds_50"]
        curr_cands_32 = rec["all_cands_25"]
        z_fov_min, z_fov_max = rec["z_fov_ras"]

        if step_idx == 0 or len(active_tracks) == 0:
            active_tracks = []
            for d in curr_dets_50:
                pts = d["ras"].copy()
                active_tracks.append({
                    "track_id": next_track_id,
                    "ras": pts,
                    "points": pts,
                    "confidence": float(d["confidence"]),
                    "first_seen_step": step_idx,
                    "last_seen_step": step_idx,
                    "missed_count": 0,
                    "is_new": True,
                    "dz_tip_advance_mm": 0.0,
                })
                next_track_id += 1

            rec["causal_4d_preds"] = [dict(t) for t in active_tracks]
            step_analytics.append({
                "step_index": step_idx,
                "series_name": rec["series_name"],
                "n_indep_50": len(curr_dets_50),
                "n_causal_4d": len(active_tracks),
                "n_persisted": 0,
                "n_newly_inserted": len(active_tracks),
                "n_recovered_dropout": 0,
                "rigid_shift_xyz_mm": [0.0, 0.0, 0.0],
                "mean_tip_advance_mm": 0.0,
                "max_tip_advance_mm": 0.0,
                "tissue_push_near_le_12mm_median": None,
                "tissue_push_far_gt_12mm_median": None,
            })
            continue

        prev_z_fov = timepoint_records[step_idx - 1]["z_fov_ras"]
        shift_xyz = estimate_interscan_rigid_xy_z_shift(active_tracks, curr_dets_50, prev_z_fov=prev_z_fov, curr_z_fov=(z_fov_min, z_fov_max))

        prop_tracks = []
        for tr in active_tracks:
            p_shifted = tr["ras"] + shift_xyz[None, :]
            prop_tracks.append(dict(tr, ras_prop=p_shifted))

        valid_cands = [
            c for c in curr_cands_32
            if float(c.get("confidence", 0.0)) >= 32.0 and (get_traj_pts(c)[-1, 2] - get_traj_pts(c)[0, 2]) >= 10.0
        ]
        cand_pts_list = [regularize_poly3_trajectory(get_traj_pts(c), degree=3) for c in valid_cands]
        cand_confs = [float(c.get("confidence", 0.0)) for c in valid_cands]

        N_prev = len(prop_tracks)
        N_cand = len(cand_pts_list)
        cost_mat = np.full((N_prev, N_cand), 999.0, dtype=np.float64)

        for i, tr in enumerate(prop_tracks):
            p_pr = tr["ras_prop"]
            for j, c_pts in enumerate(cand_pts_list):
                z0 = max(float(p_pr[0, 2]), float(c_pts[0, 2]))
                z1 = min(float(p_pr[-1, 2]), float(c_pts[-1, 2]))
                if z1 > z0 + 3.0:
                    zs = np.linspace(z0, z1, 10)
                    dx = np.interp(zs, c_pts[:, 2], c_pts[:, 0]) - np.interp(zs, p_pr[:, 2], p_pr[:, 0])
                    dy = np.interp(zs, c_pts[:, 2], c_pts[:, 1]) - np.interp(zs, p_pr[:, 2], p_pr[:, 1])
                    d_xy = float(np.mean(np.hypot(dx, dy)))
                else:
                    d_xy = float(np.linalg.norm(c_pts[len(c_pts)//2, :2] - p_pr[len(p_pr)//2, :2]))
                if d_xy <= 4.0:
                    cost_mat[i, j] = d_xy - 0.01 * cand_confs[j]

        ri, ci = linear_sum_assignment(cost_mat)
        matched_prev = {}
        used_cands = set()
        for r, c in zip(ri, ci):
            if cost_mat[r, c] <= 3.8:
                matched_prev[r] = c
                used_cands.add(c)

        new_active_tracks = []
        persisted_shifts = []
        tip_advances_step = []
        n_recovered = 0

        for r, tr in enumerate(prop_tracks):
            p_pr = tr["ras_prop"]
            if r in matched_prev:
                c_idx = matched_prev[r]
                c_pts = cand_pts_list[c_idx]
                c_conf = cand_confs[c_idx]
                if c_conf < 50.0:
                    n_recovered += 1

                z_ov0 = max(float(p_pr[0, 2]), float(c_pts[0, 2]))
                z_ov1 = min(float(p_pr[-1, 2]), float(c_pts[-1, 2]))
                if z_ov1 > z_ov0 + 4.0:
                    zm = 0.5 * (z_ov0 + z_ov1)
                    xy_prev = np.array([float(np.interp(zm, p_pr[:, 2], p_pr[:, 0])), float(np.interp(zm, p_pr[:, 2], p_pr[:, 1]))])
                    xy_curr = np.array([float(np.interp(zm, c_pts[:, 2], c_pts[:, 0])), float(np.interp(zm, c_pts[:, 2], c_pts[:, 1]))])
                else:
                    xy_prev = p_pr[len(p_pr)//2, :2]
                    xy_curr = c_pts[len(c_pts)//2, :2]
                lat_shift = float(np.linalg.norm(xy_curr - xy_prev))
                persisted_shifts.append((xy_curr, lat_shift))

                z_bot_new = min(float(c_pts[0, 2]), max(float(z_fov_min), float(p_pr[0, 2])))
                z_top_prev_fov = min(float(z_fov_max) - 1.0, float(p_pr[-1, 2]))
                z_top_new = max(float(c_pts[-1, 2]), z_top_prev_fov - 1.0)
                dz_adv = max(0.0, float(c_pts[-1, 2]) - float(p_pr[-1, 2]))
                tip_advances_step.append(dz_adv)
                all_tip_advancements.append(dz_adv)

                zs_kalman = np.linspace(z_bot_new, z_top_new, 25)
                x_c = np.interp(zs_kalman, c_pts[:, 2], c_pts[:, 0])
                y_c = np.interp(zs_kalman, c_pts[:, 2], c_pts[:, 1])
                x_p = np.interp(zs_kalman, p_pr[:, 2], p_pr[:, 0])
                y_p = np.interp(zs_kalman, p_pr[:, 2], p_pr[:, 1])
                alpha_k = 0.75
                pts_k = np.column_stack([alpha_k * x_c + (1.0 - alpha_k) * x_p, alpha_k * y_c + (1.0 - alpha_k) * y_p, zs_kalman])

                new_active_tracks.append({
                    "track_id": tr["track_id"],
                    "ras": pts_k,
                    "points": pts_k,
                    "confidence": max(float(c_conf), 0.92 * float(tr["confidence"])),
                    "first_seen_step": tr["first_seen_step"],
                    "last_seen_step": step_idx,
                    "missed_count": 0,
                    "is_new": False,
                    "dz_tip_advance_mm": dz_adv,
                    "lateral_shift_mm": lat_shift,
                })
            else:
                if tr["missed_count"] == 0 and tr["confidence"] >= 55.0:
                    z_b = max(float(z_fov_min), float(p_pr[0, 2]))
                    z_t = min(float(z_fov_max), float(p_pr[-1, 2]))
                    if z_t > z_b + 10.0:
                        zs_k = np.linspace(z_b, z_t, 20)
                        pts_k = np.column_stack([
                            np.interp(zs_k, p_pr[:, 2], p_pr[:, 0]),
                            np.interp(zs_k, p_pr[:, 2], p_pr[:, 1]),
                            zs_k,
                        ])
                        n_recovered += 1
                        new_active_tracks.append({
                            "track_id": tr["track_id"],
                            "ras": pts_k,
                            "points": pts_k,
                            "confidence": 0.85 * float(tr["confidence"]),
                            "first_seen_step": tr["first_seen_step"],
                            "last_seen_step": tr["last_seen_step"],
                            "missed_count": tr["missed_count"] + 1,
                            "is_new": False,
                            "dz_tip_advance_mm": 0.0,
                            "lateral_shift_mm": 0.0,
                        })

        newly_inserted_xy = []
        n_new = 0
        for j, c_pts in enumerate(cand_pts_list):
            if j in used_cands or cand_confs[j] < 50.0:
                continue
            if any(traj_gt_dist(c_pts, t["ras"]) < 2.20 for t in new_active_tracks):
                continue
            n_new += 1
            newly_inserted_xy.append(c_pts[len(c_pts)//2, :2])
            new_active_tracks.append({
                "track_id": next_track_id,
                "ras": c_pts,
                "points": c_pts,
                "confidence": cand_confs[j],
                "first_seen_step": step_idx,
                "last_seen_step": step_idx,
                "missed_count": 0,
                "is_new": True,
                "dz_tip_advance_mm": float(c_pts[-1, 2] - c_pts[0, 2]),
                "lateral_shift_mm": 0.0,
            })
            next_track_id += 1

        near_pushes, far_pushes = [], []
        if len(newly_inserted_xy) > 0 and len(persisted_shifts) > 0:
            new_xy_arr = np.array(newly_inserted_xy)
            for xy_p, shift_mm in persisted_shifts:
                min_d_new = float(np.min(np.linalg.norm(new_xy_arr - xy_p[None, :], axis=1)))
                all_tissue_push_samples.append((min_d_new, shift_mm, f"{rec['case_id']}_{rec['series_name']}"))
                if min_d_new <= 12.0:
                    near_pushes.append(shift_mm)
                else:
                    far_pushes.append(shift_mm)

        active_tracks = new_active_tracks
        rec["causal_4d_preds"] = [dict(t) for t in active_tracks]
        step_analytics.append({
            "step_index": step_idx,
            "series_name": rec["series_name"],
            "n_indep_50": len(curr_dets_50),
            "n_causal_4d": len(active_tracks),
            "n_persisted": len(active_tracks) - n_new,
            "n_newly_inserted": n_new,
            "n_recovered_dropout": n_recovered,
            "rigid_shift_xyz_mm": [round(float(x), 2) for x in shift_xyz],
            "mean_tip_advance_mm": round(float(np.mean(tip_advances_step)), 2) if tip_advances_step else 0.0,
            "max_tip_advance_mm": round(float(np.max(tip_advances_step)), 2) if tip_advances_step else 0.0,
            "tissue_push_near_le_12mm_median": round(float(np.median(near_pushes)), 3) if near_pushes else None,
            "tissue_push_far_gt_12mm_median": round(float(np.median(far_pushes)), 3) if far_pushes else None,
        })

    return step_analytics, all_tissue_push_samples, all_tip_advancements


def analyze_depth_dependent_bending(trajs_list):
    """
    Quantifies depth-dependent needle bending / deflection from the straight basal entry tangent:
      - Fits initial entry tangent over the bottom 20 mm (or bottom 35% of shaft)
      - Measures lateral deviation ||r_xy(z) - r_straight_xy(z)|| (mm) as a function of insertion depth (z - z_base)
      - Reports bending across depth bins: [0-30mm, 30-60mm, 60-90mm, >90mm] and tip deflection vs total insertion depth.
    """
    depth_bins = {"0_30mm": [], "30_60mm": [], "60_90mm": [], "gt_90mm": []}
    catheter_records = []

    for t in trajs_list:
        pts = get_traj_pts(t)
        if len(pts) < 4:
            continue
        z_base = float(pts[0, 2])
        z_tip = float(pts[-1, 2])
        L = z_tip - z_base
        if L < 20.0:
            continue

        z_entry_hi = z_base + min(22.0, 0.35 * L)
        zs_ent = np.linspace(z_base, z_entry_hi, 8)
        xs_ent = np.interp(zs_ent, pts[:, 2], pts[:, 0])
        ys_ent = np.interp(zs_ent, pts[:, 2], pts[:, 1])
        vx_0 = float(np.polyfit(zs_ent - z_base, xs_ent, 1)[0])
        vy_0 = float(np.polyfit(zs_ent - z_base, ys_ent, 1)[0])
        x0, y0 = float(pts[0, 0]), float(pts[0, 1])

        zs_eval = np.linspace(z_base, z_tip, 30)
        depths = zs_eval - z_base
        xs_actual = np.interp(zs_eval, pts[:, 2], pts[:, 0])
        ys_actual = np.interp(zs_eval, pts[:, 2], pts[:, 1])
        xs_straight = x0 + vx_0 * depths
        ys_straight = y0 + vy_0 * depths
        devs = np.hypot(xs_actual - xs_straight, ys_actual - ys_straight)

        for d_val, dev_val in zip(depths, devs):
            if d_val <= 30.0:
                depth_bins["0_30mm"].append(float(dev_val))
            elif d_val <= 60.0:
                depth_bins["30_60mm"].append(float(dev_val))
            elif d_val <= 90.0:
                depth_bins["60_90mm"].append(float(dev_val))
            else:
                depth_bins["gt_90mm"].append(float(dev_val))

        tip_dev = float(devs[-1])
        max_dev = float(np.max(devs))
        catheter_records.append({
            "insertion_depth_mm": round(L, 2),
            "tip_deflection_from_entry_tangent_mm": round(tip_dev, 3),
            "max_deflection_mm": round(max_dev, 3),
        })

    summary_bins = {}
    for k, vals in depth_bins.items():
        if vals:
            summary_bins[k] = {
                "n_samples": len(vals),
                "mean_deflection_mm": round(float(np.mean(vals)), 3),
                "median_deflection_mm": round(float(np.median(vals)), 3),
                "iqr_deflection_mm": [round(float(np.percentile(vals, 25)), 3), round(float(np.percentile(vals, 75)), 3)],
                "p90_deflection_mm": round(float(np.percentile(vals, 90)), 3),
            }
    depths_all = [r["insertion_depth_mm"] for r in catheter_records]
    tips_all = [r["tip_deflection_from_entry_tangent_mm"] for r in catheter_records]
    rho, p_val = spearmanr(depths_all, tips_all) if len(depths_all) >= 5 else (0.0, 1.0)

    return {
        "n_catheters_analyzed": len(catheter_records),
        "deflection_by_depth_bin": summary_bins,
        "overall_tip_deflection_mean_mm": round(float(np.mean(tips_all)), 3) if tips_all else 0.0,
        "overall_tip_deflection_median_mm": round(float(np.median(tips_all)), 3) if tips_all else 0.0,
        "overall_tip_deflection_p90_mm": round(float(np.percentile(tips_all, 90)), 3) if tips_all else 0.0,
        "spearman_depth_vs_tip_deflection_r": round(float(rho), 4),
        "spearman_p_value": float(p_val),
    }


# ==============================================================================
# PART 4B: PELVIC CT CROSS-MODALITY TRACKING & 3D RIGID BUNDLE REGISTRATION
# ==============================================================================

def make_ct_polarity_volume(vol_hu, mode="dual"):
    """
    Maps a Pelvic CT volume (in Hounsfield Units) into a pseudo-T2W dark-core tubular volume [0, 220]
    for zero-retraining Stage 0 (rust_compute_3d_sym_response) + Stage 1/2 inference:
      - mode == 'dark_air': hollow air-filled plastic catheter lumens (< -90 HU inside soft tissue) -> dark troughs
      - mode == 'bright_stylet': hyperdense metallic dummy markers/stylets (> +55 HU above local soft tissue) -> dark troughs
      - mode == 'dual': max-fusion of narrow bright stylet cores and narrow dark air-lumen cores, suppressing wide bone/bowel air.
    """
    vol_f = vol_hu.astype(np.float32)
    loc_mean = uniform_filter(vol_f, size=(1, 11, 11))
    in_body = (loc_mean > -260.0) & (vol_f > -990.0)

    bright_dev = np.clip(vol_f - np.maximum(35.0, loc_mean + 20.0), 0.0, 420.0) / 420.0
    dark_dev = np.where(loc_mean > -220.0, np.clip((-75.0) - vol_f, 0.0, 650.0) / 650.0, 0.0)

    if mode == "bright_stylet":
        tube_ness = bright_dev
    elif mode == "dark_air":
        tube_ness = dark_dev
    else:
        tube_ness = np.maximum(bright_dev, 0.90 * dark_dev)

    pseudo_t2 = np.where(in_body, 175.0 * (1.0 - 0.88 * tube_ness), 0.0)
    return np.ascontiguousarray(np.clip(pseudo_t2, 0.0, 220.0), dtype=np.float64)


def register_ct_bundle_to_mri_bundle(ct_trajs, mri_trajs):
    """
    Fast vectorized 6-DoF 3D rigid bundle registration (R, t) aligning CT trajectories to MRI trajectories.
    """
    if not ct_trajs or not mri_trajs:
        return [], np.eye(3), np.zeros(3), 999.0

    def sample_pts(t_list, n_p=15):
        out = []
        for t in t_list:
            p = get_traj_pts(t)
            if len(p) >= 2 and (p[-1, 2] - p[0, 2]) >= 8.0:
                zs = np.linspace(p[0, 2], p[-1, 2], n_p)
                xs = np.interp(zs, p[:, 2], p[:, 0])
                ys = np.interp(zs, p[:, 2], p[:, 1])
                out.append(np.column_stack([xs, ys, zs]))
        return out

    C_list = sample_pts(ct_trajs)
    M_list = sample_pts(mri_trajs)
    if not C_list or not M_list:
        return [], np.eye(3), np.zeros(3), 999.0

    c_all = np.vstack(C_list)
    m_all = np.vstack(M_list)
    c_cent = np.median(c_all, axis=0)
    m_cent = np.median(m_all, axis=0)

    c_tips = np.array([c[-1] for c in C_list])
    m_tips = np.array([m[-1] for m in M_list])
    tz_init = float(np.median(m_tips[:, 2]) - np.median(c_tips[:, 2]))
    tx_init = float(m_cent[0] - c_cent[0])
    ty_init = float(m_cent[1] - c_cent[1])

    def make_rot(rx, ry, rz):
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    m_mids = np.array([m[len(m)//2] for m in M_list])
    c_mids = np.array([c[len(c)//2] for c in C_list])

    best_p = np.array([0.0, 0.0, 0.0, tx_init, ty_init, tz_init], dtype=np.float64)
    best_loss = 1e9
    for dx in np.linspace(-22.0, 22.0, 7):
        for dy in np.linspace(-22.0, 22.0, 7):
            for dz in np.linspace(-25.0, 25.0, 5):
                t_vec = np.array([tx_init + dx, ty_init + dy, tz_init + dz])
                c_sh = c_mids + t_vec
                dmat = np.linalg.norm(c_sh[:, None, :2] - m_mids[None, :, :2], axis=-1) + 0.20 * np.abs(c_sh[:, None, 2] - m_mids[None, :, 2])
                ri, ci = linear_sum_assignment(dmat)
                vals = np.sort(dmat[ri, ci])
                k_keep = max(2, int(0.75 * len(vals)))
                loss = float(np.mean(vals[:k_keep]))
                if loss < best_loss:
                    best_loss = loss
                    best_p = np.array([0.0, 0.0, 0.0, t_vec[0], t_vec[1], t_vec[2]])

    C_arr = np.stack(C_list, axis=0)
    M_arr = np.stack(M_list, axis=0)
    C_centered = C_arr - c_cent[None, None, :]

    def obj_fn(params):
        rx, ry, rz, tx, ty, tz = params
        if abs(rx) > 0.35 or abs(ry) > 0.35 or abs(rz) > 0.35:
            return 1e6
        R = make_rot(rx, ry, rz)
        t_v = c_cent + np.array([tx, ty, tz])
        C_tr = np.einsum("ij,nkj->nki", R, C_centered) + t_v[None, None, :]
        c_sub = C_tr[:, [2, 5, 8, 11, 14], :2]
        m_sub = M_arr[:, [2, 5, 8, 11, 14], :2]
        d_xy = np.mean(np.linalg.norm(c_sub[:, None, :, :] - m_sub[None, :, :, :], axis=-1), axis=-1)
        d_tip_z = np.abs(C_tr[:, -1, 2][:, None] - M_arr[:, -1, 2][None, :])
        cost = d_xy + 0.22 * d_tip_z
        ri, ci = linear_sum_assignment(cost)
        vals = np.sort(cost[ri, ci])
        k_keep = max(2, int(0.80 * len(vals)))
        return float(np.mean(vals[:k_keep]))

    res = minimize(obj_fn, best_p, method="Powell", options={"maxiter": 120, "ftol": 1e-3})
    rx, ry, rz, tx, ty, tz = res.x
    R_opt = make_rot(rx, ry, rz)
    t_opt = np.array([tx, ty, tz])

    aligned_ct = []
    for t in ct_trajs:
        p = get_traj_pts(t)
        p_tr = (R_opt @ (p - c_cent).T).T + c_cent + t_opt
        p_tr = p_tr[np.argsort(p_tr[:, 2])]
        aligned_ct.append({
            "ras": p_tr,
            "points": p_tr,
            "confidence": float(t.get("confidence", 50.0)) if isinstance(t, dict) else 50.0,
        })

    return aligned_ct, R_opt, t_opt, float(res.fun)


def main():
    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Exp 4] Running on {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})", flush=True)

    meta_list = pickle.load(open(CLEAN_META_PATH, "rb"))
    oof_preds_pkl = pickle.load(open(CLEAN_OOF_PREDS_PATH, "rb"))
    nf_oof_all_50 = oof_preds_pkl["all_tau50"]
    nf_oof_all_35 = oof_preds_pkl["all_tau35"]

    bench_idx_by_cnum = {int(m["case_num"]): idx for idx, m in enumerate(meta_list)}
    logic = StandaloneEvalLogic()

    # Load all 5 outer-fold Stage 1 ensembles
    outer_models = {f_out: load_outer_fold_stage1_ensemble(f_out, device) for f_out in range(5)}

    # ==========================================================================
    # PART 4A: 4D INTRA-PROCEDURAL SERIAL INSERTION & DEFLECTION TRACKING
    # ==========================================================================
    print("\n==================================================================", flush=True)
    print("PART 4A: 4D Intra-Procedural Catheter Insertion & Deflection (Cases 68-71)", flush=True)
    print("==================================================================", flush=True)

    case_series_4d = {
        68: [
            "6 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "7 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "8 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "9 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "11 AX space trial 1.6 SLICE COVER TEMPLATE.nrrd",
            "12 AX space trial 1.6 SLICE COVER TEMPLATE.nrrd",
        ],
        69: [
            "9 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "11 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "13 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "15 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "17 AX space trial 1.6 SLICE COVER TEMPLATE.nrrd",
        ],
        70: [
            "6 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "11 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "12 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "13 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "14 AX space trial 1.6 SLICE COVER TEMPLATE.nrrd",
        ],
        71: [
            "7 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "9 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "11 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "13 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "15 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "18 AX TSE T2 MONITORING CATH PLACEMENT.nrrd",
            "19 AX space trial 1.6 SLICE COVER TEMPLATE.nrrd",
        ],
    }

    part4a_patients = {}
    all_4d_push_samples = []
    all_4d_tip_advances = []
    all_final_3d_trajs = []
    indep_vs_causal_evals = []
    all_monitoring_runtimes = []

    for cnum, s_list in case_series_4d.items():
        b_idx = bench_idx_by_cnum[cnum]
        f_out = b_idx % 5
        m_info = meta_list[b_idx]
        gts_cy_ht = get_gts(m_info["gts_all"])
        eh_dir = os.path.join(GYN_DIR, f"Case  0{cnum}", "NRRD", "Manual_EH")
        gts_eh = load_vtk_trajectories_from_dir(eh_dir) if os.path.exists(eh_dir) else []

        print(f"\n--- Case {cnum} (bench_idx={b_idx}, strictly held-out outer_fold={f_out}, GT_CY+HT={len(gts_cy_ht)}, GT_EH={len(gts_eh)}) ---", flush=True)

        tp_records = []
        for s_idx, fn in enumerate(s_list):
            p = os.path.join(HT_DIR, f"{cnum} NRRD-MRI", fn)
            vol, sp_ras, ijk2ras = read_nrrd_local(p)[:3]
            c0 = (ijk2ras @ np.array([0, 0, 0, 1]))[:3]
            c1 = (ijk2ras @ np.array([vol.shape[2]-1, vol.shape[1]-1, vol.shape[0]-1, 1]))[:3]
            z_fov = (float(min(c0[2], c1[2])), float(max(c0[2], c1[2])))

            sel_50, cands_25, _, _, elapsed = run_heldout_outer_fold_inference_on_mri(
                vol, sp_ras, ijk2ras, outer_models[f_out], logic, device, min_conf=50.0
            )
            if "TSE" in fn:
                all_monitoring_runtimes.append(elapsed)
            tp_records.append({
                "case_num": cnum,
                "case_id": f"Case {cnum:03d}",
                "step_index": s_idx,
                "series_name": fn.replace(".nrrd", ""),
                "shape_kji": list(vol.shape),
                "spacing_ras_mm": [round(float(x), 3) for x in sp_ras],
                "z_fov_ras": [round(z_fov[0], 2), round(z_fov[1], 2)],
                "runtime_sec": round(elapsed, 3),
                "indep_preds_50": sel_50,
                "all_cands_25": cands_25,
            })

        step_analytics, push_samples, tip_advs = run_causal_4d_kalman_tracker(tp_records)
        all_4d_push_samples.extend(push_samples)
        all_4d_tip_advances.extend(tip_advs)

        for sa in step_analytics:
            print(
                f"  [t={sa['step_index']}] {sa['series_name']:46s} | Indep={sa['n_indep_50']:2d} -> Causal4D={sa['n_causal_4d']:2d} "
                f"(persisted={sa['n_persisted']:2d}, new={sa['n_newly_inserted']:2d}, recov={sa['n_recovered_dropout']:1d}) | "
                f"dZ_tip_mean={sa['mean_tip_advance_mm']:4.1f}mm | push_near={sa['tissue_push_near_le_12mm_median']}mm",
                flush=True,
            )

        # Evaluate against per-timepoint & final ground-truth annotations
        eval_annotations = {}

        # 1. Final 3D SPACE scan vs Rater 1+2 (CY + HT, clipped to pelvic ROI in benchmark)
        final_rec = tp_records[-1]
        res_indep_final, _ = evaluate_method_unified([final_rec["indep_preds_50"]], [gts_cy_ht], thresh=3.5, n_boot=50)
        res_causal_final, _ = evaluate_method_unified([final_rec["causal_4d_preds"]], [gts_cy_ht], thresh=3.5, n_boot=50)
        eval_annotations["final_3d_space_vs_cy_ht"] = {
            "n_gt": len(gts_cy_ht),
            "indep_recall_pct": res_indep_final["hungarian_1to1_recall_pct"],
            "indep_precision_pct": res_indep_final["precision_1to1_pct"],
            "indep_f1_pct": res_indep_final["f1_1to1_pct"],
            "causal_4d_recall_pct": res_causal_final["hungarian_1to1_recall_pct"],
            "causal_4d_precision_pct": res_causal_final["precision_1to1_pct"],
            "causal_4d_f1_pct": res_causal_final["f1_1to1_pct"],
            "causal_4d_shaft_median_mm": round(float(res_causal_final["metrics"]["mean_shaft_xy"]["median"]), 3),
            "causal_4d_tip_median_mm": round(float(res_causal_final["metrics"]["tip_3d"]["median"]), 3),
        }
        indep_vs_causal_evals.append(("final_3d_" + str(cnum), res_indep_final["f1_1to1_pct"], res_causal_final["f1_1to1_pct"]))
        all_final_3d_trajs.extend(final_rec["causal_4d_preds"])

        # 2. Check Hongtao's intra-procedural subset annotations:
        if cnum == 68 and len(gts_eh) > 0:
            # Series 11 is in Manual_EH coordinate frame
            s11_rec = tp_records[-2]
            r_eh_c, _ = evaluate_method_unified([s11_rec["causal_4d_preds"]], [gts_eh], thresh=3.5, n_boot=20)
            eval_annotations["series_11_space_vs_manual_eh"] = {
                "n_gt_eh": len(gts_eh),
                "causal_4d_recall_pct": r_eh_c["hungarian_1to1_recall_pct"],
                "causal_4d_f1_pct": r_eh_c["f1_1to1_pct"],
                "shaft_median_mm": round(float(r_eh_c["metrics"]["mean_shaft_xy"]["median"]), 3),
            }
        elif cnum == 70:
            # 70 NRRD-MRI/needle registration/manual-seg_1..11.vtk (10 trajectories on Series 6)
            ht_70_vtks = load_vtk_trajectories_from_dir(os.path.join(HT_DIR, "70 NRRD-MRI", "needle registration"))
            s6_indep = tp_records[0]["indep_preds_50"]
            s6_causal = tp_records[0]["causal_4d_preds"]
            r_s6_i, _ = evaluate_method_unified([s6_indep], [ht_70_vtks], thresh=3.5, n_boot=20)
            r_s6_c, _ = evaluate_method_unified([s6_causal], [ht_70_vtks], thresh=3.5, n_boot=20)
            eval_annotations["series_6_monitoring_vs_ht_10_needles"] = {
                "n_gt_annotated_subset": len(ht_70_vtks),
                "indep_recall_pct": r_s6_i["hungarian_1to1_recall_pct"],
                "causal_4d_recall_pct": r_s6_c["hungarian_1to1_recall_pct"],
                "shaft_median_mm": round(float(r_s6_c["metrics"]["mean_shaft_xy"]["median"]), 3),
                "tip_median_mm": round(float(r_s6_c["metrics"]["tip_3d"]["median"]), 3),
            }
        elif cnum == 71:
            # 71 NRRD-MRI/obturator registration/manual-seg_1..6.vtk (6 trajectories segmented on transformed Series 9/11)
            ht_71_obt = load_vtk_trajectories_from_dir(os.path.join(HT_DIR, "71 NRRD-MRI", "obturator registration"))
            for s_i, s_tag in [(1, "series_9"), (2, "series_11")]:
                aligned_obt_gt, _, _, _ = register_ct_bundle_to_mri_bundle(
                    [{"ras": g, "points": g} for g in ht_71_obt],
                    [t["ras"] for t in tp_records[s_i]["causal_4d_preds"]],
                )
                obt_gt_ras = [g["ras"] for g in aligned_obt_gt] if aligned_obt_gt else ht_71_obt
                r_i, _ = evaluate_method_unified([tp_records[s_i]["indep_preds_50"]], [obt_gt_ras], thresh=3.8, n_boot=20)
                r_c, _ = evaluate_method_unified([tp_records[s_i]["causal_4d_preds"]], [obt_gt_ras], thresh=3.8, n_boot=20)
                eval_annotations[f"{s_tag}_monitoring_vs_ht_6_obturator_needles"] = {
                    "n_gt_annotated_subset": len(ht_71_obt),
                    "indep_recall_pct": r_i["hungarian_1to1_recall_pct"],
                    "causal_4d_recall_pct": r_c["hungarian_1to1_recall_pct"],
                    "shaft_median_mm": round(float(r_c["metrics"]["mean_shaft_xy"]["median"]), 3) if r_c["hungarian_1to1_matched"] > 0 else None,
                }
            ht_71_s19 = load_vtk_trajectories_from_dir(os.path.join(HT_DIR, "71 NRRD-MRI"))
            aligned_s19_gt, _, _, _ = register_ct_bundle_to_mri_bundle(
                [{"ras": g, "points": g} for g in ht_71_s19],
                [t["ras"] for t in tp_records[-1]["causal_4d_preds"]],
            )
            s19_gt_ras = [g["ras"] for g in aligned_s19_gt] if aligned_s19_gt else ht_71_s19
            r_19, _ = evaluate_method_unified([tp_records[-1]["causal_4d_preds"]], [s19_gt_ras], thresh=3.8, n_boot=20)
            eval_annotations["series_19_vs_ht_4_tracked_needles"] = {
                "n_gt_annotated_subset": len(ht_71_s19),
                "recall_pct": r_19["hungarian_1to1_recall_pct"],
                "shaft_median_mm": round(float(r_19["metrics"]["mean_shaft_xy"]["median"]), 3) if r_19["hungarian_1to1_matched"] > 0 else None,
            }
        elif cnum == 69:
            s9_trajs = [t["ras"] for t in tp_records[0]["indep_preds_50"]]
            r_s11_indep, _ = evaluate_method_unified([tp_records[1]["indep_preds_50"]], [s9_trajs], thresh=3.5, n_boot=20)
            r_s11_causal, _ = evaluate_method_unified([tp_records[1]["causal_4d_preds"]], [s9_trajs], thresh=3.5, n_boot=20)
            eval_annotations["series_11_dropout_recovery_vs_series_9_bundle"] = {
                "n_series_9_bundle": len(s9_trajs),
                "indep_series_11_recall_pct": r_s11_indep["hungarian_1to1_recall_pct"],
                "causal_4d_series_11_recall_pct": r_s11_causal["hungarian_1to1_recall_pct"],
            }

        part4a_patients[f"Case_{cnum:03d}"] = {
            "case_num": cnum,
            "bench_idx": b_idx,
            "held_out_outer_fold": f_out,
            "n_timepoints": len(tp_records),
            "step_by_step_timeline": step_analytics,
            "annotation_evaluations": eval_annotations,
            "bending_analysis": analyze_depth_dependent_bending(final_rec["causal_4d_preds"]),
        }

    # Summarize Tissue Push across all 4D transitions
    near_8 = [s[1] for s in all_4d_push_samples if s[0] <= 10.0]
    mid_10_20 = [s[1] for s in all_4d_push_samples if 10.0 < s[0] <= 20.0]
    far_20 = [s[1] for s in all_4d_push_samples if s[0] > 20.0]
    near_12 = [s[1] for s in all_4d_push_samples if s[0] <= 12.0]
    far_12 = [s[1] for s in all_4d_push_samples if s[0] > 12.0]

    mw_u, mw_p = mannwhitneyu(near_12, far_12, alternative="greater") if (len(near_12) >= 3 and len(far_12) >= 3) else (0.0, 1.0)
    dists_push = [s[0] for s in all_4d_push_samples]
    vals_push = [s[1] for s in all_4d_push_samples]
    rho_push, p_push = spearmanr(dists_push, vals_push) if len(dists_push) >= 5 else (0.0, 1.0)

    overall_bending = analyze_depth_dependent_bending(all_final_3d_trajs)

    part4a_summary = {
        "n_patients_4d": 4,
        "patient_ids": ["Case 068", "Case 069", "Case 070", "Case 071"],
        "total_serial_scans_evaluated": sum(len(v) for v in case_series_4d.values()),
        "mean_monitoring_scan_runtime_sec": round(float(np.mean(all_monitoring_runtimes or [1.05])), 3),
        "tissue_push_biomechanics": {
            "total_persisted_catheter_transitions": len(all_4d_push_samples),
            "near_newly_inserted_le_10mm": {
                "n": len(near_8),
                "mean_lateral_shift_mm": round(float(np.mean(near_8)), 3) if near_8 else 0.0,
                "median_lateral_shift_mm": round(float(np.median(near_8)), 3) if near_8 else 0.0,
                "iqr_lateral_shift_mm": [round(float(np.percentile(near_8, 25)), 3), round(float(np.percentile(near_8, 75)), 3)] if near_8 else [0.0, 0.0],
                "p90_lateral_shift_mm": round(float(np.percentile(near_8, 90)), 3) if near_8 else 0.0,
            },
            "intermediate_10_to_20mm": {
                "n": len(mid_10_20),
                "mean_lateral_shift_mm": round(float(np.mean(mid_10_20)), 3) if mid_10_20 else 0.0,
                "median_lateral_shift_mm": round(float(np.median(mid_10_20)), 3) if mid_10_20 else 0.0,
                "iqr_lateral_shift_mm": [round(float(np.percentile(mid_10_20, 25)), 3), round(float(np.percentile(mid_10_20, 75)), 3)] if mid_10_20 else [0.0, 0.0],
            },
            "far_from_newly_inserted_gt_20mm": {
                "n": len(far_20),
                "mean_lateral_shift_mm": round(float(np.mean(far_20)), 3) if far_20 else 0.0,
                "median_lateral_shift_mm": round(float(np.median(far_20)), 3) if far_20 else 0.0,
                "iqr_lateral_shift_mm": [round(float(np.percentile(far_20, 25)), 3), round(float(np.percentile(far_20, 75)), 3)] if far_20 else [0.0, 0.0],
            },
            "mann_whitney_near_le_12mm_vs_far_gt_12mm_p": float(mw_p),
            "spearman_distance_vs_lateral_push_r": round(float(rho_push), 4),
            "spearman_distance_vs_lateral_push_p": float(p_push),
        },
        "depth_dependent_bending_biomechanics": overall_bending,
        "per_patient_4d_results": part4a_patients,
    }

    print("\n=== PART 4A BIOMECHANICS SUMMARY ===", flush=True)
    print(json.dumps({
        "tissue_push": part4a_summary["tissue_push_biomechanics"],
        "depth_bending": part4a_summary["depth_dependent_bending_biomechanics"],
    }, indent=2), flush=True)

    if "--part4a-only" in sys.argv and os.path.exists(OUT_JSON_PATH):
        existing_report = json.load(open(OUT_JSON_PATH))
        existing_report["part4a_4d_intraprocedural_tracking"] = part4a_summary
        with open(OUT_JSON_PATH, "w") as f:
            json.dump(existing_report, f, indent=2)
        print(f"\n[Done] Updated Part 4A in {OUT_JSON_PATH} in {time.time() - t_start:.1f}s!", flush=True)
        return

    # ==========================================================================
    # PART 4B: PAIRED POST-IMPLANT PELVIC CT (40 NRRD-CT VOLUMES) EVALUATION
    # ==========================================================================
    print("\n==================================================================", flush=True)
    print("PART 4B: Paired Post-Implant Pelvic CT (40 NRRD-CT Volumes) Cross-Modality Evaluation", flush=True)
    print("==================================================================", flush=True)

    ct_dirs = sorted(glob.glob(os.path.join(GYN_DIR, "Case*", "NRRD*CT*")))
    case_ct_map = {}
    for d in ct_dirs:
        m = re.search(r"Case\s*(\d+)", d)
        if not m:
            continue
        cnum = int(m.group(1))
        nrrds = sorted(glob.glob(os.path.join(d, "*.nrrd")))
        pelvis_nrrds = [p for p in nrrds if "topogram" not in os.path.basename(p).lower()]
        if not pelvis_nrrds:
            continue
        pelvis_nrrds.sort(key=lambda p: (0 if "1.5" in os.path.basename(p) else 1, -os.path.getsize(p)))
        if cnum not in case_ct_map or ("NRRD-CT" in d):
            case_ct_map[cnum] = pelvis_nrrds[0]

    print(f"Discovered {len(case_ct_map)} unique patients with 3D Pelvic CT volumes.", flush=True)

    ct_case_records = []
    all_reg_shaft_errs = []
    all_reg_tip_z_errs = []

    adjudication_nf_extras_total = 0
    adjudication_nf_extras_confirmed_by_ct = 0
    adjudication_lb_extras_total = 0
    adjudication_lb_extras_confirmed_by_ct = 0

    tip_disagreement_cases = []

    for idx_ct, (cnum, ct_path) in enumerate(sorted(case_ct_map.items())):
        t_c0 = time.perf_counter()
        vol_ct, sp_ct, ijk2ras_ct = read_nrrd_local(ct_path)[:3]
        K_ct, J_ct, I_ct = vol_ct.shape

        # Crop bottom thigh slices if K_ct > 120 to focus on the pelvic implant slab
        k_lim_ct = int(0.22 * K_ct) if K_ct > 120 else 0

        # Run Stage 0 + Stage 2 on Bright-Core Stylet, Dark-Core Air-Lumen, and Dual-Polarity Fusion
        vol_bright = make_ct_polarity_volume(vol_ct, mode="bright_stylet")
        vol_dark = make_ct_polarity_volume(vol_ct, mode="dark_air")
        vol_dual = make_ct_polarity_volume(vol_ct, mode="dual")

        # Fast Rust 3D radial symmetry + multi-slice graph tracker on all 3 polarity modes (~2.5s total per CT volume)
        trajs_bright = logic.autoDetectAllNeedles(MockVolumeNode(vol_bright, sp_ct, ijk2ras_ct), minConfidence=35.0, kLim=k_lim_ct, renderNeedles=False, script=True)
        trajs_dark = logic.autoDetectAllNeedles(MockVolumeNode(vol_dark, sp_ct, ijk2ras_ct), minConfidence=35.0, kLim=k_lim_ct, renderNeedles=False, script=True)
        trajs_dual = logic.autoDetectAllNeedles(MockVolumeNode(vol_dual, sp_ct, ijk2ras_ct), minConfidence=32.0, kLim=k_lim_ct, renderNeedles=False, script=True)

        # Combine non-duplicate trajectories across dual + bright + dark polarities
        combined_ct = list(trajs_dual)
        for t_extra in trajs_bright + trajs_dark:
            p_ex = get_traj_pts(t_extra)
            if not any(traj_gt_dist(p_ex, get_traj_pts(s)) < 2.20 for s in combined_ct):
                combined_ct.append(t_extra)

        ct_elapsed = time.perf_counter() - t_c0

        in_bench = cnum in bench_idx_by_cnum
        rec_ct = {
            "case_num": cnum,
            "case_id": f"Case {cnum:03d}",
            "in_54_mri_benchmark": in_bench,
            "ct_filename": os.path.basename(ct_path),
            "shape_kji": [int(K_ct), int(J_ct), int(I_ct)],
            "spacing_ras_mm": [round(float(x), 3) for x in sp_ct],
            "runtime_sec": round(ct_elapsed, 2),
            "n_detected_bright_stylet_mode": len(trajs_bright),
            "n_detected_dark_air_lumen_mode": len(trajs_dark),
            "n_detected_dual_polarity_unified": len(trajs_dual),
            "n_detected_combined_ct_pool": len(combined_ct),
        }

        if in_bench:
            b_idx = bench_idx_by_cnum[cnum]
            m_info = meta_list[b_idx]
            gts_mri_all = get_gts(m_info["gts_all"])
            gts_mri_cy = get_gts(m_info["gts_long"])
            lb_gts = [np.asarray(p, dtype=np.float64)[np.argsort(np.asarray(p, dtype=np.float64)[:, 2])] for p in m_info.get("lb_gts", [])]
            nf_mri_preds = nf_oof_all_50[b_idx]

            ref_mri_bundle = gts_mri_all if len(gts_mri_all) >= 5 else [get_traj_pts(p) for p in nf_mri_preds]
            aligned_ct, R_opt, t_opt, reg_loss = register_ct_bundle_to_mri_bundle(combined_ct, ref_mri_bundle)

            # Also register bright-only and dark-only to compare single-polarity vs dual-polarity recall
            aligned_bright, _, _, _ = register_ct_bundle_to_mri_bundle(trajs_bright, ref_mri_bundle)
            aligned_dark, _, _, _ = register_ct_bundle_to_mri_bundle(trajs_dark, ref_mri_bundle)

            def match_bundle_to_gt(aligned_b):
                cost_b = np.full((len(gts_mri_all), len(aligned_b)), 999.0)
                tip_b = np.full((len(gts_mri_all), len(aligned_b)), 999.0)
                for i_g, g in enumerate(gts_mri_all):
                    for j_c, c in enumerate(aligned_b):
                        pm = compute_unified_pair_metrics(c["ras"], g)
                        if pm is not None:
                            cost_b[i_g, j_c] = pm["mean_shaft_xy"]
                            tip_b[i_g, j_c] = pm["tip_z"]
                ri_b, ci_b = linear_sum_assignment(cost_b)
                m35_b = [(r, c) for r, c in zip(ri_b, ci_b) if cost_b[r, c] <= 3.5]
                m50_b = [(r, c) for r, c in zip(ri_b, ci_b) if cost_b[r, c] <= 5.0]
                return m35_b, m50_b, cost_b, tip_b

            m_35, m_50, cost, tip_z_mat = match_bundle_to_gt(aligned_ct)
            m_35_br, m_50_br, _, _ = match_bundle_to_gt(aligned_bright)
            m_35_dk, m_50_dk, _, _ = match_bundle_to_gt(aligned_dark)

            sh_errs = [float(cost[r, c]) for r, c in m_50]
            tz_errs = [float(tip_z_mat[r, c]) for r, c in m_50]
            all_reg_shaft_errs.extend(sh_errs)
            all_reg_tip_z_errs.extend(tz_errs)

            rec_ct["n_mri_gt_all"] = len(gts_mri_all)
            rec_ct["n_matched_le_3p5mm"] = len(m_35)
            rec_ct["n_matched_le_5p0mm"] = len(m_50)
            rec_ct["n_matched_bright_only_le_5p0mm"] = len(m_50_br)
            rec_ct["n_matched_dark_only_le_5p0mm"] = len(m_50_dk)
            rec_ct["recall_le_3p5mm_pct"] = round(100.0 * len(m_35) / max(1, len(gts_mri_all)), 2)
            rec_ct["recall_le_5p0mm_pct"] = round(100.0 * len(m_50) / max(1, len(gts_mri_all)), 2)
            rec_ct["recall_bright_only_le_5p0mm_pct"] = round(100.0 * len(m_50_br) / max(1, len(gts_mri_all)), 2)
            rec_ct["recall_dark_only_le_5p0mm_pct"] = round(100.0 * len(m_50_dk) / max(1, len(gts_mri_all)), 2)
            rec_ct["inlier_shaft_xy_median_mm"] = round(float(np.median(sh_errs)), 3) if sh_errs else None
            rec_ct["inlier_tip_z_median_mm"] = round(float(np.median(tz_errs)), 3) if tz_errs else None

            # ADJUDICATION 1: Check NF 2026 "extra" predictions on MRI (not matched to CY+HT gts_mri_all within 3.5mm)
            nf_pts_list = [get_traj_pts(p) for p in nf_mri_preds]
            cost_nf_gt = np.full((len(gts_mri_all), len(nf_pts_list)), 999.0)
            for i_g, g in enumerate(gts_mri_all):
                for j_p, p_nf in enumerate(nf_pts_list):
                    pm = compute_unified_pair_metrics(p_nf, g)
                    if pm is not None:
                        cost_nf_gt[i_g, j_p] = pm["mean_shaft_xy"]
            r_ng, c_ng = linear_sum_assignment(cost_nf_gt)
            matched_nf_indices = {c for r, c in zip(r_ng, c_ng) if cost_nf_gt[r, c] <= 3.5}

            for j_p, p_nf in enumerate(nf_pts_list):
                if j_p not in matched_nf_indices:
                    adjudication_nf_extras_total += 1
                    confirmed = False
                    for c_ct in aligned_ct:
                        pm_ct = compute_unified_pair_metrics(c_ct["ras"], p_nf)
                        if pm_ct is not None and pm_ct["mean_shaft_xy"] <= 4.0:
                            confirmed = True
                            break
                    if confirmed:
                        adjudication_nf_extras_confirmed_by_ct += 1

            # ADJUDICATION 2: Check Rater 2 (LB/AMe) "extra" traces not in CY+HT
            for g_ex in m_info.get("gts_extra_lb", []):
                p_ex = get_traj_pts(g_ex)
                adjudication_lb_extras_total += 1
                confirmed = False
                for c_ct in aligned_ct:
                    pm_ct = compute_unified_pair_metrics(c_ct["ras"], p_ex)
                    if pm_ct is not None and pm_ct["mean_shaft_xy"] <= 4.0:
                        confirmed = True
                        break
                if confirmed:
                    adjudication_lb_extras_confirmed_by_ct += 1

            # ADJUDICATION 3: When CY and LB/AMe disagree on a catheter's cranial tip Z by >= 5.0 mm
            if len(lb_gts) > 0 and len(gts_mri_cy) > 0:
                cost_cy_lb = np.full((len(gts_mri_cy), len(lb_gts)), 999.0)
                for i_c, g_c in enumerate(gts_mri_cy):
                    for j_l, g_l in enumerate(lb_gts):
                        pm = compute_unified_pair_metrics(g_l, g_c)
                        if pm is not None and pm["mean_shaft_xy"] <= 3.5:
                            cost_cy_lb[i_c, j_l] = pm["mean_shaft_xy"]
                r_cl, c_cl = linear_sum_assignment(cost_cy_lb)
                for i_c, j_l in zip(r_cl, c_cl):
                    if cost_cy_lb[i_c, j_l] <= 3.5:
                        g_c = gts_mri_cy[i_c]
                        g_l = lb_gts[j_l]
                        dz_raters = abs(float(g_c[-1, 2] - g_l[-1, 2]))
                        if dz_raters >= 5.0:
                            best_nf = None
                            best_d_nf = 3.5
                            for p_nf in nf_pts_list:
                                pm_n = compute_unified_pair_metrics(p_nf, g_c)
                                if pm_n is not None and pm_n["mean_shaft_xy"] < best_d_nf:
                                    best_d_nf = pm_n["mean_shaft_xy"]
                                    best_nf = p_nf
                            best_ct = None
                            best_d_ct = 4.0
                            for c_ct in aligned_ct:
                                pm_ct = compute_unified_pair_metrics(c_ct["ras"], g_c)
                                if pm_ct is not None and pm_ct["mean_shaft_xy"] < best_d_ct:
                                    best_d_ct = pm_ct["mean_shaft_xy"]
                                    best_ct = c_ct["ras"]
                            if best_nf is not None and best_ct is not None:
                                z_ct = float(best_ct[-1, 2])
                                z_nf = float(best_nf[-1, 2])
                                z_cy = float(g_c[-1, 2])
                                z_lb = float(g_l[-1, 2])
                                z_cons = 0.5 * (z_cy + z_lb)
                                tip_disagreement_cases.append({
                                    "case_id": f"Case {cnum:03d}",
                                    "dz_cy_vs_lb_mm": round(dz_raters, 2),
                                    "err_nf_vs_ct_mm": round(abs(z_nf - z_ct), 2),
                                    "err_cy_vs_ct_mm": round(abs(z_cy - z_ct), 2),
                                    "err_lb_vs_ct_mm": round(abs(z_lb - z_ct), 2),
                                    "err_consensus_vs_ct_mm": round(abs(z_cons - z_ct), 2),
                                })

            print(
                f"  [{idx_ct+1:02d}/40] Case {cnum:03d} | CT_pool={len(combined_ct):2d} (Bright={len(trajs_bright):2d}, Dark={len(trajs_dark):2d}) | "
                f"MRI_GT={len(gts_mri_all):2d} -> Matched(<=3.5/5.0mm)={len(m_35):2d}/{len(m_50):2d} ({rec_ct['recall_le_5p0mm_pct']:5.1f}%) | "
                f"ShaftMed={rec_ct['inlier_shaft_xy_median_mm']}mm | {ct_elapsed:.1f}s",
                flush=True,
            )
        else:
            print(
                f"  [{idx_ct+1:02d}/40] Case {cnum:03d} (non-bench) | CT_pool={len(combined_ct):2d} (Bright={len(trajs_bright):2d}, Dark={len(trajs_dark):2d}) | {ct_elapsed:.1f}s",
                flush=True,
            )

        ct_case_records.append(rec_ct)

    bench_ct_recs = [r for r in ct_case_records if r["in_54_mri_benchmark"]]
    tot_gt_ct = sum(r["n_mri_gt_all"] for r in bench_ct_recs)
    tot_m35 = sum(r["n_matched_le_3p5mm"] for r in bench_ct_recs)
    tot_m50 = sum(r["n_matched_le_5p0mm"] for r in bench_ct_recs)
    tot_m50_br = sum(r["n_matched_bright_only_le_5p0mm"] for r in bench_ct_recs)
    tot_m50_dk = sum(r["n_matched_dark_only_le_5p0mm"] for r in bench_ct_recs)

    part4b_summary = {
        "total_unique_ct_patients": len(ct_case_records),
        "paired_with_54_mri_benchmark_patients": len(bench_ct_recs),
        "non_benchmark_ct_patients": len(ct_case_records) - len(bench_ct_recs),
        "total_mri_gt_catheters_across_38_paired_cases": tot_gt_ct,
        "polarity_mode_detections_mean_per_case": {
            "bright_core_metallic_stylet_mode": round(float(np.mean([r["n_detected_bright_stylet_mode"] for r in ct_case_records])), 2),
            "dark_core_air_lumen_mode": round(float(np.mean([r["n_detected_dark_air_lumen_mode"] for r in ct_case_records])), 2),
            "dual_polarity_unified_mode": round(float(np.mean([r["n_detected_dual_polarity_unified"] for r in ct_case_records])), 2),
            "combined_dual_polarity_pool": round(float(np.mean([r["n_detected_combined_ct_pool"] for r in ct_case_records])), 2),
        },
        "cross_modality_ct_to_mri_registration_metrics": {
            "bright_only_matched_le_5p0mm_recall_pct": round(100.0 * tot_m50_br / max(1, tot_gt_ct), 2),
            "dark_only_matched_le_5p0mm_recall_pct": round(100.0 * tot_m50_dk / max(1, tot_gt_ct), 2),
            "dual_polarity_matched_le_3p5mm_count": tot_m35,
            "dual_polarity_matched_le_3p5mm_recall_pct": round(100.0 * tot_m35 / max(1, tot_gt_ct), 2),
            "dual_polarity_matched_le_5p0mm_count": tot_m50,
            "dual_polarity_matched_le_5p0mm_recall_pct": round(100.0 * tot_m50 / max(1, tot_gt_ct), 2),
            "inlier_shaft_xy_mean_mm": round(float(np.mean(all_reg_shaft_errs)), 3) if all_reg_shaft_errs else 0.0,
            "inlier_shaft_xy_median_mm": round(float(np.median(all_reg_shaft_errs)), 3) if all_reg_shaft_errs else 0.0,
            "inlier_shaft_xy_iqr_mm": [round(float(np.percentile(all_reg_shaft_errs, 25)), 3), round(float(np.percentile(all_reg_shaft_errs, 75)), 3)] if all_reg_shaft_errs else [0.0, 0.0],
            "inlier_tip_z_mean_mm": round(float(np.mean(all_reg_tip_z_errs)), 3) if all_reg_tip_z_errs else 0.0,
            "inlier_tip_z_median_mm": round(float(np.median(all_reg_tip_z_errs)), 3) if all_reg_tip_z_errs else 0.0,
            "inlier_tip_z_iqr_mm": [round(float(np.percentile(all_reg_tip_z_errs, 25)), 3), round(float(np.percentile(all_reg_tip_z_errs, 75)), 3)] if all_reg_tip_z_errs else [0.0, 0.0],
        },
        "cross_modality_ct_adjudication": {
            "nf2026_unmatched_extra_mri_detections_total": adjudication_nf_extras_total,
            "nf2026_unmatched_extra_confirmed_by_ct_count": adjudication_nf_extras_confirmed_by_ct,
            "nf2026_unmatched_extra_confirmed_by_ct_pct": round(100.0 * adjudication_nf_extras_confirmed_by_ct / max(1, adjudication_nf_extras_total), 2),
            "lb_extra_mri_traces_total": adjudication_lb_extras_total,
            "lb_extra_confirmed_by_ct_count": adjudication_lb_extras_confirmed_by_ct,
            "lb_extra_confirmed_by_ct_pct": round(100.0 * adjudication_lb_extras_confirmed_by_ct / max(1, adjudication_lb_extras_total), 2),
            "tip_disagreements_ge_5mm_adjudicated_count": len(tip_disagreement_cases),
            "tip_disagreement_nf_vs_ct_median_mm": round(float(np.median([x["err_nf_vs_ct_mm"] for x in tip_disagreement_cases])), 3) if tip_disagreement_cases else None,
            "tip_disagreement_cy_vs_ct_median_mm": round(float(np.median([x["err_cy_vs_ct_mm"] for x in tip_disagreement_cases])), 3) if tip_disagreement_cases else None,
            "tip_disagreement_lb_vs_ct_median_mm": round(float(np.median([x["err_lb_vs_ct_mm"] for x in tip_disagreement_cases])), 3) if tip_disagreement_cases else None,
            "tip_disagreement_consensus_vs_ct_median_mm": round(float(np.median([x["err_consensus_vs_ct_mm"] for x in tip_disagreement_cases])), 3) if tip_disagreement_cases else None,
            "nf_closer_to_ct_than_single_rater_pct": round(100.0 * float(np.mean([
                x["err_nf_vs_ct_mm"] <= max(x["err_cy_vs_ct_mm"], x["err_lb_vs_ct_mm"]) for x in tip_disagreement_cases
            ])), 2) if tip_disagreement_cases else None,
        },
        "per_case_ct_results": ct_case_records,
    }

    print("\n=== PART 4B PELVIC CT SUMMARY ===", flush=True)
    print(json.dumps({
        "registration_metrics": part4b_summary["cross_modality_ct_to_mri_registration_metrics"],
        "adjudication": part4b_summary["cross_modality_ct_adjudication"],
    }, indent=2), flush=True)

    full_report = {
        "experiment": "Experiment 4: 4D Intra-Procedural Serial Insertion & Deflection Tracking (Cases 68-71) & Paired Post-Implant Pelvic CT (40 NRRD-CT Volumes) Cross-Modality Evaluation",
        "date": "2026-10-23",
        "script": "vault/paper/benchmarks_clean/run_exp4_4d_insertion_and_ct.py",
        "elapsed_sec": round(time.time() - t_start, 2),
        "part4a_4d_intraprocedural_tracking": part4a_summary,
        "part4b_paired_pelvic_ct_cross_modality": part4b_summary,
    }

    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[Done] Saved full Experiment 4 JSON report to {OUT_JSON_PATH} in {time.time() - t_start:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
