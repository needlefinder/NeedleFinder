#!/usr/bin/env python3
"""
Experiment 2: Public Pelvic MRI Zero-Needle Specificity Benchmark
Dataset: Medical Segmentation Decathlon (MSD) Task05_Prostate (48 clinical 3T axial T2W TSE pelvic MRI volumes: 32 imagesTr + 16 imagesTs).
All 48 volumes contain 0 brachytherapy needles (showing prostate, rectum, bladder, obturator internus, pelvic vasculature, and perineal tissue).

Evaluates:
1. Raw 3D U-Net () Voxel & Connected-Tube Detections (without physics prior or Graph Transformer)
2. Stage 1  + Tangent Tubes + Physics Candidate Pool (before  pruning)
3. Pure 3D CV +  ( physics tracker)
4. Unified  ()
   across confidence thresholds tau in {25%, 35%, 45%, 50%, 60%}.
"""

import glob
import json
import math
import os
import sys
import time
import types
import nibabel as nib
import numpy as np
from scipy.ndimage import maximum_filter

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
from NeedleFinderLib.RustCore import rust_compute_3d_sym_response
from NeedleFinderLib.NeuralNeedleNet import (
    run_nn_3d_segmentation,
    extract_nn_labelmap_tubes,
    run_mpnn_and_graph_transformer,
)

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
        return "MockVolumeNode_PublicMRI"


def trace_raw_unet_tubes_no_physics(nn_preds, k0, i0, j0, sp_i, sp_j, sp_k, ijk_to_ras, ras_to_ijk):
    """
    Baseline: Standard 3D U-Net + Tangent Tube Tracing WITHOUT Stage 0 symmetry gate (resp_roi > 2.5)
    or Stage 2B DenseGraphTransformer suppression.
    Traces tubes from local peaks of p_mask / p_peak / p_obt across >= 4 slices.
    """
    p_mask = nn_preds["p_mask"]
    p_peak = nn_preds["p_peak"]
    vx_map = nn_preds["vx"]
    vy_map = nn_preds["vy"]
    p_obt = nn_preds["p_obt"]
    K_sub, H, W = p_mask.shape
    rot_mat = ijk_to_ras[:3, :3]
    trans_vec = ijk_to_ras[:3, 3]

    work_peak = np.maximum(p_peak * (0.45 + 0.55 * p_mask), p_peak * p_obt * 1.15).copy()
    nms_px = max(1, int(round(1.35 / min(sp_i, sp_j))))
    lm = work_peak == maximum_filter(work_peak, size=(1, 2 * nms_px + 1, 2 * nms_px + 1))
    seed_mask = lm & (work_peak >= 0.30) & ((p_mask >= 0.35) | (p_obt >= 0.35))
    sk, sj, si = np.where(seed_mask)
    if len(sk) == 0:
        return []

    scores = work_peak[sk, sj, si]
    order_seeds = np.argsort(-scores)
    visited = np.zeros_like(work_peak, dtype=bool)
    search_r = max(2, int(round(2.4 / min(sp_i, sp_j))))
    ri_m = max(1, int(round(1.85 / sp_i)))
    rj_m = max(1, int(round(1.85 / sp_j)))
    yy_m, xx_m = np.mgrid[-rj_m:rj_m + 1, -ri_m:ri_m + 1]
    disk_m = ((xx_m * sp_i) ** 2 + (yy_m * sp_j) ** 2) <= (1.85 ** 2)

    raw_tubes = []
    min_slices = 4 if sp_k >= 3.0 else 5

    for idx_s in order_seeds[:350]:
        ks0, js0, is0 = int(sk[idx_s]), int(sj[idx_s]), int(si[idx_s])
        if visited[ks0, js0, is0]:
            continue
        pts = {ks0: (float(is0), float(js0), float(work_peak[ks0, js0, is0]))}
        for direction in (1, -1):
            curr_k = ks0
            curr_i, curr_j = float(is0), float(js0)
            missed = 0
            while True:
                next_k = curr_k + direction
                if not (0 <= next_k < K_sub):
                    break
                ci_int = int(np.clip(round(curr_i), 0, W - 1))
                cj_int = int(np.clip(round(curr_j), 0, H - 1))
                vx_loc = float(vx_map[curr_k, cj_int, ci_int])
                vy_loc = float(vy_map[curr_k, cj_int, ci_int])
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
                if weighted.size == 0 or float(np.max(weighted)) < 0.20:
                    missed += 1
                    if missed > 2:
                        break
                    curr_k = next_k
                    curr_i, curr_j = pred_i, pred_j
                    continue
                lj, li = np.unravel_index(int(np.argmax(weighted)), weighted.shape)
                best_j, best_i = j_lo + lj, i_lo + li
                raw_p = float(work_peak[next_k, best_j, best_i])
                if raw_p < 0.20:
                    missed += 1
                    if missed > 2:
                        break
                    curr_k = next_k
                    curr_i, curr_j = pred_i, pred_j
                    continue
                pts[next_k] = (float(best_i), float(best_j), raw_p)
                curr_k = next_k
                curr_i, curr_j = float(best_i), float(best_j)
                missed = 0
        if len(pts) < min_slices:
            continue
        ks = sorted(pts.keys())
        arr_pts = np.array([[pts[k][0] + i0, pts[k][1] + j0, k + k0, pts[k][2]] for k in ks], dtype=np.float64)
        ras_pts = (rot_mat @ arr_pts[:, :3].T).T + trans_vec
        order_z = np.argsort(ras_pts[:, 2])
        ras_pts = ras_pts[order_z]
        z_vals = ras_pts[:, 2]
        if float(z_vals[-1] - z_vals[0]) < 10.0:
            continue
        for k_v in ks:
            iv, jv = int(round(pts[k_v][0])), int(round(pts[k_v][1]))
            if 0 <= iv < W and 0 <= jv < H:
                j0_p, j1_p = max(0, jv - rj_m), min(H, jv + rj_m + 1)
                i0_p, i1_p = max(0, iv - ri_m), min(W, iv + ri_m + 1)
                d_sub = disk_m[(j0_p - (jv - rj_m)):(j1_p - (jv - rj_m)), (i0_p - (iv - ri_m)):(i0_p - (iv - ri_m))]
                visited[k_v, j0_p:j1_p, i0_p:i1_p][d_sub] = True
                work_peak[k_v, j0_p:j1_p, i0_p:i1_p][d_sub] = 0.0
        raw_tubes.append(ras_pts)
    return raw_tubes


def main():
    tr_files = sorted([p for p in glob.glob("/tmp/msd_prostate/Task05_Prostate/imagesTr/*.nii.gz") if "/._" not in p])
    ts_files = sorted([p for p in glob.glob("/tmp/msd_prostate/Task05_Prostate/imagesTs/*.nii.gz") if "/._" not in p])
    all_files = [("imagesTr", p) for p in tr_files] + [("imagesTs", p) for p in ts_files]
    print(f"Loaded {len(all_files)} clinical 3T pelvic T2W MRI volumes from MSD Task05_Prostate.")

    logic = StandaloneEvalLogic()
    taus = [25.0, 35.0, 45.0, 50.0, 60.0]

    per_case_results = []
    t_start_all = time.perf_counter()

    for idx, (split_name, nii_path) in enumerate(all_files):
        case_id = os.path.basename(nii_path).replace(".nii.gz", "")
        t0 = time.perf_counter()
        img = nib.load(nii_path)
        data = img.get_fdata()
        # Modality 0 is T2W MRI, shape in NIfTI is (I, J, K, C)
        t2_ijk = data[..., 0] if data.ndim == 4 else data
        # Convert to (K, J, I) for Slicer / NeedleFinder
        vol = np.ascontiguousarray(np.transpose(t2_ijk, (2, 1, 0)), dtype=np.float64)
        zooms = img.header.get_zooms()
        sp_ras = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
        ijk_to_ras = np.array(img.affine, dtype=np.float64)
        ras_to_ijk = np.linalg.inv(ijk_to_ras)

        K, J, I = vol.shape
        sp_i, sp_j, sp_k = sp_ras
        k0 = 0
        i0, i1 = int(0.16 * I), int(0.84 * I)
        j0, j1 = int(0.16 * J), int(0.84 * J)

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

        # Stage 0 + Stage 1 3D U-Net + Tangent Tubes
        t_s1 = time.perf_counter()
        res1 = rust_compute_3d_sym_response(vol_work, sp_i, sp_j, i0, i1, j0, j1, k0, K, body_thresh)
        resp_full, cs_full, rsym_full = res1
        resp1 = resp_full[k0:K, j0:j1, i0:i1].copy()
        cs1 = cs_full[k0:K, j0:j1, i0:i1]
        rsym1 = rsym_full[k0:K, j0:j1, i0:i1]

        nn_preds = run_nn_3d_segmentation(vol_work, sp_i, sp_j, sp_k, i0, i1, j0, j1, k0, K, resp1, cs1, rsym1)
        s1_time = time.perf_counter() - t_s1

        # 1. Raw Stage 1 U-Net + Tangent Tubes (without physics filter or DenseGraphTransformer)
        raw_unet_no_phys = trace_raw_unet_tubes_no_physics(nn_preds, k0, i0, j0, sp_i, sp_j, sp_k, ijk_to_ras, ras_to_ijk)
        # 2. Stage 1 U-Net + Tangent Tubes with Stage 0 symmetry gate (extract_nn_labelmap_tubes directly)
        s1_tubes_sym = extract_nn_labelmap_tubes(
            nn_preds, resp1, k0, i0, j0, sp_i, sp_j, sp_k,
            ijk_to_ras, ras_to_ijk, [], include_obturator=True,
        )

        # 3. Full Production Pipeline () at minimum threshold tau=25%
        vnode = MockVolumeNode(vol, sp_ras, ijk_to_ras)
        t_pipe = time.perf_counter()
        trajs_unified_25 = logic.autoDetectAllNeedles(
            vnode, minConfidence=25.0, kLim=0, renderNeedles=False, script=False
        )
        pipe_time = time.perf_counter() - t_pipe

        # Also check Stage 1 + Physics Candidate Pool before DenseGraphTransformer:
        # We can see how many candidate trajectories existed before DenseGraphTransformer by running script=True at minConfidence=25.0 + s1_tubes_sym
        trajs_phys_raw = logic.autoDetectAllNeedles(
            vnode, minConfidence=25.0, kLim=0, renderNeedles=False, script=True
        )
        cand_pool_before_gt = len(trajs_phys_raw) + len(s1_tubes_sym)

        # Count detections for Unified NeedleFinder 2026 across thresholds tau in {25, 35, 45, 50, 60}
        unified_counts = {}
        for tau in taus:
            unified_counts[str(int(tau))] = sum(1 for t in trajs_unified_25 if float(t.get("confidence", 0.0)) >= tau)

        phys_counts = {}
        for tau in taus:
            phys_counts[str(int(tau))] = sum(1 for t in trajs_phys_raw if float(t.get("confidence", 0.0)) >= tau)

        vox_fp_frac = float(np.mean(nn_preds["p_mask"] >= 0.50))
        vox_obt_frac = float(np.mean(nn_preds["p_obt"] >= 0.50))

        rec = {
            "case_id": case_id,
            "split": split_name,
            "shape_kji": [int(K), int(J), int(I)],
            "spacing_ras_mm": [round(sp_i, 4), round(sp_j, 4), round(sp_k, 4)],
            "runtime_sec": round(pipe_time, 4),
            "vox_p_mask_ge_50_frac": round(vox_fp_frac, 6),
            "vox_p_obt_ge_50_frac": round(vox_obt_frac, 6),
            "raw_unet_tubes_no_physics_or_gt": len(raw_unet_no_phys),
            "stage1_unet_sym_tubes_only": len(s1_tubes_sym),
            "candidate_pool_before_graph_transformer": cand_pool_before_gt,
            "pure_physics_hybridnet_by_tau": phys_counts,
            "unified_needlefinder_2026_by_tau": unified_counts,
            "detected_confs_unified": [round(float(t.get("confidence", 0.0)), 2) for t in trajs_unified_25],
        }
        per_case_results.append(rec)
        print(
            f"[{idx+1:02d}/48] {case_id} ({K}x{J}x{I}, {sp_i:.2f}x{sp_j:.2f}x{sp_k:.1f}mm) | "
            f"RawUNet={len(raw_unet_no_phys)} | CandPool={cand_pool_before_gt} | "
            f"Unified(35/45/50/60%)={unified_counts['35']}/{unified_counts['45']}/{unified_counts['50']}/{unified_counts['60']} | "
            f"{pipe_time:.2f}s"
        )

    n_cases = len(per_case_results)
    summary = {
        "benchmark": "Public Clinical Pelvic 3T T2W MRI Zero-Needle Specificity Benchmark (MSD Task05_Prostate)",
        "date": "2026-10-22",
        "dataset": {
            "name": "Medical Segmentation Decathlon (MSD) Task05_Prostate",
            "source": "s3://msd-for-monai/Task05_Prostate.tar (CC-BY-SA 4.0, Radboudumc Nijmegen)",
            "modality": "3T Axial T2-Weighted TSE Pelvic MRI (0 brachytherapy needles)",
            "total_volumes": n_cases,
            "imagesTr_volumes": len(tr_files),
            "imagesTs_volumes": len(ts_files),
            "in_plane_resolution_mm": "0.60 - 0.75 mm",
            "slice_thickness_mm": "3.0 - 4.0 mm",
        },
        "methods_compared": {
            "1_raw_stage1_unet_tangent_tubes_no_physics_or_gt": {
                "description": "Raw Stage 1 NeedleUNet3D + Tangent Tube Tracking (without Stage 0 symmetry gate or Stage 2B DenseGraphTransformer)",
                "total_fp_tracks": int(sum(r["raw_unet_tubes_no_physics_or_gt"] for r in per_case_results)),
                "mean_fp_per_volume": round(float(np.mean([r["raw_unet_tubes_no_physics_or_gt"] for r in per_case_results])), 3),
                "median_fp_per_volume": round(float(np.median([r["raw_unet_tubes_no_physics_or_gt"] for r in per_case_results])), 3),
                "max_fp_per_volume": int(np.max([r["raw_unet_tubes_no_physics_or_gt"] for r in per_case_results])),
                "zero_fp_patients": int(sum(1 for r in per_case_results if r["raw_unet_tubes_no_physics_or_gt"] == 0)),
                "patient_level_specificity_pct": round(100.0 * sum(1 for r in per_case_results if r["raw_unet_tubes_no_physics_or_gt"] == 0) / n_cases, 2),
            },
            "2_candidate_pool_before_graph_transformer": {
                "description": "Stage 0 Physics + Stage 1 NeedleUNet3D + Tangent Tubes Candidate Pool (without Stage 2B DenseGraphTransformer pruning)",
                "total_fp_tracks": int(sum(r["candidate_pool_before_graph_transformer"] for r in per_case_results)),
                "mean_fp_per_volume": round(float(np.mean([r["candidate_pool_before_graph_transformer"] for r in per_case_results])), 3),
                "median_fp_per_volume": round(float(np.median([r["candidate_pool_before_graph_transformer"] for r in per_case_results])), 3),
                "max_fp_per_volume": int(np.max([r["candidate_pool_before_graph_transformer"] for r in per_case_results])),
                "zero_fp_patients": int(sum(1 for r in per_case_results if r["candidate_pool_before_graph_transformer"] == 0)),
                "patient_level_specificity_pct": round(100.0 * sum(1 for r in per_case_results if r["candidate_pool_before_graph_transformer"] == 0) / n_cases, 2),
            },
            "3_stage1_unet_with_stage0_sym_gate_only": {
                "description": "Stage 1 NeedleUNet3D + Tangent Tubes gated by Stage 0 Rust 3D Radial Symmetry (without DenseGraphTransformer)",
                "total_fp_tracks": int(sum(r["stage1_unet_sym_tubes_only"] for r in per_case_results)),
                "mean_fp_per_volume": round(float(np.mean([r["stage1_unet_sym_tubes_only"] for r in per_case_results])), 3),
                "median_fp_per_volume": round(float(np.median([r["stage1_unet_sym_tubes_only"] for r in per_case_results])), 3),
                "max_fp_per_volume": int(np.max([r["stage1_unet_sym_tubes_only"] for r in per_case_results])),
                "zero_fp_patients": int(sum(1 for r in per_case_results if r["stage1_unet_sym_tubes_only"] == 0)),
                "patient_level_specificity_pct": round(100.0 * sum(1 for r in per_case_results if r["stage1_unet_sym_tubes_only"] == 0) / n_cases, 2),
            },
            "4_unified_needlefinder_2026_by_threshold": {},
        },
        "mean_runtime_sec_per_volume": round(float(np.mean([r["runtime_sec"] for r in per_case_results])), 3),
        "median_runtime_sec_per_volume": round(float(np.median([r["runtime_sec"] for r in per_case_results])), 3),
        "per_case_results": per_case_results,
    }

    for tau in taus:
        k_str = str(int(tau))
        cnts = [r["unified_needlefinder_2026_by_tau"][k_str] for r in per_case_results]
        z_pts = sum(1 for c in cnts if c == 0)
        summary["methods_compared"]["4_unified_needlefinder_2026_by_threshold"][f"tau_{k_str}pct"] = {
            "confidence_threshold_pct": tau,
            "total_fp_tracks": int(sum(cnts)),
            "mean_fp_per_volume": round(float(np.mean(cnts)), 3),
            "sd_fp_per_volume": round(float(np.std(cnts)), 3),
            "median_fp_per_volume": round(float(np.median(cnts)), 3),
            "max_fp_per_volume": int(np.max(cnts)),
            "zero_fp_patients": int(z_pts),
            "patient_level_specificity_pct": round(100.0 * z_pts / n_cases, 2),
        }

    out_path = "/tmp/2026-10-22-public-pelvic-mri-specificity-benchmark.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary["methods_compared"], indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
