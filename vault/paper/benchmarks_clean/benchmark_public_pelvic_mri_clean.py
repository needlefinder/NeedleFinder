#!/usr/bin/env python3
"""
Experiment 2 (Clean Rerun, Zero-Nibabel Pure-NumPy NIfTI Reader):
Public Pelvic MRI Zero-Needle Specificity Benchmark (`AUD-501`)
Dataset: Medical Segmentation Decathlon (MSD) Task05_Prostate
(48 clinical 3T axial T2W TSE pelvic MRI volumes: 32 imagesTr + 16 imagesTs, 0 implanted needles).

Evaluates across all 48 volumes:
  1. Raw Stage 1 `NeedleUNet3D` + Tangent Tube Tracking (without Stage 0 physics gate or Stage 2B Graph Transformer)
  2. Candidate Pool before Stage 2B Graph Transformer (Stage 0 Physics + Stage 1 Tubes)
  3. Stage 1 `NeedleUNet3D` + Tangent Tubes gated by Stage 0 Rust 3D Radial Symmetry (`extract_nn_labelmap_tubes`)
  4. Unified `NeedleFinder 2026` (`autoDetectAllNeedles`) across confidence thresholds `tau in {25, 35, 45, 50, 55, 60}%`,
     reporting both:
       (a) Ungated `autoDetectAllNeedles` (where Stage 0 seeds on relative 30th-percentile pelvic vessels/folds)
       (b) Stage-0/Stage-1 Dark-Core Corroborated `NeedleFinder 2026` (where trajectories must be corroborated by the
           Stage 0 radial symmetry + Stage 1 dark-core susceptibility gate)
     and also evaluating the newly trained Clean SSL + Multi-Rater Stage 1 `NeedleUNet3D` ensemble.
"""

import glob
import gzip
import json
import math
import os
import struct
import sys
import time
import types
import numpy as np
import torch
from scipy.ndimage import maximum_filter

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

sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean")

from NeedleFinderLib.TrackingAlgorithms import TrackingAlgorithmsMixin
import NeedleFinderLib.TrackingAlgorithms as ta_mod
from NeedleFinderLib.RustCore import rust_compute_3d_sym_response
from NeedleFinderLib.NeuralNeedleNet import (
    run_nn_3d_segmentation,
    extract_nn_labelmap_tubes,
)
from train_and_eval_unet3d import NeedleUNet3D
from train_nested_stage1 import infer_single_view_3d, raw_out_to_probs

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


def read_nifti_gz_pure_numpy(nii_path: str):
    """Zero-dependency pure-Python NIfTI-1 (.nii.gz) reader using gzip + struct + numpy."""
    with gzip.open(nii_path, "rb") as f:
        raw = f.read()

    sizeof_hdr = struct.unpack("<i", raw[0:4])[0]
    endian = "<" if sizeof_hdr == 348 else ">"
    dims = struct.unpack(endian + "8h", raw[40:56])
    ndim = dims[0]
    nx, ny, nz = int(dims[1]), int(dims[2]), int(dims[3])
    nt = int(dims[4]) if ndim >= 4 and dims[4] > 0 else 1

    datatype, bitpix = struct.unpack(endian + "2h", raw[70:74])
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    scl_slope, scl_inter = struct.unpack(endian + "2f", raw[112:120])

    sform_code = struct.unpack(endian + "h", raw[254:256])[0]
    if sform_code > 0:
        srow_x = struct.unpack(endian + "4f", raw[280:296])
        srow_y = struct.unpack(endian + "4f", raw[296:312])
        srow_z = struct.unpack(endian + "4f", raw[312:328])
        ijk_to_ras = np.array([srow_x, srow_y, srow_z, [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        ijk_to_ras = np.diag([float(pixdim[1]), float(pixdim[2]), float(pixdim[3]), 1.0])

    dtype_map = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        512: np.uint16,
    }
    np_dtype = np.dtype(dtype_map[datatype]).newbyteorder(endian)
    count = nx * ny * nz * nt
    arr = np.frombuffer(raw, dtype=np_dtype, count=count, offset=vox_offset)
    arr = arr.reshape((nx, ny, nz, nt) if nt > 1 else (nx, ny, nz), order="F")
    if nt > 1:
        arr = arr[..., 0]
    arr = arr.astype(np.float64)
    if scl_slope != 0.0 and not np.isnan(scl_slope):
        arr = arr * float(scl_slope) + (float(scl_inter) if not np.isnan(scl_inter) else 0.0)

    vol_kji = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)), dtype=np.float64)
    sp_ras = (abs(float(pixdim[1])), abs(float(pixdim[2])), abs(float(pixdim[3])))
    return vol_kji, sp_ras, ijk_to_ras


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


def is_corroborated_by_sym_tube(traj_dict, sym_tubes, dist_thresh_mm: float = 3.5) -> bool:
    """Checks whether a predicted trajectory is corroborated by a Stage 0 + Stage 1 dark-core symmetry tube."""
    if not sym_tubes:
        return False
    raw_t = traj_dict.get("pts", traj_dict.get("ras", traj_dict.get("points")))
    pts_t = np.asarray(raw_t, dtype=np.float64)
    pts_t = pts_t[np.argsort(pts_t[:, 2])]
    for st in sym_tubes:
        raw_s = st.get("pts", st.get("ras", st.get("points"))) if isinstance(st, dict) else st
        pts_s = np.asarray(raw_s, dtype=np.float64)
        pts_s = pts_s[np.argsort(pts_s[:, 2])]
        z_lo = max(float(pts_t[0, 2]), float(pts_s[0, 2]))
        z_hi = min(float(pts_t[-1, 2]), float(pts_s[-1, 2]))
        if z_hi <= z_lo + 2.0:
            continue
        zs = np.linspace(z_lo, z_hi, 10)
        xt = np.interp(zs, pts_t[:, 2], pts_t[:, 0])
        yt = np.interp(zs, pts_t[:, 2], pts_t[:, 1])
        xs = np.interp(zs, pts_s[:, 2], pts_s[:, 0])
        ys = np.interp(zs, pts_s[:, 2], pts_s[:, 1])
        if float(np.mean(np.hypot(xt - xs, yt - ys))) <= dist_thresh_mm:
            return True
    return False


def compute_fp_distribution_stats(counts_list, seed: int = 42, n_boot: int = 1000):
    arr = np.asarray(counts_list, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)
    boot_means = [float(np.mean(arr[rng.integers(0, n, size=n)])) for _ in range(n_boot)]
    z_pts = int(np.sum(arr == 0))
    le1_pts = int(np.sum(arr <= 1))
    return {
        "total_fp_tracks": int(np.sum(arr)),
        "mean_fp_per_volume": round(float(np.mean(arr)), 3),
        "mean_fp_ci95": [round(float(np.percentile(boot_means, 2.5)), 3), round(float(np.percentile(boot_means, 97.5)), 3)],
        "sd_fp_per_volume": round(float(np.std(arr, ddof=1)), 3),
        "pop_sd_fp_per_volume": round(float(np.std(arr, ddof=0)), 3),
        "median_fp_per_volume": round(float(np.median(arr)), 3),
        "iqr_fp_per_volume": [round(float(np.percentile(arr, 25)), 3), round(float(np.percentile(arr, 75)), 3)],
        "max_fp_per_volume": int(np.max(arr)),
        "zero_fp_patients": z_pts,
        "patient_level_specificity_pct": round(100.0 * z_pts / n, 2),
        "le_1_fp_patients": le1_pts,
        "le_1_fp_specificity_pct": round(100.0 * le1_pts / n, 2),
    }


def main():
    tr_files = sorted([p for p in glob.glob("/tmp/msd_prostate/Task05_Prostate/imagesTr/*.nii.gz") if "/._" not in p])
    ts_files = sorted([p for p in glob.glob("/tmp/msd_prostate/Task05_Prostate/imagesTs/*.nii.gz") if "/._" not in p])
    all_files = [("imagesTr", p) for p in tr_files] + [("imagesTs", p) for p in ts_files]
    print(f"Loaded {len(all_files)} clinical 3T pelvic T2W MRI volumes from MSD Task05_Prostate (pure-NumPy NIfTI reader).", flush=True)

    # Load Clean Stage 1 SSL + Multi-Rater ensemble (5 outer fold m=0 models) on CPU / secondary GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clean_s1_models = []
    for f_out in range(5):
        ckpt = f"/home/guillaume/github/needlefinder/nn_models_clean/unet3d_outer{f_out}_inner0.pt"
        if os.path.exists(ckpt):
            m = NeedleUNet3D(6, 24).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            m.eval()
            clean_s1_models.append(m)

    logic = StandaloneEvalLogic()
    taus = [25.0, 35.0, 45.0, 50.0, 55.0, 60.0]

    per_case_results = []
    for idx, (split_name, nii_path) in enumerate(all_files):
        case_id = os.path.basename(nii_path).replace(".nii.gz", "")
        vol, sp_ras, ijk_to_ras = read_nifti_gz_pure_numpy(nii_path)
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

        res1 = rust_compute_3d_sym_response(vol_work, sp_i, sp_j, i0, i1, j0, j1, k0, K, body_thresh)
        resp_full, cs_full, rsym_full = res1
        resp1 = resp_full[k0:K, j0:j1, i0:i1].copy()
        cs1 = cs_full[k0:K, j0:j1, i0:i1]
        rsym1 = rsym_full[k0:K, j0:j1, i0:i1]

        nn_preds = run_nn_3d_segmentation(vol_work, sp_i, sp_j, sp_k, i0, i1, j0, j1, k0, K, resp1, cs1, rsym1)

        # 1. Raw Stage 1 U-Net + Tangent Tubes (without physics filter or DenseGraphTransformer)
        raw_unet_no_phys = trace_raw_unet_tubes_no_physics(nn_preds, k0, i0, j0, sp_i, sp_j, sp_k, ijk_to_ras, ras_to_ijk)

        # 2. Stage 1 U-Net + Tangent Tubes with Stage 0 symmetry gate (extract_nn_labelmap_tubes directly)
        s1_tubes_sym = extract_nn_labelmap_tubes(
            nn_preds, resp1, k0, i0, j0, sp_i, sp_j, sp_k,
            ijk_to_ras, ras_to_ijk, [], include_obturator=True,
        )

        # Also evaluate Clean Stage 1 (SSL + Multi-Rater) with Stage 0 symmetry gate on the exact same ROI
        clean_s1_sym_count = len(s1_tubes_sym)
        if clean_s1_models:
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
            outs = [infer_single_view_3d(m, x_ssl, device)[:, :K_c, :H_c, :W_c] for m in clean_s1_models]
            probs_c = raw_out_to_probs(torch.stack(outs, dim=0).mean(dim=0), temp_pos=1.0)
            nn_clean = {
                "p_mask": probs_c[0].numpy().astype(np.float32),
                "p_peak": probs_c[1].numpy().astype(np.float32),
                "vx": probs_c[2].numpy().astype(np.float32),
                "vy": probs_c[3].numpy().astype(np.float16).astype(np.float32),
                "p_obt": probs_c[4].numpy().astype(np.float32),
            }
            s1_clean_tubes = extract_nn_labelmap_tubes(
                nn_clean, resp1, k0, i0, j0, sp_i, sp_j, sp_k,
                ijk_to_ras, ras_to_ijk, [], include_obturator=True,
            )
            clean_s1_sym_count = len(s1_clean_tubes)

        # 3. Full Production Pipeline at minimum threshold tau=25%
        vnode = MockVolumeNode(vol, sp_ras, ijk_to_ras)
        t_pipe = time.perf_counter()
        trajs_unified_25 = logic.autoDetectAllNeedles(
            vnode, minConfidence=25.0, kLim=0, renderNeedles=False, script=False
        )
        pipe_time = time.perf_counter() - t_pipe

        trajs_phys_raw = logic.autoDetectAllNeedles(
            vnode, minConfidence=25.0, kLim=0, renderNeedles=False, script=True
        )
        cand_pool_before_gt = len(trajs_phys_raw) + len(s1_tubes_sym)

        unified_counts = {}
        unified_sym_gated_counts = {}
        phys_counts = {}
        for tau in taus:
            k_str = str(int(tau))
            u_tau = [t for t in trajs_unified_25 if float(t.get("confidence", 0.0)) >= tau]
            unified_counts[k_str] = len(u_tau)
            unified_sym_gated_counts[k_str] = sum(1 for t in u_tau if is_corroborated_by_sym_tube(t, s1_tubes_sym))
            phys_counts[k_str] = sum(1 for t in trajs_phys_raw if float(t.get("confidence", 0.0)) >= tau)

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
            "clean_ssl_stage1_sym_tubes_only": int(clean_s1_sym_count),
            "candidate_pool_before_graph_transformer": cand_pool_before_gt,
            "pure_physics_hybridnet_by_tau": phys_counts,
            "unified_needlefinder_2026_by_tau": unified_counts,
            "unified_needlefinder_2026_sym_gated_by_tau": unified_sym_gated_counts,
            "detected_confs_unified": [round(float(t.get("confidence", 0.0)), 2) for t in trajs_unified_25],
        }
        per_case_results.append(rec)
        print(
            f"[{idx+1:02d}/48] {case_id} ({K}x{J}x{I}) | RawUNet={len(raw_unet_no_phys)} | "
            f"CandPool={cand_pool_before_gt} | SymGate={len(s1_tubes_sym)} (CleanSSL={clean_s1_sym_count}) | "
            f"Unified(50%)={unified_counts['50']} (Gated={unified_sym_gated_counts['50']}) | {pipe_time:.2f}s",
            flush=True,
        )

    n_cases = len(per_case_results)
    summary = {
        "benchmark": "Public Clinical Pelvic 3T T2W MRI Zero-Needle Specificity Benchmark (MSD Task05_Prostate)",
        "date": "2026-10-23",
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
                **compute_fp_distribution_stats([r["raw_unet_tubes_no_physics_or_gt"] for r in per_case_results]),
            },
            "2_candidate_pool_before_graph_transformer": {
                "description": "Stage 0 Physics + Stage 1 NeedleUNet3D + Tangent Tubes Candidate Pool (without Stage 2B DenseGraphTransformer pruning)",
                **compute_fp_distribution_stats([r["candidate_pool_before_graph_transformer"] for r in per_case_results]),
            },
            "3_stage1_unet_with_stage0_sym_gate_only": {
                "description": "Stage 1 NeedleUNet3D + Tangent Tubes gated by Stage 0 Rust 3D Radial Symmetry (without DenseGraphTransformer)",
                **compute_fp_distribution_stats([r["stage1_unet_sym_tubes_only"] for r in per_case_results]),
            },
            "3b_clean_ssl_stage1_with_stage0_sym_gate_only": {
                "description": "Clean SSL + Multi-Rater Stage 1 NeedleUNet3D + Tangent Tubes gated by Stage 0 Rust 3D Radial Symmetry",
                **compute_fp_distribution_stats([r["clean_ssl_stage1_sym_tubes_only"] for r in per_case_results]),
            },
            "4_unified_needlefinder_2026_by_threshold": {},
            "5_unified_needlefinder_2026_sym_gated_by_threshold": {},
        },
        "mean_runtime_sec_per_volume": round(float(np.mean([r["runtime_sec"] for r in per_case_results])), 3),
        "median_runtime_sec_per_volume": round(float(np.median([r["runtime_sec"] for r in per_case_results])), 3),
        "per_case_results": per_case_results,
    }

    for tau in taus:
        k_str = str(int(tau))
        cnts_u = [r["unified_needlefinder_2026_by_tau"][k_str] for r in per_case_results]
        cnts_g = [r["unified_needlefinder_2026_sym_gated_by_tau"][k_str] for r in per_case_results]
        summary["methods_compared"]["4_unified_needlefinder_2026_by_threshold"][f"tau_{k_str}pct"] = {
            "confidence_threshold_pct": tau,
            **compute_fp_distribution_stats(cnts_u),
        }
        summary["methods_compared"]["5_unified_needlefinder_2026_sym_gated_by_threshold"][f"tau_{k_str}pct"] = {
            "confidence_threshold_pct": tau,
            **compute_fp_distribution_stats(cnts_g),
        }

    out_paths = [
        "/home/guillaume/github/needlefinder/vault/sources/2026-10-23-public-pelvic-mri-specificity-clean.json",
        "/home/guillaume/github/needlefinder/vault/sources/2026-10-22-public-pelvic-mri-specificity-benchmark.json",
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(summary, f, indent=2)

    print("\n=== SPECIFICITY SUMMARY ===")
    print(json.dumps(summary["methods_compared"], indent=2))


if __name__ == "__main__":
    main()
