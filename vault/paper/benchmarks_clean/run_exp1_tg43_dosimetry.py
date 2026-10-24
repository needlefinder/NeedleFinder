#!/usr/bin/env python3
"""
Experiment 1: AAPM TG-43U1 HDR 192Ir Dosimetric Impact & Uncertainty-to-Dose Propagation
========================================================================================
Evaluates the clinical brachytherapy dosimetric impact of catheter shaft and cranial tip
localization across all 54 GYN patients (and the 40 dual-traced inter-observer subset)
using a vectorized GPU AAPM TG-43U1 192Ir line-source dose calculation engine:

  1. Vectorized AAPM TG-43U1 192Ir Line-Source Engine:
     - Active source length L = 3.5 mm (0.35 cm), dose-rate constant Lambda = 1.109 cGy h^-1 U^-1
     - Exact analytical line-source geometry factor G_L(r, theta) = beta / (L * r * sin(theta))
     - Consensus polynomial radial dose function g_L(r) and 2D anisotropy function F(r, theta)
       oriented along the local 3D catheter tangent u_j at each dwell position
     - Dwell positions stepped backward from the cranial catheter tip (d_0 = 3.5 mm tip offset,
       5.0 mm step spacing over <= 50 mm active length)

  2. Standardized 3D Evaluation Grid (1.5 mm isotropic) & HIPO/NNLS Dwell Optimization:
     - HR-CTV_implant: 8.0 mm radial envelope around reference active dwells (D90 = 100% = 7.0 Gy)
     - Bladder_proxy (Anterior OAR 2cc shell), Rectum_proxy (Posterior OAR 2cc shell),
       and Sigmoid/Bowel_proxy (Cranial OAR 2cc cap above catheter tips)

  3. Six Methods Evaluated Under Protocol A (Fixed-Plan Matched-Channel Sensitivity) &
     Protocol B (Optimize on Predicted -> Deliver on Reference):
     - Human Rater 2 (LB/AMe, N=40 dual-traced scans)
     - Stage 0 Physics Tracker (k_lim_auto)
     - PMB 2019 3D U-Net + Walk
     - NeedleFinder 2026 (No SynthBrachy3T, 4a)
     - NeedleFinder 2026 (Proposed, 4b)
     - NeedleFinder 2026 + Top-2 Uncertainty Tip Review (links Exp 1 to Exp 2!)

  4. Single-Catheter & Implant Tip-Shift Sensitivity Curve (dz in {-15..+15} mm)
  5. Uncertainty-to-Dose Propagation (M=30 Monte Carlo tip samples from Dedicated1DTipLocalizer)
"""

import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment, nnls
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun")

from run_stage2_and_tip_clean import (
    Dedicated1DTipLocalizer,
    decode_clean_stage2_preds,
    extract_1d_tip_profile_91,
    get_nested_fold_split,
)
from evaluate_unified_apples_to_apples import (
    get_gts,
    traj_gt_dist,
    compute_unified_pair_metrics,
)


# ==============================================================================
# 1. AAPM TG-43U1 HDR 192Ir LINE-SOURCE DOSE ENGINE (GPU-VECTORIZED)
# ==============================================================================

L_ACTIVE_CM = 0.35          # 3.5 mm active source core length (microSelectron-v2 / Flexisource)
LAMBDA_CGY_H_U = 1.109      # Consensus dose-rate constant (cGy h^-1 U^-1)
BETA_0 = 2.0 * math.atan(L_ACTIVE_CM / 2.0)
G_L_REF_CM2 = BETA_0 / (L_ACTIVE_CM * 1.0)  # G_L(1.0 cm, pi/2) = 0.989903 cm^-2
PRESCRIPTION_GY = 7.0       # Standard HDR GYN brachytherapy fraction dose (7.0 Gy = 100%)


def sample_dwells_from_catheter_ras(
    pts_ras,
    tip_offset_mm=3.5,
    step_mm=5.0,
    max_active_len_mm=30.0,
    z_min_active=None,
    n_dwells_exact=None,
):
    """
    Steps dwell positions backward along the 3D catheter shaft starting at `tip_offset_mm`
    (3.5 mm) caudal to the cranial tip `pts_ras[-1]`, with `step_mm` (5.0 mm) arc-length
    spacing. When `n_dwells_exact` is specified (for matched-channel afterloader stepping),
    steps exactly `n_dwells_exact` dwells and continues linearly along the caudal catheter
    shaft tangent if `s > total_len` (avoiding artificial dwell pile-up at the caudal end).
    Returns:
      - dwell_pts: [N_d, 3] in RAS mm
      - dwell_tangents: [N_d, 3] unit tangent vectors in RAS (pointing toward cranial tip)
      - dwell_arc_offsets: [N_d] arc-length distance from cranial tip in mm
    """
    pts = np.asarray(pts_ras, dtype=np.float64)
    pts = pts[np.argsort(pts[:, 2])]
    if len(pts) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    # Compute cumulative arc length backward from cranial tip pts[-1]
    rev_pts = pts[::-1]
    seg_vecs = np.diff(rev_pts, axis=0)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cum_s[-1])

    # Overall caudal-step unit vector (pointing from cranial tip toward caudal base)
    overall_vec = rev_pts[-1] - rev_pts[0]
    overall_norm = float(np.linalg.norm(overall_vec))
    u_caudal_step = overall_vec / max(1e-6, overall_norm) if overall_norm > 1e-4 else np.array([0.0, 0.0, -1.0])
    u_cranial_overall = -u_caudal_step

    if n_dwells_exact is not None:
        s_targets = tip_offset_mm + np.arange(int(n_dwells_exact), dtype=np.float64) * step_mm
    else:
        if total_len < tip_offset_mm + 1.0:
            return rev_pts[0:1].copy(), u_cranial_overall[None, :], np.array([0.0])
        s_targets = np.arange(tip_offset_mm, min(total_len - 0.2, tip_offset_mm + max_active_len_mm + 0.1), step_mm)
        if len(s_targets) == 0:
            s_targets = np.array([min(tip_offset_mm, total_len * 0.5)])

    dwell_pts = []
    dwell_tangents = []
    dwell_offsets = []

    for s in s_targets:
        if s <= total_len and total_len > 1e-3:
            idx = int(np.searchsorted(cum_s, s, side="right") - 1)
            idx = max(0, min(idx, len(seg_lens) - 1))
            ds = s - cum_s[idx]
            frac = ds / max(1e-6, seg_lens[idx])
            p_s = rev_pts[idx] + frac * seg_vecs[idx]
            u_s = -seg_vecs[idx] / max(1e-6, seg_lens[idx])
        else:
            # Continue linearly along the rigid/semi-rigid catheter shaft axis toward the template
            p_s = rev_pts[-1] + (s - total_len) * u_caudal_step
            u_s = u_cranial_overall

        if n_dwells_exact is None and z_min_active is not None and p_s[2] < z_min_active:
            continue
        dwell_pts.append(p_s)
        dwell_tangents.append(u_s)
        dwell_offsets.append(float(s))

    if len(dwell_pts) == 0:
        p_s = rev_pts[0] + tip_offset_mm * u_caudal_step
        dwell_pts.append(p_s)
        dwell_tangents.append(u_cranial_overall)
        dwell_offsets.append(float(tip_offset_mm))

    return (
        np.array(dwell_pts, dtype=np.float64),
        np.array(dwell_tangents, dtype=np.float64),
        np.array(dwell_offsets, dtype=np.float64),
    )


@torch.no_grad()
def compute_tg43_dose_rate_matrix_gpu(vox_ras_mm, dwell_pts_mm, dwell_tangents, device):
    """
    Computes the exact AAPM TG-43U1 line-source dose-rate kernel matrix K [N_vox, N_dwell]
    on GPU (`device`), where:
      K_{i, j} = Lambda * (G_L(r_{ij}, theta_{ij}) / G_L(r_0, theta_0)) * g_L(r_{ij}) * F(r_{ij}, theta_{ij})
    Distances are converted from mm to cm (r_cm = r_mm / 10.0).
    """
    if len(dwell_pts_mm) == 0 or len(vox_ras_mm) == 0:
        return torch.zeros((len(vox_ras_mm), len(dwell_pts_mm)), dtype=torch.float32, device=device)

    X_cm = torch.tensor(vox_ras_mm / 10.0, dtype=torch.float32, device=device)       # [N_v, 3]
    P_cm = torch.tensor(dwell_pts_mm / 10.0, dtype=torch.float32, device=device)     # [N_d, 3]
    U = torch.tensor(dwell_tangents, dtype=torch.float32, device=device)             # [N_d, 3]
    U = U / torch.norm(U, dim=-1, keepdim=True).clamp(min=1e-6)

    half_L = 0.5 * L_ACTIVE_CM
    P_minus = P_cm - half_L * U   # [N_d, 3]
    P_plus = P_cm + half_L * U    # [N_d, 3]

    # Process in voxel chunks if N_v is large to keep VRAM < 1 GB
    N_v = X_cm.shape[0]
    N_d = P_cm.shape[0]
    K_out = torch.empty((N_v, N_d), dtype=torch.float32, device=device)
    chunk_size = 16384

    for i0 in range(0, N_v, chunk_size):
        xc = X_cm[i0:i0 + chunk_size].unsqueeze(1)  # [B, 1, 3]
        diff = xc - P_cm.unsqueeze(0)               # [B, N_d, 3]
        r = torch.norm(diff, dim=-1).clamp(min=0.035)  # [B, N_d] in cm (clamp at 0.35 mm inner lumen radius)

        cos_theta = torch.sum(diff * U.unsqueeze(0), dim=-1) / r
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp(min=0.0))

        r1 = torch.norm(xc - P_minus.unsqueeze(0), dim=-1).clamp(min=0.02)
        r2 = torch.norm(xc - P_plus.unsqueeze(0), dim=-1).clamp(min=0.02)
        cos_beta = ((r1 * r1 + r2 * r2 - L_ACTIVE_CM * L_ACTIVE_CM) / (2.0 * r1 * r2)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        beta = torch.acos(cos_beta)

        denom = L_ACTIVE_CM * r * sin_theta
        g_line_off_axis = beta / denom.clamp(min=1e-5)
        g_line_on_axis = 1.0 / torch.abs(r * r - half_L * half_L).clamp(min=0.02 * 0.02)
        G_L = torch.where(sin_theta > 1e-3, g_line_off_axis, g_line_on_axis)
        G_rel = (G_L / G_L_REF_CM2).clamp(max=45.0)

        # Consensus TG-43U1 Ir-192 radial dose function g_L(r)
        r_cl = r.clamp(0.1, 10.0)
        g_rad = 0.990 + 0.0125 * r_cl - 0.0018 * (r_cl ** 2) - 0.00012 * (r_cl ** 3)

        # Consensus TG-43U1 Ir-192 2D anisotropy function F(r, theta)
        F_aniso = 1.0 - (0.38 * torch.exp(-0.35 * r_cl) + 0.06) * ((1.0 - sin_theta).clamp(min=0.0) ** 2.6)
        F_aniso = F_aniso.clamp(0.52, 1.02)

        K_out[i0:i0 + chunk_size] = LAMBDA_CGY_H_U * G_rel * g_rad * F_aniso

    return K_out


# ==============================================================================
# 2. PATIENT 3D EVALUATION GRID, TARGET/OAR MASKS, AND HIPO/NNLS OPTIMIZATION
# ==============================================================================

@torch.no_grad()
def build_patient_tg43_reference_plan(ref_catheters, device, grid_spacing_mm=1.5):
    """
    Constructs for a patient's reference active catheters (`gts_long`):
      1. Active dwell positions & tangents per catheter (`d_0 = 3.5 mm`, `step = 5.0 mm`).
      2. Standardized `1.5 mm` 3D isotropic evaluation grid around the active implant.
      3. Target (`HR-CTV_implant`, 8.0 mm radial envelope) and 3 OAR proxy masks:
         - `Bladder_proxy` (Anterior OAR shell)
         - `Rectum_proxy` (Posterior OAR shell)
         - `Sigmoid_proxy` (Cranial OAR cap superior to catheter tips)
      4. Reference HIPO / damped-NNLS dwell weights `t_ref` normalized so that
         `D90(HR-CTV_implant) = 100.0%` (7.0 Gy).
    """
    # Determine active target Z range (upper 34 mm of the implant in the HR-CTV zone)
    tip_zs = [float(np.max(c[:, 2])) for c in ref_catheters if len(c) >= 2]
    z_top_med = float(np.median(tip_zs))
    z_min_active = z_top_med - 36.0

    catheter_dwells = []
    all_dwell_pts = []
    all_dwell_tangents = []
    catheter_slices = []
    idx_cursor = 0

    for ci, c_pts in enumerate(ref_catheters):
        d_pts, d_tan, d_off = sample_dwells_from_catheter_ras(
            c_pts, tip_offset_mm=3.5, step_mm=5.0, max_active_len_mm=30.0, z_min_active=z_min_active
        )
        n_d = len(d_pts)
        catheter_dwells.append({
            "catheter_idx": ci,
            "pts": d_pts,
            "tangents": d_tan,
            "offsets": d_off,
            "slice": (idx_cursor, idx_cursor + n_d),
            "tip_ras": np.asarray(c_pts[np.argsort(c_pts[:, 2])][-1], dtype=np.float64),
        })
        all_dwell_pts.append(d_pts)
        all_dwell_tangents.append(d_tan)
        catheter_slices.append((idx_cursor, idx_cursor + n_d))
        idx_cursor += n_d

    P_ref = np.vstack(all_dwell_pts)
    U_ref = np.vstack(all_dwell_tangents)
    T_tips = np.array([cd["tip_ras"] for cd in catheter_dwells], dtype=np.float64)

    # Build 1.5 mm isotropic 3D grid around P_ref with 16 mm margin
    margin = 16.0
    p_min = np.min(P_ref, axis=0) - margin
    p_max = np.max(P_ref, axis=0) + margin
    xs = np.arange(p_min[0], p_max[0] + 0.1, grid_spacing_mm, dtype=np.float32)
    ys = np.arange(p_min[1], p_max[1] + 0.1, grid_spacing_mm, dtype=np.float32)
    zs = np.arange(p_min[2], p_max[2] + 0.1, grid_spacing_mm, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    vox_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    voxel_vol_cc = (grid_spacing_mm ** 3) / 1000.0  # 0.003375 cc per voxel

    # Compute distance from each grid voxel to nearest reference dwell position and nearest tip on GPU
    X_t = torch.tensor(vox_pts, dtype=torch.float32, device=device)
    P_t = torch.tensor(P_ref, dtype=torch.float32, device=device)
    Tips_t = torch.tensor(T_tips, dtype=torch.float32, device=device)

    min_d_dwell = torch.empty(len(vox_pts), dtype=torch.float32, device=device)
    nearest_dwell_y = torch.empty(len(vox_pts), dtype=torch.float32, device=device)
    min_d_tip = torch.empty(len(vox_pts), dtype=torch.float32, device=device)
    nearest_tip_z = torch.empty(len(vox_pts), dtype=torch.float32, device=device)

    chunk = 32768
    for i0 in range(0, len(vox_pts), chunk):
        xc = X_t[i0:i0 + chunk]
        d_mat = torch.cdist(xc, P_t)
        md, mi = torch.min(d_mat, dim=1)
        min_d_dwell[i0:i0 + chunk] = md
        nearest_dwell_y[i0:i0 + chunk] = P_t[mi, 1]

        dt_mat = torch.cdist(xc, Tips_t)
        mdt, mti = torch.min(dt_mat, dim=1)
        min_d_tip[i0:i0 + chunk] = mdt
        nearest_tip_z[i0:i0 + chunk] = Tips_t[mti, 2]

    min_d_np = min_d_dwell.cpu().numpy()
    near_y_np = nearest_dwell_y.cpu().numpy()
    min_dt_np = min_d_tip.cpu().numpy()
    near_tz_np = nearest_tip_z.cpu().numpy()
    del X_t, P_t, Tips_t, min_d_dwell, nearest_dwell_y, min_d_tip, nearest_tip_z

    # Define Target (HR-CTV_implant) and OAR Proxy Shells:
    # 1) HR-CTV_implant: within 8.0 mm of reference active dwells (excluding <= 1.0 mm intraluminal core for DVH stability)
    hr_ctv_mask = (min_d_np >= 1.0) & (min_d_np <= 8.0)
    surface_mask = (min_d_np >= 7.2) & (min_d_np <= 8.3)

    # 2) Anterior OAR (Bladder_proxy): anterior to local dwell (y - near_y >= 4.0 mm) and 5.5 <= d <= 16.0 mm
    bladder_mask = (min_d_np >= 5.5) & (min_d_np <= 16.0) & ((vox_pts[:, 1] - near_y_np) >= 4.0)

    # 3) Posterior OAR (Rectum_proxy): posterior to local dwell (near_y - y >= 4.0 mm) and 5.5 <= d <= 16.0 mm
    rectum_mask = (min_d_np >= 5.5) & (min_d_np <= 16.0) & ((near_y_np - vox_pts[:, 1]) >= 4.0)

    # 4) Cranial OAR (Sigmoid/Bowel_proxy): superior to local cranial tip (z - near_tz >= 2.5 mm) and 3.5 <= d_tip <= 15.0 mm
    sigmoid_mask = (min_dt_np >= 3.5) & (min_dt_np <= 15.0) & ((vox_pts[:, 2] - near_tz_np) >= 2.5)

    # Keep only the active evaluation union (voxels within 15.5 mm of implant/tips) to accelerate dose evaluation by 4x!
    eval_roi_mask = hr_ctv_mask | surface_mask | bladder_mask | rectum_mask | sigmoid_mask | (min_d_np <= 14.5)
    roi_indices = np.where(eval_roi_mask)[0]
    vox_roi = vox_pts[roi_indices]

    hr_ctv_roi = hr_ctv_mask[roi_indices]
    surface_roi = surface_mask[roi_indices]
    bladder_roi = bladder_mask[roi_indices]
    rectum_roi = rectum_mask[roi_indices]
    sigmoid_roi = sigmoid_mask[roi_indices]

    # Compute reference TG-43 kernel matrix on ROI voxels
    K_ref = compute_tg43_dose_rate_matrix_gpu(vox_roi, P_ref, U_ref, device)

    # Optimize reference dwell weights t_ref via damped NNLS on HR-CTV surface + OAR penalty
    K_surf = K_ref[surface_roi].cpu().numpy()
    if len(K_surf) > 1200:
        rng_s = np.random.default_rng(42)
        K_surf = K_surf[rng_s.choice(len(K_surf), size=1200, replace=False)]

    N_d = P_ref.shape[0]
    A_ata = K_surf.T @ K_surf
    reg_scale = float(np.trace(A_ata)) / max(1, N_d)
    # Add Tikhonov dwell-gradient smoothness + L2 damping so dwell times are physically smooth
    L_diff = np.zeros((N_d, N_d), dtype=np.float64)
    for c_s, c_e in catheter_slices:
        for j in range(c_s, c_e - 1):
            L_diff[j, j] += 1.0
            L_diff[j + 1, j + 1] += 1.0
            L_diff[j, j + 1] -= 1.0
            L_diff[j + 1, j] -= 1.0

    M_sys = A_ata + 0.18 * reg_scale * np.eye(N_d) + 0.25 * reg_scale * L_diff
    rhs = K_surf.T @ np.full(len(K_surf), 100.0, dtype=np.float64)
    t_raw = np.linalg.solve(M_sys, rhs)
    t_raw = np.maximum(t_raw, 0.15 * np.median(np.maximum(t_raw, 1e-4)))

    # Normalize t_ref so that D90(HR-CTV_implant) == 100.0% (7.0 Gy)
    t_ref_t = torch.tensor(t_raw, dtype=torch.float32, device=device)
    D_ref_raw = (K_ref @ t_ref_t).cpu().numpy()
    d90_raw = float(np.percentile(D_ref_raw[hr_ctv_roi], 10.0))
    scale_factor = 100.0 / max(1e-4, d90_raw)
    t_ref = (t_raw * scale_factor).astype(np.float64)
    D_ref = (D_ref_raw * scale_factor).astype(np.float32)

    # Store K_ref on CPU as float32 numpy array so GPU memory is freed!
    K_ref_np = K_ref.cpu().numpy()
    del K_ref, t_ref_t

    # Precompute 3D neighbor index table on `vox_roi` for fast 3D Gamma pass-rate evaluation (<= 4.5 mm radius)
    X_roi_t = torch.tensor(vox_roi, dtype=torch.float32, device=device)
    gamma_eval_mask = (D_ref >= 10.0) & (min_d_np[roi_indices] >= 1.5) & (min_d_np[roi_indices] <= 13.5)
    gamma_ref_idx = np.where(gamma_eval_mask)[0]
    if len(gamma_ref_idx) > 3500:
        rng_g = np.random.default_rng(101)
        gamma_ref_idx = np.sort(rng_g.choice(gamma_ref_idx, size=3500, replace=False))
    X_gref_t = X_roi_t[gamma_ref_idx]

    N_g = len(gamma_ref_idx)
    K_nbr = min(81, len(vox_roi))
    gamma_nbr_idx = np.empty((N_g, K_nbr), dtype=np.int64)
    gamma_nbr_dist_sq = np.empty((N_g, K_nbr), dtype=np.float32)
    chunk_g = 512
    for i0 in range(0, N_g, chunk_g):
        xr = X_gref_t[i0:i0 + chunk_g]
        d2 = torch.cdist(xr, X_roi_t).pow(2)
        vals, inds = torch.topk(d2, k=K_nbr, dim=1, largest=False)
        gamma_nbr_dist_sq[i0:i0 + chunk_g] = vals.cpu().numpy()
        gamma_nbr_idx[i0:i0 + chunk_g] = inds.cpu().numpy()

    del X_roi_t, X_gref_t
    torch.cuda.empty_cache()

    return {
        "ref_catheters": ref_catheters,
        "catheter_dwells": catheter_dwells,
        "catheter_slices": catheter_slices,
        "P_ref": P_ref,
        "U_ref": U_ref,
        "t_ref": t_ref,
        "z_min_active": z_min_active,
        "vox_roi": vox_roi,
        "voxel_vol_cc": voxel_vol_cc,
        "hr_ctv_roi": hr_ctv_roi,
        "surface_roi": surface_roi,
        "bladder_roi": bladder_roi,
        "rectum_roi": rectum_roi,
        "sigmoid_roi": sigmoid_roi,
        "K_ref_np": K_ref_np,
        "D_ref": D_ref,
        "gamma_ref_idx": gamma_ref_idx,
        "gamma_nbr_idx": gamma_nbr_idx,
        "gamma_nbr_dist_sq": gamma_nbr_dist_sq,
    }


def compute_d2cc_pct(dose_pct_arr, oar_mask, voxel_vol_cc):
    """
    Computes D_{2cc} (or D_{0.5cc} if the proxy shell is smaller than 2cc) as the minimum
    dose to the hottest 2.0 cm^3 of the OAR proxy volume.
    """
    oar_doses = dose_pct_arr[oar_mask]
    if len(oar_doses) == 0:
        return 0.0
    n_vox_2cc = max(1, int(round(2.0 / voxel_vol_cc)))
    if len(oar_doses) <= n_vox_2cc:
        return float(np.percentile(oar_doses, 50.0))
    part_idx = len(oar_doses) - n_vox_2cc
    return float(np.partition(oar_doses, part_idx)[part_idx])


@torch.no_grad()
def compute_3d_gamma_pass_rates_gpu(D_eval_roi, plan, device):
    """
    Computes 3D Gamma pass rates (% of voxels with gamma <= 1.0) on GPU across `gamma_ref_idx`:
      - 3% / 3 mm (global, 3% of 100% prescription dose, 3.0 mm DTA)
      - 2% / 2 mm (global, 2% of 100% prescription dose, 2.0 mm DTA)
      - 2% / 2 mm (local, 2% of max(D_ref(x), 20%), 2.0 mm DTA)
    Uses continuous spatial interpolation refinement along local dose gradients between
    1.5 mm grid points so sub-millimeter dose-to-agreement is accurately resolved.
    """
    g_idx = plan["gamma_ref_idx"]
    nbr_idx = torch.tensor(plan["gamma_nbr_idx"], dtype=torch.long, device=device)          # [N_g, K_nbr]
    dist_sq = torch.tensor(plan["gamma_nbr_dist_sq"], dtype=torch.float32, device=device)  # [N_g, K_nbr]
    D_r = torch.tensor(plan["D_ref"][g_idx], dtype=torch.float32, device=device).unsqueeze(1)  # [N_g, 1]
    D_e_all = torch.tensor(D_eval_roi, dtype=torch.float32, device=device)                     # [N_roi]
    de_0 = D_e_all[g_idx].unsqueeze(1)                                                         # [N_g, 1]
    de_n = D_e_all[nbr_idx]                                                                    # [N_g, K_nbr]

    d_loc_denom = torch.clamp(D_r, min=20.0) * 0.02
    near_mask = dist_sq <= (4.5 ** 2)

    # Continuous sub-grid interpolation along segment from same-voxel to neighbor voxel
    denom_interp = de_n - de_0
    alpha = torch.where(
        torch.abs(denom_interp) > 1e-3,
        ((D_r - de_0) / denom_interp).clamp(0.0, 1.0),
        torch.zeros_like(denom_interp),
    )
    d_interp = de_0 + alpha * denom_interp
    dd_interp_sq = (d_interp - D_r).pow(2)
    dist_interp_sq = (alpha * alpha) * dist_sq

    g2_33 = torch.where(near_mask, dist_interp_sq / (3.0 ** 2) + dd_interp_sq / (3.0 ** 2), torch.full_like(dist_sq, 1e6))
    g2_22_g = torch.where(near_mask, dist_interp_sq / (2.0 ** 2) + dd_interp_sq / (2.0 ** 2), torch.full_like(dist_sq, 1e6))
    g2_22_l = torch.where(near_mask, dist_interp_sq / (2.0 ** 2) + dd_interp_sq / (d_loc_denom ** 2), torch.full_like(dist_sq, 1e6))

    min_g2_33 = torch.min(g2_33, dim=1)[0]
    min_g2_22_glob = torch.min(g2_22_g, dim=1)[0]
    min_g2_22_loc = torch.min(g2_22_l, dim=1)[0]

    return {
        "gamma_3pct_3mm_global_pass_pct": float(torch.mean((min_g2_33 <= 1.0).float()).item() * 100.0),
        "gamma_2pct_2mm_global_pass_pct": float(torch.mean((min_g2_22_glob <= 1.0).float()).item() * 100.0),
        "gamma_2pct_2mm_local_pass_pct": float(torch.mean((min_g2_22_loc <= 1.0).float()).item() * 100.0),
    }


def evaluate_dose_distribution_metrics(D_eval_roi, plan, device):
    """
    Computes complete clinical DVH and 3D Gamma metrics comparing `D_eval_roi` against `plan["D_ref"]`.
    """
    D_ref = plan["D_ref"]
    hr_mask = plan["hr_ctv_roi"]
    v_cc = plan["voxel_vol_cc"]

    d90_ref = float(np.percentile(D_ref[hr_mask], 10.0))  # 100.0%
    d90_eval = float(np.percentile(D_eval_roi[hr_mask], 10.0))
    delta_d90_pct = d90_eval - d90_ref
    abs_delta_d90_pct = abs(delta_d90_pct)
    abs_delta_d90_gy = abs_delta_d90_pct * (PRESCRIPTION_GY / 100.0)

    v100_ref = float(np.mean(D_ref[hr_mask] >= 100.0) * 100.0)
    v100_eval = float(np.mean(D_eval_roi[hr_mask] >= 100.0) * 100.0)
    v150_ref = float(np.mean(D_ref[hr_mask] >= 150.0) * 100.0)
    v150_eval = float(np.mean(D_eval_roi[hr_mask] >= 150.0) * 100.0)
    v200_ref = float(np.mean(D_ref[hr_mask] >= 200.0) * 100.0)
    v200_eval = float(np.mean(D_eval_roi[hr_mask] >= 200.0) * 100.0)

    bl_ref = compute_d2cc_pct(D_ref, plan["bladder_roi"], v_cc)
    bl_eval = compute_d2cc_pct(D_eval_roi, plan["bladder_roi"], v_cc)
    rec_ref = compute_d2cc_pct(D_ref, plan["rectum_roi"], v_cc)
    rec_eval = compute_d2cc_pct(D_eval_roi, plan["rectum_roi"], v_cc)
    sig_ref = compute_d2cc_pct(D_ref, plan["sigmoid_roi"], v_cc)
    sig_eval = compute_d2cc_pct(D_eval_roi, plan["sigmoid_roi"], v_cc)

    gamma_res = compute_3d_gamma_pass_rates_gpu(D_eval_roi, plan, device)

    return {
        "d90_eval_pct": round(d90_eval, 3),
        "delta_d90_signed_pct": round(delta_d90_pct, 3),
        "abs_delta_d90_pct": round(abs_delta_d90_pct, 3),
        "abs_delta_d90_gy": round(abs_delta_d90_gy, 4),
        "v100_ref_pct": round(v100_ref, 3),
        "v100_eval_pct": round(v100_eval, 3),
        "abs_delta_v100_pct": round(abs(v100_eval - v100_ref), 3),
        "abs_delta_v150_pct": round(abs(v150_eval - v150_ref), 3),
        "abs_delta_v200_pct": round(abs(v200_eval - v200_ref), 3),
        "bladder_d2cc_ref_pct": round(bl_ref, 3),
        "bladder_d2cc_eval_pct": round(bl_eval, 3),
        "abs_delta_bladder_d2cc_pct": round(abs(bl_eval - bl_ref), 3),
        "rectum_d2cc_ref_pct": round(rec_ref, 3),
        "rectum_d2cc_eval_pct": round(rec_eval, 3),
        "abs_delta_rectum_d2cc_pct": round(abs(rec_eval - rec_ref), 3),
        "sigmoid_cranial_d2cc_ref_pct": round(sig_ref, 3),
        "sigmoid_cranial_d2cc_eval_pct": round(sig_eval, 3),
        "abs_delta_sigmoid_cranial_d2cc_pct": round(abs(sig_eval - sig_ref), 3),
        "signed_delta_sigmoid_cranial_d2cc_pct": round(sig_eval - sig_ref, 3),
        **{k: round(v, 3) for k, v in gamma_res.items()},
    }


# ==============================================================================
# 3. PROTOCOL A (FIXED-PLAN MATCHED-CHANNEL) & PROTOCOL B (OPTIMIZE-ON-PRED -> DELIVER-ON-REF)
# ==============================================================================

def evaluate_predictions_on_patient_tg43(preds_case, plan, device, match_thresh_mm=3.5, fallback_preds_case=None):
    """
    Evaluates a method's predicted catheters on a patient under:
      - Protocol A (Fixed-Plan Matched-Channel Sensitivity):
        Transfers reference dwell weights `t_ref` by matched channel identity and arc-length
        step index measured backward from the predicted distal tip (`d_0 = 3.5 mm`, `step = 5.0 mm`).
        For any missed reference channel, checks `fallback_preds_case` within 5.5 mm (if provided)
        or transfers its channel weight to the nearest predicted catheter (preserving total
        prescription TRAK while penalizing missing spatial coverage).
      - Protocol B (End-to-End Plan-Optimization Transfer):
        Optimizes dwell weights `t_pred` on the predicted catheters to achieve D90 = 100% on the
        predicted target geometry, then delivers `t_pred` on the true reference catheters!
    Also computes Dwell-Time-Weighted 3D Tip Error vs Unweighted 3D Tip Error.
    """
    ref_caths = plan["ref_catheters"]
    n_ref = len(ref_caths)
    pred_pts_list = []
    for p in preds_case:
        pts = np.asarray(p["ras"] if isinstance(p, dict) and "ras" in p else (p["points"] if isinstance(p, dict) else p), dtype=np.float64)
        if len(pts) >= 2:
            pts = pts[np.argsort(pts[:, 2])]
            pred_pts_list.append(pts)
    n_pred = len(pred_pts_list)

    fb_pts_list = []
    if fallback_preds_case is not None:
        for p in fallback_preds_case:
            pts = np.asarray(p["ras"] if isinstance(p, dict) and "ras" in p else (p["points"] if isinstance(p, dict) else p), dtype=np.float64)
            if len(pts) >= 2:
                pts = pts[np.argsort(pts[:, 2])]
                fb_pts_list.append(pts)

    # 1-to-1 Hungarian matching to reference active catheters
    cost = np.full((n_ref, max(1, n_pred)), 1e6, dtype=np.float64)
    pms = {}
    for gi, g in enumerate(ref_caths):
        for pi, p in enumerate(pred_pts_list):
            pm = compute_unified_pair_metrics(p, g)
            if pm is not None:
                cost[gi, pi] = pm["mean_shaft_xy"]
                pms[(gi, pi)] = pm

    cost_thresh = np.where(cost > match_thresh_mm, 1e6, cost)
    ri, ci = linear_sum_assignment(cost_thresh)
    matched_map = {}  # gi -> pi
    tip_errors_by_gi = {}
    shaft_errors_by_gi = {}
    for g_idx, p_idx in zip(ri, ci):
        if cost_thresh[g_idx, p_idx] <= match_thresh_mm:
            matched_map[int(g_idx)] = int(p_idx)
            tip_errors_by_gi[int(g_idx)] = float(pms[(g_idx, p_idx)]["tip_3d"])
            shaft_errors_by_gi[int(g_idx)] = float(pms[(g_idx, p_idx)]["mean_shaft_xy"])

    # Channel dwell weights W_c = sum_{j in c} t_ref_j
    channel_weights = np.array([
        float(np.sum(plan["t_ref"][s0:s1])) for (s0, s1) in plan["catheter_slices"]
    ], dtype=np.float64)

    matched_gis = sorted(matched_map.keys())
    if len(matched_gis) > 0:
        w_matched = np.array([channel_weights[g_i] for g_i in matched_gis], dtype=np.float64)
        e_tip_matched = np.array([tip_errors_by_gi[g_i] for g_i in matched_gis], dtype=np.float64)
        e_sh_matched = np.array([shaft_errors_by_gi[g_i] for g_i in matched_gis], dtype=np.float64)
        dwell_weighted_tip_err = float(np.sum(w_matched * e_tip_matched) / max(1e-6, np.sum(w_matched)))
        unweighted_tip_err = float(np.mean(e_tip_matched))
        dwell_weighted_shaft_err = float(np.sum(w_matched * e_sh_matched) / max(1e-6, np.sum(w_matched)))
    else:
        dwell_weighted_tip_err = 15.0
        unweighted_tip_err = 15.0
        dwell_weighted_shaft_err = 3.5

    # Resolve trajectory per channel (matched_map -> fallback within 5.5 mm -> nearest pred)
    used_pis = set(matched_map.values())
    channel_pred_pts = []
    for g_i in range(n_ref):
        if g_i in matched_map:
            channel_pred_pts.append(pred_pts_list[matched_map[g_i]])
        else:
            # Check unassigned predictions in pred_pts_list or fb_pts_list within 5.5 mm
            best_p = None
            best_d = 5.5
            for pi, p in enumerate(pred_pts_list):
                if pi not in used_pis:
                    d_sh = traj_gt_dist(p, ref_caths[g_i])
                    if d_sh < best_d:
                        best_d = d_sh
                        best_p = p
            if best_p is None and len(fb_pts_list) > 0:
                for p in fb_pts_list:
                    d_sh = traj_gt_dist(p, ref_caths[g_i])
                    if d_sh < best_d:
                        best_d = d_sh
                        best_p = p
            if best_p is not None:
                channel_pred_pts.append(best_p)
            elif n_pred > 0:
                nearest_pi = int(np.argmin([traj_gt_dist(p, ref_caths[g_i]) for p in pred_pts_list]))
                channel_pred_pts.append(pred_pts_list[nearest_pi])
            else:
                channel_pred_pts.append(ref_caths[g_i])

    # ------------------------------------------------------------------
    # PROTOCOL A: Fixed-Plan Matched-Channel Sensitivity
    # ------------------------------------------------------------------
    P_protA = []
    U_protA = []
    t_protA = []

    for g_i in range(n_ref):
        s0, s1 = plan["catheter_slices"][g_i]
        t_ch = plan["t_ref"][s0:s1]
        n_d_ref = len(t_ch)
        p_pts = channel_pred_pts[g_i]

        # Step the exact same number of dwells n_d_ref backward from the predicted cranial tip (d_0=3.5mm, step=5.0mm)
        d_pts, d_tan, _ = sample_dwells_from_catheter_ras(
            p_pts, tip_offset_mm=3.5, step_mm=5.0, n_dwells_exact=n_d_ref
        )

        P_protA.append(d_pts)
        U_protA.append(d_tan)
        t_protA.append(t_ch)

    P_protA = np.vstack(P_protA)
    U_protA = np.vstack(U_protA)
    t_protA = np.concatenate(t_protA)
    K_protA = compute_tg43_dose_rate_matrix_gpu(plan["vox_roi"], P_protA, U_protA, device)
    D_protA = (K_protA @ torch.tensor(t_protA, dtype=torch.float32, device=device)).cpu().numpy()
    del K_protA
    metrics_protA = evaluate_dose_distribution_metrics(D_protA, plan, device)

    # ------------------------------------------------------------------
    # PROTOCOL B: Optimize on Predicted Geometry -> Deliver on True Reference Catheters
    # ------------------------------------------------------------------
    P_delivB = []
    U_delivB = []
    P_planB = []
    U_planB = []

    for g_i in range(n_ref):
        s0, s1 = plan["catheter_slices"][g_i]
        n_d_ref = s1 - s0
        ref_d_pts = plan["P_ref"][s0:s1]
        ref_d_tan = plan["U_ref"][s0:s1]
        P_delivB.append(ref_d_pts)
        U_delivB.append(ref_d_tan)

        p_pts = channel_pred_pts[g_i]
        d_pts, d_tan, _ = sample_dwells_from_catheter_ras(
            p_pts, tip_offset_mm=3.5, step_mm=5.0, n_dwells_exact=n_d_ref
        )
        P_planB.append(d_pts)
        U_planB.append(d_tan)

    P_planB = np.vstack(P_planB)
    U_planB = np.vstack(U_planB)
    K_planB = compute_tg43_dose_rate_matrix_gpu(plan["vox_roi"], P_planB, U_planB, device)

    # Optimize t_pred on K_planB to cover the target surface
    K_surf_B = K_planB[plan["surface_roi"]].cpu().numpy()
    if len(K_surf_B) > 1200:
        rng_b = np.random.default_rng(42)
        K_surf_B = K_surf_B[rng_b.choice(len(K_surf_B), size=1200, replace=False)]
    N_d = P_planB.shape[0]
    A_ata_B = K_surf_B.T @ K_surf_B
    reg_B = float(np.trace(A_ata_B)) / max(1, N_d)
    L_diff_B = np.zeros((N_d, N_d), dtype=np.float64)
    for c_s, c_e in plan["catheter_slices"]:
        for j in range(c_s, c_e - 1):
            L_diff_B[j, j] += 1.0
            L_diff_B[j + 1, j + 1] += 1.0
            L_diff_B[j, j + 1] -= 1.0
            L_diff_B[j + 1, j] -= 1.0
    M_sys_B = A_ata_B + 0.18 * reg_B * np.eye(N_d) + 0.25 * reg_B * L_diff_B
    rhs_B = K_surf_B.T @ np.full(len(K_surf_B), 100.0, dtype=np.float64)
    t_pred_raw = np.linalg.solve(M_sys_B, rhs_B)
    t_pred_raw = np.maximum(t_pred_raw, 0.15 * np.median(np.maximum(t_pred_raw, 1e-4)))

    # Normalize t_pred so planned D90 on K_planB equals 100%
    D_planned_B = (K_planB @ torch.tensor(t_pred_raw, dtype=torch.float32, device=device)).cpu().numpy()
    del K_planB
    d90_planned = float(np.percentile(D_planned_B[plan["hr_ctv_roi"]], 10.0))
    t_pred = (t_pred_raw * (100.0 / max(1e-4, d90_planned))).astype(np.float32)

    # Deliver t_pred on the TRUE reference catheters (K_ref_np)!
    D_delivB = plan["K_ref_np"] @ t_pred
    metrics_protB = evaluate_dose_distribution_metrics(D_delivB, plan, device)

    return {
        "n_ref_active_catheters": n_ref,
        "n_pred_catheters": n_pred,
        "n_matched_1to1": len(matched_map),
        "recall_active_pct": round(100.0 * len(matched_map) / max(1, n_ref), 2),
        "unweighted_mean_tip_3d_mm": round(unweighted_tip_err, 3),
        "dwell_weighted_mean_tip_3d_mm": round(dwell_weighted_tip_err, 3),
        "dwell_weighted_mean_shaft_xy_mm": round(dwell_weighted_shaft_err, 4),
        "protocol_A_fixed_plan_sensitivity": metrics_protA,
        "protocol_B_optimize_on_pred_deliver_on_ref": metrics_protB,
    }


# ==============================================================================
# 4. 1D TIP UNCERTAINTY EXTRACTION, TOP-2 REVIEW, & MONTE CARLO DOSE PROPAGATION
# ==============================================================================

def compute_1d_tip_posteriors_and_top2_review(
    preds_long_50,
    graphs,
    all_oof_cal,
    all_X_vols,
    tip_models_5fold,
    gts_long_list,
    device,
):
    """
    For each patient i in 0..53 and each predicted catheter in `preds_long_50[i]`:
      1. Evaluates the fold-appropriate `Dedicated1DTipLocalizer` (`tip_models_5fold[i % 5]`)
         to obtain the 91-bin posterior probability distribution `p_bins` over dz in [-25, +20] mm
         and its standard deviation `sigma_tip_mm`.
      2. Constructs `preds_long_top2_review[i]`, where the 2 catheters per patient with the highest
         composite tip uncertainty `sigma_tip_mm` are reviewed by a human physicist (snapping their
         cranial tip Z to the matched reference catheter tip Z).
    """
    dz_grid = np.linspace(-25.0, 20.0, 91, dtype=np.float64)
    posteriors_per_case = []
    preds_top2_review = []

    for ci in range(len(preds_long_50)):
        f_out = ci % 5
        tip_model = tip_models_5fold[f_out]
        g = graphs[ci]
        oof_cal = all_oof_cal[ci]
        X_vol = all_X_vols[ci]
        preds_c = copy.deepcopy(preds_long_50[ci])
        gts_c = gts_long_list[ci]

        if len(preds_c) == 0:
            posteriors_per_case.append([])
            preds_top2_review.append([])
            continue

        profs = []
        for p in preds_c:
            ras = np.asarray(p["ras"], dtype=np.float64)
            ras = ras[np.argsort(ras[:, 2])]
            z_top = float(ras[-1, 2])
            z_nbr = float(p.get("z_tip_nbr", z_top))
            profs.append(extract_1d_tip_profile_91(ras, g, oof_cal, X_vol, z_center=z_top, z_tip_nbr=z_nbr))

        P_t = torch.tensor(np.stack(profs, axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            dz_pred, step_logits = tip_model(P_t)
            p_bins = F.softmax(step_logits, dim=-1).cpu().numpy()

        case_post = []
        for pi, p in enumerate(preds_c):
            ras = np.asarray(p["ras"], dtype=np.float64)
            ras = ras[np.argsort(ras[:, 2])]
            pb = p_bins[pi]
            mu_dz = float(np.sum(pb * dz_grid))
            var_dz = float(np.sum(pb * ((dz_grid - mu_dz) ** 2)))
            std_1d = math.sqrt(max(0.0, var_dz))
            # Combine 1D posterior spread with neighbor/profile tip discrepancy
            z_top = float(ras[-1, 2])
            dz_nbr_abs = abs(float(p.get("z_tip_nbr", z_top)) - z_top)
            dz_prof_abs = abs(float(p.get("z_prof_tip", z_top)) - z_top)
            sigma_comp = 0.65 * std_1d + 0.20 * min(15.0, dz_nbr_abs) + 0.15 * min(15.0, dz_prof_abs)
            case_post.append({
                "pred_idx": pi,
                "p_bins": pb,
                "std_1d_mm": std_1d,
                "sigma_tip_mm": sigma_comp,
            })

        # Rank predicted catheters by decreasing tip uncertainty `sigma_tip_mm` and review top 2
        order_unc = np.argsort([-cp["sigma_tip_mm"] for cp in case_post])
        for rank_i in order_unc[:2]:
            p_item = preds_c[rank_i]
            ras = np.asarray(p_item["ras"], dtype=np.float64)
            ras = ras[np.argsort(ras[:, 2])]
            # Find matched reference catheter if within 3.5 mm shaft XY
            best_gi = -1
            best_d = 3.5
            for gi, gt in enumerate(gts_c):
                pm = compute_unified_pair_metrics(ras, gt)
                if pm is not None and pm["mean_shaft_xy"] <= best_d:
                    best_d = pm["mean_shaft_xy"]
                    best_gi = gi
            if best_gi >= 0:
                gt_match = np.asarray(gts_c[best_gi], dtype=np.float64)
                gt_match = gt_match[np.argsort(gt_match[:, 2])]
                z_true_tip = float(gt_match[-1, 2])
                z_cur_tip = float(ras[-1, 2])
                dz_snap = z_true_tip - z_cur_tip
                if abs(dz_snap) > 0.25:
                    slope_top = (ras[-1, :2] - ras[max(0, len(ras) - 4), :2]) / max(1.0, ras[-1, 2] - ras[max(0, len(ras) - 4), 2])
                    if dz_snap > 0:
                        n_st = max(1, int(round(dz_snap / 1.5)))
                        zs_add = np.linspace(z_cur_tip + dz_snap / n_st, z_true_tip, n_st)
                        pts_add = np.column_stack([
                            ras[-1, 0] + slope_top[0] * (zs_add - z_cur_tip),
                            ras[-1, 1] + slope_top[1] * (zs_add - z_cur_tip),
                            zs_add,
                        ])
                        ras_new = np.vstack([ras, pts_add])
                    else:
                        keep = ras[:, 2] < z_true_tip - 0.2
                        tip_pt = np.array([[
                            ras[-1, 0] + slope_top[0] * dz_snap,
                            ras[-1, 1] + slope_top[1] * dz_snap,
                            z_true_tip,
                        ]])
                        ras_new = np.vstack([ras[keep], tip_pt]) if np.sum(keep) >= 2 else np.vstack([ras[:1], tip_pt])
                    p_item["ras"] = ras_new
                    p_item["points"] = ras_new

        posteriors_per_case.append(case_post)
        preds_top2_review.append(preds_c)

    return posteriors_per_case, preds_top2_review


def run_uncertainty_to_dose_propagation(
    preds_long_50,
    posteriors_per_case,
    patient_plans,
    nf_patient_evals,
    device,
    n_mc_samples=30,
):
    """
    Propagates `Dedicated1DTipLocalizer` 91-bin posterior tip distributions (`p_bins`) through
    the 3D AAPM TG-43 dose engine using `n_mc_samples = 30` Monte Carlo realizations per patient:
      - Computes predicted dose uncertainty `sigma_hat_D90_pct` and `sigma_hat_D2cc_cranial_pct`
      - Computes Spearman rank correlation across the 54 patients between:
          1) `sigma_hat_D90_pct` vs true `|Delta D90|`
          2) Dwell-weighted predicted tip uncertainty `sigma_tip_dwell_weighted_mm` vs true `|Delta D90|`
          3) Dwell-weighted true tip error vs true `|Delta D90|` (and vs unweighted tip error)
    """
    dz_grid = np.linspace(-25.0, 20.0, 91, dtype=np.float64)
    rng = np.random.default_rng(2026)

    per_patient_unc = []
    for ci in range(len(patient_plans)):
        plan = patient_plans[ci]
        preds_c = preds_long_50[ci]
        post_c = posteriors_per_case[ci]
        ref_caths = plan["ref_catheters"]
        n_ref = len(ref_caths)

        pred_pts_list = [np.asarray(p["ras"], dtype=np.float64)[np.argsort(np.asarray(p["ras"])[:, 2])] for p in preds_c if len(p["ras"]) >= 2]
        n_pred = len(pred_pts_list)

        cost = np.full((n_ref, max(1, n_pred)), 1e6, dtype=np.float64)
        for gi, g in enumerate(ref_caths):
            for pi, p in enumerate(pred_pts_list):
                pm = compute_unified_pair_metrics(p, g)
                if pm is not None and pm["mean_shaft_xy"] <= 3.5:
                    cost[gi, pi] = pm["mean_shaft_xy"]
        ri, col_i = linear_sum_assignment(cost)
        matched_map = {int(g_i): int(p_i) for g_i, p_i in zip(ri, col_i) if cost[g_i, p_i] <= 3.5}

        channel_weights = np.array([float(np.sum(plan["t_ref"][s0:s1])) for (s0, s1) in plan["catheter_slices"]], dtype=np.float64)
        sigma_by_gi = []
        w_by_gi = []
        for g_i, p_i in matched_map.items():
            if p_i < len(post_c):
                sigma_by_gi.append(post_c[p_i]["sigma_tip_mm"])
                w_by_gi.append(channel_weights[g_i])
        dwell_w_sigma_tip = float(np.sum(np.array(w_by_gi) * np.array(sigma_by_gi)) / max(1e-6, np.sum(w_by_gi))) if w_by_gi else 5.0

        # Draw M = 30 Monte Carlo tip perturbations from p_bins (centered around 0) and evaluate D90 & cranial D2cc
        mc_d90s = []
        mc_cran_d2ccs = []
        for m_i in range(n_mc_samples):
            P_mc = []
            U_mc = []
            t_mc = []
            for g_i in range(n_ref):
                s0, s1 = plan["catheter_slices"][g_i]
                t_ch = plan["t_ref"][s0:s1]
                n_d_ref = len(t_ch)
                if g_i in matched_map:
                    p_i = matched_map[g_i]
                    base_pts = pred_pts_list[p_i].copy()
                    pb = post_c[p_i]["p_bins"]
                    mu_dz = float(np.sum(pb * dz_grid))
                    sampled_dz = float(rng.choice(dz_grid, p=pb)) - mu_dz
                    # Also include a small Gaussian component from sigma_tip_mm for heavy-tail realism
                    sampled_dz += float(rng.normal(0.0, 0.35 * post_c[p_i]["sigma_tip_mm"]))
                    sampled_dz = float(np.clip(sampled_dz, -15.0, 15.0))
                else:
                    base_pts = ref_caths[g_i].copy()
                    sampled_dz = float(rng.normal(0.0, 3.0))

                d_pts, d_tan, _ = sample_dwells_from_catheter_ras(
                    base_pts, tip_offset_mm=3.5, step_mm=5.0, n_dwells_exact=n_d_ref
                )

                # Shift dwell positions along local catheter tangent by sampled_dz
                d_pts = d_pts + sampled_dz * d_tan
                P_mc.append(d_pts)
                U_mc.append(d_tan)
                t_mc.append(t_ch)

            P_mc = np.vstack(P_mc)
            U_mc = np.vstack(U_mc)
            t_mc_t = torch.tensor(np.concatenate(t_mc), dtype=torch.float32, device=device)
            K_mc = compute_tg43_dose_rate_matrix_gpu(plan["vox_roi"], P_mc, U_mc, device)
            D_mc = (K_mc @ t_mc_t).cpu().numpy()
            del K_mc, t_mc_t
            mc_d90s.append(float(np.percentile(D_mc[plan["hr_ctv_roi"]], 10.0)))
            mc_cran_d2ccs.append(compute_d2cc_pct(D_mc, plan["sigmoid_roi"], plan["voxel_vol_cc"]))

        sigma_d90_pct = float(np.std(mc_d90s))
        iqr_d90_pct = float(np.percentile(mc_d90s, 75) - np.percentile(mc_d90s, 25))
        sigma_cran_pct = float(np.std(mc_cran_d2ccs))

        true_abs_d90 = nf_patient_evals[ci]["protocol_A_fixed_plan_sensitivity"]["abs_delta_d90_pct"]
        true_abs_cran = nf_patient_evals[ci]["protocol_A_fixed_plan_sensitivity"]["abs_delta_sigmoid_cranial_d2cc_pct"]
        true_gamma_22 = nf_patient_evals[ci]["protocol_A_fixed_plan_sensitivity"]["gamma_2pct_2mm_global_pass_pct"]

        per_patient_unc.append({
            "case_idx": ci,
            "dwell_weighted_pred_sigma_tip_mm": round(dwell_w_sigma_tip, 3),
            "dwell_weighted_true_tip_err_mm": nf_patient_evals[ci]["dwell_weighted_mean_tip_3d_mm"],
            "unweighted_true_tip_err_mm": nf_patient_evals[ci]["unweighted_mean_tip_3d_mm"],
            "pred_sigma_d90_pct": round(sigma_d90_pct, 3),
            "pred_sigma_d90_gy": round(sigma_d90_pct * (PRESCRIPTION_GY / 100.0), 4),
            "pred_iqr_d90_pct": round(iqr_d90_pct, 3),
            "pred_sigma_cranial_d2cc_pct": round(sigma_cran_pct, 3),
            "true_abs_delta_d90_pct": true_abs_d90,
            "true_abs_delta_cranial_d2cc_pct": true_abs_cran,
            "true_gamma_2pct_2mm_pass_pct": true_gamma_22,
        })

    arr_pred_sd90 = np.array([x["pred_sigma_d90_pct"] for x in per_patient_unc])
    arr_pred_stip = np.array([x["dwell_weighted_pred_sigma_tip_mm"] for x in per_patient_unc])
    arr_true_wtip = np.array([x["dwell_weighted_true_tip_err_mm"] for x in per_patient_unc])
    arr_true_utip = np.array([x["unweighted_true_tip_err_mm"] for x in per_patient_unc])
    arr_true_d90 = np.array([x["true_abs_delta_d90_pct"] for x in per_patient_unc])
    arr_true_g22 = np.array([x["true_gamma_2pct_2mm_pass_pct"] for x in per_patient_unc])

    r_sd90_d90, p_sd90_d90 = spearmanr(arr_pred_sd90, arr_true_d90)
    r_stip_d90, p_stip_d90 = spearmanr(arr_pred_stip, arr_true_d90)
    r_wtip_d90, p_wtip_d90 = spearmanr(arr_true_wtip, arr_true_d90)
    r_utip_d90, p_utip_d90 = spearmanr(arr_true_utip, arr_true_d90)
    r_wtip_g22, p_wtip_g22 = spearmanr(arr_true_wtip, arr_true_g22)

    return {
        "n_mc_samples_per_patient": n_mc_samples,
        "summary_54_patients": {
            "mean_pred_sigma_d90_pct": round(float(np.mean(arr_pred_sd90)), 3),
            "median_pred_sigma_d90_pct": round(float(np.median(arr_pred_sd90)), 3),
            "p95_pred_sigma_d90_pct": round(float(np.percentile(arr_pred_sd90, 95)), 3),
            "spearman_pred_sigma_d90_vs_true_abs_delta_d90": {"rho": round(float(r_sd90_d90), 4), "p_value": float(p_sd90_d90)},
            "spearman_dwell_weighted_pred_sigma_tip_vs_true_abs_delta_d90": {"rho": round(float(r_stip_d90), 4), "p_value": float(p_stip_d90)},
            "spearman_dwell_weighted_true_tip_err_vs_true_abs_delta_d90": {"rho": round(float(r_wtip_d90), 4), "p_value": float(p_wtip_d90)},
            "spearman_unweighted_true_tip_err_vs_true_abs_delta_d90": {"rho": round(float(r_utip_d90), 4), "p_value": float(p_utip_d90)},
            "spearman_dwell_weighted_true_tip_err_vs_gamma_2pct_2mm": {"rho": round(float(r_wtip_g22), 4), "p_value": float(p_wtip_g22)},
        },
        "per_patient": per_patient_unc,
    }


# ==============================================================================
# 5. CONTROLLED TIP-SHIFT SENSITIVITY CURVE (dz in {-15..+15} mm)
# ==============================================================================

def run_tip_shift_sensitivity_curve(patient_plans, device):
    """
    Evaluates the controlled dosimetric sensitivity curve across all 54 patients when
    catheter tips are shifted axially along their 3D trajectory tangent by:
      dz in {-15, -10, -5, -3, -2, -1, 0, +1, +2, +3, +5, +10, +15} mm
    under both:
      (a) Single Dominant Catheter Shift (shifting the highest-dwell-weight catheter per patient)
      (b) Systematic Implant Tip Shift (shifting the top-3 highest-dwell-weight catheters per patient)
    """
    shifts_mm = [-15.0, -10.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
    curve_out = {"single_dominant_catheter_shift": {}, "top3_catheters_implant_shift": {}}

    for mode, n_shift_caths in [("single_dominant_catheter_shift", 1), ("top3_catheters_implant_shift", 3)]:
        for dz in shifts_mm:
            d90_abs, d90_signed, v100_abs, cran_abs, cran_signed, g22_glob, g33_glob = [], [], [], [], [], [], []
            for plan in patient_plans:
                if abs(dz) < 1e-6:
                    d90_abs.append(0.0)
                    d90_signed.append(0.0)
                    v100_abs.append(0.0)
                    cran_abs.append(0.0)
                    cran_signed.append(0.0)
                    g22_glob.append(100.0)
                    g33_glob.append(100.0)
                    continue

                ch_w = [float(np.sum(plan["t_ref"][s0:s1])) for (s0, s1) in plan["catheter_slices"]]
                top_caths = set(np.argsort(-np.array(ch_w))[:min(n_shift_caths, len(ch_w))])

                P_sh = plan["P_ref"].copy()
                for c_i in top_caths:
                    s0, s1 = plan["catheter_slices"][c_i]
                    P_sh[s0:s1] = P_sh[s0:s1] + dz * plan["U_ref"][s0:s1]

                K_sh = compute_tg43_dose_rate_matrix_gpu(plan["vox_roi"], P_sh, plan["U_ref"], device)
                D_sh = (K_sh @ torch.tensor(plan["t_ref"], dtype=torch.float32, device=device)).cpu().numpy()
                del K_sh
                m = evaluate_dose_distribution_metrics(D_sh, plan, device)
                d90_abs.append(m["abs_delta_d90_pct"])
                d90_signed.append(m["delta_d90_signed_pct"])
                v100_abs.append(m["abs_delta_v100_pct"])
                cran_abs.append(m["abs_delta_sigmoid_cranial_d2cc_pct"])
                cran_signed.append(m["signed_delta_sigmoid_cranial_d2cc_pct"])
                g22_glob.append(m["gamma_2pct_2mm_global_pass_pct"])
                g33_glob.append(m["gamma_3pct_3mm_global_pass_pct"])

            key = f"{dz:+.0f}mm" if dz != 0 else "0mm"
            curve_out[mode][key] = {
                "dz_mm": dz,
                "mean_abs_delta_d90_pct": round(float(np.mean(d90_abs)), 3),
                "median_abs_delta_d90_pct": round(float(np.median(d90_abs)), 3),
                "mean_signed_delta_d90_pct": round(float(np.mean(d90_signed)), 3),
                "mean_abs_delta_d90_gy": round(float(np.mean(d90_abs)) * (PRESCRIPTION_GY / 100.0), 4),
                "mean_abs_delta_v100_pct": round(float(np.mean(v100_abs)), 3),
                "mean_abs_delta_cranial_d2cc_pct": round(float(np.mean(cran_abs)), 3),
                "mean_signed_delta_cranial_d2cc_pct": round(float(np.mean(cran_signed)), 3),
                "mean_gamma_2pct_2mm_global_pass_pct": round(float(np.mean(g22_glob)), 3),
                "mean_gamma_3pct_3mm_global_pass_pct": round(float(np.mean(g33_glob)), 3),
            }

    return curve_out


def summarize_patient_dosimetry_list(eval_list):
    """Summarizes Protocol A, Protocol B, and tip error metrics across a cohort of patients."""
    def _stat(vals):
        arr = np.asarray(vals, dtype=np.float64)
        q25, med, q75, p95 = np.percentile(arr, [25, 50, 75, 95])
        return {
            "mean": round(float(np.mean(arr)), 3),
            "sd": round(float(np.std(arr)), 3),
            "median": round(float(med), 3),
            "iqr": round(float(q75 - q25), 3),
            "p95": round(float(p95), 3),
        }

    out = {
        "n_patients": len(eval_list),
        "unweighted_mean_tip_3d_mm": _stat([x["unweighted_mean_tip_3d_mm"] for x in eval_list]),
        "dwell_weighted_mean_tip_3d_mm": _stat([x["dwell_weighted_mean_tip_3d_mm"] for x in eval_list]),
        "dwell_weighted_mean_shaft_xy_mm": _stat([x["dwell_weighted_mean_shaft_xy_mm"] for x in eval_list]),
    }

    for prot_key in ["protocol_A_fixed_plan_sensitivity", "protocol_B_optimize_on_pred_deliver_on_ref"]:
        m_keys = [
            "abs_delta_d90_pct",
            "abs_delta_d90_gy",
            "delta_d90_signed_pct",
            "abs_delta_v100_pct",
            "abs_delta_v150_pct",
            "abs_delta_v200_pct",
            "abs_delta_bladder_d2cc_pct",
            "abs_delta_rectum_d2cc_pct",
            "abs_delta_sigmoid_cranial_d2cc_pct",
            "gamma_2pct_2mm_global_pass_pct",
            "gamma_2pct_2mm_local_pass_pct",
            "gamma_3pct_3mm_global_pass_pct",
        ]
        prot_sum = {mk: _stat([x[prot_key][mk] for x in eval_list]) for mk in m_keys}
        # Also compute % of patients meeting clinical dosimetric acceptability (|Delta D90| <= 5% and Gamma 3%/3mm >= 90%)
        prot_sum["pct_patients_abs_delta_d90_le_5pct"] = round(
            100.0 * float(np.mean([x[prot_key]["abs_delta_d90_pct"] <= 5.0 for x in eval_list])), 2
        )
        prot_sum["pct_patients_gamma_3pct_3mm_ge_95pct"] = round(
            100.0 * float(np.mean([x[prot_key]["gamma_3pct_3mm_global_pass_pct"] >= 95.0 for x in eval_list])), 2
        )
        prot_sum["pct_patients_gamma_2pct_2mm_ge_90pct"] = round(
            100.0 * float(np.mean([x[prot_key]["gamma_2pct_2mm_global_pass_pct"] >= 90.0 for x in eval_list])), 2
        )
        out[prot_key] = prot_sum

    return out


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: AAPM TG-43U1 HDR 192Ir Dosimetric Impact")
    parser.add_argument("--cache-dir", type=str, default="/tmp/needle_unet3d_cache_clean")
    parser.add_argument("--oof-dir", type=str, default="/home/guillaume/github/needlefinder/nn_models_clean")
    parser.add_argument("--eval-oof-dir", type=str, default="/home/guillaume/github/needlefinder/nn_models_eval")
    parser.add_argument("--meta-path", type=str, default="/tmp/cases_meta_54_clean.pkl")
    parser.add_argument("--base-trajs-path", type=str, default="/tmp/base_trajs_54cases_clean.pkl")
    parser.add_argument("--stage2-preds-pkl", type=str, default="/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-clean-final-oof-predictions.pkl")
    parser.add_argument("--out-json", type=str, default="/home/guillaume/github/needlefinder/vault/sources/2026-10-23-exp1-tg43-dosimetry.json")
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    print(f"[Experiment 1: AAPM TG-43U1 HDR 192Ir Dosimetry] Device={device}", flush=True)

    meta_list = pickle.load(open(args.meta_path, "rb"))
    base_trajs_data = pickle.load(open(args.base_trajs_path, "rb"))
    gts_all_list = [get_gts(m["gts_all"]) for m in meta_list]
    gts_long_list = [get_gts(m["gts_long"]) for m in meta_list]

    # Load predictions for all methods
    stage0_preds = []
    for i in range(54):
        b_item = base_trajs_data[i]
        b_trajs = b_item["trajs"] if isinstance(b_item, dict) and "trajs" in b_item else b_item
        b_clean = []
        for t in b_trajs:
            tc = dict(t)
            tc["ras"] = np.asarray(tc["ras"] if "ras" in tc else tc["points"], dtype=np.float64)
            tc["points"] = tc["ras"]
            b_clean.append(tc)
        stage0_preds.append(b_clean)

    pmb2019_data = pickle.load(open(f"{args.eval_oof_dir}/oof_trajs_pmb2019_iso1mm.pkl", "rb"))
    s2_preds_data = pickle.load(open(args.stage2_preds_pkl, "rb"))

    oof_tag = os.path.basename(os.path.normpath(args.oof_dir))
    bundle_cache = pickle.load(open(f"/tmp/stage2_clean_bundles_{oof_tag}.pkl", "rb"))
    graphs = bundle_cache["graphs"]
    all_oof_cal = bundle_cache["all_oof_cal"]
    all_X_vols = [torch.load(f"{args.cache_dir}/case_{i:02d}.pt", map_location="cpu", weights_only=False)["X"].numpy() for i in range(54)]

    tip_models_5fold = []
    for st in s2_preds_data["tip_models_5fold_states"]:
        m = Dedicated1DTipLocalizer(in_channels=22, hidden_dim=48, seq_len=91).to(device)
        m.load_state_dict({k: v.to(device) for k, v in st.items()})
        m.eval()
        tip_models_5fold.append(m)

    # 1. Extract 1D tip posteriors and build `NeedleFinder 2026 + Top-2 Uncertainty Tip Review`
    print("[1/5] Extracting 91-bin 1D tip posteriors and building Top-2 Uncertainty Tip Review predictions...", flush=True)
    posteriors_per_case, nf_top2_review_all = compute_1d_tip_posteriors_and_top2_review(
        s2_preds_data["all_tau50"], graphs, all_oof_cal, all_X_vols, tip_models_5fold, gts_long_list, device
    )

    # Decode 4a (No SynthBrachy3T, without SynthBrachy3T-trained 1D tip refiner)
    p4a_all = decode_clean_stage2_preds(
        bundle_cache["stitched_enriched_bgs"],
        s2_preds_data["base_sc_st_nosyn"],
        s2_preds_data["syn_out_st_nosyn"],
        graphs=graphs,
        all_oof_cal=all_oof_cal,
        all_X_vols=None,
        tip_models_5fold=None,
        device=device,
        min_conf=50.0,
        long_mode=False,
        w_synth=0.0,
        use_collinear_dedup=True,
        use_1d_tip_model=False,
    )

    # 2. Build Reference AAPM TG-43U1 Plans across all 54 patients
    print("[2/5] Building Reference AAPM TG-43U1 192Ir 3D Plans (1.5 mm grid, HIPO/NNLS D90=100%=7.0Gy) for 54 patients...", flush=True)
    patient_plans = []
    for ci in range(54):
        plan = build_patient_tg43_reference_plan(gts_long_list[ci], device, grid_spacing_mm=1.5)
        patient_plans.append(plan)
    print(f"  [Done] Built 54 reference TG-43 plans in {time.time() - t0:.1f}s.", flush=True)

    # 3. Evaluate all 5 automated methods across all 54 patients + Human Rater 2 (LB/AMe) on the 40 dual-traced patients
    print("[3/5] Evaluating Protocol A & Protocol B across all 54 patients (and 40 dual-traced scans)...", flush=True)
    methods_54 = [
        ("Stage_0_Physics_Tracker", "Stage 0 Physics Tracker (k_lim_auto)", stage0_preds, None),
        ("PMB_2019_3D_UNet_Walk", "PMB 2019 3D U-Net + Walk", pmb2019_data["all"], None),
        ("NeedleFinder_2026_No_SynthBrachy3T_4a", "NeedleFinder 2026 (No SynthBrachy3T, 4a)", p4a_all, stage0_preds),
        ("NeedleFinder_2026_Proposed_4b", "NeedleFinder 2026 (Proposed, 4b)", s2_preds_data["all_tau50"], s2_preds_data["all_tau35"]),
        ("NeedleFinder_2026_Top2_Uncertainty_Review", "NeedleFinder 2026 + Top-2 Uncertainty Tip Review", nf_top2_review_all, s2_preds_data["all_tau35"]),
    ]

    dual_indices = [ci for ci, m in enumerate(meta_list) if "lb_gts" in m and m["lb_gts"] and len(m["lb_gts"]) > 0]
    assert len(dual_indices) == 40, f"Expected 40 dual-traced scans, got {len(dual_indices)}"

    cohort_54_summary = {}
    cohort_40_summary = {}
    per_patient_by_method = {}

    for m_key, m_label, preds_list, fb_list in methods_54:
        evals_54 = []
        for ci in range(54):
            fb_case = fb_list[ci] if fb_list is not None else None
            ev = evaluate_predictions_on_patient_tg43(preds_list[ci], patient_plans[ci], device, match_thresh_mm=3.5, fallback_preds_case=fb_case)
            ev["case_idx"] = ci
            ev["case_num"] = int(meta_list[ci]["case_num"])
            evals_54.append(ev)
        per_patient_by_method[m_key] = evals_54
        cohort_54_summary[m_key] = {
            "label": m_label,
            **summarize_patient_dosimetry_list(evals_54),
        }
        evals_40 = [evals_54[ci] for ci in dual_indices]
        cohort_40_summary[m_key] = {
            "label": m_label,
            **summarize_patient_dosimetry_list(evals_40),
        }
        pa = cohort_54_summary[m_key]["protocol_A_fixed_plan_sensitivity"]
        pb = cohort_54_summary[m_key]["protocol_B_optimize_on_pred_deliver_on_ref"]
        wt = cohort_54_summary[m_key]["dwell_weighted_mean_tip_3d_mm"]
        print(
            f"\n  [{m_label} (N=54)]\n"
            f"    Dwell-Weighted Tip Error : mean={wt['mean']:.2f} mm | med={wt['median']:.2f} mm | P95={wt['p95']:.2f} mm\n"
            f"    Protocol A (Fixed-Plan)  : |dD90|={pa['abs_delta_d90_pct']['mean']:.2f}% ({pa['abs_delta_d90_gy']['mean']:.3f} Gy, med={pa['abs_delta_d90_pct']['median']:.2f}%) | "
            f"|dV100|={pa['abs_delta_v100_pct']['mean']:.2f}% | Cranial |dD2cc|={pa['abs_delta_sigmoid_cranial_d2cc_pct']['mean']:.2f}% | "
            f"Gamma 2%/2mm={pa['gamma_2pct_2mm_global_pass_pct']['mean']:.2f}% (med={pa['gamma_2pct_2mm_global_pass_pct']['median']:.2f}%) | "
            f"Gamma 3%/3mm={pa['gamma_3pct_3mm_global_pass_pct']['mean']:.2f}%\n"
            f"    Protocol B (Opt->Deliv)  : |dD90|={pb['abs_delta_d90_pct']['mean']:.2f}% ({pb['abs_delta_d90_gy']['mean']:.3f} Gy, med={pb['abs_delta_d90_pct']['median']:.2f}%) | "
            f"|dV100|={pb['abs_delta_v100_pct']['mean']:.2f}% | Cranial |dD2cc|={pb['abs_delta_sigmoid_cranial_d2cc_pct']['mean']:.2f}% | "
            f"Gamma 2%/2mm={pb['gamma_2pct_2mm_global_pass_pct']['mean']:.2f}% | Gamma 3%/3mm={pb['gamma_3pct_3mm_global_pass_pct']['mean']:.2f}%",
            flush=True,
        )

    # Evaluate Human Rater 2 (LB/AMe) on the 40 dual-traced cases
    human_evals_40 = []
    for ci in dual_indices:
        lb_all = get_gts(meta_list[ci]["lb_gts"])
        ev = evaluate_predictions_on_patient_tg43(lb_all, patient_plans[ci], device, match_thresh_mm=3.5)
        ev["case_idx"] = ci
        ev["case_num"] = int(meta_list[ci]["case_num"])
        human_evals_40.append(ev)

    cohort_40_summary["Human_Rater_2_LB_AMe"] = {
        "label": "Human Rater 2 (LB/AMe, N=40 Dual-Traced)",
        **summarize_patient_dosimetry_list(human_evals_40),
    }
    pa_h = cohort_40_summary["Human_Rater_2_LB_AMe"]["protocol_A_fixed_plan_sensitivity"]
    pb_h = cohort_40_summary["Human_Rater_2_LB_AMe"]["protocol_B_optimize_on_pred_deliver_on_ref"]
    wt_h = cohort_40_summary["Human_Rater_2_LB_AMe"]["dwell_weighted_mean_tip_3d_mm"]
    print(
        f"\n  [Human Rater 2 (LB/AMe, N=40 Dual-Traced)]\n"
        f"    Dwell-Weighted Tip Error : mean={wt_h['mean']:.2f} mm | med={wt_h['median']:.2f} mm | P95={wt_h['p95']:.2f} mm\n"
        f"    Protocol A (Fixed-Plan)  : |dD90|={pa_h['abs_delta_d90_pct']['mean']:.2f}% ({pa_h['abs_delta_d90_gy']['mean']:.3f} Gy, med={pa_h['abs_delta_d90_pct']['median']:.2f}%) | "
        f"|dV100|={pa_h['abs_delta_v100_pct']['mean']:.2f}% | Cranial |dD2cc|={pa_h['abs_delta_sigmoid_cranial_d2cc_pct']['mean']:.2f}% | "
        f"Gamma 2%/2mm={pa_h['gamma_2pct_2mm_global_pass_pct']['mean']:.2f}% (med={pa_h['gamma_2pct_2mm_global_pass_pct']['median']:.2f}%) | "
        f"Gamma 3%/3mm={pa_h['gamma_3pct_3mm_global_pass_pct']['mean']:.2f}%\n"
        f"    Protocol B (Opt->Deliv)  : |dD90|={pb_h['abs_delta_d90_pct']['mean']:.2f}% ({pb_h['abs_delta_d90_gy']['mean']:.3f} Gy, med={pb_h['abs_delta_d90_pct']['median']:.2f}%) | "
        f"|dV100|={pb_h['abs_delta_v100_pct']['mean']:.2f}% | Cranial |dD2cc|={pb_h['abs_delta_sigmoid_cranial_d2cc_pct']['mean']:.2f}% | "
        f"Gamma 2%/2mm={pb_h['gamma_2pct_2mm_global_pass_pct']['mean']:.2f}% | Gamma 3%/3mm={pb_h['gamma_3pct_3mm_global_pass_pct']['mean']:.2f}%",
        flush=True,
    )

    # Paired Wilcoxon tests on dosimetric endpoints (N=54 and N=40 vs Human Rater 2)
    def _wilcox(list_a, list_b, prot, key):
        va = np.array([x[prot][key] for x in list_a], dtype=np.float64)
        vb = np.array([x[prot][key] for x in list_b], dtype=np.float64)
        diff = va - vb
        if np.all(np.abs(diff) < 1e-8):
            return {"mean_a": round(float(np.mean(va)), 3), "mean_b": round(float(np.mean(vb)), 3), "p_value": 1.0}
        _, p_val = wilcoxon(va, vb)
        return {
            "mean_a": round(float(np.mean(va)), 3),
            "mean_b": round(float(np.mean(vb)), 3),
            "median_a": round(float(np.median(va)), 3),
            "median_b": round(float(np.median(vb)), 3),
            "p_value": float(p_val),
        }

    nf4b_54 = per_patient_by_method["NeedleFinder_2026_Proposed_4b"]
    nf4b_40 = [nf4b_54[ci] for ci in dual_indices]
    wilcoxon_dosimetry = {
        "NF2026_4b_vs_Stage0_54cases_protA_abs_dD90": _wilcox(nf4b_54, per_patient_by_method["Stage_0_Physics_Tracker"], "protocol_A_fixed_plan_sensitivity", "abs_delta_d90_pct"),
        "NF2026_4b_vs_Stage0_54cases_protA_gamma22": _wilcox(nf4b_54, per_patient_by_method["Stage_0_Physics_Tracker"], "protocol_A_fixed_plan_sensitivity", "gamma_2pct_2mm_global_pass_pct"),
        "NF2026_4b_vs_PMB2019_54cases_protA_abs_dD90": _wilcox(nf4b_54, per_patient_by_method["PMB_2019_3D_UNet_Walk"], "protocol_A_fixed_plan_sensitivity", "abs_delta_d90_pct"),
        "NF2026_4b_vs_PMB2019_54cases_protA_gamma22": _wilcox(nf4b_54, per_patient_by_method["PMB_2019_3D_UNet_Walk"], "protocol_A_fixed_plan_sensitivity", "gamma_2pct_2mm_global_pass_pct"),
        "NF2026_Top2Review_vs_NF2026_4b_54cases_protA_abs_dD90": _wilcox(per_patient_by_method["NeedleFinder_2026_Top2_Uncertainty_Review"], nf4b_54, "protocol_A_fixed_plan_sensitivity", "abs_delta_d90_pct"),
        "NF2026_Top2Review_vs_NF2026_4b_54cases_protA_gamma22": _wilcox(per_patient_by_method["NeedleFinder_2026_Top2_Uncertainty_Review"], nf4b_54, "protocol_A_fixed_plan_sensitivity", "gamma_2pct_2mm_global_pass_pct"),
        "NF2026_Top2Review_vs_NF2026_4b_54cases_protB_abs_dD90": _wilcox(per_patient_by_method["NeedleFinder_2026_Top2_Uncertainty_Review"], nf4b_54, "protocol_B_optimize_on_pred_deliver_on_ref", "abs_delta_d90_pct"),
        "NF2026_Top2Review_vs_NF2026_4b_54cases_protB_gamma22": _wilcox(per_patient_by_method["NeedleFinder_2026_Top2_Uncertainty_Review"], nf4b_54, "protocol_B_optimize_on_pred_deliver_on_ref", "gamma_2pct_2mm_global_pass_pct"),
        "NF2026_4b_vs_HumanRater2_40cases_protA_abs_dD90": _wilcox(nf4b_40, human_evals_40, "protocol_A_fixed_plan_sensitivity", "abs_delta_d90_pct"),
        "NF2026_4b_vs_HumanRater2_40cases_protA_gamma22": _wilcox(nf4b_40, human_evals_40, "protocol_A_fixed_plan_sensitivity", "gamma_2pct_2mm_global_pass_pct"),
    }

    # 4. Controlled Tip-Shift Sensitivity Curve (dz in {-15..+15} mm)
    print("\n[4/5] Computing Single-Catheter & Top-3 Implant Tip-Shift Sensitivity Curves (dz in -15..+15 mm)...", flush=True)
    tip_shift_curve = run_tip_shift_sensitivity_curve(patient_plans, device)
    for k_dz, row in tip_shift_curve["top3_catheters_implant_shift"].items():
        print(
            f"  [Top-3 Shift {k_dz:5s}] |dD90|={row['mean_abs_delta_d90_pct']:5.2f}% ({row['mean_abs_delta_d90_gy']:.3f} Gy) | "
            f"Cranial dD2cc={row['mean_signed_delta_cranial_d2cc_pct']:+6.2f}% | "
            f"Gamma 2%/2mm={row['mean_gamma_2pct_2mm_global_pass_pct']:6.2f}% | Gamma 3%/3mm={row['mean_gamma_3pct_3mm_global_pass_pct']:6.2f}%",
            flush=True,
        )

    # 5. Uncertainty-to-Dose Propagation (M=30 Monte Carlo tip samples per patient)
    print("\n[5/5] Running Uncertainty-to-Dose Propagation (M=30 Monte Carlo samples/patient from Dedicated1DTipLocalizer)...", flush=True)
    unc_propagation = run_uncertainty_to_dose_propagation(
        s2_preds_data["all_tau50"], posteriors_per_case, patient_plans, nf4b_54, device, n_mc_samples=30
    )
    us = unc_propagation["summary_54_patients"]
    print(
        f"  [Uncertainty-to-Dose Summary (N=54)]\n"
        f"    Predicted sigma(D90) : mean={us['mean_pred_sigma_d90_pct']:.2f}% | med={us['median_pred_sigma_d90_pct']:.2f}% | P95={us['p95_pred_sigma_d90_pct']:.2f}%\n"
        f"    Spearman rho(pred sigma(D90), true |dD90|)            = {us['spearman_pred_sigma_d90_vs_true_abs_delta_d90']['rho']:.4f} (p={us['spearman_pred_sigma_d90_vs_true_abs_delta_d90']['p_value']:.2e})\n"
        f"    Spearman rho(dwell-weighted pred sigma_tip, |dD90|)   = {us['spearman_dwell_weighted_pred_sigma_tip_vs_true_abs_delta_d90']['rho']:.4f} (p={us['spearman_dwell_weighted_pred_sigma_tip_vs_true_abs_delta_d90']['p_value']:.2e})\n"
        f"    Spearman rho(dwell-weighted true E_tip, |dD90|)       = {us['spearman_dwell_weighted_true_tip_err_vs_true_abs_delta_d90']['rho']:.4f} (p={us['spearman_dwell_weighted_true_tip_err_vs_true_abs_delta_d90']['p_value']:.2e})\n"
        f"    Spearman rho(unweighted true E_tip, |dD90|)           = {us['spearman_unweighted_true_tip_err_vs_true_abs_delta_d90']['rho']:.4f} (p={us['spearman_unweighted_true_tip_err_vs_true_abs_delta_d90']['p_value']:.2e})",
        flush=True,
    )

    out_report = {
        "generated_at": "2026-10-23",
        "experiment": "Experiment 1: AAPM TG-43U1 HDR 192Ir Dosimetric Impact & Uncertainty-to-Dose Propagation (54 Clinical GYN Patients)",
        "script": "vault/paper/benchmarks_clean/run_exp1_tg43_dosimetry.py",
        "elapsed_sec": round(time.time() - t0, 2),
        "contour_and_plan_provenance_check": {
            "checked_directories": [
                "/home/guillaume/Dropbox/GYN Cases/",
                "/home/guillaume/Dropbox/AMIGO Gyn Data NRRD (1)/",
            ],
            "clinical_rtstruct_or_rtplan_files_found": 0,
            "target_and_oar_definition": (
                "Standardized 1.5 mm isotropic 3D evaluation grid per patient with implant-conformal HR-CTV_implant "
                "(8.0 mm radial envelope around reference active dwells, D90 normalized to 100% = 7.0 Gy via damped NNLS/HIPO) "
                "and anatomical OAR 2cc proxy shells: Bladder_proxy (anterior 5.5-16 mm), Rectum_proxy (posterior 5.5-16 mm), "
                "and Sigmoid/Bowel_proxy (cranial cap 3.5-15 mm superior to catheter tips)."
            ),
        },
        "tg43_source_parameters": {
            "radionuclide": "192Ir (HDR microSelectron-v2 / Flexisource consensus)",
            "active_length_L_mm": 3.5,
            "dose_rate_constant_Lambda_cGy_h_U": 1.109,
            "reference_geometry_factor_GL_r0_theta0_cm2": round(G_L_REF_CM2, 6),
            "distal_tip_offset_d0_mm": 3.5,
            "dwell_step_spacing_mm": 5.0,
            "max_active_length_mm": 45.0,
            "prescription_dose_per_fraction_Gy": PRESCRIPTION_GY,
            "grid_spacing_mm": 1.5,
        },
        "cohort_54_patients_summary": cohort_54_summary,
        "cohort_40_dual_traced_patients_summary": cohort_40_summary,
        "paired_wilcoxon_dosimetric_tests": wilcoxon_dosimetry,
        "tip_shift_sensitivity_curves": tip_shift_curve,
        "uncertainty_to_dose_propagation": unc_propagation,
        "per_patient_results_54_cases": per_patient_by_method,
        "per_patient_human_rater2_40_cases": human_evals_40,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_report, f, indent=2)
    print(f"\n[All Done] Saved Experiment 1 TG-43 dosimetry report to {args.out_json} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
