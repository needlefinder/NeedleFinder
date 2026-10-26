#!/usr/bin/env python3
"""
Train & Evaluate Stage 1 (3D Multi-Head Residual U-Net: Labelmap + Sharp Core + 3D Tangent Field)
and Stage 2 (Tangent-Guided 3D Needle Trajectory Reconstructor) on edelweiss GPUs.

Evaluates both:
1. Official 54-case CY benchmark (759 long needles)
2. Full 54-case Multi-Observer benchmark (1,079 needles: 783 CY long + 296 HT short obturator needles)
3. Full-FOV deep scan benchmark (case.nrrd / Case 016 NRRD 640x416x144 with k_lim=16 and k_lim=92)
Exports the trained model to ONNX (`needle_unet3d.onnx`) for sub-second inference in 3D Slicer.
"""

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path

import glob
import math
import os
import pickle
import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder"))
from prepare_and_inspect_dataset import (
    load_all_dataset_cases,
    read_nrrd_local,
    read_vtk_centerline_local,
    rust_compute_3d_sym_response,
)


# ==============================================================================
# 1. 3D MULTI-HEAD RESIDUAL U-NET ARCHITECTURE (Stage 1: Labelmap + Tangent Field)
# ==============================================================================

class AnisoResBlock3D(nn.Module):
    """Anisotropic 3D Residual Block with GroupNorm and SiLU."""
    def __init__(self, in_ch, out_ch, z_kernel=3):
        super().__init__()
        zk = z_kernel
        zp = zk // 2
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=(zk, 3, 3), padding=(zp, 1, 1), bias=False)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=(zk, 3, 3), padding=(zp, 1, 1), bias=False)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + residual)


class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation channel attention."""
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(4, ch // reduction)
        self.fc1 = nn.Conv3d(ch, mid, kernel_size=1)
        self.fc2 = nn.Conv3d(mid, ch, kernel_size=1)

    def forward(self, x):
        w = x.mean(dim=(2, 3, 4), keepdim=True)
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(w))))
        return x * w


class NeedleUNet3D(nn.Module):
    """
    3D Multi-Head Residual U-Net for Interstitial Brachytherapy Needle Segmentation.
    Input:  [B, 6, K, H, W]
      0: norm_mri
      1: norm_rsym_resp
      2: norm_ring_cs
      3: dark_contrast
      4: z_rel_norm
      5: radial_dist_norm
    Output: [B, 5, K, H, W]
      0: logit_mask  (binary/soft needle labelmap, r <= 1.25 mm)
      1: logit_peak  (sharp sub-voxel centerline Gaussian, sigma = 0.70 mm)
      2: vx          (physical slope dx/dz in mm/mm)
      3: vy          (physical slope dy/dz in mm/mm)
      4: logit_obt   (short/obturator needle probability map)
    """
    def __init__(self, in_ch=6, base_ch=24):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 6

        self.enc1 = nn.Sequential(
            AnisoResBlock3D(in_ch, c1, z_kernel=3),
            AnisoResBlock3D(c1, c1, z_kernel=3),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc2 = nn.Sequential(
            AnisoResBlock3D(c1, c2, z_kernel=3),
            SEBlock3D(c2),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc3 = nn.Sequential(
            AnisoResBlock3D(c2, c3, z_kernel=3),
            SEBlock3D(c3),
        )
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Bottleneck with dilated in-plane & axial context
        self.bottleneck = nn.Sequential(
            AnisoResBlock3D(c3, c4, z_kernel=3),
            nn.Conv3d(c4, c4, kernel_size=(3, 3, 3), padding=(1, 2, 2), dilation=(1, 2, 2), bias=False),
            nn.GroupNorm(min(8, c4), c4),
            nn.SiLU(inplace=True),
            SEBlock3D(c4),
        )

        self.dec3 = AnisoResBlock3D(c4 + c3, c3, z_kernel=3)
        self.dec2 = AnisoResBlock3D(c3 + c2, c2, z_kernel=3)
        self.dec1 = nn.Sequential(
            AnisoResBlock3D(c2 + c1, c1, z_kernel=3),
            AnisoResBlock3D(c1, c1, z_kernel=3),
        )

        self.head = nn.Conv3d(c1, 5, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        u3 = F.interpolate(b, size=e3.shape[2:], mode="trilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(d3, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.head(d1)
        return out


# ==============================================================================
# 2. FEATURE & GROUND-TRUTH TARGET GENERATION FOR EACH 3D VOLUME
# ==============================================================================

def build_case_tensors(case_dict):
    """
    Extract 6-channel 3D input tensor X [6, K_sub, H, W] and
    5-channel 3D ground-truth target tensor Y [5, K_sub, H, W] over the pelvic ROI.
    """
    vol = case_dict["vol"]
    K, J, I = vol.shape
    sp_i, sp_j, sp_k = float(case_dict["sp_ras"][0]), float(case_dict["sp_ras"][1]), float(case_dict["sp_ras"][2])
    ijk_to_ras = case_dict["ijk_to_ras"]
    ras_to_ijk = np.linalg.inv(ijk_to_ras)
    rot_mat = ijk_to_ras[:3, :3]
    trans_vec = ijk_to_ras[:3, 3]

    k0 = int(np.clip(case_dict["k_lim_gt"] - 1, 0, max(0, K - 5)))
    K_sub = K - k0

    # Pelvic ROI crop (~130x130 mm around image center, rounded to multiple of 8)
    margin_i = min(I // 2 - 4, int(round(68.0 / sp_i)))
    margin_j = min(J // 2 - 4, int(round(68.0 / sp_j)))
    half_w = ((margin_i + 7) // 8) * 8
    half_h = ((margin_j + 7) // 8) * 8
    ci, cj = I // 2, J // 2
    i0 = max(0, ci - half_w)
    i1 = min(I, i0 + 2 * half_w)
    if (i1 - i0) % 8 != 0:
        i1 = i0 + ((i1 - i0) // 8) * 8
    j0 = max(0, cj - half_h)
    j1 = min(J, j0 + 2 * half_h)
    if (j1 - j0) % 8 != 0:
        j1 = j0 + ((j1 - j0) // 8) * 8

    H, W = j1 - j0, i1 - i0

    sub_roi = vol[k0:K, j0:j1, i0:i1]
    non_zero = sub_roi[sub_roi > 0]
    body_thresh = float(np.percentile(non_zero, 15)) if len(non_zero) > 100 else 10.0
    p10 = float(np.percentile(non_zero, 10)) if len(non_zero) > 100 else 0.0
    p95 = float(np.percentile(non_zero, 95)) if len(non_zero) > 100 else 100.0

    res = rust_compute_3d_sym_response(vol, sp_i, sp_j, i0, i1, j0, j1, k0, K, body_thresh)
    resp_full, cs_full, rsym_full = res
    resp = resp_full[k0:K, j0:j1, i0:i1]
    cs = cs_full[k0:K, j0:j1, i0:i1]
    rsym = rsym_full[k0:K, j0:j1, i0:i1]

    # Channel 0: Normalized MRI
    ch0 = np.clip((sub_roi - p10) / max(1.0, p95 - p10), -0.5, 2.5).astype(np.float32)

    # Channel 1: Log-normalized 3D Radial Symmetry Response
    pos_r = resp[resp > 0]
    r_scale = float(np.percentile(pos_r, 99)) if len(pos_r) > 20 else 50.0
    ch1 = np.clip(np.log1p(np.maximum(0.0, resp)) / max(1.0, math.log1p(r_scale)), 0.0, 1.5).astype(np.float32)

    # Channel 2: Normalized Ring Contrast
    pos_c = cs[cs > 0]
    c_scale = float(np.percentile(pos_c, 99)) if len(pos_c) > 20 else 25.0
    ch2 = np.clip(cs / max(1.0, c_scale), 0.0, 1.5).astype(np.float32)

    # Channel 3: Darkness ratio relative to ring background
    ch3 = np.clip((rsym - sub_roi) / (rsym + 15.0), -0.5, 1.0).astype(np.float32)

    # Channel 4: Normalized Z height above k_lim
    ks_rel = np.arange(K_sub, dtype=np.float32)[:, None, None]
    ch4 = np.broadcast_to((ks_rel * sp_k) / 120.0, (K_sub, H, W)).copy()

    # Channel 5: Radial distance from ROI center in mm / 50.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy_loc, cx_loc = H * 0.5, W * 0.5
    r_mm = np.hypot((xx - cx_loc) * sp_i, (yy - cy_loc) * sp_j) / 50.0
    ch5 = np.broadcast_to(r_mm[None, :, :], (K_sub, H, W)).copy()

    X = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0)  # [6, K_sub, H, W]

    # Build Ground-Truth Targets Y [5, K_sub, H, W]
    target_mask = np.zeros((K_sub, H, W), dtype=np.float32)
    target_peak = np.zeros((K_sub, H, W), dtype=np.float32)
    target_vx = np.zeros((K_sub, H, W), dtype=np.float32)
    target_vy = np.zeros((K_sub, H, W), dtype=np.float32)
    target_obt = np.zeros((K_sub, H, W), dtype=np.float32)
    min_dist_map = np.full((K_sub, H, W), 999.0, dtype=np.float32)

    ri_pad = int(math.ceil(2.5 / sp_i))
    rj_pad = int(math.ceil(2.5 / sp_j))

    for g in case_dict["gts_all"]:
        pts_ras = g["pts"]  # sorted by Z
        is_obt = 1.0 if g["type"] == "ht_obturator" else 0.35
        # Convert RAS points to IJK
        pts_h = np.column_stack([pts_ras, np.ones(len(pts_ras))])
        pts_ijk = (ras_to_ijk @ pts_h.T).T[:, :3]
        order = np.argsort(pts_ijk[:, 2])
        pts_ijk = pts_ijk[order]
        pts_ras_s = pts_ras[order]

        k_min_f = float(pts_ijk[0, 2])
        k_max_f = float(pts_ijk[-1, 2])
        kr_start = max(0, int(math.floor(k_min_f)) - k0)
        kr_end = min(K_sub - 1, int(math.ceil(k_max_f)) - k0)
        if kr_end < kr_start:
            continue

        # Compute physical slope (dx/dz, dy/dz) along the needle
        dz_total = float(pts_ras_s[-1, 2] - pts_ras_s[0, 2])
        if abs(dz_total) < 1.0:
            continue

        for kr in range(kr_start, kr_end + 1):
            k_abs = float(kr + k0)
            ic = float(np.interp(k_abs, pts_ijk[:, 2], pts_ijk[:, 0])) - i0
            jc = float(np.interp(k_abs, pts_ijk[:, 2], pts_ijk[:, 1])) - j0
            if not (0 <= ic < W and 0 <= jc < H):
                continue

            # Local physical slope dx/dz, dy/dz around k_abs
            k_lo = max(float(pts_ijk[0, 2]), k_abs - 2.0)
            k_hi = min(float(pts_ijk[-1, 2]), k_abs + 2.0)
            if k_hi <= k_lo + 0.2:
                vx = float(pts_ras_s[-1, 0] - pts_ras_s[0, 0]) / dz_total
                vy = float(pts_ras_s[-1, 1] - pts_ras_s[0, 1]) / dz_total
            else:
                x_lo = float(np.interp(k_lo, pts_ijk[:, 2], pts_ras_s[:, 0]))
                x_hi = float(np.interp(k_hi, pts_ijk[:, 2], pts_ras_s[:, 0]))
                y_lo = float(np.interp(k_lo, pts_ijk[:, 2], pts_ras_s[:, 1]))
                y_hi = float(np.interp(k_hi, pts_ijk[:, 2], pts_ras_s[:, 1]))
                z_lo = float(np.interp(k_lo, pts_ijk[:, 2], pts_ras_s[:, 2]))
                z_hi = float(np.interp(k_hi, pts_ijk[:, 2], pts_ras_s[:, 2]))
                dz_loc = z_hi - z_lo if abs(z_hi - z_lo) > 0.5 else dz_total
                vx = float(np.clip((x_hi - x_lo) / dz_loc, -0.75, 0.75))
                vy = float(np.clip((y_hi - y_lo) / dz_loc, -0.75, 0.75))

            ic_int = int(round(ic))
            jc_int = int(round(jc))
            i_lo = max(0, ic_int - ri_pad)
            i_hi = min(W, ic_int + ri_pad + 1)
            j_lo = max(0, jc_int - rj_pad)
            j_hi = min(H, jc_int + rj_pad + 1)

            jj_p, ii_p = np.mgrid[j_lo:j_hi, i_lo:i_hi].astype(np.float32)
            d_mm = np.hypot((ii_p - ic) * sp_i, (jj_p - jc) * sp_j)

            # Mask target: 1.0 inside 1.15 mm, smooth falloff to 0 at 1.90 mm
            m_val = np.clip((1.90 - d_mm) / (1.90 - 1.15), 0.0, 1.0)
            # Sharp Gaussian peak target (sigma = 0.70 mm) for instance separation
            p_val = np.exp(-0.5 * (d_mm / 0.70) ** 2)

            target_mask[kr, j_lo:j_hi, i_lo:i_hi] = np.maximum(target_mask[kr, j_lo:j_hi, i_lo:i_hi], m_val)
            target_peak[kr, j_lo:j_hi, i_lo:i_hi] = np.maximum(target_peak[kr, j_lo:j_hi, i_lo:i_hi], p_val)

            closer = (d_mm < min_dist_map[kr, j_lo:j_hi, i_lo:i_hi]) & (d_mm <= 1.85)
            min_dist_map[kr, j_lo:j_hi, i_lo:i_hi] = np.minimum(min_dist_map[kr, j_lo:j_hi, i_lo:i_hi], d_mm)
            target_vx[kr, j_lo:j_hi, i_lo:i_hi][closer] = vx
            target_vy[kr, j_lo:j_hi, i_lo:i_hi][closer] = vy
            if is_obt > 0.5:
                target_obt[kr, j_lo:j_hi, i_lo:i_hi] = np.maximum(target_obt[kr, j_lo:j_hi, i_lo:i_hi], m_val)

    Y = np.stack([target_mask, target_peak, target_vx, target_vy, target_obt], axis=0)
    meta = {
        "case_num": case_dict["case_num"],
        "case_id": case_dict["case_id"],
        "K": K, "J": J, "I": I,
        "k0": k0, "K_sub": K_sub,
        "i0": i0, "i1": i1, "j0": j0, "j1": j1,
        "H": H, "W": W,
        "sp_i": sp_i, "sp_j": sp_j, "sp_k": sp_k,
        "ijk_to_ras": ijk_to_ras,
        "ras_to_ijk": ras_to_ijk,
        "rot_mat": rot_mat,
        "trans_vec": trans_vec,
        "resp": resp.astype(np.float32),
        "cs": cs.astype(np.float32),
        "rsym": rsym.astype(np.float32),
        "gts_long": case_dict["gts_long"],
        "gts_ht": case_dict["gts_ht"],
        "gts_all": case_dict["gts_all"],
    }
    return X, Y, meta


# ==============================================================================
# 3. MULTI-TASK TOPOLOGICAL + HARD-NEGATIVE LOSS FUNCTION
# ==============================================================================

def compute_multitask_loss(out, target, x_in):
    """
    out:    [B, 5, K, H, W] -> (logit_mask, logit_peak, vx, vy, logit_obt)
    target: [B, 5, K, H, W] -> (t_mask, t_peak, t_vx, t_vy, t_obt)
    x_in:   [B, 6, K, H, W] -> x_in[:, 1] is normalized rsym response (for hard-negative mining)
    """
    logit_mask = out[:, 0]
    logit_peak = out[:, 1]
    pred_vx = torch.tanh(out[:, 2]) * 0.75
    pred_vy = torch.tanh(out[:, 3]) * 0.75
    logit_obt = out[:, 4]

    t_mask = target[:, 0]
    t_peak = target[:, 1]
    t_vx = target[:, 2]
    t_vy = target[:, 3]
    t_obt = target[:, 4]
    rsym_norm = x_in[:, 1]

    # 1. Weighted BCE + Hard-Negative Focal Weight on Mask
    prob_mask = torch.sigmoid(logit_mask)
    bce_raw = F.binary_cross_entropy_with_logits(logit_mask, t_mask, reduction="none")
    pt = torch.where(t_mask >= 0.5, prob_mask, 1.0 - prob_mask)
    focal_w = (1.0 - pt).pow(2.0)
    # Boost positive needle voxels (18x) and hard-negative non-needle dark voids (5x where rsym is high but t_mask == 0)
    pos_w = 1.0 + 18.0 * t_mask + 5.0 * (1.0 - t_mask) * torch.clamp(rsym_norm, 0.0, 1.0)
    loss_bce = (bce_raw * focal_w * pos_w).mean()

    # 2. Soft Dice Loss on Mask
    inter = (prob_mask * t_mask).sum(dim=(1, 2, 3))
    union = prob_mask.sum(dim=(1, 2, 3)) + t_mask.sum(dim=(1, 2, 3))
    loss_dice = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()

    # 3. Z-Continuity Topological Loss (soft-clDice along Z on centerline peaks)
    prob_peak = torch.sigmoid(logit_peak)
    peak_err = (prob_peak - t_peak).abs()
    peak_w = 1.0 + 25.0 * t_peak + 4.0 * (1.0 - (t_mask > 0.1).float()) * torch.clamp(rsym_norm, 0.0, 1.0)
    loss_peak = (peak_err.pow(2.0) * peak_w).mean()

    # Z-gradient consistency along active needle tracts
    if prob_mask.shape[1] > 2:
        dz_pred = prob_mask[:, 1:] - prob_mask[:, :-1]
        dz_true = t_mask[:, 1:] - t_mask[:, :-1]
        tract_w = 1.0 + 10.0 * torch.maximum(t_mask[:, 1:], t_mask[:, :-1])
        loss_z_cont = ((dz_pred - dz_true).abs() * tract_w).mean()
    else:
        loss_z_cont = 0.0

    # 4. 3D Tangent Vector Field Loss (Masked Smooth-L1 on active needle voxels)
    fg_mask = (t_mask >= 0.25).float()
    fg_sum = fg_mask.sum().clamp(min=1.0)
    loss_vec = (
        (F.smooth_l1_loss(pred_vx, t_vx, reduction="none") + F.smooth_l1_loss(pred_vy, t_vy, reduction="none"))
        * fg_mask
    ).sum() / fg_sum

    # 5. Obturator / Short-Needle Auxiliary Head Loss
    prob_obt = torch.sigmoid(logit_obt)
    bce_obt = F.binary_cross_entropy_with_logits(logit_obt, t_obt, reduction="none")
    loss_obt = (bce_obt * (1.0 + 20.0 * t_obt)).mean()

    return loss_bce + loss_dice + 1.5 * loss_peak + 0.4 * loss_z_cont + 0.8 * loss_vec + 0.4 * loss_obt


if __name__ == "__main__":
    print("NeedleUNet3D parameters:", sum(p.numel() for p in NeedleUNet3D().parameters()))
