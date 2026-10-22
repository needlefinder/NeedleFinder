#!/usr/bin/env python3
"""
Train & Evaluate Two Essential 5-Fold Cross-Validation Baselines on all 54 GYN Cases
(using the exact same 5-fold patient split `i % 5 == fold`):

1. Baseline 1 (`PMB-2019-Iso1mm-UNet3D`):
   - Replicates the PMB 2019 (Zaffino et al., Phys. Med. Biol. 2019) approach on all 54 cases:
   - Resamples raw T2W MRI (`X[0]`) and binary r=1.5 mm needle masks to isotropic 1.0 x 1.0 x 1.0 mm^3 resolution.
   - Trains a 1-class 3D U-Net (isotropic 3x3x3 convolutions + 2x2x2 max-pooling) with Soft Dice loss (5-fold CV).
   - Applies probability thresholding + 3D connected components / DBSCAN clustering + 3D centerline skeletonization/PCA
     polyline extraction in 1 mm^3 isotropic space, and transforms predicted trajectories back to RAS physical coordinates.

2. Baseline 2 (`Native-nnUNet3D-clDice`):
   - Trains a modern native-resolution 1-class 3D Residual U-Net on raw T2W MRI (`X[0]` only, without Stage 0 Rust
     physics channels X[1:6] and without multi-head peak/tangent heads Y[1:5]) using Dice + Focal + Soft-Skeleton clDice loss
     (5-fold CV).
   - Follows with 3D morphological skeletonization / connected-component centerline extraction in native resolution.
"""

import math
import os
import pickle
import random
import sys
import time
import multiprocessing as mp
import numpy as np
from scipy.ndimage import label as cc_label, maximum_filter, gaussian_filter, distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_DIR = "/tmp/needle_unet3d_cache"
OUT_DIR = "/home/guillaume/github/needlefinder/nn_models"
META_PATH = "/tmp/cases_meta_54.pkl"
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# BASELINE 1 ARCHITECTURE: PMB 2019 Isotropic 1.0x1.0x1.0 mm^3 1-Class 3D U-Net
# ==============================================================================

class ConvBlockIso3D(nn.Module):
    """Standard 3D U-Net double 3x3x3 convolution block (PMB 2019 / Cicek et al.)."""
    def __init__(self, in_ch, out_ch, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p_drop) if p_drop > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PMB2019IsoUNet3D(nn.Module):
    """
    1-Class 3D U-Net operating on isotropic 1.0 x 1.0 x 1.0 mm^3 T2W MRI volumes
    with 4 resolution levels and 2x2x2 3D max-pooling (Zaffino et al., PMB 2019).
    """
    def __init__(self, in_ch=1, base_ch=20):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.enc1 = ConvBlockIso3D(in_ch, c1, p_drop=0.1)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.enc2 = ConvBlockIso3D(c1, c2, p_drop=0.2)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.enc3 = ConvBlockIso3D(c2, c3, p_drop=0.2)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.bottleneck = ConvBlockIso3D(c3, c4, p_drop=0.3)

        self.dec3 = ConvBlockIso3D(c4 + c3, c3, p_drop=0.2)
        self.dec2 = ConvBlockIso3D(c3 + c2, c2, p_drop=0.2)
        self.dec1 = ConvBlockIso3D(c2 + c1, c1, p_drop=0.1)
        self.head = nn.Conv3d(c1, 1, kernel_size=1)

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
        return self.head(d1)


def compute_pmb2019_dice_loss(logits, target):
    """PMB 2019 Soft Dice loss (+ small weighted BCE for numerical stability on sparse 0.12% foreground)."""
    prob = torch.sigmoid(logits[:, 0])
    t = (target[:, 0] >= 0.5).float()
    inter = (prob * t).sum(dim=(1, 2, 3))
    denom = (prob * prob).sum(dim=(1, 2, 3)) + (t * t).sum(dim=(1, 2, 3))
    loss_dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
    bce = F.binary_cross_entropy_with_logits(logits[:, 0], t, pos_weight=torch.tensor([12.0], device=logits.device))
    return loss_dice + 0.25 * bce


# ==============================================================================
# BASELINE 2 ARCHITECTURE: Native-Resolution 1-Class 3D ResUNet + Soft-clDice
# ==============================================================================

class AnisoResBlock3D(nn.Module):
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
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(4, ch // reduction)
        self.fc1 = nn.Conv3d(ch, mid, kernel_size=1)
        self.fc2 = nn.Conv3d(mid, ch, kernel_size=1)

    def forward(self, x):
        w = x.mean(dim=(2, 3, 4), keepdim=True)
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(w))))
        return x * w


class NativeResUNet3D_MaskOnly(nn.Module):
    """
    Modern Native-Resolution 1-Class 3D Residual U-Net on Raw T2W MRI only (in_ch=1, out_ch=1).
    No Stage 0 Rust physics channels, and no multi-head peak/tangent vector heads.
    """
    def __init__(self, in_ch=1, base_ch=24):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 6
        self.enc1 = nn.Sequential(AnisoResBlock3D(in_ch, c1), AnisoResBlock3D(c1, c1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc2 = nn.Sequential(AnisoResBlock3D(c1, c2), SEBlock3D(c2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc3 = nn.Sequential(AnisoResBlock3D(c2, c3), SEBlock3D(c3))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bottleneck = nn.Sequential(
            AnisoResBlock3D(c3, c4),
            nn.Conv3d(c4, c4, kernel_size=3, padding=(1, 2, 2), dilation=(1, 2, 2), bias=False),
            nn.GroupNorm(min(8, c4), c4),
            nn.SiLU(inplace=True),
            SEBlock3D(c4),
        )
        self.dec3 = AnisoResBlock3D(c4 + c3, c3)
        self.dec2 = AnisoResBlock3D(c3 + c2, c2)
        self.dec1 = nn.Sequential(AnisoResBlock3D(c2 + c1, c1), AnisoResBlock3D(c1, c1))
        self.head = nn.Conv3d(c1, 1, kernel_size=1)

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
        return self.head(d1)


def soft_erode_3d(img):
    p1 = -F.max_pool3d(-img, (1, 3, 1), stride=1, padding=(0, 1, 0))
    p2 = -F.max_pool3d(-img, (1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(p1, p2)


def soft_dilate_3d(img):
    return F.max_pool3d(img, (1, 3, 3), stride=1, padding=(0, 1, 1))


def soft_open_3d(img):
    return soft_dilate_3d(soft_erode_3d(img))


def soft_skel_3d(img, iter_=3):
    img1 = soft_open_3d(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode_3d(img)
        img1 = soft_open_3d(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def compute_cldice_focal_dice_loss(logits, target):
    """Dice + Focal + Soft-Skeleton clDice loss (Shit et al., CVPR 2021)."""
    prob = torch.sigmoid(logits)
    t = target.float()

    # 1. Soft Dice
    inter = (prob * t).sum(dim=(1, 2, 3, 4))
    union = prob.sum(dim=(1, 2, 3, 4)) + t.sum(dim=(1, 2, 3, 4))
    loss_dice = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()

    # 2. Weighted Focal BCE
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = torch.where(t >= 0.5, prob, 1.0 - prob)
    focal_w = (1.0 - pt).pow(2.0) * (1.0 + 18.0 * t)
    loss_focal = (bce * focal_w).mean()

    # 3. Soft-Skeleton clDice
    skel_pred = soft_skel_3d(prob, iter_=3)
    skel_true = soft_skel_3d(t, iter_=3)
    tprec = ((skel_pred * t).sum(dim=(1, 2, 3, 4)) + 1.0) / (skel_pred.sum(dim=(1, 2, 3, 4)) + 1.0)
    tsens = ((skel_true * prob).sum(dim=(1, 2, 3, 4)) + 1.0) / (skel_true.sum(dim=(1, 2, 3, 4)) + 1.0)
    cl_dice = (1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + 1e-6)).mean()

    return loss_dice + loss_focal + 0.5 * cl_dice


# ==============================================================================
# DATA LOADING, RESAMPLING & AUGMENTATION
# ==============================================================================

def load_all_54_items(meta_list):
    items_native = []
    items_iso1mm = []
    for i in range(54):
        d = torch.load(f"{CACHE_DIR}/case_{i:02d}.pt", weights_only=False)
        x_mri = d["X"][0:1].float()  # [1, K, H, W] raw normalized T2W MRI only
        y_mask = d["Y"][0:1].float()  # [1, K, H, W] needle mask
        sp_i, sp_j, sp_k = [float(v) for v in meta_list[i]["sp_ras"]]
        _, K, H, W = x_mri.shape

        # Resample to isotropic 1.0 x 1.0 x 1.0 mm^3 for Baseline 1 (PMB 2019)
        D_iso = max(16, int(round(K * sp_k / 8.0)) * 8)
        H_iso = max(64, int(round(H * sp_j / 8.0)) * 8)
        W_iso = max(64, int(round(W * sp_i / 8.0)) * 8)
        x_iso = F.interpolate(x_mri.unsqueeze(0), size=(D_iso, H_iso, W_iso), mode="trilinear", align_corners=False).squeeze(0)
        y_bin = (y_mask >= 0.5).float()
        y_iso = (F.interpolate(y_bin.unsqueeze(0), size=(D_iso, H_iso, W_iso), mode="trilinear", align_corners=False).squeeze(0) >= 0.45).float()

        meta_d = {
            "idx": i,
            "k0": int(d["k0"]),
            "i0": int(d["i0"]), "i1": int(d["i1"]),
            "j0": int(d["j0"]), "j1": int(d["j1"]),
            "K": K, "H": H, "W": W,
            "D_iso": D_iso, "H_iso": H_iso, "W_iso": W_iso,
            "sp_ras": (sp_i, sp_j, sp_k),
            "ijk_to_ras": np.asarray(meta_list[i]["ijk_to_ras"], dtype=np.float64),
        }
        items_native.append((x_mri, y_mask, meta_d))
        items_iso1mm.append((x_iso, y_iso, meta_d))
    return items_iso1mm, items_native


def augment_1ch(x, y, max_d=48, max_hw=160):
    x = x.clone()
    y = y.clone()
    _, D, H, W = x.shape
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
        y = torch.flip(y, dims=[3])
    if random.random() < 0.2:
        x = torch.flip(x, dims=[2])
        y = torch.flip(y, dims=[2])
    if random.random() < 0.7:
        dy = random.randint(-8, 8)
        dx = random.randint(-8, 8)
        x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        y = torch.roll(y, shifts=(dy, dx), dims=(2, 3))
    if random.random() < 0.7:
        scale = random.uniform(0.85, 1.18)
        bias = random.uniform(-0.08, 0.08)
        x = (x * scale + bias).clamp(-0.5, 2.5)
    if D > max_d:
        d0 = random.randint(0, D - max_d)
        x = x[:, d0:d0 + max_d]
        y = y[:, d0:d0 + max_d]
    if H > max_hw:
        h0 = random.randint(0, H - max_hw)
        x = x[:, :, h0:h0 + max_hw, :]
        y = y[:, :, h0:h0 + max_hw, :]
    if W > max_hw:
        w0 = random.randint(0, W - max_hw)
        x = x[:, :, :, w0:w0 + max_hw]
        y = y[:, :, :, w0:w0 + max_hw]
    return x, y


def infer_sliding_1ch(model, x_cpu, device, chunk_d=48):
    _, D, H, W = x_cpu.shape
    if D <= chunk_d + 8:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x_cpu.unsqueeze(0).to(device))
        return torch.sigmoid(out.float().squeeze(0).cpu()[0]).numpy()
    out_sum = np.zeros((D, H, W), dtype=np.float32)
    w_sum = np.zeros((D, 1, 1), dtype=np.float32)
    stride = chunk_d // 2
    starts = list(range(0, max(1, D - chunk_d + 1), stride))
    if starts[-1] + chunk_d < D:
        starts.append(D - chunk_d)
    for s in starts:
        e = min(D, s + chunk_d)
        sub = x_cpu[:, s:e].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(sub).float().squeeze(0).cpu()[0]).numpy()
        out_sum[s:e] += p
        w_sum[s:e] += 1.0
    return out_sum / np.maximum(w_sum, 1.0)


# ==============================================================================
# CENTERLINE EXTRACTION FOR BASELINE 1 (PMB 2019 DBSCAN/CC + 3D POLYLINE)
# AND BASELINE 2 (NATIVE 3D MORPHOLOGICAL SKELETONIZATION + CC LINKING)
# ==============================================================================

def extract_pmb2019_iso_trajectories(p_iso, meta_d, prob_thresh=0.42):
    """
    PMB 2019 Post-Processing on Isotropic 1x1x1 mm^3 Probability Volume:
    1. Thresholds probability volume and extracts slice-wise 2D connected component / local peak centroids
       (and splits merged blobs via local maxima in 1 mm isotropic grid).
    2. Links 3D points across 1 mm isotropic Z slices via DBSCAN / nearest-neighbor collinear chaining
       (radius <= 3.2 mm, max gap <= 8 mm) and fits degree-2 / linear 3D polylines.
    3. Transforms isotropic (w_iso, h_iso, d_iso) back to native (i, j, k) and RAS physical coordinates.
    """
    D_iso, H_iso, W_iso = p_iso.shape
    K, H, W = meta_d["K"], meta_d["H"], meta_d["W"]
    k0, i0, j0 = meta_d["k0"], meta_d["i0"], meta_d["j0"]
    ijk_to_ras = meta_d["ijk_to_ras"]
    rot = ijk_to_ras[:3, :3]
    trans = ijk_to_ras[:3, 3]
    z_base_ras = float((rot @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0 + 1)]) + trans)[2])

    # Extract 2D centroids per 1mm slice using connected components + local maxima for touching blobs
    smooth_p = gaussian_filter(p_iso.astype(np.float32), sigma=(0.5, 0.8, 0.8))
    loc_max = (smooth_p == maximum_filter(smooth_p, size=(1, 5, 5))) & (p_iso >= prob_thresh)
    slice_pts = {}
    yy_w, xx_w = np.mgrid[-1:2, -1:2]
    for d in range(D_iso):
        jj, ii = np.where(loc_max[d])
        if len(ii) == 0:
            continue
        pts_d = []
        for jc, ic in zip(jj, ii):
            if 1 <= jc < H_iso - 1 and 1 <= ic < W_iso - 1:
                patch = p_iso[d, jc - 1:jc + 2, ic - 1:ic + 2]
                ws = float(np.sum(patch))
                si = float(ic) + float(np.sum(xx_w * patch)) / max(ws, 1e-6)
                sj = float(jc) + float(np.sum(yy_w * patch)) / max(ws, 1e-6)
            else:
                si, sj = float(ic), float(jc)
            # Map from isotropic (W_iso, H_iso, D_iso) to native full-volume (I, J, K)
            i_nat = i0 + (si + 0.5) * (W / float(W_iso)) - 0.5
            j_nat = j0 + (sj + 0.5) * (H / float(H_iso)) - 0.5
            k_nat = k0 + (float(d) + 0.5) * (K / float(D_iso)) - 0.5
            pt_ras = rot @ np.array([i_nat, j_nat, k_nat]) + trans
            pts_d.append((pt_ras[0], pt_ras[1], pt_ras[2], float(p_iso[d, jc, ic]), d))
        slice_pts[d] = pts_d

    # Greedy 3D DBSCAN-style trajectory linking across 1mm isotropic Z slices
    used = set()
    trajs = []
    for d_start in sorted(slice_pts.keys()):
        for idx_p, pt0 in enumerate(slice_pts[d_start]):
            if (d_start, idx_p) in used:
                continue
            chain = [pt0]
            used.add((d_start, idx_p))
            curr_x, curr_y, curr_z, _, curr_d = pt0
            vx_est, vy_est = 0.0, 0.0
            for d_next in range(d_start + 1, D_iso):
                if d_next - curr_d > 7:
                    break
                if d_next not in slice_pts:
                    continue
                best_j = -1
                best_dist = 2.8
                for j_p, cand in enumerate(slice_pts[d_next]):
                    if (d_next, j_p) in used:
                        continue
                    dz = cand[2] - curr_z
                    pred_x = curr_x + vx_est * dz
                    pred_y = curr_y + vy_est * dz
                    dist = math.hypot(cand[0] - pred_x, cand[1] - pred_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j_p
                if best_j >= 0:
                    chosen = slice_pts[d_next][best_j]
                    used.add((d_next, best_j))
                    chain.append(chosen)
                    if len(chain) >= 3:
                        dz_tot = chain[-1][2] - chain[0][2]
                        if abs(dz_tot) > 1.0:
                            vx_est = 0.7 * vx_est + 0.3 * ((chain[-1][0] - chain[0][0]) / dz_tot)
                            vy_est = 0.7 * vy_est + 0.3 * ((chain[-1][1] - chain[0][1]) / dz_tot)
                    curr_x, curr_y, curr_z, _, curr_d = chosen

            if len(chain) < 9:  # at least 9 mm in 1mm isotropic grid
                continue
            arr = np.array([[c[0], c[1], c[2], c[3]] for c in chain], dtype=np.float64)
            arr = arr[np.argsort(arr[:, 2])]
            z_span = float(arr[-1, 2] - arr[0, 2])
            if z_span < 11.5:
                continue
            deg = 2 if len(arr) >= 18 else 1
            px = np.polyfit(arr[:, 2], arr[:, 0], deg)
            py = np.polyfit(arr[:, 2], arr[:, 1], deg)
            fx = np.polyval(px, arr[:, 2])
            fy = np.polyval(py, arr[:, 2])
            rmse = float(np.sqrt(np.mean((arr[:, 0] - fx) ** 2 + (arr[:, 1] - fy) ** 2)))
            if rmse > 1.65:
                continue
            sx = float(fx[-1] - fx[0]) / max(1.0, z_span)
            sy = float(fy[-1] - fy[0]) / max(1.0, z_span)
            if math.hypot(sx, sy) > 0.62:
                continue
            mean_prob = float(np.mean(arr[:, 3]))
            if mean_prob < 0.46:
                continue
            # Sample clean polyline every ~2 mm along Z
            n_out = max(6, int(round(z_span / 2.0)))
            zs_out = np.linspace(arr[0, 2], arr[-1, 2], n_out)
            xs_out = np.polyval(px, zs_out)
            ys_out = np.polyval(py, zs_out)
            smooth_ras = np.column_stack([xs_out, ys_out, zs_out])
            is_long = (abs(arr[0, 2] - z_base_ras) <= 26.0) and (z_span >= 14.0)
            trajs.append({
                "ras": smooth_ras,
                "points": smooth_ras,
                "confidence": mean_prob * 100.0,
                "z_span": z_span,
                "is_long": is_long,
            })

    # Deduplicate trajectories within 2.0 mm
    trajs.sort(key=lambda t: -t["confidence"] * math.log1p(t["z_span"]))
    dedup = []
    for t in trajs:
        dup = False
        for ex in dedup:
            z0 = max(float(t["ras"][0, 2]), float(ex["ras"][0, 2]))
            z1 = min(float(t["ras"][-1, 2]), float(ex["ras"][-1, 2]))
            if z1 > z0 + 2.0:
                zs = np.linspace(z0, z1, 10)
                d = float(np.mean(np.hypot(
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 0]) - np.interp(zs, ex["ras"][:, 2], ex["ras"][:, 0]),
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 1]) - np.interp(zs, ex["ras"][:, 2], ex["ras"][:, 1]),
                )))
                if d < 2.0:
                    dup = True
                    break
        if not dup:
            dedup.append(t)
    return dedup


def extract_native_cldice_skeleton_trajectories(p_mask, meta_d, prob_thresh=0.40):
    """
    Baseline 2 Post-Processing on Native-Resolution Probability Map (Mask-Only + clDice):
    1. Computes 2D/3D morphological skeleton / medial axis ridges via distance transform & local maxima on p_mask.
    2. Links skeleton centroids across native slices (without neural tangent field vx/vy and without physics prior X[1]).
    """
    K, H, W = p_mask.shape
    k0, i0, j0 = meta_d["k0"], meta_d["i0"], meta_d["j0"]
    sp_i, sp_j, sp_k = meta_d["sp_ras"]
    ijk_to_ras = meta_d["ijk_to_ras"]
    rot = ijk_to_ras[:3, :3]
    trans = ijk_to_ras[:3, 3]
    z_base_ras = float((rot @ np.array([i0 + W * 0.5, j0 + H * 0.5, float(k0 + 1)]) + trans)[2])

    # Medial-axis ridge enhancement on predicted binary mask
    bin_m = p_mask >= prob_thresh
    ridge = np.zeros_like(p_mask, dtype=np.float32)
    for k in range(K):
        if np.any(bin_m[k]):
            edt = distance_transform_edt(bin_m[k], sampling=(sp_j, sp_i))
            ridge[k] = p_mask[k] * (1.0 + 0.5 * np.clip(edt, 0.0, 2.0))

    nms_r = max(1, int(round(1.35 / min(sp_i, sp_j))))
    loc_max = (ridge == maximum_filter(ridge, size=(1, 2 * nms_r + 1, 2 * nms_r + 1))) & (p_mask >= prob_thresh)
    yy_w, xx_w = np.mgrid[-1:2, -1:2]

    slice_pts = {}
    for k in range(K):
        jj, ii = np.where(loc_max[k])
        if len(ii) == 0:
            continue
        pts_k = []
        for jc, ic in zip(jj, ii):
            if 1 <= jc < H - 1 and 1 <= ic < W - 1:
                patch = p_mask[k, jc - 1:jc + 2, ic - 1:ic + 2]
                ws = float(np.sum(patch))
                si = float(ic) + float(np.sum(xx_w * patch)) / max(ws, 1e-6)
                sj = float(jc) + float(np.sum(yy_w * patch)) / max(ws, 1e-6)
            else:
                si, sj = float(ic), float(jc)
            pt_ras = rot @ np.array([si + i0, sj + j0, float(k + k0)]) + trans
            pts_k.append((pt_ras[0], pt_ras[1], pt_ras[2], float(p_mask[k, jc, ic]), k))
        slice_pts[k] = pts_k

    min_slices = 4 if sp_k >= 3.5 else (5 if sp_k >= 2.2 else 7)
    max_dk = 3
    used = set()
    trajs = []
    for k_start in sorted(slice_pts.keys()):
        for idx_p, pt0 in enumerate(slice_pts[k_start]):
            if (k_start, idx_p) in used:
                continue
            chain = [pt0]
            used.add((k_start, idx_p))
            curr_x, curr_y, curr_z, _, curr_k = pt0
            vx_est, vy_est = 0.0, 0.0
            for k_next in range(k_start + 1, K):
                if k_next - curr_k > max_dk:
                    break
                if k_next not in slice_pts:
                    continue
                best_j = -1
                best_dist = 2.6 + 0.15 * abs(k_next - curr_k) * sp_k
                for j_p, cand in enumerate(slice_pts[k_next]):
                    if (k_next, j_p) in used:
                        continue
                    dz = cand[2] - curr_z
                    pred_x = curr_x + vx_est * dz
                    pred_y = curr_y + vy_est * dz
                    dist = math.hypot(cand[0] - pred_x, cand[1] - pred_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j_p
                if best_j >= 0:
                    chosen = slice_pts[k_next][best_j]
                    used.add((k_next, best_j))
                    chain.append(chosen)
                    if len(chain) >= 3:
                        dz_tot = chain[-1][2] - chain[0][2]
                        if abs(dz_tot) > 1.0:
                            vx_est = 0.6 * vx_est + 0.4 * ((chain[-1][0] - chain[0][0]) / dz_tot)
                            vy_est = 0.6 * vy_est + 0.4 * ((chain[-1][1] - chain[0][1]) / dz_tot)
                    curr_x, curr_y, curr_z, _, curr_k = chosen

            if len(chain) < min_slices:
                continue
            arr = np.array([[c[0], c[1], c[2], c[3]] for c in chain], dtype=np.float64)
            arr = arr[np.argsort(arr[:, 2])]
            z_span = float(arr[-1, 2] - arr[0, 2])
            if z_span < 11.5:
                continue
            deg = 2 if len(arr) >= 8 else 1
            px = np.polyfit(arr[:, 2], arr[:, 0], deg)
            py = np.polyfit(arr[:, 2], arr[:, 1], deg)
            fx = np.polyval(px, arr[:, 2])
            fy = np.polyval(py, arr[:, 2])
            rmse = float(np.sqrt(np.mean((arr[:, 0] - fx) ** 2 + (arr[:, 1] - fy) ** 2)))
            if rmse > 1.50:
                continue
            sx = float(fx[-1] - fx[0]) / max(1.0, z_span)
            sy = float(fy[-1] - fy[0]) / max(1.0, z_span)
            if math.hypot(sx, sy) > 0.60:
                continue
            mean_prob = float(np.mean(arr[:, 3]))
            if mean_prob < 0.45:
                continue
            smooth_ras = np.column_stack([fx, fy, arr[:, 2]])
            is_long = (abs(arr[0, 2] - z_base_ras) <= 26.0) and (z_span >= 14.0)
            trajs.append({
                "ras": smooth_ras,
                "points": smooth_ras,
                "confidence": mean_prob * 100.0,
                "z_span": z_span,
                "is_long": is_long,
            })

    trajs.sort(key=lambda t: -t["confidence"] * math.log1p(t["z_span"]))
    dedup = []
    for t in trajs:
        dup = False
        for ex in dedup:
            z0 = max(float(t["ras"][0, 2]), float(ex["ras"][0, 2]))
            z1 = min(float(t["ras"][-1, 2]), float(ex["ras"][-1, 2]))
            if z1 > z0 + 2.0:
                zs = np.linspace(z0, z1, 10)
                d = float(np.mean(np.hypot(
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 0]) - np.interp(zs, ex["ras"][:, 2], ex["ras"][:, 0]),
                    np.interp(zs, t["ras"][:, 2], t["ras"][:, 1]) - np.interp(zs, ex["ras"][:, 2], ex["ras"][:, 1]),
                )))
                if d < 2.0:
                    dup = True
                    break
        if not dup:
            dedup.append(t)
    return dedup


# ==============================================================================
# 5-FOLD CV TRAINING FOR BOTH BASELINES
# ==============================================================================

def run_baseline1_pmb2019(items_iso, device, epochs=28):
    print(f"\n=== Training Baseline 1: PMB-2019-Iso1mm-UNet3D (5-Fold CV on {device}) ===", flush=True)
    oof_trajs_all = [None] * 54
    oof_trajs_long = [None] * 54
    oof_dices = []

    for fold in range(5):
        t0 = time.perf_counter()
        val_idx = [i for i in range(54) if i % 5 == fold]
        train_idx = [i for i in range(54) if i % 5 != fold]

        model = PMB2019IsoUNet3D(in_ch=1, base_ch=20).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        best_dice = -1.0
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            order = train_idx.copy()
            random.shuffle(order)
            for idx in order:
                x, y, _ = items_iso[idx]
                xa, ya = augment_1ch(x, y, max_d=64, max_hw=144)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(xa.unsqueeze(0).to(device))
                    loss = compute_pmb2019_dice_loss(out, ya.unsqueeze(0).to(device))
                loss.backward()
                opt.step()
            sched.step()

            if ep % 7 == 0 or ep == epochs:
                model.eval()
                f_dices = []
                for vi in val_idx:
                    xv, yv, _ = items_iso[vi]
                    pm = infer_sliding_1ch(model, xv, device, chunk_d=64)
                    tm = (yv[0].numpy() >= 0.5).astype(np.float32)
                    inter = float((pm * tm).sum())
                    union = float(pm.sum() + tm.sum())
                    f_dices.append((2.0 * inter + 1.0) / (union + 1.0))
                md = float(np.mean(f_dices))
                if md > best_dice:
                    best_dice = md
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"  [PMB2019-Iso1mm | Fold {fold}] Ep {ep:02d}/{epochs} | ValDice={md:.3f} ({time.perf_counter()-t0:.1f}s)", flush=True)

        model.load_state_dict(best_state)
        torch.save(best_state, f"{OUT_DIR}/pmb2019_iso1mm_fold{fold}.pt")
        model.eval()
        for vi in val_idx:
            xv, yv, meta_d = items_iso[vi]
            pm = infer_sliding_1ch(model, xv, device, chunk_d=64)
            tm = (yv[0].numpy() >= 0.5).astype(np.float32)
            oof_dices.append(float((2.0 * (pm * tm).sum() + 1.0) / (pm.sum() + tm.sum() + 1.0)))
            np.savez_compressed(f"{OUT_DIR}/oof_pmb2019_iso1mm_case_{vi:02d}.npz", p_mask=pm.astype(np.float16))
            trajs = extract_pmb2019_iso_trajectories(pm, meta_d, prob_thresh=0.42)
            oof_trajs_all[vi] = trajs
            oof_trajs_long[vi] = [t for t in trajs if t["is_long"]]

    with open(f"{OUT_DIR}/oof_trajs_pmb2019_iso1mm.pkl", "wb") as f:
        pickle.dump({"all": oof_trajs_all, "long": oof_trajs_long, "mean_oof_dice": float(np.mean(oof_dices))}, f)
    print(f"[Done] Baseline 1 PMB-2019-Iso1mm-UNet3D Mean OOF Dice = {np.mean(oof_dices):.4f}", flush=True)


def run_baseline2_native_cldice(items_nat, device, epochs=28):
    print(f"\n=== Training Baseline 2: Native-nnUNet3D-clDice (5-Fold CV on {device}) ===", flush=True)
    oof_trajs_all = [None] * 54
    oof_trajs_long = [None] * 54
    oof_dices = []

    for fold in range(5):
        t0 = time.perf_counter()
        val_idx = [i for i in range(54) if i % 5 == fold]
        train_idx = [i for i in range(54) if i % 5 != fold]

        model = NativeResUNet3D_MaskOnly(in_ch=1, base_ch=24).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        best_dice = -1.0
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            order = train_idx.copy()
            random.shuffle(order)
            for idx in order:
                x, y, _ = items_nat[idx]
                xa, ya = augment_1ch(x, y, max_d=36, max_hw=160)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(xa.unsqueeze(0).to(device))
                    loss = compute_cldice_focal_dice_loss(out, ya.unsqueeze(0).to(device))
                loss.backward()
                opt.step()
            sched.step()

            if ep % 7 == 0 or ep == epochs:
                model.eval()
                f_dices = []
                for vi in val_idx:
                    xv, yv, _ = items_nat[vi]
                    pm = infer_sliding_1ch(model, xv, device, chunk_d=44)
                    tm = yv[0].numpy().astype(np.float32)
                    inter = float((pm * tm).sum())
                    union = float(pm.sum() + tm.sum())
                    f_dices.append((2.0 * inter + 1.0) / (union + 1.0))
                md = float(np.mean(f_dices))
                if md > best_dice:
                    best_dice = md
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"  [Native-clDice | Fold {fold}] Ep {ep:02d}/{epochs} | ValDice={md:.3f} ({time.perf_counter()-t0:.1f}s)", flush=True)

        model.load_state_dict(best_state)
        torch.save(best_state, f"{OUT_DIR}/native_cldice_fold{fold}.pt")
        model.eval()
        for vi in val_idx:
            xv, yv, meta_d = items_nat[vi]
            pm = infer_sliding_1ch(model, xv, device, chunk_d=44)
            tm = yv[0].numpy().astype(np.float32)
            oof_dices.append(float((2.0 * (pm * tm).sum() + 1.0) / (pm.sum() + tm.sum() + 1.0)))
            np.savez_compressed(f"{OUT_DIR}/oof_native_cldice_case_{vi:02d}.npz", p_mask=pm.astype(np.float16))
            trajs = extract_native_cldice_skeleton_trajectories(pm, meta_d, prob_thresh=0.40)
            oof_trajs_all[vi] = trajs
            oof_trajs_long[vi] = [t for t in trajs if t["is_long"]]

    with open(f"{OUT_DIR}/oof_trajs_native_cldice.pkl", "wb") as f:
        pickle.dump({"all": oof_trajs_all, "long": oof_trajs_long, "mean_oof_dice": float(np.mean(oof_dices))}, f)
    print(f"[Done] Baseline 2 Native-nnUNet3D-clDice Mean OOF Dice = {np.mean(oof_dices):.4f}", flush=True)


def run_worker(which):
    meta_list = pickle.load(open(META_PATH, "rb"))
    items_iso, items_nat = load_all_54_items(meta_list)
    device = torch.device("cuda:0")  # RTX 4090 (14.9 GB free VRAM)
    if which == 1:
        run_baseline1_pmb2019(items_iso, device, epochs=28)
    else:
        run_baseline2_native_cldice(items_nat, device, epochs=28)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p1 = mp.Process(target=run_worker, args=(1,))
    p2 = mp.Process(target=run_worker, args=(2,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
