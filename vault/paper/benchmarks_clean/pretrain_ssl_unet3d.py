#!/usr/bin/env python3
"""
Step 2: Self-Supervised 3D Encoder Pretraining on 51 Strictly External Pelvic MRI Volumes (Item 5).

External Cohort (51 volumes, 0 overlap with the 54 benchmark cases):
  - 48 MSD Task05_Prostate 3T T2W MRI volumes (/tmp/msd_prostate/Task05_Prostate/imagesTr/*.nii.gz & imagesTs/*.nii.gz)
  - 3 BWH non-benchmark pelvic T2W NRRD volumes (Case 061 3D SPACE, Case 061 2D TSE, Case 100 Prostate 3D SPACE)

Dual Self-Supervised Pretraining Objective (45 epochs on RTX 4090):
  a. 3D Masked Volume & Physics Feature Autoencoding (3D MAE):
     Masks 25% of 3D patches (8x16x16 blocks) + random axial slices in channels 0..3 and trains
     `ssl_recon_head(d1)` to reconstruct the unmasked clean channels 0..3 (L1 + MSE loss).
  b. Physics-Guided Self-Supervised Synthetic 3D Susceptibility Tube Pretext Task on Real Pelvic Anatomy:
     Dynamically injects 8-20 realistic curved 3D susceptibility dark-core tubes (r ~ U(1.0, 1.5) mm,
     vx, vy ~ N(0, 0.15), partial-volume tip taper, slice dropouts) into the real pelvic MRI volumes,
     constructing exact 5-channel ground-truth targets Y_syn (mask, peak, vx, vy, obt), while all real
     pelvic vessels, bowel loops, and fascial planes with high radial symmetry (X_raw[1] > 0.2) outside
     the injected tubes serve as authentic anatomical hard negatives (Y_syn = 0).

Outputs:
  - /tmp/ssl_pelvic_51_cache/ssl_vol_{idx:02d}.pt (idx in 0..50)
  - /home/guillaume/github/needlefinder/nn_models_clean/unet3d_ssl_pretrained_51vols.pt
  - /home/guillaume/github/needlefinder/vault/paper/benchmarks_clean/step2_ssl_pretrain_summary.json
"""

import glob
import gzip
import json
import math
import os
import random
import struct
import sys
import time
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

for mod in (
    "vtk", "vtk.util", "vtk.util.numpy_support", "slicer", "slicer.util",
    "qt", "ctk", "SimpleITK", "sitkUtils", "slicer.ScriptedLoadableModule"
):
    sys.modules[mod] = types.ModuleType(mod)

sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean")

from NeedleFinderLib.RustCore import rust_compute_3d_sym_response
from prepare_and_inspect_dataset import read_nrrd_local
from train_and_eval_unet3d import NeedleUNet3D, compute_multitask_loss
from prepare_clean_dataset import estimate_axial_limit_slice_sym_safe, build_clean_input_tensor

SSL_CACHE_DIR = "/tmp/ssl_pelvic_51_cache"
OUT_MODEL_DIR = "/home/guillaume/github/needlefinder/nn_models_clean"
SSL_CKPT_PATH = os.path.join(OUT_MODEL_DIR, "unet3d_ssl_pretrained_51vols.pt")
SSL_SUMMARY_PATH = "/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean/step2_ssl_pretrain_summary.json"


def read_nifti_t2w_pure_numpy(path: str):
    """
    Pure-NumPy gzip + struct NIfTI-1 (.nii.gz) reader extracting channel 0 (T2W MRI)
    in (K, J, I) layout and voxel spacing (sp_i, sp_j, sp_k) in mm.
    """
    with gzip.open(path, "rb") as f:
        raw = f.read()

    sizeof_hdr = struct.unpack("<i", raw[0:4])[0]
    endian = "<" if sizeof_hdr == 348 else ">"
    dims = struct.unpack(endian + "8h", raw[40:56])
    ndim = dims[0]
    nx, ny, nz = int(dims[1]), int(dims[2]), int(dims[3])
    nt = int(dims[4]) if ndim >= 4 and dims[4] > 0 else 1

    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    scl_slope = struct.unpack(endian + "f", raw[112:116])[0]
    scl_inter = struct.unpack(endian + "f", raw[116:120])[0]

    dt_map = {
        2: "u1",
        4: "i2",
        8: "i4",
        16: "f4",
        64: "f8",
        256: "i1",
        512: "u2",
        768: "u4",
    }
    np_dt = endian + dt_map[datatype]
    n_vox = nx * ny * nz * nt
    itemsize = np.dtype(np_dt).itemsize
    arr = np.frombuffer(raw[vox_offset:vox_offset + n_vox * itemsize], dtype=np_dt)
    if nt > 1:
        # Fortran order (I, J, K, C) -> take channel 0 = T2W -> transpose to (K, J, I)
        arr = arr.reshape((nx, ny, nz, nt), order="F")[..., 0]
    else:
        arr = arr.reshape((nx, ny, nz), order="F")

    vol_kji = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)), dtype=np.float64)
    if scl_slope != 0.0 and not (math.isnan(scl_slope) or math.isnan(scl_inter)):
        vol_kji = vol_kji * float(scl_slope) + float(scl_inter)

    sp_ras = (abs(float(pixdim[1])), abs(float(pixdim[2])), abs(float(pixdim[3])))
    return vol_kji, sp_ras


def prepare_ssl_51_cache():
    os.makedirs(SSL_CACHE_DIR, exist_ok=True)
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)

    tr_files = sorted([p for p in glob.glob("/tmp/msd_prostate/Task05_Prostate/imagesTr/*.nii.gz") if "/._" not in p])
    ts_files = sorted([p for p in glob.glob("/tmp/msd_prostate/Task05_Prostate/imagesTs/*.nii.gz") if "/._" not in p])
    msd_files = [("msd_prostate_tr", p) for p in tr_files] + [("msd_prostate_ts", p) for p in ts_files]
    assert len(msd_files) == 48, f"Expected 48 MSD Task05_Prostate volumes, found {len(msd_files)}"

    bwh_files = [
        ("bwh_nonbenchmark_case061_3d_space", "/home/guillaume/Dropbox/GYN Cases/Case  061/NRRD/34 AX space trial 1.6 SLICE COVER TEMPLATE.MR._.34.1.2014.01.29.12.10.33.826.56237582.nrrd"),
        ("bwh_nonbenchmark_case061_2d_tse", "/home/guillaume/Dropbox/GYN Cases/Case  061/NRRD/Manual_LB/32 AX TSE T2 MONITORING CATH PLACEMENT.MR._.32.22.2014.01.29.12.10.33.826.56232830.nrrd"),
        ("bwh_nonbenchmark_case100_prostate_3d_space", "/home/guillaume/Dropbox/GYN Cases/Case 100 Prostate/NRRD-MR/33 AX space trial 1.6 SLICE COVER TEMPLATE.nrrd"),
    ]
    for _, bp in bwh_files:
        assert os.path.exists(bp), f"Missing BWH external volume: {bp}"

    all_51 = msd_files + bwh_files
    print(f"[SSL Cache] Preparing 51 external pelvic MRI volumes (48 MSD Task05_Prostate + 3 BWH non-benchmark)...")

    cached_items = []
    t0 = time.perf_counter()
    for idx, (source_tag, fpath) in enumerate(all_51):
        pt_path = os.path.join(SSL_CACHE_DIR, f"ssl_vol_{idx:02d}.pt")
        if os.path.exists(pt_path):
            d = torch.load(pt_path, weights_only=False)
            cached_items.append(d)
            continue

        if fpath.endswith(".nii.gz"):
            vol, sp_ras = read_nifti_t2w_pure_numpy(fpath)
            K, J, I = vol.shape
            sp_i, sp_j, sp_k = sp_ras
            k0 = 0
        else:
            vol, sp_ras, _, _, _ = read_nrrd_local(fpath, acsv_path=None)
            K, J, I = vol.shape
            sp_i, sp_j, sp_k = float(sp_ras[0]), float(sp_ras[1]), float(sp_ras[2])
            k_lim_auto = estimate_axial_limit_slice_sym_safe(vol, sp_k)
            k0 = int(np.clip(k_lim_auto - 1, 0, max(0, K - 8)))

        margin_i = min(I // 2 - 4, int(round(68.0 / sp_i)))
        margin_j = min(J // 2 - 4, int(round(68.0 / sp_j)))
        half_w = ((margin_i + 7) // 8) * 8
        half_h = ((margin_j + 7) // 8) * 8
        ci_v, cj_v = I // 2, J // 2
        i0 = max(0, ci_v - half_w)
        i1 = min(I, i0 + 2 * half_w)
        if (i1 - i0) % 8 != 0:
            i1 = i0 + ((i1 - i0) // 8) * 8
        j0 = max(0, cj_v - half_h)
        j1 = min(J, j0 + 2 * half_h)
        if (j1 - j0) % 8 != 0:
            j1 = j0 + ((j1 - j0) // 8) * 8

        # Rescale high-intensity 3T scanners before Rust 3D symmetry computation so body_thresh is calibrated
        roi_raw = vol[k0:K, j0:j1, i0:i1]
        pos_raw = roi_raw[roi_raw > 5.0]
        med_tissue = float(np.percentile(pos_raw, 55)) if len(pos_raw) else 85.0
        if med_tissue > 450.0:
            vol_work = np.ascontiguousarray(vol * (160.0 / med_tissue), dtype=np.float64)
        else:
            vol_work = vol

        X_unlab, _, _, _ = build_clean_input_tensor(vol_work, (sp_i, sp_j, sp_k), k0, K, i0, i1, j0, j1)

        # For the 3 BWH clinical volumes (idx >= 48), suppress any pre-existing unannotated needle-like cores
        # so that only our synthetic injected tubes are positive and background is guaranteed needle-free
        is_zero_needle_msd = (idx < 48)
        if not is_zero_needle_msd:
            suspect = X_unlab[1] > 0.38
            X_unlab[0][suspect] = np.median(X_unlab[0])
            X_unlab[1][suspect] *= 0.15
            X_unlab[2][suspect] *= 0.15
            X_unlab[3][suspect] = np.minimum(X_unlab[3][suspect], 0.05)

        item = {
            "idx": idx,
            "source_tag": source_tag,
            "file_name": os.path.basename(fpath),
            "is_zero_needle_msd": is_zero_needle_msd,
            "X": torch.from_numpy(X_unlab),
            "sp_ras": (float(sp_i), float(sp_j), float(sp_k)),
            "shape": list(X_unlab.shape),
        }
        torch.save(item, pt_path)
        cached_items.append(item)

        if (idx + 1) % 10 == 0 or idx == 50:
            print(f"  Cached {idx+1}/51 SSL volumes ({time.perf_counter()-t0:.1f}s)", flush=True)

    return cached_items


def forward_unet_with_dec1(model: NeedleUNet3D, x: torch.Tensor):
    """Runs NeedleUNet3D and returns both the 5-channel prediction `out` and the 24-channel `d1` feature map."""
    e1 = model.enc1(x)
    e2 = model.enc2(model.pool1(e1))
    e3 = model.enc3(model.pool2(e2))
    b = model.bottleneck(model.pool3(e3))

    u3 = F.interpolate(b, size=e3.shape[2:], mode="trilinear", align_corners=False)
    d3 = model.dec3(torch.cat([u3, e3], dim=1))

    u2 = F.interpolate(d3, size=e2.shape[2:], mode="trilinear", align_corners=False)
    d2 = model.dec2(torch.cat([u2, e2], dim=1))

    u1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
    d1 = model.dec1(torch.cat([u1, e1], dim=1))

    out = model.head(d1)
    return out, d1


def synthesize_ssl_batch_gpu(x_raw_cpu: torch.Tensor, sp_ras, device: torch.device):
    """
    Given a cached 6-channel external pelvic MRI tensor `x_raw_cpu` [6, K_full, H_full, W_full]:
    1. Crops a 3D training slab (up to 36 x 192 x 192, multiple of 8) and applies random flips/intensity jitter.
    2. Dynamically injects 8-20 realistic curved 3D susceptibility dark-core tubes:
       - r ~ U(1.0, 1.5) mm
       - physical slopes vx, vy ~ N(0, 0.15)
       - smoothly curved quadratic trajectories
       - partial-volume tip taper & random 1-2 slice dropouts
       - constructs exact 5-channel ground-truth tensor Y_syn (mask, peak, vx, vy, obt)
       - preserves all real pelvic vessels/bowel/fascia outside the injected tubes with X_raw[1] > 0.2
         as authentic anatomical hard negatives (Y_syn = 0)!
    3. Constructs a 3D Masked Autoencoding (MAE) input `x_mae` by randomly masking 25% of 8x16x16 3D patches
       and random axial slices in channels 0..3.
    """
    sp_i, sp_j, sp_k = float(sp_ras[0]), float(sp_ras[1]), float(sp_ras[2])
    _, K, H, W = x_raw_cpu.shape

    x = x_raw_cpu.to(device, non_blocking=True).clone()
    if K > 36:
        ks = random.randint(0, K - 36)
        x = x[:, ks:ks + 36]
        K = 36
    if H > 192:
        hs = random.randint(0, H - 192)
        x = x[:, :, hs:hs + 192, :]
        H = 192
    if W > 192:
        ws = random.randint(0, W - 192)
        x = x[:, :, :, ws:ws + 192]
        W = 192

    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])

    # Mild intensity jitter on clean volume
    scale = random.uniform(0.90, 1.12)
    bias = random.uniform(-0.04, 0.04)
    x[0] = (x[0] * scale + bias).clamp(-0.5, 2.5)

    # Save clean target for 3D MAE reconstruction of channels 0..3
    x_clean_target = x[0:4].clone()

    # Build 3D MAE masked input (25% random 8x16x16 blocks + 1-2 random axial slices masked in ch 0..3)
    x_mae = x.clone()
    kz_blk = max(1, min(8, K // 3))
    nb_k = max(1, math.ceil(K / kz_blk))
    nb_h = max(1, math.ceil(H / 16))
    nb_w = max(1, math.ceil(W / 16))
    blk_mask = (torch.rand((1, nb_k, nb_h, nb_w), device=device) < 0.25).float()
    blk_mask = F.interpolate(blk_mask.unsqueeze(0), size=(K, H, W), mode="nearest")[0]  # [1, K, H, W]
    if K >= 8:
        n_sl = random.randint(1, 2)
        for _ in range(n_sl):
            k_m = random.randint(0, K - 1)
            blk_mask[:, k_m, :, :] = 1.0

    noise_fill = torch.randn_like(x_mae[0:4]) * 0.08
    x_mae[0:4] = x_mae[0:4] * (1.0 - blk_mask) + noise_fill * blk_mask

    # Inject 8-20 synthetic 3D susceptibility tubes into x_syn
    x_syn = x.clone()
    n_tubes = random.randint(8, 20)
    ks_vec = torch.arange(K, device=device, dtype=torch.float32)  # [K]
    zs_mm = ks_vec * sp_k  # [K]

    # Sample bundle center in central 55% of H, W
    bundle_cx = random.uniform(0.32 * W, 0.68 * W)
    bundle_cy = random.uniform(0.32 * H, 0.68 * H)

    ic_nk = torch.zeros((n_tubes, K), device=device, dtype=torch.float32)
    jc_nk = torch.zeros((n_tubes, K), device=device, dtype=torch.float32)
    vx_nk = torch.zeros((n_tubes, K), device=device, dtype=torch.float32)
    vy_nk = torch.zeros((n_tubes, K), device=device, dtype=torch.float32)
    active_nk = torch.zeros((n_tubes, K), device=device, dtype=torch.bool)
    taper_nk = torch.ones((n_tubes, K), device=device, dtype=torch.float32)
    drop_nk = torch.ones((n_tubes, K), device=device, dtype=torch.float32)
    r_n = torch.empty((n_tubes, 1, 1, 1), device=device, dtype=torch.float32).uniform_(1.0, 1.5)
    is_obt_n = torch.zeros((n_tubes,), device=device, dtype=torch.float32)

    for ti in range(n_tubes):
        is_short_obt = (random.random() < 0.28) and (K >= 10)
        if is_short_obt:
            is_obt_n[ti] = 1.0
            k_start = random.randint(int(0.20 * K), max(int(0.20 * K), int(0.55 * K)))
            min_len = max(4, int(round(13.0 / max(1.0, sp_k))))
            max_len = max(min_len + 1, int(round(32.0 / max(1.0, sp_k))))
            k_end = min(K - 1, k_start + random.randint(min_len, max_len))
            # Obturator needles lie more laterally
            side = -1.0 if random.random() < 0.5 else 1.0
            x0_px = bundle_cx + side * random.uniform(18.0 / sp_i, min(0.38 * W, 42.0 / sp_i))
            y0_px = bundle_cy + random.uniform(-25.0 / sp_j, 25.0 / sp_j)
        else:
            is_obt_n[ti] = 0.0
            k_start = random.randint(0, min(3, max(0, K // 5)))
            k_end = random.randint(max(k_start + 4, int(0.45 * K)), K - 1)
            x0_px = bundle_cx + random.gauss(0.0, 18.0 / sp_i)
            y0_px = bundle_cy + random.gauss(0.0, 18.0 / sp_j)

        x0_px = float(np.clip(x0_px, 8.0, W - 9.0))
        y0_px = float(np.clip(y0_px, 8.0, H - 9.0))

        # Physical slopes vx, vy ~ N(0, 0.15) mm/mm + smooth quadratic curvature
        vx0 = float(np.clip(random.gauss(0.0, 0.15), -0.50, 0.50))
        vy0 = float(np.clip(random.gauss(0.0, 0.15), -0.50, 0.50))
        cx = random.gauss(0.0, 0.0012)
        cy = random.gauss(0.0, 0.0012)

        dz = zs_mm - float(zs_mm[k_start])
        dx_mm = vx0 * dz + 0.5 * cx * (dz ** 2)
        dy_mm = vy0 * dz + 0.5 * cy * (dz ** 2)

        ic_t = x0_px + dx_mm / sp_i
        jc_t = y0_px + dy_mm / sp_j
        vx_t = torch.clamp(vx0 + cx * dz, -0.75, 0.75)
        vy_t = torch.clamp(vy0 + cy * dz, -0.75, 0.75)

        in_bounds = (ks_vec >= k_start) & (ks_vec <= k_end) & (ic_t >= 3.0) & (ic_t < W - 3.0) & (jc_t >= 3.0) & (jc_t < H - 3.0)
        if int(in_bounds.sum()) < 4:
            continue

        ic_nk[ti] = ic_t
        jc_nk[ti] = jc_t
        vx_nk[ti] = vx_t
        vy_nk[ti] = vy_t
        active_nk[ti] = in_bounds

        # Partial-volume tip taper on the top 1-2 slices of the trajectory
        taper_nk[ti, k_end] = random.uniform(0.45, 0.75)
        if k_end - 1 > k_start:
            taper_nk[ti, k_end - 1] = random.uniform(0.75, 0.95)

        # Occasional 1-slice signal dropout along the shaft so encoder learns axial continuity
        if k_end - k_start >= 6 and random.random() < 0.35:
            k_drop = random.randint(k_start + 2, k_end - 2)
            drop_nk[ti, k_drop] = random.uniform(0.15, 0.35)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Distance [n_tubes, K, H, W] in mm
    d_mm = torch.hypot(
        (xx[None, None, :, :] - ic_nk[:, :, None, None]) * sp_i,
        (yy[None, None, :, :] - jc_nk[:, :, None, None]) * sp_j,
    )
    d_mm = torch.where(active_nk[:, :, None, None], d_mm, torch.full_like(d_mm, 999.0))

    # Construct 5-channel ground-truth target Y_syn [5, K, H, W]
    m_nk = torch.clamp((1.90 - d_mm) / (1.90 - 1.15), 0.0, 1.0)
    p_nk = torch.exp(-0.5 * (d_mm / 0.70) ** 2) * (d_mm <= 2.50)

    t_mask = m_nk.max(dim=0).values
    t_peak = p_nk.max(dim=0).values

    min_d, argmin_n = d_mm.min(dim=0)  # [K, H, W]
    fg_v = (min_d <= 1.85).float()
    vx_exp = vx_nk[:, :, None, None].expand(-1, -1, H, W)
    vy_exp = vy_nk[:, :, None, None].expand(-1, -1, H, W)
    t_vx = torch.gather(vx_exp, 0, argmin_n.unsqueeze(0)).squeeze(0) * fg_v
    t_vy = torch.gather(vy_exp, 0, argmin_n.unsqueeze(0)).squeeze(0) * fg_v
    t_obt = (m_nk * is_obt_n[:, None, None, None]).max(dim=0).values

    Y_syn = torch.stack([t_mask, t_peak, t_vx, t_vy, t_obt], dim=0)  # [5, K, H, W]

    # Render realistic 3D susceptibility dark-core + bright-rim physics signature into x_syn[0..3]
    eff_amp = taper_nk[:, :, None, None] * drop_nk[:, :, None, None]
    core_prof = (torch.exp(-0.5 * (d_mm / r_n) ** 2) * eff_amp * (d_mm <= 3.2)).max(dim=0).values
    rim_prof = (torch.exp(-0.5 * ((d_mm - 1.75 * r_n) / 0.60) ** 2) * eff_amp * (d_mm <= 3.8)).max(dim=0).values

    dark_str = random.uniform(0.72, 0.92)
    sym_str = random.uniform(0.68, 1.12)
    cs_str = random.uniform(0.60, 1.05)
    dr_str = random.uniform(0.55, 0.88)

    x_syn[0] = torch.clamp(x_syn[0] * (1.0 - dark_str * core_prof) - 0.22 * core_prof + 0.08 * rim_prof, -0.5, 2.5)
    speckle = (torch.randn_like(core_prof) * 0.05) * (core_prof > 0.15).float()
    x_syn[1] = torch.clamp(torch.maximum(x_syn[1], core_prof * sym_str + speckle), 0.0, 1.5)
    x_syn[2] = torch.clamp(torch.maximum(x_syn[2], core_prof * cs_str + speckle), 0.0, 1.5)
    x_syn[3] = torch.clamp(torch.maximum(x_syn[3], core_prof * dr_str + speckle * 0.5), -0.5, 1.0)

    return x_syn, Y_syn, x_mae, x_clean_target, blk_mask


def main():
    t0 = time.perf_counter()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cached_items = prepare_ssl_51_cache()
    n_vols = len(cached_items)

    # Select RTX 4090 automatically
    best_gpu = 0
    for g_i in range(torch.cuda.device_count()):
        if "4090" in torch.cuda.get_device_name(g_i):
            best_gpu = g_i
            break
    torch.cuda.set_device(best_gpu)
    device = torch.device(f"cuda:{best_gpu}")
    print(f"[SSL Pretrain] Training on {device} ({torch.cuda.get_device_name(best_gpu)}) across {n_vols} external pelvic MRI volumes for 45 epochs...", flush=True)

    model = NeedleUNet3D(in_ch=6, base_ch=24).to(device)
    ssl_recon_head = nn.Conv3d(24, 4, kernel_size=1).to(device)

    params = list(model.parameters()) + list(ssl_recon_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=2.0e-3, weight_decay=1e-4)
    epochs = 45
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    # Measure authentic anatomical hard-negative fraction across the 51 volumes
    hard_neg_fracs = [float((it["X"][1] > 0.20).float().mean()) for it in cached_items]
    print(f"[SSL Pretrain] Mean fraction of voxels with high radial symmetry (X_raw[1] > 0.20) acting as anatomical hard negatives: {100.0*np.mean(hard_neg_fracs):.2f}%")

    history = []
    best_ssl_score = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        ssl_recon_head.train()
        order = list(range(n_vols))
        random.shuffle(order)

        ep_loss_tot = 0.0
        ep_loss_tube = 0.0
        ep_loss_mae = 0.0
        ep_syn_dice = 0.0

        optimizer.zero_grad()
        for step_i, idx in enumerate(order):
            item = cached_items[idx]
            x_syn, y_syn, x_mae, x_clean_t, blk_mask = synthesize_ssl_batch_gpu(
                item["X"], item["sp_ras"], device
            )

            x_syn_b = x_syn.unsqueeze(0)
            y_syn_b = y_syn.unsqueeze(0)
            x_mae_b = x_mae.unsqueeze(0)
            x_clean_b = x_clean_t.unsqueeze(0)
            w_mae = 0.25 + 0.75 * blk_mask.unsqueeze(0)  # [1, 1, K, H, W]

            with torch.amp.autocast("cuda"):
                # Branch B: Synthetic 3D susceptibility tube segmentation + real pelvic hard-negative suppression
                out_syn, d1_syn = forward_unet_with_dec1(model, x_syn_b)
                loss_tube = compute_multitask_loss(out_syn, y_syn_b, x_syn_b)

                # Branch A: 3D Masked Volume & Physics Feature Autoencoding (3D MAE)
                _, d1_mae = forward_unet_with_dec1(model, x_mae_b)
                pred_recon = ssl_recon_head(d1_mae)
                l1_mae = (F.l1_loss(pred_recon, x_clean_b, reduction="none") * w_mae).mean()
                mse_mae = (F.mse_loss(pred_recon, x_clean_b, reduction="none") * w_mae).mean()
                loss_mae = l1_mae + mse_mae

                loss = loss_tube + 0.30 * loss_mae

            scaler.scale(loss).backward()
            if (step_i + 1) % 2 == 0 or (step_i + 1) == len(order):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                pm = torch.sigmoid(out_syn[0, 0])
                tm = y_syn[0]
                dice_step = float((2.0 * (pm * tm).sum() + 1.0) / (pm.sum() + tm.sum() + 1.0))

            ep_loss_tot += float(loss.detach().cpu())
            ep_loss_tube += float(loss_tube.detach().cpu())
            ep_loss_mae += float(loss_mae.detach().cpu())
            ep_syn_dice += dice_step

        scheduler.step()
        m_tot = ep_loss_tot / n_vols
        m_tube = ep_loss_tube / n_vols
        m_mae = ep_loss_mae / n_vols
        m_dice = ep_syn_dice / n_vols

        score = m_dice - 0.1 * m_mae
        if score > best_ssl_score or ep >= epochs - 5:
            best_ssl_score = max(best_ssl_score, score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            rec_entry = {
                "epoch": ep,
                "loss_total": round(m_tot, 4),
                "loss_tube": round(m_tube, 4),
                "loss_mae": round(m_mae, 4),
                "syn_tube_dice": round(m_dice, 4),
                "elapsed_sec": round(time.perf_counter() - t0, 1),
            }
            history.append(rec_entry)
            print(
                f"[SSL Pretrain] Ep {ep:02d}/{epochs} | Loss={m_tot:.3f} (Tube={m_tube:.3f}, MAE={m_mae:.4f}) | "
                f"SynTubeDice={m_dice:.3f} | ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    torch.save(best_state, SSL_CKPT_PATH)
    torch.save(
        {"unet_state_dict": best_state, "ssl_recon_head_state_dict": ssl_recon_head.cpu().state_dict()},
        SSL_CKPT_PATH.replace(".pt", "_full_with_mae_head.pt"),
    )

    summary = {
        "step": "Step 2: Self-Supervised 3D Encoder Pretraining on 51 External Pelvic MRI Volumes (Item 5)",
        "n_external_volumes": n_vols,
        "n_msd_prostate_3t_t2w_volumes": 48,
        "n_bwh_non_benchmark_pelvic_t2w_volumes": 3,
        "overlap_with_54_benchmark_cases": 0,
        "mean_anatomical_hard_negative_voxel_fraction_rsym_gt_0p2": round(float(np.mean(hard_neg_fracs)), 4),
        "epochs": epochs,
        "gpu_used": torch.cuda.get_device_name(best_gpu),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "checkpoint_path": SSL_CKPT_PATH,
        "final_epoch_metrics": history[-1],
        "training_history": history,
    }
    with open(SSL_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] SSL pretrained weights to {SSL_CKPT_PATH} and summary to {SSL_SUMMARY_PATH} in {time.perf_counter()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
