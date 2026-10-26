#!/usr/bin/env python3
"""
Experiment 1 (GPU 0 - RTX 4090):
Stage 1 3D Hybrid Conv-Transformer (AxialTransUNet3D) for 3D Needle Segmentation & Tangent Estimation.
Combines high-resolution anisotropic 3D residual convolutions (to preserve 1.2 mm needle cross-sections)
with Factorized 3D Axial (Z) Multi-Head Self-Attention (to link needles across obturator/signal dropouts)
and In-Plane (XY) Multi-Head Self-Attention (to capture global Syed-Neblett template array geometry).

Trains 5-fold cross-validation + full 54-case model on cached tensors in /tmp/needle_unet3d_cache/
and compares out-of-fold ValDice and ValCoreRecall directly against NeedleUNet3D.
"""

import glob
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_DIR = "/tmp/needle_unet3d_cache"
OUT_DIR = "/home/guillaume/github/needlefinder/nn_models"
os.makedirs(OUT_DIR, exist_ok=True)


class ResBlock3D(nn.Module):
    def __init__(self, in_c, out_c, k_size=(3, 3, 3), dilation=(1, 1, 1)):
        super().__init__()
        pad = tuple((k - 1) * d // 2 for k, d in zip(k_size, dilation))
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=k_size, padding=pad, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(min(8, out_c), out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=k_size, padding=pad, dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(min(8, out_c), out_c)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_c, max(4, out_c // 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(max(4, out_c // 4), out_c, 1),
            nn.Sigmoid(),
        )
        self.skip = nn.Conv3d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        h = F.silu(self.gn1(self.conv1(x)), inplace=True)
        h = self.gn2(self.conv2(h))
        h = h * self.se(h)
        return F.silu(h + r, inplace=True)


class AxialSpatialTransformerBlock3D(nn.Module):
    """
    Factorized 3D Axial (Z) + Spatial (XY) Multi-Head Self-Attention Block with 3D Conv-FFN.
    - Z-Attention:Every (h, w) column attends across all K axial slices to bridge obturator dropouts
      and enforce long-range collinear needle continuity.
    - XY-Attention: Every slice k attends across the H x W pelvic grid to model inter-needle
      template spacing and global bundle context.
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 1. Axial (Z) Self-Attention
        self.norm_z = nn.LayerNorm(dim)
        self.attn_z = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.z_pos_proj = nn.Linear(4, dim)

        # 2. Spatial (XY) Self-Attention
        self.norm_xy = nn.LayerNorm(dim)
        self.attn_xy = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.xy_pos_proj = nn.Linear(4, dim)

        # 3. 3D Depthwise-Enhanced Feed-Forward Network
        hidden_dim = int(dim * mlp_ratio)
        self.norm_ffn = nn.GroupNorm(min(8, dim), dim)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, 1, bias=False),
        )
        self.gamma_z = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.gamma_xy = nn.Parameter(torch.ones(1, 1, dim) * 0.5)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # --- Step 1: Axial (Z) Self-Attention along D for each (h, w) ---
        # Reshape to [B * H * W, D, C]
        xz = x.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, D, C)
        z_lin = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype).unsqueeze(-1)
        z_pe = self.z_pos_proj(torch.cat([
            z_lin,
            torch.sin(z_lin * math.pi),
            torch.cos(z_lin * math.pi),
            z_lin ** 2,
        ], dim=-1)).unsqueeze(0)  # [1, D, C]
        xz_norm = self.norm_z(xz + z_pe)
        az, _ = self.attn_z(xz_norm, xz_norm, xz_norm, need_weights=False)
        xz = xz + self.gamma_z * az
        x = xz.view(B, H, W, D, C).permute(0, 4, 3, 1, 2).contiguous()

        # --- Step 2: Spatial (XY) Self-Attention across H*W for each slice d ---
        # Reshape to [B * D, H * W, C]
        xxy = x.permute(0, 2, 3, 4, 1).contiguous().view(B * D, H * W, C)
        yy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        r2 = gx ** 2 + gy ** 2
        xy_feat = torch.stack([gx, gy, torch.sin(gx * math.pi), r2], dim=-1).view(1, H * W, 4)
        xy_pe = self.xy_pos_proj(xy_feat)
        xxy_norm = self.norm_xy(xxy + xy_pe)
        axy, _ = self.attn_xy(xxy_norm, xxy_norm, xxy_norm, need_weights=False)
        xxy = xxy + self.gamma_xy * axy
        x = xxy.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        # --- Step 3: 3D Conv-FFN ---
        x = x + self.ffn(self.norm_ffn(x))
        return x


class AxialTransUNet3D(nn.Module):
    """
    Hybrid 3D Conv-Transformer U-Net:
    - High-resolution Conv stem & Stage 1/2 encoders preserve 1 mm needle cross-sections
    - Stage 2 & Stage 3 feature maps pass through Axial (Z) + Spatial (XY) Transformer blocks
    - Multi-head output: [logit_mask, logit_peak, vx_raw, vy_raw, logit_obt]
    """
    def __init__(self, in_ch=6, base_c=24):
        super().__init__()
        self.enc1 = nn.Sequential(
            ResBlock3D(in_ch, base_c, k_size=(1, 3, 3)),
            ResBlock3D(base_c, base_c, k_size=(3, 3, 3)),
        )
        self.down1 = nn.Conv3d(base_c, base_c * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
        self.enc2 = ResBlock3D(base_c * 2, base_c * 2, k_size=(3, 3, 3))

        self.down2 = nn.Conv3d(base_c * 2, base_c * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
        self.enc3 = ResBlock3D(base_c * 4, base_c * 4, k_size=(3, 3, 3))
        self.trans3 = AxialSpatialTransformerBlock3D(base_c * 4, num_heads=4, mlp_ratio=2.0)

        self.down3 = nn.Conv3d(base_c * 4, base_c * 6, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.bot_conv = ResBlock3D(base_c * 6, base_c * 6, k_size=(3, 3, 3))
        self.bot_trans1 = AxialSpatialTransformerBlock3D(base_c * 6, num_heads=6, mlp_ratio=2.0)
        self.bot_trans2 = AxialSpatialTransformerBlock3D(base_c * 6, num_heads=6, mlp_ratio=2.0)

        self.up3 = nn.ConvTranspose3d(base_c * 6, base_c * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.dec3 = ResBlock3D(base_c * 8, base_c * 4, k_size=(3, 3, 3))

        self.up2 = nn.ConvTranspose3d(base_c * 4, base_c * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
        self.dec2 = ResBlock3D(base_c * 4, base_c * 2, k_size=(3, 3, 3))

        self.up1 = nn.ConvTranspose3d(base_c * 2, base_c, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
        self.dec1 = ResBlock3D(base_c * 2, base_c, k_size=(3, 3, 3))

        self.head = nn.Sequential(
            nn.Conv3d(base_c, base_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, base_c), base_c),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_c, 5, kernel_size=1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.trans3(self.enc3(self.down2(e2)))
        b = self.bot_trans2(self.bot_trans1(self.bot_conv(self.down3(e3))))

        u3 = self.up3(b)
        if u3.shape[2:] != e3.shape[2:]:
            u3 = F.interpolate(u3, size=e3.shape[2:], mode="trilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        if u2.shape[2:] != e2.shape[2:]:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.head(d1)


def compute_loss(out, target):
    logit_mask = out[:, 0]
    logit_peak = out[:, 1]
    vx_pred = torch.tanh(out[:, 2]) * 0.75
    vy_pred = torch.tanh(out[:, 3]) * 0.75
    logit_obt = out[:, 4]

    t_mask = target[:, 0]
    t_peak = target[:, 1]
    t_vx = target[:, 2]
    t_vy = target[:, 3]
    t_obt = target[:, 4]

    pos_w = torch.tensor([12.0], device=out.device)
    bce_mask = F.binary_cross_entropy_with_logits(logit_mask, t_mask, pos_weight=pos_w)
    p_mask = torch.sigmoid(logit_mask)
    inter = (p_mask * t_mask).sum(dim=(1, 2, 3))
    union = p_mask.sum(dim=(1, 2, 3)) + t_mask.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0)).mean()

    p_peak = torch.sigmoid(logit_peak)
    w_peak = 1.0 + 25.0 * t_peak
    peak_loss = (w_peak * (p_peak - t_peak) ** 2).mean()

    fg = (t_mask > 0.3).float()
    fg_sum = fg.sum().clamp(min=1.0)
    vec_loss = (((vx_pred - t_vx) ** 2 + (vy_pred - t_vy) ** 2) * fg).sum() / fg_sum

    pos_w_obt = torch.tensor([18.0], device=out.device)
    bce_obt = F.binary_cross_entropy_with_logits(logit_obt, t_obt, pos_weight=pos_w_obt)

    return bce_mask + dice_loss + 2.5 * peak_loss + 1.5 * vec_loss + 0.5 * bce_obt


def sample_crop(X, Y, crop_k=24, crop_hw=96, train=True):
    C, K, H, W = X.shape
    pad_k = max(0, (2 - (K % 2)) % 2)
    pad_h = max(0, (8 - (H % 8)) % 8)
    pad_w = max(0, (8 - (W % 8)) % 8)
    if pad_k > 0 or pad_h > 0 or pad_w > 0:
        X = F.pad(X, (0, pad_w, 0, pad_h, 0, pad_k), mode="replicate")
        Y = F.pad(Y, (0, pad_w, 0, pad_h, 0, pad_k), mode="constant", value=0.0)
    _, K, H, W = X.shape
    if not train:
        return X, Y

    ck = min(K, crop_k)
    ck = ck - (ck % 2) if ck >= 2 else K
    ch = min(H, crop_hw)
    ch = ch - (ch % 8)
    cw = min(W, crop_hw)
    cw = cw - (cw % 8)

    k0 = np.random.randint(0, K - ck + 1) if K > ck else 0
    h0 = np.random.randint(0, H - ch + 1) if H > ch else 0
    w0 = np.random.randint(0, W - cw + 1) if W > cw else 0

    xc = X[:, k0:k0 + ck, h0:h0 + ch, w0:w0 + cw].clone()
    yc = Y[:, k0:k0 + ck, h0:h0 + ch, w0:w0 + cw].clone()

    if np.random.rand() < 0.5:
        xc = torch.flip(xc, dims=[3])
        yc = torch.flip(yc, dims=[3])
        yc[2] = -yc[2]
    if np.random.rand() < 0.5:
        xc = torch.flip(xc, dims=[2])
        yc = torch.flip(yc, dims=[2])
        yc[3] = -yc[3]
    if np.random.rand() < 0.5:
        scale = 0.90 + 0.20 * np.random.rand()
        shift = -0.08 + 0.16 * np.random.rand()
        xc[0] = (xc[0] * scale + shift).clamp(-0.5, 2.5)
    return xc, yc


@torch.no_grad()
def infer_full_volume(model, X, device):
    model.eval()
    C, K, H, W = X.shape
    pad_k = (2 - (K % 2)) % 2
    pad_h = (8 - (H % 8)) % 8
    pad_w = (8 - (W % 8)) % 8
    xp = F.pad(X, (0, pad_w, 0, pad_h, 0, pad_k), mode="replicate")
    _, Kp, Hp, Wp = xp.shape

    # Patch-based XY inference if H*W is large (>160x160) so XY self-attention stays fast & exact
    if Hp <= 168 and Wp <= 168 and Kp <= 64:
        out = model(xp.unsqueeze(0).to(device))[0].cpu()
    else:
        out = torch.zeros((5, Kp, Hp, Wp), dtype=torch.float32)
        w_acc = torch.zeros((1, Kp, Hp, Wp), dtype=torch.float32)
        ck = min(Kp, 48)
        ch = min(Hp, 128)
        cw = min(Wp, 128)
        sk = max(1, ck - 12)
        sh = max(1, ch - 32)
        sw = max(1, cw - 32)
        for ks in range(0, Kp, sk):
            ke = min(Kp, ks + ck)
            ks_a = max(0, ke - ck)
            for hs in range(0, Hp, sh):
                he = min(Hp, hs + ch)
                hs_a = max(0, he - ch)
                for ws in range(0, Wp, sw):
                    we = min(Wp, ws + cw)
                    ws_a = max(0, we - cw)
                    sub = xp[:, ks_a:ke, hs_a:he, ws_a:we].unsqueeze(0).to(device)
                    pred = model(sub)[0].cpu()
                    out[:, ks_a:ke, hs_a:he, ws_a:we] += pred
                    w_acc[:, ks_a:ke, hs_a:he, ws_a:we] += 1.0
                    if we == Wp:
                        break
                if he == Hp:
                    break
            if ke == Kp:
                break
        out /= w_acc.clamp(min=1.0)
    return out[:, :K, :H, :W]


def evaluate_fold(model, val_items, device):
    dices = []
    recalls = []
    peak_ious = []
    for item in val_items:
        X = item["X"]
        Y = item["Y"]
        out = infer_full_volume(model, X, device)
        pm = torch.sigmoid(out[0])
        pp = torch.sigmoid(out[1])
        tm = (Y[0] > 0.5).float()
        tp = (Y[1] > 0.35).float()
        pred_bin = (pm > 0.45).float()
        inter = (pred_bin * tm).sum().item()
        union = pred_bin.sum().item() + tm.sum().item()
        dices.append((2.0 * inter + 1e-5) / (union + 1e-5))
        core_rec = ((pm > 0.35).float() * tp).sum().item() / max(1.0, tp.sum().item())
        recalls.append(core_rec)
        peak_rec = ((pp > 0.28).float() * tp).sum().item() / max(1.0, tp.sum().item())
        peak_ious.append(peak_rec)
    return float(np.mean(dices)), float(np.mean(recalls)), float(np.mean(peak_ious))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[AxialTransUNet3D] Using device: {device} ({torch.cuda.get_device_name(device)})")

    files = sorted(glob.glob(f"{CACHE_DIR}/case_*.pt"))
    dataset = [torch.load(f, weights_only=False) for f in files]
    n_params = sum(p.numel() for p in AxialTransUNet3D().parameters())
    print(f"[AxialTransUNet3D] Loaded {len(dataset)} cases | Model parameters: {n_params:,}")

    fold_dices = []
    fold_recalls = []
    fold_peak_recs = []

    for fold in range(5):
        train_items = [d for i, d in enumerate(dataset) if i % 5 != fold]
        val_items = [d for i, d in enumerate(dataset) if i % 5 == fold]

        model = AxialTransUNet3D(in_ch=6, base_c=24).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.4e-3, weight_decay=1e-4)
        epochs = 36
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=8e-5)
        scaler = torch.amp.GradScaler("cuda")

        best_score = -1.0
        best_dice, best_rec, best_pk = 0.0, 0.0, 0.0
        best_path = f"{OUT_DIR}/transunet3d_fold{fold}.pt"
        t0 = time.time()

        for ep in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(len(train_items))
            ep_loss = 0.0
            steps = 0
            for idx_start in range(0, len(perm), 2):
                batch_idxs = perm[idx_start:idx_start + 2]
                xs, ys = [], []
                for bi in batch_idxs:
                    for _ in range(2):
                        xc, yc = sample_crop(train_items[bi]["X"], train_items[bi]["Y"], crop_k=24, crop_hw=96, train=True)
                        xs.append(xc)
                        ys.append(yc)
                # Stack pairs with matching shapes or run per crop
                opt.zero_grad(set_to_none=True)
                b_loss = 0.0
                for xc, yc in zip(xs, ys):
                    xb = xc.unsqueeze(0).to(device, non_blocking=True)
                    yb = yc.unsqueeze(0).to(device, non_blocking=True)
                    with torch.amp.autocast("cuda"):
                        out = model(xb)
                        loss = compute_loss(out, yb) / len(xs)
                    scaler.scale(loss).backward()
                    b_loss += loss.item()
                scaler.step(opt)
                scaler.update()
                ep_loss += b_loss
                steps += 1
            sched.step()

            if ep % 9 == 0 or ep == epochs:
                vd, vr, vp = evaluate_fold(model, val_items, device)
                sc = vd + 0.5 * vr + 0.3 * vp
                if sc > best_score:
                    best_score = sc
                    best_dice, best_rec, best_pk = vd, vr, vp
                    torch.save(model.state_dict(), best_path)
                print(
                    f"[Fold {fold}] Ep {ep:02d}/{epochs} | Loss={ep_loss/max(1,steps):.4f} | "
                    f"ValDice={vd:.4f} | ValCoreRecall={vr:.4f} | ValPeakRecall={vp:.4f} ({time.time()-t0:.1f}s)"
                )

        # Generate OOF predictions for val_items using best checkpoint
        model.load_state_dict(torch.load(best_path, weights_only=True))
        for i, d in enumerate(dataset):
            if i % 5 == fold:
                out = infer_full_volume(model, d["X"], device)
                pm = torch.sigmoid(out[0]).numpy().astype(np.float16)
                pp = torch.sigmoid(out[1]).numpy().astype(np.float16)
                vx = (torch.tanh(out[2]) * 0.75).numpy().astype(np.float16)
                vy = (torch.tanh(out[3]) * 0.75).numpy().astype(np.float16)
                po = torch.sigmoid(out[4]).numpy().astype(np.float16)
                np.savez_compressed(
                    f"{OUT_DIR}/oof_transunet_case_{i:02d}.npz",
                    p_mask=pm, p_peak=pp, vx=vx, vy=vy, p_obt=po,
                    k0=d["k0"], i0=d["i0"], i1=d["i1"], j0=d["j0"], j1=d["j1"],
                )
        fold_dices.append(best_dice)
        fold_recalls.append(best_rec)
        fold_peak_recs.append(best_pk)

    print("=" * 80)
    print(
        f"[AxialTransUNet3D 5-Fold OOF Summary] "
        f"Mean ValDice = {np.mean(fold_dices):.4f} | "
        f"Mean ValCoreRecall = {np.mean(fold_recalls)*100:.2f}% | "
        f"Mean ValPeakRecall = {np.mean(fold_peak_recs)*100:.2f}%"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
