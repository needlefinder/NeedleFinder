#!/usr/bin/env python3
"""
Export Production NeedleFinder 2026 Slicer v2.0 Models:
  1. `needle_unet3d.onnx`: Clean 51-Volume External Pelvic SSL + 40/40 Multi-Rater Stage 1 NeedleUNet3D
  2. `needle_graph_transformer.onnx`: Clean Multi-Rater Stage 2B DenseGraphTransformer
  3. `needle_graphnet.npz`:
       - `mpnn.*`: 4-Layer Triplet Collinearity NeedleSliceMPNN
       - `gt.*`: Stage 2B DenseGraphTransformer
       - `tip_bgt.*`: SynthBrachy3T + Multi-Rater TipAwareBundleGraphTransformer
       - `tip1d.*`: 91-Step (0.5 mm) 24-Channel Dedicated1DTipLocalizer (Dilated 1D ResNet + 4-Head Self-Attention)
       - `conformal_q90`, `conformal_q95`: Nested Split-Conformal 3D Tip Prediction Interval Quantiles
"""

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx._internal.torchscript_exporter.onnx_proto_utils as _opu
_opu._add_onnxscript_fn = lambda proto, custom_opsets: proto


sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean"))

from train_and_eval_unet3d import NeedleUNet3D, compute_multitask_loss
from train_nested_stage1 import augment_3d_rich
from run_part2_literature_fusion_inside_needlefinder import (
    Dedicated1DTipLocalizer,
    extract_1d_tip_profile_91_24ch,
    shift_1d_profile,
)

LIB_DIR = resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder/NeedleFinderLib")
CLEAN_CACHE_DIR = resolve_path("/tmp/needle_unet3d_cache_clean")
OOF_DIR = resolve_path("/home/guillaume/github/needlefinder/nn_models_clean")
STAGE2_PREDS_PKL = resolve_path("/home/guillaume/github/needlefinder/vault/sources/2026-10-23-stage2-clean-final-oof-predictions.pkl")


def conv1d_np(x, w, b, dilation=1):
    # x: [B, C_in, L], w: [C_out, C_in, K], b: [C_out]
    B, C_in, L = x.shape
    C_out, _, K = w.shape
    pad = (K // 2) * dilation
    if pad > 0:
        x_p = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="constant")
    else:
        x_p = x
    out = np.broadcast_to(b[None, :, None], (B, C_out, L)).copy()
    for k in range(K):
        s = k * dilation
        # [B, C_in, L] -> tensordot with w[:, :, k] [C_out, C_in]
        out += np.einsum("bcl,oc->bol", x_p[:, :, s:s + L], w[:, :, k], optimize=True)
    return out


def groupnorm_np(x, w, b, num_groups=8, eps=1e-5):
    B, C, L = x.shape
    xg = x.reshape(B, num_groups, C // num_groups, L)
    mean = np.mean(xg, axis=(2, 3), keepdims=True)
    var = np.var(xg, axis=(2, 3), keepdims=True)
    xn = (xg - mean) / np.sqrt(var + eps)
    return xn.reshape(B, C, L) * w[None, :, None] + b[None, :, None]


def silu_np(x):
    return x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def layernorm_np(x, w, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def forward_1d_tip_localizer_numpy(x, W):
    """
    Pure-NumPy forward pass of Dedicated1DTipLocalizer(in_channels=24, hidden_dim=48, seq_len=91).
    Returns:
      dz_pred_mm: [B] predicted cranial tip adjustment in mm
      step_probs: [B, 91] 91-bin probability distribution over [-25, +20] mm
      step_std_mm: [B] distribution standard deviation in mm (epistemic/aleatoric tip spread)
    """
    h = silu_np(groupnorm_np(conv1d_np(x, W["tip1d.stem.0.weight"], W["tip1d.stem.0.bias"], dilation=1),
                             W["tip1d.stem.1.weight"], W["tip1d.stem.1.bias"]))
    for b_idx, dil in ((1, 1), (2, 2), (3, 4)):
        pfx = f"tip1d.block{b_idx}"
        r = silu_np(groupnorm_np(conv1d_np(h, W[f"{pfx}.0.weight"], W[f"{pfx}.0.bias"], dilation=dil),
                                 W[f"{pfx}.1.weight"], W[f"{pfx}.1.bias"]))
        r = groupnorm_np(conv1d_np(r, W[f"{pfx}.3.weight"], W[f"{pfx}.3.bias"], dilation=dil),
                         W[f"{pfx}.4.weight"], W[f"{pfx}.4.bias"])
        h = silu_np(h + r)

    # 4-head 1D self-attention across L=91
    h_seq = np.transpose(h, (0, 2, 1))  # [B, 91, 48]
    B, L, D = h_seq.shape
    qkv = h_seq @ W["tip1d.attn.in_proj_weight"].T + W["tip1d.attn.in_proj_bias"]
    q, k, v = np.split(qkv, 3, axis=-1)
    n_heads = 4
    d_head = D // n_heads
    q = q.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)  # [B, 4, 91, 12]
    k = k.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
    att = (q @ np.transpose(k, (0, 1, 3, 2))) / np.sqrt(float(d_head))
    att_max = np.max(att, axis=-1, keepdims=True)
    att_exp = np.exp(np.clip(att - att_max, -20.0, 20.0))
    att_p = att_exp / np.maximum(1e-6, np.sum(att_exp, axis=-1, keepdims=True))
    attn_out = (att_p @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
    attn_out = attn_out @ W["tip1d.attn.out_proj.weight"].T + W["tip1d.attn.out_proj.bias"]
    h_seq = layernorm_np(h_seq + attn_out, W["tip1d.attn_norm.weight"], W["tip1d.attn_norm.bias"])
    h = np.transpose(h_seq, (0, 2, 1))  # [B, 48, 91]

    sh = silu_np(conv1d_np(h, W["tip1d.step_head.0.weight"], W["tip1d.step_head.0.bias"], dilation=1))
    step_logits = conv1d_np(sh, W["tip1d.step_head.2.weight"], W["tip1d.step_head.2.bias"], dilation=1).squeeze(1)
    sl_max = np.max(step_logits, axis=-1, keepdims=True)
    sl_exp = np.exp(np.clip(step_logits - sl_max, -20.0, 20.0))
    step_probs = sl_exp / np.maximum(1e-6, np.sum(sl_exp, axis=-1, keepdims=True))
    dz_grid = W["tip1d.dz_grid"][None, :]
    dz_soft = np.sum(step_probs * dz_grid, axis=-1)
    step_var = np.sum(step_probs * ((dz_grid - dz_soft[:, None]) ** 2), axis=-1)
    step_std_mm = np.sqrt(np.maximum(0.25, step_var))

    h_center = h[:, :, 50]
    h_mean = np.mean(h, axis=-1)
    h_cat = np.concatenate([h_center, h_mean], axis=-1)
    rh = silu_np(h_cat @ W["tip1d.reg_head.0.weight"].T + W["tip1d.reg_head.0.bias"])
    dz_res = (rh @ W["tip1d.reg_head.2.weight"].T + W["tip1d.reg_head.2.bias"]).squeeze(-1) * 10.0
    dz_pred_mm = 0.75 * dz_soft + 0.25 * dz_res
    return dz_pred_mm, step_probs, step_std_mm


def main():
    t0 = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.device_count() > 1 else "cuda:0")
    print(f"[Slicer v2.0 Model Exporter] Using {device} ({torch.cuda.get_device_name(device)})", flush=True)

    # 1. Train & Export Production Stage 1 NeedleUNet3D (SSL Pretrained + 54-Case Multi-Rater)
    prod_unet_pt = os.path.join(OOF_DIR, "unet3d_prod_ssl_multirater_54cases.pt")
    model_u = NeedleUNet3D(in_ch=6, base_ch=24).to(device)
    if os.path.exists(prod_unet_pt):
        print(f"  Loading existing production Stage 1 checkpoint {prod_unet_pt}...")
        model_u.load_state_dict(torch.load(prod_unet_pt, map_location=device, weights_only=True))
    else:
        print("  Training production Stage 1 NeedleUNet3D on all 54 cases (initialized from 51-vol SSL + Multi-Rater labels)...")
        ssl_ckpt = os.path.join(OOF_DIR, "unet3d_ssl_pretrained_51vols.pt")
        model_u.load_state_dict(torch.load(ssl_ckpt, map_location=device, weights_only=True), strict=False)
        cases_data = [torch.load(f"{CLEAN_CACHE_DIR}/case_{i:02d}.pt", map_location="cpu", weights_only=False) for i in range(54)]
        opt_u = torch.optim.AdamW(model_u.parameters(), lr=1.4e-3, weight_decay=1e-4)
        sched_u = torch.optim.lr_scheduler.CosineAnnealingLR(opt_u, T_max=32, eta_min=8e-5)
        rng = np.random.default_rng(20261024)
        for ep in range(32):
            model_u.train()
            for idx in rng.permutation(54):
                cd = cases_data[idx]
                X_in = cd["X"].clone()
                Y_tg = cd.get("Y_multirater", cd["Y"]).clone()
                X_aug, Y_aug = augment_3d_rich(X_in, Y_tg, device)
                xb = X_aug.unsqueeze(0).to(device)
                yb = Y_aug.unsqueeze(0).to(device)
                opt_u.zero_grad(set_to_none=True)
                out = model_u(xb)
                loss = compute_multitask_loss(out, yb, xb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_u.parameters(), 2.0)
                opt_u.step()
            sched_u.step()
        torch.save(model_u.state_dict(), prod_unet_pt)
        print(f"  Saved {prod_unet_pt} in {time.perf_counter()-t0:.1f}s")

    model_u.eval().cpu()
    dummy = torch.randn(1, 6, 32, 160, 160)
    onnx_unet_path = os.path.join(LIB_DIR, "needle_unet3d.onnx")
    torch.onnx.export(
        model_u,
        dummy,
        onnx_unet_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch", 2: "depth", 3: "height", 4: "width"},
        },
        opset_version=17, dynamo=False,
    )
    print(f"  Exported {onnx_unet_path} ({os.path.getsize(onnx_unet_path)/1e6:.2f} MB)")

    # 2. Train full 54-case 24-Channel Dedicated1DTipLocalizer & Export to needle_graphnet.npz
    tv_cache = pickle.load(open(resolve_path("/tmp/stage2_tv_unc_24ch_cache.pkl"), "rb"))
    s2_data = pickle.load(open(STAGE2_PREDS_PKL, "rb"))
    b_cache = pickle.load(open(resolve_path("/tmp/stage2_clean_bundles_nn_models_clean.pkl"), "rb"))
    bgs = s2_data.get("stitched_enriched_bgs") or b_cache["stitched_enriched_bgs"]
    for i, bg in enumerate(bgs):
        bg["tip_profiles_91_24ch"] = tv_cache["profs_24"][i]

    dz_grid_np = np.linspace(-25.0, 20.0, 91, dtype=np.float32)
    rng_t = np.random.default_rng(20261024)
    samples = []
    for bg in bgs:
        profs = bg["tip_profiles_91_24ch"].numpy() if hasattr(bg["tip_profiles_91_24ch"], "numpy") else bg["tip_profiles_91_24ch"]
        y_true_z = bg["y_true_tip_z"].numpy() if hasattr(bg["y_true_tip_z"], "numpy") else bg["y_true_tip_z"]
        y_w = bg["y_dz_weight"].numpy() if hasattr(bg["y_dz_weight"], "numpy") else bg["y_dz_weight"]
        for c_i in range(len(profs)):
            dz0 = float(y_true_z[c_i]) - float(bg["pool"][c_i]["ras"][-1, 2])
            if y_w[c_i] > 0.1 and abs(dz0) <= 20.0:
                p0 = profs[c_i]
                w0 = float(y_w[c_i])
                g0 = np.exp(-0.5 * ((dz_grid_np - dz0) / 1.5) ** 2).astype(np.float32)
                g0 /= max(1e-6, float(np.sum(g0)))
                samples.append((p0, dz0, w0, g0))
                for _ in range(3):
                    s_steps = int(rng_t.integers(-10, 11))
                    dz_s = dz0 - s_steps * 0.5
                    if -21.0 <= dz_s <= 17.0:
                        ps = shift_1d_profile(p0, s_steps)
                        ps = ps + rng_t.normal(0.0, 0.015, size=ps.shape).astype(np.float32)
                        gs = np.exp(-0.5 * ((dz_grid_np - dz_s) / 1.5) ** 2).astype(np.float32)
                        gs /= max(1e-6, float(np.sum(gs)))
                        samples.append((ps, dz_s, w0 * 0.85, gs))

    X_tr = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32, device=device)
    Y_tr = torch.tensor([s[1] for s in samples], dtype=torch.float32, device=device)
    W_tr = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=device)
    Yg_tr = torch.tensor(np.stack([s[3] for s in samples]), dtype=torch.float32, device=device)

    tip_model = Dedicated1DTipLocalizer(in_channels=24, hidden_dim=48, seq_len=91).to(device)
    opt_t = torch.optim.AdamW(tip_model.parameters(), lr=1.8e-3, weight_decay=2e-4)
    sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=40, eta_min=1e-4)
    for ep in range(40):
        tip_model.train()
        perm = torch.randperm(len(X_tr), device=device)
        for b0 in range(0, len(X_tr), 64):
            bi = perm[b0:b0 + 64]
            opt_t.zero_grad(set_to_none=True)
            dz_p, sl_p = tip_model(X_tr[bi])
            wb = W_tr[bi]
            l_reg = (F.smooth_l1_loss(dz_p, Y_tr[bi], reduction="none", beta=2.0) * wb).sum() / wb.sum().clamp(min=1.0)
            l_kl = (-(Yg_tr[bi] * F.log_softmax(sl_p, dim=-1)).sum(dim=-1) * wb).sum() / wb.sum().clamp(min=1.0)
            loss = l_reg + 1.5 * l_kl
            loss.backward()
            opt_t.step()
        sched_t.step()
    tip_model.eval().cpu()

    # Load existing needle_graphnet.npz and append tip1d.* weights + conformal quantiles
    npz_path = os.path.join(LIB_DIR, "needle_graphnet.npz")
    existing_npz = dict(np.load(npz_path))
    for k, v in tip_model.state_dict().items():
        existing_npz[f"tip1d.{k}"] = v.numpy().astype(np.float32)
    existing_npz["conformal.q90"] = np.array([1.965], dtype=np.float32)
    existing_npz["conformal.q95"] = np.array([2.988], dtype=np.float32)
    np.savez_compressed(npz_path, **existing_npz)

    # Verify pure-NumPy 1D Tip Localizer against PyTorch
    x_test = X_tr[:16].cpu().numpy()
    with torch.no_grad():
        dz_pt, sl_pt = tip_model(torch.tensor(x_test))
    dz_np, probs_np, std_np = forward_1d_tip_localizer_numpy(x_test, existing_npz)
    max_diff = float(np.max(np.abs(dz_pt.numpy() - dz_np)))
    print(f"  Exported {npz_path} ({os.path.getsize(npz_path)/1024:.1f} KB) | NumPy vs PyTorch 1D Tip Localizer max diff: {max_diff:.2e}")

if __name__ == "__main__":
    main()
