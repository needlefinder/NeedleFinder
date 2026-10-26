#!/usr/bin/env python3
"""
Step 3: Strictly Nested Stage 1 Training + Rich 3D Augmentations + Dual-Rater Labels
        + 5-Fold Inner Ensembling & 4-View 3D TTA + Stage 1 Ablation Study (Items 1, 2, 3, 5).

Protocol:
  - Outer 5-fold split: `outer_fold = idx % 5` (idx in 0..53), identical to the benchmark protocol.
  - Inside each outer fold `f_out in 0..4`:
    - `outer_test = [i for i in range(54) if i % 5 == f_out]` (NEVER touched during training or checkpoint selection!)
    - `outer_train = [i for i in range(54) if i % 5 != f_out]` (43 or 44 cases)
    - Partition `outer_train` into 5 inner folds `m in 0..4`:
      - `inner_val = outer_train[m::5]` (8 or 9 cases)
      - `inner_subtrain = [i for i in outer_train if i not in inner_val]` (34 or 35 cases)
  - Trains all 25 inner models `(f_out, m)`:
    - Initializes from `/home/guillaume/github/needlefinder/nn_models_clean/unet3d_ssl_pretrained_51vols.pt` (Item 5)
    - Supervises on `Y_multirater` (Item 3: CY + HT + LB/AMe consensus/soft-extra targets on `inner_subtrain`)
    - Applies Rich 3D Data Augmentation (Item 2)
    - Strictly Nested Checkpoint Selection (Item 1): Evaluates every 6 epochs strictly on `inner_val`
      (`m_dice + 0.5 * m_rec` against `Y[v_idx]`) and keeps the best checkpoint from `inner_val`.
  - Generates:
    - Inner-OOF predictions for `outer_train`:
      `/home/guillaume/github/needlefinder/nn_models_clean/inner_oof_f{f_out}_case_{v_idx:02d}.npz`
    - 5-Fold Inner-Ensembled + 4-View 3D TTA Outer-OOF predictions for `outer_test`:
      `/home/guillaume/github/needlefinder/nn_models_clean/oof_case_{t_idx:02d}.npz`
  - Runs comprehensive Stage 1 Voxel & Trajectory Ablations isolating Items 1, 2, 3, and 5, saving:
    `/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean/step3_nested_stage1_summary.json`
"""

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from reproducibility.paths import resolve_path
from reproducibility.checkpoints import write_checkpoint_record, read_checkpoint_record

import glob
import json
import math
import os
import pickle
import random
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

sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun"))
sys.path.insert(0, resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean"))

from train_and_eval_unet3d import NeedleUNet3D, compute_multitask_loss
from NeedleFinderLib.DatasetSplits import nested_split
from NeedleFinderLib.NeuralNeedleNet import extract_nn_labelmap_tubes
from evaluate_unified_apples_to_apples import evaluate_method_unified

CLEAN_CACHE_DIR = resolve_path("/tmp/needle_unet3d_cache_clean")
CLEAN_META_PATH = resolve_path("/tmp/cases_meta_54_clean.pkl")
CLEAN_BASE_TRAJS_PATH = resolve_path("/tmp/base_trajs_54cases_clean.pkl")
OLD_CACHE_DIR = resolve_path("/tmp/needle_unet3d_cache")
OLD_OOF_DIR = resolve_path("/home/guillaume/github/needlefinder/nn_models_eval")
OUT_DIR = resolve_path("/home/guillaume/github/needlefinder/nn_models_clean")
SSL_CKPT_PATH = os.path.join(OUT_DIR, "unet3d_ssl_pretrained_51vols.pt")
SUMMARY_JSON_PATH = resolve_path("/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean/step3_nested_stage1_summary.json")


# ==============================================================================
# 1. RICH 3D MRI DATA AUGMENTATION (Item 2) & BASIC AUGMENTATION (for Ablation)
# ==============================================================================

def augment_3d_rich(x_cpu: torch.Tensor, y_cpu: torch.Tensor, device: torch.device):
    """
    Fast GPU-accelerated Sharp Rich 3D MRI Data Augmentation on tensor x [6, K, H, W], y [5, K, H, W] (Item 2):
      1. Crop 3D training slab (K <= 36, H <= 192, W <= 192) and move to GPU
      2. Random X-flip (dims=[3], y[2] = -y[2]) and Y-flip (dims=[2], y[3] = -y[3])
      3. Random in-plane spatial shift (+/- 12 px, preserving exact sub-voxel target sharpness)
      4. Smooth 3D multiplicative MRI bias field B(z,y,x) interpolated from a 3x4x4 random grid in [0.82, 1.20]
      5. Intensity scaling, contrast gamma jitter, and Gaussian noise
      6. 1-to-2 slice signal dropout in x[1:4]
    """
    _, K, H, W = x_cpu.shape
    k_s = random.randint(0, K - 36) if K > 36 else 0
    k_e = min(K, k_s + 36)
    h_s = random.randint(0, H - 192) if H > 192 else 0
    h_e = min(H, h_s + 192)
    w_s = random.randint(0, W - 192) if W > 192 else 0
    w_e = min(W, w_s + 192)

    x = x_cpu[:, k_s:k_e, h_s:h_e, w_s:w_e].to(device, non_blocking=True).clone()
    y = y_cpu[:, k_s:k_e, h_s:h_e, w_s:w_e].to(device, non_blocking=True).clone()
    _, K, H, W = x.shape

    # 1. Random X-flip (left-right, W dimension)
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
        y = torch.flip(y, dims=[3])
        y[2] = -y[2]

    # 2. Random Y-flip (anterior-posterior, H dimension)
    if random.random() < 0.25:
        x = torch.flip(x, dims=[2])
        y = torch.flip(y, dims=[2])
        y[3] = -y[3]

    # 3. Random in-plane spatial shift (+/- 12 px)
    if random.random() < 0.75:
        dy = random.randint(-12, 12)
        dx = random.randint(-12, 12)
        x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        y = torch.roll(y, shifts=(dy, dx), dims=(2, 3))

    # 4. Smooth 3D multiplicative MRI bias field B(z,y,x) interpolated from 3x4x4 grid in [0.82, 1.20]
    if random.random() < 0.65:
        bias_grid = torch.empty((1, 1, 3, 4, 4), dtype=x.dtype, device=device).uniform_(0.82, 1.20)
        bias_field = F.interpolate(bias_grid, size=(K, H, W), mode="trilinear", align_corners=False)[0, 0]
        x[0] = torch.clamp(x[0] * bias_field, -0.5, 2.5)

    # 5. Intensity scaling, contrast gamma jitter, and Gaussian noise
    if random.random() < 0.70:
        scale = random.uniform(0.84, 1.18)
        shift = random.uniform(-0.07, 0.07)
        gamma = random.uniform(0.90, 1.12)
        x0_pos = torch.clamp(x[0] + 0.5, 0.0, 3.0)
        x[0] = torch.clamp((x0_pos.pow(gamma) - 0.5) * scale + shift, -0.5, 2.5)
        scale_r = random.uniform(0.86, 1.16)
        x[1] = torch.clamp(x[1] * scale_r, 0.0, 1.5)
        x[2] = torch.clamp(x[2] * scale_r, 0.0, 1.5)

    if random.random() < 0.45:
        sigma = random.uniform(0.01, 0.03)
        x[0] = torch.clamp(x[0] + torch.randn_like(x[0]) * sigma, -0.5, 2.5)

    # 6. Slice signal dropout in x[1:4]
    if K > 10 and random.random() < 0.50:
        k_drop = random.randint(2, K - 3)
        x[1:4, k_drop] *= random.uniform(0.12, 0.25)

    return x, y


def augment_3d_basic(x_cpu: torch.Tensor, y_cpu: torch.Tensor, device: torch.device):
    """Original basic augmentation on GPU for ablation comparison."""
    _, K, H, W = x_cpu.shape
    k_s = random.randint(0, K - 36) if K > 36 else 0
    k_e = min(K, k_s + 36)
    h_s = random.randint(0, H - 192) if H > 192 else 0
    h_e = min(H, h_s + 192)
    w_s = random.randint(0, W - 192) if W > 192 else 0
    w_e = min(W, w_s + 192)

    x = x_cpu[:, k_s:k_e, h_s:h_e, w_s:w_e].to(device, non_blocking=True).clone()
    y = y_cpu[:, k_s:k_e, h_s:h_e, w_s:w_e].to(device, non_blocking=True).clone()
    _, K, H, W = x.shape

    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
        y = torch.flip(y, dims=[3])
        y[2] = -y[2]
    if random.random() < 0.2:
        x = torch.flip(x, dims=[2])
        y = torch.flip(y, dims=[2])
        y[3] = -y[3]
    if random.random() < 0.7:
        dy = random.randint(-10, 10)
        dx = random.randint(-10, 10)
        x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        y = torch.roll(y, shifts=(dy, dx), dims=(2, 3))
    if random.random() < 0.7:
        scale = random.uniform(0.82, 1.20)
        bias = random.uniform(-0.08, 0.08)
        x[0] = (x[0] * scale + bias).clamp(-0.5, 2.5)
        scale_r = random.uniform(0.85, 1.18)
        x[1] = (x[1] * scale_r).clamp(0.0, 1.5)
        x[2] = (x[2] * scale_r).clamp(0.0, 1.5)
    if K > 10 and random.random() < 0.5:
        k_drop = random.randint(2, K - 3)
        x[1:4, k_drop] *= 0.15
    return x, y


# ==============================================================================
# 2. SLIDING-WINDOW 3D INFERENCE & 4-VIEW 3D TEST-TIME AUGMENTATION (TTA)
# ==============================================================================

def infer_single_view_3d(model: nn.Module, x_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Sliding Z-chunk inference returning raw 5-channel outputs [5, K, H, W] on CPU."""
    _, K, H, W = x_cpu.shape
    if K <= 40:
        with torch.no_grad(), torch.amp.autocast("cuda"):
            return model(x_cpu.unsqueeze(0).to(device))[0].float().cpu()
    out_cpu = torch.zeros((5, K, H, W), dtype=torch.float32)
    w_cpu = torch.zeros((1, K, 1, 1), dtype=torch.float32)
    chunk = 36
    stride = 24
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for ks in range(0, K, stride):
            ke = min(K, ks + chunk)
            if ke - ks < 8:
                ks = max(0, ke - chunk)
            sub = x_cpu[:, ks:ke].unsqueeze(0).to(device)
            o_sub = model(sub)[0].float().cpu()
            out_cpu[:, ks:ke] += o_sub
            w_cpu[:, ks:ke] += 1.0
            if ke == K:
                break
    return out_cpu / w_cpu.clamp(min=1.0)


def raw_out_to_probs(ov: torch.Tensor, temp_pos: float = 1.0) -> torch.Tensor:
    """
    Converts raw model output [5, K, H, W] to calibrated (p_mask, p_peak, vx, vy, p_obt) [5, K, H, W].
    When ensembling logits across multiple models/views, `temp_pos > 1.0` compensates for ensemble logit
    variance shrinkage on foreground catheters while preserving logit-space background suppression.
    """
    if temp_pos != 1.0:
        o0 = torch.where(ov[0] > -1.0, ov[0] * temp_pos, ov[0])
        o1 = torch.where(ov[1] > -1.0, ov[1] * temp_pos, ov[1])
        o4 = torch.where(ov[4] > -1.0, ov[4] * temp_pos, ov[4])
    else:
        o0, o1, o4 = ov[0], ov[1], ov[4]
    p_mask = torch.sigmoid(o0)
    p_peak = torch.sigmoid(o1)
    vx = torch.tanh(ov[2]) * 0.75
    vy = torch.tanh(ov[3]) * 0.75
    p_obt = torch.sigmoid(o4)
    return torch.stack([p_mask, p_peak, vx, vy, p_obt], dim=0)


def infer_4view_tta_logits(model: nn.Module, x_cpu: torch.Tensor, device: torch.device, use_tta: bool = True) -> torch.Tensor:
    """
    Runs 4-view 3D Test-Time Augmentation (TTA) in raw logit space and returns combined [5, K, H, W] logits on CPU:
      - View 0 (w=0.42): identity x (anatomically native orientation)
      - View 1 (w=0.42): X-flip flip(x, [3]) -> unflip & negate vx (anatomically left-right symmetric)
      - View 2 (w=0.08): Y-flip flip(x, [2]) -> unflip & negate vy (anterior-posterior reflection)
      - View 3 (w=0.08): XY-flip flip(x, [2, 3]) -> unflip & negate both vx, vy
    """
    o0 = infer_single_view_3d(model, x_cpu, device)
    if not use_tta:
        return o0

    x1 = torch.flip(x_cpu, dims=[3])
    o1 = torch.flip(infer_single_view_3d(model, x1, device), dims=[3])
    o1[2] = -o1[2]

    x2 = torch.flip(x_cpu, dims=[2])
    o2 = torch.flip(infer_single_view_3d(model, x2, device), dims=[2])
    o2[3] = -o2[3]

    x3 = torch.flip(x_cpu, dims=[2, 3])
    o3 = torch.flip(infer_single_view_3d(model, x3, device), dims=[2, 3])
    o3[2] = -o3[2]
    o3[3] = -o3[3]

    return 0.42 * o0 + 0.42 * o1 + 0.08 * o2 + 0.08 * o3


def infer_both_single_and_4view_tta_logits(model: nn.Module, x_cpu: torch.Tensor, device: torch.device):
    """Returns both single-view logits `o0` and 4-view TTA logits `o_tta` without redundant forward passes."""
    o0 = infer_single_view_3d(model, x_cpu, device)

    x1 = torch.flip(x_cpu, dims=[3])
    o1 = torch.flip(infer_single_view_3d(model, x1, device), dims=[3])
    o1[2] = -o1[2]

    x2 = torch.flip(x_cpu, dims=[2])
    o2 = torch.flip(infer_single_view_3d(model, x2, device), dims=[2])
    o2[3] = -o2[3]

    x3 = torch.flip(x_cpu, dims=[2, 3])
    o3 = torch.flip(infer_single_view_3d(model, x3, device), dims=[2, 3])
    o3[2] = -o3[2]
    o3[3] = -o3[3]

    o_tta = 0.42 * o0 + 0.42 * o1 + 0.08 * o2 + 0.08 * o3
    return o0, o_tta


def infer_4view_tta_probs(model: nn.Module, x_cpu: torch.Tensor, device: torch.device, use_tta: bool = True) -> torch.Tensor:
    """Runs single-view or 4-view 3D TTA and returns calibrated probabilities [5, K, H, W] on CPU."""
    ov = infer_4view_tta_logits(model, x_cpu, device, use_tta=use_tta)
    return raw_out_to_probs(ov, temp_pos=(1.04 if use_tta else 1.0))


# ==============================================================================
# 3. STRICTLY NESTED INNER-FOLD TRAINING WORKER
# ==============================================================================

def train_single_inner_model(
    items,
    inner_subtrain,
    inner_val,
    device: torch.device,
    ssl_ckpt_path: str = SSL_CKPT_PATH,
    use_multirater: bool = True,
    use_rich_aug: bool = True,
    epochs: int = 42,
    eval_every: int = 6,
    seed: int = 42,
    tag: str = "",
):
    """
    Trains a single inner-fold NeedleUNet3D model on `inner_subtrain` and selects the best checkpoint
    strictly on `inner_val` (`m_dice + 0.5 * m_rec` against `Y[v_idx]`), NEVER touching `outer_test`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = NeedleUNet3D(in_ch=6, base_ch=24).to(device)
    if ssl_ckpt_path and os.path.exists(ssl_ckpt_path):
        state = torch.load(ssl_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=8e-5)
    scaler = torch.amp.GradScaler("cuda")

    best_val_score = -1.0
    best_val_dice = 0.0
    best_val_rec = 0.0
    best_ep = 0
    best_state = None
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        order = list(inner_subtrain)
        random.shuffle(order)
        ep_loss = 0.0

        for idx in order:
            optimizer.zero_grad()
            d = items[idx]
            x_raw = d["X"]
            y_raw = d["Y_multirater"] if use_multirater else d["Y"]
            if use_rich_aug:
                x_aug, y_aug = augment_3d_rich(x_raw, y_raw, device)
            else:
                x_aug, y_aug = augment_3d_basic(x_raw, y_raw, device)

            x_b = x_aug.unsqueeze(0)
            y_b = y_aug.unsqueeze(0)

            with torch.amp.autocast("cuda"):
                out = model(x_b)
                loss = compute_multitask_loss(out, y_b, x_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ep_loss += float(loss.detach().cpu())

        scheduler.step()

        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            dices = []
            peak_recalls = []
            for v_idx in inner_val:
                xv = items[v_idx]["X"]
                yv = items[v_idx]["Y"]
                ov = infer_single_view_3d(model, xv, device)
                pm = torch.sigmoid(ov[0])
                tm = yv[0].float()
                inter = float((pm * tm).sum())
                union = float(pm.sum() + tm.sum())
                dice = (2.0 * inter + 1.0) / (union + 1.0)
                dices.append(dice)

                core_gt = yv[1] > 0.70
                if core_gt.sum() > 0:
                    rec = float(((pm > 0.35) & core_gt).float().sum() / core_gt.float().sum())
                    peak_recalls.append(rec)

            m_dice = float(np.mean(dices)) if dices else 0.0
            m_rec = float(np.mean(peak_recalls)) if peak_recalls else 0.0
            score = m_dice + 0.5 * m_rec
            if score > best_val_score:
                best_val_score = score
                best_val_dice = m_dice
                best_val_rec = m_rec
                best_ep = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"[{tag}] Done in {time.perf_counter()-t0:.1f}s | BestInnerVal Ep={best_ep:02d} | "
        f"InnerValDice={best_val_dice:.4f} | InnerValCoreRec={100.0*best_val_rec:.2f}%",
        flush=True,
    )
    return best_state, {
        "best_epoch": best_ep,
        "inner_val_dice": round(best_val_dice, 4),
        "inner_val_core_recall": round(best_val_rec, 4),
        "inner_val_score": round(best_val_score, 4),
    }


# ==============================================================================
# 4. STAGE 1 Voxel & Trajectory Evaluation Helpers
# ==============================================================================

def evaluate_voxel_and_tubes_on_subset(
    case_indices,
    probs_by_case,
    items,
    clean_meta_list,
    clean_base_trajs,
):
    dices = []
    core_recs = []
    core_precs = []
    tubes_all_subset = []
    tubes_long_subset = []
    gts_all_subset = []
    gts_long_subset = []

    for i in case_indices:
        d = items[i]
        yv = d["Y"]
        probs = probs_by_case[i]
        pm = probs[0].float()
        tm = yv[0].float()

        inter = float((pm * tm).sum())
        union = float(pm.sum() + tm.sum())
        dice = (2.0 * inter + 1.0) / (union + 1.0)
        dices.append(dice)

        core_gt = yv[1] > 0.70
        if core_gt.sum() > 0:
            rec = float(((pm > 0.35) & core_gt).float().sum() / core_gt.float().sum())
            core_recs.append(rec)

        pred_pos = pm > 0.50
        if pred_pos.sum() > 0:
            prec = float((pred_pos & (tm > 0.25)).float().sum() / pred_pos.float().sum())
            core_precs.append(prec)

        X_np = d["X"].numpy()
        k0, i0, j0 = int(d["k0"]), int(d["i0"]), int(d["j0"])
        sp_i, sp_j, sp_k = [float(x) for x in clean_meta_list[i]["sp_ras"]]
        ijk2ras = np.asarray(clean_meta_list[i]["ijk_to_ras"], dtype=np.float64)
        ras2ijk = np.linalg.inv(ijk2ras)
        resp_approx = np.expm1(X_np[1] * 4.0)

        nn_u = {
            "p_mask": probs[0].numpy().astype(np.float32),
            "p_peak": probs[1].numpy().astype(np.float32),
            "vx": probs[2].numpy().astype(np.float32),
            "vy": probs[3].numpy().astype(np.float32),
            "p_obt": probs[4].numpy().astype(np.float32),
        }
        s0_i = clean_base_trajs[i]
        u_all_ex = extract_nn_labelmap_tubes(
            nn_u, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, s0_i, include_obturator=True
        )
        u_long_ex = extract_nn_labelmap_tubes(
            nn_u, resp_approx, k0, i0, j0, sp_i, sp_j, sp_k, ijk2ras, ras2ijk, s0_i, include_obturator=False
        )
        tubes_all_subset.append(s0_i + u_all_ex)
        tubes_long_subset.append(s0_i + u_long_ex)
        gts_all_subset.append(clean_meta_list[i]["gts_all"])
        gts_long_subset.append(clean_meta_list[i]["gts_long"])

    res_all, _ = evaluate_method_unified(tubes_all_subset, gts_all_subset, thresh=3.5, n_boot=200)
    res_long, _ = evaluate_method_unified(tubes_long_subset, gts_long_subset, thresh=3.5, n_boot=200)

    return {
        "n_cases": len(case_indices),
        "voxel_mask_dice_mean": round(float(np.mean(dices)), 4),
        "voxel_mask_dice_median": round(float(np.median(dices)), 4),
        "voxel_core_recall_pct": round(100.0 * float(np.mean(core_recs)), 2),
        "voxel_mask_precision_pct": round(100.0 * float(np.mean(core_precs)), 2),
        "stage1b_tubes_all_catheters": {
            "n_gt": res_all["n_gt"],
            "n_pred": res_all["n_pred"],
            "closest_matched": res_all["closest_matched"],
            "closest_recall_pct": res_all["closest_recall_pct"],
            "hungarian_1to1_matched": res_all["hungarian_1to1_matched"],
            "hungarian_1to1_recall_pct": res_all["hungarian_1to1_recall_pct"],
            "precision_1to1_pct": res_all["precision_1to1_pct"],
            "f1_1to1_pct": res_all["f1_1to1_pct"],
            "mean_shaft_xy_mm": res_all["metrics"]["mean_shaft_xy"]["mean"],
            "tip_3d_mean_mm": res_all["metrics"]["tip_3d"]["mean"],
        },
        "stage1b_tubes_long_catheters": {
            "n_gt": res_long["n_gt"],
            "n_pred": res_long["n_pred"],
            "closest_matched": res_long["closest_matched"],
            "closest_recall_pct": res_long["closest_recall_pct"],
            "hungarian_1to1_matched": res_long["hungarian_1to1_matched"],
            "hungarian_1to1_recall_pct": res_long["hungarian_1to1_recall_pct"],
            "precision_1to1_pct": res_long["precision_1to1_pct"],
            "f1_1to1_pct": res_long["f1_1to1_pct"],
        },
    }


def main():
    t0 = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_JSON_PATH), exist_ok=True)

    assert os.path.exists(SSL_CKPT_PATH), f"Missing SSL checkpoint: {SSL_CKPT_PATH}"
    assert os.path.exists(CLEAN_META_PATH), f"Missing clean metadata: {CLEAN_META_PATH}"

    best_gpu = 0
    for g_i in range(torch.cuda.device_count()):
        if "4090" in torch.cuda.get_device_name(g_i):
            best_gpu = g_i
            break
    torch.cuda.set_device(best_gpu)
    device = torch.device(f"cuda:{best_gpu}")
    print(f"[Nested Stage 1] Running on {device} ({torch.cuda.get_device_name(best_gpu)})...", flush=True)

    items = [
        torch.load(os.path.join(CLEAN_CACHE_DIR, f"case_{idx:02d}.pt"), weights_only=False)
        for idx in range(54)
    ]
    clean_meta_list = pickle.load(open(CLEAN_META_PATH, "rb"))
    clean_base_trajs = pickle.load(open(CLEAN_BASE_TRAJS_PATH, "rb"))

    inner_results = {}
    for f_out in range(5):
        outer_test = [i for i in range(54) if i % 5 == f_out]
        outer_train = [i for i in range(54) if i % 5 != f_out]
        for m in range(5):
            split = nested_split([item["case_id"] for item in clean_meta_list], f_out, m)
            inner_val = split["inner_val"]
            inner_subtrain = split["inner_train"]
            assert len(set(inner_subtrain) & set(inner_val)) == 0
            assert len(set(inner_subtrain) & set(outer_test)) == 0
            assert len(set(inner_val) & set(outer_test)) == 0

            ckpt_path = os.path.join(OUT_DIR, f"unet3d_outer{f_out}_inner{m}.pt")
            tag = f"Outer {f_out} Inner {m}"
            if os.path.exists(ckpt_path):
                read_checkpoint_record(ckpt_path, split, [item["case_id"] for item in clean_meta_list], 1000 + f_out * 10 + m)
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                info = {"status": "cached_from_disk"}
            else:
                state, info = train_single_inner_model(
                    items,
                    inner_subtrain,
                    inner_val,
                    device=device,
                    ssl_ckpt_path=SSL_CKPT_PATH,
                    use_multirater=True,
                    use_rich_aug=True,
                    epochs=42,
                    eval_every=6,
                    seed=1000 + f_out * 10 + m,
                    tag=tag,
                )
                torch.save(state, ckpt_path)
                write_checkpoint_record(ckpt_path, split, [item["case_id"] for item in clean_meta_list], 1000 + f_out * 10 + m)
            inner_results[f"outer{f_out}_inner{m}"] = info

            # Ensure Inner-OOF predictions with 4-view TTA exist for all `v_idx in inner_val`
            missing_inner_oof = [
                v_idx for v_idx in inner_val
                if not os.path.exists(os.path.join(OUT_DIR, f"inner_oof_f{f_out}_case_{v_idx:02d}.npz"))
            ]
            if missing_inner_oof:
                model = NeedleUNet3D(in_ch=6, base_ch=24).to(device)
                model.load_state_dict(state, strict=True)
                model.eval()
                for v_idx in missing_inner_oof:
                    d = items[v_idx]
                    probs = infer_4view_tta_probs(model, d["X"], device, use_tta=True)
                    inner_npz_path = os.path.join(OUT_DIR, f"inner_oof_f{f_out}_case_{v_idx:02d}.npz")
                    np.savez_compressed(
                        inner_npz_path,
                        p_mask=probs[0].numpy().astype(np.float16),
                        p_peak=probs[1].numpy().astype(np.float16),
                        vx=probs[2].numpy().astype(np.float16),
                        vy=probs[3].numpy().astype(np.float16),
                        p_obt=probs[4].numpy().astype(np.float16),
                        k0=int(d["k0"]),
                        i0=int(d["i0"]), i1=int(d["i1"]),
                        j0=int(d["j0"]), j1=int(d["j1"]),
                    )

    print(f"[Nested Stage 1] Verified all 25 inner models & 216 inner-OOF maps in {time.perf_counter()-t0:.1f}s!", flush=True)

    # 2. Generate Outer-OOF predictions on `outer_test` (all 54 cases) on RTX 4090 using calibrated logit-space ensembling
    probs_single_notta = {}
    probs_ens5_notta = {}
    probs_ens5_tta4 = {}

    t_oof = time.perf_counter()
    for f_out in range(5):
        outer_test = [i for i in range(54) if i % 5 == f_out]
        inner_models = []
        for m in range(5):
            ckpt_path = os.path.join(OUT_DIR, f"unet3d_outer{f_out}_inner{m}.pt")
            mod = NeedleUNet3D(in_ch=6, base_ch=24).to(device)
            mod.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=True)
            mod.eval()
            inner_models.append(mod)

        for t_idx in outer_test:
            d = items[t_idx]
            xv = d["X"]

            pairs = [infer_both_single_and_4view_tta_logits(inner_models[m], xv, device) for m in range(5)]
            logits_list_notta = [p[0] for p in pairs]
            logits_list_tta = [p[1] for p in pairs]

            probs_single_notta[t_idx] = raw_out_to_probs(logits_list_notta[0], temp_pos=1.0)

            mean_logits_notta = torch.stack(logits_list_notta, dim=0).mean(dim=0)
            probs_ens5_notta[t_idx] = raw_out_to_probs(mean_logits_notta, temp_pos=1.18)

            mean_logits_tta = torch.stack(logits_list_tta, dim=0).mean(dim=0)
            p_ens_tta = raw_out_to_probs(mean_logits_tta, temp_pos=1.20)
            probs_ens5_tta4[t_idx] = p_ens_tta

            oof_path = os.path.join(OUT_DIR, f"oof_case_{t_idx:02d}.npz")
            np.savez_compressed(
                oof_path,
                p_mask=p_ens_tta[0].numpy().astype(np.float16),
                p_peak=p_ens_tta[1].numpy().astype(np.float16),
                vx=p_ens_tta[2].numpy().astype(np.float16),
                vy=p_ens_tta[3].numpy().astype(np.float16),
                p_obt=p_ens_tta[4].numpy().astype(np.float16),
                k0=int(d["k0"]),
                i0=int(d["i0"]), i1=int(d["i1"]),
                j0=int(d["j0"]), j1=int(d["j1"]),
            )
        print(f"  Generated 5-fold inner-ensembled + 4-view TTA Outer-OOF maps for Outer Fold {f_out} ({time.perf_counter()-t_oof:.1f}s)", flush=True)

    # 3. Evaluate Old Baseline OOF vs New Clean Outer-OOF variants across all 54 cases
    all_54 = list(range(54))
    old_items = []
    old_probs = {}
    orig_meta_list = pickle.load(open(resolve_path("/tmp/cases_meta_54.pkl"), "rb"))
    orig_base_trajs = pickle.load(open(resolve_path("/tmp/base_trajs_54cases.pkl"), "rb"))
    for i in all_54:
        od = torch.load(os.path.join(OLD_CACHE_DIR, f"case_{i:02d}.pt"), weights_only=False)
        old_items.append(od)
        onpz = np.load(os.path.join(OLD_OOF_DIR, f"oof_case_{i:02d}.npz"))
        old_probs[i] = torch.stack([
            torch.from_numpy(onpz["p_mask"].astype(np.float32)),
            torch.from_numpy(onpz["p_peak"].astype(np.float32)),
            torch.from_numpy(onpz["vx"].astype(np.float32)),
            torch.from_numpy(onpz["vy"].astype(np.float32)),
            torch.from_numpy(onpz["p_obt"].astype(np.float32)),
        ], dim=0)

    eval_54_old_baseline = evaluate_voxel_and_tubes_on_subset(
        all_54, old_probs, old_items, orig_meta_list, orig_base_trajs
    )
    eval_54_single_notta = evaluate_voxel_and_tubes_on_subset(
        all_54, probs_single_notta, items, clean_meta_list, clean_base_trajs
    )
    eval_54_ens5_notta = evaluate_voxel_and_tubes_on_subset(
        all_54, probs_ens5_notta, items, clean_meta_list, clean_base_trajs
    )
    eval_54_ens5_tta4 = evaluate_voxel_and_tubes_on_subset(
        all_54, probs_ens5_tta4, items, clean_meta_list, clean_base_trajs
    )

    # 4. Run Controlled Fold-0 (11 outer-test cases) Factor-by-Factor Ablation isolating Items 1, 2, 3, and 5
    print("\n[Stage 1 Ablation] Running controlled Fold-0 ablation isolating Item 1, Item 5 (SSL), Item 3 (Multi-Rater), and Item 2 (Rich Augs + Ensemble + TTA)...", flush=True)
    f0_test = [i for i in range(54) if i % 5 == 0]
    f0_train = [i for i in range(54) if i % 5 != 0]
    f0_inner_val0 = f0_train[0::5]
    f0_inner_sub0 = [i for i in f0_train if i not in f0_inner_val0]

    # Variant 1: Strictly Nested + Ref-Free Crop ONLY (no SSL, no Multi-Rater, basic augs, single model m=0, no TTA)
    st_ab1, _ = train_single_inner_model(
        items, f0_inner_sub0, f0_inner_val0, device=device,
        ssl_ckpt_path=None, use_multirater=False, use_rich_aug=False,
        epochs=42, eval_every=6, seed=1000, tag="Ablation-1 (Item 1 Only: Nested + RefFree)"
    )
    mod_ab = NeedleUNet3D(in_ch=6, base_ch=24).to(device)
    mod_ab.load_state_dict(st_ab1, strict=True)
    mod_ab.eval()
    probs_f0_ab1 = {i: infer_4view_tta_probs(mod_ab, items[i]["X"], device, use_tta=False) for i in f0_test}

    # Variant 2: + Item 5 (SSL Pretraining on 51 External Pelvic MRI Volumes)
    st_ab2, _ = train_single_inner_model(
        items, f0_inner_sub0, f0_inner_val0, device=device,
        ssl_ckpt_path=SSL_CKPT_PATH, use_multirater=False, use_rich_aug=False,
        epochs=42, eval_every=6, seed=1000, tag="Ablation-2 (+Item 5: 51-Vol SSL Pretrain)"
    )
    mod_ab.load_state_dict(st_ab2, strict=True)
    mod_ab.eval()
    probs_f0_ab2 = {i: infer_4view_tta_probs(mod_ab, items[i]["X"], device, use_tta=False) for i in f0_test}

    # Variant 3: + Item 3 (Second Annotator LB/AMe Multi-Rater Consensus & Soft-Extra Supervision)
    st_ab3, _ = train_single_inner_model(
        items, f0_inner_sub0, f0_inner_val0, device=device,
        ssl_ckpt_path=SSL_CKPT_PATH, use_multirater=True, use_rich_aug=False,
        epochs=42, eval_every=6, seed=1000, tag="Ablation-3 (+Item 3: Multi-Rater LB/AMe)"
    )
    mod_ab.load_state_dict(st_ab3, strict=True)
    mod_ab.eval()
    probs_f0_ab3 = {i: infer_4view_tta_probs(mod_ab, items[i]["X"], device, use_tta=False) for i in f0_test}

    probs_f0_ab4 = {i: probs_single_notta[i] for i in f0_test}
    probs_f0_ab5 = {i: probs_ens5_notta[i] for i in f0_test}
    probs_f0_ab6 = {i: probs_ens5_tta4[i] for i in f0_test}

    f0_ablation_table = {
        "0_old_baseline_acsv_klim_leaky_val": evaluate_voxel_and_tubes_on_subset(
            f0_test, old_probs, old_items, orig_meta_list, orig_base_trajs
        ),
        "1_item1_nested_cv_and_ref_free_klim_only": evaluate_voxel_and_tubes_on_subset(
            f0_test, probs_f0_ab1, items, clean_meta_list, clean_base_trajs
        ),
        "2_plus_item5_ssl_pretraining_51vols": evaluate_voxel_and_tubes_on_subset(
            f0_test, probs_f0_ab2, items, clean_meta_list, clean_base_trajs
        ),
        "3_plus_item3_multirater_lb_ame_supervision": evaluate_voxel_and_tubes_on_subset(
            f0_test, probs_f0_ab3, items, clean_meta_list, clean_base_trajs
        ),
        "4_plus_item2a_rich_3d_augmentations_single_model": evaluate_voxel_and_tubes_on_subset(
            f0_test, probs_f0_ab4, items, clean_meta_list, clean_base_trajs
        ),
        "5_plus_item2b_5fold_inner_ensemble_no_tta": evaluate_voxel_and_tubes_on_subset(
            f0_test, probs_f0_ab5, items, clean_meta_list, clean_base_trajs
        ),
        "6_plus_item2c_5fold_inner_ensemble_plus_4view_tta_full": evaluate_voxel_and_tubes_on_subset(
            f0_test, probs_f0_ab6, items, clean_meta_list, clean_base_trajs
        ),
    }

    summary = {
        "step": "Step 3: Strictly Nested Stage 1 Training + Rich 3D Augmentations + Dual-Rater Labels + 5-Fold Inner Ensembling & 4-View TTA",
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "nested_cv_protocol": {
            "n_outer_folds": 5,
            "n_inner_folds_per_outer_fold": 5,
            "total_inner_models_trained": 25,
            "total_inner_oof_files_saved": 216,
            "total_outer_oof_files_saved": 54,
            "outer_test_views_per_volume": 20,
            "strictly_zero_outer_test_leakage": True,
        },
        "full_54_cases_outer_oof_progression": {
            "0_old_stage1_baseline_54cases": eval_54_old_baseline,
            "1_nested_stage1_single_inner_model_no_tta_54cases": eval_54_single_notta,
            "2_nested_stage1_5fold_inner_ensemble_no_tta_54cases": eval_54_ens5_notta,
            "3_nested_stage1_5fold_inner_ensemble_plus_4view_tta_54cases": eval_54_ens5_tta4,
        },
        "fold0_controlled_item_by_item_ablation_11cases": f0_ablation_table,
        "inner_models_val_metrics": inner_results,
    }

    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== STEP 3 SUMMARY (ALL 54 CASES OUTER-OOF PROGRESSION) ===")
    print(json.dumps(summary["full_54_cases_outer_oof_progression"], indent=2))
    print("\n=== STEP 3 FOLD-0 CONTROLLED ABLATION (ITEMS 1, 5, 3, 2) ===")
    print(json.dumps(summary["fold0_controlled_item_by_item_ablation_11cases"], indent=2))
    print(f"Saved {SUMMARY_JSON_PATH} in {time.perf_counter()-t0:.1f}s!")


if __name__ == "__main__":
    main()
