#!/usr/bin/env python3
"""
Parallel 5-Fold Cross-Validation + Full Training of NeedleUNet3D on edelweiss
(using both GPU 0: RTX 4070 Ti SUPER and GPU 1: RTX 4090).

1. Builds & caches 3D tensors (X: 6ch, Y: 5ch) for all 54 cases + full-FOV case.nrrd.
2. Trains 5 cross-validation folds in parallel across GPU 0 and GPU 1.
3. Saves out-of-fold 3D probability maps & tangent fields for all 54 cases.
4. Trains the production NeedleUNet3D model on all cases and exports `needle_unet3d.onnx`.
"""

import math
import os
import pickle
import random
import sys
import time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/guillaume/github/needlefinder")
from prepare_and_inspect_dataset import load_all_dataset_cases, read_nrrd_local, read_vtk_centerline_local
from train_and_eval_unet3d import NeedleUNet3D, build_case_tensors, compute_multitask_loss

CACHE_DIR = "/tmp/needle_unet3d_cache"
OUT_DIR = "/home/guillaume/github/needlefinder/nn_models"


def prepare_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    done_flag = os.path.join(CACHE_DIR, "done.flag")
    if os.path.exists(done_flag):
        print("[Cache] Using existing cached tensors in", CACHE_DIR)
        return

    print("[Cache] Building 3D input/target tensors for 54 cases...")
    t0 = time.perf_counter()
    cases = load_all_dataset_cases()
    manifest = []
    for idx, c in enumerate(cases):
        X, Y, meta = build_case_tensors(c)
        fn = os.path.join(CACHE_DIR, f"case_{idx:02d}.pt")
        torch.save({
            "X": torch.from_numpy(X),
            "Y": torch.from_numpy(Y),
            "case_num": c["case_num"],
            "case_id": c["case_id"],
            "k0": meta["k0"],
            "i0": meta["i0"], "i1": meta["i1"],
            "j0": meta["j0"], "j1": meta["j1"],
            "sp_ras": c["sp_ras"],
        }, fn)
        manifest.append({"idx": idx, "case_num": c["case_num"], "case_id": c["case_id"], "file": fn})
        if (idx + 1) % 10 == 0 or idx == len(cases) - 1:
            print(f"  Cached {idx+1}/{len(cases)} cases ({time.perf_counter()-t0:.1f}s)")

    with open(os.path.join(CACHE_DIR, "manifest.pkl"), "wb") as f:
        pickle.dump(manifest, f)
    with open(done_flag, "w") as f:
        f.write("ok\n")


def augment_3d_sample(x, y):
    """
    3D Data Augmentation on GPU tensor x [6, K, H, W], y [5, K, H, W]:
    - Random X-flip (negates vx = y[2])
    - Random in-plane shift
    - Random MRI intensity scaling & contrast jitter on ch 0..3
    - Random 1-slice dropout simulation on input to train gap-inpainting
    """
    x = x.clone()
    y = y.clone()
    _, K, H, W = x.shape

    # 1. Random horizontal flip (left-right in I, which corresponds to W dimension)
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
        y = torch.flip(y, dims=[3])
        y[2] = -y[2]  # flip dx/dz

    # 2. Random small vertical flip (A-P) with 20% prob
    if random.random() < 0.2:
        x = torch.flip(x, dims=[2])
        y = torch.flip(y, dims=[2])
        y[3] = -y[3]  # flip dy/dz

    # 3. Random spatial roll in H, W (+/- 10 pixels)
    if random.random() < 0.7:
        dy = random.randint(-10, 10)
        dx = random.randint(-10, 10)
        x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        y = torch.roll(y, shifts=(dy, dx), dims=(2, 3))

    # 4. Random intensity/contrast scaling
    if random.random() < 0.7:
        scale = random.uniform(0.82, 1.20)
        bias = random.uniform(-0.08, 0.08)
        x[0] = (x[0] * scale + bias).clamp(-0.5, 2.5)
        scale_r = random.uniform(0.85, 1.18)
        x[1] = (x[1] * scale_r).clamp(0.0, 1.5)
        x[2] = (x[2] * scale_r).clamp(0.0, 1.5)

    # 5. Simulated single-slice signal dropout (teaches 3D UNet to bridge obturator/artifact gaps)
    if K > 10 and random.random() < 0.5:
        k_drop = random.randint(2, K - 3)
        x[1:4, k_drop] *= 0.15

    # Crop or pad Z if K > 36 and H, W > 192 during training to keep VRAM < 1.5 GB
    if K > 36:
        k_start = random.randint(0, K - 36)
        x = x[:, k_start:k_start + 36]
        y = y[:, k_start:k_start + 36]
    if H > 192:
        h_start = random.randint(0, H - 192)
        x = x[:, :, h_start:h_start + 192, :]
        y = y[:, :, h_start:h_start + 192, :]
    if W > 192:
        w_start = random.randint(0, W - 192)
        x = x[:, :, :, w_start:w_start + 192]
        y = y[:, :, :, w_start:w_start + 192]

    return x, y


def infer_full_3d(model, x_cpu, device):
    """Sliding Z-chunk inference for arbitrary 3D volume size with minimal VRAM."""
    _, K, H, W = x_cpu.shape
    if K <= 44:
        with torch.no_grad(), torch.amp.autocast("cuda"):
            return model(x_cpu.unsqueeze(0).to(device))[0].float().cpu()
    out_cpu = torch.zeros((5, K, H, W), dtype=torch.float32)
    w_cpu = torch.zeros((1, K, 1, 1), dtype=torch.float32)
    chunk = 40
    stride = 28
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


def train_one_fold(fold_idx, train_indices, val_indices, gpu_id, epochs=40):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    t0 = time.perf_counter()

    # Keep cached tensors in 251 GiB CPU RAM (move only single batch to GPU)
    items = []
    for idx in range(54):
        d = torch.load(os.path.join(CACHE_DIR, f"case_{idx:02d}.pt"), weights_only=False)
        items.append((d["X"], d["Y"], d))

    model = NeedleUNet3D(in_ch=6, base_ch=24).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    best_val_dice = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        order = list(train_indices)
        random.shuffle(order)
        ep_loss = 0.0

        optimizer.zero_grad()
        for step_i, idx in enumerate(order):
            x_raw, y_raw, _ = items[idx]
            x_aug, y_aug = augment_3d_sample(x_raw, y_raw)
            x_b = x_aug.unsqueeze(0).to(device, non_blocking=True)
            y_b = y_aug.unsqueeze(0).to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                out = model(x_b)
                loss = compute_multitask_loss(out, y_b, x_b)

            scaler.scale(loss).backward()
            if (step_i + 1) % 2 == 0 or (step_i + 1) == len(order):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            ep_loss += float(loss.detach().cpu())

        scheduler.step()

        if ep % 8 == 0 or ep == epochs:
            model.eval()
            dices = []
            peak_Recalls = []
            for v_idx in val_indices:
                xv, yv, _ = items[v_idx]
                ov = infer_full_3d(model, xv, device)
                pm = torch.sigmoid(ov[0])
                tm = yv[0].float()
                inter = float((pm * tm).sum())
                union = float(pm.sum() + tm.sum())
                dice = (2.0 * inter + 1.0) / (union + 1.0)
                dices.append(dice)

                core_gt = yv[1] > 0.70
                if core_gt.sum() > 0:
                    rec = float(((pm > 0.35) & core_gt).float().sum() / core_gt.float().sum())
                    peak_Recalls.append(rec)

            m_dice = float(np.mean(dices)) if dices else 0.0
            m_rec = float(np.mean(peak_Recalls)) if peak_Recalls else 0.0
            if m_dice + 0.5 * m_rec > best_val_dice:
                best_val_dice = m_dice + 0.5 * m_rec
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[GPU {gpu_id} | Fold {fold_idx}] Ep {ep:02d}/{epochs} | "
                f"Loss={ep_loss/len(order):.3f} | ValDice={m_dice:.3f} | ValCoreRecall={m_rec*100:.1f}% "
                f"({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT_DIR, f"unet3d_fold{fold_idx}.pt"))

    # Generate out-of-fold predictions for val_indices
    model.eval()
    for v_idx in val_indices:
        xv, _, meta_d = items[v_idx]
        ov = infer_full_3d(model, xv, device)
        p_mask = torch.sigmoid(ov[0]).numpy().astype(np.float16)
        p_peak = torch.sigmoid(ov[1]).numpy().astype(np.float16)
        vx = (torch.tanh(ov[2]) * 0.75).numpy().astype(np.float16)
        vy = (torch.tanh(ov[3]) * 0.75).numpy().astype(np.float16)
        p_obt = torch.sigmoid(ov[4]).numpy().astype(np.float16)
        np.savez_compressed(
            os.path.join(OUT_DIR, f"oof_case_{v_idx:02d}.npz"),
            p_mask=p_mask,
            p_peak=p_peak,
            vx=vx,
            vy=vy,
            p_obt=p_obt,
            k0=meta_d["k0"],
            i0=meta_d["i0"], i1=meta_d["i1"],
            j0=meta_d["j0"], j1=meta_d["j1"],
        )
    print(f"[GPU {gpu_id} | Fold {fold_idx}] Finished and saved OOF predictions in {time.perf_counter()-t0:.1f}s!", flush=True)


def worker_gpu(gpu_id, fold_tasks):
    for fold_idx, train_idx, val_idx, epochs in fold_tasks:
        train_one_fold(fold_idx, train_idx, val_idx, gpu_id, epochs=epochs)


def export_production_onnx():
    print("[ONNX] Training production NeedleUNet3D on all 54 cases on GPU 0 (RTX 4090)...")
    all_idx = list(range(54))
    train_one_fold(fold_idx=99, train_indices=all_idx, val_indices=all_idx[:10], gpu_id=0, epochs=40)

    device = torch.device("cuda:0")
    model = NeedleUNet3D(in_ch=6, base_ch=24).to(device)
    state = torch.load(os.path.join(OUT_DIR, "unet3d_fold99.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().cpu()

    dummy = torch.randn(1, 6, 32, 160, 160)
    onnx_path = os.path.join(OUT_DIR, "needle_unet3d.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch", 2: "depth", 3: "height", 4: "width"},
        },
        opset_version=17,
    )
    print(f"[ONNX] Successfully exported dynamic 3D ONNX model to {onnx_path} ({os.path.getsize(onnx_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    prepare_cache()

    # Create 5 stratified folds across the 54 cases
    # Note: In PyTorch CUDA_DEVICE_ORDER=FASTEST_FIRST, cuda:0 is RTX 4090 (24GB) and cuda:1 is RTX 4070 Ti SUPER (16GB)
    indices = list(range(54))
    folds = [indices[i::5] for i in range(5)]
    tasks_gpu0 = []  # cuda:0 RTX 4090 (Folds 0, 1, 2)
    tasks_gpu1 = []  # cuda:1 RTX 4070 Ti SUPER (Folds 3, 4)

    for f_i in range(5):
        if os.path.exists(os.path.join(OUT_DIR, f"unet3d_fold{f_i}.pt")):
            print(f"[Skip] Fold {f_i} already trained.")
            continue
        val_idx = folds[f_i]
        train_idx = [x for x in indices if x not in val_idx]
        tasks_gpu0.append((f_i, train_idx, val_idx, 36))

    if tasks_gpu0:
        worker_gpu(0, tasks_gpu0)

    if not os.path.exists(os.path.join(OUT_DIR, "needle_unet3d.onnx")):
        export_production_onnx()
