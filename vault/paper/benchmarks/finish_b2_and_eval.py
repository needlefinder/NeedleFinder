#!/usr/bin/env python3
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import pickle, random, time, torch
import numpy as np
from train_baselines_pmb2026 import (
    META_PATH, OUT_DIR, load_all_54_items, NativeResUNet3D_MaskOnly,
    augment_1ch, compute_cldice_focal_dice_loss, extract_native_cldice_skeleton_trajectories
)
from run_baseline2_gpu1 import infer_low_vram_1ch

def main():
    device = torch.device("cuda:0")
    print(f"=== Accelerating Remaining Folds of Baseline 2 on {device} ({torch.cuda.get_device_name(device)}) ===", flush=True)
    meta_list = pickle.load(open(META_PATH, "rb"))
    _, items_nat = load_all_54_items(meta_list)
    epochs = 28

    # Run in reverse fold order (4, 3, 2) so we meet GPU 1 (which runs 0, 1, 2) in the middle!
    for fold in [4, 3, 2, 1, 0]:
        ckpt_path = f"{OUT_DIR}/native_cldice_fold{fold}.pt"
        if os.path.exists(ckpt_path):
            print(f"  [Skip] Fold {fold} already completed by GPU 1.")
            continue
        t0 = time.perf_counter()
        val_idx = [i for i in range(54) if i % 5 == fold]
        train_idx = [i for i in range(54) if i % 5 != fold]

        model = NativeResUNet3D_MaskOnly(in_ch=1, base_ch=24).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
        best_dice = -1.0
        best_state = None

        for ep in range(1, epochs + 1):
            if os.path.exists(ckpt_path):
                print(f"  [Info] Fold {fold} just finished on GPU 1, stopping duplicate work on GPU 0.")
                break
            model.train()
            order = train_idx.copy()
            random.shuffle(order)
            for idx in order:
                x, y, _ = items_nat[idx]
                xa, ya = augment_1ch(x, y, max_d=24, max_hw=144)
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
                    pm = infer_low_vram_1ch(model, xv, device, chunk_d=16, max_hw=160)
                    tm = yv[0].numpy().astype(np.float32)
                    inter = float((pm * tm).sum())
                    union = float(pm.sum() + tm.sum())
                    f_dices.append((2.0 * inter + 1.0) / (union + 1.0))
                md = float(np.mean(f_dices))
                if md > best_dice:
                    best_dice = md
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"  [Native-clDice GPU0 | Fold {fold}] Ep {ep:02d}/{epochs} | ValDice={md:.3f} ({time.perf_counter()-t0:.1f}s)", flush=True)

        if not os.path.exists(ckpt_path) and best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, ckpt_path)
            model.eval()
            for vi in val_idx:
                xv, _, _ = items_nat[vi]
                pm = infer_low_vram_1ch(model, xv, device, chunk_d=16, max_hw=160)
                np.savez_compressed(f"{OUT_DIR}/oof_native_cldice_case_{vi:02d}.npz", p_mask=pm.astype(np.float16))
        torch.cuda.empty_cache()

    # Wait briefly if any case npz is still being written by GPU 1
    for _ in range(60):
        if all(os.path.exists(f"{OUT_DIR}/oof_native_cldice_case_{i:02d}.npz") for i in range(54)):
            break
        time.sleep(2)

    print("=== Extracting OOF Skeleton Trajectories for all 54 cases (Baseline 2) ===", flush=True)
    oof_trajs_all = [None] * 54
    oof_trajs_long = [None] * 54
    oof_dices = []
    for vi in range(54):
        _, yv, meta_d = items_nat[vi]
        pm = np.load(f"{OUT_DIR}/oof_native_cldice_case_{vi:02d}.npz")["p_mask"].astype(np.float32)
        tm = yv[0].numpy().astype(np.float32)
        oof_dices.append(float((2.0 * (pm * tm).sum() + 1.0) / (pm.sum() + tm.sum() + 1.0)))
        trajs = extract_native_cldice_skeleton_trajectories(pm, meta_d, prob_thresh=0.40)
        oof_trajs_all[vi] = trajs
        oof_trajs_long[vi] = [t for t in trajs if t["is_long"]]

    with open(f"{OUT_DIR}/oof_trajs_native_cldice.pkl", "wb") as f:
        pickle.dump({"all": oof_trajs_all, "long": oof_trajs_long, "mean_oof_dice": float(np.mean(oof_dices))}, f)
    print(f"[Done] Saved oof_trajs_native_cldice.pkl (Mean OOF Dice = {np.mean(oof_dices):.4f})", flush=True)

if __name__ == "__main__":
    main()
