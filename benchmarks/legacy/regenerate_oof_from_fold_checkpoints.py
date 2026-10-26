#!/usr/bin/env python3
"""Regenerate Stage 1 (NeedleUNet3D) out-of-fold maps for the cases whose oof_case_XX.npz was overwritten by the
production (all-data) run of run_edelweiss_training.py (cases 0,1,2,5,6,7 -> folds 0,1,2 by i % 5).
Writes to OUT_FIXED (never overwrites nn_models/) and reports how much the overwritten maps differ."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder")
sys.path.insert(0, "/home/guillaume/github/needlefinder")
from run_edelweiss_training import infer_full_3d, CACHE_DIR, OUT_DIR
try:
    from run_edelweiss_training import NeedleUNet3D
except ImportError:
    from train_and_eval_unet3d import NeedleUNet3D
OUT_FIXED = os.path.join(os.path.dirname(OUT_DIR), "nn_models_oof_fixed")
os.makedirs(OUT_FIXED, exist_ok=True)
device = torch.device("cuda:0")
cases = [int(c) for c in (sys.argv[1:] or ["0", "1", "2", "5", "6", "7"])]
keys = ["p_mask", "p_peak", "vx", "vy", "p_obt"]
report = {}
for ci in cases:
    fold = ci % 5
    model = NeedleUNet3D(in_ch=6, base_ch=24).to(device)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"unet3d_fold{fold}.pt"), map_location=device, weights_only=True))
    model.eval()
    d = torch.load(os.path.join(CACHE_DIR, f"case_{ci:02d}.pt"), weights_only=False)
    with torch.no_grad():
        out = infer_full_3d(model, d["X"].float(), device)
    out = out.numpy()
    # same packing as run_edelweiss_training.train_one_fold (sigmoid on mask/peak/obt heads, raw vx/vy)
    pm, pp, vx, vy, po = 1/(1+np.exp(-out[0])), 1/(1+np.exp(-out[1])), out[2], out[3], 1/(1+np.exp(-out[4]))
    old = np.load(os.path.join(OUT_DIR, f"oof_case_{ci:02d}.npz"))
    def dice(a, b, t=0.5):
        A, B = a > t, b > t
        return 2 * (A & B).sum() / max(1, A.sum() + B.sum())
    report[ci] = {"fold": fold, "k0": int(d["k0"]), "dice_mask_old_vs_fold": float(dice(old["p_mask"], pm)), "dice_peak_old_vs_fold": float(dice(old["p_peak"], pp, 0.3)),
                  "old_mask_vox": int((old["p_mask"] > 0.5).sum()), "fold_mask_vox": int((pm > 0.5).sum())}
    np.savez_compressed(os.path.join(OUT_FIXED, f"oof_case_{ci:02d}.npz"), p_mask=pm.astype(np.float16), p_peak=pp.astype(np.float16),
                        vx=vx.astype(np.float16), vy=vy.astype(np.float16), p_obt=po.astype(np.float16), k0=d["k0"], i0=d["i0"], i1=d.get("i1", -1), j0=d["j0"], j1=d.get("j1", -1))
    print(ci, report[ci], flush=True)
import json; json.dump(report, open(os.path.join(OUT_FIXED, "regeneration_report.json"), "w"), indent=1)
print("[done]", OUT_FIXED)
