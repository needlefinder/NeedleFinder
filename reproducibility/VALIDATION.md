# Paper Validation Summary (`paper-media-2026-v1`)

All six release gates (`fast`, `parity`, `coverage`, `clinical`, `slicer`, and
`paper_reproduction`) pass on tag `paper-media-2026-v1` across the CPU/Slicer
validation host (`CPython 3.10.6`, `3D Slicer 5.12.4`) and the CUDA GPU
reproduction host (`CPython 3.13.2`, `PyTorch 2.11.0+cu130`, `NVIDIA GeForce RTX 4090`).

| Check | Result |
| --- | --- |
| Non-clinical unit & integration tests (`fast` + `parity`) | `68 passed` (`fast`: `52 passed`, `parity`: `16 passed`, `0 skipped`) |
| PyTorch / NumPy / ONNX model parity | Stage 1 3D U-Net, Stage 2A MPNN, Stage 2B Graph Transformer, and 5-tap 1D Tip Localizer match within `< 2e-5` |
| Critical statement + branch coverage (`coverage`) | `EvaluationMetrics.py`: `98.41%`; `DatasetSplits.py`: `100.0%` ($\ge 90\%$ floors; whole-source Python coverage ~30%) |
| Clinical regression (`clinical`) | 4 clinical MRI cases, 67 reference catheters; frozen trajectory (`0.0001 mm` atol) and orientation checks passed |
| 3D Slicer smoke test (`slicer`) | Slicer 5.12.4 headless MRML transforms, VTK array/bounds, and native C++/Rust core passed |
| Archived OOF trajectory re-evaluation (`verify_archived_oof.py`) | 54 cases, $\tau=50\%$ and $\tau=35\%$: counts, Hungarian $1\text{:}1$ F1, and shaft/tip means and medians match `reproducibility/results/` |
| Stage 1 25-checkpoint & split provenance (`paper_reproduction`) | All 25 `unet3d_outer{0..4}_inner{0..4}.pt` checkpoints and `.provenance.json` sidecars verified against `splits-54.json` (`44d6be12...`) |
| Stage 1 live GPU inference & 4-view TTA parity (`paper_reproduction`) | `unet3d_outer0_inner0.pt` across 9 inner-validation cases: `val_dice=0.4857`, `val_rec=0.7739`, `0.0` max diff vs. cached `inner_oof_f0_case_*.npz` |
| 54-case live GPU Stage 2 + 5-tap 1D-tip reproduction (`paper_reproduction`) | All 7 primary/ablation rows (`4a` `88.72%`, `4b` `89.13%` [95% CI `87.02–91.33%`], `0.5885 mm` shaft, `2.265 mm` median tip; long `90.00%`; adhesion `91.33%`; `tau=35%` `88.21%`; `4c` `89.10%`; `4d` `89.38%`) match identically |
| Peer-reviewed baselines 1–6 & Wilcoxon (`paper_reproduction`) | All 6 baselines (`83.06%`, `38.68%`, `54.19%`, `58.17%`, `66.55%`, `69.45%`) and paired Wilcoxon signed-rank tests verified across 54 patients |
| Table 3, MSD specificity & Experiments 1–4 (`paper_reproduction`) | 40 dual-traced cases, 48-volume MSD specificity (`6` / `16` FP), TG-43 (`2.83%`), conformal/Grubbs/TOST, prostate transfer (`96.3%`), 4D (`82.76%`) & CT (`71.91%` / `82.77%`) verified |
| Artifact SHA-256 & byte-size lock (`tools/release.py verify`) | All `30` locked production models, clinical baselines, and archived result files verified |

## Scope of Automated Validation Gates

- **Archive & CPU verification (`verify_paper_results.py`, `verify_archived_oof.py`, `fast`, `parity`, `coverage`):**
  Runs without private clinical images in ~15 seconds on CPU. Recomputes Hungarian $1\text{:}1$ matching and shaft/tip error summaries directly from the archived out-of-fold trajectory predictions and per-needle distributions in `reproducibility/results/`.
- **Live GPU reproduction (`tools/validate.py paper_reproduction`):**
  Verifies all 25 nested $5 \times 5$ Stage 1 checkpoint SHA-256 digests and `.provenance.json` sidecars against `reproducibility/splits-54.json`, runs live GPU Stage 1 3D U-Net inference + 4-view TTA parity on `unet3d_outer0_inner0.pt`, and executes full 54-case live GPU Stage 2 + 5-tap 1D-tip inference from the pinned Stage 1 out-of-fold probability maps and Stage 0 caches (~90 seconds on RTX 4090). Retraining all 25 3D U-Net checkpoints from raw imaging volumes is performed via `benchmarks/clean/train_nested_stage1.py`.
- **4-case clinical regression (`tools/validate.py clinical`):**
  Verifies numerical trajectory regression (`0.0001 mm` tolerance) across 4 clinical MRI test cases (`67` reference catheters); full 54-case clinical cohort evaluation is covered by `paper_reproduction`.

## Commands

Run from the repository root:

```sh
python tools/release.py verify
python tools/validate.py fast
python tools/validate.py parity
python tools/validate.py coverage
python tools/validate.py clinical
python tools/validate.py paper_reproduction
python tools/verify_paper_results.py --output reproducibility/runs/evidence/archived-summary.csv
python tools/verify_archived_oof.py
python tools/release.py check
```

Headless 3D Slicer smoke test:

```sh
QT_QPA_PLATFORM=offscreen \
QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/qt/plugins/platforms \
/path/to/Slicer --no-main-window --no-splash --disable-cli-modules \
  --additional-module-paths "$PWD/NeedleFinder" \
  --python-script "$PWD/tests/slicer_smoke.py"
```
