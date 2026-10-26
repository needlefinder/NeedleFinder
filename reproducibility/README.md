# Reproducing and Verifying NeedleFinder (`paper-media-2026-v1`)

This directory contains pinned validation environments, frozen 54-patient cross-validation
splits (`splits-54.json`), artifact SHA-256 locks (`artifacts.lock.json`), and
archived machine-readable benchmark results (`results/`) for the 2026 NeedleFinder
manuscript.

## Reproducibility Tiers

Reproduction and verification are organized into three explicit tiers according to
hardware and clinical data requirements:

| Tier | Command(s) | Requirements | What It Verifies |
| --- | --- | --- | --- |
| **1. Fast CPU & Archive Verification** (~15 s) | `tools/release.py verify`<br>`tools/validate.py fast`<br>`tools/validate.py parity`<br>`tools/validate.py coverage`<br>`tools/verify_paper_results.py`<br>`tools/verify_archived_oof.py` | Python 3.10+, CPU PyTorch (`requirements-parity.lock`), C++17 or Rust toolchain; **no private clinical images required** | All 30 locked artifact SHA-256 digests; geometry, spline, Hungarian $1\text{:}1$ matching, and split-isolation unit tests; synthetic 3D volume + native C++/Rust tracking; PyTorch/NumPy/ONNX parity (`< 2e-5`); and re-evaluation of the archived 54-case out-of-fold (OOF) trajectories in `reproducibility/results/`. |
| **2. Live GPU Checkpoint & Inference Reproduction** (~90 s) | `tools/validate.py clinical`<br>`tests/slicer_smoke.py`<br>`tools/validate.py paper_reproduction`<br>`tools/release.py check` | 4-case clinical regression set (`NEEDLEFINDER_TEST_DATA`), 3D Slicer 5.12+, and CUDA GPU with the 25 pinned Stage 1 checkpoints (`NEEDLEFINDER_WORK_DIR`) | 4-case clinical MRI regression (`67` reference trajectories); headless 3D Slicer MRML/VTK smoke test; SHA-256 & split-provenance verification of all 25 nested $5 \times 5$ Stage 1 3D U-Net checkpoints against `splits-54.json`; live GPU Stage 1 inference + 4-view TTA parity; and full 54-case live GPU Stage 2 + 5-tap 1D-tip inference, peer-reviewed baselines 1–6, Wilcoxon tests, Table 3, MSD specificity, and Experiments 1–4. |
| **3. End-to-End Training from Scratch** (multi-GPU hours) | `benchmarks/clean/prepare_clean_dataset.py` $\rightarrow$ `benchmarks/clean/run_exp4_4d_insertion_and_ct.py` | Full clinical MRI/CT cohort (`NEEDLEFINDER_DATA_DIR`) and CUDA GPU workstation | Full dataset preprocessing, 51-volume self-supervised 3D U-Net pre-training, strictly nested $5 \times 5$ Stage 1 3D U-Net training (25 models), Stage 2A/2B + 1D tip localizer training, baseline training, and all clinical translation experiments. |

## Quick Start (Tier 1 — CPU & Archive Verification)

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r reproducibility/requirements-parity.lock
.venv/bin/python tools/release.py verify
.venv/bin/python tools/validate.py fast
.venv/bin/python tools/validate.py parity
.venv/bin/python tools/validate.py coverage
.venv/bin/python tools/verify_paper_results.py \
  --output reproducibility/runs/archived-results.csv
.venv/bin/python tools/verify_archived_oof.py
```

- `pytest.ini` restricts test discovery to `tests/`.
- `coverage` enforces $\ge 90\%$ statement + branch coverage floors on `EvaluationMetrics.py` (`98.41%`) and `DatasetSplits.py` (`100%`).

## Data Paths & Frozen Splits

`reproducibility/paths.py` maps historical workstation paths via environment variables:

- `NEEDLEFINDER_WORK_DIR`: working directory for caches and model checkpoints (default: `reproducibility/runs/work`).
- `NEEDLEFINDER_DATA_DIR`: root directory containing the clinical imaging datasets (`GYN Cases`, etc.).
- `NEEDLEFINDER_TEST_DATA`: directory containing the four clinical regression cases (`Case 12`, `Case 29`, `Case 44`, `Case 66`).
- `NEEDLEFINDER_CLINICAL_BASELINE`: frozen trajectory regression baseline (default: `NeedleFinder/Testing/clinical_mri_frozen_baseline.json`).

`reproducibility/splits-54.json` freezes the 54-patient ordering, all 25 nested $5 \times 5$ inner training/validation splits, per-fold random seeds (`1000 + 10 * outer_fold + inner_fold`), and the SHA-256 digest of the 54-case metadata (`44d6be12b0f8b3e7e85327b70108a83106ff899e69f555a5516377ba905252a4`). Stage 1 training and verification (`reproducibility/checkpoints.py`) bind every checkpoint (`unet3d_outer{0..4}_inner{0..4}.pt`) to a `.provenance.json` sidecar that fails closed if the patient ordering, split indices, or seed differ.

## Clinical, Slicer, and GPU Reproduction (Tier 2)

```bash
NEEDLEFINDER_TEST_DATA=/path/to/cases \
NEEDLEFINDER_CLINICAL_BASELINE=NeedleFinder/Testing/clinical_mri_frozen_baseline.json \
  .venv/bin/python tools/validate.py clinical

/path/to/Slicer --no-main-window --no-splash --disable-cli-modules \
  --additional-module-paths "$PWD/NeedleFinder" \
  --python-script "$PWD/tests/slicer_smoke.py"

.venv/bin/python tools/validate.py paper_reproduction
.venv/bin/python tools/release.py check
```

## Full Paper Training Pipeline (Tier 3)

To retrain the entire pipeline from scratch on a GPU workstation with `NEEDLEFINDER_DATA_DIR` and `NEEDLEFINDER_WORK_DIR` configured, run the scripts in [`benchmarks/clean/`](../benchmarks/README.md) in order:

1. `benchmarks/clean/prepare_clean_dataset.py`
2. `benchmarks/clean/pretrain_ssl_unet3d.py`
3. `benchmarks/clean/train_nested_stage1.py`
4. `benchmarks/clean/run_stage2_and_tip_clean.py`
5. `benchmarks/clean/run_part1_peer_baselines_and_unc_maps.py`
6. `benchmarks/clean/run_part2_literature_fusion_inside_needlefinder.py`
7. `benchmarks/clean/export_clean_unified_and_table3.py`
8. `benchmarks/clean/benchmark_public_pelvic_mri_clean.py`
9. `benchmarks/clean/run_exp1_tg43_dosimetry.py` through `benchmarks/clean/run_exp4_4d_insertion_and_ct.py`

To re-export the production 1D tip localizer ONNX weights without retraining:

```bash
.venv/bin/python tools/export_tip_model.py \
  --output NeedleFinder/NeedleFinderLib/dedicated_1d_tip_localizer.onnx
```

## Release Manifest & Companion Repository

Every validation suite writes a receipt in `reproducibility/runs/evidence/` bound to the exact source fingerprint (`tools/release.py`).

```bash
.venv/bin/python tools/release.py check
.venv/bin/python tools/release.py manifest --output dist/paper-media-2026-v1-manifest.json
.venv/bin/python tools/create_paper_companion.py --output dist/paper-companion
```
