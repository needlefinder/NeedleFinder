# NeedleFinder 2026 Paper Benchmarks

This directory contains all training, cross-validation, baseline reproduction,
specificity, and clinical translation scripts for the 2026 NeedleFinder manuscript
(tag `paper-media-2026-v1`).

## Directory Structure

- **`benchmarks/clean/`** — Primary strictly nested $5 \times 5$ cross-validation
  pipeline (`54` patients, `1,063` catheters), peer-reviewed baseline reproductions,
  literature fusion, specificity evaluation, timing/calibration audits, and
  clinical translation Experiments 1–4:
  1. `prepare_clean_dataset.py` — Build multi-rater consensus supervision (`40/40`
     dual-traced scenes), automatic reference-free `k_lim_auto` slab bounds, and
     54-case metadata (`step1_clean_dataset_summary.json`).
  2. `pretrain_ssl_unet3d.py` — Self-supervised 3D U-Net pre-training on 51 external
     pelvic MRI volumes (`step2_ssl_pretrain_summary.json`).
  3. `train_nested_stage1.py` — Strictly nested $5 \times 5$ Stage 1 3D U-Net
     training (`25` inner models, frozen patient splits in
     `reproducibility/splits-54.json`, checkpoint provenance sidecars via
     `reproducibility/checkpoints.py`, `step3_nested_stage1_summary.json`).
  4. `run_stage2_and_tip_clean.py` — Stage 2A `NeedleSliceMPNN`, Stage 2B
     `TipAwareBundleGraphTransformer` (`SynthBrachy3T` pre-training), and
     five-tap `Dedicated1DTipLocalizer` training and evaluation.
  5. `test_stage2_transfer.py` — Stage 2 transfer utilities, weighted degree-3
     polynomial fitting, and 5-tap 1D tip inference helper (`run_stage2_inference`).
  6. `run_part1_peer_baselines_and_unc_maps.py` — Reproduction of peer-reviewed
     MRI brachytherapy baselines (`Dai et al. 2020`, `Kostoulas et al. 2025`,
     `Aleong et al. 2024 / Kim et al. 2026`, `part1_peer_baselines_summary.json`).
  7. `run_part2_literature_fusion_inside_needlefinder.py` — Integration of
     literature-inspired upgrades inside `NeedleFinder 2026` (`4d`).
  8. `export_clean_unified_and_table3.py` — Unified 54-case evaluation, paired
     Wilcoxon signed-rank tests, and 40-patient dual-observer (`Table 3`) export.
  9. `benchmark_public_pelvic_mri_clean.py` — Zero-needle specificity evaluation
     across 48 public `MSD Task05_Prostate` 3T pelvic MRI volumes.
  10. `benchmark_54cases_timing_clean.py` — End-to-end per-stage CPU/GPU timing.
  11. `calibration_and_failure_clean.py` — Stage 2B confidence calibration,
      reason-code attribution, and failure-mode taxonomy.
  12. `export_slicer_production_models.py` — Export production ONNX/NPZ weights
      into `NeedleFinder/NeedleFinderLib/`.
  13. `run_exp1_tg43_dosimetry.py` — Experiment 1: AAPM TG-43U1 $^{192}\text{Ir}$
      HDR dosimetric sensitivity (`Protocol A` and `Protocol B`) and uncertainty-guided
      clinician review (`k=2`).
  14. `run_exp2_uncertainty_triage_conformal_grubbs.py` — Experiment 2: Longitudinal
      vs. lateral tip error decomposition (`98.76%` along-shaft), nested split-conformal
      tip intervals, review-budget curves, and Grubbs' three-cornered-hat / TOST
      non-inferiority analysis.
  15. `run_exp3_prostate_and_cross_sequence.py` — Experiment 3: Zero-shot transfer
      to 3T prostate MRI (`Case 100 Prostate`) and 297 multi-sequence MRI scans.
  16. `run_exp4_4d_insertion_and_ct.py` — Experiment 4: 4D causal Kalman serial
      insertion tracking (`Cases 68–71`), tissue-push/deflection biomechanics, and
      zero-configuration paired pelvic CT evaluation (`38` paired patients).

- **`benchmarks/rerun/`** — Shared evaluation, graph construction, and baseline
  training modules imported by `benchmarks/clean/` (`evaluate_unified_apples_to_apples.py`,
  `train_baselines_pmb2026.py`, `train_and_eval_unet3d.py`, `train_graphnet3d.py`,
  `train_export_graphnet_prod.py`, `prepare_and_inspect_dataset.py`, etc.).

- **`benchmarks/legacy/`** — Earlier 5-fold cross-validation scripts preserved for
  reproducibility.

## Archived Results & Automated Verification

All machine-readable JSON/NPZ/PKL benchmark outputs are archived in
`reproducibility/results/` and locked by SHA-256 in
`reproducibility/artifacts.lock.json`.

See [`reproducibility/README.md`](../reproducibility/README.md) and
[`reproducibility/VALIDATION.md`](../reproducibility/VALIDATION.md) for full
verification instructions (`tools/validate.py` and `tools/release.py`).
