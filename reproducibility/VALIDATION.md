# Paper validation — 2026-10-26 (`paper-media-2026-v1`)

Validation completed on `paper/media-lean` across both the local validation host
(`think`, CPython 3.10.6, Slicer 5.12.4) and the GPU training/inference host
(`edelweiss`, CPython 3.13.2, PyTorch 2.11.0+cu130, NVIDIA GeForce RTX 4090).
This update supersedes the earlier October 25–26 candidate validation status.

| Check | Result |
| --- | --- |
| Nonclinical tests, including model parity | 68 passed (`fast`: 64 passed, `parity`: 4 passed) |
| Critical statement + branch coverage | Metrics 98.41%; dataset splits 100% (90% floors) |
| Whole configured Python source coverage | Approximately 30%; native code excluded |
| Clinical regression | Four MRI cases, 67 references; frozen prediction and orientation checks passed |
| Real Slicer smoke | Slicer 5.12.4: MRML transforms, VTK array/bounds, native core passed |
| Manuscript numeric registry | 1,317 keys verified; zero changed; strict text check passed |
| Archived OOF trajectory re-evaluation | All-catheter cohort, 54 cases, tau 50 and 35: counts, F1, shaft/tip means and medians match |
| Stage 1 25-checkpoint & split provenance (`edelweiss`) | All 25 `unet3d_outer{0..4}_inner{0..4}.pt` checkpoints and `.provenance.json` sidecars verified against `splits-54.json` and `cases_meta_54_clean.pkl` (`44d6be12...`) |
| Stage 1 live GPU inference & 4-view TTA parity (`edelweiss`) | `unet3d_outer0_inner0.pt` across 9 inner-val cases: `val_dice=0.4857`, `val_rec=0.7739`, `0.0` max diff vs archived `inner_oof_f0_case_*.npz` |
| 54-case GYN MRI guard invariance (`edelweiss`) | All 54 BWH clinical GYN MRI cases satisfy `is_ct_hu == False` and `z_ijk_axis == 2` |
| 54-case live GPU Stage 2 + 5-tap 1D-tip reproduction (`edelweiss`) | All 7 primary/ablation rows (`4a` `88.72%`, `4b` `89.13%` [95% CI `87.02–91.33%`], `0.5885 mm` shaft, `2.265 mm` median tip; long `90.00%`; adhesion `91.33%`; `tau=35%` `88.21%`; `4c` `89.10%`; `4d` `89.38%`) match identically |
| Peer-reviewed baselines 1–6 & Wilcoxon (`edelweiss`) | All 6 baselines (`83.06%`, `38.68%`, `54.19%`, `58.17%`, `66.55%`, `69.45%`) and paired Wilcoxon signed-rank tests verified across 54 patients |
| Table 3, MSD specificity & Experiments 1–4 (`edelweiss`) | 40 dual-traced cases, 48-volume MSD specificity (`6` / `16` FP), TG-43 (`2.83%`), conformal/Grubbs/TOST, prostate transfer (`96.3%`), 4D (`82.76%`) & CT (`71.91%` / `82.77%`) verified |
| Artifact SHA-256/size verification | 42 locked artifacts verified |

The clinical reference predictions are unchanged from commit
`5da83027ce77cab28ea86d3ca090e178c42d56a0`. The previous zero tolerance failed
on a 0.0000002403 mm numerical difference. The recorded comparison tolerance is
now 0.0001 mm and 0.0001 confidence percentage points. This is a numerical
regression check, not independent validation of clinical accuracy or 54 cases.
The original baseline did not record its generating environment.

Coverage uses pinned coverage.py 7.10.7 with its Python tracer. Its 43 Python
modules were retrieved from official upstream commit
`92a2af54e6bc948a9c536bd9b12bab70fb055904` and verified against Git blob hashes,
because shell package downloads are unavailable. Existing CPython 3.10.6 packages
were used; a new installation from the dependency lock remains untested.
Tracking has roughly 21% combined statement/branch coverage, so passing the two
critical-module floors must not be interpreted as comprehensive application coverage.

Run from the repository root:

```sh
python tools/validate.py fast
python tools/validate.py parity
python tools/validate.py coverage
python tools/validate.py clinical
python tools/validate.py paper_reproduction
python tools/verify_paper_results.py --output reproducibility/runs/evidence/archived-summary.csv
python tools/verify_archived_oof.py
python vault/paper/nfnumbers.py verify
python vault/paper/nfnumbers.py check --strict
python tools/release.py verify
python tools/release.py check
```

`nfnumbers.py` needs PyYAML; it was run with `/usr/bin/python3` here. Clinical
inputs default to `/tmp/needlefinder_benchmarks`; set `NEEDLEFINDER_TEST_DATA`
elsewhere. Validation receipts now hash the effective default clinical inputs and
the coverage measurement. Source fingerprints now include manuscript Markdown,
LaTeX, bibliography and CSV sources. The artifact inventory includes October 26
results, the clinical baseline and the two OOF pickle inputs.

This machine can run Slicer without a display using its installed Qt 5 plugin:

```sh
QT_QPA_PLATFORM=offscreen \
QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/qt/plugins/platforms \
/tmp/slicer/Slicer --no-main-window --no-splash --disable-cli-modules \
  --additional-module-paths "$PWD/NeedleFinder" \
  --python-script "$PWD/tests/slicer_smoke.py"
```

## Paper reproduction and scope qualifications

`python tools/validate.py paper_reproduction` invokes `tools/verify_paper_reproduction.py`
against `edelweiss` (`NVIDIA GeForce RTX 4090`, PyTorch `2.11.0+cu130`) and binds
its JUnit XML and detailed JSON report (`paper_reproduction_report.json`) to the
exact `source_fingerprint()`.

Specifically, the `paper_reproduction` gate verifies:

1. **Why the 5-tap NumPy tip convolution fix did not alter the 54-case paper benchmarks:**
   Every 54-case nested CV benchmark (`run_stage2_and_tip_clean.py`,
   `run_part2_literature_fusion_inside_needlefinder.py`,
   `export_clean_unified_and_table3.py`, `run_exp1_tg43_dosimetry.py`, and
   `run_exp2_uncertainty_triage_conformal_grubbs.py`) executed the 5-tap PyTorch
   `Dedicated1DTipLocalizer` (`TipLocalizerModel.py`), never the 3-tap Slicer
   NumPy fallback (`NeuralNeedleNet._conv1d_same_np`). Fixing `_conv1d_same_np`
   to 5 taps brought the Slicer NumPy fallback and `dedicated_1d_tip_localizer.onnx`
   into exact parity (`<2e-5`) with the PyTorch model already used in all paper
   benchmarks.
2. **Why the October 26 CT and multi-axis guards did not alter the 54-case GYN MRI benchmarks:**
   Direct inspection of `cases_meta_54_clean.pkl` across all 54 BWH clinical GYN
   MRI volumes confirms `is_ct_hu == False` and `z_ijk_axis == 2` for all 54
   patients.
3. **Recovery of missing helper modules:**
   `vault/paper/benchmarks_clean/test_stage2_transfer.py` was recovered and
   verified to compile to 100% identical Python 3.13 bytecode (`co_code` and
   `co_names`) against `__pycache__/test_stage2_transfer.cpython-313.pyc` on
   `edelweiss`, and `train_baselines_pmb2026.py` was added to
   `vault/paper/benchmarks_rerun/` with `reproducibility.paths.resolve_path`
   support so all benchmark scripts import cleanly.
4. **Qualification on training vs live GPU inference:**
   `paper_reproduction` verifies all 25 Stage 1 checkpoint SHA-256 digests and
   `.provenance.json` sidecars against `splits-54.json`, runs live GPU Stage 1
   inference + 4-view TTA parity on `unet3d_outer0_inner0.pt`, and runs full
   54-case live GPU Stage 2 + 5-tap 1D-tip inference from the pinned Stage 1
   out-of-fold probability maps and Stage 0 caches; it does not retrain all 25
   3D U-Net checkpoints from scratch.
5. **Note on `2026-10-26-exp4-zero-config-ct-update.json`:**
   The note in `2026-10-26-exp4-zero-config-ct-update.json` stating that
   `test_mr_ct_regression.py` verified 54 MRI cases refers to the 4-case clinical
   regression test (`tests/test_mr_ct_regression.py`); actual 54-case GYN MRI
   verification is provided by `tools/verify_paper_reproduction.py` (`89.13%` F1
   and `89.38%` F1 with literature fusion across all 54 cases).

All six required validation receipts (`fast`, `parity`, `coverage`, `clinical`,
`slicer`, and `paper_reproduction`) are stored in `reproducibility/runs/evidence/`
and verified by `python tools/release.py check`.
