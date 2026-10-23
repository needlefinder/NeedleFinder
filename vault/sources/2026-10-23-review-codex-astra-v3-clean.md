# NeedleFinder v3 clean benchmark verification audit

Audit date: 2026-10-23. Scope: the eight Python programs under `vault/paper/benchmarks_clean/`, their three step summaries, the requested clean evaluation JSON artifacts, and `vault/paper/needlefinder_2026_pmb.tex`, including the five tables actually included by that manuscript. Paths below are relative to `/home/guillaume/github/needlefinder/` unless absolute.

**Verdict: the implementation is materially improved, but the five-point plan is not fully implemented and the benchmark cannot presently be certified as strictly nested, reference-free, or leakage-free end to end.** Several failures are directly demonstrable from executable code, rather than merely missing documentation. The manuscript has not been synchronized with the new artifacts. The numerical integrity of the exported matched-error arrays is substantially better than the methodological integrity of the full pipeline.

This is a source/artifact audit, not a fresh training or clinical validation run. I parsed all eight Python files, inspected the relevant training/inference paths, independently recalculated JSON statistics, enumerated the actual model/map files, and inspected `/tmp/cases_meta_54_clean.pkl`. I did not retrain the networks, rerun MRI inference, verify raw patient identities across institutions, or independently reproduce the trajectory matching from all saved predictions. Existing checkpoint files establish availability, not their complete training provenance. No manuscript, benchmark source, or evaluation artifact was changed for this audit.

## 1. Plan verification matrix

| Plan requirement | Verdict | Evidence and qualification |
|---|---|---|
| Stage 1 outer 5 × inner 5 splits | Implemented | 25 checkpoint files, 216 expected inner-OOF maps, 54 outer-OOF maps exist; index partitions are disjoint. |
| Whole pipeline strictly nested / zero leakage | Fails | Stage 2A and the tip localizer use globally generated outer-OOF features; Stage 2B's optional inner refresh retains contaminated candidate geometry and MPNN outputs. |
| Inner-OOF prediction generation | Implemented with qualification | Held-out inner cases do not enter gradient training, but their labels select the checkpoint used to generate their maps. |
| Stage 2 GBDTs and tip models consume inner-OOF maps | Not as specified | Stage 2B graph transformers optionally consume refreshed features; no GBDT implementation found; tip training does not consume these refreshed bundles. |
| Intensity-only, tilt-safe crop | Fails | Active `k_lim_auto` uses a scene marker or legacy `k_lim_gt`; the intensity-only estimate is a separate unused alternative for this path. |
| Few-shot supports restricted to outer training cases | Implemented locally | `outer_train[:k]` never selects an outer-test case. Full-pipeline few-shot / leakage-free interpretation remains unsupported. |
| Frozen specificity on 51 independent needle-free scans | Fails as stated | Artifact has 48 scans. Clean SSL models were pretrained on those same 48 scans; full production evaluation is a different pipeline. |
| Unified arrays / Table 3 consistency | Substantial pass, bounded | All 18 exported method/cohort distribution entries checked agree; Table 3 is byte-content-equivalent as a parsed object to the embedded copy. Not all methods have exported raw arrays. |
| 54-case measured runtime | Partial; GPU claim fails | 54 CPU records exist. GPU totals contain synthetic-input timings and scaled CPU estimates. Neither timing path measures the complete clean ensemble. |
| Five inner models plus four-view TTA | Implemented | 20 model/view contributions per outer-test volume; weighted in-plane reflections of 3D volumes. |
| Rich MRI augmentation including Z-blurring | Partial | Bias field, flips, roll shifts, gamma/noise, and feature-slice attenuation exist; partial-volume Z-blurring does not. |
| 40-case multi-rater supervision | Implemented with nesting caveat | 40 metadata cases have LB/AMe traces; training targets include consensus and extras. Primary references are CY + HT, not a verified annotator named AS. |
| 91 × 22 tip localizer, 0.5 mm spacing | Implemented | Dilated residual Conv1D plus self-attention; not the requested BiGRU / GBDT / cross-fit ensemble architecture. |
| Collinear split-fragment stitching | Implemented | Explicit gap, collinearity, support and duplicate checks. |
| 51-volume external pretraining | Implemented with qualifications | 48 MSD + three BWH volumes; masked reconstruction plus synthetic-tube supervised multitask learning; BWH case numbers do not overlap benchmark metadata. |

## 2. Item 1: methodological hygiene

### 2.1 Stage 1 nesting is real, but does not establish whole-pipeline nesting

`train_nested_stage1.py:526–593` constructs outer test indices with `i % 5 == f_out`, then five inner validation partitions from `outer_train[m::5]`. The code asserts pairwise disjointness from the outer test set. Four outer training sets contain 43 cases and one contains 44; inner subtraining sets contain 34 or 35, and inner validation sets 8 or 9. Actual metadata contain 54 distinct case numbers. This supports case-blocking; a patient-ID manifest is still needed to prove that no person has multiple case identifiers.

I independently found **25** `unet3d_outer*_inner*.pt`, **216** `inner_oof_f*_case_*.npz`, and **54** `oof_case_*.npz` files. Every expected `(outer fold, training case)` map exists. The Stage 1 summary reports the same counts and 20 views per outer test volume. Its `strictly_zero_outer_test_leakage: true` is a self-description, not an independent verification result.

Inner labels select the best epoch through `m_dice + 0.5*m_rec` in `train_single_inner_model`, and the selected network predicts that same inner validation set. These are out-of-gradient-training predictions, but not predictions from an estimator selected without those cases' labels. This distinction does not by itself contaminate outer-test Stage 1 predictions, since all selection stays within outer training. For strict cross-fitting of downstream training data, use a fixed epoch chosen elsewhere or another training-only selection split.

Checkpoint/map reuse is based on file existence. The NPZ inspected contains predictions and crop coordinates, but no training IDs, validation IDs, checkpoint hash, source hash, or configuration hash. Many model entries in the summary are only `cached_from_disk`. Consequently, source correctness cannot prove that every existing cached file was generated by the current code and split configuration.

### 2.2 Critical: upstream contamination remains in Stage 2

The relevant chain is explicit:

1. `run_stage2_and_tip_clean.py:2157–2209` builds every case's graph from global `oof_case_{i}.npz` maps.
2. `train_nested_stage2a_mpnn` uses those graphs. There is no fold-specific rebuild of Stage 2A training graphs from `inner_oof_f{f_out}`.
3. Candidate trajectories, MPNN probabilities, and stitched pools are constructed from these outputs.
4. `build_fold_inner_oof_bundles` (`:1250–1376`) refreshes selected probability/profile features on **the same pre-existing trajectories**. Its docstring explicitly says it reuses the pool and Hungarian assignments. Features such as `conf_heur`, `from_mpnn`, `mpnn_pa`, `mpnn_pl`, and `mpnn_pe` are copied from that pool (`:1317–1340`). Geometry and some adjacency content are also retained.
5. `train_nested_stage2c_1d_tip_localizer` (`:1608–1720`) builds `case_samples` from global `stitched_enriched_bgs`. The main function trains it **before** constructing inner-refreshed bundles (`:2339–2357`). It has no fold-specific bundle-map argument.

For outer fold F, an outer-training case i belongs to another global fold G. Its global OOF model excludes G but ordinarily includes patients in F. Thus Stage 2 training geometry/features can encode information learned from F's labels. Refreshing only some features does not erase that dependency. The same problem affects tip training and few-shot training bundles.

There is also a silent fallback at `:1267–1270`: a missing inner map substitutes the original global-OOF bundle. The current 216-file inventory is complete, so this is a robustness defect rather than an observed missing-file event. Nevertheless, a strict protocol must abort on missing or stale maps.

`--use-inner-oof-crossfit` is optional; the refreshed model is additionally exported as a separate comparison. The final JSON contains both ordinary and inner-refresh rows, and their different results show that the primary published row has not simply become the cross-fit row. Stage 2's helper selects **one** inner holdout per outer fold (`:1159–1171`), rather than training all five inner Stage 2 models. It is nested holdout selection locally, not 5 × 5 Stage 2 training.

**Required fix:** build the complete upstream training DAG separately inside each outer training split: Stage 1 maps, graph inputs, Stage 2A models/predictions, candidate geometry, stitching, Stage 2B features/labels and tip profiles. Cross-fit each learned upstream dependency used to train the next component. Make artifact provenance mandatory and reject unsafe fallbacks. Merely selecting the current `4c` row cannot substantiate zero leakage.

### 2.3 Critical: `k_lim_auto` is not MRI-only

The source contradicts its own title:

- `prepare_clean_dataset.py:181–209`, `compute_tilt_safe_pelvic_klim`, first uses `z_lim_ras` when present, otherwise `k_lim_orig`, and only otherwise the intensity estimator.
- Main discovers `template slice position.acsv` files, including Manual_LB locations, and passes the marker to the NRRD reader (`:469–482`, approximately).
- It sets `k_lim_gt_orig = int(orig_m['k_lim_gt'])` and calls the crop function with **both** marker and legacy crop metadata (`:489–494`). Because the legacy integer is always supplied, the intensity-only fallback is not reached in this call.
- The inherited loader in root `prepare_and_inspect_dataset.py:233–236` derives a missing legacy crop from the median reference base slice.
- The independently calculated `k_lim_sym_safe` is saved but is not the active crop. In the actual clean metadata, **48/54** `k_lim_auto` values differ from `k_lim_sym_safe`.

This is marker-assisted cropping with a possible reference-centerline fallback, not 100% reference-free cropping. NRRD is the MRI container here; the important reference dependency is the ACSV marker / inherited reference-derived metadata, not simply whether a filename ends in `.nrrd`.

The tilt correction projects the boundary at one representative pelvic location `(0.50 I, 0.43 J)`. It is not a mathematical guarantee over the entire oblique ROI. The special `K <= 39` rule retains the whole Stage 1 volume. Only indices 51 and 52 rerun Stage 0; the other 52 retain legacy trajectories (`:638` onward). Thus even substituting an intensity-only Stage 1 crop would not make the complete cached pipeline reference-free without regenerating Stage 0.

The summary's 0 truncated bases, 100% in-FOV coverage and 99.651% total coverage are assigned constants near the end of the script, not computed coverage assertions in that run. Do not present these constants as newly measured validation. Recompute coverage per case from the chosen crop, preserve unmodified primary reference targets, and test all downstream stages with the same bounds.

### 2.4 Few-shot: support exclusion passes, broader claim does not

`export_clean_unified_and_table3.py:388–405` selects `k_indices = outer_train[:k_shot]` for k = 1, 5, 10, 20. Both bundle scorers are trained on these cases. This fixes the explicit use of a full-data base scorer in the small-k scoring blend. The code has a separate synthetic-only k=0 model on Stage 0 candidate bundles, plus an analytical Stage 0 baseline.

However:

- The bundles for k>0 still come from fully trained Stage 1 / Stage 2A and the globally constructed pools described above.
- The budget is a **Stage 2B fine-tuning budget**, not a whole-system annotation budget.
- `outer_train[:k]` is deterministic first-k selection, not random support sampling. One support draw per fold does not measure support-selection variability.
- The zero-shot Stage 0 path inherits assisted cropping and legacy Stage 0 provenance; its “0 clinical labels” claim is not certified by this exporter alone.
- `43_shot` actually denotes 43–44 available outer-training cases, with Stage 2 fitting on 35 and checkpoint selection on 8–9; it is not 43 gradient-training cases in every fold.

The manuscript already appropriately limits the budget to Stage 2B, but its description of blending with a full-fold scorer and its old numbers are stale. Preserve the scope qualification while updating the mechanism and values. Use repeated patient-level support draws if making general claims about one-shot performance.

### 2.5 Specificity: 48 volumes, changed model, and pretraining overlap

The specificity JSON contains **48** records: 32 MSD training-image files and 16 test-image files. The script does not include the three BWH pretraining scans. Those BWH scans are not established needle-free; the SSL code explicitly suppresses suspected pre-existing cores in them. There is no 51-volume needle-free specificity evaluation here.

`benchmark_public_pelvic_mri_clean.py:315` onward loads only `inner0` from each of five outer folds for the additional clean Stage 1 experiment. It uses single-view inference and `temp_pos=1.0`, unlike the benchmark's five-inner-model, four-view, temperature-1.20 configuration. The full-pipeline rows call the production `autoDetectAllNeedles` path, not the clean Stage 2/1D-tip decoder. Post-hoc corroboration uses the production symmetry tubes; the clean Stage 1 count is a separate row.

Most importantly, the clean SSL Stage 1 models were pretrained on **these same 48 MSD images**, including synthetic-tube supervision on their real backgrounds. They are external to the 54-case benchmark, but not held-out from this model's training. Freezing weights during evaluation is insufficient to call this an independent held-out specificity test. Report it as pretraining-cohort background evaluation, or reserve a new external cohort and freeze the full model/thresholds before accessing it.

An additional reproducibility defect exists in `trace_raw_unet_tubes_no_physics`: the second slice of `disk_m` uses the same expression for its start and end, producing a zero-width mask. Boolean indexing of a nonempty patch with that mask is invalid. The provided JSON's positive raw-tube counts therefore require reconciliation with the exact source version that ran. Python parsing passes; it cannot detect this runtime shape error.

### 2.6 Runtime: CPU observations exist, GPU totals are estimates

`benchmark_54cases_timing_clean.py` contains an actual per-case CPU wall-clock call to production `autoDetectAllNeedles`. The artifact has 54 distinct case records with 20 standard-resolution and 34 high-resolution scans. All reported overall stage means agree with independently recomputed per-case means to rounding precision.

But the claimed “exact measured uncached GPU” benchmark is contradicted at `:258–289`:

- GPU Stage 1 runs one network on `torch.zeros`, with dimensions capped at 36 × 112 × 128, at most twice. It does not infer on the actual MRI tensor or run the five-model/four-view ensemble.
- GPU Stage 2A is derived from CPU timing by subtracting `0.62 * s2a_mpnn_measured`.
- GPU Stage 2B is `0.52 * s2b_cpu`.
- GPU total is the sum of these mixed measurements and estimates.

CPU Stage 2A is a residual, not a direct stage measurement. A fallback adds a fixed 0.008 s, and floors are applied. Model loading/warmup and MRI disk loading are outside the main timer. CPU inference uses production ONNX/graph weights, not demonstrably the clean 25-model/1D-tip system. The hardware is a **24-core Threadripper PRO 7965WX with 48 logical threads**, not the manuscript's generic 8-core machine; setting one ONNX session to eight threads does not constrain every Rust/native component.

**Required fix:** benchmark the exact accuracy-producing configuration on real inputs, synchronize GPU work, measure full pipeline wall time, report warm/cold scope and thread counts, and distinguish deployment configuration from experimental ensemble. Relabel existing GPU values as estimates if retained.

## 3. Item 2: ensembling, TTA and augmentation

Verified five inner networks per outer fold and four views per network in `train_nested_stage1.py:599–619`. TTA weights are identity 0.42, X reflection 0.42, Y reflection 0.08, XY reflection 0.08, with spatial unflipping and direction-sign correction. These are in-plane flips of a 3D volume, not axial Z reflections. Logits are averaged and selectively rescaled before sigmoid: outer ensemble TTA uses 1.20, whereas an individual inner-OOF TTA map uses 1.04. This creates a training/inference feature-distribution difference that should be disclosed and assessed without tuning on outer-test outcomes.

`augment_3d_rich` implements cropped training slabs, X/Y flips, spatial shifts via `torch.roll`, a trilinearly interpolated 3 × 4 × 4 multiplicative bias grid, intensity/gamma changes, Gaussian noise, and attenuation of **one** axial slice in derived channels 1–3. No partial-volume Z-blurring exists in the clean training function. The latter is feature-channel attenuation, not dropout of an acquired MRI slice in channel 0. Circular roll introduces wraparound; it is not a padded translation. MRI modifications are applied to cached channels without consistently recomputing all physics-derived features from the altered intensity volume.

These augmentations are meaningful, but the requested list overstates what is implemented. Add and validate Z-blurring or remove it from claims. Describe the actual attenuation and boundary behavior. The controlled item-by-item Stage 1 ablation is only on outer fold 0's 11 cases; it cannot by itself establish a cohort-wide benefit of each intervention.

## 4. Item 3: multi-rater supervision

`prepare_clean_dataset.py` loads LB/AMe through `load_human_interobserver_40cases`, clips traces, matches them to CY trajectories within 3.5 mm, builds consensus shafts over shared support, retains HT obturators, and adds unmatched LB-only targets with weight 0.75. Stage 1 uses `Y_multirater` on inner subtraining cases, but primary `Y` for checkpoint scoring. Actual clean metadata have LB traces in **40 cases**.

Stage 2 constructs primary and multi-rater targets. Tip supervision averages close rater endpoints when their axial difference is at most 4 mm, and otherwise chooses a profile-supported endpoint with downweighting (`run_stage2_and_tip_clean.py:839–941`). LB-only candidates can receive soft positives. The local training loops exclude outer-test target rows. Precomputing labels for all cases is not itself leakage when those labels remain unused by inference; the problem is the upstream learned-feature provenance already documented.

The brief's primary reference **AS** is not established by this code. The implemented benchmark uses **CY long trajectories + HT short obturators**, with CY alone for the long subset. The manuscript also calls the primary team CY. Do not silently equate AS with CY; document the intended naming mapping if one exists.

Table 3 now compares unlike annotation inventories unless carefully labeled: all-primary CY+HT contains 723 trajectories on these 40 scenes, versus LB/AMe's 513; the CY-long subset has 505. The “long” LB subset is selected using a ≥28 mm span rule, leaving 460 traces, not all 513. Low recall against all CY+HT is not simply human disagreement on an identical annotation task. Also, since LB labels train the model on other patients, held-out LB agreement is valid conditional on correct nesting but should not be marketed as agreement with an entirely unseen annotation style.

## 5. Item 4: dedicated tip model and stitching

Verified in `run_stage2_and_tip_clean.py:385–601`:

- 91 samples on `linspace(-25, 20, 91)`, exactly 0.5 mm apart, centered at the candidate tip with zero at index 50.
- 22 channels containing neural probabilities, tangent quantities, symmetry/ring/intensity information, relative positions, in-bounds status and axial derivatives.
- Hidden width 48, residual Conv1D blocks with dilations 1, 2 and 4, four-head self-attention, a soft-argmax distribution head and a residual regressor.
- Training includes profile shifts, Gaussian step targets and weighted regression/distribution losses.
- One selected model per outer fold. The decoder uses that fold's model, not a cross-fit tip-model ensemble.

**Not implemented:** BiGRU, GBDT tabular blend, and the specified inner-cross-fit ensemble. There is no `GRU`, `GBDT`, `HistGradientBoosting`, or gradient-boosting training implementation in the eight clean files. Stage 2 uses MPNN and graph transformers. The existing decoder blends neural, graph-tip, profile and neighborhood estimates; that is not a GBDT blend.

Stitching is real (`:281–379`): allowed axial gap up to 20 mm and overlap up to 6 mm, extrapolated midpoint separation ≤2.2 mm, slope discrepancy ≤0.14, upper-fragment support ≥0.26, followed by duplicate checks. These thresholds and the subsequent cranial extension need training-only selection provenance. A slope-vector difference of 0.14 is only approximately an eight-degree angle; label it by its implemented quantity. Examine false joins between close parallel catheters and tips outside the profile window. A 0.5 mm interpolation grid does not imply 0.5 mm acquired resolution or localization accuracy.

The final and unified JSON files agree on the following whole-cohort rows (TP, prediction count and complete metric dictionaries checked):

| Configuration, τ=50% | TP / predictions | F1 (%) | Mean 3D tip error (mm) | Median (mm) |
|---|---:|---:|---:|---:|
| Stitched + multi-rater, before dedicated tip model | 972 / 1118 | 89.13 | 6.9169 | 2.5568 |
| Dedicated tip, without SynthBrachy3T | 983 / 1153 | 88.72 | 6.3929 | 2.2612 |
| Dedicated tip, with SynthBrachy3T | 972 / 1118 | 89.13 | 6.2335 | 2.2650 |
| Optional inner-feature-refresh row | 981 / 1139 | 89.10 | 6.2519 | 2.2420 |

The full synthetic row has tip P95 **27.2682 mm**; the refreshed row **28.4562 mm**. These tails remain material. Comparing stitched and dedicated-tip rows supports an observed tip improvement in this artifact, but does not repair the nesting defects or prove every requested architecture component exists.

## 6. Item 5: external pretraining

`pretrain_ssl_unet3d.py:111–129` enumerates 48 MSD files and three BWH volumes: two from Case 061 and one from Case 100 Prostate. The actual 54-case metadata exclude both 61 and 100. Thus BWH case-number separation is verified, but these are **51 volumes, not 51 independent patients**: the two Case 061 scans share a case. MSD-to-BWH identity separation is asserted by dataset provenance, not checked with patient identifiers by the script. The summary's `overlap_with_54_benchmark_cases: 0` is hard-coded rather than computed from a manifest.

The reconstruction branch masks roughly 25% of spatial blocks plus one or two axial slices, replaces masked channels 0–3 with noise, and reconstructs clean channels with weighted L1 + MSE. This implements masked reconstruction / noise replacement. Mild global intensity jitter occurs before the reconstruction target is saved; that jitter is not separately removed by reconstruction.

Pretraining is **not solely self-supervised encoder pretraining**. It jointly optimizes synthetic susceptibility-tube multitask segmentation on the real backgrounds and masked reconstruction, with `loss_tube + 0.30 * loss_mae` (`:483`). The full U-Net state, including decoder/task heads, is saved and loaded strictly into Stage 1. Name the method accordingly: synthetic-tube multitask pretraining plus masked reconstruction on external MRI backgrounds.

The summary records 45 epochs, RTX 4090, 370.96 seconds, final total loss 0.1719, tube loss 0.1504, reconstruction loss 0.0718, and synthetic-tube Dice 0.8667. These are training diagnostics, not held-out efficacy results. The same MSD backgrounds are later used for clean-model specificity, as discussed above. Suppressing suspected cores in BWH images does not certify that those images were originally needle-free.

## 7. Numerical and manuscript reconciliation

### 7.1 Verified internal consistency and its limits

I recomputed counts, means, medians and P95 from every exported shaft/tip error array: **18 method/cohort entries**, nine methods in each of all-1063 and long-759. Every array length equals its row's Hungarian match count; numerical summaries agree within 0.00015 mm, consistent with four-decimal array rounding. Each checked patient F1 array has 54 entries. The standalone Table 3 object exactly equals `human_interobserver_agreement_40_overlapping_cases` in the unified JSON.

I also recomputed false-positive totals and sample SD for all **16 specificity configurations**, and overall per-stage timing means for both CPU and GPU columns. These arithmetic checks pass. They establish internal consistency, not independent inference provenance or measured GPU validity.

Limitations of the “single transaction” claim:

- The exporter calls the evaluator and a separate distribution extractor on the same predictions and asserts counts. It does not literally share one stored matching result.
- Only nine methods per cohort retain distributions; 20 summary rows exist per cohort.
- Table 3 has summaries, not raw match arrays to independently reproduce every statistic.
- The long-cohort distribution extractor does not accept `aux_short_gts_per_case`, so its patient precision/F1 arrays are strict long-only measures. The evaluator also reports obturator-adjusted measures. These are different estimands: the synthetic τ50 long row has strict F1 **84.77%**, adjusted F1 **90.00%**. Label plots/tests explicitly.
- The exporter deep-copies an old JSON template; inherited metadata need auditing rather than assuming the new date updates everything.
- Keys say `bootstrap_95ci_1000_patients`, but clean unified/Table 3 calls request **300 bootstrap draws**, and Stage 2 progression **200**. Neither means 1,000 patients. Use explicit `n_bootstrap`, resampling unit, seed and CI method fields.

### 7.2 Main accuracy results: source, prose and tables disagree

The manuscript includes `tables/tab2_main_results.tex`, not its `_v1` alternative. The active table and prose contain different historical results, neither synchronized to the clean artifacts.

| Quantity | Manuscript prose / abstract | Active Table 2 where applicable | Clean unified artifact |
|---|---:|---:|---:|
| Synthetic-tip τ50 recall (%) | 89.8 | 91.3 | **91.44** |
| Precision (%) | 85.6 | 85.3 | **86.94** |
| F1 (%) | 87.7 | 88.2 | **89.13** |
| Median shaft error (mm) | 0.46 | 0.46 | **0.4695** |
| Median tip error (mm) | 2.77 | 2.61 | **2.2650** |
| PMB2019-style F1 (%) | 83.1 | 81.3 | **83.06** |
| ResUNet/clDice F1 (%) | — | 37.9 | **38.68** |
| Xiao-style F1 (%) | 54.2 | 53.7 | **54.19** |
| Stage 0 F1 (%) | 76.5 in budget discussion | 69.8 | **75.34** |

Sources: manuscript `:368–379`, `:1080` onward and `:1150–1175`; active Table 2 rows. These clean values describe the current artifact, not a certification that it is unbiased.

There is also a labeling problem in the exporter: `Unified NeedleFinder 2026 (tau=50%)` and `Unified NeedleFinder 2026 + IRGM + SyncTrack + Poly3 (tau=50%)` both use the same `nosyn_by_tau[50]` predictions (`export_clean_unified_and_table3.py:714–724`). They cannot support separate ablation claims for adding IRGM/SyncTrack/Poly3. Run a genuine toggled ablation or merge/rename the duplicate rows. The active AxialTransUNet3D table row has no corresponding result in the clean unified cohort dictionary; supply a traceable artifact or remove it from this clean comparison.

### 7.3 Synthetic pretraining significance claim is no longer supported

Manuscript `:373–374` and `:1107–1108` state mean tip error improves from 9.11 to 7.70 mm with p=0.003 and F1 p=0.11. The clean Stage 2 artifact instead reports:

- Mean matched-catheter tip error: **6.3929 → 6.2335 mm**.
- Patient-paired tip Wilcoxon p = **0.172341**.
- F1: **88.72 → 89.13%**; patient-paired F1 p = **0.319904**.
- Median tip error: **2.2612 → 2.2650 mm**, a tiny increase.

Remove the old significant-tip-benefit claim for this experiment. Distinguish pooled catheter means from patient-paired means: the latter are 6.0194 versus 5.9115 mm in the test record. Recompute all table footnote p-values and comparisons from one selected, corrected experiment. Changing matched sets between methods further limits interpretation as a paired catheter-level localization effect.

### 7.4 Few-shot values

| Stage 2B budget | Clean all-catheter F1 (%) |
|---|---:|
| Analytical Stage 0 | 75.34 |
| Synthetic-only Stage 2B, zero clinical fine-tuning | 78.19 |
| 1 case | 79.74 |
| 5 cases | 83.75 |
| 10 cases | 85.69 |
| 20 cases | 86.61 |
| Full available outer fold | 89.13 |

The manuscript's 84.9% one-shot result and statement that it beats the 83.1% PMB-style model do not hold for this artifact: **79.74 < 83.06**. Correct the abstract, Figure pareto caption and discussion. Distinguish the two zero-shot baselines; the artifact aliases `0_shot` to the synthetic-only model, whereas the old text uses that name for analytical processing.

### 7.5 Table 3 must be rebuilt with explicit denominators

| Clean Table 3 comparison | Reference / predicted traces | Matches | Recall (%) | Mean shaft (mm) | Median tip (mm) |
|---|---:|---:|---:|---:|---:|
| LB/AMe vs CY+HT | 723 / 513 | 476 | 65.84 | 0.7260 | 3.2078 |
| LB long subset vs CY long | 505 / 460 | 424 | 83.96 | 0.7005 | 3.2407 |
| NF τ50 vs CY+HT | 723 / 768 | 653 | 90.32 | 0.5817 | 2.0966 |
| NF τ35 vs CY+HT | 723 / 825 | 669 | 92.53 | 0.5862 | 2.1969 |
| NF τ50 vs primary-first union | 759 / 768 | 669 | 88.14 | 0.5862 | 2.1660 |
| NF τ50 vs LB/AMe | 513 / 768 | 465 | 90.64 | 0.7700 | 3.9689 |

The active table still cites `2026-10-22-table3-observers-and-nf.json` and uses 464 human matches and 469 NF-versus-CY matches; prose uses 465 human matches. None is a faithful transcription of the new all-primary comparison. Do not divide 476 by 505 to repair this: the 476 matches were obtained against 723 references.

The union is operationally primary-first with LB traces appended only if they are farther than 3.5 mm from existing references. It is not a symmetric consensus reference or independent third annotation. Its 759 references on 40 scenes must not be confused with the separate 759-long-catheter cohort on 54 scenes. A lower discrepancy against the training annotation style is not evidence of absolute superiority to human experts.

### 7.6 Specificity values and configuration labels

| Configuration | Clean FP total | Mean FP/volume | Zero-FP volumes |
|---|---:|---:|---:|
| Raw production Stage 1 tubes | 306 | 6.375 | 7/48 |
| Script-defined pre-transformer candidate count | 294 | 6.125 | 2/48 |
| Production Stage 1 + symmetry gate only | 18 | 0.375 | 31/48 |
| Clean SSL Stage 1 + symmetry gate only | 6 | 0.125 | 42/48 |
| Production full pipeline, τ50 | 197 | 4.104 | 8/48 |
| Production full pipeline, τ60 | 171 | 3.562 | 8/48 |
| Production pipeline + tube corroboration, τ50 | 16 | 0.333 | 38/48 |

The active Table 4 still reports a candidate pool of 985 and τ60 full-pipeline total of 188. The manuscript's 4.7 FP/scan default result is stale. Its 18 tracks / 31 zero-FP scans correspond to the **Stage 1 gated-only row**, not the new full-pipeline-plus-corroboration row. The old 98.2% rejection claim uses the old 985 denominator and cannot be retained for the new 294 count. Rejection comparisons also require matching candidate populations; simple counts of independently traced pools are not a track-level rejection audit.

Report both FP/volume and the proportion of volumes with zero predictions, with uncertainty. Avoid treating the ≤1-FP proportion as conventional diagnostic specificity. The 6-track clean-model result is from a different inference configuration on its pretraining cohort and must not replace a held-out full-pipeline result.

### 7.7 Runtime values

| Configuration | All 54 (s) | Standard 20 (s) | High-resolution 34 (s) |
|---|---:|---:|---:|
| CPU artifact, production run | **4.1512** | **1.2600** | **5.8519** |
| GPU column, mixed estimates | **2.6744** | **0.6749** | **3.8506** |
| Manuscript CPU claim | 2.74 | 1.20 | 3.47 |

Clean CPU stage means are 0.2669, 2.5750, 1.2482, and 0.0611 seconds for Stage 0, Stage 1, Stage 2A and Stage 2B. The old manuscript's 0.41/0.96/1.37 breakdown and 1.25-second GPU result do not match. Even the old weighted CPU subgroup calculation `(20×1.20 + 34×3.47)/54 = 2.6293`, not 2.74. Update performance comparisons only after obtaining matched configuration/hardware timings.

### 7.8 Adhesion subgroup and protocol prose

The JSON has **23 severe-adhesion cases containing 571 total reference catheters**, of which **233 are classified as adhering** across the broader analysis. Its long-catheter evaluation in those selected cases has 437 references. The `adhesion_233` key in Stage 2 actually evaluates all catheters in selected cases, not a 233-catheter-only subset. The active Table 2 heading correctly says 571, but stale names/captions/footnotes must be reconciled.

The artifact's severe-case criterion is “≥4 long or ≥6 total catheters passing within ≤3.0 mm on shared axial slices”; the table footnote describes “≥2 catheters ... ≥3 consecutive slices.” These are different definitions. State the implemented rule and distinguish patient selection from catheter-level subset evaluation.

Manuscript `:923–937` still explicitly describes reference-assisted cropping, test-fold checkpoint selection, and non-nested stacking. Do not simply replace it with a blanket zero-leakage statement: Stage 1 checkpoint selection has improved, but crop and stacking problems remain. Write a component-specific protocol after fixing the remaining paths.

## 8. Required actions before claiming completion

1. **Repair data isolation first.** Regenerate intensity-only crops and all affected Stage 0/1/2 inputs; reconstruct every downstream training dependency within each outer training split. Remove permissive fallback to global OOF data. Store split and hash manifests with every checkpoint/map/cache.
2. **Choose and name the implemented architecture honestly.** Either implement the planned Z-blurring, BiGRU, cross-fit tip ensemble and GBDT blend, or formally revise the plan to the actual residual/self-attention tip model and graph-transformer pipeline.
3. **Separate pretraining, model selection and external testing.** Reserve an untouched needle-free cohort for the complete frozen clean model. Report 51 pretraining volumes and 48 current specificity volumes separately; audit patient overlap and annotation status explicitly.
4. **Measure the evaluated system.** Replace dummy-input/scaled GPU figures with real end-to-end inference measurements, and benchmark deployment separately if its smaller architecture is the clinically relevant runtime.
5. **Regenerate the manuscript from a single selected artifact set.** Update abstract, active tables, plotted data, captions, discussion, denominators, p-values and runtime hardware together. Remove duplicate ablation labels and stale significance claims. Do not mix the ordinary and optional inner-refresh rows.
6. **Improve inferential presentation.** Report patient-clustered confidence intervals, number of bootstrap draws, predefined primary threshold, exploratory threshold sweeps, matched-only error denominators, missed-catheter penalties where used, and heavy-tail tip statistics. Do not interpret interpolation spacing as accuracy or use “human-level” language without a matched annotation task and explicit equivalence analysis.

The present artifacts provide useful evidence of working ensembling, auxiliary multi-rater supervision, a dedicated axial tip model, stitching, external pretraining and improved export consistency. They do not yet provide the claimed comprehensive verification of all five plan items. The critical blockers are active reference-assisted cropping, incomplete upstream nesting, reuse of the SSL cohort for clean specificity, estimated GPU runtime, and widespread manuscript/artifact version mismatch.

## Audit snapshot identifiers

SHA-256 hashes of the principal audited inputs (to distinguish subsequent edits from this review):

```text
7c7c753ffbb45fa2a9347d136043fc046c3951f836baf870ca5e812aad7bccd9  vault/paper/benchmarks_clean/benchmark_54cases_timing_clean.py
aaa9129dce15393493752f4dc3d5c381be7b5f1f19c226b91debb19ffdee3854  vault/paper/benchmarks_clean/benchmark_public_pelvic_mri_clean.py
549530291f1e0c97eb60b65519b56c1c62cc63bfea61bf32a5c15628d35097f9  vault/paper/benchmarks_clean/export_clean_unified_and_table3.py
3fd7a9d634a6ab5aae410120128b4e2dd33a6fbf3f3c589e04d3b296889a7365  vault/paper/benchmarks_clean/prepare_clean_dataset.py
1e54b803d12e567d6521a6170ca109c957b95b44c9e03f2eae0ea9e3c0162bb0  vault/paper/benchmarks_clean/pretrain_ssl_unet3d.py
84b4cbe2e5fe2903ce292048b24c5cb92a994b8ae4a0f9fc1e140aed6f5884f8  vault/paper/benchmarks_clean/run_stage2_and_tip_clean.py
d89d15fe53fbd874ed8b32741abd39e929a87c31472fcc278bd772127a8be867  vault/paper/benchmarks_clean/train_nested_stage1.py
d6e71748af0ac570e8b5df63889e1d2c17c88c4feb3d8f3bc0a93fd9a6e3bdc4  vault/sources/2026-10-23-unified-clean-nested-benchmark.json
a59a4dcc107f25a9fe53652cd44459fd31c955da6b6218f1c3f8d60ca2e4fbc8  vault/sources/2026-10-23-table3-observers-and-nf-clean.json
15a69ca6df146dc6c9115885ee5dcde0aa835f050c0678a393b744f5f5c5475f  vault/sources/2026-10-23-stage2-and-tip-clean-final.json
c3ff9b274c798e99d13859e45d0deeea343bd2967771ddb988f2f472926f4713  vault/sources/2026-10-23-public-pelvic-mri-specificity-clean.json
d86f3375c29257e6f5402f0c0c2343b19cfd7b99a2599d1b1c93adb7cbd3f0fe  vault/sources/2026-10-23-per-case-timing-benchmark.json
515083582415c860a40c80cea2c72f89fc50f4df5a72d607506ac35d99a94770  vault/paper/needlefinder_2026_pmb.tex
```
