# Audit: `manuscript.md`, Tables 1–6, vs. `vault/sources/2026-10-23-*.json`

**Bottom line.** Most headline numbers reproduce from the JSONs. That covers Table 2 (every row, with p-values), Table 3 (detections and shaft/tip), Table 4 (every row), Table 6 Part A (every dosimetry cell), the review-queue rows, and the prostate and cross-sequence rows. The paper is not submission-safe yet. Eight issues would fail a careful reviewer: a unit-of-analysis error behind the TOST p-values, a k_lim claim the JSON contradicts, a review simulation that uses ground truth but is presented as a workflow result, selective 4D reporting, a "zero-shot / out-of-distribution" framing that conflicts with how the SSL model was trained, timing numbers that don't match the timing JSON, a false-positive taxonomy no current JSON produces, and prose numbers that disagree with the tables. I did not edit any files.

## Blocking issues

1. **TOST / Wilcoxon p-values use the wrong unit.** Abstract, Results (l.224) and Table 3 fn e report p = 1.6e-41, 5.5e-10, 2.2e-14 and 6.8e-19. `run_exp2_uncertainty_triage_conformal_grubbs.py` (`run_tost_analysis`) feeds one difference per matched catheter (n = 432) and treats them as independent. Methods say the tests are over the 54 patients. Only the patient-clustered bootstrap (p = 0.0014, tip mean) is at patient level. The JSON also does not support "equivalence": shaft-mean `is_equivalent_tost_at_alpha_0p05` is `false`, parametric TOST p = 0.0024. "TOST equivalence p = 5.5e-10" reads as equivalence when it is a non-inferiority test. *Fix:* rerun on patient-level statistics (or a cluster bootstrap), report non-inferiority only.

2. **k_lim_auto claims contradict the audit JSON.** Abstract: "preserves 100.0% of in-FOV catheter depth". Methods (l.95): "mean difference of −0.06 slices (median 0.0, IQR 1.0)". In `failure-analysis.json` → `task2…/policy_comparison_54cases`, the reference-free policy `reference_free_sym_safe` preserves 99.65% of z-span, and its crop differs from the ACSV marker by a mean of −22.8 slices (median −24.5). The −0.06 / 0.0 values appear only for the ACSV reference row, which is zero by construction. I am assuming `reference_free_sym_safe` is k_lim_auto; please confirm. If so, both claims fail.

3. **The review-budget results use ground truth, and the paper does not say so.** In `run_exp2…py` (queue 1), for each inspected tip with error > 2 mm the post-review error is `hypot(dr_perp, min(|ds_par|, 0.85 mm))`, computed from the reference trajectory. That assumes a perfect one-click correction. The Abstract, Results (l.260), Table 6 Part B and Discussion (l.280) present 3.72 and 2.77 mm as expected clinical outcomes. Also, in Table 6 Part B the Nom./Emp. and width columns for the k = 2, 5, 8 rows are the unreviewed k = 0 intervals (84.4%, 90.2%, 95.4%; 8.71, 12.11, 18.42 mm), not post-review values. *Fix:* state the oracle assumption in Results, Abstract and the table caption, and move the interval columns to a separate "k = 0 intervals" block.

4. **4D results are selectively reported.** From `exp4` per-case final-space F1 (independent → causal 4D):
   - Case 068: 77.19 → 58.67 (−18.5 pp). Omitted.
   - Case 069: 84.71 → 92.86 (+8.2 pp).
   - Case 070: 85.71 → 91.30 (+5.6 pp).
   - Case 071: 76.19 → 76.47 (+0.3 pp).

   Results (l.264) and the Abstract report only the two gains. Table 6's "4D Causal Kalman (Cases 69/70)" row says 23 volumes and 108 catheters. Those two cases have 10 scans and 42 + 22 = 64 reference catheters. Its shaft value (0.87) is Case 069's and its tip value (4.79) is Case 070's. *Fix:* report all four cases with per-case rows.

5. **"Zero-shot / out-of-distribution" conflicts with the SSL training data.** The Methods (Patient Cohorts), Table 1 fn d, `step2_ssl_pretrain_summary.json` and `pretrain_ssl_unet3d.py` (lines 5–7, 115–128) show that the main checkpoint `unet3d_ssl_pretrained_51vols.pt` was pre-trained on all 48 MSD volumes and on Case 100. That creates two problems:
   - The Table 4 "Clean SSL" row (0.125 FP/scan, the headline) had MSD in its pretraining pool. Footnote f admits this, but the Table 4 caption, Table 1 fn d and the Abstract still call the 48 MSD scans "out-of-distribution" and report 0.125–0.33 as zero-shot.
   - The prostate claim ("zero prostate exposure of any kind (excluding Case 100 from SSL pre-training)", l.262) comes from a different model. `run_exp3…py` line 17 calls it "Zero-Exposure GYN Ensemble, No-SSL". The manuscript never describes a no-SSL model. The Table 6 prostate row is at τ = 35 while the main results use τ = 50, and the table does not say so.

   *Fix:* describe both model families and state which rows used MSD in pretraining. Drop "out-of-distribution" for the MSD set or label those numbers as the no-SSL test.

6. **Timing numbers conflict with `per-case-timing-benchmark.json`.** `stage_breakdown` gives these CPU means:
   - Stage 0: paper 0.41 s (0.12 std / 0.59 high-res); JSON 0.27 s (0.07 / 0.38).
   - Stage 1: paper 2.28 s (1.88 GPU); JSON 2.58 s (1.33 GPU).
   - Stage 2A: paper 1.22 s (0.54 GPU); JSON 1.25 s (1.05 GPU).
   - Stage 2B: paper 0.24 s; JSON 0.06 s (0.03 GPU).
   - End-to-end CPU: paper "4.15 ± 2.38 s, median 5.35 s"; JSON SD 2.81 s, median 4.41 s. Standard-res median: paper 1.26 s; JSON 1.07 s. High-res median: paper 5.94 s; JSON 5.56 s.
   - GPU end-to-end: paper "2.67 ± 1.44 s, median 3.45 s"; JSON SD 1.95 s, median 2.93 s.

   The means (4.15, 1.26, 5.85, 2.67) match. The stage values appear in Methods, Results (l.246), the Fig. 1 caption (l.37) and Table 5 notes.
   Table 2's baseline times (PMB 14.80, clDice 3.85, Xiao 3.42) have no source JSON; the per-case file covers only NF stages. Table 2 Part B times are cumulative (0.27 → 2.84 → 4.09 → 4.15) but are not labelled as such.

7. **Stage-by-stage prose doesn't match Table 2 Part B.** Results l.204:
   - Stage 0: prose "83.2% recall, 0.65 mm median shaft, 3.92 mm median tip". JSON and Table: 82.8%, 0.47 mm, 3.08 mm.
   - Stage 1 ("2a"): prose "95.3% recall (97.5% long), 72.1% precision (82.1% F1), 0.45 mm, 2.98 mm". JSON Stage 1B: 94.2% (97.0% long), 61.5%, 74.4%, 0.46 mm, 2.95 mm. The 95.3% matches Stage 0's long-shaft recall (95.26), which suggests a mix-up.
   - "3c": prose "84.2% precision (86.4% F1), 6.48 / 2.33 mm". No JSON row matches. Item 3 is 81.7 / 86.4 / 7.38 / 2.57; the collinear row is 86.9 / 89.1 / 6.92 / 2.56.
   - Section 2B (l.131): "recall above 95% (97.5% long)" (JSON 94.2 / 97.0). "about seven false tracks per patient" after Stage 1 (JSON (1627 − 1001) / 54 = 11.6). "Stage 2A removes more than half" (JSON 11.6 → 6.6, a 43% cut).

   The text uses labels 2a, 3c, 4a, 4b; Table 2 uses Item 1, 3, 4a, 4b. Use one scheme.

8. **The FP/FN taxonomy cannot be reproduced from any current JSON.** Results l.234–236 gives 66 / 48 / 16 / 9 / 7 false positives, 66 / 25 misses, 47 / 21 / 23 for adhesion/short/dropout, and "16 confirmed by LB/AMe". `failure-analysis.json` is a stale run: 1,116 predictions, 955 matched, 108 misses, 161 unmatched FPs, 190 tail tips > 10 mm. The clean run gives 1,118 / 972 / 91 / 146 / 162. The parts that do check out are 66 long and 25 HT misses (759 − 693; 304 − 279) and the 16-trajectory union gain (669 − 653). The tail taxonomy 67 / 41 / 27 / 20 / 7 is in `exp2` part 2a and matches. *Fix:* regenerate the failure analysis on the clean run, or remove the unsupported percentages.

## 1. Internal inconsistencies (paper vs. paper)

| Item | Where | Conflict |
|---|---|---|
| CT pairs | Abstract, Contributions, Discussion: "40 paired" | Results (l.264) and Fig. 8 caption: 38 paired (40 scanned in total). Table 6 Part C: n_vol = 40 with 801 catheters, which come from 38 pairs |
| Table 2 "Part C" | Results l.184 | Table 2 has only Parts A and B; the long shafts are a block inside Part A |
| Stage numbers | Results l.204 vs Table 2 | See blocking issue 7 |
| LB/AMe extras | Table 1 fn c: "35 extra traces" | Table 3 fn b: 36 unique LB/AMe trajectories (759 − 723) |
| SSL pool size | Table 1 row "External SSL": Pts = 51 | Text: 51 volumes from 50 subjects |
| SSL resolution | Table 1 row: XY 0.60–0.88 mm, Δz 3.0–4.8 mm | Case 100 is 0.47 × 0.47 × 1.6 mm (exp3 metadata), outside both ranges |
| 4D cohort | Results l.264; Table 6 row; Fig. 8 | Text says Cases 68–71; table row says "Cases 69/70" with 23 volumes |
| Tip SD in Discussion | l.280: 2.98 and 3.56 mm | These are robust MAD values; Table 3 reports winsorized 4.94 and 5.09 mm |
| Tail percentages | l.282: 99.3% longitudinal "(41.4% caudal, 25.3% cranial, 16.7% human)" | 41.4 + 25.3 + 16.7 = 83.4. The 12.3% adhesion and 4.3% boundary cases are omitted, so it reads as summing to 99.3 |
| Dose statement | Plain-language summary: "within 2.8 to 5.9 percent" | 2.8 is the median and 5.9 the mean |
| Secondary MRI count | Results l.262: "297 secondary MRI series" | 297 includes the 54 primary scans; secondary = 243 |
| Fig. 3 caption | "human IOV IQR 0.33–0.85 mm, P95 1.90 mm" | IQR is from the 432-catheter three-way set; P95 is from the 424-pair set in Table 3 |

## 2. JSON-to-paper mismatches

| Paper location | Paper says | JSON says (file → field) |
|---|---|---|
| Table 6 text (l.258) | ρ = −0.8519 (dwell-weighted tip vs γ2/2) | −0.8474, p = 6.5e-16 (`exp1` → `spearman_dwell_weighted_true_tip_err_vs_gamma_2pct_2mm`) |
| Methods / Table 6 text (l.258) | 2.5 mm dwell-step spacing | 5.0 mm (`exp1` → `tg43_source_parameters.dwell_step_spacing_mm`) |
| Few-shot (l.206) | k = 1 long F1adj 84.3% | 79.9% (`unified` → `few_shot…/1_shot/cohort_long_759…`) |
| Few-shot (l.206) | k = 5 median tip 2.33 mm | 2.49 mm |
| Few-shot (l.206) | k = 10 long F1adj 88.1%; k = 20 88.8% | 88.95%; 88.44% |
| LB/AMe long tip (l.220) | SD 10.67; IQR 4.33; P95 23.13 mm | SD 7.29; IQR 6.02; P95 21.88 mm (`table3…/2_human_inter_observer_long_subset`) |
| LB/AMe long shaft (l.220) | IQR 0.52 mm | 0.50 mm |
| Adhesion (l.198) | p = 5.4e-4, 2.1e-6, 2.4e-7 | Not in any JSON. The Wilcoxon blocks cover the whole cohort only |
| Tissue push (l.264; Table 6 fn d) | Near ≤10 mm vs far >20 mm, p = 6.1e-4 | The JSON test is ≤12 mm vs >12 mm (`mann_whitney_near_le_12mm_vs_far_gt_12mm_p`). The medians (1.41 / 0.93) do match the ≤10 / >20 bins |
| Table 6 CT row | n_vol = 40 | 38 paired scans (`exp4` → `part4b`) |
| Stage timings | See blocking issue 6 | |
| k_lim (l.95; Abstract) | See blocking issue 2 | |

**Matching:** Table 2 Part B values and the p-values in its footnote, Table 3 metrics, Table 4 rows, Table 6 Part A (every dosimetry cell), conformal coverage and widths, the queue-1 review rows, the CT adjudication (52/100, 70 disputed tips, 4.20 / 4.31 / 8.00 mm, 55/70), the Grubbs SDs, AUROC 0.805 and ρ = 0.532, the prostate rows, the wbSSFP and 2D T2 rows, and the calibration numbers.

## 3. Hard-to-verify or confusing claims

1. **"Pre-specified τ = 50%"** (Abstract, l.194). A reviewer will ask when this was fixed. Nothing in the JSON records a pre-registration. Say "τ = 50% was set as the default before the nested evaluation" only if that is documented; otherwise say it is the default and point to the sweep in Table 2.
2. **"Obturator-adjusted precision."** Add one sentence: "Predictions that match a secondary obturator track are not counted as false positives; recall is unchanged." Show the strict value (79.1%) beside it.
3. **"Dwell-weighted tip error."** Define it: each catheter's tip error weighted by its share of optimized dwell time.
4. **Protocol A / B** (Table 6 caption). Put one-line definitions in the table itself: "A: reference dwell times on predicted catheters. B: dwell times re-optimized on predicted catheters, dose evaluated on reference anatomy."
5. **"Passes within 3.0 mm"** (severe adhesion). Specify the distance (closest approach of centerlines on shared axial slices?) and confirm it against the code.
6. **"98.8% of squared tip variance is longitudinal."** This is an energy fraction, not a variance. Rewrite as: "Across the 972 matched catheters, 98.8% of the squared 3D tip error lies along the catheter direction."
7. **Grubbs columns** (Table 3, bottom block). The column headers hide three different estimators. Label them "RMS σ", "MAD σ", "winsorized σ_z" and "median 3D tip error vs CT". Explain why the Rater 1 MAD tip SD is 0.001 mm (a degenerate estimate, CI 0.001–2.02), which is why it is omitted.
8. **CT adjudication.** "52/100" needs a definition: 100 = unmatched NF detections vs CY in the 38 CT-paired patients; "confirmed" = present on post-implant CT (what distance threshold?). The 70 disputed tips are a selected subset (CY–LB disagreement ≥ 5 mm), so "78.6% closer to CT" is conditional on that disagreement. The plain-language sentence "real tubes that one human team had left untraced" conflates the two; reword it.
9. **Count-guided recovery (0.65 clicks/patient).** State where N_implant comes from. If it is the reference count, say that this is an oracle count, and say the clicks are simulated against ground truth.
10. **Conformal coverage** (79.3% at 80% nominal, etc.). Say it is marginal over the 972 matched catheters only; unmatched predictions and missed catheters are not covered.
11. **Speed comparisons** (l.246). "One to two orders of magnitude faster than our 2019 pipeline (14.8 s in our re-implementation)" is about 3.6×. Replace with "about 3.6× faster than our re-implementation, and 29–130× faster than the 2–9 minutes reported." "2.8–4.3× faster than Xiao" should say "than the published 11.6 s"; our re-implementation runs in 3.42 s, faster than NF's 4.15 s.
12. **TOST wording.** Use "non-inferior" throughout. Remove "TOST equivalence p" from Table 3 fn e (see blocking issue 1).
13. **Table 5 statistics.** The NF row reports medians, while most literature rows report means. Add the NF means (0.59 mm shaft, 6.23 mm tip) so readers compare like with like.
14. **Unverified claims in the text.** "No swapped identities" (l.210) needs a count across the 54 cases or should be removed. "Largest annotated MRI cohort" (Contributions) is a literature claim I did not check.
15. **Selection rules.** Table 6 Part C uses 49 of 70 2D AX T2 scans and 2 of 9 wbSSFP scans (the JSON's `n_scans`), and 63 aligned post-insertion axial scans. The inclusion rule is not stated.

## Limits of this audit

- I did not verify the literature values in Table 5, the citations, or the figure images (PDF/SVG).
- I did not re-run any benchmark script. The checks compare the cited fields to the JSON at full precision, and I accepted the paper's rounding.
- Two mappings are inferences that need confirming: that `reference_free_sym_safe` is k_lim_auto, and that the 0.85 mm floor in the review simulation is the intended model.
- `vault/sources/2026-10-23-audit-haiku.md` and `…audit-luna.md` exist as empty files. I did not write to either. Say the word and I will save this report to `audit-haiku.md`.
