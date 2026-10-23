File tools were disabled in this session, so I could not read the vault or check which of these pieces already exist in the repo. The brainstorm below is grounded in the brief only.

## Immediate high-ROI experiments (hours on edelweiss)

Ranked by expected impact per hour.

**1. TG-43 dosimetric impact study on all 54 patients.** This is the single addition that turns the paper from a geometry paper into a clinical-physics paper, and it is on-brand for PMB. Plan it as a plan-transfer experiment, because you almost certainly lack the clinical dwell times:

- **Geometry sources per patient:** reference consensus, NeedleFinder 2026 out-of-fold tracks, the PMB 2019 U-Net tracks, and rater 2 where available. Use only matched catheters. Missed and spurious catheters are caught in the clinic by the known implant count, so keep them out of the dose comparison and report them separately.
- **Structures:** if no clinical contours exist, define a surrogate HR-CTV as the alpha shape of the reference active dwell positions dilated by a few millimetres, shared across all geometry sources. Get bladder, rectum and sigmoid from TotalSegmentator MRI on the T2-SPACE volume for D2cc.
- **Dwells:** index from the tip with a fixed first-dwell offset and 5 mm step, activate dwells inside the CTV plus margin. Because dwells are indexed from the tip, this is exactly where your tip error enters the dose.
- **Dose engine:** line-source TG-43 for an HDR Ir-192 microSelectron v2 or Flexisource using the published consensus g(r) and F(r,θ) tables. Vectorised NumPy on a 1 mm grid takes seconds per patient. Heterogeneity corrections cancel in a difference study, so TG-43 is defensible.
- **Two counterfactuals:** optimise dwell times on the predicted geometry, deliver on the reference geometry. That is what happens if the physicist trusts the automation. Also optimise on the reference and evaluate on the predicted, as the symmetric check. Use a simple inverse optimiser with non-negative least squares plus a uniform-time baseline normalised to D90.

Report, paired per patient, against the rater-2 dosimetric difference as the noise floor:

| Metric | Structure |
|---|---|
| ΔD90, ΔV100, ΔV150, ΔV200 | surrogate HR-CTV |
| ΔD2cc, ΔD0.1cc | bladder, rectum, sigmoid |
| 3D gamma 2%/2 mm and 3%/3 mm, 10% threshold | CTV + 1 cm |
| Fraction of patients exceeding a clinical action level | all |

Add a per-catheter sensitivity figure. Perturb one catheter at a time by its actual predicted error and plot ΔD90 against tip error. The knee of that curve gives a clinically derived tip tolerance, and that tolerance becomes the target for the audit protocol in item 2. Also introduce a dwell-time-weighted tip error, so catheters that carry most of the dose dominate the headline number.

**2. Uncertainty-guided clinician-in-the-loop audit protocol.** You already have every signal needed from the out-of-fold outputs:

- **Per-track risk score:** Stage 2B calibrated probability, tip-localizer predictive entropy, tip spread across the five inner-fold ensemble members and the four TTA views in millimetres, track length as a cohort z-score, and distance from the nearest template-hole trajectory from the .acsv markers. Fit a small logistic model or just rank by ensemble tip spread, which is usually the strongest predictor.
- **Evaluation:** AUROC for "tip error above tolerance" and for "false positive". Then an inspection-budget curve: tracks reviewed in risk order on the x-axis, residual P95 tip error and recall on the y-axis, assuming each inspected track is corrected. Compare against random order and length-only ordering.
- **Recall to 100% via the count rule:** the number of inserted catheters is known. When the pipeline finds fewer, rank the rejected Stage 2B candidates and report the rank at which each missed catheter appears. That is the number of clicks to full recall.
- **Conformal tip intervals:** split conformal calibration on inner folds gives a per-tip error bound with guaranteed coverage. Report empirical coverage at 90% and 95%. This is cheap and reviewers like it.
- **Time saved:** a two-physicist timed study on five patients in Slicer, manual reconstruction versus review-and-correct of the automated output, would make the OR-time claim real. The triage curves alone are a few hours. The timed study needs a human afternoon.

**3. Gold-standard-free estimate of true tip error.** On the 40 dual-traced scans you have three measurements per tip: rater 1, rater 2 and the out-of-fold prediction. Grubbs' three-cornered-hat estimator gives each source's intrinsic variance from the three pairwise difference variances without any gold standard. This directly answers the question "is the model better than the label noise?" rather than asserting it. Caveat in the text that the model was trained on the consensus, so independence is partial.

**4. Tip-error tail dissection.** A P95 of 27 mm against a median of 2.3 mm means a small cluster of gross failures. Decompose tip error into along-axis and lateral components, since only along-axis matters for dwells. Check for a systematic along-axis bias, which a cohort offset removes. Classify the gross failures into a taxonomy: void extending past the physical tip, tip merged with air or bone, adhesion splits, and obturator confusion. Show what fraction the count rule and the risk score from item 2 catch. This figure will be the most-read one in the paper.

**5. Zero-shot prostate transfer on Case 100.** Run the frozen pipeline and report F1, shaft and tip error against the 14 manual polylines. One case is anecdotal, but "zero-shot to an unseen organ and template" is a sentence reviewers remember. Also report Stage 0 alone on it to show the physics prior is organ-agnostic.

**6. Cross-sequence robustness within the same session.** If the 2D AX TSE T2 and AX T1 TSE POST stacks share the frame of reference with the SPACE volume, the 3D reference polylines are already ground truth for them after resampling. Run the pipeline on each stack and tabulate F1 and errors per sequence. That is a cross-sequence generalisation table for the cost of a resampling script.

**7. Learning curve and stress tests.** Retrain at 10, 20, 30 and all training patients to show data efficiency and the value of the self-supervised pretraining. Separately, a robustness sweep over added noise, bias field, through-plane downsampling and cropping gives degradation curves that deployment-minded readers want.

**8. Audit the DICOM folders for RTPLAN, RTSTRUCT and RTDOSE.** Clinical plans may sit beside the CT series. If they exist for even a subset of the 40 CT cases, item 1 upgrades from surrogate plans to real clinical plans, which is a different tier of evidence.

## Broader extensions

**Serial 2D monitoring as 4D insertion tracking.** Four cases with five timepoints each is enough for a short companion paper. Track tip position over time, insertion velocity, and shaft deflection from the straight template trajectory as a function of depth. The monotonic-advancement constraint, where the catheter at one timepoint must contain the catheter at the previous one, is a label-free consistency test. Validate against the per-timepoint manual traces.

**Paired CT as a third tip reference and a cross-modality test.** Register the post-implant CT to the intraoperative MRI by iterative closest point on the catheter bundle itself, not on anatomy. CT tips are sharp, so this reduces the tip label noise argument and feeds item 3 with a fourth source. Then retrain Stage 1 on CT with a bright-core variant of the Stage 0 prior while leaving Stage 2 untouched. Stage 2 unchanged across modalities is a strong generality claim for the graph transformer.

**Automatic channel-number assignment.** Mapping each track to a template hole from the .acsv markers is what the afterloader actually needs. Accuracy of that mapping is a larger practical win than another F1 point and nobody has reported it for MRI.

**Public benchmark release.** External brachytherapy data is essentially nonexistent, which is why no one can compare to you. Releasing the de-identified 54 volumes with 1,063 traces, the evaluation harness, and the ONNX models would make this the reference dataset. That needs IRB review but is the move that makes a paper a landmark. Public tubular datasets such as MSD hepatic vessels or airway CT benchmarks can show the Stage 2 tracker generalises, but verify licensing before relying on any of them.

**Multi-sequence fusion to shrink the tail.** The 2D TSE stacks have higher in-plane resolution than SPACE. A late fusion of tip estimates across sequences is the most plausible route to cutting the gross-failure tail, and the data is already on disk.

**Temporal validation split.** Train on earlier cases and test on later ones as an approximation of deployment drift. It costs one retrain and preempts the standard reviewer question.

## How I would reframe the paper

Lead the abstract with dosimetric equivalence to a second human rater, with geometry as the supporting evidence rather than the headline. Replace "outperforms inter-observer agreement" with a TOST equivalence test against the inter-observer margin, which is more defensible. Make the audit protocol the clinical-translation section, with the inspection-budget curve and the conformal bounds as its two figures. Keep every number with a patient-level bootstrap interval. The pipeline is already strong enough. What elevates it is showing what the errors cost in gray and what it takes to catch them.
