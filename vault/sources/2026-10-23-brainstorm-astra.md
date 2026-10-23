**The biggest opportunity is to connect reconstruction errors to treatment consequences and the effort needed to correct them.** Your shaft accuracy is already compelling. The more consequential unanswered questions are: Which errors change a plan? Can the system identify them? How quickly can a physicist produce a complete, verified reconstruction?

I would prioritize **dosimetric sensitivity, uncertainty-guided review, and frozen prostate transfer**, with serial insertion tracking as the strongest exploratory extension.

The proposals below use your reported results and a limited repository inspection; I have not independently verified the server datasets or run these experiments.

**Immediate high-ROI experiments on your server**

| Priority | Experiment | First useful deliverable | Effort and dependency |
|---|---|---|---|
| 1 | Explain the tip-error tail and missing catheters | Failure taxonomy; review-budget curves | Hours using saved OOF outputs |
| 2 | Frozen GYN → prostate transfer | Complete 14-catheter case report | Hours after coordinate/provenance checks |
| 3 | Dose sensitivity pilot | Paired dose differences on representative cases | Hours **if** plans, contours and source metadata exist |
| 4 | Serial TSE feasibility | Four patient timelines and an insertion animation | Initial inference in hours; annotation review longer |
| 5 | CT adjudication pilot | Resolve selected MRI/rater disagreements | Registration and expert review required |
| 6 | Controlled robustness experiments | Performance versus spacing, contrast, tilt and noise | Hours to a day, depending on inference count |

### 1. Turn the tip-error tail into a clinically meaningful result

The **2.27 mm median versus 27.27 mm P95 tip error** deserves its own analysis before another architecture improvement.

For every large error, distinguish:

- Wrong catheter identity or adhesion-related switching.
- Correct shaft, but truncated or overextended endpoint.
- Distal/proximal endpoint convention mismatch.
- Fragment stitching failure.
- Reference disagreement or uncertain tip visibility.
- Tip outside the imaged volume.
- Coordinate or voxel-spacing error.

Decompose the tip residual into **along-shaft and transverse components**. A large axial offset on the correct catheter and a transverse jump onto a neighboring catheter demand different interventions and have different dosimetric consequences.

Report separately for long interstitial and short obturator tracks:

- Median, P90/P95 and fraction exceeding 2, 5 and 10 mm.
- Patient-level worst error.
- Complete-catheter recall, including endpoint and identity requirements.
- Results on a common matched subset **and** detection failures outside that subset.

**Memorable figure:** a shaft-error/tip-error scatterplot colored by failure mechanism, with representative MRI panels. It explains why excellent median geometry can coexist with substantial review needs.

### 2. Run Case 100 as a genuinely frozen transfer experiment

This is your easiest compelling addition.

**Protocol**

1. Verify Case 100 was absent from supervised training, SSL, synthetic-background generation and threshold development.
2. Freeze preprocessing, crop logic, checkpoints and thresholds before viewing its evaluation results.
3. Verify VTK and MRI coordinates, transforms, catheter identities and endpoint conventions.
4. Run NeedleFinder and the retrained PMB 2019 baseline.
5. Evaluate all 14 catheters: TP/FP/FN, shaft and tip errors, complete trajectories, confidence and runtime.
6. Use CT for a separately identified adjudication analysis, not as an inference input.

Show all 14 trajectories, including failures. Report exact counts rather than leading with a percentage.

**Claim:** “Single-case, cross-organ zero-shot feasibility at the same institution.” Fourteen catheters are not fourteen independent patients, and this does not demonstrate external-site generalization.

A useful mechanistic addition is a **predefined physics-channel ablation**: does physics enrichment preserve performance under organ shift better than image-only inference? Any prostate-driven tuning becomes a separate adaptation experiment.

### 3. Build dosimetry around the data you actually have

**Trajectories alone cannot recover clinical dose or DVHs.** You also need:

- Source model and air-kerma strength.
- Dwell positions, dwell times and channel identities.
- Catheter-tip-to-source-center offsets and channel direction.
- Prescription and HR-CTV/OAR contours.
- Image/plan coordinate transforms.
- Other irradiating applicator channels, where present.

First inventory DICOM **RTPLAN, RTSTRUCT and RTDOSE**, TPS exports and Slicer scenes. NRRD image pairs do not establish that these planning data exist.

**If clinical plans are available**

Implement the source-specific TG-43 line-source calculation:

\[
D(\mathbf x)=\sum_j t_j S_K\Lambda
\frac{G_L(r_j,\theta_j)}{G_L(r_0,\theta_0)}
g_L(r_j)F(r_j,\theta_j).
\]

Derive source orientation from the local catheter tangent, and use the actual source’s consensus parameters. TG-43 estimates dose in a water-based reference model; call the result **reconstruction-induced dosimetric differences under TG-43**, rather than “exact dosimetric accuracy.” The relevant high-energy source guidance includes source-specific \(^{192}\mathrm{Ir}\) datasets and interpolation conventions. [AAPM/ESTRO HEBD report](https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.3703892)

Use two distinct experiments:

| Experiment | Hold fixed | Change | Question answered |
|---|---|---|---|
| **Fixed-plan reconstruction sensitivity** | Dwell times, source, anatomy, channel identity | Reconstructed source positions | How much does reconstruction alter calculated dose? |
| **Planning consequence** | Prescription, objectives, anatomy, optimizer settings | Reconstruction used for optimization | Does planning on the reconstruction produce an acceptable plan when evaluated on reference geometry? |

For fixed-plan sensitivity, map dwell positions by **physical channel index and arc length**, respecting the commissioned distal offset. Nearest-point projection can hide a longitudinal tip error. Tip offsets and source-path reconstruction are explicitly important in current HDR practice guidance. [AAPM MPPG 13.a, Part B](https://pmc.ncbi.nlm.nih.gov/articles/PMC12256571/)

For the planning-consequence experiment, optimize on each reconstruction, then transfer its dwell times by channel/dwell index to the reference geometry for evaluation. Otherwise, a plan can look satisfactory in its own incorrect coordinate model.

**Handle detection failures explicitly**

- Matched-channel analysis isolates localization error.
- Missing active channels need a declared omission or correction scenario.
- False-positive channels have no automatic, physically meaningful dose assignment.
- Channel swaps need a separate analysis.
- Do not irradiate every annotated short track merely because it appears in the reconstruction benchmark.

Present raw-output usability and clinician-corrected planning as separate endpoints.

**Primary dose outcomes**

- HR-CTV \(D_{90}\): Gy and relative difference.
- \(V_{100},V_{150},V_{200}\): specify structure and units; percentage-point differences for percentage volumes.
- Bladder, rectum, sigmoid and bowel \(D_{2\mathrm{cc}}\), where contoured.
- Patient-level constraint crossings.
- Paired DVHs, dose-difference maps and worst-case examples.

Use gamma as a secondary spatial agreement metric. Predefine global/local normalization, reference dose, low-dose cutoff, evaluation region, interpolation and grid spacing. Report 2%/2 mm and 3%/3 mm as sensitivity analyses; neither is automatically a validated clinical acceptance criterion for this application. Whole-volume gamma can conceal localized target/OAR errors.

Validate the implementation against source QA tables and TPS calculations, and perform dose-grid convergence checks. Keep this numerical validation separate from reconstruction comparisons.

**If plans or contours are missing**

You can still run a valuable **standardized virtual-plan sensitivity study**:

- Define identical dwell spacing and active lengths on reference catheters.
- Use predefined equal-time or synthetic dwell patterns.
- Perturb geometry using actual model errors.
- Measure dose-field changes and isodose-volume changes.

Without target/OAR contours, do not label these clinical DVHs or \(D_{90}\). Without actual dwell times, do not imply reconstruction of delivered treatment.

**Best first pilot:** select cases by predefined strata—typical performance, adhesion, large tip error, missed active catheter and high resolution—then extend to every patient with adequate planning data.

### 4. Evaluate uncertainty-guided review as a workflow

**The central limitation: reviewing existing tracks cannot recover a catheter that was never proposed.**

Build three review queues:

| Queue | Candidate signals | Human action |
|---|---|---|
| Accepted-track errors | Calibrated error probability, ensemble disagreement, competing paths | Accept, delete, split or retrace |
| Tip errors | Endpoint interval width, multimodality, ensemble disagreement | Confirm or relocate tip |
| Missing tracks | Rejected candidates, unexplained image evidence, template-channel discrepancies | Recover candidate or seed a new trajectory |

Template information must come from genuinely available clinical records or image analysis, not reference catheter counts.

Your existing [calibration script](/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun/calibration_and_reason_codes.py) provides a useful starting point, but its target is proximity to a reference shaft. That is not the same as “unique, complete, correctly identified catheter with tip error below 2 mm.”

Calibrate separate outcomes:

\[
P(\text{track requires correction}),\qquad
P(E_{\mathrm{tip}}>2\text{ mm}),\qquad
P(\text{rejected candidate recovers an uncovered catheter}).
\]

Entropy is a candidate predictor, not a calibrated error probability. Report endpoint uncertainty in millimetres and test whether it predicts actual errors.

Fit calibrators and review thresholds **inside each outer training partition using inner OOF predictions**. The existing script’s calibration across pooled outer-fold outputs should not simply be inherited as the definitive nested clinical workflow evaluation.

**Retrospective experiment, runnable now**

Simulate review budgets of 0, 1, 2, 3, 5, 10 and all actions per case. Compare:

- Random review.
- Confidence-only review.
- Tip-uncertainty review.
- Combined missing-track and geometry review.
- Oracle ordering as an upper bound.

An oracle replacement measures **potential benefit**, not actual click performance. For a realistic simulation, apply a specified correction operator—such as relocating a tip while retaining the shaft—rather than replacing the entire track with ground truth.

Plot:

- Recall and precision versus review budget.
- Tip P95 and fraction within 2 mm versus review budget.
- Percentage of patients with a completely acceptable reconstruction.
- Residual dosimetric error versus review budget, once available.

Define “below 2 mm” explicitly: median, P95, or every evaluated tip. These are very different targets.

**Then conduct a timed reader study**

Use counterbalanced manual, conventional automated-review and uncertainty-guided-review conditions, with independent final adjudication and safeguards against case-memory effects.

Measure full task time:

\[
T_{\mathrm{total}}=
T_{\mathrm{load}}+T_{\mathrm{infer}}+T_{\mathrm{inspect}}
+T_{\mathrm{correct}}+T_{\mathrm{verify/export}}.
\]

Report measured reconstruction-time savings. OR-time savings require evidence that this task lies on the procedure’s critical path.

Achieving 100% observed recall in a finite study does not establish guaranteed future recall; report patient-clustered uncertainty and patient-level failure rates.

### 5. Use the serial TSE scans for a focused temporal experiment

Start with **retrospective sequential reconstruction**, then add temporal association.

For each of Cases 68–71:

1. Verify timestamps, slice spacing, coverage and annotation completeness.
2. Apply the model trained without that patient—including their final SPACE scan and other modalities.
3. Reconstruct each timepoint independently.
4. Associate tracks using template entry, overlapping shaft geometry and motion constraints.
5. Compare independent inference with a causal tracker using only current and previous scans.

Measure:

- Newly inserted catheter detection.
- Identity switches and fragmentation.
- Tip advancement error.
- Lateral deflection and curvature changes.
- Motion of previously placed catheters.
- Processing latency after acquisition.

Register using stable anatomy/template information and quantify registration uncertainty. Otherwise, patient motion can masquerade as catheter deflection. Tissue-motion claims require independent tissue landmarks.

A 2D multislice acquisition is not necessarily an instantaneous 3D snapshot. Account for slice timing and gaps; interpolation does not restore missing through-plane information.

**Best deliverable:** an animation showing insertion order, stable catheter identities, advancing tips and uncertainty. With four patients, frame this as feasibility rather than validated live guidance.

**Broader extensions**

### 6. Treat paired CT as an adjudication resource first

The 40 CT pairs may be most valuable for investigating what MRI agreement actually measures.

Start with:

- Algorithm–human disagreements.
- Human–human tip disagreements.
- Adhesion and identity-switch cases.
- Apparently confident but incorrect predictions.

Have a blinded expert reconstruct selected CT channels, verify scan chronology and stylet/marker configuration, and align images using independent landmarks or the template. Report registration error and possible interscan catheter motion.

Do not use deformable alignment that forces the evaluated catheters into agreement. With 1.5 mm CT slices, CT endpoints also have partial-volume uncertainty.

This matters because the original PMB 2019 study already used CT to help resolve ambiguous MRI annotations; CT assistance is not inherently a novel independent reference. Document which current annotations had CT access. [Zaffino et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726393/)

Later, compare:

- MRI-only reconstruction.
- CT-only reconstruction.
- Paired-modality adjudication.
- CT-assisted supervision evaluated on unseen patients.

Keep all modalities and timepoints from a patient in the same split. These are additional observations, not additional independent patients.

### 7. Choose external datasets by the claim they test

| Resource | Appropriate experiment | Limitation |
|---|---|---|
| **Independent brachytherapy cohort** | Frozen scanner/site/applicator transfer | Highest relevance; requires an independent cohort |
| **DABI implanted-electrode localization data** | Transfer graph grouping and identity preservation to CT electrodes | Requires a CT-specific frontend and electrode-aware geometry |
| **TopCoW CTA/MRA** | Test connectivity and topology components | Branching vessels differ fundamentally from nonbranching catheters |
| **ASOCA coronary CT data** | Curved centerline extraction and continuity | Useful component benchmark, weak evidence for brachytherapy generalization |
| **Biopsy-needle imaging** | Endpoint localization under a different acquisition protocol | Confirm actual needle images and endpoint labels before selecting a dataset |

DABI provides public data supporting implanted ECoG/SEEG localization workflows. Verify usable image/reference pairs before committing to a benchmark. [DABI project](https://dabi.loni.usc.edu/projects/R01NS062092)

TopCoW has a maintained public training/external-test release. [Organizer dataset release](https://zenodo.org/records/15692630) ASOCA provides annotated coronary CT data. [Dataset publication](https://www.nature.com/articles/s41597-023-02016-2)

I would not make DBS/SEEG or vascular transfer a prerequisite for this PMB submission. They require different appearance models and often different topology assumptions. They fit a broader methodological follow-up.

Likewise, check historical AMIGO/GYN public collections for overlap before calling them external validation.

### 8. Three contributions with unusually high upside

**Dose-aware review prioritization.** Once dose sensitivity is available, prioritize expected consequence per unit review time:

\[
\text{priority}_i=
\frac{
P(\text{error}_i)\,
\mathbb E[\text{dosimetric consequence}\mid\text{error}_i]
}{
\mathbb E[\text{review time}_i]
}.
\]

Compare this with uncertainty-only ordering. Before a plan exists, use clearly labeled geometric/OAR-proximity proxies; after provisional planning, use actual dwell-weighted sensitivity.

**A benchmark of complete, verified reconstruction.** Publish patient-level success, channel identity, active-dwell coverage, endpoint accuracy and correction burden alongside F1. A high F1 can still leave almost every case requiring intervention; this endpoint reveals whether the workflow is usable.

**A reusable multimodal, multirater, longitudinal resource.** Subject to data-sharing permissions, release fixed patient splits, coordinate conventions, annotation provenance, uncertainty labels, evaluation code and representative temporal cases. A durable benchmark may have more impact than another model component.

For the paper itself, I would center the contribution on **physics-guided reconstruction with quantified treatment consequences and efficient human verification**. Keep prostate transfer and temporal tracking as focused demonstrations.

Finally, replace “outperforming humans” with the narrower supported claim unless a paired, independent analysis establishes superiority. The reported 0.47 versus 0.49 mm shaft medians and nearly identical mean tip errors do not by themselves establish better clinical accuracy—especially when the model learns from those raters. A carefully adjudicated result showing **which errors remain, what they cost, and how quickly they can be corrected** would be considerably stronger.