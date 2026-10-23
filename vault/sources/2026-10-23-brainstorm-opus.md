I didn't have file or shell tools in this session, so I couldn't read `vault/wiki/index.md` or inspect the `GYN Cases/` folders. Some of these ideas may already be covered in the vault, and anything that depends on what's on disk is marked **[verify]**.

---

## Two framing risks to fix first

1. **Drop "outperforming inter-observer agreement."** A 0.47 vs 0.49 mm median shaft error and a 6.23 vs 6.28 mm mean tip error are effectively the same. Reviewers will read "outperforming" as overclaiming. Recast it as **non-inferiority to a second human rater**, tested with TOST (two one-sided tests) using a pre-stated margin (say 0.5 mm shaft, 2 mm tip) and patient-clustered bootstrap confidence intervals. That is a stronger claim and one you can defend.
2. **The tip-error tail is the real clinical weakness.** The 95th-percentile tip error is 27 mm and the mean is almost 3× the median. Dwell positions are counted back from the tip, so this tail will drive any dosimetric result. Experiments 1.4 and 1.5 below are designed to turn that weakness into a contribution.

---

## Tier 1: High-ROI experiments you can run on the server now (hours to a few days)

### 1.1 Cross-sequence robustness with labels you already have (best value per hour)
The 2D `AX/SAG/COR TSE T2` and `AX T1 TSE POST` series on the 40 GYN patients come from the same session as the SPACE volume, in the same scanner frame.
- **Label transfer:** apply the SPACE reference polylines directly through the DICOM/NRRD world coordinates. Before trusting the labels, check that the patient didn't move between series. Compare acquisition timestamps **[verify]** and run a quick rigid registration on the template or obturator. Exclude any series with more than about 1 mm of residual misalignment.
- **Run:** (a) Stage 0 alone, (b) the full pipeline zero-shot on resampled anisotropic input (expect 3–5 mm slices), and (c) a multi-view fusion of AX+SAG+COR, either by super-resolution reconstruction or by fusing per-view probability maps.
- **Result:** a table of F1, shaft error and tip error by sequence, at no annotation cost. Stage 0 is physics-based, so you can argue it should transfer better than the learned stages. This experiment tests that argument.

### 1.2 An independent reference standard from the 40 paired CTs (strongest scientific addition)
Right now the reference comes from the same human tracers you compare against, which is partly circular. On post-implant CT, catheters are hyperdense and easy to segment, with an opposite contrast mechanism to MRI.
- Segment the CT catheters (threshold plus a centerline; semi-automatic is fine) and register CT to MRI. Use rigid or affine registration on the template, obturator and pelvic bone, and report the registration residual.
- Then compare the NeedleFinder, Rater 1, Rater 2 and PMB-2019 tracks against the CT reference. Tip error against CT matters most, because CT is the historical clinical standard for reconstruction.
- **Caveat:** the catheters may shift between MRI and CT (patient transfer, bladder filling). Report the MRI-to-CT displacement of the *human* tracks as an uncertainty floor, which is itself a useful clinical finding. **[verify]** that the catheters were still in place at CT time.

### 1.3 TG-43 dosimetric impact (what turns this into a *physics* paper)
- **Engine:** a TG-43 line-source model (TG-43U1) with a 2D anisotropy function F(r,θ) and radial dose function g_L(r), using consensus Ir-192 data for the source model actually used at BWH. Use a 1 mm grid, vectorized NumPy/numba. Running 54 patients × 4 reconstructions should take minutes.
- **The scenario that matters clinically:** the plan is optimized on the *predicted* catheters and delivered on the *true* ones. Steps:
  1. Generate dwell positions on each reconstruction: 5 mm step, counted back from the tip, over the active length.
  2. Run inverse optimization (a HIPO/IPSA-like linear or quadratic program) on the predicted geometry.
  3. Transfer the dwell times to the true geometry by catheter match and arc length from the tip.
  4. Recompute the dose.
  - Also report the simpler "same dwell times, shifted geometry" case.
- **Metrics:** HR-CTV ΔD90, ΔV100, ΔV150 and ΔV200; for organs at risk, ΔD2cc for bladder, rectum and sigmoid. In GYN, D2cc to the organs at risk often limits the plan more than V150/V200. Report gamma pass rates as secondary only. Brachytherapy dose gradients are so steep that 3%/3 mm gamma is too forgiving; if you report gamma, use 2%/1 mm local.
- **Main blocker:** contours. **[verify]** whether HR-CTV and organ-at-risk structures exist anywhere (Oncentra/BrachyVision exports, RTSTRUCT). If not:
  - Organs at risk: auto-segment with an MRI model such as TotalSegmentator-MRI and spot-check.
  - CTV: use a geometric surrogate, for example the implant-envelope volume (catheter convex hull shrunk by 5 mm). State clearly that it is a surrogate.
- **Missed catheters:** you need an explicit policy. Assume a human adds them, which connects directly to 1.4. Otherwise recall errors get counted as dose errors.
- **Uncertainty propagation:** sample catheter geometries from your predictive distribution (shaft σ, tip entropy → distribution of tip shifts) and propagate them to a ΔD90 distribution per patient. The resulting sentence, "geometric uncertainty of X mm corresponds to Y% dose uncertainty," is what PMB readers will remember.
- **Calibration check:** reproduce the known sensitivity from the literature (for example ΔD90 per mm of systematic tip shift, as in the GEC-ESTRO reconstruction-uncertainty work by Hellebust 2010 and Kirisits 2014) to validate your engine before applying it.

### 1.4 Uncertainty-guided clinician-in-the-loop audit (simulated, then timed)
- **Calibration first:** report reliability diagrams and ECE for the Stage 2B track probabilities, and check that tip entropy actually correlates with tip error. Without that, the triage story doesn't hold.
- **Simulated oracle:** rank items by uncertainty. Each "click" by the oracle does one of three things:
  - Deletes a false positive (1 click).
  - Corrects a tip (1 click, snapping to the reference tip).
  - Adds a missed catheter (2 clicks, entry and tip, then re-runs your Stage 0 seeded tracker from those clicks, so you test the real interactive tool rather than pasting in ground truth).
- **Curves:** recall, tip P95 and the fraction of tips under 2 mm against the number of clicks. Compare three orderings: uncertainty-ranked, random, and spatial sweep. The headline is "*N* clicks per patient to reach 99.5% recall and P95 tip error under 2 mm, versus about *M* clicks for full manual tracing."
- **Conformal tips:** split conformal prediction on the out-of-fold residuals, with the score normalized by entropy, gives a distribution-free guarantee such as "90% of tips lie within r(entropy) mm." Display it in Slicer as a tip-uncertainty segment along the catheter. This is new for brachytherapy and cheap to compute from out-of-fold outputs you already have.
- **Selective prediction:** risk–coverage curves and AURC for automatic acceptance.
- **Minutes saved:** don't model this; measure it (see 2.2).

### 1.5 Taxonomy of tip-tail failures
Take every tip error above 10 mm and label the cause. Possible categories: premature termination in the dark-core, overshoot into air or rectal gas, tip in an adhesion bundle, an ambiguous reference (raters disagree), or obturator confusion. Cross-tabulate against tip entropy. This tells you whether the tail is *flagged* (fine with a human in the loop) or *silent* (dangerous). Reviewers will ask, so answer it in advance.

### 1.6 Case 100 Prostate: frame it as a feasibility case, not a generalization claim
With n=1 and 14 catheters, run it zero-shot (Stage 0 alone, and the full pipeline) and report per-catheter shaft and tip error. Add the CT-based reference as in 1.2. Present it as one figure and one paragraph ("transfers without retraining to prostate HDR"). Don't claim cross-organ generalization from one patient.

### 1.7 Template-hole constraint (possible recall gain)
If the `.acsv` template markers give the template pose **[verify]**, then the entry points are constrained to the hole grid, and the clinician knows which holes were used. You could use that as a count-aware decoding prior, or as an oracle upper bound on recall ("given the hole list from the OR log"). This is essentially free information in the clinical workflow.

---

## Tier 2: Extensions over a few weeks (would make it a landmark paper)

### 2.1 4D insertion monitoring (Cases 68–71) as a clinical-science result
- Track catheters across `Series 9→17` with a temporal prior: propagate the catheters at time *t* as priors for time *t+1*, and detect newly inserted catheters as the residual.
- Measure **deflection against insertion depth** and **displacement of already-placed catheters when new ones are added** (tissue push). There is very little quantitative data on this in GYN interstitial brachytherapy, so it would be a new clinical finding, not just a benchmark.
- Expect thick-slice 2D TSE, where Stage 0 plus temporal priors will likely beat the learned stages. With only 4 cases, this may fit better as a short companion paper, or as a section with "real-time" in the title.

### 2.2 A small timed reader study (turns simulation into evidence)
Two or three physicists each do 10 cases under two conditions: (a) manual tracing from scratch and (b) NeedleFinder plus uncertainty triage in Slicer. Log time and clicks, and use a crossover design with a washout period. Even a small study turns "saves OR time" from a claim into data. While you're at it, add a **blinded visual Turing test**: experts rate anonymized auto and manual tracks as acceptable for planning or not.

### 2.3 Dosimetry on real clinical plans
If the delivered plans (dwell positions and times) can be exported **[verify]**, recompute the delivered plans on the auto geometry. That is the most clinically convincing version of 1.3.

### 2.4 Release the dataset as a benchmark
Release the 54 volumes, the dual-rater traces, the zero-needle negatives protocol and the evaluation code, after IRB approval and de-identification. Of everything here, this probably does the most for citations and long-term impact, because no public interstitial-brachytherapy MRI benchmark with dual raters is known to me.

### 2.5 Reporting rigor
Include a CLAIM or TRIPOD+AI checklist, patient-clustered confidence intervals throughout, and a pre-specified primary endpoint (for example, tip-error non-inferiority).

---

## Tier 3: Defer to separate papers
- **SEEG/DBS electrodes, biopsy needles, vascular benchmarks:** these are different physics (CT metal artifact versus MR susceptibility) and would dilute a focused PMB paper. They suit a "general thin-tubular-object" methods paper in MedIA, where Stage 0's physics prior is the selling point.
- **Prospective silent deployment in AMIGO:** run in the background during live cases and compare with the clinical reconstruction. This is the clinical-translation step that comes after this paper.

---

## Recommended PMB paper scope

| Include | Supplement / companion |
|---|---|
| 1.2 CT reference, 1.3 dosimetry (with uncertainty propagation), 1.4 triage plus conformal tips, 1.5 tail taxonomy, non-inferiority reframing | 1.1 cross-sequence table, 1.6 prostate case, 1.7 template prior |
| | 2.1 4D deflection → companion paper; 2.2 reader study → follow-up if not ready |

Together these would let the paper show that the geometric accuracy holds up against an independent reference, what the remaining errors mean in dose, and how much human review is needed to make the output clinically safe.

My suggested order is 1.1 and 1.5 today (cheapest, and they show where you stand), then 1.4 (reuses out-of-fold outputs), then 1.2 and 1.3 (need registration and contours).
