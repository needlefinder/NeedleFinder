---
title: "NeedleFinder 2026 Manuscript: Research & Paper-Architecture Review"
reviewer: Claude Fable 5.1
date: 2026-10-22
scope: vault/wiki (index, overview, 4 research/synthesis notes, 4 concept pages, 5 entity pages, 2 source digests), vault/sources/2026-10-21-54cases-benchmark.json (raw per-case JSON), PMB 2019 full text (PMC6726393), 2017 MedIA citation record, 2025-2026 SOTA abstracts, dataset searches (Zenodo/TCIA/SciData/SPL/NA-MIC)
---

# NeedleFinder 2026 Manuscript Review

## 0. Read this first: blocked on you, and what I could not confirm

### Decisions only you can make
1. **Which run is canonical.** The raw JSON (`sources/2026-10-21-54cases-benchmark.json`) and the 2026-10-22 wiki table disagree on every headline number (see Section 7, Table 7.1). The paper needs one frozen, versioned results file per pipeline variant, produced by one evaluation script, with the git hash of the code in the JSON header.
2. **The thesis.** The vault currently carries two incompatible theses: the 10-21 synthesis ("zero-shot physics beats the 3D U-Net without training data") and the 10-22 benchmark ("trained multi-head U-Net + graph networks are needed to reach 95% recall on all 1,063 needles"). Section 3 recommends how to fuse them into one story. You must choose.
3. **Whether to add a dosimetric study** (re-plan ~10 cases with auto vs manual catheters, report D90 / D2cc deltas). Without it, the paper is a PMB methods paper. With it, it is a Medical Physics clinical-impact paper and the "0.62 mm vs 2.0 mm" claim becomes meaningful. I recommend doing it (Section 4, Q3).
4. **Data release path.** No public MRI brachytherapy needle dataset exists (Section 6). Releasing even the 10 MICCAI 2013 cases plus the HT/CY annotations would make this the reference benchmark for the field. That requires IRB/de-identification decisions I cannot make.

### Could not confirm (mark these before citing)
- Exact numbers in Tanderup et al. 2008 (*Radiother Oncol* 89:156-163) on reconstruction-uncertainty dose impact. I only found secondary citations. I did confirm Schindel et al. 2013 (*J Contemp Brachytherapy*): a ±4.5 mm reconstruction uncertainty produced >10% dosimetric change, and HR-CTV D90 fell to 95.7% of prescription at a 2 mm shift.
- Final EMBRACE II intracavitary/interstitial (IC/IS) usage. Confirmed: EMBRACE I used IC/IS in 43% of patients; the EMBRACE II protocol targets IC/IS in at least 20% of patients per center; an Elekta brief reports preliminary EMBRACE II uptake above 70% (industry source, treat as soft).
- Whether the AMIGO "GYN Data Collection" that Egger et al. 2014 (arXiv 1404.6116) described as freely downloadable from spl.harvard.edu is still hosted. The SPL host returned a certificate error. If it is live, part of this cohort may already be public, which changes the data-availability story.
- PMB 2019 PMID. The vault lists two different PMIDs (31272095 on the splits page, 31238315 on the entity page). PubMed blocked the fetch. Verify one.
- The PMC-rendered author list spelled the second author "Penelle". This may be a fetch artifact or a real indexing error on PMC. If real, request a correction: it breaks citation continuity for the first author across the lineage.

---

## 1. Why this is still a live problem in 2026

The clinical and technical case is stronger than the current vault text makes it. The vault argues from the 2019 paper's limitations; the paper should argue from where the field moved.

### 1.1 Clinical drivers
- **Interstitial brachytherapy went from niche to standard of care.** EMBRACE I (Pötter et al., *Lancet Oncol* 2021) treated 43% of locally advanced cervical cancer patients with combined IC/IS applicators and showed that interstitial needles are what makes the dose-planning aims reachable in large tumors (RetroEMBRACE: benefit at HR-CTV ≥ 30 cm³). EMBRACE II (Pötter et al., *Clin Transl Radiat Oncol* 2018) raises the dose-planning aims so that IC/IS becomes necessary for a large fraction of patients and mandates at least 20% IC/IS per center. Every one of those needles must be reconstructed on the planning image.
- **Hybrid applicators multiplied the needle count.** Venezia, Utrecht IC/IS, and Vienna ring-plus-needle applicators carry 6-20+ parallel or oblique needles in a few cm. Free-hand perineal template implants (the BWH Syed-Neblett cohort) carry up to 28. The reconstruction problem scales with needle count and with needle proximity, not with image quality.
- **MR-only workflows remove the CT safety net.** GEC-ESTRO/ICRU 89 recommend MRI as the planning modality. Centers moving to MR-only planning (Shaaer et al., *JACMP* 2022, Sunnybrook) report 46 ± 10 min of expert time for manual catheter reconstruction on MRI versus about 11 min with their 2D U-Net assistance. On MRI, plastic catheters are signal voids that look like vessels, sutures, and air. On CT they are bright and trivial. MR-only brachytherapy is therefore bottlenecked by exactly this step, and the patient is under anesthesia during it.
- **Reconstruction uncertainty is now the dominant residual uncertainty.** With target delineation and dose calculation uncertainties shrinking under the EMBRACE protocols, applicator and needle reconstruction (1-3 mm systematic errors) is one of the largest remaining contributors (Nesvacil et al. 2013; Schindel et al. 2013: ±4.5 mm reconstruction error exceeds 10% dose change; 2 mm shift already drops D90 to 95.7%). Steep gradients near ¹⁹²Ir dwell positions (cite a primary source for the 6-12%/mm figure in the overview page; it is currently unsourced) make a 2.0 mm mean shaft error clinically relevant and a 0.6 mm error clinically negligible, at the shaft. The tip is a different story (Section 4, Q3).
- **Time at the table matters more in 2026, not less.** Same-day MR-guided insert-image-plan-treat workflows (AMIGO, MR-brachy suites, and now low-field 0.55 T systems) compress the planning window. A reconstruction step that runs in 0.6 s on the console CPU is a workflow enabler, not a convenience.

### 1.2 Technical drivers: why "just run nnU-Net" does not solve it
This is the argument reviewers from MedIA/TMI will test. Make it precise:
1. **Thin-structure sampling.** A 1.5-2.0 mm void is 3-6 voxels wide at 0.4-0.6 mm in-plane and 1-2 voxels at 1 mm³. Isotropic resampling (the 2019 choice, and nnU-Net's default target-spacing heuristic on anisotropic data) throws away the only dimension in which the needle is well resolved. The 2019 paper reported native 0.4 × 0.4 × 1.0 mm³ for SPACE, but the raw JSON shows the cohort actually spans 0.31-0.88 mm in-plane and 1.2-6.0 mm slices (31 cases at 1.6 mm, 10 at 3.0 mm, 5 at 6.0 mm). Native-resolution processing is not an optimization; it is the difference between seeing and not seeing the needle.
2. **Dice is permutation-invariant over instances.** A voxel loss cannot distinguish "two touching needles correctly labeled" from "two touching needles merged". The union of two tubes is one connected component; the skeleton of a union is not the union of skeletons. Every voxel-mask pipeline must therefore add an instance-separation heuristic after the network, and that heuristic is where the 2019 method's 2.0 ± 3.4 mm error (and the 3.4 mm standard deviation) comes from. The 2019 Discussion says so: "non-continuous segmentations lead to higher Hausdorff errors" and "the method classifies voxels rather than identifying individual catheters".
3. **Topology losses (clDice) fix continuity, not identity.** They reduce mid-shaft breaks but still output a semantic mask. Kostoulas et al. 2025 (SPIE, Amsterdam UMC, prostate MRI) is the clearest recent statement of this: they treat the post-processing of segmentation errors as the open problem and get 0 FP / 0 FN only after adapting the association step.
4. **Signal dropouts are a multi-slice problem.** Peripheral needles lose contrast on 30-50% of slices (your root-cause note). No per-voxel or per-slice decision can recover them; coherent integration along the trajectory (your Pass-3 sheared Radon, equivalently a matched filter along the known slope field) gains SNR as √N over N slices.
5. **Deployment constraints are real.** The 2019 ensemble needed a 24 GB GPU and 9 ± 2.5 min per case (2 ± 0.5 min for the single "light" model). Clinical brachy consoles and 3D Slicer workstations in the suite are CPU machines inside the hospital network. A 10.5 MB ONNX bundle that runs in 0.6 s on CPU with no PHI leaving the room is a different deployment class.

### 1.3 Where the field is now (use as the related-work spine)
| Year | Work | Modality / site | N | Key numbers | What it lacks |
|---|---|---|---|---|---|
| 2013 | Pernelle et al., MICCAI (LNCS 8151) | 3T MRI gyn | 10 pts / 101 cath | 92.1% with max HD < 2 mm, median 0.58 mm | 1 click per needle |
| 2017 | Mastmeyer, Pernelle et al., *Med Image Anal* 42:173-188 | 3T MRI gyn | 54 pts / 760 cath | mean HD 1.49 mm, 93% identification, 3 s/cath | tip-seeded; mechanical model needs tips |
| 2019 | Zaffino, Pernelle et al., *PMB* 64:165008 | 3T MRI gyn | 50 pts / 826 cath | Dice 0.60, Davg 2.0 ± 3.4 mm (ensemble) / 1.6 ± 2.8 mm (single), FP 6.7%, FN 1.5%, 9 min GPU | 1 mm³ resampling; no instances |
| 2020 | Dai et al., *Med Phys* 47:4115 | 1.5T MRI prostate | 20 pts | tip 0.37 ± 1.68 mm, 87% tips within 2 mm | prostate geometry, not touching gyn bundles |
| 2021 | Zhang et al., SPIE | TRUS prostate | 23 pts / 339 needles | 96% detected, shaft 0.29 mm, tip 0.44 mm | ultrasound |
| 2022 | Shaaer et al., *JACMP* 23(2) | MRI gyn, MR-only | 20 pts / 360 cath | 0.97 ± 0.66 mm vs planners, 11 min vs 46 min manual | 2D U-Net, semi-automatic |
| 2025 | Kostoulas et al., SPIE (arXiv 2507.18895) | MRI prostate | 261 test needles | median tip 1.07 mm, shaft 0.75 mm, 0 FP / 0 FN | prostate; post-processing focus |
| 2026 | Xiao et al. (arXiv 2607.11290) | CT interstitial | 38 pts / 203 fractions | shaft 0.33 mm, tip 0.90 mm, F1 0.89, 11.6 s | CT (bright metal), synthetic pretraining |
| 2026 | **This work** | 3T MRI gyn | 54 pts / 1,063 needles | shaft 0.55-0.62 mm, closest recall 95-98%, 0.61 s CPU | tip error tail; precision 75-84% |

Two observations for positioning. First, your shaft accuracy is at the level of the best CT method and better than every MRI method published. Second, your tip accuracy and precision are not yet, and both are reported by every competitor. The paper must report them with the same prominence or reviewers will assume the worst.

---

## 2. The narrative: one lineage, one thesis

### 2.1 Thesis statement (proposed)
> A brachytherapy implant is not a voxel class; it is a set of K smooth curves. Methods that estimate a voxel mask and recover curves afterwards inherit an instance-association error that no amount of voxel accuracy removes. We formulate reconstruction as structure estimation at three scales (voxel evidence, slice-to-slice association, bundle-level consistency), show that a native-resolution physics filter alone already surpasses the 2019 3D U-Net in shaft accuracy without training, and that learned evidence, association, and bundle reasoning are each necessary to reach 95% recall across all 1,063 interstitial and obturator needles in the full 54-patient cohort.

This fuses both vault theses: the zero-shot physics result becomes "Ablation 0" and the explainable backbone; the learned stages become the path to completeness.

### 2.2 The three-scale argument (make this Section 2 of the paper, with Figure 3)
| Scale | Mathematical object | Failure of a pure voxel method | Your component | Why it is the principled fix |
|---|---|---|---|---|
| Voxel | Evidence that a dark tube of radius r passes through x with tangent t | Resampling destroys r; CNN receptive fields confuse voids with vessels | Stage 0 multi-radius 6-diameter + half-ring filter (Rust/C++), then Stage 1 multi-head U-Net taking the physics channels | Ring-minus-center is a matched filter for a dark cylinder under Rician noise; feeding it to the network makes the network sample-efficient on 54 cases and makes its inputs interpretable; the half-ring term handles one-sided (obturator/rectum) boundaries that break symmetric filters |
| Slice-to-slice | Data association of 2D detections into curves with bounded curvature | Skeletonization of a mask union merges or swaps neighbours; per-slice linking cannot use second-order smoothness | Stage 2A `NeedleSliceMPNN`: triplet collinearity scoring on edges (h→u, u→v) + sub-voxel (dx, dy) regression | Triplet collinearity is the discrete second derivative of the curve; this is a learned multiple-hypothesis tracker and the direct generalization of the 2013 Bézier and 2017 mechanical-model priors |
| Bundle | Set of K curves with mutual exclusivity (one needle per template hole, no shared voxels, near-parallel field) | No voxel method sees the bundle; duplicates and shadow tracks survive | Stage 2B `DenseGraphTransformer`: edge-biased GAT + Set Transformer over candidate trajectories | Set attention is permutation-equivariant, so it can encode "this track is a duplicate of that one" without a hand-tuned NMS threshold; it supplies the +12% precision that NMS could not |
| Integration | Coherent line integral along a slope field | Per-slice thresholds miss low-contrast needles | Pass-3 sheared Radon / shift-and-add | SNR gain ∝ √N over N slices; matched filtering in the direction of the implant |

Keep this table in the paper. It is the "why principled" answer reviewers ask for, and it lets you name the 2013 Bézier curve, the 2017 mechanical model, and the 2019 U-Net as the three ancestors of the three scales.

### 2.3 Lineage framing (Introduction, two paragraphs)
Paragraph 1: the problem and clinical stakes (Section 1.1 above).
Paragraph 2: "Our group has studied this exact cohort for thirteen years." 2013: interactive, curve-fit, tip-seeded. 2017: mechanical model over 760 needles, 1.49 mm, 93% identification. 2019: first fully automatic, learned voxel evidence, but at 1 mm³ and with voxel output, 2.0 mm. 2026: the first method to combine learned evidence with learned association and bundle reasoning at native resolution, evaluated on the complete cohort including the short obturator needles that none of the prior papers scored. Say explicitly that this is the first paper of the lineage to score the obturator needles; that is a genuine first.

Name discipline: the manuscript should use stage names (Stage 0 physics filter, Stage 1 evidence network, Stage 2A association network, Stage 2B bundle network) and one system name. Five code names (`HybridNeedleNet`, `NeedleUNet3D`, `AxialTransUNet3D`, `NeedleSliceMPNN`, `DenseGraphTransformer`) in the abstract will read as a product sheet. Also rename "spatio-temporal": the slice axis is spatial. "Slice-sequential association graph" is accurate.

### 2.4 Proposed section skeleton
1. Introduction (1.5 pages): stakes, lineage, contributions (4 bullets: native-resolution physics evidence; multi-head anisotropic evidence network; association + bundle networks; full-cohort evaluation incl. obturators, inter-observer floor, and dosimetric impact).
2. Why voxel segmentation is insufficient (1 page + Figure 2): the three-scale argument.
3. Methods: 3.1 cohort and annotations (Table 1); 3.2 Stage 0; 3.3 Stage 1; 3.4 Stage 2A; 3.5 Stage 2B; 3.6 training protocol and splits; 3.7 evaluation protocol (define closest-match, 1-to-1 Hungarian, shaft, max-HD, tip, base, angle, and the 3.5 mm threshold; state the matching protocol used in 2019 and why you re-evaluate the 2019 model under yours).
4. Experiments: 4.1 ablation ladder (Table 3); 4.2 comparison with the 2019 network re-run on the same cases; 4.3 inter-observer floor; 4.4 obturator vs interstitial; 4.5 robustness to slice thickness and sequence; 4.6 tip error and dosimetric impact; 4.7 runtime and deployment.
5. Discussion: what each stage buys; what remains (tip tail, precision, single-center); comparison table with the field (Section 1.3).
6. Conclusion.

---

## 3. Reviewer questions and how to pre-empt them

Ordered by how likely each is to block acceptance. Each item lists the evidence you already have (from the raw JSON unless stated), the gap, and the fix.

### Q1. "Your comparison with Zaffino 2019 is not apples to apples."
- **Reality check.** The 2019 paper did not use a 25/3/22 split (the vault says it did; the full text does not). It describes a 70/30 split by catheter count, 4-fold cross-validation, and a 5-network majority-vote ensemble. FP (55/826) and FN (12/826) were counted over all 826 catheters. The `testing.txt` with 4 cases is one fold's artifact from the archived code, not "the published test set". The JSON labels 27 cases train, 6 val, 4 test, 17 "HeldOut-Extra", while the two vault pages give two different 37-case train lists. Three inconsistent lists.
- **Also:** the 2019 single "light" model scored 1.6 ± 2.8 mm with 13.8% FP. Compare against both.
- **Also:** 2019 Dice was computed on 1.5 mm-radius cylinders at 1 mm³; yours is at native resolution with a different rasterization. "Matches 0.60 Dice" is not a defensible claim; drop it.
- **Fix.** You have the 2019 TensorFlow checkpoint. Re-run it on all 54 cases, resample its output back to native space, and score it with the 2026 matching protocol. Report both methods on identical cases with identical metrics, paired per case (Wilcoxon signed-rank, patient-level bootstrap CIs). Keep the 4-case subset only as a footnote. The claim "3.2× lower shaft error" survives this; "98.61% on the held-out PMB test set" (71/72, Wilson 95% CI 92.5-99.8%) should not be a headline.

### Q2. "Where is the leakage?" (model selection, calibration, thresholds)
- **Exposure.** The physics filter radii, tissue floor, and template-cluster filter were designed on Case 050; the dedup threshold on Case 016. Both are inside the 54-case cohort. `HybridNeedleNet`, the U-Nets, the MPNN, and the DGT were trained 5-fold on the same 54 cases. The `minConfidence` operating points (25-60%) were chosen by looking at out-of-fold results, which makes the reported F1 optimistic.
- **Fix.** (a) State patient-level folds and confirm no patient appears in two cases (multiple fractions of one patient would be leakage; say so either way). (b) Select thresholds inside each training fold (nested), or fix them a priori and report the full curve (Figure 6a) rather than the best point. (c) Report per-fold variance. (d) Report the calibration cases separately, as the JSON already does (Case 050 excluded: 97.29% closest recall on 53 cases, essentially unchanged). (e) For the zero-shot ablation, the 2-case calibration is a strength: say "two cases, 37 needles, were used to set all constants; the remaining 52 cases are unseen".

### Q3. "0.62 mm shaft error is nice, but what about the tip, and what does any of this do to dose?"
This is the question that decides Medical Physics acceptance. The brief does not mention tip error; the raw JSON does.
- **Evidence (pooled long needles, JSON):** tip Δz mean 10.8 mm, median 3.0 mm, p95 46.8 mm; 3D tip error median 3.4 mm, p95 49 mm; 22 of 54 cases have mean tip Δz > 10 mm. Inter-observer CY vs LB: tip Δz mean 6.2 mm, median 3.0 mm, p95 21 mm. So at the median you match a second human (one slice), but your tail is twice as heavy. Max slice-wise HD < 2 mm holds for only 80.9% of needles (23 cases below 80%), against 92.1% in 2013 and 93% in 2017, because max-HD is dominated by the distal segment.
- **Why it matters.** Dwell positions are indexed from the catheter tip (or from the template with known free length). A 3 mm tip error shifts every dwell on that needle by 3 mm along its axis; a 10 mm error is a mis-planned needle.
- **Fix, three parts.** (1) Implement the sub-slice tip refinement already in your plan (1D intensity profile along the tangent, ± 2 slices) and the collinear fragment stitching; both target the tail. (2) Report tip error with the same prominence as shaft error, stratified by slice thickness, and show the CDF against the inter-observer CDF (Figure 8a). (3) Do the dosimetric study: for 10-15 cases, import auto-reconstructed catheters into the TPS (or a TG-43 recomputation in SlicerRT), keep the clinical dwell times, and report ΔD90 HR-CTV and ΔD2cc bladder/rectum/sigmoid versus the manual reconstruction, with and without one-click tip correction. Bland-Altman plot. This single experiment turns "0.62 vs 2.0 mm" into a clinical statement and is the difference between a methods paper and a practice-changing one.

### Q4. "Your precision is worse than 2019's 6.7% FP rate."
- **Evidence.** Physics-only run: adjusted precision 68.6% (raw 56%); U-Net + tube tracking: 62.9%; with the bundle network: 75.1% at max recall, 83.6% at 60% confidence. 2019: ~93% precision, but FPs were "spurious voxel clusters after DBSCAN", not catheter instances, so the two are not comparable either.
- **Hidden cause.** Your own 10-21 evaluation found that 4 of 5 "false positives" on Case 030 were real obturator needles absent from the CY annotation. The JSON's physics-only run detected 165 of 315 short needles while being tuned for long ones. Incomplete ground truth inflates FP counts.
- **Fix.** (a) Use the HT short-needle set so GT is complete, and report precision against the complete 1,063 (or 1,074; see Q8) set. (b) Have one clinician adjudicate every remaining FP on a 10-case subset into {real unannotated needle, vessel, artifact, duplicate}; report the breakdown (Figure 7 insert). (c) Show the FROC and precision-recall curves rather than one operating point. (d) State the clinical cost asymmetry honestly (a false track is one click to delete; a missed needle is a planning error) but do not use it to excuse precision; the bundle network is your answer and its +12% is a real contribution.

### Q5. "How good is the ground truth, and is the algorithm just reproducing the tool that made it?"
- **Evidence.** CY annotated the 759 long needles; HT the short ones; LB/AMe 543 needles on 41 cases. The JSON has 349 inter-observer pairs on 27 cases: CY vs LB mean shaft distance 0.85 mm (median 0.59), max-HD 1.43 mm. Your algorithm-to-CY distance (0.63 mm mean, 0.48 median) is below the human-to-human distance. That is the single strongest sentence available to you: "the automatic reconstruction is closer to the reference observer than a second expert is."
- **Caveat to pre-empt.** Base-entry inter-observer error is 3.7 mm (median 2.25), far larger than the algorithm's 1.0 mm, which means observers start their trajectories on different slices. Define "base" as the intersection with the template plane, not the first annotated point, or drop the base metric from the inter-observer comparison.
- **Circularity.** State exactly how CY, LB, AMe, and HT made their annotations: per-slice manual fiducials? Semi-automatic iGyne/NeedleFinder tracking with manual correction? If the latter, the 2013 tracker shaped the GT, and the reviewer will ask whether the 2026 shaft error is partly agreement with its own ancestor. The honest mitigation is the inter-observer comparison plus the CT-verified phantom (2013) or a new phantom with CT ground truth.

### Q6. "Short obturator vs long interstitial: are these one problem or two?"
- **Evidence.** The physics-only pipeline tuned for long needles finds 74% of all needles; the U-Net with the `p_obt` head reaches 95.2%. Obturator needles are 25-55 mm in FOV, lie in a <18 mm radius ring around the obturator axis, and terminate at the obturator tip.
- **Fix.** Report a 2 × 2 class confusion matrix (predicted class vs GT class) and per-class recall/precision/shaft/tip. State the classification rule (length and radial distance) and the learned `p_obt` head as the alternative. Discuss transfer to hybrid applicators (Venezia/Utrecht): the obturator geometry differs, so the `p_obt` head is applicator-specific and the paper should say so.

### Q7. "Single scanner, single center. Does it generalize?"
- **Evidence you have.** Two sequence families (2D T2 TSE at 0.42-0.88 mm / 1.6-6.0 mm; 3D SPACE at 0.47-0.59 mm / 1.3-1.6 mm), slice thickness from 1.2 to 6.0 mm. Shaft error is flat across slice thickness (0.53-0.90 mm per stratum), which is a robustness result worth a figure.
- **What is missing.** Any other scanner, field strength, or center. The 2019 paper listed "single scanner" as its first limitation; a reviewer will note that the lineage has never left the Verio.
- **Fix, cheapest first.** (a) Leave-one-sequence-family-out training (train on TSE, test on SPACE, and vice versa). (b) Synthetic degradation: downsample in-plane, add Rician noise, change bias field, re-score. (c) A phantom (the 2013 gel-wax phantom with 12 catheters exists) imaged on a different scanner, ideally 1.5 T and a low-field system. (d) Ask one external group for a blind test on 5-10 cases (Sunnybrook's MR-only cohort, Amsterdam UMC's prostate MRI needles, McGill's interstitial cohort); report their numbers as "external, unseen, run by them". (e) Release the ONNX models and the Slicer extension so that generalization can be tested by others, and say so in the limitations.

### Q8. "Is it actually fully automatic, and are the numbers reproducible?"
- **Template plane.** The JSON shows the automatic template-limit estimate off by more than 6 mm in 44 of 54 cases (Case 002: 139 mm). If the reported metrics used the manual `k_lim`, the pipeline needs one user input and the paper must say so. Fix: a learned template-plane head (the Stage 1 network can predict it), or evaluate with auto `k_lim` and report the cost.
- **Runtime.** Three different runtimes exist for the physics stage: 0.27 s (brief), 0.39-2.76 s (10-21 note), 7.15 s mean / 8.6 s median / 12.8 s max (JSON `t_total_ms`). State machine, thread count, which passes, and whether evaluation I/O was included. Report one table with hardware.
- **Counts.** 759 vs 760 long needles (one dropped; say why). 304 (brief, wiki) vs 315 (JSON) short needles. 72 vs 40 GT needles on the same 4 test cases (the 72 probably includes 32 short needles, in which case that row is not "long needles only").
- **Statistics.** Patient-level bootstrap CIs for every rate; paired tests between variants; per-case scatter. Currently none are reported.
- **Artifacts.** Code, ONNX weights, evaluation script with the matching protocol, and a results JSON per table row. PMB and Medical Physics both have data/code availability sections; MedIA/TMI reviewers expect a repository link.

### Q9 (bonus, Medical Physics). "What does the clinician actually do with it?"
A small usability measurement goes a long way: for 10 cases, time for a physicist to reach an approved reconstruction starting from (a) nothing, (b) the automatic result. Count edits (deleted tracks, added tracks, tip corrections). Shaaer et al. reported 11 vs 46 min; you can report seconds plus N clicks.

---

## 4. Figure and table blueprint

Eight figures, three tables, one graphical abstract. Every figure has one message stated in its first caption sentence.

**Figure 1. The problem and the lineage.** Three panels across. (a) Axial T2 slice from an obturator-dense case with 20+ needle voids, 2 cm scale bar. (b) 15 × 15 mm zoom of three adjacent voids with the native voxel grid overlaid, and the same crop resampled to 1 mm³ beside it; the voids visibly merge. (c) Horizontal timeline 2013 → 2017 → 2019 → 2026 with one icon per method (click, spring, grid, graph), the cohort size, and the headline shaft error under each. Message: the needle is resolved only at native resolution, and this is the fourth generation on one cohort.

**Figure 2. Why voxel masks are not enough.** Three columns, two rows. Columns: touching needles, crossing needles in thick slices, needle against the obturator. Top row: the 2019 U-Net mask after skeletonization (merged / broken / missed), rendered as polylines over the image. Bottom row: Stage 2 output on the same crop. Color by GT identity so swaps are visible. Message: instance errors, not voxel errors, drive the 2019 error tail.

**Figure 3. Three-scale architecture.** One wide diagram, left to right: MRI volume → Stage 0 physics channels (4 small image strips) → Stage 1 multi-head network (5 output strips: mask, peak, vx, vy, obturator) → slice-detection graph (nodes per slice, edges between slices, one triplet highlighted) → Stage 2A association output (curves) → Stage 2B bundle attention (M × M attention heatmap with a duplicate pair highlighted) → final curves in 3D. Under each block: tensor shape, parameter count, CPU time. Message: pixels → curves → bundles in 0.61 s.

**Figure 4. Physics evidence.** (a) Ring-minus-center and half-ring geometry on a 1.5 mm void, with the symmetric filter failing at the obturator boundary and the half-ring succeeding (two small response maps). (b) Pass-3 sheared Radon: a low-contrast needle invisible on three single slices, the shift-and-add accumulation image with a clear peak, and the SNR-vs-N curve. Message: a matched filter along the implant direction recovers what per-slice detection cannot.

**Figure 5. Association and bundle reasoning.** (a) Triplet collinearity: three consecutive detections with the discrete second-difference vector drawn; MPNN edge scores before and after four message-passing layers on a real crossing. (b) Set-transformer attention matrix for one case (M ≈ 25 candidates) with shadow-track pairs annotated and their confidence drop after the bundle stage. Message: duplicates are a set-level property and are removed by set attention.

**Figure 6. Quantitative results.** (a) Precision-recall / FROC curves for the ablation ladder (physics only; + evidence net; + association net; + bundle net) with the 2019 network re-run as a single point, all on the same 54 cases. (b) Cumulative distribution of shaft error for each variant, with the CY-vs-LB inter-observer CDF as a dashed line. (c) Per-case paired dot plot of mean shaft error, 2019 re-run vs 2026, connected per case, log y-axis. Message: every stage moves the curve, and the final method is inside the inter-observer band.

**Figure 7. Qualitative gallery in 3D Slicer.** Three columns: best, median, worst case by F1. Rows: 3D rendering of GT (gray) and automatic (colored by shaft error, 0-2 mm colormap); one axial slice with both overlaid; a 10-word failure annotation under the worst case. Include one 6.0 mm-slice case and one obturator-dense case. Insert: pie chart of adjudicated false positives (unannotated needle / vessel / artifact / duplicate). Message: failures are at tips and in unannotated needles, not along shafts.

**Figure 8. Tip accuracy and dosimetric impact.** (a) Histogram of tip Δz for automatic vs CY, overlaid with CY vs LB, in units of slices and mm, before and after sub-slice tip refinement. (b) Bland-Altman of HR-CTV D90 and OAR D2cc, auto vs manual reconstruction, with and without one-click tip correction, for the dosimetric subset. Message: shaft accuracy is sub-dosimetric; tip accuracy is where the remaining clinical risk sits, and here is how much.

**Graphical abstract.** Figure 3 compressed to one row with four numbers: 54 patients, 1,063 needles, 0.6 mm, 0.6 s.

**Table 1. Cohort and acquisition.** Per sequence family: N cases, in-plane range, slice thickness distribution (1.2 / 1.6 / 2.0 / 2.5 / 3.0 / 3.5 / 4.0 / 4.8 / 6.0 mm counts), needles per case, observers. Generated from the NRRD headers, not typed.

**Table 2. Lineage comparison on the same cohort.** 2013, 2017, 2019 (ensemble and light), 2019 re-run, 2026 zero-shot, 2026 full. Columns: automation, training data, resolution, recall, precision, shaft, tip, max-HD < 2 mm, runtime, hardware.

**Table 3. Ablation ladder.** One row per stage added, one evaluation protocol, fixed thresholds, all 54 cases out-of-fold: closest recall, Hungarian recall, precision, F1, shaft mean/median, tip median/p95, runtime. Replace the current nine-row table that mixes variants and thresholds.

---

## 5. Public datasets and an open-benchmark strategy

### 5.1 What exists (searched Zenodo, TCIA, Scientific Data, arXiv, NA-MIC, SPL)
There is **no public MRI dataset with annotated brachytherapy needles or catheters**. This is the finding, and it is an opportunity.
- **Already public from this cohort:** Case 050 as the RSNA 2016 / 3D Slicer tutorial dataset (20 long + 6 short needles). Confirm its license; if it is openly redistributable, say in the paper that one case with ground truth is public and that the extension reproduces the paper's numbers on it.
- **Possibly public:** the AMIGO "GYN Data Collection" that Egger et al. 2014 described as free to download from SPL (unverified; see Section 0).
- **CT, not public as far as stated:** Xiao et al. 2026 (38 patients, 203 fractions) and the HKBU 140-patient applicator CT set (*SN Computer Science*, 2026). Check their data-availability statements.
- **MRI prostate needles, not public as far as stated:** Kostoulas et al. 2025 (Amsterdam UMC), Dai et al. 2020 (Emory). Worth an email for an external test.
- **TRUS needles:** several groups (Emory, Gothenburg, Western), none public.
- **Needle-free pelvic MRI that is public:** PI-CAI, ProstateX, Prostate158 (prostate T2); a few uterine/cervical MRI sets. Useful as a **zero-needle specificity test**: run the pipeline on 100 needle-free pelvic volumes and report false tracks per volume. No competitor has reported this, and it directly addresses Q4 and Q7.

### 5.2 What to release with the paper
1. **A de-identified subset with annotations** as a TCIA collection (for example the 10 MICCAI 2013 cases, 101 long + ~60 short needles, both sequence families), under the existing IRB if it allows it. This becomes the reference benchmark for MRI needle reconstruction and the lineage's fourth paper will be cited for the data alone.
2. **The 2013 gel-wax phantom** (MRI + CT, 12 catheters) and, if feasible, a new phantom scanned at 3 T and 1.5 T. Phantoms carry no PHI and can be released immediately.
3. **A synthetic MRI needle simulator** (dark tubes with susceptibility blooming, Rician noise, anisotropic sampling) with analytic ground truth, in the spirit of Xiao et al. 2026's synthetic-to-real pretraining. Use it for the slice-thickness robustness figure and release it.
4. **Evaluation code** implementing closest-match and Hungarian matching, the 3.5 mm threshold, shaft/max-HD/tip/base/angle metrics, and bootstrap CIs, as a pip package. If the field adopts your metric code, every later paper compares to you on your terms.
5. **ONNX weights (10.5 MB) and the 3D Slicer extension** in the Extension Index.

### 5.3 Benchmark strategy
- Run the **2019 checkpoint** and a **native-resolution nnU-Net v2** (with and without clDice) as learned baselines, plus Frangi vesselness + Hough as the classical baseline. Four baselines on the same 54 cases makes the ablation ladder unassailable.
- Propose a **Grand-Challenge task** with a hidden test set from the released subset after the paper; a challenge is the normal route to becoming the reference dataset.

---

## 6. Venue and title

**Primary recommendation: Physics in Medicine & Biology**, with the dosimetric subset (Q3) included. Same journal as 2019, same cohort, direct lineage; a PMB editor will see the continuity. If the dosimetric study is strong, **Medical Physics** is equally appropriate and its readership is the one that deploys in the suite. **MedIA/TMI** only if you lead with the association and bundle networks as the contribution and accept the single-center limitation being weighed heavily; I would not split the work into a methods paper and a clinical paper, the combined story is stronger.

Title candidates (short, claim-bearing):
1. *From voxels to curves: physics-informed graph reconstruction of interstitial brachytherapy needles in intraoperative MRI at sub-millimetre accuracy in under one second*
2. *Sub-millimetre, GPU-free reconstruction of 1,063 brachytherapy needles in MRI: thirteen years and four methods on one cohort*
3. *Needle reconstruction is a set-of-curves problem: native-resolution physics, multi-head 3D networks, and bundle attention for MR-guided gynecologic brachytherapy*

---

## 7. Data-integrity punch list (fix before drafting)

### 7.1 Raw JSON (2026-10-21) vs wiki table (2026-10-22) on the physics pipeline, all 54 cases, long needles
| Quantity | Raw JSON | Wiki 10-22 / brief | Action |
|---|---|---|---|
| Short needles (GT) | 315 | 304 | Decide; explain the 11 |
| Closest recall | 739/759 = 97.36% | 741/759 = 97.63% | One frozen run |
| Hungarian recall | 729/759 = 96.05% | 726/759 = 95.65% | One frozen run |
| Adjusted precision | 68.6% | 83.54% | Different pipeline version; label it |
| F1 | 80.0% | 89.19% | Same |
| Mean shaft error | 0.63 mm (median 0.48) | 0.62 mm (median 0.54) | Same |
| Max slice-wise HD < 2 mm | 80.9% of needles | not reported | Report it |
| Tip Δz | mean 10.8, median 3.0, p95 46.8 mm | not reported | Report it |
| Runtime per case | mean 7.15 s, median 8.6 s | 0.27 s | State hardware and scope |
| PMB "test" subset GT count | 40 long needles (4 cases) | 72 needles | Clarify which classes |
| PMB train list | 27 cases | two different 37-case lists in two vault pages | Read `train.txt` on mbp-2 and fix all three |

### 7.2 Citation and provenance errors in the vault
- "JCMC 2017" does not exist. The 2017 paper is Mastmeyer, Pernelle, Ma, Barber, Kapur, *Medical Image Analysis* 42:173-188, DOI 10.1016/j.media.2017.06.011 (mean HD 1.49 mm, 93% identification, 3 s/catheter). The arXiv precursor is 1705.06712 ("Model-based catheter segmentation in MRI-images", 10 patients, 101 catheters, 1.40 mm mean). Verify whether a MICCAI 2015 workshop version exists before citing "MICCAI 2015".
- MICCAI 2013 is LNCS 8151 (Part III), pp. 380-387; the splits page says 8150.
- PMB 2019: two different PMIDs in the vault; the entity page's author list is garbled ("Hesham Elhalawani,azawa"); the real list is Zaffino, Pernelle, Mastmeyer, Mehrtash, Zhang, Kikinis, Kapur, Spadea. The 2019 split is 70/30 by catheter with 4-fold CV and a 5-model ensemble, not 25/3/22. Inference was 9 ± 2.5 min (ensemble) or 2 ± 0.5 min (single model) on a 24 GB P6000, not "1-2 min".
- The 2019 paper states native 0.4 × 0.4 × 1.0 mm³; the JSON shows 0.31-0.88 mm and 1.2-6.0 mm. Table 1 must come from headers.
- Case 050 geometry is described three different ways (320×320×128 at 0.586/1.3; 320×320×78 at 0.469/2.0; 640×416×112 at 0.469/1.6, the last identical to the Case 030 row). Case 038 is 768×768×64 on one page and 512×512×92 on another. Regenerate every dimension from the files.
- The SOTA note says the 2019 method had "92.6% recall on a 20-case test set"; the paper reports FN 1.5% over 826 catheters. Remove the 92.6% figure unless its source is found.
- The 2019 "~1-2 min" and the vault's "GPU + skeletonization" description should be replaced with the paper's actual post-processing: majority vote, DBSCAN, island removal < 50 voxels, and no instance extraction (left for future work).

---

## 8. Twelve-line summary of what to do next
1. Freeze one evaluation script and one results JSON per table row, with code hash and hardware in the header.
2. Resolve the three inconsistent PMB train lists from the files on mbp-2.
3. Re-run the 2019 checkpoint on all 54 cases under the 2026 protocol.
4. Train a native-resolution nnU-Net v2 baseline (± clDice).
5. Complete the GT with the HT short needles; adjudicate residual FPs on 10 cases.
6. Implement tip refinement and fragment stitching; re-score the tip tail.
7. Run the dosimetric subset (10-15 cases, D90 / D2cc, with and without one-click tip fix).
8. Add bootstrap CIs and paired tests everywhere; select thresholds inside folds.
9. Evaluate with automatic template plane, or add a template-plane head.
10. Zero-needle specificity test on public pelvic MRI; leave-one-sequence-out; phantom on a second scanner.
11. Prepare the data release (10-case subset + phantom + simulator + metric package + ONNX + Slicer extension).
12. Write the paper around the three-scale thesis with Figures 1-8 and Tables 1-3 as specified.

---

## Appendix: sources consulted
- Zaffino, Pernelle et al., *PMB* 2019, full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC6726393/
- Mastmeyer, Pernelle et al., *Med Image Anal* 2017 record: https://research.uni-luebeck.de/en/publications/accurate-model-based-segmentation-of-gynecologic-brachytherapy-ca/ and arXiv precursor https://arxiv.org/abs/1705.06712
- Pernelle et al., MICCAI 2013 record: https://nmr.mgh.harvard.edu/publications/journal_articles/5892
- Shaaer et al., *JACMP* 2022: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8833281/
- Dai et al., *Med Phys* 2020: https://pmc.ncbi.nlm.nih.gov/articles/PMC7708403
- Kostoulas et al. 2025 (SPIE / arXiv 2507.18895): https://arxiv.org/abs/2507.18895
- Xiao et al. 2026 (arXiv 2607.11290): https://arxiv.org/abs/2607.11290
- Xiao et al. 2025/2026 tip-handle matching (arXiv 2509.17931): https://arxiv.org/abs/2509.17931
- Zhang et al., SPIE 2021 TRUS needles: https://doi.org/10.1117/12.2580802
- Schindel et al., *J Contemp Brachytherapy* 2013: https://www.termedia.pl/Original-paper-Dosimetric-impacts-of-applicator-displacements-and-applicator-reconstruction-uncertainties-on-3D-image-guided-brachytherapy-for-cervical-cancer,54,21922,1,0.html
- Nesvacil et al. 2013: https://pmc.ncbi.nlm.nih.gov/articles/PMC3675683
- EMBRACE II IC/IS target (Smolic et al. 2019): https://pmc.ncbi.nlm.nih.gov/articles/PMC7807665
- Elekta EMBRACE science brief (industry): https://www.Elekta.com/medical-affairs/bibliographies/LLFBX220222_EMBRACE_ScncBrf_A4_f3.pdf
- Review of AI in brachytherapy (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12148788/ and https://arxiv.org/abs/2409.16543
- Review of open source / open data in brachytherapy DL: https://arxiv.org/abs/2205.07516
- Egger et al. 2014 (AMIGO data statement): https://arxiv.org/abs/1404.6116
- NA-MIC 2013 Gyne Brachy project week: https://www.na-mic.org/wiki/2013_Winter_Project_Week_Gyne_Brachy
