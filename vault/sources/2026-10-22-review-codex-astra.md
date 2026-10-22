**NeedleFinder 2026: research, evidence, and manuscript architecture review**

Prepared 22 October 2026. This review reads the six requested vault documents, checks selected training, evaluation, and deployment code, inspects the archived 54-case benchmark JSON, and consults primary publications and dataset records. It does not rerun training or independently reproduce the latest benchmark. Numerical results from the 22 October note are therefore identified as reported results, with verification requirements below.

**1. Recommended scientific position**

The strongest paper is about **recovering individually identifiable, geometrically accurate catheter trajectories from ambiguous MRI**, with computation fast enough to support interactive verification. Its central contribution should be a representation and evaluation framework connecting local image evidence, continuity along each catheter, and consistency among catheters. The five explored algorithm families are experiments that test this framework; they should not become five competing stories in the manuscript.

The longitudinal BWH cohort is unusually valuable: it allows the same clinical problem to be revisited across changes in representation, automation, and computing resources. It is also unusually vulnerable to accumulated development exposure. Reusing it is scientifically legitimate, but historical familiarity, annotation changes, and model selection must be explicit. A 2026 analysis of an archival cohort is not a contemporary clinical deployment study.

The current evidence supports a promising hypothesis: geometric priors provide accurate long-shaft proposals, learned image context broadens coverage to difficult short and boundary-adjacent catheters, and relational reasoning reduces redundant candidates. It does **not yet establish** that every component is necessary, that the complete system is training-free, that a transformer is essential, or that the reported historical error ratio is a paired improvement.

Recommended title: **“From voxel evidence to catheter trajectories: geometry-guided reconstruction in MRI-guided gynecologic brachytherapy.”** Add “CPU deployment in 3D Slicer” to the subtitle or abstract once the timing and deployment configuration are reconciled. Reserve “submillimeter” for the particular shaft metric actually measured, rather than implying submillimeter tip, whole-catheter, or source-dwell accuracy.

**2. Evidence that must be reconciled before the abstract is written**

The issues below are consequential because they change what the reported numbers mean. They are repairable through provenance and evaluation work; they are not reasons to discard the research direction.

**2.1 The published 2019 comparator differs materially from the vault summary**

The primary PMB article describes 50 patients/826 catheters annotated by HZ, four-fold evaluation with unique testing patients, and five-model ensembles. Its 2.0 ± 3.4 mm result is a symmetric distance between nonzero segmentation voxels, not the current shared-z shaft-XY metric. Ensemble inference was 9 ± 2.5 minutes; the light model took 2 ± 0.5 minutes and had a different accuracy/false-positive profile. Post-processing used DBSCAN and small-island removal; individual reconstruction/channel assignment was future work, including combination with model-based tracking. The implementation used Theano/Lasagne, so the recovered TensorFlow project is not automatically the published implementation. [Zaffino et al., PMB 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726393/)

Consequently, neither the recovered four-case `testing.txt` nor the vault's 25/3/22 partition is established as the publication's evaluation protocol. Label the former **“recovered legacy-code test subset”** until provenance is demonstrated. The numerical quotient 2.0/0.62 ≈ 3.23 is not evidence of a 3.2-fold localization improvement across different metrics and annotations. Keep historical results in a separate context table until both methods are re-evaluated identically.

Also, the historical FP percentage uses reference count as denominator; it cannot simply be subtracted from 100% to obtain precision. Do not publish the vault's derived historical precision/F1 as measured values. [Zaffino et al., results and evaluation definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726393/)

**2.2 Local implementation and artifact audit**

| Finding | Evidence inspected | Consequence and required action |
|---|---|---|
| Production training can overwrite OOF predictions | [run_edelweiss_training.py:234](/home/guillaume/github/needlefinder/run_edelweiss_training.py:234) saves every validation prediction to `oof_case_XX.npz`; [line 266](/home/guillaume/github/needlefinder/run_edelweiss_training.py:266) trains fold 99 on all 54 cases while passing the first ten as validation indices. | That execution path writes in-sample predictions over OOF filenames for indices 0–9. Later fold runs could replace some files again. The current code proves the collision risk, not the provenance of every current cache. Regenerate predictions in immutable fold-specific directories and audit all downstream results. |
| Reported validation folds also select checkpoints | [run_edelweiss_training.py:202](/home/guillaume/github/needlefinder/run_edelweiss_training.py:202) evaluates validation cases repeatedly and selects the checkpoint with `Dice + 0.5 × core recall`. | These are development-validation predictions, rather than a fully untouched outer test. Separate inner checkpoint selection from outer evaluation. |
| “Stratified folds” are index interleaving | [run_edelweiss_training.py:295](/home/guillaume/github/needlefinder/run_edelweiss_training.py:295) uses `indices[i::5]`. | Verify patient independence and actual balance in sequence, spacing, short-needle prevalence, and implant density. Do not describe interleaving as statistical stratification. |
| Template-plane information enters the inputs | [train_and_eval_unet3d.py:167](/home/guillaume/github/needlefinder/train_and_eval_unet3d.py:167) crops from `k_lim_gt`; [prepare_and_inspect_dataset.py:233](/home/guillaume/github/needlefinder/prepare_and_inspect_dataset.py:233) can derive it from GT catheter bases; evaluation passes it into tracking. | A clinician-specified plane is a legitimate assisted condition if declared. A plane derived from held-out catheter labels is privileged information. Evaluate an image-only plane estimator separately from manual-plane assistance. |
| The inspected network has six input channels | [train_and_eval_unet3d.py:74](/home/guillaume/github/needlefinder/train_and_eval_unet3d.py:74) specifies MRI, radial response, ring contrast, darkness ratio, relative z, and radial position. | The four-channel `R_asym` narrative does not describe this version. Freeze a version and document the actual tensor; ablate positional priors separately from image-derived physics features. |
| Deployment conditionally subsamples its input | [NeuralNeedleNet.py:84](/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder/NeedleFinderLib/NeuralNeedleNet.py:84) uses 2× XY striding for large ROIs and 2× slice striding for certain deep volumes, then interpolates predictions back. | “Entire pipeline at native resolution” is too broad for this path. Distinguish native-grid filtering/refinement from reduced-grid neural inference; measure the accuracy/speed trade-off. Native-size output is not proof of native-resolution inference. |
| Matching thresholds differ | [evaluate_nn_pipeline.py:437](/home/guillaume/github/needlefinder/evaluate_nn_pipeline.py:437) uses 5.0 mm; the latest long-needle table specifies 3.5 mm. | Identify the evaluator that produced each table and freeze tolerance, coverage requirements, and assignment rules. |
| Shaft error excludes nonoverlapping ends | [evaluate_nn_pipeline.py:67](/home/guillaume/github/needlefinder/evaluate_nn_pipeline.py:67) averages XY discrepancies at 40 shared-z samples, allowing overlap just greater than 1 mm. | A short correct fragment can have excellent shaft error despite incomplete reconstruction. Add longitudinal coverage, full-curve distances, and 3D tip/base errors. The variables named `max_hd_xy` and `p95_hd_xy` are corresponding-z discrepancies, not automatically conventional Hausdorff metrics. |
| Model artifact size differs | The inspected `nn_models/needle_unet3d.onnx` is 13,223,595 bytes, rather than the documented 8.95 MB. | Record model hash, architecture, parameter count, precision, and exact artifact size. These may be different model versions; the number alone cannot resolve which produced the benchmark. |
| The cited raw benchmark is an older experiment | [21 October JSON](/home/guillaume/github/needlefinder/vault/sources/2026-10-21-54cases-benchmark.json) contains 315 short needles, 349 interobserver pairs, and approximately 7.15 s average total time. The 22 October note reports different counts and much faster timing. | Do not treat the older JSON as raw support for the newer table. Export the exact latest per-case predictions, metrics, counts, and stage timings. |
| Scanner-spacing descriptions are incomplete | The archived JSON spans 0.3125–0.875 mm in-plane and 1.2–6.0 mm through-plane, wider than the brief's quoted range. | Rebuild the acquisition table directly from the image manifest and verify orientations/headers. Explain any exclusions or differing datasets. |

Additional provenance questions: why 760 historical CY catheters become 759; how 315 short annotations become 304; and how the historical 543 repeat annotations relate to the 349 pairs currently evaluated. The preparation code geometrically deduplicates CY/HT annotations at 2.4 mm. This deserves manual review in densely packed bundles: proximity alone is not proof that two labels represent the same catheter. Preserve observer identities and channel IDs where available.

The archived benchmark gives an especially useful warning: its pooled shaft-XY mean is approximately 0.630 mm, while full-curve 3D mean distance is approximately 2.12 mm and mean 3D tip error is approximately 11.38 mm. These are **older-version results**, not estimates for the new unified model. They nevertheless demonstrate, using this project's own data, why shaft accuracy cannot stand in for full catheter reconstruction.

**2.3 What the latest reported results actually suggest**

The following is a faithful selection from the [22 October benchmark note](/home/guillaume/github/needlefinder/vault/wiki/research/2026-10-22-full-54case-unet-transformer-graphnet-benchmark.md), pending the audit above. All rows use the full 1,063-needle reference set except the explicitly labeled long-only row.

| Configuration | Closest recall | One-to-one recall | Precision | F1 | Mean shaft error | Reported CPU time |
|---|---:|---:|---:|---:|---:|---:|
| Physics + learned MLP, long-only, 759 references | 97.63% | 95.65% | 83.54% | 89.19% | 0.62 mm | 0.27 s |
| Physics + learned MLP, all references | 74.13% | 72.44% | 88.61% | 79.71% | 0.62 mm | 0.27 s |
| U-Net + tangent tracking | 95.20% | 93.32% | 62.90% | 75.15% | 0.66 mm | 0.52 s |
| Slice MPNN | 91.82% | 88.90% | 74.41% | 81.01% | 0.55 mm | 0.42 s |
| Unified, score threshold 25% | 95.20% | 93.70% | 75.11% | 83.38% | 0.65 mm | 0.61 s |
| Unified, score threshold 60% | 90.22% | 87.86% | 83.62% | 85.69% | 0.62 mm | 0.61 s |

The most informative within-table result is the **12.21 percentage-point precision increase** from U-Net tracking to the unified system at equal closest recall. It is not a 12.2% relative increase; strict recall also changes slightly. Because more than the bundle scorer changes between these rows, isolate its causal contribution with the identical candidate pool and an MLP/NMS comparator.

The 0.55 mm MPNN result is a localization–coverage trade-off. It is not the accuracy of the unified system, and the easiest matched catheters may differ across methods. Compare localization on common successfully matched references as well as each method's full matched set. Misses must remain visible in the detection endpoint.

The U-Net/transformer ensemble raises closest recall from 1,012 to 1,014: only two additional references, or about 0.19 percentage points, with Stage-1 CPU inference increasing from 0.21 to 0.47 s. The transformer alone does not improve every metric. Present this as evidence about complementary errors and computational trade-offs; do not force a transformer-superiority story.

At the high-recall unified operating point, the rounded precision and TP count imply roughly 330 false positives and 67 missed references across 54 cases. At 60%, they imply roughly 183 false positives and 129 misses. These are approximate deductions, not recovered counts. They explain why high F1 does not establish a planning-ready result: the clinically important outcome is the work and risk remaining after automatic reconstruction.

**3. Why this problem remains important in 2026**

**Clinical relevance follows from the workflow.** MRI-guided adaptive brachytherapy has substantial clinical evidence: EMBRACE-I reported 92% actuarial five-year local control, and EMBRACE-II explicitly incorporates combined intracavitary/interstitial techniques and dose-volume objectives. This motivates reliable reconstruction as an enabling task; it does not imply that improving a reconstruction algorithm independently improves local control. [EMBRACE-I publication record and abstract](https://researchinformation.umcutrecht.nl/en/publications/mri-guided-adaptive-brachytherapy-in-locally-advanced-cervical-ca/), [EMBRACE-II protocol](https://www.embracestudy.dk/UserUpload/PublicDocuments/EMBRACE%20II%20Protocol.pdf)

An MR-only workflow needs both anatomy and device geometry in the same coordinate system. MRI makes target/OAR assessment possible, but catheter voids can remain difficult to distinguish. An additional CT introduces another acquisition and image-registration step; avoiding it is a meaningful workflow objective. Shaaer et al. directly studied this obstacle and evaluated a semiautomatic MR-only reconstruction workflow. [Shaaer et al., JACMP 2022](https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.13494)

Modern hybrid applicators make catheter identity, length, and tip placement relevant across a wider range of geometry. However, the Syed-Neblett/vaginal-obturator cohort does not by itself validate a Venezia, Utrecht, or Vienna workflow. The recent Venezia study demonstrates both the continuing need and the need for a contemporary comparator. [Kim et al., Brachytherapy 2026, online 2025](https://pubmed.ncbi.nlm.nih.gov/41203438/)

**The technical target is more demanding than a binary mask.** The desired output is a collection of individually reviewable source paths with endpoints, confidence, and eventual channel correspondence. A semantic foreground union does not preserve which points belong to which catheter. Two different assignments through a crossing can produce the same voxel mask. Good overlap can therefore coexist with a clinically incorrect path identity. This is a representational argument; it is not a theorem that U-Nets cannot solve the task when supplied with appropriate instance outputs and decoding.

Thin signal voids intensify the problem. Their apparent diameter and center depend on sampling, sequence, orientation, susceptibility, and nearby air or plastic. A nominal 1.5 mm catheter may occupy only a few in-plane pixels and have sparse through-plane observations. Interpolation and thresholding can alter centroids, separate weak segments, or fuse neighbors. Native-grid processing preserves observations; it does not remove MR distortion or prove that the center of a susceptibility void coincides with the physical source path.

**Low CPU latency matters most when it reduces interaction cost.** It can support rapid reprocessing after a correction, deployment on existing planning workstations, and review directly in 3D Slicer. The appropriate claim is “rapid reconstruction after image acquisition.” A subsecond compute kernel does not establish live MR guidance, subsecond import/rendering, or a reduction in anesthesia time. Measure time to an approved reconstruction, including review and edits, before making those workflow claims.

**Clinical importance is not proportional to mean shaft error alone.** A missing active channel, a tip-length error, or an identity swap can matter more than a modest improvement in transverse localization. The paper should ask whether better geometry improves the fidelity and efficiency of planning, rather than assume every 0.1 mm reduction has the same value.

**4. Paper narrative and mathematical architecture**

**4.1 Use the historical sequence to expose the changing inference problem**

| Period | What the clinician supplied / algorithm learned | Scientific role in the new paper |
|---|---|---|
| Pernelle et al., MICCAI 2013 | A tip seed supplied catheter existence and initialization; image/model tracking recovered a path. | Establish the original insight that catheter geometry is valuable. The published abstract reports 93/101 correctly segmented catheters and 0.3 s per catheter. [Primary record](https://pubmed.ncbi.nlm.nih.gov/24505784/) |
| Mastmeyer, Pernelle et al., 2015/2017 | Mechanically informed trajectory models and collection-level reasoning expanded the evaluation. | Establish continuity of geometric and multi-catheter reasoning. Use the exact journal title, *Accurate Model-based Segmentation of Gynecologic Brachytherapy Catheter Collections in MRI-images*. [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5654713/) |
| Zaffino, Pernelle et al., 2019 | Learned image evidence enabled automatic foreground discovery. | Establish the automation advance and the unresolved transition from foreground to individual paths; see the corrected comparison in Section 2.1. |
| NeedleFinder 2026 | Image evidence, trajectory continuity, and bundle relations are represented at their appropriate scales. | Test whether this decomposition improves completeness, identity, localization, and correction burden under one reproducible evaluation. |

The manuscript should credit the lineage in one introduction paragraph and one compact visual. A long historical tour would distract from the hypothesis. Avoid repeating the brief's exact Hough/Rayleigh/Bézier attribution without checking the original Methods: the supplied historical-splits note instead describes conic ray search and fifth-degree Bézier fitting. The vault itself is inconsistent on this point.

Suggested transition sentence: **“Earlier model-based methods supplied trajectory structure but required initialization; learned segmentation supplied automatic image evidence. We investigate whether explicit reconstruction at the point, path, and bundle levels can unite these advantages.”**

Do not state that all 3D CNNs require 1 mm isotropic resampling. nnU-Net adapts preprocessing and network configuration to the dataset and is explicitly a semantic-segmentation framework. It is an appropriate modern baseline, including its residual-encoder presets. [Official nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet)

**4.2 Organize the method around three inference scales**

**Local evidence: where might a catheter be?** Radial and asymmetric contrast operators encode a dark-tube hypothesis in physical units. A symmetric response can fail when one side of a catheter borders a dark obturator or air; directional/half-ring evidence relaxes that assumption. Learned context distinguishes such ambiguous boundaries from ordinary tissue structures. Describe these as image-formation-inspired geometric priors, not a complete MR forward model or a physics-informed neural network in the PDE sense.

The multihead network should predict occupancy, centerline likelihood, direction, and catheter subtype. Occupancy supports localization in context; a narrow peak field supports accurate centering; slopes guide continuation; subtype evidence prevents long-needle priors from excluding short channels. If the final model has positional channels, show them explicitly and test whether they overfit this template geometry.

**Path inference: which observations belong together?** A directed graph connects plausible detections across slices, including bounded skip edges for missing observations. Triplet messages distinguish a locally plausible link from a consistent continuation. Let physical node positions be \(\mathbf r_h,\mathbf r_u,\mathbf r_v\), and

\[
\mathbf s_{hu}=\frac{\mathbf r_{u,xy}-\mathbf r_{h,xy}}{z_u-z_h},\qquad
\Delta\mathbf s_{huv}=\mathbf s_{uv}-\mathbf s_{hu}.
\]

A triplet term based on \(\|\Delta\mathbf s_{huv}\|\), augmented by physical slice separation, evidence, and uncertainty, expresses a soft bending prior. For variable spacing, a curvature-like quantity should also account for the distance between the segment midpoints. Avoid a hard straightness rule: real catheters can bend. Four MPNN layers give a finite graph receptive field; whole-trajectory reasoning additionally depends on the decoder and connectivity.

This is a **spatial slice graph**, not a temporal sequence unless actual time-resolved images are used. The 68,124 nodes and 1,243,324 edges describe computational size, not independent clinical sample size. Graph construction and offset targets must be confined to each patient's coordinate system.

The two slope outputs are not themselves a 3D unit vector. Under the increasing-z convention,

\[
\mathbf t=\frac{(v_x,v_y,1)}{\sqrt{1+v_x^2+v_y^2}}.
\]

State the coordinate frame and handle oblique scans through the image affine. This representation assumes paths are sufficiently monotonic in the chosen slice direction. A single direction per voxel is also ambiguous where unresolved catheters overlap; graph hypotheses can preserve alternatives, but cannot recover information absent from the image without additional assumptions.

**Bundle inference: which plausible paths should coexist?** An independent classifier cannot easily distinguish a second real catheter from a duplicate proposal for an already represented catheter. Relative distance, longitudinal overlap, direction, and shared evidence create a relational problem. Edge-biased attention and a permutation-equivariant set encoder can score trajectories in the context of neighbors. Attention weights are not causal explanations; inspectable inputs, counterfactual ablations, and failure analyses support transparency more credibly.

Require invariance to candidate ordering and assess robustness to varying candidate counts. Do not impose a universal no-crossing constraint: projected crossings and closely spaced physical catheters can be legitimate. Penalize incompatible identity assignments and duplicate explanations of the same observations.

**4.3 Present a unifying objective without inventing an implementation claim**

Let \(\mathcal T\) be candidate curves and \(\mathcal S\subseteq\mathcal T\) the selected set. An explanatory energy is

\[
E(\mathcal S)=
\sum_{T\in\mathcal S}\left[
 -\int_T\ell_{\mathrm{image}}(\mathbf r)\,ds
 +\lambda_\kappa\int_T\kappa(s)^2\,ds
 +\lambda_t\int_T\bigl(1-|\mathbf t_T(s)\cdot\hat{\mathbf t}(\mathbf r(s))|\bigr)\,ds
 +\lambda_e E_{\mathrm{end}}(T)
\right]
+\lambda_b\sum_{T\ne T'}\psi_{\mathrm{duplicate/identity}}(T,T')
+\lambda_n|\mathcal S|.
\]

Here image evidence supports paths, curvature and tangent consistency support plausible continuation, endpoint terms encourage adequate extent, and bundle terms control duplicate explanations. The cardinality term makes explicit that adding arbitrary paths is not free. This is a **conceptual decomposition**, not a claim that the current neural pipeline minimizes this exact energy or learns calibrated likelihoods. Label endpoint modeling as a proposed extension if it is not implemented.

The physics filter, network, MPNN, and bundle scorer are ways to approximate these factors. Dynamic programming, min-cost flow, structured instance prediction, or simpler learned scoring are credible alternatives. The paper becomes stronger when it demonstrates where its chosen approximation helps, rather than calling it the unique mathematical solution.

**4.4 Test three hypotheses, then write the manuscript around them**

| Hypothesis | Controlled experiment | Evidence that would support it |
|---|---|---|
| Preserving fine image evidence improves localization | Same backbone, labels, physical field of view, training budget, and decoder; native/fine anisotropic sampling versus 1 mm isotropic input; separately test deployment striding. | Better paired shaft and endpoint errors with documented memory/runtime trade-offs. Do not attribute all historical differences to resampling. |
| Path-aware outputs improve reconstruction beyond overlap | Mask-only versus mask+peak versus mask+peak+direction, with fixed features and decoder comparisons. Add a topology-loss baseline. | Reduced fragmentation, identity switches, and missed longitudinal coverage at comparable detection recall. |
| Bundle relations reduce duplicates beyond individual confidence | Freeze candidate generation; compare independent MLP, MLP+NMS, relational scorer, and relational scorer+same NMS. | Higher precision/FROC performance at matched strict recall, with gains concentrated in verified duplicates rather than loss of real close neighbors. |

Use one primary U-Net-based system in the main paper. Include the axial transformer and ensemble as an accuracy/cost experiment unless they produce a robust, meaningful gain. Keep the standalone MPNN result visible as evidence of the localization/coverage trade-off. Do not describe the unified pipeline as end-to-end trained unless gradients actually cross all stages.

The introduction should move through clinical need, the mismatch between foreground segmentation and catheter reconstruction, prior geometric and learned approaches, and the three hypotheses. Methods should then follow data provenance → task/metrics → pipeline → training/evaluation → deployment/dosimetry. Results should answer the three hypotheses before presenting workflow implications. The discussion should explain both gains and persistent endpoint, generalization, and correction limitations.

**5. Eight reviewer questions and concrete ways to pre-empt them**

**Question 1 — “Where is the truly held-out evaluation, including all learned and manually tuned stages?”**

An OOF U-Net does not make a graph model trained on all OOF labels itself OOF. Nor does a training-free filter become untouched after repeated all-case tuning. Address the code collision, checkpoint selection, plane inputs, and threshold sweeps documented above before claiming unbiased performance.

Use grouped outer folds by patient, keeping repeat acquisitions/fractions and all augmentations together. Within each outer training set, use inner splits to select Stage-1 checkpoints and create training predictions for downstream graph models. Train all MLP/graph/set components and choose thresholds using only that outer training population. Run the entire frozen pipeline on the outer held-out patients. Upstream models used to build downstream training features must not have seen outer-test labels, even indirectly through another global OOF fold.

Export a manifest associating each prediction with case ID, outer fold, training case IDs, checkpoint hash, preprocessing version, and scoring configuration. Keep production training outputs in a separate namespace. Make labels unavailable to inference and verify that deleting label files does not change predictions. A no-label inference check will not fix prior development exposure, but prevents this class of direct leakage.

After substantial development on all 54 cases, even repaired nested CV estimates internal performance of a selected research approach. A frozen external or later-acquired cohort is the strongest remaining test. Historical test labels should not be presented as newly untouched data.

**Question 2 — “Are the historical comparison, sample counts, and metrics genuinely comparable?”**

Publish an annotation ledger mapping patient, acquisition, catheter identity, observer, visible extent, and inclusion reason. Keep CY long and HT short labels distinct; identify HZ's legacy annotations separately. Do not add the 543 repeat annotations to the 1,063 physical catheter count. Reconcile missing annotations and deduplication explicitly.

Use one evaluator for all rerun methods. Primary detection should be one-to-one; closest recall is supplementary diagnostic coverage because multiple references may select the same prediction. Define admissible matches using physical distance and longitudinal coverage, with tip or endpoint criteria appropriate to the task. As a starting research specification, test a substantial coverage requirement such as 80%; choose and freeze it before final evaluation rather than presenting it as an established clinical standard.

Implement maximum-cardinality matching over admissible pairs, then minimize distance among equivalent-cardinality assignments, or use an equivalent correctly penalized assignment with dummy unmatched nodes. Unrestricted minimum-cost Hungarian assignment followed by threshold rejection is not guaranteed to maximize the number of valid matches.

Report tolerance curves at 1, 2, 3.5, and 5 mm; strict recall; FP per case; precision; F1; coverage; whole-curve 3D distances; and 3D tip errors. For long-only analysis, state how real short-needle detections are handled. Either evaluate an explicitly long-class output or report all-class performance; do not silently convert true short detections into false positives or remove them post hoc.

**Question 3 — “Does 0.55–0.65 mm approach human agreement, or merely agree with the training annotator?”**

Use the intersection of identifiable catheters with independent observer annotations, recording how many of the historical 543 are actually recoverable. The old JSON's 349 pairs are not automatically the complete repeat-observer study. For each matched physical catheter, compare algorithm–CY, algorithm–LB, algorithm–AMe, and observer–observer disagreement under identical extent conventions. Obtain independent short-catheter annotations if HT is the sole reference for that class.

Interobserver disagreement is an empirical reference distribution, not a hard accuracy floor. Shared observer bias, training to CY's style, and differences in visible endpoint definitions can all change the comparison. An algorithm can agree with one observer more closely than two observers agree without being closer to physical truth. Use adjudication or CT/phantom geometry on a subset for absolute accuracy, accounting for registration error and implant movement.

Report paired patient-level error distributions and confidence intervals. Use Bland–Altman analysis for signed comparable measurements, such as signed endpoint displacement or reconstructed length, rather than treating nonnegative Euclidean distances as interchangeable repeated scalar measurements. Predefine a clinically meaningful agreement margin if making an equivalence claim; a nonsignificant difference does not prove equivalence.

**Question 4 — “Does the geometric gain improve dose fidelity or clinical work?”**

Perform a retrospective plan-perturbation study on available complete plans. First hold anatomy, source model, channel identities, dwell times, and dose engine fixed while replacing reference paths with predicted paths. Recompute dose to isolate reconstruction error. Then, as a separate experiment, reoptimize plans on reconstructed geometry and evaluate against the reference geometry; otherwise reoptimization can conceal the error being studied.

Report HR-CTV D90 and relevant OAR D2cc changes, absolute and relative dose differences, constraint crossings, and changes in clinically active dwell positions. Include signed axial tip offsets, transverse shifts, curvature, missing active channels, and identity swaps. Define handling of false-positive channels before evaluation; their existence does not imply that they receive dwell time. Compare automatic output and clinician-corrected output separately.

The first-order relationship is \(\delta D(\mathbf r)\approx\nabla D(\mathbf r)\cdot\delta\mathbf r\). Direction, dwell weighting, source distance, and anatomy determine the effect. A local inverse-square intuition cannot justify a universal “2 mm means 10–20% DVH error” claim. Include empirical error perturbations and an illustrative 0.5/1/2 mm sensitivity sweep, clearly identifying the latter as simulations.

A pilot of roughly 20–30 evaluable plans could characterize effect sizes and feasibility, but the final sample should follow available plans and precision/power requirements. If plans are unavailable, write a technical reconstruction paper and state dosimetric validation as uncompleted work. Do not infer clinical benefit from the shaft ratio.

**Question 5 — “Why are all these components necessary, and is the baseline competitive in 2026?”**

Include a well-trained nnU-Net v2 baseline with a documented configuration and tuning budget. Use both a standard reconstruction decoder and the proposed decoder on its outputs where feasible; this separates better image evidence from better path reconstruction. Compare mask-only and multihead versions with equivalent capacity. Add a topology-aware loss comparator: clDice encourages tubular topology, but does not by itself encode individual catheter identity. [clDice primary paper](https://arxiv.org/abs/2003.07311)

Ablate asymmetric evidence, Radon recovery, learned MLP scoring, direction heads, triplet messages, and bundle scoring in a staged design. Do not require a full combinatorial experiment: first identify which stage addresses each documented failure mode, then test the corresponding intervention. Distinguish graph offset regression from link selection to determine the origin of the MPNN localization gain.

Use a simple global selection baseline, such as constrained path decoding or independent scoring with NMS, before attributing improvement to attention. Compare matched-recall operating points and candidate pools. Repeated training seeds are particularly useful for the two-needle ensemble gain. The modern literature comparators in Section 7 are necessary to establish what is new beyond the 2019 lineage.

**Question 6 — “What fails in short needles, crowded bundles, and complete patients?”**

Report separate long/short precision, recall, F1, coverage, shaft, and endpoint errors; then stratify by boundary adjacency, spacing, acquisition type, visible length, curvature, and nearest-neighbor distance. Annotation source and catheter class are confounded in CY-versus-HT results, so class differences cannot automatically be attributed to anatomy alone.

The low full-cohort recall of the long-tuned physics configuration is useful: it exposes the domain where its assumptions fail. Show whether learned channels recover those specific failures without sacrificing long-needle performance. Do not infer short-only recall by subtracting totals from different pipeline configurations.

Report the fraction of patients with every clinically relevant catheter found, no duplicates, correct identities, and acceptable endpoints. Also report per-case additions, deletions, and endpoint edits. A toy calculation illustrates the importance of this endpoint: if 20 needle detections were independent with 95% success each, only \(0.95^{20}\approx36\%\) of cases would be complete. Actual errors are correlated, so measure this directly rather than treating the illustration as an estimate.

Show ordinary successes, worst cases, and residual failures selected by a stated rule. Establish a review workflow that highlights ambiguous fragments and possible missing channels. Validate confidence calibration before interpreting “60% confidence” as a 60% probability of correctness.

**Question 7 — “Do 54 old single-center cases establish generalization or statistical superiority?”**

Use patients as the resampling unit for confidence intervals and paired comparisons. Report pooled catheter metrics and patient-macro metrics because patients contribute different numbers of catheters. Bootstrap entire patient bundles, preserving all needles and observer comparisons within each resample. Do not treat graph edges or thousands of sampled centerline points as independent patients.

Predefine one primary endpoint and one main comparison. For example: paired patient-level strict recall at a validation-selected FP/case operating point, with coverage and endpoints in the matching definition. Present effect sizes and uncertainty, then secondary localization and workflow metrics. Adjust or clearly label exploratory multiple comparisons; do not select the best confidence threshold on the test set.

Prioritize an independent MR cohort with another scanner/site/applicator. Within the archival cohort, acquisition-held-out tests, learning curves, intensity perturbations, and synthetic gaps probe robustness but cannot replace external validation. Geometric priors are not inherently invariant to field strength, sequence, device material, patient anatomy, or MR distortion. On a small external set, publish broad confidence intervals and every failure rather than claiming universal transfer.

**Question 8 — “Is the advertised CPU/Slicer performance reproducible, and what is actually automatic?”**

Benchmark the exact deployed artifact and preprocessing path on the same cases used for accuracy. Report CPU model, physical/logical cores, thread settings, RAM, software versions, model hashes, input dimensions, and actual effective spacings. Separate cold model loading, warm inference, filtering, graph construction, decoding, import/export, and rendering. Measure total wall-clock time independently rather than adding overlapping stage estimates.

The listed stage means sum to approximately 0.615 s, compatible with the rounded 0.61 s only if those stages are sequential, nonoverlapping, and measured consistently. The deployed source's subsampling and its older timing comment require direct measurement; neither confirms nor disproves the claimed new latency. Report median, IQR, p95, maximum, and peak RAM, including full-field-of-view cases.

Ensure the CPU-only environment cannot silently use a GPU provider. For GPU comparisons, include synchronization and transfers. Distinguish timings with cached physics/OOF maps from fresh processing. Demonstrate parity between offline evaluation and Slicer inference, including crop, feature normalization, model, and reconstruction settings.

Document all required user inputs: image selection, template plane, catheter count, tip seeds, and manual edits. Conduct a blinded or randomized-order reader crossover study comparing manual and assisted reconstruction, with washout if cases repeat. The clinically useful endpoint is time to verified output with correction counts and residual errors. Source-channel connection and treatment approval remain separate tasks unless explicitly implemented and evaluated.

**6. Exact figure blueprint: eight figures with one message each**

Use consistent colors throughout: gray anatomy, blue reference trajectories, orange predictions, teal image evidence, violet relational stages, and red errors. Use physical scale bars and display uncertainty intervals. Produce diagrams as vector graphics and quantitative plots from the frozen evaluation exports. Do not create synthetic “clinical results” panels; proposed experiments below remain empty until measured.

**Figure 1 — The clinical task and the missing representation**

Layout: three columns across a double-width figure, with a narrow lineage strip along the bottom. Left: one axial pelvic MR slice plus a small coronal inset locating template, long catheter, and short obturator catheter. Middle: two enlarged 12–20 mm patches showing a crowded pair and an obturator-boundary catheter; under each, show the semantic foreground union. Right: the same evidence rendered as separate colored centerlines with distal tips and a channel-review list. The bottom strip reads “2013 seeded path → 2017 model/collection → 2019 automatic foreground → 2026 automatic path/bundle inference.”

Caption message: **“The output required for planning is a set of individual paths with endpoints.”** Avoid a historical accuracy bar chart because the historical metrics are not commensurate. Use one anonymized case and explicit slice locations.

**Figure 2 — One readable architecture diagram**

Layout: three horizontal bands, with left-to-right flow. Top band: MRI and physical coordinate metadata → image-derived features. Middle band: a compact U-Net icon with five labeled output maps; a parallel physics proposal branch enters candidate construction. Bottom band: slice detections → directed spatial graph → candidate trajectories → bundle scoring/selection → Slicer centerlines. Show optional axial attention with a dashed box, rather than drawing a second full network. Mark manual template-plane input if the evaluated system uses it.

Each arrow carries a representation label: voxel field, slope field, point set, edge set, or curve set. Add small CPU timing boxes only after deployment verification. Display six channels if that is the frozen model. Label any reduced-resolution neural path honestly. The exact union of physics, tangent-tracker, and MPNN proposals must match code; this is a proposed layout, not proof of the current topology.

Caption message: **“Local evidence, path continuity, and bundle consistency act at different scales.”** Put kernel/channel tables in the supplement.

**Figure 3 — Why the geometric components help**

Layout: three rows × four columns. Rows are “boundary asymmetry,” “signal dropout,” and “duplicate/shadow track.” Columns are “input,” “intermediate evidence,” “without component,” and “with component.” Use identical windows and physical extents within each row. Boundary row: draw symmetric and half-ring sample locations. Dropout row: a z-versus-position view shows the weak segments and the line-integral accumulation. Duplicate row: show which observations two candidate curves share and the result of bundle selection.

Add one quantitative paired difference at the end of each row from the corresponding ablation, not just a visually attractive success. Caption message: **“Each component addresses a measured failure mechanism.”** These examples should include one case where the intervention does not work, in the supplement or Figure 8.

**Figure 4 — Triplet geometry and bundle relations**

Layout: left half shows three or four translucent slice planes containing a few detections. Highlight incoming h→u and two candidate outgoing links u→v₁/u→v₂; show the slope mismatch equation beside them. Draw a skip edge across an empty slice and a subpixel offset arrow. Right half shows six candidate curves, including two duplicates and two legitimate close neighbors, with a small relational graph and before/after selection. Include a tiny candidate-permutation test schematic below.

Use coordinates in millimeters. Call the graph spatial, not temporal. Caption message: **“Correct continuation depends on neighboring directions; duplicate suppression depends on the entire candidate set.”** Distinguish a designed regularizer from a learned attention score.

**Figure 5 — Detection and correction trade-offs**

Layout: three panels. A: strict one-to-one recall versus FP/patient (FROC), with bootstrap bands for the main methods. B: paired patient-level precision change at matched strict recall for independent versus bundle scoring. C: long and short recall at the frozen primary operating point, together with the percentage of complete patients. Mark the 25% and 60% operating points as illustrative score thresholds only if independently selected/evaluated.

Use closest recall as a faint supplementary curve or in the supplement, not the headline panel. Avoid a radar plot. Caption message: **“Higher candidate coverage becomes useful only when duplicate burden and patient completeness are visible.”**

**Figure 6 — Localization, endpoints, and observer agreement**

Layout: four panels in a 2×2 grid. A: paired patient distributions of shaft-XY error. B: 3D tip-error distributions with a visible tail or inset. C: longitudinal coverage versus shaft error, revealing short accurate fragments. D: algorithm–observer and observer–observer errors on the identical overlap subset, including the available paired sample count. Mark long/short classes with shape rather than relying only on color.

Compare methods on common matches as well as report their full matched sets in the associated table. Caption message: **“Submillimeter transverse agreement does not establish endpoint accuracy or complete reconstruction.”** An optional signed-tip Bland–Altman plot belongs in the supplement if space is tight.

**Figure 7 — Clinical effect and computation**

Layout: left half shows paired changes in HR-CTV D90 and OAR D2cc from fixed-dwell plan perturbation; right top shows manual versus assisted time-to-verified-output; right bottom shows measured total latency and its stages. Include one small dose-difference slice only if it illustrates a representative measured case, with a color bar and fixed dose scale.

If no plans or reader experiment exist, replace this figure with a clearly labeled geometric robustness and runtime figure, and remove clinical-outcome claims from the abstract. Caption message for the full version: **“Clinical value is measured through dose fidelity and verified workflow time.”**

**Figure 8 — Generalization and honest failure limits**

Layout: left half contains a forest plot of internal and external performance by site/sequence/applicator, with n patients and confidence intervals. Right half is a 2×2 failure atlas: missed short needle, merged/identity-switched pair, incorrect tip extent, and dark-anatomy false positive. Each thumbnail has one sentence describing cause and required correction. If no external cohort exists, label the forest plot as internal acquisition-stratified analysis and include an explicit “external validation pending” statement in the caption.

Caption message: **“The remaining errors define the conditions under which human review is needed.”** Select examples systematically, such as the worst endpoint case and the most common error in each class.

For a six-figure submission, merge Figures 3 and 4 and move Figure 8's atlas into the supplement. Retain the dataset/split flowchart as Supplementary Figure S1, with every training and test dependency shown. Main tables should cover cohort/annotations, final comparative results, and targeted ablations; a separate supplement should contain the complete operating-point sweep and historical comparison caveats.

**7. Public datasets and a credible 2026 SOTA strategy**

**7.1 Literature comparators that should change the manuscript**

The search below is targeted rather than an exhaustive systematic review. Verify bibliographic details again at submission, but these primary sources already rule out a manuscript that treats the 2019 result as the only relevant competitor.

| Source | Why it matters | Fair use in NeedleFinder |
|---|---|---|
| [Shaaer et al., JACMP 2022](https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.13494) | Semiautomatic MR-only gynecologic reconstruction using 2D U-Net localization and image extrema; 20 patients, five planners, reported mean reconstruction variation 0.97 ± 0.66 mm and 11 ± 1 min workflow time. | Correct the vault's journal/architecture description. Discuss observer comparison and workflow time, without equating its dwell-position variation to shaft-XY error. |
| [Aleong et al., Medical Physics 2024](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.17117) | Rapid multicatheter MRI segmentation in prostate interventions, with independent observers and dosimetric assessment. | An adjacent clinical application with a stronger validation template than Dice alone. Explain differences in anatomy, geometry, and acquisition. |
| [Kim et al., Brachytherapy 2026](https://pubmed.ncbi.nlm.nih.gov/41203438/) | Direct contemporary MRI comparator: uncertainty-aware catheter digitization with Venezia applicators; 35 MRIs/30 patients, reported shaft and tip errors 0.74 ± 0.32 mm and 2.52 ± 2.04 mm, with plan comparison. | Essential related work. Its method uses an actual catheter count during refinement and a 0.5×0.5×1.0 mm resampled grid, which should be explicit in a baseline inventory. [Primary Methods](https://www.sciencedirect.com/science/article/abs/pii/S1538472125003186) |
| [Xiao et al., July 2026 preprint](https://arxiv.org/abs/2607.11290) | CT digitization combining synthetic-to-real learning and structure-aware Hough/tracking; evaluated over 203 fractions from 38 patients. | Related geometric hybrid work, explicitly labeled a preprint and a different modality. It prevents a broad claim that combining learning with geometric tracking is itself novel. |
| [nnU-Net official implementation](https://github.com/MIC-DKFZ/nnUNet) | A strong self-configuring semantic baseline with current residual-encoder configurations. | Train on the same folds and labels; document preprocessing, resource budget, and decoder. |
| [clDice](https://arxiv.org/abs/2003.07311) | A tubular topology-aware loss. | Test continuity gains separately from instance identity and endpoint completeness. |

Do not compare these published numerical means as a leaderboard. Different cohorts, references, interaction levels, dose workflows, and matching rules prevent that. Where code is unavailable, label a reproduction as an implementation inspired by the paper and describe deviations. Avoid substituting an undertrained transformer for a carefully tuned modern U-Net baseline.

**7.2 Dataset options, with access and task fit distinguished**

I did not verify a freely downloadable, independently labeled external **gynecologic MR multicatheter** benchmark suitable for immediate paired comparison. That is a finding of this search, not proof that none exists. Public records, accessible files, useful annotations, and permission to redistribute are different properties.

| Resource | Verified contents/access status | Useful experiment and limitation |
|---|---|---|
| BWH 54-case collection and Case 050/RSNA example | Present in the project's provenance notes; this review did not establish a current public distribution URL or release license. | Highest-value release candidate, with institutional authorization. Case 050 is already development/calibration data and cannot be external validation. |
| [DeepInfer Prostate-Needle-Finder](https://www.deepinfer.org/models/prostate-needle-finder/) and [Mehrtash et al. primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6450731/) | Model/deployment resource and a 71-patient MRI needle study; a model download is not evidence that the patient images and annotations are public. | Relevant susceptibility-void MRI domain; explore authorized collaborator data access. Single biopsy needles test image localization, not dense gynecologic bundle separation. |
| [TCIA Prostate-MRI-US-Biopsy](https://www.cancerimagingarchive.net/collection/prostate-mri-us-biopsy/) | Public MR/US collection with tracked biopsy coordinates, anatomy/target surfaces, and overlays; 1,151 subjects listed. | Potential geometry or registration research. It is **not established as a visible in-situ needle-segmentation ground-truth dataset**. Coordinate records alone cannot validate extraction of a needle from MRI. Inspect actual image timing and labels before committing effort. |
| [FM-US-Video, Zenodo 21896696](https://zenodo.org/records/21896696) | Open record published August 2026 with US needle data and model-weight archives. | Inspect annotation schema, acquisition grouping, and reuse terms before use. Useful for ultrasound-video localization/tracking; temporal frames are not calibrated volumetric slice planes. |
| [Malamal and Panicker, Zenodo 7791802](https://zenodo.org/records/7791802) | Ultrasound transmission/needle-imaging data with related scripts; record is public but files are **restricted**, requiring a data-release agreement/access request. | Controlled testing of angle-dependent visibility and line evidence. This is not a ready-made open 3D multi-needle MRI benchmark, and manual labels may need preparation. |
| [Wijata/Andrzejewski/Pyciński, Mendeley Data](https://data.mendeley.com/datasets/zk6scwv52p/1) | CC BY 4.0 package containing a trained CNN, MATLAB scripts, and a few 2D ultrasound input images. | Small reproducibility/demo resource; do not describe it as the full clinical training dataset or a substantial independent validation cohort. |
| [Breast US needle tracking study, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11782214/) | Eleven-patient ultrasound video study; the paper directs data requests to the authors. | Possible clinical 2D transfer study, conditional on access. Split by patient/video, never adjacent frames. Public article availability does not mean open data. |
| [HDR breast catheter EMT/CT-reference study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5608198/) with [OSF deposit](https://osf.io/kd6ta/) | The paper declares a public deposit of tracking/reference data. The OSF file inventory could not be verified in this review. | Potential catheter geometry/assignment benchmark. Confirm whether original CT voxels and usable labels are included; do not advertise it as an established CT segmentation dataset. |
| [3D ultrasound orthogonal-plane CNN study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6132402/) | Relevant primary 3D US needle study; no current open downloadable labeled release was verified here. | A collaborator/request lead for genuine volumetric US evaluation, rather than repurposing 2D video as 3D. |

Beware keyword matches: a dataset described as “needle tracking in CT-guided surgery” can contain external RGB camera video rather than CT volumes. Generic cervical OAR datasets, prostate MR datasets, and ultrasound tumor datasets should not be advertised as catheter benchmarks without verifying image-visible devices and reference trajectories.

**7.3 Highest-impact open-benchmark strategy**

First priority is a **frozen external MRI test**, preferably another institution or contemporary hybrid applicator. A modest independently annotated cohort with transparent uncertainty is more directly relevant than a large unrelated ultrasound dataset. Run the frozen model before any adaptation; if adaptation is studied, report it separately and reserve another test subset.

Second priority is a **reproducible BWH reconstruction benchmark**. Subject to institutional permission, release de-identified volumes or an authorized subset, physical-coordinate centerlines, observer identities, acquisition metadata, catheter classes, visible-extent conventions, and a case-level annotation ledger. Include predefined patient splits, inference-only containers, production weights clearly separated from evaluation folds, and a metric implementation with tiny synthetic examples that reveal matching and endpoint behavior. If images cannot be distributed, release permitted derived data and a container-to-data evaluation mechanism. Do not promise that derived data are automatically unrestricted.

Third priority is a **paired MRI/CT phantom** with known geometry: vary intercatheter spacing, crossing angle, curvature, visible length, proximity to a dark cavity/obturator, and acquisition anisotropy. Acquire independent repeat scans and quantify MR–CT registration uncertainty. Preserve the distinction between physical catheter axis, visible artifact center, and source-dwell path. This directly tests the mechanism behind a submillimeter claim.

Fourth priority is a **synthetic geometry benchmark** for the path and bundle stages. Generate curves with controlled dropout, false detections, duplicate proposals, coordinate noise, and unresolved crossings. Split by generated bundle/phantom configuration, not by points. Make the image simulator independent enough from the proposed filter to avoid awarding it an artificial advantage. Use this to map failure boundaries and sample efficiency; it cannot establish clinical MRI generalization.

A useful benchmark should score identity switches, misses, duplicates, longitudinal coverage, 3D tips, shaft errors, calibration, and runtime together. Offer acquisition-specific breakdowns and a downloadable prediction format. A public test server should limit adaptive test-set tuning or preserve a hidden secondary test. The benchmark may become a longer-lived contribution than a particular network combination.

**8. Practical manuscript and experiment priorities**

**Before interpreting headline performance:** preserve the current artifacts; separate production and OOF storage; regenerate verified predictions; remove reference-derived inference inputs or label the assisted condition; freeze the manifest, deployed model, evaluation protocol, and annotation ledger. Export the 22 October experiment's actual per-case raw results. No further model complexity is needed to perform this work.

**Before claiming methodological improvement:** run the contemporary nnU-Net baseline and the three controlled hypothesis tests in Section 4.4, with patient-level uncertainty. Measure identity, coverage, and endpoints. Keep “training-free” only for a truly unlearned branch; an MLP with trained weights is learned even if inference uses NumPy. Treat the transformer as optional unless its incremental benefit survives these controls.

**Before claiming clinical value:** obtain independent observer review, fixed-dwell dose analysis where plans exist, and time-to-verified-output. Validate the exact CPU/Slicer path. Add a frozen external MRI or paired phantom study according to what is feasible, explicitly distinguishing generalization from physical-geometry validation.

**Venue judgment:** PMB is a coherent home for a carefully measured continuation of this reconstruction problem, particularly with geometric/dosimetric validation. Medical Physics is compelling with rigorous observer, planning, and workflow results. MedIA/TMI would require a stronger methodological argument and broader independent validation than a collection of familiar modules on one archival cohort. These are judgments about fit, not predictions of acceptance.

The durable claim to aim for is: **“Explicit modeling of catheter trajectories and their relationships improves the completeness and usability of reconstruction beyond what voxel overlap alone reveals.”** Let the controlled experiments determine how much physics, learned image context, and graph/set reasoning is needed to support that claim.

**Source and scope note**

Local context read: [index](/home/guillaume/github/needlefinder/vault/wiki/index.md), [overview](/home/guillaume/github/needlefinder/vault/wiki/overview.md), [historical splits](/home/guillaume/github/needlefinder/vault/wiki/research/historical-dataset-splits-2013-2026.md), [21 October architecture survey](/home/guillaume/github/needlefinder/vault/wiki/research/2026-10-21-sota-nn-hybrid-needle-detection.md), [22 October benchmark](/home/guillaume/github/needlefinder/vault/wiki/research/2026-10-22-full-54case-unet-transformer-graphnet-benchmark.md), and [paper roadmap](/home/guillaume/github/needlefinder/vault/wiki/syntheses/2026-paper-roadmap-explainable-zero-shot-vs-3d-cnn.md). Selected code and raw-artifact checks are linked at the relevant findings. External links support the associated literature or dataset statements; experimental designs and manuscript recommendations are this review's proposals. No patient-data release, collaborator contact, model training, or software modification was performed for this review.
