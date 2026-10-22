---
title: "NeedleFinder 2026: Sub-Millimeter Catheter Reconstruction on MRI for Interstitial Gynecologic Brachytherapy in Seconds"
author:
  - "Guillaume Pernelle, Paolo Zaffino, Lauren Barber, Alireza Mehrtash,"
  - "Christiane Yeung, Tina Kapur, and Collaborators"
date: "October 2026 (Target Journal: *Physics in Medicine & Biology*)"
abstract: |
  **Objective.** Interstitial gynecologic brachytherapy planned on MRI alone requires locating 15 to 30 thin plastic catheters on 3T T2-weighted images. Tracing them by hand takes tens of minutes while the patient is under anesthesia. Automatic methods based on voxel segmentation break catheters into fragments and mistake blood vessels for catheters. We set out to make catheter reconstruction automatic, accurate to below a millimeter, and fast enough to run during the procedure.

  **Approach.** NeedleFinder 2026 works in three stages. Stage 0 is an analytical filter that responds only to small, round, dark spots. Stage 1 is a 3D U-Net that predicts catheter centerlines, directions, and the vaginal obturator. Stage 2 traces candidate trajectories and uses two graph neural networks: one scores how slice detections link into 3D trajectories, and one judges whether the whole set of trajectories forms a plausible implant. Stage 2 is pre-trained on 300 simulated implant bundles (SynthBrachy3T). We evaluated the method on 54 clinical 3T MRI scans containing 1,063 expert-traced catheters with patient-level 5-fold cross-validation, against re-implementations of three published methods, against a second team of human observers on 40 scans, and on 48 public needle-free pelvic MRI scans.

  **Main results.** NeedleFinder 2026 found 91.3% of catheters with 85.3% precision (F1 88.2%), compared with 81.3% F1 for our 2019 3D U-Net and 53.7% for a 2026 structure-aware tracker ($p < 10^{-4}$). The median shaft error was 0.46 mm and the median tip error 2.61 mm. The method agreed with the expert reference more closely (0.56 mm mean shaft discrepancy) than two human teams agreed with each other (0.71 mm). When only its bundle-scoring stage was fine-tuned on a single annotated patient, synthetic pre-training reached 82.0% F1. On needle-free scans the full pipeline produced 4.7 false tracks per scan at the default threshold, and 0.38 per scan when its candidates were gated by the Stage 0 symmetry map instead of the bundle transformer. Full-volume reconstruction takes 2.74 s on an 8-core CPU (1.20 s on standard $320 \times 256$ scans, 3.47 s on high-resolution $640 \times 416$ / $768 \times 768$ scans) or 1.25 s on a GPU.

  **Significance.** Automatic catheter reconstruction on MRI now matches expert agreement, needs very little training data, and runs in 1 to 3 seconds per patient. It turns a manual task of tens of minutes into a short verification step.
---

# Plain-Language Summary {-}

Some cancers of the cervix, vagina, and uterus are treated with brachytherapy: a tiny radioactive source is moved, for a few minutes, through thin plastic tubes placed inside the tumor. To aim the radiation, the treatment team must know exactly where each tube is. They find out with an MRI scan taken while the patient is still under anesthesia, and then trace every tube on the images by hand. With 15 to 30 tubes per patient this takes tens of minutes, and a position error of one or two millimeters can change the dose to the tumor or to nearby organs by about ten percent.

On an MRI scan a tube does not glow. It is a small dark dot, a few pixels wide, on each image slice. Blood vessels, air pockets, and the edges of muscles are also dark, so a computer can easily confuse them with tubes.

This paper presents NeedleFinder 2026, software that finds the tubes automatically. It works in three steps. First, a simple filter keeps only dark spots that are dark all the way around, the way a tube cross-section looks, and discards one-sided edges. Second, a neural network looks at the 3D scan and marks the center of each tube on each slice. Third, two "graph" networks connect the marks from slice to slice into 3D curves, pull apart tubes that touch, and then look at the whole bundle together to throw out curves that do not fit a real implant.

We tested it on 54 patients with 1,063 tubes. It found 91 percent of them and was right 85 percent of the time when it reported one. Its traced positions were, on average, half a millimeter from the expert's, which is closer than two experts are to each other. It runs in under three seconds on an ordinary computer (about one second on standard-resolution scans) and needs no graphics card. In practice, the physicist clicks one button, checks the result, and fixes the rare miss, instead of tracing everything from scratch.

# Introduction

## The Clinical Problem

Brachytherapy delivers radiation from inside the body. For locally advanced cervical cancer it is the cornerstone of curative radiotherapy [@potter2018; @potter2021]. The EMBRACE-I study reported 92% local control at five years with a median dose of 90 Gy (EQD2) to the high-risk target volume [@potter2021], and EMBRACE-II now prescribes planning aims of 90 to 95 Gy while sparing the bladder, rectum, and sigmoid colon [@potter2018]. Reaching such doses in large or irregular tumors requires *interstitial* implants: in addition to a central tandem and a vaginal obturator or ring, 10 to 30 thin plastic catheters or needles are inserted into the tumor through a perineal template or freehand [@mastmeyer2017; @zaffino2019; @chino2025].

The tumor itself is best seen on T2-weighted (T2W) MRI, which GEC-ESTRO and consensus recommendations designate as the reference for target delineation and set reconstruction requirements for [@dimopoulos2012; @jacobsen2022]. MRI-based applicator reconstruction with marker-filled catheters was established for cervical cancer in 2009, with inter-observer standard deviations below 2 mm [@haack2009]. Historically, clinics took both an MRI and a CT scan and registered the two, which adds its own uncertainty to the reconstruction [@hellebust2010], and practical guidance for hybrid implants still advises a CT in addition to the MRI because plastic needles are hard to see on MRI [@ladbury2023]. The AAPM Task Group 303 report describes how to implement MRI in HDR brachytherapy from simulation to treatment [@prisciandaro2022], the Medical Physics Practice Guideline 13.a lists MRI among the accepted planning modalities [@richardson2025], and MRI-only interstitial workflows have been shown to be feasible [@shaaer2020; @lee2023]. Positive-contrast line markers make the catheter lumen visible on T1-weighted or balanced sequences [@ning2020; @shaaer2020] and are listed as required equipment in published implementation guides [@zhangh2019], and zero-echo-time sequences show empty catheters directly [@kaza2022; @lee2023], but both need an additional series fused to the T2W images. Identifying the source path on the T2W series alone remains a recognized challenge [@richart2018]; for rigid applicators, direct reconstruction on T2W MRI is in fact more precise than library-based or fused workflows [@malajovich2020], but interstitial needles have no library model.

Finding the catheters on MRI is the central technical challenge of an MRI-only workflow [@zaffino2019; @sargazi2026]. The dose from the $^{192}$Ir source falls steeply with distance: organ-at-risk dose parameters change by up to 5 to 6% per millimeter of applicator displacement [@tanderup2008], applicator shifts beyond 1.5 mm can change the rectal D2cc by more than 10% [@schindel2013], and in MRI-only cervical implants a systematic 1 mm error along the source path already changes organ-at-risk doses significantly [@singh2026]. Swapping two catheters during digitization changes prostate, urethra, and rectum dose metrics by 5 to 20% [@nunez2021], and needle reconstruction uncertainty alone shifts the urethral dose by about 3 Gy in prostate implants [@rylander2017]. Manual reconstruction is itself error-prone: with electromagnetic tracking as the reference, 23% of 265 needle reconstructions in ultrasound-guided prostate implants deviated by 2 mm or more [@androulakis2025]. Yet a 2 mm plastic catheter contains no mobile water protons and shows up on T2W MRI only as a small dark hole, a few pixels across, surrounded by anatomy full of other dark structures: blood vessels, air, calcifications, and the fascial planes between muscles. Tracing 15 to 30 catheters slice by slice therefore takes tens of minutes, 46 minutes on average in one MRI-only gynecologic series [@shaaer2022], within a template-based implant procedure of about 100 minutes under anesthesia [@roumeliotis2026].

![**Overview of NeedleFinder 2026.** The input is a clinical 3T T2W pelvic MRI (Case 017: 26 reference trajectories, including the obturator channels). Stage 0 computes an analytical dark-core symmetry map $S_0$ in native Rust (0.41 s cohort mean). Stage 1 applies a 3D residual U-Net with implant-region gating to predict centerline peaks, local directions, and the obturator (0.96 s on CPU, 0.22 s on GPU). Stage 2 links slice detections into 3D trajectories with native C++17/Rust seed tracking, a triplet message-passing network, and adhesion-aware duplicate merging; the last tile shows the out-of-fold predictions (vermilion) over the expert reference (grey) (1.22 s on CPU, 0.54 s on GPU), then prunes false trajectories with a bundle graph transformer and smooths each curve with a cubic polynomial (0.15 s on CPU, 0.08 s on GPU). The full patient volume is processed end-to-end in 2.74 s on an 8-core CPU (1.20 s on standard scans, 3.47 s on high-resolution scans) or 1.25 s on a GPU; the bar shows the stage breakdown.](../reports/figures/fig1_pipeline_overview.pdf){#fig:overview width=100%}

## Key Terms

Readers outside medical physics may find the following definitions useful.

- **Catheter, needle, shaft, tip.** We use *catheter* and *needle* interchangeably for the thin plastic tubes. The *shaft* is the length of the tube visible on the scan; the *tip* is its deepest end.
- **Digitization or reconstruction.** Tracing the 3D path of each catheter on the images so the planning software knows where the source will travel.
- **Slice, voxel, in-plane.** An MRI volume is a stack of 2D images (slices). Each 3D picture element is a voxel. *In-plane* means within one slice; here the in-plane pixel is 0.31 to 0.88 mm while slices are 1.2 to 6.0 mm thick.
- **Recall, precision, F1.** Recall is the fraction of true catheters the method found. Precision is the fraction of reported catheters that were real. F1 is their harmonic mean, a single score that penalizes both misses and false alarms.
- **Shaft error and tip error.** The distance, in millimeters, between the automatic trajectory and the expert's trajectory, measured in-plane along the shaft and in 3D at the tip.
- **Cross-validation, out-of-fold.** The 54 patients were split into 5 groups. Each group was tested by a model trained on the other four, so no patient was ever tested by a model that had seen them. All reported results are *out-of-fold* in this sense.
- **Inter-observer variability.** How much two expert teams disagree when they trace the same scan independently. It is the practical floor for how accurately any method can be measured.

## Previous Work

Automatic catheter digitization has gone through three generations. Two recent systematic reviews cover the field [@sargazi2026; @vijayan2026].

**Model-based tracking (2013 to 2019).** The BWH program of 3T MR-guided gynecologic brachytherapy in the AMIGO suite, which produced the cohort used here, was described in 2012 together with the first image-analysis tools for catheter identification [@kapur2012] and open-source 3D Slicer modules for template and obturator registration [@chen2014]. Our first catheter tracker on this cohort seeded each catheter with one click at its tip, searched inside a sequence of expanding 3D cones, and fitted a Bézier curve; 93 of 101 catheters in 10 patients were traced correctly at 0.3 s per catheter [@pernelle2013]. Mastmeyer *et al.* [@mastmeyer2017] constrained the cone search with a mechanical rod-and-spring catheter model, reaching 1.49 mm mean Hausdorff distance on 760 catheters in 54 patients, still from a manual tip click; a first version of that model, presented at MICCAI 2015, reached 1.40 mm mean error on the 101 catheters of the 2013 set [@mastmeyer2017a]. Template-geometry-assisted reconstruction on MRI has been described for commercial interstitial templates [@richart2015], and on ultrasound 3D Hough transforms localized single needles with about 1 mm endpoint accuracy in vivo [@qiu2013]. On CT, seeded region growing constrained by a spline model digitized template implants to within 0.4 mm of manual digitization [@dise2015], and thresholding followed by density-based clustering reaches 0.3 mm for rigid applicators [@deufel2020], but MRI offers no such intensity threshold. Related model-to-image registration reconstructs rigid applicators on MRI within inter-observer variability [@hrinivich2019], and line fitting by RANSAC digitizes straight needles on CT [@zheng2021]. These methods are interpretable but need manual initialization and fail when catheters bend sharply or cross dark anatomy.

**Voxel segmentation with heuristic post-processing (2019 to 2026).** To remove manual initialization, Zaffino *et al.* [@zaffino2019] trained an ensemble of 3D U-Nets to segment catheter voxels on 50 clinical T2W scans with 826 catheters, then clustered the voxels with DBSCAN and removed small islands. It reached a voxel Dice of 0.60, a mean distance error of 2.0 mm, 6.7% false-positive and 1.5% false-negative clusters, in 2 to 9 minutes per patient. Convolutional segmentation, often nnU-Net based [@isensee2021], has since been applied to CT brachytherapy of the cervix, prostate, and breast [@jung2019; @jung2019b; @zhangd2020; @hu2021; @weishaupt2022; @xie2023; @wang2024; @moradi2025; @quetin2025], to MRI-guided prostate implants [@dai2020; @aleong2024; @kostoulas2025], to single needles on interventional MRI [@li2020; @zhou2024], to transrectal ultrasound [@zhang2020tmi; @zhang2020pmb; @zhang2020mp; @andersen2020; @wangf2020; @liu2022; @goulet2025], and to gynecologic MRI with line markers [@shaaer2022] or with applicator mesh alignment [@kim2026]. Even with topology-preserving losses, from the Betti-number loss [@hu2019] to clDice [@shit2021] and skeleton recall [@kirchhoff2024], pure voxel segmentation has two failure modes on T2W MRI: a one-slice signal dropout splits one catheter into two to four fragments, and dark vessels and tendons are segmented as short false catheters. The ultrasound literature documents the same pattern: accuracy collapses on artifact-laden catheters [@liu2022] and tips are much harder than shafts [@wangf2020]. Keypoint heatmap regression, the choice made for our Stage 1, has recently been shown to localize an interventional needle on MRI more accurately than mask segmentation of the same images [@zhouw2026]. Electromagnetic or active MR tracking avoids the imaging problem altogether, at the cost of dedicated hardware and sequences in the operating room [@mehrtash2014; @wang2015; @dearcos2017; @durrbeck2022; @cantin2024; @deufel2024; @beaulieu2025; @roumeliotis2026]; crossed catheter paths were already identified as a common failure of image-based digitization in the BWH program when active tracking was introduced [@wang2015].

**Structure-aware tracking and graph networks (2022 to 2026).** On CT, Xiao *et al.* [@xiao2026] identified two further problems: *catheter adhesion*, where shafts touch or overlap on consecutive slices so that greedy trackers swap or merge them, and the scarcity of annotated clinical data. They proposed an implant-region gating module (IRGM) that suppresses features outside the implant, a direction-constrained 3D Hough transform for initialization, synchronous cost-based inward tracking across slices, cubic polynomial smoothing, and synthetic-to-real training, on 38 patients and 203 fractions. Their companion work handles needle adhesion in dense CT seed implants by detecting tips and handles [@xiao2026jbhi]. Outside brachytherapy, message-passing networks have replaced greedy linking in tracking-by-detection [@braso2020; @benhaim2022], transformers predict graphs of curvilinear structures directly from images [@shit2022], and synthetic-to-real training has been shown to transfer to real needle detection [@arapi2023]. Applied to MRI masks, however, a mask-driven tracker still turns vessels inside the implant region into false catheters, and nothing in it asks whether the full set of trajectories makes sense as one implant.

**Open data.** No public needle or catheter dataset for brachytherapy exists [@fechter2024], and the AI literature in brachytherapy remains largely retrospective and single-center [@vijayan2026]. The public pelvic MRI scans we use for the false-alarm test come from the Medical Segmentation Decathlon [@antonelli2022].

## Contributions

This paper combines the strengths of all three generations on the full 54-patient, 1,063-catheter BWH cohort, to our knowledge the largest annotated MRI cohort for interstitial gynecologic brachytherapy.

1. **A three-stage architecture.** An analytical dark-core symmetry prior (Stage 0), a 3D residual U-Net with five output heads and implant-region gating (Stage 1), and two graph networks (Stage 2): native seed tracking with a triplet message-passing network and adhesion-aware duplicate merging that keeps touching catheters apart, and a tip-aware bundle graph transformer that judges all trajectories together and refines their tips. Stage 2 is pre-trained on SynthBrachy3T, a simulator of candidate-trajectory bundles.
2. **A controlled comparison.** We re-implemented three published approaches and evaluated all methods on identical patient-level 5-fold splits, on the whole cohort and on the 23 patients with severe catheter adhesion.
3. **Three clinically relevant checks.** Agreement with experts compared with agreement between experts on 40 doubly traced scans; accuracy as a function of how many annotated patients were available to fine-tune the bundle-scoring stage, from zero to all; and false alarms on 48 public pelvic MRI scans containing no catheters.

\FloatBarrier

# Materials and Methods

## Patient Cohorts and Public Benchmark

\input{tables/tab1_cohorts.tex}

Table \ref{tbl:cohorts} summarizes the data. The primary cohort is 54 intraoperative 3T T2W turbo spin-echo MRI scans acquired during MRI-guided interstitial gynecologic brachytherapy in the AMIGO suite at Brigham and Women's Hospital (Harvard Medical School, Boston, MA, USA) with Institutional Review Board approval [@pernelle2013; @mastmeyer2017; @zaffino2019]. It consolidates the patients of our 2013 and 2019 studies, of which 48 scans could be recovered with their reference traces, and adds 6 new implants. Expert medical physicists traced 1,063 reference trajectories (9 to 42 per patient): 759 full-depth interstitial catheters from the primary annotation set, which we call *long shafts*, and 304 short obturator and partial-field-of-view tracks from the secondary set. In-plane pixel size ranges from 0.31 to 0.88 mm and slice thickness from 1.2 to 6.0 mm.

Xiao *et al.* [@xiao2026] analyzed separately the fractions in which catheters touch or overlap. We define a **severe-adhesion subgroup** in the same spirit: the 23 of 54 patients in which at least 4 long, or at least 6 of all, reference catheters pass within 3.0 mm of another catheter on shared axial slices. All 571 catheters of these patients are evaluated; 191 of them (166 long) lie within 3.0 mm of a neighbor, and 233 do across the whole cohort.

Two further datasets serve as checks. Forty scans were traced independently by two expert teams (CY, 505 trajectories; LB/AMe, 513 trajectories); the 464 pairs matched one-to-one within 3.5 mm define the human inter-observer floor. To measure false alarms on anatomy that contains no catheters at all, we use all 48 pelvic T2W scans of the public Medical Segmentation Decathlon prostate task [@antonelli2022] (MSD Task05_Prostate; 0.60 to 0.75 mm in-plane, 3 to 4 mm slices).

## Stage 0: Analytical Dark-Core Symmetry Prior

The physical idea is simple. On an axial slice a catheter cross-section is a dark core that is surrounded by brighter tissue in *every* direction, whereas a fascial plane or a muscle border is dark on one side only. Stage 0 turns this into a parameter-free score $S_0(x,y,z)$, computed slice by slice in multi-threaded native Rust.

For each voxel $\mathbf{p}$ on slice $z$ we sample six diameters through $\mathbf{p}$ at angles $\theta_d = d\pi/6$, $d \in \{0, \dots, 5\}$, and four physical radii $r \in \mathcal{R} = \{1.35, 1.85, 2.5, 3.3\}$ mm, converting each radius to pixels with the in-plane spacing. For diameter $d$ and radius $r$ the darker of the two ring samples is kept,
$$m_{d,r} = \min\!\big(I(\mathbf{p} + r\mathbf{u}_d),\; I(\mathbf{p} - r\mathbf{u}_d)\big),$$
so that a one-sided edge, where one end of the diameter stays dark, produces a small $m_{d,r}$. The six values are sorted, $m_{(1),r} \le \dots \le m_{(6),r}$, and the ring brightness is a robust weighted order statistic that still tolerates one dark neighbor,
$$S_{\text{ring}}(\mathbf{p}) = \max_{r \in \mathcal{R}} \big[0.45\, m_{(2),r} + 0.55\, m_{(3),r}\big].$$
The response $S_0$ grows with the positive contrast between this ring brightness and the $3 \times 3$ smoothed center intensity and with the relative darkness of the core (the exact expression, a product of the absolute contrast, the relative contrast, and a squared dark-core ratio, is in the released source), and is normalized per volume. Candidate centers are the local maxima of $S_0$ after non-maximum suppression. Stage 0 takes 0.41 s per full volume on average on an 8-core CPU (0.12 s on standard $320 \times 256$ scans and 0.59 s on high-resolution $640 \times 416$ / $768 \times 768$ scans); the exact weights are in the released source code.

![**Stage 1 and Stage 2 in detail.** **(a)** Stage 1 takes the normalized T2W volume and the Stage 0 maps into an anisotropic 3D residual U-Net with squeeze-and-excitation blocks, which predicts five outputs: tubular mask, centerline peak, two direction components, and obturator. **(b)** Stage 2 traces tracks from the seeds in native code, scores their nodes and edges with a 4-layer triplet message-passing network, then scores every trajectory in the context of the whole bundle with a graph transformer (two edge-biased attention layers and one global self-attention layer) and smooths it with a cubic polynomial.](../reports/figures/fig2_architecture_modules.pdf){#fig:arch width=100%}

## Stage 1: Multi-Head 3D U-Net with Implant-Region Gating

Resampling the scan to 1 mm isotropic voxels, as most pipelines do, blurs a 0.3 mm in-plane catheter hole by a factor of three and destroys its boundary. Working on the native grid instead, a standard $3 \times 3 \times 3$ convolution mixes information over up to ten times more distance along $z$ than in-plane. Stage 1 (NeedleUNet3D, Figure \ref{fig:arch}a) therefore operates on the native grid, starts with a $1 \times 3 \times 3$ in-plane residual block, and continues with $3 \times 3 \times 3$ residual blocks with squeeze-and-excitation attention (24, 48, 96, and 144 channels). Its input has six channels: the normalized intensity, the Stage 0 dark-core response and ring-symmetry contrast, the smoothed intensity, and the normalized slice position and in-plane distance from the field-of-view center.

The network predicts four outputs:

1. **Tubular mask** $P_{\text{mask}}$: binary cross-entropy with a positive weight of 12 plus soft Dice on the tubular label around each reference trajectory.
2. **Centerline peak** $P_{\text{peak}}$: a sharp 2D Gaussian ($\sigma = 0.70$ mm) at the sub-pixel intersection of each catheter with the slice, trained with a mean-squared-error loss up-weighted 25-fold at the peaks. Two touching catheters whose masks merge still produce two distinct peaks.
3. **Direction field** $(V_x, V_y) = (\Delta x/\Delta z, \Delta y/\Delta z)$: the local slope of the trajectory, trained with a mean-squared-error loss inside the mask.
4. **Obturator channel** $P_{\text{obt}}$: a dedicated segmentation of the large central obturator and tandem, so that its lumen and rim do not spawn false needles.

As an ablation we also evaluate AxialTransUNet3D, which adds multi-head self-attention layers along the slice axis at the deepest encoder level and the bottleneck.

## Stage 2: From Slice Detections to a Coherent 3D Bundle

Stage 2 answers two questions: which detections on neighboring slices belong to the same catheter, and which of the resulting 3D curves are real.

### Stage 2A: Native Seed Tracking and Triplet Message Passing

On each slice, local maxima of $P_{\text{peak}}$ and of $S_0$ become nodes with a 16-dimensional feature vector (normalized position, $P_{\text{mask}}$, $P_{\text{peak}}$, $S_0$, local $3 \times 3$ intensity statistics, predicted slope, and obturator proximity). Directed edges connect nodes across 1 to $\max(4, 24\text{ mm}/\Delta z)$ slices. An edge is kept when its chord slope is at most 0.56 and its in-plane displacement lies within $2.8 + 0.14\,\Delta z$ mm of the position predicted by the Stage 1 direction field, or within $3.5 + 0.22\,\Delta z$ mm outright. Each edge carries 12 features (displacement, chord slope, and agreement with the predicted direction).

A single pairwise message cannot see a kink, because a kink involves three nodes, although stacked layers accumulate wider context. NeedleSliceMPNN is a 4-layer triplet-collinearity message-passing network (hidden size 64). Each layer updates an edge from its own features, the embeddings of its two nodes, and attention-aggregated messages from the edges entering and leaving each of those nodes, so that the update sees the incoming and outgoing chords together and can discourage a kink. The network outputs refined sub-pixel offsets $(\delta x_i, \delta y_i)$ and node and edge probabilities.

Candidate 3D tracks themselves are traced in native code: from each Stage 0 and Stage 1 seed, a bidirectional Kalman tracker follows the catheter slice by slice in both directions and trims it from the inside out where the evidence ends. Tracks that run within 1.65 to 2.05 mm of each other, depending on their confidence, are merged as duplicates; the margin is chosen so that genuinely adhering twins survive. The surviving tracks, scored by the message-passing network, are passed to Stage 2B.

### Stage 2B: Bundle Graph Transformer, Synthetic Pre-Training, and Smoothing

After Stage 1, recall is above 92% (96% on long shafts), but each patient still carries about ten false tracks along vessels and obturator edges, and needle-free pelvic scans produce about twenty; Stage 2A removes roughly half of them. Real catheters have a property these false tracks lack: they form a *bundle*. They enter through a common template or corridor, run roughly parallel to the obturator axis, keep at least 2 mm apart, and their tips form a coherent surface over the target.

Each candidate trajectory becomes a 26-dimensional token (length, linear-fit residual, 75th-percentile $S_0$ and $P_{\text{mask}}$, $P_{\text{peak}}$, $P_{\text{obt}}$, the three message-passing probabilities, agreement between models, the deviation of its entry position and slope from the bundle median, and the contrast gradient at its cranial end), together with 8 pairwise features between trajectories (minimum 3D clearance, mean in-plane separation, angular divergence, $z$-overlap, and duplicate indicators). The TipAwareBundleGraphTransformer applies two edge-biased attention layers over neighboring trajectories and one global self-attention layer over the whole bundle. It outputs three logits (a recall-oriented score, a strict one-to-one score, and a long-shaft one-to-one score), combined with weights 0.45 and 0.55 into a confidence $p_m$, and one cranial tip offset $\Delta z_m$ that trims or extends the deep end. An implant-region gate inspired by the IRGM of Xiao *et al.* [@xiao2026] is applied at this point: whereas they train a separate region branch, ours is geometric. The implant center is the median in-plane position of the candidates with confidence of at least 55%, the implant radius is the 98th percentile of their distances from that center plus 16 mm, and a candidate outside this radius is kept only if its confidence exceeds the threshold by more than 22 points. Trajectories with $p_m \ge \tau$ (default $\tau = 50\%$) are kept, their cranial tips refined with the predicted offset and the local contrast drop-off, and each is smoothed with a cubic polynomial in $z$ (Poly3) to remove the voxel staircase across thick slices.

Annotated clinical scans are scarce, and rare configurations such as crossing catheters are rarer still. We therefore built **SynthBrachy3T**, a generator of synthetic candidate-trajectory bundles with the same 26 node and 8 edge features as real candidates. Each of the 300 bundles contains 12 to 23 true long shafts arranged around a common axis, of which 30% receive an adhering twin 1.65 to 2.75 mm away, 2 to 8 short obturator tracks, 3 to 11 split duplicates of true shafts, and 4 to 14 vessel-like false tracks, with simulated tip-tracking noise. SynthBrachy3T does not simulate MRI voxels; it supplies the bundle-level reasoning task with far more adhesion and tip configurations than any clinical dataset contains. Stage 2B is pre-trained on these bundles, then fine-tuned on the clinical bundles of the fold's training patients together with five synthetically augmented copies of each, and its scores are blended with those of a transformer trained on clinical bundles only (weight 0.60 on the synthetic model). For the long-shaft rows the confidence threshold is lowered to 42.5% and the long-shaft logit is weighted more heavily.

\FloatBarrier

# Experimental Setup

## Cross-Validation Protocol and Baselines

Every learned component and every baseline is trained and evaluated with patient-level 5-fold cross-validation on the 54 patients, so no label of a test patient is used to train the model that is evaluated on it. Two qualifications apply. The training epoch of each Stage 1 network was selected on its held-out fold, which can bias the Stage 1 maps slightly in its favor. And the Stage 2 networks of a fold are trained on Stage 1 maps of the other patients, which were produced by Stage 1 models whose training folds included the fold's test patients; the stacked pipeline is therefore out-of-fold with respect to labels but not fully nested with respect to upstream features. We re-implemented three published approaches on the same splits:

- **PMB 2019: 3D U-Net + DBSCAN** [@zaffino2019]. Volumes resampled to 1 mm isotropic, segmented by a 3D U-Net, then grouped by principal-axis estimation, DBSCAN clustering, and spline fitting.
- **ResUNet + clDice + skeleton** [@isensee2021; @shit2021]. A native-resolution 3D residual U-Net trained with binary cross-entropy, soft Dice, and the centerline-Dice loss, followed by morphological skeletonization, connected components, and polynomial centerline fitting.
- **Xiao 2026 Sync-Track** [@xiao2026]. Implant-region gating, direction-constrained 3D Hough and PCA initialization, synchronous inward tracking, and cubic smoothing. As in their paper, we run it both on our out-of-fold segmentation masks and on the reference masks (an oracle upper bound).

## Metrics and Statistics

Predicted and reference trajectories are matched one-to-one with the Hungarian algorithm (SciPy `linear_sum_assignment`) using a 3.5 mm in-plane cutoff over the slices they share. We report:

1. **Detection**: recall, precision, and F1 under this strict one-to-one matching.
2. **Shaft error** $E_{\text{shaft}}$: the in-plane distance between the two curves at 40 evenly spaced positions along their shared length (pairs with a mean distance above 8 mm are treated as unmatched), summarized by mean, standard deviation, median, interquartile range, and 95th percentile.
3. **3D curve distance** $D_{\text{avg}}$ and **95th-percentile Hausdorff distance** $\text{HD}_{95}$, computed symmetrically on 50 resampled points per curve.
4. **Tip error** $E_{\text{tip}}$: the 3D distance between the predicted and reference deepest (cranial) tip.

Following the GEC-ESTRO and AAPM guidance on reporting brachytherapy uncertainties [@kirisits2014], reconstruction discrepancies are reported as distributions (median, interquartile range, and 95th percentile) and separated from observer variation. Differences between methods are tested over the 54 patients with paired two-sided Wilcoxon signed-rank tests, and 95% confidence intervals come from patient-level bootstrap resamples (1,000 for the headline rows; 200 to 500 for the threshold sweep, subgroup, and few-shot rows).

## Computing Environment, Software, and Code

Training and the 5-fold benchmarks ran on one Linux workstation (Arch Linux) with a 24-core AMD Ryzen Threadripper PRO 7965WX, 256 GB of RAM, and two NVIDIA GPUs (GeForce RTX 4090, 24 GB, and RTX 4070 Ti SUPER, 16 GB), The benchmark, graph-network training, and few-shot experiments ran with Python 3.13.2, PyTorch 2.11.0 (CUDA 13.0), SciPy 1.16.3, and NumPy 2.4.3; the Stage 1 networks were trained on both GPUs in parallel and exported to ONNX in a separate environment with Python 3.12.9, PyTorch 2.4.1 (CUDA 12.1), SciPy 1.16.1, and ONNX 1.23.2, the PyTorch version being the producer recorded in the model files. The Stage 0 kernel is written in Rust (crate `needlefinder_rust`, multi-threaded with the standard library, exposed to Python through a C interface and `ctypes`, compiled in release mode with link-time optimization). Hungarian matching uses SciPy. Figures were produced with Matplotlib.

For clinical use the trained Stage 1 and Stage 2B networks are exported to ONNX and run with ONNX Runtime inside an extension for 3D Slicer 5.10 [@fedorov2012], the open-source platform that also hosted the earlier NeedleFinder modules [@mastmeyer2017; @zaffino2019]; the Stage 2A weights are stored as NumPy archives. No GPU is required. The runtimes reported in the Results were measured on an 8-core desktop CPU and, for the GPU figures, on the RTX 4090.

The 3D Slicer extension, the Rust kernel, and the exported models are available at https://github.com/needlefinder/NeedleFinder. The training scripts (Stage 1 U-Nets, graph networks, and all baselines), the benchmark and statistics code, the SynthBrachy3T generator, the prediction export used for Table \ref{tbl:interobserver} and the qualitative figures, and the figure and table generators are collected in the repository's `vault/paper/benchmarks` directory and will be made public on publication. See also the Data and Code Availability statements at the end of the paper.

\FloatBarrier

# Results

## Accuracy Against Published Methods

\input{tables/tab2_main_results.tex}

Table \ref{tbl:main} is the main result. Part A compares all methods on the whole cohort and on the severe-adhesion subgroup; Part B reports ablations on the same folds. Figure \ref{fig:raincloud} shows the distribution of per-catheter errors.

![**Per-catheter error distributions on the whole cohort** (54 patients; each method's matched catheters). (a) In-plane shaft error and (b) 3D tip error. Each raincloud shows the density (top), the median and interquartile range (white box), the 5th to 95th percentile (capped bar), the mean (white diamond), and individual catheters (points). Numbers at the right give median / 95th percentile in mm. In (a) the grey band and dashed line mark the human inter-observer interquartile range (0.34 to 0.85 mm) and 95th percentile (2.05 mm); in (b) the dashed line marks the human inter-observer median tip discrepancy (3.20 mm).](../reports/figures/fig3_raincloud_distributions.pdf){#fig:raincloud width=100%}

Four findings stand out.

**Voxel segmentation alone over-fragments.** The ResUNet + clDice baseline finds almost every catheter (90.4% recall) and is accurate where it matches (0.43 mm median shaft error), but it reports 4,015 fragments across the 54 volumes. Its precision is 23.9% and its F1 37.9%. Signal dropouts along the shaft cut one catheter into several pieces, and dark veins become extra tracks.

**Structure-aware tracking needs a clean mask.** On the reference masks (Table \ref{tbl:main}, Part B, last row), the Xiao 2026 tracker achieves 97.4% recall, 89.7% F1, and 0.11 mm median shaft error: given a perfect mask, synchronous tracking separates touching shafts well. On our out-of-fold masks its F1 drops to 53.7% (precision 38.9%), because vessels inside the implant region still become tracks and nothing rejects them. Tracking alone does improve on skeletonization, by 15.9 F1 points on the whole cohort.

**NeedleFinder 2026 has the highest F1 of the evaluated configurations.** At the pre-specified threshold $\tau = 50\%$ it reaches 91.3% recall, 85.3% precision, and 88.2% F1 (88.4% at $\tau = 55\%$; 94.7% recall at $\tau = 25\%$). The gains over the other methods are all significant: 2.9 F1 points over our own pipeline without synthetic pre-training ($p = 6 \times 10^{-5}$), 6.9 points over PMB 2019 ($p = 1 \times 10^{-5}$), 34.5 points over Xiao 2026, and 50.3 points over ResUNet + clDice (both $p < 10^{-9}$). Median tip error falls to 2.61 mm, from 3.35 mm (PMB 2019), 6.02 mm (Xiao 2026), and 2.78 mm (our pipeline without synthetic pre-training; $p = 5 \times 10^{-4}$). Median 3D curve distance is 1.03 mm and median $\text{HD}_{95}$ is 3.44 mm. The tip distribution has a heavy tail that the median hides: 20% of matched catheters have a tip error above 10 mm and the 95th percentile is 36 mm, almost always where a track stops early or runs on past the true end along the catheter axis. Across the cohort the method misses 93 reference trajectories and reports 167 unmatched ones, about 1.7 misses and 3.1 extra tracks per patient. The median shaft error is 0.46 mm on all matched catheters and 0.45 mm on the 759 long shafts, on which recall is 92.5% at the default threshold and 96.4% at $\tau = 25\%$.

**The advantage holds when catheters touch.** On the 23 severe-adhesion patients (571 catheters), NeedleFinder 2026 reaches 89.7% F1 (93.4% recall, 86.3% precision) with 0.45 mm median shaft error and 2.98 mm median tip error, compared with 85.9% F1 for PMB 2019, 63.1% for Xiao 2026, and 47.8% for ResUNet + clDice.

## Contribution of Each Stage and Learning From Few Patients

![**(a) Precision versus recall** for each pipeline stage, each baseline, and the NeedleFinder 2026 threshold sweep on the whole cohort. Dashed curves are lines of constant F1; the star marks the default threshold $\tau = 50\%$. **(b) Accuracy versus number of annotated patients per fold used to fine-tune the bundle-scoring stage**, $k \in \{0, 1, 5, 10, 20, 43\}$, with and without SynthBrachy3T pre-training; the $k = 0$ point uses no learned bundle scoring. Shaded bands are the patient-level bootstrap 95% confidence intervals of F1.](../reports/figures/fig4_pareto_and_fewshot.pdf){#fig:pareto width=100%}

Figure \ref{fig:pareto}(a) shows what each stage adds. Stage 1 alone has high recall (92.8%) but only 62.6% precision. Stage 2A raises precision by 15.1 points and tightens median shaft error to 0.41 mm. Stage 2B with implant gating and cubic smoothing reaches 85.3% F1. Pre-training Stage 2B on SynthBrachy3T adds a further 2.9 F1 points at the default threshold, to 88.2%.

Figure \ref{fig:pareto}(b) asks how many annotated patients are really needed. The budget $k$ applies to the bundle-scoring Stage 2B, which is pre-trained on SynthBrachy3T and then fine-tuned on the bundles of $k$ clinical patients; Stages 1 and 2A keep their full training folds, so the curve measures how much bundle-level supervision the method needs, not how much annotation the whole pipeline needs. With no learned bundle scoring at all ($k = 0$), the analytical Stage 0 candidates with the geometric gate, duplicate merging, and cubic smoothing alone reach 75.1% F1. With a single annotated patient ($k = 1$) the fine-tuned Stage 2B reaches 82.0% F1, which is 3.9 points better than the same stage without synthetic pre-training and better than the PMB 2019 U-Net trained on all 43 patients of a fold (81.3%). Five patients give 83.7% F1, ten give 86.6%, and all 43 give 88.2%.

![**Two out-of-fold test cases.** (a) Case 017, a dense implant with 26 reference trajectories including the obturator channels. (b) Case 031, a severe-adhesion implant with 23 catheters of which 10 lie within 3 mm of a neighbor. Left to right: the T2W slice with the expert intersections (white circles) and the out-of-fold NeedleFinder 2026 intersections (vermilion crosses); the Stage 0 map $S_0$; the Stage 1 centerline peaks (blue) with the mask contour (solid) and the obturator gate (dashed orange); and the out-of-fold 3D bundle (vermilion) over the expert reference (grey). Case 017: 27 predicted trajectories for 26 reference trajectories; Case 031: 23 for 23.](../reports/figures/fig5_qualitative_mri_3d.pdf){#fig:qualitative width=100%}

Figure \ref{fig:qualitative} makes the stages visible on two difficult cases. Stage 0 lights up every round dark core in the pelvis, including vessels. Stage 1 suppresses those outside the implant gate and resolves adjacent catheters into separate peaks even where their masks merge. Stage 2 then produces smooth bundles with no swapped identities.

\FloatBarrier

## Agreement With Experts Compared With Agreement Between Experts

\input{tables/tab3_interobserver.tex}

Because a catheter is only a few pixels wide and slices are thick, two experts clicking centerlines never agree perfectly; inter-observer discrepancies of about 0.6 mm have been reported for breast implants, and more than a third of prostate needles differ by over 1 mm between observers [@kallis2019; @karius2024], and algorithms within that range are considered clinically equivalent to a second observer [@hrinivich2019; @andersen2020]. Table \ref{tbl:interobserver} measures this on the 40 doubly traced scans and compares it with the algorithm's agreement with each team.

The two human teams matched 464 trajectory pairs, 91.9% of CY's 505 trajectories and 90.5% of LB/AMe's 513, with a mean shaft discrepancy of 0.71 mm (median 0.49 mm, 95th percentile 2.05 mm) and a median tip discrepancy of 3.20 mm. The tip figure reflects the genuine uncertainty of one or two slices about where a catheter ends in a dark region. NeedleFinder 2026, tested out-of-fold against the primary team CY, matched 92.9% of CY's trajectories with a mean shaft error of 0.56 mm (median 0.44 mm, 95th percentile 1.25 mm). That is lower than the disagreement between the two human teams and cuts the 95th-percentile tail by 39%; its median tip error against CY (2.79 mm) is also below the human-to-human value. Against the second team LB/AMe, whose traces were never used in training, the algorithm matched 90.5% of trajectories with a mean shaft error of 0.76 mm (median 0.52 mm, 95th percentile 2.09 mm), within 0.05 mm of the human-to-human mean, median, and 95th percentile (its interquartile range is wider, 0.60 versus 0.51 mm). Only the tips differ: the median tip discrepancy against LB/AMe is 4.76 mm, because LB/AMe placed catheter ends differently from both CY and the algorithm. For the shafts, the algorithm is as close to either expert team as the teams are to each other, and its errors are of the same order as the differences between manual planning-system reconstruction and electromagnetic tracking measured in gynecologic implants (0.41 mm for shafts, 0.76 mm for tips) [@deufel2025].

## False Alarms on Scans Without Catheters, and Speed

\input{tables/tab4_specificity.tex}

A method deployed in the clinic must not invent catheters. We ran NeedleFinder 2026 on 48 public pelvic MRI scans that contain no catheters (Table \ref{tbl:specificity}, Figure \ref{fig:spec_runtime}a). Before Stage 2B, the candidate pool contains 20.5 false tracks per scan on average. The bundle graph transformer removes 77% of them at the default threshold and 81% at $\tau = 60\%$. Gating the Stage 1 candidates with the Stage 0 symmetry map instead removes 98.2% of them, leaving 18 false tracks across all 48 scans: 0.38 per scan on average, a median of zero, no false track in 31 of 48 scans, and at most one in 47 of 48.

![**(a) Fraction of needle-free scans with at most $x$ false tracks** on the 48 MSD prostate volumes. With the Stage 0 symmetry gate, 64.6% of scans have no false track and 97.9% have at most one. **(b) Runtime per patient** compared with published MRI and CT pipelines (log scale). **(c) Runtime by stage** on an 8-core desktop CPU (2.74 s mean) and on one RTX 4090 GPU (1.25 s mean).](../reports/figures/fig6_specificity_and_runtime.pdf){#fig:spec_runtime width=100%}

Speed follows from the design (Figure \ref{fig:spec_runtime}b, c). Stage 0 runs in multi-threaded native Rust in 0.41 s across the full volume, Stage 1 runs in 0.96 s on an 8-core CPU (0.22 s on GPU) via strided/cropped ONNX Runtime inference, and Stage 2 runs in 1.37 s on CPU (0.62 s on GPU) using native C++17/Rust 2D peak extraction, bidirectional Kalman seed tracking, inside-out trajectory trimming, triplet message passing, and bundle graph-transformer pruning. Full-volume uncached reconstruction takes 2.74 s on average on an 8-core desktop CPU (1.20 s on standard $320 \times 256 \times 28\text{--}40$ scans and 3.47 s on high-resolution $640 \times 416 \times 112\text{--}160$ / $768 \times 768 \times 64$ scans) and 1.25 s on a single GPU. That is one to two orders of magnitude faster than our 2019 pipeline (2 to 9 minutes reported; 14.8 s in our optimized re-implementation) [@zaffino2019] and 4.2 to 9.3 times faster than Xiao *et al.* (11.6 s) [@xiao2026].

\FloatBarrier

# Discussion

\input{tables/tab5_literature.tex}

## Where This Work Sits in the Literature

Table \ref{tbl:literature} places NeedleFinder 2026 among published catheter reconstruction studies on CT, ultrasound, and MRI. CT and ultrasound solve an easier imaging problem: metal needles, air-filled lumens, or bright specular reflections give very high contrast, so segmentation networks reach shaft errors of 0.1 to 0.7 mm on CT [@weishaupt2022; @xiao2026; @quetin2025] and 0.1 to 0.6 mm on ultrasound [@zhang2020pmb; @andersen2020]. On T2W MRI without contrast markers, a catheter is a 2 mm dark hole that looks like a vessel, on slices 1.2 to 6 mm thick; the best published MRI results are 0.74 to 0.97 mm shaft error on gynecologic implants, obtained with line markers or with CT-derived reference reconstructions [@shaaer2022; @kim2026] and 0.75 to 0.93 mm on prostate implants [@dai2020; @kostoulas2025]. NeedleFinder 2026 reaches a median shaft error of 0.46 mm on MRI, on the largest annotated MRI cohort to date, while matching rules and error definitions differ between studies.

The head-to-head comparison in Table \ref{tbl:main} explains why neither voxel segmentation [@zaffino2019; @isensee2021; @shit2021] nor mask-driven tracking [@xiao2026] is enough on MRI, and why the combination works. Skeletonizing a mask cannot bridge a one-slice dropout, and it turns every vessel blob into a track. Tracking bridges the gaps but still trusts the mask. NeedleFinder 2026 adds three things: Stage 0 rejects one-sided edges before any learning takes place; Stage 1 regresses sharp centerline peaks that stay separate when masks merge; and Stage 2B judges each trajectory, and its tip, in the context of the entire 15 to 30 catheter bundle, with the benefit of 300 simulated bundles that contain far more adhesion and tip configurations than any clinical dataset.

## What It Means in the Clinic, and Limitations

NeedleFinder 2026 is a one-click module in 3D Slicer [@fedorov2012] that runs in 1 to 3 seconds without a GPU. It changes the physicist's task from tracing 15 to 30 catheters, which takes tens of minutes, into a 1 to 2 minute check: click *Detect Needles*, inspect the 3D bundle and the axial intersections, and correct the occasional tip or partial track. A similar verification workflow has already saved about 15 minutes per case in ultrasound-guided prostate brachytherapy [@goulet2025]. Because the in-plane shaft error (0.46 mm median) is below the 0.49 mm median disagreement between experts (Table \ref{tbl:interobserver}), the shafts themselves rarely need editing.

Several limitations remain. First, tips are harder than shafts, here as in ultrasound [@wangf2020], and needles bend in situ by several millimeters depending on their length [@karius2026]. Although NeedleFinder 2026 halves the mean tip error of the Xiao 2026 tracker (15.6 to 7.9 mm) and its median tip error of 2.61 mm (2.98 mm with severe adhesion) is within one slice thickness, outliers occur when a tip ends inside a dark pocket at the uterine fundus or when a partial track leaves the imaged slab. Along-axis tip uncertainty is bounded by the slice thickness, and each millimeter of along-axis displacement changes the target D90 by 1 to 2.5% [@huang2026]. Thin-slice isotropic T2W sequences (SPACE or VISTA, 1 mm or less), zero-echo-time sequences that show empty catheters with sub-millimeter agreement to CT [@kaza2022; @lee2023], or combined T1W and T2W input should bring tip error below 1.5 mm. Second, all clinical scans come from one institution and one 3T scanner; geometric distortion of plastic applicators at 3T is small (0.2 to 0.75 mm) [@vanheerden2017], so accuracy is dominated by catheter identification rather than by the scanner. The 48 external MSD scans show that the method does not hallucinate on other scanners, but a prospective multi-center study across 1.5T, 3T, and low-field systems [@major2026] from several vendors is still needed to establish generalization. Third, the reference traces come from one annotation convention, so agreement with them is not the same as agreement with physical truth, and we have not measured the editing time or the dosimetric effect of the residual errors in a planning system; the 1 to 2 minute verification workflow is therefore an expectation, not a measurement. Fourth, the three published baselines are our own adaptations to this MRI data, not the authors' code, and the evaluation matches catheters on their shared length only, so a correctly placed but incomplete track counts as detected while its tip error is reported separately.

# Conclusion

NeedleFinder 2026 reconstructs interstitial brachytherapy catheters on clinical 3T T2W MRI with 88.2% F1, 0.46 mm median shaft error, and 2.61 mm median tip error, in 2.74 s on an 8-core CPU (1.25 s on a GPU). Its agreement with expert traces is closer than the agreement between two expert teams, its bundle-scoring stage reaches 82.0% F1 when fine-tuned on a single annotated patient thanks to synthetic pre-training, and it produces almost no false catheters on scans that contain none. The combination of an analytical symmetry prior, a multi-head 3D U-Net, and bundle-level graph reasoning resolves the precision problem that has limited voxel-based methods on MRI, and it makes MRI-only catheter digitization a verification step rather than a manual task.

\FloatBarrier

# Data Availability {-}

The 54 clinical MRI scans and their expert traces were acquired at Brigham and Women's Hospital under Institutional Review Board approval and contain protected health information; they cannot be shared publicly. De-identified trajectory annotations and per-patient evaluation results may be made available by the corresponding author on reasonable request, subject to institutional and ethics approval. The 48 needle-free scans are the public Medical Segmentation Decathlon prostate task (Task05_Prostate), available at http://medicaldecathlon.com [@antonelli2022]. The SynthBrachy3T generator is released with the code, so the 300 synthetic bundles can be regenerated from a fixed random seed.

# Code Availability {-}

The 3D Slicer extension, the Rust Stage 0 kernel, and the exported ONNX and NumPy models are available at https://github.com/needlefinder/NeedleFinder under an open-source license. Training, benchmark, statistical analysis, SynthBrachy3T, prediction-export, and figure-generation code is collected in the `vault/paper/benchmarks` directory of the same repository and will be made public on publication.

# Ethics Statement {-}

This retrospective study used imaging data acquired during routine clinical care under a protocol approved by the Institutional Review Board of Brigham and Women's Hospital / Mass General Brigham [protocol number to be inserted].

# Funding {-}

[Funding sources and grant numbers to be inserted.]

# Acknowledgments {-}

[To be completed: clinical staff of the AMIGO suite, annotators, computing resources.]

# Author Contributions {-}

[To be completed following the CRediT taxonomy.]

# Conflicts of Interest {-}

[The authors declare no conflicts of interest, or list them here.]

# References {-}
