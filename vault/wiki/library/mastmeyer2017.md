---
title: "Accurate model-based segmentation of gynecologic brachytherapy catheter collections in MRI-images"
key: mastmeyer2017
type: library
authors: "Mastmeyer A, Pernelle G, Ma R, Barber L, Kapur T"
year: 2017
venue: "Medical Image Analysis"
volume: "42"
pages: "173-188"
doi: "10.1016/j.media.2017.06.011"
pmid: "28803217"
arxiv: "none"
url: "https://doi.org/10.1016/j.media.2017.06.011"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON (doi.org) + PubMed abstract (efetch) + PMC author manuscript full text (PMC5654713 via efetch db=pmc)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Semi-automatic, model-based segmentation of collections of interstitial brachytherapy catheters in T2-weighted MRI of gynecologic cancer patients. Starting from a user click on the superior catheter tip, a ray search finds the best dark-line candidate inside successive 3D search cones (as in the 2013 Bézier method, which this paper explicitly augments rather than replaces); the search is guided by a catheter-specific angular-spring mechanical model, adapted from Goksel et al., in which the catheter is a chain of short rigid rods joined by rotational/torsion springs whose bending increases from tip to base. A final quality-control step removes outliers and conflicting trajectories. Data: 54 patients, 760 catheters (14 +/- 7 per patient, mean insertion depth 74 mm), acquired at 3 Tesla in the AMIGO suite at Brigham and Women's Hospital (2012-2016); references were drawn by a physician in 3D Slicer. Mean Hausdorff error 1.49 mm; 51 outliers deviated by more than two catheter widths (3.4 mm), giving 93 % identification accuracy in a Syed-Neblett template; 3-sigma precision 2.44 mm in a multi-user tip-perturbation simulation; about 3 s per catheter on a standard PC.

## Key facts for our paper
- 54 patients, 760 catheters, 14 +/- 7 catheters per patient; 3 Tesla, AMIGO suite, BWH, 2012-2016 (Abstract; Sect. 2 / Sect. 4.1 "Reference segmentations").
- Mean Hausdorff error 1.49 mm (Abstract).
- 51 catheters deviated > 3.4 mm (two catheter widths); identification accuracy 93 % (Abstract).
- Precision: 3-sigma maximum error 2.44 mm under simulated tip-click variation (Abstract).
- 3 s per catheter on a standard PC (Abstract).
- Initialisation: one manual click per catheter on the superior tip (Sect. 3, "Specify t0, the tip, manually by clicking on it in the MRI, once per catheter").
- Mechanical model: angular-spring model, rigid rods + rotational springs (Sect. 3.2); the Bézier/cone search of Pernelle 2013 is retained and augmented ("We used a mechanical model to augment the Bezier-model").
- Reference segmentations drawn with 3D Slicer (Sect. 4.1).
- Contrast between catheters and tissue described as "not sharp" in the T2 MRI (Sect. 2).

## How our manuscript cites it
- "10 to 30 thin plastic catheters or needles are inserted ... [@mastmeyer2017; @zaffino2019; @chino2025]" (Intro). SUPPORTED (14 +/- 7 catheters per patient in this source; 10-35 in Zaffino).
- "Mastmeyer et al. replaced the curve search with a mechanical rod-and-spring catheter model, reaching 1.49 mm mean Hausdorff distance on 760 catheters in 54 patients, still from a manual tip click" (Related work). 1.49 mm mean HD, 760 catheters, 54 patients, manual tip click, rod-and-spring model: SUPPORTED. "replaced the curve search": NOT SUPPORTED. The source keeps the cone/ray search and Bézier model and augments it with the angular-spring model ("We used a mechanical model to augment the Bezier-model"). Suggested wording: "constrained the cone search with a mechanical rod-and-spring catheter model".
- "The primary cohort is 54 intraoperative 3T T2W turbo spin-echo MRI scans ... AMIGO suite at Brigham and Women's Hospital ... IRB approval [@pernelle2013; @mastmeyer2017; @zaffino2019]" (Data). 54 patients, 3 T, T2W, AMIGO, BWH: SUPPORTED. "turbo spin-echo" and "IRB approval": NOT CHECKABLE in this source (sequence name and IRB statement not found in the retrieved text; Zaffino 2019 states both).
- Table 5 row "Mastmeyer et al. 2017 | 3T T2W | GYN IC/IS | 54 | 760 | 93(f) / -- | 1.49(g) | -- | 3 s/cath." with notes (f) "Fraction within 3.4 mm", (g) "Mean Hausdorff distance", and caption "semi-automatic (one manual click per catheter)". SUPPORTED in full. "IC/IS": NOT CHECKABLE (Syed-Neblett template interstitial implants; intracavitary component not stated).

## Bib corrections
None. (Journal, volume 42, pages 173-188, year 2017, DOI verified.)
