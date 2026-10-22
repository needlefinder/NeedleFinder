---
title: "Validation of Catheter Segmentation for MR-Guided Gynecologic Cancer Brachytherapy"
key: pernelle2013
type: library
authors: "Pernelle G, Mehrtash A, Barber L, Damato A, Wang W, Seethamraju RT, Schmidt E, Cormack RA, Wells W, Viswanathan A, Kapur T"
year: 2013
venue: "Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2013, Lecture Notes in Computer Science (Springer); PubMed: Med Image Comput Comput Assist Interv 16(Pt 3)"
volume: "LNCS 8151 (PubMed: 16(Pt 3))"
pages: "380-387"
doi: "10.1007/978-3-642-40760-4_48"
pmid: "24505784"
arxiv: "none"
url: "https://doi.org/10.1007/978-3-642-40760-4_48"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON (doi.org) + PubMed abstract (efetch) + PMC author manuscript full text (PMC4005335 via efetch db=pmc)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Semi-automatic segmentation of interstitial brachytherapy catheters in MRI: starting from a physician-provided distal tip (and a user-defined proximal template plane), the algorithm iteratively searches for the direction that maximises the likelihood of 1.6 mm-diameter signal voids inside a sequence of 3D search cones whose heights grow with a Fibonacci series, then fits a 5th-degree Bézier curve through the six control points. Data were acquired at 3 Tesla (Siemens) with 2D T2-weighted FSE (0.2x0.3x2.0 mm), 3D SPACE and 3D bSSFP sequences during MRI-guided catheter insertion. Validation on a gel-wax phantom gave a maximum Hausdorff distance (HD) of 1.5 mm (MR auto vs MR manual) and 3.5 mm (MR auto vs CT manual, mean 2.3 mm). On an IRB-approved retrospective database of 10 interstitial patients with 101 catheters, a segmentation was classified as correct if HD < 2 mm; 93/101 catheters were correct at 0.3 s per catheter (3 GHz Core i7, 16 GB RAM). The stated failure mode is a tip click that is not close enough to the shaft, after which the search can follow a neighbouring catheter.

## Key facts for our paper
- 10 patients, 101 catheters, 3 T MRI, acquired intraoperatively ("during MRI-guided insertion"); IRB-approved retrospective database (Sect. 3, "Patient MRI Results").
- 93/101 catheters (92.1 %) correct, criterion: symmetric Hausdorff distance < 2 mm (Sect. 3).
- 0.3 s per catheter on a 3 GHz Intel Core i7, 16 GB RAM, Mac OS X 10.7 (Sect. 3).
- Initialisation: user-provided catheter tip c0 plus a user-defined proximal plane P (Sect. 2, "Segmentation Algorithm Implementation").
- Search: successive 3D cones (heights lambda_i following a Fibonacci recursion, radii increasing), minimising the intensity line integral; 5th-degree Bézier curve through 6 control points (Sect. 2).
- Phantom: max HD 1.5 mm (MR-MR), 3.5 mm max / 2.3 +/- 0.5 mm mean (MR-CT) (Sect. 3).
- Sequences: 2D T2W FSE TR/TE 3000/120 ms, 0.2x0.3x2.0 mm; 3D SPACE 0.4x0.4x1.0 mm; 3D bSSFP 0.6x0.6x1.6 mm (Sect. 2).
- The paper does not mention 3D Slicer, the AMIGO suite, or Brigham and Women's Hospital by name (only the affiliation line and "IRB-approved").

## How our manuscript cites it
- "Our first work on the BWH cohort seeded each catheter with one click at its tip, searched slice by slice inside an expanding cone, and fitted a Bézier curve; 93 of 101 catheters in 10 patients were traced correctly at 0.3 s per catheter" (Related work). Tip click, expanding cone, Bézier, 93/101, 10 patients, 0.3 s: SUPPORTED. "slice by slice": NOT SUPPORTED. The source searches in 3D conic volumes with Fibonacci-increasing heights, not slice by slice. Suggested wording: "searched inside a sequence of expanding 3D cones".
- "The primary cohort is 54 intraoperative 3T T2W turbo spin-echo MRI scans acquired ... in the AMIGO suite at Brigham and Women's Hospital ... with IRB approval [@pernelle2013; @mastmeyer2017; @zaffino2019]" (Data). 3 T, T2W fast/turbo spin echo, intraoperative, IRB-approved: SUPPORTED by this source. AMIGO suite and BWH named: NOT CHECKABLE from this source alone (not named in the text; the co-cited zaffino2019 and mastmeyer2017 do name AMIGO/BWH).
- "3D Slicer [@fedorov2012], the open-source platform in which the 2013 and 2019 NeedleFinder modules were also built [@pernelle2013; @zaffino2019]" (Implementation). NOT SUPPORTED by this source: the 2013 paper never mentions 3D Slicer. (Zaffino 2019 refers to "the dedicated catheter reconstruction algorithm already developed and available in 3D Slicer", and Mastmeyer 2017 says references were drawn in 3D Slicer.) Suggested fix: cite zaffino2019 and/or mastmeyer2017 for the Slicer provenance of the 2013 module, or drop pernelle2013 from that bracket.
- Table 5 row "Pernelle et al. 2013 | 3T T2W | GYN IC/IS | 10 | 101 | 92.1 / -- | --(e) | -- | 0.3 s/cath." with note (e) "Maximum Hausdorff distance below 2 mm for 92 % of catheters" and caption "semi-automatic (one manual click per catheter)". 10 patients, 101 catheters, 92.1 % (= 93/101), HD < 2 mm criterion, 0.3 s/catheter, one manual tip click: SUPPORTED. "3T": SUPPORTED. "T2W": NOT CHECKABLE (three sequences were acquired; the paper does not state which one the 101 patient catheters were segmented on). "IC/IS": NOT CHECKABLE (interstitial with obturator described; intracavitary component not stated).

## Bib corrections
None required. Note: Crossref carries no volume number for this chapter; "volume = {8151}" is the LNCS volume of the MICCAI 2013 Part III proceedings book (ISBN 978-3-642-40760-4) and PubMed indexes it as 16(Pt 3):380-7. Author "Wang, W." is "Wang, Wei" in Crossref.
