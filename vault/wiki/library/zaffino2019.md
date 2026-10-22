---
title: "Fully automatic catheter segmentation in MRI with 3D convolutional neural networks: application to MRI-guided gynecologic brachytherapy"
key: zaffino2019
type: library
authors: "Zaffino P, Pernelle G, Mastmeyer A, Mehrtash A, Zhang H, Kikinis R, Kapur T, Spadea MF"
year: 2019
venue: "Physics in Medicine & Biology"
volume: "64(16)"
pages: "165008"
doi: "10.1088/1361-6560/ab2f47"
pmid: "31272095"
arxiv: "none"
url: "https://doi.org/10.1088/1361-6560/ab2f47"
verified: 2026-10-22
verified_via: "Crossref REST API + PubMed abstract (efetch) + PMC author manuscript full text (PMC6726393 via efetch db=pmc)"
bib_status: corrected
cited_in_manuscript: yes
---

## Summary
First fully automatic segmentation of multiple closely spaced brachytherapy catheters in intraoperative MRI. Data: 50 gynecologic cancer patients treated by MRI-guided HDR brachytherapy in the AMIGO suite of Brigham and Women's Hospital (IRB approved), one intraoperative 3 Tesla scan each (Siemens Magnetom Verio, 3D T2 FSE "SPACE" protocol), 826 catheters (10 to 35 per patient) manually segmented in 3D Slicer by a radiation physicist/oncologist. Method: a 4-level 3D U-Net trained five times on different patient subsets (ensemble, majority voting), followed by DBSCAN clustering to remove low-density noise and morphological removal of islands smaller than 50 voxels; 4-fold validation over all patients. Results: average distance error 2.0 +/- 3.4 mm, Dice 0.60 +/- 0.17, false-positive catheters 55/826 (6.7 %), false-negative catheters 12/826 (1.5 %); ensemble inference 9 +/- 2.5 min per MRI, single-model "light" version 2 +/- 0.5 min with 13.8 % false positives. Stated weaknesses: non-continuous catheter segmentations (raising Hausdorff errors) and dependence on a single scanner. Deployed in 3D Slicer via the DeepInfer module.

## Key facts for our paper
- 50 patients, 826 catheters, 10-35 catheters per patient; one intraoperative MRI per patient (Abstract; Sect. 2.1).
- 3 T Siemens Magnetom Verio, 3D T2/FSE SPACE, TR/TE 3000/160 ms; AMIGO suite, Brigham and Women's Hospital; IRB approval (Sect. 2.1).
- 3D U-Net, 4 levels, 60^3 output tile from 148^3 input; 5-network ensemble with majority voting; 4-fold cross-validation (Sect. 2.2-2.3).
- Post-processing: DBSCAN removal of low-density clusters, then removal of islands < 50 voxels (Sect. 2.3).
- Average distance error 2.0 +/- 3.4 mm; Dice 0.60 +/- 0.17 (Abstract; Results).
- False positives 55/826 = 6.7 %, false negatives 12/826 = 1.5 % (Results).
- Inference time: 9 +/- 2.5 min (ensemble); 2 +/- 0.5 min (single model, 13.8 % FP, 1.2 % FN) (Results).
- Available in 3D Slicer through the DeepInfer module (Abstract; Discussion).

## How our manuscript cites it
- "10 to 30 thin plastic catheters or needles are inserted ... [@mastmeyer2017; @zaffino2019; @chino2025]" (Intro). SUPPORTED: Zaffino reports 10 to 35 catheters per patient (Mastmeyer reports 14 +/- 7 per patient).
- "Finding the catheters on MRI is the hardest step of an MRI-only workflow [@zaffino2019; @sargazi2026]" (Intro). NOT CHECKABLE: the source states that catheter appearance in MRI is variable and is the technical challenge compared with CT, but it does not rank workflow steps.
- "Zaffino et al. trained an ensemble of 3D U-Nets to segment catheter voxels on 50 clinical T2W scans with 826 catheters, then clustered the voxels with DBSCAN and removed small islands" (Related work). SUPPORTED.
- "The primary cohort is 54 intraoperative 3T T2W turbo spin-echo MRI scans ... AMIGO suite at Brigham and Women's Hospital ... IRB approval [@pernelle2013; @mastmeyer2017; @zaffino2019]" (Data). SUPPORTED for 3 T, T2 FSE (SPACE), intraoperative, AMIGO, BWH, IRB. (The 54-scan count is ours, not the source's 50.)
- "PMB 2019: 3D U-Net + DBSCAN [@zaffino2019]" (Baselines). SUPPORTED.
- "3D Slicer ... in which the 2013 and 2019 NeedleFinder modules were also built [@pernelle2013; @zaffino2019]" (Implementation). SUPPORTED for the 2019 module (deployed in 3D Slicer via DeepInfer).
- "one to two orders of magnitude faster than our 2019 pipeline (2 to 9 minutes reported; 14.8 s in our optimized re-implementation) [@zaffino2019]" (Discussion). SUPPORTED: 9 +/- 2.5 min ensemble, 2 +/- 0.5 min single model.
- "neither voxel segmentation [@zaffino2019; @isensee2021; @shit2021] nor mask-driven tracking [@xiao2026] is enough on MRI" (Discussion). NOT CHECKABLE (our own result; the source itself names non-continuous segmentations as a weakness, which is consistent).
- Table 5 row "Zaffino et al. 2019 | 3T T2W | GYN IC/IS | 50 | 826 | 98.5 / 93.3(h) | 2.0 | -- | 2--9 min", note (h) "Cluster-level: 1.5 % false negatives and 6.7 % false positives among 826 catheters". 50 patients, 826 catheters, 3 T T2W, 1.5 % FN (-> 98.5 % recall), 6.7 % FP (-> 93.3 % precision), 2.0 mm average distance error, 2-9 min: SUPPORTED. "IC/IS": NOT CHECKABLE (interstitial catheters around an obturator are described; intracavitary component not stated).

## Bib corrections
- author: "Zhang, H." -> "Zhang, Hongtao" (Crossref/PubMed give Hongtao Zhang, Hebei General Hospital).
- Optionally add note: Spadea MF and Kapur T are joint last authors ("Both authors contributed equally").
All other fields (journal, volume 64, number 16, pages 165008, year 2019, DOI) verified.
