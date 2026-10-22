---
title: "Bridging the simulation-to-real gap for AI-based needle and target detection in robot-assisted ultrasound-guided interventions"
key: arapi2023
type: library
authors: "Arapi V, Hardt-Stremayr A, Weiss S, Steinbrener J"
year: 2023
venue: "European Radiology Experimental"
volume: "7(1)"
pages: "30"
doi: "10.1186/s41747-023-00344-x"
pmid: "37332035"
arxiv: "none"
url: "https://doi.org/10.1186/s41747-023-00344-x"
verified: 2026-10-22
verified_via: "Crossref + PubMed abstract + PMC full text (PMC10277269 via Europe PMC)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Arapi et al. address the lack of annotated clinical data for AI-based needle and target detection in robot-assisted, ultrasound-guided interventions by generating synthetic ultrasound training data from real preoperative 3D data (MRI or CT). A deep learning detection algorithm for needle tip and target anatomy was trained on the synthetic data only and validated on real in vitro ultrasound data acquired from a CIRS triple-modality abdominal phantom with the iSYS Micromate robot, a Clarius C3 scanner and an OptiTrack motion capture system for ground truth. The models generalized to unseen synthetic and in vitro data, reaching a reported mAP of 98% for needle detection at 30 frames per second, and a one-time US-robot calibration allowed the tracker to fine-position the robot toward the target from 2D US alone. The authors conclude the approach bridges the simulation-to-real gap, but the real-data validation is in vitro (phantom), not patient data. An erratum was published (Eur Radiol Exp 2023;7:36).

## Key facts for our paper
- Synthetic US generated from clinical preoperative MRI/CT; detection model trained on synthetic data only (Methods).
- Validation on real in vitro phantom US (CIRS 057A), 1.52 mm coated biopsy needle, Clarius C3 2-6 MHz (Methods).
- Needle detection mAP 98% at 30 frames/s in simulated scenarios (Discussion).
- Key point: "Models trained with synthetic US data generalize well to real in vitro US data" (Key points).

## How our manuscript cites it
- "synthetic-to-real training has been shown to transfer to real needle detection [@arapi2023]": SUPPORTED, with the caveat that the "real" data are in vitro phantom ultrasound, not in vivo. Suggested precision: "to real (phantom) ultrasound needle detection".

## Bib corrections
None.
