---
title: "Clinical Application of Deep Learning-Assisted Needles Reconstruction in Prostate Ultrasound Brachytherapy"
key: goulet2025
type: library
authors: "Goulet M, Duguay-Drouin P, Mascolo-Fortin J, Mégrourèche J, Octave N, Tsui JMG"
year: 2025
venue: "International Journal of Radiation Oncology, Biology, Physics"
volume: "122(1)"
pages: "199-207"
doi: "10.1016/j.ijrobp.2024.12.026"
pmid: "39800329"
arxiv: "none"
url: "https://doi.org/10.1016/j.ijrobp.2024.12.026"
verified: 2026-10-22
verified_via: "Crossref + PubMed abstract (full text paywalled: ScienceDirect/redjournal returned 403; OpenAlex/Unpaywall report no OA copy)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Goulet et al. evaluated the clinical feasibility and time saving of AI-driven automatic reconstruction of transperineal needles for ultrasound-guided HDR prostate brachytherapy planning. The study used 102 US-planned brachytherapy images from a single institution, split into 50 for training/validation, 11 for an accuracy test set, and 41 for a clinical-implementation set. Test-set accuracy was measured against the consensus of manual reconstructions by 5 medical physicists on post-treatment 3D-US scans; the mean error between AI and consensus needle positions was (0.44 ± 0.32) mm, with 95.0% of AI needle points within 1 mm of their human counterparts. In the clinical set the algorithm reached a true positive reconstruction rate of 93.4%, with 4.5% of those needles needing manual correction before dosimetry, and total AI-assisted reconstruction time was on average 15.2 min lower (P < .01) than before AI implementation. The abstract does not name the network architecture, the number of needles, or the number of distinct patients.

## Key facts for our paper
- Cohort: 102 US-planned brachytherapy images, single institution (50 train/val, 11 test, 41 clinical) (abstract).
- Mean AI-vs-consensus needle position error 0.44 ± 0.32 mm; 95.0% of AI needle points < 1 mm from human consensus (test set, 5 physicists) (abstract).
- True positive reconstruction rate 93.4% in the clinical set; 4.5% of those needles required manual correction (abstract).
- Mean total reconstruction time reduced by 15.2 min (P < .01) with AI assistance (abstract).
- Imaging: 3D-US, HDR prostate brachytherapy (abstract).

## How our manuscript cites it
- "Convolutional segmentation ... has since been applied ... to transrectal ultrasound [...; @goulet2025]": SUPPORTED for a deep-learning needle reconstruction on 3D ultrasound in prostate HDR; the abstract calls it "AI-driven auto-reconstruction" without naming a convolutional architecture or the probe type, so those details are NOT CHECKABLE from the abstract.
- "A similar verification workflow has already saved about 15 minutes per case in ultrasound-guided prostate brachytherapy [@goulet2025]": SUPPORTED (15.2 min average reduction, P < .01).
- Table 5 row "Goulet et al. 2025 | 3D US | Prostate | 102 | -- | 93.4 / -- | 0.44 | -- | --": 93.4% recall and 0.44 mm SUPPORTED (true positive rate in the 41-case clinical set; mean point error in the 11-scan test set). "Pts 102" NOT SUPPORTED as stated: the abstract reports "102 US-planned BT images", not 102 patients, and the number of distinct patients is not given. Suggested fix: "102 scans" (or footnote that 102 is the number of 3D-US images, of which 11 were used for accuracy and 41 for the clinical test). Modality "3D US" SUPPORTED.

## Bib corrections
None (Crossref renders the journal as "International Journal of Radiation Oncology*Biology*Physics"; bib form is equivalent).
