---
title: "Approaching automated applicator digitization from a new angle: Using sagittal images to improve deep learning accuracy and robustness in high-dose-rate prostate brachytherapy"
key: weishaupt2022
type: library
authors: "Weishaupt LL, Sayed HK, Mao X, Choo R, Stish BJ, Enger SA, Deufel C"
year: 2022
venue: "Brachytherapy"
volume: "21(4)"
pages: "520-531"
doi: "10.1016/j.brachy.2022.02.005"
pmid: "35422402"
arxiv: "none"
url: "https://doi.org/10.1016/j.brachy.2022.02.005"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON + PubMed abstract (PMID 35422402) + Europe PMC author list; no full text (not in PMC)"
bib_status: corrected
cited_in_manuscript: yes
---

## Summary
Automates segmentation of titanium interstitial needles on CT for HDR prostate brachytherapy. 57 retrospective interstitial prostate treatments were split into training (n = 27), validation (n = 10) and testing (n = 20); the CT volumes were reformatted into 2D sagittal slices and segmented with a 2D U-Net, followed by density-based linkage clustering to separate individual catheters in 3D and a polynomial fitting step to handle gold seeds and intersecting needles. On the 20 test patients with 353 needles, tip agreement was -0.1 +/- 0.6 mm (range -1.8 to 1.4 mm) and shaft agreement 0.13 +/- 0.09 mm (maximum 0.96 mm); all intersecting applicators were separated. Automated digitization took approximately 1 min per case.

## Key facts for our paper
- Modality/site: CT, HDR prostate, titanium needles (abstract).
- Cohort: 57 patients; test set 20 patients, 353 needles (abstract, Results).
- Method: 2D U-Net on sagittal reformats + density-based linkage clustering + polynomial fitting (abstract, Methods). Not nnU-Net.
- Shaft agreement 0.13 +/- 0.09 mm (max 0.96 mm); tip agreement -0.1 +/- 0.6 mm (range -1.8 to 1.4 mm) (abstract, Results).
- Run time: "averaged approximately 1 min" (abstract, Results).

## How our manuscript cites it
- Manuscript line 57, "Convolutional segmentation, often nnU-Net based ... applied to CT brachytherapy of the cervix, prostate, and breast [... @weishaupt2022 ...]": SUPPORTED (CT, prostate, 2D U-Net; the "often nnU-Net based" hedge does not apply to this paper, which uses a plain 2D U-Net).
- Manuscript line 223, "segmentation networks reach shaft errors of 0.1 to 0.7 mm on CT [@weishaupt2022; ...]": SUPPORTED (0.13 mm shaft agreement).
- Table 5 row "Weishaupt et al. 2022 | CT | Prostate | 57 | 353 (a: test-set catheters) | -- | 0.13 | -- | ~60 s": SUPPORTED (57 patients; 353 needles in the 20-patient test set; shaft 0.13 mm; ~1 min). Tip column could be filled with -0.1 +/- 0.6 mm (signed agreement) if desired.

## Bib corrections
- author: second author is "Sayed, Hisham Kamal" per Crossref publisher metadata (PubMed/Europe PMC give initials "Sayed HK"); bib has "Sayed, Hossam K." -> change to "Sayed, Hisham K.". Last author given as "Deufel, Christopher" (no middle initial) in Crossref/PubMed; bib "Deufel, Christopher L." is harmless.
