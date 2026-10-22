---
title: "The Medical Segmentation Decathlon"
key: antonelli2022
type: library
authors: "Antonelli M, Reinke A, Bakas S, Farahani K, Kopp-Schneider A, Landman BA, Litjens G, Menze B, Ronneberger O, Summers RM, van Ginneken B, Bilello M, Bilic P, Christ PF, Do RKG, Gollub MJ, Heckers SH, Huisman H, Jarnagin WR, McHugo MK, Napel S, Pernicka JSG, Rhode K, Tobon-Gomez C, Vorontsov E, Meakin JA, Ourselin S, Wiesenfarth M, Arbeláez P, Bae B, Chen S, Daza L, Feng J, He B, Isensee F, Ji Y, Jia F, Kim I, Maier-Hein K, Merhof D, Pai A, Park B, Perslev M, Rezaiifar R, Rippel O, Sarasua I, Shen W, Son J, Wachinger C, Wang L, Wang Y, Xia Y, Xu D, Xu Z, Zheng Y, Simpson AL, Maier-Hein L, Cardoso MJ"
year: 2022
venue: "Nature Communications"
volume: "13(1)"
pages: "4128"
doi: "10.1038/s41467-022-30695-9"
pmid: "35840566"
arxiv: "none"
url: "https://doi.org/10.1038/s41467-022-30695-9"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON (doi.org) + PubMed abstract (efetch) + Europe PMC full text XML (PMC9287542)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Describes the Medical Segmentation Decathlon (MSD), a challenge in which algorithms compete across ten segmentation tasks and several modalities to test whether a method that performs well on multiple tasks generalises to unseen tasks. Seven datasets (brain, liver, heart, hippocampus, prostate, lung, pancreas) were released in the development phase and three held out. The results confirmed the hypothesis, and the winner (nnU-Net) kept generalising to other clinical problems over the following two years. The prostate task consists of 48 multiparametric MRI studies (T2-weighted, diffusion-weighted and T1 contrast-enhanced series; challenge data: T2 and ADC), with peripheral zone and transition zone as targets, split 32 training / 16 test. Three conclusions: state-of-the-art algorithms generalise when retrained on unseen tasks; consistent multi-task performance is a strong surrogate for generalisability; training accurate segmentation models is now commoditised.

## Key facts for our paper
- Prostate task: 48 mp-MRI studies, T2 and ADC channels, targets PZ and TZ, 32 training / 16 test "4D volumes" (Table 1 of the paper; Methods, "Prostate").
- Prostate HD tolerance used in ranking: 4 mm (Methods, evaluation metrics).
- Data released under the challenge (medicaldecathlon.com is the challenge site named in the paper's data availability).
- The paper does not give voxel spacing per dataset.

## How our manuscript cites it
- "The public pelvic MRI scans we use for the false-alarm test come from the Medical Segmentation Decathlon [@antonelli2022]" (Related work). SUPPORTED.
- "we use all 48 pelvic T2W scans of the public Medical Segmentation Decathlon prostate task [@antonelli2022] (MSD Task05_Prostate; 0.60 to 0.75 mm in-plane, 3 to 4 mm slices)" (Data). "48 scans" and "T2W": SUPPORTED (48 studies, T2 channel). "Task05_Prostate" naming and "0.60 to 0.75 mm in-plane, 3 to 4 mm slices": NOT CHECKABLE from the paper (the folder name and voxel spacing come from the released data, not the publication).
- "The 48 needle-free scans are the public Medical Segmentation Decathlon prostate task (Task05_Prostate), available at http://medicaldecathlon.com [@antonelli2022]" (Data availability). SUPPORTED for 48 scans and the challenge site; folder name NOT CHECKABLE.
- Not in Table 5.

## Bib corrections
None. (Journal, volume 13, number 1, article 4128, year 2022, DOI verified.)
