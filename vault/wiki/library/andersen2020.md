---
title: "Deep learning-based digitization of prostate brachytherapy needles in ultrasound images"
key: andersen2020
type: library
authors: "Andersén C, Rydén T, Thunberg P, Lagerlöf JH"
year: 2020
venue: "Medical Physics"
volume: "47(12)"
pages: "6414-6420"
doi: "10.1002/mp.14508"
pmid: "33012023"
arxiv: "none"
url: "https://doi.org/10.1002/mp.14508"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON + PubMed abstract (PMID 33012023) + PMC full text (PMC7821271)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
A 3D U-Net CNN was trained to find needles in transrectal ultrasound (TRUS) volumes from prostate HDR brachytherapy. Data: TRUS volumes from 1102 treatments (Örebro University Hospital, 2010-2019) with a clinical ground truth (CGT) of 24422 needles manually digitized by medical physicists; 713 treatments (17215 needles) were used for training and 389 treatments (7207 needles) for evaluation. Needle voxel predictions were combined into a straight-line fit per needle and compared with the CGT by root-mean-square distance (RMSD). Needle detection was 100%; median RMSD between AI and CGT was 0.55 mm (IQR 0.35-0.86). In a 188-needle subset re-digitized by a second physicist (G1), AI-vs-CGT RMSD was 0.52 mm versus 0.80 mm between CGT and G1 and 0.75 mm between AI and G1, so the AI performed as well as the clinical digitization relative to a second observer. Manual digitization took 10 min 11 s on average; AI time was negligible (< 2 s). Limitation: the network did not find needle tips (not trained to).

## Key facts for our paper
- Modality/site: TRUS, HDR prostate, steel needles assumed straight (Methods).
- Cohort: 1102 treatments; 24422 needles in total; evaluation on 389 treatments / 7207 needles (abstract; Methods).
- Method: 3D U-Net + least-squares line fit (not nnU-Net) (Methods).
- Median RMSD 0.55 mm (IQR 0.35-0.86) AI vs CGT; inter-observer CGT vs G1 0.80 mm (IQR 0.48-1.18) (abstract).
- Detection 100% (Results); tips not localized (Discussion).
- Manual digitization 10 min 11 s; AI < 2 s (Results).

## How our manuscript cites it
- Manuscript line 57, "... to transrectal ultrasound [@zhang2020pmb; @andersen2020; ...]": SUPPORTED.
- Manuscript line 201, "algorithms within that range are considered clinically equivalent to a second observer [@hrinivich2019; @andersen2020]": SUPPORTED (AI-CGT difference smaller than inter-observer difference; authors conclude performance "similar to that of a medical physicist").
- Manuscript line 223, "0.1 to 0.6 mm on ultrasound [@zhang2020pmb; @andersen2020]": SUPPORTED (0.55 mm median RMSD).
- Table 5 row "Andersén et al. 2020 | TRUS | Prostate | 1102 | 24422 | -- | 0.55 (c: RMS distance to clinical reconstruction) | -- | --": NOT SUPPORTED in one detail: the paper reports 1102 *treatments*, not patients, and the 24422 needles are the whole dataset while the 0.55 mm figure is on the 7207-needle evaluation set. Suggested fix: "1102 tx" (or a footnote "treatments") and "24422 (7207 evaluated)"; the recall column could legitimately show 100.

## Bib corrections
None.
