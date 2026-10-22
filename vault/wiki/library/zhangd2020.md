---
title: "Automatic segmentation and applicator reconstruction for CT-based brachytherapy of cervical cancer using 3D convolutional neural networks"
key: zhangd2020
type: library
authors: "Zhang D, Yang Z, Jiang S, Zhou Z, Meng M, Wang W"
year: 2020
venue: "Journal of Applied Clinical Medical Physics"
volume: "21(10)"
pages: "158-169"
doi: "10.1002/acm2.13024"
pmid: "32991783"
arxiv: "none"
url: "https://doi.org/10.1002/acm2.13024"
verified: 2026-10-22
verified_via: "Crossref + PubMed abstract + PMC full text (PMC7592978 via Europe PMC)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Zhang et al. propose DSD-UNET, a 3D convolutional neural network, for automatic segmentation of HR-CTV and organs at risk and for applicator reconstruction in planning CT for cervical cancer brachytherapy. The network was trained and tested on 91 patients for segmentation, and channel-path digitization of tandem-and-ovoid applicators was evaluated on 32 patients; applicator reconstruction combined DSD-UNET segmentation of the applicator components with 3D skeletonization and polynomial curve fitting. DSD-UNET outperformed 3D U-Net on all structures, with mean DSC of 86.9% (bladder), 82.9% (HR-CTV) and 82.1% (rectum). Applicator tube segmentation reached an average DSC of 92.1% and HD of 2.3 mm, and Hausdorff distances between automatic and manual channel paths were 0.88 ± 0.12 mm (intrauterine), 0.95 ± 0.16 mm (left ovoid) and 0.96 ± 0.15 mm (right ovoid). The method addresses intracavitary tandem-and-ovoid applicators, not interstitial needles.

## Key facts for our paper
- CT, cervical cancer, 91 patients (segmentation), 32 patients (applicator digitization) (abstract).
- 3D CNN (DSD-UNET) + skeletonization + polynomial curve fitting (abstract).
- Channel path HD 0.88 ± 0.12 / 0.95 ± 0.16 / 0.96 ± 0.15 mm (intrauterine / left / right ovoid) (abstract).

## How our manuscript cites it
- "Convolutional segmentation ... has since been applied to CT brachytherapy of the cervix, prostate, and breast [...; @zhangd2020; ...]": SUPPORTED (3D CNN segmentation on CT for cervix brachytherapy applicator reconstruction; a custom DSD-UNET rather than nnU-Net, consistent with the sentence's "often").

## Bib corrections
None.
