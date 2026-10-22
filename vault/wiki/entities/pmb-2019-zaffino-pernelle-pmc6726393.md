---
title: PMB 2019 (Zaffino, Pernelle et al., PMC6726393)
type: entity
category: publication
created: 2026-10-21
updated: 2026-10-22
tags:
  - entity
  - publication
  - pmb
  - unet
  - deep-learning
---

# PMB 2019 (Zaffino, Pernelle et al., PMC6726393)

## Citation
- **Title**: *Fully automatic catheter segmentation in MRI with 3D convolutional neural networks: application to MRI-guided gynecologic brachytherapy*
- **Authors**: Paolo Zaffino, Guillaume Pernelle, André Mastmeyer, Alireza Mehrtash, H. Zhang, Ron Kikinis, Tina Kapur, Maria Francesca Spadea
- **Journal**: *Physics in Medicine & Biology*, vol. 64, no. 16, p. 165008, August 2019.
- **PubMed Central**: [PMC6726393](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726393/) | **PMID**: `31238315`
- **Source Code & Data on `mbp-2`**:
  - Preprocessing & LabelMaps: `/Users/guillaume/Dropbox/2016-paolo/` (`join_needles.py`, `preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/`)
  - Recovered TensorFlow 3D U-Net: `/Users/guillaume/Dropbox/Projects/unet/` (`3d-unet-tf-tiles-1class-multiple-gpus.py`, `unet.py`, `functions.py`, `train.txt`, `valid.txt`, `testing.txt`, `unet_trained/model.cpkt`)

## Method Summary
1. **Resampling**: Clinical 3T T2W MRI volumes and tubular labelmaps ($r = 1.5\text{ mm}$) were resampled to **isotropic $1.0 \times 1.0 \times 1.0\text{ mm}^3$** spacing.
2. **3D U-Net**: 4-level 3D encoder-decoder trained with Dice loss and $\pm 10^\circ$ rotation augmentation (published study used a 5-model ensemble in Theano/Lasagne on an NVIDIA Quadro P6000 24 GB GPU, plus a single "light" model variant).
3. **Post-Processing**: Majority voting across the 5-model ensemble + DBSCAN density clustering + small-island removal ($<50$ voxels). Note: individual catheter instance separation at crossings and channel identification were explicitly left as future work.

## Dataset Splits & Quantitative Results
- **Cohort**: 50 gynecologic brachytherapy patients (826 catheters annotated by radiation physicist `HZ`) selected from the 54-case BWH cohort (`Cases 002–078`), evaluated via 4-fold cross-validation.
- **Reported Accuracy (5-Model Ensemble vs. Light Single Model)**:
  - **Voxel Dice similarity coefficient (DSC)**: **$0.60 \pm 0.17$**
  - **3D Symmetric Mean Distance ($D_{\text{avg}}$)**: **$2.0 \pm 3.4\text{ mm}$** (Ensemble) | **$1.6 \pm 2.8\text{ mm}$** (Light model)
  - **3D Hausdorff Distance (HD / 95th percentile)**: **$15.9 \pm 20.5\text{ mm}$** (HD) | **$6.9 \pm 9.5\text{ mm}$** (HD95)
  - **False Positive (FP) rate**: **$6.7\%$** (Ensemble) | **$13.8\%$** (Light model)
  - **False Negative (FN) rate**: **$1.5\%$** (Ensemble)
  - **Inference Runtime**: **$9 \pm 2.5\text{ min}$** per patient (5-model Ensemble on P6000 GPU) | **$2 \pm 0.5\text{ min}$** (Light model)

## Related Pages
- [[wiki/research/historical-dataset-splits-2013-2026]]
- [[wiki/syntheses/2026-paper-roadmap-explainable-zero-shot-vs-3d-cnn]]
- [[wiki/research/2026-10-22-full-54case-unet-transformer-graphnet-benchmark]]
- [[sources/2019-pmb-pmc6726393-summary]]
