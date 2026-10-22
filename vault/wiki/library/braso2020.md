---
title: "Learning a Neural Solver for Multiple Object Tracking"
key: braso2020
type: library
authors: "Brasó G, Leal-Taixé L"
year: 2020
venue: "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
volume: "none"
pages: "6246-6256"
doi: "10.1109/CVPR42600.2020.00628"
pmid: "none"
arxiv: "1912.07515"
url: "https://doi.org/10.1109/CVPR42600.2020.00628"
verified: 2026-10-22
verified_via: "arXiv API (1912.07515v2, 'Accepted to CVPR 2020 (oral)'); Crossref DOI 10.1109/CVPR42600.2020.00628 (pp. 6246-6256)"
bib_status: new
cited_in_manuscript: yes
---

## Summary
Brasó and Leal-Taixé formulate multiple object tracking within tracking-by-detection as a graph problem and exploit the classical network-flow formulation to define a fully differentiable framework based on message passing networks (MPNs). By operating directly on the graph domain the method reasons globally over the entire set of detections and predicts the final association, showing that learning in MOT need not be restricted to feature extraction but can also be applied to the data association step. The approach yields significant improvements in MOTA and IDF1 on three public benchmarks, and the code is public.

## Key facts for our paper
- Message-passing network replaces greedy/combinatorial association in tracking-by-detection; global reasoning over all detections.
- Direct conceptual ancestor of our Stage 2A triplet message-passing linking of slice detections.

## Suggested use in our manuscript
Introduction, 'Previous Work', third generation paragraph, together with [@benhaim2022]. Suggested sentence: 'Outside brachytherapy, message-passing networks have replaced greedy linking in tracking-by-detection, reasoning globally over all detections to decide associations [@braso2020; @benhaim2022].'

## BibTeX
```bibtex
@inproceedings{braso2020,
  author        = {Brasó, G. and Leal-Taixé, L.},
  title         = {Learning a Neural Solver for Multiple Object Tracking},
  booktitle     = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages         = {6246--6256},
  year          = {2020},
  doi           = {10.1109/CVPR42600.2020.00628},
  eprint        = {1912.07515},
  archivePrefix = {arXiv}
}
```
