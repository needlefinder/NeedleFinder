---
title: "Graph Neural Network for Cell Tracking in Microscopy Videos"
key: benhaim2022
type: library
authors: "Ben-Haim T, Riklin Raviv T"
year: 2022
venue: "Computer Vision – ECCV 2022, Lecture Notes in Computer Science vol. 13681"
volume: "13681"
pages: "610-626"
doi: "10.1007/978-3-031-19803-8_36"
pmid: "none"
arxiv: "2202.04731"
url: "https://doi.org/10.1007/978-3-031-19803-8_36"
verified: 2026-10-22
verified_via: "Crossref + arXiv API (2202.04731v2) + Springer chapter page (WebFetch)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Ben-Haim and Riklin Raviv present a graph neural network (GNN) approach to cell tracking in high-throughput microscopy videos. The whole time-lapse sequence is modeled as a directed graph whose nodes are cell instances and whose edges are candidate associations; trajectories are extracted as maximal paths by solving an edge classification problem. Deep metric learning produces cell feature vectors that separate different cells, and a new GNN block enables mutual updates of node and edge features, so that message passing (whose extent is set by the number of GNN blocks) propagates information well beyond neighboring frames. Active edges are then used to construct tracks and lineage trees. On 2D and 3D datasets (C2C12 and five Cell Tracking Challenge datasets) the framework outperforms state-of-the-art methods on most datasets; code is public (github.com/talbenha/cell-tracker-gnn). The Springer page notes that inference applies simple rules (top two outgoing edges above 0.5, highest incoming edge), so final linking is not purely global.

## Key facts for our paper
- End-to-end GNN with message passing over a sequence-wide graph; edge classification yields tracks and lineages (abstract).
- Evaluated on C2C12 (16 sequences) and Cell Tracking Challenge datasets (PhC-C2DH-U373, Fluo-N2DH-SIM+, Fluo-N3DH-SIM+, Fluo-C2DL-Huh7, Fluo-N2DL-HeLa); ranked first on three CTC datasets (Springer page summary).
- ECCV 2022, LNCS 13681, pp. 610-626, first online 23 October 2022.

## How our manuscript cites it
- "Outside brachytherapy, message-passing networks have replaced greedy linking in tracking-by-assignment [@benhaim2022]": SUPPORTED in substance: the paper replaces frame-by-frame association with a message-passing GNN that classifies edges of a sequence-wide graph. Caveat: the paper itself does not use the phrase "tracking-by-assignment" or "greedy", so "have replaced" is the manuscript's generalization; softer wording such as "message-passing networks have been used instead of greedy frame-to-frame linking" would track the source more closely.

## Bib corrections
None required. Optional additions: `volume = {13681}`, `series = {Lecture Notes in Computer Science}`, `address = {Cham}`, `publisher = {Springer Nature Switzerland}`.
