---
title: "Iterative Hough Transform for Line Detection in 3D Point Clouds"
key: dalitz2017
type: library
authors: "Dalitz C, Schramke T, Jeltsch M"
year: 2017
venue: "Image Processing On Line"
volume: "7"
pages: "184-196"
doi: "10.5201/ipol.2017.208"
pmid: "none"
arxiv: "none"
url: "https://doi.org/10.5201/ipol.2017.208"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON (doi.org) + IPOL article page (abstract and BibTeX); full PDF not retrieved"
bib_status: ok
cited_in_manuscript: no
---

## Summary
Describes the application of the Hough transform to detecting lines in three-dimensional point clouds. Parameter quantisation uses a recently proposed Hough parameter-space regularisation. Voting is iterative: the line with the most votes is selected and its supporting points are removed at each step. To overcome the inaccuracy of the discretised parameter space, each line is re-estimated by an orthogonal least-squares fit over the candidate points returned by the Hough transform. The IPOL publication includes reviewed source code (BSD-2-Clause) and an online demo. No clinical data or quantitative accuracy figures are given in the abstract.

## Key facts for our paper
- Iterative 3D Hough transform for lines: select best line, remove its points, repeat (Abstract).
- Orthogonal least-squares refinement of each detected line (Abstract).
- Reference implementation available (IPOL, BSD-2-Clause, version 1.0, 2017-03-20).
- This is the Hough method that Xiao et al. 2026 adapt ("iterative 3D Hough transform based on Dalitz et al." with direction-constrained voting).

## How our manuscript cites it
- Not cited anywhere in manuscript.md (no [@dalitz2017]) and not in Table 5. The manuscript mentions the "direction-constrained 3D Hough" of Xiao et al. without citing Dalitz; if the Sync-Track baseline description wants to credit the underlying Hough algorithm, [@dalitz2017] could be added there. Otherwise the entry is unused.

## Bib corrections
None. (Journal, volume 7, pages 184-196, year 2017, DOI verified against the IPOL page's own BibTeX.)
