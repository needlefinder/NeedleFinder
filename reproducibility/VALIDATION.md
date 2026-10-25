# Candidate validation — 2026-10-25

A fresh local Git clone (without untracked helpers, cached native libraries, or
clinical inputs from the source working directory) passed:

| Check | Result |
| --- | --- |
| Fast suite | 49 passed |
| Model parity suite | 16 passed |
| Artifact SHA-256/size verification | 28 artifacts verified |
| Archived summary recomputation | 26 method/cohort rows with distributions verified |
| Training-module import smoke check | Nested Stage 1 and Stage 2 modules imported |
| Source syntax check | No syntax errors in changed source/test/reproduction trees |

Fast tests compile/load the native core and detect two known synthetic shafts,
including world-coordinate invariance after an axis permutation. Parity tests
cover odd-kernel/dilated NumPy versus PyTorch convolution, NumPy versus ONNX MPNN
and tip-bundle inference, and NumPy/ONNX versus the actual PyTorch tip architecture.
The Stage 1 U-Net check verifies loadability, shape and finite output; it does not
compare the original PyTorch U-Net checkpoint, which is not available here.

An additional standard-library `trace` run passed all 65 non-clinical tests. Its
executed-line report was 100% for `DatasetSplits.py`, `EvaluationMetrics.py`, and
`TipLocalizerModel.py`; geometry was 22%, tracking 19%, and neural inference 13%.
These are **stdlib trace line estimates, not coverage.py branch measurements**,
and do not include C++/Rust native-code coverage. Much of the full application
remains untested. Branch coverage and its CI thresholds remain unverified locally.

Blocked gates:

- `coverage`: package absent; PyPI DNS unavailable. A version is pinned for CI.
- `clinical`: the four-case data root exists, but no independently reviewed frozen
  trajectory baseline was supplied. The test fails instead of skipping it.
- `slicer`: this installation supplies only Qt xcb; no display or Xvfb is available.
  Both offscreen and ordinary headless launch attempts failed before tests could run.
- `paper_reproduction`: not rerun. The corrected tip convolution can change
  predictions; archived paper metrics cannot be attributed to this candidate yet.
- Publication: shell GitHub DNS unavailable; the connector cannot access the
  configured 2026 repository (404). No remote branch/tag, release, protection
  setting, repository visibility, or Zenodo DOI was changed.

A fresh source checkout was validated using an existing local CPython 3.10.6
scientific environment. Installing the pinned dependencies into a clean environment
requires network access and remains unverified here. The main working tree also
contains concurrent manuscript/figure work, which is excluded from this candidate.

Machine-readable receipts and manifests are generated under
`reproducibility/runs/evidence/` and `dist/`; they are bound to source fingerprints.
Final paper publication must pass `python tools/release.py check`.
