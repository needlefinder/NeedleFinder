# Paper release (`paper-media-2026-v1`): validated test and reproducibility infrastructure

This release (`paper-media-2026-v1`) provides fast geometry/matching/leakage unit
tests, synthetic native C++/Rust tracking verification, PyTorch/NumPy/ONNX model
parity checks, CI workflows, version-pinned validation dependencies, artifact
SHA-256 locks, frozen 54-patient splits (`reproducibility/splits-54.json`),
checkpoint provenance sidecars for all 25 nested Stage 1 3D U-Net checkpoints,
and release manifests. A separate paper companion repository pins the exact core
commit via a Git submodule rather than copying algorithms.

The parity test suite exposed and fixed a production defect in the Slicer NumPy
fallback (`NeuralNeedleNet._conv1d_same_np`), which previously used only three
taps of the five-tap trained 1D tip convolutions. Both the NumPy fallback and
exported `dedicated_1d_tip_localizer.onnx` now match the five-tap PyTorch
`Dedicated1DTipLocalizer` training architecture within `2e-5` absolute tolerance.
Live GPU reproduction on `edelweiss` (`NVIDIA GeForce RTX 4090`, PyTorch
`2.11.0+cu130`) confirmed that all 54-case nested CV paper benchmarks used the
five-tap PyTorch `Dedicated1DTipLocalizer` directly and reproduce every reported
manuscript metric identically (`89.13%` F1, `95%` bootstrap CI `[87.02%, 91.33%]`,
`0.5885 mm` shaft error, `6.2335 mm` mean / `2.265 mm` median 3D tip error;
`89.38%` F1 with literature fusion). Similarly, all 54 BWH clinical GYN MRI cases
satisfy `is_ct_hu == False` and `z_ijk_axis == 2`, so the zero-configuration pelvic
CT and multi-axis insertion guards do not alter the 54-case GYN MRI tracking path.

Release status: `paper-validated` (tag `paper-media-2026-v1`). All six release
gates (`fast`, `parity`, `coverage`, `clinical`, `slicer`, and `paper_reproduction`)
and `tools/release.py check` pass on the clean release tree. Note that
`paper_reproduction` executes live GPU Stage 1 inference/TTA parity and full
54-case live GPU Stage 2 + 1D-tip inference from the 25 provenance-verified
Stage 1 checkpoints rather than retraining all 25 3D U-Net checkpoints from
scratch. No external Zenodo DOI has been minted yet; cite tag
`paper-media-2026-v1` and its exact commit SHA.
