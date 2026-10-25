# Paper release candidate: test and reproducibility infrastructure

This candidate adds fast geometry/matching/leakage tests, synthetic native
tracking, model parity checks, CI, version-pinned validation dependencies,
artifact hashes, frozen patient splits, checkpoint provenance checks and release
manifests. A separate paper companion pins an exact core commit rather than
copying algorithms. Clinical validation now requires a supplied frozen baseline
and fails when required inputs are absent.

The new parity tests exposed a production defect: NumPy tip inference used only
three taps of five-tap trained convolutions. The implementation and tip ONNX model
were corrected and compared to the shared PyTorch training architecture. This is
a behavior change; archived paper performance is not automatically attributable
to this candidate. Other pre-existing MR/CT transfer changes and their model
assets are captured in the candidate rather than omitted from its source tag.

Release status: candidate only. Final publication requires successful coverage,
clinical baseline comparison, Slicer execution and full paper revalidation, then
an immutable release and version-specific DOI. No DOI or verified-paper label is
claimed by these notes. Training/inference results must be regenerated or reconciled
before final manuscript code-availability text is updated.
