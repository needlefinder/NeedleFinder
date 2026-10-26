# Release Notes (`paper-media-2026-v1`)

Tag `paper-media-2026-v1` is the frozen code, model weight, and benchmark archive
release accompanying the 2026 NeedleFinder manuscript.

## Highlights

- **3D Slicer Extension (`NeedleFinder/`)**:
  - Four automated 3D needle/catheter tracking algorithms (`Kalman`, `Particle Filter`,
    `iGyne`, and `Minimum-Cost Path`), native C++17 / Rust acceleration
    (`NeedleFinderRust`), and two-stage hybrid deep-learning / bundle graph-transformer
    refinement (`NeuralNeedleNet`, `TipLocalizerModel`).
  - Exact parity (`< 2e-5` absolute tolerance) across PyTorch training models,
    exported ONNX production models, and NumPy fallback inference for Stage 1
    3D U-Net, Stage 2A MPNN, Stage 2B Tip-Aware Bundle Graph Transformer, and
    the five-tap 1D Tip Localizer (`dedicated_1d_tip_localizer.onnx`).
- **Complete 2026 Paper Benchmark Suite (`benchmarks/clean/` & `benchmarks/rerun/`)**:
  - End-to-end scripts for multi-rater consensus supervision (`40/40` dual-traced
    cases), 51-volume self-supervised 3D U-Net pre-training, strictly nested
    $5 \times 5$ cross-validation (`54` patients, `1,063` catheters), peer-reviewed
    baseline reproductions (`1–6`), literature fusion (`89.13%` $\rightarrow$ `89.38%` F1),
    zero-needle specificity on 48 public `MSD Task05_Prostate` 3T MRI scans,
    and clinical translation Experiments 1–4 (TG-43 dosimetry, split-conformal +
    Grubbs/TOST non-inferiority, zero-shot 3T prostate & 297 multi-sequence MRI
    transfer, and 4D serial insertion + paired pelvic CT).
- **Locked Artifacts & Automated Release Gates (`reproducibility/` & `tools/`)**:
  - Frozen 54-patient nested $5 \times 5$ splits (`reproducibility/splits-54.json`),
    SHA-256 lock (`reproducibility/artifacts.lock.json`) covering all 30 production
    models and archived benchmark outputs (`reproducibility/results/`), and six
    automated release gates (`fast`, `parity`, `coverage`, `clinical`, `slicer`,
    and `paper_reproduction`) verified by `tools/release.py check`.
