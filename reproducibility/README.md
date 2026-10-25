# Reproducing and releasing NeedleFinder

The core repository owns the algorithms. A separate paper companion pins a core
commit through a Git submodule; it contains no copied implementation. The local
checkout has two remotes: `origin` is the public extension and `private` is the
2026 working repository. Push to a named remote explicitly. Do not assume they
contain the same source or that a paper candidate belongs on the public default branch.

## Candidate status

This work fixes a real production tip-localizer bug: the NumPy convolution used
only three taps of the trained five-tap kernels. The corrected implementation and
ONNX export now match the PyTorch training architecture. This can change clinical
predictions. Existing archived paper results have not been relabeled as results
from the corrected inference. A final paper tag is blocked until clinical and
full-paper validation pass. The previous untracked ONNX export was retained locally
under `reproducibility/runs/original-models/` for comparison.

## Environment and quick checks

The test environment targets CPython 3.10.6, with exact dependency versions in
`requirements-test.lock` and `requirements-parity.lock`. The latter adds CPU
PyTorch. These locks describe the candidate validation environment, not a claim
about the original GPU training environment. They pin versions but not wheel hashes.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r reproducibility/requirements-parity.lock
.venv/bin/python tools/release.py verify
.venv/bin/python tools/validate.py fast
.venv/bin/python tools/validate.py parity
.venv/bin/python tools/validate.py coverage
.venv/bin/python tools/verify_paper_results.py \
  --output reproducibility/runs/archived-results.csv
```

`pytest.ini` confines discovery to real tests; historical research scripts named
`test_*` are not accidentally executed. Fast tests need a C++17 compiler or Rust
for the synthetic native-tracking integration. No clinical files or network are
required after dependencies are installed. Slicer GUI APIs are doubled in pytest;
the separate Slicer test uses real MRML/VTK and must run in its own process.

CI reports whole-source statement/branch coverage and enforces 90% floors for the
isolated matching metrics and split logic. These are initial requirements, not
claimed measured coverage. Coverage deliberately includes untested core modules;
there is no misleading whole-project 90% claim. Increase per-module floors as
geometry, tracking, inference, and UI tests expand.

The archive verifier recomputes detection scores and shaft/tip means and medians
for the 26 method/cohort rows with saved per-needle distributions. Other rows and
full training/inference are outside that check. It does not certify paper reproduction.

## Data and historical experiment paths

`reproducibility/paths.py` replaces hard-coded workstation paths in the clean and
rerun experiment scripts:

- `NEEDLEFINDER_WORK_DIR`: scratch caches and model outputs (default `reproducibility/runs/work`).
- `NEEDLEFINDER_DATA_DIR`: directory replacing the historical Dropbox root; for example it contains `GYN Cases`.
- `NEEDLEFINDER_TEST_DATA`: root containing the four clinical regression case directories.
- `NEEDLEFINDER_CLINICAL_BASELINE`: reviewed JSON trajectory baseline; absence fails the clinical gate.

Do not distribute clinical images merely because source code is licensed. Data
access and model/artifact permissions remain separate. Record input checksums in
the run's manifest. Existing NumPy-2 pickle archives need a NumPy-2 environment or
an explicitly verified conversion to a portable format; the candidate CPU test
lock is not suitable for loading every historical pickle.

`splits-54.json` freezes the archived patient ordering, 25 inner splits, seeds,
and the source metadata checksum. Stage 1 uses the tested split helper and writes
checkpoint provenance sidecars. Reusing a checkpoint with a different order,
seed, split or checksum fails. Historical checkpoints without sidecars are not
silently assigned fabricated provenance: retrain into a new output directory or
perform and document a separate provenance reconstruction.

## Full paper experiments

Run expensive experiments in a dedicated clone of the pinned commit, with the
correct GPU training environment, data and original checkpoint provenance. The
scripts below still require those external inputs; the CPU parity environment is
not a substitute for a validated GPU training lock. Several scripts write derived
JSON into their clone's vault, so never run the chain in the archival working copy.

The clean pipeline entry points, in order, are:

1. `vault/paper/benchmarks_clean/prepare_clean_dataset.py`
2. `vault/paper/benchmarks_clean/pretrain_ssl_unet3d.py`
3. `vault/paper/benchmarks_clean/train_nested_stage1.py`
4. `vault/paper/benchmarks_clean/run_stage2_and_tip_clean.py`
5. `vault/paper/benchmarks_clean/run_part1_peer_baselines_and_unc_maps.py`
6. `vault/paper/benchmarks_clean/run_part2_literature_fusion_inside_needlefinder.py`
7. `vault/paper/benchmarks_clean/export_clean_unified_and_table3.py`
8. `vault/paper/benchmarks_clean/benchmark_public_pelvic_mri_clean.py`
9. `vault/paper/benchmarks_clean/run_exp1_tg43_dosimetry.py` through `run_exp4_4d_insertion_and_ct.py`.

Use each script's documented inputs/options; do not infer missing training caches
from aggregate result files. Once validated, figures are built with
`python vault/paper/generate_pmb_figures.py`; the full manuscript build is
`bash vault/paper/build_paper.sh` (requires its Pandoc/TeX/fonts dependencies).

Re-export the tip model without retraining:

```bash
.venv/bin/python tools/export_tip_model.py \
  --output NeedleFinder/NeedleFinderLib/dedicated_1d_tip_localizer.onnx
```

Changing a model requires reviewing and refreshing `artifacts.lock.json`, running
parity and clinical checks, and issuing a new version. Never silently replace an
asset attached to a published paper release.

## Clinical and Slicer validation

Clinical baseline schema: `cases` maps the four case names to ordered lists of
`{"ras": [[x,y,z], ...], "confidence": number}`. Set explicit
`trajectory_atol_mm` and `confidence_atol`; record the baseline source commit,
input hashes and independent review in the baseline provenance. Do not create a
baseline from current predictions and call that a regression result.

```bash
NEEDLEFINDER_TEST_DATA=/path/to/cases \
NEEDLEFINDER_CLINICAL_BASELINE=/path/to/approved-baseline.json \
  .venv/bin/python tools/validate.py clinical

/path/to/Slicer --no-main-window --no-splash --disable-cli-modules \
  --additional-module-paths "$PWD/NeedleFinder" \
  --python-script "$PWD/tests/slicer_smoke.py"
```

Slicer needs a working display or Xvfb with the xcb plugin. `WITH_GENERIC_TESTS`
also registers the existing Slicer module self-tests in CTest; the smoke test does
not replace those full UI tests.

## Freezing and sharing

1. Verify from a fresh clone. Commit all required code, models/references, tests,
   locks and documentation. Ignore caches and bytecode.
2. Run checks using `tools/validate.py`. Receipts include source fingerprints and
   report checksums; failed, skipped, missing, or stale checks block release.
3. Generate `python tools/release.py manifest --output dist/candidate-manifest.json`.
   It includes full commit/tree IDs, source fingerprint, artifact hashes, environment
   and protocol checksum. It is labeled candidate until full validation exists.
4. `python tools/release.py check` requires fast, parity, clinical, Slicer, coverage,
   and full paper-reproduction evidence. The paper-reproduction receipt is supplied
   only after a reviewed full run, with the same receipt fields and a hashed report
   recording commands, data/checkpoint hashes, hardware, tolerances and comparisons.
5. Create an annotated immutable-version tag. Use `paper-media-2026-v1-rc1` for a
   candidate and `paper-media-2026-v1` only after all gates pass. Never move a paper tag.
6. Push the reviewed commit and tag to the intended repository. Enable GitHub
   release immutability, attach all assets to a draft, then publish. Require the
   `fast` and `parity-and-coverage` CI checks on the authoritative branch.
7. Archive the exact release with Zenodo and cite the version-specific DOI plus
   tag/SHA in the manuscript and `CITATION.cff`. A DOI has not yet been minted.

A tag identifies source; the manifest also identifies weights, inputs and outputs.
On another machine, check out the full SHA and run `tools/release.py verify` before
running experiments. Git pulls and filesystem syncing are not release identification.

Create a local separate paper companion after committing the core:

```bash
python tools/create_paper_companion.py --output dist/paper-companion
```

It pins `core` to the exact commit through a Git submodule and records the SHA in
`core-version.json`. Its default core URL is the existing 2026 remote; this commit
must be pushed there before other users can initialize it. The generator neither
publishes nor changes repository visibility.
