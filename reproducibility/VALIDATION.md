# Candidate validation — 2026-10-26

Validation started from clean source commit `fcd223a` on `paper/media-lean` in
an isolated local clone. This update supersedes the October 25 validation status.

| Check | Result |
| --- | --- |
| Nonclinical tests, including model parity | 68 passed |
| Critical statement + branch coverage | Metrics 98.41%; dataset splits 100% (90% floors) |
| Whole configured Python source coverage | Approximately 30%; native code excluded |
| Clinical regression | Four MRI cases, 67 references; frozen prediction and orientation checks passed |
| Real Slicer smoke | Slicer 5.12.4: MRML transforms, VTK array/bounds, native core passed |
| Manuscript numeric registry | 1,317 keys verified; zero changed; strict text check passed |
| Archived OOF trajectory re-evaluation | All-catheter cohort, 54 cases, tau 50 and 35: counts, F1, shaft/tip means and medians match |
| Artifact SHA-256/size verification | 42 artifacts; previous 28 verified before extending inventory |

The clinical reference predictions are unchanged from commit
`5da83027ce77cab28ea86d3ca090e178c42d56a0`. The previous zero tolerance failed
on a 0.0000002403 mm numerical difference. The recorded comparison tolerance is
now 0.0001 mm and 0.0001 confidence percentage points. This is a numerical
regression check, not independent validation of clinical accuracy or 54 cases.
The original baseline did not record its generating environment.

Coverage uses pinned coverage.py 7.10.7 with its Python tracer. Its 43 Python
modules were retrieved from official upstream commit
`92a2af54e6bc948a9c536bd9b12bab70fb055904` and verified against Git blob hashes,
because shell package downloads are unavailable. Existing CPython 3.10.6 packages
were used; a new installation from the dependency lock remains untested.
Tracking has roughly 21% combined statement/branch coverage, so passing the two
critical-module floors must not be interpreted as comprehensive application coverage.

Run from the repository root:

```sh
python tools/validate.py fast
python tools/validate.py parity
python tools/validate.py coverage
python tools/validate.py clinical
python tools/verify_paper_results.py --output reproducibility/runs/evidence/archived-summary.csv
python tools/verify_archived_oof.py
python vault/paper/nfnumbers.py verify
python vault/paper/nfnumbers.py check --strict
python tools/release.py verify
python tools/release.py check
```

`nfnumbers.py` needs PyYAML; it was run with `/usr/bin/python3` here. Clinical
inputs default to `/tmp/needlefinder_benchmarks`; set `NEEDLEFINDER_TEST_DATA`
elsewhere. Validation receipts now hash the effective default clinical inputs and
the coverage measurement. Source fingerprints now include manuscript Markdown,
LaTeX, bibliography and CSV sources. The artifact inventory includes October 26
results, the clinical baseline and the two OOF pickle inputs.

This machine can run Slicer without a display using its installed Qt 5 plugin:

```sh
QT_QPA_PLATFORM=offscreen \
QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/qt/plugins/platforms \
/tmp/slicer/Slicer --no-main-window --no-splash --disable-cli-modules \
  --additional-module-paths "$PWD/NeedleFinder" \
  --python-script "$PWD/tests/slicer_smoke.py"
```

## Remaining release blockers

The full training/inference reproduction gate remains closed. Archived trajectory
re-evaluation does not reproduce training, inference, confidence intervals, every
paper experiment, or the current source's predictions on all 54 cases. Prediction
archives rely on the original exporter's metadata ordering and lack embedded case
IDs. The corrected convolution and newer CT/freehand changes require a complete
rerun with exact source and checkpoint provenance before final paper tagging.

The 54-case images/training caches and original checkpoints are on edelweiss.
This sandbox cannot open SSH sockets to that host; local GPU drivers are unavailable.
The configured private GitHub repository returns 404 through the connector, so
remote publication, protection settings and archival DOI creation remain undone.
No final paper tag was created and the existing RC1 tag was not moved.

The immutable source `2026-10-26-exp4-zero-config-ct-update.json` says that
`test_mr_ct_regression.py` verified 54 MRI cases. That test verifies **four** MRI
cases. Do not cite the source note as evidence of a 54-case current-code rerun.
The CT update also does not identify an exact generating source commit. Resolve
these provenance gaps from the original run or rerun; do not infer a commit.

Evidence is generated in `reproducibility/runs/evidence/`, bound to the source
fingerprint. Final acceptance still requires `python tools/release.py check` to
pass with a real full-paper reproduction receipt. A clean tree and a candidate
tag alone do not validate the paper.
