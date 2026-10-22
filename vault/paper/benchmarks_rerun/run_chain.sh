#!/bin/bash
# Re-run of the full NeedleFinder 2026 evaluation chain on the corrected out-of-fold maps (AUD-498).
set -o pipefail
PY=~/.pyenv/versions/3.13.2/bin/python
cd ~/github/needlefinder/vault/paper/benchmarks_rerun || exit 1
mkdir -p /tmp/rerun
run() { echo "=== $(date '+%F %T') START $1"; $PY "$1" > "/tmp/rerun/$2.log" 2>&1; rc=$?; echo "=== $(date '+%F %T') END $1 rc=$rc"; [ $rc -ne 0 ] && { echo "FAILED at $1"; tail -20 "/tmp/rerun/$2.log"; exit $rc; }; }
run evaluate_unified_apples_to_apples.py 01_evaluate
run run_xiao2026_and_fewshot_benchmark.py 02_xiao_fewshot
run run_synthbrachy3t_full.py 03_synth_full
run finalize_synthbrachy3t_benchmark.py 04_finalize
run export_nf_predictions_and_table3.py 05_export
run benchmark_public_pelvic_mri.py 06_specificity
echo "=== CHAIN DONE $(date '+%F %T')"
