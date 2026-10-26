#!/usr/bin/env python3
"""Re-evaluate trusted repository OOF trajectories, without rerunning inference.

Only opens fixed, version-controlled pickle paths (pickle is executable input).
NumPy 2 archive names are mapped for the pinned NumPy 1 validation environment.
"""
import importlib.util
import io
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.release import sha256, source_fingerprint


class ArchiveReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda blob: torch.load(io.BytesIO(blob), map_location='cpu')
        return super().find_class(module, name)


def main():
    paths = [ROOT / 'vault/sources' / name for name in (
        '2026-10-23-cases-meta-54-clean.pkl',
        '2026-10-23-stage2-clean-final-oof-predictions.pkl',
        '2026-10-23-unified-clean-nested-benchmark.json')]
    before = source_fingerprint()
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    with paths[0].open('rb') as stream:
        metadata = ArchiveReader(stream).load()
    with paths[1].open('rb') as stream:
        predictions = ArchiveReader(stream).load()
    expected = json.loads(paths[2].read_text())['cohort_all_1063_catheters_54cases']
    spec = importlib.util.spec_from_file_location('metrics', ROOT / 'NeedleFinder/NeedleFinderLib/EvaluationMetrics.py')
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    gts = []
    for case in metadata:
        points = [np.asarray(g['pts'], dtype=np.float64) for g in case['gts_all']]
        gts.append([p[np.argsort(p[:, 2])] for p in points if len(p) >= 2])
    rows, errors = [], []
    for threshold in (50, 35):
        method = f'NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau={threshold}%)'
        preds = predictions[f'all_tau{threshold}']
        if len(preds) != len(gts) or len(gts) != 54:
            raise ValueError('Expected 54 aligned cases')
        actual, _ = metrics.evaluate_method_unified(preds, gts, n_boot=0)
        reference = expected[method]
        for key in ('n_cases', 'n_gt', 'n_pred', 'hungarian_1to1_matched', 'f1_1to1_pct'):
            if actual[key] != reference[key]:
                errors.append(f'{method}: {key}: {actual[key]} != {reference[key]}')
        for key in ('mean_shaft_xy', 'tip_3d'):
            for stat in ('mean', 'median'):
                if not np.isclose(actual['metrics'][key][stat], reference['metrics'][key][stat], atol=1.1e-4, rtol=0):
                    errors.append(f'{method}: {key}/{stat} differs')
        rows.append({'method': method, 'summary': actual})
    if before != source_fingerprint() or any(sha256(p) != hashes[str(p.relative_to(ROOT))] for p in paths):
        errors.append('Inputs changed during validation')
    report = {'status': 'failed' if errors else 'passed', 'scope': 'archived trajectories only; no training/inference rerun or confidence-interval validation',
              'case_order': 'metadata list order, as used by the archived exporter; predictions do not embed case IDs',
              'source_fingerprint': before, 'inputs': hashes, 'rows': rows, 'errors': errors}
    destination = ROOT / 'reproducibility/runs/evidence/archived-oof-report.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'errors': errors, 'report': str(destination)}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
