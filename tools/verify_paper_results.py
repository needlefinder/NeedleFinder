#!/usr/bin/env python3
"""Recompute summary statistics from archived per-case/per-needle JSON arrays.

This validates archived result consistency; it does NOT rerun training/inference.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vault/sources/2026-10-23-unified-clean-nested-benchmark.json'


def verify(source=SOURCE):
    data = json.loads(source.read_text())
    rows=[]
    for cohort, dist_key in [('cohort_all_1063_catheters_54cases','all_1063'),
                             ('cohort_long_759_cy_catheters_54cases','long_759')]:
        distributions=data['per_patient_and_needle_error_distributions'][dist_key]
        for method in distributions:
            summary = data[cohort][method]
            dist=distributions[method]
            shaft=np.asarray(dist['per_needle_mean_shaft_xy_mm'])
            tip=np.asarray(dist['per_needle_tip_3d_mm'])
            if len(shaft) != summary['hungarian_1to1_matched'] or len(tip) != len(shaft):
                raise ValueError(f'Matched-count mismatch: {method}')
            for values,key in [(shaft,'mean_shaft_xy'),(tip,'tip_3d')]:
                for stat,func in [('mean',np.mean),('median',np.median)]:
                    if not np.isclose(func(values),summary['metrics'][key][stat],atol=1.1e-4,rtol=0):
                        raise ValueError(f'Statistic mismatch: {method} {key} {stat}')
            tp=summary['hungarian_1to1_matched']; gt=summary['n_gt']; pred=summary['n_pred']
            for actual,expected in [(100*tp/max(1,gt),summary['hungarian_1to1_recall_pct']),
                                    (100*tp/max(1,pred),summary['precision_1to1_pct']),
                                    (200*tp/max(1,gt+pred),summary['f1_1to1_pct'])]:
                if abs(actual-expected)>0.0051: raise ValueError(f'Detection metric mismatch: {method}')
            rows.append(dict(cohort=dist_key,method=method,n_gt=gt,n_pred=pred,matched=tp,
                             recall=summary['hungarian_1to1_recall_pct'],precision=summary['precision_1to1_pct'],
                             f1=summary['f1_1to1_pct'],shaft_mean=float(shaft.mean()),tip_mean=float(tip.mean())))
    if not rows: raise ValueError('No archived distributions verified')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=verify();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Validated {len(rows)} archived method/cohort rows with available distributions; wrote {a.output}')
    print('Rows without archived distributions are not verified; this is not an inference/training reproduction.')

if __name__=='__main__': main()
