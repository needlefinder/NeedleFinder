#!/usr/bin/env python3
"""Enforce explicit critical-module floors; report whole-core coverage separately."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    result=json.loads((ROOT/'coverage.json').read_text())
    targets=json.loads((ROOT/'reproducibility/coverage-policy.json').read_text())
    failed=[]
    for path,floor in targets['critical_modules'].items():
        value=result['files'].get(path,{}).get('summary',{}).get('percent_covered',0)
        print(f'{path}: {value:.2f}% (required {floor}%)')
        if value<floor:failed.append(path)
    print(f"Entire measured source: {result['totals']['percent_covered']:.2f}% (statements + branches)")
    return int(bool(failed))

if __name__=='__main__':raise SystemExit(main())
