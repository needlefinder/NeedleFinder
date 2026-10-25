#!/usr/bin/env python3
"""Run a validation suite and bind its report to the exact current source files."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.release import ROOT, sha256, source_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite',choices=['fast','parity','clinical','coverage'])
    p.add_argument('--output',type=Path,default=ROOT/'reproducibility/runs/evidence')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    before=source_fingerprint()
    report=a.output/f'{a.suite}.xml'
    selection={'fast':'not clinical and not parity','parity':'parity','clinical':'clinical','coverage':'not clinical'}[a.suite]
    cmd=[sys.executable,'-m','pytest','-q','-m',selection,f'--junitxml={report}']
    if a.suite=='coverage':cmd=[sys.executable,'-m','coverage','run','-m',*cmd[2:]]
    env=os.environ.copy();env.setdefault('OMP_NUM_THREADS','1');env.setdefault('OPENBLAS_NUM_THREADS','1')
    result=subprocess.run(cmd,cwd=ROOT,env=env)
    failed=result.returncode!=0;skipped=0;tests=0
    if report.exists():
        root=ET.parse(report).getroot()
        cases=root.findall('.//testcase');tests=len(cases)
        skipped=sum(c.find('skipped') is not None for c in cases)
        failed |= any(c.find('failure') is not None or c.find('error') is not None for c in cases)
    else:failed=True
    failed |= before!=source_fingerprint() or tests==0 or skipped>0
    if a.suite=='coverage' and not failed:
        failed |= subprocess.run([sys.executable,'-m','coverage','json'],cwd=ROOT).returncode!=0
        if not failed:failed |= subprocess.run([sys.executable,'tools/check_coverage.py'],cwd=ROOT).returncode!=0
    receipt={'suite':a.suite,'status':'failed' if failed else 'passed','source_fingerprint':before,
             'tests':tests,'skipped':skipped,'command':cmd,'python':sys.version,
             'report':report.name,'report_sha256':sha256(report) if report.exists() else None}
    (a.output/f'{a.suite}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return int(failed)

if __name__=='__main__':raise SystemExit(main())
