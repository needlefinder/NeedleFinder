#!/usr/bin/env python3
"""Build and verify source/artifact manifests; fail closed before a paper release."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'reproducibility/artifacts.lock.json'
REQUIRED_CHECKS = ('fast', 'parity', 'clinical', 'slicer', 'paper_reproduction', 'coverage')
CODE_SUFFIXES = {'.py','.rs','.cpp','.toml','.cfg','.onnx','.npz','.lock','.ini','.cmake','.yml','.yaml','.json'}


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_fingerprint():
    paths=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z','--cached','--others','--exclude-standard']).decode().split('\0')
    hashes={}
    for name in sorted(set(paths)-{''}):
        p=ROOT/name
        if p.is_file() and (p.suffix in CODE_SUFFIXES or p.name in {'CMakeLists.txt','.coveragerc'}):
            hashes[name]=sha256(p)
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def lock_artifacts():
    files=list((ROOT/'NeedleFinder/NeedleFinderLib').glob('*.onnx'))+list((ROOT/'NeedleFinder/NeedleFinderLib').glob('*.npz'))
    files+=sorted((ROOT/'vault/sources').glob('2026-10-2[345]-*.json'))
    files+=sorted((ROOT/'vault/reports/figures/data').glob('*.npz'))
    return {'schema_version':1,'files':{str(p.relative_to(ROOT)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(files)}}


def verify_artifacts(lock=None):
    lock=json.loads(LOCK.read_text()) if lock is None else lock
    if not lock.get('files'):raise ValueError('Empty artifact lock')
    for name,info in lock['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT.resolve()):raise ValueError(f'Path outside repository: {name}')
        if not p.is_file() or sha256(p)!=info['sha256'] or p.stat().st_size!=info['bytes']:
            raise ValueError(f'Artifact absent or modified: {name}')
    return len(lock['files'])


def release_blockers(evidence):
    blockers=[]
    if git('status','--porcelain','--untracked-files=all'):blockers.append('Working tree is not clean')
    current=source_fingerprint()
    for check in REQUIRED_CHECKS:
        receipt=evidence/f'{check}.json'
        if not receipt.is_file():blockers.append(f'Missing {check} validation');continue
        r=json.loads(receipt.read_text())
        if r.get('status')!='passed' or r.get('source_fingerprint')!=current:
            blockers.append(f'{check}: failed or stale source fingerprint')
        if r.get('skipped',0):blockers.append(f'{check}: skipped tests are not release evidence')
        report=evidence/r.get('report','missing')
        if not report.is_file() or sha256(report)!=r.get('report_sha256'):
            blockers.append(f'{check}: missing/modified report')
    return blockers


def manifest():
    verify_artifacts()
    return {'schema_version':1,'source_commit':git('rev-parse','HEAD'),
            'source_tree':git('rev-parse','HEAD^{tree}'),'dirty':bool(git('status','--porcelain')),
            'source_fingerprint':source_fingerprint(),'artifact_lock_sha256':sha256(LOCK),
            'platform':platform.platform(),'python':sys.version,
            'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            'reproduction_config_sha256':sha256(ROOT/'reproducibility/protocol.json'),
            'status':'candidate-not-paper-validated'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['lock','verify','manifest','check'])
    p.add_argument('--output',type=Path)
    p.add_argument('--evidence',type=Path,default=ROOT/'reproducibility/runs/evidence')
    a=p.parse_args()
    if a.action=='lock':
        LOCK.write_text(json.dumps(lock_artifacts(),indent=2,sort_keys=True)+'\n');print(LOCK)
    elif a.action=='verify':print(f'Verified {verify_artifacts()} artifacts')
    elif a.action=='manifest':
        if a.output is None:p.error('--output is required')
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(manifest(),indent=2,sort_keys=True)+'\n');print(a.output)
    else:
        verify_artifacts();blockers=release_blockers(a.evidence)
        if blockers:
            print('Paper release BLOCKED:\n'+'\n'.join('- '+b for b in blockers));return 1
        print('All paper release gates passed')
    return 0

if __name__=='__main__':raise SystemExit(main())
