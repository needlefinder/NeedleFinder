"""Resolve historical workstation paths without requiring that workstation layout."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('NEEDLEFINDER_WORK_DIR', ROOT / 'reproducibility/runs/work')).resolve()
DATA = Path(os.environ.get('NEEDLEFINDER_DATA_DIR', WORK / 'data')).resolve()


def resolve_path(value):
    mappings = (
        ('/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder', ROOT / 'NeedleFinder'),
        ('/mnt/SSD/github/needlefinder/NeedleFinder/NeedleFinder', ROOT / 'NeedleFinder'),
        ('/home/guillaume/github/needlefinder/vault/paper/benchmarks_clean', ROOT / 'benchmarks/clean'),
        ('/mnt/SSD/github/needlefinder/vault/paper/benchmarks_clean', ROOT / 'benchmarks/clean'),
        ('/home/guillaume/github/needlefinder/vault/paper/benchmarks_rerun', ROOT / 'benchmarks/rerun'),
        ('/mnt/SSD/github/needlefinder/vault/paper/benchmarks_rerun', ROOT / 'benchmarks/rerun'),
        ('/home/guillaume/github/needlefinder/vault/sources', ROOT / 'reproducibility/results'),
        ('/mnt/SSD/github/needlefinder/vault/sources', ROOT / 'reproducibility/results'),
        ('/home/guillaume/github/needlefinder/vault/reports/figures/data', ROOT / 'reproducibility/results/figures_data'),
        ('/mnt/SSD/github/needlefinder/vault/reports/figures/data', ROOT / 'reproducibility/results/figures_data'),
        ('/home/guillaume/github/needlefinder', WORK),
        ('/mnt/SSD/github/needlefinder', WORK),
        ('/home/guillaume/Dropbox', DATA),
        ('/tmp', WORK),
    )
    for prefix, root in mappings:
        if value == prefix or value.startswith(prefix + '/'):
            return str(root / value[len(prefix):].lstrip('/'))
    return value
