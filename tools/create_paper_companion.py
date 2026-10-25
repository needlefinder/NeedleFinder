#!/usr/bin/env python3
"""Create a separate paper repository pinned to a core commit, without copying algorithms."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--core-url',default='https://github.com/needlefinder/needlefinder-2026.git')
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists; refusing to replace another repository')
    sha=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    a.output.mkdir(parents=True)
    subprocess.run(['git','init','-b','main',str(a.output)],check=True,capture_output=True)
    def git(*args):subprocess.run(['git','-C',str(a.output),*args],check=True)
    (a.output/'core').mkdir()
    (a.output/'.gitmodules').write_text(f'[submodule "core"]\n\tpath = core\n\turl = {a.core_url}\n')
    (a.output/'core-version.json').write_text(json.dumps({'repository':a.core_url,'commit':sha,'status':'candidate-not-paper-validated'},indent=2)+'\n')
    (a.output/'README.md').write_text(f'''# NeedleFinder paper reproduction

This repository is a focused entry point. All algorithms and experiment scripts
come from the `core` submodule pinned to `{sha}`. Do not copy or independently
edit the algorithms here. Update the submodule through a reviewed change.

This is a candidate, not a verified paper release. The source commit must be
pushed to the configured core remote before others can initialize this submodule.
The default core remote requires access; no repository visibility is changed.

```bash
git submodule update --init
cd core
python3.10 -m venv .venv
.venv/bin/python -m pip install -r reproducibility/requirements-parity.lock
.venv/bin/python tools/release.py verify
.venv/bin/python tools/validate.py fast
.venv/bin/python tools/validate.py parity
.venv/bin/python tools/verify_paper_results.py --output reproducibility/runs/results.csv
```

See `core/reproducibility/README.md` for data, full experiments, figures, release
gates and citation. The CSV command verifies archived statistics; it does not
rerun training. Cite the final version-specific DOI once it has been issued.
''')
    git('add','.gitmodules','core-version.json','README.md')
    git('update-index','--add','--cacheinfo',f'160000,{sha},core')
    print(f'Prepared paper companion at {a.output}, core={sha}; no remote publication performed.')

if __name__=='__main__':main()
