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
    p.add_argument('--core-url',default='https://github.com/needlefinder/NeedleFinder.git')
    p.add_argument('--tag',default='paper-media-2026-v1')
    p.add_argument('--status',default='paper-validated')
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists; refusing to replace another repository')
    sha=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    a.output.mkdir(parents=True)
    subprocess.run(['git','init','-b','main',str(a.output)],check=True,capture_output=True)
    def git(*args):subprocess.run(['git','-C',str(a.output),*args],check=True)
    (a.output/'core').mkdir()
    (a.output/'.gitmodules').write_text(f'[submodule "core"]\n\tpath = core\n\turl = {a.core_url}\n')
    (a.output/'core-version.json').write_text(json.dumps({'repository':a.core_url,'tag':a.tag,'commit':sha,'status':a.status},indent=2)+'\n')
    (a.output/'README.md').write_text(f'''# NeedleFinder paper reproduction
 
This repository is a focused entry point for the 2026 NeedleFinder manuscript.
All algorithms and experiment scripts come from the `core` submodule pinned to
tag `{a.tag}` (`{sha}`). Do not copy or independently edit the algorithms here.
Update the submodule through a reviewed change.

Release status: `{a.status}` (tag `{a.tag}`, commit `{sha}`).

```bash
git submodule update --init
cd core
python3.10 -m venv .venv
.venv/bin/python -m pip install -r reproducibility/requirements-parity.lock
.venv/bin/python tools/release.py verify
.venv/bin/python tools/validate.py fast
.venv/bin/python tools/validate.py parity
.venv/bin/python tools/validate.py coverage
.venv/bin/python tools/verify_paper_results.py --output reproducibility/runs/results.csv
```

See `core/reproducibility/README.md` and `core/reproducibility/VALIDATION.md` for
data configuration, full GPU reproduction (`tools/validate.py paper_reproduction`),
figures, release gates, and citation.
''')
    git('add','.gitmodules','core-version.json','README.md')
    git('update-index','--add','--cacheinfo',f'160000,{sha},core')
    print(f'Prepared paper companion at {a.output}, core={sha} (tag={a.tag}, status={a.status}).')

if __name__=='__main__':main()
