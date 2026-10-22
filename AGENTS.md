# NeedleFinder Agent Instructions (`AGENTS.md`)

## Obsidian Vault (`OVNF`) — Read First

A persistent LLM-maintained Obsidian wiki (**OVNF**: Obsidian Vault for NeedleFinder) lives at `vault/` (`/mnt/SSD/github/needlefinder/vault`, symlinked at `NeedleFinder/vault` and synced to `mbp-2:/Users/guillaume/Dropbox/NeedleFinder-obsidian/NeedleFinder` and `edelweiss:/home/guillaume/github/needlefinder/vault`). **Before exploring the codebase from scratch, running deep research, or making architectural changes, check the vault first:**

1. Read `vault/CLAUDE.md` for the vault schema, page conventions, and maintenance workflows.
2. Read `vault/wiki/index.md` to find existing pages covering tracking algorithms, 3D tubular detection kernels, clinical cohorts, benchmark evaluations, and publication plans.
3. Read the relevant `vault/wiki/` pages before grepping raw code.
4. When you complete significant work, discover new facts, or ingest new sources, **update the vault** (`vault/wiki/` pages + `vault/wiki/index.md` + `vault/wiki/log.md`) so future sessions inherit your findings.

## Report Delivery (HTML / PDF)

**Whenever you generate or update an HTML report or PDF (or any HTML/PDF artifact meant to be viewed), ALWAYS end your reply with an `rsync` command to pull it for viewing, using host `think`:**

```bash
rsync -avz think:/mnt/SSD/github/needlefinder/vault/<path-to-file>.{html,pdf} ./
```

For example:
```bash
rsync -avz think:/mnt/SSD/github/needlefinder/vault/paper/needlefinder_2026_pmb.pdf ./
```

Give the exact path to the file(s) just produced. This is non-optional — the human views reports off-box, so a report without its rsync line is unfinished.

