# NeedleFinder Brain (OVNF) — LLM Wiki Schema

This is the schema file for the **NeedleFinder** knowledge base (**OVNF**: Obsidian Vault for NeedleFinder). It tells LLM agents how to maintain and evolve this persistent research & engineering wiki.

## OVNF Overview

From the repository root, OVNF is located at `/mnt/SSD/github/needlefinder/vault` (and symlinked at `NeedleFinder/vault`, synced to `mbp-2:/Users/guillaume/Dropbox/NeedleFinder-obsidian/NeedleFinder`). Treat OVNF as the source of record for durable NeedleFinder algorithmic design, clinical datasets, benchmark evaluations, 3D Slicer extension architecture, and publication planning.

Before modifying tracking algorithms, 3D tubular detection kernels, evaluation metrics, or dataset splits, read `wiki/index.md` first, then read the relevant OVNF pages.

OVNF follows the LLM Wiki / Obsidian wiki pattern used across `audiala` (`OVAK`) and `ia` (`influence-alpha`) on `edelweiss`: immutable raw sources in `sources/`, compiled interlinked wiki pages in `wiki/`, a content catalog in `wiki/index.md`, and a reverse-chronological changelog in `wiki/log.md`.

## Architecture

```
vault/
├── CLAUDE.md           # This file — schema & maintenance conventions
├── home.md             # Human navigation hub (Obsidian entrypoint)
├── sources/            # Raw inputs, paper digests, benchmark JSONs (immutable)
├── wiki/               # LLM-maintained knowledge
│   ├── index.md        # Content catalog (LLM reads first)
│   ├── log.md          # Chronological changelog (append at top)
│   ├── overview.md     # High-level project & research synthesis
│   ├── entities/       # Software modules, clinical cohorts, hardware/applicators
│   ├── concepts/       # Mathematical filters, tracking models, geometry priors
│   ├── research/       # Quantitative evaluation reports & dataset provenance
│   └── syntheses/      # Cross-cutting comparisons & paper roadmaps
└── plans/              # Publication & engineering execution plans
```

## Navigation Files

1. **`wiki/index.md` (Content-Oriented)**: Catalog of every page in the wiki with wikilink, category, source count, and one-line summary. Read this first.
2. **`wiki/log.md` (Chronological)**: Append-only record at the top of the file using `## [YYYY-MM-DD] action | description` (`ingest`, `query`, `lint`, `update`, `create`).

## Page Frontmatter Convention

All wiki pages (except `index.md` and `log.md`) begin with YAML frontmatter:

```yaml
---
title: <Page Title>
type: <entity | concept | research | synthesis | overview | source | plan>
category: <algorithm | dataset | evaluation | architecture | publication | hardware>
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [<source-files>]
tags: [<relevant-tags>]
---
```

## Report Delivery (HTML / PDF)

**Whenever you generate or update an HTML report or PDF (or any HTML/PDF artifact meant to be viewed), ALWAYS end your reply with an `rsync` command to pull it for viewing, using host `think`:**

```bash
rsync -avz think:/mnt/SSD/github/needlefinder/vault/<path-to-file>.{html,pdf} ./
```

Give the exact path to the file(s) just produced. **Never use unquoted `*` wildcards in `rsync` commands** (since `zsh` fails with `no matches found` on local expansion); always use explicit filenames, brace expansion (`{pdf,html,tex}`), or directory paths. This is non-optional — the human views reports off-box, so a report without its rsync line is unfinished.
