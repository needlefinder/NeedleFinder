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

## Writing Style (Papers, Reports, Vault Prose)

All manuscript and report prose written for this project follows the house style set in the October 2026 V2 rewrite (see `sources/2026-10-23-pmb-v2-peer-review-and-response.md`). It blends two models, applied within reason: **readability always wins over completeness or density.**

1. **Claudia Clopath ("why before how").** Open every section with the physical or computational puzzle in plain words, then the method. Short declarative sentences. One idea per sentence, one finding per paragraph. Define a term in one sentence the first time it appears. No code identifiers, script names, or item numbers in prose.
2. **Tina Kapur (clinical workflow grounding).** Anchor claims in what the physicist actually does in the operating room and in 3D Slicer: the time, the clicks, the decision. Reason from imaging physics (proton density, susceptibility) rather than from architecture names.
3. **Numbers.** Each headline number appears once in the abstract and once in the results; elsewhere point to the table or figure. No parenthetical number walls. Give a before-and-after pair rather than a lone value when a method improves something.
4. **Qualify honestly.** Say what was simulated, modeled, or assumed (synthetic anatomy, oracle corrections, estimated click times) in the same sentence as the result, not in a distant limitations paragraph. Report only the configuration that was actually evaluated; never widen a result from one setting (e.g. k = 2 reviewed tips) into a range (2 to 5) that was not run. A smaller claim stated exactly beats a larger claim stated loosely.
5. **Every rewrite preserves every verified claim, number, and citation.** Readability edits never silently change the evidence; if the evidence does not support the prose, fix the prose and log it in `wiki/log.md`.

## Report Delivery (HTML / PDF)

**Whenever you generate or update an HTML report or PDF (or any HTML/PDF artifact meant to be viewed), ALWAYS end your reply with an `rsync` command to pull it for viewing, using host `think`:**

```bash
rsync -avz think:/mnt/SSD/github/needlefinder/vault/<path-to-file>.{html,pdf} ./
```

Give the exact path to the file(s) just produced. **Never use unquoted `*` wildcards in `rsync` commands** (since `zsh` fails with `no matches found` on local expansion); always use explicit filenames, brace expansion (`{pdf,html,tex}`), or directory paths. This is non-optional — the human views reports off-box, so a report without its rsync line is unfinished.
