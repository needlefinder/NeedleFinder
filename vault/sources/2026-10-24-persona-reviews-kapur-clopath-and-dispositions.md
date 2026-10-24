---
title: 2026-10-24 Persona Reviews (Tina Kapur, Claudia Clopath) of PMB V2 Manuscript and Dispositions
type: source
category: publication
created: 2026-10-24
updated: 2026-10-24
sources: [paper/manuscript.md, 2026-10-23-exp1-tg43-dosimetry.json, 2026-10-23-exp2-uncertainty-conformal-grubbs.json, 2026-10-23-stage2-and-tip-clean-final.json, 2026-10-23-failure-analysis.json, 2026-10-23-exp4-4d-insertion-and-ct.json]
tags: [peer-review, persona-review, claim-audit, pmb-2026, writing-style]
---

# Context

Two independent LLM persona reviews (Claude Fable 5.1 subagents, fresh context, read-only) of `paper/manuscript.md` were run on 2026-10-24 after the tip-review dose claim was narrowed to the evaluated k = 2 configuration. Both were asked to find the paper "perfect" or list every defect with an exact OLD→NEW fix. Both returned **NOT PERFECT**. Every numeric claim in the findings was re-checked against the source JSONs before the fix was applied. Dispositions below: **applied**, **applied (merged)**, **left (needs author input)**, or **declined** with reason.

# Review A: Tina Kapur persona (clinical translation, imaging physics, 3D Slicer workflow)

| # | Finding | Evidence check | Disposition |
|---|---|---|---|
| B1 | OAR proxy D2cc computed for every patient/protocol but never reported; paper gave target D90 only | Verified in exp1 JSON: Protocol B unreviewed bladder/rectum/sigmoid median 17.5/18.3/12.5% (mean 24.6/26.2/29.5%); top-2 review 13.7/13.1/10.1% (17.8/19.4/22.5%); Protocol A bladder/rectum 3.8/5.4% median; 40-scan Human Team 2 19.9/25.2% vs NF 14.3/11.4% | **applied**: new Results bullet, abstract clause, Limitations item 3 |
| M1 | Flipped-reference (Team 2 as reference) non-inferiority reported selectively | Verified: tip median p_pt 0.073, p_boot 0.046, p_cath 9.6e-5; tip mean 7.53 vs 6.10 mm, p_pt 0.12, p_boot 0.42 | **applied** |
| M2 | Three-cornered-hat 0.00 mm for Team 1 mis-explained; real cause is V12+V13≈V23 (correlated errors) | Verified: Team 1 axial MAD SD point 0.001, bootstrap CI 0.001–2.02 mm | **applied (merged with Clopath M2)** |
| M3 | Workflow never deletes false tracks; count-check click is a candidate promotion, costs precision | Verified: 146 unmatched predictions (66 genuine background FP) = 2.7/1.2 per patient; full-deficit promotion 35 tracks, 17 real, precision 86.94→85.78% | **applied**: workflow step 3 and Protocol B methods |
| M4 | Review times 16/40/64 s vs JSON 18/42/66 s | Verified | **applied** (same as Clopath B1) |
| M5 | "Clinically ready" criterion necessary not sufficient | — | **applied** |
| M6 | "Patient under anesthesia during tracing" overstated | Clinical practice varies by center | **applied**: "waits with the implant in place, often still sedated" |
| M7 | n=1 prostate, n=2 wbSSFP presented as cohorts | — | **applied** (Clopath M7 wording) |
| M8 | "Best 1-channel pipelines below 70% F1" contradicts Zaffino 2019 at 83.1% | Verified in Table 2 | **applied** (contradiction removed); **left**: why the 2019 pipeline beats three 2020–2026 methods needs a reason-code audit of the Zaffino reproduction |
| M9 | "Reproduced" → "re-implemented" | Ledger calls it re-implementation | **applied**; authors to confirm no original code was used |
| M10 | "Missed by the human team" is an inference; CT shows "present, not traced" | — | **applied** (abstract, summary, results) |
| M11 | Title "Sub-Millimeter Catheter Reconstruction" when tip is 2.3 mm median | — | **applied**: "Automatic Catheter Reconstruction with Sub-Millimeter Shaft Accuracy"; **authors may revert** |
| M12 | Code identifiers and archive codes in prose | — | **applied** throughout; archive codes kept only where defined in Section 2 |
| M13 | 4.15 s is the PyTorch harness on a 24-core CPU, not the Slicer ONNX module; interface claim unevidenced | — | **applied** wording; **left**: one timing run of the ONNX module on a planning workstation is needed |
| m1–m18 | Conformal "guarantee"; 54 vs 40 cohort mixing in Protocol B human comparison; plain-language tip numbers from different catheter sets (paired 2.33 vs 3.18 mm); τ=35% precision; "proving zero"; two oracles; "contains clinically"; "parameter-free"; "real time"; 0.003 mm median; "entire bundle"; 10–30 vs 15–30 catheters and 30–45 vs 46 min; Protocol B gamma lower than A; tandem not reconstructed; "live 4D"; vault path; three-cornered-hat ≠ non-inferiority; "CT ground truth" | 40-scan NF Protocol B 7.0/3.2% verified; paired tip 2.33/3.18 in manuscript line 269 | **applied** except: m12 time figure (30–45 vs 46 min) **left** for authors; m14 tandem **left** (needs confirmation); m18 "single-rater" definition **left** (only "CT ground truth" → "CT tip position" applied) |
| p1–p7 | Polish | — | **applied** (p5 "PMB 2019" naming unified only in the dosimetry sentence; table labels untouched) |

# Review B: Claudia Clopath persona (pedagogy, statistics, overclaiming)

| # | Finding | Evidence check | Disposition |
|---|---|---|---|
| B1 | 16/40/64 s vs 18/42/66 s | Verified | **applied** |
| M1 | Abstract and summary omit that the k=2 correction is an oracle | — | **applied** |
| M2 | Three-cornered-hat independence assumption violated by training on both teams' traces | — | **applied (merged with Kapur M2)**: Methods caveat + Results rewording + "CT adjudication depends on neither team" |
| M3 | ~40 uncorrected tests; p≈0.01–0.05 results should be labelled suggestive | — | **applied**: Methods statement; fusion gain labelled "0.3-point, uncorrected" |
| M4 | "Makes it even stronger" for a 0.006 mm change | — | **applied** |
| M5 | Code identifiers | — | **applied** |
| M6 | Headline numbers repeated in Discussion/Conclusion | — | **applied**: Discussion numbers replaced by table pointers; Conclusion rewritten |
| M7 | Prostate scan is in the 51 SSL pre-training volumes; 100% F1 only with manual crop or exposed model; CT recall 68.9% | Verified in manuscript Section 2 and Results | **applied**: abstract and results rewritten |
| M8 | Runtime always paired with the fast subgroup | 5.85 s high-res from timing JSON | **applied** |
| M9 | Six overclaiming verbs | — | **applied** |
| M10 | 4D biomechanics: test not named, 12 mm split differs from bins, 4 patients treated as independent | Verified key `mann_whitney_near_le_12mm_vs_far_gt_12mm_p` | **applied** |
| M11 | τ=35% recall without precision | Verified 83.4% precision, 88.2% F1 | **applied** |
| M12 | "Much more strongly" correlation difference untested | — | **applied** |
| m1 | 30–45 min vs 46 min | — | **left** for authors (needs a citation or local timing) |
| m2 | 10–30 vs 15–30 catheters | — | **applied** |
| m3 | "under-one-minute" vs 18–66 s + 4 + 5 | — | **applied** |
| m4–m5 | Two oracles; assumed (not estimated) times | — | **applied** |
| m6 | Count-check mechanism unexplained | Verified in exp2 queue2 (promotion of rejected candidates) | **applied** |
| m7 | Paired ranges in abstract unexplained | — | **applied** |
| m8–m10 | "parameter-free"; "2nd and 3rd order statistics"; "outperforming" on 0.7 points | — | **applied** |
| m11 | 465 vs 476 pairs, 35 vs 36 Team-2-only | — | **left** for authors (needs the matching-pass explanation) |
| m12–m14 | Re-implementations; Xiao hardware; flipped test marginal | — | **applied** |
| polish | kirisits2014 mis-cited for bootstrap (moved to the D2cc sensitivity sentence); rhetoric; 95-word sentence split | — | **applied**; Fig. 1 caption runtimes, Generation 2 citation block, case numbers **declined** (harmless) |

# Final overclaim sweep (fresh-context, after both reviews)

44 sentences in 29 entries were found and all applied; the substantive one was the omitted post-contrast T1 VIBE cross-sequence result (12 scans, 2 patients, 217 catheters: 56.1% F1, 58.5% recall, 53.8% precision, 8.08 mm median tip; Stage 0 alone 61.5%, Zaffino 2019 re-implementation 59.4%), now in Results, abstract, cohorts, conclusion, Limitations and Table 6 Part C. See `wiki/log.md` 2026-10-24 for the full list.

# Open items for the authors

1. Manual tracing time: the uncited "30 to 45 minutes" (abstract, summary, discussion) was changed in the final sweep to "about 45 minutes", matching the cited 46-minute mean [@shaaer2022]. Revert with a source or local timing if the shorter figure is preferred.
2. Why the isotropic 2019 U-Net + DBSCAN re-implementation (83.1% F1) beats three 2020–2026 re-implementations by 14–25 points on the same folds.
3. One timing run of the ONNX 3D Slicer module on a typical planning workstation; confirm the module lists uncertainty-ranked tips.
4. Confirm no original baseline code was used (all six baselines are re-implementations).
5. Confirm whether the intrauterine tandem is reconstructed or only masked.
6. Explain the 465 vs 476 consensus-pair counts and 35 vs 36 Team-2-only counts.
7. Define "closer to CT than single-rater MRI" (78.6%): closer than at least one team, or than both?
8. Title change to "...with Sub-Millimeter Shaft Accuracy..." is applied; revert if the authors prefer the shorter title.
