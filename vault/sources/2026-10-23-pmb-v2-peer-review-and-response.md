# *Physics in Medicine & Biology* (PMB) V2 Readability Transformation & Three-Reviewer Peer-Review Record

**Date:** 2026-10-23  
**Manuscript:** `vault/paper/manuscript.md` $\to$ `vault/paper/needlefinder_2026_pmb.{pdf,html,tex}` (and `needlefinder_2026_pmb_v2.{pdf,html}`; V1 dense audited edition preserved at `vault/paper/manuscript_v1_audit.md` and `needlefinder_2026_pmb_v1_audit.{pdf,html}`).

---

## 1. Summary of the V2 Readability Transformation (Claudia Clopath & Tina Kapur Style)

To make the paper effortless to read for both specialists and scientists outside medical physics or deep learning, we restructured and rewrote the prose following the complementary stylistic principles of **Prof. Claudia Clopath** (Imperial College London) and **Dr. Tina Kapur** (Brigham and Women's Hospital / Harvard Medical School):

1. **"Why Before How" Pedagogical Framing (Claudia Clopath Style):**
   - Every section and subsection opens with the intuitive physical or computational puzzle before introducing equations or network architectures:
     - *Section 1.1:* Why T2W MRI catheter reconstruction is hard and why 1 mm matters dosimetrically.
     - *Section 3:* How a human physicist traces catheters in three mental steps (2D round dark cores $\to$ slice-to-slice collinearity $\to$ whole-bundle plausibility and tip termination) and how Stages 0, 1, and 2 mirror those exact three steps.
     - *Section 3.1:* Why standard neural networks confuse muscle borders with catheters, and how sampling six diameters across four radii separates round dark cores from one-sided edges.
     - *Section 3.2:* Why resampling to $1\text{ mm}^3$ blurs 2 mm catheters, why binary masks merge when catheters touch while Gaussian centerline peaks ($P_{\text{peak}}$) stay separate, and how pre-training on 51 needle-free pelvic MRI volumes turns real blood vessels into guaranteed anatomical hard negatives.
     - *Section 3.3:* Why greedy slice linking creates kinks that a triplet message-passing network prevents, why looking at the whole 15-to-30-catheter bundle eliminates residual vascular tracks, and why 98.8% of tip error is purely longitudinal along the shaft.
2. **Operating-Room Clinical Narrative & Translational Grounding (Tina Kapur Style):**
   - Grounded the narrative in the real intraoperative workflow of the BWH AMIGO suite and **3D Slicer** (turning a 30-to-45-minute slice-by-slice manual digitization under general anesthesia into a 4-second automatic reconstruction followed by an under-one-minute uncertainty-guided tip check).
3. **Eliminating Code Jargon & Parenthetical Number Walls:**
   - Removed internal script identifiers (`Item 1..5`, `k_lim_auto`, `Dedicated1DTipLocalizer` in the Abstract, raw numerator/denominator dumps, and 10-number parenthetical clauses) from the narrative prose, letting Tables 1–6 carry the dense secondary distributions while keeping 100% of verified claims intact.

---

## 2. Point-by-Point Response to the Three *Physics in Medicine & Biology* (PMB) Reviewers

### Reviewer 1 (Senior Clinical Medical Physicist & Brachytherapy Researcher — GEC-ESTRO / TG-303 Perspective)

- **Comment R1.1 (MRI Physics of Catheter Voids, Oblique Tilt $\alpha$, and Caudo-Cranial Orientation in `k_lim_auto`):**
  *Clarify whether the 54 AMIGO scans had hollow/obturator-loaded plastic catheters; distinguish proton-density voids from $\Delta\chi$ susceptibility transitions; explain why Stage 0's 2D axial ring filter remains invariant under realistic oblique catheter tilt ($\alpha \le 25^\circ\text{--}30^\circ$) and how thick slices ($\Delta z = 4\text{--}6\text{ mm}$) are handled; and state that `k_lim_auto` standardizes caudo-cranial slice ordering via DICOM/NRRD direction cosines.*
  - **Author Response & Action Taken:** Fully adopted in **Section 1.1**, **Section 2 (Cohort 1)**, and **Section 3.1**. We now specify that patients were scanned with 6F polymeric interstitial catheters (with plastic or non-ferromagnetic obturators in situ, producing a proton-density signal void and mild $\Delta\chi$ interface susceptibility contrast), explain geometrically why taking the maximum over $r \in \{1.35, 1.85, 2.5, 3.3\}\text{ mm}$ and the 2nd/3rd order statistics ($0.45\,m_{(2),r} + 0.55\,m_{(3),r}$) tolerates elliptical cross-section elongation ($\sec\alpha \le 1.15$ for $\alpha \le 30^\circ$) while $1.6\text{ mm}$ $z$-upsampling and Stage 2A fragment stitching handle thick slices ($\Delta z = 4\text{--}6\text{ mm}$), and state explicitly that `k_lim_auto` standardizes inferior-to-superior slice orientation from the volume's direction cosines before scanning upward.

- **Comment R1.2 (Upfront Methodological Transparency for TG-43U1 Envelopes, Protocols A vs. B, and Simulated 1-Click Tip Triage):**
  *Move the definitions of the implant-conformal target envelope ($\text{HR-CTV}_{\text{implant}}$, $8.0\text{ mm}$ radial margin), the $2\text{ cm}^3$ OAR proxy shells, the handling of missed/predicted catheters in Protocols A & B, and the mathematical formula + ergonomic $\sim 8\text{ s/tip}$ timing assumption of the 1-click tip relocation oracle from Table 6 footnotes into Section 4.2 (Methods), and consistently label review timings as "simulated."*
  - **Author Response & Action Taken:** Fully adopted in **Section 4.2**, **Section 5.6.2**, **Section 6.2**, and **Section 7**. The implant-conformal $\text{HR-CTV}_{\text{implant}}$ ($8.0\text{ mm}$ envelope), the $2\text{ cm}^3$ OAR proxy shells, Protocols A and B, and the exact 1-click oracle equation $E_{\text{tip}} \to \sqrt{\Delta r_\perp^2 + \min(|\Delta s_\parallel|, 0.85\text{ mm})^2}$ ($\sim 8\text{ s/tip}$ simulated) are now defined transparently in Section 4.2.

- **Comment R1.3 (Disclosure of Unlabeled MSD Scan Overlap in Section 5.5 Main Text):**
  *State directly in the main text of Section 5.5 (matching Table 4 Footnote `f`) that `0.33 FP/scan` ($79.2\%$ 0-FP scans) is the strictly zero-exposure MSD specificity result, whereas `0.125 FP/scan` ($87.5\%$ 0-FP scans) includes the 48 needle-free MSD volumes as unlabeled background anatomy during synthetic-tube SSL pre-training.*
  - **Author Response & Action Taken:** Adopted verbatim in **Section 5.5**.

- **Comment R1.4 (Table 1–6 Clinical Polish & 4D Intra-Procedural Discussion):**
  *Expand Table 1 to include the 4 cross-domain validation cohorts; replace `Item 1 / 3 / 4a / 4b` in Table 2 Part B with descriptive stage names; define `Cran. D2cc` in Table 6 Footnote `a`; add a sentence in Section 6.2 on how the 1.31 s causal 4D Kalman mode can alert physicians to tissue push or deflection during progressive needle insertion.*
  - **Author Response & Action Taken:** Adopted across **`tab1_cohorts.tex`** (now 10 rows covering all datasets), **`tab2_main_results.tex`**, **`tab3_interobserver.tex`**, **`tab5_literature.tex`**, **`tab6_dosimetry_and_generalization.tex`**, and **Section 6.2**.

---

### Reviewer 2 (Computational Scientist & ML Educator — Claudia Clopath Perspective)

- **Comment R2.1 (Sentence-Level Cognitive Load Rewrites in Abstract, Section 2, Section 4.2, and Section 5.4):**
  *Break up the 27-word em-dashed parenthetical in the Abstract (Stage 1), separate zero-shot transfer from runtime at the end of Abstract Main Results, introduce archive codes (`CY`, `HT`, `LB/AMe`) cleanly in Section 2, split the 65-word non-inferiority and 51-word TG-43U1 sentences in Section 4.2, and streamline the 91-missed-catheter breakdown in Section 5.4.*
  - **Author Response & Action Taken:** All five sentence rewrites have been incorporated into **`vault/paper/manuscript.md`**.

- **Comment R2.2 (One-Sentence Intuitive Bridges at First Appearance for Specialized Concepts):**
  *Provide immediate 1-sentence intuitive explanations at first appearance for: (1) obturator-adjusted precision ($P_{\text{adj}}$), (2) Grubbs' three-cornered-hat estimator (including $\hat{\sigma}_1^2 = \frac{1}{2}(V_{12} + V_{13} - V_{23})$), (3) split-conformal prediction intervals ($r_i = E_{\text{tip}, i}/\hat{\sigma}_{\text{tip}, i}$ and $\hat{r}_{1-\alpha} = \hat{q}_{1-\alpha}\,\hat{\sigma}_{\text{tip}}$), (4) two-stage stacking optimism and inner out-of-fold cross-fitting (`4c`), and (5) dwell-time-weighted tip error ($E_{\text{tip}}^{\text{dwell-w}}$).*
  - **Author Response & Action Taken:** All five intuitive explanations and equations are now integrated into **Sections 3.3, 4.1, 4.2, and 5.6.1**.

- **Comment R2.3 (Mathematical & Data-Flow Completion in Section 3):**
  *Write out the explicit formula for $S_0(\mathbf{p})$ in Section 3.1, clarify the exact sequence between bidirectional Kalman seed tracking and `NeedleSliceMPNN` in Step 2A, and specify how the 1D tip offset $\Delta s$ shifts the cranial endpoint along the 3D tangent before `Poly3` smoothing in Step 2C.*
  - **Author Response & Action Taken:** Adopted in **Sections 3.1 and 3.3**.

---

### Reviewer 3 (Biostatistician & Quantitative Audit Reviewer)

- **Comment R3.1 (Five Numerical / Provenance Corrections):**
  1. *Fix the $k=2$ tip triage mean error in Section 5.6.2 from `$4.89\text{ mm}$` to **`$5.07\text{ mm}$ ($1.95\text{ mm}$ median)`** to match Table 6 Part B1 and `2026-10-23-exp2-uncertainty-conformal-grubbs.json`.*
  2. *Update the 95% Conformal Interval row in `tab6_dosimetry_and_generalization.tex` (Part B2, line 43) from `94.8% | 20.69 | 18.42 | 28.40 | 12.69` to **`95.2% | 21.05 | 18.42 | 28.39 | 12.89`** to match `part2b_nested_split_conformal_tip_intervals.target_95pct` in `2026-10-23-exp2-uncertainty-conformal-grubbs.json`.*
  3. *Fix the active slice count parenthetical in Section 3.1 (`k_lim_auto`) to **`median 2.0 slices deeper (mean 1.91 slices deeper, IQR 1.0; 64.1 vs. 62.2 active slices on average)`** (`89.0 - 24.89 = 64.11` vs. `89.0 - 26.80 = 62.20`, exact difference `1.91` slices).*
  4. *Clarify in Section 3.2 why $n=35$ Team 2-only traces receive soft weight $0.55$ during training ($513 - 465\text{ CY} - 13\text{ HT} = 35$) vs. $36$ in the evaluation union.*
  5. *Update standard-resolution GPU runtime in `tab5_literature.tex` Footnote `k` from `$0.68$\,s` to **`$0.67$\,s`** (`0.6749 s` in `2026-10-23-per-case-timing-benchmark.json`, matching `manuscript.md`).*
  - **Author Response & Action Taken:** All five corrections have been applied and verified across `manuscript.md`, `tab5_literature.tex`, and `tab6_dosimetry_and_generalization.tex`.

- **Comment R3.2 (Statistical Rigor Additions in Sections 4.2, 5.3, and 5.6):**
  *Specify patient-level paired Wilcoxon testing ($N=54$ and $N=23$) and zero-match penalty imputation ($3.5\text{ mm}$ shaft, $10\text{ mm}$ curve, $20\text{ mm}$ tip/$\text{HD}_{95}$); explain how the $n=432$ three-way matched set is formed from the $476$ all-catheter human pairs and why finite-sample Grubbs subtraction clamps Rater 1's MAD $\sigma_z$ at $0.00\text{ mm}$ while 15% Winsorized $\sigma_z$ yields $3.77, 4.94, 5.09\text{ mm}$; note why two-sided 90% CIs ($1-2\alpha$) correspond to one-sided $\alpha=0.05$ non-inferiority tests and mention the flipped-reference check against Team 2 ($p_{\text{boot}} = 0.046$); and state the held-out outer-fold assignment (`Case 068` Fold 4, `Case 069` Fold 0, `Case 070` Fold 1, `Case 071` Fold 2) for the 4D causal Kalman evaluation.*
  - **Author Response & Action Taken:** All six statistical clarifications are now incorporated in **Sections 4.2, 5.3, 5.6.2, and 5.6.4**, as well as in the footnotes of Tables 2, 3, and 6.
