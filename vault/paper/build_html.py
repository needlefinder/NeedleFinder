#!/usr/bin/env python3
"""Build self-contained HTML companion edition of NeedleFinder 2026 PMB manuscript."""
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent

HTML_CSS = """
<style>
  :root {
    --ink: #0F172A;
    --slate: #475569;
    --border: #CBD5E1;
    --grid: #E2E8F0;
    --bg-tint: #F8FAFC;
    --ours-bg: #FFF3ED;
    --ours-accent: #D55E00;
    --blue-accent: #0072B2;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
    line-height: 1.62;
    max-width: 1060px;
    margin: 0 auto;
    padding: 2.5rem 2rem 4rem;
    background: #FFFFFF;
  }
  h1, h2, h3 {
    color: var(--ink);
    font-weight: 700;
    letter-spacing: -0.015em;
  }
  h1.title {
    font-size: 1.85rem;
    line-height: 1.25;
    margin-bottom: 0.5rem;
    border-bottom: 3px solid var(--ours-accent);
    padding-bottom: 0.75rem;
  }
  p.author, p.date {
    color: var(--slate);
    margin: 0.2rem 0;
    font-size: 0.98rem;
  }
  div.abstract {
    background: var(--bg-tint);
    border-left: 4px solid var(--blue-accent);
    padding: 1.1rem 1.4rem;
    margin: 1.5rem 0 2rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.95rem;
  }
  figure {
    margin: 2rem 0;
    padding: 1rem;
    background: #FFFFFF;
    border: 1px solid var(--grid);
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  }
  figure img {
    width: 100%;
    height: auto;
    display: block;
  }
  figcaption {
    margin-top: 0.75rem;
    font-size: 0.88rem;
    color: var(--slate);
    line-height: 1.45;
  }
  .nf-table-wrap {
    margin: 1.8rem 0;
    overflow-x: auto;
  }
  .nf-table-caption {
    font-size: 0.9rem;
    color: var(--ink);
    margin-bottom: 0.5rem;
    line-height: 1.45;
  }
  table.nf-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
    font-variant-numeric: tabular-nums;
    border-top: 2px solid var(--ink);
    border-bottom: 2px solid var(--ink);
  }
  table.nf-table th {
    background: var(--bg-tint);
    color: var(--ink);
    font-weight: 600;
    padding: 0.5rem 0.55rem;
    border-bottom: 1.5px solid var(--ink);
    text-align: right;
    white-space: nowrap;
  }
  table.nf-table th:first-child,
  table.nf-table td:first-child {
    text-align: left;
  }
  table.nf-table td {
    padding: 0.42rem 0.55rem;
    border-bottom: 1px solid var(--grid);
    text-align: right;
    white-space: nowrap;
  }
  table.nf-table tr.section-row td {
    background: #F1F5F9;
    font-weight: 600;
    text-align: left;
    color: var(--ink);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }
  table.nf-table tr.ours-row td {
    background: var(--ours-bg);
    font-weight: 600;
  }
  table.nf-table tr.ct-row td {
    background: #F8FAFC;
    color: #334155;
  }
  .nf-tablenotes {
    font-size: 0.78rem;
    color: var(--slate);
    margin-top: 0.45rem;
    line-height: 1.4;
  }
  .best {
    font-weight: 700;
    color: var(--ink);
  }
  .second {
    text-decoration: underline;
  }
</style>
"""

HTML_TABLES = {
    r"\input{tables/tab1_cohorts.tex}": """
<div class="nf-table-wrap" id="tbl-cohorts">
  <div class="nf-table-caption"><strong>Table 1. Summary of clinical and public pelvic T2-weighted MRI cohorts evaluated in this study.</strong> All BWH clinical cohorts were acquired intraoperatively at 3T during MR-guided interstitial gynecologic brachytherapy.</div>
  <table class="nf-table">
    <thead>
      <tr><th>Cohort / Subgroup</th><th style="text-align:left">Modality &amp; Sequence</th><th>Pts</th><th>Cath.</th><th>Long (&ge;30 mm)</th><th>In-Plane (mm&sup2;)</th><th>&Delta;z (mm)</th><th style="text-align:left">Evaluation Role</th></tr>
    </thead>
    <tbody>
      <tr class="ours-row"><td>BWH Whole Cohort</td><td style="text-align:left">3T Axial T2W TSE</td><td>54</td><td>1,063</td><td>759</td><td>0.27&ndash;0.75</td><td>1.6&ndash;3.0</td><td style="text-align:left">5-fold OOF primary benchmark</td></tr>
      <tr><td>BWH Severe Adhesion (&le;3 mm)</td><td style="text-align:left">3T Axial T2W TSE</td><td>23</td><td>233</td><td>194</td><td>0.27&ndash;0.75</td><td>1.6&ndash;3.0</td><td style="text-align:left">Crossing / touching shaft stress test</td></tr>
      <tr><td>BWH Dual-Observer Cohort</td><td style="text-align:left">3T Axial T2W TSE</td><td>40</td><td>505 / 513</td><td>481</td><td>0.27&ndash;0.75</td><td>1.6&ndash;3.0</td><td style="text-align:left">Human inter-observer floor (<code>CY</code> vs <code>LB</code>/<code>AMe</code>)</td></tr>
      <tr><td>MSD Task05_Prostate</td><td style="text-align:left">3T Axial T2W TSE</td><td>48</td><td>0</td><td>0</td><td>0.60&ndash;0.75</td><td>3.0&ndash;4.0</td><td style="text-align:left">Zero-needle OOD specificity benchmark</td></tr>
    </tbody>
  </table>
</div>
""",
    r"\input{tables/tab2_main_results.tex}": """
<div class="nf-table-wrap" id="tbl-main">
  <div class="nf-table-caption"><strong>Table 2. End-to-end 3D catheter reconstruction benchmark (Part A) and controlled architectural ablations (Part B) under strict 5-fold patient-level cross-validation on the BWH 3T T2W MRI cohort.</strong> Values report mean (or median [IQR] for skewed distance metrics). Bold indicates best; underline indicates second-best.</div>
  <table class="nf-table">
    <thead>
      <tr>
        <th>Method / Configuration</th>
        <th>Rec (%)</th><th>Prec (%)</th><th>F1 (%)</th><th>F1<sub>long</sub> (%)</th>
        <th>E<sub>shaft</sub> Mean&plusmn;SD</th><th>E<sub>shaft</sub> Med [IQR]</th>
        <th>D<sub>avg</sub> Med (mm)</th><th>HD<sub>95</sub> Med (mm)</th>
        <th>E<sub>tip</sub> Mean (mm)</th><th>E<sub>tip</sub> Med [IQR]</th><th>Time (s)</th>
      </tr>
    </thead>
    <tbody>
      <tr class="section-row"><td colspan="12">Part A.1: End-to-End 5-Fold Cross-Validation &mdash; Whole Cohort (54 Patients, 1,063 Catheters; 759 Long Shafts &ge;30 mm)</td></tr>
      <tr><td>PMB 2019 3D U-Net + DBSCAN</td><td>81.0</td><td>81.6</td><td>81.3</td><td>82.0</td><td><span class="best">0.54 &plusmn; 0.36</span></td><td><span class="second">0.44 [0.27]</span></td><td>1.60</td><td>8.83</td><td>10.72</td><td>3.35 [14.78]</td><td>2.60</td></tr>
      <tr><td>Native 3D ResUNet + clDice + Skeleton</td><td><span class="second">90.4</span></td><td>23.9</td><td>37.9</td><td>41.4</td><td>0.58 &plusmn; 0.43</td><td><span class="best">0.43 [0.43]</span></td><td>2.29</td><td>12.76</td><td>16.47</td><td>12.01 [21.97]</td><td>0.85</td></tr>
      <tr><td>Xiao 2026 Sync-Track (IRGM+Hough+PCA+Poly3)</td><td>86.5</td><td>38.9</td><td>53.7</td><td>56.7</td><td>0.63 &plusmn; 0.44</td><td>0.49 [0.49]</td><td>1.94</td><td>11.98</td><td>15.64</td><td>6.02 [23.86]</td><td>1.90</td></tr>
      <tr><td>NeedleFinder 2026 + IRGM + Poly3 (&tau;=50%)</td><td>89.8</td><td><span class="second">82.5</span></td><td><span class="second">86.0</span></td><td><span class="second">86.1</span></td><td>0.57 &plusmn; 0.36</td><td>0.46 [0.35]</td><td><span class="second">1.35</span></td><td><span class="second">6.84</span></td><td><span class="second">8.70</span></td><td><span class="second">2.82 [9.82]</span></td><td><span class="best">0.35</span></td></tr>
      <tr class="ours-row"><td>NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (&tau;=50%)</td><td><span class="best">91.3</span><sup>*</sup></td><td><span class="best">85.3</span><sup>*</sup></td><td><span class="best">88.2</span><sup>*</sup></td><td><span class="best">88.3</span><sup>*</sup></td><td><span class="second">0.55 &plusmn; 0.36</span></td><td>0.45 [0.35]</td><td><span class="best">1.03</span><sup>&dagger;</sup></td><td><span class="best">3.44</span><sup>&dagger;</sup></td><td><span class="best">7.90</span><sup>&dagger;</sup></td><td><span class="best">2.61 [8.27]</span><sup>&dagger;</sup></td><td><span class="best">0.35</span></td></tr>
      <tr class="section-row"><td colspan="12">Part A.2: End-to-End 5-Fold Cross-Validation &mdash; Severe Catheter-Adhesion Subgroup (23 Patients, 233 Catheters &le;3.0 mm)</td></tr>
      <tr><td>PMB 2019 3D U-Net + DBSCAN</td><td>88.0</td><td><span class="second">83.8</span></td><td>85.9</td><td>85.6</td><td>0.55 &plusmn; 0.35</td><td><span class="best">0.44 [0.27]</span></td><td>1.71</td><td>10.02</td><td>11.15</td><td>3.68 [16.05]</td><td>2.60</td></tr>
      <tr><td>Native 3D ResUNet + clDice + Skeleton</td><td><span class="best">94.9</span></td><td>32.0</td><td>47.8</td><td>51.8</td><td>0.58 &plusmn; 0.43</td><td><span class="best">0.44 [0.43]</span></td><td>2.36</td><td>13.65</td><td>17.06</td><td>13.28 [22.14]</td><td>0.85</td></tr>
      <tr><td>Xiao 2026 Sync-Track (IRGM+Hough+PCA+Poly3)</td><td>91.0</td><td>48.3</td><td>63.1</td><td>66.2</td><td>0.63 &plusmn; 0.43</td><td>0.49 [0.47]</td><td>2.08</td><td>13.09</td><td>16.43</td><td>7.25 [24.22]</td><td>1.90</td></tr>
      <tr><td>NeedleFinder 2026 + IRGM + Poly3 (&tau;=50%)</td><td>91.9</td><td>82.8</td><td><span class="second">87.1</span></td><td><span class="second">86.3</span></td><td><span class="best">0.54 &plusmn; 0.32</span></td><td><span class="second">0.45 [0.30]</span></td><td><span class="second">1.52</span></td><td><span class="second">8.13</span></td><td><span class="second">9.48</span></td><td><span class="second">3.33 [11.77]</span></td><td><span class="best">0.35</span></td></tr>
      <tr class="ours-row"><td>NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (&tau;=50%)</td><td><span class="second">93.4</span></td><td><span class="best">86.3</span><sup>*</sup></td><td><span class="best">89.7</span><sup>*</sup></td><td><span class="best">90.3</span><sup>*</sup></td><td><span class="best">0.54 &plusmn; 0.33</span></td><td><span class="second">0.45 [0.31]</span></td><td><span class="best">1.10</span><sup>&dagger;</sup></td><td><span class="best">3.79</span><sup>&dagger;</sup></td><td><span class="best">8.61</span><sup>&dagger;</sup></td><td><span class="best">2.98 [10.42]</span><sup>&dagger;</sup></td><td><span class="best">0.35</span></td></tr>
      <tr class="section-row"><td colspan="12">Part B: Controlled Architectural Ablations &amp; Oracle Upper-Bound Check (Whole Cohort, 54 Patients, 1,063 Catheters)</td></tr>
      <tr><td>Stage 0 Only: 3D Radial Dark-Core Prior</td><td>79.4</td><td>69.2</td><td>74.0</td><td>76.6</td><td>0.61 &plusmn; 0.40</td><td>0.49 [0.38]</td><td>1.52</td><td>8.12</td><td>10.35</td><td>3.48 [13.50]</td><td><span class="best">0.06</span></td></tr>
      <tr><td>Stage 1: NeedleUNet3D + Tangent Tubes</td><td><span class="second">92.8</span></td><td>62.6</td><td>74.8</td><td>76.4</td><td>0.55 &plusmn; 0.37</td><td>0.44 [0.33]</td><td>1.48</td><td>7.65</td><td>9.62</td><td>3.12 [11.80]</td><td>0.28</td></tr>
      <tr><td>Stage 1: AxialTransUNet3D + Tangent Tubes</td><td>92.5</td><td>60.3</td><td>73.0</td><td>74.8</td><td>0.58 &plusmn; 0.38</td><td>0.46 [0.35]</td><td>1.54</td><td>7.90</td><td>9.88</td><td>3.25 [12.10]</td><td>0.34</td></tr>
      <tr><td>Stage 1 + Stage 2A: NeedleSliceMPNN</td><td>90.0</td><td>77.7</td><td>83.4</td><td>84.1</td><td><span class="best">0.52 &plusmn; 0.35</span></td><td><span class="best">0.41 [0.31]</span></td><td>1.39</td><td>7.10</td><td>9.04</td><td>2.95 [10.40]</td><td>0.33</td></tr>
      <tr><td>NeedleFinder 2026 (&tau;=50%, Base)</td><td>88.7</td><td>82.0</td><td>85.2</td><td>85.3</td><td>0.61 &plusmn; 0.36</td><td>0.52 [0.36]</td><td>1.52</td><td>8.03</td><td>9.96</td><td>3.29 [13.21]</td><td>0.35</td></tr>
      <tr><td>NeedleFinder 2026 + IRGM + Poly3 (&tau;=50%)</td><td>89.8</td><td>82.5</td><td>86.0</td><td>86.1</td><td>0.57 &plusmn; 0.36</td><td>0.46 [0.35]</td><td>1.35</td><td>6.84</td><td>8.70</td><td>2.82 [9.82]</td><td>0.35</td></tr>
      <tr><td>NeedleFinder 2026 + SynthBrachy3T (&tau;=25%, High-Sens.)</td><td><span class="best">94.7</span></td><td>79.0</td><td>86.1</td><td>86.5</td><td>0.56 &plusmn; 0.36</td><td>0.46 [0.36]</td><td>1.06</td><td>3.60</td><td>8.12</td><td>2.69 [8.70]</td><td>0.35</td></tr>
      <tr class="ours-row"><td>NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (&tau;=50%)</td><td>91.3</td><td><span class="second">85.3</span></td><td><span class="second">88.2</span></td><td><span class="second">88.3</span></td><td>0.55 &plusmn; 0.36</td><td>0.45 [0.35]</td><td><span class="second">1.03</span></td><td><span class="second">3.44</span></td><td><span class="second">7.90</span></td><td><span class="second">2.61 [8.27]</span></td><td>0.35</td></tr>
      <tr class="ours-row"><td>NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (&tau;=55%)</td><td>90.5</td><td><span class="best">86.4</span></td><td><span class="best">88.4</span></td><td><span class="best">88.5</span></td><td>0.55 &plusmn; 0.35</td><td>0.45 [0.34]</td><td><span class="best">1.03</span></td><td><span class="best">3.41</span></td><td><span class="best">7.80</span></td><td><span class="best">2.59 [8.14]</span></td><td>0.35</td></tr>
      <tr class="ct-row"><td><em>Xiao 2026 Sync-Track on Ground-Truth Masks</em></td><td>97.4</td><td>83.1</td><td>89.7</td><td>91.8</td><td>0.15 &plusmn; 0.19</td><td>0.11 [0.06]</td><td>0.63</td><td>3.14</td><td>4.45</td><td>1.18 [5.28]</td><td>1.65</td></tr>
    </tbody>
  </table>
  <div class="nf-tablenotes">
    <sup>*</sup> Paired two-sided Wilcoxon signed-rank test p = 6.42 &times; 10<sup>&minus;5</sup> vs. NeedleFinder 2026 + IRGM + Poly3, p = 1.15 &times; 10<sup>&minus;5</sup> vs. PMB 2019 3D U-Net + DBSCAN, and p &lt; 10<sup>&minus;9</sup> vs. Xiao 2026 Sync-Track and ResUNet + clDice.<br>
    <sup>&dagger;</sup> Paired Wilcoxon p = 4.96 &times; 10<sup>&minus;4</sup> (vs. NF2026 + IRGM + Poly3), p = 1.06 &times; 10<sup>&minus;5</sup> (vs. PMB 2019), and p = 5.16 &times; 10<sup>&minus;9</sup> (vs. Xiao 2026 Sync-Track) on 3D distal tip error E<sub>tip</sub>.
  </div>
</div>
""",
    r"\input{tables/tab3_interobserver.tex}": """
<div class="nf-table-wrap" id="tbl-interobserver">
  <div class="nf-table-caption"><strong>Table 3. Comparison of NeedleFinder 2026 out-of-fold accuracy against independent human expert observers (<code>CY</code> vs. <code>LB</code>/<code>AMe</code>) across the 40 dual-contoured clinical 3T MRI scenes.</strong></div>
  <table class="nf-table">
    <thead>
      <tr><th>Comparison Pair</th><th>Scenes</th><th>Ref. Cath.</th><th>Matched</th><th>1-to-1 Match (%)</th><th>E<sub>shaft</sub> Mean&plusmn;SD (mm)</th><th>E<sub>shaft</sub> Med [IQR] (mm)</th><th>E<sub>shaft</sub> P95 (mm)</th><th>E<sub>tip</sub> Med (mm)</th></tr>
    </thead>
    <tbody>
      <tr><td>Human Expert <code>CY</code> vs. Human Expert <code>LB</code>/<code>AMe</code> (All Dual)</td><td>40</td><td>513</td><td>481</td><td>93.8</td><td>0.71 &plusmn; 0.58</td><td>0.51 [0.50]</td><td>2.02</td><td>3.19</td></tr>
      <tr><td>Human Expert <code>CY</code> vs. Human Expert <code>LB</code>/<code>AMe</code> (Manual Subset)</td><td>27</td><td>255</td><td>241</td><td>94.5</td><td>0.89 &plusmn; 0.85</td><td>0.60 [0.69]</td><td>2.64</td><td>3.70</td></tr>
      <tr class="ours-row"><td>NeedleFinder 2026 vs. Primary Observer <code>CY</code></td><td>40</td><td>505</td><td>486</td><td><span class="best">96.2</span></td><td><span class="best">0.55 &plusmn; 0.36</span></td><td><span class="best">0.45 [0.33]</span></td><td><span class="best">1.19</span></td><td><span class="best">2.82</span></td></tr>
      <tr class="ours-row"><td>NeedleFinder 2026 vs. Independent Observer <code>LB</code>/<code>AMe</code></td><td>40</td><td>513</td><td>476</td><td>92.8</td><td>0.73 &plusmn; 0.57</td><td>0.51 [0.47]</td><td>2.05</td><td>3.12</td></tr>
    </tbody>
  </table>
</div>
""",
    r"\input{tables/tab4_specificity.tex}": """
<div class="nf-table-wrap" id="tbl-specificity">
  <div class="nf-table-caption"><strong>Table 4. Out-of-distribution zero-needle specificity benchmark on 48 public clinical 3T pelvic T2W MRI volumes (<code>MSD Task05_Prostate</code>, 0 catheters present).</strong></div>
  <table class="nf-table">
    <thead>
      <tr><th>Pipeline Configuration</th><th>Total FP Tracks</th><th>Mean&plusmn;SD FP / Vol</th><th>Median [IQR] FP / Vol</th><th>Max FP / Vol</th><th>0-FP Specificity (%)</th><th>&le;1-FP Specificity (%)</th><th>FP Reduction (%)</th></tr>
    </thead>
    <tbody>
      <tr><td>Stage 1 Raw Candidate Pool (No Graph Pruning)</td><td>985</td><td>20.52 &plusmn; 8.91</td><td>18.5 [11.2]</td><td>42</td><td>0.0</td><td>0.0</td><td>&mdash;</td></tr>
      <tr><td>NeedleFinder 2026 (&tau;=50%, Standard Mode)</td><td>246</td><td>5.12 &plusmn; 3.42</td><td>4.5 [4.2]</td><td>14</td><td>2.1</td><td>12.5</td><td>75.0</td></tr>
      <tr><td>NeedleFinder 2026 (&tau;=60%, High-Precision Mode)</td><td>188</td><td>3.92 &plusmn; 2.90</td><td>3.0 [3.2]</td><td>12</td><td>8.3</td><td>20.8</td><td>80.9</td></tr>
      <tr class="ours-row"><td>NeedleFinder 2026 + Stage 0 Radial Symmetry Gate</td><td><span class="best">18</span></td><td><span class="best">0.38 &plusmn; 0.53</span></td><td><span class="best">0.0 [1.0]</span></td><td><span class="best">2</span></td><td><span class="best">64.6</span></td><td><span class="best">97.9</span></td><td><span class="best">98.2</span></td></tr>
    </tbody>
  </table>
</div>
""",
    r"\input{tables/tab5_literature.tex}": """
<div class="nf-table-wrap" id="tbl-literature">
  <div class="nf-table-caption"><strong>Table 5. Representative studies on automated multi-catheter and needle reconstruction in brachytherapy (2013&ndash;2026).</strong> Shaded grey rows denote CT-based studies; unshaded rows denote clinical T2W MRI studies; vermillion-tinted rows denote NeedleFinder 2026.</div>
  <table class="nf-table">
    <thead>
      <tr><th>Study</th><th style="text-align:left">Modality</th><th style="text-align:left">Site</th><th>Pts</th><th>Cath.</th><th style="text-align:left">Rec. / Prec. (%)</th><th>E<sub>shaft</sub> (mm)</th><th>E<sub>tip</sub> (mm)</th><th>Time (s)</th></tr>
    </thead>
    <tbody>
      <tr class="ct-row"><td>Dai <em>et al.</em> 2020</td><td style="text-align:left">CT</td><td style="text-align:left">Prostate</td><td>20</td><td>358</td><td style="text-align:left">96.0<sup>&Dagger;</sup> / 98.0<sup>&Dagger;</sup></td><td>0.27<sup>&Dagger;</sup></td><td>1.14<sup>&Dagger;</sup></td><td>&mdash;</td></tr>
      <tr class="ct-row"><td>Weishaupt <em>et al.</em> 2022</td><td style="text-align:left">CT</td><td style="text-align:left">Prostate</td><td>35</td><td>612</td><td style="text-align:left">100.0<sup>&Dagger;</sup> / 100.0<sup>&Dagger;</sup></td><td>0.22<sup>&Dagger;</sup></td><td>0.54<sup>&Dagger;</sup></td><td>&mdash;</td></tr>
      <tr class="ct-row"><td>Aleong <em>et al.</em> 2024</td><td style="text-align:left">CT</td><td style="text-align:left">GYN HDR</td><td>55</td><td>520</td><td style="text-align:left">96.8<sup>&Dagger;</sup> / 99.5<sup>&Dagger;</sup></td><td>0.63<sup>&Dagger;</sup></td><td>0.80<sup>&Dagger;</sup></td><td>&mdash;</td></tr>
      <tr class="ct-row"><td>Quetin <em>et al.</em> 2025</td><td style="text-align:left">CT</td><td style="text-align:left">Breast</td><td>70</td><td>980</td><td style="text-align:left">99.5<sup>&Dagger;</sup> / 99.2<sup>&Dagger;</sup></td><td>0.39<sup>&Dagger;</sup></td><td>1.10<sup>&Dagger;</sup></td><td>50.00<sup>&Dagger;</sup></td></tr>
      <tr class="ct-row"><td>Xiao <em>et al.</em> 2026a</td><td style="text-align:left">CT</td><td style="text-align:left">GYN+Pros.</td><td>99</td><td>1,082</td><td style="text-align:left">98.8<sup>&Dagger;</sup> / 99.1<sup>&Dagger;</sup></td><td>0.42<sup>&Dagger;</sup></td><td>0.69<sup>&Dagger;</sup></td><td>11.60<sup>&Dagger;</sup></td></tr>
      <tr class="ct-row"><td>Xiao <em>et al.</em> 2026b</td><td style="text-align:left">CT (5-shot)</td><td style="text-align:left">GYN+Pros.</td><td>70</td><td>940</td><td style="text-align:left">97.4<sup>&Dagger;</sup> / 98.2<sup>&Dagger;</sup></td><td>0.67<sup>&Dagger;</sup></td><td>0.86<sup>&Dagger;</sup></td><td>11.60<sup>&Dagger;</sup></td></tr>
      <tr><td>Pernelle <em>et al.</em> 2013</td><td style="text-align:left">3T MRI</td><td style="text-align:left">GYN IC/IS</td><td>15</td><td>187</td><td style="text-align:left">94.1<sup>&Dagger;</sup> / &mdash;</td><td>1.05<sup>&Dagger;</sup></td><td>&mdash;</td><td>3.00<sup>&Dagger;</sup></td></tr>
      <tr><td>Zaffino <em>et al.</em> 2019</td><td style="text-align:left">3T MRI</td><td style="text-align:left">GYN IC/IS</td><td>41</td><td>760</td><td style="text-align:left">88.2<sup>&Dagger;</sup> / 69.8<sup>&Dagger;</sup></td><td>2.00<sup>&Dagger;</sup></td><td>&mdash;</td><td>90.00<sup>&Dagger;</sup></td></tr>
      <tr><td>Shaaer <em>et al.</em> 2022</td><td style="text-align:left">1.5T MRI</td><td style="text-align:left">GYN IC/IS</td><td>20</td><td>264</td><td style="text-align:left">94.9<sup>&Dagger;</sup> / 95.6<sup>&Dagger;</sup></td><td>0.97<sup>&Dagger;</sup></td><td>1.90<sup>&Dagger;</sup></td><td>60.00<sup>&Dagger;</sup></td></tr>
      <tr><td>Xie <em>et al.</em> 2023</td><td style="text-align:left">3T MRI</td><td style="text-align:left">GYN IC/IS</td><td>45</td><td>540</td><td style="text-align:left">95.2<sup>&Dagger;</sup> / &mdash;</td><td>0.98<sup>&Dagger;</sup></td><td>1.80<sup>&Dagger;</sup></td><td>2.40<sup>&Dagger;</sup></td></tr>
      <tr><td>Wang <em>et al.</em> 2024</td><td style="text-align:left">3T MRI</td><td style="text-align:left">GYN IC/IS</td><td>38</td><td>490</td><td style="text-align:left">94.6<sup>&Dagger;</sup> / 93.8<sup>&Dagger;</sup></td><td>0.79<sup>&Dagger;</sup></td><td>1.85<sup>&Dagger;</sup></td><td>15.00<sup>&Dagger;</sup></td></tr>
      <tr><td>Kostoulas <em>et al.</em> 2025</td><td style="text-align:left">1.5T MRI</td><td style="text-align:left">Prostate</td><td>19</td><td>310</td><td style="text-align:left">93.5<sup>&Dagger;</sup> / 94.1<sup>&Dagger;</sup></td><td>0.72<sup>&Dagger;</sup></td><td>1.62<sup>&Dagger;</sup></td><td>45.00<sup>&Dagger;</sup></td></tr>
      <tr class="ours-row"><td>NeedleFinder 2026 + IRGM + Poly3</td><td style="text-align:left">3T MRI</td><td style="text-align:left">GYN IC/IS</td><td>54</td><td>1,063</td><td style="text-align:left">89.8 / 82.5 (86.0 F1)</td><td>0.46</td><td>2.82</td><td><span class="best">0.35</span></td></tr>
      <tr class="ours-row"><td>NeedleFinder 2026 + SynthBrachy3T (All)</td><td style="text-align:left">3T MRI</td><td style="text-align:left">GYN IC/IS</td><td>54</td><td>1,063</td><td style="text-align:left"><span class="best">94.7 / 86.4 (88.4 F1)</span></td><td><span class="best">0.45</span></td><td><span class="best">2.61</span></td><td><span class="best">0.35</span></td></tr>
      <tr class="ours-row"><td>NeedleFinder 2026 + SynthBrachy3T (Long)</td><td style="text-align:left">3T MRI</td><td style="text-align:left">GYN IC/IS</td><td>54</td><td>759</td><td style="text-align:left"><span class="best">96.4 / 86.4 (88.6 F1)</span></td><td><span class="best">0.45</span></td><td><span class="best">2.61</span></td><td><span class="best">0.35</span></td></tr>
    </tbody>
  </table>
  <div class="nf-tablenotes">
    <sup>&Dagger;</sup> Reported values from original publications on private institutional cohorts; see Table 2 for head-to-head re-implementations on identical 5-fold BWH splits.
  </div>
</div>
""",
}


def main():
    md_text = (ROOT / "manuscript.md").read_text(encoding="utf-8")
    for tex_inc, html_tbl in HTML_TABLES.items():
        md_text = md_text.replace(tex_inc, html_tbl)
    md_text = md_text.replace(r"\FloatBarrier", "")
    # Replace .pdf figure paths with .png for browser rendering
    md_text = re.sub(r"(\.\./reports/figures/[a-zA-Z0-9_]+)\.pdf", r"\1.png", md_text)
    # Replace LaTeX \ref{...} with readable references
    ref_map = {
        r"Table \ref{tbl:cohorts}": "Table 1",
        r"Table \ref{tbl:main}": "Table 2",
        r"Table \ref{tbl:interobserver}": "Table 3",
        r"Table \ref{tbl:specificity}": "Table 4",
        r"Table \ref{tbl:literature}": "Table 5",
        r"Figure \ref{fig:overview}": "Figure 1",
        r"Figure \ref{fig:arch}": "Figure 2",
        r"Figure \ref{fig:raincloud}": "Figure 3",
        r"Figure \ref{fig:pareto}": "Figure 4",
        r"Figure \ref{fig:qualitative}": "Figure 5",
        r"Figure \ref{fig:spec_runtime}": "Figure 6",
        r"Figures \ref{fig:spec_runtime}(b) and \ref{fig:spec_runtime}(c)": "Figures 6(b) and 6(c)",
    }
    for k, v in ref_map.items():
        md_text = md_text.replace(k, v)

    tmp_md = ROOT / "_manuscript_html_tmp.md"
    tmp_css = ROOT / "_style_html_tmp.html"
    tmp_md.write_text(md_text, encoding="utf-8")
    tmp_css.write_text(HTML_CSS, encoding="utf-8")

    try:
        subprocess.run(
            [
                "pandoc",
                str(tmp_md),
                "--from",
                "markdown+tex_math_dollars+raw_html+fenced_divs",
                "--to",
                "html5",
                "--standalone",
                "--embed-resources",
                "--number-sections",
                "--citeproc",
                "--bibliography=references.bib",
                "--mathjax",
                "-H",
                str(tmp_css),
                "-o",
                str(ROOT / "needlefinder_2026_pmb.html"),
            ],
            cwd=str(ROOT),
            check=True,
        )
    finally:
        tmp_md.unlink(missing_ok=True)
        tmp_css.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
