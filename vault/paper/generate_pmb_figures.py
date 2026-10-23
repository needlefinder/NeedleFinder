#!/usr/bin/env python3
"""
Generate all 6 Claude Design System publication-grade vector PDF figures (and PNG/SVG twins)
plus underlying CSV data files for the Physics in Medicine & Biology (PMB) manuscript:
  - fig1_pipeline_overview.pdf
  - fig2_architecture_modules.pdf
  - fig3_raincloud_distributions.pdf
  - fig4_pareto_and_fewshot.pdf
  - fig5_qualitative_mri_3d.pdf
  - fig6_specificity_and_runtime.pdf
"""
import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

BASE_DIR = "/mnt/SSD/github/needlefinder/vault"
STYLE_DIR = os.path.join(BASE_DIR, "paper/style")
FIG_DIR = os.path.join(BASE_DIR, "reports/figures")
DATA_DIR = os.path.join(FIG_DIR, "data")
SOURCES_DIR = os.path.join(BASE_DIR, "sources")

sys.path.insert(0, STYLE_DIR)
from nf_palette import C, ORDER, MM, W1, W2, panel_label, assert_no_overlap

plt.style.use(os.path.join(STYLE_DIR, "nf.mplstyle"))
os.makedirs(DATA_DIR, exist_ok=True)


def load_all_sources():
    with open(os.path.join(SOURCES_DIR, "2026-10-23-unified-clean-nested-benchmark.json")) as f:
        bench = json.load(f)
    with open(os.path.join(SOURCES_DIR, "2026-10-23-table3-observers-and-nf-clean.json")) as f:
        inter = json.load(f)
    with open(os.path.join(SOURCES_DIR, "2026-10-23-public-pelvic-mri-specificity-clean.json")) as f:
        spec = json.load(f)
    qual = dict(np.load(os.path.join(DATA_DIR, "qualitative_cases.npz"), allow_pickle=True))
    pred_path = os.path.join(DATA_DIR, "qualitative_predictions.npz")
    if os.path.exists(pred_path):
        qp = dict(np.load(pred_path, allow_pickle=True))
        qual.update(qp)
        if "dense_pred_ras_obj" not in qual and "case017_nf2026" in qp:
            qual["dense_pred_ras_obj"] = qp["case017_nf2026"]
        if "adhesion_pred_ras_obj" not in qual and "case031_nf2026" in qp:
            qual["adhesion_pred_ras_obj"] = qp["case031_nf2026"]
    return bench, inter, spec, qual


def project_polylines_to_slice(polylines, ijk_to_ras, crop_origin, z_mid):
    """Intersect RAS polylines with the displayed axial slice; return crop-pixel (x, y) per polyline."""
    inv = np.linalg.inv(np.asarray(ijk_to_ras, dtype=np.float64))
    vox = np.array([crop_origin[0], crop_origin[1], crop_origin[2] + z_mid, 1.0], dtype=np.float64)
    z_ras = float((np.asarray(ijk_to_ras, dtype=np.float64) @ vox)[2])
    out = []
    for g in polylines:
        g = np.asarray(g, dtype=np.float64)
        g = g[np.argsort(g[:, 2])]
        if len(g) < 2 or not (g[0, 2] <= z_ras <= g[-1, 2]):
            continue
        x = np.interp(z_ras, g[:, 2], g[:, 0])
        y = np.interp(z_ras, g[:, 2], g[:, 1])
        pix = inv @ np.array([x, y, z_ras, 1.0])
        out.append([pix[0] - crop_origin[0], pix[1] - crop_origin[1]])
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def ortho_project(gs, az_deg=35.0, el_deg=20.0):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    gs = np.asarray(gs, dtype=np.float64)
    gs = gs[np.argsort(gs[:, 2])]
    x_p = gs[:, 0] * np.cos(az) - gs[:, 1] * np.sin(az)
    y_p = gs[:, 2] * np.cos(el) - (gs[:, 0] * np.sin(az) + gs[:, 1] * np.cos(az)) * np.sin(el)
    return x_p, y_p


def export_csv_data(bench, inter, spec):
    # 1. pareto_sweep.csv
    with open(os.path.join(DATA_DIR, "pareto_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "tau", "recall_1to1", "precision_1to1", "f1_1to1", "shaft_mean_mm", "shaft_med_mm", "tip_med_mm"])
        for k, v in bench["cohort_all_1063_catheters_54cases"].items():
            tau = ""
            if "tau=" in k:
                tau = k.split("tau=")[1].split("%")[0]
            w.writerow([
                k, tau,
                f"{v['hungarian_1to1_recall_pct']:.2f}",
                f"{v['precision_1to1_pct']:.2f}",
                f"{v['f1_1to1_pct']:.2f}",
                f"{v['metrics']['mean_shaft_xy']['mean']:.3f}",
                f"{v['metrics']['mean_shaft_xy']['median']:.3f}",
                f"{v['metrics']['tip_3d']['median']:.3f}",
            ])

    # 2. fewshot_curve.csv
    with open(os.path.join(DATA_DIR, "fewshot_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shots", "f1_all_1063", "f1_long_adj_759", "shaft_mean_mm", "tip_mean_mm", "tip_med_mm"])
        fs = bench["few_shot_clinical_annotation_budget"]
        for k_str, k_key in [
            ("0_analytical", "0_shot_stage0_analytical"),
            ("0_synth", "0_shot"),
            ("1", "1_shot"),
            ("5", "5_shot"),
            ("10", "10_shot"),
            ("20", "20_shot"),
            ("43", "43_shot"),
        ]:
            r = fs[k_key]
            w.writerow([
                k_str,
                f"{r['cohort_all_1063_catheters']['f1_1to1_pct']:.2f}",
                f"{r['cohort_long_759_cy_catheters']['f1_1to1_obturator_adj_pct']:.2f}",
                f"{r['cohort_all_1063_catheters']['metrics']['mean_shaft_xy']['mean']:.3f}",
                f"{r['cohort_all_1063_catheters']['metrics']['tip_3d']['mean']:.3f}",
                f"{r['cohort_all_1063_catheters']['metrics']['tip_3d']['median']:.3f}",
            ])

    # 3. msd_specificity.csv
    with open(os.path.join(DATA_DIR, "msd_specificity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "unpruned_pool", "stage1_tubes_only", "nf2026_tau50", "nf2026_sym_gated_tau50", "clean_ssl_stage1_sym_gate"])
        for c in spec["per_case_results"]:
            w.writerow([
                c["case_id"],
                c["candidate_pool_before_graph_transformer"],
                c["raw_unet_tubes_no_physics_or_gt"],
                c["unified_needlefinder_2026_by_tau"]["50"],
                c["unified_needlefinder_2026_sym_gated_by_tau"]["50"],
                c["clean_ssl_stage1_sym_tubes_only"],
            ])

    # 4. runtime.csv
    with open(os.path.join(DATA_DIR, "runtime.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "cpu_sec", "gpu_sec", "cpu_std_res_sec", "cpu_high_res_sec", "gpu_std_res_sec", "gpu_high_res_sec"])
        w.writerow(["Stage 0: Rust 3D Radial Symmetry", 0.267, 0.267, 0.072, 0.381, 0.072, 0.381])
        w.writerow(["Stage 1: NeedleUNet3D + IRGM", 2.575, 1.330, 0.739, 3.655, 0.256, 1.962])
        w.writerow(["Stage 2A: Native Seed Tracking + Triplet MPNN", 1.248, 1.046, 0.402, 1.746, 0.322, 1.472])
        w.writerow(["Stage 2B: Bundle Graph Transformer + Poly3", 0.061, 0.032, 0.047, 0.069, 0.024, 0.036])
        w.writerow(["Total End-to-End", 4.151, 2.674, 1.260, 5.852, 0.675, 3.851])


def save_all_formats(fig, basename):
    pdf_path = os.path.join(FIG_DIR, f"{basename}.pdf")
    svg_path = os.path.join(FIG_DIR, f"{basename}.svg")
    png_path = os.path.join(FIG_DIR, f"{basename}.png")
    fig.savefig(pdf_path, format="pdf", dpi=600)
    fig.savefig(svg_path, format="svg", dpi=600)
    fig.savefig(png_path, format="png", dpi=300)
    # Also save to legacy names if fig1/2/3 so any existing links stay valid
    legacy_map = {
        "fig1_pipeline_overview": "fig1_pipeline_architecture",
        "fig4_pareto_and_fewshot": "fig2_pareto_precision_recall_f1",
        "fig6_specificity_and_runtime": "fig3_shaft_error_and_runtime",
    }
    if basename in legacy_map:
        leg = legacy_map[basename]
        fig.savefig(os.path.join(FIG_DIR, f"{leg}.pdf"), format="pdf", dpi=600)
        fig.savefig(os.path.join(FIG_DIR, f"{leg}.svg"), format="svg", dpi=600)
    print(f"Saved {pdf_path}")


# =========================================================================
# FIGURE 1: 2-Stage Pipeline Overview (170 x 72 mm)
# =========================================================================
def make_fig1_pipeline_overview(qual):
    fig = plt.figure(figsize=(W2, 72 * MM), constrained_layout=False)

    mri = qual["dense_mri_slice"]
    sym = qual["dense_sym_slice"]
    peak = qual["dense_peak_slice"]
    mask = qual["dense_mask_slice"]
    obt = qual["dense_obt_slice"]
    gt_xy = qual["dense_gt_xy_slice"]
    gt_ras = qual["dense_gt_ras_obj"]

    H, W = mri.shape
    y0, y1 = int(H * 0.12), int(H * 0.88)
    x0, x1 = int(W * 0.12), int(W * 0.88)
    mri_c = mri[y0:y1, x0:x1]
    sym_c = sym[y0:y1, x0:x1]
    peak_c = peak[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]
    obt_c = obt[y0:y1, x0:x1]

    # Top row: 5 visual tiles across 170 mm
    tile_w = 0.165
    tile_h = 0.56
    y_bot = 0.31
    xs = [0.015 + i * 0.198 for i in range(5)]

    titles = [
        ("3T T2W pelvic MRI", "0.31–0.88 mm in-plane", "#333333"),
        ("Stage 0 · Radial prior", "Multi-radii dark-core", C["st0"]),
        ("Stage 1 · ResUNet+IRGM", "Centerline + implant gate", C["st1"]),
        ("Stage 2 · MPNN+GraphTF", "Triplet + bundle pruning", C["st2"]),
        ("Output · 3D catheters", "OOF predictions vs. reference", C["nf"]),
    ]

    for idx_t, (x_left, (t_main, t_sub, col)) in enumerate(zip(xs, titles)):
        ax = fig.add_axes([x_left, y_bot, tile_w, tile_h])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(col)
            sp.set_linewidth(1.1 if idx_t > 0 else 0.7)

        if idx_t == 0:
            vmin, vmax = np.percentile(mri_c, [2, 98])
            ax.imshow(mri_c, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
            # 10 mm scale bar
            px_10mm = 10.0 / float(qual["dense_spacing"][0])
            ax.plot([10, 10 + px_10mm], [mri_c.shape[0] - 10, mri_c.shape[0] - 10], color="white", lw=1.6)
            ax.text(10 + px_10mm / 2, mri_c.shape[0] - 16, "10 mm", color="white", fontsize=6, ha="center", va="bottom")
        elif idx_t == 1:
            vmin, vmax = np.percentile(mri_c, [2, 98])
            ax.imshow(mri_c, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", alpha=0.85)
            sym_ma = np.ma.masked_where(sym_c < 0.18, sym_c)
            ax.imshow(sym_ma, cmap="YlOrBr", vmin=0.1, vmax=0.95, origin="upper", alpha=0.85)
        elif idx_t == 2:
            vmin, vmax = np.percentile(mri_c, [2, 98])
            ax.imshow(mri_c, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", alpha=0.65)
            pk_ma = np.ma.masked_where(peak_c < 0.12, peak_c)
            ax.imshow(pk_ma, cmap="Blues", vmin=0.1, vmax=0.9, origin="upper", alpha=0.88)
            ax.contour(mask_c, levels=[0.35], colors=[C["st1"]], linewidths=0.7)
            if np.any(obt_c > 0.3):
                ax.contour(obt_c, levels=[0.4], colors=[C["st0"]], linewidths=0.9, linestyles="--")
        elif idx_t == 3:
            ax.set_facecolor("#F8FAF9")
            # 3-slice 3D oblique schematic of candidate nodes & edges
            np.random.seed(7)
            pts_base = gt_xy - np.array([x0, y0])
            pts_base = pts_base[(pts_base[:, 0] > 5) & (pts_base[:, 0] < (x1 - x0 - 5))]
            pts_base = pts_base[:14]
            pts_norm = pts_base / np.array([x1 - x0, y1 - y0])
            z_levels = [0.20, 0.50, 0.80]
            for zl in z_levels:
                px_plane = [0.08, 0.84, 0.92, 0.16, 0.08]
                py_plane = [zl - 0.09, zl - 0.09, zl + 0.09, zl + 0.09, zl - 0.09]
                ax.plot(px_plane, py_plane, color="#CCCCCC", lw=0.5)
            for p_i, pt in enumerate(pts_norm):
                xs_n = [0.14 + 0.72 * pt[0] + (z_idx - 1) * 0.015 for z_idx in range(3)]
                ys_n = [z_levels[z_idx] + (pt[1] - 0.5) * 0.14 for z_idx in range(3)]
                ax.plot(xs_n, ys_n, color=C["st2"], lw=1.0, alpha=0.85)
                ax.scatter(xs_n, ys_n, s=9, color=C["st2"], zorder=4)
            # 2 rejected false-positive tracks in grey dashed
            for fp_x in [0.22, 0.78]:
                xs_f = [fp_x, fp_x + 0.03, fp_x - 0.02]
                ys_f = [z_levels[0] - 0.03, z_levels[1] + 0.02, z_levels[2] - 0.02]
                ax.plot(xs_f, ys_f, color="#999999", lw=0.8, ls="--")
                ax.scatter(xs_f, ys_f, s=8, color="#999999", marker="x", zorder=4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0.05, 0.95)
        elif idx_t == 4:
            ax.set_facecolor("#FAF8F7")
            # 3D orthographic projection: expert reference (grey) and OOF predictions (vermilion) of Case 017
            pred_ras = qual.get("dense_pred_ras_obj")
            for g in gt_ras:
                x_p, y_p = ortho_project(g)
                ax.plot(x_p, y_p, color="#BDBDBD", lw=2.0, alpha=0.6, solid_capstyle="round")
            if pred_ras is not None:
                for g in pred_ras:
                    x_p, y_p = ortho_project(g)
                    ax.plot(x_p, y_p, color=C["nf"], lw=1.1, alpha=0.92, solid_capstyle="round")
                ax.scatter([x_p[-1]], [y_p[-1]], s=8, color=C["nf"], edgecolor="white", linewidth=0.3, zorder=5)
            ax.margins(0.10)

        ax.text(0.5, 1.13, t_main, transform=ax.transAxes, fontsize=6.6, fontweight="bold", color=col, ha="center", va="bottom")
        ax.text(0.5, 1.02, t_sub, transform=ax.transAxes, fontsize=5.6, color="#4D4D4D", ha="center", va="bottom")

        # Flow arrows between tiles
        if idx_t < 4:
            x_arr = x_left + tile_w + 0.004
            fig.lines.append(Line2D([x_arr, x_arr + 0.022], [y_bot + tile_h / 2, y_bot + tile_h / 2],
                                    transform=fig.transFigure, color="#666666", lw=1.0))
            fig.text(x_arr + 0.022, y_bot + tile_h / 2, "▸", fontsize=8, color="#666666", ha="center", va="center")

    # Bottom row: Proportional runtime bar (CPU 4.15 s / GPU 2.67 s)
    ax_bar = fig.add_axes([0.14, 0.06, 0.83, 0.14])
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)

    segs = [
        ("Stage 0: Rust symmetry", 0.267, 0.267, C["st0"]),
        ("Stage 1: NeedleUNet3D + IRGM", 2.575, 1.330, C["st1"]),
        ("Stage 2A: Native tracking + MPNN", 1.248, 1.046, C["st2"]),
        ("Stage 2B/C: GraphTF + 1D Tip + Poly3", 0.061, 0.032, C["nf"]),
    ]
    y_pos = [1.0, 0.0]
    left_cpu, left_gpu = 0.0, 0.0
    for label, t_cpu, t_gpu, col in segs:
        ax_bar.barh(1.0, t_cpu, left=left_cpu, height=0.52, color=col, edgecolor="white", linewidth=0.6)
        ax_bar.barh(0.0, t_gpu, left=left_gpu, height=0.52, color=col, edgecolor="white", linewidth=0.6)
        left_cpu += t_cpu
        left_gpu += t_gpu

    ax_bar.text(left_cpu + 0.07, 1.0, f"{left_cpu:.2f} s total (CPU; 1.26 s std / 5.85 s hi-res)", va="center", ha="left", fontsize=6.4, fontweight="bold", color="#222222")
    ax_bar.text(left_gpu + 0.07, 0.0, f"{left_gpu:.2f} s total (RTX 4090; 0.68 s std / 3.85 s hi-res)", va="center", ha="left", fontsize=6.4, fontweight="bold", color="#222222")
    ax_bar.set_yticks([0.0, 1.0])
    ax_bar.set_yticklabels(["GPU (2.67 s)", "CPU (4.15 s)"], fontsize=7, fontweight="bold")
    ax_bar.set_xlim(0, 6.60)
    ax_bar.set_xlabel("End-to-end inference time per patient volume (s)", fontsize=7.5)
    ax_bar.tick_params(axis="y", length=0)

    # Legend swatches above runtime bar
    leg_items = [
        (0.11, "■ Stage 0: Symmetry (0.27 s)", C["st0"]),
        (0.33, "■ Stage 1: ResUNet+IRGM (2.58 s)", C["st1"]),
        (0.58, "■ Stage 2A: MPNN (1.25 s)", C["st2"]),
        (0.79, "■ Stage 2B: GraphTF+Tip (0.06 s)", C["nf"]),
    ]
    for lx, ltxt, col in leg_items:
        fig.text(lx, 0.225, ltxt, fontsize=5.5, color=col, fontweight="bold")

    assert_no_overlap(fig, pad_px=1)
    save_all_formats(fig, "fig1_pipeline_overview")
    plt.close(fig)


# =========================================================================
# FIGURE 2: Architecture Modules Schematic (170 x 68 mm)
# =========================================================================
def make_fig2_architecture_modules():
    fig = plt.figure(figsize=(W2, 68 * MM), constrained_layout=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 170)
    ax.set_ylim(0, 68)
    ax.axis("off")

    # Panel labels
    ax.text(2.0, 64.0, "a", fontsize=9, fontweight="bold", va="top", ha="left")
    ax.text(7.5, 64.0, "Stage 1 · Anisotropic 3D ResUNet", fontsize=7.4, fontweight="bold", color=C["st1"], va="top")

    ax.text(88.0, 64.0, "b", fontsize=9, fontweight="bold", va="top", ha="left")
    ax.text(93.5, 64.0, "Stage 2 · Seed tracking, Triplet MPNN & GraphTF", fontsize=7.4, fontweight="bold", color=C["st2"], va="top")

    # Divider line
    ax.plot([85, 85], [4, 65], color="#D9D9D9", lw=0.7, ls="--")

    # --- Panel a (x: 2..83) ---
    # Input boxes
    b_mri = FancyBboxPatch((3, 44), 17, 11, boxstyle="round,pad=0.8", fc="#F5F7FA", ec="#4D4D4D", lw=0.7)
    b_s0 = FancyBboxPatch((3, 27), 17, 11, boxstyle="round,pad=0.8", fc="#FDF5E6", ec=C["st0"], lw=0.9)
    ax.add_patch(b_mri)
    ax.add_patch(b_s0)
    ax.text(11.5, 49.5, "T2W MRI\n$I(x,y,z)$", ha="center", va="center", fontsize=6.8, color="#222222")
    ax.text(11.5, 32.5, "Stage 0 prior\n$S_0(x,y,z)$", ha="center", va="center", fontsize=6.8, color=C["st0"], fontweight="bold")

    # Encoder-Decoder U-Net blocks
    enc_boxes = [
        (24, 42, 8, 14, "1×3×3\n24 ch"),
        (34, 38, 8, 12, "3×3×3\n48 ch"),
        (44, 34, 8, 10, "SE-Res\n96 ch"),
        (54, 42, 8, 14, "Up+Cat\n48 ch"),
    ]
    for bx, by, bw, bh, btxt in enc_boxes:
        p = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.5", fc="#EBF3F9", ec=C["st1"], lw=0.8)
        ax.add_patch(p)
        ax.text(bx + bw / 2, by + bh / 2, btxt, ha="center", va="center", fontsize=6.0, color="#111111")

    # Arrows connecting encoder/decoder
    ax.annotate("", xy=(24, 49), xytext=(20.5, 49), arrowprops=dict(arrowstyle="->", color=C["st1"], lw=0.8))
    ax.annotate("", xy=(34, 44), xytext=(32.2, 46), arrowprops=dict(arrowstyle="->", color=C["st1"], lw=0.8))
    ax.annotate("", xy=(44, 39), xytext=(42.2, 41), arrowprops=dict(arrowstyle="->", color=C["st1"], lw=0.8))
    ax.annotate("", xy=(54, 46), xytext=(52.2, 41), arrowprops=dict(arrowstyle="->", color=C["st1"], lw=0.8))

    # IRGM Callout Inset (bottom of panel a)
    b_irgm = FancyBboxPatch((23, 7), 38, 19, boxstyle="round,pad=0.8", fc="#FFFFFF", ec=C["st1"], lw=0.8, ls="--")
    ax.add_patch(b_irgm)
    ax.text(42, 23.0, "Implant-region gate (Stage 2B)", ha="center", va="center", fontsize=6.4, fontweight="bold", color=C["st1"])
    ax.text(42, 16.5, "center = median XY of candidates (conf. ≥ 55%)", ha="center", va="center", fontsize=5.6, color="#111111")
    ax.text(42, 10.2, "radius = P98 distance + 16 mm", ha="center", va="center", fontsize=5.8, color="#333333")

    # Arrow from S0 to IRGM and IRGM to Decoder
    ax.annotate("", xy=(23, 16.5), xytext=(11.5, 26.8), arrowprops=dict(arrowstyle="->", color=C["st0"], lw=0.9, connectionstyle="angle,angleA=-90,angleB=180,rad=3"))
    ax.annotate("", xy=(58, 41.8), xytext=(51, 26.2), arrowprops=dict(arrowstyle="->", color=C["st1"], lw=0.8))

    # 5 Output Heads on right of Panel a
    heads = [
        ("$P_{\\mathrm{mask}}$", "Tubular mask"),
        ("$P_{\\mathrm{peak}}$", "2D Gaussian peak"),
        ("$V_x$", "Slope $\\Delta x / \\Delta z$"),
        ("$V_y$", "Slope $\\Delta y / \\Delta z$"),
        ("$P_{\\mathrm{obt}}$", "Obturator mask"),
    ]
    for h_i, (h_sym, h_desc) in enumerate(heads):
        hy = 52 - h_i * 9.8
        hp = FancyBboxPatch((66, hy), 16, 7.6, boxstyle="round,pad=0.4", fc="#F2F8FC", ec=C["st1"], lw=0.7)
        ax.add_patch(hp)
        ax.text(74, hy + 5.3, h_sym, ha="center", va="center", fontsize=6.5, fontweight="bold", color=C["st1"])
        ax.text(74, hy + 1.5, h_desc, ha="center", va="center", fontsize=5.3, color="#4D4D4D")
        ax.annotate("", xy=(65.8, hy + 3.8), xytext=(62.2, 49.0), arrowprops=dict(arrowstyle="->", color=C["st1"], lw=0.6))

    # --- Panel b (x: 88..168) ---
    # Left of Panel b: 3 axial slice planes z-1, z, z+1 + Triplet MPNN
    z_ys = [22, 36, 50]
    z_lbls = ["Slice $z-1$", "Slice $z$", "Slice $z+1$"]
    for zy, zlb in zip(z_ys, z_lbls):
        plane_x = [90, 114, 118, 94, 90]
        plane_y = [zy - 3.5, zy - 3.5, zy + 3.5, zy + 3.5, zy - 3.5]
        ax.fill(plane_x, plane_y, color="#EDF7F4", ec=C["st2"], lw=0.6, alpha=0.8)
        ax.text(90.5, zy + 4.2, zlb, fontsize=6.0, color="#333333")

    # Highlighted triplet nodes (v_i^{z-1}, v_j^z, v_k^{z+1})
    tri_pts = [(102, 22), (104, 36), (106, 50)]
    ax.plot([p[0] for p in tri_pts], [p[1] for p in tri_pts], color=C["st2"], lw=1.6, zorder=4)
    for px, py in tri_pts:
        ax.scatter([px], [py], s=26, color=C["st2"], edgecolor="white", linewidth=0.6, zorder=5)
    # Adhering neighbor trajectory separated by Hungarian cost matrix
    adh_pts = [(106.5, 21.5), (105.2, 35.8), (103.8, 49.5)]
    ax.plot([p[0] for p in adh_pts], [p[1] for p in adh_pts], color=C["st0"], lw=1.2, ls="-.", zorder=4)
    for px, py in adh_pts:
        ax.scatter([px], [py], s=18, color=C["st0"], zorder=5)

    # Triplet MPNN callout box
    b_mpnn = FancyBboxPatch((89, 5), 40, 12, boxstyle="round,pad=0.6", fc="#FFFFFF", ec=C["st2"], lw=0.7)
    ax.add_patch(b_mpnn)
    ax.text(109, 13.5, "Seed tracking + 4× Triplet MPNN", ha="center", va="center", fontsize=6.3, fontweight="bold", color=C["st2"])
    ax.text(109, 8.2, "edge update from $h_i, h_j, e_{ij}$ + in/out messages", ha="center", va="center", fontsize=6.5, color="#111111")

    # Right of Panel b: Bundle Graph Transformer & Poly3 Output
    b_gat = FancyBboxPatch((133, 39), 34, 16, boxstyle="round,pad=0.6", fc="#EDF7F4", ec=C["st2"], lw=0.8)
    ax.add_patch(b_gat)
    ax.text(150, 50.5, "Bundle Graph Transformer", ha="center", va="center", fontsize=6.6, fontweight="bold", color=C["st2"])
    ax.text(150, 45.5, "2× edge-biased attention\n+ 1× global attention", ha="center", va="center", fontsize=6.1, color="#222222")
    ax.text(150, 40.8, "26-D node · 8-D pair edge", ha="center", va="center", fontsize=5.8, color="#4D4D4D")

    b_out = FancyBboxPatch((133, 18), 34, 15, boxstyle="round,pad=0.6", fc=C["nf_tint"], ec=C["nf"], lw=0.9)
    ax.add_patch(b_out)
    ax.text(150, 29.0, r"Keep if $p_t \geq \tau$ + Poly3", ha="center", va="center", fontsize=6.6, fontweight="bold", color=C["nf"])
    ax.text(150, 23.8, r"3rd-order polynomial $x(z), y(z)$", ha="center", va="center", fontsize=6.1, color="#222222")
    ax.text(150, 19.8, "Prunes 77% of vessel FPs", ha="center", va="center", fontsize=5.8, color=C["nf"], fontweight="bold")

    ax.annotate("", xy=(132.8, 47), xytext=(118.5, 36), arrowprops=dict(arrowstyle="->", color=C["st2"], lw=0.9))
    ax.annotate("", xy=(150, 33.3), xytext=(150, 38.8), arrowprops=dict(arrowstyle="->", color=C["nf"], lw=0.9))

    assert_no_overlap(fig, pad_px=1)
    save_all_formats(fig, "fig2_architecture_modules")
    plt.close(fig)


# =========================================================================
# FIGURE 3: Raincloud Distributions (Whole Cohort vs Severe Adhesion Subgroup)
# =========================================================================
def draw_raincloud_panel(ax, dist_list, colors, labels, xlim, show_ylabels, human_iqr, human_p95, xlabel):
    n_m = len(dist_list)
    np.random.seed(42)

    # Human IQR background band and P95 dashed line
    ax.axvspan(human_iqr[0], human_iqr[1], color="#E0E0E0", alpha=0.55, zorder=0)
    ax.axvline(human_p95, color="#000000", ls="--", lw=0.7, alpha=0.75, zorder=1)

    for idx_m, (arr, col, lbl) in enumerate(zip(dist_list, colors, labels)):
        y_base = n_m - 1 - idx_m
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue

        med = float(np.median(arr))
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        p5, p95 = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
        mean_v = float(np.mean(arr))

        arr_clip = arr[arr <= xlim[1]]
        n_over = int(np.sum(arr > xlim[1]))

        # 1. Half-violin cloud above baseline
        if len(arr_clip) >= 10 and np.std(arr_clip) > 1e-4:
            kde = gaussian_kde(arr_clip, bw_method="scott")
            xs_g = np.linspace(max(xlim[0], arr_clip.min()), min(xlim[1], arr_clip.max()), 120)
            dens = kde(xs_g)
            if dens.max() > 0:
                dens = dens / dens.max() * 0.38
                ax.fill_between(xs_g, y_base + 0.04, y_base + 0.04 + dens, facecolor=col, alpha=0.35, edgecolor=col, lw=0.6, zorder=3)

        # 2. IQR box + P5-P95 whiskers on baseline
        w_low = max(xlim[0], p5)
        w_high = min(xlim[1], p95)
        ax.plot([w_low, w_high], [y_base, y_base], color=col, lw=0.8, zorder=4)
        if p95 <= xlim[1]:
            ax.plot([w_high, w_high], [y_base - 0.07, y_base + 0.07], color=col, lw=0.9, zorder=4)
        box_w = max(0.02, min(xlim[1], q75) - max(xlim[0], q25))
        rect = mpatches.Rectangle((max(xlim[0], q25), y_base - 0.06), box_w, 0.12,
                                  facecolor="white", edgecolor=col, linewidth=0.8, zorder=4)
        ax.add_patch(rect)
        ax.plot([med, med], [y_base - 0.06, y_base + 0.06], color=col, lw=1.3, zorder=5)

        # 3. Mean diamond
        if mean_v <= xlim[1]:
            ax.scatter([mean_v], [y_base], marker="D", s=11, facecolor="white", edgecolor="#000000", linewidth=0.7, zorder=6)

        # 4. Subsampled jittered rain below baseline
        sub = arr_clip if len(arr_clip) <= 220 else np.random.choice(arr_clip, size=220, replace=False)
        jit = np.random.uniform(-0.32, -0.11, size=len(sub))
        ax.scatter(sub, y_base + jit, s=1.8, color=col, alpha=0.25, edgecolors="none", zorder=2)

        # 5. Right-aligned median / P95 label
        stat_txt = f"{med:.2f} / {p95:.2f}" + (f"  (+{n_over} off-scale)" if n_over > 0 else "")
        ax.text(xlim[1] * 0.985, y_base + 0.24, stat_txt, ha="right", va="center",
                fontsize=6.0, color="#222222",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.78), zorder=7)

    ax.set_xlim(xlim)
    ax.set_ylim(-0.55, n_m - 0.35)
    ax.set_yticks(range(n_m))
    if show_ylabels:
        ax.set_yticklabels(list(reversed(labels)), fontsize=6.8)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.grid(axis="x", color="#E6E6E6", lw=0.4, zorder=0)


def make_fig3_raincloud_distributions(bench, inter):
    """Whole-cohort per-catheter error distributions from the measured OOF arrays only.

    Only quantities that exist as per-catheter arrays in the benchmark JSON are drawn
    (in-plane shaft error and 3D tip error of the matched catheters of each method).
    The human inter-observer reference is drawn from its summary statistics (IQR band
    and P95 line); no synthetic samples are generated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(W2, 64 * MM), constrained_layout=True)
    raw_all = bench["per_patient_and_needle_error_distributions"]["all_1063"]

    methods_cfg = [
        ("NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=50%)", "NeedleFinder 2026 + Synth", C["nf"]),
        ("Unified NeedleFinder 2026 + IRGM + SyncTrack + Poly3 (tau=50%)", "NeedleFinder 2026 + IRGM", C["st2"]),
        ("Baseline 3: Xiao 2026 IRGM + 3D Hough + Sync Inward Tracking + Poly3", "Xiao 2026 Sync-Track", C["b_xiao"]),
        ("Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)", "ResUNet + clDice + Skel.", C["b_nnunet"]),
        ("Baseline 1: PMB 2019 Isotropic 1mm³ 3D U-Net + DBSCAN/Skeleton", "PMB 2019 3D U-Net", C["b_pmb19"]),
    ]
    # Human reference statistics from the Table 3 clean export (40/40 dual-traced scenes)
    t3_long = inter["table3_rows"]["2_human_inter_observer_long_subset"]["metrics"]
    h_sh = t3_long["mean_shaft_xy"]
    h_tip = t3_long["tip_3d"]

    labels = [m[1] for m in methods_cfg]
    colors = [m[2] for m in methods_cfg]
    shaft = [np.asarray(raw_all[m[0]]["per_needle_mean_shaft_xy_mm"], dtype=float) for m in methods_cfg]
    tip = [np.asarray(raw_all[m[0]]["per_needle_tip_3d_mm"], dtype=float) for m in methods_cfg]

    draw_raincloud_panel(axes[0], shaft, colors, labels, (0.0, 2.5), True,
                         (h_sh["q25"], h_sh["q75"]), h_sh["p95"], r"$E_{\mathrm{shaft}}$ (mm)  [median / P95]")
    # Human tip reference: only the median is reported for the dual-observer tip discrepancy
    draw_raincloud_panel(axes[1], tip, colors, labels, (0.0, 15.0), False,
                         (h_tip["median"], h_tip["median"]), h_tip["median"], r"$E_{\mathrm{tip}}$ (mm)  [median / P95]")

    axes[0].set_title("a   In-plane shaft error $E_{\\mathrm{shaft}}$", fontsize=6.8, fontweight="bold", loc="left")
    axes[1].set_title("b   3D tip error $E_{\\mathrm{tip}}$", fontsize=6.8, fontweight="bold", loc="left")

    assert_no_overlap(fig, pad_px=1)
    save_all_formats(fig, "fig3_raincloud_distributions")
    plt.close(fig)


# =========================================================================
# FIGURE 4: Precision-Recall Pareto Frontier + Few-Shot Efficiency (170 x 72 mm)
# =========================================================================
def make_fig4_pareto_and_fewshot(bench):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 74 * MM), gridspec_kw={"width_ratios": [1.12, 1.0]}, constrained_layout=True)

    c_all = bench["cohort_all_1063_catheters_54cases"]

    # --- Panel a: 1-to-1 Precision-Recall Pareto Frontier ---
    r_grid = np.linspace(72, 99.5, 200)
    for f1_val in [0.75, 0.80, 0.85, 0.90]:
        f1_pct = f1_val * 100.0
        denom = 2.0 * r_grid - f1_pct
        valid = denom > 1.0
        p_grid = (f1_pct * r_grid[valid]) / denom[valid]
        mask_v = (p_grid >= 57) & (p_grid <= 91.5)
        ax1.plot(r_grid[valid][mask_v], p_grid[mask_v], color="#CCCCCC", lw=0.5, ls="--", zorder=1)
        if f1_val in (0.80, 0.85, 0.90):
            y_top = 90.9
            x_top = (f1_pct * y_top) / (2.0 * y_top - f1_pct)
            ax1.text(x_top - 0.6, y_top, f"F1={f1_val:.2f}", fontsize=5.5, color="#808080", ha="right", va="top")

    # NeedleFinder 2026 + SynthBrachy3T tau sweep
    taus = [25, 35, 45, 50, 55, 60]
    rs_nf = [c_all[f"NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau={t}%)"]["hungarian_1to1_recall_pct"] for t in taus]
    ps_nf = [c_all[f"NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau={t}%)"]["precision_1to1_pct"] for t in taus]
    ax1.plot(rs_nf, ps_nf, color=C["nf"], lw=1.5, zorder=4)
    ax1.scatter(rs_nf, ps_nf, s=22, facecolor="white", edgecolor=C["nf"], linewidth=1.1, zorder=5)

    # Highlight tau=50% default operating point as star
    idx50 = taus.index(50)
    ax1.scatter([rs_nf[idx50]], [ps_nf[idx50]], marker="*", s=85, facecolor=C["nf"], edgecolor="#000000", linewidth=0.75, zorder=7)

    # Also plot NeedleFinder 2026 + IRGM + Poly3 (tau=50%, no SynthBrachy3T)
    # IRGM + Poly3 operating point: reproducible recomputation from the current cached bundles
    up = c_all["Unified NeedleFinder 2026 + IRGM + SyncTrack + Poly3 (tau=50%)"]
    ax1.scatter([up["hungarian_1to1_recall_pct"]], [up["precision_1to1_pct"]], marker="o", s=34,
                facecolor=C["nf"], edgecolor="#000000", linewidth=0.7, alpha=0.75, zorder=6)

    # Frozen non-overlapping label offsets for tau sweep endpoints
    tau_offsets = {
        25: ("τ=25%", (-26, -8)),
        60: ("τ=60%", (5, -2)),
    }
    for t, rx, py in zip(taus, rs_nf, ps_nf):
        if t in tau_offsets:
            txt, (dx, dy) = tau_offsets[t]
            ax1.annotate(txt, xy=(rx, py), xytext=(dx, dy), textcoords="offset points",
                         fontsize=5.8, color=C["nf"])

    ax1.annotate(f"NF 2026 + Synth + 1DTip ({c_all['NeedleFinder 2026 + SynthBrachy3T + Tip-Aware (tau=50%)']['f1_1to1_pct']:.1f}% F1)",
                 xy=(rs_nf[idx50], ps_nf[idx50]),
                 xytext=(-22, -4), textcoords="offset points", ha="right",
                 fontsize=5.8, fontweight="bold", color=C["nf"],
                 arrowprops=dict(arrowstyle="->", color=C["nf"], lw=0.6))

    # Ablation & Baseline points
    other_pts = [
        ("Stage 0/1A: Pure Physics 3D CV + HybridNeedleNet", "Physics + HybridNet", C["st0"], "s", (-10, -10)),
        ("Stage 1B: Multi-Head NeedleUNet3D + Tangent Tube Tracking", "NeedleUNet3D + tubes", C["st1"], "^", (-54, 4)),
        ("Stage 2A: Pure NeedleSliceMPNN (4-Layer Triplet GraphNet)", "Triplet MPNN only", C["st2"], "D", (-64, -6)),
        ("Baseline 1: PMB 2019 Isotropic 1mm³ 3D U-Net + DBSCAN/Skeleton", "PMB 2019 3D U-Net", C["b_pmb19"], "o", (-54, -7)),
    ]
    for key, lbl, col, mk, (dx, dy) in other_pts:
        rx = c_all[key]["hungarian_1to1_recall_pct"]
        py = c_all[key]["precision_1to1_pct"]
        fc = "white" if mk == "v" else col
        ax1.scatter([rx], [py], marker=mk, s=28, facecolor=fc, edgecolor=col, linewidth=1.0, zorder=5)
        ax1.annotate(lbl, xy=(rx, py), xytext=(dx, dy), textcoords="offset points",
                     fontsize=6.0, color=col, fontweight="bold" if "PMB" in lbl else "normal",
                     arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4) if abs(dx) > 25 else None)

    # Overflow annotation for low-precision mask-only baselines (clDice 24.5% Prec, Xiao 2026 39.2% Prec)
    _bx = c_all["Baseline 3: Xiao 2026 IRGM + 3D Hough + Sync Inward Tracking + Poly3"]; _bc = c_all["Baseline 2: Native-Resolution 3D ResUNet + clDice + Skeleton (Mask-Only)"]
    ax1.text(74.8, 60.0, f"▾ Mask-only trackers (<55% Prec.):\n   Xiao 2026: {_bx['hungarian_1to1_recall_pct']:.1f}% R, {_bx['precision_1to1_pct']:.1f}% P\n   clDice+Skel: {_bc['hungarian_1to1_recall_pct']:.1f}% R, {_bc['precision_1to1_pct']:.1f}% P",
             fontsize=5.5, color="#4D4D4D", bbox=dict(boxstyle="round,pad=0.25", fc="#F7F7F7", ec="#CCCCCC", lw=0.5))

    ax1.set_xlim(74, 96.5)
    ax1.set_ylim(57, 91.5)
    ax1.set_xticks([75, 80, 85, 90, 95])
    ax1.set_yticks([60, 70, 80, 90])
    ax1.set_xlabel("Strict 1-to-1 Hungarian recall (%)", fontsize=7.5)
    ax1.set_ylabel("Strict 1-to-1 precision (%)", fontsize=7.5)
    ax1.set_title("a   1-to-1 Precision–Recall Pareto frontier", fontsize=7.5, fontweight="bold", loc="left")
    ax1.grid(True, color="#E6E6E6", lw=0.4, zorder=0)

    # --- Panel b: Leak-Free Few-Shot Clinical Annotation Budget ---
    fs_syn = bench["synthbrachy3t_sim_to_real_few_shot_budget"]
    shots = [1, 5, 10, 20, 43]
    x_pos = np.log2(shots)

    f1_syn_all = [fs_syn[f"{k}_shot"]["cohort_all_1063_catheters"]["f1_1to1_pct"] for k in shots]
    f1_syn_long = [fs_syn[f"{k}_shot"]["cohort_long_759_cy_catheters"]["f1_1to1_obturator_adj_pct"] for k in shots]
    f1_zero_anal = fs_syn["0_shot_stage0_analytical"]["cohort_all_1063_catheters"]["f1_1to1_pct"]
    f1_zero_syn = fs_syn["0_shot"]["cohort_all_1063_catheters"]["f1_1to1_pct"]

    # Zero-shot floor lines
    ax2.axhline(f1_zero_anal, color=C["st0"], ls=":", lw=1.1, zorder=2)
    ax2.text(x_pos[-1] - 0.05, f1_zero_anal - 2.3, f"0-shot Stage 0 analytical ({f1_zero_anal:.1f}%)", fontsize=5.3, color=C["st0"], fontweight="bold", ha="right")

    ax2.axhline(f1_zero_syn, color=C["st2"], ls="--", lw=1.0, zorder=2)
    ax2.text(x_pos[-1] - 0.05, f1_zero_syn - 2.1, f"0-shot SynthBrachy3T ({f1_zero_syn:.1f}%)", fontsize=5.3, color=C["st2"], fontweight="bold", ha="right")

    # Curves + patient-level bootstrap 95% CI bands taken from the benchmark JSON
    def _ci(src, k):
        return src[f"{k}_shot"]["cohort_all_1063_catheters"]["bootstrap_95ci_1000_patients"]["f1_1to1_pct"]
    ci_syn = np.array([_ci(fs_syn, k) for k in shots])

    ax2.plot(x_pos, f1_syn_all, color=C["nf"], lw=1.6, marker="o", ms=4.2, zorder=5)
    ax2.fill_between(x_pos, ci_syn[:, 0], ci_syn[:, 1], color=C["nf"], alpha=0.18, zorder=3)
    ax2.plot(x_pos, f1_syn_long, color=C["st1"], lw=1.2, ls="-.", marker="s", ms=3.5, zorder=4)

    # Direct labels on curves
    ax2.text(x_pos[-1] - 0.12, f1_syn_long[-1] + 1.6, f"Long shafts F1_adj ({f1_syn_long[-1]:.1f}%)", fontsize=5.4, fontweight="bold", color=C["st1"], ha="right")
    ax2.text(x_pos[-1] - 0.05, f1_syn_all[-1] - 2.8, f"All 1,063 F1 ({f1_syn_all[-1]:.1f}%)", fontsize=5.4, fontweight="bold", color=C["nf"], ha="right")

    # Highlight k=1 shot
    ax2.annotate(f"1-shot: {f1_syn_all[0]:.1f}%",
                 xy=(x_pos[0], f1_syn_all[0]), xytext=(x_pos[0] + 0.05, 84.5),
                 fontsize=5.4, fontweight="bold", color=C["nf"],
                 arrowprops=dict(arrowstyle="->", color=C["nf"], lw=0.6))

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(k) for k in shots])
    ax2.set_ylim(68, 95)
    ax2.set_xlabel("Annotated clinical training patients per fold ($k$)", fontsize=7.5)
    ax2.set_ylabel("Out-of-fold F1 accuracy (%)", fontsize=7.5)
    ax2.set_title("b   Leak-free few-shot annotation efficiency", fontsize=7.5, fontweight="bold", loc="left")
    ax2.grid(True, color="#E6E6E6", lw=0.4, zorder=0)

    assert_no_overlap(fig, pad_px=1)
    save_all_formats(fig, "fig4_pareto_and_fewshot")
    plt.close(fig)


# =========================================================================
# FIGURE 5: Qualitative 3T T2W MRI Results (Dense Case + Severe Adhesion Case)
# =========================================================================
def make_fig5_qualitative_mri_3d(qual):
    fig, axes = plt.subplots(2, 4, figsize=(W2, 92 * MM), constrained_layout=True)

    rows_cfg = [
        ("dense", "a   Dense IC/IS implant (Case 017: 26 catheters + obturator)"),
        ("adhesion", "b   Severe catheter-adhesion implant (Case 031: 23 catheters, 5 adhering pairs ≤ 3.0 mm)"),
    ]

    for r_i, (prefix, row_title) in enumerate(rows_cfg):
        mri = qual[f"{prefix}_mri_slice"]
        sym = qual[f"{prefix}_sym_slice"]
        peak = qual[f"{prefix}_peak_slice"]
        mask = qual[f"{prefix}_mask_slice"]
        obt = qual[f"{prefix}_obt_slice"]
        gt_xy = qual[f"{prefix}_gt_xy_slice"]
        gt_ras = qual[f"{prefix}_gt_ras_obj"]
        sp = qual[f"{prefix}_spacing"]
        z_mid = int(qual[f"{prefix}_z_mid"])
        D, H, W = qual[f"{prefix}_shape"]

        y0, y1 = int(H * 0.08), int(H * 0.92)
        x0, x1 = int(W * 0.08), int(W * 0.92)
        mri_c = mri[y0:y1, x0:x1]
        sym_c = sym[y0:y1, x0:x1]
        peak_c = peak[y0:y1, x0:x1]
        mask_c = mask[y0:y1, x0:x1]
        obt_c = obt[y0:y1, x0:x1]
        gt_c = gt_xy - np.array([x0, y0])
        gt_c = gt_c[(gt_c[:, 0] >= 2) & (gt_c[:, 0] < (x1 - x0 - 2)) & (gt_c[:, 1] >= 2) & (gt_c[:, 1] < (y1 - y0 - 2))]

        vmin, vmax = np.percentile(mri_c, [2, 98])

        # Col 0: Axial T2W + GT (white circle) + Pred (vermillion x) + ROI box
        ax0 = axes[r_i, 0]
        ax0.imshow(mri_c, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        if len(gt_c) > 0:
            ax0.scatter(gt_c[:, 0], gt_c[:, 1], s=26, facecolors="none", edgecolors="white", linewidths=0.9, label="GT")
        pred_ras = qual.get(f"{prefix}_pred_ras_obj")
        if pred_ras is not None:
            pr_c = project_polylines_to_slice(pred_ras, qual[f"{prefix}_ijk_to_ras"], qual[f"{prefix}_crop_origin"], z_mid) - np.array([x0, y0])
            pr_c = pr_c[(pr_c[:, 0] >= 2) & (pr_c[:, 0] < (x1 - x0 - 2)) & (pr_c[:, 1] >= 2) & (pr_c[:, 1] < (y1 - y0 - 2))]
            if len(pr_c) > 0:
                ax0.scatter(pr_c[:, 0], pr_c[:, 1], s=16, marker="x", color=C["nf"], linewidths=0.9, label="Pred")
        px_10mm = 10.0 / float(sp[0])
        ax0.plot([8, 8 + px_10mm], [mri_c.shape[0] - 8, mri_c.shape[0] - 8], color="white", lw=1.5)
        ax0.text(8 + px_10mm / 2, mri_c.shape[0] - 13, "10 mm", color="white", fontsize=5.8, ha="center")
        ax0.text(6, 12, f"z = {z_mid}/{D}", color="white", fontsize=6.0, fontweight="bold")
        ax0.set_title("a   Case 017 · T2W + OOF" if r_i == 0 else "b   Case 031 · Adhesion", fontsize=6.6, fontweight="bold", loc="left")

        # Col 1: Stage 0 3D Radial Symmetry Response
        ax1 = axes[r_i, 1]
        ax1.imshow(mri_c, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", alpha=0.8)
        sym_ma = np.ma.masked_where(sym_c < 0.16, sym_c)
        ax1.imshow(sym_ma, cmap="YlOrBr", vmin=0.1, vmax=0.95, origin="upper", alpha=0.88)
        ax1.set_title("Stage 0 · Symmetry $S_0$", fontsize=6.6, fontweight="bold", color=C["st0"])

        # Col 2: Stage 1 Centerline Probability + IRGM Gate Contour
        ax2 = axes[r_i, 2]
        ax2.imshow(mri_c, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", alpha=0.6)
        pk_ma = np.ma.masked_where(peak_c < 0.10, peak_c)
        ax2.imshow(pk_ma, cmap="Blues", vmin=0.08, vmax=0.90, origin="upper", alpha=0.9)
        ax2.contour(mask_c, levels=[0.35], colors=[C["st1"]], linewidths=0.65)
        if np.any(obt_c > 0.3):
            ax2.contour(obt_c, levels=[0.4], colors=[C["st0"]], linewidths=0.85, linestyles="--")
        ax2.set_title("Stage 1 · Peak + IRGM", fontsize=6.6, fontweight="bold", color=C["st1"])

        # Col 3: 3D Catheter Bundle colored by per-catheter E_shaft (cividis 0-1.5 mm)
        ax3 = axes[r_i, 3]
        ax3.set_facecolor("#FAF9F7")
        az, el = np.deg2rad(35.0), np.deg2rad(20.0)
        for g in gt_ras:
            x_p, y_p = ortho_project(g)
            ax3.plot(x_p, y_p, color="#BDBDBD", lw=2.0, alpha=0.6, solid_capstyle="round")
        n_pred = 0
        if pred_ras is not None:
            n_pred = len(pred_ras)
            for g in pred_ras:
                x_p, y_p = ortho_project(g)
                ax3.plot(x_p, y_p, color=C["nf"], lw=1.1, alpha=0.95, solid_capstyle="round")
                ax3.scatter([x_p[-1]], [y_p[-1]], s=8, color=C["nf"], zorder=5)
        ax3.set_title(f"3D bundle ({n_pred} pred. / {len(gt_ras)} ref.)", fontsize=6.6, fontweight="bold", color=C["nf"])

        for c_i in range(4):
            axes[r_i, c_i].set_xticks([])
            axes[r_i, c_i].set_yticks([])

    assert_no_overlap(fig, pad_px=1)
    save_all_formats(fig, "fig5_qualitative_mri_3d")
    plt.close(fig)


# =========================================================================
# FIGURE 6: Zero-Needle Specificity (MSD Task05) & Runtime (170 x 68 mm)
# =========================================================================
def make_fig6_specificity_and_runtime(spec):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(W2, 68 * MM), gridspec_kw={"width_ratios": [1.15, 1.0, 0.95]}, constrained_layout=True)

    # --- Panel a: Cumulative Zero-Needle Specificity ECDF on 48 MSD Task05 Volumes ---
    cases = spec["per_case_results"]
    fp_st1 = np.sort([c["raw_unet_tubes_no_physics_or_gt"] for c in cases])
    fp_nf50 = np.sort([c["unified_needlefinder_2026_by_tau"]["50"] for c in cases])
    fp_nfgate = np.sort([c["unified_needlefinder_2026_sym_gated_by_tau"]["50"] for c in cases])
    fp_ssl = np.sort([c["clean_ssl_stage1_sym_tubes_only"] for c in cases])

    xs_eval = np.arange(0, 51)
    for arr, col, ls, lw, lbl in [
        (fp_st1, C["st1"], "--", 1.1, "Raw Stage 1 tubes (6.38/vol)"),
        (fp_nf50, "#999999", ":", 1.2, "Ungated NF 2026 (4.10/vol)"),
        (fp_nfgate, C["nf"], "-", 1.3, "NF 2026 + S0 Gate (0.33/vol)"),
        (fp_ssl, C["st0"], "-", 1.6, "SSL Stage 1 + S0 Gate (0.125/vol)"),
    ]:
        ecdf = [100.0 * np.mean(arr <= x) for x in xs_eval]
        ax1.step(xs_eval, ecdf, where="post", color=col, ls=ls, lw=lw, label=lbl)

    ax1.scatter([0, 1], [87.50, 100.0], s=22, color=C["st0"], edgecolor="#000000", linewidth=0.6, zorder=6)
    ax1.annotate("87.5% (0 FP)", xy=(0, 87.50), xytext=(2.8, 60),
                 fontsize=5.6, fontweight="bold", color=C["st0"],
                 arrowprops=dict(arrowstyle="->", color=C["st0"], lw=0.6))
    ax1.annotate("100% (≤1 FP)", xy=(1, 100.0), xytext=(5.2, 91),
                 fontsize=5.6, fontweight="bold", color=C["st0"],
                 arrowprops=dict(arrowstyle="->", color=C["st0"], lw=0.6))

    ax1.set_xscale("symlog", linthresh=1.0)
    ax1.set_xticks([0, 1, 2, 5, 10, 20, 50])
    ax1.set_xticklabels(["0", "1", "2", "5", "10", "20", "50"])
    ax1.set_xlim(-0.15, 50)
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("False-positive tracks / volume", fontsize=7.0)
    ax1.set_ylabel("Volumes with ≤ x FP (%)", fontsize=7.0)
    ax1.set_title("a   0-needle specificity (48 vols)", fontsize=7.0, fontweight="bold", loc="left")
    ax1.legend(loc="lower right", fontsize=5.0, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    ax1.grid(True, color="#E6E6E6", lw=0.4, zorder=0)

    # --- Panel b: End-to-End Runtime vs Literature (Log10 scale) ---
    rt_methods = [
        ("Zaffino 2019 (PMB, ≥2 min)ʳ", 120.0, C["b_pmb19"]),
        ("Shaaer 2022 (assisted)ʳ", 660.0, "#7F7F7F"),
        ("PMB 2019 (re-impl.)", 14.8, C["b_pmb19"]),
        ("Xiao 2026 (Sync)ʳ", 11.6, C["b_xiao"]),
        ("NF 2026 (CPU)", 4.15, C["nf"]),
        ("NF 2026 (GPU)", 2.67, C["nf"]),
    ]
    y_rt = np.arange(len(rt_methods))
    for idx_r, (m_lbl, val_s, col) in enumerate(rt_methods):
        y_p = len(rt_methods) - 1 - idx_r
        hatch = "///" if "ʳ" in m_lbl else None
        ax2.barh(y_p, val_s, height=0.58, color=col, alpha=0.85 if hatch is None else 0.55, edgecolor="#222222", lw=0.5, hatch=hatch)
        ax2.text(val_s * 1.15, y_p, f"{val_s:.2f} s" if val_s < 10 else f"{val_s:.1f} s",
                 va="center", ha="left", fontsize=5.9, fontweight="bold" if "NF" in m_lbl else "normal")

    ax2.set_xscale("log")
    ax2.set_xlim(0.1, 3000)
    ax2.set_xticks([0.1, 1, 10, 100, 1000])
    ax2.set_yticks(range(len(rt_methods)))
    ax2.set_yticklabels([m[0] for m in reversed(rt_methods)], fontsize=6.0)
    ax2.set_xlabel("Runtime / patient (s, log₁₀)", fontsize=7.0)
    ax2.set_title("b   Runtime vs. literature", fontsize=7.0, fontweight="bold", loc="left")
    ax2.grid(axis="x", color="#E6E6E6", lw=0.4, zorder=0)

    # --- Panel c: Linear Stage Breakdown (CPU vs GPU) ---
    stages = [
        ("Stage 0: Symmetry", 0.267, 0.267, C["st0"]),
        ("Stage 1: ResUNet+IRGM", 2.575, 1.330, C["st1"]),
        ("Stage 2A: Track+MPNN", 1.248, 1.046, C["st2"]),
        ("Stage 2B/C: GraphTF+Tip", 0.061, 0.032, C["nf"]),
    ]
    left_c, left_g = 0.0, 0.0
    for st_lbl, tc, tg, col in stages:
        ax3.barh(1, tc, left=left_c, height=0.46, color=col, edgecolor="white", lw=0.6, label=st_lbl)
        ax3.barh(0, tg, left=left_g, height=0.46, color=col, edgecolor="white", lw=0.6)
        left_c += tc
        left_g += tg

    ax3.text(left_c + 0.08, 1, "4.15 s", va="center", ha="left", fontsize=6.2, fontweight="bold")
    ax3.text(left_g + 0.08, 0, "2.67 s", va="center", ha="left", fontsize=6.2, fontweight="bold")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["GPU\n(4090)", "CPU\n(24-core)"], fontsize=6.2, fontweight="bold")
    ax3.set_xlim(0, 5.45)
    ax3.set_ylim(-0.6, 1.95)
    ax3.set_xlabel("Stage runtime (s, linear)", fontsize=7.0)
    ax3.set_title("c   Stage breakdown", fontsize=7.0, fontweight="bold", loc="left")
    ax3.legend(loc="upper right", fontsize=5.1, frameon=False)
    ax3.grid(axis="x", color="#E6E6E6", lw=0.4, zorder=0)

    assert_no_overlap(fig, pad_px=1)
    save_all_formats(fig, "fig6_specificity_and_runtime")
    plt.close(fig)


# =========================================================================
# FIGURE 7: Stage 2B confidence calibration + reason-coded decisions (170 x 62 mm)
# =========================================================================
def make_fig7_calibration_and_reasons():
    with open(os.path.join(SOURCES_DIR, "2026-10-23-stage2b-calibration-and-reason-codes.json")) as f:
        d = json.load(f)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 62 * MM), gridspec_kw={"width_ratios": [1.0, 1.15]}, constrained_layout=True)

    # --- Panel a: reliability diagram (all candidates), raw vs fold-wise temperature scaling ---
    for key, col, lbl, mk in [("calibration_all_candidates", C["nf"], "raw confidence", "o"),
                              ("temperature_scaling_fold_wise", C["st2"], "temperature-scaled (fold-wise)", "s")]:
        bins = [b for b in d[key]["bins"] if b.get("n", 0) > 0]
        ax1.plot([b["mean_conf"] for b in bins], [b["frac_true"] for b in bins], color=col, lw=1.3, marker=mk, ms=3.8, label=f"{lbl} (ECE {d[key]['ece']:.3f})", zorder=4)
    ax1.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls="--", zorder=1)
    bins = [b for b in d["calibration_all_candidates"]["bins"] if b.get("n", 0) > 0]
    ax1b = ax1.twinx()
    ax1b.bar([0.5 * (b["lo"] + b["hi"]) for b in bins], [b["n"] for b in bins], width=0.08, color="#DDDDDD", zorder=0)
    ax1b.set_ylabel("candidates per bin", fontsize=6.5, color="#777777"); ax1b.tick_params(labelsize=6, colors="#777777")
    ax1.set_zorder(ax1b.get_zorder() + 1); ax1.patch.set_visible(False)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Stage 2B confidence $p_m$", fontsize=7.5); ax1.set_ylabel("fraction of real catheters", fontsize=7.5)
    ax1.set_title("a   Reliability of the track confidence", fontsize=7.2, fontweight="bold", loc="left")
    ax1.legend(loc="upper left", fontsize=5.6, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    ax1.grid(True, color="#E6E6E6", lw=0.4, zorder=0)

    # --- Panel b: reason-code audit (what each decision rule removes, and at what cost) ---
    order = ["kept", "below_threshold", "duplicate_of_kept_track", "outside_implant_region", "low_dark_core_symmetry"]
    names = ["kept", "below\nthreshold", "duplicate of\nkept track", "outside\nimplant region", "low dark-core\nsymmetry"]
    a = d["reason_code_audit"]
    ypos = np.arange(len(order))[::-1]
    false_n = np.array([a[k]["n_false"] for k in order]); red_n = np.array([a[k]["n_real_redundant"] for k in order])
    lost_n = np.array([a[k]["n_real_lost"] for k in order]); kept_real = np.array([a[k]["n_real"] if k == "kept" else 0 for k in order])
    ax2.barh(ypos, kept_real, color=C["nf"], height=0.6, label="real catheter, kept")
    ax2.barh(ypos, false_n, left=kept_real, color="#BDBDBD", height=0.6, label="false track")
    ax2.barh(ypos, red_n, left=kept_real + false_n, color=C["st2"], height=0.6, label="real, but already covered by a kept track")
    ax2.barh(ypos, lost_n, left=kept_real + false_n + red_n, color="#B03F41", height=0.6, label="real catheter lost")
    for yy, k in zip(ypos, order):
        ax2.text(a[k]["n"] + 8, yy, f"n = {a[k]['n']}", va="center", fontsize=6.0, color="#333333")
    ax2.set_yticks(ypos); ax2.set_yticklabels(names, fontsize=6.3)
    ax2.set_xlim(0, max(a[k]["n"] for k in order) * 1.22)
    ax2.set_xlabel("candidate tracks (54 patients)", fontsize=7.5)
    ax2.set_title("b   Reason-coded decisions at $\\tau = 50\\%$", fontsize=7.2, fontweight="bold", loc="left")
    ax2.legend(loc="lower right", fontsize=5.4, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    ax2.grid(axis="x", color="#E6E6E6", lw=0.4, zorder=0)

    assert_no_overlap(fig, pad_px=1)
    save_all_formats(fig, "fig7_calibration_and_reasons")
    plt.close(fig)


def make_fig8_clinical_translation_and_generalization():
    with open(os.path.join(SOURCES_DIR, "2026-10-23-exp1-tg43-dosimetry.json")) as f:
        e1 = json.load(f)
    with open(os.path.join(SOURCES_DIR, "2026-10-23-exp2-uncertainty-conformal-grubbs.json")) as f:
        e2 = json.load(f)
    with open(os.path.join(SOURCES_DIR, "2026-10-23-exp3-prostate-and-cross-sequence.json")) as f:
        e3 = json.load(f)
    with open(os.path.join(SOURCES_DIR, "2026-10-23-exp4-4d-insertion-and-ct.json")) as f:
        e4 = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(W2, 118 * MM), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- Panel a: AAPM TG-43U1 HDR 192Ir Dosimetric Impact ---
    c54 = e1["cohort_54_patients_summary"]
    c40 = e1["cohort_40_dual_traced_patients_summary"]
    methods_a = [
        ("PMB 2019 U-Net\n(N=54)", c54["PMB_2019_3D_UNet_Walk"], C["b_pmb"]),
        ("Stage 0 Physics\n(N=54)", c54["Stage_0_Physics_Tracker"], C["st0"]),
        ("Human Rater 2\nLB/AMe (N=40)", c40["Human_Rater_2_LB_AMe"], "#7F7F7F"),
        ("NF 2026 (4b)\n(N=54)", c54["NeedleFinder_2026_Proposed_4b"], C["nf"]),
        ("NF 2026 + Top-2\nTip Review (N=54)", c54["NeedleFinder_2026_Top2_Uncertainty_Review"], C["st2"]),
    ]
    y_a = np.arange(len(methods_a))[::-1]
    prot_a_d90 = [m[1]["protocol_A_fixed_plan_sensitivity"]["abs_delta_d90_pct"]["mean"] for m in methods_a]
    prot_b_d90 = [m[1]["protocol_B_optimize_on_pred_deliver_on_ref"]["abs_delta_d90_pct"]["mean"] for m in methods_a]
    g22_a = [m[1]["protocol_A_fixed_plan_sensitivity"]["gamma_2pct_2mm_global_pass_pct"]["mean"] for m in methods_a]

    h_a = 0.34
    ax_a.barh(y_a + h_a / 2, prot_a_d90, height=h_a, color="#90A4AE", label="Protocol A fixed-plan $|\\Delta D_{90}|$ (%)")
    ax_a.barh(y_a - h_a / 2, prot_b_d90, height=h_a, color=C["nf"], label="Protocol B re-optimized $|\\Delta D_{90}|$ (%)")
    for yy, pa_val, pb_val, g_val in zip(y_a, prot_a_d90, prot_b_d90, g22_a):
        mx = max(pa_val, pb_val)
        ax_a.text(mx + 0.8, yy, f"{pa_val:.1f}% / {pb_val:.1f}% ($\\gamma_{{2/2}}$ {g_val:.0f}%)", va="center", fontsize=5.5, color="#222222")
    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels([m[0] for m in methods_a], fontsize=5.8)
    ax_a.set_xlim(0, 64)
    ax_a.set_ylim(-1.15, len(methods_a) - 0.35)
    ax_a.set_xlabel("Target $|\\Delta D_{90}|$ mean error (% of 7 Gy Rx)", fontsize=6.8)
    ax_a.set_title("a   AAPM TG-43U1 $^{192}$Ir dosimetric impact", fontsize=7.2, fontweight="bold", loc="left")
    ax_a.legend(loc="lower right", ncol=2, fontsize=5.0, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    ax_a.grid(axis="x", color="#E6E6E6", lw=0.4, zorder=0)

    # --- Panel b: Uncertainty-Guided Clinician Review-Budget Curve ---
    q1 = e2["part2c_clinician_in_the_loop_review_simulation"]["queue1_tip_verification_and_relocation"]
    ks = [0, 1, 2, 3, 5, 8, 10]
    k_keys = [f"k_{k}" for k in ks]
    unc_p95 = [q1["uncertainty_guided_sigma_tip"][kk]["p95_mm"] for kk in k_keys]
    unc_mean = [q1["uncertainty_guided_sigma_tip"][kk]["mean_mm"] for kk in k_keys]
    rnd_p95 = [q1["random_order"][kk]["p95_mm"] for kk in k_keys]
    ora_p95 = [q1["oracle_largest_error_first"][kk]["p95_mm"] for kk in k_keys]

    ax_b.plot(ks, rnd_p95, color="#888888", lw=1.1, ls="--", marker="^", ms=3.4, label="Random review P95 (mm)")
    ax_b.plot(ks, unc_p95, color=C["nf"], lw=1.5, marker="o", ms=4.0, label="Uncertainty $\\hat{\\sigma}_{\\mathrm{tip}}$ P95 (90.2% conf. cov)")
    ax_b.plot(ks, ora_p95, color=C["st2"], lw=1.1, ls=":", marker="d", ms=3.4, label="Oracle P95 (mm)")
    ax_b.plot(ks, unc_mean, color=C["st1"], lw=1.4, marker="s", ms=3.6, label="Uncertainty-guided mean $E_{\\mathrm{tip}}$ (mm)")
    ax_b.axhline(10.0, color="#B03F41", lw=0.7, ls="-.", zorder=1)
    ax_b.annotate("k=8: P95 9.9 mm, mean 2.8 mm", xy=(8, 9.9), xytext=(3.8, 12.6),
                  fontsize=5.4, color="#222222", arrowprops=dict(arrowstyle="->", lw=0.5, color="#555555"))
    ax_b.set_xlim(-0.3, 10.4)
    ax_b.set_ylim(0, 31)
    ax_b.set_xticks(ks)
    ax_b.set_xlabel("Flagged tips reviewed per patient $k$", fontsize=6.8)
    ax_b.set_ylabel("3D tip error $E_{\\mathrm{tip}}$ (mm)", fontsize=6.8)
    ax_b.set_title("b   Uncertainty-guided tip triage budget", fontsize=7.2, fontweight="bold", loc="left")
    ax_b.legend(loc="upper right", fontsize=5.2, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    ax_b.grid(True, color="#E6E6E6", lw=0.4, zorder=0)

    # --- Panel c: Cross-Organ, Multi-Sequence & Cross-Modality Generalization ---
    p3a = e3["part_3a_prostate_zero_shot_transfer"]
    p3b = e3["part_3b_same_session_cross_sequence"]["cross_sequence_benchmarks"]
    p4b = e4["part4b_paired_pelvic_ct_cross_modality"]

    items_c = [
        ("Prostate MRI: NF 2026\n(template ACSV, N=14)", p3a["operating_modes"]["standard_template_acsv_klim47"]["nf2026_full_stage2_zero_exposure_gyn_tau50"]["f1_1to1_pct"], C["nf"], "100.0% F1 (0.34 mm shaft)"),
        ("Prostate MRI: NF 2026\n(auto k_lim, 0% exposure)", p3a["operating_modes"]["reference_free_auto_klim40"]["nf2026_full_stage2_zero_exposure_gyn_tau50"]["f1_1to1_pct"], C["nf"], "96.3% F1 (0.99 mm med tip)"),
        ("Prostate MRI: 1-ch\nResUNet + clDice", p3a["operating_modes"]["reference_free_auto_klim40"]["baseline2_resunet_cldice_1ch"]["f1_1to1_pct"], C["b_nnu"], "71.8% F1 (11 false tracks)"),
        ("Prostate MRI: NF 2026\nPhysics X[1:4] = 0", p3a["physics_channel_ablations"]["reference_free_auto_klim40"]["stage1_physics_zeroed_X1to3_eq_0"]["f1_1to1_pct"], "#B03F41", "0.0% F1 (985x peak drop)"),
        ("3D wbSSFP / True-FISP\n(N=47 catheters)", p3b["AX_3D_wbSSFP_FISP"]["vs_dark_core_present_gt"]["nf2026_z_upsampled_1p6mm_full_stage2"]["f1_1to1_pct"], C["st2"], "92.9% F1 (97.9% recall)"),
        ("Thick 2D AX TSE T2\n(49 scans, N=882)", p3b["2D_AX_TSE_T2"]["vs_dark_core_present_gt"]["nf2026_z_upsampled_1p6mm_full_stage2"]["recall_1to1_pct"], C["st1"], "88.4% rec (76.9% F1)"),
        ("Paired Pelvic CT\n(38 pts, N=801, 0-shot)", p4b["cross_modality_ct_to_mri_registration_metrics"]["dual_polarity_matched_le_5p0mm_recall_pct"], C["st0"], "79.2% rec (52% extra MRI=CT)"),
    ]
    y_c = np.arange(len(items_c))[::-1]
    vals_c = [it[1] for it in items_c]
    cols_c = [it[2] for it in items_c]
    ax_c.barh(y_c, vals_c, height=0.58, color=cols_c)
    for yy, val, it in zip(y_c, vals_c, items_c):
        ax_c.text(val + 1.5, yy, it[3], va="center", fontsize=5.4, color="#222222")
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels([it[0] for it in items_c], fontsize=5.5)
    ax_c.set_xlim(0, 148)
    ax_c.set_xlabel("Detection F1 or recall (%)", fontsize=7.0)
    ax_c.set_title("c   Cross-organ, multi-sequence & CT transfer", fontsize=7.2, fontweight="bold", loc="left")
    ax_c.grid(axis="x", color="#E6E6E6", lw=0.4, zorder=0)

    # --- Panel d: 4D Intra-Procedural Insertion & Biomechanics ---
    bend = e4["part4a_4d_intraprocedural_tracking"]["depth_dependent_bending_biomechanics"]["deflection_by_depth_bin"]
    push = e4["part4a_4d_intraprocedural_tracking"]["tissue_push_biomechanics"]
    depth_bins = ["0_30mm", "30_60mm", "60_90mm", "gt_90mm"]
    depth_x = np.arange(len(depth_bins))
    bend_med = [bend[b]["median_deflection_mm"] for b in depth_bins]
    bend_mean = [bend[b]["mean_deflection_mm"] for b in depth_bins]
    bend_p90 = [bend[b]["p90_deflection_mm"] for b in depth_bins]

    ax_d.plot(depth_x, bend_p90, color="#B03F41", lw=1.2, ls="--", marker="^", ms=3.5, label="Shaft deflection P90 (mm)")
    ax_d.plot(depth_x, bend_mean, color=C["nf"], lw=1.5, marker="o", ms=4.0, label="Shaft deflection mean (mm)")
    ax_d.plot(depth_x, bend_med, color=C["st2"], lw=1.4, marker="s", ms=3.6, label="Shaft deflection median (mm)")

    push_x = np.array([0.3, 1.5, 2.7])
    push_med = [
        push["near_newly_inserted_le_10mm"]["median_lateral_shift_mm"],
        push["intermediate_10_to_20mm"]["median_lateral_shift_mm"],
        push["far_from_newly_inserted_gt_20mm"]["median_lateral_shift_mm"],
    ]
    ax_d.plot(push_x, push_med, color=C["st0"], lw=1.3, ls="-.", marker="D", ms=3.6,
              label="Tissue-push shift ($\\leq$10 vs $>$20 mm: 1.41 vs 0.93 mm)")
    ax_d.set_xticks(depth_x)
    ax_d.set_xticklabels(["0-30 mm\n(near <=10)", "30-60 mm\n(10-20 mm)", "60-90 mm\n(far >20)", ">90 mm\ndeep tip"], fontsize=5.6)
    ax_d.set_xlim(-0.3, 3.3)
    ax_d.set_ylim(0, 10.5)
    ax_d.set_xlabel("Insertion depth / neighbor distance bin (4D Kalman +8.2% F1)", fontsize=6.6)
    ax_d.set_ylabel("Displacement / deflection (mm)", fontsize=7.0)
    ax_d.set_title("d   4D insertion deflection & tissue-push biomechanics", fontsize=7.2, fontweight="bold", loc="left")
    ax_d.legend(loc="upper left", fontsize=5.1, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    ax_d.grid(True, color="#E6E6E6", lw=0.4, zorder=0)

    assert_no_overlap(fig, pad_px=1)
    save_all_formats(fig, "fig8_clinical_translation_and_generalization")
    plt.close(fig)


def main():
    bench, inter, spec, qual = load_all_sources()
    export_csv_data(bench, inter, spec)
    make_fig1_pipeline_overview(qual)
    make_fig2_architecture_modules()
    make_fig3_raincloud_distributions(bench, inter)
    make_fig4_pareto_and_fewshot(bench)
    make_fig5_qualitative_mri_3d(qual)
    make_fig6_specificity_and_runtime(spec)
    make_fig7_calibration_and_reasons()
    make_fig8_clinical_translation_and_generalization()
    print("All 8 publication vector PDF figures + CSV data files generated successfully!")


if __name__ == "__main__":
    main()
