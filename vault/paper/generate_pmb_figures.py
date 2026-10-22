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
    with open(os.path.join(SOURCES_DIR, "2026-10-22-unified-apples-to-apples-benchmark.json")) as f:
        bench = json.load(f)
    with open(os.path.join(SOURCES_DIR, "2026-10-22-interobserver-543-benchmark.json")) as f:
        inter = json.load(f)
    with open(os.path.join(SOURCES_DIR, "2026-10-22-public-pelvic-mri-specificity-benchmark.json")) as f:
        spec = json.load(f)
    qual = dict(np.load(os.path.join(DATA_DIR, "qualitative_cases.npz"), allow_pickle=True))
    pred_path = os.path.join(DATA_DIR, "qualitative_predictions.npz")
    if os.path.exists(pred_path):
        qual.update(dict(np.load(pred_path, allow_pickle=True)))
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
        w.writerow(["shots", "voxel_dice", "f1_all_1063", "f1_long_adj_759", "shaft_mean_mm", "tip_mean_mm"])
        fs = bench["few_shot_clinical_annotation_budget"]
        for k_str in ["0", "1", "5", "10", "20", "43"]:
            r = fs[f"{k_str}_shot"]
            w.writerow([
                k_str,
                f"{r['mean_oof_voxel_dice']:.4f}",
                f"{r['cohort_all_1063_catheters']['f1_1to1_pct']:.2f}",
                f"{r['cohort_long_759_cy_catheters']['f1_1to1_obturator_adj_pct']:.2f}",
                f"{r['cohort_all_1063_catheters']['metrics']['mean_shaft_xy']['mean']:.3f}",
                f"{r['cohort_all_1063_catheters']['metrics']['tip_3d']['mean']:.3f}",
            ])

    # 3. msd_specificity.csv
    with open(os.path.join(DATA_DIR, "msd_specificity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "unpruned_pool", "stage1_tubes_only", "nf2026_tau60", "nf2026_stage0_sym_gate"])
        for c in spec["per_case_results"]:
            w.writerow([
                c["case_id"],
                c["candidate_pool_before_graph_transformer"],
                c["raw_unet_tubes_no_physics_or_gt"],
                c["unified_needlefinder_2026_by_tau"]["60"],
                c["stage1_unet_sym_tubes_only"],
            ])

    # 4. runtime.csv
    with open(os.path.join(DATA_DIR, "runtime.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "cpu_sec", "gpu_sec", "cpu_std_res_sec", "cpu_high_res_sec"])
        w.writerow(["Stage 0: Rust 3D Radial Symmetry", 0.410, 0.410, 0.120, 0.590])
        w.writerow(["Stage 1: NeedleUNet3D + IRGM", 0.960, 0.220, 0.550, 1.020])
        w.writerow(["Stage 2A: Native Seed Tracking + Triplet MPNN", 1.220, 0.540, 0.450, 1.670])
        w.writerow(["Stage 2B: Bundle Graph Transformer + Poly3", 0.150, 0.080, 0.080, 0.190])
        w.writerow(["Total End-to-End", 2.740, 1.250, 1.200, 3.470])


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

    # Bottom row: Proportional runtime bar (CPU 2.74 s / GPU 1.25 s)
    ax_bar = fig.add_axes([0.14, 0.06, 0.83, 0.14])
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)

    segs = [
        ("Stage 0: Rust symmetry", 0.410, 0.410, C["st0"]),
        ("Stage 1: NeedleUNet3D + IRGM", 0.960, 0.220, C["st1"]),
        ("Stage 2A: Native tracking + MPNN", 1.220, 0.540, C["st2"]),
        ("Stage 2B: GraphTF + Poly3", 0.150, 0.080, C["nf"]),
    ]
    y_pos = [1.0, 0.0]
    left_cpu, left_gpu = 0.0, 0.0
    for label, t_cpu, t_gpu, col in segs:
        ax_bar.barh(1.0, t_cpu, left=left_cpu, height=0.52, color=col, edgecolor="white", linewidth=0.6)
        ax_bar.barh(0.0, t_gpu, left=left_gpu, height=0.52, color=col, edgecolor="white", linewidth=0.6)
        left_cpu += t_cpu
        left_gpu += t_gpu

    ax_bar.text(left_cpu + 0.05, 1.0, f"{left_cpu:.2f} s total (8-core CPU; 1.20 s std / 3.47 s hi-res)", va="center", ha="left", fontsize=6.5, fontweight="bold", color="#222222")
    ax_bar.text(left_gpu + 0.05, 0.0, f"{left_gpu:.2f} s total (RTX 4090 GPU)", va="center", ha="left", fontsize=6.5, fontweight="bold", color="#222222")
    ax_bar.set_yticks([0.0, 1.0])
    ax_bar.set_yticklabels(["GPU (1.25 s)", "CPU (2.74 s)"], fontsize=7, fontweight="bold")
    ax_bar.set_xlim(0, 4.30)
    ax_bar.set_xlabel("End-to-end inference time per patient volume (s)", fontsize=7.5)
    ax_bar.tick_params(axis="y", length=0)

    # Legend swatches above runtime bar
    leg_items = [
        (0.12, "■ Stage 0: Symmetry (0.41 s)", C["st0"]),
        (0.35, "■ Stage 1: ResUNet+IRGM (0.96 s)", C["st1"]),
        (0.61, "■ Stage 2A: MPNN (1.22 s)", C["st2"]),
        (0.81, "■ Stage 2B: GraphTF (0.15 s)", C["nf"]),
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
    # Human reference statistics from the single-loader Table 3 export (same scenes, matching rule and metrics as the NF rows)
    with open(os.path.join(SOURCES_DIR, "2026-10-22-table3-observers-and-nf.json")) as f:
        t3 = json.load(f)["rows"]["CY_vs_LBAMe_ref_3.5mm"]
    h_sh = t3["shaft_mm"]
    h_tip = t3["tip_3d_mm"]

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
    with open(os.path.join(SOURCES_DIR, "2026-10-22-irgm-poly3-rows-recheck.json")) as f:
        _rc = json.load(f)["rows"]["irgm_whole_cohort"]
    up = {"hungarian_1to1_recall_pct": _rc["rec"], "precision_1to1_pct": _rc["prec"], "f1_1to1_pct": _rc["f1"]}
    ax1.scatter([up["hungarian_1to1_recall_pct"]], [up["precision_1to1_pct"]], marker="o", s=34,
                facecolor=C["nf"], edgecolor="#000000", linewidth=0.7, alpha=0.75, zorder=6)

    # Frozen non-overlapping label offsets for tau sweep endpoints
    tau_offsets = {
        25: ("τ=25%", (-28, 3)),
        60: ("τ=60%", (-24, 3)),
    }
    for t, rx, py in zip(taus, rs_nf, ps_nf):
        if t in tau_offsets:
            txt, (dx, dy) = tau_offsets[t]
            ax1.annotate(txt, xy=(rx, py), xytext=(dx, dy), textcoords="offset points",
                         fontsize=5.8, color=C["nf"])

    ax1.annotate("NF 2026 + SynthBrachy3T (88.2% F1)",
                 xy=(rs_nf[idx50], ps_nf[idx50]),
                 xytext=(-18, -3), textcoords="offset points", ha="right",
                 fontsize=5.8, fontweight="bold", color=C["nf"],
                 arrowprops=dict(arrowstyle="->", color=C["nf"], lw=0.6))

    ax1.annotate(f"NF 2026 + IRGM ({up['f1_1to1_pct']:.1f}% F1)",
                 xy=(up["hungarian_1to1_recall_pct"], up["precision_1to1_pct"]),
                 xytext=(-18, -3), textcoords="offset points", ha="right",
                 fontsize=5.6, color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#777777", lw=0.5))

    # Ablation & Baseline points
    other_pts = [
        ("Stage 0/1A: Pure Physics 3D CV + HybridNeedleNet", "Physics + HybridNet", C["st0"], "s", (-10, -10)),
        ("Stage 1B: Multi-Head NeedleUNet3D + Tangent Tube Tracking", "NeedleUNet3D + tubes", C["st1"], "^", (-48, 5)),
        ("Stage 1B: Hybrid AxialTransUNet3D + Tangent Tube Tracking", "AxialTransUNet3D", C["st1"], "v", (-48, -9)),
        ("Stage 2A: Pure NeedleSliceMPNN (4-Layer Triplet GraphNet)", "Triplet MPNN only", C["st2"], "D", (-54, 1)),
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

    # Overflow annotation for low-precision mask-only baselines (clDice 23.9% Prec, Xiao 2026 38.9% Prec)
    ax1.text(74.8, 69.0, "▾ Mask-only trackers (<55% Prec.):\n   Xiao 2026: 86.9% R, 38.9% P\n   clDice+Skel: 90.4% R, 23.9% P",
             fontsize=5.5, color="#4D4D4D", bbox=dict(boxstyle="round,pad=0.25", fc="#F7F7F7", ec="#CCCCCC", lw=0.5))

    ax1.set_xlim(74, 96.5)
    ax1.set_ylim(57, 91.5)
    ax1.set_xticks([75, 80, 85, 90, 95])
    ax1.set_yticks([60, 70, 80, 90])
    ax1.set_xlabel("Strict 1-to-1 Hungarian recall (%)", fontsize=7.5)
    ax1.set_ylabel("Strict 1-to-1 precision (%)", fontsize=7.5)
    ax1.set_title("a   1-to-1 Precision–Recall Pareto frontier", fontsize=7.5, fontweight="bold", loc="left")
    ax1.grid(True, color="#E6E6E6", lw=0.4, zorder=0)

    # --- Panel b: Few-Shot Clinical Annotation Budget (with vs without SynthBrachy3T) ---
    fs = bench["few_shot_clinical_annotation_budget"]
    fs_syn = bench["synthbrachy3t_sim_to_real_few_shot_budget"]
    shots = [1, 5, 10, 20, 43]
    x_pos = np.log2(shots)

    f1_syn_all = [fs_syn[f"{k}_shot"]["cohort_all_1063_catheters"]["f1_1to1_pct"] for k in shots]
    f1_zero = fs_syn["0_shot"]["cohort_all_1063_catheters"]["f1_1to1_pct"]

    # Zero-shot floor line
    ax2.axhline(f1_zero, color=C["st0"], ls=":", lw=1.1, zorder=2)
    ax2.text(x_pos[-1] - 0.05, f1_zero - 5.2, f"0-shot, no learned scoring ({f1_zero:.1f}%)", fontsize=5.5, color=C["st0"], fontweight="bold", ha="right")

    # Curves + patient-level bootstrap 95% CI bands taken from the benchmark JSON
    def _ci(src, k):
        return src[f"{k}_shot"]["cohort_all_1063_catheters"]["bootstrap_95ci_1000_patients"]["f1_1to1_pct"]
    ci_syn = np.array([_ci(fs_syn, k) for k in shots])

    ax2.plot(x_pos, f1_syn_all, color=C["nf"], lw=1.6, marker="o", ms=4.2, zorder=5)
    ax2.fill_between(x_pos, ci_syn[:, 0], ci_syn[:, 1], color=C["nf"], alpha=0.18, zorder=3)



    # Direct labels on curves
    ax2.text(x_pos[-1] - 0.05, f1_syn_all[-1] + 2.6, "NF 2026 + SynthBrachy3T (88.2%)", fontsize=5.6, fontweight="bold", color=C["nf"], ha="right")

    # Highlight k=1 shot Sim-to-Real jump
    ax2.annotate(f"1-shot: {f1_syn_all[0]:.1f}%",
                 xy=(x_pos[0], f1_syn_all[0]), xytext=(x_pos[0] + 0.05, 58.0),
                 fontsize=5.6, fontweight="bold", color=C["nf"],
                 arrowprops=dict(arrowstyle="->", color=C["nf"], lw=0.6))

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(k) for k in shots])
    ax2.set_ylim(12, 95)
    ax2.set_xlabel("Annotated clinical training patients per fold ($k$)", fontsize=7.5)
    ax2.set_ylabel("Out-of-fold accuracy (%)", fontsize=7.5)
    ax2.set_title("b   Few-shot clinical annotation efficiency", fontsize=7.5, fontweight="bold", loc="left")
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
    fp_pool = np.sort([c["candidate_pool_before_graph_transformer"] for c in cases])
    fp_st1 = np.sort([c["raw_unet_tubes_no_physics_or_gt"] for c in cases])
    fp_nf60 = np.sort([c["unified_needlefinder_2026_by_tau"]["60"] for c in cases])
    fp_sym = np.sort([c["stage1_unet_sym_tubes_only"] for c in cases])

    xs_eval = np.arange(0, 51)
    for arr, col, ls, lw, lbl in [
        (fp_pool, "#999999", ":", 1.1, "Unpruned (20.5/vol)"),
        (fp_st1, C["st1"], "--", 1.2, "Stage 1 tubes (6.4/vol)"),
        (fp_nf60, C["nf"], "-", 1.3, "NF 2026 τ=60% (3.9/vol)"),
        (fp_sym, C["st0"], "-", 1.6, "Stage 0+1 gate (0.38/vol)"),
    ]:
        ecdf = [100.0 * np.mean(arr <= x) for x in xs_eval]
        ax1.step(xs_eval, ecdf, where="post", color=col, ls=ls, lw=lw, label=lbl)

    ax1.scatter([0, 1], [64.58, 97.92], s=22, color=C["st0"], edgecolor="#000000", linewidth=0.6, zorder=6)
    ax1.annotate("64.6% (0 FP)", xy=(0, 64.58), xytext=(3.8, 56),
                 fontsize=5.8, fontweight="bold", color=C["st0"],
                 arrowprops=dict(arrowstyle="->", color=C["st0"], lw=0.6))
    ax1.annotate("97.9% (≤1 FP)", xy=(1, 97.92), xytext=(4.8, 86),
                 fontsize=5.8, fontweight="bold", color=C["st0"],
                 arrowprops=dict(arrowstyle="->", color=C["st0"], lw=0.6))

    ax1.set_xscale("symlog", linthresh=1.0)
    ax1.set_xticks([0, 1, 2, 5, 10, 20, 50])
    ax1.set_xticklabels(["0", "1", "2", "5", "10", "20", "50"])
    ax1.set_xlim(-0.15, 50)
    ax1.set_ylim(0, 104)
    ax1.set_xlabel("False-positive tracks / volume", fontsize=7.0)
    ax1.set_ylabel("Volumes with ≤ x FP (%)", fontsize=7.0)
    ax1.set_title("a   0-needle specificity (48 vols)", fontsize=7.0, fontweight="bold", loc="left")
    ax1.legend(loc="lower right", fontsize=5.2, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    ax1.grid(True, color="#E6E6E6", lw=0.4, zorder=0)

    # --- Panel b: End-to-End Runtime vs Literature (Log10 scale) ---
    rt_methods = [
        ("Zaffino 2019 (PMB, ≥2 min)ʳ", 120.0, C["b_pmb19"]),
        ("Shaaer 2022 (assisted)ʳ", 660.0, "#7F7F7F"),
        ("PMB 2019 (re-impl.)", 14.8, C["b_pmb19"]),
        ("Xiao 2026 (Sync)ʳ", 11.6, C["b_xiao"]),
        ("NF 2026 (CPU)", 2.74, C["nf"]),
        ("NF 2026 (GPU)", 1.25, C["nf"]),
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
        ("Stage 0: Symmetry", 0.410, 0.410, C["st0"]),
        ("Stage 1: ResUNet+IRGM", 0.960, 0.220, C["st1"]),
        ("Stage 2A: Track+MPNN", 1.220, 0.540, C["st2"]),
        ("Stage 2B: GraphTF", 0.150, 0.080, C["nf"]),
    ]
    left_c, left_g = 0.0, 0.0
    for st_lbl, tc, tg, col in stages:
        ax3.barh(1, tc, left=left_c, height=0.46, color=col, edgecolor="white", lw=0.6, label=st_lbl)
        ax3.barh(0, tg, left=left_g, height=0.46, color=col, edgecolor="white", lw=0.6)
        left_c += tc
        left_g += tg

    ax3.text(left_c + 0.07, 1, "2.74 s", va="center", ha="left", fontsize=6.2, fontweight="bold")
    ax3.text(left_g + 0.07, 0, "1.25 s", va="center", ha="left", fontsize=6.2, fontweight="bold")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["GPU\n(4090)", "CPU\n(8-core)"], fontsize=6.2, fontweight="bold")
    ax3.set_xlim(0, 3.65)
    ax3.set_ylim(-0.6, 1.95)
    ax3.set_xlabel("Stage runtime (s, linear)", fontsize=7.0)
    ax3.set_title("c   Stage breakdown", fontsize=7.0, fontweight="bold", loc="left")
    ax3.legend(loc="upper right", fontsize=5.2, frameon=False)
    ax3.grid(axis="x", color="#E6E6E6", lw=0.4, zorder=0)

    assert_no_overlap(fig, pad_px=1)
    save_all_formats(fig, "fig6_specificity_and_runtime")
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
    print("All 6 publication vector PDF figures + CSV data files generated successfully!")


if __name__ == "__main__":
    main()
