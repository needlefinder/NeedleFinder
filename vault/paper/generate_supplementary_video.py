#!/usr/bin/env python3
"""
Generate Supplementary Video 1 (MP4) for Medical Image Analysis (MedIA):
360-degree 3D Orbital Rotation of Reconstructed Catheter Bundle, Split-Conformal 90% Tip Spheres,
Multi-Scale 3T MRI Progression, and 3D Slicer / TG-43U1 Dosimetric Triage.
"""
import os
import subprocess
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from generate_pmb_figures import (
    load_all_sources,
    render_true_3d_catheter_bundle,
    C,
)


def build_video():
    bench, inter, spec, qual = load_all_sources()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_mp4 = os.path.join(out_dir, "supplementary_video1_3d_catheter_bundle.mp4")

    mri = qual["dense_mri_slice"]
    sym = qual["dense_sym_slice"]
    peak = qual["dense_peak_slice"]
    mask = qual["dense_mask_slice"]
    obt = qual["dense_obt_slice"]
    gt_ras = qual["dense_gt_ras_obj"]
    pred_ras = qual.get("dense_pred_ras_obj")

    H, W = mri.shape
    y0, y1 = int(H * 0.12), int(H * 0.88)
    x0, x1 = int(W * 0.12), int(W * 0.88)
    mri_c = mri[y0:y1, x0:x1]
    sym_c = sym[y0:y1, x0:x1]
    peak_c = peak[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]
    obt_c = obt[y0:y1, x0:x1]
    vmin, vmax = np.percentile(mri_c, [2, 98])

    n_frames = 120  # 8 seconds at 15 fps (encoded at 30 fps via frame interpolation / pts)
    tmpdir = tempfile.mkdtemp(prefix="nf_vid_")

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)  # 1920 x 1080 px HD

    # Static background & side panels
    ax_bg = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_bg.set_xlim(0, 192)
    ax_bg.set_ylim(0, 108)
    ax_bg.axis("off")

    banner = FancyBboxPatch((2.0, 97.5), 188.0, 8.5, boxstyle="round,pad=0.4", fc="#0F172A", ec="#1E293B", lw=1.0)
    ax_bg.add_patch(banner)
    ax_bg.text(96.0, 101.8,
               "NeedleFinder 2026 — Supplementary Video 1: 3D Catheter Bundle, Split-Conformal Tip Spheres & 3D Slicer Triage",
               fontsize=11.0, fontweight="bold", color="#FFFFFF", ha="center", va="center")

    # Left Card: 3T T2W MRI + Stage 0 + Stage 1
    c_left = FancyBboxPatch((2.0, 3.0), 50.0, 92.0, boxstyle="round,pad=0.5", fc="#F8FAFC", ec="#CBD5E1", lw=1.0)
    c_mid = FancyBboxPatch((54.5, 3.0), 83.0, 92.0, boxstyle="round,pad=0.5", fc="#FFFFFF", ec=C["nf_border"], lw=1.2)
    c_right = FancyBboxPatch((140.0, 3.0), 50.0, 92.0, boxstyle="round,pad=0.5", fc=C["nf_tint"], ec=C["nf_border"], lw=1.0)
    ax_bg.add_patch(c_left)
    ax_bg.add_patch(c_mid)
    ax_bg.add_patch(c_right)

    ax_bg.text(27.0, 91.5, "1. Multi-Scale 3T MRI Slices (Case 017)", fontsize=9.8, fontweight="bold", color="#1E293B", ha="center")
    ax_bg.text(96.0, 91.5, "2. 360° 3D Reconstructed Bundle & 90% Conformal Tip Spheres", fontsize=10.2, fontweight="bold", color=C["nf_text"], ha="center")
    ax_bg.text(165.0, 91.5, "3. Clinical & Dosimetric Summary", fontsize=9.8, fontweight="bold", color="#1E293B", ha="center")

    # Left 3 static image panels
    ax_l1 = fig.add_axes([0.045, 0.565, 0.18, 0.25])
    ax_l1.imshow(mri_c, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    ax_l1.set_xticks([]); ax_l1.set_yticks([])
    ax_l1.set_title("Native 3T T2W MRI (6F signal voids)", fontsize=8.4, fontweight="bold", color="#1E293B")

    ax_l2 = fig.add_axes([0.045, 0.295, 0.18, 0.25])
    ax_l2.imshow(mri_c, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", alpha=0.72)
    sym_ma = np.ma.masked_where(sym_c < 0.18, sym_c)
    ax_l2.imshow(sym_ma, cmap="YlOrBr", vmin=0.1, vmax=0.95, origin="upper", alpha=0.92)
    ax_l2.set_xticks([]); ax_l2.set_yticks([])
    ax_l2.set_title("Stage 0: 3D Dark-Core Symmetry S0 (0.27 s)", fontsize=8.4, fontweight="bold", color=C["st0_text"])

    ax_l3 = fig.add_axes([0.045, 0.025, 0.18, 0.25])
    ax_l3.imshow(mri_c, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", alpha=0.62)
    pk_ma = np.ma.masked_where(peak_c < 0.12, peak_c)
    ax_l3.imshow(pk_ma, cmap="Blues", vmin=0.1, vmax=0.9, origin="upper", alpha=0.90)
    ax_l3.contour(mask_c, levels=[0.35], colors=[C["st1"]], linewidths=0.8)
    if np.any(obt_c > 0.3):
        ax_l3.contour(obt_c, levels=[0.4], colors=[C["st0"]], linewidths=1.0, linestyles="--")
    ax_l3.set_xticks([]); ax_l3.set_yticks([])
    ax_l3.set_title("Stage 1: Centerline Peaks P_peak + Obturator", fontsize=8.4, fontweight="bold", color=C["st1_text"])

    # Right summary cards
    cards_r = [
        (68.5, "#FFFFFF", C["st1_border"], "Strictly Nested 5×5 OOF Benchmark",
         ["• 54 BWH 3T T2W MRI patients (1,063 catheters)",
          "• 91.4% Recall · 87.4% Precision · 89.4% F1",
          "• Outperforms 6 methods by +6.0 to +50.7 F1",
          "• Shaft: 0.47 mm median (0.44 mm long shafts)",
          "• 3D Tip: 2.26 mm median (98.8% longitudinal)"]),
        (42.5, "#FFFFFF", C["st2_border"], "Human Equivalence & Paired CT Audit",
         ["• 40 dual-expert scans (Grubbs 3-cornered hat)",
          "• Non-inferior to human experts (p = 1.3e-8)",
          "• Latent shaft error: 0.35 mm vs 0.56 mm Rater 2",
          "• 38 paired Pelvic CT scans: 52.0% of apparent",
          "  MRI 'false alarms' are real catheters on CT"]),
        (16.5, "#FFFFFF", C["st0_border"], "Conformal Tip Check & 3D Slicer (4.15 s)",
         ["• Split-conformal 90% 3D spheres: 90.2% cov.",
          "• 18 s* check of top-2 tips cuts TG-43 |ΔD90|",
          "  from 4.2% to 2.8% median (74.3% γ 2%/2mm)",
          "• Zero-shot transfer: Prostate 3T (96.3% F1),",
          "  wbSSFP (92.9% F1), 4D Insertion (82.8% F1)"]),
    ]
    for cy_r, fc_r, ec_r, title_r, lines_r in cards_r:
        cr = FancyBboxPatch((142.0, cy_r), 46.0, 22.5, boxstyle="round,pad=0.4", fc=fc_r, ec=ec_r, lw=0.9)
        ax_bg.add_patch(cr)
        ax_bg.text(144.0, cy_r + 19.2, title_r, fontsize=8.8, fontweight="bold", color="#0F172A", va="center")
        for li_idx, line_s in enumerate(lines_r):
            ax_bg.text(144.0, cy_r + 14.8 - li_idx * 3.2, line_s, fontsize=8.0, color="#1E293B", va="center")

    ax_bg.text(96.0, 5.8,
               "Legend:  — NeedleFinder 2026 3D tube (vermilion)   -- Expert reference (charcoal)   (o) 90% Conformal tip sphere (gold = top-2 review)",
               fontsize=8.3, fontweight="bold", color="#1E293B", ha="center", va="center")

    ax_3d = fig.add_axes([0.300, 0.105, 0.400, 0.725])

    for f_idx in range(n_frames):
        ax_3d.clear()
        ax_3d.axis("off")
        az = (36.0 + 360.0 * f_idx / n_frames) % 360.0
        el = 22.0 + 6.0 * np.sin(2.0 * np.pi * f_idx / n_frames)
        render_true_3d_catheter_bundle(ax_3d, gt_ras, pred_ras, az_deg=az, el_deg=el,
                                       show_anatomy=True, show_spheres=True, compact=False)
        ax_3d.margins(0.06)
        ax_3d.set_title(f"Case 017 Out-of-Fold 3D Bundle  (Azimuth = {az:5.1f}°, Elevation = {el:4.1f}°)",
                        fontsize=9.2, fontweight="bold", color="#334155", pad=4)
        frame_path = os.path.join(tmpdir, f"frame_{f_idx:04d}.png")
        fig.savefig(frame_path, dpi=150)

    plt.close(fig)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", "15",
        "-i", os.path.join(tmpdir, "frame_%04d.png"),
        "-vf", "fps=30,scale=1920:1080:flags=lanczos",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        out_mp4,
    ]
    subprocess.run(cmd, check=True)
    for fn in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, fn))
    os.rmdir(tmpdir)
    print(f"Generated Supplementary Video 1: {out_mp4}")


if __name__ == "__main__":
    build_video()
