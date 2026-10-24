# NeedleFinder 2026 · PMB Scientific Visual Design System (CVD-safe + 1:1 160mm scale)
"""
Single source of truth for colors, typography, marker shapes, line styles,
hatching patterns, and automated figure geometry/font-size linting.

Colorblind (CVD) safety guarantees:
- Okabe-Ito + Paul Tol high-contrast tokens tested under Machado (2009)
  protanopia, deuteranopia, tritanopia (severity 1.0), and luminance grayscale.
- Semantic lock:
    * Stage 0 (Physics Dark-Core): Warm Amber (#E69F00 fill, #8A5F00 text)
    * Stage 1 (3D ResUNet + SSL):  Ocean Blue (#0072B2 fill, #005A8C text)
    * Stage 2 (Graph + 1D Tip):    Bluish Green (#009E73 fill, #00704F text)
    * NeedleFinder 2026 (Ours):    Vermilion (#D55E00 fill, #9E4400 text)
    * Human Expert Reference:      Deep Charcoal (#1A1A1A)
    * Published Baselines:         Monotonic luminance grey ramp (#383838 -> #A6A6A6)
                                   always paired with distinct marker shapes.
    * False Positives / Rejected:  Reddish Purple (#CC79A7 fill, #882255 edge/text) + '////' hatch
    * Redundant / Duplicate:       Sky Blue (#56B4E9 fill, #1E6B99 edge/text) + '...' hatch
"""

NF = dict(
    # Pipeline stage semantic tokens (Okabe-Ito)
    st0="#E69F00", st0_text="#7A5200", st0_tint="#FDF4DC", st0_border="#C88700",
    st1="#0072B2", st1_text="#004E7A", st1_tint="#E6F2F8", st1_border="#005A8C",
    st2="#009E73", st2_text="#006346", st2_tint="#E5F6F0", st2_border="#007A58",
    st2c="#007A58", st2c_text="#00523B", st2c_tint="#DDF2EB",
    # NeedleFinder (proposed) tokens
    nf="#D55E00",  nf_text="#9E4200",  nf_tint="#FCEEE4", nf_border="#B34B00",
    nf_alt="#E69F00", nf_fusion="#B34800",
    # Published baseline greyscale ramp (monotonically ordered + shape double-encoded)
    b_pmb="#383838", b_pmb19="#383838",
    b_kim="#505050",
    b_kost="#666666",
    b_dai="#7A7A7A",
    b_nnu="#8E8E8E", b_nnunet="#8E8E8E",
    b_xiao="#A2A2A2",
    # Human reference & clinical bands
    human="#1A1A1A", human_band="#E2E6EA",
    # 3D & outcome tokens (never put brick red next to vermilion!)
    gt3d="#595959",
    fn="#1A1A1A",
    fp="#CC79A7", fp_text="#7A204D", fp_tint="#F9EBF2",
    dup="#56B4E9", dup_text="#1B5E85", dup_tint="#E8F5FC",
    roi="#56B4E9",
    # Structural chrome
    axis="#2B2B2B", grid="#E2E5E9", arrow="#3A3A3A",
    card_bg="#F8FAFC", card_border="#CBD5E1",
    muted="#5A6572",
)
C = NF

# Exact 1:1 physical dimensions matching preamble.tex (\textwidth = 160 mm)
MM = 1.0 / 25.4
W1 = 78.0 * MM   # single-column width (3.07 in)
W2 = 160.0 * MM  # full textwidth (6.30 in) -- 0% LaTeX shrinkage!

MARKERS = dict(
    nf="o",
    nf_4d="*",
    nf_4a="P",
    b_pmb="s",
    b_kim="p",
    b_kost="v",
    b_dai="<",
    b_nnu="^",
    b_xiao="D",
    human="_",
)

ORDER = ["human", "b_xiao", "b_nnu", "b_dai", "b_kost", "b_kim", "b_pmb", "nf"]
LABELS = {
    "human": "Human Inter-Observer (Team 1 vs Team 2)",
    "b_pmb": "PMB 2019 3D U-Net (Zaffino et al.)",
    "b_kim": "Aleong 2024 / Kim 2026 (5-Model Unc.)",
    "b_kost": "Kostoulas et al. 2025 (Gap-Bridge)",
    "b_dai": "Dai et al. 2020 (Att. U-Net + TV)",
    "b_nnu": "ResUNet + clDice (Shit et al. 2021)",
    "b_xiao": "Xiao et al. 2026 Sync-Track (preprint)",
    "nf": "NeedleFinder 2026 (Ours)",
}


def panel_label(ax, letter, x=-0.08, y=1.05, fontsize=10.5):
    """Consistent bold lowercase panel tag (a, b, c...) at top-left of axes."""
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        color="#1A1A1A",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def assert_no_overlap(fig, pad_px=1.5, min_fontsize=7.4):
    """
    Strict automated figure linter:
      1. Fails if any visible Text artist has font size < min_fontsize pt.
      2. Fails if any two visible Text artists' bounding boxes overlap.
    """
    import matplotlib.text as mtext
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().execute(fig)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [
        t for t in fig.findobj(mtext.Text)
        if t.get_visible() and t.get_text().strip()
    ]
    small = [
        (t.get_text().replace("\n", " ")[:40], round(t.get_fontsize(), 2))
        for t in texts
        if t.get_fontsize() < min_fontsize
    ]
    if small:
        raise AssertionError(f"Font size below {min_fontsize}pt detected: {small[:8]}")

    boxes = [(t, t.get_window_extent(r).expanded(1.0, 1.0).padded(pad_px)) for t in texts]
    bad = [
        (a.get_text().replace("\n", " ")[:35], b.get_text().replace("\n", " ")[:35])
        for i, (a, ba) in enumerate(boxes)
        for (b, bb) in boxes[i + 1:]
        if ba.overlaps(bb)
    ]
    if bad:
        raise AssertionError(f"Overlapping labels detected: {bad[:10]}")
