# NeedleFinder 2026 · PMB design tokens (single source of truth)
NF = dict(
    st0="#E69F00", st0_text="#8A5F00", st0_tint="#F7ECCC",
    st1="#0072B2", st1_text="#005A8C", st1_tint="#CCE3F0",
    st2="#009E73", st2_text="#00704F", st2_tint="#CCECE3",
    nf="#D55E00",  nf_text="#9E4400",  nf_tint="#FBEFE6",
    b_pmb="#4D4D4D", b_pmb19="#4D4D4D", b_nnu="#7F7F7F", b_nnunet="#7F7F7F", b_xiao="#ABABAB",
    human="#000000", human_band="#E0E0E0",
    gt3d="#BDBDBD", fn="#000000", fp="#CC79A7", roi="#56B4E9",
    axis="#333333", grid="#E6E6E6", arrow="#4D4D4D",
)
C = NF
MM = 1.0 / 25.4
W1 = 84.0 * MM
W2 = 170.0 * MM


def panel_label(ax, letter):
    ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
MARKERS = dict(nf="o", b_pmb="s", b_nnu="^", b_xiao="D", human="_")
ORDER = ["human", "b_pmb", "b_nnu", "b_xiao", "nf"]  # bottom->top in rainclouds; ours at top
LABELS = {
    "human": "Inter-observer (CY vs LB/AMe)",
    "b_pmb": "PMB 2019 3D U-Net",
    "b_nnu": "nnU-Net + clDice",
    "b_xiao": "Xiao 2026 Sync-Track",
    "nf": "NeedleFinder 2026 (Ours)",
}


def assert_no_overlap(fig, pad_px=2):
    """Fail if any two visible Text artists' bounding boxes overlap (after padding)."""
    import matplotlib.text as mtext
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [
        t for t in fig.findobj(mtext.Text)
        if t.get_visible() and t.get_text().strip()
    ]
    boxes = [(t, t.get_window_extent(r).expanded(1.0, 1.0).padded(pad_px)) for t in texts]
    bad = [
        (a.get_text(), b.get_text())
        for i, (a, ba) in enumerate(boxes)
        for (b, bb) in boxes[i + 1:]
        if ba.overlaps(bb)
    ]
    if bad:
        raise AssertionError(f"Overlapping labels detected: {bad[:10]}")
