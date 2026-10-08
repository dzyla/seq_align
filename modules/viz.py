from io import BytesIO

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import ListedColormap

ALPHABET = "-ACDEFGHIKLMNPQRSTVWYXBZJOU*"
GAP_CODE = 0
_UNKNOWN_CODE = ALPHABET.index("X")
_LUT = np.full(256, _UNKNOWN_CODE, dtype=np.uint8)
for _i, _ch in enumerate(ALPHABET):
    _LUT[ord(_ch)] = _i
    _LUT[ord(_ch.lower())] = _i
_LUT[ord(".")] = GAP_CODE
_LETTERS = np.array(list(ALPHABET))

# Clustal X-style residue classes; nucleotides use the common A/C/G/T scheme.
PROTEIN_COLORS = {
    "-": "#ffffff",
    **dict.fromkeys("AILMFWV", "#80a0f0"),
    "C": "#f08080",
    **dict.fromkeys("KR", "#f01505"),
    **dict.fromkeys("DE", "#c048c0"),
    **dict.fromkeys("NQST", "#15c015"),
    "G": "#f09048",
    "P": "#c0c000",
    **dict.fromkeys("HY", "#15a4a4"),
}
DNA_COLORS = {"-": "#ffffff", "A": "#64f73f", "C": "#ffb340", "G": "#eb413c", "T": "#3c88ee", "U": "#3c88ee"}
OTHER_COLOR = "#bbbbbb"
HOVER_LIMIT = 200_000      # cells; above this, no per-cell hover text is sent to the browser
PLOTLY_LIMIT = 2_000_000   # cells; above this, a static image is rendered instead
LETTER_LIMIT = 3_000        # cells; up to this, residue letters are drawn inside the heatmap cells


def palette(seq_type):
    """Hex color for every code in ALPHABET."""
    colors = DNA_COLORS if seq_type == "DNA" else PROTEIN_COLORS
    return [colors.get(ch, OTHER_COLOR) for ch in ALPHABET]


def alignment_to_codes(seqs):
    """Encode equal-length aligned strings as a uint8 matrix of ALPHABET indices."""
    if not seqs:
        return np.zeros((0, 0), dtype=np.uint8)
    length = len(seqs[0])
    if any(len(s) != length for s in seqs):
        raise ValueError("All aligned sequences must have the same length.")
    raw = np.frombuffer("".join(seqs).encode("ascii", errors="replace"), dtype=np.uint8)
    return _LUT[raw.reshape(len(seqs), length)]


def codes_to_letters(codes):
    return _LETTERS[codes]


def render_mode(n_rows, n_cols, method):
    cells = n_rows * n_cols
    if method.startswith("Matplotlib") or cells > PLOTLY_LIMIT:
        return "static"
    return "interactive_hover" if cells <= HOVER_LIMIT else "interactive"


def msa_figure_plotly(codes, ids, seq_type, hover=True):
    colors = palette(seq_type)
    k = len(colors)
    colorscale = []
    for i, color in enumerate(colors):
        colorscale += [[i / k, color], [(i + 1) / k, color]]
    heatmap = dict(z=codes, x=np.arange(1, codes.shape[1] + 1), y=list(ids), colorscale=colorscale,
                   zmin=-0.5, zmax=k - 0.5, showscale=False)
    if hover:
        heatmap["text"] = codes_to_letters(codes)
        heatmap["hovertemplate"] = "Sequence: %{y}<br>Position: %{x}<br>Residue: %{text}<extra></extra>"
        if codes.size <= LETTER_LIMIT:
            heatmap["texttemplate"] = "%{text}"
            heatmap["textfont"] = dict(size=10, color="#222222")
    else:
        heatmap["hovertemplate"] = "Sequence: %{y}<br>Position: %{x}<extra></extra>"
    fig = go.Figure(go.Heatmap(**heatmap))
    fig.update_layout(
        title="Multiple Sequence Alignment",
        xaxis_title="Alignment position",
        yaxis=dict(type="category", autorange="reversed", showticklabels=len(ids) <= 60),
        height=min(900, 200 + 14 * len(ids)),
        plot_bgcolor="white",
    )
    return fig


def msa_figure_matplotlib(codes, ids, seq_type):
    colors = palette(seq_type)
    k = len(colors)
    n_rows, n_cols = codes.shape
    fig, ax = plt.subplots(figsize=(20, max(3.5, min(30, 0.25 * len(ids) + 2))))
    # extent makes column k (1-based) sit at x=k, matching the Plotly view.
    ax.imshow(codes, cmap=ListedColormap(colors), vmin=-0.5, vmax=k - 0.5, aspect="auto",
              interpolation="nearest", extent=(0.5, n_cols + 0.5, n_rows - 0.5, -0.5))
    ax.set_title("Multiple Sequence Alignment", fontsize=14)
    ax.set_xlabel("Alignment position", fontsize=12)
    if len(ids) <= 60:
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids, fontsize=8)
    else:
        ax.set_ylabel("Sequence", fontsize=12)
    # One legend entry per distinct color, labelled with the residues that share it.
    groups = {}
    for c in np.unique(codes):
        groups.setdefault(colors[c], []).append(ALPHABET[c])
    handles = [mpatches.Patch(facecolor=color, edgecolor="#999999",
                              label="gap" if letters == ["-"] else "".join(letters))
               for color, letters in groups.items()]
    ax.legend(handles=handles, title="Residues", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout()
    return fig


@st.cache_data(max_entries=4, show_spinner=False)
def render_msa_png(codes, ids, seq_type):
    fig = msa_figure_matplotlib(codes, list(ids), seq_type)
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return buf.getvalue()


def plot_msa(codes, ids, method="Plotly (Interactive)", seq_type="Protein"):
    """Show an MSA heatmap sized to the alignment, plus a PNG download."""
    if codes.size == 0:
        st.error("No alignment to plot.")
        return
    mode = render_mode(codes.shape[0], codes.shape[1], method)
    png = render_msa_png(codes, tuple(ids), seq_type)
    if mode == "static":
        if not method.startswith("Matplotlib"):
            st.info("Alignment is too large for the interactive view; showing a static image.")
        st.image(png, width="stretch")
    else:
        st.plotly_chart(msa_figure_plotly(codes, ids, seq_type, hover=mode == "interactive_hover"), width="stretch")
    st.download_button("📥 Download MSA Heatmap (PNG)", data=png, file_name="msa_heatmap.png",
                       mime="image/png", on_click="ignore")
