from io import BytesIO

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import ListedColormap
from modules.msa_colors import DESCRIPTIONS, SCHEMES, Coloring, color_alignment
from modules.ui import show_plotly

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
LETTER_MAX_COLUMNS = 150  # wider alignments get no letters: cells are too narrow to read
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


def _discrete_colorscale(colors):
    k = len(colors)
    scale = []
    for i, color in enumerate(colors):
        scale += [[i / k, color], [(i + 1) / k, color]]
    return scale


def msa_figure_plotly(codes, ids, seq_type, hover=True, coloring=None):
    """Overview heatmap; `coloring` (msa_colors.Coloring) overrides the per-residue palette."""
    colors = coloring.colors if coloring is not None else palette(seq_type)
    z = coloring.index if coloring is not None else codes
    k = len(colors)
    heatmap = dict(z=z, x=np.arange(1, codes.shape[1] + 1), y=list(ids), colorscale=_discrete_colorscale(colors),
                   zmin=-0.5, zmax=k - 0.5, showscale=False)
    if hover:
        heatmap["text"] = codes_to_letters(codes)
        heatmap["hovertemplate"] = "Sequence: %{y}<br>Position: %{x}<br>Residue: %{text}<extra></extra>"
        if codes.size <= LETTER_LIMIT and codes.shape[1] <= LETTER_MAX_COLUMNS:
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


def msa_figure_matplotlib(codes, ids, seq_type, coloring=None):
    colors = coloring.colors if coloring is not None else palette(seq_type)
    z = coloring.index if coloring is not None else codes
    k = len(colors)
    n_rows, n_cols = codes.shape
    fig, ax = plt.subplots(figsize=(20, max(3.5, min(30, 0.25 * len(ids) + 2))))
    # extent makes column k (1-based) sit at x=k, matching the Plotly view.
    ax.imshow(z, cmap=ListedColormap(colors), vmin=-0.5, vmax=k - 0.5, aspect="auto",
              interpolation="nearest", extent=(0.5, n_cols + 0.5, n_rows - 0.5, -0.5))
    ax.set_title("Multiple Sequence Alignment", fontsize=14)
    ax.set_xlabel("Alignment position", fontsize=12)
    if len(ids) <= 60:
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids, fontsize=8)
    else:
        ax.set_ylabel("Sequence", fontsize=12)
    if coloring is not None and coloring.legend:
        handles = [mpatches.Patch(facecolor=color, edgecolor="#999999", label=label)
                   for label, color in coloring.legend]
    else:  # one legend entry per distinct color, labelled with the residues that share it
        groups = {}
        for c in np.unique(codes):
            groups.setdefault(colors[c], []).append(ALPHABET[c])
        handles = [mpatches.Patch(facecolor=color, edgecolor="#999999",
                                  label="gap" if letters == ["-"] else "".join(letters))
                   for color, letters in groups.items()]
    if handles:
        ax.legend(handles=handles, title="Colours", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout()
    return fig


@st.cache_data(max_entries=4, show_spinner=False)
def render_msa_png(codes, ids, seq_type, scheme=None, reference=0):
    coloring = color_alignment(["".join(r) for r in codes_to_letters(codes)], scheme, seq_type, reference) \
        if scheme else None
    fig = msa_figure_matplotlib(codes, list(ids), seq_type, coloring)
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return buf.getvalue()


@st.cache_data(max_entries=8, show_spinner=False)
def _cached_coloring(seqs, scheme, seq_type, reference):
    """(index, colors, legend) of color_alignment, cached on the alignment content and the scheme."""
    coloring = color_alignment(list(seqs), scheme, seq_type, reference)
    return coloring.index, coloring.colors, coloring.legend


def plot_msa(codes, ids, method="Plotly (Interactive)", seq_type="Protein", scheme=None, reference=0):
    """Show an MSA heatmap sized to the alignment, plus a PNG download. `scheme` is an msa_colors scheme name."""
    if codes.size == 0:
        st.error("No alignment to plot.")
        return
    mode = render_mode(codes.shape[0], codes.shape[1], method)
    png = render_msa_png(codes, tuple(ids), seq_type, scheme, reference)
    if mode == "static":
        if not method.startswith("Matplotlib"):
            st.info("Alignment is too large for the interactive view; showing a static image.")
        st.image(png, width="stretch")
    else:
        coloring = Coloring(*_cached_coloring(tuple("".join(r) for r in codes_to_letters(codes)), scheme, seq_type,
                                              reference)) if scheme else None
        show_plotly(msa_figure_plotly(codes, ids, seq_type, hover=mode == "interactive_hover", coloring=coloring),
                    width="stretch")
    st.download_button("Download MSA Heatmap (PNG)", data=png, file_name="msa_heatmap.png",
                       mime="image/png", on_click="ignore", icon=":material/download:")


VIEWER_WIDTHS = (40, 150, 80)      # min, max, default columns in the sequence viewer window
VIEWER_MAX_ROWS = 120              # rows shown at once; more sequences get a row window


def reference_numbering(ref_aligned):
    """Reference residue number for every alignment column (None where the reference has a gap)."""
    numbers, k = [], 0
    for c in ref_aligned:
        k += c not in "-."
        numbers.append(k if c not in "-." else None)
    return numbers


def _text_colour(hex_colour):
    """Dark or white letter colour, whichever contrasts more with the WCAG relative luminance of the background."""
    h = hex_colour.lstrip("#")
    lin = [(c / 12.92) if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
           for c in (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#ffffff" if luminance < 0.45 else "#1a1a1a"


def msa_viewer_figure(codes, ids, coloring, start, width, ref_numbers, conservation, consensus):
    """Sequence viewer: a window of columns with letters in every cell, a conservation bar and a consensus row
    above, the x axis labelled in reference numbering every 10 residues."""
    from plotly.subplots import make_subplots

    end = min(codes.shape[1], start + width)
    cols = np.arange(start, end)
    letters = codes_to_letters(codes[:, start:end])
    z = coloring.index[:, start:end]
    k = len(coloring.colors)
    rows = ["consensus"] + list(ids)
    z = np.vstack([np.zeros((1, len(cols)), dtype=z.dtype), z])
    text = np.vstack([np.array(list(consensus[start:end]))[None, :], letters])
    refpos = np.array([[("" if ref_numbers[c] is None else str(ref_numbers[c])) for c in cols]] * len(rows))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.12, 0.88], vertical_spacing=0.01)
    cons = np.nan_to_num(conservation[start:end])
    fig.add_trace(go.Bar(x=cols + 1, y=cons, marker_color="#0f766e", showlegend=False,
                         hovertemplate="Column %{x}: conservation %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=z, x=cols + 1, y=rows, colorscale=_discrete_colorscale(coloring.colors),
                             zmin=-0.5, zmax=k - 0.5, showscale=False, text=text, customdata=refpos,
                             xgap=1, ygap=1,
                             hovertemplate="%{y}<br>Column %{x} (reference %{customdata})<br>%{text}<extra></extra>"),
                  row=2, col=1)
    # letters sit on their own trace so each one can contrast with its cell's background
    background = np.asarray(coloring.colors, dtype=object)[z]
    fig.add_trace(go.Scatter(x=np.tile(cols + 1, len(rows)), y=np.repeat(rows, len(cols)), mode="text",
                             text=text.ravel(), hoverinfo="skip", showlegend=False,
                             textfont=dict(size=11, family="IBM Plex Mono, monospace",
                                           color=[_text_colour(c) for c in background.ravel()])),
                  row=2, col=1)
    ticks = [c + 1 for c in cols if ref_numbers[c] is not None and ref_numbers[c] % 10 == 0]
    fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=[str(ref_numbers[t - 1]) for t in ticks],
                     row=2, col=1, title_text="Reference position")
    fig.update_yaxes(range=[0, 1], showticklabels=False, row=1, col=1)
    fig.update_yaxes(type="category", autorange="reversed", row=2, col=1)
    fig.update_layout(height=min(4000, 120 + 22 * len(rows)), margin=dict(l=10, r=10, t=10, b=40),
                      plot_bgcolor="white", bargap=0.1)
    return fig


def _jump_to_residue(ref_numbers, n_cols):
    """Callback for the jump box: centre the window on the reference residue, once per change of the box."""
    jump = st.session_state.msa_view_jump
    if not jump:
        return
    width = st.session_state.get("msa_view_width", VIEWER_WIDTHS[2])
    target = next((c for c, n in enumerate(ref_numbers) if n == jump), 0)
    st.session_state.msa_view_start = max(1, min(n_cols - width + 1, target + 1 - width // 2))
    st.session_state.msa_view_jump = 0  # so the same residue can be jumped to again


def alignment_view(pairs, seq_type, plot_method, reference_id):
    """MSA tab view: colour scheme + overview heatmap or sequence viewer."""
    from modules.conservation import column_conservation
    from modules.espript import column_classes, consensus_line

    ids = [i for i, _ in pairs]
    seqs = [s for _, s in pairs]
    c1, c2 = st.columns([1, 1])
    mode = c1.segmented_control("View", ["Overview heatmap", "Sequence viewer"], default="Overview heatmap",
                                required=True, key="msa_view_mode")
    schemes = SCHEMES["DNA" if seq_type == "DNA" else "Protein"]
    scheme = c2.selectbox("Colour scheme", schemes, key="msa_color_scheme")
    st.caption(DESCRIPTIONS[scheme])
    reference = ids.index(reference_id) if reference_id in ids else 0
    codes = alignment_to_codes(seqs)
    if mode == "Overview heatmap":
        plot_msa(codes, ids, plot_method, seq_type, scheme, reference)
        return
    coloring = Coloring(*_cached_coloring(tuple(seqs), scheme, seq_type, reference))
    ref_numbers = reference_numbering(seqs[reference])
    lo, hi, default = VIEWER_WIDTHS
    c1, c2, c3 = st.columns([1, 2, 1])
    width = c1.slider("Columns shown", lo, hi, default, 10, key="msa_view_width")
    n_cols = codes.shape[1]
    c3.number_input(f"Jump to {ids[reference]} residue", 0, max(n for n in ref_numbers if n) if any(ref_numbers)
                    else 0, 0, key="msa_view_jump", help="0 keeps the slider position.",
                    on_change=_jump_to_residue, args=(ref_numbers, n_cols))
    # the slider range does not depend on the width; the window is clamped so it always ends at the last column
    start = c2.slider("Start column", 1, n_cols, key="msa_view_start") if n_cols > 1 else 1
    start = min(start, max(1, n_cols - width + 1))
    rows = np.arange(len(ids))
    if len(ids) > VIEWER_MAX_ROWS:
        first = st.slider("First sequence", 1, len(ids) - VIEWER_MAX_ROWS + 1, 1, key="msa_view_row")
        rows = rows[first - 1:first - 1 + VIEWER_MAX_ROWS]
    conservation = column_conservation(seqs, seq_type)
    consensus = consensus_line(seqs, column_classes(seqs, 0.7, seq_type))
    # column statistics come from the whole alignment; a row window only slices the colouring
    sub = Coloring(coloring.index[rows], coloring.colors, coloring.legend)
    show_plotly(msa_viewer_figure(codes[rows], [ids[r] for r in rows], sub, start - 1, width, ref_numbers,
                                  conservation, consensus), width="stretch")
