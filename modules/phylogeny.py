import math
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from io import BytesIO, StringIO

from modules.pairwise import get_aligner
from modules.seqtools import clean_sequence

TREE_METHODS = {"Neighbor-Joining": "nj", "UPGMA": "upgma"}
MAX_DISTANCE = 10.0


class TreeError(RuntimeError):
    """Raised when a tree cannot be built."""


LAYOUT_OPTIONS = {"Rectangular": "rectangular", "Circular": "circular", "Unrooted": "unrooted"}
ROOTING_OPTIONS = {"Midpoint": "midpoint", "Outgroup": "outgroup", "As built": "as_built"}


@st.cache_data(max_entries=16, show_spinner=False)
def tree_figure_outputs(newick, layout, cladogram, rooting, outgroup, ladderize, n_clades, style_items):
    """Rooted tree, Drawing, 150-dpi preview PNG, SVG, PDF and PNG export dpi for one set of settings (cached)."""
    from modules.tree_layout import assign_clades, compute_geometry, prepare_tree, recursion_headroom
    from modules.figures import png_dpi
    from modules.tree_render import TreeStyle, export_bytes, place, quiet_fonts, render_matplotlib

    style = TreeStyle(**{k: list(v) if k == "clade_names" else v for k, v in style_items})
    tree = Phylo.read(StringIO(newick), "newick")
    with recursion_headroom(tree):
        tree = prepare_tree(tree, rooting, outgroup, ladderize)
        geometry = compute_geometry(tree, layout, cladogram, assign_clades(tree, n_clades))
        drawing = place(geometry, style)
        fig = render_matplotlib(drawing)
        preview = BytesIO()
        with quiet_fonts():  # framed exactly like the downloads
            fig.savefig(preview, format="png", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.05)
        return {"drawing": drawing, "preview": preview.getvalue(), "svg": export_bytes(fig, "svg"),
                "pdf": export_bytes(fig, "pdf"), "newick": tree_to_newick(tree), "png_dpi": png_dpi(fig)}


def _png(drawing):
    from modules.tree_render import export_bytes, render_matplotlib
    return export_bytes(render_matplotlib(drawing), "png")


def phylogenetic_tree_section(tree, key_prefix="tree"):
    """Publication tree figure: layout, style, rooting and label settings, a Figure/Explore view and
    SVG/PDF/PNG/Newick downloads. `key_prefix` keeps settings separate per page."""
    from dataclasses import asdict
    from modules.tree_layout import recursion_headroom
    from modules.tree_render import FONTS, STYLES, TreeStyle, render_plotly
    from modules.ui import show_plotly, theme_type

    if not tree:
        st.warning("No phylogenetic tree to display.")
        return
    key = lambda name: f"{key_prefix}_{name}"
    with recursion_headroom(tree):
        leaves = [t.name for t in tree.get_terminals()]
        newick_in = tree_to_newick(tree)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        layout = c1.segmented_control("Layout", list(LAYOUT_OPTIONS), default="Rectangular", required=True,
                                      key=key("layout"))
        style_label = c2.segmented_control("Style", list(STYLES), default="Classic ink", required=True,
                                           key=key("style"))
        c1, c2, c3 = st.columns(3)
        rooting = c1.selectbox("Rooting", list(ROOTING_OPTIONS), key=key("rooting"))
        cladogram = c2.toggle("Cladogram", key=key("cladogram"), disabled=layout == "Unrooted",
                              help="Ignore branch lengths and align the tips.")
        ladderize = c3.toggle("Ladderize", value=True, key=key("ladderize"))
        outgroup = []
        if rooting == "Outgroup":
            outgroup = st.multiselect("Outgroup leaves", leaves, key=key("outgroup"))
            if not outgroup:
                st.info("Pick one or more outgroup leaves; midpoint rooting is used until then.")
        n_clades, clade_names = 1, []
        if STYLES[style_label] == "clades":
            c1, c2 = st.columns([1, 2])
            if len(leaves) > 2:
                n_clades = c1.slider("Clades", 2, min(12, len(leaves)), 3, key=key("clades"))
            else:  # a slider needs min < max
                n_clades = len(leaves)
            store = st.session_state.setdefault(key("clade_name_store"), [])  # names survive changes of k
            seed = [store[i] if i < len(store) else f"Clade {i + 1}" for i in range(n_clades)]
            table = c2.data_editor(pd.DataFrame({"Clade name": seed}), hide_index=True, width="stretch",
                                   key=key(f"clade_names_{n_clades}"))
            clade_names = table["Clade name"].fillna("").astype(str).tolist()
            st.session_state[key("clade_name_store")] = clade_names + store[n_clades:]
        support_min = st.slider("Show support ≥", 0, 100, 50, key=key("support"))
        with st.expander("Labels and size"):
            c1, c2, c3 = st.columns(3)
            font = c1.selectbox("Font", list(FONTS), key=key("font"))
            font_size = c1.slider("Font size (pt)", 5, 16, 9, key=key("font_size"))
            line_width = c1.slider("Line width (pt)", 0.5, 3.0, 1.0, 0.25, key=key("line_width"))
            italic = c2.toggle("Italic labels", value=True, key=key("italic"))
            spaces = c2.toggle("Underscores as spaces", value=True, key=key("spaces"))
            truncate = c2.number_input("Truncate names at (0 = off)", 0, 60, 0, key=key("truncate"))
            width_cm = c3.slider("Width (cm)", 8.0, 40.0, 16.0, 0.5, key=key("width"))
            height_cm = None
            if layout == "Rectangular" and not c3.toggle("Automatic height", value=True, key=key("auto_height")):
                height_cm = c3.slider("Height (cm)", 4.0, 120.0, 20.0, 0.5, key=key("height"))

    style = TreeStyle(style=STYLES[style_label], support_min=float(support_min), font_family=font,
                      font_size=float(font_size), italic=italic, underscores_to_spaces=spaces,
                      truncate=int(truncate), line_width=float(line_width), width_cm=float(width_cm),
                      height_cm=height_cm, clade_names=clade_names)
    style_items = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in asdict(style).items())
    args = (newick_in, LAYOUT_OPTIONS[layout], cladogram and layout != "Unrooted")
    rooting_mode = "midpoint" if rooting == "Outgroup" and not outgroup else ROOTING_OPTIONS[rooting]
    with st.spinner("Drawing tree..."):
        try:
            out = tree_figure_outputs(*args, rooting_mode, tuple(outgroup), ladderize, n_clades, style_items)
        except TreeError as e:
            st.warning(f"{e} Using midpoint rooting instead.")
            try:
                out = tree_figure_outputs(*args, "midpoint", (), ladderize, n_clades, style_items)
            except TreeError as e2:
                st.error(str(e2))
                return
    drawing = out["drawing"]
    if drawing.warning:
        st.warning(drawing.warning)
    if layout != "Rectangular" and drawing.width > width_cm / 2.54 + 0.05:
        st.caption(f"The figure is {drawing.width * 2.54:.0f} cm wide so labels fit; reduce the font size to make "
                   "it smaller.")

    view = st.segmented_control("View", ["Figure", "Explore"], default="Figure", required=True, key=key("view"),
                                help="Figure is exactly what you download. Explore adds hover and zoom.")
    if view == "Figure":
        st.image(out["preview"])
    else:
        dark = theme_type() == "dark" and st.session_state.get("appearance_charts") != "Publication white"
        show_plotly(render_plotly(drawing, dark=dark), width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("SVG", out["svg"], "phylogenetic_tree.svg", "image/svg+xml", icon=":material/download:",
                       on_click="ignore", width="stretch", key=key("dl_svg"))
    c2.download_button("PDF", out["pdf"], "phylogenetic_tree.pdf", "application/pdf",
                       icon=":material/download:", on_click="ignore", width="stretch", key=key("dl_pdf"))
    c3.download_button(f"PNG ({out['png_dpi']} dpi)", lambda: _png(drawing), "phylogenetic_tree.png", "image/png",
                       icon=":material/download:", on_click="ignore", width="stretch", key=key("dl_png"))
    c4.download_button("Newick", out["newick"], "phylogenetic_tree.nwk", "text/plain",
                       icon=":material/download:", on_click="ignore", width="stretch", key=key("dl_nwk"),
                       help="The tree as rooted and ladderized above.")


def tree_to_newick(tree):
    out = StringIO()
    Phylo.write(tree, out, "newick")
    return out.getvalue()


def p_distance(aligned1, aligned2):
    """Fraction of differing residues over columns where neither sequence has a gap."""
    valid = diffs = 0
    for a, b in zip(aligned1, aligned2):
        if a != '-' and b != '-':
            valid += 1
            diffs += a != b
    return diffs / valid if valid else None


def correct_distance(p, seq_type):
    """Jukes–Cantor (DNA) or Kimura (protein) correction for multiple substitutions, capped."""
    if seq_type == "DNA":
        x = 1 - 4 * p / 3
        return MAX_DISTANCE if x <= 0 else min(MAX_DISTANCE, -0.75 * math.log(x))
    x = 1 - p - 0.2 * p * p
    return MAX_DISTANCE if x <= 0 else min(MAX_DISTANCE, -math.log(x))


def _finish(p, seq_type, correction):
    if p is None:
        return MAX_DISTANCE if correction else 1.0
    return correct_distance(p, seq_type) if correction else p


def alignment_distance_matrix(aligned, seq_type, correction=True):
    """Symmetric distance matrix from aligned strings (pairwise deletion of gap columns), vectorized."""
    n = len(aligned)
    matrix = np.zeros((n, n))
    if n < 2:
        return matrix
    if not aligned[0]:
        return np.full((n, n), _finish(None, seq_type, correction)) * (1 - np.eye(n))
    arr = np.frombuffer("".join(s.upper() for s in aligned).encode(), dtype=np.uint8).reshape(n, -1)
    residue = arr != ord('-')
    for i in range(n - 1):
        valid = residue[i] & residue[i + 1:]
        diffs = ((arr[i] != arr[i + 1:]) & valid).sum(axis=1)
        counts = valid.sum(axis=1)
        for k, (d, c) in enumerate(zip(diffs, counts)):
            value = _finish(d / c if c else None, seq_type, correction)
            matrix[i, i + 1 + k] = matrix[i + 1 + k, i] = value
    return matrix


def compute_distance(seq1, seq2, seq_type, correction=True, aligner=None):
    """Distance between two unaligned sequences via global pairwise alignment."""
    s1 = clean_sequence(seq1, seq_type)
    s2 = clean_sequence(seq2, seq_type)
    if not s1 or not s2:
        return _finish(None, seq_type, correction)
    aligner = aligner or get_aligner(seq_type, mode="global")
    try:
        alignment = next(iter(aligner.align(s1, s2)))
    except StopIteration:
        return _finish(None, seq_type, correction)
    return _finish(p_distance(str(alignment[0]), str(alignment[1])), seq_type, correction)


def _tree_from_matrix(names, matrix, method):
    lower = [[float(matrix[i][j]) for j in range(i + 1)] for i in range(len(names))]
    constructor = DistanceTreeConstructor()
    dm = DistanceMatrix(names=list(names), matrix=lower)
    tree = constructor.nj(dm) if method == "nj" else constructor.upgma(dm)
    for clade in tree.get_nonterminals():
        clade.name = None  # Biopython's "Inner1" names would be glued to supports in Newick
    if method == "nj":
        for clade in tree.find_clades():
            if clade.branch_length is not None and clade.branch_length < 0:
                clade.branch_length = 0.0
    if method == "nj" and len(names) > 2 and max(max(row) for row in lower) > 0:
        tree.root_at_midpoint()
    return tree


def build_tree_from_alignment(alignment, seq_type, method="nj", correction=True):
    """Tree from an MSA (Biopython MultipleSeqAlignment)."""
    try:
        names = [record.id for record in alignment]
        matrix = alignment_distance_matrix([str(r.seq) for r in alignment], seq_type, correction)
        return _tree_from_matrix(names, matrix, method)
    except Exception as e:
        raise TreeError(f"Tree construction from MSA failed: {e}") from e


def build_phylogenetic_tree(sequences, seq_type, method="nj", correction=True):
    """Tree from unaligned SeqRecords using pairwise global alignments."""
    try:
        aligner = get_aligner(seq_type, mode="global")
        n = len(sequences)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = matrix[j, i] = compute_distance(
                    sequences[i].seq, sequences[j].seq, seq_type, correction, aligner)
        return _tree_from_matrix([s.id for s in sequences], matrix, method)
    except Exception as e:
        raise TreeError(f"Tree construction failed: {e}") from e


def _normalized_split(names, all_names, anchor):
    side = all_names - names if anchor in names else names
    if len(side) < 2 or len(side) > len(all_names) - 2:
        return None
    return frozenset(side)


def tree_splits(tree, all_names, anchor):
    """Non-trivial bipartitions, expressed as the side without `anchor` (independent of rooting)."""
    splits = set()
    for clade in tree.get_nonterminals():
        split = _normalized_split(frozenset(t.name for t in clade.get_terminals()), all_names, anchor)
        if split is not None:
            splits.add(split)
    return splits


def bootstrap_tree(aligned_pairs, seq_type, method="nj", correction=True, replicates=100, seed=0):
    """Tree from the full alignment; internal clades carry % bootstrap support (column resampling)."""
    try:
        names = [i for i, _ in aligned_pairs]
        seqs = [s.upper() for _, s in aligned_pairs]
        target = _tree_from_matrix(names, alignment_distance_matrix(seqs, seq_type, correction), method)
        if len(seqs) < 3 or not seqs[0]:
            return target
        arr = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), -1)
        rng = np.random.default_rng(seed)
        all_names, anchor = frozenset(names), names[0]
        counts = Counter()
        for _ in range(replicates):
            cols = rng.integers(0, arr.shape[1], arr.shape[1])
            replicate = [row.tobytes().decode() for row in arr[:, cols]]
            tree = _tree_from_matrix(names, alignment_distance_matrix(replicate, seq_type, correction), method)
            counts.update(tree_splits(tree, all_names, anchor))
        for clade in target.get_nonterminals():
            split = _normalized_split(frozenset(t.name for t in clade.get_terminals()), all_names, anchor)
            clade.confidence = round(100 * counts[split] / replicates) if split is not None else None
        return target
    except TreeError:
        raise
    except Exception as e:
        raise TreeError(f"Bootstrap tree failed: {e}") from e
