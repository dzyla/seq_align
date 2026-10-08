import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
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


def phylogenetic_tree_section(tree):
    """
    Handles the phylogenetic tree visualization workflow.
    """
    st.header("🌳 Phylogenetic Tree")
    st.info("This section visualizes the phylogenetic tree and provides the Newick format for download.")

    if not tree:
        st.warning("No phylogenetic tree to display.")
        return

    # Visualization
    st.subheader("📊 Tree Visualization")

    view = st.radio("View", ("Interactive", "Static (Matplotlib)"), horizontal=True, key="tree_view")
    if view == "Interactive":
        st.plotly_chart(tree_figure(tree), width="stretch")
    else:
        # Calculate reasonable default height based on number of terminals (leaves)
        terminals = tree.get_terminals()
        default_height = max(8, len(terminals) * 0.2)

        with st.expander("Tree Display Settings"):
            col_set1, col_set2 = st.columns(2)
            with col_set1:
                tree_height = st.number_input("Tree Height", value=float(default_height), min_value=5.0, help="Increase this if labels are overlapping vertically.")
                tree_width = st.number_input("Tree Width", value=12.0, min_value=5.0, help="Width of the plot.")
            with col_set2:
                font_size = st.slider("Font Size", 6, 20, 10, help="Font size for labels.")
                show_labels = st.checkbox("Show Labels", value=True)
                show_inner_labels = st.checkbox("Show Inner Node Labels", value=False, help="Show labels for internal nodes (e.g., bootstrap values or node names).")

        with plt.rc_context({'font.size': font_size}):
            fig, ax = plt.subplots(figsize=(tree_width, tree_height))
        try:
            def label_func(node):
                if node.is_terminal():
                    return str(node) if show_labels else ''
                return str(node) if show_inner_labels else ''

            with plt.rc_context({'font.size': font_size}):
                Phylo.draw(tree, axes=ax, do_show=False, show_confidence=True, label_func=label_func)
            ax.set_title("Phylogenetic Tree", fontsize=16)
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            ax.get_yaxis().set_visible(False)
            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
            buf.seek(0)
            st.download_button(label="📥 Download Tree Image (PNG)", data=buf,
                               file_name="phylogenetic_tree.png", mime="image/png", on_click="ignore")
        except Exception as e:
            st.error(f"Could not draw the tree graphically: {e}")
            st.info("Displaying ASCII version of the tree instead.")
            tree_io = StringIO()
            Phylo.draw_ascii(tree, out=tree_io)
            st.code(tree_io.getvalue())
        finally:
            plt.close(fig)

    # Newick format
    st.subheader("📄 Newick Format")
    newick = tree_to_newick(tree)
    st.code(newick)
    st.download_button(label="📥 Download Newick File", data=newick, file_name="phylogenetic_tree.nwk",
                       mime="text/plain", on_click="ignore")


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


def tree_layout(tree):
    """Rectangular layout: (segments [(x0,y0,x1,y1)], leaves [(x,y,name)], internals [(x,y,confidence)])."""
    has_lengths = any(c.branch_length for c in tree.find_clades())
    depths = tree.depths(unit_branch_lengths=not has_lengths)
    terminals = tree.get_terminals()
    y = {t: i for i, t in enumerate(terminals)}

    def assign(clade):
        if clade not in y:
            ys = [assign(c) for c in clade.clades]
            y[clade] = (min(ys) + max(ys)) / 2
        return y[clade]

    assign(tree.root)
    segments, internals = [], []
    for clade in tree.find_clades(order="preorder"):
        for child in clade.clades:
            segments.append((depths[clade], y[child], depths[child], y[child]))
        if clade.clades:
            ys = [y[c] for c in clade.clades]
            segments.append((depths[clade], min(ys), depths[clade], max(ys)))
            internals.append((depths[clade], y[clade], clade.confidence))
    leaves = [(depths[t], y[t], t.name) for t in terminals]
    return segments, leaves, internals


def tree_figure(tree):
    segments, leaves, internals = tree_layout(tree)
    xs, ys = [], []
    for x0, y0, x1, y1 in segments:
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#555555", width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[l[0] for l in leaves], y=[l[1] for l in leaves], mode="markers+text",
                             text=[l[2] for l in leaves], textposition="middle right",
                             marker=dict(size=6, color="#4e79a7"), hovertemplate="%{text}<extra></extra>"))
    labelled = [(x, yy, c) for x, yy, c in internals if c is not None]
    if labelled:
        fig.add_trace(go.Scatter(x=[p[0] for p in labelled], y=[p[1] for p in labelled], mode="text",
                                 text=[str(int(p[2])) for p in labelled], textposition="top left",
                                 textfont=dict(size=10, color="#d62728"), hoverinfo="skip"))
    max_x = max([l[0] for l in leaves] + [0.0])
    fig.update_layout(showlegend=False, height=max(300, 22 * len(leaves) + 80), plot_bgcolor="white",
                      xaxis=dict(title="Branch length", range=[0, max_x * 1.35 or 1]),
                      yaxis=dict(visible=False, autorange="reversed"), margin=dict(l=20, r=20, t=30, b=40))
    return fig
