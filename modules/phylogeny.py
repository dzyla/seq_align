import math

import matplotlib.pyplot as plt
import numpy as np
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
            Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False, label_func=label_func)
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
    """Symmetric distance matrix from aligned strings (pairwise deletion of gap columns)."""
    n = len(aligned)
    matrix = np.zeros((n, n))
    upper = [s.upper() for s in aligned]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = _finish(p_distance(upper[i], upper[j]), seq_type, correction)
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
