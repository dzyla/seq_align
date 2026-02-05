import streamlit as st
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix, DistanceCalculator
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from modules.pairwise import get_aligner

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

    # Update font size
    plt.rcParams.update({'font.size': font_size})

    fig, ax = plt.subplots(figsize=(tree_width, tree_height))
    try:
        # Define label function to control visibility
        def label_func(node):
            if node.is_terminal():
                return str(node) if show_labels else ''
            else:
                return str(node) if show_inner_labels else ''

        Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False, label_func=label_func)

        ax.set_title("Phylogenetic Tree", fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)

        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        buf.seek(0)
        st.download_button(
            label="📥 Download Tree Image (PNG)",
            data=buf,
            file_name="phylogenetic_tree.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Could not draw the tree graphically: {e}")
        st.info("Displaying ASCII version of the tree instead.")
        tree_io = StringIO()
        Phylo.draw_ascii(tree, out=tree_io)
        st.code(tree_io.getvalue())

    # Newick format
    st.subheader("📄 Newick Format")
    if 'tree_newick' not in st.session_state or not st.session_state.get('tree_newick'):
        tree_newick_io = StringIO()
        Phylo.write(tree, tree_newick_io, "newick")
        st.session_state.tree_newick = tree_newick_io.getvalue()

    st.code(st.session_state.tree_newick)
    st.download_button(
        label="📥 Download Newick File",
        data=st.session_state.tree_newick,
        file_name="phylogenetic_tree.nwk",
        mime="text/plain"
    )

def build_tree_from_alignment(alignment, seq_type):
    """
    Build a phylogenetic tree from a Multiple Sequence Alignment.
    """
    try:
        if seq_type == 'DNA':
            calculator = DistanceCalculator('identity')
        else: # Protein
            calculator = DistanceCalculator('blosum62')

        dm = calculator.get_distance(alignment)
        constructor = DistanceTreeConstructor()
        tree = constructor.upgma(dm)
        return tree
    except Exception as e:
        st.error(f"Tree construction from MSA failed: {e}")
        return None


def build_phylogenetic_tree(sequences, seq_type):
    """
    Build a simple phylogenetic tree from sequences using a distance matrix.

    Parameters:
        sequences: List of sequence records
        seq_type (str): Type of sequences ('DNA' or 'Protein')

    Returns:
        Tree: BioPython Tree object
    """
    try:
        ids = [seq.id for seq in sequences]
        matrix = []
        for i, seq1 in enumerate(sequences):
            row = []
            for seq2 in sequences[:i + 1]:
                distance = compute_distance(seq1.seq, seq2.seq, seq_type)
                row.append(distance)
            matrix.append(row)
        distance_matrix = DistanceMatrix(names=ids, matrix=matrix)
        constructor = DistanceTreeConstructor()
        tree = constructor.upgma(distance_matrix)
        return tree
    except Exception as e:
        st.error(f"Tree construction failed: {e}")
        return None


def compute_distance(seq1, seq2, seq_type):
    """
    Compute a simple distance between two sequences.

    Parameters:
        seq1: First sequence
        seq2: Second sequence
        seq_type (str): Type of sequences ('DNA' or 'Protein')

    Returns:
        float: Distance value between 0 and 1
    """
    # Use standard gap scores for distance computation
    aligner = get_aligner(seq_type, mode='global', open_gap_score=-0.5, extend_gap_score=-0.1)

    alignments = aligner.align(seq1, seq2)
    if not alignments:
        return 1.0

    alignment = next(alignments)
    aligned_seq1 = str(alignment[0])
    aligned_seq2 = str(alignment[1])

    # Count differences (excluding positions with gaps)
    differences = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != b and a != '-' and b != '-')

    # Count alignment positions without gaps in both sequences
    valid_positions = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != '-' and b != '-')

    # Calculate distance as proportion of differences
    if valid_positions == 0:
        return 1.0

    distance = differences / valid_positions
    return distance
