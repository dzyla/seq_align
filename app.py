import streamlit as st
from Bio import AlignIO, SeqIO, Phylo, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices, MultipleSeqAlignment, AlignInfo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix, DistanceCalculator
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder
from io import StringIO, BytesIO
import traceback
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt # Retained for MSA matplotlib plot, not tree
from matplotlib.colors import ListedColormap # Retained for MSA matplotlib plot
import matplotlib.patches as mpatches # Retained for MSA matplotlib plot
import os
import pandas as pd
import tempfile
import random

# Import pyFAMSA
from pyfamsa import Aligner as PyFAMSAAligner, Sequence as PyFAMSASequence

# --- Helper function for manual tree layout ---
def _get_node_layout(tree: Phylo.BaseTree.Tree):
    """
    Calculates x and y coordinates for each node in the tree for manual Plotly plotting.
    Returns:
        node_positions (dict): {clade: (x, y)}
        max_x (float): Maximum x coordinate, useful for label positioning and axis scaling.
    """
    node_positions = {}
    
    # Assign initial y positions to terminal nodes (tips)
    y_current = 0
    tip_spacing = 10 
    
    processed_tips_for_y = set()
    for tip in tree.get_terminals(order='preorder'): # Ensures consistent Y ordering for tips
        if tip not in processed_tips_for_y:
            # Store y directly on the clade object temporarily for easy access during parent calculation
            tip.y_coord_temp_manual = y_current 
            processed_tips_for_y.add(tip)
            y_current += tip_spacing

    max_x = 0.0
    
    # --- Build a parent map first ---
    # This allows us to look up the parent of any clade efficiently.
    parents = {child: parent for parent in tree.find_clades(order='preorder') for child in parent.clades}

    # Calculate y for internal nodes using a postorder traversal.
    # This ensures that children's y-coordinates (which might be tips or other internal nodes)
    # are already computed and stored on the temporary attribute.
    for node in tree.find_clades(order='postorder'):
        if node.is_terminal():
            # y_coord_temp_manual was already set for tips.
            pass 
        else:
            # For internal nodes, y is the average of its children's y-coordinates.
            child_ys = [getattr(child, 'y_coord_temp_manual', 0) for child in node.clades]
            if child_ys:
                node.y_coord_temp_manual = sum(child_ys) / len(child_ys)
            else: 
                # This case should ideally not be reached for a valid tree structure if tips are handled.
                # If a non-terminal node has no children with y_coord_temp_manual, default its y.
                node.y_coord_temp_manual = 0 

    # Calculate final x and y coordinates, and populate node_positions using a preorder traversal.
    # Preorder ensures that a parent's x-coordinate is known when its children are processed.
    for node in tree.find_clades(order='preorder'):
        parent_x = 0
        node_parent = parents.get(node) # Use the precomputed parent map

        if node_parent and node_parent in node_positions: 
            parent_x = node_positions[node_parent][0]
        
        # Ensure branch_length is not None; use a small default if it is for visualization.
        branch_length = node.branch_length if node.branch_length is not None else 0.01
        current_x = parent_x + branch_length
        
        if current_x > max_x:
            max_x = current_x
            
        # Get the y-coordinate that was computed (either for tips or averaged for internal nodes).
        current_y = getattr(node, 'y_coord_temp_manual', 0) 
        node_positions[node] = (current_x, current_y)

    # Clean up the temporary y-coordinate attribute from all clades.
    for node in tree.find_clades():
        if hasattr(node, 'y_coord_temp_manual'):
            delattr(node, 'y_coord_temp_manual')
            
    return node_positions, max_x

# --- Phylogenetic Tree Plotting Function (Manual Layout) ---
def plot_phylogenetic_tree_plotly(tree: Phylo.BaseTree.Tree):
    """ Plots a Biopython tree using manually calculated layout and Plotly for rendering. """
    if not tree or not tree.root:
        st.warning("Tree object is empty or invalid.")
        return None

    num_terminals = tree.count_terminals()
    if num_terminals == 0:
        st.warning("Tree has no terminal nodes (sequences) to plot.")
        return None
    
    # Handle single sequence tree (single terminal node)
    if num_terminals == 1 and (not tree.root.clades or len(tree.root.clades) == 0):
        term_node = list(tree.get_terminals())[0]
        fig_single = go.Figure(data=[go.Scatter(x=[0], y=[0], mode='markers+text',
                                                text=[term_node.name or "Sequence"],
                                                textposition="bottom center", marker_size=10)])
        fig_single.update_layout(title=f"Tree: {term_node.name or 'Sequence'}", height=200, plot_bgcolor='white',
                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False))
        return fig_single
    
    if num_terminals < 2: # Should be caught by single terminal, but defensive
        st.warning(f"Tree has only {num_terminals} terminal(s). Cannot draw a meaningful phylogenetic graph.")
        return None

    node_positions, max_x_coord = _get_node_layout(tree)
    
    # Build parent map again for drawing, or pass it from _get_node_layout if preferred
    parents_for_drawing = {child: parent for parent in tree.find_clades(order='preorder') for child in parent.clades}


    edge_x_coords, edge_y_coords = [], []
    node_x_plot, node_y_plot, node_plot_labels, node_hover_texts = [], [], [], []

    for clade, (x_clade, y_clade) in node_positions.items():
        node_x_plot.append(x_clade)
        node_y_plot.append(y_clade)
        
        label_on_plot = ""
        hover_text = str(clade.name) if clade.name else ("Internal Node" if not clade.is_terminal() else "Tip")
        if clade.is_terminal() and clade.name:
            label_on_plot = str(clade.name)
        
        if hasattr(clade, 'confidence') and clade.confidence is not None:
            hover_text += f" (Conf: {clade.confidence:.2f})" if isinstance(clade.confidence, float) else f" (Conf: {clade.confidence})"

        node_plot_labels.append(label_on_plot)
        node_hover_texts.append(hover_text)

        # Draw lines to parent for rectangular phylogram
        clade_parent_for_drawing = parents_for_drawing.get(clade)
        if clade_parent_for_drawing and clade_parent_for_drawing in node_positions:
            x_parent, y_parent = node_positions[clade_parent_for_drawing]
            
            # 1. Horizontal line for the branch: from (x_parent, y_clade) to (x_clade, y_clade)
            edge_x_coords.extend([x_parent, x_clade, None])
            edge_y_coords.extend([y_clade, y_clade, None])
            
            # 2. Vertical line connecting parent's y-level to this branch's y-level, at parent's x:
            #    from (x_parent, y_parent) to (x_parent, y_clade)
            edge_x_coords.extend([x_parent, x_parent, None])
            edge_y_coords.extend([y_parent, y_clade, None])

    fig = go.Figure()

    # Add Edges (Branches)
    fig.add_trace(go.Scatter(
        x=edge_x_coords, y=edge_y_coords,
        mode='lines',
        line=dict(color='slategray', width=1.5), # CORRECTED COLOR
        hoverinfo='none'
    ))

    # Add Nodes and Tip Labels
    node_colors = ['orange' if clade.is_terminal() else 'lightblue' for clade in node_positions.keys()]
    
    show_text_on_nodes = num_terminals <= 50 

    fig.add_trace(go.Scatter(
        x=node_x_plot, y=node_y_plot,
        mode='markers' + ('+text' if show_text_on_nodes else ''),
        text=[lbl if show_text_on_nodes else '' for lbl in node_plot_labels], 
        textposition="middle right", 
        textfont=dict(size=9),
        marker=dict(size=6, color=node_colors, line=dict(width=0.5, color='darkslategrey')),
        hovertext=node_hover_texts,
        hoverinfo='text'
    ))
    
    y_coords_all = [pos[1] for pos in node_positions.values()]
    y_min_plot = min(y_coords_all) - 10 if y_coords_all else -10
    y_max_plot = max(y_coords_all) + 10 if y_coords_all else 10
    
    effective_max_x = max(max_x_coord, 0.1)


    fig.update_layout(
        title_text="Phylogenetic Tree",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=150, t=40), 
        xaxis=dict(
            showgrid=False, zeroline=True, showticklabels=True,
            title="Branch Length (Distance)",
            range=[-effective_max_x * 0.05, effective_max_x * 1.25] 
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[y_min_plot, y_max_plot]
        ),
        height=max(400, num_terminals * 20 + 60), 
        plot_bgcolor='white'
    )
    return fig

# --- [Existing code from your snippet: msa_to_image, plot_msa_image, etc. up to init_session_state] ---
def msa_to_image(alignment_text: str, format_str: str) -> tuple: # Renamed format to format_str
    """
    Converts Multiple Sequence Alignment (MSA) to numerical image data and amino acid array.

    Parameters:
        alignment_text (str): The MSA text in the specified format
        format_str (str): The format of the MSA text (e.g., 'fasta', 'clustal')

    Returns:
        tuple: (msa_image, msa_letters) - numerical representation and letter representation
    """
    try:
        alignment = AlignIO.read(StringIO(alignment_text), format_str)
    except Exception as e:
        st.error(f"An error occurred while parsing the MSA alignment: {e}")
        raise e

    AA_CODES = {
        "-": 0, "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
        "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
        "X": 21, "B": 22, "J": 23, "O": 24, "Z": 25,
    }

    msa_image = np.zeros((len(alignment), alignment.get_alignment_length()), dtype=int)
    msa_letters = np.empty((len(alignment), alignment.get_alignment_length()), dtype=object)

    for i, record in enumerate(alignment):
        for j, aa in enumerate(str(record.seq)):
            code = AA_CODES.get(aa.upper(), 0)
            msa_image[i, j] = code
            msa_letters[i, j] = aa.upper()

    return msa_image, msa_letters


def plot_msa_image(msa_image: np.ndarray, msa_letters: np.ndarray, plot_method: str):
    """
    Plots the Multiple Sequence Alignment (MSA) as a heatmap.

    Parameters:
        msa_image (np.ndarray): Numerical representation of the MSA
        msa_letters (np.ndarray): Letter representation of the MSA
        plot_method (str): The plotting method ('Plotly (Interactive)' or 'Matplotlib (Static)')
    """
    if msa_image is None or msa_letters is None:
        st.error("No MSA image data to plot.")
        return

    if msa_image.shape != msa_letters.shape:
        st.error("Mismatch between msa_image and msa_letters dimensions.")
        return

    if plot_method == "Plotly (Interactive)":
        plot_msa_image_plotly(msa_image, msa_letters)
    elif plot_method == "Matplotlib (Static)":
        plot_msa_image_matplotlib(msa_image, msa_letters)
    else:
        st.error("Unsupported plotting method selected.")


def plot_msa_image_plotly(msa_image: np.ndarray, msa_letters: np.ndarray):
    """
    Plots the MSA as an interactive heatmap using Plotly.

    Parameters:
        msa_image (np.ndarray): Numerical representation of the MSA
        msa_letters (np.ndarray): Letter representation of the MSA
    """
    msa_image_list = msa_image.tolist()
    msa_letters_list = msa_letters.tolist()

    hover_text = [
        [f"Position: {x}<br>Sequence: {y}<br>Amino Acid: {aa}"
         for x, aa in enumerate(row, start=1)]
        for y, row in enumerate(msa_letters_list, start=1)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=msa_image_list,
            text=hover_text,
            hoverinfo="text",
            colorscale="Spectral",
            showscale=False
        )
    )

    fig.update_traces(hovertemplate="%{text}<extra></extra>")
    fig.update_layout(
        title="Multiple Sequence Alignment Heatmap (Plotly)",
        xaxis_title="MSA Residue Position",
        yaxis_title="Sequence Number",
        xaxis=dict(ticks='', showticklabels=True),
        yaxis=dict(ticks='', showticklabels=True),
        plot_bgcolor='white',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_msa_image_matplotlib(msa_image: np.ndarray, msa_letters: np.ndarray):
    """
    Plots the MSA as a static heatmap using Matplotlib.

    Parameters:
        msa_image (np.ndarray): Numerical representation of the MSA
        msa_letters (np.ndarray): Letter representation of the MSA
    """
    _plot_msa_image_matplotlib_subset(msa_image, msa_letters, 0, 0)


CODE_TO_AA = {
    0: "-", 1: "A", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "K", 10: "L",
    11: "M", 12: "N", 13: "P", 14: "Q", 15: "R", 16: "S", 17: "T", 18: "V", 19: "W", 20: "Y",
    21: "X", 22: "B", 23: "J", 24: "O", 25: "Z",
}


def _plot_msa_image_matplotlib_subset(msa_image_subset: np.ndarray, msa_letters_subset: np.ndarray, seq_offset: int, res_offset: int):
    """
    Helper function to plot a subset of the MSA using Matplotlib.

    Parameters:
        msa_image_subset (np.ndarray): Subset of the numerical MSA representation
        msa_letters_subset (np.ndarray): Subset of the letter MSA representation
        seq_offset (int): Sequence offset for labeling
        res_offset (int): Residue offset for labeling
    """
    fig, ax = plt.subplots(figsize=(20, 7))
    cax = ax.imshow(msa_image_subset, cmap='Spectral', aspect='auto', interpolation='nearest')
    cmap = plt.get_cmap('Spectral', 26)
    ax.set_title("Multiple Sequence Alignment", fontsize=14)
    ax.set_xlabel("MSA Residue Position", fontsize=12)
    ax.set_ylabel("Sequence Number", fontsize=12)

    unique_values = np.unique(msa_image_subset)
    handles = []
    labels = []
    for val in unique_values:
        label = CODE_TO_AA.get(val, "?")
        handles.append(mpatches.Patch(color=cmap(val), label=label))
        labels.append(label)

    ax.legend(handles=handles, labels=labels, title="Amino Acids", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    buf.seek(0)
    img = buf.getvalue()
    st.image(img, caption="MSA Heatmap", use_container_width=True)


def init_session_state():
    """
    Initialize session state variables if they don't exist.
    """
    # MSA-related variables
    if 'msa_result' not in st.session_state:
        st.session_state.msa_result = None
    if 'mutations' not in st.session_state:
        st.session_state.mutations = None
    if 'msa_image' not in st.session_state:
        st.session_state.msa_image = None
    if 'msa_letters' not in st.session_state:
        st.session_state.msa_letters = None
    if 'consensus_data' not in st.session_state:
        st.session_state.consensus_data = None
    if 'alignment_text' not in st.session_state: # Used by pairwise
        st.session_state.alignment_text = None
    if 'pairwise_mutations' not in st.session_state:
        st.session_state.pairwise_mutations = None
    if 'phylogenetic_tree_from_msa' not in st.session_state: # For MSA tree
        st.session_state.phylogenetic_tree_from_msa = None


    # Input-related variables
    if 'sequences' not in st.session_state:
        st.session_state.sequences = None
    if 'seq_type' not in st.session_state:
        st.session_state.seq_type = None
    if 'last_file' not in st.session_state:
        st.session_state.last_file = None
    if 'last_tree_file' not in st.session_state: # For Newick upload
        st.session_state.last_tree_file = None
    if 'tree' not in st.session_state: # For Newick upload
        st.session_state.tree = None
    if 'tree_ascii' not in st.session_state: # For Newick upload display
        st.session_state.tree_ascii = None
    if 'tree_newick' not in st.session_state: # For Newick upload download
        st.session_state.tree_newick = None

    # Format conversion variables
    if 'converted_data' not in st.session_state:
        st.session_state.converted_data = None
    if 'conversion_error' not in st.session_state:
        st.session_state.conversion_error = None
    if 'last_conversion_params' not in st.session_state:
        st.session_state.last_conversion_params = {}

    # Pairwise alignment variables
    # 'selected_seqs' removed as direct selectbox values are used

    # MSA calculation state tracking
    if 'last_msa_params' not in st.session_state:
        st.session_state.last_msa_params = {}

    # Pairwise alignment parameters tracking
    if 'last_pairwise_params' not in st.session_state:
        st.session_state.last_pairwise_params = {}

    # Text input tracking
    if 'last_text_hash' not in st.session_state:
        st.session_state.last_text_hash = None

def calculate_representative_sequence(alignment, threshold=0.7):
    """
    Calculate the consensus sequence and find the most representative sequence in the alignment.

    Parameters:
        alignment: Biopython Alignment object
        threshold (float): Consensus threshold (default: 0.7)

    Returns:
        tuple: (consensus_record, closest_record, min_differences, closest_seq_id)
    """
    if not alignment or len(alignment) == 0: # Handle empty or invalid alignment
        empty_seq = Seq("")
        return (SeqRecord(empty_seq, id="Consensus", description="Consensus sequence"),
                SeqRecord(empty_seq, id="N/A", description="Most representative sequence"),
                0, "N/A")

    summary = AlignInfo.SummaryInfo(alignment)
    consensus = summary.dumb_consensus(threshold=threshold, ambiguous='X')
    consensus_seq_str = str(consensus)

    def count_differences(seq1_str, seq2_str):
        return sum(1 for a, b in zip(seq1_str, seq2_str) if a != b)

    min_differences = float('inf')
    closest_sequence_str = ""
    closest_seq_id = alignment[0].id # Default to first sequence if no better one is found

    for record in alignment:
        record_seq_str = str(record.seq)
        differences = count_differences(consensus_seq_str, record_seq_str)
        if differences < min_differences:
            min_differences = differences
            closest_sequence_str = record_seq_str
            closest_seq_id = record.id
    
    if not closest_sequence_str and alignment : # Fallback if all were identical to consensus (min_diff=0 but vars not set)
        closest_sequence_str = str(alignment[0].seq)


    consensus_record = SeqRecord(Seq(consensus_seq_str), id="Consensus", description="Consensus sequence")
    closest_record = SeqRecord(Seq(closest_sequence_str), id=closest_seq_id, description="Most representative sequence")

    return consensus_record, closest_record, min_differences, closest_seq_id

def main():
    st.set_page_config(page_title="Advanced Sequence Alignment Ability", layout="wide")
    init_session_state()

    st.title("🔬 Advanced Sequence Alignment and Format Conversion Ability")
    st.write("""
    ### Welcome to the Advanced Sequence Alignment Tool!
    This application allows you to perform **pairwise** and **multiple sequence alignments (MSA)** on **DNA** or **Protein** sequences.
    You can choose between **Global**, **Local**, and **Overlap** alignments. Additionally, you can convert sequence files between various formats.
    For MSA, point mutations relative to a reference sequence are reported, and a phylogenetic tree can be generated.
    **How to Use:**
    1. **Upload your sequences** either by uploading a file or pasting them directly.
    2. **Select your preferences** from the sidebar.
    3. **Run** the desired analysis.
    4. **Download** your results!
    """)

    st.sidebar.header("📥 Upload and Settings")
    input_format = st.sidebar.selectbox(
        "Select Input Format",
        ("Text (FASTA)", "FASTA", "Clustal", "Phylip", "EMBL", "GenBank", "Newick", "PDB", "mmCIF"),
        help="Choose the format of your input sequences."
    )

    sequences = None
    tree_uploaded = None # For uploaded Newick tree
    uploaded_file = None

    if input_format == "Text (FASTA)":
        seq_text = st.sidebar.text_area(
            "Paste Sequences Here (FASTA Format)", height=250,
            placeholder=">Sequence1\nATGCGTA...\n>Sequence2\nATGCGTC...",
            help="Enter your sequences in FASTA format"
        )
        if seq_text:
            current_hash = hash(seq_text)
            if st.session_state.get('last_text_hash') != current_hash:
                st.session_state.last_text_hash = current_hash
                sequences, error = parse_sequences_from_text(seq_text)
                if error: st.sidebar.error(error)
                else:
                    st.sidebar.success(f"Successfully loaded {len(sequences)} sequences.")
                    st.session_state.sequences = sequences
            sequences = st.session_state.get('sequences')
    elif input_format in ["PDB", "mmCIF"]:
        uploaded_file = st.sidebar.file_uploader("Upload PDB/mmCIF File", type=get_file_extensions(input_format), key=f"uploader_{input_format}")
        if uploaded_file:
            if st.session_state.get('last_file') != uploaded_file.name:
                st.session_state.last_file = uploaded_file.name
                sequences, error = parse_sequences_from_structure(uploaded_file, input_format)
                if error: st.sidebar.error(error)
                else:
                    st.sidebar.success(f"Extracted {len(sequences)} sequences.")
                    st.session_state.sequences = sequences
            sequences = st.session_state.get('sequences')
    elif input_format == "Newick":
        uploaded_file = st.sidebar.file_uploader("Upload Newick Tree File", type=["nwk", "newick", "tree", "tre"], key="uploader_newick")
        if uploaded_file:
            if st.session_state.get('last_tree_file') != uploaded_file.name:
                st.session_state.last_tree_file = uploaded_file.name
                try:
                    tree_uploaded = Phylo.read(uploaded_file, "newick")
                    st.session_state.tree = tree_uploaded # Store uploaded tree
                    st.session_state.tree_ascii = None # Reset cache
                    st.session_state.tree_newick = None
                    st.sidebar.success("Newick tree loaded.")
                except Exception as e:
                    st.sidebar.error(f"Error reading Newick: {e}")
            tree_uploaded = st.session_state.get('tree')
    else: # Other sequence file formats
        uploaded_file = st.sidebar.file_uploader("Upload Sequence File", type=get_file_extensions(input_format), key=f"uploader_{input_format}")
        if uploaded_file:
            if st.session_state.get('last_file') != uploaded_file.name:
                st.session_state.last_file = uploaded_file.name
                sequences, error = parse_sequences_from_file(uploaded_file, input_format)
                if error: st.sidebar.error(error)
                else:
                    st.sidebar.success(f"Loaded {len(sequences)} sequences.")
                    st.session_state.sequences = sequences
            sequences = st.session_state.get('sequences')
    
    # Fallback to session state if needed
    if sequences is None and 'sequences' in st.session_state: sequences = st.session_state.sequences
    if tree_uploaded is None and 'tree' in st.session_state and input_format == "Newick": tree_uploaded = st.session_state.tree


    # --- Main application logic based on loaded data ---
    if (sequences and len(sequences) > 0) or (tree_uploaded and input_format == "Newick"):
        current_seq_type = st.session_state.get('seq_type', "Protein") # Default to protein
        seq_type_idx = 0 if current_seq_type == "DNA" else 1

        if sequences: # Common settings if sequences are loaded
            selected_seq_type = st.sidebar.selectbox(
                "🔬 Select Sequence Type", ("DNA", "Protein"), index=seq_type_idx,
                help="Choose DNA or Protein. This affects alignment scoring and analysis."
            )
            if selected_seq_type != current_seq_type:
                st.session_state.seq_type = selected_seq_type
                # Invalidate results that depend on seq_type
                st.session_state.msa_result = None
                st.session_state.mutations = None
                st.session_state.phylogenetic_tree_from_msa = None
                st.session_state.alignment_text = None # Pairwise
                st.session_state.last_msa_params = {}
                st.session_state.last_pairwise_params = {}
        
        final_seq_type = st.session_state.get('seq_type', "Protein")

        # Determine main mode of operation
        analysis_mode = None
        if sequences and input_format not in ["PDB", "mmCIF"]:
            analysis_mode = st.sidebar.selectbox(
                "🛠️ Select Analysis Mode", ("Pairwise", "MSA", "Convert Formats"),
                help="Choose the type of analysis."
            )
        elif sequences and input_format in ["PDB", "mmCIF"]: # PDB/mmCIF shows extracted FASTA
            st.header("📄 Extracted FASTA Sequences")
            fasta_io = StringIO()
            SeqIO.write(sequences, fasta_io, "fasta")
            fasta_str = fasta_io.getvalue()
            st.text_area("FASTA Sequences", value=fasta_str, height=300)
            if uploaded_file: # Check if uploaded_file is not None
                 file_basename = os.path.splitext(os.path.basename(uploaded_file.name))[0]
                 st.download_button(
                    label="📥 Download FASTA", data=fasta_str,
                    file_name=f"{file_basename}_sequences.fasta", mime="text/plain"
                )
        elif tree_uploaded and input_format == "Newick":
            analysis_mode = "Phylogenetic Tree (Uploaded)"
        
        # Execute based on analysis_mode
        if analysis_mode == "Pairwise" and sequences:
            pairwise_alignment_section(sequences, final_seq_type)
        elif analysis_mode == "MSA" and sequences:
            msa_section(sequences, final_seq_type)
        elif analysis_mode == "Convert Formats" and sequences:
            format_conversion_section(sequences, input_format)
        elif analysis_mode == "Phylogenetic Tree (Uploaded)" and tree_uploaded:
            phylogenetic_tree_section(tree_uploaded)
            
    else:
        st.info("👋 Please upload sequences or a Newick tree file in the sidebar to begin your analysis.")

amino_acid_map = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XLE": "J", "XAA": "X", # XAA for unknown
    "MSE": "M", "UNK": "X", "MLE": "L", "CSD": "C", "HYP": "P",
    "KCX": "K", "CSO": "C", "TPO": "T", "SEP": "S", "MLY": "K",
    "M3L": "K", "OCS": "C", "PTR": "Y", "PCA": "E", "SAC": "S", "MLZ": "K"
}

def get_file_extensions(format_name):
    format_extensions = {
        "FASTA": ["fasta", "fa", "fna", "ffn", "faa", "frn", "fsa", "seq"],
        "Clustal": ["clustal", "aln", "clw"],
        "Phylip": ["phy", "ph", "phylip"],
        "EMBL": ["embl", "ebl", "emb"],
        "GenBank": ["gb", "gbk", "genbank"],
        "Newick": ["nwk", "newick", "tree", "tre"], # Added tre
        "PDB": ["pdb", "ent"],
        "mmCIF": ["cif", "mmcif", "mcif"]
    }
    return format_extensions.get(format_name, [])

def parse_sequences_from_text(text):
    try:
        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [line.strip() for line in normalized_text.split('\n') if line.strip()] # Remove empty lines
        cleaned_text = '\n'.join(lines)
        if not cleaned_text.startswith('>'):
            return None, "FASTA format error: Must start with '>'. Please check your input."
        seq_io = StringIO(cleaned_text)
        sequences = list(SeqIO.parse(seq_io, "fasta"))
        if not sequences:
            return None, "No valid FASTA sequences found. Please check sequence content."
        
        sequence_ids = set()
        for seq_rec in sequences:
            if not seq_rec.id: return None, "Found a sequence without an ID. All sequences must have an ID."
            if seq_rec.id in sequence_ids:
                return None, f"Duplicate sequence ID: '{seq_rec.id}'. IDs must be unique."
            sequence_ids.add(seq_rec.id)
            if not str(seq_rec.seq).strip(): # Check for empty sequence string
                return None, f"Sequence '{seq_rec.id}' is empty. Please provide sequence data."
        return sequences, None
    except Exception as e:
        # traceback.print_exc()
        return None, f"Error parsing text input as FASTA: {e}. Ensure correct FASTA format."

def parse_sequences_from_file(file, format_name):
    try:
        file.seek(0) # Reset file pointer
        file_content_str = file.read().decode("utf-8", errors='replace') # Handle potential decoding errors
        seq_io = StringIO(file_content_str)
        
        sequences = list(SeqIO.parse(seq_io, format_name.lower()))
        if not sequences:
            return None, f"No sequences found in the uploaded {format_name} file. Check file content and format."

        sequence_ids = set()
        for seq_rec in sequences:
            if not seq_rec.id: return None, f"Found a sequence without an ID in the {format_name} file."
            if seq_rec.id in sequence_ids:
                return None, f"Duplicate sequence ID '{seq_rec.id}' in {format_name} file. IDs must be unique."
            sequence_ids.add(seq_rec.id)
            if not str(seq_rec.seq).strip():
                 return None, f"Sequence '{seq_rec.id}' in {format_name} file is empty."
        return sequences, None
    except Exception as e:
        # traceback.print_exc()
        return None, f"Error parsing {format_name} file: {e}. Ensure correct format and encoding (UTF-8 recommended)."


def parse_sequences_from_structure(file, format_name):
    try:
        file.seek(0)
        file_content = file.read() # Read as bytes first
        file_basename = os.path.splitext(os.path.basename(file.name))[0] if file.name else "structure"

        # Use temp file for Biopython parsers as they often expect file paths
        temp_suffix = f".{format_name.lower()}"
        with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=temp_suffix) as temp_file_obj: # Write as bytes
            temp_file_obj.write(file_content)
            temp_filepath = temp_file_obj.name
        
        structure = None
        try:
            parser_class = PDBParser if format_name == "PDB" else MMCIFParser
            parser = parser_class(QUIET=True)
            structure = parser.get_structure(file_basename, temp_filepath)
        except Exception as e_parse_struct:
            os.unlink(temp_filepath) # Clean up temp file
            return None, f"Failed to parse {format_name} structure: {e_parse_struct}"

        extracted_sequences = []
        ppb = PPBuilder()
        processed_chain_ids_model_specific = set() # To avoid re-processing same chain in a model

        for model in structure:
            for chain in model:
                # Create a unique ID for this model-chain combination
                model_chain_id = f"model{model.id}_chain{chain.id}"
                if model_chain_id in processed_chain_ids_model_specific:
                    continue # Already processed this chain in this model

                # Attempt 1: Use PPBuilder (generally robust for standard peptides)
                try:
                    peptides = ppb.build_peptides(chain)
                    chain_seq_str = "".join(str(pp.get_sequence()) for pp in peptides)
                    if chain_seq_str:
                        seq_id = f"{file_basename}_M{model.id}_C{chain.id}_ppb"
                        desc = f"Model {model.id}, Chain {chain.id} (PPBuilder) from {file.name or 'uploaded file'}"
                        extracted_sequences.append(SeqRecord(Seq(chain_seq_str), id=seq_id, description=desc))
                        processed_chain_ids_model_specific.add(model_chain_id)
                        continue # Successfully extracted, move to next chain
                except Exception: # Catch errors from PPBuilder for specific chains
                    pass # Will try manual extraction next

                # Attempt 2: Manual residue extraction (fallback)
                manual_seq_list = []
                for residue in chain:
                    res_name = residue.get_resname().strip().upper()
                    # Filter for standard amino acids and known modified ones mapped in amino_acid_map
                    # Also ensure it's an amino acid residue (e.g., skip HETATM like HOH, LIG)
                    if residue.id[0] == ' ' and res_name in amino_acid_map: # ' ' indicates standard residue record
                        manual_seq_list.append(amino_acid_map[res_name])
                    elif residue.id[0] == ' ' and len(res_name) == 3 and res_name.isalpha(): # Unknown 3-letter AA code
                        manual_seq_list.append('X')


                if len(manual_seq_list) >= 5: # Heuristic: only consider if at least 5 AAs found
                    manual_seq_str = "".join(manual_seq_list)
                    seq_id = f"{file_basename}_M{model.id}_C{chain.id}_manual"
                    desc = f"Model {model.id}, Chain {chain.id} (Manual) from {file.name or 'uploaded file'}"
                    extracted_sequences.append(SeqRecord(Seq(manual_seq_str), id=seq_id, description=desc))
                    processed_chain_ids_model_specific.add(model_chain_id)
        
        os.unlink(temp_filepath) # Clean up temp file

        if not extracted_sequences:
            return None, f"No protein sequences could be extracted from the {format_name} file. Ensure it contains protein chains."
        
        # Ensure unique IDs across all extracted sequences before returning
        final_sequences = []
        final_ids = set()
        for i, seq_rec in enumerate(extracted_sequences):
            original_id = seq_rec.id
            counter = 1
            while seq_rec.id in final_ids: # If ID conflict (e.g. from different methods for same chain name)
                seq_rec.id = f"{original_id}_{counter}"
                counter += 1
            final_ids.add(seq_rec.id)
            final_sequences.append(seq_rec)

        return final_sequences, None
    except Exception as e:
        # traceback.print_exc() # For developer debugging
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.unlink(temp_filepath)
        return None, f"An error occurred while processing the {format_name} file: {e}"


def pairwise_alignment_section(sequences, seq_type):
    st.header("🔀 Pairwise Alignment")
    if len(sequences) < 2:
        st.warning("Please upload at least two sequences for pairwise alignment.")
        return

    align_mode_pairwise = st.selectbox(
        "Select Alignment Mode", ("Global", "Local", "Overlap"),
        help="Global: Align entire sequences. Local: Find best local regions. Overlap: Allow end gaps.",
        key="pairwise_align_mode"
    )
    seq_names = [seq.id for seq in sequences]
    col1_pw, col2_pw = st.columns(2)
    
    default_ref_idx = 0
    if 'reference_id_pw' in st.session_state and st.session_state.reference_id_pw in seq_names:
        default_ref_idx = seq_names.index(st.session_state.reference_id_pw)
    
    with col1_pw:
        reference_id_pw = st.selectbox("Select Reference Sequence", seq_names, index=default_ref_idx, key="reference_seq_select_pw")
    
    # Ensure target is different from reference
    available_targets = [name for name in seq_names if name != reference_id_pw]
    if not available_targets:
        st.warning("Only one unique sequence ID available. Cannot perform pairwise alignment.")
        return
        
    default_target_idx_in_available = 0
    if 'target_id_pw' in st.session_state and st.session_state.target_id_pw in available_targets:
         default_target_idx_in_available = available_targets.index(st.session_state.target_id_pw)

    with col2_pw:
        target_id_pw = st.selectbox("Select Target Sequence", available_targets, index=default_target_idx_in_available, key="target_seq_select_pw")

    st.session_state.reference_id_pw = reference_id_pw # Store for persistence
    st.session_state.target_id_pw = target_id_pw

    selection_changed_pw = (st.session_state.get('last_selected_seqs_pw') != (reference_id_pw, target_id_pw))
    if selection_changed_pw: # Reset results if selection changes
        st.session_state.alignment_text = None 
        st.session_state.pairwise_mutations = None
        st.session_state.last_pairwise_params = {} # Force re-calc
        st.session_state.last_selected_seqs_pw = (reference_id_pw, target_id_pw)

    c1_gap, c2_gap = st.columns(2)
    open_gap_score = c1_gap.number_input("Open Gap Score", value=-0.5, step=0.1, help="Penalty for opening a gap.")
    extend_gap_score = c2_gap.number_input("Extend Gap Score", value=-0.1, step=0.1, help="Penalty for extending a gap.")

    seq1_pw = next((s for s in sequences if s.id == reference_id_pw), None)
    seq2_pw = next((s for s in sequences if s.id == target_id_pw), None)

    if not seq1_pw or not seq2_pw: # Should not happen if selectbox logic is correct
        st.error("Selected sequence(s) not found. Please re-select.")
        return

    current_params_pw = {
        'ref_id': reference_id_pw, 'target_id': target_id_pw, 'seq_type': seq_type,
        'align_mode': align_mode_pairwise.lower(), 'open_gap': open_gap_score, 'extend_gap': extend_gap_score
    }
    params_changed_pw = (st.session_state.get('last_pairwise_params') != current_params_pw)

    if st.button("Run Pairwise Alignment", key="run_pairwise_btn"):
        if params_changed_pw or not st.session_state.get('alignment_text'): # Recalculate if params changed or no result
            with st.spinner("Aligning sequences..."):
                alignment_text_res, mutations_res = perform_pairwise_alignment(
                    seq1_pw, seq2_pw, seq_type, align_mode_pairwise.lower(), open_gap_score, extend_gap_score
                )
                st.session_state.alignment_text = alignment_text_res
                st.session_state.pairwise_mutations = mutations_res
                st.session_state.last_pairwise_params = current_params_pw
    
    if st.session_state.get('alignment_text'):
        st.subheader("🧬 Alignment Result")
        st.code(st.session_state.alignment_text)
        st.subheader("🔍 Point Mutations Relative to Reference")
        if st.session_state.get('pairwise_mutations'):
            st.write(f"**Reference:** `{reference_id_pw}`, **Target:** `{target_id_pw}`")
            mutations_str = ', '.join(st.session_state.pairwise_mutations)
            st.write(f"**Mutations:** {mutations_str if mutations_str else 'None / Identical in aligned regions'}")
            
            mutation_positions = [int(''.join(filter(str.isdigit, m))) for m in st.session_state.pairwise_mutations if ''.join(filter(str.isdigit,m))]
            if mutation_positions:
                fig_pw_mut = go.Figure(go.Scatter(
                    x=mutation_positions, y=[1]*len(mutation_positions), mode='markers',
                    marker=dict(size=10, color='red'), text=st.session_state.pairwise_mutations, hoverinfo='text'
                ))
                fig_pw_mut.update_layout(title="Point Mutations Plot", xaxis_title="Position in Reference", yaxis=dict(showticklabels=False,showgrid=False,zeroline=False), height=200)
                st.plotly_chart(fig_pw_mut, use_container_width=True)
        else:
            st.write("No point mutations detected or alignment/mutation analysis failed.")
        st.download_button("📥 Download Alignment", st.session_state.alignment_text, f"pairwise_{reference_id_pw}_vs_{target_id_pw}.txt", key="download_pairwise")


def msa_section(sequences, seq_type):
    st.header("📏 Multiple Sequence Alignment (MSA)")
    st.info("Align multiple sequences to find conserved regions and evolutionary relationships. A phylogenetic tree can also be generated.")

    if len(sequences) < 2: 
        st.warning("Please upload at least two sequences for MSA.")
        return

    ref_seq_id_msa = st.selectbox(
        "Select Reference Sequence for Mutation Reporting",
        [seq.id for seq in sequences], key="ref_seq_msa_select",
        help="Mutations will be reported relative to this sequence."
    )
    msa_output_format = st.selectbox(
        "Select MSA Output Format", ("fasta", "clustal", "phylip", "stockholm"),
        help="Format for MSA results and download."
    )
    plot_method_msa = st.selectbox(
        "Select MSA Heatmap Plotting Method", ("Plotly (Interactive)", "Matplotlib (Static)"),
        help="Choose how the MSA heatmap is displayed."
    )
    
    generate_tree_checkbox = st.checkbox("Generate Phylogenetic Tree from MSA", value=False, key="gen_tree_cb_msa")
    tree_build_method_selected = 'nj' 
    if generate_tree_checkbox:
        tree_build_method_selected = st.selectbox("Tree Construction Method (for generated tree)", 
                                                  ["nj", "upgma"], index=0, key="msa_tree_method_select")

    calculate_representative = st.checkbox("Calculate Most Representative Sequence (from MSA)", value=False)
    consensus_threshold = 0.7
    if calculate_representative:
        consensus_threshold = st.slider("Consensus Threshold for Representative Seq", 0.5, 1.0, 0.7, 0.05)

    current_msa_params = {
        'seq_ids': tuple(s.id for s in sequences), 'ref_id': ref_seq_id_msa, 'seq_type': seq_type,
        'out_fmt': msa_output_format, 'calc_repr': calculate_representative,
        'gen_tree': generate_tree_checkbox, 'tree_method': tree_build_method_selected if generate_tree_checkbox else None
    }
    if calculate_representative: current_msa_params['cons_thresh'] = consensus_threshold
    
    msa_params_changed = (st.session_state.get('last_msa_params') != current_msa_params)

    if st.button("Run MSA Analysis", key="run_msa_btn"):
        if msa_params_changed or not st.session_state.get('msa_result'):
            with st.spinner("Performing MSA... This may take a moment."):
                msa_result_text, mutations_dict = perform_msa(sequences, ref_seq_id_msa, seq_type, msa_output_format)
                st.session_state.msa_result = msa_result_text
                st.session_state.mutations = mutations_dict
                st.session_state.last_msa_params = current_msa_params
                st.session_state.consensus_data = None 
                st.session_state.phylogenetic_tree_from_msa = None 

                if msa_result_text:
                    try:
                        img, letters = msa_to_image(msa_result_text, msa_output_format)
                        st.session_state.msa_image = img
                        st.session_state.msa_letters = letters
                    except Exception as e_img:
                        st.error(f"Failed to generate MSA heatmap data: {e_img}")
                        st.session_state.msa_image, st.session_state.msa_letters = None, None
    
    if st.session_state.get('msa_result'):
        st.subheader("📄 MSA Result")
        try:
            # Use the selected output format for parsing for stats
            current_msa_output_format_for_stats = st.session_state.get('last_msa_params',{}).get('out_fmt','fasta')
            parsed_msa_for_stats = list(AlignIO.read(StringIO(st.session_state.msa_result), current_msa_output_format_for_stats))
            st.write(f"Total sequences in MSA: {len(parsed_msa_for_stats)}. Alignment length: {parsed_msa_for_stats[0].get_alignment_length() if parsed_msa_for_stats else 'N/A'}")
        except Exception as e_parse_stats: 
             st.warning(f"Could not parse MSA for stats (format: {current_msa_output_format_for_stats}): {e_parse_stats}")
             st.write("MSA generated. Displaying heatmap if possible.")

        # FIX APPLIED HERE: Check if NumPy arrays are not None before using them in a boolean context
        if st.session_state.get('msa_image') is not None and st.session_state.get('msa_letters') is not None:
            plot_msa_image(st.session_state.msa_image, st.session_state.msa_letters, plot_method_msa)
        
        st.subheader("🔍 Point Mutations Relative to Reference")
        if st.session_state.get('mutations'):
            seq_ids_with_muts = sorted([k for k,v in st.session_state.mutations.items() if v]) # Only those with mutations
            
            with st.expander("All Sequence Mutations Summary (relative to reference)", expanded=True):
                if st.session_state.mutations: # Check again, as it might be cleared by other actions
                    output_lines = []
                    all_analyzed_seq_ids = st.session_state.mutations.keys()
                    for seq_id_mut in sorted(all_analyzed_seq_ids):
                        mut_list_item = st.session_state.mutations.get(seq_id_mut, []) 
                        if not mut_list_item:
                            formatted_muts_str = "No mutations relative to reference"
                        else:
                            formatted_muts = [f"{ref_aa}{pos}{var_aa}" for pos, ref_aa, var_aa in mut_list_item]
                            formatted_muts_str = ', '.join(formatted_muts)
                        output_lines.append(f"{seq_id_mut}: {formatted_muts_str}")
                    
                    st.text_area(
                        "Formatted Mutations List:", 
                        "\n".join(output_lines), 
                        height=max(100, min(400, len(output_lines) * 20 + 20)),
                        key="all_mutations_summary_text_area_msa"
                    )
                else: st.write("No mutation data available.")

            if seq_ids_with_muts: 
                default_sel_muts = seq_ids_with_muts[:min(3, len(seq_ids_with_muts))] if len(seq_ids_with_muts) <=10 else []
                sel_all_muts_detail = False
                if len(seq_ids_with_muts) > 10:
                    sel_all_muts_detail = st.checkbox("Show all sequences for detailed mutation plots", False, key="sel_all_detail_muts_msa")
                
                sel_seq_ids_detail = []
                if sel_all_muts_detail: sel_seq_ids_detail = seq_ids_with_muts
                elif seq_ids_with_muts: 
                     sel_seq_ids_detail = st.multiselect(
                        "Select sequences for detailed mutation plots:",
                        seq_ids_with_muts, default=default_sel_muts,
                        key="msa_mut_seq_sel_detail"
                    )

                for seq_id_detail in sel_seq_ids_detail:
                    mut_list_detail = st.session_state.mutations.get(seq_id_detail, []) 
                    if mut_list_detail: 
                        df_mut = pd.DataFrame(mut_list_detail, columns=["Position", "Reference", "Variant"])
                        with st.expander(f"Detailed Mutations for {seq_id_detail} ({len(mut_list_detail)} mutations)"):
                            st.dataframe(df_mut)
                            if len(mut_list_detail) <= 300 : 
                                fig_mut_detail = go.Figure(go.Scatter(
                                    x=[pos for pos, _, _ in mut_list_detail], y=[1]*len(mut_list_detail),
                                    mode='markers', marker=dict(size=8, color='red'),
                                    text=[f"{r}{p}{v}" for p, r, v in mut_list_detail], hoverinfo='text'
                                ))
                                fig_mut_detail.update_layout(title=f"Mutations in {seq_id_detail}", xaxis_title="Position in Ref.", 
                                                             yaxis=dict(showticklabels=False,showgrid=False,zeroline=False), height=200, margin=dict(l=20,r=20,t=30,b=20))
                                st.plotly_chart(fig_mut_detail, use_container_width=True)
            elif not st.session_state.get('mutations'): 
                 st.write("Mutation analysis not run or no mutations found.")
            


        # Representative Sequence Calculation
        if st.session_state.get('last_msa_params', {}).get('calc_repr'):
            st.subheader("🧬 Representative Sequence Analysis")
            if st.session_state.get('consensus_data') is None or \
               st.session_state.get('consensus_data', {}).get('threshold') != st.session_state.get('last_msa_params', {}).get('cons_thresh'):
                with st.spinner("Calculating consensus and representative sequence..."):
                    try:
                        current_msa_output_format_repr = st.session_state.get('last_msa_params',{}).get('out_fmt','fasta')
                        alignment_obj_repr = AlignIO.read(StringIO(st.session_state.msa_result), current_msa_output_format_repr)
                        
                        current_consensus_threshold_repr = st.session_state.get('last_msa_params', {}).get('cons_thresh', 0.7)
                        cons_rec, closest_rec, min_diff, closest_id = calculate_representative_sequence(
                            alignment_obj_repr, threshold=current_consensus_threshold_repr
                        )
                        st.session_state.consensus_data = {
                            'threshold': current_consensus_threshold_repr,
                            'consensus_record': cons_rec, 'closest_record': closest_rec,
                            'min_differences': min_diff, 'closest_seq_id': closest_id,
                            'alignment_length': alignment_obj_repr.get_alignment_length(),
                            'seq_count': len(alignment_obj_repr)
                        }
                    except Exception as e_repr:
                        st.error(f"Failed to calculate representative sequence: {e_repr}")
            
            if st.session_state.get('consensus_data'):
                data_repr = st.session_state.consensus_data
                st.write(f"**Consensus threshold:** {data_repr['threshold']:.2f}, **Sequences:** {data_repr['seq_count']}, **Alignment length:** {data_repr['alignment_length']}")
                col1_repr, col2_repr = st.columns(2)
                with col1_repr:
                    st.markdown("###### Consensus Sequence")
                    st.code(f">{data_repr['consensus_record'].id}\n{data_repr['consensus_record'].seq}")
                with col2_repr:
                    st.markdown(f"###### Most Representative Sequence (`{data_repr['closest_seq_id']}`)")
                    st.code(f">{data_repr['closest_record'].id}\n{data_repr['closest_record'].seq}")
                    st.caption(f"Differences from consensus: {data_repr['min_differences']}")
                
                repr_fasta_io = StringIO()
                SeqIO.write([data_repr['consensus_record'], data_repr['closest_record']], repr_fasta_io, "fasta")
                st.download_button("📥 Download Consensus & Representative", repr_fasta_io.getvalue(), "consensus_repr.fasta", key="download_repr_seq")

        # Phylogenetic Tree Generation (from MSA)
        if st.session_state.get('last_msa_params', {}).get('gen_tree'):
            st.subheader("🌳 Phylogenetic Tree from MSA")
            if st.session_state.get('phylogenetic_tree_from_msa') is None: 
                try:
                    current_msa_output_format_tree_gen = st.session_state.get('last_msa_params',{}).get('out_fmt','fasta')
                    alignment_obj_tree_source = AlignIO.read(StringIO(st.session_state.msa_result), current_msa_output_format_tree_gen)
                    num_seqs_in_msa = len(alignment_obj_tree_source)
                    
                    alignment_for_tree_build = alignment_obj_tree_source
                    tree_gen_action = 'all_sequences' 

                    if num_seqs_in_msa > 200: 
                        tree_gen_action = st.radio(
                            "Large MSA Detected for Tree Building:",
                            ('Use all sequences (can be slow for >200)', 
                             'Use random subset of 50 sequences', 
                             'Skip tree generation for this MSA'),
                            index=2, key=f"msa_tree_action_radio_{len(sequences)}" 
                        )
                    
                    proceed_building_tree = True
                    if tree_gen_action == 'Skip tree generation for this MSA':
                        proceed_building_tree = False
                        st.info("Tree generation from MSA skipped by user choice.")
                    elif tree_gen_action == 'Use random subset of 50 sequences':
                        if num_seqs_in_msa > 50:
                            random.seed(42) 
                            subset_indices = sorted(random.sample(range(num_seqs_in_msa), 50))
                            subset_records = [alignment_obj_tree_source[i] for i in subset_indices]
                            alignment_for_tree_build = MultipleSeqAlignment(subset_records)
                            st.info(f"Using a random subset of 50 (from {num_seqs_in_msa}) sequences for tree generation.")
                        else:
                            st.info(f"Number of sequences ({num_seqs_in_msa}) not significantly larger than 50, using all.")
                    
                    if proceed_building_tree:
                        if len(alignment_for_tree_build) < 2:
                            st.warning("Need at least 2 sequences in the (subsetted) alignment to build a tree.")
                        else:
                            with st.spinner("Generating phylogenetic tree from MSA..."):
                                first_seq_tree_str = str(alignment_for_tree_build[0].seq).upper()
                                dna_chars_tree = "ATGCUNRYKMSWBDHV-" 
                                is_dna_tree = all(c in dna_chars_tree for c in first_seq_tree_str if c != '-')
                                
                                model_dist_calc = 'blastn' if is_dna_tree and (len(set(first_seq_tree_str)-set("-N")) <= 4) else 'blosum62' 
                                try:
                                    calculator = DistanceCalculator(model_dist_calc)
                                    dm_for_tree = calculator.get_distance(alignment_for_tree_build)
                                except ValueError: 
                                    st.warning(f"Distance model '{model_dist_calc}' failed, trying 'identity'.")
                                    calculator = DistanceCalculator('identity')
                                    dm_for_tree = calculator.get_distance(alignment_for_tree_build)

                                tree_constructor = DistanceTreeConstructor()
                                tree_method_to_run_msa_tree = st.session_state.get('last_msa_params',{}).get('tree_method', 'nj')
                                
                                generated_msa_tree = None
                                if tree_method_to_run_msa_tree == 'nj':
                                    generated_msa_tree = tree_constructor.nj(dm_for_tree)
                                else: 
                                    generated_msa_tree = tree_constructor.upgma(dm_for_tree)
                                
                                st.session_state.phylogenetic_tree_from_msa = generated_msa_tree
                except Exception as e_tree_gen:
                    st.error(f"Error during phylogenetic tree generation from MSA: {e_tree_gen}")
                    # traceback.print_exc()
            
            if st.session_state.get('phylogenetic_tree_from_msa'):
                msa_tree_to_plot = st.session_state.phylogenetic_tree_from_msa
                
                newick_tree_io = StringIO()
                Phylo.write(msa_tree_to_plot, newick_tree_io, "newick")
                st.download_button(
                    label="📥 Download Generated Tree (Newick)",
                    data=newick_tree_io.getvalue(),
                    file_name=f"msa_phylo_tree_{st.session_state.get('last_msa_params',{}).get('tree_method','nj')}.nwk",
                    mime="text/plain", key="download_msa_tree_newick_btn"
                )

                st.write("Visualizing generated phylogenetic tree with Plotly...")
                fig_msa_tree = plot_phylogenetic_tree_plotly(msa_tree_to_plot) 
                if fig_msa_tree:
                    st.plotly_chart(fig_msa_tree, use_container_width=True)
                else: 
                    st.error("Failed to plot MSA tree with Plotly. Displaying ASCII tree as fallback.")
                    tree_ascii_buffer = StringIO()
                    Phylo.draw_ascii(msa_tree_to_plot, file=tree_ascii_buffer)
                    st.text_area("ASCII Tree (from MSA)", tree_ascii_buffer.getvalue(), height=max(200, msa_tree_to_plot.count_terminals()*15 + 50))
            elif st.session_state.get('last_msa_params', {}).get('gen_tree'): 
                st.info("Phylogenetic tree from MSA was not generated (e.g., skipped or error).")

        st.download_button(
            label="📥 Download MSA Results", data=st.session_state.msa_result,
            file_name=f"msa_results.{msa_output_format}",
            mime=f"text/{msa_output_format}", key="download_msa_main_btn"
        )
    elif st.session_state.get('last_msa_params'): 
        st.info("Click 'Run MSA Analysis' to perform alignment and generate results based on current settings.")


def format_conversion_section(sequences, input_format):
    st.header("🔄 Sequence Format Conversion")
    st.info("Convert sequences between different file formats.")

    conversion_output_format = st.selectbox(
        "Select Output Format For Conversion",
        ("fasta", "clustal", "phylip", "embl", "genbank"), 
        key="conv_out_fmt_select"
    )
    current_conv_params = {
        'seq_ids': tuple(s.id for s in sequences), 
        'in_fmt': input_format, 'out_fmt': conversion_output_format
    }
    params_changed_conv = (st.session_state.get('last_conversion_params') != current_conv_params)

    if st.button("Convert Format", key="convert_fmt_btn"):
        if params_changed_conv or not st.session_state.get('converted_data'):
            if input_format.lower() == "newick": 
                st.warning("Newick is a tree format, cannot convert to sequence formats directly here.")
                st.session_state.converted_data = None
                st.session_state.conversion_error = "Incompatible: Newick to sequence format."
            else:
                with st.spinner("Converting..."):
                    conv_data, conv_err = convert_format(sequences, conversion_output_format)
                    st.session_state.converted_data = conv_data
                    st.session_state.conversion_error = conv_err
                    st.session_state.last_conversion_params = current_conv_params
    
    if st.session_state.get('converted_data'):
        st.success("Conversion successful!")
        st.text_area("Converted Sequences", st.session_state.converted_data, height=300)
        st.download_button(
            "📥 Download Converted File", st.session_state.converted_data,
            f"converted_sequences.{conversion_output_format}",
            mime=f"text/{conversion_output_format}", key="download_conv_btn"
        )
    elif st.session_state.get('conversion_error'):
        st.error(f"Conversion failed: {st.session_state.conversion_error}")


def phylogenetic_tree_section(uploaded_tree: Phylo.BaseTree.Tree): 
    st.header("🌳 Phylogenetic Tree Visualization (from Uploaded Newick)")
    st.info("Displaying the phylogenetic tree from your uploaded Newick file.")

    if not uploaded_tree: 
        st.warning("No phylogenetic tree data found from the uploaded file.")
        return

    st.subheader("Uploaded Tree Structure")
    fig_uploaded_tree = plot_phylogenetic_tree_plotly(uploaded_tree) 
    if fig_uploaded_tree:
        st.plotly_chart(fig_uploaded_tree, use_container_width=True)
    else: 
        st.error("Failed to plot uploaded tree with Plotly. Displaying ASCII tree as fallback.")
        if not st.session_state.get('tree_ascii'): 
            tree_io_ascii_uploaded = StringIO()
            Phylo.draw_ascii(uploaded_tree, out=tree_io_ascii_uploaded)
            st.session_state.tree_ascii = tree_io_ascii_uploaded.getvalue()
        st.text_area("ASCII Representation of Uploaded Tree", st.session_state.tree_ascii, height=max(200, uploaded_tree.count_terminals()*15 + 50))

    if not st.session_state.get('tree_newick'):
        tree_io_newick_uploaded = StringIO()
        Phylo.write(uploaded_tree, tree_io_newick_uploaded, "newick")
        st.session_state.tree_newick = tree_io_newick_uploaded.getvalue()
    
    st.download_button(
        "📥 Download Uploaded Tree (Newick)", st.session_state.tree_newick,
        "uploaded_phylogenetic_tree.nwk", mime="text/plain", key="download_uploaded_newick_btn"
    )


def perform_pairwise_alignment(seq1_rec, seq2_rec, seq_type, mode="global", open_gap=-0.5, extend_gap=-0.1):
    try:
        valid_chars_dna = set('ATGCUNRYKMSWBDHV')
        valid_chars_protein = set('ACDEFGHIKLMNPQRSTVWYBXZJUO')
        valid_chars = valid_chars_dna if seq_type == "DNA" else valid_chars_protein
        
        s1_str_orig = str(seq1_rec.seq)
        s2_str_orig = str(seq2_rec.seq)

        s1_cleaned_str = ''.join(c.upper() for c in s1_str_orig if c.upper() in valid_chars or c == '-')
        s2_cleaned_str = ''.join(c.upper() for c in s2_str_orig if c.upper() in valid_chars or c == '-')
        
        if not s1_cleaned_str or not s2_cleaned_str:
            return "One or both sequences became empty after cleaning (removing non-standard characters). Please check input.", []

        aligner = Align.PairwiseAligner()
        aligner.mode = mode
        aligner.substitution_matrix = substitution_matrices.load("NUC.4.4" if seq_type == "DNA" else "BLOSUM62")
        aligner.open_gap_score = open_gap
        aligner.extend_gap_score = extend_gap
        
        alignments = list(aligner.align(Seq(s1_cleaned_str), Seq(s2_cleaned_str))) 
        
        if not alignments: 
            return "No alignment found. Sequences might be too dissimilar or parameters too stringent.", []
        
        best_alignment = alignments[0] 
        aligned_s1_str_from_biopython = str(best_alignment[0])
        aligned_s2_str_from_biopython = str(best_alignment[1])
        
        match_line_str = generate_match_line(aligned_s1_str_from_biopython, aligned_s2_str_from_biopython, aligner.substitution_matrix)
        mutations_list = report_mutations_from_alignment(aligned_s1_str_from_biopython, aligned_s2_str_from_biopython) 
        
        formatted_alignment_text = format_alignment_display(
            seq1_rec.id, aligned_s1_str_from_biopython, match_line_str, 
            seq2_rec.id, aligned_s2_str_from_biopython, best_alignment.score
        )
        return formatted_alignment_text, mutations_list
    except Exception as e:
        # traceback.print_exc() 
        return f"Pairwise alignment error: {e}. Check sequence characters and parameters.", []

def generate_match_line(aligned_seq1_str, aligned_seq2_str, sub_matrix, similar_threshold=1):
    match_chars = []
    for char1, char2 in zip(aligned_seq1_str, aligned_seq2_str):
        if char1 == '-' or char2 == '-': 
            match_chars.append(' ')
        elif char1 == char2: 
            match_chars.append('|')
        else: 
            score = -float('inf') 
            try: 
                score = sub_matrix[(char1, char2)] 
            except KeyError: 
                try: score = sub_matrix[(char2, char1)]
                except KeyError: pass 
            
            if score >= similar_threshold:
                match_chars.append(':') 
            else:
                match_chars.append(' ') 
    return "".join(match_chars)

def format_alignment_display(id1, aln1_str, match_str, id2, aln2_str, score, line_len=70):
    aln_len_total = len(aln1_str)
    if not (aln_len_total == len(match_str) == len(aln2_str)):
        return "Error: Aligned string lengths mismatch in display formatting."

    output_display_str = f"Alignment Score: {score:.2f}\n\n"
    max_id_len = max(len(id1), len(id2), len("Match"), len("Position")) + 2 

    seq1_ungapped_pos = 0
    seq2_ungapped_pos = 0

    for i in range(0, aln_len_total, line_len):
        chunk_aln1 = aln1_str[i : i + line_len]
        chunk_match = match_str[i : i + line_len]
        chunk_aln2 = aln2_str[i : i + line_len]

        start_pos1_display = seq1_ungapped_pos + 1
        start_pos2_display = seq2_ungapped_pos + 1
        
        seq1_ungapped_pos += len(chunk_aln1.replace('-', ''))
        seq2_ungapped_pos += len(chunk_aln2.replace('-', ''))

        end_pos1_display = seq1_ungapped_pos
        end_pos2_display = seq2_ungapped_pos
        
        output_display_str += f"{id1.ljust(max_id_len)}{start_pos1_display:<7} {chunk_aln1} {end_pos1_display}\n"
        output_display_str += f"{'Match'.ljust(max_id_len)}{''.ljust(7)} {chunk_match}\n"
        output_display_str += f"{id2.ljust(max_id_len)}{start_pos2_display:<7} {chunk_aln2} {end_pos2_display}\n\n"
        
    return output_display_str.strip()


def report_mutations_from_alignment(aligned_ref_str, aligned_query_str):
    """ Reports mutations based on Biopython's aligned strings. Ref positions are ungapped. """
    mutations_found = []
    ref_ungapped_position_counter = 0
    for i in range(len(aligned_ref_str)): 
        ref_char_at_pos = aligned_ref_str[i]
        query_char_at_pos = aligned_query_str[i]
        
        if ref_char_at_pos != '-':
            ref_ungapped_position_counter += 1
        
        if ref_char_at_pos != query_char_at_pos:
            if ref_char_at_pos != '-' and query_char_at_pos != '-':
                mutations_found.append(f"{ref_char_at_pos}{ref_ungapped_position_counter}{query_char_at_pos}")
    return mutations_found


def perform_msa(sequences_list, ref_id_msa, seq_type, out_format): 
    try:
        pyfamsa_input_sequences = [PyFAMSASequence(s.id.encode(), str(s.seq).encode()) for s in sequences_list]
        famsa_aligner = PyFAMSAAligner(guide_tree="sl") 
        msa_from_pyfamsa = famsa_aligner.align(pyfamsa_input_sequences)
        
        biopython_aligned_records = [
            SeqRecord(Seq(s.sequence.decode()), id=s.id.decode(), description="") 
            for s in msa_from_pyfamsa
        ]
        
        msa_output_text_io = StringIO()
        SeqIO.write(biopython_aligned_records, msa_output_text_io, out_format)
        msa_final_text = msa_output_text_io.getvalue()

        mutations_report_dict = {}
        aligned_reference_sequence_str = None
        for record in biopython_aligned_records:
            if record.id == ref_id_msa:
                aligned_reference_sequence_str = str(record.seq)
                break
        
        if aligned_reference_sequence_str is None: 
            st.error(f"Reference sequence ID '{ref_id_msa}' not found within MSA results. Cannot report mutations accurately.")
            return msa_final_text, {} 

        for query_record in biopython_aligned_records:
            aligned_query_sequence_str = str(query_record.seq)
            individual_seq_mutations_str_list = report_mutations_from_alignment(
                aligned_reference_sequence_str, 
                aligned_query_sequence_str
            )
            
            parsed_mutations_for_storage = []
            for mut_str in individual_seq_mutations_str_list:
                if len(mut_str) >=3 and mut_str[1:-1].isdigit(): # Basic check for "A12G" format
                    ref_base = mut_str[0]
                    var_base = mut_str[-1]
                    pos_str = mut_str[1:-1]
                    parsed_mutations_for_storage.append((int(pos_str), ref_base, var_base))
                else:
                    pass 
            
            mutations_report_dict[query_record.id] = parsed_mutations_for_storage
            
        return msa_final_text, mutations_report_dict

    except Exception as e_msa_main:
        # traceback.print_exc() 
        st.error(f"An error occurred during Multiple Sequence Alignment: {e_msa_main}")
        return "", {}


def convert_format(sequences_to_convert, target_format):
    try:
        output_conversion_io = StringIO()
        SeqIO.write(sequences_to_convert, output_conversion_io, target_format)
        return output_conversion_io.getvalue(), None
    except Exception as e_format_conv:
        return None, f"Error during sequence format conversion: {e_format_conv}"


if __name__ == "__main__":
    main()