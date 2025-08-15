import streamlit as st
from Bio import AlignIO, SeqIO, Phylo, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder, PDBIO
from io import StringIO, BytesIO
import traceback
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
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
        return None, None

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

    img_bytes = fig.to_image(format="png")
    st.download_button(
        label="📥 Download MSA Heatmap (PNG)",
        data=img_bytes,
        file_name="msa_heatmap.png",
        mime="image/png"
    )


def plot_msa_image_matplotlib(msa_image: np.ndarray, msa_letters: np.ndarray):
    """
    Plots the MSA as a static heatmap using Matplotlib.

    Parameters:
        msa_image (np.ndarray): Numerical representation of the MSA
        msa_letters (np.ndarray): Letter representation of the MSA
    """
    img_buf = _plot_msa_image_matplotlib_subset(msa_image, msa_letters, 0, 0)
    if img_buf:
        st.download_button(
            label="📥 Download MSA Heatmap (PNG)",
            data=img_buf,
            file_name="msa_heatmap.png",
            mime="image/png"
        )


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
    st.image(buf, caption="MSA Heatmap", use_container_width=True)
    return buf


def init_session_state():
    """
    Initialize session state variables if they don't exist.

    This function ensures all required session state variables are properly initialized
    to prevent KeyError exceptions and enable proper state tracking.
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
    if 'alignment_text' not in st.session_state:
        st.session_state.alignment_text = None
    if 'pairwise_mutations' not in st.session_state:
        st.session_state.pairwise_mutations = None

    # Input-related variables
    if 'sequences' not in st.session_state:
        st.session_state.sequences = None
    if 'seq_type' not in st.session_state:
        st.session_state.seq_type = None
    if 'last_file' not in st.session_state:
        st.session_state.last_file = None
    if 'last_tree_file' not in st.session_state:
        st.session_state.last_tree_file = None
    if 'tree' not in st.session_state:
        st.session_state.tree = None
    if 'tree_ascii' not in st.session_state:
        st.session_state.tree_ascii = None
    if 'tree_newick' not in st.session_state:
        st.session_state.tree_newick = None

    # Format conversion variables
    if 'converted_data' not in st.session_state:
        st.session_state.converted_data = None
    if 'conversion_error' not in st.session_state:
        st.session_state.conversion_error = None
    if 'last_conversion_params' not in st.session_state:
        st.session_state.last_conversion_params = {}

    # Pairwise alignment variables
    if 'selected_seqs' not in st.session_state:
        st.session_state.selected_seqs = None

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

    # Initialize session state
    init_session_state()

    st.title("🔬 Advanced Sequence Alignment and Format Conversion Ability")
    st.write("""
    ### Welcome to the Advanced Sequence Alignment Tool!
    This application allows you to perform **pairwise** and **multiple sequence alignments (MSA)** on **DNA** or **Protein** sequences.
    You can choose between **Global**, **Local**, and **Overlap** alignments. Additionally, you can convert sequence files between various formats.
    For MSA, point mutations relative to a reference sequence are reported.
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
        uploaded_file = st.sidebar.file_uploader(
            "Upload PDB/mmCIF File",
            type=get_file_extensions(input_format),
            key=f"uploader_{input_format}",
            help="Upload a PDB or mmCIF file to extract protein sequences"
        )
        if uploaded_file:
            try:
                # Only process if file is new or changed
                file_changed = ('last_file' not in st.session_state or
                               st.session_state.last_file != uploaded_file.name)

                if file_changed:
                    st.session_state.last_file = uploaded_file.name
                    sequences, error = parse_sequences_from_structure(uploaded_file, input_format)
                    if error:
                        st.sidebar.error(error)
                    else:
                        st.sidebar.success(f"Successfully extracted {len(sequences)} sequences from the uploaded file.")
                        st.session_state.sequences = sequences
                else:
                    sequences = st.session_state.sequences
            except Exception as e:
                st.sidebar.error(f"Error parsing {input_format} file: {e}")
    elif input_format == "Newick":
        uploaded_file = st.sidebar.file_uploader("Upload Newick Tree File", type=["nwk", "newick", "tree", "tre"], key="uploader_newick")
        if uploaded_file:
            try:
                # Only process if file is new or changed
                file_changed = ('last_tree_file' not in st.session_state or
                               st.session_state.last_tree_file != uploaded_file.name)

                if file_changed:
                    st.session_state.last_tree_file = uploaded_file.name
                    tree = Phylo.read(uploaded_file, "newick")
                    st.session_state.tree = tree

                    # Reset tree visualization cache
                    if 'tree_ascii' in st.session_state:
                        del st.session_state.tree_ascii
                    if 'tree_newick' in st.session_state:
                        del st.session_state.tree_newick
                else:
                    tree = st.session_state.tree

                st.sidebar.success("Successfully loaded Newick tree.")
            except Exception as e:
                st.sidebar.error(f"Error reading Newick file: {e}")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Sequence File",
            type=get_file_extensions(input_format),
            key=f"uploader_{input_format}",
            help=f"Upload sequence file in {input_format} format"
        )
        if uploaded_file:
            try:
                # Only process if file is new or changed
                file_changed = ('last_file' not in st.session_state or
                               st.session_state.last_file != uploaded_file.name)

                if file_changed:
                    st.session_state.last_file = uploaded_file.name
                    sequences, error = parse_sequences_from_file(uploaded_file, input_format)
                    if error:
                        st.sidebar.error(error)
                    else:
                        st.sidebar.success(f"Successfully loaded {len(sequences)} sequences.")
                        st.session_state.sequences = sequences
                else:
                    sequences = st.session_state.sequences
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")

    # Get sequences from session state if available
    if sequences is None and 'sequences' in st.session_state:
        sequences = st.session_state.sequences

    if tree is None and 'tree' in st.session_state:
        tree = st.session_state.tree

    if (sequences and len(sequences) > 0) or input_format == "Newick":
        if sequences:
            seq_type = st.sidebar.selectbox(
                "🔬 Select Sequence Type",
                ("DNA", "Protein"),
                help="Choose whether your sequences are DNA or Protein.",
                index=1
            )

            # Only update the session state if the selection has changed
            if 'seq_type' not in st.session_state or st.session_state.seq_type != seq_type:
                st.session_state.seq_type = seq_type
                # If sequence type changes, invalidate previous results
                st.session_state.msa_result = None
                st.session_state.mutations = None
                st.session_state.msa_image = None
                st.session_state.msa_letters = None
                st.session_state.consensus_data = None
                st.session_state.alignment_text = None
                st.session_state.pairwise_mutations = None
                st.session_state.last_msa_params = {}
                st.session_state.last_pairwise_params = {}
        else:
            seq_type = st.session_state.seq_type if 'seq_type' in st.session_state else None

        if sequences and input_format not in ["PDB", "mmCIF"]:
            alignment_mode = st.sidebar.selectbox(
                "🛠️ Select Alignment Mode",
                ("Pairwise", "MSA", "Convert Formats"),
                help="Choose the type of analysis you want to perform."
            )
        elif sequences and input_format in ["PDB", "mmCIF"]:
            alignment_mode = None
        else:
            alignment_mode = "Phylogenetic Tree"

        if sequences:
            if input_format in ["PDB", "mmCIF"]:
                st.header("📄 Extracted FASTA Sequences")
                fasta_io = StringIO()
                SeqIO.write(sequences, fasta_io, "fasta")
                fasta_str = fasta_io.getvalue()
                st.text_area("FASTA Sequences", value=fasta_str, height=500)
                file_basename = os.path.splitext(os.path.basename(uploaded_file.name))[0]
                st.download_button(
                    label="📥 Download FASTA",
                    data=fasta_str,
                    file_name=f"{file_basename}_sequences.fasta",
                    mime="text/plain"
                )
            else:
                if alignment_mode == "Pairwise":
                    pairwise_alignment_section(sequences, seq_type)
                elif alignment_mode == "MSA":
                    msa_section(sequences, seq_type)
                elif alignment_mode == "Convert Formats":
                    format_conversion_section(sequences, input_format)
        else:
            if alignment_mode == "Phylogenetic Tree":
                phylogenetic_tree_section(tree)
    else:
        st.info("Please upload sequences or a Newick tree in the sidebar to begin.")

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
            return None, f"No sequences found in the uploaded {format_name} file."
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

            with st.sidebar.expander("Structure processing details"):
                for model_idx, model in enumerate(structure, 1):
                    st.write(f"Processing model {model_idx}...")

                    for chain in model:
                        chain_id = chain.id
                        if chain_id == " " or not list(chain.get_residues()):
                            continue

                        st.write(f"Extracting chain {chain_id}...")

                        # Method 1: Use PPBuilder
                        peptides = list(ppb.build_peptides(chain))
                        if peptides:
                            sequence = "".join(str(pp.get_sequence()) for pp in peptides)
                            if sequence:
                                seq_record = SeqRecord(
                                    Seq(sequence),
                                    id=f"{file_basename}_{chain_id}",
                                    description=f"Chain {chain_id} from {file.name}"
                                )
                                sequences.append(seq_record)
                                st.write(f"✅ Extracted {len(sequence)} residues")
                                continue

                        # Method 2: Manual extraction for problematic chains
                        st.write("Standard extraction failed, trying manual extraction...")
                        try:
                            aa_sequence = ""
                            valid_residues = 0

                            for res in chain:
                                # Skip non-standard residues and waters
                                if res.id[0] != " " and not res.id[0].startswith("H_"):
                                    continue

                                resname = res.resname.strip()
                                aa = amino_acid_map.get(resname, None)
                                if aa:
                                    aa_sequence += aa
                                    valid_residues += 1
                                elif len(resname) == 3 and resname[0].isalpha():
                                    aa_sequence += "X"
                                    valid_residues += 1

                            if valid_residues >= 5:  # At least 5 valid residues
                                seq_record = SeqRecord(
                                    Seq(aa_sequence),
                                    id=f"{file_basename}_{chain_id}_manual",
                                    description=f"Manually extracted from chain {chain_id}"
                                )
                                sequences.append(seq_record)
                                st.write(f"✅ Manually extracted {valid_residues} residues")
                        except Exception as chain_err:
                            st.write(f"❌ Failed manual extraction: {chain_err}")

            if not sequences:
                return None, "No protein sequences could be extracted from the structure file."

            return sequences, None
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_filepath)
            except:
                pass

    except Exception as e:
        traceback.print_exc()  # For debugging
        return None, f"An error occurred while parsing the {format_name} file: {str(e)}"


def save_msa_to_fasta(msa: List[PyFAMSASequence], output_path: str) -> bool:
    """
    Converts pyFAMSA MSA output to a FASTA file compatible with Biopython.

    Parameters:
        msa (List[PyFAMSASequence]): List of aligned sequences from pyFAMSA
        output_path (str): Path to save the FASTA file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        seq_records = []
        for seq in msa:
            seq_id = seq.id.decode('utf-8')
            sequence = seq.sequence.decode('utf-8')
            record = SeqRecord(Seq(sequence), id=seq_id, description="")
            seq_records.append(record)
        SeqIO.write(seq_records, output_path, "fasta")
        print(f"MSA successfully saved to {output_path}")
        return True
    except Exception as e:
        # traceback.print_exc() # For developer debugging
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.unlink(temp_filepath)
        return None, f"An error occurred while processing the {format_name} file: {e}"


import re


def convert_mutations_to_csv(mutations: List[str]) -> str:
    """
    Converts a list of mutation strings to a CSV formatted string.
    """
    if not mutations:
        return ""

    mutation_data = []
    for mut in mutations:
        match = re.match(r"([A-Z-])(\d+)([A-Z-])", mut)
        if match:
            original, position, mutated = match.groups()
            mutation_data.append([original, int(position), mutated])

    if not mutation_data:
        return ""

    df = pd.DataFrame(mutation_data, columns=["Original", "Position", "Mutated"])
    output = StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def convert_msa_mutations_to_csv(mutations: Dict[str, List[Tuple[int, str, str]]]) -> str:
    """
    Converts a dictionary of MSA mutations to a CSV formatted string.
    """
    if not mutations:
        return ""

    all_mutations = []
    for seq_id, mut_list in mutations.items():
        for pos, ref, var in mut_list:
            all_mutations.append([seq_id, pos, ref, var])

    if not all_mutations:
        return ""

    df = pd.DataFrame(all_mutations, columns=["Sequence_ID", "Position", "Reference", "Variant"])
    output = StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()


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

    # Check if parameters have changed
    params_changed = ('last_pairwise_params' not in st.session_state or
                      st.session_state.last_pairwise_params != current_params)

    run_alignment = st.button("Run Pairwise Alignment")

    # Perform alignment if button is clicked or if we have results and parameters haven't changed
    if run_alignment or (st.session_state.get('alignment_text') is not None and not params_changed):
        # Only recalculate if parameters changed or explicitly requested
        if run_alignment or params_changed:
            with st.spinner("Aligning sequences..."):
                alignment_text, mutations = perform_pairwise_alignment(
                    seq1, seq2, seq_type, align_mode.lower(), open_gap_score, extend_gap_score
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
            st.write(f"**Mutations:** {mutations_str}")

            mutation_positions = []
            mutation_labels = []
            for mut in st.session_state.pairwise_mutations:
                pos = ''.join(filter(str.isdigit, mut))
                mutation_positions.append(int(pos))
                mutation_labels.append(mut)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mutation_positions,
                y=[1]*len(mutation_positions),
                mode='markers',
                marker=dict(size=10, color='red'),
                text=mutation_labels,
                hoverinfo='text'
            ))
            fig.update_layout(
                title="Point Mutations Plot",
                xaxis_title="Position in Reference Sequence",
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
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

    if calculate_representative:
        current_msa_params['consensus_threshold'] = consensus_threshold

    # Check if MSA parameters have changed
    msa_params_changed = ('last_msa_params' not in st.session_state or
                         st.session_state.last_msa_params != current_msa_params)

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
                        msa_image, msa_letters = msa_to_image(msa_result, msa_output_format)
                        st.session_state.msa_image = msa_image
                        st.session_state.msa_letters = msa_letters
                    except Exception as e:
                        st.error(f"Failed to generate MSA heatmap: {e}")

        if st.session_state.msa_result:
            st.subheader("📄 MSA Result")
            seqs_total = parse_sequences_from_text(st.session_state.msa_result)[0]
            st.write(f"Total sequences: {len(seqs_total)}. MSA length: {len(seqs_total[0].seq)}")

            if st.session_state.msa_image is not None and st.session_state.msa_letters is not None:
                plot_msa_image(st.session_state.msa_image, st.session_state.msa_letters, plot_method)

            st.subheader("🔍 Point Mutations Relative to Reference")
            if st.session_state.mutations:
                # Allow user to select which sequences to view mutations for
                seq_ids = sorted(list(st.session_state.mutations.keys()))

                if len(seq_ids) > 0:
                    # Add a "Select All" option if there are many sequences
                    if len(seq_ids) > 10:
                        select_all = st.checkbox(
                            "Select all sequences",
                            value=False,
                            key="select_all_mutations",
                            help="Show mutations for all sequences"
                        )

                        if select_all:
                            selected_seq_ids = seq_ids
                        else:
                            # Default to showing first few sequences if count is high
                            default_selection = seq_ids[:min(5, len(seq_ids))]
                            selected_seq_ids = st.multiselect(
                                "Select sequences to view mutations for:",
                                seq_ids,
                                default=default_selection,
                                key="mutation_sequence_selector",
                                help="Choose which sequences to display mutations for"
                            )
                    else:
                        # For fewer sequences, just show the multiselect with all selected by default
                        selected_seq_ids = st.multiselect(
                            "Select sequences to view mutations for:",
                            seq_ids,
                            default=seq_ids[:min(10, len(seq_ids))],
                            key="mutation_sequence_selector",
                            help="Choose which sequences to display mutations for"
                        )

                    if selected_seq_ids:
                        for seq_id in selected_seq_ids:
                            mut_list = st.session_state.mutations.get(seq_id, [])
                            if mut_list:
                                # Each mutation tuple is (position, ref_base, variant)
                                df = pd.DataFrame(mut_list, columns=["Position", "Reference", "Variant"])

                                with st.expander(f"Mutations for {seq_id} ({len(mut_list)} mutations)"):
                                    st.dataframe(df)

                                    # Plot mutations if there aren't too many
                                    if len(mut_list) <= 300:  # Limit for performance
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=[pos for pos, _, _ in mut_list],
                                            y=[1]*len(mut_list),
                                            mode='markers',
                                            marker=dict(size=8, color='red'),
                                            text=[f"{ref}{pos}{var}" for pos, ref, var in mut_list],
                                            hoverinfo='text'
                                        ))
                                        fig.update_layout(
                                            title=f"Mutations in {seq_id}",
                                            xaxis_title="Position in Reference Sequence",
                                            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                                            height=200,
                                            margin=dict(l=20, r=20, t=30, b=20),
                                            showlegend=False
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # write the name of the sequence followed by the single point mutation
                                        string_mutations = ', '.join([f"{ref}{pos}{var}" for pos, ref, var in mut_list])
                                        st.write(f"**{seq_id}:** {string_mutations}")
                                            
                    else:
                        st.info("Select at least one sequence to view mutations.")
                else:
                    st.info("No sequences with mutations detected.")
            else:
                st.write("No point mutations detected relative to the reference sequence.")

            # Add representative sequence calculation if requested
            if calculate_representative:
                # Only calculate if not already done or if user changes threshold
                recalculate = (st.session_state.consensus_data is None or
                              'threshold' not in st.session_state.consensus_data or
                              st.session_state.consensus_data['threshold'] != consensus_threshold)

                st.subheader("🧬 Representative Sequence Analysis")

                if recalculate:
                    with st.spinner("Calculating consensus and representative sequence..."):
                        try:
                            # Get alignment object from MSA result
                            alignment = AlignIO.read(StringIO(st.session_state.msa_result), msa_output_format)

                            # Calculate representative sequence with user-selected threshold
                            consensus_record, closest_record, min_differences, closest_seq_id = calculate_representative_sequence(
                                alignment, threshold=consensus_threshold
                            )

                            # Store in session state
                            st.session_state.consensus_data = {
                                'threshold': consensus_threshold,
                                'consensus_record': consensus_record,
                                'closest_record': closest_record,
                                'min_differences': min_differences,
                                'closest_seq_id': closest_seq_id,
                                'alignment_length': alignment.get_alignment_length(),
                                'seq_count': len(alignment)
                            }
                        except Exception as e:
                            st.error(f"Failed to calculate representative sequence: {str(e)}")
                            st.info("Try adjusting the consensus threshold or check if your sequences are properly aligned.")

                # Display results if available
                if st.session_state.consensus_data:
                    data = st.session_state.consensus_data
                    st.write(f"**Consensus threshold:** {data['threshold']}")
                    st.write(f"**Number of sequences:** {data['seq_count']}")
                    st.write(f"**Alignment length:** {data['alignment_length']} positions")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Consensus Sequence")
                        st.info("Consensus shows the most common residue at each position based on the threshold")
                        st.code(f">Consensus\n{data['consensus_record'].seq}")

                    with col2:
                        st.markdown("### Most Representative Sequence")
                        st.info("This sequence is most similar to the consensus sequence")
                        st.write(f"**Sequence ID:** `{data['closest_seq_id']}`")
                        st.write(f"**Differences from consensus:** {data['min_differences']}")
                        st.code(f">{data['closest_seq_id']}\n{data['closest_record'].seq}")

                    # Create downloadable file with consensus and representative sequence
                    consensus_fasta_io = StringIO()
                    SeqIO.write([data['consensus_record'], data['closest_record']], consensus_fasta_io, "fasta")
                    consensus_fasta = consensus_fasta_io.getvalue()

                    st.download_button(
                        label="📥 Download Consensus and Representative Sequence",
                        data=consensus_fasta,
                        file_name="consensus_and_representative.fasta",
                        mime="text/plain"
                    )

            st.download_button(
                label="📥 Download MSA",
                data=st.session_state.msa_result,
                file_name=f"msa_alignment.{msa_output_format}",
                mime=f"text/{msa_output_format}"
            )


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

    # Check if parameters have changed
    params_changed = ('last_conversion_params' not in st.session_state or
                      st.session_state.last_conversion_params != current_params)

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


def phylogenetic_tree_section(tree):
    """
    Handles the phylogenetic tree visualization workflow.

    Parameters:
        tree: BioPython Tree object to visualize
    """
    st.header("🌳 Phylogenetic Tree")
    st.info(
        "Phylogenetic trees show evolutionary relationships among sequences. "
        "This visualization displays the hierarchical clustering of your sequences."
    )

    if not uploaded_tree: 
        st.warning("No phylogenetic tree data found from the uploaded file.")
        return

    st.subheader("📄 Phylogenetic Tree")

    # Only redraw the tree when needed
    if 'tree_ascii' not in st.session_state:
        tree_io = StringIO()
        Phylo.draw_ascii(tree, out=tree_io)
        st.session_state.tree_ascii = tree_io.getvalue()

        tree_newick_io = StringIO()
        Phylo.write(tree, tree_newick_io, "newick")
        st.session_state.tree_newick = tree_newick_io.getvalue()

    st.text(st.session_state.tree_ascii)

    st.download_button(
        label="📥 Download Phylogenetic Tree (Newick)",
        data=st.session_state.tree_newick,
        file_name="phylogenetic_tree.newick",
        mime="text/plain"
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
        msa_io = StringIO()
        SeqIO.write(aligned_seq_records, msa_io, output_format)
        msa_text = msa_io.getvalue()
        # Get reference sequence from aligned records
        ref_seq = next((seq for seq in aligned_seq_records if seq.id == reference_id), None)
        if not ref_seq:
            st.error("Reference sequence not found in MSA results.")
            return msa_text, {}
        mutations = {}
        for seq in aligned_seq_records:
            if seq.id == reference_id:
                continue
            seq_mutations = []
            ref_position = 0
            for i, (ref_base, seq_base) in enumerate(zip(ref_seq.seq, seq.seq), start=1):
                if ref_base != '-':
                    ref_position += 1
                # Report mutation only if both residues are not gaps.
                if ref_base != seq_base and ref_base != '-' and seq_base != '-':
                    seq_mutations.append((ref_position, ref_base, seq_base))
            mutations[seq.id] = seq_mutations
        return msa_text, mutations
    except Exception as e:
        st.error(f"An error occurred during MSA: {e}")
        print(traceback.format_exc())
        return "", {}


def convert_format(sequences, output_format):
    """
    Convert sequences from the current format to the desired output format.

    Parameters:
        sequences: List of sequence records to convert
        output_format (str): Target format for conversion

    Returns:
        tuple: (converted_data, error) - Converted data as string and error message or None
    """
    try:
        output = StringIO()
        SeqIO.write(sequences, output, output_format)
        converted_data = output.getvalue()
        return converted_data, None
    except Exception as e:
        return None, f"Error during format conversion: {e}"


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
            for seq2 in sequences[:i]:
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
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    if seq_type == "DNA":
        aligner.substitution_matrix = substitution_matrices.load("NUC.4.4")
    else:
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

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


if __name__ == "__main__":
    main()