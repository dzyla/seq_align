import streamlit as st
from Bio import AlignIO
from Bio import Align
from Bio import SeqIO
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
from Bio.Align import AlignInfo

# Import pyFAMSA
from pyfamsa import Aligner as PyFAMSAAligner, Sequence as PyFAMSASequence


def msa_to_image(alignment_text: str, format: str) -> tuple:
    """
    Converts Multiple Sequence Alignment (MSA) to numerical image data and amino acid array.

    Parameters:
        alignment_text (str): The MSA text in the specified format
        format (str): The format of the MSA text (e.g., 'fasta', 'clustal')

    Returns:
        tuple: (msa_image, msa_letters) - numerical representation and letter representation
    """
    try:
        alignment = AlignIO.read(StringIO(alignment_text), format)
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
    plt.close(fig)
    buf.seek(0)
    img = buf.getvalue()
    st.image(img, caption="MSA Heatmap", use_container_width=True)


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
    # Calculate the consensus sequence
    summary = AlignInfo.SummaryInfo(alignment)
    consensus = summary.dumb_consensus(threshold=threshold, ambiguous='X')
    consensus_seq = str(consensus)

    # Find the sequence closest to the consensus
    def count_differences(seq1, seq2):
        """Count the number of differences between two sequences."""
        return sum(1 for a, b in zip(seq1, seq2) if a != b)

    min_differences = None
    closest_sequence = None
    closest_seq_id = None

    for record in alignment:
        seq = str(record.seq)
        differences = count_differences(consensus_seq, seq)

        if (min_differences is None) or (differences < min_differences):
            min_differences = differences
            closest_sequence = seq
            closest_seq_id = record.id

    # Create SeqRecord objects
    consensus_record = SeqRecord(Seq(consensus_seq), id="Consensus", description="Consensus sequence")
    closest_record = SeqRecord(Seq(closest_sequence), id=closest_seq_id, description="Most representative sequence")

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
    tree = None
    uploaded_file = None

    # Handle different input methods based on format selection
    if input_format == "Text (FASTA)":
        seq_text = st.sidebar.text_area(
            "Paste Sequences Here (FASTA Format)",
            height=250,
            placeholder=">Sequence1\nATGCGTA...\n>Sequence2\nATGCGTC...",
            help="Enter your sequences in FASTA format"
        )

        # Only parse if text has changed
        if seq_text:
            current_hash = hash(seq_text)
            if 'last_text_hash' not in st.session_state or st.session_state.last_text_hash != current_hash:
                st.session_state.last_text_hash = current_hash
                sequences, error = parse_sequences_from_text(seq_text)
                if error:
                    st.sidebar.error(error)
                else:
                    st.sidebar.success(f"Successfully loaded {len(sequences)} sequences.")
                    st.session_state.sequences = sequences
            else:
                sequences = st.session_state.sequences
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
        uploaded_file = st.sidebar.file_uploader(
            "Upload Newick Tree File",
            type=["nwk", "newick", "tree"],
            key="uploader_newick",
            help="Upload a Newick format tree file for visualization"
        )
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


# Define amino acid 3-letter to 1-letter mapping
amino_acid_map = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Non-standard amino acids
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XLE": "J",
    "MSE": "M",  # Selenomethionine
    "UNK": "X",  # Unknown
    # Modified residues - map to their standard counterparts
    "MLE": "L", "CSD": "C", "HYP": "P", "KCX": "K", "CSO": "C",
    "TPO": "T", "SEP": "S", "MLY": "K", "M3L": "K", "OCS": "C",
    "PTR": "Y", "PCA": "E", "SAC": "S", "MLZ": "K"
}


def get_file_extensions(format_name):
    """
    Return appropriate file extensions based on the selected format.

    Parameters:
        format_name (str): Name of the file format

    Returns:
        list: List of file extensions associated with the format
    """
    format_extensions = {
        "FASTA": ["fasta", "fa", "fna", "ffn", "faa", "frn", "fsa", "seq"],
        "Clustal": ["clustal", "aln", "clw"],
        "Phylip": ["phy", "ph", "phylip"],
        "EMBL": ["embl", "ebl", "emb"],
        "GenBank": ["gb", "gbk", "genbank"],
        "Newick": ["nwk", "newick", "tree"],
        "PDB": ["pdb", "ent"],
        "mmCIF": ["cif", "mmcif", "mcif"]  # Added mcif as alternative extension
    }
    return format_extensions.get(format_name, [])


def parse_sequences_from_text(text):
    """
    Parse sequences from pasted text in FASTA format.

    Parameters:
        text (str): FASTA formatted text with sequences

    Returns:
        tuple: (sequences, error) - List of sequence records or None, and error message or None
    """
    try:
        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [line.strip() for line in normalized_text.split('\n') if line.strip() != '']
        cleaned_text = '\n'.join(lines)
        if not cleaned_text.startswith('>'):
            return None, "FASTA format should start with '>'. Please check your input."
        seq_io = StringIO(cleaned_text)
        sequences = list(SeqIO.parse(seq_io, "fasta"))
        if not sequences:
            return None, "No valid FASTA sequences found. Please check your input."
        sequence_ids = set()
        for seq in sequences:
            if seq.id in sequence_ids:
                return None, f"Duplicate sequence ID found: '{seq.id}'. Each sequence must have a unique ID."
            sequence_ids.add(seq.id)
            if not str(seq.seq).strip():
                return None, f"Sequence '{seq.id}' is empty. Please provide valid sequences."
        return sequences, None
    except Exception as e:
        return None, f"An error occurred while parsing the text input: {e}"


def parse_sequences_from_file(file, format_name):
    """
    Parse sequences from an uploaded file based on the selected format.

    Parameters:
        file: Uploaded file object
        format_name (str): Format of the file

    Returns:
        tuple: (sequences, error) - List of sequence records or None, and error message or None
    """
    try:
        if format_name.lower() == "newick":
            return None, "Newick format is for phylogenetic trees, not sequences."
        file.seek(0)
        file_content = file.read().decode("utf-8")
        seq_io = StringIO(file_content)
        sequences = list(SeqIO.parse(seq_io, format_name.lower()))
        if not sequences:
            return None, f"No sequences found in the uploaded {format_name} file."
        return sequences, None
    except Exception as e:
        return None, f"Error parsing {format_name} file: {e}"


def parse_sequences_from_structure(file, format_name):
    """
    Parse sequences from PDB or mmCIF structure files.

    Parameters:
        file: Uploaded file object
        format_name (str): Format of the file ('PDB' or 'mmCIF')

    Returns:
        tuple: (sequences, error) - List of sequence records or None, and error message or None
    """
    try:
        file.seek(0)
        file_content = file.read()
        file_basename = os.path.splitext(os.path.basename(file.name))[0]

        # Use temporary files for reliable parsing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_name.lower()}") as temp_file:
            temp_file.write(file_content)
            temp_filepath = temp_file.name

        try:
            # Parse the structure
            if format_name == "PDB":
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(file_basename, temp_filepath)
            elif format_name == "mmCIF":
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure(file_basename, temp_filepath)
            else:
                return None, f"Unsupported format: {format_name}"

            # Extract sequences using multiple methods
            sequences = []
            ppb = PPBuilder()

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
        print(f"An error occurred while saving MSA to FASTA: {e}")
        return False


def get_msa_as_fasta(msa: List[PyFAMSASequence]) -> str:
    """
    Converts pyFAMSA MSA output to a FASTA-formatted string compatible with Biopython.

    Parameters:
        msa (List[PyFAMSASequence]): List of aligned sequences from pyFAMSA

    Returns:
        str: FASTA-formatted string
    """
    try:
        seq_records = []
        for seq in msa:
            seq_id = seq.id.decode('utf-8')
            sequence = seq.sequence.decode('utf-8')
            record = SeqRecord(Seq(sequence), id=seq_id, description="")
            seq_records.append(record)
        fasta_io = StringIO()
        SeqIO.write(seq_records, fasta_io, "fasta")
        fasta_content = fasta_io.getvalue()
        return fasta_content
    except Exception as e:
        st.error(f"An error occurred while converting MSA to FASTA: {e}")
        return ""


def pairwise_alignment_section(sequences, seq_type):
    """
    Handles the pairwise alignment workflow including UI and computation.

    Parameters:
        sequences: List of sequence records to align
        seq_type (str): Type of sequences ('DNA' or 'Protein')
    """
    st.header("🔀 Pairwise Alignment")

    if len(sequences) < 2:
        st.warning("Please upload at least two sequences for pairwise alignment.")
        return

    align_mode = st.selectbox(
        "Select Alignment Mode",
        ("Global", "Local", "Overlap"),
        help="Global: Align entire sequences. Local: Find best local regions. Overlap: Allow end gaps."
    )

    seq_names = [seq.id for seq in sequences]

    # Replace multiselect with two separate selectboxes
    col1, col2 = st.columns(2)

    # Default to first sequence or previously selected
    default_ref_idx = 0
    if 'reference_id' in st.session_state and st.session_state.reference_id in seq_names:
        default_ref_idx = seq_names.index(st.session_state.reference_id)

    with col1:
        reference_id = st.selectbox(
            "Select Reference Sequence",
            seq_names,
            index=default_ref_idx,
            key="reference_seq_select",
            help="First sequence for alignment"
        )

    # Default to second sequence or previously selected
    default_target_idx = min(1, len(seq_names)-1)  # Default to second sequence if available
    if 'target_id' in st.session_state and st.session_state.target_id in seq_names:
        default_target_idx = seq_names.index(st.session_state.target_id)

    with col2:
        target_id = st.selectbox(
            "Select Target Sequence",
            seq_names,
            index=default_target_idx,
            key="target_seq_select",
            help="Second sequence for alignment"
        )

    # Store the current selections in session state
    st.session_state.reference_id = reference_id
    st.session_state.target_id = target_id

    # Check if selection has changed from last time
    selection_changed = ('last_selected_seqs' not in st.session_state or
                         st.session_state.last_selected_seqs != (reference_id, target_id))

    if selection_changed:
        # Reset alignment results when selection changes
        if 'alignment_text' in st.session_state:
            st.session_state.alignment_text = None
            st.session_state.pairwise_mutations = None
        # Reset the parameters tracking
        if 'last_pairwise_params' in st.session_state:
            st.session_state.last_pairwise_params = {}
        st.session_state.last_selected_seqs = (reference_id, target_id)

    c1, c2 = st.columns(2)
    open_gap_score = c1.number_input(
        "Open Gap Score",
        value=-0.5,
        step=0.1,
        help="Penalty for opening a gap in the alignment. More negative = higher penalty."
    )

    extend_gap_score = c2.number_input(
        "Extend Gap Score",
        value=-0.1,
        step=0.1,
        help="Penalty for extending an existing gap. More negative = higher penalty."
    )

    if reference_id == target_id:
        st.warning("Please select two different sequences for alignment.")
        return

    seq1 = next(seq for seq in sequences if seq.id == reference_id)
    seq2 = next(seq for seq in sequences if seq.id == target_id)

    # Create a unique parameter set to detect changes
    current_params = {
        'ref_id': reference_id,
        'target_id': target_id,
        'seq_type': seq_type,
        'align_mode': align_mode.lower(),
        'open_gap_score': open_gap_score,
        'extend_gap_score': extend_gap_score
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
                st.session_state.alignment_text = alignment_text
                st.session_state.pairwise_mutations = mutations
                st.session_state.last_pairwise_params = current_params

        st.subheader("🧬 Alignment Result")
        st.code(st.session_state.alignment_text)

        st.subheader("🔍 Point Mutations Relative to Reference")
        if st.session_state.pairwise_mutations:
            st.write(f"**Reference Sequence:** `{reference_id}`")
            st.write(f"**Target Sequence:** `{target_id}`")
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
            st.write("No point mutations detected.")

        st.download_button(
            label="📥 Download Alignment",
            data=st.session_state.alignment_text,
            file_name=f"pairwise_alignment_{reference_id}_vs_{target_id}.txt",
            mime="text/plain"
        )


def msa_section(sequences, seq_type):
    """
    Handles the Multiple Sequence Alignment (MSA) workflow including UI and computation.

    Parameters:
        sequences: List of sequence records to align
        seq_type (str): Type of sequences ('DNA' or 'Protein')
    """
    st.header("📏 Multiple Sequence Alignment (MSA)")
    st.info(
        "Multiple Sequence Alignment aligns three or more sequences together, "
        "identifying conserved regions and structural similarities."
    )

    if len(sequences) < 2:
        st.warning("Please upload at least two sequences for MSA.")
        return

    ref_seq_id = st.selectbox(
        "Select Reference Sequence for Mutation Reporting",
        [seq.id for seq in sequences],
        key="ref_seq_select",
        help="Mutations in other sequences will be reported relative to this sequence"
    )

    msa_output_format = st.selectbox(
        "Select MSA Output Format",
        ("fasta", "clustal", "phylip", "stockholm"),
        help="Choose the format for saving the alignment results"
    )

    plot_method = st.selectbox(
        "Select Plotting Method",
        ("Plotly (Interactive)", "Matplotlib (Static)"),
        help="Plotly provides interactive visualization. Matplotlib creates static images."
    )

    # Add option to calculate representative sequence
    calculate_representative = st.checkbox(
        "Calculate Most Representative Sequence",
        help="Find the consensus sequence and sequence that best represents the entire alignment"
    )

    if calculate_representative:
        consensus_threshold = st.slider(
            "Consensus Threshold",
            0.5, 1.0, 0.7, 0.05,
            help="Minimum fraction of sequences that must have the same residue at a position to be included in consensus"
        )

    # Create a unique parameter set to track MSA settings
    current_msa_params = {
        'sequences_ids': tuple(seq.id for seq in sequences),
        'ref_seq_id': ref_seq_id,
        'seq_type': seq_type,
        'msa_output_format': msa_output_format,
        'calculate_representative': calculate_representative
    }

    if calculate_representative:
        current_msa_params['consensus_threshold'] = consensus_threshold

    # Check if MSA parameters have changed
    msa_params_changed = ('last_msa_params' not in st.session_state or
                         st.session_state.last_msa_params != current_msa_params)

    run_msa = st.button("Run MSA")

    # Use session state to store and retrieve MSA results
    if run_msa or (st.session_state.msa_result is not None and not msa_params_changed):
        # Only recalculate if parameters changed or explicitly requested
        if run_msa or msa_params_changed:
            with st.spinner("Performing MSA - this may take a moment for large datasets..."):
                msa_result, mutations = perform_msa(sequences, ref_seq_id, seq_type, msa_output_format)
                st.session_state.msa_result = msa_result
                st.session_state.mutations = mutations
                st.session_state.last_msa_params = current_msa_params

                # Reset consensus data when running new MSA
                st.session_state.consensus_data = None

                # Generate image data for plotting
                if msa_result:
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
    """
    Handles the sequence format conversion workflow.

    Parameters:
        sequences: List of sequence records to convert
        input_format (str): Original format of the sequences
    """
    st.header("🔄 Sequence Format Conversion")
    st.info(
        "Convert your sequences between different file formats. "
        "This is useful for compatibility with different analysis abilities."
    )

    conversion_output_format = st.selectbox(
        "Select Output Format",
        ("fasta", "clustal", "phylip", "embl", "genbank"),
        help="Choose the desired format for the converted sequences"
    )

    # Create a hash of the current parameters
    current_params = {
        'sequences_ids': tuple(seq.id for seq in sequences),
        'input_format': input_format,
        'output_format': conversion_output_format
    }

    # Check if parameters have changed
    params_changed = ('last_conversion_params' not in st.session_state or
                      st.session_state.last_conversion_params != current_params)

    convert_button = st.button("Convert Format")

    # Use session state to store conversion results
    if 'converted_data' not in st.session_state:
        st.session_state.converted_data = None
        st.session_state.conversion_error = None

    # Only run conversion if button is clicked or we have results and parameters haven't changed
    if convert_button or (st.session_state.converted_data is not None and not params_changed):
        # Only recalculate if parameters changed or explicitly requested
        if convert_button or params_changed:
            if input_format.lower() == "newick" and conversion_output_format.lower() != "newick":
                st.warning("Newick format is a tree format and cannot be converted to sequence formats directly.")
                st.session_state.converted_data = None
                st.session_state.conversion_error = "Incompatible formats"
            else:
                with st.spinner("Converting format..."):
                    converted_data, error = convert_format(sequences, conversion_output_format)
                    st.session_state.converted_data = converted_data
                    st.session_state.conversion_error = error
                    st.session_state.last_conversion_params = current_params

        # Display results if available
        if st.session_state.converted_data:
            st.success("Format conversion successful.")
            st.text(st.session_state.converted_data)
            st.download_button(
                label="📥 Download Converted File",
                data=st.session_state.converted_data,
                file_name=f"converted_sequences.{conversion_output_format}",
                mime=f"text/{conversion_output_format}"
            )
        elif st.session_state.conversion_error:
            st.error(f"Format conversion failed: {st.session_state.conversion_error}")


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

    if not tree:
        st.warning("No phylogenetic tree to display.")
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


def perform_pairwise_alignment(seq1, seq2, seq_type, mode="global", open_gap_score=-0.5, extend_gap_score=-0.1):
    """
    Perform pairwise alignment using Biopython's Align.PairwiseAligner.

    Parameters:
        seq1: First sequence record
        seq2: Second sequence record
        seq_type (str): Type of sequences ('DNA' or 'Protein')
        mode (str): Alignment mode ('global', 'local', or 'overlap')
        open_gap_score (float): Score for opening a gap
        extend_gap_score (float): Score for extending a gap

    Returns:
        tuple: (alignment_text, mutations) - Formatted alignment text and list of mutations
    """
    try:
        # Clean sequences by converting to uppercase and removing invalid characters
        if seq_type == "DNA":
            valid_chars = set('ATGCNRYKMSWBDHV')
        else:  # Protein
            valid_chars = set('ACDEFGHIKLMNPQRSTVWYBXZJUO')

        # Clean the sequences by removing invalid characters
        seq1_str = ''.join(c.upper() for c in str(seq1.seq) if c.upper() in valid_chars or c == '-')
        seq2_str = ''.join(c.upper() for c in str(seq2.seq) if c.upper() in valid_chars or c == '-')

        # Create new Seq objects with the cleaned sequences
        cleaned_seq1 = Seq(seq1_str)
        cleaned_seq2 = Seq(seq2_str)

        # Configure the aligner
        aligner = Align.PairwiseAligner()
        aligner.mode = mode

        if seq_type == "DNA":
            aligner.substitution_matrix = substitution_matrices.load("NUC.4.4")
        else:
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

        aligner.open_gap_score = open_gap_score
        aligner.extend_gap_score = extend_gap_score

        # Perform the alignment with cleaned sequences
        alignments = aligner.align(cleaned_seq1, cleaned_seq2)

        if not alignments:
            return "No alignments found. Sequences may be too dissimilar.", []

        alignment = next(alignments)

        # Get the alignment as strings with gaps
        aligned_seq1_str = str(alignment[0])
        aligned_seq2_str = str(alignment[1])

        # Generate match line
        match_line = generate_match_line(aligned_seq1_str, aligned_seq2_str, aligner.substitution_matrix)

        # Report mutations
        mutations = report_mutations(aligned_seq1_str, aligned_seq2_str)

        # Format alignment display
        alignment_text = format_alignment_display(seq1.id, aligned_seq1_str, match_line, seq2.id, aligned_seq2_str, alignment.score)

        return alignment_text, mutations
    except Exception as e:
        error_msg = f"An error occurred during pairwise alignment: {str(e)}"
        print(traceback.format_exc())
        return error_msg, []

def generate_match_line(aligned_seq1, aligned_seq2, substitution_matrix, threshold=1):
    """
    Generate a match line for two aligned sequences.

    Parameters:
        aligned_seq1 (str): First aligned sequence with gaps
        aligned_seq2 (str): Second aligned sequence with gaps
        substitution_matrix: Substitution matrix for scoring
        threshold (int): Threshold for similarity

    Returns:
        str: Match line with | for identical, : for similar, and space for different
    """
    match_line = []
    for a, b in zip(aligned_seq1, aligned_seq2):
        if a == b and a != '-':
            match_line.append('|')  # Identical
        elif a != '-' and b != '-' and are_similar(a, b, substitution_matrix, threshold):
            match_line.append(':')  # Similar
        else:
            match_line.append(' ')  # Different or gap

    return ''.join(match_line)


def are_similar(a, b, substitution_matrix, threshold=1):
    """
    Determine if two residues are similar based on the substitution matrix.

    Parameters:
        a (str): First residue
        b (str): Second residue
        substitution_matrix: Substitution matrix for scoring
        threshold (int): Threshold for similarity

    Returns:
        bool: True if residues are similar, False otherwise
    """
    try:
        if a == '-' or b == '-':
            return False
        score = substitution_matrix[a.upper()][b.upper()]
        return score >= threshold
    except KeyError:
        return False


def format_alignment_display(id1, aligned1, match_line, id2, aligned2, score, interval=10):
    """
    Create a formatted string for alignment display, including match lines and location markers.

    Parameters:
        id1 (str): ID of first sequence
        aligned1 (str): First aligned sequence with gaps
        match_line (str): Line showing matches between sequences
        id2 (str): ID of second sequence
        aligned2 (str): Second aligned sequence with gaps
        score (float): Alignment score
        interval (int): Interval for position markers

    Returns:
        str: Formatted alignment text
    """
    def generate_location_line(aligned_seq, interval=10):
        location = [' '] * len(aligned_seq)
        residue_count = 0
        next_mark = interval
        for i, char in enumerate(aligned_seq):
            if char != '-':
                residue_count += 1
                if residue_count == next_mark:
                    mark_str = str(next_mark)
                    mark_len = len(mark_str)
                    start_pos = i - mark_len + 1
                    if start_pos < 0:
                        start_pos = 0
                    for j, digit in enumerate(mark_str):
                        pos = start_pos + j
                        if pos < len(aligned_seq) and location[pos] == ' ':
                            location[pos] = digit
                    next_mark += interval
        return ''.join(location)

    max_label_length = max(len(id1), len("Match"), len(id2))
    id1_padded = id1.ljust(max_label_length)
    match_padded = "Match".ljust(max_label_length)
    id2_padded = id2.ljust(max_label_length)
    location1 = generate_location_line(aligned1, interval)
    location2 = generate_location_line(aligned2, interval)
    padding = ' ' * (max_label_length + 2)
    location1_padded = padding + location1
    location2_padded = padding + location2
    alignment_text = (
        "Pairwise Alignment:\n"
        f"{location1_padded}\n"
        f"{id1_padded}: {aligned1}\n"
        f"{match_padded}: {match_line}\n"
        f"{id2_padded}: {aligned2}\n"
        f"{location2_padded}\n"
        f"{'Score'.ljust(max_label_length)}: {score:.2f}\n"
    )
    return alignment_text


def report_mutations(aligned1, aligned2):
    """
    Report point mutations between two aligned sequences in the format 'S18I'.

    Parameters:
        aligned1 (str): First aligned sequence with gaps
        aligned2 (str): Second aligned sequence with gaps

    Returns:
        list: List of mutation strings (e.g., 'S18I')
    """
    mutations = []
    position = 0
    for i, (a, b) in enumerate(zip(aligned1, aligned2), start=1):
        if a != '-':
            position += 1
        if a != b and a != '-' and b != '-':
            mutations.append(f"{a}{position}{b}")
    return mutations


def perform_msa(sequences, reference_id, seq_type, output_format):
    """
    Perform Multiple Sequence Alignment using pyFAMSA and report mutations relative to the reference sequence.

    Parameters:
        sequences: List of sequence records to align
        reference_id (str): ID of the reference sequence for mutation detection
        seq_type (str): Type of sequences ('DNA' or 'Protein')
        output_format (str): Format for the MSA output

    Returns:
        tuple: (msa_text, mutations) - MSA text and dictionary of mutations per sequence
    """
    try:
        pyfamsa_sequences = [
            PyFAMSASequence(seq.id.encode(), str(seq.seq).encode()) for seq in sequences
        ]
        aligner = PyFAMSAAligner(guide_tree="upgma")
        msa = aligner.align(pyfamsa_sequences)
        aligned_seq_records = [
            SeqRecord(
                Seq(seq.sequence.decode()),
                id=seq.id.decode(),
                description=""
            )
            for seq in msa
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