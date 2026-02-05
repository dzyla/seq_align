import streamlit as st
import os
from io import StringIO
from Bio import SeqIO, Phylo
from modules.utils import init_session_state, reset_results, get_file_extensions
from modules.parsers import parse_sequences_from_text, parse_sequences_from_file, parse_sequences_from_structure
from modules.pairwise import pairwise_alignment_section
from modules.msa import msa_section
from modules.best_match import best_match_finder_section
from modules.clustering import sequence_clustering_section
from modules.antibody import antibody_prediction_section
from modules.conversion import format_conversion_section
from modules.phylogeny import phylogenetic_tree_section, build_phylogenetic_tree
from modules.dna import dna_translation_section

def handle_input(input_format):
    """
    Handles user input for sequences and trees based on the selected format.
    """
    sequences = None
    tree = None
    uploaded_file = None

    if input_format == "Text (FASTA)":
        seq_text = st.sidebar.text_area(
            "Paste Sequences Here (FASTA Format)",
            height=250,
            placeholder=">Sequence1\nATGCGTA...\n>Sequence2\nATGCGTC...",
            help="Enter your sequences in FASTA format"
        )
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
        uploaded_files = st.sidebar.file_uploader(
            f"Upload {input_format} File(s)",
            type=get_file_extensions(input_format),
            key=f"uploader_{input_format}",
            help=f"Upload one or more {input_format} files to extract protein sequences",
            accept_multiple_files=True
        )
        if uploaded_files:
            # Process multiple files
            all_sequences = []
            for uploaded_file in uploaded_files:
                sequences, error = parse_sequences_from_structure(uploaded_file, input_format)
                if error:
                    st.sidebar.error(f"Error in file {uploaded_file.name}: {error}")
                else:
                    all_sequences.extend(sequences)

            if all_sequences:
                st.sidebar.success(f"Successfully extracted {len(all_sequences)} sequences from {len(uploaded_files)} file(s).")
                st.session_state.sequences = all_sequences
        else:
            # Clear sequences if no files are uploaded
            if 'sequences' in st.session_state:
                st.session_state.sequences = None
    elif input_format == "Newick":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Newick Tree File",
            type=["nwk", "newick", "tree"],
            key="uploader_newick",
            help="Upload a Newick format tree file for visualization"
        )
        if uploaded_file:
            if 'last_tree_file' not in st.session_state or st.session_state.last_tree_file != uploaded_file.name:
                st.session_state.last_tree_file = uploaded_file.name
                try:
                    tree = Phylo.read(uploaded_file, "newick")
                    st.session_state.tree = tree
                    st.sidebar.success("Successfully loaded Newick tree.")
                    # Reset tree visualization cache
                    if 'tree_newick' in st.session_state: del st.session_state.tree_newick
                except Exception as e:
                    st.sidebar.error(f"Error reading Newick file: {e}")
            else:
                tree = st.session_state.tree
    else:
        uploaded_file = st.sidebar.file_uploader(
            f"Upload {input_format} File",
            type=get_file_extensions(input_format),
            key=f"uploader_{input_format}",
            help=f"Upload sequence file in {input_format} format"
        )
        if uploaded_file:
            if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
                st.session_state.last_file = uploaded_file.name
                sequences, error = parse_sequences_from_file(uploaded_file, input_format)
                if error:
                    st.sidebar.error(error)
                else:
                    st.sidebar.success(f"Successfully loaded {len(sequences)} sequences.")
                    st.session_state.sequences = sequences
            else:
                sequences = st.session_state.sequences

    return sequences, tree, uploaded_file


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
        ("Text (FASTA)", "FASTA", "Clustal", "GenBank", "Newick", "PDB", "mmCIF"),
        help="Choose the format of your input sequences."
    )

    sequences, tree, uploaded_file = handle_input(input_format)

    # Retrieve from session state if not loaded from input
    if sequences is None and 'sequences' in st.session_state:
        sequences = st.session_state.sequences
    if tree is None and 'tree' in st.session_state:
        tree = st.session_state.tree

    if (sequences and len(sequences) > 0) or tree:
        seq_type = None
        if sequences:
            seq_type = st.sidebar.selectbox(
                "🔬 Select Sequence Type",
                ("DNA", "Protein"),
                help="Choose whether your sequences are DNA or Protein.",
                index=1
            )
            if 'seq_type' not in st.session_state or st.session_state.seq_type != seq_type:
                st.session_state.seq_type = seq_type
                # Invalidate results if sequence type changes
                keys_to_reset = [
                    'msa_result', 'mutations', 'msa_image', 'msa_letters',
                    'consensus_data', 'alignment_text', 'pairwise_mutations',
                    'last_msa_params', 'last_pairwise_params',
                    'best_match_results', 'best_match_params'
                ]
                reset_results(keys_to_reset)

        alignment_mode = None
        if sequences:
            options = ("Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Antibody Prediction", "Convert Formats", "Phylogenetic Tree", "Translate DNA")
            if input_format in ["PDB", "mmCIF"]:
                options = ("Extracted Sequences", "Sequence Clustering", "Antibody Prediction", "Phylogenetic Tree")
            alignment_mode = st.sidebar.selectbox("🛠️ Select Analysis", options)
        elif tree:
            alignment_mode = "Phylogenetic Tree"

        if alignment_mode == "Extracted Sequences":
            st.header("📄 Extracted FASTA Sequences")
            fasta_io = StringIO()
            SeqIO.write(sequences, fasta_io, "fasta")
            fasta_str = fasta_io.getvalue()
            st.text_area("FASTA Sequences", value=fasta_str, height=500)

            file_name = "extracted_sequences.fasta"
            # The 'uploaded_file' variable is only available for single file uploads.
            # For multiple files, we use a generic name.
            if uploaded_file and hasattr(uploaded_file, 'name'):
                 file_name = f"{os.path.splitext(os.path.basename(uploaded_file.name))[0]}_sequences.fasta"

            st.download_button(label="📥 Download FASTA", data=fasta_str, file_name=file_name, mime="text/plain")
        elif alignment_mode == "Pairwise":
            pairwise_alignment_section(sequences, seq_type)
        elif alignment_mode == "MSA":
            msa_section(sequences, seq_type)
        elif alignment_mode == "Best Match Finder":
            best_match_finder_section(sequences, seq_type)
        elif alignment_mode == "Sequence Clustering":
            sequence_clustering_section(sequences, seq_type)
        elif alignment_mode == "Antibody Prediction":
            antibody_prediction_section(sequences)
        elif alignment_mode == "Translate DNA":
            if seq_type == "DNA":
                dna_translation_section(sequences)
            else:
                st.warning("Translation is only available for DNA sequences. Please select DNA as sequence type.")
        elif alignment_mode == "Convert Formats":
            format_conversion_section(sequences, input_format)
        elif alignment_mode == "Phylogenetic Tree":
            if tree:
                phylogenetic_tree_section(tree)
            elif sequences:
                with st.spinner("Building phylogenetic tree..."):
                    new_tree = build_phylogenetic_tree(sequences, seq_type)
                    if new_tree:
                        st.session_state.tree = new_tree
                        if 'tree_ascii' in st.session_state: del st.session_state.tree_ascii
                        if 'tree_newick' in st.session_state: del st.session_state.tree_newick
                        phylogenetic_tree_section(new_tree)
                    else:
                        st.error("Could not build the phylogenetic tree.")
    else:
        st.info("Please upload sequences or a Newick tree in the sidebar to begin.")


if __name__ == "__main__":
    main()
