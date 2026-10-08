import hashlib
from io import StringIO

import streamlit as st
from Bio import Phylo, SeqIO

from modules.antibody import antibody_prediction_section
from modules.best_match import best_match_finder_section
from modules.clustering import sequence_clustering_section
from modules.conversion import format_conversion_section
from modules.dna import dna_translation_section
from modules.msa import msa_section
from modules.pairwise import pairwise_alignment_section
from modules.parsers import (
    make_unique_ids, parse_sequences_from_file, parse_sequences_from_structure, parse_sequences_from_text,
)
from modules.pdb_renumber import pdb_renumber_section
from modules.phylogeny import TREE_METHODS, TreeError, build_phylogenetic_tree, phylogenetic_tree_section
from modules.seqtools import fingerprint
from modules.structures import STRUCTURE_FORMATS, StructureFile
from modules.utils import get_file_extensions, init_session_state, reset_all_results, stale_notice, track_dataset

INPUT_FORMATS = ("Text (FASTA)", "FASTA", "Clustal", "GenBank", "Newick", "PDB", "mmCIF")


def _digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _load_once(key, loader, state=None):
    """Run `loader` only when the input identity `key` changes; otherwise reuse its cached result."""
    state = st.session_state if state is None else state
    if state.get("input_key") != key:
        state["input_key"] = key
        state["input_cache"] = loader()
    return state["input_cache"]


def _clear_structures(state=None):
    """Empty the structure store and drop the input cache that produced it (only if it was non-empty)."""
    state = st.session_state if state is None else state
    if state.get("structures"):
        state["structures"] = {}
        state["input_key"] = None
        state["input_cache"] = None


def _load_structures(files, input_format, all_models):
    records, errors, store = [], [], {}
    for f in files:
        seqs, err = parse_sequences_from_structure(f, input_format, all_models=all_models)
        if err:
            errors.append(f"{f.name}: {err}")
        else:
            records.extend(seqs)
        store[f.name] = StructureFile(name=f.name, data=f.getvalue(), fmt=input_format)
    st.session_state.structures = store
    return (make_unique_ids(records) if records else None), ("\n".join(errors) or None)


def _read_newick(uploaded):
    try:
        return Phylo.read(StringIO(uploaded.getvalue().decode("utf-8")), "newick"), None
    except Exception as e:
        return None, f"Error reading Newick file: {e}"


def handle_input(input_format):
    """Return (sequences, tree) for the current sidebar input. Parsing reruns only when content changes."""
    result, error = None, None
    if input_format not in STRUCTURE_FORMATS:
        _clear_structures()

    if input_format == "Text (FASTA)":
        seq_text = st.sidebar.text_area(
            "Paste Sequences Here (FASTA Format)", height=250,
            placeholder=">Sequence1\nATGCGTA...\n>Sequence2\nATGCGTC...",
            help="Enter your sequences in FASTA format", key="fasta_text",
        )
        if seq_text:
            result, error = _load_once(("text", _digest(seq_text.encode())),
                                       lambda: parse_sequences_from_text(seq_text))
    elif input_format in STRUCTURE_FORMATS:
        all_models = st.sidebar.checkbox("Extract all models (NMR ensembles)", value=False)
        files = st.sidebar.file_uploader(
            f"Upload {input_format} File(s)", type=get_file_extensions(input_format),
            key=f"uploader_{input_format}", accept_multiple_files=True,
            help=f"Upload one or more {input_format} files to extract protein sequences",
        )
        if files:
            key = (input_format, all_models, tuple((f.name, _digest(f.getvalue())) for f in files))
            result, error = _load_once(key, lambda: _load_structures(files, input_format, all_models))
        else:
            _clear_structures()
    elif input_format == "Newick":
        uploaded = st.sidebar.file_uploader(
            "Upload Newick Tree File", type=["nwk", "newick", "tree"], key="uploader_newick",
            help="Upload a Newick format tree file for visualization",
        )
        if uploaded:
            result, error = _load_once(("newick", _digest(uploaded.getvalue())), lambda: _read_newick(uploaded))
    else:
        uploaded = st.sidebar.file_uploader(
            f"Upload {input_format} File", type=get_file_extensions(input_format),
            key=f"uploader_{input_format}", help=f"Upload sequence file in {input_format} format",
        )
        if uploaded:
            result, error = _load_once((input_format, _digest(uploaded.getvalue())),
                                       lambda: parse_sequences_from_file(uploaded, input_format))

    if error:
        st.sidebar.error(error)
    if input_format == "Newick":
        return None, result
    if result:
        st.sidebar.success(f"Successfully loaded {len(result)} sequences.")
    return result, None


def main():
    st.set_page_config(page_title="Advanced Sequence Alignment Ability", layout="wide")
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
    input_format = st.sidebar.selectbox("Select Input Format", INPUT_FORMATS, key="input_format",
                                        help="Choose the format of your input sequences.")
    sequences, tree = handle_input(input_format)

    if tree is not None:
        phylogenetic_tree_section(tree)
        return
    if not sequences:
        st.info("Please upload sequences or a Newick tree in the sidebar to begin.")
        return

    track_dataset(fingerprint(sequences))
    structures = st.session_state.get("structures", {})

    seq_type = st.sidebar.selectbox("🔬 Select Sequence Type", ("DNA", "Protein"), index=1, key="seq_type_select",
                                    help="Choose whether your sequences are DNA or Protein.")
    if st.session_state.get("seq_type") != seq_type:
        st.session_state.seq_type = seq_type
        reset_all_results()

    if input_format in STRUCTURE_FORMATS:
        options = ("Extracted Sequences", "Sequence Clustering", "Antibody Prediction", "Phylogenetic Tree",
                   "PDB Residue Renumbering")
    else:
        options = ("Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Antibody Prediction",
                   "Convert Formats", "Phylogenetic Tree", "Translate DNA")
    alignment_mode = st.sidebar.selectbox("🛠️ Select Analysis", options, key="analysis_select")

    if alignment_mode == "Extracted Sequences":
        st.header("📄 Extracted FASTA Sequences")
        fasta_io = StringIO()
        SeqIO.write(sequences, fasta_io, "fasta")
        fasta_str = fasta_io.getvalue()
        st.text_area("FASTA Sequences", value=fasta_str, height=500)
        stem = next(iter(structures.values())).stem if len(structures) == 1 else "extracted"
        st.download_button(label="📥 Download FASTA", data=fasta_str, file_name=f"{stem}_sequences.fasta",
                           mime="text/plain")
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
        format_conversion_section(sequences, input_format, seq_type)
    elif alignment_mode == "PDB Residue Renumbering":
        pdb_renumber_section(structures)
    elif alignment_mode == "Phylogenetic Tree":
        c_m, c_c = st.columns(2)
        tree_method = c_m.selectbox("Tree method", list(TREE_METHODS))
        correction = c_c.checkbox("Correct for multiple substitutions", value=True,
                                  help="Jukes–Cantor (DNA) or Kimura (protein) distance correction.")
        params = {"dataset": st.session_state.dataset_fp, "seq_type": seq_type,
                  "method": tree_method, "correction": correction}
        if st.button("Build Tree", key="build_tree"):
            with st.spinner("Building phylogenetic tree..."):
                try:
                    tree = build_phylogenetic_tree(sequences, seq_type, TREE_METHODS[tree_method], correction)
                    st.session_state.tree_result = {"params": params, "tree": tree}
                except TreeError as e:
                    st.session_state.tree_result = None
                    st.error(str(e))
        stored = st.session_state.get("tree_result")
        if stored:
            if stored["params"] != params:
                stale_notice()
            else:
                phylogenetic_tree_section(stored["tree"])


if __name__ == "__main__":
    main()
