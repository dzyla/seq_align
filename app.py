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
from modules.navigation import available_analyses
from modules.pairwise import pairwise_alignment_section
from modules.parsers import (
    make_unique_ids, parse_sequences_from_file, parse_sequences_from_structure, parse_sequences_from_text,
)
from modules.pdb_renumber import pdb_renumber_section
from modules.phylogeny import TREE_METHODS, TreeError, build_phylogenetic_tree, phylogenetic_tree_section
from modules.overview import EXAMPLES, load_example_callback, show_dataset_overview
from modules.seqtools import detect_seq_type, fingerprint
from modules.structures import STRUCTURE_FORMATS, StructureFile
from modules.utils import get_file_extensions, init_session_state, reset_all_results, stale_notice, track_dataset

INPUT_FORMATS = ("Text (FASTA)", "FASTA", "Clustal", "GenBank", "Newick", "PDB", "mmCIF")


def _digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _load_once(key, loader, state=None):
    """Run `loader` only when the input identity `key` changes; otherwise reuse its cached result."""
    state = st.session_state if state is None else state
    if state.get("input_key") != key:
        result = loader()
        state["input_cache"] = result
        state["input_key"] = key
    return state["input_cache"]


def _clear_structures(state=None):
    """Empty the structure store and drop the input cache that produced it (only if it was non-empty)."""
    state = st.session_state if state is None else state
    if state.get("structures"):
        state["structures"] = {}
        state["input_key"] = None
        state["input_cache"] = None


def _unique_name(name, existing):
    """Return `name`, or `name` with _2, _3, ... before the extension, so it is not in `existing`."""
    if name not in existing:
        return name
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    n = 2
    while True:
        candidate = f"{stem}_{n}{dot}{ext}"
        if candidate not in existing:
            return candidate
        n += 1


def _load_structures(files, input_format, all_models):
    records, errors, store = [], [], {}
    for f in files:
        seqs, err = parse_sequences_from_structure(f, input_format, all_models=all_models)
        if err:
            errors.append(f"{f.name}: {err}")
        else:
            records.extend(seqs)
        name = _unique_name(f.name, store)
        store[name] = StructureFile(name=name, data=f.getvalue(), fmt=input_format)
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
    st.set_page_config(page_title="Advanced Sequence Alignment Tool", page_icon="🔬", layout="wide")
    init_session_state()

    st.title("🔬 Advanced Sequence Alignment Tool")
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
Align, compare and annotate **DNA** or **protein** sequences and structures.

1. **Load data** in the sidebar: paste FASTA, upload a sequence or structure file, or load an example.
2. **Check the sequence type.** It is detected automatically; change it if needed.
3. **Pick an analysis** and adjust its settings.
4. **Run** it and **download** the results.

Pairwise alignments support Global, Local and Overlap (free end gaps) modes. MSA uses FAMSA.
Mutation positions are always reported in reference-sequence coordinates.
""")

    st.sidebar.header("📥 Upload and Settings")
    c_ex, c_btn = st.sidebar.columns([3, 1], vertical_alignment="bottom")
    c_ex.selectbox("Example data", ["—"] + list(EXAMPLES), key="example_choice")
    c_btn.button("Load", key="load_example", on_click=load_example_callback,
                 disabled=st.session_state.get("example_choice", "—") == "—")
    input_format = st.sidebar.selectbox("Select Input Format", INPUT_FORMATS, key="input_format",
                                        help="Choose the format of your input sequences.")
    sequences, tree = handle_input(input_format)

    if tree is not None:
        phylogenetic_tree_section(tree)
        return
    if not sequences:
        st.info("Please upload sequences or a Newick tree in the sidebar to begin.")
        return

    changed = track_dataset(fingerprint(sequences))
    detected = detect_seq_type([str(s.seq) for s in sequences])
    if changed or "seq_type_select" not in st.session_state:
        st.session_state.seq_type_select = detected
    seq_type = st.sidebar.selectbox("🔬 Sequence Type", ("DNA", "Protein"), key="seq_type_select",
                                    help=f"Auto-detected as {detected}. Change it if the detection is wrong.")
    if st.session_state.get("seq_type") != seq_type:
        st.session_state.seq_type = seq_type
        reset_all_results()
    structures = st.session_state.get("structures", {})
    show_dataset_overview(sequences, seq_type)

    options = available_analyses(bool(sequences), bool(structures), seq_type)
    if st.session_state.get("analysis_select") not in options:
        st.session_state.analysis_select = options[0]
    alignment_mode = st.sidebar.selectbox("🛠️ Select Analysis", options, key="analysis_select")

    if alignment_mode == "Extracted Sequences":
        st.header("📄 Extracted FASTA Sequences")
        fasta_io = StringIO()
        SeqIO.write(sequences, fasta_io, "fasta")
        fasta_str = fasta_io.getvalue()
        st.text_area("FASTA Sequences", value=fasta_str, height=500)
        stem = next(iter(structures.values())).stem if len(structures) == 1 else "extracted"
        st.download_button(label="📥 Download FASTA", data=fasta_str, file_name=f"{stem}_sequences.fasta",
                           mime="text/plain", on_click="ignore")
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
        dna_translation_section(sequences)
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
