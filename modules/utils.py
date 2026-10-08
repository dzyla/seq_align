import copy

import streamlit as st
from typing import List

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

# Every analysis result and its parameter snapshot. Cleared together when the dataset changes.
RESULT_KEYS = {
    "pairwise_result": None,
    "last_selected_seqs": None,
    "msa_aligned": None,
    "consensus_data": None,
    "msa_tree": None,
    "best_match_results": None,
    "clustering_matrix": None,
    "clustering_ids": None,
    "clustering_msa": None,
    "converted_data": None,
    "conversion_error": None,
    "antibody_results": None,
    "orf_results": None,
    "pdb_renumber_results": None,
    "pdb_renumber_hash": None,
    "tree_result": None,
    "last_msa_params": {},
    "last_pairwise_params": {},
    "best_match_params": {},
    "last_conversion_params": {},
}

STALE_MESSAGE = "Settings changed since the last run. Click the run button to update the results."


def init_session_state():
    """Create every session key the app reads, without overwriting existing values."""
    defaults = {
        **RESULT_KEYS,
        "seq_type": None,
        "dataset_fp": None,
        "input_key": None,
        "input_cache": None,
        "structures": {},
        "extra_structures": {},
        "fetched": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = copy.copy(value)


def reset_all_results(state=None):
    """Restore every result key to its default (fresh copies of mutable defaults)."""
    state = st.session_state if state is None else state
    for key, default in RESULT_KEYS.items():
        state[key] = copy.copy(default)


def track_dataset(fp, state=None):
    """Reset all results when the dataset fingerprint changes. Returns True on change."""
    state = st.session_state if state is None else state
    if state.get("dataset_fp") == fp:
        return False
    reset_all_results(state)
    state["dataset_fp"] = fp
    return True


def stale_notice(message=STALE_MESSAGE):
    """Tell the user the shown inputs differ from the ones used for the stored result."""
    st.info(message, icon=":material/refresh:")


def have_params_changed(current_params: dict, last_params_key: str) -> bool:
    """
    Checks if the current parameters are different from the last stored parameters.
    """
    return st.session_state.get(last_params_key) != current_params

def reset_results(keys_to_reset: List[str]):
    """
    Resets specified session state variables to their default values.
    """
    for key in keys_to_reset:
        if key.endswith('_params'):
            st.session_state[key] = {}
        else:
            st.session_state[key] = None

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
        "GenBank": ["gb", "gbk", "genbank"],
        "Newick": ["nwk", "newick", "tree"],
        "PDB": ["pdb", "ent"],
        "mmCIF": ["cif", "mmcif", "mcif"]  # Added mcif as alternative extension
    }
    return format_extensions.get(format_name, [])
