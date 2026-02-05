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

def init_session_state():
    """
    Initialize session state variables if they don't exist.
    This function ensures all required session state variables are properly initialized
    to prevent KeyError exceptions and enable proper state tracking.
    """
    # Define the default values for session state variables
    defaults = {
        "msa_result": None,
        "mutations": None,
        "msa_image": None,
        "msa_letters": None,
        "consensus_data": None,
        "alignment_text": None,
        "pairwise_mutations": None,
        "sequences": None,
        "seq_type": None,
        "last_file": None,
        "last_tree_file": None,
        "tree": None,
        "tree_newick": None,
        "converted_data": None,
        "conversion_error": None,
        "last_conversion_params": {},
        "selected_seqs": None,
        "last_msa_params": {},
        "last_pairwise_params": {},
        "last_text_hash": None,
        "msa_tree": None,
        "best_match_results": None,
        "best_match_params": {},
    }

    # Initialize session state variables if they don't exist
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
