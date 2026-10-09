"""Which analyses apply to the loaded data."""

NEEDS_MULTIPLE = ("Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Phylogenetic Tree")
MIN_ANCESTRAL_SEQUENCES = 4


def available_analyses(has_sequences, has_structures, seq_type, n_sequences=None):
    """Analyses for the loaded data; n_sequences == 1 drops those that need at least two sequences."""
    options = []
    if has_structures:
        options.append("Extracted Sequences")
    if has_sequences:
        if n_sequences != 1:
            options += list(NEEDS_MULTIPLE)
        options.append("Coevolution")               # its upload source works with any dataset
        if n_sequences is not None and n_sequences >= MIN_ANCESTRAL_SEQUENCES:
            options.append("Ancestral Reconstruction")
        if seq_type == "DNA":
            options += ["Translate DNA"]
        else:
            options += ["Antibody Prediction", "Sequence Properties"]
        options.append("Convert Formats")
    if has_structures:
        options += ["Structure Viewer", "PDB Residue Renumbering"]
    return options
