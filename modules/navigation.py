"""Which analyses apply to the loaded data."""


def available_analyses(has_sequences, has_structures, seq_type):
    options = []
    if has_structures:
        options.append("Extracted Sequences")
    if has_sequences:
        options += ["Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Phylogenetic Tree"]
        if seq_type == "DNA":
            options.append("Translate DNA")
        else:
            options += ["Antibody Prediction", "Sequence Properties"]
        options.append("Convert Formats")
    if has_structures:
        options += ["Structure Viewer", "PDB Residue Renumbering"]
    return options
