from modules.navigation import available_analyses


def test_protein_sequences():
    assert available_analyses(True, False, "Protein") == [
        "Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Phylogenetic Tree",
        "Antibody Prediction", "Convert Formats",
    ]


def test_dna_sequences_get_translation_not_antibody():
    options = available_analyses(True, False, "DNA")
    assert "Translate DNA" in options
    assert "Antibody Prediction" not in options


def test_structures_add_extraction_and_renumbering():
    options = available_analyses(True, True, "Protein")
    assert options[0] == "Extracted Sequences"
    assert options[-1] == "PDB Residue Renumbering"
    assert "Pairwise" in options
