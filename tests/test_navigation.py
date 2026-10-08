from modules.navigation import available_analyses


def test_protein_sequences():
    assert available_analyses(True, False, "Protein") == [
        "Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Phylogenetic Tree",
        "Antibody Prediction", "Sequence Properties", "Convert Formats",
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


def test_structures_offer_viewer():
    assert "Structure Viewer" in available_analyses(True, True, "Protein")
    assert "Structure Viewer" not in available_analyses(True, False, "Protein")


def test_dna_tools_listed():
    options = available_analyses(True, False, "DNA")
    assert options[-3:] == ["Translate DNA", "DNA Tools", "Convert Formats"]


def test_single_sequence_omits_multi_sequence_analyses():
    options = available_analyses(True, False, "DNA", n_sequences=1)
    assert options == ["Translate DNA", "DNA Tools", "Convert Formats"]
    for name in ("Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Phylogenetic Tree"):
        assert name not in options


def test_two_sequences_unchanged():
    assert available_analyses(True, False, "DNA", n_sequences=2) == available_analyses(True, False, "DNA")
