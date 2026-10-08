from modules.seqtools import clean_sequence


def test_clean_sequence_protein_maps_rare_residues_and_case():
    assert clean_sequence("mkuOj*x", "Protein") == "MKCKX*X"


def test_clean_sequence_protein_drops_gaps_unless_kept():
    assert clean_sequence("MK-VL", "Protein") == "MKVL"
    assert clean_sequence("MK-VL", "Protein", keep_gaps=True) == "MK-VL"


def test_clean_sequence_dna_converts_rna_and_drops_junk():
    assert clean_sequence("acgu-n 12", "DNA") == "ACGTN"


def test_clean_sequence_accepts_seq_objects():
    from Bio.Seq import Seq
    assert clean_sequence(Seq("atg"), "DNA") == "ATG"
