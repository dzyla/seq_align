from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.conversion import convert_format


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def test_fasta_to_genbank_sets_molecule_type():
    data, error = convert_format([rec("s1", "ATGC"), rec("s2", "ATGCAA")], "genbank", seq_type="DNA")
    assert error is None
    assert "LOCUS" in data and "DNA" in data


def test_embl_protein():
    data, error = convert_format([rec("p1", "MKV")], "embl", seq_type="Protein")
    assert error is None
    assert "PROTEIN" in data.upper()


def test_alignment_format_needs_equal_lengths():
    data, error = convert_format([rec("s1", "ATGC"), rec("s2", "ATGCAA")], "clustal", seq_type="DNA")
    assert data is None
    assert "aligned" in error


def test_input_records_not_mutated():
    r = rec("s1", "ATGC")
    convert_format([r], "genbank", seq_type="DNA")
    assert "molecule_type" not in r.annotations
