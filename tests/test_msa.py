import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.msa import MSAError, mutations_from_alignment, perform_msa, run_msa, serialize_alignment


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def test_run_msa_uppercases_and_aligns():
    aligned = dict(run_msa([("a", "mkvlaagivg"), ("b", "MKVLAGIVG")], "Protein"))
    assert set(aligned) == {"a", "b"}
    assert len(aligned["a"]) == len(aligned["b"])
    assert aligned["a"].replace("-", "") == "MKVLAAGIVG"


def test_run_msa_rejects_empty_sequence():
    with pytest.raises(MSAError, match="empty"):
        run_msa([("a", "MKV"), ("b", "123")], "Protein")


def test_phylip_relaxed_keeps_long_ids():
    seqs = [rec("sample_0001_x", "MKVLAAGIVG"), rec("sample_0001_y", "MKVLAGIVG"), rec("sample_0002", "MKVAAGIVG")]
    text, _ = perform_msa(seqs, "sample_0001_x", "Protein", "phylip-relaxed")
    assert "sample_0001_y" in text


def test_strict_phylip_failure_raises_msa_error():
    with pytest.raises(MSAError):
        serialize_alignment([("sample_0001_x", "MK"), ("sample_0001_y", "MK")], "phylip")


def test_mutations_case_insensitive():
    muts = mutations_from_alignment([("ref", "MKV-L"), ("q", "MRVAL")], "ref")
    assert muts == {"q": [(2, "K", "R")]}
