from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.overview import EXAMPLES, dataset_summary, summary_table
from modules.parsers import parse_sequences_from_text
from modules.seqtools import detect_seq_type


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id)


def test_detect_dna_rna_protein():
    assert detect_seq_type(["ATGCNNATGC"]) == "DNA"
    assert detect_seq_type(["AUGCCC"]) == "DNA"
    assert detect_seq_type(["MKTAYIAKQR"]) == "Protein"
    assert detect_seq_type([]) == "Protein"


def test_detect_mixed_majority():
    assert detect_seq_type(["ATGCATGCATGCATGC", "ATGCATGCATGC", "MKV"]) == "DNA"


def test_dataset_summary_and_table():
    records = [rec("a", "MKV"), rec("b", "MKVLA"), rec("c", "M" * 50)]
    assert dataset_summary(records) == {"count": 3, "min_length": 3, "median_length": 5, "max_length": 50}
    table = summary_table(records, preview=10)
    assert list(table.columns) == ["ID", "Length", "Preview"]
    assert table.loc[2, "Preview"] == "M" * 10 + "…"


def test_examples_parse_and_detect():
    expected = {"Globins (protein)": "Protein", "pUC19 plasmid (DNA)": "DNA", "Antibody VH/VL (protein)": "Protein"}
    for name, kind in expected.items():
        records, error = parse_sequences_from_text(EXAMPLES[name].read_text())
        assert error is None, (name, error)
        assert detect_seq_type([str(r.seq) for r in records]) == kind
