import math

import pytest
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.phylogeny import (
    MAX_DISTANCE, alignment_distance_matrix, build_phylogenetic_tree, build_tree_from_alignment,
    compute_distance, correct_distance, p_distance,
)


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def test_p_distance_ignores_gaps():
    assert p_distance("AC-T", "AGGT") == pytest.approx(1 / 3)
    assert p_distance("--", "AA") is None


def test_jukes_cantor_and_kimura():
    assert correct_distance(0.0, "DNA") == 0.0
    assert correct_distance(0.1, "DNA") == pytest.approx(-0.75 * math.log(1 - 0.4 / 3))
    assert correct_distance(0.8, "DNA") == MAX_DISTANCE
    assert correct_distance(0.1, "Protein") == pytest.approx(-math.log(1 - 0.1 - 0.2 * 0.01))


def test_alignment_distance_matrix_symmetric():
    d = alignment_distance_matrix(["ACGT", "ACGA", "TCGA"], "DNA", correction=False)
    assert d[0, 1] == pytest.approx(0.25)
    assert d[1, 0] == d[0, 1]
    assert d[0, 0] == 0.0


def test_tree_lowercase_input():
    assert compute_distance(Seq("mkvlaa"), Seq("MKVLAA"), "Protein") == 0.0


def test_nj_tree_has_all_leaves():
    seqs = [rec("a", "MKVLAAGIVG"), rec("b", "MKVLAAGIVA"), rec("c", "MRVLSAGLVA"), rec("d", "MRILSAGLVA")]
    tree = build_phylogenetic_tree(seqs, "Protein", method="nj")
    assert sorted(t.name for t in tree.get_terminals()) == ["a", "b", "c", "d"]


def test_tree_from_alignment_upgma():
    aln = MultipleSeqAlignment([rec("a", "ACGT-A"), rec("b", "ACGTTA"), rec("c", "TCGTTA")])
    tree = build_tree_from_alignment(aln, "DNA", method="upgma")
    assert sorted(t.name for t in tree.get_terminals()) == ["a", "b", "c"]
