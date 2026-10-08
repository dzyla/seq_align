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


@pytest.mark.parametrize("method", ["nj", "upgma"])
def test_identical_sequences_give_tree_with_all_leaves(method):
    tree = build_phylogenetic_tree([rec("a", "MKV"), rec("b", "MKV"), rec("c", "MKV")], "Protein", method=method)
    assert sorted(t.name for t in tree.get_terminals()) == ["a", "b", "c"]


def test_tree_to_newick_roundtrip():
    from io import StringIO
    from Bio import Phylo
    from modules.phylogeny import tree_to_newick
    tree = Phylo.read(StringIO("((A:0.1,B:0.2):0.3,C:0.5);"), "newick")
    text = tree_to_newick(tree)
    assert sorted(t.name for t in Phylo.read(StringIO(text), "newick").get_terminals()) == ["A", "B", "C"]


from modules.phylogeny import bootstrap_tree, tree_figure, tree_layout, tree_splits

GROUPED = [("a", "A" * 19 + "C"), ("b", "A" * 20), ("c", "T" * 19 + "G"), ("d", "T" * 20)]


def test_vectorized_distance_matches_definition():
    d = alignment_distance_matrix(["AC-T", "AGGT", "TC-T"], "DNA", correction=False)
    assert d[0, 1] == pytest.approx(1 / 3)
    assert d[0, 2] == pytest.approx(1 / 3)


def test_bootstrap_supports_true_split():
    tree = bootstrap_tree(GROUPED, "DNA", method="nj", correction=False, replicates=50, seed=1)
    supports = [c.confidence for c in tree.get_nonterminals() if c.confidence is not None]
    assert 100 in supports


def test_tree_splits_rooting_invariant():
    from io import StringIO
    from Bio import Phylo
    t1 = Phylo.read(StringIO("((a,b),(c,d));"), "newick")
    t2 = Phylo.read(StringIO("(a,(b,(c,d)));"), "newick")
    names = frozenset("abcd")
    assert tree_splits(t1, names, "a") == tree_splits(t2, names, "a") == {frozenset("cd")}


def test_tree_figure_has_all_leaves():
    from io import StringIO
    from Bio import Phylo
    tree = Phylo.read(StringIO("((A:0.1,B:0.2):0.3,C:0.5);"), "newick")
    segments, leaves, _ = tree_layout(tree)
    assert sorted(name for _, _, name in leaves) == ["A", "B", "C"]
    fig = tree_figure(tree)
    assert sorted(fig.data[1].text) == ["A", "B", "C"]


def test_bootstrap_tree_empty_alignment_returns_tree():
    tree = bootstrap_tree([("a", ""), ("b", ""), ("c", "")], "DNA", replicates=10)
    assert sorted(t.name for t in tree.get_terminals()) == ["a", "b", "c"]


def test_bootstrap_tree_same_seed_same_supports():
    def supports():
        t = bootstrap_tree(GROUPED, "DNA", correction=False, replicates=20, seed=3)
        return [c.confidence for c in t.get_nonterminals()]
    assert supports() == supports()


def test_bootstrap_supports_survive_newick_round_trip():
    from io import StringIO
    from Bio import Phylo
    from modules.phylogeny import tree_to_newick
    tree = bootstrap_tree(GROUPED, "DNA", method="nj", correction=False, replicates=20, seed=1)
    back = Phylo.read(StringIO(tree_to_newick(tree)), "newick")
    inner = [c for c in back.get_nonterminals() if c is not back.root]
    assert inner
    for clade in back.get_nonterminals():
        assert clade.name is None
    assert all(isinstance(c.confidence, (int, float)) for c in inner)
