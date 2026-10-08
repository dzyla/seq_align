import math
from io import StringIO

import pytest
from Bio import Phylo

from modules.phylogeny import TreeError
from modules.tree_layout import assign_clades, compute_geometry, prepare_tree

NEWICK = "((A:0.1,B:0.2)90:0.3,(C:0.25,(D:0.1,E:0.15)70:0.05)100:0.2,F:0.6);"


def tree():
    return Phylo.read(StringIO(NEWICK), "newick")


def test_rectangular_rows_and_parent_midpoints():
    geo = compute_geometry(prepare_tree(tree(), "as_built", ladderize=False), "rectangular")
    assert [leaf.b for leaf in geo.leaves] == [0, 1, 2, 3, 4, 5]
    assert [leaf.name for leaf in geo.leaves] == list("ABCDEF")
    parent_ab = next(n for n in geo.nodes if not n.is_leaf and n.support == 90)
    assert parent_ab.b == pytest.approx(0.5)
    assert geo.max_depth == pytest.approx(0.6)


def test_circular_cladogram_puts_tips_on_one_circle_with_even_spacing():
    geo = compute_geometry(prepare_tree(tree(), "as_built", ladderize=False), "circular", cladogram=True)
    radii = {round(leaf.a, 9) for leaf in geo.leaves}
    assert radii == {geo.max_depth}
    step = 2 * math.pi / (len(geo.leaves) + 1)
    angles = [leaf.b for leaf in geo.leaves]
    assert all(a - b == pytest.approx(step) for a, b in zip(angles, angles[1:]))


def test_unrooted_edges_keep_branch_lengths():
    t = prepare_tree(tree(), "as_built", ladderize=False)
    geo = compute_geometry(t, "unrooted")
    lengths = sorted(round(math.hypot(x1 - x0, y1 - y0), 9) for x0, y0, x1, y1, _ in geo.edges)
    expected = sorted(round(c.branch_length, 9) for c in t.find_clades() if c is not t.root)
    assert lengths == expected
    assert len(geo.leaf_angles) == len(geo.leaves) == 6


def test_outgroup_rooting_makes_outgroup_a_root_child():
    t = prepare_tree(tree(), "outgroup", outgroup=["F"])
    assert "F" in [c.name for c in t.root.clades]


@pytest.mark.parametrize("outgroup", [[], ["Z"], list("ABCDEF")])
def test_invalid_outgroup_raises_tree_error(outgroup):
    with pytest.raises(TreeError):
        prepare_tree(tree(), "outgroup", outgroup=outgroup)


def test_ladderize_puts_smaller_clades_first():
    t = prepare_tree(tree(), "as_built", ladderize=True)
    sizes = [c.count_terminals() for c in t.root.clades]
    assert sizes == sorted(sizes)


def test_prepare_tree_does_not_modify_the_input():
    original = tree()
    prepare_tree(original, "outgroup", outgroup=["A"])
    assert [c.name for c in original.root.clades] == [None, None, "F"]


@pytest.mark.parametrize("k", [2, 3, 4])
def test_assign_clades_gives_k_groups_covering_all_leaves(k):
    t = prepare_tree(tree(), "midpoint")
    clades = assign_clades(t, k)
    leaf_groups = [clades[leaf] for leaf in t.get_terminals()]
    assert sorted(set(leaf_groups)) == list(range(k))
    assert leaf_groups == sorted(leaf_groups)  # numbered top to bottom


def test_assign_clades_one_group_means_no_colouring():
    assert assign_clades(prepare_tree(tree()), 1) == {}


def test_unknown_layout_rejected():
    with pytest.raises(ValueError):
        compute_geometry(prepare_tree(tree()), "radial")


def caterpillar(n_leaves):
    """Ladder-like tree built iteratively (depth n_leaves - 1)."""
    from Bio.Phylo.BaseTree import Clade, Tree
    node = Clade(branch_length=0.1, name="A0")
    for i in range(1, n_leaves):
        node = Clade(branch_length=0.1, clades=[node, Clade(branch_length=0.1, name=f"A{i}")])
    node.branch_length = 0.0
    return Tree(root=node, rooted=True)


@pytest.mark.parametrize("layout", ["rectangular", "circular"])
def test_deep_caterpillar_tree_lays_out_and_restores_recursion_limit(layout):
    import sys
    before = sys.getrecursionlimit()
    t = prepare_tree(caterpillar(400), "midpoint")
    geo = compute_geometry(t, layout)
    assert len(geo.leaves) == 400
    assert sys.getrecursionlimit() == before


def test_assign_clades_never_exceeds_k_with_trifurcating_root():
    t = prepare_tree(Phylo.read(StringIO("(A:0.1,B:0.2,C:0.3);"), "newick"), "as_built")
    two = assign_clades(t, 2)
    assert len(set(two.values())) <= 2
    assert len(set(assign_clades(t, 3).values())) == 3
