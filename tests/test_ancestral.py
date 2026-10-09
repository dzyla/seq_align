import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import modules.ancestral as anc
from modules.ancestral import (ASRError, IQTreeResult, altall_sequence, ancestor_options, ancestor_sequence,
                               compare_to_extant, mrca_node, parse_report, parse_state, parse_tree,
                               reconstruct_gaps, uncertain_sites)
from modules.tree_layout import prepare_tree

DATA = Path(__file__).parent / "data" / "asr"


def read_fasta(path):
    rows, name = [], None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
            rows.append([name, ""])
        elif line.strip():
            rows[-1][1] += line.strip()
    return [tuple(r) for r in rows]


def fixture_result():
    alphabet, states = parse_state((DATA / "globins.state").read_text())
    model, loglik = parse_report((DATA / "globins.iqtree").read_text())
    return IQTreeResult(parse_tree((DATA / "globins.treefile").read_text()), states, alphabet, model, loglik,
                        "IQ-TREE multicore version 2.0.3", {})


ROWS = read_fasta(DATA / "globins_aln.fasta")


def test_parse_state_report_and_tree_labels():
    r = fixture_result()
    assert r.alphabet == "ARNDCQEGHILKMFPSTWYV" and sorted(r.states) == ["Node1", "Node2", "Node3", "Node4"]
    states, post = r.states["Node4"]
    assert len(states) == len(ROWS[0][1]) == post.shape[0] and post.shape[1] == 20
    assert np.allclose(post.sum(axis=1), 1, atol=1e-3)
    assert (r.model, round(r.loglik, 2)) == ("LG+G4", -1478.58)
    node4 = next(c for c in r.tree.get_nonterminals() if c.name == "Node4")
    assert node4.confidence == 48.0                     # 'Node4/48' split into name + support


def test_mrca_and_single_leaf_rejected():
    r = fixture_result()
    tree = prepare_tree(r.tree, "midpoint")
    assert mrca_node(tree, ["HBB_HUMAN", "HBB1_MOUSE"], r.states) == "Node4"
    with pytest.raises(ASRError, match="at least two"):
        mrca_node(tree, ["HBB_HUMAN"], r.states)
    with pytest.raises(ASRError, match="root of the displayed tree"):
        mrca_node(tree, [t.name for t in tree.get_terminals()], r.states)
    labels = [o[2] for o in ancestor_options(tree, r.states)]
    assert any(l.startswith("Node4: 2 sequences") for l in labels)


def test_reconstruct_gaps_rules():
    from io import StringIO
    from Bio import Phylo
    tree = Phylo.read(StringIO("(((a,b)N2,c)N1,d)N0;"), "newick")
    rows = [("a", "M-K"), ("b", "M-K"), ("c", "MRK"), ("d", "M-R")]
    gaps = reconstruct_gaps(tree, rows)
    assert gaps["N2"].tolist() == [True, False, True]      # gap shared by the a+b clade -> absent
    assert gaps["N1"][1] == False                           # noqa: E712 - c alone has the residue
    rows2 = [("a", "MRK"), ("b", "M-K"), ("c", "MRK"), ("d", "MRK")]
    assert reconstruct_gaps(tree, rows2)["N2"].tolist() == [True, True, True]   # single-leaf gap -> present


def test_uncertain_sites_altall_and_compare():
    r = fixture_result()
    tree = prepare_tree(r.tree, "midpoint")
    gaps = reconstruct_gaps(tree, ROWS)
    seq, pp, cols = ancestor_sequence(r, gaps, "Node4")
    assert len(seq) == len(pp) == len(cols) and 0 <= pp.min() and pp.max() <= 1
    table = uncertain_sites(r, gaps, "Node4", 0.8, 0.2)
    assert (table["Posterior"] < 0.8).all() and len(table) == int((pp < 0.8).sum())
    alt = altall_sequence(r, gaps, "Node4", 0.8, 0.2)
    assert len(alt) == len(seq) and alt != seq
    changed = [k for k, (a, b) in enumerate(zip(seq, alt)) if a != b]
    assert all(pp[k] < 0.8 for k in changed)
    subs, ins, dels, identity = compare_to_extant(r, gaps, "Node4", ROWS, "HBB_HUMAN")
    assert 0.8 < identity < 1.0 and all(label[0] != label[-1] for label, _ in subs)


def test_missing_binary_and_timeout(monkeypatch):
    monkeypatch.setattr(anc, "find_iqtree", lambda: None)
    with pytest.raises(ASRError, match="not installed"):
        anc.run_iqtree(ROWS, "Protein")
    monkeypatch.setattr(anc, "find_iqtree", lambda: ("/bin/iqtree", "IQ-TREE 2"))

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="iqtree", timeout=1)
    monkeypatch.setattr(anc.subprocess, "run", slow)
    with pytest.raises(ASRError, match="did not finish"):
        anc.run_iqtree(ROWS, "Protein", timeout=1)


def test_too_few_sequences_rejected(monkeypatch):
    monkeypatch.setattr(anc, "find_iqtree", lambda: ("/bin/iqtree", "IQ-TREE 2"))
    with pytest.raises(ASRError, match="4-300"):
        anc.run_iqtree(ROWS[:3], "Protein")


@pytest.mark.skipif(anc.find_iqtree() is None, reason="IQ-TREE not installed")
def test_end_to_end_iqtree_run_keeps_original_ids():
    r = anc.run_iqtree(ROWS, "Protein", "LG+G4")
    assert sorted(t.name for t in r.tree.get_terminals()) == sorted(i for i, _ in ROWS)
    assert len(r.states) == 4
    for states, post in r.states.values():
        assert np.allclose(post.sum(axis=1), 1, atol=1e-3)
    assert "IQ-TREE" in r.version and {"iqtree.treefile", "iqtree.state"} <= set(r.files)


@pytest.mark.skipif(not __import__("os").environ.get("CI"), reason="only enforced in CI (GitHub Actions sets CI)")
def test_ci_has_the_cloud_system_binaries():
    # packages.txt installs these on Streamlit Cloud; CI installs the same Debian/Ubuntu packages, so a missing
    # binary must fail here instead of silently skipping the IQ-TREE and DSSP tests
    from modules.espript import find_mkdssp
    assert anc.find_iqtree() is not None, "iqtree package not found (iqtree2/iqtree3/iqtree)"
    assert find_mkdssp() is not None, "dssp package not found (mkdssp)"


def test_check_site_count():
    from modules.ancestral import check_site_count
    check_site_count({"Node1": ("MKV", None)}, 3)
    with pytest.raises(ASRError, match="2 sites for an alignment of 3 columns"):
        check_site_count({"Node1": ("MK", None)}, 3)


def test_iqtree_major_version():
    from modules.ancestral import iqtree_major
    assert iqtree_major("IQ-TREE multicore version 2.0.3 for Linux 64-bit built Dec 20 2020") == 2
    assert iqtree_major("IQ-TREE version 3.0.1 for Linux x86 64-bit built May 5 2025") == 3
    assert iqtree_major("IQ-TREE multicore version 1.6.12 for Linux 64-bit built Aug 15 2019") == 1
    assert iqtree_major("IQ-TREE something") is None


def test_iqtree_1x_treated_as_missing(monkeypatch):
    monkeypatch.setattr(anc.shutil, "which", lambda name: "/opt/iqtree" if name == "iqtree" else None)

    class Done:
        stdout = "IQ-TREE multicore version 1.6.12 for Linux 64-bit built Aug 15 2019\n"
    monkeypatch.setattr(anc.subprocess, "run", lambda *a, **k: Done())
    assert anc.find_iqtree() is None
    with pytest.raises(ASRError, match="IQ-TREE 2 or newer is required"):
        anc.run_iqtree(ROWS, "Protein")


def test_failure_message_lists_last_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(anc, "find_iqtree", lambda: ("/bin/iqtree", "IQ-TREE 2"))
    log = ("IQ-TREE starting\nWARNING: 3 sequences contain more than 50% gaps\nERROR: Sequence s0001 has an "
           "invalid character\nERROR: Alignment has errors\nsome trailing line\n")

    class Done:
        returncode, stdout, stderr = 2, "", ""

    def fake_run(cmd, cwd, **k):
        (Path(cwd) / "run.log").write_text(log)
        return Done()
    monkeypatch.setattr(anc.subprocess, "run", fake_run)
    with pytest.raises(ASRError) as e:
        anc.run_iqtree(ROWS, "Protein")
    message = str(e.value)
    assert "invalid character" in message and "Alignment has errors" in message and "50% gaps" in message


def test_compare_rna_extant_counts_u_as_t():
    from io import StringIO
    from Bio import Phylo
    tree = Phylo.read(StringIO("((a,b)N1,(c,d)N2)N0;"), "newick")
    result = IQTreeResult(tree, {"N1": ("ACGTA", np.ones((5, 4), np.float32))}, "ACGT", "GTR+G4", None, "IQ-TREE 2")
    rows = [("a", "ACGUA"), ("b", "ACGUA"), ("c", "ACGUC"), ("d", "ACGUC")]
    subs, ins, dels, identity = compare_to_extant(result, {}, "N1", rows, "a")
    assert subs == [] and identity == 1.0
    subs, *_ = compare_to_extant(result, {}, "N1", rows, "c")
    assert [label for label, _ in subs] == ["A5C"]


def test_fasta_header_names_the_node_once():
    from modules.ancestral_page import fasta_record
    assert fasta_record("Node4", "ML ancestor", "LG+G4", "-1478.58", "MKV") == \
        ">Node4 ML ancestor; model=LG+G4; lnL=-1478.58\nMKV\n"
