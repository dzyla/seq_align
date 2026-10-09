import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from modules.coevolution import (ALPHABET, MF_MAX_COLUMNS, PLM_MAX_COLUMNS, contact_precision, mean_field_dca,
                                 parse_alignment, plm_dca, prepare_msa, sequence_weights, structure_distances,
                                 top_pairs)
from modules.structures import StructureFile

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"


def test_a3m_insertions_removed_and_query_first():
    rows = parse_alignment(">query\nMKV-L\n>hit1 desc\nMkkKVAL\n>hit2\nM.KV-L\n")
    assert rows == [("query", "MKV-L"), ("hit1", "MKVAL"), ("hit2", "MKV-L")]


def test_stockholm_blocks_concatenate():
    text = "# STOCKHOLM 1.0\nseqA MKV\nseqB MRV\n#=GC SS_cons ...\n\nseqA LLE\nseqB LIE\n//\n"
    assert parse_alignment(text) == [("seqA", "MKVLLE"), ("seqB", "MRVLIE")]


def test_prepare_drops_query_gaps_length_mismatch_and_gappy_rows():
    rows = [("q", "M-KVL"), ("a", "MAKVL"), ("short", "MKV"), ("gappy", "--K--"), ("b", "MRKIL")]
    p = prepare_msa(rows)
    assert p.query_seq == "MKVL" and p.n_dropped_length == 1 and p.n_dropped_gappy == 1
    assert p.msa.shape == (3, 4) and "".join(ALPHABET[k] for k in p.msa[1]) == "MKVL"


def test_query_kept_when_gappy_and_subsampling_is_deterministic():
    rows = [("q", "M----")] + [(f"s{i}", "MKVLA") for i in range(50)]
    p1 = prepare_msa(rows, max_seqs=10, seed=1)
    p2 = prepare_msa(rows, max_seqs=10, seed=1)
    assert p1.query_seq == "M" and len(p1.msa) == 10 and p1.n_subsampled == 41
    assert np.array_equal(p1.msa, p2.msa)


def test_sequence_weights_by_hand():
    # a, b identical; c differs at 3 of 4 positions -> weights 1/2, 1/2, 1
    msa = prepare_msa([("a", "MKVL"), ("b", "MKVL"), ("c", "MRIA")]).msa
    assert sequence_weights(msa, theta=0.8).tolist() == [0.5, 0.5, 1.0]


def _coupled_msa(n=2000, L=30, i=5, j=20, seed=0):
    rng = np.random.default_rng(seed)
    msa = rng.integers(0, 20, size=(n, L)).astype(np.int8)
    pairs = np.array([[0, 5], [3, 9], [11, 2], [17, 14]], dtype=np.int8)
    pick = pairs[rng.integers(0, len(pairs), size=n)]
    msa[:, i], msa[:, j] = pick[:, 0], pick[:, 1]
    return msa


@pytest.mark.parametrize("method", [mean_field_dca, plm_dca])
def test_synthetic_coupling_ranked_first(method):
    msa = _coupled_msa()
    scores = method(msa, sequence_weights(msa))
    assert top_pairs(scores, min_sep=1, k=1)[0][:2] == (5, 20)


def test_contact_precision_on_toy_matrices():
    scores = np.zeros((10, 10))
    scores[0, 9] = scores[9, 0] = 3.0
    scores[1, 8] = scores[8, 1] = 2.0
    D = np.full((10, 10), 20.0)
    D[0, 9] = D[9, 0] = 5.0
    D[1, 8] = D[8, 1] = np.nan          # unmodelled pairs are skipped
    assert contact_precision(scores, D, min_sep=6, top_k=1) == 1.0
    assert contact_precision(scores, D, min_sep=6, top_k=2) == 0.5


def test_structure_distances_crambin_helix():
    sf = StructureFile(name="1crn.pdb", data=(DATA / "1crn.pdb").read_bytes(), fmt="PDB")
    query = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
    D, mapping, coords = structure_distances(sf, "A", query)
    assert D.shape == (46, 46) and len(mapping) == 46
    assert 4.0 < D[7, 11] < 7.5              # i, i+4 within the first helix (residues 7-19)
    assert np.allclose(np.diag(D), 0)


def test_limits_are_cloud_sized():
    assert PLM_MAX_COLUMNS == 200 and MF_MAX_COLUMNS == 400


@pytest.mark.parametrize("method,L,n,limit_gb", [("plm_dca", 200, 1000, 1.5), ("mean_field_dca", 400, 2000, 1.5)])
def test_memory_stays_within_cloud_budget(method, L, n, limit_gb):
    code = (f"import sys, resource, numpy as np; sys.path.insert(0, {str(ROOT)!r});"
            f"from modules.coevolution import {method}, sequence_weights;"
            f"msa = np.random.default_rng(0).integers(0, 21, size=({n}, {L})).astype(np.int8);"
            f"{method}(msa, sequence_weights(msa));"
            "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert int(done.stdout.strip().splitlines()[-1]) / 1e6 < limit_gb


def test_aligned_fasta_and_stockholm_keep_lowercase_residues():
    assert parse_alignment(">a\nmkvl\n>b\nmrvl\n") == [("a", "MKVL"), ("b", "MRVL")]
    text = "# STOCKHOLM 1.0\nseqA MKV.L\nseqB mrvaL\n//\n"
    assert parse_alignment(text) == [("seqA", "MKV-L"), ("seqB", "MRVAL")]


def test_plm_dca_robust_with_many_iterations():
    msa = _coupled_msa(n=300, L=12, i=2, j=9)
    scores = plm_dca(msa, sequence_weights(msa), iters=100)
    assert np.isfinite(scores).all()
    assert top_pairs(scores, min_sep=1, k=1)[0][:2] == (2, 9)


def test_hhsuite_annotation_rows_dropped():
    text = ">ss_pred\nCCHHHHCC\n>ss_conf\n98877789\n>query\nMKVLAAGI\n>hit\nMKVL-AGI\n"
    rows = parse_alignment(text)
    assert rows == [("query", "MKVLAAGI"), ("hit", "MKVL-AGI")]
    assert prepare_msa(rows).query_seq == "MKVLAAGI"
    assert parse_alignment(">Consensus\nMKV\n>sa_x\nABC\n>q\nMKV\n>h\nMRV\n") == [("q", "MKV"), ("h", "MRV")]


def test_clustal_alignment_parsed():
    text = ("CLUSTAL W (1.83) multiple sequence alignment\n\n"
            "seqA            MKV-LA\nseqB            MRVQLA\n                * * **\n\n"
            "seqA            GI\nseqB            GV\n                * \n")
    assert parse_alignment(text) == [("seqA", "MKV-LAGI"), ("seqB", "MRVQLAGV")]
