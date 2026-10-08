import io

import numpy as np
import pandas as pd
import json
import zipfile
from io import StringIO

from Bio import Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.export import build_session_zip, has_results
from modules.pairwise import align_pair


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def make_tree(newick="(a:0.1,b:0.2);"):
    return Phylo.read(StringIO(newick), "newick")


def test_zip_contents():
    seqs = [rec("a", "MKVLAA"), rec("b", "MKVLAG")]
    state = {
        "pairwise_result": align_pair(seqs[0], seqs[1], "Protein"),
        "msa_aligned": [("a", "MKVLAA"), ("b", "MKVLAG")],
        "tree_result": {"params": {}, "tree": make_tree()},
        "last_pairwise_params": {"ref_id": "a", "target_id": "b"},
    }
    assert has_results(state)
    archive = zipfile.ZipFile(io.BytesIO(build_session_zip(state, seqs, "Protein")))
    names = set(archive.namelist())
    assert {"sequences.fasta", "session.json", "pairwise_alignment.txt", "msa_alignment.fasta", "tree.nwk",
            "README.txt"} <= names
    meta = json.loads(archive.read("session.json"))
    assert meta["seq_type"] == "Protein"
    assert meta["parameters"]["last_pairwise_params"]["ref_id"] == "a"


def test_msa_tree_is_exported_from_its_dict_wrapper():
    seqs = [rec("a", "MKVLAA"), rec("b", "MKVLAG")]
    state = {"msa_tree": {"params": {}, "tree": make_tree("(a:0.3,b:0.4);")}}
    assert has_results(state)
    archive = zipfile.ZipFile(io.BytesIO(build_session_zip(state, seqs, "Protein")))
    assert "msa_tree.nwk" in archive.namelist()
    assert "a" in archive.read("msa_tree.nwk").decode()


def test_has_results_empty():
    assert not has_results({})
    assert not has_results({"msa_aligned": [], "tree_result": None})


def _zip(state, seqs=None):
    seqs = seqs or [rec("a", "MKVLAA"), rec("b", "MKVLAG")]
    return zipfile.ZipFile(io.BytesIO(build_session_zip(state, seqs, "Protein")))


def test_result_parameters_and_msa_view_in_session_json():
    state = {
        "tree_result": {"params": {"method": "nj"}, "tree": make_tree()},
        "msa_tree": {"params": {"replicates": 100}, "tree": make_tree()},
        "orf_results": {"params": {"min_len": 30}, "orfs": [{"Frame": "+1"}]},
        "antibody_results": {"params": {"scheme": "imgt"},
                             "df": pd.DataFrame({"Sequence ID": ["x"], "_labels": [[1]]})},
        "msa_trim": 0.5, "msa_redundancy": 0.9,
    }
    meta = json.loads(_zip(state).read("session.json"))
    assert meta["result_parameters"] == {"tree_result": {"method": "nj"}, "msa_tree": {"replicates": 100},
                                         "orf_results": {"min_len": 30}, "antibody_results": {"scheme": "imgt"}}
    assert meta["msa_view"] == {"max_gap": 0.5, "max_id": 0.9}


def test_result_parameters_skip_missing():
    meta = json.loads(_zip({"msa_aligned": [("a", "MK"), ("b", "MR")]}).read("session.json"))
    assert meta["result_parameters"] == {}
    assert "msa_view" not in meta


def test_consensus_representative_fasta_exported():
    from modules.msa import aligned_to_msa, calculate_representative_sequence
    consensus, closest, diff, closest_id = calculate_representative_sequence(
        aligned_to_msa([("a", "MKV"), ("b", "MKL"), ("c", "MKV")]))
    state = {"consensus_data": {"consensus_record": consensus, "closest_record": closest}}
    text = _zip(state).read("consensus_representative.fasta").decode()
    assert text.startswith(">Consensus\nMKX\n>a\nMKV")


def test_other_result_branches_export_expected_files():
    df = pd.DataFrame({"Sequence ID": ["x"], "_labels": [[1]]})
    state = {
        "best_match_results": (None, [{"Sequence ID": "b", "Score": 1.0}]),
        "clustering_matrix": np.array([[100.0, 50.0], [50.0, 100.0]]),
        "clustering_ids": ["a", "b"],
        "antibody_results": {"params": {}, "df": df},
        "orf_results": {"params": {}, "orfs": [{"Frame": "+1"}]},
    }
    archive = _zip(state)
    assert {"best_match.csv", "clustering_identity_matrix.csv", "antibody_domains.csv", "orfs.csv"} <= set(
        archive.namelist())
    assert "_labels" not in archive.read("antibody_domains.csv").decode()
    assert "Sequence ID" in archive.read("best_match.csv").decode()
    assert archive.read("clustering_identity_matrix.csv").decode().startswith(",a,b")
