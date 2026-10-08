import io
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
