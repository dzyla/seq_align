from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.seqtools import fingerprint
from modules.utils import RESULT_KEYS, reset_all_results, track_dataset


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id)


def test_fingerprint_sensitive_to_content():
    a = fingerprint([rec("x", "MKV"), rec("y", "MKL")])
    assert a == fingerprint([rec("x", "MKV"), rec("y", "MKL")])
    assert a != fingerprint([rec("x", "MKV"), rec("y", "MKI")])
    assert a != fingerprint([rec("x", "MKV"), rec("z", "MKL")])


def test_reset_all_results_restores_defaults():
    state = {"pairwise_result": object(), "last_msa_params": {"a": 1}}
    reset_all_results(state)
    for key, default in RESULT_KEYS.items():
        assert state[key] == default
    state["last_msa_params"]["x"] = 1
    assert RESULT_KEYS["last_msa_params"] == {}   # defaults are copied, never shared


def test_track_dataset_resets_only_on_change():
    state = {}
    assert track_dataset("fp1", state) is True
    state["pairwise_result"] = "kept"
    assert track_dataset("fp1", state) is False
    assert state["pairwise_result"] == "kept"
    assert track_dataset("fp2", state) is True
    assert state["pairwise_result"] is None


def test_same_name_new_content_reparses():
    import app
    calls = []
    state = {}

    def loader():
        calls.append(1)
        return ("parsed", None)

    key_v1 = ("FASTA", app._digest(b">a\nMKV\n"))
    key_v2 = ("FASTA", app._digest(b">a\nMKL\n"))
    app._load_once(key_v1, loader, state)
    app._load_once(key_v1, loader, state)
    app._load_once(key_v2, loader, state)
    assert len(calls) == 2


def test_clear_structures_invalidates_input_cache():
    import app
    state = {"structures": {"a.pdb": object()}, "input_key": ("PDB",), "input_cache": ("x", None)}
    app._clear_structures(state)
    assert state == {"structures": {}, "input_key": None, "input_cache": None}


def test_clear_structures_noop_when_already_empty():
    import app
    state = {"structures": {}, "input_key": ("text", "h"), "input_cache": ("x", None)}
    app._clear_structures(state)
    assert state["input_key"] == ("text", "h")
    assert state["input_cache"] == ("x", None)


def test_load_once_failing_loader_keeps_previous_key():
    import pytest
    import app
    state = {}
    app._load_once("k1", lambda: ("one", None), state)

    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        app._load_once("k2", boom, state)
    assert state["input_key"] == "k1"
    assert app._load_once("k1", boom, state) == ("one", None)


def test_unique_name():
    import app
    assert app._unique_name("x.pdb", set()) == "x.pdb"
    assert app._unique_name("x.pdb", {"x.pdb"}) == "x_2.pdb"
    assert app._unique_name("x.pdb", {"x.pdb", "x_2.pdb"}) == "x_3.pdb"
    assert app._unique_name("noext", {"noext"}) == "noext_2"
