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


from modules.clustering import calculate_sequence_coverage
from modules.msa import aligned_to_msa


def test_aligned_to_msa():
    aln = aligned_to_msa([("a", "MK-V"), ("b", "MKLV")])
    assert len(aln) == 2
    assert aln.get_alignment_length() == 4


def test_coverage_vectorized():
    assert calculate_sequence_coverage(["MK-V", "M--V", "MKLV"]).tolist() == [3, 2, 1, 3]


import shutil

from modules.msa import alignment_views, remove_redundant, trim_columns


def test_trim_columns():
    trimmed, kept = trim_columns([("a", "M-KV"), ("b", "M-KL"), ("c", "MAK-")], 0.5)
    assert kept == [0, 2, 3]
    assert dict(trimmed)["a"] == "MKV"


def test_remove_redundant_keeps_reference():
    pairs = [("b", "MKVLA"), ("a", "MKVLA"), ("c", "WWWWW")]
    kept, removed = remove_redundant(pairs, 0.95, keep_ids=("a",))
    assert [i for i, _ in kept] == ["a", "c"]
    assert removed == {"b": "a"}


def test_alignment_views_keep_coordinates():
    pairs = [("ref", "M-KV"), ("x", "MAKV"), ("y", "MAKV")]
    rows, view, removed = alignment_views(pairs, "ref", max_gap_fraction=0.3, max_identity=0.99)
    assert dict(rows)["ref"] == "M-KV"
    assert [i for i, _ in rows] == ["ref", "x"]
    assert dict(view)["ref"] == "MKV"
    assert removed == {"y": "x"}


def test_mafft_missing_raises(monkeypatch):
    import modules.msa as msa
    monkeypatch.setattr(msa.shutil, "which", lambda name: None)
    with pytest.raises(MSAError, match="MAFFT"):
        run_msa([("a", "MKV"), ("b", "MKL")], "Protein", method="MAFFT")


@pytest.mark.skipif(shutil.which("mafft") is None, reason="MAFFT not installed")
def test_mafft_alignment_preserves_ids_and_residues():
    aligned = run_msa([("a", "mkvlaagivg"), ("b", "MKVLAGIVG")], "Protein", method="MAFFT")
    assert [i for i, _ in aligned] == ["a", "b"]
    assert aligned[0][1].replace("-", "") == "MKVLAAGIVG"


def test_single_residue_sequence_rejected_before_aligning(monkeypatch):
    import modules.msa as msa

    def boom(*a, **k):
        raise AssertionError("must not reach FAMSA")
    monkeypatch.setattr(msa, "run_famsa", boom)
    with pytest.raises(MSAError, match="too short.*a"):
        run_msa([("a", "M"), ("b", "MKV")], "Protein")


@pytest.mark.parametrize("pairs", [
    [("a", "AAAA"), ("b", "WWWW")],
    [("a", "MKVL"), ("b", "WWWW")],
    [("a", "MKVLAAGIVGLLAQWERTYASDFGKLCV"), ("b", "HHHHHHHHHH")],
])
def test_famsa_crash_inputs_do_not_kill_the_process(pairs):
    try:
        aligned = run_msa(pairs, "Protein")
        assert [i for i, _ in aligned] == [i for i, _ in pairs]
    except MSAError:
        pass  # a clean error is acceptable; the point is that the test process survives


def test_famsa_keeps_input_ids():
    aligned = run_msa([("seq one", "MKVLAAGIVG"), ("seq_two", "MKVLAGIVG")], "Protein")
    assert [i for i, _ in aligned] == ["seq one", "seq_two"]


def test_famsa_works_from_an_unguarded_main_script(tmp_path):
    # `streamlit run app.py` executes the page as an unguarded __main__; FAMSA isolation must not re-run it.
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    script = tmp_path / "page.py"
    script.write_text(
        f"import sys\nsys.path.insert(0, {str(repo)!r})\n"
        "from modules.msa import run_msa\n"
        "print(run_msa([('a', 'MKVLAAGIVGLL'), ('b', 'MKVLAGIVGLL')], 'Protein'))\n"
    )
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "('a', 'MKVLAAGIVGLL')" in done.stdout


def test_famsa_crash_is_reported_as_crash():
    with pytest.raises(MSAError, match="crashed"):
        run_msa([("a", "AAAA"), ("b", "WWWW")], "Protein")
