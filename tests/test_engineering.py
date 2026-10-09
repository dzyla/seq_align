import pytest

from modules.engineering import (MOTIF_LIBRARY, compile_motifs, consensus_suggestions, find_motifs, prosite_to_regex,
                                 site_summary)

ROWS = [("t", "MKVAE-LW"), ("a", "MKILEALW"), ("b", "MKILEGLW"), ("c", "MRILE-LF"), ("d", "MKILE-LW")]


def test_consensus_mismatch_suggested_in_target_numbering():
    table = consensus_suggestions(ROWS, "t")
    assert {"V3I", "A4L"} <= set(table["Suggestion"])
    row = table.set_index("Suggestion").loc["V3I"]
    assert row["Consensus frequency"] == pytest.approx(0.8) and row["Sequences"] == 5


def test_consensus_respects_frequency_occupancy_and_gaps():
    table = consensus_suggestions(ROWS, "t", min_freq=0.85)
    assert "V3I" not in set(table["Suggestion"])          # 0.8 < 0.85
    gappy = [("t", "MKA"), ("a", "--L"), ("b", "--L"), ("c", "--L"), ("d", "MKL")]
    assert set(consensus_suggestions(gappy, "t")["Suggestion"]) == {"A3L"}  # columns 1-2: occupancy 0.4 < 0.5
    with_gap = [("t", "M-W"), ("a", "MKW"), ("b", "MKW")]
    assert consensus_suggestions(with_gap, "t").empty     # target gap is never suggested


def test_consensus_sorted_by_score():
    table = consensus_suggestions(ROWS, "t")
    assert list(table["Score"]) == sorted(table["Score"], reverse=True)


@pytest.mark.parametrize("pattern,regex", [
    ("N-{P}-[ST]-{P}", "N[^P][ST][^P]"), ("R-x-[KR]-R", "R.[KR]R"), ("x(2,4)-C", ".{2,4}C"),
    ("C-x(2)-C", "C.{2}C"), ("<M-x(2)", "^M.{2}"), ("[ST]-x-[RK]>", "[ST].[RK]$"), ("N-G.", "NG"),
])
def test_prosite_to_regex(pattern, regex):
    assert prosite_to_regex(pattern) == regex


@pytest.mark.parametrize("bad", ["", "N-{P", "N--G", "1-2", "[st]"])
def test_prosite_invalid(bad):
    with pytest.raises(ValueError):
        prosite_to_regex(bad)


def test_library_compiles():
    assert set(compile_motifs(MOTIF_LIBRARY)) == set(MOTIF_LIBRARY)


def test_find_motifs_overlapping_case_insensitive_and_columns():
    rows = [("x", "aa-aA"), ("y", "-AAAA")]
    hits = find_motifs(rows, compile_motifs({"AA": "A-A"}))
    x = hits[hits["Sequence"] == "x"]
    assert list(x["Start"]) == [1, 2, 3] and list(x["Column"]) == [0, 1, 3]
    y = hits[hits["Sequence"] == "y"]
    assert list(y["Start"]) == [1, 2, 3] and list(y["Column"]) == [1, 2, 3]


def test_regex_syntax_and_bad_regex():
    assert "C..C" in compile_motifs({"c": "C..C"}, "regex")["c"].pattern
    with pytest.raises(ValueError, match="bad"):
        compile_motifs({"bad": "C(("}, "regex")


def test_site_summary_counts_and_reference_position():
    rows = [("r", "-NGT"), ("s", "ANGT"), ("t", "ADGT")]
    hits = find_motifs(rows, compile_motifs({"NG": "N-G"}))
    summary = site_summary(hits, rows, "r")
    assert summary.iloc[0]["Sequences with site"] == "2/3" and summary.iloc[0]["Reference position"] == 1
    assert site_summary(hits.iloc[0:0], rows, "r").empty


def test_dna_alignment_consensus_has_no_blosum():
    # Review Focus: DNA alignments must not crash the protein-oriented tabs
    rows = [("t", "ATGA"), ("a", "ATGC"), ("b", "ATGC"), ("c", "ATGC")]
    table = consensus_suggestions(rows, "t", seq_type="DNA")
    assert list(table["Suggestion"]) == ["A4C"] and table["BLOSUM62"].isna().all()


def test_zero_length_regex_matches_are_ignored():
    # Review Focus: a custom regex that can match the empty string must not flood the table
    hits = find_motifs([("x", "CAAC")], compile_motifs({"z": "A*"}, "regex"))
    assert list(hits["Match"]) == ["AA", "A"]
