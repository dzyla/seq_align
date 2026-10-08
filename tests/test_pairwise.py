from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.pairwise import (
    DEFAULT_EXTEND_GAP,
    DEFAULT_OPEN_GAP,
    calculate_alignment_score,
    get_aligner,
    perform_pairwise_alignment,
)

REF = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKW"
    "KRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIK"
)


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def test_default_gap_scores_are_needle_like():
    aligner = get_aligner("Protein")
    assert DEFAULT_OPEN_GAP == -10.0
    assert DEFAULT_EXTEND_GAP == -0.5
    assert aligner.open_gap_score == -10.0
    assert aligner.extend_gap_score == -0.5


def test_overlap_mode_aligns_fragment():
    text, mutations = perform_pairwise_alignment(
        rec("ref", REF), rec("frag", "LEERLGLIEVQAPWLSRV"), "Protein", "overlap"
    )
    assert text.startswith("Pairwise Alignment"), text
    assert mutations == ["I36W"]


def test_selenocysteine_and_lowercase_do_not_crash():
    text, _ = perform_pairwise_alignment(rec("a", "mkuvlA"), rec("b", "MKCVLA"), "Protein")
    assert text.startswith("Pairwise Alignment"), text


def test_calculate_alignment_score_ignores_gaps_in_input():
    aligner = get_aligner("Protein")
    gapped = calculate_alignment_score(aligner, rec("a", "MK-VL"), rec("b", "MKVL"), "Protein")
    plain = calculate_alignment_score(aligner, rec("a", "MKVL"), rec("b", "MKVL"), "Protein")
    assert gapped == plain


import pytest

from modules.pairwise import AlignmentError, align_pair, format_result_text, report_mutations

CIRC = "GATTACAGGCCTTAAGCTAGCATCGGATCCAGTC"


def test_local_alignment_reports_reference_coordinates():
    result = align_pair(rec("ref", REF), rec("q", "RQLEERLGLIEVQAPWLSRVGDG"), "Protein", "local")
    assert result.ref_start == 20
    assert result.mutations == ["I36W"]


def test_circular_mutations_use_original_reference_coordinates():
    mutated = CIRC[:2] + "C" + CIRC[3:]          # T3C
    rotated = mutated[7:] + mutated[:7]
    for circ_ref, circ_tgt in [(True, False), (False, True), (True, True)]:
        result = align_pair(
            rec("ref", CIRC), rec("t", rotated), "DNA", "global",
            is_circular1=circ_ref, is_circular2=circ_tgt,
        )
        assert result.mutations == ["T3C"], (circ_ref, circ_tgt, result.mutations)


def test_align_pair_rejects_empty():
    with pytest.raises(AlignmentError, match="empty"):
        align_pair(rec("a", ""), rec("b", "ATGG"), "DNA")


def test_report_mutations_with_rotation():
    assert report_mutations("CDEAK", "CDEWK", ref_rotation=2, ref_length=5) == ["A1W"]


def test_format_result_text_contains_score_and_notes():
    result = align_pair(rec("ref", CIRC), rec("t", CIRC[5:] + CIRC[:5]), "DNA", is_circular2=True)
    text = format_result_text(result)
    assert "Score" in text
    assert "circularly shifted" in text


from modules.pairwise import compute_alignment_stats, format_alignment_blocks, mutation_lollipop


def test_alignment_stats():
    stats = compute_alignment_stats("MKV-L", "MRVAL", "|:| |")
    assert stats["length"] == 5
    assert stats["identical"] == 3
    assert stats["similar"] == 4
    assert stats["gaps"] == 1
    assert stats["identity_pct"] == 60.0


def test_blocks_wrap_and_use_reference_coordinates():
    result = align_pair(rec("ref", REF), rec("q", "RQLEERLGLIEVQAPWLSRVGDG"), "Protein", "local")
    text = format_alignment_blocks(result, width=10)
    first = text.splitlines()[0].split()
    assert first[0] == "ref" and first[1] == "21"
    assert len([l for l in text.splitlines() if l.startswith("ref")]) == 3


def test_blocks_align_with_long_ids():
    long_id = "a_really_long_sequence_identifier"
    result = align_pair(rec(long_id, "MKVLAAGIVG"), rec("q", "MKVLAAGIVG"), "Protein")
    lines = format_alignment_blocks(result).splitlines()
    assert lines[0].index("MKVLAAGIVG") == lines[2].index("MKVLAAGIVG")


def test_lollipop_none_without_mutations():
    assert mutation_lollipop([], 100) is None
    fig = mutation_lollipop(["K2R", "V30A"], 100)
    assert list(fig.data[1].x) == [2, 30]


def test_lollipop_labels_only_for_few_mutations():
    many = [f"A{i}G" for i in range(1, 31)]
    assert mutation_lollipop(many, 100).data[1].mode == "markers"
    assert mutation_lollipop(["K2R", "V30A", "A5G"], 100).data[1].mode == "markers+text"


def test_lollipop_height_is_substitution_score():
    from modules.pairwise import load_matrix
    fig = mutation_lollipop(["W2C"], 10, "Protein")
    assert list(fig.data[1].y) == [load_matrix("BLOSUM62")["W"]["C"]] == [-2.0]
    assert fig.layout.yaxis.title.text == "Substitution score (BLOSUM62)"


def test_blocks_query_labels_use_original_coordinates_when_circular():
    import random
    random.seed(1)
    ref = "".join(random.choice("ACGT") for _ in range(60))
    target = ref[7:] + ref[:7]
    result = align_pair(rec("ref", ref), rec("t", target), "DNA", is_circular2=True)
    assert result.query_rotation
    first_q = [l for l in format_alignment_blocks(result).splitlines() if l.startswith("t ")][0].split()
    expected = ((1 - 1 + result.query_rotation) % result.query_length) + 1
    assert int(first_q[1]) == expected != 1
