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

from modules.pairwise import AlignmentError, align_pair, format_result_text, location_line, report_mutations

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


def test_location_line_respects_start_offset():
    assert location_line("A" * 15, 10, start=20) == "        30     "


def test_report_mutations_with_rotation():
    assert report_mutations("CDEAK", "CDEWK", ref_rotation=2, ref_length=5) == ["A1W"]


def test_format_result_text_contains_score_and_notes():
    result = align_pair(rec("ref", CIRC), rec("t", CIRC[5:] + CIRC[:5]), "DNA", is_circular2=True)
    text = format_result_text(result)
    assert "Score" in text
    assert "circularly shifted" in text
