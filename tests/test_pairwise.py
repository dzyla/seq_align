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
