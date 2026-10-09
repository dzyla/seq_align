import numpy as np
import pytest

from modules.msa_colors import (CLUSTALX_COLOURS, CONSERVATION_RAMP, HYDRO_RAMP, IDENTITY_SHADES, SCHEMES, WHITE,
                                color_alignment)
from modules.viz import _text_colour, msa_viewer_figure, reference_numbering


def colour_of(coloring, row, col):
    return coloring.colors[coloring.index[row, col]]


def column(seqs_column):
    """Rows of a one-column alignment."""
    return list(seqs_column)


def test_clustalx_rules():
    c = color_alignment(column("GGGG"), "ClustalX")
    assert colour_of(c, 0, 0) == CLUSTALX_COLOURS["ORANGE"]               # glycine always orange
    c = color_alignment(column("LLLLLLLKKK"), "ClustalX")
    assert colour_of(c, 0, 0) == CLUSTALX_COLOURS["BLUE"]                 # 70% hydrophobic
    assert colour_of(c, 9, 0) == WHITE                                    # K: KR only 30%
    assert colour_of(color_alignment(column("CCCC"), "ClustalX"), 0, 0) == CLUSTALX_COLOURS["PINK"]
    c = color_alignment(column("KKKKKKKAAA"), "ClustalX")
    assert colour_of(c, 0, 0) == CLUSTALX_COLOURS["RED"] and colour_of(c, 9, 0) == WHITE
    c = color_alignment(column("NNNNNNDDDD"), "ClustalX")
    assert colour_of(c, 0, 0) == CLUSTALX_COLOURS["GREEN"]                # N > 50%
    assert colour_of(color_alignment(column("--"), "ClustalX"), 0, 0) == WHITE


def test_percent_identity_shades():
    for seqs, expected in (("AAAAAAAAAL", IDENTITY_SHADES[0][1]), ("AAAAAAALLV", IDENTITY_SHADES[1][1]),
                           ("AAAAALLVVW", IDENTITY_SHADES[2][1]), ("AAAALLLVVW", WHITE)):
        c = color_alignment(column(seqs), "Percent identity")
        assert colour_of(c, 0, 0) == expected, seqs
    c = color_alignment(column("AAAAAAAAAL"), "Percent identity")
    assert colour_of(c, 9, 0) == WHITE                                    # non-consensus residue stays white


def test_conservation_and_similarity():
    c = color_alignment(["MK-", "MKA", "MRA", "MRG"], "Conservation")
    assert colour_of(c, 0, 0) == CONSERVATION_RAMP[-1]                    # invariant column
    assert colour_of(c, 0, 2) == WHITE                                    # gap
    c = color_alignment(["LLD-", "ILDA", "DLDA"], "Similarity to reference", reference=1)
    assert [colour_of(c, 1, j) for j in range(3)] == [c.colors[1]] * 3    # reference identical to itself
    assert colour_of(c, 0, 0) == c.colors[2] and colour_of(c, 2, 0) == c.colors[3]   # L~I similar, D vs I not
    assert colour_of(c, 0, 3) == WHITE                                    # gap


def test_hydrophobicity_and_schemes_list():
    c = color_alignment(["IR"], "Hydrophobicity")
    assert colour_of(c, 0, 0) == HYDRO_RAMP[-1] and colour_of(c, 0, 1) == HYDRO_RAMP[0]
    for seq_type, schemes in SCHEMES.items():
        for scheme in schemes:
            c = color_alignment(["ACGT-", "ACGA-"] if seq_type == "DNA" else ["MKV-L", "MRVAL"], scheme, seq_type)
            assert c.index.shape == (2, 5) and c.index.max() < len(c.colors)
    with pytest.raises(ValueError):
        color_alignment(["A"], "Rainbow")


def test_viewer_window_and_reference_ruler():
    seqs = ["M-KVLAAGIVGLLAQWERTYASDFGKLCV" * 2, "MKKVLAAGIVGLLAQWERTYASDFGKLCV" * 2]
    from modules.viz import alignment_to_codes
    ref_numbers = reference_numbering(seqs[0])
    assert ref_numbers[:3] == [1, None, 2]
    coloring = color_alignment(seqs, "ClustalX")
    fig = msa_viewer_figure(alignment_to_codes(seqs), ["a", "b"], coloring, start=5, width=40, ref_numbers=ref_numbers,
                            conservation=np.ones(len(seqs[0])), consensus="x" * len(seqs[0]))
    heat = next(t for t in fig.data if t.type == "heatmap")
    assert np.array(heat.z).shape == (3, 40) and list(heat.y) == ["consensus", "a", "b"]
    letters = next(t for t in fig.data if t.type == "scatter" and t.mode == "text")
    assert len(letters.text) == 3 * 40 and len(letters.textfont.color) == 3 * 40
    assert all(c in ("#ffffff", "#1a1a1a") for c in letters.textfont.color)
    ticks = fig.layout.xaxis2.ticktext
    assert all(int(t) % 10 == 0 for t in ticks)


def test_text_colour_contrasts_with_background():
    assert _text_colour("#134e4a") == "#ffffff"
    assert _text_colour("#0000ff") == "#ffffff"
    assert _text_colour("#ffffff") == "#1a1a1a"
    assert _text_colour("#ccccff") == "#1a1a1a"
    assert _text_colour("#14b8a6") == "#1a1a1a" and _text_colour("#ff0000") == "#1a1a1a"   # light teal, red: dark letters
