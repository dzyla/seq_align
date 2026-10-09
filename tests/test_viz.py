import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from modules.viz import (
    ALPHABET, GAP_CODE, alignment_to_codes, msa_figure_matplotlib, msa_figure_plotly, palette, render_mode,
)


def test_codes_are_uint8_and_map_gaps_and_unknowns():
    codes = alignment_to_codes(["MK-v", "MK@V"])
    assert codes.dtype == np.uint8
    assert codes[0, 2] == GAP_CODE
    assert ALPHABET[codes[1, 2]] == "X"          # unknown character
    assert codes[0, 3] == codes[1, 3]            # case-insensitive


def test_unequal_lengths_rejected():
    try:
        alignment_to_codes(["MK", "M"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_matplotlib_colors_match_palette():
    codes = alignment_to_codes(["ACDEFGHIKL", "MNPQRSTVWY"])
    fig = msa_figure_matplotlib(codes, ["a", "b"], "Protein")
    image = fig.axes[0].images[0]
    colors = palette("Protein")
    for code in np.unique(codes):
        assert to_hex(image.cmap(image.norm(code))) == colors[code]
    plt.close(fig)


def test_gap_is_white_in_both_palettes():
    assert palette("Protein")[GAP_CODE] == "#ffffff"
    assert palette("DNA")[GAP_CODE] == "#ffffff"


def test_render_mode_thresholds():
    assert render_mode(10, 100, "Plotly (Interactive)") == "interactive_hover"
    assert render_mode(500, 1000, "Plotly (Interactive)") == "interactive"
    assert render_mode(2000, 2000, "Plotly (Interactive)") == "static"
    assert render_mode(10, 10, "Matplotlib (Static)") == "static"


def test_plotly_figure_has_no_hover_text_when_disabled():
    codes = alignment_to_codes(["MK", "ML"])
    assert msa_figure_plotly(codes, ["a", "b"], "Protein", hover=False).data[0].text is None
    assert msa_figure_plotly(codes, ["a", "b"], "Protein", hover=True).data[0].text is not None


def test_matplotlib_xticks_are_one_based():
    codes = alignment_to_codes(["ACDEFGHIKL"])
    fig = msa_figure_matplotlib(codes, ["a"], "Protein")
    assert tuple(fig.axes[0].images[0].get_extent()) == (0.5, 10.5, 0.5, -0.5)
    plt.close(fig)


def test_matplotlib_legend_is_compact_and_figure_is_tall_enough():
    seqs = ["MKVLAAGIVG--KRDEST", "MKVLA-GIVGWWKRDEST", "MRVLSAGLVA--HYNQCP"]
    codes = alignment_to_codes(seqs)
    fig = msa_figure_matplotlib(codes, ["a", "b", "c"], "Protein")
    legend = fig.axes[0].get_legend()
    assert len(legend.get_texts()) < len(np.unique(codes))
    assert fig.get_size_inches()[1] >= 3.5
    plt.close(fig)


def test_plotly_y_axis_is_categorical_for_numeric_ids():
    from modules.viz import alignment_to_codes, msa_figure_plotly
    fig = msa_figure_plotly(alignment_to_codes(["MK", "ML", "MV"]), ["1", "2", "10"], "Protein")
    assert fig.layout.yaxis.type == "category"


def test_letters_shown_only_for_small_alignments():
    small = alignment_to_codes(["MKV", "MKL"])
    assert msa_figure_plotly(small, ["a", "b"], "Protein").data[0].texttemplate == "%{text}"
    big = alignment_to_codes(["M" * 2000, "K" * 2000])
    assert msa_figure_plotly(big, ["a", "b"], "Protein").data[0].texttemplate is None


def test_letters_hidden_for_wide_alignments():
    wide = alignment_to_codes(["M" * 1000, "K" * 1000, "A" * 1000])
    assert msa_figure_plotly(wide, ["a", "b", "c"], "Protein").data[0].texttemplate is None


def test_conservation_plots_do_not_zoom():
    import numpy as np
    from modules.msa_colors import color_alignment
    from modules.viz import alignment_to_codes, conservation_line_figure, msa_viewer_figure, reference_numbering
    fig = conservation_line_figure({1: 0.2, 3: 0.9})
    assert list(fig.data[0].y) == [0.2, None, 0.9]
    assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and fig.layout.dragmode is False
    seqs = ["MKVLAAGIVG", "MRVLA-GIVG"]
    fig = msa_viewer_figure(alignment_to_codes(seqs), ["a", "b"], color_alignment(seqs, "ClustalX"), 0, 10,
                            reference_numbering(seqs[0]), np.ones(10), "x" * 10)
    axes = [fig.layout[k] for k in fig.layout if k.startswith(("xaxis", "yaxis"))]
    assert axes and all(a.fixedrange for a in axes)
