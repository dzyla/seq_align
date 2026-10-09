import pytest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from modules.figures import EXPORT_FORMATS, export_bytes, preview_png


def _fig():
    fig = Figure(figsize=(2, 1))
    FigureCanvasAgg(fig)
    fig.text(0.1, 0.5, "label")
    return fig


def test_shared_exports_and_preview():
    fig = _fig()
    assert set(EXPORT_FORMATS) == {"svg", "pdf", "png"}
    assert b">label<" in export_bytes(fig, "svg")
    assert preview_png(fig).startswith(b"\x89PNG")


def test_tree_render_reexports_shared_helpers():
    import modules.figures as figures
    import modules.tree_render as tree_render
    assert tree_render.export_bytes is figures.export_bytes and tree_render.quiet_fonts is figures.quiet_fonts


def test_png_dpi_is_capped_for_tall_figures():
    from io import BytesIO

    from PIL import Image

    from modules.figures import png_dpi
    assert png_dpi(_fig()) == 600
    tall = Figure(figsize=(8.9, 122.5))
    FigureCanvasAgg(tall)
    tall.text(0.1, 0.5, "label")
    tall.text(0.9, 0.99, "top")
    tall.text(0.1, 0.01, "bottom")
    dpi = png_dpi(tall)
    assert dpi < 600
    image = Image.open(BytesIO(export_bytes(tall, "png")))
    assert max(image.size) <= 20000 and image.size[0] * image.size[1] <= 150e6
    assert image.info["dpi"][0] == pytest.approx(dpi, abs=1)
