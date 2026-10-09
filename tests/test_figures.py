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
