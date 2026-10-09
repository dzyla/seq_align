"""Shared export helpers for publication figures (matplotlib.figure.Figure without pyplot)."""
import logging
import math
from contextlib import contextmanager
from io import BytesIO

import matplotlib

EXPORT_FORMATS = {"svg": "image/svg+xml", "pdf": "application/pdf", "png": "image/png"}
PNG_DPI, PNG_MAX_SIDE, PNG_MAX_PIXELS = 600, 20000, 150e6
PAD_INCHES = 0.05


@contextmanager
def quiet_fonts():
    """Silence matplotlib's findfont warnings for listed fallback families that are not installed."""
    logger = logging.getLogger("matplotlib.font_manager")
    old = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(old)


def png_dpi(fig):
    """Export resolution of a figure: 600 dpi unless that would exceed PNG_MAX_SIDE pixels on the longest side
    or PNG_MAX_PIXELS in total (both measured on the tight bounding box the export uses)."""
    try:
        with quiet_fonts():
            bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = bbox.width, bbox.height
    except AttributeError:  # canvas without a renderer: fall back to the figure size
        w, h = fig.get_size_inches()
    w, h = w + 2 * PAD_INCHES, h + 2 * PAD_INCHES
    cap = min(PNG_MAX_SIDE / max(w, h), math.sqrt(PNG_MAX_PIXELS / (w * h)))
    return int(min(PNG_DPI, math.floor(cap * 0.995)))


def export_bytes(fig, fmt):
    """SVG (editable text), PDF (embedded TrueType fonts) or PNG bytes (at png_dpi(fig), 600 dpi when the
    figure is small enough) of a figure."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown export format {fmt!r}")
    buf = BytesIO()
    dpi = png_dpi(fig) if fmt == "png" else 100
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}), quiet_fonts():
        fig.savefig(buf, format=fmt, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=PAD_INCHES)
    return buf.getvalue()


def preview_png(fig, dpi=150):
    """On-screen preview PNG with the same framing as export_bytes."""
    buf = BytesIO()
    with quiet_fonts():
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=PAD_INCHES)
    return buf.getvalue()
