"""Shared export helpers for publication figures (matplotlib.figure.Figure without pyplot)."""
import logging
from contextlib import contextmanager
from io import BytesIO

import matplotlib

EXPORT_FORMATS = {"svg": "image/svg+xml", "pdf": "application/pdf", "png": "image/png"}


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


def export_bytes(fig, fmt):
    """SVG (editable text), PDF (embedded TrueType fonts) or 600-dpi PNG bytes of a figure."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown export format {fmt!r}")
    buf = BytesIO()
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}), quiet_fonts():
        fig.savefig(buf, format=fmt, dpi=600 if fmt == "png" else 100, facecolor="white", bbox_inches="tight",
                    pad_inches=0.05)
    return buf.getvalue()


def preview_png(fig, dpi=150):
    """On-screen preview PNG with the same framing as export_bytes."""
    buf = BytesIO()
    with quiet_fonts():
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    return buf.getvalue()
