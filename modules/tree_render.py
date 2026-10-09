"""Publication tree figures: place a TreeGeometry in inches (measured labels, no overlaps), then draw it
with matplotlib (exported figure) or plotly (interactive view). Uses matplotlib.figure.Figure directly,
never pyplot, so no global figure registry or GUI backend is involved."""
import html
import math
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from modules.figures import EXPORT_FORMATS, export_bytes, quiet_fonts  # noqa: F401 (re-exported)

STYLES = {"Classic ink": "classic", "Support dots": "dots", "Clade colour": "clades"}
# 12 colour-blind-aware colours (Okabe-Ito, then Tol muted/bright), no black or yellow on white
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255", "#999933",
           "#CC6677", "#117733", "#44AA99")
FONTS = {
    "Arial": ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"),
    "Inter": ("Inter", "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"),
    "Times New Roman": ("Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"),
}
INK, SUPPORT_GREY, LEADER_GREY = "#1a1a1a", "#6b6b6b", "#b5b5b5"
NODE_GREY, HIGHLIGHT = "#4a6a8a", "#d6246e"      # internal node names (muted blue-grey), selected node (accent)
NIGHT_COLOURS = {INK: "#ece8df", SUPPORT_GREY: "#a59f91", NODE_GREY: "#9db5cf",
                 "white": "#1a1916"}             # Explore view, night theme
SUPPORT_OVERLAP = "Support numbers would overlap; support is shown as dots."
MIN_FONT, LABEL_LIMIT = 5.0, 500


@dataclass
class TreeStyle:
    style: str = "classic"          # classic | dots | clades
    support_min: float = 50.0
    font_family: str = "Arial"
    font_size: float = 9.0
    italic: bool = True
    underscores_to_spaces: bool = True
    truncate: int = 0               # 0 = off
    line_width: float = 1.0
    width_cm: float = 16.0
    height_cm: float | None = None  # None = automatic (rectangular only)
    clade_names: list = field(default_factory=list)
    internal_labels: bool = False   # draw each named internal node's name next to it
    highlight: str | None = None    # node name marked with an accent dot


@dataclass
class Text:
    x: float
    y: float
    text: str
    size: float
    color: str = INK
    ha: str = "left"
    va: str = "center"
    rotation: float = 0.0
    italic: bool = False
    halo: bool = False
    kind: str = "leaf"              # leaf | support | scale | legend | clade | node
    hover: str = ""
    width: float = 0.0              # measured, inches (unrotated)
    height: float = 0.0


@dataclass
class Drawing:
    width: float
    height: float
    lines: list = field(default_factory=list)   # (xs, ys, color, width, dashed)
    dots: list = field(default_factory=list)    # (x, y, face, edge, size_pt, hover)
    texts: list = field(default_factory=list)   # Text
    fills: list = field(default_factory=list)   # (xs, ys, color, alpha) closed polygons
    family: str = "DejaVu Sans"     # resolved (installed) font, used for measuring
    warning: str | None = None
    font_choice: str = "Arial"      # the requested family, written first in exports (SVG keeps it)
    node_hover: list = field(default_factory=list)  # (x, y, name) of named internal nodes, for Explore hover


def _shift(drawing, dy):
    """Move everything up by dy inches and grow the figure to match."""
    drawing.lines = [(xs, [y + dy for y in ys], *rest) for xs, ys, *rest in drawing.lines]
    drawing.dots = [(x, y + dy, *rest) for x, y, *rest in drawing.dots]
    drawing.fills = [(xs, [y + dy for y in ys], *rest) for xs, ys, *rest in drawing.fills]
    drawing.node_hover = [(x, y + dy, name) for x, y, name in drawing.node_hover]
    for t in drawing.texts:
        t.y += dy
    drawing.height += dy


def resolve_family(choice):
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONTS.get(choice, FONTS["Arial"]) if f in available), "DejaVu Sans")


class _Measurer:
    """Unrotated text extents in inches, from matplotlib's Agg renderer."""

    def __init__(self, family):
        self.fig = Figure(dpi=72)
        self.renderer = FigureCanvasAgg(self.fig).get_renderer()
        self.family = family

    def __call__(self, text, size, italic=False):
        t = self.fig.text(0, 0, text, fontsize=size, family=self.family, style="italic" if italic else "normal")
        box = t.get_window_extent(self.renderer)
        t.remove()
        return box.width / 72, box.height / 72


def display_name(name, style):
    name = name or ""
    if style.underscores_to_spaces:
        name = name.replace("_", " ")
    if style.truncate and len(name) > style.truncate:
        name = name[:style.truncate] + "…"
    return name


def _nice(value):
    """A 1/2/5 x 10^k number close to `value`, for scale bars."""
    if value <= 0:
        return 0.0
    exp = math.floor(math.log10(value))
    for m in (5, 2, 1):
        if m * 10 ** exp <= value:
            return m * 10 ** exp
    return 10 ** exp


def _corner(x0, y0, x1, y1, r):
    """Vertical from (x0, y0) to y1, a rounded corner, then horizontal to x1, as a polyline."""
    if abs(y1 - y0) < 1e-9:
        return [x0, x1], [y0, y1]
    if r <= 0:
        return [x0, x0, x1], [y0, y1, y1]
    r = min(r, abs(y1 - y0), max(abs(x1 - x0), 1e-9))
    s = 1 if y1 > y0 else -1
    xs, ys = [x0], [y0]
    for t in np.linspace(0, 1, 6):  # quadratic Bezier from (x0, y1 - s r) via (x0, y1) to (x0 + r, y1)
        xs.append((1 - t) ** 2 * x0 + 2 * (1 - t) * t * x0 + t ** 2 * (x0 + r))
        ys.append((1 - t) ** 2 * (y1 - s * r) + 2 * (1 - t) * t * y1 + t ** 2 * y1)
    xs.append(x1)
    ys.append(y1)
    return xs, ys


def _color(clade, style):
    return PALETTE[clade % len(PALETTE)] if style.style == "clades" and clade is not None else INK


def _corners(t, pad=0.0):
    """Corner points (inches) of a measured text box anchored at (x, y) by ha/va, rotated about the anchor."""
    w, h = t.width + 2 * pad, t.height + 2 * pad
    x0 = {"left": 0.0, "right": -w, "center": -w / 2}[t.ha] + (pad if t.ha == "left" else -pad if t.ha == "right"
                                                                 else 0.0)
    y0 = {"bottom": 0.0, "top": -h, "center": -h / 2}[t.va] + (pad if t.va == "bottom" else -pad if t.va == "top"
                                                                 else 0.0)
    a = math.radians(t.rotation)
    c, s = math.cos(a), math.sin(a)
    return np.array([(t.x + px * c - py * s, t.y + px * s + py * c)
                     for px, py in ((x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h))])


def _polygons_overlap(p, q):
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (p, q):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pp, qq = p @ axis, q @ axis
            if min(pp.max(), qq.max()) <= max(pp.min(), qq.min()):
                return False
    return True


def _any_overlap(texts, others, pad=0.0):
    """True if any of `texts` (padded) overlaps another of `texts` or any of `others`."""
    polys = [_corners(t, pad) for t in texts] + [_corners(t) for t in others]
    lo = np.array([p.min(axis=0) for p in polys])
    hi = np.array([p.max(axis=0) for p in polys])
    for i in range(len(texts)):
        near = np.nonzero((lo[i + 1:, 0] < hi[i, 0]) & (hi[i + 1:, 0] > lo[i, 0]) &
                          (lo[i + 1:, 1] < hi[i, 1]) & (hi[i + 1:, 1] > lo[i, 1]))[0] + i + 1
        if any(_polygons_overlap(polys[i], polys[j]) for j in near):
            return True
    return False


def _add_supports(drawing, marks, style, size, dots, dot_size, measure):
    """Support marks at internal nodes `marks` [(x, y, support)], drawn after the leaf labels: grey numbers
    (classic) or dots. Numbers that would overlap each other or a leaf label switch the whole figure to dots,
    with a warning. Returns True when support is drawn as dots."""
    shown = [(x, y, s) for x, y, s in marks if s is not None and s >= style.support_min]
    if not dots and shown:
        texts = []
        for x, y, support in shown:
            label = f"{support:.0f}"
            w, h = measure(label, size * 0.8)
            texts.append(Text(x - 0.03, y + 0.025, label, size * 0.8, SUPPORT_GREY, "right", "bottom", halo=True,
                              kind="support", width=w, height=h))
        if not _any_overlap(texts, [t for t in drawing.texts if t.kind == "leaf"], pad=size * 0.8 * 0.08 / 72):
            drawing.texts.extend(texts)
            return False
        dots = True
        drawing.warning = " ".join(filter(None, [drawing.warning, SUPPORT_OVERLAP]))
    for x, y, support in shown:
        face = INK if support >= 95 else "#a8a8a8" if support >= 70 else "white"
        drawing.dots.append((x, y, face, INK, dot_size, f"support {support:.0f}"))
    return dots


NODE_SPOTS = ((-0.03, -0.03, "right", "top"), (-0.03, 0.025, "right", "bottom"), (0.04, -0.03, "left", "top"),
              (0.04, 0.03, "left", "bottom"))
NODE_HIDDEN = "{n} node names would overlap other labels and are hidden (Explore shows them on hover)."


def _mark_nodes(drawing, points, style, size, dot_size, measure):
    """Internal node names and the highlighted node, after leaf labels and support marks. points: (x, y, name,
    is_leaf) of every node. A name takes the first spot around its node (below-left, above-left, then right)
    inside the figure that overlaps no leaf, support, clade or other node label; names that fit nowhere are left
    out with a warning (the highlighted node is placed first, so its name is kept whenever a spot exists)."""
    if style.internal_labels:
        named = [(x, y, name) for x, y, name, leaf in points if name and not leaf]
        named.sort(key=lambda p: p[2] != style.highlight)
        drawing.node_hover.extend(named)
        blockers = [t for t in drawing.texts if t.kind in ("leaf", "support", "clade")]
        label_size, hidden = size * 0.75, 0
        pad = label_size * 0.08 / 72
        for x, y, name in named:
            w, h = measure(name, label_size)
            colour = HIGHLIGHT if name == style.highlight else NODE_GREY
            for dx, dy, ha, va in NODE_SPOTS:
                text = Text(x + dx, y + dy, name, label_size, colour, ha, va, halo=True, kind="node", hover=name,
                            width=w, height=h)
                corners = _corners(text)
                inside = corners.min() >= 0 and (corners.max(axis=0) <= (drawing.width, drawing.height)).all()
                if inside and not (_any_overlap([text], blockers) or _any_overlap([text], blockers, pad=pad)):
                    drawing.texts.append(text)
                    blockers.append(text)
                    break
            else:
                hidden += 1
        if hidden:
            drawing.warning = " ".join(filter(None, [drawing.warning, NODE_HIDDEN.format(n=hidden)]))
    if style.highlight:
        spot = next(((x, y) for x, y, name, _ in points if name == style.highlight), None)
        if spot is not None:
            drawing.dots.append((*spot, HIGHLIGHT, "white", max(dot_size * 1.8, 7.0),
                                 f"{style.highlight} (selected)"))


def _has_supports(geo):
    return any(nd.support is not None for nd in geo.nodes if not nd.is_leaf)


def _scale_bar(drawing, x, y, units_per_inch, size, max_depth):
    """Draw a scale bar starting at (x, y); returns the x where it ends (x when no bar is drawn)."""
    length = _nice(max_depth / 5)
    if length <= 0:
        return x
    w = length / units_per_inch
    drawing.lines.append(([x, x + w], [y, y], INK, 1.0, False))
    drawing.texts.append(Text(x + w / 2, y - 0.04, f"{length:g}", size * 0.85, INK, "center", "top", kind="scale"))
    return x + w


LEGEND_Y, LEGEND_MARGIN = 0.25, 0.15


def _legend_row(size):
    return max(0.22, size * 1.6 / 72)


def _legend_rows(drawing, x0, y, items, size, measure):
    """Place legend items [(swatch(x, y), label, text size, text offset, gap)] left to right from (x0, y),
    starting a new row below whenever an item would cross the right margin. Returns the height used."""
    row, x, rows = _legend_row(size), x0, 1
    right = drawing.width - LEGEND_MARGIN
    for swatch, label, text_size, dx, gap in items:
        w, h = measure(label, text_size)
        if x > x0 and x + dx + w > right:
            x, y, rows = x0, y - row, rows + 1
        swatch(x, y)
        drawing.texts.append(Text(x + dx, y, label, text_size, INK, kind="legend", width=w, height=h))
        x += dx + w + gap
    return rows * row


def _dot_legend(drawing, x, y, size, dot_size, measure):
    def dot(face):
        return lambda px, py: drawing.dots.append((px, py, face, INK, dot_size, ""))
    items = [(dot(face), label, size * 0.85, 0.07, 0.15)
             for face, label in ((INK, "≥95"), ("#a8a8a8", "70–94"), ("white", "<70"))]
    return _legend_rows(drawing, x, y, items, size, measure)


def _clade_legend(drawing, x, y, names, size, measure):
    def swatch(color):
        return lambda px, py: drawing.fills.append(([px, px + 0.1, px + 0.1, px], [py - 0.05, py - 0.05, py + 0.05,
                                                                                      py + 0.05], color, 1.0))
    items = [(swatch(PALETTE[i % len(PALETTE)]), name, size * 0.9, 0.14, 0.2) for i, name in enumerate(names)]
    return _legend_rows(drawing, x, y, items, size, measure)


def _legends(drawing, x, size, dot_size, measure, clade_names=(), dots=False):
    """Bottom legends from (x, LEGEND_Y): clade names, then the support-dot key on its own row. When they need
    more than one row, the figure grows and everything above them moves up."""
    used = 0.0
    if clade_names:
        used += _clade_legend(drawing, x, LEGEND_Y, clade_names, size, measure)
    if dots:
        used += _dot_legend(drawing, x, LEGEND_Y - used, size, dot_size, measure)
    extra = used - _legend_row(size)
    if extra > 1e-9:
        _shift(drawing, extra)


def clade_labels(style, n):
    return [style.clade_names[i] if i < len(style.clade_names) and style.clade_names[i] else f"Clade {i + 1}"
            for i in range(n)]


def place(geo, style):
    """Lay the geometry out in inches with every leaf label measured. Returns a Drawing."""
    family = resolve_family(style.font_family)
    measure = _Measurer(family)
    n = len(geo.leaves)
    names = [display_name(leaf.name, style) for leaf in geo.leaves]
    show_labels = n <= LABEL_LIMIT
    if geo.layout == "rectangular":
        d = _place_rectangular(geo, style, names, measure, show_labels)
    elif geo.layout == "circular":
        d = _place_circular(geo, style, names, measure, show_labels)
    else:
        d = _place_unrooted(geo, style, names, measure, show_labels)
    d.family = family
    d.font_choice = style.font_family
    if not show_labels:
        d.warning = " ".join(filter(None, [f"{n} leaves: labels are hidden in the figure (hover in Explore shows "
                                           "names).", d.warning]))
    return d


def _groups(geo):
    groups = {}
    for leaf_index, leaf in enumerate(geo.leaves):
        if leaf.clade is not None:
            groups.setdefault(leaf.clade, []).append(leaf_index)
    return groups


def _place_rectangular(geo, style, names, measure, show_labels):
    n = len(geo.leaves)
    font = style.font_size
    warning = None
    top, left = 0.15, 0.15
    clades = style.style == "clades" and bool(_groups(geo))
    bottom = 0.5
    if style.height_cm:
        height = style.height_cm / 2.54
        row = max(height - top - bottom, 0.1) / max(n - 1, 1)
        font = min(font, row * 72 / 1.4)
        if font < MIN_FONT:
            font, warning = MIN_FONT, "Labels overlap at this height; increase the height or reduce the font size."
    else:
        row = font * 1.4 / 72
        height = top + bottom + row * max(n - 1, 1)
    # support numbers need about 1.2 x their own height between rows; otherwise show support as dots
    dots = style.style != "classic" or row < 1.2 * (font * 0.8 * 1.2 / 72)
    if dots and style.style == "classic" and _has_supports(geo):
        warning = " ".join(filter(None, [warning, "Rows are too tight for support numbers; support is shown as dots."]))
    sizes = [measure(name, font, style.italic) for name in names] if show_labels else [(0.0, 0.0)] * n
    label_w = (max(w for w, _ in sizes) + 0.08) if show_labels else 0.05
    groups = _groups(geo)
    clade_names = clade_labels(style, max(groups) + 1) if clades else []
    clade_w = (0.16 + max(measure(clade_names[g], font)[0] for g in groups)) if clades else 0.0
    width = style.width_cm / 2.54
    span = width - left - label_w - clade_w - 0.1
    if span < 1.0:
        span = 1.0
        width = left + span + label_w + clade_w + 0.1
    X = lambda depth: left + depth / geo.max_depth * span
    Y = lambda r: height - top - r * row
    d = Drawing(width, height, warning=warning)
    lw = style.line_width
    aligned = style.style in ("dots", "clades") and not geo.cladogram
    max_x = X(geo.max_depth)
    if clades:  # background band per clade, from the clade root to the right edge
        for g, leaf_rows in groups.items():
            root_depth = min(nd.a for nd in geo.nodes if nd.clade == g)
            x0, x1 = X(root_depth) - 0.03, width - clade_w - 0.05
            y0, y1 = Y(max(leaf_rows)) - row / 2 + 0.01, Y(min(leaf_rows)) + row / 2 - 0.01
            d.fills.append(([x0, x1, x1, x0], [y0, y0, y1, y1], PALETTE[g % len(PALETTE)], 0.08))
    radius = min(0.04, row / 3)
    for pa, pb, ca, cb, clade in geo.edges:
        xs, ys = _corner(X(pa), Y(pb), X(ca), Y(cb), radius)
        d.lines.append((xs, ys, _color(clade, style), lw, False))
    dot_size = max(font * 0.55, 3.0)
    for (w, h), leaf, name in zip(sizes, geo.leaves, names):
        x, y = X(leaf.a), Y(leaf.b)
        lx = max_x + 0.06 if aligned else x + 0.06
        if aligned and lx - x > 0.1:
            d.lines.append(([x + 0.03, lx - 0.03], [y, y], LEADER_GREY, 0.6, True))
        if show_labels:
            d.texts.append(Text(lx, y, name, font, _color(leaf.clade, style), "left", "center", italic=style.italic,
                                hover=leaf.name, width=w, height=h))
    if clades:
        bar_x = width - clade_w + 0.02
        for g, leaf_rows in groups.items():
            y0, y1 = Y(max(leaf_rows)) - row * 0.35, Y(min(leaf_rows)) + row * 0.35
            d.lines.append(([bar_x, bar_x], [y0, y1], PALETTE[g % len(PALETTE)], 3.0, False))
            d.texts.append(Text(bar_x + 0.08, (y0 + y1) / 2, clade_names[g], font, PALETTE[g % len(PALETTE)],
                                kind="clade"))
    dots = _add_supports(d, [(X(nd.a), Y(nd.b), nd.support) for nd in geo.nodes if not nd.is_leaf], style, font,
                         dots, dot_size, measure)
    _mark_nodes(d, [(X(nd.a), Y(nd.b), nd.name, nd.is_leaf) for nd in geo.nodes], style, font, dot_size, measure)
    bar_end = left
    if geo.has_lengths and not geo.cladogram:
        bar_end = _scale_bar(d, left, 0.3, geo.max_depth / span, font, geo.max_depth)
    if dots and _has_supports(geo):
        _legends(d, max(bar_end + 0.4, width - 1.6), font, dot_size, measure, dots=True)
    return d


def _place_circular(geo, style, names, measure, show_labels):
    n = len(geo.leaves)
    font = style.font_size
    sizes = [measure(name, font, style.italic) for name in names] if show_labels else [(0.0, 0.0)] * n
    label_w = max(w for w, _ in sizes) if show_labels else 0.0
    label_h = max((h for _, h in sizes), default=font * 1.2 / 72) or font * 1.2 / 72
    step = 2 * math.pi / (n + 1)
    gap = 0.08
    needed = 1.25 * label_h / (2 * math.sin(min(step, math.pi / 2) / 2)) - gap if show_labels else 0.0
    groups = _groups(geo)
    clades = style.style == "clades" and bool(groups)
    ring = 0.18 if clades else 0.0
    radius = max(needed, 1.0, style.width_cm / 2.54 / 2 - label_w - gap - ring - 0.15)
    half = radius + gap + label_w + ring + 0.15
    legend_h = 0.45 if (geo.has_lengths and not geo.cladogram) or clades or _has_supports(geo) else 0.0
    width, height = 2 * half, 2 * half + legend_h
    cx, cy = half, half + legend_h
    R = lambda depth: depth / geo.max_depth * radius
    P = lambda r, t: (cx + r * math.cos(t), cy + r * math.sin(t))
    d = Drawing(width, height)
    lw = style.line_width
    if clades:
        for g, leaf_index in groups.items():
            root_depth = min(nd.a for nd in geo.nodes if nd.clade == g)
            t1 = geo.leaves[min(leaf_index)].b + step / 2
            t0 = geo.leaves[max(leaf_index)].b - step / 2
            ts = np.linspace(t0, t1, 40)
            r0, r1 = R(root_depth), radius + gap + label_w + 0.04
            xs = [P(r1, t)[0] for t in ts] + [P(r0, t)[0] for t in ts[::-1]]
            ys = [P(r1, t)[1] for t in ts] + [P(r0, t)[1] for t in ts[::-1]]
            d.fills.append((xs, ys, PALETTE[g % len(PALETTE)], 0.08))
            ring_r = radius + gap + label_w + 0.1
            d.lines.append(([P(ring_r, t)[0] for t in ts], [P(ring_r, t)[1] for t in ts],
                            PALETTE[g % len(PALETTE)], 3.0, False))
    for pr, pt, cr, ct, clade in geo.edges:
        color = _color(clade, style)
        if abs(ct - pt) > 1e-9:
            ts = np.linspace(pt, ct, max(3, int(abs(ct - pt) / 0.03)))
            d.lines.append(([P(R(pr), t)[0] for t in ts], [P(R(pr), t)[1] for t in ts], color, lw, False))
        (x0, y0), (x1, y1) = P(R(pr), ct), P(R(cr), ct)
        d.lines.append(([x0, x1], [y0, y1], color, lw, False))
    dot_size = max(font * 0.55, 3.0)
    for (w, h), leaf, name in zip(sizes, geo.leaves, names):
        tip = P(R(leaf.a), leaf.b)
        lx, ly = P(radius + gap, leaf.b)
        if R(leaf.a) < radius - 0.02:
            end = P(radius + gap - 0.03, leaf.b)
            d.lines.append(([tip[0], end[0]], [tip[1], end[1]], LEADER_GREY, 0.6, True))
        deg = math.degrees(leaf.b) % 360
        flip = 90 < deg < 270
        if show_labels:
            d.texts.append(Text(lx, ly, name, font, _color(leaf.clade, style), "right" if flip else "left", "center",
                                rotation=deg - 180 if flip else (deg if deg <= 90 else deg - 360),
                                italic=style.italic, hover=leaf.name, width=w, height=h))
    dots = _add_supports(d, [(*P(R(nd.a), nd.b), nd.support) for nd in geo.nodes if not nd.is_leaf], style, font,
                         style.style != "classic", dot_size, measure)
    _mark_nodes(d, [(*P(R(nd.a), nd.b), nd.name, nd.is_leaf) for nd in geo.nodes], style, font, dot_size, measure)
    bar_end = 0.2
    if geo.has_lengths and not geo.cladogram:
        bar_end = _scale_bar(d, 0.2, 0.3, geo.max_depth / radius, font, geo.max_depth)
    _legends(d, max(1.2, bar_end + 0.4), font, dot_size, measure,
             clade_labels(style, max(groups) + 1) if clades else (), dots and _has_supports(geo))
    return d


def _box(x, y, w, h, ha, va):
    x0 = x if ha == "left" else x - w if ha == "right" else x - w / 2
    y0 = y if va == "bottom" else y - h if va == "top" else y - h / 2
    return [x0, y0, x0 + w, y0 + h]


ANGLE_TRIES = (0, 20, -20, 40, -40, 60, -60)


def _anchor_for(angle):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return c, s, "left" if c > 0.3 else "right" if c < -0.3 else "center", \
        "bottom" if s > 0.3 else "top" if s < -0.3 else "center"


def _place_outward(tips, sizes, angles, obstacles=None, pad=0.02, gap=0.05, step_frac=0.5, max_steps=80):
    """Greedy label placement around tips. Labels are placed in order; each tries spots at increasing
    distance from its tip along its branch direction (and up to 60 degrees either side), taking the first
    spot clear of (1) labels already placed, (2) branch points in `obstacles` and leaders already drawn,
    and (3) whose own leader line crosses no placed label. If none is found the constraints are relaxed
    in that order, ending with labels-only along the branch direction, which always succeeds: labels never
    overlap each other. Returns (boxes [x0, y0, x1, y1], leader end points or None per label)."""
    placed = np.empty((0, 4))
    points = np.empty((0, 2)) if obstacles is None else np.asarray(obstacles, dtype=float).reshape(-1, 2)
    leaders = np.empty((0, 2))

    def inside(px, py, boxes, margin=0.0):
        return ((px[:, None] > boxes[None, :, 0] - margin) & (px[:, None] < boxes[None, :, 2] + margin) &
                (py[:, None] > boxes[None, :, 1] - margin) & (py[:, None] < boxes[None, :, 3] + margin)).any()

    def labels_clear(box, _ray):
        return ((placed[:, 0] >= box[2] + pad) | (placed[:, 2] <= box[0] - pad) |
                (placed[:, 1] >= box[3] + pad) | (placed[:, 3] <= box[1] - pad)).all()

    def branches_clear(box, _ray):
        pts = np.vstack([points, leaders])
        return not len(pts) or not inside(pts[:, 0], pts[:, 1], box[None, :])

    def leader_clear(_box, ray):
        x, y, c, s, dist = ray
        if dist <= gap + 1e-9 or not len(placed):
            return True
        ts = np.arange(0.06, dist, 0.02)
        return not len(ts) or not inside(x + ts * c, y + ts * s, placed)

    boxes, ends = [], []
    for (x, y), (w, h), base in zip(tips, sizes, angles):
        step = max(h * step_frac, 0.01)
        box = ray = None
        for checks in ((labels_clear, branches_clear, leader_clear), (labels_clear, branches_clear)):
            for k in range(max_steps):
                for delta in ANGLE_TRIES:
                    c, s, ha, va = _anchor_for(base + delta)
                    dist = gap + k * step
                    candidate = np.array(_box(x + dist * c, y + dist * s, w, h, ha, va))
                    if all(check(candidate, (x, y, c, s, dist)) for check in checks):
                        box, ray = candidate, (x, y, c, s, dist)
                        break
                if box is not None:
                    break
            if box is not None:
                break
        k = 0
        while box is None:  # labels-only along the branch direction: terminates (finitely many labels)
            c, s, ha, va = _anchor_for(base)
            dist = gap + k * step
            candidate = np.array(_box(x + dist * c, y + dist * s, w, h, ha, va))
            if labels_clear(candidate, None):
                box, ray = candidate, (x, y, c, s, dist)
            k += 1
        placed = np.vstack([placed, box])
        boxes.append(box.tolist())
        _, _, c, s, dist = ray
        if dist > gap + h / 2:
            ts = np.arange(0.0, dist, 0.02)
            leaders = np.vstack([leaders, np.column_stack([x + ts * c, y + ts * s])])
            ends.append((x + dist * c, y + dist * s))
        else:
            ends.append(None)
    return boxes, ends


def _place_unrooted(geo, style, names, measure, show_labels):
    font = style.font_size
    sizes = [measure(name, font, style.italic) for name in names] if show_labels else [(0.0, 0.0)] * len(names)
    label_w = max((w for w, _ in sizes), default=0.0)
    xs = [nd.a for nd in geo.nodes]
    ys = [nd.b for nd in geo.nodes]
    extent = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    label_h = max((h for _, h in sizes), default=0.0)
    ring = len(names) * label_h * 1.3 / (2 * math.pi) if show_labels else 0.0  # perimeter fits the labels
    scale = max((style.width_cm / 2.54 - 2 * label_w - 0.3) / extent, 1.5 / extent, 2 * ring / extent)
    boxes, ends = [], []
    if show_labels:
        obstacles = []
        for x0, y0, x1, y1, _ in geo.edges:  # branch points every ~0.03 in, except the last 0.06 in at each tip
            length = math.hypot(x1 - x0, y1 - y0) * scale
            for t in np.linspace(0, 1, max(2, int(length / 0.03))):
                if (1 - t) * length > 0.06:
                    obstacles.append(((x0 + t * (x1 - x0)) * scale, (y0 + t * (y1 - y0)) * scale))
        tips = [(leaf.a * scale, leaf.b * scale) for leaf in geo.leaves]
        boxes, ends = _place_outward(tips, sizes, geo.leaf_angles, obstacles)
    margin = 0.15
    legend_h = 0.45
    all_x = [x * scale for x in xs] + [b[0] for b in boxes] + [b[2] for b in boxes]
    all_y = [y * scale for y in ys] + [b[1] for b in boxes] + [b[3] for b in boxes]
    ox, oy = margin - min(all_x), margin + legend_h - min(all_y)
    width = max(all_x) - min(all_x) + 2 * margin
    height = max(all_y) - min(all_y) + 2 * margin + legend_h
    d = Drawing(width, height)
    for x0, y0, x1, y1, clade in geo.edges:
        d.lines.append(([x0 * scale + ox, x1 * scale + ox], [y0 * scale + oy, y1 * scale + oy],
                        _color(clade, style), style.line_width, False))
    dot_size = max(font * 0.55, 3.0)
    if show_labels:
        for (w, h), leaf, name, box, end in zip(sizes, geo.leaves, names, boxes, ends):
            bx0, by0 = box[0] + ox, box[1] + oy
            if end is not None:
                tx, ty = leaf.a * scale + ox, leaf.b * scale + oy
                d.lines.append(([tx, end[0] + ox], [ty, end[1] + oy], LEADER_GREY, 0.6, False))
            d.texts.append(Text(bx0, by0 + h / 2, name, font, _color(leaf.clade, style), "left", "center",
                                italic=style.italic, hover=leaf.name, width=w, height=h))
    dots = _add_supports(d, [(nd.a * scale + ox, nd.b * scale + oy, nd.support) for nd in geo.nodes
                             if not nd.is_leaf], style, font, style.style != "classic", dot_size, measure)
    _mark_nodes(d, [(nd.a * scale + ox, nd.b * scale + oy, nd.name, nd.is_leaf) for nd in geo.nodes], style, font,
                dot_size, measure)
    bar_end = 0.2
    if geo.has_lengths:
        bar_end = _scale_bar(d, 0.2, 0.3, 1 / scale, font, extent)
    groups = _groups(geo)
    _legends(d, max(1.2, bar_end + 0.4), font, dot_size, measure,
             clade_labels(style, max(groups) + 1) if style.style == "clades" and groups else (),
             dots and _has_supports(geo))
    return d


def font_families(drawing):
    """The requested family first, then its fallbacks: exports name it (SVG text keeps 'Arial', ...) while
    glyphs come from the first installed one, which is also the font the labels were measured with."""
    return list(dict.fromkeys([drawing.font_choice, *FONTS.get(drawing.font_choice, ()), drawing.family]))


def render_matplotlib(drawing):
    """The publication figure (white background), as a matplotlib Figure without pyplot."""
    with quiet_fonts():
        return _render_matplotlib(drawing)


def _render_matplotlib(drawing):
    fig = Figure(figsize=(drawing.width, drawing.height), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, drawing.width)
    ax.set_ylim(0, drawing.height)
    ax.set_axis_off()
    for xs, ys, color, alpha in drawing.fills:
        ax.add_patch(Polygon(np.column_stack([xs, ys]), closed=True, facecolor=color, edgecolor="none",
                             alpha=alpha, zorder=0))
    for xs, ys, color, width, dashed in drawing.lines:
        ax.plot(xs, ys, color=color, linewidth=width, linestyle=(0, (1, 2)) if dashed else "-",
                solid_capstyle="round", solid_joinstyle="round", zorder=1)
    for x, y, face, edge, size, _ in drawing.dots:
        ax.scatter([x], [y], s=size ** 2, facecolor=face, edgecolor=edge, linewidth=0.8, zorder=3)
    families = font_families(drawing)
    for t in drawing.texts:  # halo as a white box: path effects would export support numbers as outlines
        ax.text(t.x, t.y, t.text, fontsize=t.size, color=t.color, ha=t.ha, va=t.va, rotation=t.rotation,
                rotation_mode="anchor", family=families, style="italic" if t.italic else "normal",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none") if t.halo else None,
                gid=t.kind, zorder=4)
    return fig


def _text_center(t):
    """Centre of a (possibly rotated) text box anchored at (x, y) with ha/va, in inches."""
    w, h = t.width, t.height
    dx = w / 2 if t.ha == "left" else -w / 2 if t.ha == "right" else 0.0
    dy = h / 2 if t.va == "bottom" else -h / 2 if t.va == "top" else 0.0
    a = math.radians(t.rotation)
    return t.x + dx * math.cos(a) - dy * math.sin(a), t.y + dx * math.sin(a) + dy * math.cos(a)


def render_plotly(drawing, dark=False):
    """Interactive view of the same drawing: hover on leaves and nodes, zoom and pan."""
    px_per_in = 96
    recolor = (lambda c: NIGHT_COLOURS.get(c, c)) if dark else (lambda c: c)
    fig = go.Figure()
    shapes = [dict(type="path", path="M " + " L ".join(f"{x},{y}" for x, y in zip(xs, ys)) + " Z",
                   fillcolor=color, opacity=alpha, line=dict(width=0), layer="below")
              for xs, ys, color, alpha in drawing.fills]
    by_style = {}
    for xs, ys, color, width, dashed in drawing.lines:
        bucket = by_style.setdefault((recolor(color), width, dashed), ([], []))
        bucket[0].extend(list(xs) + [None])
        bucket[1].extend(list(ys) + [None])
    for (color, width, dashed), (xs, ys) in by_style.items():
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=color, width=width * 1.3, dash="dot" if dashed else "solid")))
    if drawing.dots:
        fig.add_trace(go.Scatter(x=[d[0] for d in drawing.dots], y=[d[1] for d in drawing.dots], mode="markers",
                                 marker=dict(color=[recolor(d[2]) for d in drawing.dots],
                                             size=[d[4] * 1.3 for d in drawing.dots],
                                             line=dict(color=[recolor(d[3]) for d in drawing.dots], width=1)),
                                 text=[d[5] for d in drawing.dots], hovertemplate="%{text}<extra></extra>",
                                 showlegend=False))
    leaves = [t for t in drawing.texts if t.kind == "leaf"]
    if leaves:  # invisible hover targets on the label centres
        centres = [_text_center(t) for t in leaves]
        fig.add_trace(go.Scatter(x=[c[0] for c in centres], y=[c[1] for c in centres], mode="markers",
                                 marker=dict(size=10, opacity=0), text=[t.hover for t in leaves],
                                 hovertemplate="%{text}<extra></extra>", showlegend=False))
    if drawing.node_hover:  # every named internal node, including names left out of the figure
        fig.add_trace(go.Scatter(x=[p[0] for p in drawing.node_hover], y=[p[1] for p in drawing.node_hover],
                                 mode="markers", marker=dict(size=10, opacity=0),
                                 text=[p[2] for p in drawing.node_hover], hovertemplate="%{text}<extra></extra>",
                                 showlegend=False))
    annotations = []
    for t in drawing.texts:
        cx, cy = _text_center(t) if t.width else (t.x, t.y)
        annotations.append(dict(
            x=cx, y=cy, text=f"<i>{html.escape(t.text)}</i>" if t.italic else html.escape(t.text), showarrow=False,
            textangle=-t.rotation,
            xanchor="center" if t.width else t.ha, yanchor="middle" if t.width else
            {"center": "middle", "bottom": "bottom", "top": "top"}[t.va],
            font=dict(size=t.size * px_per_in / 72, color=recolor(t.color), family=drawing.family)))
    fig.update_layout(
        shapes=shapes, annotations=annotations, width=None, height=max(320, drawing.height * px_per_in),
        margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, drawing.width]),
        yaxis=dict(visible=False, range=[0, drawing.height], scaleanchor="x", scaleratio=1))
    return fig
