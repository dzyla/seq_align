"""Publication tree figures: place a TreeGeometry in inches (measured labels, no overlaps), then draw it
with matplotlib (exported figure) or plotly (interactive view). Uses matplotlib.figure.Figure directly,
never pyplot, so no global figure registry or GUI backend is involved."""
import html
import math
from dataclasses import dataclass, field
from io import BytesIO

import matplotlib
import numpy as np
import plotly.graph_objects as go
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

STYLES = {"Classic ink": "classic", "Support dots": "dots", "Clade colour": "clades"}
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#B8860B")
FONTS = {
    "Arial": ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"),
    "Inter": ("Inter", "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"),
    "Times New Roman": ("Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"),
}
INK, SUPPORT_GREY, LEADER_GREY = "#1a1a1a", "#6b6b6b", "#b5b5b5"
MIN_FONT, LABEL_LIMIT = 5.0, 500
EXPORT_FORMATS = {"svg": "image/svg+xml", "pdf": "application/pdf", "png": "image/png"}


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
    kind: str = "leaf"              # leaf | support | scale | legend | clade
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
    family: str = "DejaVu Sans"
    warning: str | None = None


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


def _support_marks(drawing, x, y, support, style, size, dots, dot_size):
    if support is None or support < style.support_min:
        return
    if dots:
        face = INK if support >= 95 else "#a8a8a8" if support >= 70 else "white"
        drawing.dots.append((x, y, face, INK, dot_size, f"support {support:.0f}"))
    else:
        drawing.texts.append(Text(x - 0.03, y + 0.025, f"{support:.0f}", size * 0.8, SUPPORT_GREY, "right",
                                  "bottom", halo=True, kind="support"))


def _scale_bar(drawing, x, y, units_per_inch, size, max_depth):
    """Draw a scale bar starting at (x, y); returns the x where it ends (x when no bar is drawn)."""
    length = _nice(max_depth / 5)
    if length <= 0:
        return x
    w = length / units_per_inch
    drawing.lines.append(([x, x + w], [y, y], INK, 1.0, False))
    drawing.texts.append(Text(x + w / 2, y - 0.04, f"{length:g}", size * 0.85, INK, "center", "top", kind="scale"))
    return x + w


def _dot_legend(drawing, x, y, size, dot_size, measure):
    for face, label in ((INK, "≥95"), ("#a8a8a8", "70–94"), ("white", "<70")):
        drawing.dots.append((x, y, face, INK, dot_size, ""))
        w, _ = measure(label, size * 0.85)
        drawing.texts.append(Text(x + 0.07, y, label, size * 0.85, INK, kind="legend", width=w))
        x += 0.07 + w + 0.15


def _clade_legend(drawing, x, y, names, size, measure):
    for i, name in enumerate(names):
        drawing.fills.append(([x, x + 0.1, x + 0.1, x], [y - 0.05, y - 0.05, y + 0.05, y + 0.05],
                              PALETTE[i % len(PALETTE)], 1.0))
        w, _ = measure(name, size * 0.9)
        drawing.texts.append(Text(x + 0.14, y, name, size * 0.9, INK, kind="legend", width=w))
        x += 0.14 + w + 0.2


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
    if not show_labels:
        d.warning = f"{n} leaves: labels are hidden in the figure (hover in Explore shows names)."
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
    if dots and style.style == "classic" and any(nd.support is not None for nd in geo.nodes if not nd.is_leaf):
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
    for node in geo.nodes:
        if not node.is_leaf:
            _support_marks(d, X(node.a), Y(node.b), node.support, style, font, dots, dot_size)
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
    if geo.has_lengths and not geo.cladogram:
        _scale_bar(d, left, 0.3, geo.max_depth / span, font, geo.max_depth)
    if dots and any(nd.support is not None for nd in geo.nodes if not nd.is_leaf):
        _dot_legend(d, width - 1.6, 0.25, font, dot_size, measure)
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
    legend_h = 0.45 if (geo.has_lengths and not geo.cladogram) or style.style != "classic" or clades else 0.0
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
    for node in geo.nodes:
        if not node.is_leaf:
            x, y = P(R(node.a), node.b)
            _support_marks(d, x, y, node.support, style, font, style.style != "classic", dot_size)
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
    bar_end = 0.2
    if geo.has_lengths and not geo.cladogram:
        bar_end = _scale_bar(d, 0.2, 0.3, geo.max_depth / radius, font, geo.max_depth)
    if clades:
        _clade_legend(d, max(1.2, bar_end + 0.4), 0.25, clade_labels(style, max(groups) + 1), font, measure)
    elif style.style != "classic":
        _dot_legend(d, width - 1.6, 0.25, font, dot_size, measure)
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
    for node in geo.nodes:
        if not node.is_leaf:
            _support_marks(d, node.a * scale + ox, node.b * scale + oy, node.support, style, font,
                           style.style != "classic", dot_size)
    if show_labels:
        for (w, h), leaf, name, box, end in zip(sizes, geo.leaves, names, boxes, ends):
            bx0, by0 = box[0] + ox, box[1] + oy
            if end is not None:
                tx, ty = leaf.a * scale + ox, leaf.b * scale + oy
                d.lines.append(([tx, end[0] + ox], [ty, end[1] + oy], LEADER_GREY, 0.6, False))
            d.texts.append(Text(bx0, by0 + h / 2, name, font, _color(leaf.clade, style), "left", "center",
                                italic=style.italic, hover=leaf.name, width=w, height=h))
    bar_end = 0.2
    if geo.has_lengths:
        bar_end = _scale_bar(d, 0.2, 0.3, 1 / scale, font, extent)
    groups = _groups(geo)
    if style.style == "clades" and groups:
        _clade_legend(d, max(1.2, bar_end + 0.4), 0.25, clade_labels(style, max(groups) + 1), font, measure)
    elif style.style != "classic":
        _dot_legend(d, width - 1.6, 0.25, font, dot_size, measure)
    return d


def render_matplotlib(drawing):
    """The publication figure (white background), as a matplotlib Figure without pyplot."""
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
    for t in drawing.texts:  # halo as a white box: path effects would export support numbers as outlines
        ax.text(t.x, t.y, t.text, fontsize=t.size, color=t.color, ha=t.ha, va=t.va, rotation=t.rotation,
                rotation_mode="anchor", family=drawing.family, style="italic" if t.italic else "normal",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none") if t.halo else None,
                gid=t.kind, zorder=4)
    return fig


def export_bytes(fig, fmt):
    """SVG (editable text), PDF (embedded TrueType fonts) or 600-dpi PNG bytes of a figure."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown export format {fmt!r}")
    buf = BytesIO()
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        fig.savefig(buf, format=fmt, dpi=600 if fmt == "png" else 100, facecolor="white", bbox_inches="tight",
                    pad_inches=0.05)
    return buf.getvalue()


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
    ink_swap = {INK: "#ece8df", SUPPORT_GREY: "#a59f91"} if dark else {}
    recolor = lambda c: ink_swap.get(c, c)
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
                                 marker=dict(color=[d[2] for d in drawing.dots], size=[d[4] * 1.3 for d in drawing.dots],
                                             line=dict(color=[recolor(d[3]) for d in drawing.dots], width=1)),
                                 text=[d[5] for d in drawing.dots], hovertemplate="%{text}<extra></extra>",
                                 showlegend=False))
    leaves = [t for t in drawing.texts if t.kind == "leaf"]
    if leaves:  # invisible hover targets on the label centres
        centres = [_text_center(t) for t in leaves]
        fig.add_trace(go.Scatter(x=[c[0] for c in centres], y=[c[1] for c in centres], mode="markers",
                                 marker=dict(size=10, opacity=0), text=[t.hover for t in leaves],
                                 hovertemplate="%{text}<extra></extra>", showlegend=False))
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
