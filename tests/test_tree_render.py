import math
import random
from io import BytesIO, StringIO

import numpy as np
import pytest
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from PIL import Image

from modules.tree_layout import assign_clades, compute_geometry, prepare_tree
from modules.tree_render import (LABEL_LIMIT, TreeStyle, display_name, export_bytes, place, render_matplotlib,
                                 render_plotly)


def random_tree(n, seed):
    """Seeded random binary tree: mixed-length names, exponential branch lengths, supports 30-100."""
    rng = random.Random(seed)
    nodes = []
    for i in range(n):
        name = "".join(rng.choice("ABCDEFGHIKLMNPQRSTVWY") for _ in range(rng.randint(3, 18)))
        nodes.append(Clade(branch_length=rng.expovariate(10), name=f"{name}_{i}"))
    while len(nodes) > 1:
        a = nodes.pop(rng.randrange(len(nodes)))
        b = nodes.pop(rng.randrange(len(nodes)))
        nodes.append(Clade(branch_length=rng.expovariate(10), clades=[a, b], confidence=float(rng.randint(30, 100))))
    nodes[0].branch_length = 0.0
    return Tree(root=nodes[0], rooted=True)


def leaf_rectangles(fig):
    """Rotated rectangles (corner points, pixels) of every leaf label in a rendered figure."""
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    rects = []
    for t in ax.texts:
        if t.get_gid() != "leaf":
            continue
        rotation = t.get_rotation()
        t.set_rotation(0)
        box = t.get_window_extent(renderer)
        t.set_rotation(rotation)
        x, y = ax.transData.transform(t.get_position())
        w, h = box.width, box.height
        x0 = {"left": 0, "right": -w, "center": -w / 2}[t.get_ha()]
        y0 = {"bottom": 0, "top": -h, "center": -h / 2}[t.get_va()]
        corners = np.array([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)])
        a = math.radians(rotation)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        rects.append(corners @ rot.T + (x, y))
    return rects


def overlaps(p, q, tol=0.5):
    """Separating-axis test for two convex quadrilaterals; True if they overlap by more than tol pixels."""
    for poly in (p, q):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]]) / (np.hypot(*edge) or 1.0)
            pp, qq = p @ axis, q @ axis
            if min(pp.max(), qq.max()) - max(pp.min(), qq.min()) <= tol:
                return False
    return True


def draw(n, layout, cladogram=False, style="classic", seed=None, **style_kw):
    t = prepare_tree(random_tree(n, seed if seed is not None else n), "midpoint")
    clades = assign_clades(t, 4) if style == "clades" else None
    drawing = place(compute_geometry(t, layout, cladogram, clades), TreeStyle(style=style, **style_kw))
    return drawing, render_matplotlib(drawing)


LAYOUTS = [("rectangular", False), ("rectangular", True), ("circular", False), ("unrooted", False)]


@pytest.mark.parametrize("n", [10, 50, 150])
@pytest.mark.parametrize("layout,cladogram", LAYOUTS)
def test_leaf_labels_never_overlap(n, layout, cladogram):
    _, fig = draw(n, layout, cladogram)
    rects = leaf_rectangles(fig)
    assert len(rects) == n
    bad = [(i, j) for i in range(n) for j in range(i + 1, n) if overlaps(rects[i], rects[j])]
    assert not bad, f"{len(bad)} overlapping label pairs"


@pytest.mark.parametrize("layout,cladogram", LAYOUTS)
@pytest.mark.parametrize("style", ["dots", "clades"])
def test_styles_keep_labels_apart(layout, cladogram, style):
    _, fig = draw(40, layout, cladogram, style)
    rects = leaf_rectangles(fig)
    assert not [(i, j) for i in range(40) for j in range(i + 1, 40) if overlaps(rects[i], rects[j])]


def test_forced_small_height_shrinks_font_then_warns():
    drawing, _ = draw(30, "rectangular", height_cm=12.0)
    sizes = {t.size for t in drawing.texts if t.kind == "leaf"}
    assert max(sizes) < 9.0 and drawing.warning is None
    drawing, _ = draw(60, "rectangular", height_cm=3.0)
    assert {t.size for t in drawing.texts if t.kind == "leaf"} == {5.0}
    assert "overlap" in drawing.warning


def test_tight_rows_switch_classic_support_to_dots():
    roomy, _ = draw(30, "rectangular")
    assert [t for t in roomy.texts if t.kind == "support"] and not roomy.dots
    tight, _ = draw(60, "rectangular", height_cm=3.0)
    assert not [t for t in tight.texts if t.kind == "support"] and tight.dots
    assert "shown as dots" in tight.warning


def test_support_values_follow_style_and_threshold():
    t = Phylo.read(StringIO("((A:0.1,B:0.2)99:0.3,(C:0.2,D:0.1)40:0.2);"), "newick")
    geo = compute_geometry(prepare_tree(t, "as_built"), "rectangular")
    classic = place(geo, TreeStyle(style="classic", support_min=50))
    assert [x.text for x in classic.texts if x.kind == "support"] == ["99"]
    dots = place(geo, TreeStyle(style="dots", support_min=0))
    assert len([d for d in dots.dots if d[5]]) == 2 and not [x for x in dots.texts if x.kind == "support"]


def test_display_name_options():
    assert display_name("HBA_HUMAN", TreeStyle()) == "HBA HUMAN"
    assert display_name("HBA_HUMAN", TreeStyle(underscores_to_spaces=False, truncate=3)) == "HBA…"


def test_labels_hidden_above_limit():
    drawing, _ = draw(LABEL_LIMIT + 5, "rectangular", seed=1)
    assert not [t for t in drawing.texts if t.kind == "leaf"]
    assert "hidden" in drawing.warning


def test_exports_have_correct_headers_and_editable_text():
    _, fig = draw(8, "rectangular")
    svg = export_bytes(fig, "svg")
    assert svg.lstrip().startswith(b"<?xml") and b"<text" in svg
    assert export_bytes(fig, "pdf").startswith(b"%PDF")
    png = export_bytes(fig, "png")
    assert png.startswith(b"\x89PNG")
    assert Image.open(BytesIO(png)).info["dpi"][0] == pytest.approx(600, abs=1)
    with pytest.raises(ValueError):
        export_bytes(fig, "eps")


def test_plotly_view_has_every_leaf_and_hover_names():
    drawing, _ = draw(12, "circular", style="clades")
    fig = render_plotly(drawing, dark=True)
    leaf_texts = [t for t in drawing.texts if t.kind == "leaf"]
    assert len(fig.layout.annotations) == len(drawing.texts)
    hover = next(tr for tr in fig.data if tr.mode == "markers" and tr.marker.opacity == 0)
    assert sorted(hover.text) == sorted(t.hover for t in leaf_texts)


@pytest.mark.parametrize("newick", [
    "(A:0.1,B:0.2);",                       # two leaves
    "((A:0,B:0):0,(C:0,D:0):0);",           # all-zero branch lengths (identical sequences)
    "((A,B),(C,(D,E)),F);",                 # topology only (uploaded Newick without lengths)
    "(A:0.1,B:0.2,C:0.3,D:0.1,E:0.05);",    # polytomy
])
@pytest.mark.parametrize("layout,cladogram", LAYOUTS)
@pytest.mark.parametrize("rooting", ["midpoint", "as_built"])
def test_degenerate_trees_render_every_layout(newick, layout, cladogram, rooting):
    t = prepare_tree(Phylo.read(StringIO(newick), "newick"), rooting)
    drawing = place(compute_geometry(t, layout, cladogram, assign_clades(t, 2)), TreeStyle(style="clades"))
    assert len([x for x in drawing.texts if x.kind == "leaf"]) == len(t.get_terminals())
    export_bytes(render_matplotlib(drawing), "svg")


def test_topology_only_tree_has_no_scale_bar():
    t = prepare_tree(Phylo.read(StringIO("((A,B),(C,(D,E)),F);"), "newick"), "as_built")
    drawing = place(compute_geometry(t, "rectangular"), TreeStyle())
    assert not [x for x in drawing.texts if x.kind == "scale"]


def test_night_explore_view_swaps_black_ink():
    drawing, _ = draw(6, "rectangular")
    colours = {tr.line.color for tr in render_plotly(drawing, dark=True).data if tr.mode == "lines"}
    assert "#1a1a1a" not in colours and "#ece8df" in colours
