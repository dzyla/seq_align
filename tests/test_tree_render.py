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


def test_support_numbers_are_editable_svg_text():
    t = Phylo.read(StringIO("((A:0.1,B:0.2)90:0.3,(C:0.2,D:0.1)70:0.2);"), "newick")
    geo = compute_geometry(prepare_tree(t, "as_built"), "rectangular")
    svg = export_bytes(render_matplotlib(place(geo, TreeStyle(style="classic", support_min=50))), "svg")
    assert b">90<" in svg and b">70<" in svg


def test_palette_has_twelve_distinct_colours_without_black():
    from modules.tree_render import PALETTE
    assert len(set(PALETTE)) == 12 and "#000000" not in PALETTE


def test_night_explore_recolours_dot_faces():
    t = Phylo.read(StringIO("((A:0.1,B:0.2)99:0.3,(C:0.2,D:0.1)40:0.2);"), "newick")
    drawing = place(compute_geometry(prepare_tree(t, "as_built"), "rectangular"), TreeStyle(style="dots",
                                                                                           support_min=0))
    marker = next(tr for tr in render_plotly(drawing, dark=True).data
                  if tr.mode == "markers" and tr.marker.opacity != 0)
    faces = dict(zip(marker.text, marker.marker.color))
    assert faces["support 99"] == "#ece8df" and faces["support 40"] == "#1a1916"
    assert "#1a1a1a" not in set(marker.marker.line.color)


def support_rectangles(fig):
    """Corner points (pixels) of every rendered support number (unrotated)."""
    renderer = fig.canvas.get_renderer()
    rects = []
    for t in fig.axes[0].texts:
        if t.get_gid() == "support":
            b = t.get_window_extent(renderer)
            rects.append(np.array([(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)]))
    return rects


@pytest.mark.parametrize("layout", ["rectangular", "circular", "unrooted"])
@pytest.mark.parametrize("n,seed", [(50, 50), (50, 7), (10, 10)])
def test_classic_support_numbers_never_overlap(layout, n, seed):
    _, fig = draw(n, layout, seed=seed)
    supports, leaves = support_rectangles(fig), leaf_rectangles(fig)
    bad = [(i, j) for i in range(len(supports)) for j in range(i + 1, len(supports))
           if overlaps(supports[i], supports[j])]
    bad += [(i, -j) for i, s in enumerate(supports) for j, leaf in enumerate(leaves) if overlaps(s, leaf)]
    assert not bad, f"{len(bad)} overlapping support boxes"


@pytest.mark.parametrize("layout", ["circular", "unrooted"])
def test_crowded_classic_support_falls_back_to_dots_with_warning(layout):
    roomy, _ = draw(10, layout)
    assert [t for t in roomy.texts if t.kind == "support"] and roomy.warning is None
    crowded, _ = draw(50, layout)
    assert not [t for t in crowded.texts if t.kind == "support"] and crowded.dots
    assert crowded.warning == "Support numbers would overlap; support is shown as dots."
    assert "≥95" in {t.text for t in crowded.texts if t.kind == "legend"}


def test_clade_legend_wraps_inside_narrow_figure():
    t = prepare_tree(random_tree(50, 50), "midpoint")
    drawing = place(compute_geometry(t, "circular", False, assign_clades(t, 12)),
                    TreeStyle(style="clades", width_cm=8.0))
    legend = [x for x in drawing.texts if x.kind == "legend"]
    assert {f"Clade {i}" for i in range(1, 13)} <= {x.text for x in legend}
    assert all(x.x + x.width <= drawing.width + 1e-9 for x in legend)
    for xs, ys, _, alpha in drawing.fills:
        if alpha == 1.0:
            assert min(xs) >= 0 and max(xs) <= drawing.width and min(ys) >= 0 and max(ys) <= drawing.height
    assert all(0 <= x.y <= drawing.height for x in drawing.texts)


@pytest.mark.parametrize("layout", ["circular", "unrooted"])
def test_dot_legend_only_when_tree_has_supports(layout):
    t = prepare_tree(Phylo.read(StringIO("((A:0.1,B:0.2):0.3,(C:0.2,D:0.1):0.2,E:0.4);"), "newick"), "midpoint")
    for style in ("dots", "clades"):
        drawing = place(compute_geometry(t, layout, False, assign_clades(t, 2)), TreeStyle(style=style))
        assert "≥95" not in {x.text for x in drawing.texts}


@pytest.mark.parametrize("layout", ["circular", "unrooted"])
def test_clade_colour_shows_dot_legend_below_clade_legend(layout):
    drawing, _ = draw(20, layout, style="clades")
    legend = {x.text: x for x in drawing.texts if x.kind == "legend"}
    assert "≥95" in legend and "Clade 1" in legend
    assert legend["≥95"].y < legend["Clade 1"].y


def test_svg_records_the_chosen_font_family():
    drawing, _ = draw(6, "rectangular", font_family="Arial")
    assert drawing.font_choice == "Arial"
    svg = export_bytes(render_matplotlib(drawing), "svg").decode()
    assert "'Arial'" in svg


def test_render_and_export_emit_no_findfont_warnings(caplog):
    drawing, _ = draw(6, "rectangular", font_family="Times New Roman")
    with caplog.at_level("WARNING", logger="matplotlib.font_manager"):
        export_bytes(render_matplotlib(drawing), "svg")
        export_bytes(render_matplotlib(drawing), "pdf")
    assert not [r for r in caplog.records if "findfont" in r.getMessage()]


NAMED = "((A:0.1,B:0.2)N1/95:0.3,((C:0.2,D:0.1)N2/80:0.2,E:0.3)N3/70:0.2)N0;"


def named_geometry(layout):
    from modules.tree_layout import join_node_labels, split_node_labels
    t = split_node_labels(Phylo.read(StringIO(NAMED), "newick"))
    assert {c.name: c.confidence for c in t.get_nonterminals()} == {"N0": None, "N1": 95, "N3": 70, "N2": 80}
    out = StringIO()
    Phylo.write(join_node_labels(t), out, "newick")                     # the Newick round trip keeps both
    again = split_node_labels(Phylo.read(StringIO(out.getvalue()), "newick"))
    assert {c.name: c.confidence for c in again.get_nonterminals()} == {"N0": None, "N1": 95, "N3": 70, "N2": 80}
    return compute_geometry(prepare_tree(t, "as_built"), layout)


@pytest.mark.parametrize("layout", ["rectangular", "circular", "unrooted"])
def test_internal_labels_and_highlight(layout):
    geo = named_geometry(layout)
    plain = place(geo, TreeStyle())
    assert not [t for t in plain.texts if t.kind == "node"]               # default output unchanged
    labelled = place(geo, TreeStyle(internal_labels=True))
    names = {t.text for t in labelled.texts if t.kind == "node"}
    assert {"N1", "N2", "N3"} <= names
    svg = export_bytes(render_matplotlib(labelled), "svg")
    assert b">N1<" in svg and b">N2<" in svg
    fig = render_plotly(labelled)
    assert {"N1", "N2", "N3"} <= {a.text for a in fig.layout.annotations}
    marked = place(geo, TreeStyle(internal_labels=True, highlight="N2"))
    extra = [d for d in marked.dots if d not in labelled.dots]
    assert len(extra) == 1 and "N2" in extra[0][5] and len(marked.dots) == len(labelled.dots) + 1
    assert place(geo, TreeStyle(highlight="nonexistent")).dots == plain.dots


def test_node_labels_do_not_overlap_leaf_or_support_labels():
    from modules.tree_render import _any_overlap, _corners
    t = prepare_tree(random_tree(30, 3), "midpoint")
    for k, clade in enumerate(t.get_nonterminals()):
        clade.name = f"Node{k + 1}"
    for layout in ("rectangular", "circular", "unrooted"):
        d = place(compute_geometry(t, layout), TreeStyle(internal_labels=True))
        nodes = [x for x in d.texts if x.kind == "node"]
        others = [x for x in d.texts if x.kind in ("leaf", "support")]
        assert nodes and not _any_overlap(nodes, others)
        for x in nodes:
            corners = _corners(x)                                         # inside the figure
            assert corners.min() >= 0 and (corners.max(axis=0) <= (d.width, d.height)).all()
