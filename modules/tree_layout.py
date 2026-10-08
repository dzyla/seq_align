"""Tree geometry for publication figures: rooting, clade groups and layout coordinates. No plotting here."""
import copy
import math
from dataclasses import dataclass, field

LAYOUTS = ("rectangular", "circular", "unrooted")


@dataclass
class NodeInfo:
    """A node in layout coordinates (see TreeGeometry)."""
    a: float
    b: float
    support: float | None
    clade: int | None
    is_leaf: bool
    name: str | None
    branch_length: float


@dataclass
class TreeGeometry:
    """Layout coordinates, by layout:
    - rectangular: a = depth, b = leaf index (0 = top row; parents sit midway between their outer children)
    - circular:    a = radius (depth), b = angle in radians (leaf i at pi/2 - i * 2pi/(n+1), clockwise)
    - unrooted:    a, b = cartesian x, y with the root at (0, 0)
    `edges` are (parent a, parent b, child a, child b, clade): drawn as an L (rectangular: vertical then
    horizontal; circular: arc then radial) or a straight line (unrooted). `leaves` are the leaf NodeInfo
    in display order; `leaf_angles` (unrooted only) give the outward direction at each tip in degrees."""
    layout: str
    cladogram: bool
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    leaves: list = field(default_factory=list)
    leaf_angles: list = field(default_factory=list)
    max_depth: float = 1.0
    has_lengths: bool = True


def prepare_tree(tree, rooting="midpoint", outgroup=(), ladderize=True):
    """Deep copy of `tree`, rooted (midpoint, outgroup or as built) and optionally ladderized."""
    from modules.phylogeny import TreeError  # local: phylogeny imports the renderer, which imports this module

    tree = copy.deepcopy(tree)
    names = [t.name for t in tree.get_terminals()]
    if rooting == "outgroup":
        outgroup = list(outgroup)
        unknown = [o for o in outgroup if o not in names]
        if not outgroup or unknown or len(outgroup) >= len(names):
            raise TreeError("The outgroup must be one or more (but not all) leaves of the tree"
                            + (f"; unknown: {', '.join(unknown)}" if unknown else "") + ".")
        try:
            tree.root_with_outgroup(*outgroup)
        except ValueError as e:
            raise TreeError(f"Cannot root on this outgroup: {e}") from e
    elif rooting == "midpoint":
        if len(names) > 2 and max(tree.distance(t) for t in tree.get_terminals()) > 0:
            tree.root_at_midpoint()
    if ladderize:
        tree.ladderize()
    return tree


def _height(clade, has_lengths):
    """Longest path from `clade` down to a tip (branch lengths, or edge counts when there are none)."""
    if clade.is_terminal():
        return 0.0
    return max(((c.branch_length or 0.0) if has_lengths else 1.0) + _height(c, has_lengths) for c in clade.clades)


def assign_clades(tree, k):
    """Split the tree into k groups by repeatedly splitting the group whose root is highest.
    Returns {clade: group index} for every clade inside a group (none above them), numbered top to bottom."""
    if k <= 1:
        return {}
    has_lengths = any(c.branch_length for c in tree.find_clades())
    groups = [tree.root]
    while len(groups) < k:
        splittable = [g for g in groups if not g.is_terminal()]
        if not splittable:
            break
        top = max(splittable, key=lambda g: _height(g, has_lengths))
        i = groups.index(top)
        groups[i:i + 1] = list(top.clades)
    order = {t: n for n, t in enumerate(tree.get_terminals())}
    groups.sort(key=lambda g: order[g.get_terminals()[0]])
    return {c: n for n, g in enumerate(groups) for c in g.find_clades()}


def _depths(tree, cladogram):
    has_lengths = any(c.branch_length for c in tree.find_clades())
    if not cladogram and has_lengths:
        return tree.depths(), True
    depth = {}

    def walk(clade, d):
        depth[clade] = d
        for c in clade.clades:
            walk(c, d + 1)
    walk(tree.root, 0)
    if cladogram:  # all tips at the same depth; each parent one step left of its closest child
        tip = max(depth.values())

        def align(clade):
            depth[clade] = tip if clade.is_terminal() else min(align(c) for c in clade.clades) - 1
            return depth[clade]
        align(tree.root)
        shift = depth[tree.root]
        depth = {c: d - shift for c, d in depth.items()}
    return depth, has_lengths


def _rows(tree):
    row = {t: float(i) for i, t in enumerate(tree.get_terminals())}

    def assign(clade):
        if clade not in row:
            rs = [assign(c) for c in clade.clades]
            row[clade] = (min(rs) + max(rs)) / 2
        return row[clade]
    assign(tree.root)
    return row


def _node(clade, a, b, clades):
    return NodeInfo(a, b, clade.confidence, clades.get(clade), clade.is_terminal(), clade.name,
                    clade.branch_length or 0.0)


def compute_geometry(tree, layout="rectangular", cladogram=False, clades=None):
    """Geometry of an already prepared tree (see prepare_tree); `clades` is an assign_clades mapping."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}")
    clades = clades or {}
    if layout == "unrooted":
        return _unrooted(tree, clades)
    depth, has_lengths = _depths(tree, cladogram)
    row = _rows(tree)
    n = len(tree.get_terminals())
    if layout == "rectangular":
        coord = {c: (depth[c], row[c]) for c in row}
    else:
        step = 2 * math.pi / (n + 1)
        coord = {c: (depth[c], math.pi / 2 - row[c] * step) for c in row}
    geo = TreeGeometry(layout, cladogram, max_depth=max(depth.values()) or 1.0, has_lengths=has_lengths)
    node_of = {}
    for clade in tree.find_clades(order="preorder"):
        node_of[clade] = _node(clade, *coord[clade], clades)
        geo.nodes.append(node_of[clade])
        for child in clade.clades:
            geo.edges.append((*coord[clade], *coord[child], clades.get(child)))
    geo.leaves = [node_of[t] for t in tree.get_terminals()]
    return geo


def _unrooted(tree, clades):
    """Equal-angle layout: every subtree gets an angular wedge proportional to its number of leaves."""
    has_lengths = any(c.branch_length for c in tree.find_clades())
    lengths = [c.branch_length for c in tree.find_clades() if c.branch_length]
    floor = 0.02 * max(lengths) if lengths else 0.0
    geo = TreeGeometry("unrooted", False, has_lengths=has_lengths)
    count = {c: c.count_terminals() for c in tree.find_clades()}
    node_of, angle_of = {}, {}

    def place(clade, x, y, a0, a1, direction):
        node_of[clade] = _node(clade, x, y, clades)
        geo.nodes.append(node_of[clade])
        if clade.is_terminal():
            angle_of[clade] = math.degrees(direction) % 360
            return
        a = a0
        for child in clade.clades:
            wedge = (a1 - a0) * count[child] / count[clade]
            mid = a + wedge / 2
            length = max(child.branch_length or 0.0, floor) if has_lengths else 1.0
            cx, cy = x + length * math.cos(mid), y + length * math.sin(mid)
            geo.edges.append((x, y, cx, cy, clades.get(child)))
            place(child, cx, cy, a, a + wedge, mid)
            a += wedge
    place(tree.root, 0.0, 0.0, 0.0, 2 * math.pi, 0.0)
    xs = [n.a for n in geo.nodes]
    ys = [n.b for n in geo.nodes]
    geo.max_depth = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    geo.leaves = [node_of[t] for t in tree.get_terminals()]
    geo.leaf_angles = [angle_of[t] for t in tree.get_terminals()]
    return geo
