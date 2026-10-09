"""ESPript-style publication alignment figure: column classes, secondary structure and rendering.

Pure functions (no Streamlit). The figure is drawn in inch coordinates on a matplotlib Figure without
pyplot, so it exports through modules.figures like the tree figures.
"""
import bisect
import math
import os
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrow, Rectangle

from modules.figures import quiet_fonts

GROUPS = ("FYW", "ILVM", "KRH", "DE", "STA", "NQ", "G", "P", "C")
STYLES = {"Classic ESPript": "classic", "Lab notebook": "lab", "Greyscale": "grey"}
PALETTES = {
    "classic": dict(ident_bg="#d6312b", ident_fg="#ffffff", sim_fg="#d6312b", sim_bg=None, frame="#2a5db0",
                    ink="#111111", marker2="#2a5db0"),
    "lab": dict(ident_bg="#0f766e", ident_fg="#ffffff", sim_fg="#0f766e", sim_bg="#d7efe9", frame="#0f766e",
                ink="#1c1b19", marker2="#b45309"),
    "grey": dict(ident_bg="#1c1c1c", ident_fg="#ffffff", sim_fg="#111111", sim_bg="#c4c4c4", frame=None,
                 ink="#111111", marker2="#555555"),
}
RESIDUE_FONTS = ["Courier New", "Liberation Mono", "Nimbus Mono PS", "DejaVu Sans Mono"]
LABEL_FONTS = ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
SYMBOL_FONTS = ["DejaVu Sans"]
MAX_CELLS = 30000
GAP_CHARS = "-."
SS_NAMES = {"alpha": "α", "eta": "η", "beta": "β"}


@dataclass
class FigureOptions:
    style: str = "classic"        # classic | lab | grey
    block: int = 60               # columns per block
    font: float = 8.0             # residue font size, pt
    threshold: float = 0.7        # similarity threshold
    consensus: bool = False       # consensus line under each block
    conservation: bool = True     # conservation bar under each block
    truncate: int = 0             # name truncation (0 = off)
    underscores: bool = True      # show underscores in names as spaces
    seq_type: str = "Protein"     # Protein: identity + similarity groups; DNA/RNA: identity only


def residue_group(residue):
    return next((g for g in GROUPS if residue in g), residue)


def column_classes(seqs, threshold=0.7, seq_type="Protein"):
    """Per column (kind, fraction, group): kind 'ident' when every sequence has the same residue and no gap;
    'sim' when at least `threshold` of the sequences share one residue group; otherwise 'none'.
    Nucleotide alignments (seq_type other than 'Protein') use identity only: every other column is 'none', with
    the fraction of its most common base."""
    n = len(seqs)
    out = []
    for col in zip(*seqs):
        residues = [c.upper() for c in col if c not in GAP_CHARS]
        if n and len(residues) == n and len(set(residues)) == 1:
            out.append(("ident", 1.0, residue_group(residues[0])))
            continue
        if not residues:
            out.append(("none", 0.0, None))
            continue
        counts = {}
        for c in residues:
            g = residue_group(c) if seq_type == "Protein" else c
            counts[g] = counts.get(g, 0) + 1
        group, k = max(counts.items(), key=lambda kv: kv[1])
        frac = k / n
        out.append(("sim", frac, group) if frac >= threshold and seq_type == "Protein" else ("none", frac, group))
    return out


def consensus_line(seqs, classes):
    """ESPript-like consensus: upper case for identical columns, lower case for similar, '.' otherwise."""
    line = []
    for col, (kind, _, group) in zip(zip(*seqs), classes):
        residues = [c.upper() for c in col if c not in GAP_CHARS and residue_group(c.upper()) == group]
        top = max(set(residues), key=residues.count) if residues else "."
        line.append(top if kind == "ident" else top.lower() if kind == "sim" else ".")
    return "".join(line)


def ss_elements(ss, breaks=()):
    """Secondary-structure elements in a per-residue DSSP string (8-state; ' ' or '-' = none):
    [(kind, start, end, label)] with 0-based inclusive positions. H/I runs >= 4 -> alpha (α1...),
    G runs >= 3 -> eta (η1...), E runs >= 2 -> beta (β1...), T runs >= 2 -> turn ('TT').
    breaks: positions where a new element starts even inside a run (touching HELIX/SHEET records)."""
    rules = {"H": ("alpha", 4), "I": ("alpha", 4), "G": ("eta", 3), "E": ("beta", 2), "T": ("turn", 2)}
    counters = {"alpha": 0, "eta": 0, "beta": 0}
    elements, i = [], 0
    while i < len(ss):
        c = ss[i]
        j = i
        while j + 1 < len(ss) and j + 1 not in breaks and (ss[j + 1] == c or (c in "HI" and ss[j + 1] in "HI")):
            j += 1
        if c in rules and j - i + 1 >= rules[c][1]:
            kind = rules[c][0]
            if kind == "turn":
                label = "TT"
            else:
                counters[kind] += 1
                label = f"{SS_NAMES[kind]}{counters[kind]}"
            elements.append((kind, i, j, label))
        i = j + 1
    return elements


def _residue_ss(ref_aligned, chain_seq, chain_ss, breaks=()):
    """Per-reference-residue SS ('-' where the chain has no residue), the reference residue -> alignment column
    index, and the reference residues where a chain break (a new SS record) falls."""
    from modules.pdb_renumber import align_target_to_chain  # local: pdb_renumber imports pairwise

    columns = [j for j, c in enumerate(ref_aligned) if c not in GAP_CHARS]
    ref = "".join(ref_aligned[j] for j in columns)
    if not ref or not chain_seq:
        return "-" * len(ref), columns, set()
    aligned_ref, aligned_chain, _ = align_target_to_chain(ref, chain_seq)
    ref_to_chain, ri, ci = {}, 0, 0
    for a, b in zip(aligned_ref, aligned_chain):
        if a != "-" and b != "-":
            ref_to_chain[ri] = ci
        ri += a != "-"
        ci += b != "-"
    ss = "".join(chain_ss[ref_to_chain[i]] if i in ref_to_chain else "-" for i in range(len(ref)))
    ref_breaks = {i for i in range(len(ref)) if i in ref_to_chain and
                  (ref_to_chain[i] in breaks or
                   (i - 1 in ref_to_chain and any(ref_to_chain[i - 1] < b <= ref_to_chain[i] for b in breaks)))}
    return ss, columns, ref_breaks


def map_ss_to_alignment(ref_aligned, chain_seq, chain_ss):
    """Transfer per-residue secondary structure of a structure chain onto the columns of the aligned reference.
    Returns a string as long as `ref_aligned`; columns without a mapped residue are ' '."""
    ss, columns, _ = _residue_ss(ref_aligned, chain_seq, chain_ss)
    out = [" "] * len(ref_aligned)
    for col, code in zip(columns, ss):
        out[col] = " " if code == "-" else code
    return "".join(out)


def ss_elements_by_column(ref_aligned, chain_seq, chain_ss, breaks=()):
    """Secondary-structure elements of the reference in alignment-column coordinates: built on the per-residue
    SS of the reference (so a reference gap never splits or shortens an element), then each element runs from its
    first to its last residue's column. breaks: chain residue indices where a new SS record starts."""
    ss, columns, ref_breaks = _residue_ss(ref_aligned, chain_seq, chain_ss, breaks)
    return [(kind, columns[a], columns[b], label) for kind, a, b, label in ss_elements(ss, ref_breaks)]


def _column_string_elements(ref_aligned, ss_by_column):
    """Elements of a per-column SS string (map_ss_to_alignment), read per reference residue (gaps bridged)."""
    columns = [j for j, c in enumerate(ref_aligned) if c not in GAP_CHARS]
    ss = "".join(ss_by_column[j] if ss_by_column[j] != " " else "-" for j in columns)
    return [(kind, columns[a], columns[b], label) for kind, a, b, label in ss_elements(ss)]


def find_mkdssp():
    return shutil.which("mkdssp") or shutil.which("dssp")


def secondary_structure(sf, chain_id):
    """(chain sequence, per-residue SS string, source, reason, breaks) for one protein chain of a StructureFile.
    Uses DSSP (mkdssp); falls back to the file's HELIX/SHEET (PDB) or _struct_conf/_struct_sheet_range (mmCIF)
    records; source is 'DSSP', 'file records' or 'none' (then `reason` says why). breaks: chain residue indices
    where a file record starts, so touching records stay separate elements (empty for DSSP)."""
    from modules.structures import chain_residue_table, parse_structure

    structure = parse_structure(sf)
    table = chain_residue_table(structure)
    if chain_id not in table:
        return "", "", "none", f"Chain {chain_id} has no protein residues.", ()
    residues = table[chain_id]
    seq = "".join(r.aa for r in residues)
    exe = find_mkdssp()
    dssp_error = ""
    if exe:
        try:
            ss = _dssp(sf, structure, chain_id, residues, exe)
            if ss.strip("-"):
                return seq, ss, "DSSP", "", ()
        except Exception as e:  # fall back to file records below
            dssp_error = f"DSSP failed ({e.__class__.__name__}: {e})."
    ss, breaks = _ss_from_records(sf, chain_id, residues)
    if ss.strip("-"):
        return (seq, ss, "file records", "" if exe else "mkdssp is not installed; using the file's HELIX/SHEET records.",
                breaks)
    if not exe:
        reason = "DSSP is not available and the file has no secondary-structure records."
    else:
        reason = dssp_error or "DSSP found no secondary structure and the file has no records."
    return seq, "-" * len(seq), "none", reason, ()


def _dssp(sf, structure, chain_id, residues, exe):
    from Bio.PDB.DSSP import DSSP

    suffix = ".cif" if sf.fmt == "mmCIF" else ".pdb"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(sf.data)
        path = handle.name
    try:
        dssp = DSSP(list(structure)[0], path, dssp=exe)
        by_residue = {(key[1][1], key[1][2].strip()): dssp[key][2] for key in dssp.keys() if key[0] == chain_id}
    finally:
        os.unlink(path)
    return "".join(by_residue.get((r.number, r.icode), "-") for r in residues)


def _ss_from_records(sf, chain_id, residues):
    ranges = []  # (code, start, end) in author numbering
    if sf.fmt == "mmCIF":
        from io import StringIO

        from Bio.PDB.MMCIF2Dict import MMCIF2Dict

        d = MMCIF2Dict(StringIO(sf.text))

        def column(name):
            v = d.get(name, [])
            return v if isinstance(v, list) else [v]
        for chain, beg, end in zip(column("_struct_conf.beg_auth_asym_id"), column("_struct_conf.beg_auth_seq_id"),
                                   column("_struct_conf.end_auth_seq_id")):
            if chain == chain_id:
                ranges.append(("H", int(beg), int(end)))
        for chain, beg, end in zip(column("_struct_sheet_range.beg_auth_asym_id"),
                                   column("_struct_sheet_range.beg_auth_seq_id"),
                                   column("_struct_sheet_range.end_auth_seq_id")):
            if chain == chain_id:
                ranges.append(("E", int(beg), int(end)))
    else:
        for line in sf.text.splitlines():
            try:
                if line.startswith("HELIX ") and line[19] == chain_id:
                    ranges.append(("H", int(line[21:25]), int(line[33:37])))
                elif line.startswith("SHEET ") and line[21] == chain_id:
                    ranges.append(("E", int(line[22:26]), int(line[33:37])))
            except (ValueError, IndexError):
                continue
    ss, breaks, previous = [], [], None
    for i, r in enumerate(residues):
        record = next(((c, a, b) for c, a, b in ranges if a <= r.number <= b), None)
        ss.append(record[0] if record else "-")
        if record and record != previous:
            breaks.append(i)
        previous = record
    return "".join(ss), tuple(breaks)


def layout_blocks(c0, c1, block):
    """Column ranges [(start, end_exclusive)] of the wrapped blocks covering columns c0..c1-1."""
    return [(s, min(c1, s + block)) for s in range(c0, c1, block)]


def region_columns(ref_aligned, start, end):
    """Alignment column range (c0, c1_exclusive) covering reference residues start..end (1-based, inclusive)."""
    cols = [j for j, c in enumerate(ref_aligned) if c not in GAP_CHARS]
    if not cols:
        return 0, len(ref_aligned)
    start = max(1, min(start, len(cols)))
    end = max(start, min(end, len(cols)))
    return cols[start - 1], cols[end - 1] + 1


def cell_count(n_rows, c0, c1):
    return n_rows * max(0, c1 - c0)


def display_name(name, options):
    name = name.replace("_", " ") if options.underscores else name
    if options.truncate and len(name) > options.truncate:
        name = name[:options.truncate] + "…"
    return name


def figure_caption(options, ss_source, ss_label=None):
    """Plain-language description of the colouring rule and the secondary-structure source."""
    if options.seq_type == "Protein":
        rule = (f"Identical columns: same residue in every sequence, no gaps. Similar residues: at least "
                f"{options.threshold:.0%} of sequences in one group (FYW, ILVM, KRH, DE, STA, NQ, G, P, C).")
    else:
        rule = "Identical columns: same residue in every sequence, no gaps (DNA: identity only)."
    if ss_source == "none" or not ss_label:
        return rule + " No secondary structure shown."
    return rule + f" Secondary structure of {ss_label} from {ss_source}."


def _text_width(renderer, fig, text, size, family, style="normal"):
    t = fig.text(0, 0, text, fontsize=size, family=family, style=style)
    width = t.get_window_extent(renderer).width / fig.dpi
    t.remove()
    return width


def render_alignment_figure(rows, ref_id, ss=None, options=None, markers=None, ss_label=None, columns=None):
    """ESPript-style figure of aligned (id, sequence) rows, wrapped in blocks.

    ss: the reference's secondary-structure elements in alignment columns (ss_elements_by_column), or a
    per-column SS string (map_ss_to_alignment), or None. Elements run across reference-gap columns.
    markers: {legend label: set of alignment columns}; at most two sets are drawn (star, then triangle); only
    sets with a column in the shown region appear in the legend.
    columns: (c0, c1_exclusive) alignment columns to show; default all. Columns that are gaps in every row are
    dropped; the ruler keeps the reference numbering."""
    with quiet_fonts():
        return _render(rows, ref_id, ss, options or FigureOptions(), markers or {}, ss_label, columns)


def _render(rows, ref_id, ss, o, markers, ss_label, columns):
    pal = PALETTES[o.style]
    full = dict(rows)
    L = len(rows[0][1]) if rows else 0
    c0, c1 = columns or (0, L)
    kept = [j for j in range(c0, c1) if any(s[j] not in GAP_CHARS for _, s in rows)]
    position = {j: p for p, j in enumerate(kept)}
    rows = [(i, "".join(s[j] for j in kept)) for i, s in rows]
    seqs = [s for _, s in rows]
    classes = column_classes(seqs, o.threshold, o.seq_type)
    cons = consensus_line(seqs, classes) if o.consensus else ""
    resnum, k = {}, 0
    for j, c in enumerate(full[ref_id]):
        k += c not in GAP_CHARS
        if c not in GAP_CHARS and j in position:
            resnum[position[j]] = k
    if isinstance(ss, str):
        ss = _column_string_elements(full[ref_id], ss) if ss.strip() else []
    elements = []  # (kind, first, last, label, starts inside the shown columns) in shown-column positions
    for kind, a, b, label in ss or []:
        pa, pb = bisect.bisect_left(kept, a), bisect.bisect_right(kept, b) - 1
        if pa <= pb:
            elements.append((kind, pa, pb, label, a in position))
    symbols = ("\u2605", "\u25b2")  # black star, black up-pointing triangle
    marker_sets = [(label, {position[j] for j in cols if j in position}, symbol, colour)
                   for (label, cols), symbol, colour in zip(markers.items(), symbols, (pal["ident_bg"], pal["marker2"]))]
    marker_sets = [m for m in marker_sets if m[1]]
    c0, c1 = 0, len(kept)

    font = o.font
    cw = font * 0.62 / 72 * 1.25
    rh = font * 1.45 / 72
    probe = Figure(dpi=72)
    renderer = FigureCanvasAgg(probe).get_renderer()
    names = [display_name(i, o) for i, _ in rows]
    name_w = max([_text_width(renderer, probe, n, font * 0.9, LABEL_FONTS, "italic") for n in names] +
                 [_text_width(renderer, probe, ss_label or ref_id, font * 0.85, LABEL_FONTS)]) + 0.15
    has_ss = bool(elements)
    head = rh * (3.2 if has_ss else 1.6)
    extra = (rh if o.consensus else 0) + (rh * 0.9 if o.conservation else 0) + rh * 1.0 * len(marker_sets)
    blocks = layout_blocks(c0, c1, o.block)
    block_h = head + len(rows) * rh + extra + 0.2
    legend_h = rh * 1.4 if marker_sets else 0
    width = name_w + min(o.block, c1 - c0) * cw + 0.45
    height = max(len(blocks), 1) * block_h + legend_h + 0.1
    fig = Figure(figsize=(width, height), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    mono = dict(family=RESIDUE_FONTS, fontsize=font)

    for bi, (b0, b1) in enumerate(blocks):
        y0 = 0.05 + bi * block_h
        X = lambda j: name_w + (j - b0) * cw  # noqa: E731
        if has_ss:
            yss = y0 + rh * 0.9
            ax.text(0.05, yss, ss_label or ref_id, fontsize=font * 0.85, family=LABEL_FONTS, va="center",
                    color="#555555", gid="ss_label")
            for kind, a, b, label, starts_here in elements:
                if b < b0 or a >= b1:
                    continue
                xa, xb = X(max(a, b0)), X(min(b, b1 - 1)) + cw
                if kind in ("alpha", "eta"):
                    amp = rh * (0.32 if kind == "alpha" else 0.2)
                    xs = np.linspace(xa, xb, max(8, int((xb - xa) / 0.01)))
                    ax.plot(xs, yss + amp * np.sin((xs - xa) / (cw * 0.9) * math.pi), color=pal["ink"], lw=1.0)
                elif kind == "beta":
                    ax.add_patch(FancyArrow(xa, yss, xb - xa, 0, width=rh * 0.28, head_width=rh * 0.6,
                                            head_length=min(cw * 1.2, (xb - xa) * 0.5), length_includes_head=True,
                                            color=pal["ink"], lw=0))
                else:
                    ax.text((xa + xb) / 2, yss, "TT", ha="center", va="center", fontsize=font * 0.75,
                            color=pal["ink"], family=LABEL_FONTS)
                if kind != "turn" and starts_here and a >= b0:
                    ax.text((xa + xb) / 2, yss - rh * 0.75, label, ha="center", va="center", fontsize=font * 0.8,
                            color=pal["ink"], family=LABEL_FONTS, gid="ss_element")
        yr = y0 + head - rh * 0.9
        for j in range(b0, b1):
            if resnum.get(j) and resnum[j] % 10 == 0:
                ax.text(X(j) + cw / 2, yr, str(resnum[j]), ha="center", va="center", fontsize=font * 0.8,
                        color="#444444", family=LABEL_FONTS, gid="ruler")
        ys = y0 + head
        if pal["frame"]:
            j = b0
            while j < b1:
                if classes[j][0] != "none":
                    e = j
                    while e + 1 < b1 and classes[e + 1][0] != "none":
                        e += 1
                    ax.add_patch(Rectangle((X(j), ys - 0.01), (e - j + 1) * cw, len(rows) * rh + 0.02, fill=False,
                                           ec=pal["frame"], lw=0.8))
                    j = e + 1
                else:
                    j += 1
        for r, ((_, seq), name) in enumerate(zip(rows, names)):
            y = ys + r * rh
            ax.text(0.05, y + rh / 2, name, fontsize=font * 0.9, family=LABEL_FONTS, style="italic", va="center",
                    color=pal["ink"], gid="name")
            for j in range(b0, b1):
                c = seq[j]
                kind, _, group = classes[j]
                fg, bg, weight = pal["ink"], None, "normal"
                if kind == "ident":
                    fg, bg, weight = pal["ident_fg"], pal["ident_bg"], "bold"
                elif kind == "sim" and c not in GAP_CHARS and residue_group(c.upper()) == group:
                    fg, bg, weight = pal["sim_fg"], pal["sim_bg"], "bold"
                if bg:
                    ax.add_patch(Rectangle((X(j), y), cw, rh, color=bg, lw=0))
                ax.text(X(j) + cw / 2, y + rh / 2, "." if c in GAP_CHARS else c, ha="center", va="center", color=fg,
                        fontweight=weight, gid="residue", **mono)
        y = ys + len(rows) * rh
        if o.consensus:
            ax.text(0.05, y + rh / 2, "consensus", fontsize=font * 0.8, family=LABEL_FONTS, va="center",
                    color="#555555")
            for j in range(b0, b1):
                ax.text(X(j) + cw / 2, y + rh / 2, cons[j], ha="center", va="center", color=pal["ink"],
                        gid="consensus", **mono)
            y += rh
        if o.conservation:
            for j in range(b0, b1):
                h = rh * 0.75 * classes[j][1]
                if h > 0:
                    ax.add_patch(Rectangle((X(j) + cw * 0.1, y + rh * 0.8 - h), cw * 0.8, h, color=pal["ident_bg"],
                                           alpha=0.55, lw=0))
            y += rh * 0.9
        for label, cols, symbol, colour in marker_sets:
            for j in sorted(cols):
                if b0 <= j < b1:
                    ax.text(X(j) + cw / 2, y + rh / 2, symbol, ha="center", va="center", fontsize=font,
                            color=colour, family=SYMBOL_FONTS, gid="marker")
            y += rh
    if marker_sets:
        x, y = name_w, height - legend_h / 2 - 0.05
        for label, _, symbol, colour in marker_sets:
            ax.text(x, y, symbol, fontsize=font, color=colour, family=SYMBOL_FONTS, va="center", gid="legend")
            ax.text(x + cw * 1.3, y, label, fontsize=font * 0.85, family=LABEL_FONTS, va="center", color="#333333",
                    gid="legend")
            x += cw * 1.3 + _text_width(renderer, probe, label, font * 0.85, LABEL_FONTS) + 0.3
    return fig
