"""Protein-engineering aids from an MSA: back-to-consensus suggestions and a motif/site map. Pure functions."""
import re

import pandas as pd

GAP_CHARS = "-."
CONSENSUS_COLUMNS = ["Position", "Suggestion", "Target", "Consensus", "Consensus frequency", "Conservation",
                     "BLOSUM62", "Sequences", "Score", "Column"]
HIT_COLUMNS = ["Sequence", "Motif", "Start", "End", "Match", "Column"]
SITE_COLUMNS = ["Motif", "Reference position", "Sequences with site", "Fraction", "Sequence IDs", "Column"]
MOTIF_LIBRARY = {
    "N-glycosylation sequon": "N-{P}-[ST]-{P}",
    "Furin site": "R-x-[KR]-R",
    "TEV protease site": "E-N-L-Y-F-Q-[GS]",
    "Thrombin site": "L-V-P-R-G-S",
    "PreScission (HRV 3C) site": "L-E-V-L-F-Q-G-P",
    "Factor Xa site": "I-[ED]-G-R",
    "Enterokinase site": "D-D-D-D-K",
    "Deamidation (NG)": "N-G",
    "Isomerisation (DG)": "D-G",
    "Cysteine": "C",
}
FEW_SEQUENCES = 20


def consensus_suggestions(rows, target_id, min_freq=0.6, min_occupancy=0.5, seq_type="Protein"):
    """Positions where the target departs from a strongly conserved column consensus.

    A column qualifies when its non-gap share is >= min_occupancy, its most frequent residue c* has
    frequency >= min_freq among non-gap residues, and the target has a different residue there (not a gap).
    Ranked by BLOSUM62 score of target->consensus (descending, then consensus frequency).
    Positions are in target numbering."""
    from modules.conservation import column_conservation  # local: keeps the import graph shallow
    from modules.pairwise import load_matrix

    seqs = [s.upper() for _, s in rows]
    target = dict(rows)[target_id].upper()
    conservation = column_conservation(seqs, seq_type)
    blosum = load_matrix("BLOSUM62") if seq_type == "Protein" else None
    out, pos = [], 0
    for j, t in enumerate(target):
        if t in GAP_CHARS:
            continue
        pos += 1
        column = [s[j] for s in seqs]
        residues = [c for c in column if c not in GAP_CHARS]
        if len(residues) / len(column) < min_occupancy:
            continue
        top = max(sorted(set(residues)), key=residues.count)
        freq = residues.count(top) / len(residues)
        if freq < min_freq or top == t:
            continue
        cons = conservation[j]
        cons = 0.0 if cons != cons else float(cons)  # NaN (gap-rich column) -> 0
        try:
            score = int(blosum[t][top]) if blosum is not None else None
        except (KeyError, IndexError):
            score = None
        out.append([pos, f"{t}{pos}{top}", t, top, round(freq, 3), round(cons, 3), score, len(residues),
                    round(freq * cons, 3), j])
    frame = pd.DataFrame(out, columns=CONSENSUS_COLUMNS)
    return frame.sort_values(["Score", "Consensus frequency"], ascending=False, kind="stable").reset_index(drop=True)


def prosite_to_regex(pattern):
    """Convert a PROSITE pattern (e.g. 'N-{P}-[ST]-{P}', 'x(2,4)', '<M', 'K>') to a Python regex."""
    text = pattern.strip().rstrip(".")
    if not text:
        raise ValueError("Empty motif pattern.")
    out = []
    for element in text.split("-"):
        if not element:
            raise ValueError(f"Invalid PROSITE pattern: {pattern!r}")
        anchor_start = element.startswith("<")
        anchor_end = element.endswith(">")
        core = element.lstrip("<").rstrip(">")
        m = re.fullmatch(r"(x|[A-Z]|\[[A-Z]+>?\]|\{[A-Z]+\})(?:\((\d+)(?:,(\d+))?\))?", core)
        if not m:
            raise ValueError(f"Invalid PROSITE element {element!r} in {pattern!r}")
        unit, lo, hi = m.groups()
        if unit == "x":
            regex = "."
        elif unit.startswith("["):
            regex = "[" + unit[1:-1].replace(">", "") + "]"
        elif unit.startswith("{"):
            regex = "[^" + unit[1:-1] + "]"
        else:
            regex = unit
        if lo:
            regex += "{" + lo + ("," + hi if hi else "") + "}"
        out.append(("^" if anchor_start else "") + regex + ("$" if anchor_end else ""))
    return "".join(out)


def compile_motifs(motifs, syntax="prosite"):
    """{name: pattern} -> {name: compiled regex}; raises ValueError naming the bad motif."""
    compiled = {}
    for name, pattern in motifs.items():
        try:
            regex = prosite_to_regex(pattern) if syntax == "prosite" else pattern
            compiled[name] = re.compile(regex, re.IGNORECASE)
        except (ValueError, re.error) as e:
            raise ValueError(f"{name}: {e}") from e
    return compiled


def find_motifs(rows, compiled):
    """All (overlapping) motif hits in each ungapped sequence, mapped back to alignment columns.
    Start/End are 1-based positions in that sequence; Column is the alignment column of the first residue."""
    out = []
    for seq_id, aligned in rows:
        columns = [j for j, c in enumerate(aligned) if c not in GAP_CHARS]
        ungapped = "".join(aligned[j] for j in columns)
        for name, regex in compiled.items():
            for m in re.finditer(f"(?=({regex.pattern}))", ungapped, regex.flags):
                text = m.group(1)
                if not text:
                    continue
                start = m.start()
                out.append([seq_id, name, start + 1, start + len(text), text, columns[start]])
    return pd.DataFrame(out, columns=HIT_COLUMNS)


def site_summary(hits, rows, ref_id):
    """One row per (motif, alignment column): how many sequences carry the site there and the reference position."""
    if hits.empty:
        return pd.DataFrame(columns=SITE_COLUMNS)
    ref = dict(rows)[ref_id]
    ref_pos, k = {}, 0
    for j, c in enumerate(ref):
        if c not in GAP_CHARS:
            k += 1
            ref_pos[j] = k
    n = len(rows)
    out = []
    for (motif, column), group in hits.groupby(["Motif", "Column"], sort=True):
        ids = list(dict.fromkeys(group["Sequence"]))
        out.append([motif, ref_pos.get(column), f"{len(ids)}/{n}", round(len(ids) / n, 3), ", ".join(ids), column])
    frame = pd.DataFrame(out, columns=SITE_COLUMNS)
    return frame.sort_values(["Fraction", "Column"], ascending=[False, True], kind="stable").reset_index(drop=True)
