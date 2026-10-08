"""Per-column conservation from an MSA and export as B-factors."""
import io
import math
from collections import Counter

import numpy as np
from Bio.PDB import PDBIO
from Bio.PDB.mmcifio import MMCIFIO

from modules.pdb_renumber import extract_header_records
from modules.structures import parse_structure

# Nine grades from variable (1) to conserved (9); ConSurf-style colors.
CONSERVATION_COLORS = ["#10c8d1", "#8cffff", "#d7ffff", "#eaffff", "#ffffff",
                       "#fcedf4", "#fac9de", "#f07dab", "#a02560"]


def column_conservation(aligned, seq_type="Protein", max_gap_fraction=0.5):
    """1 - Shannon entropy / log2(alphabet size) per column; NaN if gaps exceed max_gap_fraction."""
    if not aligned:
        return np.zeros(0)
    n_rows = len(aligned)
    h_max = math.log2(4 if seq_type == "DNA" else 20)
    scores = np.full(len(aligned[0]), np.nan)
    for i in range(len(aligned[0])):
        column = [s[i].upper() for s in aligned]
        residues = [c for c in column if c not in "-."]
        if not residues or 1 - len(residues) / n_rows > max_gap_fraction:
            continue
        counts = Counter(residues)
        total = len(residues)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        scores[i] = max(0.0, 1.0 - entropy / h_max)
    return scores


def reference_conservation(aligned_pairs, reference_id, seq_type="Protein"):
    aligned = dict(aligned_pairs)
    ref = aligned[reference_id]
    scores = column_conservation([s for _, s in aligned_pairs], seq_type)
    result, pos = {}, 0
    for col, char in enumerate(ref):
        if char in "-.":
            continue
        pos += 1
        if not np.isnan(scores[col]):
            result[pos] = float(scores[col])
    return result


def conservation_grade(score):
    return 1 + min(8, int(score * 9))


def write_bfactors(sf, values, default=0.0):
    """Structure text with B-factors set from {(chain, number, icode): value}; others get `default`."""
    structure = parse_structure(sf)
    for model in structure:
        for chain in model:
            for res in chain:
                value = values.get((chain.id, res.id[1], res.id[2].strip()), default)
                for atom in res.get_atoms():
                    atom.set_bfactor(value)
    writer = MMCIFIO() if sf.fmt == "mmCIF" else PDBIO()
    writer.set_structure(structure)
    out = io.StringIO()
    writer.save(out)
    text = out.getvalue()
    return text if sf.fmt == "mmCIF" else extract_header_records(sf.text) + text
