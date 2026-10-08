"""Per-column conservation from an MSA and export as B-factors."""
import io
import math
from collections import Counter

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


PROTEIN_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
DNA_ALPHABET = list("ACGT")


def logo_matrix(aligned, seq_type="Protein", start=1, end=None):
    """Information content (bits) per residue for alignment columns start..end (1-based, inclusive),
    against a uniform background over the full alphabet, without pseudocounts."""
    alphabet = DNA_ALPHABET if seq_type == "DNA" else PROTEIN_ALPHABET
    window = [s[start - 1:end].upper() for s in aligned]
    width = len(window[0]) if window else 0
    if not any(c in alphabet for s in window for c in s):
        # logomaker rejects windows with no countable residues (all gaps or ambiguity codes)
        matrix = pd.DataFrame(0.0, index=range(width), columns=alphabet)
    else:
        counts = logomaker.alignment_to_matrix(window, to_type="counts", characters_to_ignore=".-XNBZJOU*")
        counts = counts.reindex(columns=alphabet, fill_value=0)
        # Information per column against a uniform background (logomaker's definition); gap-only rows are zero.
        values = counts.to_numpy(dtype=float)
        totals = values.sum(axis=1, keepdims=True)
        probs = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
        q = 1.0 / len(alphabet)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(probs > 0, probs * np.log2(probs / q), 0.0)
        info = terms.sum(axis=1, keepdims=True)
        matrix = pd.DataFrame(probs * info, index=counts.index, columns=alphabet)
    matrix.index = matrix.index + start
    return matrix


def logo_figure(matrix, seq_type):
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(matrix)), 3))
    logomaker.Logo(matrix, ax=ax, color_scheme="classic" if seq_type == "DNA" else "chemistry")
    ax.set_xlabel("Alignment position")
    ax.set_ylabel("Bits")
    fig.tight_layout()
    return fig


def identity_matrix_frame(aligned_pairs):
    from modules.clustering import msa_identity_matrix  # local: clustering -> msa -> conservation cycle

    ids = [i for i, _ in aligned_pairs]
    values = msa_identity_matrix([s for _, s in aligned_pairs]) * 100
    return pd.DataFrame(values.round(1), index=ids, columns=ids)
