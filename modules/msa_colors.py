"""MSA colour schemes (Jalview-style): each maps an alignment to a per-cell colour index, a colour list and a
legend. Pure functions, shared by the overview heatmap, the sequence viewer and the static PNG."""
from dataclasses import dataclass

import numpy as np

GAP_CHARS = "-."
WHITE = "#ffffff"
SCHEMES = {
    "Protein": ("Residue", "ClustalX", "Zappo", "Taylor", "Hydrophobicity", "Percent identity", "Conservation",
                "Similarity to reference"),
    "DNA": ("Nucleotide", "Percent identity", "Conservation", "Similarity to reference"),
}
DESCRIPTIONS = {
    "Residue": "Every residue type in its usual Clustal colour, regardless of the column.",
    "ClustalX": "Jalview's ClustalX rules: a residue is coloured only when its column's composition supports it.",
    "Zappo": "Physicochemical groups.",
    "Taylor": "A distinct colour for every amino acid (Taylor 1997).",
    "Hydrophobicity": "Kyte-Doolittle scale: blue hydrophilic to red hydrophobic.",
    "Percent identity": "Residues matching their column's most common residue, shaded by how many sequences share it.",
    "Conservation": "Every residue shaded by its column's conservation (1 - entropy); gap-rich columns blank.",
    "Similarity to reference": "Identical to the reference, similar (BLOSUM62 > 0) or different.",
    "Nucleotide": "A, C, G and T/U in their usual colours.",
}

RESIDUE = {**dict.fromkeys("AILMFWV", "#80a0f0"), "C": "#f08080", **dict.fromkeys("KR", "#f01505"),
           **dict.fromkeys("DE", "#c048c0"), **dict.fromkeys("NQST", "#15c015"), "G": "#f09048", "P": "#c0c000",
           **dict.fromkeys("HY", "#15a4a4")}
NUCLEOTIDE = {"A": "#64f73f", "C": "#ffb340", "G": "#eb413c", "T": "#3c88ee", "U": "#3c88ee"}
ZAPPO = {**dict.fromkeys("ILVAM", "#ffafaf"), **dict.fromkeys("FWY", "#ffc800"), **dict.fromkeys("KRH", "#6464ff"),
         **dict.fromkeys("DE", "#ff0000"), **dict.fromkeys("STNQ", "#00ff00"), **dict.fromkeys("PG", "#ff00ff"),
         "C": "#ffff00"}
ZAPPO_LEGEND = [("Aliphatic ILVAM", "#ffafaf"), ("Aromatic FWY", "#ffc800"), ("Positive KRH", "#6464ff"),
                ("Negative DE", "#ff0000"), ("Hydrophilic STNQ", "#00ff00"), ("Conformational PG", "#ff00ff"),
                ("Cysteine C", "#ffff00")]
TAYLOR = {"A": "#ccff00", "R": "#0000ff", "N": "#cc00ff", "D": "#ff0000", "C": "#ffff00", "Q": "#ff00cc",
          "E": "#ff0066", "G": "#ff9900", "H": "#0066ff", "I": "#66ff00", "L": "#33ff00", "K": "#6600ff",
          "M": "#00ff00", "F": "#00ff66", "P": "#ffcc00", "S": "#ff3300", "T": "#ff6600", "W": "#00ccff",
          "Y": "#00ffcc", "V": "#99ff00"}
KYTE_DOOLITTLE = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
                  "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5,
                  "N": -3.5, "K": -3.9, "R": -4.5}
HYDRO_RAMP = ("#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b")
IDENTITY_SHADES = ((0.8, "#6464ff", "> 80%"), (0.6, "#9999ff", "> 60%"), (0.4, "#ccccff", "> 40%"))
CONSERVATION_RAMP = ("#f0fdfa", "#ccfbf1", "#99f6e4", "#5eead4", "#2dd4bf", "#14b8a6", "#0d9488", "#0f766e",
                     "#115e59", "#134e4a")
SIMILARITY = (("Identical", "#5fb8a8"), ("Similar (BLOSUM62 > 0)", "#cde9e3"), ("Different", "#f6c9c4"))

# Jalview ClustalX colours and rules: (colour, residues coloured, [(threshold, group, combined?)])
# A "combined" group means the summed fraction of its residues; otherwise any single residue of the group.
CLUSTALX_COLOURS = {"BLUE": "#80a0f0", "RED": "#f01505", "MAGENTA": "#c048c0", "GREEN": "#15c015",
                    "PINK": "#f08080", "ORANGE": "#f09048", "YELLOW": "#c0c000", "CYAN": "#15a4a4"}
HYDROPHOBIC = "WLVIMAFCHP"
CLUSTALX_RULES = [
    ("PINK", "C", [(0.85, "C", True)]),
    ("BLUE", "AILMFWVC", [(0.60, HYDROPHOBIC, True)]),
    ("RED", "KR", [(0.60, "KR", True), (0.80, "KRQ", False)]),
    ("MAGENTA", "E", [(0.60, "KR", True), (0.50, "QE", True), (0.50, "ED", True), (0.85, "EQD", False)]),
    ("MAGENTA", "D", [(0.60, "KR", True), (0.85, "KRQ", False), (0.50, "ED", True)]),
    ("GREEN", "N", [(0.50, "N", True), (0.85, "NY", False)]),
    ("GREEN", "Q", [(0.60, "KR", True), (0.50, "QE", True), (0.85, "QEKR", False)]),
    ("GREEN", "ST", [(0.60, HYDROPHOBIC, True), (0.50, "TS", True), (0.85, "ST", False)]),
    ("ORANGE", "G", [(0.0, "G", True)]),
    ("YELLOW", "P", [(0.0, "P", True)]),
    ("CYAN", "HY", [(0.60, HYDROPHOBIC, True), (0.85, "WYACPQFHILMV", False)]),
]
CLUSTALX_LEGEND = [("Hydrophobic", "#80a0f0"), ("Positive", "#f01505"), ("Negative", "#c048c0"),
                   ("Polar", "#15c015"), ("Cysteine", "#f08080"), ("Glycine", "#f09048"), ("Proline", "#c0c000"),
                   ("Aromatic", "#15a4a4"), ("Not conserved", WHITE)]


@dataclass
class Coloring:
    index: np.ndarray        # uint8 N x L: index into colors (0 = white)
    colors: list             # hex colours; colors[0] is white
    legend: list             # [(label, hex)]


def _letters(seqs):
    return np.array([list(s.upper()) for s in seqs]) if seqs else np.zeros((0, 0), dtype="<U1")


def _from_lookup(letters, lookup, legend):
    colors = [WHITE] + sorted(set(lookup.values()) - {WHITE})
    pos = {c: k for k, c in enumerate(colors)}
    table = np.zeros(256, dtype=np.uint8)
    for residue, color in lookup.items():
        table[ord(residue)] = pos[color]
    codes = np.frombuffer("".join("".join(r) for r in letters).encode("ascii", "replace"), dtype=np.uint8)
    return Coloring(table[codes].reshape(letters.shape), colors, legend)


def _rule_met(fraction, threshold, group, combined):
    if combined:
        return sum(fraction.get(r, 0.0) for r in group) > threshold
    return any(fraction.get(r, 0.0) > threshold for r in group)


def clustalx_colors(letters):
    n, L = letters.shape
    colors = [WHITE] + list(dict.fromkeys(CLUSTALX_COLOURS.values()))
    pos = {c: k for k, c in enumerate(colors)}
    index = np.zeros((n, L), dtype=np.uint8)
    for j in range(L):
        column = letters[:, j]
        residues, counts = np.unique(column, return_counts=True)
        fraction = {r: c / n for r, c in zip(residues, counts) if r not in GAP_CHARS}
        for residue in fraction:
            color = None
            for name, applies_to, conditions in CLUSTALX_RULES:
                if residue in applies_to and any(_rule_met(fraction, t, g, comb) for t, g, comb in conditions):
                    color = CLUSTALX_COLOURS[name]
                    break
            if color:
                index[column == residue, j] = pos[color]
    return Coloring(index, colors, CLUSTALX_LEGEND)


def percent_identity_colors(letters):
    n, L = letters.shape
    colors = [WHITE] + [c for _, c, _ in IDENTITY_SHADES]
    index = np.zeros((n, L), dtype=np.uint8)
    for j in range(L):
        column = letters[:, j]
        residues = [r for r in column if r not in GAP_CHARS]
        if not residues:
            continue
        values, counts = np.unique(residues, return_counts=True)
        top = values[np.argmax(counts)]
        frac = counts.max() / n
        shade = next((k + 1 for k, (t, _, _) in enumerate(IDENTITY_SHADES) if frac > t), 0)
        index[column == top, j] = shade
    return Coloring(index, colors, [(label, c) for _, c, label in IDENTITY_SHADES])


def conservation_colors(letters, seq_type):
    from modules.conservation import column_conservation  # local: keeps viz importable without the MSA stack

    scores = column_conservation(["".join(r) for r in letters], seq_type)
    colors = [WHITE] + list(CONSERVATION_RAMP)
    bins = np.where(np.isnan(scores), -1, np.minimum((np.nan_to_num(scores) * 10).astype(int), 9))
    index = np.zeros(letters.shape, dtype=np.uint8)
    residue = ~np.isin(letters, list(GAP_CHARS))
    for j, b in enumerate(bins):
        if b >= 0:
            index[residue[:, j], j] = b + 1
    legend = [("Low conservation", CONSERVATION_RAMP[0]), ("High conservation", CONSERVATION_RAMP[-1])]
    return Coloring(index, colors, legend)


def similarity_colors(letters, reference_row, seq_type):
    from modules.pairwise import load_matrix

    colors = [WHITE] + [c for _, c in SIMILARITY]
    codes = np.frombuffer("".join("".join(r) for r in letters).encode("ascii", "replace"),
                          dtype=np.uint8).reshape(letters.shape)
    ref = codes[reference_row][None, :]
    gap = np.isin(codes, [ord(c) for c in GAP_CHARS]) | np.isin(ref, [ord(c) for c in GAP_CHARS])
    similar = np.zeros((256, 256), dtype=bool)
    if seq_type == "Protein":
        blosum = load_matrix("BLOSUM62")
        for a in blosum.alphabet:
            for b in blosum.alphabet:
                if a != "*" and b != "*" and blosum[a][b] > 0:
                    similar[ord(a), ord(b)] = True
    index = np.where(codes == ref, 1, np.where(similar[codes, np.broadcast_to(ref, codes.shape)], 2, 3))
    index[gap] = 0
    return Coloring(index.astype(np.uint8), colors, list(SIMILARITY))


def hydrophobicity_colors(letters):
    lookup = {r: HYDRO_RAMP[min(8, int((v + 4.5) / 9.0 * 9))] for r, v in KYTE_DOOLITTLE.items()}
    return _from_lookup(letters, lookup, [("Hydrophilic", HYDRO_RAMP[0]), ("Hydrophobic", HYDRO_RAMP[-1])])


def color_alignment(seqs, scheme, seq_type="Protein", reference=0):
    """Coloring for aligned sequences under a scheme (see SCHEMES); `reference` is a row index."""
    letters = _letters(seqs)
    if letters.size == 0:
        return Coloring(np.zeros((0, 0), dtype=np.uint8), [WHITE], [])
    if scheme == "Residue":
        return _from_lookup(letters, RESIDUE, CLUSTALX_LEGEND[:-1])
    if scheme == "Nucleotide":
        return _from_lookup(letters, NUCLEOTIDE, [("A", NUCLEOTIDE["A"]), ("C", NUCLEOTIDE["C"]),
                                                  ("G", NUCLEOTIDE["G"]), ("T/U", NUCLEOTIDE["T"])])
    if scheme == "ClustalX":
        return clustalx_colors(letters)
    if scheme == "Zappo":
        return _from_lookup(letters, ZAPPO, ZAPPO_LEGEND)
    if scheme == "Taylor":
        return _from_lookup(letters, TAYLOR, [])
    if scheme == "Hydrophobicity":
        return hydrophobicity_colors(letters)
    if scheme == "Percent identity":
        return percent_identity_colors(letters)
    if scheme == "Conservation":
        return conservation_colors(letters, seq_type)
    if scheme == "Similarity to reference":
        return similarity_colors(letters, reference, seq_type)
    raise ValueError(f"Unknown colour scheme {scheme!r}")
