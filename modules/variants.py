"""Substitutions and indels between aligned sequences, labelled in HGVS-like protein notation."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Variant:
    kind: str    # "sub" | "del" | "ins" | "ext"
    start: int   # reference position (1-based); for "ins", the residue before the insertion
    end: int
    ref: str     # reference residues; for "ext", "N" or "C" (the terminus extended)
    alt: str

    @property
    def label(self):
        if self.kind == "sub":
            return f"{self.ref}{self.start}{self.alt}"
        if self.kind == "del":
            if self.start == self.end:
                return f"{self.ref}{self.start}del"
            return f"{self.ref[0]}{self.start}_{self.ref[-1]}{self.end}del"
        if self.kind == "ins":
            return f"{self.ref[0]}{self.start}_{self.ref[1]}{self.end}ins{self.alt}"
        return f"{self.ref}-terminal +{self.alt}"


def call_variants(ref_aln, qry_aln, ref_start=0, ref_rotation=0, ref_length=None):
    """Substitutions, deletions, insertions and terminal extensions of the query vs the reference.

    Positions are reference coordinates: `ref_start` counts reference residues before the first
    column (local alignments) and `ref_rotation` undoes a circular shift.
    """
    def original(pos):
        if ref_rotation and ref_length and pos > 0:
            return ((pos - 1 + ref_rotation) % ref_length) + 1
        return pos

    ref_aln, qry_aln = ref_aln.upper(), qry_aln.upper()
    ref_residues = ref_aln.replace("-", "")
    variants, pos, i, n = [], ref_start, 0, len(ref_aln)
    while i < n:
        a, b = ref_aln[i], qry_aln[i]
        if a != "-" and b != "-":
            pos += 1
            if a != b:
                variants.append(Variant("sub", original(pos), original(pos), a, b))
            i += 1
        elif a != "-":
            first, deleted = pos + 1, ""
            while i < n and ref_aln[i] != "-" and qry_aln[i] == "-":
                pos += 1
                deleted += ref_aln[i]
                i += 1
            variants.append(Variant("del", original(first), original(pos), deleted, ""))
        elif b != "-":
            inserted = ""
            while i < n and ref_aln[i] == "-":
                if qry_aln[i] != "-":
                    inserted += qry_aln[i]
                i += 1
            consumed = pos - ref_start
            if 0 < consumed < len(ref_residues):
                variants.append(Variant("ins", original(pos), original(pos + 1),
                                        ref_residues[consumed - 1] + ref_residues[consumed], inserted))
            else:
                variants.append(Variant("ext", original(pos), original(pos), "N" if consumed == 0 else "C", inserted))
        else:
            i += 1
    return variants


def variants_long(aligned_pairs, reference_id):
    ref = dict(aligned_pairs)[reference_id]
    rows = [{"Sequence ID": seq_id, "Kind": v.kind, "Start": v.start, "End": v.end, "Ref": v.ref,
             "Alt": v.alt, "Label": v.label}
            for seq_id, seq in aligned_pairs if seq_id != reference_id for v in call_variants(ref, seq)]
    return pd.DataFrame(rows, columns=["Sequence ID", "Kind", "Start", "End", "Ref", "Alt", "Label"])


def variant_matrix(aligned_pairs, reference_id):
    """Sequences × reference positions; each cell lists the variant labels starting there."""
    ref = dict(aligned_pairs)[reference_id]
    rows = {}
    for seq_id, seq in aligned_pairs:
        if seq_id == reference_id:
            continue
        cells = {}
        for v in call_variants(ref, seq):
            cells[v.start] = "; ".join(filter(None, [cells.get(v.start), v.label]))
        rows[seq_id] = cells
    positions = sorted({p for cells in rows.values() for p in cells})
    matrix = pd.DataFrame.from_dict(rows, orient="index").reindex(index=list(rows), columns=positions).fillna("")
    matrix.index.name = "Sequence ID"
    return matrix
