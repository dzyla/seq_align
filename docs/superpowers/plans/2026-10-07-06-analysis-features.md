# Plan 06 — Analysis Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the remaining approved features:

- variant tables with indels
- sequence logos and identity-matrix export
- MSA trimming, redundancy filtering and MAFFT
- bootstrap-supported interactive trees
- protein properties (ProtParam)
- DNA tools (reverse complement, codon usage, restriction map)
- one-click session export

**Architecture:** Each feature is a pure module or set of functions with unit tests, plus a small UI hook in the existing section or a new analysis page registered in `modules/navigation.py`. Coordinates shown to users are always reference-sequence coordinates computed on the untrimmed alignment.

**Tech Stack:** Biopython (`Bio.SeqUtils.ProtParam`, `Bio.Restriction`, `Bio.Phylo`), NumPy, logomaker 0.8.7, MAFFT (optional system binary), Plotly, Streamlit.

**Spec:** `docs/superpowers/specs/2026-10-07-codebase-review.md` (items F4–F9)

**Prerequisite:** Plan 04 merged to `main` (Plan 05 is independent of this plan).

## Global Constraints

- All earlier constraints apply.
- Add `logomaker==0.8.7` to `requirements.txt`. Add `mafft` to `packages.txt` and to the CI `apt-get install` line.
- Import cycles: `clustering → msa → conservation`. `modules/conservation.py` and `modules/msa.py` must import from `modules.clustering` only inside functions.
- MAFFT is optional at runtime. Missing MAFFT disables the option with an explanation; it never errors.
- Trimmed alignments are used for display, export and trees only. Mutation, variant and conservation coordinates come from the untrimmed alignment.
- Branch: `feat/06-analysis`. Commit after every task with the `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer.

## Review Focus

1. **Insertions at the alignment ends** (terminal extensions). Expected: labelled `N-terminal +AA` / `C-terminal +AA`, never as an insertion between non-existent residues. Covered in Task 1 (`test_terminal_extensions`).
2. **MAFFT not installed.** Expected: `run_msa(method="MAFFT")` raises a clear `MSAError` and the UI falls back to FAMSA. Covered in Task 3 (`test_mafft_missing_raises`).
3. **Trimming would shift reference coordinates.** Expected: coordinate-bearing outputs use untrimmed rows. Covered in Task 3 (`test_alignment_views_keep_coordinates`).
4. **ProtParam on non-standard residues** (X, U, B, Z, `*`). Expected: removed, counted and reported; no crash. Covered in Task 5 (`test_properties_nonstandard`).
5. **Restriction sites spanning the origin of a circular plasmid.** Expected: found only when "circular" is checked. Covered in Task 6 (`test_restriction_circular_origin`).

---

### Task 1: Variant calling with indels and a variant table

**Files:**
- Create: `modules/variants.py`
- Modify: `modules/msa.py` ("Variant Table" tab), `modules/pairwise.py` (indels list)
- Test: `tests/test_variants.py`

**Interfaces:**
- Consumes: `PairwiseResult`, aligned pairs from `msa_aligned`
- Produces:
  - `Variant(kind, start, end, ref, alt)`, frozen, with property `label`. Kinds: `sub`, `del`, `ins`, `ext`.
  - `call_variants(ref_aln, qry_aln, ref_start=0, ref_rotation=0, ref_length=None) -> list[Variant]`
  - `variant_matrix(aligned_pairs, reference_id) -> DataFrame` (index `Sequence ID`, columns = reference positions)
  - `variants_long(aligned_pairs, reference_id) -> DataFrame` (columns `Sequence ID, Kind, Start, End, Ref, Alt, Label`)

- [ ] **Step 1: Create the branch**

```bash
git checkout main && git pull --ff-only && git checkout -b feat/06-analysis
```

- [ ] **Step 2: Write the failing tests**

`tests/test_variants.py`:

```python
from modules.variants import call_variants, variant_matrix, variants_long


def labels(ref, qry, **kw):
    return [v.label for v in call_variants(ref, qry, **kw)]


def test_substitution_insertion_deletion():
    assert labels("MKV-LAAG", "MRVWL--G") == ["K2R", "V3_L4insW", "A5_A6del"]


def test_single_deletion():
    assert labels("MKVL", "MK-L") == ["V3del"]


def test_terminal_extensions():
    assert labels("--MKV", "AAMKV") == ["N-terminal +AA"]
    assert labels("MKV--", "MKVAA") == ["C-terminal +AA"]


def test_reference_offset_and_rotation():
    assert labels("MKV", "MRV", ref_start=10) == ["K12R"]
    assert labels("CDEAK", "CDEWK", ref_rotation=2, ref_length=5) == ["A1W"]


def test_double_gap_columns_ignored():
    assert labels("MK--V", "MK--L") == ["V3L"]


def test_variant_matrix_and_long():
    pairs = [("ref", "MKV-L"), ("a", "MRVWL"), ("b", "MKV-L")]
    matrix = variant_matrix(pairs, "ref")
    assert list(matrix.columns) == [2, 3]
    assert matrix.loc["a", 2] == "K2R" and matrix.loc["a", 3] == "V3_L4insW"
    assert matrix.loc["b", 2] == ""
    long = variants_long(pairs, "ref")
    assert list(long["Label"]) == ["K2R", "V3_L4insW"]
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_variants.py -v`
Expected: FAIL with `ModuleNotFoundError: modules.variants`.

- [ ] **Step 4: Implement `modules/variants.py`**

```python
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
```

- [ ] **Step 5: Wire into the UI**

`modules/msa.py`: import `variant_matrix, variants_long`. Insert `"Variant Table"` into `tab_titles` right after `"Point Mutations"`, and add:

```python
    with tabs[tab_titles.index("Variant Table")]:
        st.subheader("🧾 Variant Table (substitutions and indels)")
        matrix = variant_matrix(aligned, ref_seq_id)
        if matrix.empty or matrix.shape[1] == 0:
            st.write("No variants relative to the reference.")
        else:
            st.caption(f"Rows: sequences. Columns: {ref_seq_id} positions. Indels use HGVS-like notation "
                       "(e.g. V3_L4insW, A5_A6del).")
            st.dataframe(matrix, width="stretch")
            st.download_button("📥 Variant matrix (CSV)", matrix.to_csv(), "variant_matrix.csv", "text/csv",
                               on_click="ignore")
            st.download_button("📥 Variants, one per row (CSV)", variants_long(aligned, ref_seq_id).to_csv(index=False),
                               "variants_long.csv", "text/csv", on_click="ignore")
```

Insert the tab before any code that indexes tabs by position, and switch remaining positional indexing (`tabs[1]`, `tabs[2]`, `tabs[-1]`) to `tabs[tab_titles.index("...")]`.

`modules/pairwise.py`: import `call_variants` and add after the substitution list:

```python
    indels = [v.label for v in call_variants(result.aligned_ref, result.aligned_query, result.ref_start,
                                             result.ref_rotation, result.ref_length) if v.kind != "sub"]
    if indels:
        st.write(f"**Indels and terminal extensions ({len(indels)}):** {', '.join(indels)}")
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add modules/variants.py modules/msa.py modules/pairwise.py tests/test_variants.py
git commit -m "feat: variant calling with indels and a sequences-by-position variant table

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Sequence logo and percent-identity matrix download

**Files:**
- Modify: `modules/conservation.py` (`logo_matrix`, `logo_figure`), `modules/msa.py`
- Modify: `requirements.txt`
- Test: `tests/test_conservation.py`

**Interfaces:**
- Consumes: `msa_identity_matrix`
- Produces:
  - `logo_matrix(aligned, seq_type="Protein", start=1, end=None) -> DataFrame` (information content in bits; index = alignment columns, 1-based)
  - `logo_figure(matrix, seq_type) -> matplotlib Figure`
  - `identity_matrix_frame(aligned_pairs) -> DataFrame` (percent identity, IDs as index and columns)

- [ ] **Step 1: Add the dependency**

Append `logomaker==0.8.7` to `requirements.txt`, then run `pip install -r requirements-dev.txt`.

- [ ] **Step 2: Write the failing tests** (append to `tests/test_conservation.py`)

```python
from modules.conservation import identity_matrix_frame, logo_figure, logo_matrix


def test_logo_matrix_window_and_argmax():
    m = logo_matrix(["MKV-", "MKL-", "MRLA"])
    assert m.shape[0] == 4
    assert m.idxmax(axis=1).tolist() == ["M", "K", "L", "A"]
    window = logo_matrix(["MKV-", "MKL-", "MRLA"], start=2, end=3)
    assert window.index.tolist() == [2, 3]


def test_logo_figure_closes_cleanly():
    import matplotlib.pyplot as plt
    fig = logo_figure(logo_matrix(["ACGT", "ACGA"], seq_type="DNA"), "DNA")
    assert fig.axes
    plt.close(fig)


def test_identity_matrix_frame():
    df = identity_matrix_frame([("a", "MKV"), ("b", "MKL")])
    assert df.loc["a", "b"] == round(200 / 3, 1)
    assert df.loc["a", "a"] == 100.0
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_conservation.py -v`
Expected: FAIL with `ImportError: logo_matrix`.

- [ ] **Step 4: Implement** (append to `modules/conservation.py`)

```python
import logomaker
import matplotlib.pyplot as plt
import pandas as pd


def logo_matrix(aligned, seq_type="Protein", start=1, end=None):
    """Information content (bits) per residue for alignment columns start..end (1-based, inclusive)."""
    window = [s[start - 1:end].upper() for s in aligned]
    ignore = ".-X" if seq_type == "Protein" else ".-N"
    matrix = logomaker.alignment_to_matrix(window, to_type="information", characters_to_ignore=ignore)
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
```

In `modules/msa.py`:

- In the "Alignment & Heatmap" tab, after the MSA download, add:

```python
        st.download_button("📥 Percent identity matrix (CSV)", identity_matrix_frame(aligned).to_csv(),
                           "percent_identity.csv", "text/csv", on_click="ignore")
```

- In the "Conservation" tab, after the line chart, add:

```python
        length = len(aligned[0][1])
        start, end = st.slider("Logo window (alignment columns)", 1, length, (1, min(length, 60)), key="logo_window")
        if end - start + 1 > 120:
            st.warning("Showing the first 120 columns of the window.")
            end = start + 119
        fig = logo_figure(logo_matrix([s for _, s in aligned], seq_type, start, end), seq_type)
        st.pyplot(fig)
        plt.close(fig)
```

Import `identity_matrix_frame, logo_figure, logo_matrix` from `modules.conservation` and `import matplotlib.pyplot as plt`.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add modules/conservation.py modules/msa.py requirements.txt tests/test_conservation.py
git commit -m "feat: sequence logo and percent-identity matrix export

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: MSA tools: MAFFT option, column trimming, redundancy filter

**Files:**
- Modify: `modules/msa.py`
- Modify: `packages.txt`, `.github/workflows/tests.yml`
- Test: `tests/test_msa.py`

**Interfaces:**
- Consumes: `clean_sequence`, `msa_identity_matrix`
- Produces:
  - `MSA_METHODS = ("FAMSA", "MAFFT")`, `mafft_available() -> bool`
  - `run_mafft(pairs, seq_type) -> list[tuple[str, str]]`
  - `run_msa(pairs, seq_type, method="FAMSA")`; `cached_run_msa(pairs, seq_type, method="FAMSA")`
  - `trim_columns(aligned_pairs, max_gap_fraction) -> tuple[list, list[int]]`
  - `remove_redundant(aligned_pairs, identity_threshold, keep_ids=()) -> tuple[list, dict[str, str]]`
  - `alignment_views(aligned_pairs, reference_id, max_gap_fraction=1.0, max_identity=1.0) -> tuple[rows, view, removed]`, where `rows` keeps coordinates (rows filtered, no columns trimmed) and `view` is for display and export

- [ ] **Step 1: Write the failing tests** (append to `tests/test_msa.py`)

```python
import shutil

from modules.msa import alignment_views, remove_redundant, run_msa, trim_columns


def test_trim_columns():
    trimmed, kept = trim_columns([("a", "M-KV"), ("b", "M-KL"), ("c", "MAK-")], 0.5)
    assert kept == [0, 2, 3]
    assert dict(trimmed)["a"] == "MKV"


def test_remove_redundant_keeps_reference():
    pairs = [("b", "MKVLA"), ("a", "MKVLA"), ("c", "WWWWW")]
    kept, removed = remove_redundant(pairs, 0.95, keep_ids=("a",))
    assert [i for i, _ in kept] == ["a", "c"]
    assert removed == {"b": "a"}


def test_alignment_views_keep_coordinates():
    pairs = [("ref", "M-KV"), ("x", "MAKV"), ("y", "MAKV")]
    rows, view, removed = alignment_views(pairs, "ref", max_gap_fraction=0.3, max_identity=0.99)
    assert dict(rows)["ref"] == "M-KV"            # untrimmed
    assert [i for i, _ in rows] == ["ref", "x"]   # y identical to x -> removed
    assert dict(view)["ref"] == "MKV"             # trimmed for display
    assert removed == {"y": "x"}


def test_mafft_missing_raises(monkeypatch):
    import modules.msa as msa
    monkeypatch.setattr(msa.shutil, "which", lambda name: None)
    with pytest.raises(MSAError, match="MAFFT"):
        run_msa([("a", "MKV"), ("b", "MKL")], "Protein", method="MAFFT")


@pytest.mark.skipif(shutil.which("mafft") is None, reason="MAFFT not installed")
def test_mafft_alignment_preserves_ids_and_residues():
    aligned = run_msa([("a", "mkvlaagivg"), ("b", "MKVLAGIVG")], "Protein", method="MAFFT")
    assert [i for i, _ in aligned] == ["a", "b"]
    assert aligned[0][1].replace("-", "") == "MKVLAAGIVG"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_msa.py -v`
Expected: FAIL with `ImportError: alignment_views`.

- [ ] **Step 3: Implement**

In `modules/msa.py`, add `import os`, `import shutil`, `import subprocess` and `import tempfile`. Do **not** import `modules.clustering` at module level: clustering imports msa, so `remove_redundant` imports `msa_identity_matrix` inside the function.

```python
MSA_METHODS = ("FAMSA", "MAFFT")


def mafft_available():
    return shutil.which("mafft") is not None


def run_mafft(pairs, seq_type):
    """Align with MAFFT --auto; input order, IDs and residue case are preserved."""
    if not mafft_available():
        raise MSAError("MAFFT is not installed on this server; use FAMSA.")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "input.fasta")
        with open(path, "w") as handle:
            handle.write("".join(f">{seq_id}\n{seq}\n" for seq_id, seq in pairs))
        try:
            proc = subprocess.run(
                ["mafft", "--auto", "--quiet", "--preservecase", "--inputorder", "--anysymbol", path],
                capture_output=True, text=True, timeout=600, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise MSAError(f"MAFFT failed: {e}") from e
    aligned = [(r.id, str(r.seq)) for r in SeqIO.parse(StringIO(proc.stdout), "fasta")]
    if [i for i, _ in aligned] != [i for i, _ in pairs]:
        raise MSAError("MAFFT output did not match the input sequences.")
    return aligned
```

Change `run_msa` to take a method. Keep the existing cleaning and validation, and dispatch at the end:

```python
def run_msa(pairs, seq_type, method="FAMSA"):
    """Align (id, sequence) pairs with FAMSA (default) or MAFFT and return [(id, aligned_sequence)]."""
    cleaned = [(seq_id, clean_sequence(seq, seq_type).replace("*", "")) for seq_id, seq in pairs]
    empty = [seq_id for seq_id, seq in cleaned if not seq]
    if empty:
        raise MSAError(f"Sequences empty after cleaning: {', '.join(empty)}")
    if len(cleaned) < 2:
        raise MSAError("At least two sequences are required for an MSA.")
    if method == "MAFFT":
        return run_mafft(cleaned, seq_type)
    famsa_input = [PyFAMSASequence(seq_id.encode(), seq.encode()) for seq_id, seq in cleaned]
    try:
        msa = PyFAMSAAligner(guide_tree="upgma").align(famsa_input)
    except Exception as e:
        raise MSAError(f"FAMSA failed: {e}") from e
    return [(s.id.decode(), s.sequence.decode()) for s in msa]


@st.cache_data(max_entries=8, show_spinner=False)
def cached_run_msa(pairs, seq_type, method="FAMSA"):
    return run_msa(list(pairs), seq_type, method)


def trim_columns(aligned_pairs, max_gap_fraction):
    """Drop columns whose gap fraction exceeds max_gap_fraction; returns (pairs, kept column indices)."""
    seqs = [s for _, s in aligned_pairs]
    n = len(seqs)
    kept = [c for c in range(len(seqs[0])) if sum(s[c] in "-." for s in seqs) / n <= max_gap_fraction]
    return [(i, "".join(s[c] for c in kept)) for i, s in aligned_pairs], kept


def remove_redundant(aligned_pairs, identity_threshold, keep_ids=()):
    """Greedy filter: keep `keep_ids` first, then longest sequences; drop any sequence whose identity
    to an already kept sequence is >= threshold. Returns (kept pairs, {removed_id: representative_id})."""
    from modules.clustering import msa_identity_matrix  # local: clustering imports msa

    ids = [i for i, _ in aligned_pairs]
    identity = msa_identity_matrix([s for _, s in aligned_pairs])
    ungapped = {i: len(s.replace("-", "")) for i, s in aligned_pairs}
    order = [i for i in keep_ids if i in ids] + sorted((i for i in ids if i not in keep_ids),
                                                       key=lambda i: -ungapped[i])
    index = {i: k for k, i in enumerate(ids)}
    kept, removed = [], {}
    for seq_id in order:
        rep = next((k for k in kept if identity[index[seq_id], index[k]] >= identity_threshold), None)
        if rep is None:
            kept.append(seq_id)
        else:
            removed[seq_id] = rep
    pairs_by_id = dict(aligned_pairs)
    return [(k, pairs_by_id[k]) for k in kept], removed


def alignment_views(aligned_pairs, reference_id, max_gap_fraction=1.0, max_identity=1.0):
    """(rows for coordinate-bearing analyses, trimmed view for display/export, removed map)."""
    rows, removed = aligned_pairs, {}
    if max_identity < 1.0:
        rows, removed = remove_redundant(aligned_pairs, max_identity, keep_ids=(reference_id,))
    view = trim_columns(rows, max_gap_fraction)[0] if max_gap_fraction < 1.0 else rows
    return rows, view, removed
```

`remove_redundant` returns kept pairs **in keep order** (reference first, then longest first).

UI in `msa_section`:

1. Before "Run MSA", add the aligner choice and include it in `current_params`:

```python
    method_options = [m for m in MSA_METHODS if m == "FAMSA" or mafft_available()]
    msa_method = st.selectbox("Aligner", method_options, key="msa_method",
                              index=method_options.index("MAFFT") if seq_type == "DNA" and "MAFFT" in method_options else 0,
                              help="FAMSA is fast for large protein sets. MAFFT (--auto) is recommended for DNA."
                                   + ("" if mafft_available() else " MAFFT is not installed on this server."))
```

`current_params = {"fingerprint": ..., "seq_type": seq_type, "method": msa_method}`, and call `cached_run_msa(tuple(...), seq_type, msa_method)`.

2. Right after `aligned` is confirmed fresh, add the tools and derived views:

```python
    with st.expander("✂️ Alignment tools (trim columns / remove redundant sequences)"):
        max_gap = st.slider("Drop columns with gap fraction above", 0.0, 1.0, 1.0, 0.05, key="msa_trim",
                            help="1.0 keeps all columns. Trimming affects the heatmap, downloads and trees only; "
                                 "mutation, variant and conservation positions always use the full alignment.")
        max_id = st.slider("Drop sequences with identity at or above", 0.5, 1.0, 1.0, 0.01, key="msa_redundancy",
                           help="1.0 keeps all sequences. The reference sequence is always kept.")
    rows, view, removed = alignment_views(aligned, ref_seq_id, max_gap, max_id)
    if removed:
        st.caption(f"{len(removed)} redundant sequence(s) hidden: "
                   + ", ".join(f"{r} (≈ {k})" for r, k in list(removed.items())[:20]))
```

3. Use `rows` for `cached_mutations`, the variant table, conservation and the representative sequence. Use `view` for the heatmap, the MSA download, the identity matrix, the logo and the tree.

`packages.txt`:

```
hmmer
mafft
```

In `.github/workflows/tests.yml`, change the install line to `sudo apt-get install -y hmmer mafft`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add modules/msa.py packages.txt .github/workflows/tests.yml tests/test_msa.py
git commit -m "feat: MAFFT option, gap-column trimming and redundancy filtering for MSAs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Bootstrap support and an interactive tree

**Files:**
- Modify: `modules/phylogeny.py` (vectorized `alignment_distance_matrix`, `tree_splits`, `bootstrap_tree`, `tree_layout`, `tree_figure`, view toggle)
- Modify: `modules/msa.py` (bootstrap replicates input)
- Test: `tests/test_phylogeny.py`

**Interfaces:**
- Consumes: `_tree_from_matrix`, `correct_distance`
- Produces:
  - `alignment_distance_matrix(aligned, seq_type, correction=True)`: same contract, now NumPy-vectorized
  - `tree_splits(tree, all_names, anchor) -> set[frozenset]` (non-trivial bipartitions, rooting-invariant)
  - `bootstrap_tree(aligned_pairs, seq_type, method="nj", correction=True, replicates=100, seed=0) -> Tree` (`clade.confidence` = % support)
  - `tree_layout(tree) -> (segments, leaves, internals)`
  - `tree_figure(tree) -> go.Figure`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_phylogeny.py`)

```python
from modules.phylogeny import bootstrap_tree, tree_figure, tree_layout, tree_splits

GROUPED = [("a", "A" * 19 + "C"), ("b", "A" * 20), ("c", "T" * 19 + "G"), ("d", "T" * 20)]


def test_vectorized_distance_matches_definition():
    d = alignment_distance_matrix(["AC-T", "AGGT", "TC-T"], "DNA", correction=False)
    assert d[0, 1] == pytest.approx(1 / 3)
    assert d[0, 2] == pytest.approx(1 / 3)


def test_bootstrap_supports_true_split():
    tree = bootstrap_tree(GROUPED, "DNA", method="nj", correction=False, replicates=50, seed=1)
    supports = [c.confidence for c in tree.get_nonterminals() if c.confidence is not None]
    assert 100 in supports


def test_tree_splits_rooting_invariant():
    from io import StringIO
    from Bio import Phylo
    t1 = Phylo.read(StringIO("((a,b),(c,d));"), "newick")
    t2 = Phylo.read(StringIO("(a,(b,(c,d)));"), "newick")
    names = frozenset("abcd")
    assert tree_splits(t1, names, "a") == tree_splits(t2, names, "a") == {frozenset("cd")}


def test_tree_figure_has_all_leaves():
    from io import StringIO
    from Bio import Phylo
    tree = Phylo.read(StringIO("((A:0.1,B:0.2):0.3,C:0.5);"), "newick")
    segments, leaves, _ = tree_layout(tree)
    assert sorted(name for _, _, name in leaves) == ["A", "B", "C"]
    fig = tree_figure(tree)
    assert sorted(fig.data[1].text) == ["A", "B", "C"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_phylogeny.py -v`
Expected: FAIL with `ImportError: bootstrap_tree`.

- [ ] **Step 3: Implement**

In `modules/phylogeny.py`, add `from collections import Counter` and `import plotly.graph_objects as go`. Replace `alignment_distance_matrix`:

```python
def alignment_distance_matrix(aligned, seq_type, correction=True):
    """Symmetric distance matrix from aligned strings (pairwise deletion of gap columns), vectorized."""
    n = len(aligned)
    matrix = np.zeros((n, n))
    if n < 2:
        return matrix
    arr = np.frombuffer("".join(s.upper() for s in aligned).encode(), dtype=np.uint8).reshape(n, -1)
    residue = arr != ord('-')
    for i in range(n - 1):
        valid = residue[i] & residue[i + 1:]
        diffs = ((arr[i] != arr[i + 1:]) & valid).sum(axis=1)
        counts = valid.sum(axis=1)
        for k, (d, c) in enumerate(zip(diffs, counts)):
            value = _finish(d / c if c else None, seq_type, correction)
            matrix[i, i + 1 + k] = matrix[i + 1 + k, i] = value
    return matrix
```

Add:

```python
def _normalized_split(names, all_names, anchor):
    side = all_names - names if anchor in names else names
    if len(side) < 2 or len(side) > len(all_names) - 2:
        return None
    return frozenset(side)


def tree_splits(tree, all_names, anchor):
    """Non-trivial bipartitions, expressed as the side without `anchor` (independent of rooting)."""
    splits = set()
    for clade in tree.get_nonterminals():
        split = _normalized_split(frozenset(t.name for t in clade.get_terminals()), all_names, anchor)
        if split is not None:
            splits.add(split)
    return splits


def bootstrap_tree(aligned_pairs, seq_type, method="nj", correction=True, replicates=100, seed=0):
    """Tree from the full alignment; internal clades carry % bootstrap support (column resampling)."""
    names = [i for i, _ in aligned_pairs]
    seqs = [s.upper() for _, s in aligned_pairs]
    target = _tree_from_matrix(names, alignment_distance_matrix(seqs, seq_type, correction), method)
    arr = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), -1)
    rng = np.random.default_rng(seed)
    all_names, anchor = frozenset(names), names[0]
    counts = Counter()
    for _ in range(replicates):
        cols = rng.integers(0, arr.shape[1], arr.shape[1])
        replicate = [row.tobytes().decode() for row in arr[:, cols]]
        tree = _tree_from_matrix(names, alignment_distance_matrix(replicate, seq_type, correction), method)
        counts.update(tree_splits(tree, all_names, anchor))
    for clade in target.get_nonterminals():
        split = _normalized_split(frozenset(t.name for t in clade.get_terminals()), all_names, anchor)
        clade.confidence = round(100 * counts[split] / replicates) if split is not None else None
    return target


def tree_layout(tree):
    """Rectangular layout: (segments [(x0,y0,x1,y1)], leaves [(x,y,name)], internals [(x,y,confidence)])."""
    has_lengths = any(c.branch_length for c in tree.find_clades())
    depths = tree.depths(unit_branch_lengths=not has_lengths)
    terminals = tree.get_terminals()
    y = {t: i for i, t in enumerate(terminals)}

    def assign(clade):
        if clade not in y:
            ys = [assign(c) for c in clade.clades]
            y[clade] = (min(ys) + max(ys)) / 2
        return y[clade]

    assign(tree.root)
    segments, internals = [], []
    for clade in tree.find_clades(order="preorder"):
        for child in clade.clades:
            segments.append((depths[clade], y[child], depths[child], y[child]))
        if clade.clades:
            ys = [y[c] for c in clade.clades]
            segments.append((depths[clade], min(ys), depths[clade], max(ys)))
            internals.append((depths[clade], y[clade], clade.confidence))
    leaves = [(depths[t], y[t], t.name) for t in terminals]
    return segments, leaves, internals


def tree_figure(tree):
    segments, leaves, internals = tree_layout(tree)
    xs, ys = [], []
    for x0, y0, x1, y1 in segments:
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#555555", width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[l[0] for l in leaves], y=[l[1] for l in leaves], mode="markers+text",
                             text=[l[2] for l in leaves], textposition="middle right",
                             marker=dict(size=6, color="#4e79a7"), hovertemplate="%{text}<extra></extra>"))
    labelled = [(x, yy, c) for x, yy, c in internals if c is not None]
    if labelled:
        fig.add_trace(go.Scatter(x=[p[0] for p in labelled], y=[p[1] for p in labelled], mode="text",
                                 text=[str(int(p[2])) for p in labelled], textposition="top left",
                                 textfont=dict(size=10, color="#d62728"), hoverinfo="skip"))
    max_x = max([l[0] for l in leaves] + [0.0])
    fig.update_layout(showlegend=False, height=max(300, 22 * len(leaves) + 80), plot_bgcolor="white",
                      xaxis=dict(title="Branch length", range=[0, max_x * 1.35 or 1]),
                      yaxis=dict(visible=False, autorange="reversed"), margin=dict(l=20, r=20, t=30, b=40))
    return fig
```

In `phylogenetic_tree_section`, right after `st.subheader("📊 Tree Visualization")`, add a view toggle. Interactive is the default; the existing Matplotlib code runs only for the static view (pass `show_confidence=True` to `Phylo.draw`):

```python
    view = st.radio("View", ("Interactive", "Static (Matplotlib)"), horizontal=True, key="tree_view")
    if view == "Interactive":
        st.plotly_chart(tree_figure(tree), width="stretch")
    else:
        ...  # existing settings expander + Matplotlib drawing block, indented under this else
```

In `modules/msa.py`'s tree tab, add `replicates = st.number_input("Bootstrap replicates", 0, 1000, 0, 50, key="msa_bootstrap", help="0 = no bootstrap. 100 is a common choice.")`. When `replicates > 0`, build with `bootstrap_tree(view, seq_type, TREE_METHODS[tree_method], correction, replicates)`; otherwise use the existing `build_tree_from_alignment(aligned_to_msa(view), ...)`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add modules/phylogeny.py modules/msa.py tests/test_phylogeny.py
git commit -m "feat: bootstrap-supported trees and an interactive Plotly tree view

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Protein properties page

**Files:**
- Create: `modules/properties.py`
- Modify: `modules/navigation.py`, `app.py`
- Test: `tests/test_properties.py`, `tests/test_navigation.py`

**Interfaces:**
- Consumes: `Bio.SeqUtils.ProtParam.ProteinAnalysis`
- Produces:
  - `STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")`
  - `protein_properties(seq: str) -> dict` with keys `Length, MW (Da), pI, Ext. coeff. (reduced), Ext. coeff. (cystines), A280 0.1% (reduced), A280 0.1% (cystines), GRAVY, Instability index, Aromaticity, Non-standard removed`
  - `properties_table(records) -> DataFrame` (first column `Sequence ID`)
  - `sequence_properties_section(sequences)`

- [ ] **Step 1: Write the failing tests**

`tests/test_properties.py`:

```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.properties import properties_table, protein_properties


def test_protein_properties_known_values():
    p = protein_properties("MKWVTFISLLLLFSSAYS")
    assert p["Length"] == 18
    assert round(p["MW (Da)"], 1) == 2106.5
    assert round(p["pI"], 2) == 8.34
    assert p["Ext. coeff. (reduced)"] == 6990
    assert round(p["GRAVY"], 3) == 1.289
    assert round(p["A280 0.1% (reduced)"], 3) == round(6990 / 2106.5, 3)


def test_properties_nonstandard():
    p = protein_properties("MKX*UBZW")
    assert p["Non-standard removed"] == 5
    assert p["Length"] == 3


def test_properties_table():
    df = properties_table([SeqRecord(Seq("MKW"), id="a"), SeqRecord(Seq("XX"), id="b")])
    assert list(df["Sequence ID"]) == ["a", "b"]
    assert df.loc[1, "Length"] == 0
```

In `tests/test_navigation.py`, replace `test_protein_sequences` with:

```python
def test_protein_sequences():
    assert available_analyses(True, False, "Protein") == [
        "Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Phylogenetic Tree",
        "Antibody Prediction", "Sequence Properties", "Convert Formats",
    ]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_properties.py tests/test_navigation.py -v`
Expected: FAIL with `ModuleNotFoundError: modules.properties`.

- [ ] **Step 3: Implement**

`modules/properties.py`:

```python
"""Protein physico-chemical properties (Biopython ProtParam)."""
import pandas as pd
import streamlit as st
from Bio.SeqUtils.ProtParam import ProteinAnalysis

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
COLUMNS = ["Length", "MW (Da)", "pI", "Ext. coeff. (reduced)", "Ext. coeff. (cystines)",
           "A280 0.1% (reduced)", "A280 0.1% (cystines)", "GRAVY", "Instability index", "Aromaticity",
           "Non-standard removed"]


def protein_properties(seq):
    """ProtParam values on the standard residues only; non-standard residues are counted and removed."""
    text = str(seq).upper()
    clean = "".join(c for c in text if c in STANDARD_AA)
    removed = sum(1 for c in text if c.isalpha() or c == "*") - len(clean)
    if not clean:
        return {**dict.fromkeys(COLUMNS), "Length": 0, "Non-standard removed": removed}
    analysis = ProteinAnalysis(clean)
    mw = analysis.molecular_weight()
    reduced, cystines = analysis.molar_extinction_coefficient()
    return {
        "Length": len(clean),
        "MW (Da)": mw,
        "pI": analysis.isoelectric_point(),
        "Ext. coeff. (reduced)": reduced,
        "Ext. coeff. (cystines)": cystines,
        "A280 0.1% (reduced)": reduced / mw,
        "A280 0.1% (cystines)": cystines / mw,
        "GRAVY": analysis.gravy(),
        "Instability index": analysis.instability_index(),
        "Aromaticity": analysis.aromaticity(),
        "Non-standard removed": removed,
    }


def properties_table(records):
    return pd.DataFrame([{"Sequence ID": r.id, **protein_properties(r.seq)} for r in records])


def sequence_properties_section(sequences):
    st.header("⚗️ Sequence Properties")
    st.info("Molecular weight, pI, extinction coefficients (M⁻¹cm⁻¹ at 280 nm), A280 of a 1 mg/mL solution, "
            "GRAVY, instability index and aromaticity (Biopython ProtParam). Non-standard residues are excluded.")
    df = properties_table(sequences)
    st.dataframe(df.round(3), hide_index=True, width="stretch")
    st.download_button("📥 Properties (CSV)", df.to_csv(index=False), "sequence_properties.csv", "text/csv",
                       on_click="ignore")
    if (df["Non-standard removed"] > 0).any():
        st.caption("Some sequences contained non-standard residues (X, U, B, Z, *), which were excluded.")
```

`modules/navigation.py`: in the protein branch, use `options += ["Antibody Prediction", "Sequence Properties"]` instead of appending only "Antibody Prediction", and keep `Translate DNA` for DNA. The final version is shown in Task 6.

`app.py`: import `sequence_properties_section` and dispatch `elif alignment_mode == "Sequence Properties": sequence_properties_section(sequences)`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add modules/properties.py modules/navigation.py app.py tests/test_properties.py tests/test_navigation.py
git commit -m "feat: protein properties page (MW, pI, extinction coefficients, A280, GRAVY)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: DNA tools: reverse complement, codon usage, restriction map

**Files:**
- Modify: `modules/dna.py` (`reverse_complement`, `codon_usage`, `POPULAR_ENZYMES`, `restriction_sites`, `restriction_map_figure`, `dna_tools_section`)
- Modify: `modules/navigation.py`, `app.py`
- Test: `tests/test_dna_tools.py`, `tests/test_navigation.py`

**Interfaces:**
- Consumes: `clean_sequence`, `CODON_TABLES`, `Bio.Restriction`
- Produces:
  - `reverse_complement(seq) -> str`
  - `codon_usage(seq, frame=1, table=1) -> DataFrame` (columns `Codon, AA, Count, Fraction of AA`)
  - `POPULAR_ENZYMES: list[str]`
  - `restriction_sites(seq, circular=False, enzyme_set="Popular") -> DataFrame` (columns `Enzyme, Site, Cuts, Positions`; positions are 1-based nucleotides before which the top strand is cut)
  - `restriction_map_figure(sites_df, seq_length, enzymes) -> go.Figure`
  - `dna_tools_section(sequences)`
  - Final `available_analyses` as listed below

- [ ] **Step 1: Write the failing tests**

`tests/test_dna_tools.py`:

```python
from modules.dna import codon_usage, restriction_map_figure, restriction_sites, reverse_complement


def test_reverse_complement_rna_and_case():
    assert reverse_complement("augc") == "GCAT"


def test_codon_usage():
    df = codon_usage("ATGAAAAAGTAA")
    row = df[df["Codon"] == "AAA"].iloc[0]
    assert row["AA"] == "K" and row["Count"] == 1 and row["Fraction of AA"] == 0.5


def test_restriction_sites_ecori():
    df = restriction_sites("AAGAATTCAA")
    eco = df[df["Enzyme"] == "EcoRI"].iloc[0]
    assert eco["Cuts"] == 1 and eco["Positions"] == [4]


def test_restriction_circular_origin():
    seq = "AATTCAAAAAAAAAG"      # GAATTC spans the origin when circular
    assert "EcoRI" not in set(restriction_sites(seq, circular=False)["Enzyme"])
    assert "EcoRI" in set(restriction_sites(seq, circular=True)["Enzyme"])


def test_restriction_map_figure():
    df = restriction_sites("AAGAATTCAAGGATCCAA")
    fig = restriction_map_figure(df, 18, ["EcoRI", "BamHI"])
    assert len(fig.data) == 2
```

Append to `tests/test_navigation.py`:

```python
def test_dna_tools_listed():
    options = available_analyses(True, False, "DNA")
    assert options[-3:] == ["Translate DNA", "DNA Tools", "Convert Formats"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_dna_tools.py tests/test_navigation.py -v`
Expected: FAIL with `ImportError: codon_usage`.

- [ ] **Step 3: Implement** (append to `modules/dna.py`; add `import plotly.graph_objects as go` and `from Bio.Restriction import Analysis, CommOnly, RestrictionBatch`)

```python
POPULAR_ENZYMES = ["AgeI", "ApaI", "AscI", "AvrII", "BamHI", "BglII", "BsaI", "BsmBI", "BspEI", "ClaI", "EcoRI",
                   "EcoRV", "HindIII", "KpnI", "MfeI", "MluI", "NcoI", "NdeI", "NheI", "NotI", "NsiI", "PacI",
                   "PstI", "SacI", "SalI", "SmaI", "SpeI", "SphI", "XbaI", "XhoI"]
ENZYME_SETS = {"Popular": POPULAR_ENZYMES, "All commercial": None}


def reverse_complement(seq):
    return str(Seq(clean_sequence(seq, "DNA")).reverse_complement())


def codon_usage(seq, frame=1, table=1):
    clean = clean_sequence(seq, "DNA")[frame - 1:]
    codons = [clean[i:i + 3] for i in range(0, len(clean) - len(clean) % 3, 3)]
    counts = pd.Series(codons).value_counts()
    rows = [{"Codon": codon, "AA": str(Seq(codon).translate(table=table)), "Count": int(count)}
            for codon, count in counts.items()]
    df = pd.DataFrame(rows, columns=["Codon", "AA", "Count"])
    df["Fraction of AA"] = df["Count"] / df.groupby("AA")["Count"].transform("sum")
    return df.sort_values(["AA", "Codon"]).reset_index(drop=True)


def restriction_sites(seq, circular=False, enzyme_set="Popular"):
    names = ENZYME_SETS[enzyme_set]
    batch = RestrictionBatch(CommOnly) if names is None else RestrictionBatch(names)
    hits = Analysis(batch, Seq(clean_sequence(seq, "DNA")), linear=not circular).full()
    rows = [{"Enzyme": str(enzyme), "Site": enzyme.site, "Cuts": len(positions), "Positions": sorted(positions)}
            for enzyme, positions in hits.items() if positions]
    df = pd.DataFrame(rows, columns=["Enzyme", "Site", "Cuts", "Positions"])
    return df.sort_values(["Cuts", "Enzyme"]).reset_index(drop=True)


def restriction_map_figure(sites_df, seq_length, enzymes):
    fig = go.Figure()
    chosen = sites_df[sites_df["Enzyme"].isin(enzymes)]
    for k, (_, row) in enumerate(chosen.iterrows()):
        y = (k % 6) + 1
        fig.add_trace(go.Scatter(x=row["Positions"], y=[y] * len(row["Positions"]), mode="markers+text",
                                 text=[row["Enzyme"]] * len(row["Positions"]), textposition="top center",
                                 marker=dict(symbol="line-ns-open", size=16), name=row["Enzyme"],
                                 hovertemplate=f"{row['Enzyme']} ({row['Site']}) cuts at %{{x}}<extra></extra>"))
    fig.add_shape(type="line", x0=1, x1=seq_length, y0=0, y1=0, line=dict(width=4, color="#555555"))
    fig.update_layout(height=320, showlegend=False, xaxis=dict(title="Position (nt)", range=[0, seq_length + 1]),
                      yaxis=dict(visible=False, range=[-0.5, 7.5]), margin=dict(t=30))
    return fig


def dna_tools_section(sequences):
    st.header("🧰 DNA Tools")
    selected_id = st.selectbox("Sequence", [s.id for s in sequences], key="dna_tools_seq")
    seq = str(next(s for s in sequences if s.id == selected_id).seq)
    tab_rc, tab_codon, tab_re = st.tabs(["Reverse complement", "Codon usage", "Restriction map"])

    with tab_rc:
        rc = reverse_complement(seq)
        st.code(rc)
        st.download_button("📥 Reverse complement (FASTA)", f">{selected_id}_revcomp\n{rc}\n",
                           f"{selected_id}_revcomp.fasta", "text/plain", on_click="ignore")

    with tab_codon:
        c1, c2 = st.columns(2)
        frame = c1.selectbox("Reading frame", (1, 2, 3), key="codon_frame")
        table = CODON_TABLES[c2.selectbox("Codon table", list(CODON_TABLES), key="codon_table")]
        usage = codon_usage(seq, frame, table)
        st.dataframe(usage.round(3), hide_index=True, width="stretch")
        st.download_button("📥 Codon usage (CSV)", usage.to_csv(index=False), "codon_usage.csv", "text/csv",
                           on_click="ignore")

    with tab_re:
        c1, c2 = st.columns(2)
        circular = c1.checkbox("Circular (plasmid)", key="re_circular")
        enzyme_set = c2.selectbox("Enzymes", list(ENZYME_SETS), key="re_set")
        sites = restriction_sites(seq, circular, enzyme_set)
        if sites.empty:
            st.write("No sites found.")
            return
        unique = sites[sites["Cuts"] == 1]["Enzyme"].tolist()
        shown = st.multiselect("Enzymes on map", sites["Enzyme"].tolist(), default=unique[:15], key="re_shown")
        st.plotly_chart(restriction_map_figure(sites, len(clean_sequence(seq, "DNA")), shown), width="stretch")
        st.dataframe(sites, hide_index=True, width="stretch")
        st.caption(f"{len(unique)} unique cutter(s). Positions are the first nucleotide after the top-strand cut.")
        st.download_button("📥 Restriction sites (CSV)", sites.to_csv(index=False), "restriction_sites.csv",
                           "text/csv", on_click="ignore")
```

Final `modules/navigation.py`:

```python
"""Which analyses apply to the loaded data."""


def available_analyses(has_sequences, has_structures, seq_type):
    options = []
    if has_structures:
        options.append("Extracted Sequences")
    if has_sequences:
        options += ["Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Phylogenetic Tree"]
        if seq_type == "DNA":
            options += ["Translate DNA", "DNA Tools"]
        else:
            options += ["Antibody Prediction", "Sequence Properties"]
        options.append("Convert Formats")
    if has_structures:
        options += ["Structure Viewer", "PDB Residue Renumbering"]
    return options
```

`app.py`: import `dna_tools_section` and dispatch `elif alignment_mode == "DNA Tools": dna_tools_section(sequences)`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add modules/dna.py modules/navigation.py app.py tests/test_dna_tools.py tests/test_navigation.py
git commit -m "feat: DNA tools page with reverse complement, codon usage and restriction map

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Session export (ZIP of all results and parameters)

**Files:**
- Create: `modules/export.py`
- Modify: `app.py` (sidebar download), `README.md` (one line under Analyses)
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `format_result_text`, `serialize_alignment`, `public_columns`, `tree_to_newick`, `fingerprint`
- Produces: `build_session_zip(state: Mapping, sequences, seq_type) -> bytes`, `has_results(state) -> bool`

- [ ] **Step 1: Write the failing tests**

`tests/test_export.py`:

```python
import io
import json
import zipfile
from io import StringIO

from Bio import Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.export import build_session_zip, has_results
from modules.pairwise import align_pair


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def test_zip_contents():
    seqs = [rec("a", "MKVLAA"), rec("b", "MKVLAG")]
    state = {
        "pairwise_result": align_pair(seqs[0], seqs[1], "Protein"),
        "msa_aligned": [("a", "MKVLAA"), ("b", "MKVLAG")],
        "tree_result": {"params": {}, "tree": Phylo.read(StringIO("(a:0.1,b:0.2);"), "newick")},
        "last_pairwise_params": {"ref_id": "a", "target_id": "b"},
    }
    assert has_results(state)
    archive = zipfile.ZipFile(io.BytesIO(build_session_zip(state, seqs, "Protein")))
    names = set(archive.namelist())
    assert {"sequences.fasta", "session.json", "pairwise_alignment.txt", "msa_alignment.fasta", "tree.nwk",
            "README.txt"} <= names
    meta = json.loads(archive.read("session.json"))
    assert meta["seq_type"] == "Protein"
    assert meta["parameters"]["last_pairwise_params"]["ref_id"] == "a"


def test_has_results_empty():
    assert not has_results({})
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: modules.export`.

- [ ] **Step 3: Implement**

`modules/export.py`:

```python
"""Bundle the current dataset, results and parameters into one ZIP."""
import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
from Bio import SeqIO

from modules.seqtools import fingerprint

PARAM_KEYS = ("last_pairwise_params", "last_msa_params", "best_match_params", "last_conversion_params")
RESULT_KEYS_EXPORTED = ("pairwise_result", "msa_aligned", "best_match_results", "clustering_matrix",
                        "antibody_results", "orf_results", "tree_result", "msa_tree", "converted_data")


def has_results(state):
    return any(state.get(k) for k in RESULT_KEYS_EXPORTED)


def build_session_zip(state, sequences, seq_type):
    from modules.antibody import public_columns
    from modules.msa import serialize_alignment
    from modules.pairwise import format_result_text
    from modules.phylogeny import tree_to_newick

    files = {}
    fasta = io.StringIO()
    SeqIO.write(sequences, fasta, "fasta")
    files["sequences.fasta"] = fasta.getvalue()

    if state.get("pairwise_result"):
        files["pairwise_alignment.txt"] = format_result_text(state["pairwise_result"])
    if state.get("msa_aligned"):
        files["msa_alignment.fasta"] = serialize_alignment(state["msa_aligned"], "fasta")
    if state.get("best_match_results"):
        files["best_match.csv"] = pd.DataFrame(state["best_match_results"][1]).to_csv(index=False)
    if state.get("clustering_matrix") is not None and state.get("clustering_ids"):
        ids = state["clustering_ids"]
        files["clustering_identity_matrix.csv"] = pd.DataFrame(state["clustering_matrix"], index=ids,
                                                               columns=ids).to_csv()
    if state.get("antibody_results"):
        df = state["antibody_results"]["df"]
        files["antibody_domains.csv"] = df[public_columns(df)].to_csv(index=False)
    if state.get("orf_results"):
        files["orfs.csv"] = pd.DataFrame(state["orf_results"]["orfs"]).to_csv(index=False)
    if state.get("tree_result"):
        files["tree.nwk"] = tree_to_newick(state["tree_result"]["tree"])
    if state.get("msa_tree"):
        files["msa_tree.nwk"] = tree_to_newick(state["msa_tree"])
    if state.get("converted_data"):
        files["converted_sequences.txt"] = state["converted_data"]

    meta = {
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seq_type": seq_type,
        "dataset_fingerprint": fingerprint(sequences),
        "sequence_count": len(sequences),
        "parameters": {k: state[k] for k in PARAM_KEYS if state.get(k)},
    }
    files["session.json"] = json.dumps(meta, indent=2, default=str)
    files["README.txt"] = ("Advanced Sequence Alignment Tool: session export\n\n"
                           + "\n".join(f"- {name}" for name in sorted(files)) + "\n")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()
```

`app.py`: import `build_session_zip, has_results`, and at the end of `main()` (after the analysis dispatch) add:

```python
    if has_results(st.session_state):
        st.sidebar.divider()
        st.sidebar.download_button("📦 Export session (ZIP)", data=build_session_zip(st.session_state, sequences, seq_type),
                                   file_name="seq_align_session.zip", mime="application/zip", on_click="ignore",
                                   help="All current results, the input sequences and the parameters used.")
```

Add a final row to the README analyses table: `| Session export | One ZIP with input sequences, every computed result and the parameters used |`.

- [ ] **Step 4: Run the full suite and the gate**

Run: `python -m pytest -q && python precommit_check.py`
Expected: all pass; `Pre-commit passed.`

- [ ] **Step 5: Manual smoke test of the whole plan**

1. Load Globins. Run MSA with MAFFT, open Variant Table, Conservation (logo) and an NJ tree with 100 bootstraps (interactive view shows red support values). Check Sequence Properties.
2. Load pUC19 and open DNA Tools → Restriction map (circular): EcoRI and HindIII appear as unique cutters.
3. Export the session ZIP and inspect its contents.

- [ ] **Step 6: Commit**

```bash
git add modules/export.py app.py README.md tests/test_export.py
git commit -m "feat: one-click session export of sequences, results and parameters

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
