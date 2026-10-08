# Plan 01 — Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every existing analysis return scientifically correct results, never crash on ordinary input, and run on a pinned, tested environment.

**Architecture:** Shared cleaning logic moves into a new pure module, `modules/seqtools.py`. Each analysis module gets pure, unit-tested compute functions (no `st.*` calls) that its existing `*_section` UI function calls. Public function names used by the current tests stay; where a return type has to change, a new function is added and the old one becomes a thin compatibility wrapper.

**Tech Stack:** Python 3.12, Streamlit 1.60.0, Biopython 1.84, NumPy, SciPy, scikit-learn, pyFAMSA, ANARCI (vendored) plus HMMER, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-codebase-review.md` (items S1–S14, C1–C3, C6, C9, and C7 for antibody)

## Global Constraints

- Python ≥ 3.10 (development and CI use 3.12).
- Pinned versions: `streamlit==1.60.0`, `biopython==1.84`, `numpy==1.26.4`, `pandas==2.2.3`, `plotly==5.24.1`, `matplotlib==3.9.2`, `scipy==1.14.1`, `scikit-learn==1.8.0`, `pyfamsa==0.3.2`, `kaleido==0.2.1` (removed in Plan 02), `pytest==9.1.1` (dev).
- Default gap scores everywhere: open `-10.0`, extend `-0.5` (`DEFAULT_OPEN_GAP`, `DEFAULT_EXTEND_GAP` in `modules/pairwise.py`).
- Pure compute functions must not call `st.*`. UI lives in `*_section` functions.
- Use `width="stretch"` / `width="content"`; never `use_container_width`.
- Branch: `fix/01-correctness`. Commit after every task. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run tests with `python -m pytest -q` from the repo root.

## Review Focus

Inputs and failure modes that the spec implies but no single feature test would naturally cover. Each one is pinned by a test in the task listed.

1. **Lowercase, RNA (`U`) and stop (`*`) characters** in any analysis. Expected: cleaned and analysed, never a crash. Covered in Task 2 (`test_clean_sequence_*`), Task 6 (`test_pairwise_matrix_lowercase`), Task 7 (`test_tree_lowercase_input`) and Task 8 (`test_orf_rna_input`).
2. **Gap characters inside FASTA input** (sequences pasted from an alignment). Expected: gaps stripped before pairwise alignment, so the score equals the ungapped score. Covered in Task 2 (`test_calculate_alignment_score_ignores_gaps_in_input`).
3. **Structures with insertion codes** (Kabat-numbered antibody PDBs). Expected: renumbered residues get an empty insertion code; insertions relative to the target get A, B, C… Covered in Task 11 (`test_insertion_codes_reset_and_assigned`).
4. **Very short or empty-after-cleaning sequences.** Expected: a clear error message, not a traceback. Covered in Task 3 (`test_align_pair_rejects_empty`) and Task 5 (`test_run_msa_rejects_empty_sequence`).
5. **Duplicate IDs from merged structure files or multi-model files.** Expected: unique IDs, no "Duplicate sequence ID" failure. Covered in Task 10 (`test_make_unique_ids`, `test_nmr_first_model_only`).

---

### Task 1: Pinned environment, test infrastructure, CI and housekeeping

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`, `pytest.ini`, `conftest.py`, `.github/workflows/tests.yml`
- Rewrite: `precommit_check.py`
- Modify: `.gitignore`
- Delete: `streamlit.log`, `streamlit_run.log`
- Modify: `modules/pairwise.py`, `modules/viz.py`, `modules/clustering.py`, `modules/best_match.py` (deprecation sweep)

**Interfaces:**
- Consumes: nothing
- Produces: a pinned environment; `python -m pytest -q` runs from the repo root without `sys.path` hacks; `python precommit_check.py` exits non-zero on failure.

- [ ] **Step 1: Create the branch**

```bash
git checkout main && git pull --ff-only && git checkout -b fix/01-correctness
```

- [ ] **Step 2: Pin requirements**

`requirements.txt` (full replacement):

```
streamlit==1.60.0
biopython==1.84
numpy==1.26.4
pandas==2.2.3
plotly==5.24.1
matplotlib==3.9.2
scipy==1.14.1
scikit-learn==1.8.0
kaleido==0.2.1
# pyFAMSA wheel is pinned for compatibility on hosted servers
pyfamsa==0.3.2
```

`requirements-dev.txt`:

```
-r requirements.txt
pytest==9.1.1
```

- [ ] **Step 3: Install and confirm the starting state**

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Expected: `1 failed, 19 passed`. The failure is `test_get_aligner_overlap` (spec S1); Task 2 fixes it.

- [ ] **Step 4: Add pytest configuration and a root conftest**

`pytest.ini`:

```ini
[pytest]
testpaths = tests
filterwarnings =
    ignore::DeprecationWarning
```

`conftest.py` (repo root):

```python
import os
import sys

# Make `import modules.*` and `import anarci` work from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

- [ ] **Step 5: Make the pre-commit check fail loudly**

`precommit_check.py` (full replacement):

```python
"""Local gate: byte-compile all first-party code and run the test suite. Exits non-zero on failure."""
import pathlib
import py_compile
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> int:
    sources = [ROOT / "app.py", *sorted((ROOT / "modules").glob("*.py"))]
    for path in sources:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            print(f"Compile error: {exc}")
            return 1
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    if result.returncode != 0:
        print("Pre-commit failed.")
        return result.returncode
    print("Pre-commit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Add CI**

`.github/workflows/tests.yml`:

```yaml
name: tests
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install HMMER (needed by ANARCI)
        run: sudo apt-get update && sudo apt-get install -y hmmer
      - name: Install Python dependencies
        run: pip install -r requirements-dev.txt
      - name: Run tests
        run: python -m pytest -q
```

- [ ] **Step 7: Remove committed logs and ignore them**

```bash
git rm streamlit.log streamlit_run.log
```

Append to `.gitignore`:

```
*.log
.venv/
.pytest_cache/
```

- [ ] **Step 8: Replace the deprecated `use_container_width`**

```bash
grep -rln "use_container_width" modules app.py | xargs sed -i \
  -e 's/use_container_width=True/width="stretch"/g' \
  -e 's/use_container_width=False/width="content"/g'
grep -rn "use_container_width" modules app.py || echo "clean"
```

Expected: `clean`.

- [ ] **Step 9: Verify nothing else regressed**

Run: `python -m pytest -q`
Expected: `1 failed, 19 passed` (still only `test_get_aligner_overlap`).

Run: `python precommit_check.py; echo "exit=$?"`
Expected: ends with `Pre-commit failed.` and `exit=1`. This proves the gate now fails loudly.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "chore: pin dependencies, add pytest config and CI, drop committed logs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Sequence cleaning module and aligner fixes (Overlap mode, gap defaults, U/O/J)

**Files:**
- Create: `modules/seqtools.py`
- Modify: `modules/pairwise.py` (imports, `get_aligner`, `calculate_alignment_score`, the cleaning lines in `perform_pairwise_alignment`, default arguments)
- Modify: `tests/test_best_match.py` (fix the wrong attribute name in `test_get_aligner_overlap`)
- Test: `tests/test_seqtools.py`, `tests/test_pairwise.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `modules.seqtools.clean_sequence(seq, seq_type: str, keep_gaps: bool = False) -> str`
  - `modules.seqtools.DNA_LETTERS`, `PROTEIN_LETTERS`, `PROTEIN_SUBSTITUTIONS`
  - `modules.pairwise.DEFAULT_OPEN_GAP = -10.0`, `DEFAULT_EXTEND_GAP = -0.5`
  - `modules.pairwise.load_matrix(name: str)`, which is cached
  - `get_aligner(seq_type, mode="global", open_gap_score=DEFAULT_OPEN_GAP, extend_gap_score=DEFAULT_EXTEND_GAP)`

- [ ] **Step 1: Write the failing tests**

`tests/test_seqtools.py`:

```python
from modules.seqtools import clean_sequence


def test_clean_sequence_protein_maps_rare_residues_and_case():
    assert clean_sequence("mkuOj*x", "Protein") == "MKCKX*X"


def test_clean_sequence_protein_drops_gaps_unless_kept():
    assert clean_sequence("MK-VL", "Protein") == "MKVL"
    assert clean_sequence("MK-VL", "Protein", keep_gaps=True) == "MK-VL"


def test_clean_sequence_dna_converts_rna_and_drops_junk():
    assert clean_sequence("acgu-n 12", "DNA") == "ACGTN"


def test_clean_sequence_accepts_seq_objects():
    from Bio.Seq import Seq
    assert clean_sequence(Seq("atg"), "DNA") == "ATG"
```

`tests/test_pairwise.py`:

```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.pairwise import (
    DEFAULT_EXTEND_GAP,
    DEFAULT_OPEN_GAP,
    calculate_alignment_score,
    get_aligner,
    perform_pairwise_alignment,
)

REF = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKW"
    "KRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIK"
)


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def test_default_gap_scores_are_needle_like():
    aligner = get_aligner("Protein")
    assert DEFAULT_OPEN_GAP == -10.0
    assert DEFAULT_EXTEND_GAP == -0.5
    assert aligner.open_gap_score == -10.0
    assert aligner.extend_gap_score == -0.5


def test_overlap_mode_aligns_fragment():
    text, mutations = perform_pairwise_alignment(
        rec("ref", REF), rec("frag", "LEERLGLIEVQAPWLSRV"), "Protein", "overlap"
    )
    assert text.startswith("Pairwise Alignment"), text
    assert mutations == ["I36W"]


def test_selenocysteine_and_lowercase_do_not_crash():
    text, _ = perform_pairwise_alignment(rec("a", "mkuvlA"), rec("b", "MKCVLA"), "Protein")
    assert text.startswith("Pairwise Alignment"), text


def test_calculate_alignment_score_ignores_gaps_in_input():
    aligner = get_aligner("Protein")
    gapped = calculate_alignment_score(aligner, rec("a", "MK-VL"), rec("b", "MKVL"), "Protein")
    plain = calculate_alignment_score(aligner, rec("a", "MKVL"), rec("b", "MKVL"), "Protein")
    assert gapped == plain
```

In `tests/test_best_match.py`, change the first line of `test_find_best_match_normalized` to `aligner = get_aligner("DNA", mode="overlap")`. The test assumes nearly free terminal gaps (a short perfect match scores best per residue); with the new −10 open-gap default, only free end gaps preserve that premise.

Also fix the wrong attribute names in `test_get_aligner_overlap`:

```python
    def test_get_aligner_overlap(self):
        aligner = get_aligner("DNA", mode="overlap", open_gap_score=-0.5, extend_gap_score=-0.1)
        self.assertEqual(aligner.mode, "global")
        self.assertEqual(aligner.end_open_gap_score, 0.0)
        self.assertEqual(aligner.end_extend_gap_score, 0.0)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_seqtools.py tests/test_pairwise.py tests/test_best_match.py -v`
Expected: FAIL. `ModuleNotFoundError: modules.seqtools`, `ImportError: DEFAULT_OPEN_GAP`, and the overlap tests fail.

- [ ] **Step 3: Create `modules/seqtools.py`**

```python
"""Sequence helpers shared by all analysis modules. No Streamlit calls."""

DNA_LETTERS = frozenset("ACGTNRYKMSWBDHV")
# Alphabet of BLOSUM62 as shipped with Biopython.
PROTEIN_LETTERS = frozenset("ARNDCQEGHILKMFPSTWYVBZX*")
# Valid FASTA residues absent from BLOSUM62, mapped to the closest standard residue.
PROTEIN_SUBSTITUTIONS = {"U": "C", "O": "K", "J": "X"}


def clean_sequence(seq, seq_type, keep_gaps=False):
    """Upper-case a sequence and reduce it to the scoring alphabet.

    DNA: U becomes T; anything outside IUPAC nucleotide codes is dropped.
    Protein: U->C, O->K, J->X; anything outside the BLOSUM62 alphabet is dropped.
    Gaps ('-') are dropped unless keep_gaps is True.
    """
    text = str(seq).upper()
    if seq_type == "DNA":
        text = text.replace("U", "T")
        allowed = DNA_LETTERS
    else:
        text = "".join(PROTEIN_SUBSTITUTIONS.get(c, c) for c in text)
        allowed = PROTEIN_LETTERS
    return "".join(c for c in text if c in allowed or (keep_gaps and c == "-"))
```

- [ ] **Step 4: Fix the aligner in `modules/pairwise.py`**

Add to the imports at the top of the file:

```python
from functools import lru_cache
from modules.seqtools import clean_sequence
```

Add these definitions directly below the imports:

```python
DEFAULT_OPEN_GAP = -10.0
DEFAULT_EXTEND_GAP = -0.5


@lru_cache(maxsize=None)
def load_matrix(name):
    """Load a substitution matrix once per process."""
    return substitution_matrices.load(name)
```

Replace `get_aligner` entirely:

```python
def get_aligner(seq_type, mode="global", open_gap_score=DEFAULT_OPEN_GAP, extend_gap_score=DEFAULT_EXTEND_GAP):
    """
    Configure and return a PairwiseAligner.

    Parameters:
        seq_type (str): Type of sequences ('DNA' or 'Protein')
        mode (str): Alignment mode ('global', 'local', or 'overlap')
        open_gap_score (float): Score for opening a gap
        extend_gap_score (float): Score for extending a gap

    Returns:
        Align.PairwiseAligner: Configured aligner object
    """
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = load_matrix("NUC.4.4" if seq_type == "DNA" else "BLOSUM62")
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score

    if mode == "overlap":
        # Global alignment with free end gaps (semi-global). Must be set after the
        # general gap scores, which also overwrite the end-gap scores.
        aligner.mode = "global"
        aligner.end_open_gap_score = 0.0
        aligner.end_extend_gap_score = 0.0
    else:
        aligner.mode = mode

    return aligner
```

Replace the body of `calculate_alignment_score` from `if seq_type == "DNA":` to the end of the function:

```python
    s1_str = str(seq1.seq) if hasattr(seq1, 'seq') else str(seq1)
    s2_str = str(seq2.seq) if hasattr(seq2, 'seq') else str(seq2)

    seq1_clean = clean_sequence(s1_str, seq_type)
    seq2_clean = clean_sequence(s2_str, seq_type)

    if not seq1_clean or not seq2_clean:
        return None

    return aligner.score(seq1_clean, seq2_clean)
```

In `perform_pairwise_alignment`, change the signature defaults to the constants and replace the cleaning block (from `# Clean sequences by converting to uppercase` through the `seq2_str = ...` line):

```python
def perform_pairwise_alignment(seq1, seq2, seq_type, mode="global", open_gap_score=DEFAULT_OPEN_GAP, extend_gap_score=DEFAULT_EXTEND_GAP, is_circular1=False, is_circular2=False):
```

```python
        seq1_str = clean_sequence(seq1.seq, seq_type)
        seq2_str = clean_sequence(seq2.seq, seq_type)
```

In `pairwise_alignment_section`, change the two `number_input` defaults: `value=-0.5` becomes `value=DEFAULT_OPEN_GAP`, and `value=-0.1` becomes `value=DEFAULT_EXTEND_GAP`. In `modules/best_match.py`, import `DEFAULT_OPEN_GAP, DEFAULT_EXTEND_GAP` from `modules.pairwise` and change its two gap `number_input` defaults the same way. In `modules/phylogeny.py:150`, remove the explicit `open_gap_score=-0.5, extend_gap_score=-0.1` arguments so the defaults apply.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add modules/seqtools.py modules/pairwise.py modules/best_match.py modules/phylogeny.py tests/test_seqtools.py tests/test_pairwise.py tests/test_best_match.py
git commit -m "fix: overlap mode, needle-like gap defaults, shared sequence cleaning

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `align_pair` with correct local and circular coordinates; UI shows errors as errors

**Files:**
- Modify: `modules/pairwise.py` (new dataclass, functions and wrapper; rewrite `pairwise_alignment_section`)
- Modify: `modules/utils.py` (session defaults)
- Modify: `app.py` (reset-key list)
- Test: `tests/test_pairwise.py`

**Interfaces:**
- Consumes: `clean_sequence`, `get_aligner`, `DEFAULT_*` from Task 2
- Produces:
  - `AlignmentError(ValueError)`
  - `PairwiseResult` dataclass with fields `ref_id, query_id, aligned_ref, aligned_query, match_line, score, ref_start, query_start, ref_rotation, ref_length, mutations, notes`
  - `align_pair(seq1, seq2, seq_type, mode="global", open_gap_score=DEFAULT_OPEN_GAP, extend_gap_score=DEFAULT_EXTEND_GAP, is_circular1=False, is_circular2=False) -> PairwiseResult`
  - `location_line(aligned_seq, interval=10, start=0, rotation=0, length=None) -> str`
  - `report_mutations(aligned1, aligned2, ref_start=0, ref_rotation=0, ref_length=None) -> list[str]`
  - `format_result_text(result: PairwiseResult) -> str`
  - Session key `pairwise_result` replaces `alignment_text` and `pairwise_mutations`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pairwise.py`)

```python
import pytest

from modules.pairwise import AlignmentError, align_pair, format_result_text, location_line, report_mutations

CIRC = "GATTACAGGCCTTAAGCTAGCATCGGATCCAGTC"


def test_local_alignment_reports_reference_coordinates():
    result = align_pair(rec("ref", REF), rec("q", "RQLEERLGLIEVQAPWLSRVGDG"), "Protein", "local")
    assert result.ref_start == 20
    assert result.mutations == ["I36W"]


def test_circular_mutations_use_original_reference_coordinates():
    mutated = CIRC[:2] + "C" + CIRC[3:]          # T3C
    rotated = mutated[7:] + mutated[:7]
    for circ_ref, circ_tgt in [(True, False), (False, True), (True, True)]:
        result = align_pair(
            rec("ref", CIRC), rec("t", rotated), "DNA", "global",
            is_circular1=circ_ref, is_circular2=circ_tgt,
        )
        assert result.mutations == ["T3C"], (circ_ref, circ_tgt, result.mutations)


def test_align_pair_rejects_empty():
    with pytest.raises(AlignmentError, match="empty"):
        align_pair(rec("a", ""), rec("b", "ATGG"), "DNA")


def test_location_line_respects_start_offset():
    assert location_line("A" * 15, 10, start=20) == "        30     "


def test_report_mutations_with_rotation():
    assert report_mutations("CDEAK", "CDEWK", ref_rotation=2, ref_length=5) == ["A1W"]


def test_format_result_text_contains_score_and_notes():
    result = align_pair(rec("ref", CIRC), rec("t", CIRC[5:] + CIRC[:5]), "DNA", is_circular2=True)
    text = format_result_text(result)
    assert "Score" in text
    assert "circularly shifted" in text
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_pairwise.py -v`
Expected: FAIL with `ImportError: cannot import name 'AlignmentError'`.

- [ ] **Step 3: Implement**

In `modules/pairwise.py`, add `import logging` and `from dataclasses import dataclass, field` to the imports, plus `logger = logging.getLogger(__name__)` below the constants. Then add the following after `get_aligner`:

```python
class AlignmentError(ValueError):
    """Raised when two sequences cannot be aligned (e.g. empty after cleaning)."""


@dataclass
class PairwiseResult:
    ref_id: str
    query_id: str
    aligned_ref: str
    aligned_query: str
    match_line: str
    score: float
    ref_start: int          # reference residues before the first aligned column (0 unless local)
    query_start: int
    ref_rotation: int       # circular shift applied to the reference (0 if none)
    ref_length: int
    mutations: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _circular_rotations(s1, s2, seq_type, open_gap_score, extend_gap_score, circular1, circular2):
    """Find rotations that bring the best local match of the two sequences into register."""
    opt_aligner = get_aligner(seq_type, "local", open_gap_score, extend_gap_score)
    doubled1 = s1 + s1 if circular1 else s1
    doubled2 = s2 + s2 if circular2 else s2
    try:
        opt_aln = next(iter(opt_aligner.align(doubled1, doubled2)))
    except StopIteration:
        return 0, 0
    start1 = int(opt_aln.coordinates[0][0])
    start2 = int(opt_aln.coordinates[1][0])
    if circular2:
        return 0, (start2 - start1) % len(s2)
    return (start1 - start2) % len(s1), 0


def align_pair(seq1, seq2, seq_type, mode="global", open_gap_score=DEFAULT_OPEN_GAP,
               extend_gap_score=DEFAULT_EXTEND_GAP, is_circular1=False, is_circular2=False):
    """Align two SeqRecords. Mutation positions are always in original reference coordinates."""
    s1 = clean_sequence(seq1.seq, seq_type)
    s2 = clean_sequence(seq2.seq, seq_type)
    if not s1 or not s2:
        raise AlignmentError("One or both sequences are empty after cleaning.")

    notes = []
    rot1 = rot2 = 0
    if is_circular1 or is_circular2:
        rot1, rot2 = _circular_rotations(s1, s2, seq_type, open_gap_score, extend_gap_score,
                                         is_circular1, is_circular2)
        if rot1:
            s1 = s1[rot1:] + s1[:rot1]
            notes.append(f"Reference sequence was circularly shifted by {rot1} positions; "
                         f"reported positions refer to the original reference.")
        if rot2:
            s2 = s2[rot2:] + s2[:rot2]
            notes.append(f"Target sequence was circularly shifted by {rot2} positions to optimize alignment.")

    aligner = get_aligner(seq_type, mode, open_gap_score, extend_gap_score)
    try:
        alignment = next(iter(aligner.align(s1, s2)))
    except StopIteration:
        raise AlignmentError("No alignment found. Sequences may be too dissimilar.")

    aligned_ref = str(alignment[0])
    aligned_query = str(alignment[1])
    ref_start = int(alignment.coordinates[0][0])
    query_start = int(alignment.coordinates[1][0])

    return PairwiseResult(
        ref_id=seq1.id,
        query_id=seq2.id,
        aligned_ref=aligned_ref,
        aligned_query=aligned_query,
        match_line=generate_match_line(aligned_ref, aligned_query, aligner.substitution_matrix),
        score=float(alignment.score),
        ref_start=ref_start,
        query_start=query_start,
        ref_rotation=rot1,
        ref_length=len(s1),
        mutations=report_mutations(aligned_ref, aligned_query, ref_start=ref_start,
                                   ref_rotation=rot1, ref_length=len(s1)),
        notes=notes,
    )


def format_result_text(result):
    """Render a PairwiseResult as the plain-text block shown in the UI and downloads."""
    text = format_alignment_display(
        result.ref_id, result.aligned_ref, result.match_line,
        result.query_id, result.aligned_query, result.score,
        start1=result.ref_start, start2=result.query_start,
        rotation1=result.ref_rotation, length1=result.ref_length,
    )
    if result.notes:
        text += "\n" + "\n".join(f"Note: {n}" for n in result.notes) + "\n"
    return text
```

Replace `perform_pairwise_alignment` entirely with a compatibility wrapper:

```python
def perform_pairwise_alignment(seq1, seq2, seq_type, mode="global", open_gap_score=DEFAULT_OPEN_GAP,
                               extend_gap_score=DEFAULT_EXTEND_GAP, is_circular1=False, is_circular2=False):
    """Backward-compatible wrapper around align_pair returning (alignment_text, mutations)."""
    try:
        result = align_pair(seq1, seq2, seq_type, mode, open_gap_score, extend_gap_score,
                            is_circular1, is_circular2)
    except AlignmentError as e:
        return str(e), []
    except Exception as e:
        logger.exception("Pairwise alignment failed")
        return f"An error occurred during pairwise alignment: {e}", []
    return format_result_text(result), result.mutations
```

Replace `format_alignment_display` (including its nested `generate_location_line`) with a module-level `location_line` plus the new display function:

```python
def location_line(aligned_seq, interval=10, start=0, rotation=0, length=None):
    """Ruler marking every `interval`-th residue in original sequence coordinates."""
    location = [' '] * len(aligned_seq)
    count = start
    for i, char in enumerate(aligned_seq):
        if char == '-':
            continue
        count += 1
        pos = ((count - 1 + rotation) % length) + 1 if rotation and length else count
        if pos % interval == 0:
            mark = str(pos)
            start_pos = max(0, i - len(mark) + 1)
            for j, digit in enumerate(mark):
                k = start_pos + j
                if k < len(aligned_seq) and location[k] == ' ':
                    location[k] = digit
    return ''.join(location)


def format_alignment_display(id1, aligned1, match_line, id2, aligned2, score, interval=10,
                             start1=0, start2=0, rotation1=0, length1=None):
    """Formatted alignment block with position rulers in original sequence coordinates."""
    max_label_length = max(len(id1), len("Match"), len(id2))
    padding = ' ' * (max_label_length + 2)
    return (
        "Pairwise Alignment:\n"
        f"{padding}{location_line(aligned1, interval, start1, rotation1, length1)}\n"
        f"{id1.ljust(max_label_length)}: {aligned1}\n"
        f"{'Match'.ljust(max_label_length)}: {match_line}\n"
        f"{id2.ljust(max_label_length)}: {aligned2}\n"
        f"{padding}{location_line(aligned2, interval, start2)}\n"
        f"{'Score'.ljust(max_label_length)}: {score:.2f}\n"
    )
```

Replace `report_mutations`:

```python
def report_mutations(aligned1, aligned2, ref_start=0, ref_rotation=0, ref_length=None):
    """
    Report substitutions between two aligned sequences as e.g. 'S18I'.

    Positions are in original reference coordinates: `ref_start` is the number of
    reference residues before the first aligned column (local alignments) and
    `ref_rotation` undoes a circular shift of the reference.
    """
    mutations = []
    position = ref_start
    for a, b in zip(aligned1, aligned2):
        if a != '-':
            position += 1
        if a != b and a != '-' and b != '-':
            pos = position
            if ref_rotation and ref_length:
                pos = ((position - 1 + ref_rotation) % ref_length) + 1
            mutations.append(f"{a}{pos}{b}")
    return mutations
```

Rewrite `pairwise_alignment_section`. Keep everything from the start of the function through the `extend_gap_score = c2.number_input(...)` call unchanged, except for the reset block inside `if selection_changed:`, which becomes:

```python
    if selection_changed:
        # Reset alignment results when selection changes
        st.session_state.pairwise_result = None
        st.session_state.last_pairwise_params = {}
        st.session_state.last_selected_seqs = (reference_id, target_id)
```

Replace everything from `run_alignment = st.button("Run Pairwise Alignment")` to the end of the function:

```python
    run_alignment = st.button("Run Pairwise Alignment")

    if run_alignment:
        with st.spinner("Performing pairwise alignment..."):
            try:
                st.session_state.pairwise_result = align_pair(
                    seq1, seq2, seq_type, align_mode.lower(), open_gap_score, extend_gap_score,
                    is_circular_ref, is_circular_target
                )
                st.session_state.last_pairwise_params = current_params
            except AlignmentError as e:
                st.session_state.pairwise_result = None
                st.session_state.last_pairwise_params = {}
                st.error(str(e))
                return

    result = st.session_state.get('pairwise_result')
    if result is None or (params_changed and not run_alignment):
        return

    alignment_text = format_result_text(result)
    st.subheader("🧬 Alignment Result")
    st.code(alignment_text)

    st.subheader("🔍 Point Mutations Relative to Reference")
    if result.mutations:
        st.write(f"**Reference Sequence:** `{reference_id}`")
        st.write(f"**Target Sequence:** `{target_id}`")
        st.write(f"**Mutations:** {', '.join(result.mutations)}")

        mutation_positions = [int(re.search(r"\d+", mut).group()) for mut in result.mutations]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mutation_positions,
            y=[1] * len(mutation_positions),
            mode='markers',
            marker=dict(size=10, color='red'),
            text=result.mutations,
            hoverinfo='text'
        ))
        fig.update_layout(
            title="Point Mutations Plot",
            xaxis_title="Position in Reference Sequence",
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, width="stretch")

        csv_mutations = convert_mutations_to_csv(result.mutations)
        if csv_mutations:
            st.download_button(
                label="📥 Download Mutations (CSV)",
                data=csv_mutations,
                file_name=f"pairwise_mutations_{reference_id}_vs_{target_id}.csv",
                mime="text/csv"
            )
    else:
        st.write("No point mutations detected.")

    st.download_button(
        label="📥 Download Alignment",
        data=alignment_text,
        file_name=f"pairwise_alignment_{reference_id}_vs_{target_id}.txt",
        mime="text/plain"
    )
```

In `modules/utils.py` `init_session_state`, replace the two default entries `"alignment_text": None,` and `"pairwise_mutations": None,` with one entry, `"pairwise_result": None,`.

In `app.py`, in the `keys_to_reset` list, replace `'alignment_text', 'pairwise_mutations',` with `'pairwise_result',`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_pairwise.py tests/test_logic.py -v`
Expected: PASS (the existing `test_perform_pairwise_alignment_empty` still finds "One or both sequences are empty").

- [ ] **Step 5: Manual smoke check**

Run `streamlit run app.py`, paste two protein sequences, choose **Local**, and run. The mutation numbers should match the reference residue numbering. Choose **Overlap** and run; it should no longer show an error.

- [ ] **Step 6: Commit**

```bash
git add modules/pairwise.py modules/utils.py app.py tests/test_pairwise.py
git commit -m "fix: correct mutation coordinates for local and circular alignments; surface alignment errors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Best Match with comparable metrics (relative score, identity, coverage)

**Files:**
- Modify: `modules/best_match.py`
- Test: `tests/test_best_match.py`

**Interfaces:**
- Consumes: `align_pair`, `AlignmentError`, `format_result_text`, `get_aligner`, `DEFAULT_*`, `clean_sequence`
- Produces:
  - `identity_and_coverage(aligned_ref: str, aligned_query: str, ref_length: int) -> tuple[float, float]` (percentages)
  - `sort_results(results: list[dict], sort_key: str) -> list[dict]` (sorts in place, sets `Rank`, returns the list)
  - `find_best_match(...)`: same signature; each result dict gains `Relative Score`, `Identity (%)` and `Coverage (%)`. `Normalized Score` (score / query length) is kept for compatibility.

- [ ] **Step 1: Write the failing tests** (append to `TestBestMatchFinder` in `tests/test_best_match.py`)

```python
    def test_relative_score_identity_and_coverage(self):
        from modules.best_match import identity_and_coverage
        aligner = get_aligner("Protein")
        ref = SeqRecord(Seq("MVLSPADKTN"), id="Ref")
        s1 = SeqRecord(Seq("MVLSPADKTN"), id="S1")
        s2 = SeqRecord(Seq("MVLSPADK"), id="S2")
        _, results = find_best_match([ref, s1, s2], "Ref", aligner, "Protein", sort_key="Relative Score")
        by_id = {r["Sequence ID"]: r for r in results}
        self.assertAlmostEqual(by_id["S1"]["Relative Score"], 1.0)
        self.assertEqual(by_id["S2"]["Identity (%)"], 100.0)
        self.assertEqual(by_id["S2"]["Coverage (%)"], 80.0)
        self.assertEqual(identity_and_coverage("MKV-", "MRVL", 3), (2 / 3 * 100, 100.0))

    def test_sort_results_reassigns_rank(self):
        from modules.best_match import sort_results
        rows = [{"Score": 1, "Identity (%)": 90}, {"Score": 5, "Identity (%)": 10}]
        sort_results(rows, "Identity (%)")
        self.assertEqual([r["Rank"] for r in rows], [1, 2])
        self.assertEqual(rows[0]["Identity (%)"], 90)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_best_match.py -v`
Expected: FAIL with `KeyError: 'Relative Score'` / `ImportError: identity_and_coverage`.

- [ ] **Step 3: Implement**

At the top of `modules/best_match.py`, replace the import line with:

```python
from modules.pairwise import (
    DEFAULT_EXTEND_GAP, DEFAULT_OPEN_GAP, AlignmentError, align_pair, format_result_text, get_aligner,
)
from modules.seqtools import clean_sequence

SORT_OPTIONS = {
    "Relative Score": "Score as a fraction of the reference's self-alignment score (comparable across queries)",
    "Identity (%)": "Identical residues over aligned (non-gap) columns",
    "Coverage (%)": "Fraction of reference residues aligned to a query residue",
    "Score": "Raw alignment score",
    "Normalized Score": "Score divided by query length",
}
```

Replace `find_best_match` and add the two helpers:

```python
def identity_and_coverage(aligned_ref, aligned_query, ref_length):
    """Percent identity over aligned columns and percent of the reference covered."""
    aligned_cols = sum(1 for a, b in zip(aligned_ref, aligned_query) if a != '-' and b != '-')
    matches = sum(1 for a, b in zip(aligned_ref, aligned_query) if a == b and a != '-')
    identity = matches / aligned_cols * 100 if aligned_cols else 0.0
    coverage = aligned_cols / ref_length * 100 if ref_length else 0.0
    return identity, coverage


def sort_results(results, sort_key):
    """Sort result rows by `sort_key` (descending) and renumber their Rank."""
    results.sort(key=lambda x: x.get(sort_key, x['Score']), reverse=True)
    for i, res in enumerate(results):
        res['Rank'] = i + 1
    return results


def find_best_match(sequences, reference_id, aligner, seq_type, sort_key="Score"):
    """
    Find the best matching sequence against a reference sequence.

    Returns:
        tuple: (best_match_record or None, results_list sorted by sort_key)
    """
    ref_seq = next((s for s in sequences if s.id == reference_id), None)
    if not ref_seq:
        return None, []
    ref_clean = clean_sequence(ref_seq.seq, seq_type)
    if not ref_clean:
        return None, []
    self_score = aligner.score(ref_clean, ref_clean)

    results = []
    for seq in sequences:
        if seq.id == reference_id:
            continue
        query_clean = clean_sequence(seq.seq, seq_type)
        if not query_clean:
            continue
        try:
            alignment = next(iter(aligner.align(ref_clean, query_clean)))
        except StopIteration:
            continue
        score = float(alignment.score)
        identity, coverage = identity_and_coverage(str(alignment[0]), str(alignment[1]), len(ref_clean))
        length = len(seq.seq)
        results.append({
            'Sequence ID': seq.id,
            'Score': score,
            'Length': length,
            'Normalized Score': score / length if length > 0 else 0,
            'Relative Score': score / self_score if self_score > 0 else 0.0,
            'Identity (%)': identity,
            'Coverage (%)': coverage,
        })

    if not results:
        return None, []

    sort_results(results, sort_key)
    best_match_record = next(s for s in sequences if s.id == results[0]['Sequence ID'])
    return best_match_record, results
```

UI changes in `best_match_finder_section`:

1. Replace `if len(sequences) < 3:` and its warning with:

```python
    if len(sequences) < 2:
        st.warning("Please upload at least two sequences (1 reference + 1 query) for this analysis.")
        return
```

2. In the reference `selectbox`, change `[seq.id for seq in sequences][::-1]` to `[seq.id for seq in sequences]`.
3. Replace the `sort_metric` selectbox with:

```python
    sort_metric = st.selectbox(
        "Rank by",
        list(SORT_OPTIONS),
        help="\n".join(f"**{k}**: {v}" for k, v in SORT_OPTIONS.items()),
    )
```

4. Remove `'sort_metric': sort_metric` from `current_params`, so re-sorting never reruns the alignments.
5. Directly after `best_match, results = st.session_state.best_match_results`, add:

```python
        results = sort_results([dict(r) for r in results], sort_metric)
        best_match = next(s for s in sequences if s.id == results[0]['Sequence ID']) if results else None
```

6. Change the success line to report the chosen metric:

```python
        st.success(f"🏆 Best Match: **{best_match.id}** — {sort_metric}: **{results[0][sort_metric]:.2f}**")
```

7. Replace the detailed alignment computation inside the expander:

```python
            try:
                detail = align_pair(ref_seq, selected_match_record, seq_type, align_mode.lower(),
                                    open_gap_score, extend_gap_score)
                st.code(format_result_text(detail))
                if detail.mutations:
                    st.write(f"**Mutations ({len(detail.mutations)}):** {', '.join(detail.mutations)}")
            except AlignmentError as e:
                st.error(str(e))
```

8. Change `display_cols` to:

```python
        display_cols = ['Rank', 'Sequence ID', 'Relative Score', 'Identity (%)', 'Coverage (%)', 'Score', 'Length', 'Normalized Score']
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_best_match.py -v`
Expected: PASS (all 9 tests).

- [ ] **Step 5: Commit**

```bash
git add modules/best_match.py tests/test_best_match.py
git commit -m "fix: best match ranks by comparable metrics (relative score, identity, coverage)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: MSA input cleaning, PHYLIP-relaxed output, correct consensus

**Files:**
- Modify: `modules/msa.py`
- Modify: `modules/clustering.py` (handle `MSAError` at the `perform_msa` call)
- Test: `tests/test_msa.py`, `tests/test_consensus.py`

**Interfaces:**
- Consumes: `clean_sequence`
- Produces:
  - `MSAError(RuntimeError)`
  - `MSA_OUTPUT_FORMATS = ("fasta", "clustal", "phylip-relaxed", "stockholm")`
  - `run_msa(pairs: list[tuple[str, str]], seq_type: str) -> list[tuple[str, str]]`
  - `serialize_alignment(aligned_pairs, output_format) -> str`
  - `mutations_from_alignment(aligned_pairs, reference_id) -> dict[str, list[tuple[int, str, str]]]`
  - `perform_msa(...)`: same signature; now raises `MSAError` instead of calling `st.error`
  - `calculate_representative_sequence(alignment, threshold=0.7, seq_type="Protein")`

- [ ] **Step 1: Write the failing tests**

`tests/test_msa.py`:

```python
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.msa import MSAError, mutations_from_alignment, perform_msa, run_msa, serialize_alignment


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def test_run_msa_uppercases_and_aligns():
    aligned = dict(run_msa([("a", "mkvlaagivg"), ("b", "MKVLAGIVG")], "Protein"))
    assert set(aligned) == {"a", "b"}
    assert len(aligned["a"]) == len(aligned["b"])
    assert aligned["a"].replace("-", "") == "MKVLAAGIVG"


def test_run_msa_rejects_empty_sequence():
    with pytest.raises(MSAError, match="empty"):
        run_msa([("a", "MKV"), ("b", "123")], "Protein")


def test_phylip_relaxed_keeps_long_ids():
    seqs = [rec("sample_0001_x", "MKVLAAGIVG"), rec("sample_0001_y", "MKVLAGIVG"), rec("sample_0002", "MKVAAGIVG")]
    text, _ = perform_msa(seqs, "sample_0001_x", "Protein", "phylip-relaxed")
    assert "sample_0001_y" in text


def test_strict_phylip_failure_raises_msa_error():
    with pytest.raises(MSAError):
        serialize_alignment([("sample_0001_x", "MK"), ("sample_0001_y", "MK")], "phylip")


def test_mutations_case_insensitive():
    muts = mutations_from_alignment([("ref", "MKV-L"), ("q", "MRVAL")], "ref")
    assert muts == {"q": [(2, "K", "R")]}
```

Append to `tests/test_consensus.py` inside the class:

```python
    def test_consensus_dna_uses_n_and_drops_gap_columns(self):
        alignment = MultipleSeqAlignment([
            SeqRecord(Seq("A-CT"), id="s1"),
            SeqRecord(Seq("A-CA"), id="s2"),
            SeqRecord(Seq("AGCG"), id="s3"),
        ])
        consensus, closest, min_diff, closest_id = calculate_representative_sequence(
            alignment, threshold=0.6, seq_type="DNA")
        self.assertEqual(str(consensus.seq), "ACN")     # gap-majority column removed, ambiguous -> N
        self.assertEqual(min_diff, 0)                    # N columns are not counted as differences
        self.assertNotIn("-", str(closest.seq))
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_msa.py tests/test_consensus.py -v`
Expected: FAIL with `ImportError: cannot import name 'MSAError'`.

- [ ] **Step 3: Implement**

In `modules/msa.py`, add `from collections import Counter` and `from modules.seqtools import clean_sequence` to the imports. Replace `perform_msa` and `calculate_representative_sequence`, and add the new helpers:

```python
class MSAError(RuntimeError):
    """Raised when the multiple sequence alignment cannot be computed or written."""


MSA_OUTPUT_FORMATS = ("fasta", "clustal", "phylip-relaxed", "stockholm")


def run_msa(pairs, seq_type):
    """Align (id, sequence) pairs with FAMSA and return [(id, aligned_sequence)]."""
    cleaned = [(seq_id, clean_sequence(seq, seq_type).replace("*", "")) for seq_id, seq in pairs]
    empty = [seq_id for seq_id, seq in cleaned if not seq]
    if empty:
        raise MSAError(f"Sequences empty after cleaning: {', '.join(empty)}")
    if len(cleaned) < 2:
        raise MSAError("At least two sequences are required for an MSA.")
    famsa_input = [PyFAMSASequence(seq_id.encode(), seq.encode()) for seq_id, seq in cleaned]
    try:
        msa = PyFAMSAAligner(guide_tree="upgma").align(famsa_input)
    except Exception as e:
        raise MSAError(f"FAMSA failed: {e}") from e
    return [(s.id.decode(), s.sequence.decode()) for s in msa]


def serialize_alignment(aligned_pairs, output_format):
    """Write aligned (id, sequence) pairs in an alignment format."""
    records = [SeqRecord(Seq(seq), id=seq_id, description="") for seq_id, seq in aligned_pairs]
    out = StringIO()
    try:
        SeqIO.write(records, out, output_format)
    except ValueError as e:
        raise MSAError(f"Could not write the alignment as {output_format}: {e}") from e
    return out.getvalue()


def mutations_from_alignment(aligned_pairs, reference_id):
    """Substitutions of every sequence relative to the reference, in reference coordinates."""
    aligned = dict(aligned_pairs)
    if reference_id not in aligned:
        raise MSAError(f"Reference sequence '{reference_id}' not found in the alignment.")
    ref = aligned[reference_id].upper()
    mutations = {}
    for seq_id, seq in aligned_pairs:
        if seq_id == reference_id:
            continue
        seq = seq.upper()
        seq_mutations = []
        ref_position = 0
        for ref_base, seq_base in zip(ref, seq):
            if ref_base != '-':
                ref_position += 1
            if ref_base != seq_base and ref_base != '-' and seq_base != '-':
                seq_mutations.append((ref_position, ref_base, seq_base))
        mutations[seq_id] = seq_mutations
    return mutations


def perform_msa(sequences, reference_id, seq_type, output_format):
    """
    Align SeqRecords with FAMSA and report mutations relative to the reference.

    Returns (msa_text, mutations). Raises MSAError on failure.
    """
    aligned = run_msa([(s.id, str(s.seq)) for s in sequences], seq_type)
    return serialize_alignment(aligned, output_format), mutations_from_alignment(aligned, reference_id)


def calculate_representative_sequence(alignment, threshold=0.7, seq_type="Protein"):
    """
    Consensus sequence and the alignment member closest to it.

    Columns whose most frequent symbol is a gap are left out of the exported consensus.
    Columns below `threshold` get the ambiguity code (N for DNA, X for protein) and are not
    counted as differences when choosing the most representative sequence.

    Returns:
        tuple: (consensus_record, closest_record, min_differences, closest_seq_id)
    """
    ambiguous = "N" if seq_type == "DNA" else "X"
    seqs = [str(record.seq).upper() for record in alignment]
    n_seqs = len(seqs)

    consensus_aligned = []
    for i in range(alignment.get_alignment_length()):
        char, count = Counter(s[i] for s in seqs).most_common(1)[0]
        consensus_aligned.append(char if count / n_seqs >= threshold else ambiguous)
    consensus_aligned = "".join(consensus_aligned)

    def count_differences(seq):
        return sum(1 for c, s in zip(consensus_aligned, seq) if c != ambiguous and c != s)

    closest_index = min(range(n_seqs), key=lambda k: count_differences(seqs[k]))
    closest_record_src = alignment[closest_index]
    min_differences = count_differences(seqs[closest_index])

    consensus_record = SeqRecord(Seq(consensus_aligned.replace("-", "")), id="Consensus",
                                 description="Consensus sequence")
    closest_record = SeqRecord(Seq(str(closest_record_src.seq).replace("-", "")),
                               id=closest_record_src.id, description="Most representative sequence")
    return consensus_record, closest_record, min_differences, closest_record_src.id
```

Delete the unused `save_msa_to_fasta` and `get_msa_as_fasta`; `grep -rn "save_msa_to_fasta\|get_msa_as_fasta" .` must return nothing outside `msa.py`.

UI edits in `msa_section`:

1. The output format selectbox's options become `MSA_OUTPUT_FORMATS`.
2. Replace the `perform_msa(...)` call inside the spinner with:

```python
                try:
                    msa_result, mutations = perform_msa(sequences, ref_seq_id, seq_type, msa_output_format)
                except MSAError as e:
                    st.error(str(e))
                    msa_result, mutations = "", {}
```

3. Replace the two lines that call `parse_sequences_from_text(st.session_state.msa_result)` and write the totals:

```python
                try:
                    _aln = AlignIO.read(StringIO(st.session_state.msa_result), msa_output_format)
                    st.write(f"Total sequences: {len(_aln)}. MSA length: {_aln.get_alignment_length()}")
                except ValueError:
                    pass
```

4. Change both `mime=f"text/{msa_output_format}"` to `mime="text/plain"`.
5. Change `calculate_representative_sequence(alignment, threshold=consensus_threshold)` to `calculate_representative_sequence(alignment, threshold=consensus_threshold, seq_type=seq_type)`.
6. Remove the now-unused import `from modules.parsers import parse_sequences_from_text`.

In `modules/clustering.py`, change the import to `from modules.msa import MSAError, perform_msa` and wrap the call:

```python
            with st.spinner("Calculating MSA..."):
                try:
                    msa_text, _ = perform_msa(sequences, sequences[0].id, seq_type, "fasta")
                    st.session_state.clustering_msa = msa_text
                except MSAError as e:
                    st.error(str(e))
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_msa.py tests/test_consensus.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/msa.py modules/clustering.py tests/test_msa.py tests/test_consensus.py
git commit -m "fix: clean MSA input, relaxed PHYLIP output, gap-aware consensus

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Clustering science: unbiased identity, average linkage, PCoA, Otsu threshold

**Files:**
- Modify: `modules/clustering.py`
- Modify: `requirements.txt` (scipy is already pinned; no change needed)
- Test: `tests/test_clustering.py`

**Interfaces:**
- Consumes: `clean_sequence`, `get_aligner`, `DEFAULT_*`
- Produces:
  - `msa_identity_matrix(aligned: list[str]) -> np.ndarray`
  - `pairwise_identity(aligned1, aligned2, mode, len1, len2) -> float`
  - `calculate_similarity_matrix_pairwise(sequences, seq_type, align_mode, progress_callback=None) -> (matrix, ids)`
  - `suggest_threshold(similarities) -> float`
  - `hierarchical_clusters(similarity, ids, threshold, method="average") -> (clusters: list[list[str]], leaf_order: list[int])`
  - `perform_pcoa(matrix, n_components=2) -> np.ndarray` (replaces `perform_pca`)
  - Constants `HIERARCHICAL`, `CONNECTED`, `KMEANS`, `CLUSTERING_METHODS`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_clustering.py` inside the class)

```python
    def test_msa_identity_ignores_double_gap_columns(self):
        from modules.clustering import msa_identity_matrix
        m = msa_identity_matrix(["MKV---LAA", "MKV---LAA", "MKVGGGLAA", "MKA------"])
        self.assertEqual(m[0, 1], 1.0)
        self.assertAlmostEqual(m[0, 2], 6 / 9)
        self.assertAlmostEqual(m[0, 3], 2 / 6)

    def test_local_identity_normalized_by_shorter_sequence(self):
        a = SeqRecord(Seq("W" * 8 + "H" * 32), id="a")
        b = SeqRecord(Seq("W" * 8 + "K" * 33), id="b")
        matrix, _ = calculate_similarity_matrix_pairwise([a, b], "Protein", "Local")
        self.assertAlmostEqual(matrix[0, 1], 8 / 40)

    def test_pairwise_matrix_lowercase(self):
        a = SeqRecord(Seq("mkvlaa"), id="a")
        b = SeqRecord(Seq("MKVLAA"), id="b")
        matrix, _ = calculate_similarity_matrix_pairwise([a, b], "Protein", "Global")
        self.assertEqual(matrix[0, 1], 1.0)

    def test_hierarchical_clusters(self):
        from modules.clustering import hierarchical_clusters
        sim = np.array([[1, .95, .2], [.95, 1, .2], [.2, .2, 1]])
        clusters, order = hierarchical_clusters(sim, ["a", "b", "c"], 0.9)
        self.assertEqual(sorted(map(sorted, clusters)), [["a", "b"], ["c"]])
        self.assertEqual(sorted(order), [0, 1, 2])

    def test_suggest_threshold_separates_modes(self):
        from modules.clustering import suggest_threshold
        rng = np.random.default_rng(0)
        values = np.clip(np.r_[rng.normal(0.3, 0.05, 200), rng.normal(0.9, 0.03, 50)], 0, 1)
        t = suggest_threshold(values)
        self.assertTrue(0.35 < t < 0.85, t)
        self.assertEqual(suggest_threshold(np.array([])), 0.8)

    def test_pcoa_preserves_neighbourhoods(self):
        from modules.clustering import perform_pcoa
        coords = perform_pcoa(np.array([[1, .9, .1], [.9, 1, .1], [.1, .1, 1]]))
        self.assertEqual(coords.shape, (3, 2))
        self.assertLess(np.linalg.norm(coords[0] - coords[1]), np.linalg.norm(coords[0] - coords[2]))
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_clustering.py -v`
Expected: FAIL with `ImportError: msa_identity_matrix` and `ValueError: sequence contains letters not in the alphabet`.

- [ ] **Step 3: Implement the compute functions**

Imports at the top of `modules/clustering.py`: remove `from sklearn.decomposition import PCA`; change `from modules.pairwise import get_aligner, calculate_alignment_score` to `from modules.pairwise import get_aligner` (the `MSAError, perform_msa` import from Task 5 stays); then add:

```python
from collections import deque
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from modules.seqtools import clean_sequence

HIERARCHICAL = "Hierarchical (average linkage)"
CONNECTED = "Connected components (single linkage)"
KMEANS = "K-Means on PCoA coordinates"
CLUSTERING_METHODS = (HIERARCHICAL, CONNECTED, KMEANS)
```

Replace `calculate_similarity_matrix_from_msa`, `calculate_similarity_matrix_pairwise`, `perform_pca` and `find_clusters`, and add the new functions:

```python
def msa_identity_matrix(aligned):
    """Pairwise identity from aligned strings: identical residues / columns where either has a residue."""
    n = len(aligned)
    matrix = np.eye(n)
    if n == 0:
        return matrix
    arr = np.frombuffer("".join(s.upper() for s in aligned).encode(), dtype=np.uint8).reshape(n, -1)
    residue = arr != ord('-')
    for i in range(n - 1):
        same = (arr[i] == arr[i + 1:]) & residue[i]
        cols = residue[i] | residue[i + 1:]
        counts = cols.sum(axis=1)
        values = np.divide(same.sum(axis=1), counts, out=np.zeros(len(counts)), where=counts > 0)
        matrix[i, i + 1:] = values
        matrix[i + 1:, i] = values
    return matrix


def calculate_similarity_matrix_from_msa(alignment, sequences):
    """N×N identity matrix from a (sliced) MSA, in input-sequence order."""
    aln_dict = {rec.id: str(rec.seq) for rec in alignment}
    ids = [seq.id for seq in sequences if seq.id in aln_dict]
    return msa_identity_matrix([aln_dict[i] for i in ids]), ids


def pairwise_identity(aligned1, aligned2, mode, len1, len2):
    """Identity of a pairwise alignment.

    Global/overlap: matches / alignment columns. Local: matches / length of the shorter
    sequence, so a short shared motif does not score as identical sequences.
    """
    matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
    if mode == "local":
        denominator = min(len1, len2)
    else:
        denominator = sum(1 for a, b in zip(aligned1, aligned2) if a != '-' or b != '-')
    return matches / denominator if denominator else 0.0


def calculate_similarity_matrix_pairwise(sequences, seq_type, align_mode, progress_callback=None):
    """N×N identity matrix from all-against-all pairwise alignments."""
    mode = align_mode.lower()
    ids = [seq.id for seq in sequences]
    cleaned = [clean_sequence(seq.seq, seq_type) for seq in sequences]
    n = len(sequences)
    matrix = np.eye(n)
    aligner = get_aligner(seq_type, mode)
    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = 0.0
            if cleaned[i] and cleaned[j]:
                try:
                    aln = next(iter(aligner.align(cleaned[i], cleaned[j])))
                    sim = pairwise_identity(str(aln[0]), str(aln[1]), mode, len(cleaned[i]), len(cleaned[j]))
                except StopIteration:
                    sim = 0.0
            matrix[i, j] = matrix[j, i] = sim
            done += 1
            if progress_callback and total_pairs and done % 10 == 0:
                progress_callback(done / total_pairs)
    return matrix, ids


def suggest_threshold(similarities, bins=50):
    """Otsu threshold on the similarity histogram; 0.8 when there is nothing to split."""
    values = np.asarray(similarities, dtype=float)
    if values.size < 2 or np.ptp(values) == 0:
        return float(values.mean()) if values.size else 0.8
    hist, edges = np.histogram(values, bins=bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = w0 * w1 * (m0 / w0 - (m0[-1] - m0) / w1) ** 2
    return float(round(edges[int(np.argmax(np.nan_to_num(between))) + 1], 2))


def hierarchical_clusters(similarity, ids, threshold, method="average"):
    """Cut an average-linkage tree at distance 1 - threshold.

    Returns (clusters as lists of IDs, leaf order of the dendrogram as row indices).
    """
    if len(ids) < 2:
        return [list(ids)], list(range(len(ids)))
    distance = np.clip(1.0 - np.asarray(similarity, dtype=float), 0, None)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method=method)
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")
    groups = {}
    for seq_id, label in zip(ids, labels):
        groups.setdefault(label, []).append(seq_id)
    return list(groups.values()), leaves_list(tree).tolist()


def perform_pcoa(matrix, n_components=2):
    """Classical multidimensional scaling (PCoA) of the distance 1 - similarity."""
    distance = 1.0 - np.asarray(matrix, dtype=float)
    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * centering @ (distance ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, order] * np.sqrt(np.clip(eigenvalues[order], 0, None))


def find_clusters(similarity_matrix, ids, threshold):
    """Connected components of the graph with edges where similarity >= threshold (single linkage)."""
    n = len(ids)
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        component = []
        queue = deque([i])
        visited[i] = True
        while queue:
            curr = queue.popleft()
            component.append(ids[curr])
            for j in range(n):
                if not visited[j] and similarity_matrix[curr, j] >= threshold:
                    visited[j] = True
                    queue.append(j)
        clusters.append(component)
    return clusters
```

- [ ] **Step 4: Update the UI in `sequence_clustering_section`**

Replace the pairwise-matrix call in the "Accurate Pairwise" branch:

```python
        if st.button("Calculate Pairwise Similarity"):
            with st.spinner("Calculating pairwise similarity matrix (this may take time)..."):
                progress_bar = st.progress(0.0)
                matrix, sorted_ids = calculate_similarity_matrix_pairwise(
                    sequences, seq_type, align_mode, progress_callback=progress_bar.progress)
                progress_bar.empty()
                st.session_state.clustering_matrix = matrix
                st.session_state.clustering_ids = sorted_ids
```

Replace the block from `        # Choose Clustering Method` through `                clusters = list(cluster_map.values())`:

```python
        clustering_method = st.radio(
            "Clustering Method",
            CLUSTERING_METHODS,
            horizontal=True,
            help="Average linkage avoids the chaining effect of single linkage. "
                 "K-Means runs on classical MDS (PCoA) coordinates of the distance matrix."
        )

        clusters = []
        threshold = None
        similarities = matrix[np.triu_indices_from(matrix, k=1)]

        if clustering_method in (HIERARCHICAL, CONNECTED):
            threshold = st.slider(
                "Similarity Threshold for Clustering",
                0.0, 1.0, suggest_threshold(similarities), 0.01,
                format="%.2f",
                help="Sequences are grouped when their similarity is at or above the threshold. "
                     "The default comes from Otsu's method on the similarity histogram."
            )
            if clustering_method == HIERARCHICAL:
                clusters, _ = hierarchical_clusters(matrix, ids, threshold)
            else:
                clusters = find_clusters(matrix, ids, threshold)
        elif len(ids) < 3:
            st.warning("Need at least 3 sequences for K-Means clustering.")
            clusters = [list(ids)]
        else:
            coords = perform_pcoa(matrix)
            col_k1, col_k2 = st.columns([3, 2])
            if 'pca_k_val' not in st.session_state:
                st.session_state.pca_k_val = min(3, len(ids) - 1)
            st.session_state.pca_k_val = min(max(2, st.session_state.pca_k_val), max(2, len(ids) - 1))
            with col_k1:
                n_clusters = st.number_input(
                    "Number of Clusters (k)", min_value=2, max_value=max(2, len(ids) - 1), key='pca_k_val'
                )
            with col_k2:
                st.write("")
                st.write("")
                if st.button("Guess Best Number"):
                    best_k, best_score = 2, -1
                    for k in range(2, min(len(ids), 10)):
                        labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(coords)
                        if len(set(labels)) > 1:
                            score = silhouette_score(coords, labels)
                            if score > best_score:
                                best_k, best_score = k, score
                    st.session_state.pca_k_val = best_k
                    st.rerun()
            labels = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit_predict(coords)
            cluster_map = {}
            for idx, label in enumerate(labels):
                cluster_map.setdefault(label, []).append(ids[idx])
            clusters = list(cluster_map.values())
```

In the first tab, replace the PCA block (from `st.markdown("### PCA of Sequences")` to the `st.plotly_chart(fig_pca, ...)` line):

```python
            st.markdown("### PCoA of Sequences")
            if len(ids) >= 3:
                df_pca = pd.DataFrame(perform_pcoa(matrix), columns=['PCoA 1', 'PCoA 2'])
                df_pca['Sequence ID'] = ids
                df_pca['Cluster'] = [id_to_cluster.get(x, "Unknown") for x in ids]
                title_text = f"PCoA of Sequences ({clustering_method})"
                if threshold is not None:
                    title_text += f", threshold {threshold:.2f}"
                fig_pca = px.scatter(df_pca, x='PCoA 1', y='PCoA 2', color='Cluster',
                                     hover_data=['Sequence ID'], title=title_text)
                fig_pca.update_traces(marker=dict(size=12))
                fig_pca.update_yaxes(scaleanchor="x", scaleratio=1)
                fig_pca.update_layout(width=700, height=700)
                st.plotly_chart(fig_pca, width="content")
```

In the histogram tab, change `if clustering_method == "Similarity Threshold" and threshold is not None:` to `if threshold is not None:`.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_clustering.py -v`
Expected: PASS (all 8).

- [ ] **Step 6: Commit**

```bash
git add modules/clustering.py tests/test_clustering.py
git commit -m "fix: unbiased identity metrics, average-linkage clustering, PCoA and Otsu threshold

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Trees: corrected distances, Neighbor-Joining, cleaning, figure lifecycle

**Files:**
- Modify: `modules/phylogeny.py`
- Modify: `modules/msa.py` (tree tab call), `app.py` (tree call)
- Test: `tests/test_phylogeny.py`

**Interfaces:**
- Consumes: `clean_sequence`, `get_aligner`
- Produces:
  - `TreeError(RuntimeError)`, `TREE_METHODS = {"Neighbor-Joining": "nj", "UPGMA": "upgma"}`, `MAX_DISTANCE = 10.0`
  - `p_distance(aligned1, aligned2) -> float | None`
  - `correct_distance(p, seq_type) -> float` (Jukes–Cantor for DNA, Kimura for protein)
  - `alignment_distance_matrix(aligned: list[str], seq_type, correction=True) -> np.ndarray`
  - `compute_distance(seq1, seq2, seq_type, correction=True, aligner=None) -> float`
  - `build_phylogenetic_tree(sequences, seq_type, method="nj", correction=True) -> Tree` (raises `TreeError`)
  - `build_tree_from_alignment(alignment, seq_type, method="nj", correction=True) -> Tree` (raises `TreeError`)

- [ ] **Step 1: Write the failing tests**

`tests/test_phylogeny.py`:

```python
import math

import pytest
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.phylogeny import (
    MAX_DISTANCE, alignment_distance_matrix, build_phylogenetic_tree, build_tree_from_alignment,
    compute_distance, correct_distance, p_distance,
)


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def test_p_distance_ignores_gaps():
    assert p_distance("AC-T", "AGGT") == pytest.approx(1 / 3)
    assert p_distance("--", "AA") is None


def test_jukes_cantor_and_kimura():
    assert correct_distance(0.0, "DNA") == 0.0
    assert correct_distance(0.1, "DNA") == pytest.approx(-0.75 * math.log(1 - 0.4 / 3))
    assert correct_distance(0.8, "DNA") == MAX_DISTANCE
    assert correct_distance(0.1, "Protein") == pytest.approx(-math.log(1 - 0.1 - 0.2 * 0.01))


def test_alignment_distance_matrix_symmetric():
    d = alignment_distance_matrix(["ACGT", "ACGA", "TCGA"], "DNA", correction=False)
    assert d[0, 1] == pytest.approx(0.25)
    assert d[1, 0] == d[0, 1]
    assert d[0, 0] == 0.0


def test_tree_lowercase_input():
    assert compute_distance(Seq("mkvlaa"), Seq("MKVLAA"), "Protein") == 0.0


def test_nj_tree_has_all_leaves():
    seqs = [rec("a", "MKVLAAGIVG"), rec("b", "MKVLAAGIVA"), rec("c", "MRVLSAGLVA"), rec("d", "MRILSAGLVA")]
    tree = build_phylogenetic_tree(seqs, "Protein", method="nj")
    assert sorted(t.name for t in tree.get_terminals()) == ["a", "b", "c", "d"]


def test_tree_from_alignment_upgma():
    aln = MultipleSeqAlignment([rec("a", "ACGT-A"), rec("b", "ACGTTA"), rec("c", "TCGTTA")])
    tree = build_tree_from_alignment(aln, "DNA", method="upgma")
    assert sorted(t.name for t in tree.get_terminals()) == ["a", "b", "c"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_phylogeny.py -v`
Expected: FAIL with `ImportError: cannot import name 'MAX_DISTANCE'`.

- [ ] **Step 3: Implement**

`modules/phylogeny.py`: replace the imports and every function except `phylogenetic_tree_section` with:

```python
import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from io import BytesIO, StringIO

from modules.pairwise import get_aligner
from modules.seqtools import clean_sequence

TREE_METHODS = {"Neighbor-Joining": "nj", "UPGMA": "upgma"}
MAX_DISTANCE = 10.0


class TreeError(RuntimeError):
    """Raised when a tree cannot be built."""


def p_distance(aligned1, aligned2):
    """Fraction of differing residues over columns where neither sequence has a gap."""
    valid = diffs = 0
    for a, b in zip(aligned1, aligned2):
        if a != '-' and b != '-':
            valid += 1
            diffs += a != b
    return diffs / valid if valid else None


def correct_distance(p, seq_type):
    """Jukes–Cantor (DNA) or Kimura (protein) correction for multiple substitutions, capped."""
    if seq_type == "DNA":
        x = 1 - 4 * p / 3
        return MAX_DISTANCE if x <= 0 else min(MAX_DISTANCE, -0.75 * math.log(x))
    x = 1 - p - 0.2 * p * p
    return MAX_DISTANCE if x <= 0 else min(MAX_DISTANCE, -math.log(x))


def _finish(p, seq_type, correction):
    if p is None:
        return MAX_DISTANCE if correction else 1.0
    return correct_distance(p, seq_type) if correction else p


def alignment_distance_matrix(aligned, seq_type, correction=True):
    """Symmetric distance matrix from aligned strings (pairwise deletion of gap columns)."""
    n = len(aligned)
    matrix = np.zeros((n, n))
    upper = [s.upper() for s in aligned]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = _finish(p_distance(upper[i], upper[j]), seq_type, correction)
    return matrix


def compute_distance(seq1, seq2, seq_type, correction=True, aligner=None):
    """Distance between two unaligned sequences via global pairwise alignment."""
    s1 = clean_sequence(seq1, seq_type)
    s2 = clean_sequence(seq2, seq_type)
    if not s1 or not s2:
        return _finish(None, seq_type, correction)
    aligner = aligner or get_aligner(seq_type, mode="global")
    try:
        alignment = next(iter(aligner.align(s1, s2)))
    except StopIteration:
        return _finish(None, seq_type, correction)
    return _finish(p_distance(str(alignment[0]), str(alignment[1])), seq_type, correction)


def _tree_from_matrix(names, matrix, method):
    lower = [[float(matrix[i][j]) for j in range(i + 1)] for i in range(len(names))]
    constructor = DistanceTreeConstructor()
    dm = DistanceMatrix(names=list(names), matrix=lower)
    tree = constructor.nj(dm) if method == "nj" else constructor.upgma(dm)
    if method == "nj" and len(names) > 2:
        tree.root_at_midpoint()
    return tree


def build_tree_from_alignment(alignment, seq_type, method="nj", correction=True):
    """Tree from an MSA (Biopython MultipleSeqAlignment)."""
    try:
        names = [record.id for record in alignment]
        matrix = alignment_distance_matrix([str(r.seq) for r in alignment], seq_type, correction)
        return _tree_from_matrix(names, matrix, method)
    except Exception as e:
        raise TreeError(f"Tree construction from MSA failed: {e}") from e


def build_phylogenetic_tree(sequences, seq_type, method="nj", correction=True):
    """Tree from unaligned SeqRecords using pairwise global alignments."""
    try:
        aligner = get_aligner(seq_type, mode="global")
        n = len(sequences)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = matrix[j, i] = compute_distance(
                    sequences[i].seq, sequences[j].seq, seq_type, correction, aligner)
        return _tree_from_matrix([s.id for s in sequences], matrix, method)
    except Exception as e:
        raise TreeError(f"Tree construction failed: {e}") from e
```

In `phylogenetic_tree_section`, replace everything from `# Update font size` through the `except Exception as e:` handler's `st.code(...)`, so the figure is always closed and the font setting no longer leaks into other sessions:

```python
    with plt.rc_context({'font.size': font_size}):
        fig, ax = plt.subplots(figsize=(tree_width, tree_height))
    try:
        def label_func(node):
            if node.is_terminal():
                return str(node) if show_labels else ''
            return str(node) if show_inner_labels else ''

        with plt.rc_context({'font.size': font_size}):
            Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False, label_func=label_func)
        ax.set_title("Phylogenetic Tree", fontsize=16)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.get_yaxis().set_visible(False)
        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        buf.seek(0)
        st.download_button(label="📥 Download Tree Image (PNG)", data=buf,
                           file_name="phylogenetic_tree.png", mime="image/png")
    except Exception as e:
        st.error(f"Could not draw the tree graphically: {e}")
        st.info("Displaying ASCII version of the tree instead.")
        tree_io = StringIO()
        Phylo.draw_ascii(tree, out=tree_io)
        st.code(tree_io.getvalue())
    finally:
        plt.close(fig)
```

In `modules/msa.py`, replace the tree tab body:

```python
            with tabs[-1]:
                st.subheader("🌳 Phylogenetic Tree from MSA")
                c_m, c_c = st.columns(2)
                tree_method = c_m.selectbox("Tree method", list(TREE_METHODS), key="msa_tree_method")
                correction = c_c.checkbox("Correct for multiple substitutions", value=True, key="msa_tree_corr",
                                          help="Jukes–Cantor (DNA) or Kimura (protein) distance correction.")
                if st.button("Calculate Phylogenetic Tree from MSA"):
                    with st.spinner("Building tree from MSA..."):
                        alignment = AlignIO.read(StringIO(st.session_state.msa_result), msa_output_format)
                        try:
                            st.session_state.msa_tree = build_tree_from_alignment(
                                alignment, seq_type, TREE_METHODS[tree_method], correction)
                        except TreeError as e:
                            st.error(str(e))

                if st.session_state.get('msa_tree'):
                    phylogenetic_tree_section(st.session_state.msa_tree)
```

Change its import to `from modules.phylogeny import TREE_METHODS, TreeError, build_tree_from_alignment, phylogenetic_tree_section`.

In `app.py`, change the import to `from modules.phylogeny import TREE_METHODS, TreeError, build_phylogenetic_tree, phylogenetic_tree_section` and replace the `elif sequences:` branch of the Phylogenetic Tree mode:

```python
            elif sequences:
                c_m, c_c = st.columns(2)
                tree_method = c_m.selectbox("Tree method", list(TREE_METHODS))
                correction = c_c.checkbox("Correct for multiple substitutions", value=True)
                with st.spinner("Building phylogenetic tree..."):
                    try:
                        new_tree = build_phylogenetic_tree(sequences, seq_type, TREE_METHODS[tree_method], correction)
                    except TreeError as e:
                        st.error(str(e))
                        new_tree = None
                if new_tree:
                    st.session_state.tree = new_tree
                    if 'tree_newick' in st.session_state: del st.session_state.tree_newick
                    phylogenetic_tree_section(new_tree)
```

(Plan 02 Task 4 replaces this rebuild-every-rerun flow with an explicit button.)

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_phylogeny.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add modules/phylogeny.py modules/msa.py app.py tests/test_phylogeny.py
git commit -m "fix: NJ trees with corrected distances, clean input, close figures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: ORF finder: one ORF per stop, nucleotide coordinates, RNA and codon tables

**Files:**
- Modify: `modules/dna.py`
- Test: `tests/test_translation.py`

**Interfaces:**
- Consumes: `clean_sequence`
- Produces:
  - `CODON_TABLES = {"1 Standard": 1, "2 Vertebrate mitochondrial": 2, "4 Mold/Mycoplasma": 4, "5 Invertebrate mitochondrial": 5, "11 Bacterial/Plastid": 11}`
  - `find_orfs(sequence, min_len=30, table=1) -> list[dict]`. Keys: `Frame`, `Strand`, `Start (nt)`, `End (nt)`, `Length (AA)`, `Sequence`. Coordinates are 1-based and inclusive on the input strand; End includes the stop codon.

- [ ] **Step 1: Write the failing tests** (append to `TestTranslation`)

```python
    def test_nested_methionines_give_one_orf(self):
        seq = "ATG" + "GCC" * 5 + "ATG" + "GCC" * 40 + "TAA"
        orfs = find_orfs(seq, min_len=10)
        self.assertEqual(len(orfs), 1)
        self.assertEqual(orfs[0]["Length (AA)"], 47)

    def test_nucleotide_coordinates_forward_and_reverse(self):
        fwd = find_orfs("ATGCCCTGA", min_len=1)[0]
        self.assertEqual((fwd["Start (nt)"], fwd["End (nt)"], fwd["Strand"]), (1, 9, "+"))
        rev = find_orfs("GG" + "TCAGGGCAT", min_len=1)[0]
        self.assertEqual((rev["Start (nt)"], rev["End (nt)"], rev["Strand"]), (3, 11, "-"))

    def test_orf_rna_input(self):
        orfs = find_orfs("AUGCCCUGA", min_len=1)
        self.assertEqual(orfs[0]["Sequence"], "MP")

    def test_codon_table(self):
        self.assertEqual(find_orfs("ATGAGATAA", min_len=1, table=1)[0]["Sequence"], "MR")
        self.assertEqual(find_orfs("ATGAGATAA", min_len=1, table=2)[0]["Sequence"], "M")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_translation.py -v`
Expected: FAIL (nested ORF count 2; `KeyError: 'Start (nt)'`; table argument unknown).

- [ ] **Step 3: Implement**

Replace `find_orfs` in `modules/dna.py`, and add `import re`, `from modules.seqtools import clean_sequence` and the table constant:

```python
CODON_TABLES = {
    "1 Standard": 1,
    "2 Vertebrate mitochondrial": 2,
    "4 Mold/Mycoplasma": 4,
    "5 Invertebrate mitochondrial": 5,
    "11 Bacterial/Plastid": 11,
}


def find_orfs(sequence, min_len=30, table=1):
    """
    Find ORFs (first ATG to stop) in all six frames; one ORF per stop codon.

    Coordinates are 1-based, inclusive, on the input strand; End includes the stop codon.
    """
    seq = clean_sequence(sequence, "DNA")
    length = len(seq)
    orfs = []
    for strand, strand_seq in (("+", Seq(seq)), ("-", Seq(seq).reverse_complement())):
        for offset in range(3):
            frame_seq = strand_seq[offset:]
            frame_seq = frame_seq[:len(frame_seq) - len(frame_seq) % 3]
            if not frame_seq:
                continue
            protein = str(frame_seq.translate(table=table))
            segment_start = 0
            for match in re.finditer(r"\*", protein):
                stop = match.start()
                met = protein.find("M", segment_start, stop)
                if met != -1 and stop - met >= min_len:
                    start_nt = offset + 3 * met + 1
                    end_nt = offset + 3 * (stop + 1)
                    if strand == "-":
                        start_nt, end_nt = length - end_nt + 1, length - start_nt + 1
                    orfs.append({
                        "Frame": f"{strand}{offset + 1}",
                        "Strand": strand,
                        "Start (nt)": start_nt,
                        "End (nt)": end_nt,
                        "Length (AA)": stop - met,
                        "Sequence": protein[met:stop],
                    })
                segment_start = stop + 1
    return orfs
```

UI changes in `dna_translation_section`:
- Add a selectbox before the translation: `table_label = st.selectbox("Codon table", list(CODON_TABLES))` and `table = CODON_TABLES[table_label]`.
- Replace the cleaning block (`dna_seq = ...` through `clean_seq = ''.join(...)`) with `clean_seq = clean_sequence(selected_seq.seq, "DNA")`, and translate with `Seq(clean_seq_tr).translate(table=table)`.
- Call `find_orfs(str(selected_seq.seq), min_len=min_len, table=table)`.
- In the expander title, change `f"Frame {row['Frame']}, Start {row['Start (AA)']}, Length {row['Length (AA)']}"` to `f"Frame {row['Frame']}, {row['Start (nt)']}–{row['End (nt)']} nt, {row['Length (AA)']} aa"`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_translation.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add modules/dna.py tests/test_translation.py
git commit -m "fix: ORF finder reports one ORF per stop with nucleotide coordinates; RNA and codon tables

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Antibody regions for IMGT, Kabat, Chothia and Martin; species default; persistent results

**Files:**
- Rewrite: `modules/antibody.py`
- Test: `tests/test_antibody.py`

**Interfaces:**
- Consumes: vendored `anarci.run_anarci`
- Produces:
  - `ALL_SPECIES = ["human", "mouse", "rat", "rabbit", "rhesus", "pig", "alpaca"]`
  - `REGION_ORDER = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]`
  - `REGION_DEFINITIONS: dict[scheme, dict["H"|"L", list[tuple[region, start, end]]]]`
  - `extract_regions(numbering, scheme: str, chain_type: str) -> dict[str, str]` (empty dict if the scheme has no definition)
  - `extract_cdr_imgt(numbering)`, kept as a wrapper
  - `run_numbering(pairs, scheme, species, threshold) -> pandas.DataFrame`
  - Session key `antibody_results = {"params": dict, "df": DataFrame}`

- [ ] **Step 1: Write the failing tests**

`tests/test_antibody.py`:

```python
import shutil

import pytest

from modules.antibody import extract_regions, run_numbering

VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQ"
      "MNSLRAEDTAVYYCAKDRGYSSGWYFDYWGQGTLVTVSS")
VL = ("DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFA"
      "TYYCQQSYSTPLTFGQGTKVEIK")

needs_hmmer = pytest.mark.skipif(shutil.which("hmmscan") is None, reason="HMMER not installed")


def test_extract_regions_unknown_scheme_returns_empty():
    assert extract_regions([((1, ' '), 'E')], "aho", "H") == {}


@needs_hmmer
@pytest.mark.parametrize("scheme,expected_h,expected_l", [
    ("IMGT", ("GFTFSSYA", "ISGSGGST", "AKDRGYSSGWYFDY"), ("QSISSY", "AAS", "QQSYSTPLT")),
    ("Kabat", ("SYAMS", "AISGSGGSTYYADSVKG", "DRGYSSGWYFDY"), ("RASQSISSYLN", "AASSLQS", "QQSYSTPLT")),
    ("Chothia", ("GFTFSSY", "SGSGGS", "DRGYSSGWYFDY"), ("RASQSISSYLN", "AASSLQS", "QQSYSTPLT")),
])
def test_cdrs_match_reference_definitions(scheme, expected_h, expected_l):
    df = run_numbering([("vh", VH), ("vl", VL)], scheme, species=None, threshold=80)
    vh = df[df["Sequence ID"] == "vh"].iloc[0]
    vl = df[df["Sequence ID"] == "vl"].iloc[0]
    assert (vh["CDR1"], vh["CDR2"], vh["CDR3"]) == expected_h
    assert (vl["CDR1"], vl["CDR2"], vl["CDR3"]) == expected_l
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_antibody.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_regions'`.

- [ ] **Step 3: Implement**

`modules/antibody.py` (full replacement):

```python
import traceback

import anarci
import pandas as pd
import streamlit as st

ALL_SPECIES = ["human", "mouse", "rat", "rabbit", "rhesus", "pig", "alpaca"]
SCHEMES = ["IMGT", "Chothia", "Kabat", "Martin", "AHo", "Wolfguy"]
REGION_ORDER = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]

_IMGT = [("FR1", 1, 26), ("CDR1", 27, 38), ("FR2", 39, 55), ("CDR2", 56, 65),
         ("FR3", 66, 104), ("CDR3", 105, 117), ("FR4", 118, 128)]
_KABAT_H = [("FR1", 1, 30), ("CDR1", 31, 35), ("FR2", 36, 49), ("CDR2", 50, 65),
            ("FR3", 66, 94), ("CDR3", 95, 102), ("FR4", 103, 113)]
_KABAT_L = [("FR1", 1, 23), ("CDR1", 24, 34), ("FR2", 35, 49), ("CDR2", 50, 56),
            ("FR3", 57, 88), ("CDR3", 89, 97), ("FR4", 98, 109)]
_CHOTHIA_H = [("FR1", 1, 25), ("CDR1", 26, 32), ("FR2", 33, 51), ("CDR2", 52, 56),
              ("FR3", 57, 94), ("CDR3", 95, 102), ("FR4", 103, 113)]

# Region boundaries are inclusive on the integer position; insertion codes (e.g. 35A, 100B)
# belong to their integer position. Martin numbering uses Chothia CDR definitions.
REGION_DEFINITIONS = {
    "imgt": {"H": _IMGT, "L": _IMGT},
    "kabat": {"H": _KABAT_H, "L": _KABAT_L},
    "chothia": {"H": _CHOTHIA_H, "L": _KABAT_L},
    "martin": {"H": _CHOTHIA_H, "L": _KABAT_L},
}
REGION_COLORS = {"FR1": "#e0e0e0", "CDR1": "#ffcccc", "FR2": "#e0e0e0", "CDR2": "#ccffcc",
                 "FR3": "#e0e0e0", "CDR3": "#ccccff", "FR4": "#e0e0e0"}


def extract_regions(numbering, scheme, chain_type):
    """Split an ANARCI numbering into framework and CDR sequences for supported schemes."""
    definitions = REGION_DEFINITIONS.get(scheme.lower())
    if not definitions:
        return {}
    if scheme.lower() == "imgt":
        bounds = definitions["H"]
    elif chain_type == "H":
        bounds = definitions["H"]
    elif chain_type in ("K", "L"):
        bounds = definitions["L"]
    else:
        return {}
    regions = {name: [] for name in REGION_ORDER}
    for (pos, _ins), aa in numbering:
        if aa == '-':
            continue
        for name, start, end in bounds:
            if start <= pos <= end:
                regions[name].append(aa)
                break
    return {k: "".join(v) for k, v in regions.items()}


def extract_cdr_imgt(numbering):
    """Backward-compatible IMGT region extraction."""
    return extract_regions(numbering, "imgt", "H")


def run_numbering(pairs, scheme, species, threshold):
    """Number (id, sequence) pairs with ANARCI and return one row per detected domain."""
    _, numbered, alignment_details, _ = anarci.run_anarci(
        list(pairs),
        scheme=scheme.lower(),
        allowed_species=species or None,
        bit_score_threshold=threshold,
        output=False,
    )
    rows = []
    for i, seq_res in enumerate(numbered):
        if not seq_res:
            continue
        for j, (numbering, start, end) in enumerate(seq_res):
            details = alignment_details[i][j]
            row = {
                "Sequence ID": pairs[i][0],
                "Domain Index": j + 1,
                "Chain Type": details.get('chain_type', 'Unknown'),
                "Species": details.get('species', 'Unknown'),
                "E-value": float(details.get('evalue', 0)),
                "Score": float(details.get('bitscore', 0)),
                "Start": start,
                "End": end,
            }
            row.update(extract_regions(numbering, scheme, row["Chain Type"]))
            row["Numbering"] = " ".join(f"{pos}{ins.strip()}{aa}" for (pos, ins), aa in numbering if aa != '-')
            row["Sequence"] = "".join(aa for _, aa in numbering if aa != '-')
            rows.append(row)
    return pd.DataFrame(rows)


def _regions_html(row):
    parts = []
    for region in REGION_ORDER:
        segment = row.get(region, "")
        if isinstance(segment, str) and segment:
            parts.append(
                f"<span style='background-color: {REGION_COLORS[region]}; padding: 2px; border-radius: 3px; "
                f"margin-right: 2px;' title='{region}'><strong>{region}:</strong> {segment}</span>"
            )
    return " ".join(parts)


def antibody_prediction_section(sequences):
    st.header("🛡️ Antibody Functional Domain Prediction")
    st.info("Predict functional domains (CDRs, Framework Regions) of antibodies using ANARCI.")

    if not sequences:
        st.error("No sequences available for prediction.")
        return

    col1, col2 = st.columns(2)
    with col1:
        scheme = st.selectbox("Numbering Scheme", SCHEMES, index=0)
    with col2:
        species = st.multiselect("Allowed Species", ALL_SPECIES, default=ALL_SPECIES,
                                 help="All species by default, so camelid VHH are not dropped.")
    threshold = st.slider("Bit Score Threshold", 0, 200, 80, help="Minimum score for HMM alignment")

    params = {"ids": tuple(s.id for s in sequences), "seqs": tuple(str(s.seq) for s in sequences),
              "scheme": scheme, "species": tuple(species), "threshold": threshold}

    if st.button("Predict Domains"):
        with st.spinner("Running ANARCI..."):
            try:
                df = run_numbering([(s.id, str(s.seq)) for s in sequences], scheme,
                                   None if set(species) == set(ALL_SPECIES) else species, threshold)
                st.session_state.antibody_results = {"params": params, "df": df}
            except Exception as e:
                st.session_state.antibody_results = None
                st.error(f"Error running ANARCI: {e}")
                st.text(traceback.format_exc())
                return

    stored = st.session_state.get("antibody_results")
    if not stored:
        return
    if stored["params"] != params:
        st.info("Settings changed since the last run. Click **Predict Domains** to update the results.")
        return

    df = stored["df"]
    if df.empty:
        st.warning("No domains found with the current settings.")
        return

    st.success(f"Found {len(df)} domains.")
    has_regions = "CDR1" in df.columns
    if has_regions:
        cols = ["Sequence ID", "Chain Type", "Species", "CDR1", "CDR2", "CDR3", "Score", "E-value", "Sequence"]
        st.dataframe(df[[c for c in cols if c in df.columns]])
    else:
        st.caption(f"CDR definitions are not available for the {scheme} scheme; showing numbering only.")
        st.dataframe(df)

    st.download_button("Download CSV", df.to_csv(index=False), "antibody_domains.csv", "text/csv")

    st.subheader("Detailed Domain View")
    for _, row in df.iterrows():
        with st.expander(f"{row['Sequence ID']} - Domain {row['Domain Index']} ({row['Chain Type']})"):
            st.json({k: (v if not hasattr(v, "item") else v.item()) for k, v in row.to_dict().items()})
            if has_regions:
                st.markdown(f"### Regions ({scheme})")
                st.markdown(_regions_html(row), unsafe_allow_html=True)
```

Add `"antibody_results": None,` to the defaults in `modules/utils.py` `init_session_state`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_antibody.py -v`
Expected: PASS (4 tests: 1 plain, 3 parametrised; the HMMER tests are skipped only if `hmmscan` is missing).

- [ ] **Step 5: Commit**

```bash
git add modules/antibody.py modules/utils.py tests/test_antibody.py
git commit -m "fix: antibody CDRs for IMGT/Kabat/Chothia/Martin, all species by default, persistent results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Structure parsing: models, ligands, unique IDs, no Streamlit in the parser

**Files:**
- Modify: `modules/parsers.py`
- Modify: `app.py` (PDB/mmCIF branch of `handle_input`)
- Test: `tests/test_parsers.py`

**Interfaces:**
- Consumes: `amino_acid_map` from `modules.utils`
- Produces:
  - `polymer_residues(chain) -> list[Residue]`: standard residues plus modified amino acids in `amino_acid_map` (e.g. `MSE`); never ligands or water
  - `parse_sequences_from_structure(file, format_name, all_models=False) -> (records | None, error | None)`, pure
  - `make_unique_ids(records) -> list[SeqRecord]`: suffixes `_2`, `_3`, … on repeats

- [ ] **Step 1: Write the failing tests**

`tests/test_parsers.py`:

```python
import io

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.parsers import make_unique_ids, parse_sequences_from_structure


def pdb_lines(residues, chain="A"):
    """residues: list of (record, resname, number, icode)."""
    lines = []
    for serial, (record, resname, num, icode) in enumerate(residues, 1):
        lines.append(f"{record:<6}{serial:>5}  CA  {resname:>3} {chain}{num:>4}{icode:1}   "
                     f"{serial * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    return "\n".join(lines) + "\n"


class Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


PEPTIDE = [("ATOM", r, i, " ") for i, r in enumerate(["MET", "LYS", "VAL", "LEU", "ALA", "GLY"], 1)]


def test_nmr_first_model_only():
    text = "".join(f"MODEL        {m}\n" + pdb_lines(PEPTIDE) + "ENDMDL\n" for m in (1, 2)) + "END\n"
    records, error = parse_sequences_from_structure(Upload(text.encode(), "nmr.pdb"), "PDB")
    assert error is None
    assert [r.id for r in records] == ["nmr_A"]
    assert str(records[0].seq) == "MKVLAG"


def test_nmr_all_models():
    text = "".join(f"MODEL        {m}\n" + pdb_lines(PEPTIDE) + "ENDMDL\n" for m in (1, 2)) + "END\n"
    records, error = parse_sequences_from_structure(Upload(text.encode(), "nmr.pdb"), "PDB", all_models=True)
    assert error is None
    assert [r.id for r in records] == ["nmr_m1_A", "nmr_m2_A"]


def test_ligands_excluded_and_mse_included():
    residues = PEPTIDE + [("HETATM", "MSE", 7, " "), ("HETATM", "NAG", 101, " "), ("HETATM", "HOH", 201, " ")]
    records, error = parse_sequences_from_structure(Upload((pdb_lines(residues) + "END\n").encode(), "x.pdb"), "PDB")
    assert error is None
    assert str(records[0].seq) == "MKVLAGM"


def test_make_unique_ids():
    recs = [SeqRecord(Seq("A"), id="x_A"), SeqRecord(Seq("C"), id="x_A"), SeqRecord(Seq("D"), id="y_A")]
    assert [r.id for r in make_unique_ids(recs)] == ["x_A", "x_A_2", "y_A"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_parsers.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_unique_ids'`.

- [ ] **Step 3: Implement**

In `modules/parsers.py`, remove the `tempfile`, `traceback` and `PPBuilder` imports, and remove `import streamlit as st` if nothing else uses it. Replace `parse_sequences_from_structure` and add the helpers:

```python
def polymer_residues(chain):
    """Residues of the polymer: standard residues plus modified amino acids such as MSE.

    Waters and ligands (e.g. NAG, HEM) are excluded.
    """
    return [res for res in chain
            if res.id[0] == " " or (res.id[0].startswith("H_") and res.resname.strip() in amino_acid_map)]


def make_unique_ids(records):
    """Return records with repeated IDs suffixed _2, _3, ... (first occurrence unchanged)."""
    seen = {}
    unique = []
    for record in records:
        count = seen.get(record.id, 0) + 1
        seen[record.id] = count
        if count > 1:
            record = SeqRecord(record.seq, id=f"{record.id}_{count}", name=record.name,
                               description=record.description, annotations=dict(record.annotations))
        unique.append(record)
    return unique


def parse_sequences_from_structure(file, format_name, all_models=False):
    """
    Extract one protein sequence per chain from a PDB or mmCIF upload.

    Only the first model is used unless all_models is True (NMR ensembles).
    Returns (records, None) or (None, error message).
    """
    try:
        file.seek(0)
        raw = file.read()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        base = os.path.splitext(os.path.basename(file.name))[0]
        if format_name == "PDB":
            parser = PDBParser(QUIET=True)
        elif format_name == "mmCIF":
            parser = MMCIFParser(QUIET=True)
        else:
            return None, f"Unsupported format: {format_name}"
        structure = parser.get_structure(base, StringIO(text))

        models = list(structure) if all_models else list(structure)[:1]
        multi = len(models) > 1
        sequences = []
        for model_idx, model in enumerate(models, 1):
            for chain in model:
                residues = polymer_residues(chain)
                letters = [amino_acid_map.get(res.resname.strip(), "X") for res in residues]
                if not letters or all(letter == "X" for letter in letters):
                    continue  # empty, nucleic acid, or unknown polymer
                chain_label = chain.id.strip() or "_"
                seq_id = f"{base}_m{model_idx}_{chain_label}" if multi else f"{base}_{chain_label}"
                description = f"Chain {chain_label} from {file.name}" + (f", model {model_idx}" if multi else "")
                sequences.append(SeqRecord(Seq("".join(letters)), id=seq_id, description=description))

        if not sequences:
            return None, "No protein sequences could be extracted from the structure file."
        return make_unique_ids(sequences), None
    except Exception as e:
        return None, f"An error occurred while parsing the {format_name} file: {e}"
```

In `app.py`, PDB/mmCIF branch of `handle_input`: add a checkbox before the uploader, pass it through, and de-duplicate across files:

```python
        all_models = st.sidebar.checkbox("Extract all models (NMR ensembles)", value=False)
```

Change the parse call to `parse_sequences_from_structure(uploaded_file, input_format, all_models=all_models)`, and replace `st.session_state.sequences = all_sequences` with `st.session_state.sequences = make_unique_ids(all_sequences)`. Import `make_unique_ids` from `modules.parsers`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_parsers.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add modules/parsers.py app.py tests/test_parsers.py
git commit -m "fix: structure parsing handles NMR models, ligands and duplicate IDs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: PDB renumbering: no duplicate IDs, insertion codes, ligands, models, header

**Files:**
- Rewrite: `modules/pdb_renumber.py`
- Test: `tests/test_pdb_renumber.py`

**Interfaces:**
- Consumes: `polymer_residues`, `amino_acid_map`, `get_aligner`, `clean_sequence`
- Produces:
  - `align_target_to_chain(target_seq: str, chain_seq: str) -> tuple[str, str, float]` (semi-global, BLOSUM62; returns aligned target, aligned chain, identity)
  - `get_mapping_from_alignment(aligned_target, aligned_chain) -> dict[int, int]` (unchanged)
  - `plan_residue_numbers(n_residues: int, mapping: dict[int, int]) -> list[tuple[int, str]]`
  - `prepare_renumbering(structure, target_sequences) -> (mappings, logs, visuals)`: same shape as before; first model only
  - `apply_renumbering(structure, mappings) -> Structure`: applied to every model
  - `extract_header_records(pdb_text: str) -> str`

- [ ] **Step 1: Write the failing tests** (insert into `tests/test_pdb_renumber.py` above the `if __name__ == '__main__':` guard; the existing `test_renumber` must keep passing)

```python
import io as _io

from Bio.PDB import PDBParser as _PDBParser

from modules.pdb_renumber import extract_header_records, plan_residue_numbers


def _pdb(residues, chain="A"):
    lines = []
    for serial, (record, resname, num, icode) in enumerate(residues, 1):
        lines.append(f"{record:<6}{serial:>5}  CA  {resname:>3} {chain}{num:>4}{icode:1}   "
                     f"{serial * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    return "\n".join(lines) + "\nEND\n"


def _structure(text):
    return _PDBParser(QUIET=True).get_structure("t", _io.StringIO(text))


class TestRenumberingPlan(unittest.TestCase):
    def test_no_duplicates_when_unmapped_tail_follows(self):
        self.assertEqual(plan_residue_numbers(6, {0: 4, 1: 5, 2: 6}),
                         [(4, ' '), (5, ' '), (6, ' '), (7, ' '), (8, ' '), (9, ' ')])

    def test_internal_insertion_gets_insertion_code(self):
        self.assertEqual(plan_residue_numbers(4, {0: 1, 1: 2, 3: 3}),
                         [(1, ' '), (2, ' '), (2, 'A'), (3, ' ')])

    def test_n_terminal_overhang_counts_down(self):
        self.assertEqual(plan_residue_numbers(4, {2: 1, 3: 2}),
                         [(-1, ' '), (0, ' '), (1, ' '), (2, ' ')])

    def test_insertion_codes_reset_and_assigned(self):
        # Chain numbered with Kabat-style insertion codes; target equals the chain sequence.
        residues = [("ATOM", "ALA", 52, " "), ("ATOM", "CYS", 52, "A"), ("ATOM", "GLY", 53, " ")]
        structure = _structure(_pdb(residues))
        mappings, _, _ = prepare_renumbering(structure, [SeqRecord(Seq("ACG"), id="t")])
        out = apply_renumbering(structure, mappings)
        self.assertEqual([r.id[1:] for r in out[0]["A"]], [(1, ' '), (2, ' '), (3, ' ')])

    def test_ligands_keep_their_ids(self):
        residues = [("ATOM", "ALA", 1, " "), ("ATOM", "CYS", 2, " "), ("HETATM", "NAG", 2, " ")]
        structure = _structure(_pdb(residues))
        mappings, _, _ = prepare_renumbering(structure, [SeqRecord(Seq("MACG"), id="t")])
        out = apply_renumbering(structure, mappings)
        ids = [r.id for r in out[0]["A"]]
        self.assertIn(("H_NAG", 2, " "), ids)
        self.assertIn((" ", 2, " "), ids)
        self.assertIn((" ", 3, " "), ids)

    def test_all_models_renumbered(self):
        body = _pdb([("ATOM", "ALA", 1, " "), ("ATOM", "CYS", 2, " ")]).replace("END\n", "")
        text = "MODEL        1\n" + body + "ENDMDL\nMODEL        2\n" + body + "ENDMDL\nEND\n"
        structure = _structure(text)
        mappings, _, _ = prepare_renumbering(structure, [SeqRecord(Seq("MAC"), id="t")])
        out = apply_renumbering(structure, mappings)
        for model in out:
            self.assertEqual([r.id[1] for r in model["A"]], [2, 3])

    def test_extract_header_records(self):
        text = "HEADER    X\nCRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\nATOM ...\n"
        self.assertEqual(extract_header_records(text).splitlines()[0][:6], "CRYST1")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_pdb_renumber.py -v`
Expected: FAIL with `ImportError: cannot import name 'plan_residue_numbers'`.

- [ ] **Step 3: Implement**

`modules/pdb_renumber.py` (full replacement):

```python
import copy
import io
import string
import traceback

import streamlit as st
from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Structure import Structure

from modules.pairwise import get_aligner
from modules.parsers import parse_sequences_from_file, parse_sequences_from_text, polymer_residues
from modules.seqtools import clean_sequence
from modules.utils import amino_acid_map

HEADER_RECORDS = ("CRYST1", "SCALE", "ORIGX", "MTRIX")


def chain_sequence(residues):
    """One-letter sequence of polymer residues (length always equals len(residues))."""
    return "".join(amino_acid_map.get(res.resname.strip(), "X") for res in residues)


def align_target_to_chain(target_seq, chain_seq):
    """Semi-global (free end gaps) BLOSUM62 alignment of target vs chain.

    Returns (aligned_target, aligned_chain, identity over aligned columns).
    `chain_seq` must keep its length through cleaning, which holds for amino_acid_map letters.
    """
    aligner = get_aligner("Protein", mode="overlap")
    target = clean_sequence(target_seq, "Protein")
    chain = clean_sequence(chain_seq, "Protein")
    alignment = next(iter(aligner.align(target, chain)))
    aligned_t, aligned_c = str(alignment[0]), str(alignment[1])
    aligned_cols = sum(1 for a, b in zip(aligned_t, aligned_c) if a != '-' and b != '-')
    matches = sum(1 for a, b in zip(aligned_t, aligned_c) if a == b and a != '-')
    return aligned_t, aligned_c, (matches / aligned_cols if aligned_cols else 0.0)


def get_mapping_from_alignment(aligned_target, aligned_chain):
    """Map 0-based chain residue index -> 1-based target position for aligned columns."""
    mapping = {}
    target_pos = 1
    chain_pos = 0
    for t_char, c_char in zip(aligned_target, aligned_chain):
        if c_char != '-':
            if t_char != '-':
                mapping[chain_pos] = target_pos
            chain_pos += 1
        if t_char != '-':
            target_pos += 1
    return mapping


def plan_residue_numbers(n_residues, mapping):
    """Final (residue number, insertion code) for each polymer residue of a chain.

    Mapped residues get their target position with no insertion code. Unmapped residues
    between two mapped ones are insertions relative to the target and get codes A, B, C...
    after the preceding target position. N-terminal overhangs count down from the first
    mapped position; C-terminal overhangs count up from the last. Raises ValueError rather
    than produce duplicate IDs.
    """
    if not mapping:
        raise ValueError("No residue of this chain aligned to the target.")
    mapped = sorted(mapping)
    first, last = mapped[0], mapped[-1]
    numbers = [None] * n_residues
    for i in mapped:
        numbers[i] = (mapping[i], " ")
    for k, i in enumerate(range(first - 1, -1, -1), start=1):
        numbers[i] = (mapping[first] - k, " ")
    for k, i in enumerate(range(last + 1, n_residues), start=1):
        numbers[i] = (mapping[last] + k, " ")
    previous = mapping[first]
    run = 0
    for i in range(first, last + 1):
        if i in mapping:
            previous = mapping[i]
            run = 0
            continue
        if run >= len(string.ascii_uppercase):
            raise ValueError(f"More than 26 consecutive insertions after target position {previous}.")
        numbers[i] = (previous, string.ascii_uppercase[run])
        run += 1
    if len(set(numbers)) != len(numbers):
        raise ValueError("Renumbering would create duplicate residue IDs.")
    return numbers


def prepare_renumbering(structure: Structure, target_sequences: list):
    """Choose the best target for each chain of the first model and compute its mapping.

    Returns (mappings: {chain_id: {residue_index: target_pos}}, log messages, alignment visuals).
    """
    log_messages, alignment_visuals, mappings = [], [], {}
    model = next(iter(structure))
    for chain in model:
        residues = polymer_residues(chain)
        chain_seq = chain_sequence(residues)
        if not chain_seq or set(chain_seq) == {"X"}:
            log_messages.append(f"Chain {chain.id}: no protein residues found, skipping.")
            continue

        best = None
        for target in target_sequences:
            aligned_t, aligned_c, identity = align_target_to_chain(str(target.seq), chain_seq)
            matches = sum(1 for a, b in zip(aligned_t, aligned_c) if a == b and a != '-')
            if best is None or matches > best[0]:
                best = (matches, identity, target, aligned_t, aligned_c)
        if best is None:
            log_messages.append(f"Chain {chain.id}: no target sequences provided.")
            continue

        matches, identity, target, aligned_t, aligned_c = best
        mapping = get_mapping_from_alignment(aligned_t, aligned_c)
        if not mapping:
            log_messages.append(f"Chain {chain.id}: no residues aligned to {target.id}, skipping.")
            continue
        mappings[chain.id] = mapping
        alignment_visuals.append((chain.id, target.id,
                                  f"Target ({target.id}): {aligned_t}\nChain  ({chain.id})       : {aligned_c}"))
        message = (f"Chain {chain.id}: best match is {target.id} with {matches} identical residues "
                   f"({identity:.1%} identity over aligned columns).")
        if identity < 0.5:
            message += " ⚠️ Low identity: check that this is the right target."
        log_messages.append(message)
    return mappings, log_messages, alignment_visuals


def apply_renumbering(structure: Structure, mappings: dict):
    """Return a renumbered copy; every model uses the mapping computed on the first model."""
    renumbered = copy.deepcopy(structure)
    for model in renumbered:
        for chain in model:
            if chain.id not in mappings:
                continue
            residues = polymer_residues(chain)
            mapping = mappings[chain.id]
            if mapping and max(mapping) >= len(residues):
                raise ValueError(f"Model {model.id}, chain {chain.id}: residue count differs from the first model.")
            new_ids = plan_residue_numbers(len(residues), mapping)
            # Two passes so no intermediate ID collides with a residue not yet renumbered.
            for k, res in enumerate(residues):
                res.id = (res.id[0], -100000 - k, " ")
            for res, (number, icode) in zip(residues, new_ids):
                res.id = (res.id[0], number, icode)
    return renumbered


def extract_header_records(pdb_text):
    """Crystallographic records (CRYST1, SCALE, ORIGX, MTRIX) that PDBIO does not write."""
    return "".join(line + "\n" for line in pdb_text.splitlines() if line.startswith(HEADER_RECORDS))


def pdb_renumber_section(pdb_file, pdb_format):
    """UI and workflow for PDB Residue Renumbering based on a target sequence."""
    st.header("🔢 PDB Residue Renumbering")
    st.info(
        "Provide target sequences (FASTA). Each chain is matched to its best target and renumbered so "
        "residue numbers equal target positions (1-based). Insertions relative to the target get "
        "insertion codes (e.g. 52A); terminal overhangs continue the numbering outward. "
        "Ligands and waters keep their original numbers."
    )

    if not pdb_file:
        st.warning(f"Please upload a {pdb_format} file in the sidebar first.")
        return

    target_sequences = None
    target_fasta_text = st.text_area("Paste Target Sequences (FASTA format)")
    target_fasta_file = st.file_uploader("Or Upload Target FASTA File", type=["fasta", "fas", "fa", "txt"])
    if target_fasta_file:
        target_sequences, error = parse_sequences_from_file(target_fasta_file, "fasta")
        if error:
            st.error(error)
    elif target_fasta_text:
        target_sequences, error = parse_sequences_from_text(target_fasta_text)
        if error:
            st.error(error)
    if not target_sequences:
        return

    pdb_file.seek(0)
    raw_text = pdb_file.read().decode("utf-8", errors="replace")
    current_hash = hash((raw_text, tuple(str(s.seq) for s in target_sequences)))
    if st.session_state.get('pdb_renumber_hash') != current_hash:
        with st.spinner("Matching chains to target sequences..."):
            try:
                parser = MMCIFParser(QUIET=True) if pdb_format == "mmCIF" else PDBParser(QUIET=True)
                structure = parser.get_structure("renumber", io.StringIO(raw_text))
                mappings, logs, visuals = prepare_renumbering(structure, target_sequences)
                st.session_state.pdb_renumber_results = {
                    'structure': structure, 'mappings': mappings, 'logs': logs, 'visuals': visuals,
                }
                st.session_state.pdb_renumber_hash = current_hash
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.text(traceback.format_exc())
                return

    results = st.session_state.pdb_renumber_results
    st.subheader("Interactive Review & Confirmation")
    with st.expander("Review Sequence Alignments (PDB vs Target)", expanded=True):
        for chain_id, target_id, aln_text in results['visuals']:
            st.markdown(f"**Chain {chain_id}** matched to **{target_id}**")
            st.text(aln_text)
    with st.expander("Renumbering Logs", expanded=True):
        for msg in results['logs']:
            st.write(msg)

    if pdb_format == "mmCIF":
        st.caption("mmCIF output contains coordinates only; other mmCIF categories are not carried over.")
    st.warning("Please review the alignment above. If it looks correct, confirm the renumbering.")

    if st.button("Confirm Renumbering & Generate File"):
        try:
            renumbered = apply_renumbering(results['structure'], results['mappings'])
        except ValueError as e:
            st.error(f"Renumbering aborted: {e}")
            return
        writer = MMCIFIO() if pdb_format == "mmCIF" else PDBIO()
        writer.set_structure(renumbered)
        out_stream = io.StringIO()
        writer.save(out_stream)
        content = out_stream.getvalue()
        if pdb_format != "mmCIF":
            content = extract_header_records(raw_text) + content
        ext = ".cif" if pdb_format == "mmCIF" else ".pdb"
        st.success("Renumbering completed successfully!")
        st.download_button(
            label=f"📥 Download Renumbered {pdb_format}",
            data=content,
            file_name=f"{pdb_file.name.rsplit('.', 1)[0]}_renumbered{ext}",
            mime="chemical/x-mmcif" if pdb_format == "mmCIF" else "chemical/x-pdb",
        )
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_pdb_renumber.py -v`
Expected: PASS (8 tests including the original `test_renumber`).

- [ ] **Step 5: Commit**

```bash
git add modules/pdb_renumber.py tests/test_pdb_renumber.py
git commit -m "fix: renumbering never duplicates IDs; insertion codes, ligands, all models, CRYST1 kept

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Format conversion: molecule type, alignment formats, optional alignment

**Files:**
- Rewrite: `modules/conversion.py`
- Modify: `app.py` (pass `seq_type`)
- Test: `tests/test_conversion.py`

**Interfaces:**
- Consumes: `run_msa`, `MSAError`
- Produces:
  - `OUTPUT_FORMATS = ("fasta", "clustal", "phylip-relaxed", "stockholm", "nexus", "embl", "genbank")`
  - `ALIGNMENT_FORMATS = {"clustal", "phylip-relaxed", "stockholm", "nexus"}`
  - `convert_format(sequences, output_format, seq_type="Protein") -> (data | None, error | None)`
  - `format_conversion_section(sequences, input_format, seq_type="Protein")`

- [ ] **Step 1: Write the failing tests**

`tests/test_conversion.py`:

```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.conversion import convert_format


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def test_fasta_to_genbank_sets_molecule_type():
    data, error = convert_format([rec("s1", "ATGC"), rec("s2", "ATGCAA")], "genbank", seq_type="DNA")
    assert error is None
    assert "LOCUS" in data and "DNA" in data


def test_embl_protein():
    data, error = convert_format([rec("p1", "MKV")], "embl", seq_type="Protein")
    assert error is None
    assert "PROTEIN" in data.upper()


def test_alignment_format_needs_equal_lengths():
    data, error = convert_format([rec("s1", "ATGC"), rec("s2", "ATGCAA")], "clustal", seq_type="DNA")
    assert data is None
    assert "aligned" in error


def test_input_records_not_mutated():
    r = rec("s1", "ATGC")
    convert_format([r], "genbank", seq_type="DNA")
    assert "molecule_type" not in r.annotations
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_conversion.py -v`
Expected: FAIL (`missing molecule_type`, unexpected `seq_type` keyword).

- [ ] **Step 3: Implement**

`modules/conversion.py` (full replacement):

```python
import streamlit as st
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from io import StringIO

from modules.msa import MSAError, run_msa
from modules.utils import have_params_changed

OUTPUT_FORMATS = ("fasta", "clustal", "phylip-relaxed", "stockholm", "nexus", "embl", "genbank")
ALIGNMENT_FORMATS = {"clustal", "phylip-relaxed", "stockholm", "nexus"}
FILE_EXTENSIONS = {"fasta": "fasta", "clustal": "aln", "phylip-relaxed": "phy", "stockholm": "sto",
                   "nexus": "nex", "embl": "embl", "genbank": "gb"}


def convert_format(sequences, output_format, seq_type="Protein"):
    """Write sequences in `output_format`. Returns (data, None) or (None, error)."""
    molecule_type = "DNA" if seq_type == "DNA" else "protein"
    records = []
    for record in sequences:
        annotations = dict(record.annotations)
        annotations.setdefault("molecule_type", molecule_type)
        name = record.id if record.name in ("", "<unknown name>") else record.name
        records.append(SeqRecord(record.seq, id=record.id, name=name,
                                 description=record.description, annotations=annotations))
    if output_format in ALIGNMENT_FORMATS and len({len(r.seq) for r in records}) > 1:
        return None, (f"{output_format} requires aligned sequences of equal length. "
                      "Enable 'Align with FAMSA before converting' or export from the MSA tool.")
    try:
        output = StringIO()
        SeqIO.write(records, output, output_format)
        return output.getvalue(), None
    except Exception as e:
        return None, f"Error during format conversion: {e}"


def format_conversion_section(sequences, input_format, seq_type="Protein"):
    """Handles the sequence format conversion workflow."""
    st.header("🔄 Sequence Format Conversion")
    st.info("Convert your sequences between different file formats.")

    output_format = st.selectbox("Select Output Format", OUTPUT_FORMATS,
                                 help="Choose the desired format for the converted sequences")
    align_first = False
    if output_format in ALIGNMENT_FORMATS:
        align_first = st.checkbox("Align with FAMSA before converting", value=True,
                                  help="Alignment formats need sequences of equal length.")

    current_params = {
        'sequences': tuple((s.id, str(s.seq)) for s in sequences),
        'input_format': input_format,
        'output_format': output_format,
        'seq_type': seq_type,
        'align_first': align_first,
    }
    params_changed = have_params_changed(current_params, 'last_conversion_params')

    if st.button("Convert Format"):
        with st.spinner("Converting format..."):
            records = sequences
            error = None
            if align_first:
                try:
                    aligned = run_msa([(s.id, str(s.seq)) for s in sequences], seq_type)
                    records = [SeqRecord(Seq(seq), id=seq_id, description="") for seq_id, seq in aligned]
                except MSAError as e:
                    error = str(e)
            if error:
                st.session_state.converted_data, st.session_state.conversion_error = None, error
            else:
                st.session_state.converted_data, st.session_state.conversion_error = convert_format(
                    records, output_format, seq_type)
            st.session_state.last_conversion_params = current_params
        params_changed = False

    if params_changed:
        return
    if st.session_state.get('converted_data'):
        st.success("Format conversion successful.")
        st.text(st.session_state.converted_data)
        st.download_button(
            label="📥 Download Converted File",
            data=st.session_state.converted_data,
            file_name=f"converted_sequences.{FILE_EXTENSIONS[output_format]}",
            mime="text/plain",
        )
    elif st.session_state.get('conversion_error'):
        st.error(f"Format conversion failed: {st.session_state.conversion_error}")
```

In `app.py`, change `format_conversion_section(sequences, input_format)` to `format_conversion_section(sequences, input_format, seq_type)`.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass. Run `python precommit_check.py; echo "exit=$?"` and expect `Pre-commit passed.` with `exit=0`.

- [ ] **Step 5: Manual smoke test of the whole plan**

`streamlit run app.py`, then:

1. Paste 3 protein sequences.
2. Run Pairwise in Global, Local and Overlap modes.
3. Run MSA with phylip-relaxed output, and build an NJ tree from the MSA.
4. Run Clustering (Accurate, Hierarchical).
5. Run Convert → GenBank.
6. Switch the input to DNA and run Translate DNA → Find ORFs.

None of these should produce a traceback.

- [ ] **Step 6: Commit**

```bash
git add modules/conversion.py app.py tests/test_conversion.py
git commit -m "fix: conversion sets molecule type, validates alignment formats, optional FAMSA alignment

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
