# Plan 03 — GUI and Usability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The app explains itself: correct naming, auto-detected sequence type, a dataset overview, example data, a consistent analysis menu, readable alignments with standard statistics, and clearer plots.

**Architecture:** New small pure modules: `modules/overview.py` (dataset summary) and `modules/navigation.py` (which analyses apply). Display helpers live next to the analysis they serve and are unit-tested as pure functions. Example datasets ship in `examples/`.

**Tech Stack:** Streamlit 1.60, Plotly (incl. `figure_factory.create_dendrogram`), SciPy, pandas.

**Spec:** `docs/superpowers/specs/2026-10-07-codebase-review.md` (items G1–G10)

**Prerequisite:** Plan 02 merged to `main`.

## Global Constraints

- All Plan 01 and 02 constraints apply.
- Product name everywhere: **"Advanced Sequence Alignment Tool"**. The word "Ability" must not appear in UI text (`grep -rni "abilit" app.py modules` returns nothing).
- Example data is fetched once from UniProt/NCBI and committed under `examples/`. The app never fetches at runtime in this plan.
- Branch: `feat/03-gui`. Commit after every task with the `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer.

## Review Focus

1. **Mixed DNA and protein sequences in one input.** Expected: the majority type is detected and the user can override it. Covered in Task 2 (`test_detect_mixed_majority`).
2. **The selected analysis disappears after switching sequence type** (e.g. Translate DNA → Protein). Expected: falls back to the first available analysis without an exception. Covered in Task 3 (`test_analysis_falls_back_when_unavailable`).
3. **Very long sequence IDs in the alignment display.** Expected: block columns stay aligned. Covered in Task 4 (`test_blocks_align_with_long_ids`).
4. **A pairwise alignment with no substitutions.** Expected: a "no point mutations" message and no empty plot. Covered in Task 4 (`test_lollipop_none_without_mutations`).
5. **Renumbering where the user excludes a chain.** Expected: that chain keeps its original numbering. Covered in Task 6 (`test_override_skip_chain`).

---

### Task 1: Naming, welcome text and leftover wording

**Files:**
- Modify: `app.py`, any module with "Ability"/"abilities"
- Test: `tests/test_app_flow.py`

**Interfaces:**
- Consumes: nothing
- Produces: page title "Advanced Sequence Alignment Tool"; the welcome text moves into an expander "ℹ️ How to use this tool"

- [ ] **Step 1: Create the branch**

```bash
git checkout main && git pull --ff-only && git checkout -b feat/03-gui
```

- [ ] **Step 2: Write the failing test** (append to `tests/test_app_flow.py`)

```python
def test_title_has_no_ability_wording():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    assert at.title[0].value == "🔬 Advanced Sequence Alignment Tool"
    assert all("Abilit" not in m.value for m in at.markdown)
```

- [ ] **Step 3: Run the test and confirm it fails**

Run: `python -m pytest tests/test_app_flow.py::test_title_has_no_ability_wording -v`
Expected: FAIL (`'🔬 Advanced Sequence Alignment and Format Conversion Ability'`).

- [ ] **Step 4: Implement**

In `app.py` `main()`, replace the `st.set_page_config`, `st.title` and `st.write(...)` welcome block with:

```python
    st.set_page_config(page_title="Advanced Sequence Alignment Tool", page_icon="🔬", layout="wide")
    init_session_state()

    st.title("🔬 Advanced Sequence Alignment Tool")
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
Align, compare and annotate **DNA** or **protein** sequences and structures.

1. **Load data** in the sidebar: paste FASTA, upload a sequence or structure file, or load an example.
2. **Check the sequence type.** It is detected automatically; change it if needed.
3. **Pick an analysis** and adjust its settings.
4. **Run** it and **download** the results.

Pairwise alignments support Global, Local and Overlap (free end gaps) modes. MSA uses FAMSA.
Mutation positions are always reported in reference-sequence coordinates.
""")
```

Run `grep -rni "abilit" app.py modules` and rewrite every remaining hit in plain English (for example, "This is useful for compatibility with different analysis abilities." becomes "Useful for exchanging data with other tools.").

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "ui: fix product naming and move welcome text into an expander

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Auto-detected sequence type, dataset overview, example data

**Files:**
- Modify: `modules/seqtools.py` (add `detect_seq_type`)
- Create: `modules/overview.py`, `examples/globins.fasta`, `examples/puc19.fasta`, `examples/antibody_fv.fasta`
- Modify: `app.py`
- Test: `tests/test_overview.py`, `tests/test_app_flow.py`

**Interfaces:**
- Consumes: `fingerprint`, `track_dataset`
- Produces:
  - `detect_seq_type(seqs: list[str]) -> "DNA" | "Protein"`
  - `modules.overview.EXAMPLES: dict[str, pathlib.Path]`
  - `dataset_summary(records) -> dict` with keys `count`, `min_length`, `median_length`, `max_length`
  - `summary_table(records, preview=40) -> pandas.DataFrame` with columns `ID`, `Length`, `Preview`
  - `show_dataset_overview(records, seq_type)` (UI)
  - `load_example_callback()`: an `on_click` callback that sets `st.session_state.input_format` and `st.session_state.fasta_text`

- [ ] **Step 1: Add the example files**

```bash
mkdir -p examples
curl -sf "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=accession:P69905+OR+accession:P68871+OR+accession:P02144+OR+accession:P01942+OR+accession:P02088+OR+accession:P69891" -o examples/globins.fasta
curl -sf "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=L09137&rettype=fasta&retmode=text" -o examples/puc19.fasta
grep -c ">" examples/globins.fasta examples/puc19.fasta
```

Expected: `examples/globins.fasta:6` and `examples/puc19.fasta:1`.

`examples/antibody_fv.fasta` (germline-like human VH3-23/VK1 pair, used in the antibody tests):

```
>example_VH germline-like human IGHV3-23 heavy chain variable domain
EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGYSSGWYFDYWGQGTLVTVSS
>example_VL germline-like human IGKV1-39 light chain variable domain
DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGQGTKVEIK
```

- [ ] **Step 2: Write the failing tests**

`tests/test_overview.py`:

```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.overview import EXAMPLES, dataset_summary, summary_table
from modules.parsers import parse_sequences_from_text
from modules.seqtools import detect_seq_type


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id)


def test_detect_dna_rna_protein():
    assert detect_seq_type(["ATGCNNATGC"]) == "DNA"
    assert detect_seq_type(["AUGCCC"]) == "DNA"
    assert detect_seq_type(["MKTAYIAKQR"]) == "Protein"
    assert detect_seq_type([]) == "Protein"


def test_detect_mixed_majority():
    assert detect_seq_type(["ATGCATGCATGCATGC", "ATGCATGCATGC", "MKV"]) == "DNA"


def test_dataset_summary_and_table():
    records = [rec("a", "MKV"), rec("b", "MKVLA"), rec("c", "M" * 50)]
    assert dataset_summary(records) == {"count": 3, "min_length": 3, "median_length": 5, "max_length": 50}
    table = summary_table(records, preview=10)
    assert list(table.columns) == ["ID", "Length", "Preview"]
    assert table.loc[2, "Preview"] == "M" * 10 + "…"


def test_examples_parse_and_detect():
    expected = {"Globins (protein)": "Protein", "pUC19 plasmid (DNA)": "DNA", "Antibody VH/VL (protein)": "Protein"}
    for name, kind in expected.items():
        records, error = parse_sequences_from_text(EXAMPLES[name].read_text())
        assert error is None, (name, error)
        assert detect_seq_type([str(r.seq) for r in records]) == kind
```

Append to `tests/test_app_flow.py`:

```python
def test_dna_detected_automatically():
    at = load(">x\nATGCATGCATGCATGC\n>y\nATGCATGAATGCATGC\n")
    assert at.selectbox(key="seq_type_select").value == "DNA"


def test_load_example_button():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.selectbox(key="example_choice").select("Globins (protein)").run()
    at.button(key="load_example").click().run()
    assert not at.exception
    assert len(at.session_state["input_cache"][0]) == 6
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_overview.py tests/test_app_flow.py -v`
Expected: FAIL with `ModuleNotFoundError: modules.overview`.

- [ ] **Step 4: Implement**

Append to `modules/seqtools.py`:

```python
_NUCLEOTIDE_LETTERS = frozenset("ACGTUN")


def detect_seq_type(seqs, dna_fraction=0.9):
    """'DNA' if at least `dna_fraction` of all letters are A/C/G/T/U/N, else 'Protein'."""
    letters = [c for s in seqs for c in str(s).upper() if c.isalpha()]
    if not letters:
        return "Protein"
    nucleotide = sum(1 for c in letters if c in _NUCLEOTIDE_LETTERS)
    return "DNA" if nucleotide / len(letters) >= dna_fraction else "Protein"
```

`modules/overview.py`:

```python
"""Dataset overview panel and bundled example datasets."""
import statistics
from pathlib import Path

import pandas as pd
import streamlit as st

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = {
    "Globins (protein)": EXAMPLE_DIR / "globins.fasta",
    "pUC19 plasmid (DNA)": EXAMPLE_DIR / "puc19.fasta",
    "Antibody VH/VL (protein)": EXAMPLE_DIR / "antibody_fv.fasta",
}


def dataset_summary(records):
    lengths = [len(r.seq) for r in records]
    return {
        "count": len(records),
        "min_length": min(lengths) if lengths else 0,
        "median_length": int(statistics.median(lengths)) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def summary_table(records, preview=40):
    rows = []
    for r in records:
        s = str(r.seq)
        rows.append({"ID": r.id, "Length": len(s), "Preview": s[:preview] + ("…" if len(s) > preview else "")})
    return pd.DataFrame(rows, columns=["ID", "Length", "Preview"])


def show_dataset_overview(records, seq_type):
    summary = dataset_summary(records)
    with st.expander(f"📊 Dataset: {summary['count']} {seq_type} sequences", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sequences", summary["count"])
        c2.metric("Shortest", summary["min_length"])
        c3.metric("Median length", summary["median_length"])
        c4.metric("Longest", summary["max_length"])
        st.dataframe(summary_table(records), hide_index=True, width="stretch")


def load_example_callback():
    """on_click callback: put the chosen example into the FASTA text box."""
    name = st.session_state.get("example_choice")
    if name in EXAMPLES:
        st.session_state.input_format = "Text (FASTA)"
        st.session_state.fasta_text = EXAMPLES[name].read_text()
```

In `app.py`:

1. Import `from modules.overview import EXAMPLES, load_example_callback, show_dataset_overview` and add `detect_seq_type` to the seqtools import.
2. Directly after `st.sidebar.header("📥 Upload and Settings")`, add:

```python
    c_ex, c_btn = st.sidebar.columns([3, 1], vertical_alignment="bottom")
    c_ex.selectbox("Example data", ["—"] + list(EXAMPLES), key="example_choice")
    c_btn.button("Load", key="load_example", on_click=load_example_callback,
                 disabled=st.session_state.get("example_choice", "—") == "—")
```

3. Replace the block from `track_dataset(fingerprint(sequences))` through the `seq_type` selectbox and its `if` reset with:

```python
    changed = track_dataset(fingerprint(sequences))
    detected = detect_seq_type([str(s.seq) for s in sequences])
    if changed or "seq_type_select" not in st.session_state:
        st.session_state.seq_type_select = detected
    seq_type = st.sidebar.selectbox("🔬 Sequence Type", ("DNA", "Protein"), key="seq_type_select",
                                    help=f"Auto-detected as {detected}. Change it if the detection is wrong.")
    if st.session_state.get("seq_type") != seq_type:
        st.session_state.seq_type = seq_type
        reset_all_results()
    structures = st.session_state.get("structures", {})
    show_dataset_overview(sequences, seq_type)
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add examples modules/seqtools.py modules/overview.py app.py tests/test_overview.py tests/test_app_flow.py
git commit -m "ui: auto-detect sequence type, dataset overview, bundled example data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: One analysis menu for every input

**Files:**
- Create: `modules/navigation.py`
- Modify: `app.py`
- Test: `tests/test_navigation.py`, `tests/test_app_flow.py`

**Interfaces:**
- Consumes: nothing
- Produces: `available_analyses(has_sequences: bool, has_structures: bool, seq_type: str) -> list[str]`. Plans 04 and 06 extend this function.

- [ ] **Step 1: Write the failing tests**

`tests/test_navigation.py`:

```python
from modules.navigation import available_analyses


def test_protein_sequences():
    assert available_analyses(True, False, "Protein") == [
        "Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Phylogenetic Tree",
        "Antibody Prediction", "Convert Formats",
    ]


def test_dna_sequences_get_translation_not_antibody():
    options = available_analyses(True, False, "DNA")
    assert "Translate DNA" in options
    assert "Antibody Prediction" not in options


def test_structures_add_extraction_and_renumbering():
    options = available_analyses(True, True, "Protein")
    assert options[0] == "Extracted Sequences"
    assert options[-1] == "PDB Residue Renumbering"
    assert "Pairwise" in options
```

Append to `tests/test_app_flow.py`:

```python
def test_analysis_falls_back_when_unavailable():
    at = load(">x\nATGCATGCATGCATGC\n>y\nATGCATGAATGCATGC\n")
    at.selectbox(key="analysis_select").select("Translate DNA").run()
    at.selectbox(key="seq_type_select").select("Protein").run()
    assert not at.exception
    assert at.selectbox(key="analysis_select").value != "Translate DNA"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_navigation.py tests/test_app_flow.py -v`
Expected: FAIL with `ModuleNotFoundError: modules.navigation`.

- [ ] **Step 3: Implement**

`modules/navigation.py`:

```python
"""Which analyses apply to the loaded data."""


def available_analyses(has_sequences, has_structures, seq_type):
    options = []
    if has_structures:
        options.append("Extracted Sequences")
    if has_sequences:
        options += ["Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Phylogenetic Tree"]
        options.append("Translate DNA" if seq_type == "DNA" else "Antibody Prediction")
        options.append("Convert Formats")
    if has_structures:
        options.append("PDB Residue Renumbering")
    return options
```

In `app.py`, import `available_analyses` and replace the `if input_format in STRUCTURE_FORMATS: options = ... else: options = ...` block and the `alignment_mode` selectbox with:

```python
    options = available_analyses(bool(sequences), bool(structures), seq_type)
    if st.session_state.get("analysis_select") not in options:
        st.session_state.analysis_select = options[0]
    alignment_mode = st.sidebar.selectbox("🛠️ Select Analysis", options, key="analysis_select")
```

Replace the Translate DNA branch body with `dna_translation_section(sequences)`; the menu only offers it for DNA.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add modules/navigation.py app.py tests/test_navigation.py tests/test_app_flow.py
git commit -m "ui: single analysis menu driven by loaded data; structures get all sequence analyses

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Readable pairwise output with standard statistics and a lollipop plot

**Files:**
- Modify: `modules/pairwise.py` (`compute_alignment_stats`, `format_alignment_blocks`, `mutation_lollipop`, new `format_result_text`, UI; delete `format_alignment_display` and `location_line`)
- Modify: `tests/test_pairwise.py` (replace `test_location_line_respects_start_offset`)

**Interfaces:**
- Consumes: `PairwiseResult`
- Produces:
  - `compute_alignment_stats(aligned_ref, aligned_query, match_line) -> dict` with keys `length, identical, similar, gaps, identity_pct, similarity_pct, gaps_pct`
  - `format_alignment_blocks(result, width=60) -> str` (EMBOSS-like blocks; reference positions in original coordinates)
  - `format_result_text(result) -> str`: header (`Pairwise Alignment`), statistics line with `Score`, blocks, notes
  - `mutation_lollipop(mutations, ref_length) -> go.Figure | None`

- [ ] **Step 1: Write the failing tests**

In `tests/test_pairwise.py`, delete `test_location_line_respects_start_offset` and remove `location_line` from the import. Append:

```python
from modules.pairwise import compute_alignment_stats, format_alignment_blocks, mutation_lollipop


def test_alignment_stats():
    stats = compute_alignment_stats("MKV-L", "MRVAL", "|:| |")
    assert stats["length"] == 5
    assert stats["identical"] == 3
    assert stats["similar"] == 4
    assert stats["gaps"] == 1
    assert stats["identity_pct"] == 60.0


def test_blocks_wrap_and_use_reference_coordinates():
    result = align_pair(rec("ref", REF), rec("q", "RQLEERLGLIEVQAPWLSRVGDG"), "Protein", "local")
    text = format_alignment_blocks(result, width=10)
    first = text.splitlines()[0].split()
    assert first[0] == "ref" and first[1] == "21"
    assert len([l for l in text.splitlines() if l.startswith("ref")]) == 3


def test_blocks_align_with_long_ids():
    long_id = "a_really_long_sequence_identifier"
    result = align_pair(rec(long_id, "MKVLAAGIVG"), rec("q", "MKVLAAGIVG"), "Protein")
    lines = format_alignment_blocks(result).splitlines()
    assert lines[0].index("MKVLAAGIVG") == lines[2].index("MKVLAAGIVG")


def test_lollipop_none_without_mutations():
    assert mutation_lollipop([], 100) is None
    fig = mutation_lollipop(["K2R", "V30A"], 100)
    assert list(fig.data[1].x) == [2, 30]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_pairwise.py -v`
Expected: FAIL with `ImportError: compute_alignment_stats`.

- [ ] **Step 3: Implement**

In `modules/pairwise.py`, delete `location_line` and `format_alignment_display`, then add:

```python
def compute_alignment_stats(aligned_ref, aligned_query, match_line):
    """EMBOSS-style statistics: identity and similarity over the full alignment length."""
    length = len(aligned_ref)
    identical = match_line.count("|")
    similar = identical + match_line.count(":")
    gaps = sum(1 for a, b in zip(aligned_ref, aligned_query) if a == "-" or b == "-")

    def pct(n):
        return round(100.0 * n / length, 1) if length else 0.0

    return {"length": length, "identical": identical, "similar": similar, "gaps": gaps,
            "identity_pct": pct(identical), "similarity_pct": pct(similar), "gaps_pct": pct(gaps)}


def format_alignment_blocks(result, width=60):
    """Alignment wrapped in blocks of `width` columns with start/end positions per line."""
    name_width = max(len(result.ref_id), len(result.query_id))

    def ref_label(pos):
        if result.ref_rotation and result.ref_length and pos > 0:
            return ((pos - 1 + result.ref_rotation) % result.ref_length) + 1
        return pos

    ref_count, query_count = result.ref_start, result.query_start
    lines = []
    for i in range(0, len(result.aligned_ref), width):
        r = result.aligned_ref[i:i + width]
        q = result.aligned_query[i:i + width]
        m = result.match_line[i:i + width]
        r_res = sum(c != "-" for c in r)
        q_res = sum(c != "-" for c in q)
        r_start, r_end = ref_count + (1 if r_res else 0), ref_count + r_res
        q_start, q_end = query_count + (1 if q_res else 0), query_count + q_res
        lines.append(f"{result.ref_id:<{name_width}} {ref_label(r_start):>6} {r} {ref_label(r_end)}")
        lines.append(f"{'':<{name_width}} {'':>6} {m}")
        lines.append(f"{result.query_id:<{name_width}} {q_start:>6} {q} {q_end}")
        lines.append("")
        ref_count += r_res
        query_count += q_res
    return "\n".join(lines)


def format_result_text(result):
    """Plain-text report: header, statistics, wrapped alignment and notes."""
    s = compute_alignment_stats(result.aligned_ref, result.aligned_query, result.match_line)
    header = (
        "Pairwise Alignment\n"
        f"Reference: {result.ref_id}\nQuery:     {result.query_id}\n"
        f"Length: {s['length']}   Identity: {s['identical']}/{s['length']} ({s['identity_pct']}%)   "
        f"Similarity: {s['similar']}/{s['length']} ({s['similarity_pct']}%)   "
        f"Gaps: {s['gaps']}/{s['length']} ({s['gaps_pct']}%)   Score: {result.score:.2f}\n\n"
    )
    notes = "".join(f"Note: {n}\n" for n in result.notes)
    return header + format_alignment_blocks(result) + ("\n" + notes if notes else "")


def mutation_lollipop(mutations, ref_length):
    """Stem-and-marker plot of substitutions along the reference; None if there are none."""
    if not mutations:
        return None
    positions = [int(re.search(r"\d+", m).group()) for m in mutations]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=positions, y=[1] * len(positions), width=max(1, ref_length / 400),
                         marker_color="#bbbbbb", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=positions, y=[1] * len(positions), mode="markers+text", text=mutations,
                             textposition="top center", marker=dict(size=10, color="#d62728"),
                             hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title="Substitutions along the reference", xaxis=dict(title="Reference position",
                      range=[0, ref_length + 1]), yaxis=dict(visible=False, range=[0, 1.6]),
                      height=260, showlegend=False, margin=dict(t=40, b=40))
    return fig
```

In `pairwise_alignment_section`, replace everything after the stale-notice early return:

```python
    alignment_text = format_result_text(result)
    stats = compute_alignment_stats(result.aligned_ref, result.aligned_query, result.match_line)
    st.subheader("🧬 Alignment Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Identity", f"{stats['identity_pct']}%", f"{stats['identical']}/{stats['length']}", delta_color="off")
    c2.metric("Similarity", f"{stats['similarity_pct']}%", f"{stats['similar']}/{stats['length']}", delta_color="off")
    c3.metric("Gaps", f"{stats['gaps_pct']}%", f"{stats['gaps']}/{stats['length']}", delta_color="off")
    c4.metric("Score", f"{result.score:.1f}")
    st.code(format_alignment_blocks(result))
    for note in result.notes:
        st.caption(note)

    st.subheader("🔍 Point Mutations Relative to Reference")
    fig = mutation_lollipop(result.mutations, result.ref_length)
    if fig is None:
        st.write("No point mutations detected.")
    else:
        st.write(f"**{len(result.mutations)} substitutions:** {', '.join(result.mutations)}")
        st.plotly_chart(fig, width="stretch")
        st.download_button(label="📥 Download Mutations (CSV)", data=convert_mutations_to_csv(result.mutations),
                           file_name=f"pairwise_mutations_{reference_id}_vs_{target_id}.csv", mime="text/csv",
                           on_click="ignore")

    st.download_button(label="📥 Download Alignment", data=alignment_text,
                       file_name=f"pairwise_alignment_{reference_id}_vs_{target_id}.txt", mime="text/plain",
                       on_click="ignore")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass, including `test_format_result_text_contains_score_and_notes` and `test_logic.py::test_perform_pairwise_alignment_robustness` (the text still contains "Pairwise Alignment").

- [ ] **Step 5: Commit**

```bash
git add modules/pairwise.py tests/test_pairwise.py
git commit -m "ui: wrapped pairwise alignment with identity/similarity/gap stats and lollipop plot

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Clustering heatmap ordered by cluster, dendrogram, letters on small MSAs

**Files:**
- Modify: `modules/clustering.py` (`cluster_order`, heatmap tab, dendrogram)
- Modify: `modules/viz.py` (`LETTER_LIMIT`, letters overlay)
- Test: `tests/test_clustering.py`, `tests/test_viz.py`

**Interfaces:**
- Consumes: `hierarchical_clusters` (its `leaf_order`), `msa_figure_plotly`
- Produces:
  - `cluster_order(ids, clusters, leaf_order=None) -> list[int]`
  - `msa_figure_plotly(..., hover=True)` shows residue letters in cells when `codes.size <= LETTER_LIMIT` (3000)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_clustering.py` (inside the class):

```python
    def test_cluster_order(self):
        from modules.clustering import cluster_order
        ids = ["a", "b", "c", "d"]
        self.assertEqual(cluster_order(ids, [["c", "a"], ["d"], ["b"]]), [2, 0, 3, 1])
        self.assertEqual(cluster_order(ids, [], leaf_order=[3, 1, 0, 2]), [3, 1, 0, 2])
```

Append to `tests/test_viz.py`:

```python
def test_letters_shown_only_for_small_alignments():
    small = alignment_to_codes(["MKV", "MKL"])
    assert msa_figure_plotly(small, ["a", "b"], "Protein").data[0].texttemplate == "%{text}"
    big = alignment_to_codes(["M" * 2000, "K" * 2000])
    assert msa_figure_plotly(big, ["a", "b"], "Protein").data[0].texttemplate is None
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_clustering.py tests/test_viz.py -v`
Expected: FAIL with `ImportError: cluster_order` and `texttemplate` being `None` for the small alignment.

- [ ] **Step 3: Implement**

`modules/viz.py`: add `LETTER_LIMIT = 3_000` next to the other limits. Inside `msa_figure_plotly`'s `if hover:` block, after setting `hovertemplate`, add:

```python
        if codes.size <= LETTER_LIMIT:
            heatmap["texttemplate"] = "%{text}"
            heatmap["textfont"] = dict(size=10, color="#222222")
```

`modules/clustering.py`: add `import plotly.figure_factory as ff` and:

```python
def cluster_order(ids, clusters, leaf_order=None):
    """Row order for heatmaps: dendrogram leaf order if given, else grouped by cluster."""
    if leaf_order is not None:
        return list(leaf_order)
    index = {seq_id: i for i, seq_id in enumerate(ids)}
    return [index[seq_id] for cluster in clusters for seq_id in cluster]
```

In the UI, change `clusters, _ = hierarchical_clusters(matrix, ids, threshold)` to `clusters, leaf_order = hierarchical_clusters(matrix, ids, threshold)`, and initialise `leaf_order = None` next to `threshold = None`. Replace the "Matrix Heatmap" tab body:

```python
        with tab3:
            st.markdown("### Similarity Matrix")
            order = cluster_order(ids, clusters, leaf_order)
            ordered_ids = [ids[i] for i in order]
            fig_hm = go.Figure(data=go.Heatmap(
                z=matrix[np.ix_(order, order)], x=ordered_ids, y=ordered_ids, colorscale="Viridis",
                zmin=0, zmax=1, hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>"))
            show_labels = len(ids) <= 60
            fig_hm.update_layout(title="Pairwise Similarity Matrix (ordered by cluster)", height=700,
                                 xaxis_showticklabels=show_labels, yaxis_showticklabels=show_labels,
                                 yaxis_autorange="reversed")
            st.plotly_chart(fig_hm, width="stretch")
            if clustering_method == HIERARCHICAL and 2 < len(ids) <= 200:
                condensed = squareform(np.clip(1 - matrix, 0, None), checks=False)
                fig_dg = ff.create_dendrogram(np.arange(len(ids)).reshape(-1, 1), labels=ids,
                                              distfun=lambda _x: condensed,
                                              linkagefun=lambda d: linkage(d, method="average"))
                fig_dg.update_layout(title="Average-linkage dendrogram (distance = 1 − similarity)", height=450)
                st.plotly_chart(fig_dg, width="stretch")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add modules/clustering.py modules/viz.py tests/test_clustering.py tests/test_viz.py
git commit -m "ui: cluster-ordered similarity heatmap with dendrogram; residue letters on small MSAs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Renumbering review with per-chain target choice

**Files:**
- Modify: `modules/pdb_renumber.py` (`score_targets`, `build_mappings`, `prepare_renumbering(..., overrides=None)`, UI)
- Test: `tests/test_pdb_renumber.py`

**Interfaces:**
- Consumes: `align_target_to_chain`, `get_mapping_from_alignment`, `polymer_residues`, `chain_sequence`
- Produces:
  - `SKIP = "(keep original numbering)"`
  - `score_targets(structure, target_sequences) -> dict[str, list[dict]]`: per chain, candidates sorted best-first; keys `target_id, matches, identity, aligned_t, aligned_c`
  - `build_mappings(candidates, overrides=None) -> (mappings, logs, visuals)`. `overrides` maps chain → target_id or `SKIP`.
  - `prepare_renumbering(structure, target_sequences, overrides=None)`: same return as before

- [ ] **Step 1: Write the failing tests** (insert above the `if __name__` guard)

```python
class TestRenumberOverrides(unittest.TestCase):
    def setUp(self):
        self.structure = _structure(_pdb([("ATOM", "ALA", 1, " "), ("ATOM", "CYS", 2, " "), ("ATOM", "GLY", 3, " ")]))
        self.targets = [SeqRecord(Seq("MACG"), id="best"), SeqRecord(Seq("WWWWACG"), id="other")]

    def test_default_picks_best(self):
        mappings, _, _ = prepare_renumbering(self.structure, self.targets)
        self.assertEqual(mappings["A"], {0: 2, 1: 3, 2: 4})

    def test_override_other_target(self):
        mappings, _, _ = prepare_renumbering(self.structure, self.targets, overrides={"A": "other"})
        self.assertEqual(mappings["A"], {0: 5, 1: 6, 2: 7})

    def test_override_skip_chain(self):
        from modules.pdb_renumber import SKIP
        mappings, logs, _ = prepare_renumbering(self.structure, self.targets, overrides={"A": SKIP})
        self.assertNotIn("A", mappings)
        out = apply_renumbering(self.structure, mappings)
        self.assertEqual([r.id[1] for r in out[0]["A"]], [1, 2, 3])
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_pdb_renumber.py -v`
Expected: FAIL with `TypeError: prepare_renumbering() got an unexpected keyword argument 'overrides'`.

- [ ] **Step 3: Implement**

Replace `prepare_renumbering` in `modules/pdb_renumber.py` with:

```python
SKIP = "(keep original numbering)"


def score_targets(structure, target_sequences):
    """Align every protein chain of the first model to every target; candidates sorted best-first."""
    candidates = {}
    for chain in next(iter(structure)):
        chain_seq = chain_sequence(polymer_residues(chain))
        if not chain_seq or set(chain_seq) == {"X"}:
            continue
        rows = []
        for target in target_sequences:
            aligned_t, aligned_c, identity = align_target_to_chain(str(target.seq), chain_seq)
            matches = sum(1 for a, b in zip(aligned_t, aligned_c) if a == b and a != '-')
            rows.append({"target_id": target.id, "matches": matches, "identity": identity,
                         "aligned_t": aligned_t, "aligned_c": aligned_c})
        rows.sort(key=lambda r: (r["matches"], r["identity"]), reverse=True)
        candidates[chain.id] = rows
    return candidates


def build_mappings(candidates, overrides=None):
    """Mappings, log messages and alignment visuals from scored candidates and optional overrides."""
    overrides = overrides or {}
    mappings, logs, visuals = {}, [], []
    for chain_id, rows in candidates.items():
        if not rows:
            logs.append(f"Chain {chain_id}: no target sequences provided.")
            continue
        choice = overrides.get(chain_id)
        if choice == SKIP:
            logs.append(f"Chain {chain_id}: kept original numbering (excluded by user).")
            continue
        row = next((r for r in rows if r["target_id"] == choice), rows[0])
        mapping = get_mapping_from_alignment(row["aligned_t"], row["aligned_c"])
        if not mapping:
            logs.append(f"Chain {chain_id}: no residues aligned to {row['target_id']}, skipping.")
            continue
        mappings[chain_id] = mapping
        visuals.append((chain_id, row["target_id"],
                        f"Target ({row['target_id']}): {row['aligned_t']}\nChain  ({chain_id})       : {row['aligned_c']}"))
        message = (f"Chain {chain_id}: {'selected' if choice else 'best match is'} {row['target_id']} with "
                   f"{row['matches']} identical residues ({row['identity']:.1%} identity over aligned columns).")
        if row["identity"] < 0.5:
            message += " ⚠️ Low identity: check that this is the right target."
        logs.append(message)
    return mappings, logs, visuals


def prepare_renumbering(structure: Structure, target_sequences: list, overrides=None):
    """Backward-compatible: score all targets, then build mappings honouring overrides."""
    return build_mappings(score_targets(structure, target_sequences), overrides)
```

In `pdb_renumber_section`, cache candidates instead of mappings and add the per-chain choice. Replace from `key = (structure_file.digest, ...)` through the end of the "Renumbering Logs" expander:

```python
    key = (structure_file.digest, tuple((s.id, str(s.seq)) for s in target_sequences))
    if st.session_state.get('pdb_renumber_hash') != key:
        with st.spinner("Matching chains to target sequences..."):
            try:
                st.session_state.pdb_renumber_results = {
                    "candidates": score_targets(parse_structure(structure_file), target_sequences)}
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.text(traceback.format_exc())
                return
            st.session_state.pdb_renumber_hash = key

    candidates = st.session_state.pdb_renumber_results["candidates"]
    st.subheader("Interactive Review & Confirmation")
    overrides = {}
    for chain_id, rows in candidates.items():
        labels = {r["target_id"]: f"{r['target_id']} — {r['identity']:.1%} identity, {r['matches']} identical"
                  for r in rows}
        options = [r["target_id"] for r in rows] + [SKIP]
        overrides[chain_id] = st.selectbox(f"Chain {chain_id} target", options,
                                           format_func=lambda v, labels=labels: labels.get(v, v),
                                           key=f"renumber_target_{chain_id}")
    mappings, logs, visuals = build_mappings(candidates, overrides)
    results = {"mappings": mappings}

    with st.expander("Review Sequence Alignments (PDB vs Target)", expanded=True):
        for chain_id, target_id, aln_text in visuals:
            st.markdown(f"**Chain {chain_id}** matched to **{target_id}**")
            st.text(aln_text)
    with st.expander("Renumbering Logs", expanded=True):
        for msg in logs:
            st.write(msg)
```

The confirm block keeps using `results['mappings']`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass (all earlier renumbering tests plus 3 new).

- [ ] **Step 5: Commit**

```bash
git add modules/pdb_renumber.py tests/test_pdb_renumber.py
git commit -m "ui: per-chain target choice (or exclusion) in PDB renumbering review

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: README matches the app

**Files:**
- Rewrite: `README.md`

**Interfaces:** none

- [ ] **Step 1: Replace `README.md`**

````markdown
# Advanced Sequence Alignment Tool

A Streamlit app for aligning, comparing and annotating DNA and protein sequences and structures.

## Inputs

- Paste FASTA text, or upload **FASTA**, **Clustal**, **GenBank**, **PDB** or **mmCIF** files (several PDB/mmCIF files at once).
- **Newick** trees for visualization.
- Built-in examples: human/mouse globins (protein), the pUC19 plasmid (DNA), a germline-like antibody VH/VL pair.

The sequence type (DNA/protein) is detected automatically and can be overridden.

## Analyses

| Analysis | What it does |
|---|---|
| Pairwise | Global, Local or Overlap (free end gaps) alignment; identity, similarity, gaps; substitutions in reference coordinates; circular DNA support |
| MSA | FAMSA alignment; heatmap; substitutions vs a reference; consensus and most representative sequence; NJ/UPGMA tree |
| Best Match Finder | Rank sequences against a reference by relative score, identity or coverage |
| Sequence Clustering | MSA- or pairwise-based identity; average-linkage, single-linkage or K-Means (PCoA); cluster FASTA export |
| Phylogenetic Tree | Neighbor-Joining or UPGMA on Jukes–Cantor (DNA) / Kimura (protein) corrected distances |
| Antibody Prediction | ANARCI numbering (IMGT, Kabat, Chothia, Martin, AHo, Wolfguy); FR/CDR regions for IMGT, Kabat, Chothia, Martin |
| Translate DNA | Six-frame ORF finder with nucleotide coordinates and selectable codon table |
| Convert Formats | FASTA, Clustal, PHYLIP (relaxed), Stockholm, NEXUS, EMBL, GenBank |
| PDB Residue Renumbering | Renumber chains to match target sequences, with insertion codes and per-chain review |

## Install

Requires Python ≥ 3.10. Antibody numbering also requires HMMER (`hmmscan`).

```bash
git clone https://github.com/dzyla/seq_align.git
cd seq_align
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# HMMER: conda install -c bioconda hmmer   or   sudo apt-get install hmmer
streamlit run app.py
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python precommit_check.py
```

Default gap scores are −10 (open) and −0.5 (extend), the EMBOSS needle convention.
````

- [ ] **Step 2: Verify, then commit**

Run: `python -m pytest -q && python precommit_check.py`
Expected: all pass.

```bash
git add README.md
git commit -m "docs: README describes current inputs, analyses, install and tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
