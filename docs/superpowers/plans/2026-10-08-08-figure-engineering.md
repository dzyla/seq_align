# Alignment Figure, Engineering Aids and Batch UniProt — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the work below to seq_align's MSA and Fetch-by-ID pages, and replace the DNA Tools page with a link to Bio-Bench.
- An ESPript-style publication alignment figure, with DSSP secondary structure from a structure.
- Back-to-consensus suggestions.
- A motif/site map.
- Batch UniProt download by IDs, a search query, or genes + organism.

**Architecture:**
- **Pure modules:**
  - `modules/espript.py`: column classes, secondary structure, rendering.
  - `modules/engineering.py`: consensus, PROSITE conversion, motif search.
  - New batched UniProt functions in `modules/fetch.py`.
- **Streamlit layers:**
  - `modules/alignment_tabs.py`: the Figure, Consensus and Motifs & sites tabs.
  - `modules/uniprot_batch.py`: the sidebar controls and main-area preview.
- **Export:** figure export moves to a shared `modules/figures.py`, which the tree renderer re-uses.

**Tech Stack:**
- Streamlit 1.60.0, matplotlib 3.9.2 (`Figure` + Agg, no pyplot), Biopython 1.84 (`Bio.PDB.DSSP`), pandas, requests.
- `mkdssp` as a system binary: `/home/dzyla/miniconda3/bin/mkdssp` 3.1.2 locally, and the `dssp` apt package on Streamlit Cloud.

**Spec:** `docs/superpowers/specs/2026-10-08-alignment-figure-and-engineering-design.md`

**Prototype provenance:** every code block and diff below comes from a working prototype on a scratch branch of this repo, at base `8265d2a`.
- **Tests:** the full suite passed there (423 tests + Review Focus additions) and `precommit_check.py` passed.
- **Live checks:**
  - The UniProt endpoints were checked against the live API.
  - DSSP was run on 4HHB, 1CRN (PDB and mmCIF) and 1A02.
  - The globin figure was rendered with 4HHB chain B DSSP and inspected visually.

Transcribe the code exactly. Diffs apply with `git apply` from the repository root.

## Global Constraints

- Streamlit stays pinned at 1.60.0. The only new runtime dependency is the system package `dssp` in `packages.txt`; there are no new Python packages.
- Pure computation lives in functions without `st.*` and is unit-tested. Streamlit code lives in `*_section` / tab functions.
- New figure code uses `matplotlib.figure.Figure` + Agg, never pyplot, and no global rcParams mutation (`rc_context` only). Figures are drawn on white.
- Exports:
  - SVG with editable text (`svg.fonttype="none"`).
  - PDF with TrueType fonts (`pdf.fonttype=42`).
  - PNG at 600 dpi, generated on click through a callable `data=`.
- Network code is tested with mocked HTTP. No test may depend on UniProt or RCSB being reachable.
- The no-emoji source test (`tests/test_ui.py::test_no_emoji_in_app_source`) must keep passing. Write glyphs such as the star marker as escapes (`"★"`), never as literal characters.
- Python 3.13/3.14 requirement markers stay unchanged.
- Tests run with `PATH=/home/dzyla/miniconda3/bin:$PATH /home/dzyla/seq_align/.venv/bin/python -m pytest -q`. Run `.venv/bin/python precommit_check.py` before every commit.
- Every commit message ends with `Co-Authored-By: <the model that wrote the code> <noreply@anthropic.com>`.
- Import cycles:
  - `structure_view → pdb_renumber → pairwise`.
  - `clustering → msa → conservation`.
  - `espript` imports `pdb_renumber` and `structures` inside functions.
  - `engineering` imports `conservation` and `pairwise` inside functions.
  - `alignment_tabs` imports `structure_view` inside functions.

## Review Focus

These are inputs that are likely to hit real users. Each one has a test in the named task.

1. **Multi-chain mmCIF structures:** every protein chain of `tests/data/1a02.cif` gets a secondary-structure string exactly as long as its residues (Task 3).
2. **DNA alignments in the new tabs:** consensus works with no BLOSUM score, and the figure renders (Tasks 3 and 4).
3. **A region of one residue:** the figure renders exactly that column (Task 3).
4. **A custom regex that can match empty text (`A*`):** only non-empty hits are reported (Task 4).
5. **An ID list or upload with nothing UniProt-like in it:** nothing is sent to UniProt, and the user is told (Task 2).

---

### Task 1: Shared figure exports; replace DNA Tools with a Bio-Bench link

**Files:**
- Create: `modules/figures.py`
- Modify: `modules/tree_render.py` (re-export from `figures`), `modules/dna.py` (drop the DNA Tools code), `modules/navigation.py`, `app.py`, `README.md`
- Delete: `tests/test_dna_tools.py`
- Test: `tests/test_navigation.py`, `tests/test_figures.py` (new)

**Interfaces:**
- Produces, in `modules/figures.py`:
  - `EXPORT_FORMATS`
  - `quiet_fonts()` (a context manager)
  - `export_bytes(fig, fmt) -> bytes`
  - `preview_png(fig, dpi=150) -> bytes`
- `modules.tree_render` still exposes `export_bytes`, `EXPORT_FORMATS` and `quiet_fonts`, so existing imports keep working.
- `app.BIO_BENCH_URL = "https://dzyla.github.io/toolbox/"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_figures.py`:

```python
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from modules.figures import EXPORT_FORMATS, export_bytes, preview_png


def _fig():
    fig = Figure(figsize=(2, 1))
    FigureCanvasAgg(fig)
    fig.text(0.1, 0.5, "label")
    return fig


def test_shared_exports_and_preview():
    fig = _fig()
    assert set(EXPORT_FORMATS) == {"svg", "pdf", "png"}
    assert b">label<" in export_bytes(fig, "svg")
    assert preview_png(fig).startswith(b"\x89PNG")


def test_tree_render_reexports_shared_helpers():
    import modules.figures as figures
    import modules.tree_render as tree_render
    assert tree_render.export_bytes is figures.export_bytes and tree_render.quiet_fonts is figures.quiet_fonts
```

Apply this diff to `tests/test_navigation.py`:

```diff
diff --git a/tests/test_navigation.py b/tests/test_navigation.py
index 930b56b..30b2e3d 100644
--- a/tests/test_navigation.py
+++ b/tests/test_navigation.py
@@ -26,14 +26,15 @@ def test_structures_offer_viewer():
     assert "Structure Viewer" not in available_analyses(True, False, "Protein")
 
 
-def test_dna_tools_listed():
+def test_dna_tools_page_removed():
     options = available_analyses(True, False, "DNA")
-    assert options[-3:] == ["Translate DNA", "DNA Tools", "Convert Formats"]
+    assert options[-2:] == ["Translate DNA", "Convert Formats"]
+    assert "DNA Tools" not in options
 
 
 def test_single_sequence_omits_multi_sequence_analyses():
     options = available_analyses(True, False, "DNA", n_sequences=1)
-    assert options == ["Translate DNA", "DNA Tools", "Convert Formats"]
+    assert options == ["Translate DNA", "Convert Formats"]
     for name in ("Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Phylogenetic Tree"):
         assert name not in options
 
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_figures.py tests/test_navigation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.figures'`, and "DNA Tools" is still listed.

- [ ] **Step 3: Create `modules/figures.py`**

```python
"""Shared export helpers for publication figures (matplotlib.figure.Figure without pyplot)."""
import logging
from contextlib import contextmanager
from io import BytesIO

import matplotlib

EXPORT_FORMATS = {"svg": "image/svg+xml", "pdf": "application/pdf", "png": "image/png"}


@contextmanager
def quiet_fonts():
    """Silence matplotlib's findfont warnings for listed fallback families that are not installed."""
    logger = logging.getLogger("matplotlib.font_manager")
    old = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(old)


def export_bytes(fig, fmt):
    """SVG (editable text), PDF (embedded TrueType fonts) or 600-dpi PNG bytes of a figure."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown export format {fmt!r}")
    buf = BytesIO()
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}), quiet_fonts():
        fig.savefig(buf, format=fmt, dpi=600 if fmt == "png" else 100, facecolor="white", bbox_inches="tight",
                    pad_inches=0.05)
    return buf.getvalue()


def preview_png(fig, dpi=150):
    """On-screen preview PNG with the same framing as export_bytes."""
    buf = BytesIO()
    with quiet_fonts():
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    return buf.getvalue()
```

- [ ] **Step 4: Make `tree_render` use it**

Apply this diff (it removes the moved helpers and their now-unused imports):

```diff
diff --git a/modules/tree_render.py b/modules/tree_render.py
index 23a9e52..f674f18 100644
--- a/modules/tree_render.py
+++ b/modules/tree_render.py
@@ -2,13 +2,9 @@
 with matplotlib (exported figure) or plotly (interactive view). Uses matplotlib.figure.Figure directly,
 never pyplot, so no global figure registry or GUI backend is involved."""
 import html
-import logging
 import math
-from contextlib import contextmanager
 from dataclasses import dataclass, field
-from io import BytesIO
 
-import matplotlib
 import numpy as np
 import plotly.graph_objects as go
 from matplotlib import font_manager
@@ -16,6 +12,8 @@ from matplotlib.backends.backend_agg import FigureCanvasAgg
 from matplotlib.figure import Figure
 from matplotlib.patches import Polygon
 
+from modules.figures import EXPORT_FORMATS, export_bytes, quiet_fonts  # noqa: F401 (re-exported)
+
 STYLES = {"Classic ink": "classic", "Support dots": "dots", "Clade colour": "clades"}
 # 12 colour-blind-aware colours (Okabe-Ito, then Tol muted/bright), no black or yellow on white
 PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255", "#999933",
@@ -29,7 +27,6 @@ INK, SUPPORT_GREY, LEADER_GREY = "#1a1a1a", "#6b6b6b", "#b5b5b5"
 NIGHT_COLOURS = {INK: "#ece8df", SUPPORT_GREY: "#a59f91", "white": "#1a1916"}  # Explore view, night theme
 SUPPORT_OVERLAP = "Support numbers would overlap; support is shown as dots."
 MIN_FONT, LABEL_LIMIT = 5.0, 500
-EXPORT_FORMATS = {"svg": "image/svg+xml", "pdf": "application/pdf", "png": "image/png"}
 
 
 @dataclass
@@ -580,18 +577,6 @@ def _place_unrooted(geo, style, names, measure, show_labels):
     return d
 
 
-@contextmanager
-def quiet_fonts():
-    """Silence matplotlib's findfont warnings for listed fallback families that are not installed."""
-    logger = logging.getLogger("matplotlib.font_manager")
-    old = logger.level
-    logger.setLevel(logging.ERROR)
-    try:
-        yield
-    finally:
-        logger.setLevel(old)
-
-
 def font_families(drawing):
     """The requested family first, then its fallbacks: exports name it (SVG text keeps 'Arial', ...) while
     glyphs come from the first installed one, which is also the font the labels were measured with."""
@@ -628,17 +613,6 @@ def _render_matplotlib(drawing):
     return fig
 
 
-def export_bytes(fig, fmt):
-    """SVG (editable text), PDF (embedded TrueType fonts) or 600-dpi PNG bytes of a figure."""
-    if fmt not in EXPORT_FORMATS:
-        raise ValueError(f"Unknown export format {fmt!r}")
-    buf = BytesIO()
-    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}), quiet_fonts():
-        fig.savefig(buf, format=fmt, dpi=600 if fmt == "png" else 100, facecolor="white", bbox_inches="tight",
-                    pad_inches=0.05)
-    return buf.getvalue()
-
-
 def _text_center(t):
     """Centre of a (possibly rotated) text box anchored at (x, y) with ha/va, in inches."""
     w, h = t.width, t.height
```

- [ ] **Step 5: Remove DNA Tools**

Apply these diffs to `modules/dna.py` and `modules/navigation.py`:

```diff
diff --git a/modules/dna.py b/modules/dna.py
index 59a6bd2..ccf462b 100644
--- a/modules/dna.py
+++ b/modules/dna.py
@@ -2,13 +2,11 @@ import re
 
 import streamlit as st
 import pandas as pd
-import plotly.graph_objects as go
-from Bio.Restriction import Analysis, CommOnly, RestrictionBatch
 from Bio.Seq import Seq
 
 from modules.seqtools import clean_sequence
 from modules.utils import stale_notice
-from modules.ui import page_header, show_plotly
+from modules.ui import page_header
 
 CODON_TABLES = {
     "1 Standard": 1,
@@ -124,87 +122,3 @@ def find_orfs(sequence, min_len=30, table=1):
                     })
                 segment_start = stop + 1
     return orfs
-
-
-POPULAR_ENZYMES = ["AgeI", "ApaI", "AscI", "AvrII", "BamHI", "BglII", "BsaI", "BsmBI", "BspEI", "ClaI", "EcoRI",
-                   "EcoRV", "HindIII", "KpnI", "MfeI", "MluI", "NcoI", "NdeI", "NheI", "NotI", "NsiI", "PacI",
-                   "PstI", "SacI", "SalI", "SmaI", "SpeI", "SphI", "XbaI", "XhoI"]
-ENZYME_SETS = {"Popular": POPULAR_ENZYMES, "All commercial": None}
-
-
-def reverse_complement(seq):
-    return str(Seq(clean_sequence(seq, "DNA")).reverse_complement())
-
-
-def codon_usage(seq, frame=1, table=1):
-    clean = clean_sequence(seq, "DNA")[frame - 1:]
-    codons = [clean[i:i + 3] for i in range(0, len(clean) - len(clean) % 3, 3)]
-    counts = pd.Series(codons).value_counts()
-    rows = [{"Codon": codon, "AA": str(Seq(codon).translate(table=table)), "Count": int(count)}
-            for codon, count in counts.items()]
-    df = pd.DataFrame(rows, columns=["Codon", "AA", "Count"])
-    df["Fraction of AA"] = df["Count"] / df.groupby("AA")["Count"].transform("sum")
-    return df.sort_values(["AA", "Codon"]).reset_index(drop=True)
-
-
-def restriction_sites(seq, circular=False, enzyme_set="Popular"):
-    names = ENZYME_SETS[enzyme_set]
-    batch = RestrictionBatch(CommOnly) if names is None else RestrictionBatch(names)
-    hits = Analysis(batch, Seq(clean_sequence(seq, "DNA")), linear=not circular).full()
-    rows = [{"Enzyme": str(enzyme), "Site": enzyme.site, "Cuts": len(positions), "Positions": sorted(positions)}
-            for enzyme, positions in hits.items() if positions]
-    df = pd.DataFrame(rows, columns=["Enzyme", "Site", "Cuts", "Positions"])
-    return df.sort_values(["Cuts", "Enzyme"]).reset_index(drop=True)
-
-
-def restriction_map_figure(sites_df, seq_length, enzymes):
-    fig = go.Figure()
-    chosen = sites_df[sites_df["Enzyme"].isin(enzymes)]
-    for k, (_, row) in enumerate(chosen.iterrows()):
-        y = (k % 6) + 1
-        fig.add_trace(go.Scatter(x=row["Positions"], y=[y] * len(row["Positions"]), mode="markers+text",
-                                 text=[row["Enzyme"]] * len(row["Positions"]), textposition="top center",
-                                 marker=dict(symbol="line-ns-open", size=16), name=row["Enzyme"],
-                                 hovertemplate=f"{row['Enzyme']} ({row['Site']}) cuts at %{{x}}<extra></extra>"))
-    fig.add_shape(type="line", x0=1, x1=seq_length, y0=0, y1=0, line=dict(width=4, color="#555555"))
-    fig.update_layout(height=320, showlegend=False, xaxis=dict(title="Position (nt)", range=[0, seq_length + 1]),
-                      yaxis=dict(visible=False, range=[-0.5, 7.5]), margin=dict(t=30))
-    return fig
-
-
-def dna_tools_section(sequences):
-    page_header("DNA tools", "Reverse complement, codon usage and restriction map.")
-    selected_id = st.selectbox("Sequence", [s.id for s in sequences], key="dna_tools_seq")
-    seq = str(next(s for s in sequences if s.id == selected_id).seq)
-    tab_rc, tab_codon, tab_re = st.tabs(["Reverse complement", "Codon usage", "Restriction map"])
-
-    with tab_rc:
-        rc = reverse_complement(seq)
-        st.code(rc)
-        st.download_button("Reverse complement (FASTA)", f">{selected_id}_revcomp\n{rc}\n",
-                           f"{selected_id}_revcomp.fasta", "text/plain", on_click="ignore", icon=":material/download:")
-
-    with tab_codon:
-        c1, c2 = st.columns(2)
-        frame = c1.selectbox("Reading frame", (1, 2, 3), key="codon_frame")
-        table = CODON_TABLES[c2.selectbox("Codon table", list(CODON_TABLES), key="codon_table")]
-        usage = codon_usage(seq, frame, table)
-        st.dataframe(usage.round(3), hide_index=True, width="stretch")
-        st.download_button("Codon usage (CSV)", usage.to_csv(index=False), "codon_usage.csv", "text/csv",
-                           on_click="ignore", icon=":material/download:")
-
-    with tab_re:
-        c1, c2 = st.columns(2)
-        circular = c1.checkbox("Circular (plasmid)", key="re_circular")
-        enzyme_set = c2.selectbox("Enzymes", list(ENZYME_SETS), key="re_set")
-        sites = restriction_sites(seq, circular, enzyme_set)
-        if sites.empty:
-            st.write("No sites found.")
-            return
-        unique = sites[sites["Cuts"] == 1]["Enzyme"].tolist()
-        shown = st.multiselect("Enzymes on map", sites["Enzyme"].tolist(), default=unique[:15], key="re_shown")
-        show_plotly(restriction_map_figure(sites, len(clean_sequence(seq, "DNA")), shown), width="stretch")
-        st.dataframe(sites, hide_index=True, width="stretch")
-        st.caption(f"{len(unique)} unique cutter(s). Positions are the first nucleotide after the top-strand cut.")
-        st.download_button("Restriction sites (CSV)", sites.to_csv(index=False), "restriction_sites.csv",
-                           "text/csv", on_click="ignore", icon=":material/download:")
```

```diff
diff --git a/modules/navigation.py b/modules/navigation.py
index 7d11288..441a43b 100644
--- a/modules/navigation.py
+++ b/modules/navigation.py
@@ -12,7 +12,7 @@ def available_analyses(has_sequences, has_structures, seq_type, n_sequences=None
         if n_sequences != 1:
             options += list(NEEDS_MULTIPLE)
         if seq_type == "DNA":
-            options += ["Translate DNA", "DNA Tools"]
+            options += ["Translate DNA"]
         else:
             options += ["Antibody Prediction", "Sequence Properties"]
         options.append("Convert Formats")
```

Then run `git rm tests/test_dna_tools.py`.

Make these edits in `app.py`:
1. Change `from modules.dna import dna_tools_section, dna_translation_section` to `from modules.dna import dna_translation_section`.
2. In `main()`'s dispatch, delete the two lines `elif alignment_mode == "DNA Tools":` and `dna_tools_section(sequences)`.
3. Directly above `GUIDE = """`, add:
   ```python
   BIO_BENCH_URL = "https://dzyla.github.io/toolbox/"

   ```
   Replace the end of the `GUIDE` string, i.e. the lines
   ```python
   Mutation positions are always reported in reference-sequence coordinates.
   """
   ```
   with:
   ```python
   Mutation positions are always reported in reference-sequence coordinates.

   Quick calculators (restriction maps, codon usage, primers, cloning) live in [Bio-Bench](%s).
   """ % BIO_BENCH_URL
   ```
4. In `_sidebar_footer`, after `appearance_popover()`, add:
   ```python
       st.sidebar.caption(f"Quick calculators (restriction maps, codon usage, primers, cloning): "
                          f"[Bio-Bench]({BIO_BENCH_URL})")
   ```

In `README.md`, delete the table row that starts with `| DNA Tools |`. Under the analyses table, add:

```markdown
Quick bench calculators (restriction maps, codon usage, primers, cloning) live in the companion app
[Bio-Bench](https://dzyla.github.io/toolbox/).
```

- [ ] **Step 6: Run the tests to verify they pass**

Run `grep -rn "dna_tools\|restriction_sites\|codon_usage\|restriction_map_figure" app.py modules tests`. It must print nothing.

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest -q`, then `.venv/bin/python precommit_check.py`.
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add modules/figures.py modules/tree_render.py modules/dna.py modules/navigation.py app.py README.md tests/test_figures.py tests/test_navigation.py
git commit -m "refactor: shared figure exports; DNA Tools page replaced by a Bio-Bench link"
```

---

### Task 2: Batch UniProt download

**Files:**
- Modify: `modules/fetch.py` (batched functions; `FETCH_SOURCES` gains "UniProt batch")
- Create: `modules/uniprot_batch.py`
- Modify: `app.py` (the Fetch-by-ID branch and the GUIDE sentence)
- Test: `tests/test_fetch.py` (diff below), `tests/test_app_flow.py` (append)

**Interfaces:**
- Produces, in `modules/fetch.py`:
  - Constants: `UNIPROT_REST`, `UNIPROT_FIELDS`, `PREVIEW_COLUMNS`, `UNIPROT_HINTS`, `ORGANISMS`, `ACCESSION_CHUNK=500`, `NAME_CHUNK=100`, `GENE_CHUNK=100`, `DEFAULT_QUERY_CAP=2000`, `MAX_QUERY_CAP=10000`.
  - `extract_uniprot_ids(text) -> (accessions, entry_names)`
  - `uniprot_batch_tsv(accessions, entry_names=()) -> (DataFrame, missing)`
  - `uniprot_query_count(query) -> int`
  - `uniprot_query_tsv(query, isoforms=False) -> DataFrame`
  - `gene_query(genes, taxon, reviewed=True) -> str`
  - `uniprot_gene_tsv(genes, taxon, reviewed=True, isoforms=False) -> (DataFrame, missing_genes)`
  - `fetch_uniprot_fasta(ids)`: now batched; it raises `FetchError` naming any missing accession.
- Produces, in `modules/uniprot_batch.py`: `uniprot_batch_sidebar()` and `uniprot_batch_preview()`.
  - The preview lives in `st.session_state.uniprot_preview`.
  - Loading sets `st.session_state.fetched = FetchResult(fasta=...)` for the existing `_load_once` path.
- Live API facts this code relies on (checked 2026-10-08):
  - `/uniprotkb/accessions` takes comma-separated accessions and silently omits unknown ones. One malformed accession makes the whole request fail with 400, so IDs are pattern-checked first.
  - `/uniprotkb/stream?query=(id:A OR id:B)` is **fuzzy**: `NOPE_HUMAN` returned `IGDC4_HUMAN`. Entry-name results are therefore filtered to exact matches.
  - `/uniprotkb/search?size=1` returns the count in the `x-total-results` header.

- [ ] **Step 1: Write the failing tests**

Apply this diff to `tests/test_fetch.py`. It updates the three tests that mocked the old one-request-per-accession URL, adds `headers` to `FakeResponse`, and adds the batch tests:

```diff
diff --git a/tests/test_fetch.py b/tests/test_fetch.py
index 40f5dc4..470d6f3 100644
--- a/tests/test_fetch.py
+++ b/tests/test_fetch.py
@@ -9,11 +9,12 @@ DATA = pathlib.Path(__file__).parent / "data"
 
 
 class FakeResponse:
-    def __init__(self, status=200, text="", content=b"", json_data=None):
+    def __init__(self, status=200, text="", content=b"", json_data=None, headers=None):
         self.status_code = status
         self.text = text
         self.content = content or text.encode()
         self._json = json_data
+        self.headers = headers or {}
 
     def json(self):
         return self._json
@@ -47,8 +48,10 @@ def test_invalid_id_rejected_before_network(monkeypatch):
 
 
 def test_uniprot_fasta(monkeypatch):
-    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/P69905.fasta": FakeResponse(text=">sp|P69905|HBA\nMVLS\n")})
+    calls = install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/accessions":
+                                  FakeResponse(text=">sp|P69905|HBA_HUMAN x\nMVLS\n")})
     assert fetch_any("UniProt", ["P69905"]).fasta.startswith(">sp|P69905")
+    assert calls[0][1]["accessions"] == "P69905" and calls[0][1]["format"] == "fasta"
 
 
 def test_ncbi_uses_efetch(monkeypatch):
@@ -102,8 +105,81 @@ def test_ncbi_rate_limit_and_bad_request(monkeypatch):
 
 
 def test_uniprot_records_not_merged(monkeypatch):
-    install(monkeypatch, {
-        "https://rest.uniprot.org/uniprotkb/P69905.fasta": FakeResponse(text=">a\nMVLS"),
-        "https://rest.uniprot.org/uniprotkb/P68871.fasta": FakeResponse(text=">b\nMVHL"),
-    })
-    assert ">a\nMVLS\n>b\nMVHL" in fetch_any("UniProt", ["P69905", "P68871"]).fasta
+    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/accessions":
+                          FakeResponse(text=">sp|P69905|A x\nMVLS\n>sp|P68871|B y\nMVHL\n")})
+    assert ">sp|P69905|A x\nMVLS\n>sp|P68871|B y\nMVHL" in fetch_any("UniProt", ["P69905", "P68871"]).fasta
+
+
+# --- UniProt batch -------------------------------------------------------------------------------------------
+TSV_HEADER = "Entry\tEntry Name\tProtein names\tGene Names (primary)\tOrganism\tLength\n"
+
+
+def tsv(*rows):
+    return TSV_HEADER + "".join("\t".join(map(str, r)) + "\n" for r in rows)
+
+
+def test_extract_uniprot_ids_from_messy_text():
+    from modules.fetch import extract_uniprot_ids
+    acc, names = extract_uniprot_ids('P69905, hba_human; P68871\tfoo,"Q9Y6K9" P69905 123 P69905-2')
+    assert acc == ["P69905", "P68871", "Q9Y6K9", "P69905-2"] and names == ["HBA_HUMAN"]
+
+
+def test_batch_tsv_chunks_accessions_and_reports_missing(monkeypatch):
+    from modules.fetch import uniprot_batch_tsv
+    calls = install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/accessions":
+                                  FakeResponse(text=tsv(("P69905", "HBA_HUMAN", "Hb alpha", "HBA1", "Human", 142)))})
+    accessions = ["P69905"] + [f"Q{i:05d}" for i in range(1199)]
+    frame, missing = uniprot_batch_tsv(accessions, [])
+    assert len(calls) == 3 and [len(c[1]["accessions"].split(",")) for c in calls] == [500, 500, 200]
+    assert list(frame["Accession"]) == ["P69905"] and frame["Length"].iloc[0] == 142
+    assert len(missing) == 1199 and missing[0] == "Q00000"
+
+
+def test_entry_name_lookup_keeps_exact_matches_only(monkeypatch):
+    # UniProt's id: search is fuzzy: a request for NOPE_HUMAN returned IGDC4_HUMAN in a live probe
+    from modules.fetch import uniprot_batch_tsv
+    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/stream": FakeResponse(text=tsv(
+        ("P69905", "HBA_HUMAN", "Hb alpha", "HBA1", "Human", 142),
+        ("Q8TDY8", "IGDC4_HUMAN", "IGDC4", "IGDC4", "Human", 1250)))})
+    frame, missing = uniprot_batch_tsv([], ["HBA_HUMAN", "NOPE_HUMAN"])
+    assert list(frame["Entry name"]) == ["HBA_HUMAN"] and missing == ["NOPE_HUMAN"]
+
+
+def test_gene_query_and_missing_genes(monkeypatch):
+    from modules.fetch import gene_query, uniprot_gene_tsv
+    assert gene_query(["HBB", "TP53"], 9606) == "(gene_exact:HBB OR gene_exact:TP53) AND organism_id:9606 AND reviewed:true"
+    assert gene_query(["HBB"], 10090, reviewed=False) == "(gene_exact:HBB) AND organism_id:10090"
+    calls = install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/stream": FakeResponse(text=tsv(
+        ("P69905", "HBA_HUMAN", "Hb alpha", "HBA1; HBA2", "Human", 142)))})
+    frame, missing = uniprot_gene_tsv(["hba2", "NOTAGENE"], 9606, isoforms=True)
+    assert missing == ["NOTAGENE"] and calls[0][1]["includeIsoform"] == "true"
+
+
+def test_query_count_reads_header(monkeypatch):
+    from modules.fetch import uniprot_query_count
+    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/search":
+                          FakeResponse(text="Entry\n", headers={"x-total-results": "13"})})
+    assert uniprot_query_count("family:globin") == 13
+
+
+def test_uniprot_bad_query_and_rate_limit_messages(monkeypatch):
+    from modules.fetch import uniprot_query_count
+    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/search": FakeResponse(status=400)})
+    with pytest.raises(FetchError, match="query syntax"):
+        uniprot_query_count("family:(globin")
+    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/search": FakeResponse(status=429)})
+    with pytest.raises(FetchError, match="rate limit"):
+        uniprot_query_count("family:globin")
+
+
+def test_fasta_missing_accession_named(monkeypatch):
+    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/accessions":
+                          FakeResponse(text=">sp|P69905|HBA_HUMAN x\nMVLS\n")})
+    with pytest.raises(FetchError, match="Q00000"):
+        fetch.fetch_uniprot_fasta(["P69905", "Q00000"])
+
+
+def test_extract_ids_with_nothing_valid():
+    # Review Focus: an upload with no UniProt IDs is reported, not sent to UniProt
+    from modules.fetch import extract_uniprot_ids
+    assert extract_uniprot_ids("gene name, 12345, hello world") == ([], [])
```

Append to `tests/test_app_flow.py`:

```python
def test_uniprot_batch_preview_and_load(monkeypatch):
    import pandas as pd
    import modules.uniprot_batch as ub
    frame = pd.DataFrame({"Accession": ["P69905", "P68871"], "Entry name": ["HBA_HUMAN", "HBB_HUMAN"],
                          "Protein": ["Hb alpha", "Hb beta"], "Gene": ["HBA1", "HBB"],
                          "Organism": ["Human", "Human"], "Length": [142, 147]})
    monkeypatch.setattr(ub, "uniprot_batch_tsv", lambda acc, names: (frame, ["Q00000"]))
    monkeypatch.setattr(ub, "fetch_uniprot_fasta", lambda accs: "".join(
        f">sp|{a}|X_HUMAN x\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK\n" for a in accs))
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.selectbox(key="input_format").select("Fetch by ID").run()
    at.selectbox(key="fetch_source").select("UniProt batch").run()
    at.text_area(key="uniprot_ids").input("P69905 P68871 Q00000").run()
    at.button(key="uniprot_search").click().run()
    assert not at.exception
    assert any("Q00000" in w.value for w in at.warning)
    at.button(key="uniprot_load").click().run()
    assert not at.exception
    assert at.metric[0].value == "2"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_fetch.py tests/test_app_flow.py::test_uniprot_batch_preview_and_load -q`
Expected: FAIL. `extract_uniprot_ids` cannot be imported, the old-URL tests 404, and the app has no "UniProt batch" source.

- [ ] **Step 3: Implement the fetch functions**

Apply this diff to `modules/fetch.py`:

```diff
diff --git a/modules/fetch.py b/modules/fetch.py
index 88b9a42..f67e28b 100644
--- a/modules/fetch.py
+++ b/modules/fetch.py
@@ -9,7 +9,7 @@ from modules.structures import StructureFile
 
 TIMEOUT = 30
 HEADERS = {"User-Agent": "seq_align/1.0 (https://github.com/dzyla/seq_align)"}
-FETCH_SOURCES = ("UniProt", "NCBI Protein", "NCBI Nucleotide", "PDB (structure)", "AlphaFold DB (structure)")
+FETCH_SOURCES = ("UniProt", "UniProt batch", "NCBI Protein", "NCBI Nucleotide", "PDB (structure)", "AlphaFold DB (structure)")
 _UNIPROT = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$")
 _PATTERNS = {
     "UniProt": _UNIPROT,
@@ -71,9 +71,132 @@ def _get(url, what, params=None, hints=None):
     return response
 
 
+UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
+UNIPROT_FIELDS = "accession,id,protein_name,gene_primary,organism_name,length"
+PREVIEW_COLUMNS = ["Accession", "Entry name", "Protein", "Gene", "Organism", "Length"]
+UNIPROT_HINTS = {429: "UniProt rate limit reached; wait a few seconds and try again.",
+                 400: "UniProt rejected the request; check the IDs or the query syntax."}
+ORGANISMS = {"Human": 9606, "Mouse": 10090, "Rat": 10116, "Yeast (S. cerevisiae)": 559292, "E. coli K-12": 83333,
+             "Zebrafish": 7955, "Fruit fly": 7227, "C. elegans": 6239, "Arabidopsis": 3702}
+ACCESSION_CHUNK, NAME_CHUNK, GENE_CHUNK = 500, 100, 100
+DEFAULT_QUERY_CAP, MAX_QUERY_CAP = 2000, 10000
+_ENTRY_NAME = re.compile(r"^[A-Z0-9]{1,10}_[A-Z0-9]{1,5}$")
+
+
+def _chunks(items, size):
+    return [items[i:i + size] for i in range(0, len(items), size)]
+
+
+def _short(items, n=5):
+    return ", ".join(items[:n]) + (f" and {len(items) - n} more" if len(items) > n else "")
+
+
+def extract_uniprot_ids(text):
+    """UniProt accessions and entry names found in free text or a CSV/TSV (any separator), upper-cased and
+    de-duplicated in order: (accessions, entry_names)."""
+    accessions, names, seen = [], [], set()
+    for token in re.split(r"[\s,;|\t\"']+", text or ""):
+        token = token.strip().upper()
+        if not token or token in seen:
+            continue
+        if _UNIPROT.match(token):
+            accessions.append(token)
+        elif _ENTRY_NAME.match(token):
+            names.append(token)
+        else:
+            continue
+        seen.add(token)
+    return accessions, names
+
+
+def _tsv_frame(text):
+    import pandas as pd
+    from io import StringIO
+
+    if not text.strip():
+        return pd.DataFrame(columns=PREVIEW_COLUMNS)
+    frame = pd.read_csv(StringIO(text), sep="\t", dtype=str).fillna("")
+    frame.columns = PREVIEW_COLUMNS[:len(frame.columns)]
+    if "Length" in frame:
+        frame["Length"] = pd.to_numeric(frame["Length"], errors="coerce").astype("Int64")
+    return frame
+
+
+def _concat(frames):
+    import pandas as pd
+
+    frames = [f for f in frames if not f.empty]
+    if not frames:
+        return pd.DataFrame(columns=PREVIEW_COLUMNS)
+    return pd.concat(frames, ignore_index=True).drop_duplicates("Accession").reset_index(drop=True)
+
+
+def uniprot_batch_tsv(accessions, entry_names=()):
+    """Preview table for accessions (batched, <= 500 per request) and entry names (exact matches only; UniProt's
+    id: search is fuzzy). Returns (frame, missing IDs in input order)."""
+    frames = []
+    for chunk in _chunks(list(accessions), ACCESSION_CHUNK):
+        response = _get(f"{UNIPROT_REST}/accessions", f"UniProt {_short(chunk)}", hints=UNIPROT_HINTS,
+                        params={"accessions": ",".join(chunk), "format": "tsv", "fields": UNIPROT_FIELDS})
+        frames.append(_tsv_frame(response.text))
+    for chunk in _chunks(list(entry_names), NAME_CHUNK):
+        query = "(" + " OR ".join(f"id:{name}" for name in chunk) + ")"
+        response = _get(f"{UNIPROT_REST}/stream", f"UniProt {_short(chunk)}", hints=UNIPROT_HINTS,
+                        params={"query": query, "format": "tsv", "fields": UNIPROT_FIELDS})
+        frame = _tsv_frame(response.text)
+        frames.append(frame[frame["Entry name"].str.upper().isin(chunk)])
+    frame = _concat(frames)
+    found = set(frame["Accession"].str.upper()) | set(frame["Entry name"].str.upper())
+    missing = [i for i in [*accessions, *entry_names] if i not in found]
+    return frame, missing
+
+
+def uniprot_query_count(query):
+    """Number of UniProtKB entries matching a query (from the X-Total-Results header)."""
+    response = _get(f"{UNIPROT_REST}/search", "UniProt search", hints=UNIPROT_HINTS,
+                    params={"query": query, "size": 1, "format": "tsv", "fields": "accession"})
+    return int(response.headers.get("x-total-results", 0))
+
+
+def uniprot_query_tsv(query, isoforms=False):
+    """Preview table for every entry matching a query (call uniprot_query_count first to respect the cap)."""
+    response = _get(f"{UNIPROT_REST}/stream", "UniProt search", hints=UNIPROT_HINTS,
+                    params={"query": query, "format": "tsv", "fields": UNIPROT_FIELDS,
+                            "includeIsoform": "true" if isoforms else "false"})
+    return _tsv_frame(response.text)
+
+
+def gene_query(genes, taxon, reviewed=True):
+    """UniProt query for exact gene names in one organism."""
+    query = "(" + " OR ".join(f"gene_exact:{g}" for g in genes) + f") AND organism_id:{int(taxon)}"
+    return query + (" AND reviewed:true" if reviewed else "")
+
+
+def uniprot_gene_tsv(genes, taxon, reviewed=True, isoforms=False):
+    """Preview table for gene names in one organism (batched, <= 100 genes per request).
+    Returns (frame, genes with no entry)."""
+    genes = list(dict.fromkeys(g.strip().upper() for g in genes if g.strip()))
+    frame = _concat([uniprot_query_tsv(gene_query(chunk, taxon, reviewed), isoforms)
+                     for chunk in _chunks(genes, GENE_CHUNK)])
+    found = {g.strip().upper() for cell in frame["Gene"] for g in str(cell).split(";")}
+    return frame, [g for g in genes if g not in found]
+
+
 def fetch_uniprot_fasta(ids):
-    texts = [_get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", f"UniProt {acc}").text.rstrip("\n") for acc in ids]
-    return "\n".join(texts) + "\n"
+    """FASTA for UniProt accessions, batched (<= 500 per request). Raises FetchError naming any missing ID."""
+    texts = []
+    for chunk in _chunks(list(ids), ACCESSION_CHUNK):
+        response = _get(f"{UNIPROT_REST}/accessions", f"UniProt {_short(chunk)}", hints=UNIPROT_HINTS,
+                        params={"accessions": ",".join(chunk), "format": "fasta"})
+        texts.append(response.text.strip())
+    fasta = "\n".join(t for t in texts if t) + "\n"
+    returned = {line.split("|")[1].upper() for line in fasta.splitlines() if line.startswith(">") and "|" in line}
+    missing = [i for i in ids if i.upper() not in returned]
+    if missing and len(missing) == len(ids):
+        raise FetchError(f"UniProt returned no entry for {_short(list(ids))}.")
+    if missing:
+        raise FetchError(f"UniProt returned no entry for: {_short(missing)}")
+    return fasta
 
 
 def _accession_key(token):
```

- [ ] **Step 4: Create `modules/uniprot_batch.py`**

```python
"""Batch UniProt download: look up many accessions, a search query, or gene names in one organism, preview the
entries, then load the ticked ones as sequences."""
import hashlib
import re

import streamlit as st

from modules.fetch import (DEFAULT_QUERY_CAP, MAX_QUERY_CAP, ORGANISMS, PREVIEW_COLUMNS, FetchError, FetchResult,
                           extract_uniprot_ids, fetch_uniprot_fasta, uniprot_batch_tsv, uniprot_gene_tsv,
                           uniprot_query_count, uniprot_query_tsv)

MODES = ("IDs", "Search query", "Genes + organism")
OTHER_TAXON = "Other taxon ID"


def _uploaded_text(key):
    upload = st.sidebar.file_uploader("…or upload a list (.txt, .csv, .tsv)", type=["txt", "csv", "tsv"], key=key)
    return upload.getvalue().decode("utf-8", "replace") if upload else ""


def _store(frame, missing=(), missing_label=""):
    st.session_state.uniprot_preview = {"frame": frame, "missing": list(missing), "missing_label": missing_label}


def uniprot_batch_sidebar():
    """Sidebar controls. 'Search UniProt' stores the preview in st.session_state.uniprot_preview."""
    mode = st.sidebar.radio("Look up by", MODES, key="uniprot_mode")
    try:
        if mode == "IDs":
            text = st.sidebar.text_area("UniProt accessions or entry names", key="uniprot_ids", height=120,
                                        placeholder="P69905, P68871\nHBA_HUMAN")
            text = f"{text}\n{_uploaded_text('uniprot_ids_file')}"
            if st.sidebar.button("Search UniProt", key="uniprot_search", icon=":material/search:"):
                accessions, names = extract_uniprot_ids(text)
                if not accessions and not names:
                    raise FetchError("No UniProt accessions or entry names found in the input.")
                with st.spinner(f"Looking up {len(accessions) + len(names)} IDs in UniProt..."):
                    frame, missing = uniprot_batch_tsv(accessions, names)
                _store(frame, missing, "IDs not found")
        elif mode == "Search query":
            query = st.sidebar.text_input("UniProt query", key="uniprot_query",
                                          placeholder="family:globin AND reviewed:true AND organism_id:9606")
            cap = st.sidebar.number_input("Maximum entries", 100, MAX_QUERY_CAP, DEFAULT_QUERY_CAP, 100,
                                          key="uniprot_cap")
            isoforms = st.sidebar.toggle("Include isoforms", key="uniprot_query_isoforms")
            if st.sidebar.button("Search UniProt", key="uniprot_search", icon=":material/search:"):
                if not query.strip():
                    raise FetchError("Enter a UniProt query.")
                with st.spinner("Counting matches..."):
                    count = uniprot_query_count(query)
                if count == 0:
                    raise FetchError("The query matched no UniProt entries.")
                if count > cap:
                    raise FetchError(f"The query matches {count:,} entries, above the limit of {cap:,}. Narrow the "
                                     f"query or raise the limit (up to {MAX_QUERY_CAP:,}).")
                with st.spinner(f"Downloading {count:,} entries..."):
                    _store(uniprot_query_tsv(query, isoforms))
        else:
            text = st.sidebar.text_area("Gene names", key="uniprot_genes", height=100, placeholder="HBB, TP53, EGFR")
            text = f"{text}\n{_uploaded_text('uniprot_genes_file')}"
            organism = st.sidebar.selectbox("Organism", [*ORGANISMS, OTHER_TAXON], key="uniprot_organism")
            taxon = ORGANISMS.get(organism) or st.sidebar.number_input("Taxon ID", 1, 10 ** 8, 9606,
                                                                       key="uniprot_taxon")
            reviewed = st.sidebar.toggle("Reviewed (Swiss-Prot) only", value=True, key="uniprot_reviewed")
            isoforms = st.sidebar.toggle("Include isoforms", key="uniprot_gene_isoforms")
            if st.sidebar.button("Search UniProt", key="uniprot_search", icon=":material/search:"):
                genes = [g for g in re.split(r"[\s,;\t]+", text) if g]
                if not genes:
                    raise FetchError("Enter at least one gene name.")
                with st.spinner(f"Looking up {len(genes)} genes in UniProt..."):
                    frame, missing = uniprot_gene_tsv(genes, taxon, reviewed, isoforms)
                _store(frame, missing, "Genes with no entry")
    except FetchError as e:
        st.sidebar.error(str(e))


def uniprot_batch_preview():
    """Main-area preview: tick entries, then load them (sets st.session_state.fetched for the normal pipeline)."""
    preview = st.session_state.get("uniprot_preview")
    if not preview:
        return
    frame, missing = preview["frame"], preview["missing"]
    with st.expander(f"UniProt results: {len(frame)} entries", expanded=st.session_state.get("fetched") is None):
        if missing:
            shown = ", ".join(missing[:50]) + (" …" if len(missing) > 50 else "")
            st.warning(f"{preview['missing_label']} ({len(missing)}): {shown}")
        if frame.empty:
            st.info("No entries to load.")
            return
        table = frame.copy()
        table.insert(0, "Load", True)
        digest = hashlib.sha1(",".join(frame["Accession"]).encode()).hexdigest()[:10]
        edited = st.data_editor(table, hide_index=True, width="stretch", disabled=PREVIEW_COLUMNS,
                                key=f"uniprot_editor_{digest}")
        chosen = edited.loc[edited["Load"], "Accession"].tolist()
        c1, c2, c3 = st.columns(3)
        if c1.button(f"Load {len(chosen)} selected", key="uniprot_load", type="primary", disabled=not chosen,
                     icon=":material/play_arrow:"):
            try:
                with st.spinner(f"Downloading {len(chosen)} sequences..."):
                    st.session_state.fetched = FetchResult(fasta=fetch_uniprot_fasta(chosen))
            except FetchError as e:
                st.error(str(e))
        c2.download_button("Table (TSV)", edited[edited["Load"]].drop(columns=["Load"]).to_csv(sep="\t", index=False),
                           "uniprot_entries.tsv", "text/tab-separated-values", on_click="ignore",
                           icon=":material/download:", disabled=not chosen)
        c3.download_button("FASTA", lambda: fetch_uniprot_fasta(chosen), "uniprot_entries.fasta", "text/plain",
                           on_click="ignore", icon=":material/download:", disabled=not chosen)
```

- [ ] **Step 5: Wire it into `app.py`**

1. After the `from modules.fetch import ...` line, add `from modules.uniprot_batch import uniprot_batch_preview, uniprot_batch_sidebar`.
2. In `handle_input`, replace the Fetch-by-ID controls (the `ids_text = st.sidebar.text_input(...)` line through the end of its `except FetchError` block) with the block below. Keep the `source = st.sidebar.selectbox(...)` line above it and the `fetched = st.session_state.get("fetched")` lines below it unchanged.
   ```python
           if source == "UniProt batch":
               uniprot_batch_sidebar()
               uniprot_batch_preview()
           else:
               ids_text = st.sidebar.text_input("ID(s)", key="fetch_ids",
                                                placeholder="e.g. P69905, P68871  or  1CRN",
                                                help="Separate several IDs with commas or spaces.")
               if st.sidebar.button("Fetch", key="fetch_button", icon=":material/cloud_download:"):
                   try:
                       with st.spinner(f"Fetching from {source}..."):
                           st.session_state.fetched = fetch_any(source, parse_id_list(ids_text))
                   except FetchError as e:
                       st.sidebar.error(str(e))
   ```
3. In `GUIDE`, replace `Use **Fetch by ID** to pull sequences or structures from UniProt, NCBI, PDB or AlphaFold DB. Structures open in` with these two lines:
   ```
   Use **Fetch by ID** to pull sequences or structures from UniProt, NCBI, PDB or AlphaFold DB; **UniProt batch**
   looks up hundreds of accessions, a UniProt search query, or gene names in one organism at once. Structures open in
   ```

The preview is rendered inside `handle_input` before `fetched` is read. Clicking **Load selected** therefore loads the sequences in the same run, with no `st.rerun()`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest -q`, then `.venv/bin/python precommit_check.py`.
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add modules/fetch.py modules/uniprot_batch.py app.py tests/test_fetch.py tests/test_app_flow.py
git commit -m "feat: batch UniProt download by IDs, search query or genes + organism, with preview and missing-ID report"
```

---

### Task 3: ESPript figure core (`modules/espript.py`)

**Files:**
- Create: `modules/espript.py`
- Modify: `packages.txt` (add `dssp`)
- Test: `tests/test_espript.py`

**Interfaces:**
- Consumes:
  - `modules.figures.quiet_fonts` (Task 1)
  - `modules.pdb_renumber.align_target_to_chain`
  - `modules.structures.parse_structure` and `chain_residue_table`
- Produces:
  - Constants: `GROUPS`, `STYLES = {"Classic ESPript": "classic", "Lab notebook": "lab", "Greyscale": "grey"}`, `PALETTES`, `MAX_CELLS = 30000`
  - `FigureOptions(style, block, font, threshold, consensus, conservation, truncate, underscores)`
  - Column and line helpers:
    - `column_classes(seqs, threshold) -> [(kind, frac, group)]`
    - `consensus_line(seqs, classes)`
  - Secondary structure:
    - `ss_elements(ss) -> [(kind, start, end, label)]`
    - `map_ss_to_alignment(ref_aligned, chain_seq, chain_ss) -> str`
    - `find_mkdssp()`
    - `secondary_structure(sf, chain_id) -> (chain_seq, ss, source, reason)`
  - Layout and labels:
    - `layout_blocks(c0, c1, block)`
    - `region_columns(ref_aligned, start, end) -> (c0, c1)`
    - `cell_count(n_rows, c0, c1)`
    - `display_name(name, options)`
    - `figure_caption(options, ss_source, ss_label=None)`
  - Rendering: `render_alignment_figure(rows, ref_id, ss_by_column=None, options=None, markers=None, ss_label=None, columns=None) -> Figure`
- Text gids the tests rely on: `residue`, `ruler`, `marker`, `legend`, `name`, `consensus`, `ss_element`, `ss_label`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_espript.py`:

```python
import shutil
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from modules.espript import (MAX_CELLS, FigureOptions, cell_count, column_classes, consensus_line, figure_caption,
                             layout_blocks, map_ss_to_alignment, region_columns, render_alignment_figure,
                             secondary_structure, ss_elements)
from modules.figures import export_bytes
from modules.structures import StructureFile

DATA = Path(__file__).parent / "data"
ROWS = [("a", "MKV-LIDE"), ("b", "MKVALVDE"), ("c", "MKIALIEQ")]


def test_column_classes_identical_similar_none():
    kinds = [k for k, _, _ in column_classes([s for _, s in ROWS], 0.7)]
    assert kinds[:2] == ["ident", "ident"]          # M, K everywhere
    assert kinds[2] == "sim"                        # V, V, I: one ILVM group
    assert kinds[3] == "none"                       # -, A, A: 2/3 = 0.67 < 0.7
    # with a gap the column is never identical
    assert column_classes(["A", "-"], 0.5)[0][0] == "sim"


def test_similarity_threshold_boundary():
    seqs = ["DK", "EK", "QK"]           # D, E in DE (2/3), Q not
    assert column_classes(seqs, 0.66)[0][0] == "sim"
    assert column_classes(seqs, 0.7)[0][0] == "none"


def test_consensus_line_case():
    seqs = [s for _, s in ROWS]
    line = consensus_line(seqs, column_classes(seqs, 0.6))
    assert line[:2] == "MK" and line[2] in "vi" and line[2].islower()


def test_ss_elements_labels_and_minimum_lengths():
    ss = "-HHHH--HHH--GGG-EE-TT-IIIIHH-E-"
    els = ss_elements(ss)
    assert [(k, label) for k, _, _, label in els] == [("alpha", "α1"), ("eta", "η1"), ("beta", "β1"), ("turn", "TT"),
                                                      ("alpha", "α2")]
    assert els[0][1:3] == (1, 4)                  # 0-based inclusive
    assert all(label != "α" for *_, label in els)


def test_map_ss_through_gaps_and_offset():
    ref_aligned = "M-KVLA--W"                     # reference residues MKVLAW
    chain_seq, chain_ss = "KVLAW", "HHHE-"        # chain lacks the initiator M
    assert map_ss_to_alignment(ref_aligned, chain_seq, chain_ss) == "  HHHE   "


def test_layout_and_region():
    assert layout_blocks(0, 130, 60) == [(0, 60), (60, 120), (120, 130)]
    assert region_columns("M-KVLA--W", 2, 4) == (2, 5)
    assert region_columns("M-KVLA--W", 0, 99) == (0, 9)
    assert cell_count(3, 2, 5) == 9


@pytest.mark.skipif(shutil.which("mkdssp") is None and shutil.which("dssp") is None, reason="mkdssp not installed")
@pytest.mark.parametrize("name,fmt", [("1crn.pdb", "PDB"), ("1crn.cif", "mmCIF")])
def test_dssp_crambin(name, fmt):
    sf = StructureFile(name=name, data=(DATA / name).read_bytes(), fmt=fmt)
    seq, ss, source, reason = secondary_structure(sf, "A")
    assert source == "DSSP" and len(seq) == len(ss) == 46 and not reason
    kinds = [k for k, *_ in ss_elements(ss)]
    assert kinds.count("alpha") == 2 and kinds.count("beta") == 2


def test_file_records_fallback(monkeypatch):
    import modules.espript as espript
    monkeypatch.setattr(espript, "find_mkdssp", lambda: None)
    text = (DATA / "1crn.pdb").read_text()
    records = ("HELIX    1  H1 ILE A    7  PRO A   19  1                                  13\n"
               "SHEET    1   S1 2 THR A   1  CYS A   4  0\n")
    sf = StructureFile(name="x.pdb", data=(records + text).encode(), fmt="PDB")
    seq, ss, source, reason = secondary_structure(sf, "A")
    assert source == "file records" and "mkdssp" in reason
    assert ss[6:19] == "H" * 13 and ss[:4] == "EEEE"


def test_no_secondary_structure_explains_why(monkeypatch):
    import modules.espript as espript
    monkeypatch.setattr(espript, "find_mkdssp", lambda: None)
    text = "\n".join(l for l in (DATA / "1crn.pdb").read_text().splitlines() if not l.startswith(("HELIX", "SHEET")))
    sf = StructureFile(name="x.pdb", data=text.encode(), fmt="PDB")
    _, ss, source, reason = secondary_structure(sf, "A")
    assert source == "none" and set(ss) == {"-"} and "not available" in reason


def test_render_and_export():
    rows = [("seq_one", "MKVLAWGKV" * 8), ("seq_two", "MKVIAWGRV" * 8)]
    ss = "  HHHHH  " * 8
    fig = render_alignment_figure(rows, "seq_one", ss, FigureOptions(block=30, consensus=True),
                                  markers={"Consensus suggestion": {3}, "Motif site": {10, 11}}, ss_label="1ABC chain A")
    texts = [t for t in fig.axes[0].texts]
    assert sum(t.get_gid() == "residue" for t in texts) == 2 * 72
    assert sum(t.get_gid() == "marker" for t in texts) == 3
    assert any(t.get_text() == "α1" for t in texts)
    svg = export_bytes(fig, "svg")
    assert svg.lstrip().startswith(b"<?xml") and b"Courier New" in svg and b">W<" in svg
    assert export_bytes(fig, "pdf").startswith(b"%PDF")
    assert Image.open(BytesIO(export_bytes(fig, "png"))).info["dpi"][0] == pytest.approx(600, abs=1)


def test_region_render_numbers_follow_reference():
    rows = [("r", "MKVLAWGKVE" * 3), ("s", "MKVLAWGRVE" * 3)]
    fig = render_alignment_figure(rows, "r", None, FigureOptions(), columns=region_columns(rows[0][1], 11, 30))
    ruler = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == "ruler"]
    assert ruler == ["20", "30"]


def test_styles_and_caption():
    for style in ("classic", "lab", "grey"):
        render_alignment_figure(ROWS, "a", None, FigureOptions(style=style))
    caption = figure_caption(FigureOptions(threshold=0.8), "DSSP", "1CRN chain A")
    assert "80%" in caption and "DSSP" in caption and "1CRN chain A" in caption
    assert "No secondary structure" in figure_caption(FigureOptions(), "none")
    assert MAX_CELLS == 30000


@pytest.mark.skipif(shutil.which("mkdssp") is None and shutil.which("dssp") is None, reason="mkdssp not installed")
def test_secondary_structure_every_chain_of_a_complex():
    # Review Focus: multi-chain mmCIF (author chain IDs); every protein chain maps residue for residue
    from modules.structure_view import cached_residue_table
    sf = StructureFile(name="1a02.cif", data=(DATA / "1a02.cif").read_bytes(), fmt="mmCIF")
    table = cached_residue_table(sf.data, sf.fmt, sf.name)
    assert len(table) >= 2
    for chain_id, residues in table.items():
        seq, ss, source, _ = secondary_structure(sf, chain_id)
        assert len(seq) == len(ss) == len(residues) and source in ("DSSP", "file records", "none")


def test_dna_rows_and_single_column_region_render():
    # Review Focus: DNA alignments use the same rules (identity still highlighted); a one-residue region renders
    rows = [("x", "ATGC-A"), ("y", "ATGCTA")]
    fig = render_alignment_figure(rows, "x", None, FigureOptions(), columns=region_columns("ATGC-A", 3, 3))
    assert sum(t.get_gid() == "residue" for t in fig.axes[0].texts) == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_espript.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.espript'`.

- [ ] **Step 3: Implement `modules/espript.py`**

```python
"""ESPript-style publication alignment figure: column classes, secondary structure and rendering.

Pure functions (no Streamlit). The figure is drawn in inch coordinates on a matplotlib Figure without
pyplot, so it exports through modules.figures like the tree figures.
"""
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


def residue_group(residue):
    return next((g for g in GROUPS if residue in g), residue)


def column_classes(seqs, threshold=0.7):
    """Per column (kind, fraction, group): kind 'ident' when every sequence has the same residue and no gap;
    'sim' when at least `threshold` of the sequences share one residue group; otherwise 'none'."""
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
            g = residue_group(c)
            counts[g] = counts.get(g, 0) + 1
        group, k = max(counts.items(), key=lambda kv: kv[1])
        frac = k / n
        out.append(("sim", frac, group) if frac >= threshold else ("none", frac, group))
    return out


def consensus_line(seqs, classes):
    """ESPript-like consensus: upper case for identical columns, lower case for similar, '.' otherwise."""
    line = []
    for col, (kind, _, group) in zip(zip(*seqs), classes):
        residues = [c.upper() for c in col if c not in GAP_CHARS and residue_group(c.upper()) == group]
        top = max(set(residues), key=residues.count) if residues else "."
        line.append(top if kind == "ident" else top.lower() if kind == "sim" else ".")
    return "".join(line)


def ss_elements(ss):
    """Secondary-structure elements in a per-residue DSSP string (8-state; ' ' or '-' = none):
    [(kind, start, end, label)] with 0-based inclusive positions. H/I runs >= 4 -> alpha (α1...),
    G runs >= 3 -> eta (η1...), E runs >= 2 -> beta (β1...), T runs >= 2 -> turn ('TT')."""
    rules = {"H": ("alpha", 4), "I": ("alpha", 4), "G": ("eta", 3), "E": ("beta", 2), "T": ("turn", 2)}
    counters = {"alpha": 0, "eta": 0, "beta": 0}
    elements, i = [], 0
    while i < len(ss):
        c = ss[i]
        j = i
        while j + 1 < len(ss) and (ss[j + 1] == c or (c in "HI" and ss[j + 1] in "HI")):
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


def map_ss_to_alignment(ref_aligned, chain_seq, chain_ss):
    """Transfer per-residue secondary structure of a structure chain onto the columns of the aligned reference.
    Returns a string as long as `ref_aligned`; columns without a mapped residue are ' '."""
    from modules.pdb_renumber import align_target_to_chain  # local: pdb_renumber imports pairwise

    ref = "".join(c for c in ref_aligned if c not in GAP_CHARS)
    out = [" "] * len(ref_aligned)
    if not ref or not chain_seq:
        return "".join(out)
    aligned_ref, aligned_chain, _ = align_target_to_chain(ref, chain_seq)
    ref_to_chain, ri, ci = {}, 0, 0
    for a, b in zip(aligned_ref, aligned_chain):
        if a != "-" and b != "-":
            ref_to_chain[ri] = ci
        ri += a != "-"
        ci += b != "-"
    ri = 0
    for col, c in enumerate(ref_aligned):
        if c in GAP_CHARS:
            continue
        if ri in ref_to_chain:
            s = chain_ss[ref_to_chain[ri]]
            out[col] = " " if s == "-" else s
        ri += 1
    return "".join(out)


def find_mkdssp():
    return shutil.which("mkdssp") or shutil.which("dssp")


def secondary_structure(sf, chain_id):
    """(chain sequence, per-residue SS string, source, reason) for one protein chain of a StructureFile.
    Uses DSSP (mkdssp); falls back to the file's HELIX/SHEET (PDB) or _struct_conf/_struct_sheet_range (mmCIF)
    records; source is 'DSSP', 'file records' or 'none' (then `reason` says why)."""
    from modules.structures import chain_residue_table, parse_structure

    structure = parse_structure(sf)
    table = chain_residue_table(structure)
    if chain_id not in table:
        return "", "", "none", f"Chain {chain_id} has no protein residues."
    residues = table[chain_id]
    seq = "".join(r.aa for r in residues)
    exe = find_mkdssp()
    if exe:
        try:
            ss = _dssp(sf, structure, chain_id, residues, exe)
            if ss.strip("-"):
                return seq, ss, "DSSP", ""
        except Exception:  # fall back to file records below
            pass
    ss = _ss_from_records(sf, chain_id, residues)
    if ss.strip("-"):
        return seq, ss, "file records", "" if exe else "mkdssp is not installed; using the file's HELIX/SHEET records."
    reason = "DSSP is not available and the file has no secondary-structure records." if not exe else \
        "DSSP found no secondary structure and the file has no records."
    return seq, "-" * len(seq), "none", reason


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
    ss = []
    for r in residues:
        code = next((c for c, a, b in ranges if a <= r.number <= b), "-")
        ss.append(code)
    return "".join(ss)


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
    rule = (f"Identical columns: same residue in every sequence, no gaps. Similar residues: at least "
            f"{options.threshold:.0%} of sequences in one group (FYW, ILVM, KRH, DE, STA, NQ, G, P, C).")
    if ss_source == "none" or not ss_label:
        return rule + " No secondary structure shown."
    return rule + f" Secondary structure of {ss_label} from {ss_source}."


def _text_width(renderer, fig, text, size, family, style="normal"):
    t = fig.text(0, 0, text, fontsize=size, family=family, style=style)
    width = t.get_window_extent(renderer).width / fig.dpi
    t.remove()
    return width


def render_alignment_figure(rows, ref_id, ss_by_column=None, options=None, markers=None, ss_label=None,
                            columns=None):
    """ESPript-style figure of aligned (id, sequence) rows, wrapped in blocks.

    ss_by_column: per-column DSSP codes for the reference (see map_ss_to_alignment) or None.
    markers: {legend label: set of alignment columns}; at most two sets are drawn (star, then triangle).
    columns: (c0, c1_exclusive) alignment columns to show; default all."""
    with quiet_fonts():
        return _render(rows, ref_id, ss_by_column, options or FigureOptions(), markers or {}, ss_label, columns)


def _render(rows, ref_id, ss_by_column, o, markers, ss_label, columns):
    pal = PALETTES[o.style]
    seqs = [s for _, s in rows]
    L = len(seqs[0]) if seqs else 0
    c0, c1 = columns or (0, L)
    classes = column_classes(seqs, o.threshold)
    cons = consensus_line(seqs, classes) if o.consensus else ""
    ref = dict(rows)[ref_id]
    resnum, k = [], 0
    for c in ref:
        k += c not in GAP_CHARS
        resnum.append(k if c not in GAP_CHARS else None)
    elements = ss_elements(ss_by_column) if ss_by_column and ss_by_column.strip() else []
    marker_sets = list(markers.items())[:2]
    symbols = ("\u2605", "\u25b2")  # black star, black up-pointing triangle
    marker_colours = (pal["ident_bg"], pal["marker2"])

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
            for kind, a, b, label in elements:
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
                if kind != "turn" and a >= b0:
                    ax.text((xa + xb) / 2, yss - rh * 0.75, label, ha="center", va="center", fontsize=font * 0.8,
                            color=pal["ink"], family=LABEL_FONTS, gid="ss_element")
        yr = y0 + head - rh * 0.9
        for j in range(b0, b1):
            if resnum[j] and resnum[j] % 10 == 0:
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
        for (label, cols), symbol, colour in zip(marker_sets, symbols, marker_colours):
            for j in sorted(cols):
                if b0 <= j < b1:
                    ax.text(X(j) + cw / 2, y + rh / 2, symbol, ha="center", va="center", fontsize=font,
                            color=colour, family=SYMBOL_FONTS, gid="marker")
            y += rh
    if marker_sets:
        x, y = name_w, height - legend_h / 2 - 0.05
        for (label, _), symbol, colour in zip(marker_sets, symbols, marker_colours):
            ax.text(x, y, symbol, fontsize=font, color=colour, family=SYMBOL_FONTS, va="center", gid="legend")
            ax.text(x + cw * 1.3, y, label, fontsize=font * 0.85, family=LABEL_FONTS, va="center", color="#333333",
                    gid="legend")
            x += cw * 1.3 + _text_width(renderer, probe, label, font * 0.85, LABEL_FONTS) + 0.3
    return fig
```

- [ ] **Step 4: Add DSSP for Streamlit Cloud**

`packages.txt` becomes:

```
hmmer
mafft
dssp
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_espript.py -q`
Expected: all pass. The DSSP tests run because `mkdssp` is on the PATH above. Then run the full suite and precommit: green.

- [ ] **Step 6: Commit**

```bash
git add modules/espript.py tests/test_espript.py packages.txt
git commit -m "feat: ESPript-style alignment figure core with DSSP secondary structure and file-record fallback"
```

---

### Task 4: Engineering aids core (`modules/engineering.py`)

**Files:**
- Create: `modules/engineering.py`
- Test: `tests/test_engineering.py`

**Interfaces:**
- Consumes: `modules.conservation.column_conservation`, `modules.pairwise.load_matrix` (both imported inside the function).
- Produces:
  - Constants: `MOTIF_LIBRARY` (10 motifs, PROSITE syntax), `FEW_SEQUENCES = 20`.
  - `consensus_suggestions(rows, target_id, min_freq=0.6, min_occupancy=0.5, seq_type="Protein") -> DataFrame` with columns `Position, Suggestion, Target, Consensus, Consensus frequency, Conservation, BLOSUM62, Sequences, Score, Column`.
  - `prosite_to_regex(pattern) -> str`, which raises `ValueError`.
  - `compile_motifs(motifs, syntax="prosite") -> {name: compiled}`, which raises `ValueError` naming the motif.
  - `find_motifs(rows, compiled) -> DataFrame[Sequence, Motif, Start, End, Match, Column]`
  - `site_summary(hits, rows, ref_id) -> DataFrame[Motif, Reference position, Sequences with site, Fraction, Sequence IDs, Column]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engineering.py`:

```python
import pytest

from modules.engineering import (MOTIF_LIBRARY, compile_motifs, consensus_suggestions, find_motifs, prosite_to_regex,
                                 site_summary)

ROWS = [("t", "MKVAE-LW"), ("a", "MKILEALW"), ("b", "MKILEGLW"), ("c", "MRILE-LF"), ("d", "MKILE-LW")]


def test_consensus_mismatch_suggested_in_target_numbering():
    table = consensus_suggestions(ROWS, "t")
    assert list(table["Suggestion"])[:2] == ["V3I", "A4L"] or set(table["Suggestion"]) >= {"V3I", "A4L"}
    row = table.set_index("Suggestion").loc["V3I"]
    assert row["Consensus frequency"] == pytest.approx(0.8) and row["Sequences"] == 5


def test_consensus_respects_frequency_occupancy_and_gaps():
    table = consensus_suggestions(ROWS, "t", min_freq=0.85)
    assert "V3I" not in set(table["Suggestion"])          # 0.8 < 0.85
    gappy = [("t", "MKA"), ("a", "--L"), ("b", "--L"), ("c", "--L"), ("d", "MKL")]
    assert set(consensus_suggestions(gappy, "t")["Suggestion"]) == {"A3L"}  # columns 1-2: occupancy 0.4 < 0.5
    with_gap = [("t", "M-W"), ("a", "MKW"), ("b", "MKW")]
    assert consensus_suggestions(with_gap, "t").empty     # target gap is never suggested


def test_consensus_sorted_by_score():
    table = consensus_suggestions(ROWS, "t")
    assert list(table["Score"]) == sorted(table["Score"], reverse=True)


@pytest.mark.parametrize("pattern,regex", [
    ("N-{P}-[ST]-{P}", "N[^P][ST][^P]"), ("R-x-[KR]-R", "R.[KR]R"), ("x(2,4)-C", ".{2,4}C"),
    ("C-x(2)-C", "C.{2}C"), ("<M-x(2)", "^M.{2}"), ("[ST]-x-[RK]>", "[ST].[RK]$"), ("N-G.", "NG"),
])
def test_prosite_to_regex(pattern, regex):
    assert prosite_to_regex(pattern) == regex


@pytest.mark.parametrize("bad", ["", "N-{P", "N--G", "1-2", "[st]"])
def test_prosite_invalid(bad):
    with pytest.raises(ValueError):
        prosite_to_regex(bad)


def test_library_compiles():
    assert set(compile_motifs(MOTIF_LIBRARY)) == set(MOTIF_LIBRARY)


def test_find_motifs_overlapping_case_insensitive_and_columns():
    rows = [("x", "aa-aA"), ("y", "-AAAA")]
    hits = find_motifs(rows, compile_motifs({"AA": "A-A"}))
    x = hits[hits["Sequence"] == "x"]
    assert list(x["Start"]) == [1, 2, 3] and list(x["Column"]) == [0, 1, 3]
    y = hits[hits["Sequence"] == "y"]
    assert list(y["Start"]) == [1, 2, 3] and list(y["Column"]) == [1, 2, 3]


def test_regex_syntax_and_bad_regex():
    assert "C..C" in compile_motifs({"c": "C..C"}, "regex")["c"].pattern
    with pytest.raises(ValueError, match="bad"):
        compile_motifs({"bad": "C(("}, "regex")


def test_site_summary_counts_and_reference_position():
    rows = [("r", "-NGT"), ("s", "ANGT"), ("t", "ADGT")]
    hits = find_motifs(rows, compile_motifs({"NG": "N-G"}))
    summary = site_summary(hits, rows, "r")
    assert summary.iloc[0]["Sequences with site"] == "2/3" and summary.iloc[0]["Reference position"] == 1
    assert site_summary(hits.iloc[0:0], rows, "r").empty


def test_dna_alignment_consensus_has_no_blosum():
    # Review Focus: DNA alignments must not crash the protein-oriented tabs
    rows = [("t", "ATGA"), ("a", "ATGC"), ("b", "ATGC"), ("c", "ATGC")]
    table = consensus_suggestions(rows, "t", seq_type="DNA")
    assert list(table["Suggestion"]) == ["A4C"] and table["BLOSUM62"].isna().all()


def test_zero_length_regex_matches_are_ignored():
    # Review Focus: a custom regex that can match the empty string must not flood the table
    hits = find_motifs([("x", "CAAC")], compile_motifs({"z": "A*"}, "regex"))
    assert list(hits["Match"]) == ["AA", "A"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_engineering.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.engineering'`.

- [ ] **Step 3: Implement `modules/engineering.py`**

```python
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
    Ranked by consensus frequency x conservation (descending). Positions are in target numbering."""
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_engineering.py -q`
Expected: all pass. Then run the full suite and precommit: green.

- [ ] **Step 5: Commit**

```bash
git add modules/engineering.py tests/test_engineering.py
git commit -m "feat: back-to-consensus suggestions and PROSITE/regex motif search with family site summary"
```

---

### Task 5: MSA tabs — Figure, Consensus, Motifs & sites

**Files:**
- Create: `modules/alignment_tabs.py`
- Modify: `modules/msa.py` (tab list and calls), `README.md`
- Test: `tests/test_app_flow.py` (append)

**Interfaces:**
- Consumes:
  - Task 3: `modules.espript` (`STYLES`, `FigureOptions`, `MAX_CELLS`, `cell_count`, `figure_caption`, `map_ss_to_alignment`, `region_columns`, `render_alignment_figure`, `secondary_structure`).
  - Task 4: `modules.engineering`.
  - Task 1: `modules.figures` (`export_bytes`, `preview_png`).
  - `modules.structure_view`, imported inside functions: `_add_structure_controls`, `available_structures`, `cached_residue_table`, `rank_chains`, `structure_highlight_panel`.
- Produces:
  - `figure_tab(rows, ref_id, seq_type)`, `consensus_tab(rows, ref_id, seq_type)`, `motifs_tab(rows, ref_id)`.
  - `consensus_settings(rows, ref_id)` and `motif_settings()`. The Figure tab reads the other tabs' settings through these for its markers, so markers are always current.
- Widget keys:
  - Figure tab: `fig_rows`, `fig_region`, `fig_style`, `fig_block`, `fig_font`, `fig_threshold`, `fig_truncate`, `fig_consensus`, `fig_conservation`, `fig_underscores`, `fig_mark_cons`, `fig_mark_motifs`, `fig_ss_use`, `fig_ss_structure`, `fig_ss_chain`.
  - Other tabs: `cons_target`, `cons_min_freq`, `motif_select`, `motif_syntax`, `motif_custom`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_app_flow.py`:

```python
def test_msa_figure_consensus_and_motif_tabs():
    at = _load_globins_msa()
    seg = lambda key: next(w for w in at.get("button_group") if w.key == key)  # noqa: E731
    for style in ("Classic ESPript", "Lab notebook", "Greyscale"):
        seg("fig_style").set_value(style).run()
        assert not at.exception, (style, at.exception)
    labels = [d.label for d in at.get("download_button")]
    assert {"SVG", "PDF", "PNG (600 dpi)"} <= set(labels)
    assert any("Identical columns" in c.value for c in at.caption)
    assert "Consensus suggestions (CSV)" in labels and "Sites (CSV)" in labels
    at.text_input(key="motif_custom").input("C-x(2)-{P").run()
    assert not at.exception and any("Invalid motif" in e.value for e in at.error)


def test_msa_figure_with_structure_secondary_structure():
    import pathlib
    from modules.structures import StructureFile
    at = _load_globins_msa()
    data = (pathlib.Path(__file__).parent / "data" / "1crn.pdb").read_bytes()
    at.session_state["extra_structures"] = {"1crn.pdb": StructureFile(name="1crn.pdb", data=data, fmt="PDB")}
    at.run()
    at.radio(key="fig_ss_use").set_value("From a structure").run()
    assert not at.exception
    assert any("1crn chain A" in c.value for c in at.caption)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_app_flow.py -k "figure_consensus or figure_with_structure" -q`
Expected: FAIL. There is no button group with key `fig_style`.

- [ ] **Step 3: Create `modules/alignment_tabs.py`**

```python
"""MSA tabs: publication figure (ESPript style), back-to-consensus suggestions and motifs & sites."""
import streamlit as st

from modules.engineering import (FEW_SEQUENCES, MOTIF_LIBRARY, compile_motifs, consensus_suggestions, find_motifs,
                                 site_summary)
from modules.espript import (MAX_CELLS, STYLES, FigureOptions, cell_count, figure_caption, map_ss_to_alignment,
                             region_columns, render_alignment_figure, secondary_structure)
from modules.figures import export_bytes, preview_png

CONSENSUS_COLOUR, MOTIF_COLOUR = "#d6312b", "#2a5db0"


def consensus_settings(rows, ref_id):
    """Current Consensus-tab settings (also read by the Figure tab for its star markers)."""
    ids = [i for i, _ in rows]
    target = st.session_state.get("cons_target", ref_id)
    return (target if target in ids else ref_id), st.session_state.get("cons_min_freq", 0.6)


def motif_settings():
    """Current Motifs-tab settings: ({name: pattern}, syntax) for the chosen library motifs plus a custom one."""
    chosen = st.session_state.get("motif_select", ["N-glycosylation sequon"])
    motifs = {name: MOTIF_LIBRARY[name] for name in chosen if name in MOTIF_LIBRARY}
    custom = (st.session_state.get("motif_custom") or "").strip()
    syntax = "prosite" if st.session_state.get("motif_syntax", "PROSITE pattern") == "PROSITE pattern" else "regex"
    if custom:
        motifs["Custom motif"] = custom
    return motifs, syntax


def _compiled_motifs(motifs, syntax):
    """Library motifs are always PROSITE; only the custom motif follows the chosen syntax."""
    library = {k: v for k, v in motifs.items() if k != "Custom motif"}
    compiled = compile_motifs(library, "prosite")
    if "Custom motif" in motifs:
        compiled.update(compile_motifs({"Custom motif": motifs["Custom motif"]}, syntax))
    return compiled


@st.cache_data(max_entries=8, show_spinner=False)
def cached_secondary_structure(name, data, fmt, chain_id):
    from modules.structures import StructureFile

    return secondary_structure(StructureFile(name=name, data=data, fmt=fmt), chain_id)


@st.cache_data(max_entries=8, show_spinner=False)
def figure_outputs(rows, ref_id, ss_by_column, options, markers, ss_label, columns):
    """Preview PNG, SVG and PDF of the alignment figure (cached on every input)."""
    fig = render_alignment_figure(list(rows), ref_id, ss_by_column, FigureOptions(**dict(options)),
                                  {k: set(v) for k, v in markers}, ss_label, columns)
    return {"preview": preview_png(fig), "svg": export_bytes(fig, "svg"), "pdf": export_bytes(fig, "pdf")}


def _png_600(rows, ref_id, ss_by_column, options, markers, ss_label, columns):
    fig = render_alignment_figure(list(rows), ref_id, ss_by_column, FigureOptions(**dict(options)),
                                  {k: set(v) for k, v in markers}, ss_label, columns)
    return export_bytes(fig, "png")


def _ss_controls(ref_seq):
    """Secondary-structure source for the figure: a structure (loaded, uploaded, or fetched by PDB ID / UniProt
    accession for AlphaFold) and chain, or none. Returns (chain_seq, ss, source, label, reason)."""
    from modules.structure_view import _add_structure_controls, available_structures, cached_residue_table, rank_chains

    use = st.radio("Secondary structure", ["From a structure", "None"], horizontal=True, key="fig_ss_use",
                   index=0 if available_structures() else 1)
    if use == "None":
        return "", "", "none", None, ""
    _add_structure_controls("fig_ss")
    structures = available_structures()
    if not structures:
        st.caption("Load, upload or fetch a structure to show its secondary structure.")
        return "", "", "none", None, ""
    c1, c2 = st.columns(2)
    name = c1.selectbox("Structure", list(structures), key="fig_ss_structure")
    sf = structures[name]
    table = cached_residue_table(sf.data, sf.fmt, sf.name)
    if not table:
        st.warning("No protein chains in this structure.")
        return "", "", "none", None, ""
    ranked = rank_chains(ref_seq, table)
    chain_id = c2.selectbox("Chain", [c for c, _ in ranked], key="fig_ss_chain",
                            format_func=lambda c: f"{c} ({dict(ranked)[c]:.1%} identity)")
    seq, ss, source, reason = cached_secondary_structure(sf.name, sf.data, sf.fmt, chain_id)
    return seq, ss, source, f"{sf.stem} chain {chain_id}", reason


def figure_tab(rows, ref_id, seq_type):
    """Publication alignment figure in the ESPript style."""
    ids = [i for i, _ in rows]
    st.subheader("Publication figure")
    chosen = st.multiselect("Sequences (in this order)", ids, default=[ref_id] + [i for i in ids if i != ref_id],
                            key="fig_rows")
    if not chosen:
        st.info("Choose at least one sequence.")
        return
    shown = [(i, dict(rows)[i]) for i in chosen]
    shown_ref = ref_id if ref_id in chosen else chosen[0]
    shown_ref_aligned = dict(rows)[shown_ref]
    n_res = sum(c not in "-." for c in shown_ref_aligned)
    c1, c2, c3 = st.columns(3)
    start, end = c1.slider(f"Region ({shown_ref} numbering)", 1, max(n_res, 2), (1, max(n_res, 2)),
                           key="fig_region")
    style_label = c2.segmented_control("Style", list(STYLES), default="Classic ESPript", required=True,
                                       key="fig_style")
    block = c3.slider("Columns per block", 40, 120, 60, 10, key="fig_block")
    c1, c2, c3 = st.columns(3)
    font = c1.slider("Font size (pt)", 6, 12, 8, key="fig_font")
    threshold = c2.slider("Similarity threshold", 0.5, 1.0, 0.7, 0.05, key="fig_threshold")
    truncate = c3.number_input("Truncate names at (0 = off)", 0, 60, 0, key="fig_truncate")
    c1, c2, c3, c4, c5 = st.columns(5)
    consensus = c1.toggle("Consensus line", key="fig_consensus")
    conservation = c2.toggle("Conservation bar", value=True, key="fig_conservation")
    underscores = c3.toggle("Underscores as spaces", value=True, key="fig_underscores")
    mark_cons = c4.toggle("Mark consensus suggestions", value=True, key="fig_mark_cons")
    mark_motifs = c5.toggle("Mark motif sites", value=True, key="fig_mark_motifs")

    seq, ss, source, ss_label, reason = _ss_controls("".join(c for c in shown_ref_aligned if c not in "-."))
    if reason:
        st.caption(reason)
    ss_by_column = map_ss_to_alignment(shown_ref_aligned, seq, ss) if source != "none" else None

    markers = {}
    if mark_cons:
        target, min_freq = consensus_settings(rows, ref_id)
        suggestions = consensus_suggestions(rows, target, min_freq, seq_type=seq_type)
        if not suggestions.empty:
            markers["Consensus suggestion"] = tuple(sorted(suggestions["Column"]))
    if mark_motifs:
        motifs, syntax = motif_settings()
        try:
            hits = find_motifs(rows, _compiled_motifs(motifs, syntax)) if motifs else None
        except ValueError:
            hits = None
        if hits is not None and not hits.empty:
            markers["Motif site"] = tuple(sorted(set(hits["Column"])))

    c0, c1_ = region_columns(shown_ref_aligned, start, end)
    cells = cell_count(len(shown), c0, c1_)
    if cells > MAX_CELLS:
        st.warning(f"The figure would show {cells:,} residues (limit {MAX_CELLS:,}). Narrow the region or "
                   "choose fewer sequences.")
        return
    options = FigureOptions(style=STYLES[style_label], block=block, font=float(font), threshold=float(threshold),
                            consensus=consensus, conservation=conservation, truncate=int(truncate),
                            underscores=underscores)
    args = (tuple(shown), shown_ref, ss_by_column, tuple(vars(options).items()),
            tuple((k, tuple(v)) for k, v in markers.items()), ss_label, (c0, c1_))
    with st.spinner("Drawing the figure..."):
        out = figure_outputs(*args)
    st.image(out["preview"])
    st.caption(figure_caption(options, source, ss_label))
    d1, d2, d3 = st.columns(3)
    d1.download_button("SVG", out["svg"], "alignment_figure.svg", "image/svg+xml", icon=":material/download:",
                       on_click="ignore", width="stretch", key="fig_dl_svg")
    d2.download_button("PDF", out["pdf"], "alignment_figure.pdf", "application/pdf", icon=":material/download:",
                       on_click="ignore", width="stretch", key="fig_dl_pdf")
    d3.download_button("PNG (600 dpi)", lambda: _png_600(*args), "alignment_figure.png", "image/png",
                       icon=":material/download:", on_click="ignore", width="stretch", key="fig_dl_png")


def consensus_tab(rows, ref_id, seq_type):
    """Back-to-consensus mutation suggestions for one sequence."""
    from modules.structure_view import structure_highlight_panel

    ids = [i for i, _ in rows]
    st.subheader("Back-to-consensus suggestions")
    st.caption("Positions where the chosen sequence differs from a strongly conserved column consensus: a common "
               "source of stabilising mutations. Ranked by consensus frequency × conservation.")
    c1, c2 = st.columns(2)
    target = c1.selectbox("Sequence", ids, index=ids.index(ref_id), key="cons_target")
    min_freq = c2.slider("Minimum consensus frequency", 0.4, 0.95, 0.6, 0.05, key="cons_min_freq")
    if len(rows) < FEW_SEQUENCES:
        st.warning(f"Consensus from fewer than {FEW_SEQUENCES} sequences is unreliable "
                   f"({len(rows)} sequences in this alignment).")
    table = consensus_suggestions(rows, target, min_freq, seq_type=seq_type)
    if table.empty:
        st.info("No suggestions: the sequence matches the consensus at every qualifying column.")
        return
    st.dataframe(table.drop(columns=["Column"]), hide_index=True, width="stretch")
    st.download_button("Consensus suggestions (CSV)", table.drop(columns=["Column"]).to_csv(index=False),
                       "consensus_suggestions.csv", "text/csv", on_click="ignore", icon=":material/download:")
    target_seq = "".join(c for c in dict(rows)[target] if c not in "-.")
    colors = {int(r.Position): (CONSENSUS_COLOUR, r.Suggestion) for r in table.itertuples()}
    structure_highlight_panel(target_seq, colors, key="cons_3d", title="Show suggestions on structure (Mol*)")


def motifs_tab(rows, ref_id):
    """Motif and site map across the family."""
    from modules.structure_view import structure_highlight_panel

    st.subheader("Motifs and sites")
    st.multiselect("Built-in motifs", list(MOTIF_LIBRARY), default=["N-glycosylation sequon"], key="motif_select",
                   format_func=lambda k: f"{k}  ({MOTIF_LIBRARY[k]})")
    c1, c2 = st.columns([1, 3])
    c1.radio("Custom motif syntax", ["PROSITE pattern", "Regular expression"], key="motif_syntax")
    c2.text_input("Custom motif (optional)", key="motif_custom", placeholder="e.g. C-x(2)-C  or  C..C")
    motifs, syntax = motif_settings()
    if not motifs:
        st.info("Choose a built-in motif or enter a custom one.")
        return
    try:
        hits = find_motifs(rows, _compiled_motifs(motifs, syntax))
    except ValueError as e:
        st.error(f"Invalid motif: {e}")
        return
    if hits.empty:
        st.info("No matches in any sequence.")
        return
    summary = site_summary(hits, rows, ref_id)
    st.markdown("**Sites across the family**")
    st.dataframe(summary.drop(columns=["Column"]), hide_index=True, width="stretch")
    st.markdown("**Hits per sequence**")
    st.dataframe(hits.drop(columns=["Column"]), hide_index=True, width="stretch")
    d1, d2 = st.columns(2)
    d1.download_button("Sites (CSV)", summary.drop(columns=["Column"]).to_csv(index=False), "motif_sites.csv",
                       "text/csv", on_click="ignore", icon=":material/download:")
    d2.download_button("Hits (CSV)", hits.drop(columns=["Column"]).to_csv(index=False), "motif_hits.csv",
                       "text/csv", on_click="ignore", icon=":material/download:")
    ref_seq = "".join(c for c in dict(rows)[ref_id] if c not in "-.")
    colors = {int(r[1]): (MOTIF_COLOUR, f"{r[0]} ({r[2]})")
              for r in summary[["Motif", "Reference position", "Sequences with site"]].itertuples(index=False)
              if r[1] is not None and r[1] == r[1]}
    structure_highlight_panel(ref_seq, colors, key="motif_3d", title="Show sites on structure (Mol*)")
```

- [ ] **Step 4: Add the tabs to the MSA page**

Apply this diff to `modules/msa.py`:

```diff
diff --git a/modules/msa.py b/modules/msa.py
index 69d93e0..c7b1cb3 100644
--- a/modules/msa.py
+++ b/modules/msa.py
@@ -15,6 +15,7 @@ from io import StringIO
 from collections import Counter
 import pandas as pd
 from typing import List, Dict, Tuple
+from modules.alignment_tabs import consensus_tab, figure_tab, motifs_tab
 from modules.utils import have_params_changed, stale_notice
 from modules.viz import alignment_to_codes, plot_msa
 from modules.seqtools import clean_sequence, fingerprint
@@ -107,7 +108,8 @@ def msa_section(sequences, seq_type):
         st.error(str(e))
         mutations = {}
 
-    tab_titles = ["Alignment & Heatmap", "Point Mutations", "Variant Table", "Conservation", "Phylogenetic Tree"]
+    tab_titles = ["Alignment & Heatmap", "Figure", "Point Mutations", "Variant Table", "Conservation", "Consensus",
+                  "Motifs & sites", "Phylogenetic Tree"]
     if calculate_representative:
         tab_titles.insert(tab_titles.index("Variant Table") + 1, "Representative Sequence")
     tabs = st.tabs(tab_titles)
@@ -126,6 +128,15 @@ def msa_section(sequences, seq_type):
             st.download_button("Percent identity matrix (CSV)", identity_matrix_frame(view).to_csv(),
                                "percent_identity.csv", "text/csv", on_click="ignore", icon=":material/download:")
 
+    with tabs[tab_titles.index("Figure")]:
+        figure_tab(rows, ref_seq_id, seq_type)
+
+    with tabs[tab_titles.index("Consensus")]:
+        consensus_tab(rows, ref_seq_id, seq_type)
+
+    with tabs[tab_titles.index("Motifs & sites")]:
+        motifs_tab(rows, ref_seq_id)
+
     with tabs[tab_titles.index("Point Mutations")]:
         st.subheader("Point mutations relative to the reference")
         if any(mutations.values()):
```

The tabs use `rows`, the untrimmed, coordinate-bearing alignment after the redundancy filter. That is what the spec requires.

- [ ] **Step 5: README**

Add these rows to the analyses table, after the "Similarity matrix" row:

```markdown
| Alignment figure | ESPript-style publication figure on the MSA page: Classic ESPript, Lab notebook or Greyscale; secondary structure from a loaded, uploaded or fetched structure (DSSP, or the file's HELIX/SHEET records); region and sequence selection; consensus line, conservation bar and markers; SVG (editable text), PDF or 600-dpi PNG |
| Consensus suggestions | Back-to-consensus mutations for any sequence of the MSA, ranked by consensus frequency × conservation, with BLOSUM62, CSV export and 3D view |
| Motifs & sites | Built-in sites (N-glycosylation sequon, furin, TEV, thrombin, PreScission, Factor Xa, enterokinase, NG/DG, cysteines) plus your own PROSITE pattern or regex; how many sequences carry each site; CSV export and 3D view |
| UniProt batch | Fetch by ID → UniProt batch: hundreds of accessions or entry names (paste or upload), a UniProt search query (with a size limit), or gene names + organism; preview, tick and load |
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest -q`, then `.venv/bin/python precommit_check.py`.
Expected: all green, about 430 tests.

- [ ] **Step 7: Commit**

```bash
git add modules/alignment_tabs.py modules/msa.py README.md tests/test_app_flow.py
git commit -m "feat: MSA Figure, Consensus and Motifs & sites tabs with markers and 3D views"
```
