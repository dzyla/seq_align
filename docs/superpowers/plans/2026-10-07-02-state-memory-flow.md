# Plan 02 — State, Memory and Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Results always belong to the data on screen, expensive work runs once, and the session holds only what it needs.

**Architecture:** Input loading is keyed by a content hash, so parsing reruns only when the bytes change. A dataset fingerprint resets every analysis result when the data changes. Expensive pure functions (FAMSA, PNG rendering, mutation tables) go behind `st.cache_data` with bounded `max_entries`. The session stores compact data (aligned `(id, seq)` tuples, raw structure bytes) and derives views on demand. Uploaded structures live in a structure store that the renumbering tool (and later the Mol\* viewer) reads.

**Tech Stack:** Streamlit 1.60 (`st.cache_data`, `streamlit.testing.v1.AppTest`), NumPy, Matplotlib, Plotly, Biopython.

**Spec:** `docs/superpowers/specs/2026-10-07-codebase-review.md` (items C4, C5, C7, C8, M1–M5)

**Prerequisite:** Plan 01 merged to `main`.

## Global Constraints

- All Plan 01 constraints still apply (pinned versions, gap defaults, pure compute functions, `width="stretch"`).
- `kaleido` is removed from `requirements.txt` in this plan; PNG export uses Matplotlib only.
- Every `st.cache_data` call sets `max_entries` (≤ 16) and `show_spinner=False`.
- Sidebar widgets get stable keys: `input_format`, `fasta_text`, `seq_type_select`, `analysis_select`. Tests address widgets by these keys.
- Branch: `fix/02-state-memory`. Commit after every task with the `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer.

## Review Focus

1. **Re-uploading an edited file with the same filename.** Expected: the new content is parsed and old results are cleared. Covered in Task 1 (`test_same_name_new_content_reparses`).
2. **Switching the input format back and forth.** Expected: no sequences from the previous input are shown. Covered in Task 1 (`test_switching_input_format_clears_sequences`).
3. **Invalid FASTA pasted after a valid dataset.** Expected: an error, and the previous dataset is not silently reused. Covered in Task 1 (`test_parse_error_does_not_show_previous_dataset`).
4. **Very large alignments** (≥ 2 M cells). Expected: no per-cell hover payload; a static image instead of a frozen browser. Covered in Task 3 (`test_render_mode_thresholds`).
5. **Same IDs, different residues.** Expected: results reset because the fingerprint covers sequence content. Covered in Task 1 (`test_fingerprint_sensitive_to_content`).

---

### Task 1: Input layer: content-hashed loading, structure store, dataset reset

**Files:**
- Create: `modules/structures.py`
- Modify: `modules/seqtools.py` (add `fingerprint`)
- Modify: `modules/utils.py` (`RESULT_KEYS`, `reset_all_results`, `track_dataset`, `stale_notice`, defaults)
- Rewrite: `app.py`
- Modify: `modules/pdb_renumber.py` (`pdb_renumber_section` reads the structure store)
- Test: `tests/test_state.py`, `tests/test_structures.py`, `tests/test_app_flow.py`

**Interfaces:**
- Consumes: Plan 01 parsers (`parse_sequences_from_text/file/structure`, `make_unique_ids`), `prepare_renumbering`, `apply_renumbering`, `extract_header_records`
- Produces:
  - `modules.seqtools.fingerprint(records) -> str` (SHA-1 hex over IDs and sequences, order-sensitive)
  - `modules.utils.RESULT_KEYS: dict[str, default]`, `reset_all_results(state=None)`, `track_dataset(fp, state=None) -> bool`, `stale_notice(message=STALE_MESSAGE)`
  - `modules.structures.StructureFile(name: str, data: bytes, fmt: str)`, frozen, with properties `text`, `digest`, `stem`
  - `modules.structures.parse_structure(sf: StructureFile) -> Bio.PDB.Structure`
  - `st.session_state.structures: dict[str, StructureFile]` (current input's structures; `{}` otherwise)
  - `pdb_renumber_section(structures: dict[str, StructureFile])`, a new signature

- [ ] **Step 1: Create the branch**

```bash
git checkout main && git pull --ff-only && git checkout -b fix/02-state-memory
```

- [ ] **Step 2: Write the failing tests**

`tests/test_state.py`:

```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.seqtools import fingerprint
from modules.utils import RESULT_KEYS, reset_all_results, track_dataset


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id)


def test_fingerprint_sensitive_to_content():
    a = fingerprint([rec("x", "MKV"), rec("y", "MKL")])
    assert a == fingerprint([rec("x", "MKV"), rec("y", "MKL")])
    assert a != fingerprint([rec("x", "MKV"), rec("y", "MKI")])
    assert a != fingerprint([rec("x", "MKV"), rec("z", "MKL")])


def test_reset_all_results_restores_defaults():
    state = {"pairwise_result": object(), "last_msa_params": {"a": 1}}
    reset_all_results(state)
    for key, default in RESULT_KEYS.items():
        assert state[key] == default
    state["last_msa_params"]["x"] = 1
    assert RESULT_KEYS["last_msa_params"] == {}   # defaults are copied, never shared


def test_track_dataset_resets_only_on_change():
    state = {}
    assert track_dataset("fp1", state) is True
    state["pairwise_result"] = "kept"
    assert track_dataset("fp1", state) is False
    assert state["pairwise_result"] == "kept"
    assert track_dataset("fp2", state) is True
    assert state["pairwise_result"] is None
```

`tests/test_structures.py`:

```python
from modules.structures import StructureFile, parse_structure

PDB = ("ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
       "ATOM      2  CA  GLY B   1       4.800   0.000   0.000  1.00  0.00           C\nEND\n")


def test_structure_file_properties():
    sf = StructureFile(name="model.pdb", data=PDB.encode(), fmt="PDB")
    assert sf.stem == "model"
    assert len(sf.digest) == 40
    assert sf.text.startswith("ATOM")


def test_parse_structure_chains():
    structure = parse_structure(StructureFile(name="m.pdb", data=PDB.encode(), fmt="PDB"))
    assert [c.id for c in structure[0]] == ["A", "B"]
```

`tests/test_app_flow.py`:

```python
from streamlit.testing.v1 import AppTest

FASTA_A = ">a\nMKVLAAGIVG\n>b\nMKVLAGIVGA\n>c\nMRVLSAGLVA\n"
FASTA_B = ">a\nMKVLAAGIVG\n>b\nMKVLAGIVGA\n>c\nMRVLSAGLVW\n"


def load(text):
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.text_area(key="fasta_text").input(text).run()
    return at


def run_pairwise(at):
    at.selectbox(key="analysis_select").select("Pairwise").run()
    at.button[0].click().run()
    assert not at.exception
    return at


def test_changed_dataset_clears_results():
    at = run_pairwise(load(FASTA_A))
    assert at.session_state["pairwise_result"] is not None
    at.text_area(key="fasta_text").input(FASTA_B).run()
    assert at.session_state["pairwise_result"] is None


def test_parse_error_does_not_show_previous_dataset():
    at = load(FASTA_A)
    at.text_area(key="fasta_text").input("not fasta").run()
    assert len(at.sidebar.error) == 1
    assert any("Please upload" in i.value for i in at.info)


def test_switching_input_format_clears_sequences():
    at = load(FASTA_A)
    at.selectbox(key="input_format").select("FASTA").run()
    assert any("Please upload" in i.value for i in at.info)
```

Also add this unit test for the same-name re-upload case to `tests/test_state.py` (it exercises the cache key directly, since `AppTest` cannot drive file uploads):

```python
def test_same_name_new_content_reparses():
    import app
    calls = []
    state = {}

    def loader():
        calls.append(1)
        return ("parsed", None)

    key_v1 = ("FASTA", app._digest(b">a\nMKV\n"))
    key_v2 = ("FASTA", app._digest(b">a\nMKL\n"))
    app._load_once(key_v1, loader, state)
    app._load_once(key_v1, loader, state)
    app._load_once(key_v2, loader, state)
    assert len(calls) == 2
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_state.py tests/test_structures.py tests/test_app_flow.py -v`
Expected: FAIL with `ImportError: cannot import name 'fingerprint'`, `ModuleNotFoundError: modules.structures`, and `KeyError` for the `fasta_text` widget.

- [ ] **Step 4: Implement `fingerprint` and the state helpers**

Append to `modules/seqtools.py`:

```python
import hashlib


def fingerprint(records):
    """Order-sensitive SHA-1 over sequence IDs and residues; identifies a dataset."""
    digest = hashlib.sha1()
    for record in records:
        digest.update(record.id.encode())
        digest.update(b"\t")
        digest.update(str(record.seq).encode())
        digest.update(b"\n")
    return digest.hexdigest()
```

In `modules/utils.py`, add `import copy` and replace `init_session_state` with:

```python
# Every analysis result and its parameter snapshot. Cleared together when the dataset changes.
RESULT_KEYS = {
    "pairwise_result": None,
    "last_selected_seqs": None,
    "msa_result": None,
    "mutations": None,
    "msa_image": None,
    "msa_letters": None,
    "consensus_data": None,
    "msa_tree": None,
    "best_match_results": None,
    "clustering_matrix": None,
    "clustering_ids": None,
    "clustering_msa": None,
    "converted_data": None,
    "conversion_error": None,
    "antibody_results": None,
    "orf_results": None,
    "pdb_renumber_results": None,
    "pdb_renumber_hash": None,
    "tree": None,
    "tree_newick": None,
    "last_msa_params": {},
    "last_pairwise_params": {},
    "best_match_params": {},
    "last_conversion_params": {},
}

STALE_MESSAGE = "Settings changed since the last run. Click the run button to update the results."


def init_session_state():
    """Create every session key the app reads, without overwriting existing values."""
    defaults = {
        **RESULT_KEYS,
        "seq_type": None,
        "dataset_fp": None,
        "input_key": None,
        "input_cache": None,
        "structures": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = copy.copy(value)


def reset_all_results(state=None):
    """Restore every result key to its default (fresh copies of mutable defaults)."""
    state = st.session_state if state is None else state
    for key, default in RESULT_KEYS.items():
        state[key] = copy.copy(default)


def track_dataset(fp, state=None):
    """Reset all results when the dataset fingerprint changes. Returns True on change."""
    state = st.session_state if state is None else state
    if state.get("dataset_fp") == fp:
        return False
    reset_all_results(state)
    state["dataset_fp"] = fp
    return True


def stale_notice(message=STALE_MESSAGE):
    """Tell the user the shown inputs differ from the ones used for the stored result."""
    st.info(message, icon="🔄")
```

Keep `have_params_changed`, `reset_results`, `get_file_extensions` and `amino_acid_map` unchanged.

- [ ] **Step 5: Create `modules/structures.py`**

```python
"""Uploaded or fetched structure files kept as raw bytes; parsed on demand."""
import hashlib
from dataclasses import dataclass
from io import StringIO

from Bio.PDB import MMCIFParser, PDBParser

STRUCTURE_FORMATS = ("PDB", "mmCIF")


@dataclass(frozen=True)
class StructureFile:
    name: str
    data: bytes
    fmt: str  # "PDB" or "mmCIF"

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.data).hexdigest()

    @property
    def stem(self) -> str:
        return self.name.rsplit(".", 1)[0]


def parse_structure(structure_file):
    """Parse a StructureFile into a Biopython Structure (not cached: Structures are large)."""
    parser = MMCIFParser(QUIET=True) if structure_file.fmt == "mmCIF" else PDBParser(QUIET=True)
    return parser.get_structure(structure_file.stem, StringIO(structure_file.text))
```

- [ ] **Step 6: Rewrite `app.py`**

The user-facing strings stay as they are; Plan 03 Task 1 rewrites the copy.

```python
import hashlib
from io import StringIO

import streamlit as st
from Bio import Phylo, SeqIO

from modules.antibody import antibody_prediction_section
from modules.best_match import best_match_finder_section
from modules.clustering import sequence_clustering_section
from modules.conversion import format_conversion_section
from modules.dna import dna_translation_section
from modules.msa import msa_section
from modules.pairwise import pairwise_alignment_section
from modules.parsers import (
    make_unique_ids, parse_sequences_from_file, parse_sequences_from_structure, parse_sequences_from_text,
)
from modules.pdb_renumber import pdb_renumber_section
from modules.phylogeny import TREE_METHODS, TreeError, build_phylogenetic_tree, phylogenetic_tree_section
from modules.seqtools import fingerprint
from modules.structures import STRUCTURE_FORMATS, StructureFile
from modules.utils import get_file_extensions, init_session_state, reset_all_results, track_dataset

INPUT_FORMATS = ("Text (FASTA)", "FASTA", "Clustal", "GenBank", "Newick", "PDB", "mmCIF")


def _digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _load_once(key, loader, state=None):
    """Run `loader` only when the input identity `key` changes; otherwise reuse its cached result."""
    state = st.session_state if state is None else state
    if state.get("input_key") != key:
        state["input_key"] = key
        state["input_cache"] = loader()
    return state["input_cache"]


def _load_structures(files, input_format, all_models):
    records, errors, store = [], [], {}
    for f in files:
        seqs, err = parse_sequences_from_structure(f, input_format, all_models=all_models)
        if err:
            errors.append(f"{f.name}: {err}")
        else:
            records.extend(seqs)
        store[f.name] = StructureFile(name=f.name, data=f.getvalue(), fmt=input_format)
    st.session_state.structures = store
    return (make_unique_ids(records) if records else None), ("\n".join(errors) or None)


def _read_newick(uploaded):
    try:
        return Phylo.read(StringIO(uploaded.getvalue().decode("utf-8")), "newick"), None
    except Exception as e:
        return None, f"Error reading Newick file: {e}"


def handle_input(input_format):
    """Return (sequences, tree) for the current sidebar input. Parsing reruns only when content changes."""
    result, error = None, None
    if input_format not in STRUCTURE_FORMATS:
        st.session_state.structures = {}

    if input_format == "Text (FASTA)":
        seq_text = st.sidebar.text_area(
            "Paste Sequences Here (FASTA Format)", height=250,
            placeholder=">Sequence1\nATGCGTA...\n>Sequence2\nATGCGTC...",
            help="Enter your sequences in FASTA format", key="fasta_text",
        )
        if seq_text:
            result, error = _load_once(("text", _digest(seq_text.encode())),
                                       lambda: parse_sequences_from_text(seq_text))
    elif input_format in STRUCTURE_FORMATS:
        all_models = st.sidebar.checkbox("Extract all models (NMR ensembles)", value=False)
        files = st.sidebar.file_uploader(
            f"Upload {input_format} File(s)", type=get_file_extensions(input_format),
            key=f"uploader_{input_format}", accept_multiple_files=True,
            help=f"Upload one or more {input_format} files to extract protein sequences",
        )
        if files:
            key = (input_format, all_models, tuple((f.name, _digest(f.getvalue())) for f in files))
            result, error = _load_once(key, lambda: _load_structures(files, input_format, all_models))
        else:
            st.session_state.structures = {}
    elif input_format == "Newick":
        uploaded = st.sidebar.file_uploader(
            "Upload Newick Tree File", type=["nwk", "newick", "tree"], key="uploader_newick",
            help="Upload a Newick format tree file for visualization",
        )
        if uploaded:
            result, error = _load_once(("newick", _digest(uploaded.getvalue())), lambda: _read_newick(uploaded))
    else:
        uploaded = st.sidebar.file_uploader(
            f"Upload {input_format} File", type=get_file_extensions(input_format),
            key=f"uploader_{input_format}", help=f"Upload sequence file in {input_format} format",
        )
        if uploaded:
            result, error = _load_once((input_format, _digest(uploaded.getvalue())),
                                       lambda: parse_sequences_from_file(uploaded, input_format))

    if error:
        st.sidebar.error(error)
    if input_format == "Newick":
        return None, result
    if result:
        st.sidebar.success(f"Successfully loaded {len(result)} sequences.")
    return result, None


def main():
    st.set_page_config(page_title="Advanced Sequence Alignment Ability", layout="wide")
    init_session_state()

    st.title("🔬 Advanced Sequence Alignment and Format Conversion Ability")
    st.write("""
    ### Welcome to the Advanced Sequence Alignment Tool!
    This application allows you to perform **pairwise** and **multiple sequence alignments (MSA)** on **DNA** or **Protein** sequences.
    You can choose between **Global**, **Local**, and **Overlap** alignments. Additionally, you can convert sequence files between various formats.
    For MSA, point mutations relative to a reference sequence are reported.
    **How to Use:**
    1. **Upload your sequences** either by uploading a file or pasting them directly.
    2. **Select your preferences** from the sidebar.
    3. **Run** the desired analysis.
    4. **Download** your results!
    """)

    st.sidebar.header("📥 Upload and Settings")
    input_format = st.sidebar.selectbox("Select Input Format", INPUT_FORMATS, key="input_format",
                                        help="Choose the format of your input sequences.")
    sequences, tree = handle_input(input_format)

    if tree is not None:
        phylogenetic_tree_section(tree)
        return
    if not sequences:
        st.info("Please upload sequences or a Newick tree in the sidebar to begin.")
        return

    track_dataset(fingerprint(sequences))
    structures = st.session_state.get("structures", {})

    seq_type = st.sidebar.selectbox("🔬 Select Sequence Type", ("DNA", "Protein"), index=1, key="seq_type_select",
                                    help="Choose whether your sequences are DNA or Protein.")
    if st.session_state.get("seq_type") != seq_type:
        st.session_state.seq_type = seq_type
        reset_all_results()

    if input_format in STRUCTURE_FORMATS:
        options = ("Extracted Sequences", "Sequence Clustering", "Antibody Prediction", "Phylogenetic Tree",
                   "PDB Residue Renumbering")
    else:
        options = ("Pairwise", "MSA", "Best Match Finder", "Sequence Clustering", "Antibody Prediction",
                   "Convert Formats", "Phylogenetic Tree", "Translate DNA")
    alignment_mode = st.sidebar.selectbox("🛠️ Select Analysis", options, key="analysis_select")

    if alignment_mode == "Extracted Sequences":
        st.header("📄 Extracted FASTA Sequences")
        fasta_io = StringIO()
        SeqIO.write(sequences, fasta_io, "fasta")
        fasta_str = fasta_io.getvalue()
        st.text_area("FASTA Sequences", value=fasta_str, height=500)
        stem = next(iter(structures.values())).stem if len(structures) == 1 else "extracted"
        st.download_button(label="📥 Download FASTA", data=fasta_str, file_name=f"{stem}_sequences.fasta",
                           mime="text/plain")
    elif alignment_mode == "Pairwise":
        pairwise_alignment_section(sequences, seq_type)
    elif alignment_mode == "MSA":
        msa_section(sequences, seq_type)
    elif alignment_mode == "Best Match Finder":
        best_match_finder_section(sequences, seq_type)
    elif alignment_mode == "Sequence Clustering":
        sequence_clustering_section(sequences, seq_type)
    elif alignment_mode == "Antibody Prediction":
        antibody_prediction_section(sequences)
    elif alignment_mode == "Translate DNA":
        if seq_type == "DNA":
            dna_translation_section(sequences)
        else:
            st.warning("Translation is only available for DNA sequences. Please select DNA as sequence type.")
    elif alignment_mode == "Convert Formats":
        format_conversion_section(sequences, input_format, seq_type)
    elif alignment_mode == "PDB Residue Renumbering":
        pdb_renumber_section(structures)
    elif alignment_mode == "Phylogenetic Tree":
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
            phylogenetic_tree_section(new_tree)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Point the renumbering UI at the structure store**

In `modules/pdb_renumber.py`, change the imports: remove `MMCIFParser, PDBParser` from the `Bio.PDB` import (keep `PDBIO`) and add `from modules.structures import parse_structure`. Then replace `pdb_renumber_section` with the version below. It no longer keeps a `Structure` in the session, only the mappings, and re-parses from bytes on confirm.

```python
def pdb_renumber_section(structures):
    """UI and workflow for PDB Residue Renumbering based on a target sequence."""
    st.header("🔢 PDB Residue Renumbering")
    st.info(
        "Provide target sequences (FASTA). Each chain is matched to its best target and renumbered so "
        "residue numbers equal target positions (1-based). Insertions relative to the target get "
        "insertion codes (e.g. 52A); terminal overhangs continue the numbering outward. "
        "Ligands and waters keep their original numbers."
    )
    if not structures:
        st.warning("Upload a PDB or mmCIF file in the sidebar first.")
        return

    name = st.selectbox("Structure to renumber", list(structures), key="renumber_structure")
    structure_file = structures[name]

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

    key = (structure_file.digest, tuple((s.id, str(s.seq)) for s in target_sequences))
    if st.session_state.get('pdb_renumber_hash') != key:
        with st.spinner("Matching chains to target sequences..."):
            try:
                mappings, logs, visuals = prepare_renumbering(parse_structure(structure_file), target_sequences)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.text(traceback.format_exc())
                return
            st.session_state.pdb_renumber_results = {'mappings': mappings, 'logs': logs, 'visuals': visuals}
            st.session_state.pdb_renumber_hash = key

    results = st.session_state.pdb_renumber_results
    st.subheader("Interactive Review & Confirmation")
    with st.expander("Review Sequence Alignments (PDB vs Target)", expanded=True):
        for chain_id, target_id, aln_text in results['visuals']:
            st.markdown(f"**Chain {chain_id}** matched to **{target_id}**")
            st.text(aln_text)
    with st.expander("Renumbering Logs", expanded=True):
        for msg in results['logs']:
            st.write(msg)

    is_cif = structure_file.fmt == "mmCIF"
    if is_cif:
        st.caption("mmCIF output contains coordinates only; other mmCIF categories are not carried over.")
    st.warning("Please review the alignment above. If it looks correct, confirm the renumbering.")

    if st.button("Confirm Renumbering & Generate File"):
        try:
            renumbered = apply_renumbering(parse_structure(structure_file), results['mappings'])
        except ValueError as e:
            st.error(f"Renumbering aborted: {e}")
            return
        writer = MMCIFIO() if is_cif else PDBIO()
        writer.set_structure(renumbered)
        out_stream = io.StringIO()
        writer.save(out_stream)
        content = out_stream.getvalue()
        if not is_cif:
            content = extract_header_records(structure_file.text) + content
        ext = ".cif" if is_cif else ".pdb"
        st.success("Renumbering completed successfully!")
        st.download_button(
            label=f"📥 Download Renumbered {structure_file.fmt}",
            data=content,
            file_name=f"{structure_file.stem}_renumbered{ext}",
            mime="chemical/x-mmcif" if is_cif else "chemical/x-pdb",
        )
```

- [ ] **Step 8: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass, including the 3 AppTest flow tests and `test_same_name_new_content_reparses`.

- [ ] **Step 9: Manual check**

`streamlit run app.py`:

1. Upload a FASTA file and run Pairwise.
2. Edit the file on disk (same name) and upload it again. The result disappears.
3. Upload two PDB files, choose **PDB Residue Renumbering**, and confirm both files appear in the structure selectbox.

- [ ] **Step 10: Commit**

```bash
git add app.py modules/structures.py modules/seqtools.py modules/utils.py modules/pdb_renumber.py tests/test_state.py tests/test_structures.py tests/test_app_flow.py
git commit -m "fix: content-hashed input, dataset-scoped results, structure store for renumbering

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: MSA computed once and stored compactly; output format no longer reruns FAMSA

**Files:**
- Modify: `modules/msa.py` (cached wrappers, `aligned_to_msa`, rewrite `msa_section`)
- Modify: `modules/clustering.py` (use `cached_run_msa`; vectorized coverage; identity from aligned strings)
- Modify: `modules/utils.py` (`RESULT_KEYS`)
- Test: `tests/test_msa.py`, `tests/test_app_flow.py`

**Interfaces:**
- Consumes: `run_msa`, `serialize_alignment`, `mutations_from_alignment`, `MSAError`, `calculate_representative_sequence`, `build_tree_from_alignment`, `fingerprint`, `stale_notice`, `msa_identity_matrix`
- Produces:
  - `cached_run_msa(pairs: tuple[tuple[str, str], ...], seq_type: str) -> list[tuple[str, str]]` (`st.cache_data`, `max_entries=8`)
  - `cached_mutations(aligned: tuple, reference_id: str) -> dict` (`st.cache_data`, `max_entries=16`)
  - `aligned_to_msa(aligned_pairs) -> MultipleSeqAlignment`
  - `calculate_sequence_coverage(aligned: list[str]) -> np.ndarray` (non-gap count per column)
  - Session key `msa_aligned` (list of `(id, aligned_seq)`) replaces `msa_result`, `mutations`, `msa_image` and `msa_letters`
  - Session key `clustering_msa` now holds aligned pairs, not FASTA text

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_msa.py`:

```python
from modules.clustering import calculate_sequence_coverage
from modules.msa import aligned_to_msa


def test_aligned_to_msa():
    aln = aligned_to_msa([("a", "MK-V"), ("b", "MKLV")])
    assert len(aln) == 2
    assert aln.get_alignment_length() == 4


def test_coverage_vectorized():
    assert calculate_sequence_coverage(["MK-V", "M--V", "MKLV"]).tolist() == [3, 2, 1, 3]
```

Append to `tests/test_app_flow.py`:

```python
def test_changing_msa_output_format_does_not_rerun():
    at = load(FASTA_A)
    at.selectbox(key="analysis_select").select("MSA").run()
    at.button[0].click().run()
    aligned = at.session_state["msa_aligned"]
    assert aligned
    at.selectbox(key="msa_output_format").select("clustal").run()
    assert not at.exception
    assert at.session_state["msa_aligned"] == aligned
    assert not any("Settings changed" in i.value for i in at.info)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_msa.py tests/test_app_flow.py -v`
Expected: FAIL with `ImportError: aligned_to_msa`, and `KeyError` for `msa_output_format` / `msa_aligned`.

- [ ] **Step 3: Implement in `modules/msa.py`**

Add to the imports: `from Bio.Align import MultipleSeqAlignment`, `from modules.seqtools import fingerprint`, and change the utils import to `from modules.utils import have_params_changed, stale_notice`. Keep the existing `from modules.viz import msa_to_image, plot_msa_image` line; Task 3 replaces it. Add below `perform_msa`:

```python
@st.cache_data(max_entries=8, show_spinner=False)
def cached_run_msa(pairs, seq_type):
    """FAMSA alignment memoized on the exact (id, sequence) tuples and sequence type."""
    return run_msa(list(pairs), seq_type)


@st.cache_data(max_entries=16, show_spinner=False)
def cached_mutations(aligned, reference_id):
    return mutations_from_alignment(list(aligned), reference_id)


def aligned_to_msa(aligned_pairs):
    """Biopython MultipleSeqAlignment from (id, aligned_sequence) pairs."""
    return MultipleSeqAlignment([SeqRecord(Seq(seq), id=seq_id, description="") for seq_id, seq in aligned_pairs])
```

Replace `msa_section` entirely:

```python
def msa_section(sequences, seq_type):
    """Multiple Sequence Alignment workflow. The alignment is computed once per dataset."""
    st.header("📏 Multiple Sequence Alignment (MSA)")
    st.info("Multiple Sequence Alignment aligns three or more sequences together, "
            "identifying conserved regions and structural similarities.")
    if len(sequences) < 2:
        st.warning("Please upload at least two sequences for MSA.")
        return

    ref_seq_id = st.selectbox("Select Reference Sequence for Mutation Reporting", [s.id for s in sequences],
                              key="ref_seq_select",
                              help="Mutations in other sequences will be reported relative to this sequence")
    msa_output_format = st.selectbox("Select MSA Output Format", MSA_OUTPUT_FORMATS, key="msa_output_format",
                                     help="Format of the downloaded alignment. Changing it does not rerun the MSA.")
    plot_method = st.selectbox("Select Plotting Method", ("Plotly (Interactive)", "Matplotlib (Static)"),
                               help="Plotly provides interactive visualization. Matplotlib creates static images.")
    calculate_representative = st.checkbox(
        "Calculate Most Representative Sequence",
        help="Find the consensus sequence and sequence that best represents the entire alignment")
    consensus_threshold = 0.7
    if calculate_representative:
        consensus_threshold = st.slider(
            "Consensus Threshold", 0.5, 1.0, 0.7, 0.05,
            help="Minimum fraction of sequences that must share a residue for it to enter the consensus")

    current_params = {"fingerprint": fingerprint(sequences), "seq_type": seq_type}
    params_changed = have_params_changed(current_params, "last_msa_params")

    if st.button("Run MSA"):
        with st.spinner("Performing MSA - this may take a moment for large datasets..."):
            try:
                st.session_state.msa_aligned = cached_run_msa(
                    tuple((s.id, str(s.seq)) for s in sequences), seq_type)
            except MSAError as e:
                st.session_state.msa_aligned = None
                st.error(str(e))
                return
        st.session_state.last_msa_params = current_params
        st.session_state.consensus_data = None
        st.session_state.msa_tree = None
        params_changed = False

    aligned = st.session_state.get("msa_aligned")
    if not aligned:
        return
    if params_changed:
        stale_notice()
        return

    try:
        mutations = cached_mutations(tuple(aligned), ref_seq_id)
    except MSAError as e:
        st.error(str(e))
        mutations = {}

    tab_titles = ["Alignment & Heatmap", "Point Mutations", "Phylogenetic Tree"]
    if calculate_representative:
        tab_titles.insert(2, "Representative Sequence")
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader("📄 MSA Result")
        st.write(f"Total sequences: {len(aligned)}. MSA length: {len(aligned[0][1])}")
        msa_image, msa_letters = msa_to_image(serialize_alignment(aligned, "fasta"), "fasta")
        plot_msa_image(msa_image, msa_letters, plot_method)
        try:
            st.download_button(label="📥 Download MSA", data=serialize_alignment(aligned, msa_output_format),
                               file_name=f"msa_alignment.{msa_output_format}", mime="text/plain")
        except MSAError as e:
            st.error(str(e))

    with tabs[1]:
        st.subheader("🔍 Point Mutations Relative to Reference")
        if any(mutations.values()):
            seq_ids = sorted(mutations)
            select_all = st.checkbox("Select all sequences", value=True, key="select_all_mutations")
            selected_seq_ids = seq_ids if select_all else st.multiselect(
                "Select sequences to view mutations for:", seq_ids,
                default=seq_ids[:min(5, len(seq_ids))], key="mutation_sequence_selector")
            for seq_id in selected_seq_ids:
                mut_list = mutations.get(seq_id, [])
                if mut_list:
                    with st.expander(f"Mutations for {seq_id} ({len(mut_list)} mutations)"):
                        st.dataframe(pd.DataFrame(mut_list, columns=["Position", "Reference", "Variant"]))
                        st.write(f"**Mutations:** {', '.join(f'{r}{p}{v}' for p, r, v in mut_list)}")
            csv_mutations = convert_msa_mutations_to_csv(mutations)
            if csv_mutations:
                st.download_button(label="📥 Download All Mutations (CSV)", data=csv_mutations,
                                   file_name="msa_mutations.csv", mime="text/csv")
        else:
            st.write("No point mutations detected relative to the reference sequence.")

    if calculate_representative:
        with tabs[2]:
            st.subheader("🧬 Representative Sequence Analysis")
            data = st.session_state.consensus_data
            if data is None or data['threshold'] != consensus_threshold:
                alignment = aligned_to_msa(aligned)
                consensus_record, closest_record, min_diff, closest_seq_id = calculate_representative_sequence(
                    alignment, threshold=consensus_threshold, seq_type=seq_type)
                data = st.session_state.consensus_data = {
                    'threshold': consensus_threshold, 'consensus_record': consensus_record,
                    'closest_record': closest_record, 'min_differences': min_diff,
                    'closest_seq_id': closest_seq_id, 'seq_count': len(alignment),
                }
            st.write(f"**Consensus threshold:** {data['threshold']}")
            st.write(f"**Number of sequences:** {data['seq_count']}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Consensus Sequence")
                st.code(f">{data['consensus_record'].id}\n{data['consensus_record'].seq}")
            with col2:
                st.markdown("#### Most Representative Sequence")
                st.write(f"**Sequence ID:** `{data['closest_seq_id']}`")
                st.write(f"**Differences from consensus:** {data['min_differences']}")
                st.code(f">{data['closest_record'].id}\n{data['closest_record'].seq}")
            st.download_button(
                label="📥 Download Consensus/Representative (FASTA)",
                data=(f">{data['consensus_record'].id}\n{data['consensus_record'].seq}\n"
                      f">{data['closest_record'].id}\n{data['closest_record'].seq}\n"),
                file_name="consensus_representative.fasta", mime="text/plain")

    with tabs[-1]:
        st.subheader("🌳 Phylogenetic Tree from MSA")
        c_m, c_c = st.columns(2)
        tree_method = c_m.selectbox("Tree method", list(TREE_METHODS), key="msa_tree_method")
        correction = c_c.checkbox("Correct for multiple substitutions", value=True, key="msa_tree_corr",
                                  help="Jukes–Cantor (DNA) or Kimura (protein) distance correction.")
        if st.button("Calculate Phylogenetic Tree from MSA"):
            with st.spinner("Building tree from MSA..."):
                try:
                    st.session_state.msa_tree = build_tree_from_alignment(
                        aligned_to_msa(aligned), seq_type, TREE_METHODS[tree_method], correction)
                except TreeError as e:
                    st.error(str(e))
        if st.session_state.get('msa_tree'):
            phylogenetic_tree_section(st.session_state.msa_tree)
```

In `modules/utils.py` `RESULT_KEYS`, remove `"msa_result"`, `"mutations"`, `"msa_image"` and `"msa_letters"`, and add `"msa_aligned": None,`.

- [ ] **Step 4: Update clustering to reuse the cached MSA**

In `modules/clustering.py`: change the MSA import to `from modules.msa import MSAError, aligned_to_msa, cached_run_msa`. Replace `calculate_sequence_coverage`:

```python
def calculate_sequence_coverage(aligned):
    """Number of non-gap residues per alignment column."""
    if not aligned:
        return np.zeros(0, dtype=int)
    arr = np.frombuffer("".join(aligned).encode(), dtype=np.uint8).reshape(len(aligned), -1)
    return (arr != ord('-')).sum(axis=0)
```

Replace the "Fast (MSA-based)" branch, from `if method == "Fast (MSA-based)":` through the end of the `with st.form(...)` block:

```python
    if method == "Fast (MSA-based)":
        if st.button("Run/Update MSA"):
            with st.spinner("Calculating MSA..."):
                try:
                    st.session_state.clustering_msa = cached_run_msa(
                        tuple((s.id, str(s.seq)) for s in sequences), seq_type)
                except MSAError as e:
                    st.error(str(e))

        aligned = st.session_state.clustering_msa
        if aligned:
            st.success("MSA calculated.")
            aligned_ids = [seq_id for seq_id, _ in aligned]
            aligned_seqs = [seq for _, seq in aligned]
            align_len = len(aligned_seqs[0])

            msa_image, msa_letters = msa_to_image(
                "".join(f">{i}\n{s}\n" for i, s in aligned), "fasta")
            if msa_image is not None:
                with st.expander("View MSA Image", expanded=True):
                    plot_msa_image(msa_image, msa_letters, "Plotly (Interactive)")

            coverage = calculate_sequence_coverage(aligned_seqs)
            fig_cov = px.area(pd.DataFrame({"Position": range(1, align_len + 1), "Coverage": coverage}),
                              x="Position", y="Coverage", title="Sequence Coverage per Position")
            fig_cov.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_cov, width="stretch")

            st.subheader("MSA Slicing")
            st.write(f"MSA Length: {align_len}")
            s_start, s_end = suggest_msa_range(coverage.tolist())
            with st.form("msa_slicing_form"):
                start_pos, end_pos = st.slider("Select MSA Region to Analyze", min_value=1, max_value=align_len,
                                               value=(s_start, s_end),
                                               help="Suggested range based on sequence coverage density.")
                if st.form_submit_button("Calculate Similarity from Slice"):
                    with st.spinner("Calculating similarity matrix..."):
                        st.session_state.clustering_matrix = msa_identity_matrix(
                            [s[start_pos - 1:end_pos] for s in aligned_seqs])
                        st.session_state.clustering_ids = aligned_ids
```

Delete `calculate_similarity_matrix_from_msa` (now unused) after confirming with `grep -rn calculate_similarity_matrix_from_msa modules tests` that nothing references it. Remove the `AlignIO` and `StringIO` imports if they are unused (`grep -n "AlignIO\|StringIO" modules/clustering.py`).

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add modules/msa.py modules/clustering.py modules/utils.py tests/test_msa.py tests/test_app_flow.py
git commit -m "perf: cache FAMSA per dataset, store aligned pairs, serialize output on demand

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: MSA heatmap: compact codes, residue palette, correct legend, size-aware rendering

**Files:**
- Rewrite: `modules/viz.py`
- Modify: `modules/msa.py`, `modules/clustering.py` (call `plot_msa`)
- Modify: `requirements.txt` (remove `kaleido`)
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: nothing new
- Produces:
  - `ALPHABET = "-ACDEFGHIKLMNPQRSTVWYXBZJOU*"`, `GAP_CODE = 0`
  - `alignment_to_codes(seqs: list[str]) -> np.ndarray[uint8]`
  - `codes_to_letters(codes) -> np.ndarray[str]`
  - `palette(seq_type) -> list[str]` (hex color per code)
  - `render_mode(n_rows, n_cols, method) -> "static" | "interactive" | "interactive_hover"`
  - `msa_figure_plotly(codes, ids, seq_type, hover=True) -> go.Figure`
  - `msa_figure_matplotlib(codes, ids, seq_type) -> matplotlib.figure.Figure`
  - `render_msa_png(codes, ids: tuple, seq_type) -> bytes` (cached)
  - `plot_msa(codes, ids, method="Plotly (Interactive)", seq_type="Protein")`

- [ ] **Step 1: Write the failing tests**

`tests/test_viz.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from modules.viz import (
    ALPHABET, GAP_CODE, alignment_to_codes, msa_figure_matplotlib, msa_figure_plotly, palette, render_mode,
)


def test_codes_are_uint8_and_map_gaps_and_unknowns():
    codes = alignment_to_codes(["MK-v", "MK@V"])
    assert codes.dtype == np.uint8
    assert codes[0, 2] == GAP_CODE
    assert ALPHABET[codes[1, 2]] == "X"          # unknown character
    assert codes[0, 3] == codes[1, 3]            # case-insensitive


def test_unequal_lengths_rejected():
    try:
        alignment_to_codes(["MK", "M"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_matplotlib_colors_match_palette():
    codes = alignment_to_codes(["ACDEFGHIKL", "MNPQRSTVWY"])
    fig = msa_figure_matplotlib(codes, ["a", "b"], "Protein")
    image = fig.axes[0].images[0]
    colors = palette("Protein")
    for code in np.unique(codes):
        assert to_hex(image.cmap(image.norm(code))) == colors[code]
    plt.close(fig)


def test_gap_is_white_in_both_palettes():
    assert palette("Protein")[GAP_CODE] == "#ffffff"
    assert palette("DNA")[GAP_CODE] == "#ffffff"


def test_render_mode_thresholds():
    assert render_mode(10, 100, "Plotly (Interactive)") == "interactive_hover"
    assert render_mode(500, 1000, "Plotly (Interactive)") == "interactive"
    assert render_mode(2000, 2000, "Plotly (Interactive)") == "static"
    assert render_mode(10, 10, "Matplotlib (Static)") == "static"


def test_plotly_figure_has_no_hover_text_when_disabled():
    codes = alignment_to_codes(["MK", "ML"])
    assert msa_figure_plotly(codes, ["a", "b"], "Protein", hover=False).data[0].text is None
    assert msa_figure_plotly(codes, ["a", "b"], "Protein", hover=True).data[0].text is not None
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_viz.py -v`
Expected: FAIL with `ImportError: cannot import name 'ALPHABET'`.

- [ ] **Step 3: Implement**

`modules/viz.py` (full replacement):

```python
from io import BytesIO

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import ListedColormap

ALPHABET = "-ACDEFGHIKLMNPQRSTVWYXBZJOU*"
GAP_CODE = 0
_UNKNOWN_CODE = ALPHABET.index("X")
_LUT = np.full(256, _UNKNOWN_CODE, dtype=np.uint8)
for _i, _ch in enumerate(ALPHABET):
    _LUT[ord(_ch)] = _i
    _LUT[ord(_ch.lower())] = _i
_LUT[ord(".")] = GAP_CODE
_LETTERS = np.array(list(ALPHABET))

# Clustal X-style residue classes; nucleotides use the common A/C/G/T scheme.
PROTEIN_COLORS = {
    "-": "#ffffff",
    **dict.fromkeys("AILMFWV", "#80a0f0"),
    "C": "#f08080",
    **dict.fromkeys("KR", "#f01505"),
    **dict.fromkeys("DE", "#c048c0"),
    **dict.fromkeys("NQST", "#15c015"),
    "G": "#f09048",
    "P": "#c0c000",
    **dict.fromkeys("HY", "#15a4a4"),
}
DNA_COLORS = {"-": "#ffffff", "A": "#64f73f", "C": "#ffb340", "G": "#eb413c", "T": "#3c88ee", "U": "#3c88ee"}
OTHER_COLOR = "#bbbbbb"
HOVER_LIMIT = 200_000      # cells; above this, no per-cell hover text is sent to the browser
PLOTLY_LIMIT = 2_000_000   # cells; above this, a static image is rendered instead


def palette(seq_type):
    """Hex color for every code in ALPHABET."""
    colors = DNA_COLORS if seq_type == "DNA" else PROTEIN_COLORS
    return [colors.get(ch, OTHER_COLOR) for ch in ALPHABET]


def alignment_to_codes(seqs):
    """Encode equal-length aligned strings as a uint8 matrix of ALPHABET indices."""
    if not seqs:
        return np.zeros((0, 0), dtype=np.uint8)
    length = len(seqs[0])
    if any(len(s) != length for s in seqs):
        raise ValueError("All aligned sequences must have the same length.")
    raw = np.frombuffer("".join(seqs).encode("ascii", errors="replace"), dtype=np.uint8)
    return _LUT[raw.reshape(len(seqs), length)]


def codes_to_letters(codes):
    return _LETTERS[codes]


def render_mode(n_rows, n_cols, method):
    cells = n_rows * n_cols
    if method.startswith("Matplotlib") or cells > PLOTLY_LIMIT:
        return "static"
    return "interactive_hover" if cells <= HOVER_LIMIT else "interactive"


def msa_figure_plotly(codes, ids, seq_type, hover=True):
    colors = palette(seq_type)
    k = len(colors)
    colorscale = []
    for i, color in enumerate(colors):
        colorscale += [[i / k, color], [(i + 1) / k, color]]
    heatmap = dict(z=codes, x=np.arange(1, codes.shape[1] + 1), y=list(ids), colorscale=colorscale,
                   zmin=-0.5, zmax=k - 0.5, showscale=False)
    if hover:
        heatmap["text"] = codes_to_letters(codes)
        heatmap["hovertemplate"] = "Sequence: %{y}<br>Position: %{x}<br>Residue: %{text}<extra></extra>"
    else:
        heatmap["hovertemplate"] = "Sequence: %{y}<br>Position: %{x}<extra></extra>"
    fig = go.Figure(go.Heatmap(**heatmap))
    fig.update_layout(
        title="Multiple Sequence Alignment",
        xaxis_title="Alignment position",
        yaxis=dict(autorange="reversed", showticklabels=len(ids) <= 60),
        height=min(900, 200 + 14 * len(ids)),
        plot_bgcolor="white",
    )
    return fig


def msa_figure_matplotlib(codes, ids, seq_type):
    colors = palette(seq_type)
    k = len(colors)
    fig, ax = plt.subplots(figsize=(20, max(3, min(30, 0.25 * len(ids) + 2))))
    ax.imshow(codes, cmap=ListedColormap(colors), vmin=-0.5, vmax=k - 0.5, aspect="auto", interpolation="nearest")
    ax.set_title("Multiple Sequence Alignment", fontsize=14)
    ax.set_xlabel("Alignment position", fontsize=12)
    if len(ids) <= 60:
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids, fontsize=8)
    else:
        ax.set_ylabel("Sequence", fontsize=12)
    handles = [mpatches.Patch(facecolor=colors[c], edgecolor="#999999", label=ALPHABET[c])
               for c in np.unique(codes)]
    ax.legend(handles=handles, title="Residues", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout()
    return fig


@st.cache_data(max_entries=4, show_spinner=False)
def render_msa_png(codes, ids, seq_type):
    fig = msa_figure_matplotlib(codes, list(ids), seq_type)
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return buf.getvalue()


def plot_msa(codes, ids, method="Plotly (Interactive)", seq_type="Protein"):
    """Show an MSA heatmap sized to the alignment, plus a PNG download."""
    if codes.size == 0:
        st.error("No alignment to plot.")
        return
    mode = render_mode(codes.shape[0], codes.shape[1], method)
    png = render_msa_png(codes, tuple(ids), seq_type)
    if mode == "static":
        if not method.startswith("Matplotlib"):
            st.info("Alignment is too large for the interactive view; showing a static image.")
        st.image(png, width="stretch")
    else:
        st.plotly_chart(msa_figure_plotly(codes, ids, seq_type, hover=mode == "interactive_hover"), width="stretch")
    st.download_button("📥 Download MSA Heatmap (PNG)", data=png, file_name="msa_heatmap.png",
                       mime="image/png", on_click="ignore")
```

In `modules/msa.py`, change the viz import to `from modules.viz import alignment_to_codes, plot_msa` and replace the two heatmap lines in tab 0:

```python
        plot_msa(alignment_to_codes([s for _, s in aligned]), [i for i, _ in aligned], plot_method, seq_type)
```

In `modules/clustering.py`, change the viz import to `from modules.viz import alignment_to_codes, plot_msa` and replace the MSA image block:

```python
            with st.expander("View MSA Image", expanded=True):
                plot_msa(alignment_to_codes(aligned_seqs), aligned_ids, "Plotly (Interactive)", seq_type)
```

Remove the `kaleido==0.2.1` line from `requirements.txt` and run `pip uninstall -y kaleido`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass. `grep -rn "msa_to_image\|plot_msa_image\|kaleido\|to_image" modules app.py requirements.txt` returns nothing.

- [ ] **Step 5: Manual check**

Run an MSA on 3 proteins. Gaps are white and hydrophobics blue. Switch to Matplotlib: the legend colors match the image.

- [ ] **Step 6: Commit**

```bash
git add modules/viz.py modules/msa.py modules/clustering.py requirements.txt tests/test_viz.py
git commit -m "perf: compact MSA heatmap with residue palette, correct legend, size-aware rendering

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Trees built on request, scoped to the dataset; no shared Newick cache

**Files:**
- Modify: `app.py` (Phylogenetic Tree branch)
- Modify: `modules/phylogeny.py` (`tree_to_newick`; `phylogenetic_tree_section` stops using `tree_newick`)
- Modify: `modules/utils.py` (`RESULT_KEYS`: drop `tree`, `tree_newick`; add `tree_result`)
- Test: `tests/test_phylogeny.py`, `tests/test_app_flow.py`

**Interfaces:**
- Consumes: `build_phylogenetic_tree`, `TREE_METHODS`, `TreeError`, `stale_notice`
- Produces:
  - `tree_to_newick(tree) -> str`
  - Session key `tree_result = {"params": dict, "tree": Tree}`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_phylogeny.py`:

```python
def test_tree_to_newick_roundtrip():
    from io import StringIO
    from Bio import Phylo
    from modules.phylogeny import tree_to_newick
    tree = Phylo.read(StringIO("((A:0.1,B:0.2):0.3,C:0.5);"), "newick")
    text = tree_to_newick(tree)
    assert sorted(t.name for t in Phylo.read(StringIO(text), "newick").get_terminals()) == ["A", "B", "C"]
```

Append to `tests/test_app_flow.py`:

```python
def test_tree_built_only_on_button_and_reset_with_dataset():
    at = load(FASTA_A)
    at.selectbox(key="analysis_select").select("Phylogenetic Tree").run()
    assert at.session_state["tree_result"] is None
    at.button(key="build_tree").click().run()
    assert not at.exception
    assert at.session_state["tree_result"] is not None
    at.text_area(key="fasta_text").input(FASTA_B).run()
    assert at.session_state["tree_result"] is None
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_phylogeny.py tests/test_app_flow.py -v`
Expected: FAIL with `ImportError: tree_to_newick` and `KeyError: 'tree_result'`.

- [ ] **Step 3: Implement**

In `modules/phylogeny.py`, add:

```python
def tree_to_newick(tree):
    out = StringIO()
    Phylo.write(tree, out, "newick")
    return out.getvalue()
```

and replace the Newick block at the end of `phylogenetic_tree_section`:

```python
    st.subheader("📄 Newick Format")
    newick = tree_to_newick(tree)
    st.code(newick)
    st.download_button(label="📥 Download Newick File", data=newick, file_name="phylogenetic_tree.nwk",
                       mime="text/plain")
```

In `modules/utils.py` `RESULT_KEYS`: delete `"tree"` and `"tree_newick"`, and add `"tree_result": None,`.

In `app.py`, import `stale_notice` from `modules.utils` and replace the Phylogenetic Tree branch:

```python
    elif alignment_mode == "Phylogenetic Tree":
        c_m, c_c = st.columns(2)
        tree_method = c_m.selectbox("Tree method", list(TREE_METHODS))
        correction = c_c.checkbox("Correct for multiple substitutions", value=True,
                                  help="Jukes–Cantor (DNA) or Kimura (protein) distance correction.")
        params = {"dataset": st.session_state.dataset_fp, "seq_type": seq_type,
                  "method": tree_method, "correction": correction}
        if st.button("Build Tree", key="build_tree"):
            with st.spinner("Building phylogenetic tree..."):
                try:
                    tree = build_phylogenetic_tree(sequences, seq_type, TREE_METHODS[tree_method], correction)
                    st.session_state.tree_result = {"params": params, "tree": tree}
                except TreeError as e:
                    st.session_state.tree_result = None
                    st.error(str(e))
        stored = st.session_state.get("tree_result")
        if stored:
            if stored["params"] != params:
                stale_notice()
            else:
                phylogenetic_tree_section(stored["tree"])
```

`grep -rn "tree_newick\|session_state.tree\b" modules app.py` must return nothing.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app.py modules/phylogeny.py modules/utils.py tests/test_phylogeny.py tests/test_app_flow.py
git commit -m "fix: trees built on request and scoped to the dataset; Newick derived on the fly

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Results survive reruns; a stale-settings notice replaces silent disappearance

**Files:**
- Modify: `modules/dna.py` (persist ORF results)
- Modify: `modules/pairwise.py`, `modules/best_match.py`, `modules/conversion.py` (stale notice)
- Test: `tests/test_app_flow.py`

**Interfaces:**
- Consumes: `stale_notice`, `find_orfs`, `CODON_TABLES`
- Produces: session key `orf_results = {"params": dict, "orfs": list[dict]}`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_app_flow.py`)

```python
DNA = ">g\n" + "ATG" + "GCC" * 40 + "TAA" + "\n>h\nATGAAACCCGGGTTTTAA\n"


def test_orf_results_survive_rerun():
    at = load(DNA)
    at.selectbox(key="seq_type_select").select("DNA").run()
    at.selectbox(key="analysis_select").select("Translate DNA").run()
    at.button(key="find_orfs").click().run()
    assert len(at.dataframe) == 1
    at.run()                                   # any rerun, e.g. from another widget
    assert len(at.dataframe) == 1


def test_pairwise_shows_stale_notice_after_param_change():
    at = run_pairwise(load(FASTA_A))
    at.selectbox(key="pairwise_mode").select("Local").run()
    assert any("Settings changed" in i.value for i in at.info)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_app_flow.py -v`
Expected: FAIL with `KeyError` for the widget keys `find_orfs` and `pairwise_mode`.

- [ ] **Step 3: Implement**

`modules/dna.py`: import `from modules.utils import stale_notice` and replace the ORF finder block (from `st.markdown("### 🕵️ ORF Finder")` to the end of `dna_translation_section`):

```python
    st.markdown("### 🕵️ ORF Finder")
    min_len = st.number_input("Minimum ORF Length (AA)", value=30, min_value=10, step=5)
    params = {"id": selected_seq.id, "seq": str(selected_seq.seq), "min_len": min_len, "table": table}

    if st.button("Find ORFs", key="find_orfs"):
        st.session_state.orf_results = {
            "params": params, "orfs": find_orfs(str(selected_seq.seq), min_len=min_len, table=table)}

    stored = st.session_state.get("orf_results")
    if not stored:
        return
    if stored["params"] != params:
        stale_notice()
        return
    orfs = stored["orfs"]
    if not orfs:
        st.warning("No ORFs found with current settings.")
        return
    st.success(f"Found {len(orfs)} ORFs.")
    df = pd.DataFrame(orfs)
    st.dataframe(df)
    st.download_button("Download ORFs (CSV)", df.to_csv(index=False), "orfs.csv", "text/csv", on_click="ignore")
    st.subheader("ORF Sequences")
    for _, row in df.iterrows():
        with st.expander(f"Frame {row['Frame']}, {row['Start (nt)']}–{row['End (nt)']} nt, {row['Length (AA)']} aa"):
            st.code(row['Sequence'])
```

`modules/pairwise.py`: import `stale_notice` from `modules.utils`. Give the mode selectbox `key="pairwise_mode"`, and change the early return after the run block to:

```python
    result = st.session_state.get('pairwise_result')
    if result is None:
        return
    if params_changed and not run_alignment:
        stale_notice()
        return
```

`modules/best_match.py`: import `stale_notice` and add, directly before `if run_analysis or (...)`:

```python
    if st.session_state.best_match_results is not None and params_changed and not run_analysis:
        stale_notice()
        return
```

`modules/conversion.py`: import `stale_notice` and change `if params_changed: return` to:

```python
    if params_changed:
        if st.session_state.get('converted_data') or st.session_state.get('conversion_error'):
            stale_notice()
        return
```

Add `on_click="ignore"` to every `st.download_button` in `modules/` and `app.py`, so downloads never trigger a rerun:

```bash
grep -rn "st.download_button(" modules app.py | wc -l
```

Edit each call, then check that the counts match:

```bash
grep -rn 'on_click="ignore"' modules app.py | wc -l
```

- [ ] **Step 4: Run the full suite and the pre-commit gate**

Run: `python -m pytest -q && python precommit_check.py`
Expected: all pass; `Pre-commit passed.`

- [ ] **Step 5: Manual smoke test of the whole plan**

`streamlit run app.py` and go through every analysis. Results stay put when unrelated widgets change, a 🔄 notice appears when analysis settings change, and downloads don't flash the page.

- [ ] **Step 6: Commit**

```bash
git add modules/dna.py modules/pairwise.py modules/best_match.py modules/conversion.py tests/test_app_flow.py
git commit -m "fix: results persist across reruns; stale-settings notice; downloads do not rerun

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
