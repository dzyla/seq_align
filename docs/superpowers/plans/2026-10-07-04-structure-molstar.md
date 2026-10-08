# Plan 04 — Structures, Fetch by ID and Mol\* Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch sequences and structures by accession, choose SEQRES or ATOM sequences, and see mutations, conservation and pLDDT on the 3D structure in an embedded Mol\* viewer, with conservation exportable as B-factors.

**Architecture:**
- `modules/fetch.py` wraps the UniProt, NCBI, RCSB and AlphaFold DB REST APIs behind `FetchError`.
- `modules/structures.py` gains a per-residue table (chain, auth number, insertion code, residue, B-factor) built from polymer residues.
- `modules/structure_view.py` builds MolViewSpec states (pure, unit-tested) and renders them with `molviewspec.molstar_streamlit`. Structure bytes are packed into an MVSX archive, so nothing is hosted.
- A reusable `structure_highlight_panel` maps reference-sequence positions to structure residues by alignment and colors them. It accepts structures from the main input or its own upload/fetch, because the main input holds either sequences or structures, never both.

**Tech Stack:** molviewspec 1.8.1 (Mol\* 5.10.1 from jsDelivr), requests, Biopython PDB, Streamlit components.

**Spec:** `docs/superpowers/specs/2026-10-07-codebase-review.md` (items F1, F2)

**Prerequisite:** Plan 03 merged to `main`.

## Global Constraints

- All earlier constraints apply.
- Add `molviewspec==1.8.1` and `requests==2.32.3` to `requirements.txt`.
- `MOLSTAR_VERSION = "5.10.1"` is the only Mol\* version used; never `latest`.
- Network calls only happen in `modules/fetch.py`; every call has `timeout=30` and a descriptive `User-Agent`. Tests never touch the network (they monkeypatch `requests.get`).
- `modules/pairwise.py` must never import `modules.structure_view` at module level (circular: structure_view → pdb_renumber → pairwise). Use a function-level import.
- Structure test fixtures live in `tests/data/` (`1crn.pdb`, `1crn.cif`, PDB entry 1CRN, crambin, 46 residues).
- Branch: `feat/04-molstar`. Commit after every task with the `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer.

## Review Focus

1. **An invalid accession, or a network failure.** Expected: a sidebar error naming the ID, with the previous dataset untouched. Covered in Task 1 (`test_invalid_id_rejected_before_network`, `test_http_404_raises`).
2. **Structures with insertion codes** (antibodies). Expected: highlights select `auth_seq_id` together with `pdbx_PDB_ins_code`. Covered in Task 3 (`test_insertion_code_in_selector`).
3. **A reference sequence that only partly matches the chosen chain.** Expected: unmapped positions are counted and reported, not silently dropped. Covered in Task 4 (`test_unmapped_positions_reported`).
4. **mmCIF structures in the viewer and the B-factor export.** Expected: the MVS parse format is `mmcif`, and the export round-trips. Covered in Task 3 (`test_mmcif_format_name`) and Task 5 (`test_write_bfactors_mmcif`).
5. **Gap-rich alignment columns in conservation scoring.** Expected: no score (NaN), so the residue is left uncolored instead of misleadingly marked as conserved. Covered in Task 5 (`test_gappy_columns_are_nan`).

---

### Task 1: Fetch sequences and structures by ID

**Files:**
- Create: `modules/fetch.py`, `tests/data/1crn.pdb`, `tests/data/1crn.cif`
- Modify: `modules/parsers.py` (split the text-based core out of `parse_sequences_from_structure`)
- Modify: `app.py` (new input "Fetch by ID")
- Modify: `requirements.txt`
- Test: `tests/test_fetch.py`, `tests/test_app_flow.py`

**Interfaces:**
- Consumes: `StructureFile`, `parse_sequences_from_text`, `make_unique_ids`, `_load_once`
- Produces:
  - `FetchError(RuntimeError)`
  - `FETCH_SOURCES = ("UniProt", "NCBI Protein", "NCBI Nucleotide", "PDB (structure)", "AlphaFold DB (structure)")`
  - `parse_id_list(text) -> list[str]`
  - `validate_ids(source, ids) -> None` (raises `FetchError`)
  - `fetch_uniprot_fasta(ids) -> str`, `fetch_ncbi_fasta(ids, db) -> str`
  - `fetch_pdb_structure(pdb_id, fmt="mmCIF") -> StructureFile`, `fetch_alphafold_structure(uniprot_id) -> StructureFile`
  - `FetchResult(fasta: str = "", structures: tuple[StructureFile, ...] = ())` with property `digest`
  - `fetch_any(source, ids) -> FetchResult`
  - `modules.parsers.parse_structure_text(text, name, format_name, all_models=False) -> (records | None, error | None)`

- [ ] **Step 1: Create the branch and fixtures**

```bash
git checkout main && git pull --ff-only && git checkout -b feat/04-molstar
mkdir -p tests/data
curl -sf https://files.rcsb.org/download/1CRN.pdb -o tests/data/1crn.pdb
curl -sf https://files.rcsb.org/download/1CRN.cif -o tests/data/1crn.cif
grep -c "^ATOM" tests/data/1crn.pdb
```

Expected: `327`.

Add to `requirements.txt` (alphabetical position is not required):

```
molviewspec==1.8.1
requests==2.32.3
```

Then run `pip install -r requirements-dev.txt`.

- [ ] **Step 2: Write the failing tests**

`tests/test_fetch.py`:

```python
import pathlib

import pytest

import modules.fetch as fetch
from modules.fetch import FetchError, fetch_any, parse_id_list, validate_ids

DATA = pathlib.Path(__file__).parent / "data"


class FakeResponse:
    def __init__(self, status=200, text="", content=b"", json_data=None):
        self.status_code = status
        self.text = text
        self.content = content or text.encode()
        self._json = json_data

    def json(self):
        return self._json


def install(monkeypatch, routes):
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append((url, params))
        for prefix, response in routes.items():
            if url.startswith(prefix):
                return response
        return FakeResponse(status=404)

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    return calls


def test_parse_id_list():
    assert parse_id_list(" p69905, P68871\n1crn 1CRN ") == ["P69905", "P68871", "1CRN"]


def test_invalid_id_rejected_before_network(monkeypatch):
    calls = install(monkeypatch, {})
    with pytest.raises(FetchError, match="XYZ"):
        validate_ids("PDB (structure)", ["XYZ"])
    with pytest.raises(FetchError):
        fetch_any("UniProt", ["not-an-accession"])
    assert calls == []


def test_uniprot_fasta(monkeypatch):
    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/P69905.fasta": FakeResponse(text=">sp|P69905|HBA\nMVLS\n")})
    assert fetch_any("UniProt", ["P69905"]).fasta.startswith(">sp|P69905")


def test_ncbi_uses_efetch(monkeypatch):
    calls = install(monkeypatch, {"https://eutils.ncbi.nlm.nih.gov": FakeResponse(text=">NP_000549.1 x\nMVLS\n")})
    result = fetch_any("NCBI Protein", ["NP_000549.1"])
    assert result.fasta.startswith(">NP_000549.1")
    assert calls[0][1]["db"] == "protein"


def test_pdb_structure(monkeypatch):
    install(monkeypatch, {"https://files.rcsb.org/download/1CRN.cif": FakeResponse(content=(DATA / "1crn.cif").read_bytes())})
    result = fetch_any("PDB (structure)", ["1CRN"])
    assert result.structures[0].name == "1CRN.cif"
    assert result.structures[0].fmt == "mmCIF"


def test_alphafold_uses_api_cif_url(monkeypatch):
    api = FakeResponse(json_data=[{"cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v6.cif"}])
    install(monkeypatch, {
        "https://alphafold.ebi.ac.uk/api/prediction/P69905": api,
        "https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v6.cif": FakeResponse(content=(DATA / "1crn.cif").read_bytes()),
    })
    result = fetch_any("AlphaFold DB (structure)", ["P69905"])
    assert result.structures[0].name == "AF-P69905-F1-model_v6.cif"


def test_http_404_raises(monkeypatch):
    install(monkeypatch, {})
    with pytest.raises(FetchError, match="P69905"):
        fetch_any("UniProt", ["P69905"])
```

Append to `tests/test_app_flow.py`:

```python
def test_fetch_pdb_offers_structure_tools(monkeypatch):
    import pathlib
    import modules.fetch as fetch

    data = (pathlib.Path(__file__).parent / "data" / "1crn.cif").read_bytes()

    class R:
        status_code = 200
        content = data
        text = data.decode()

    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: R())
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.selectbox(key="input_format").select("Fetch by ID").run()
    at.selectbox(key="fetch_source").select("PDB (structure)").run()
    at.text_input(key="fetch_ids").input("1CRN").run()
    at.button(key="fetch_button").click().run()
    assert not at.exception
    assert "PDB Residue Renumbering" in at.selectbox(key="analysis_select").options
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_fetch.py tests/test_app_flow.py -v`
Expected: FAIL with `ModuleNotFoundError: modules.fetch`.

- [ ] **Step 4: Implement `modules/fetch.py`**

```python
"""Fetch sequences and structures from public databases. All network access lives here."""
import hashlib
import re
from dataclasses import dataclass

import requests

from modules.structures import StructureFile

TIMEOUT = 30
HEADERS = {"User-Agent": "seq_align/1.0 (https://github.com/dzyla/seq_align)"}
FETCH_SOURCES = ("UniProt", "NCBI Protein", "NCBI Nucleotide", "PDB (structure)", "AlphaFold DB (structure)")
_UNIPROT = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$")
_PATTERNS = {
    "UniProt": _UNIPROT,
    "AlphaFold DB (structure)": _UNIPROT,
    "NCBI Protein": re.compile(r"^[A-Z]{1,6}_?[0-9]+(\.[0-9]+)?$"),
    "NCBI Nucleotide": re.compile(r"^[A-Z]{1,6}_?[0-9]+(\.[0-9]+)?$"),
    "PDB (structure)": re.compile(r"^[0-9][A-Z0-9]{3}$"),
}


class FetchError(RuntimeError):
    """Raised for invalid IDs, network errors and missing entries."""


@dataclass(frozen=True)
class FetchResult:
    fasta: str = ""
    structures: tuple = ()

    @property
    def digest(self):
        h = hashlib.sha1(self.fasta.encode())
        for sf in self.structures:
            h.update(sf.name.encode())
            h.update(sf.digest.encode())
        return h.hexdigest()


def parse_id_list(text):
    """Split on commas/whitespace, upper-case, de-duplicate keeping order."""
    seen, ids = set(), []
    for token in re.split(r"[\s,;]+", text.strip()):
        token = token.upper()
        if token and token not in seen:
            seen.add(token)
            ids.append(token)
    return ids


def validate_ids(source, ids):
    if not ids:
        raise FetchError("Enter at least one ID.")
    bad = [i for i in ids if not _PATTERNS[source].match(i)]
    if bad:
        raise FetchError(f"Not a valid {source} ID: {', '.join(bad)}")


def _get(url, what, params=None):
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
    except requests.RequestException as e:
        raise FetchError(f"Network error while fetching {what}: {e}") from e
    if response.status_code == 404:
        raise FetchError(f"{what} not found.")
    if response.status_code != 200:
        raise FetchError(f"Fetching {what} failed (HTTP {response.status_code}).")
    return response


def fetch_uniprot_fasta(ids):
    return "".join(_get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", f"UniProt {acc}").text for acc in ids)


def fetch_ncbi_fasta(ids, db):
    params = {"db": db, "id": ",".join(ids), "rettype": "fasta", "retmode": "text"}
    text = _get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", f"NCBI {', '.join(ids)}", params).text
    if not text.lstrip().startswith(">"):
        raise FetchError(f"NCBI returned no FASTA for {', '.join(ids)}.")
    return text


def fetch_pdb_structure(pdb_id, fmt="mmCIF"):
    ext = "cif" if fmt == "mmCIF" else "pdb"
    response = _get(f"https://files.rcsb.org/download/{pdb_id}.{ext}", f"PDB {pdb_id}")
    return StructureFile(name=f"{pdb_id}.{ext}", data=response.content, fmt=fmt)


def fetch_alphafold_structure(uniprot_id):
    entries = _get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}", f"AlphaFold model {uniprot_id}").json()
    if not entries or not entries[0].get("cifUrl"):
        raise FetchError(f"No AlphaFold model listed for {uniprot_id}.")
    url = entries[0]["cifUrl"]
    response = _get(url, f"AlphaFold model {uniprot_id}")
    return StructureFile(name=url.rsplit("/", 1)[-1], data=response.content, fmt="mmCIF")


def fetch_any(source, ids):
    validate_ids(source, ids)
    if source == "UniProt":
        return FetchResult(fasta=fetch_uniprot_fasta(ids))
    if source == "NCBI Protein":
        return FetchResult(fasta=fetch_ncbi_fasta(ids, "protein"))
    if source == "NCBI Nucleotide":
        return FetchResult(fasta=fetch_ncbi_fasta(ids, "nuccore"))
    if source == "PDB (structure)":
        return FetchResult(structures=tuple(fetch_pdb_structure(i) for i in ids))
    return FetchResult(structures=tuple(fetch_alphafold_structure(i) for i in ids))
```

- [ ] **Step 5: Split the structure parser so it works on text**

In `modules/parsers.py`, rename the body of `parse_sequences_from_structure` into a text-based core and keep the file wrapper:

```python
def parse_structure_text(text, name, format_name, all_models=False):
    """Extract one protein sequence per chain from PDB/mmCIF text. Returns (records, error)."""
    try:
        base = os.path.splitext(os.path.basename(name))[0]
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
                    continue
                chain_label = chain.id.strip() or "_"
                seq_id = f"{base}_m{model_idx}_{chain_label}" if multi else f"{base}_{chain_label}"
                description = f"Chain {chain_label} from {name}" + (f", model {model_idx}" if multi else "")
                record = SeqRecord(Seq("".join(letters)), id=seq_id, description=description)
                record.annotations["structure"] = {"file": name, "chain": chain.id, "model": model_idx}
                sequences.append(record)
        if not sequences:
            return None, "No protein sequences could be extracted from the structure file."
        return make_unique_ids(sequences), None
    except Exception as e:
        return None, f"An error occurred while parsing the {format_name} file: {e}"


def parse_sequences_from_structure(file, format_name, all_models=False):
    """File-upload wrapper around parse_structure_text."""
    file.seek(0)
    raw = file.read()
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    return parse_structure_text(text, file.name, format_name, all_models)
```

- [ ] **Step 6: Add the "Fetch by ID" input to `app.py`**

Imports: `from modules.fetch import FETCH_SOURCES, FetchError, fetch_any, parse_id_list` and `from modules.parsers import parse_structure_text` (added to the existing parsers import).

Change `INPUT_FORMATS` to include `"Fetch by ID"` after `"Text (FASTA)"`. Change the structure-clearing guard at the top of `handle_input` to:

```python
    if input_format not in STRUCTURE_FORMATS and input_format != "Fetch by ID":
        st.session_state.structures = {}
```

Add the loader helper:

```python
def _load_fetched(result):
    if result.fasta:
        st.session_state.structures = {}
        return parse_sequences_from_text(result.fasta)
    records, errors = [], []
    for sf in result.structures:
        seqs, err = parse_structure_text(sf.text, sf.name, sf.fmt)
        if err:
            errors.append(f"{sf.name}: {err}")
        else:
            records.extend(seqs)
    st.session_state.structures = {sf.name: sf for sf in result.structures}
    return (make_unique_ids(records) if records else None), ("\n".join(errors) or None)
```

Add this branch to `handle_input` before the final `else:`:

```python
    elif input_format == "Fetch by ID":
        source = st.sidebar.selectbox("Database", FETCH_SOURCES, key="fetch_source")
        ids_text = st.sidebar.text_input("ID(s)", key="fetch_ids",
                                         placeholder="e.g. P69905, P68871  or  1CRN",
                                         help="Separate several IDs with commas or spaces.")
        if st.sidebar.button("Fetch", key="fetch_button"):
            try:
                with st.spinner(f"Fetching from {source}..."):
                    st.session_state.fetched = fetch_any(source, parse_id_list(ids_text))
            except FetchError as e:
                st.sidebar.error(str(e))
        fetched = st.session_state.get("fetched")
        if fetched:
            result, error = _load_once(("fetch", fetched.digest), lambda: _load_fetched(fetched))
```

Add `"fetched": None,` to the non-result defaults in `init_session_state` (it is an input, not a result).

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add modules/fetch.py modules/parsers.py modules/utils.py app.py requirements.txt tests/data tests/test_fetch.py tests/test_app_flow.py
git commit -m "feat: fetch sequences and structures from UniProt, NCBI, RCSB and AlphaFold DB

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Per-residue structure table and SEQRES vs ATOM sequences

**Files:**
- Modify: `modules/structures.py` (`ResidueInfo`, `chain_residue_table`)
- Modify: `modules/parsers.py` (`source` argument: `"atom"` | `"seqres"`)
- Modify: `app.py` (sequence-source radio for PDB/mmCIF uploads)
- Test: `tests/test_structures.py`, `tests/test_parsers.py`

**Interfaces:**
- Consumes: `polymer_residues`, `amino_acid_map`, `parse_structure`
- Produces:
  - `ResidueInfo(chain: str, number: int, icode: str, resname: str, aa: str, bfactor: float)`, frozen
  - `chain_residue_table(structure, model_index=0) -> dict[str, list[ResidueInfo]]` (protein chains only)
  - `parse_structure_text(text, name, format_name, all_models=False, source="atom")`
  - `parse_sequences_from_structure(file, format_name, all_models=False, source="atom")`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_structures.py`:

```python
import pathlib

from modules.structures import ResidueInfo, chain_residue_table

DATA = pathlib.Path(__file__).parent / "data"
CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def test_chain_residue_table_1crn():
    sf = StructureFile(name="1crn.pdb", data=(DATA / "1crn.pdb").read_bytes(), fmt="PDB")
    table = chain_residue_table(parse_structure(sf))
    assert list(table) == ["A"]
    residues = table["A"]
    assert "".join(r.aa for r in residues) == CRAMBIN
    assert residues[0] == ResidueInfo("A", 1, "", "THR", "T", residues[0].bfactor)
    assert residues[0].bfactor > 0
```

Append to `tests/test_parsers.py`:

```python
import pathlib

from modules.parsers import parse_structure_text

DATA = pathlib.Path(__file__).parent / "data"


def test_seqres_and_atom_sources():
    text = (DATA / "1crn.pdb").read_text()
    atom, err = parse_structure_text(text, "1crn.pdb", "PDB", source="atom")
    seqres, err2 = parse_structure_text(text, "1crn.pdb", "PDB", source="seqres")
    assert err is None and err2 is None
    assert [r.id for r in seqres] == ["1crn_A"]
    assert str(seqres[0].seq) == str(atom[0].seq)
    assert atom[0].annotations["structure"]["chain"] == "A"
    assert "structure" not in seqres[0].annotations


def test_seqres_cif():
    seqres, err = parse_structure_text((DATA / "1crn.cif").read_text(), "1crn.cif", "mmCIF", source="seqres")
    assert err is None and len(seqres[0].seq) == 46


def test_seqres_missing_reports_error():
    text = "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\nEND\n"
    records, err = parse_structure_text(text, "x.pdb", "PDB", source="seqres")
    assert records is None and "SEQRES" in err
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_structures.py tests/test_parsers.py -v`
Expected: FAIL with `ImportError: ResidueInfo` and an unexpected `source` keyword.

- [ ] **Step 3: Implement**

Append to `modules/structures.py`:

```python
from modules.parsers import polymer_residues
from modules.utils import amino_acid_map


@dataclass(frozen=True)
class ResidueInfo:
    chain: str
    number: int      # author residue number (auth_seq_id)
    icode: str       # insertion code, "" if none
    resname: str
    aa: str
    bfactor: float   # CA B-factor (pLDDT in AlphaFold models); mean over atoms if no CA


def chain_residue_table(structure, model_index=0):
    """Polymer residues per protein chain of one model, in file order."""
    model = list(structure)[model_index]
    table = {}
    for chain in model:
        rows = []
        for res in polymer_residues(chain):
            atoms = list(res.get_atoms())
            if "CA" in res:
                bfactor = res["CA"].get_bfactor()
            else:
                bfactor = sum(a.get_bfactor() for a in atoms) / len(atoms) if atoms else 0.0
            resname = res.resname.strip()
            rows.append(ResidueInfo(chain.id, res.id[1], res.id[2].strip(), resname,
                                    amino_acid_map.get(resname, "X"), float(bfactor)))
        if any(r.aa != "X" for r in rows):
            table[chain.id] = rows
    return table
```

In `modules/parsers.py`, add a `source` parameter. At the top of `parse_structure_text`'s `try:`, after computing `base`, add:

```python
        if source == "seqres":
            seq_format = "pdb-seqres" if format_name == "PDB" else "cif-seqres"
            records = []
            for record in SeqIO.parse(StringIO(text), seq_format):
                chain = record.annotations.get("chain", "_")
                records.append(SeqRecord(record.seq, id=f"{base}_{chain}",
                                         description=f"SEQRES chain {chain} from {name}"))
            if not records:
                return None, "No SEQRES records in this file; use the ATOM sequence source instead."
            return make_unique_ids(records), None
```

Update the signatures to `parse_structure_text(text, name, format_name, all_models=False, source="atom")` and `parse_sequences_from_structure(file, format_name, all_models=False, source="atom")`; the wrapper passes `source` through.

In `app.py`, PDB/mmCIF branch: add a radio before the uploader, include it in the cache key, and pass it to `_load_structures` (which passes it to `parse_sequences_from_structure`):

```python
        source = st.sidebar.radio("Sequence source", ("atom", "seqres"), horizontal=True, key="structure_seq_source",
                                  format_func={"atom": "ATOM (modelled residues)",
                                               "seqres": "SEQRES (full construct)"}.get)
```

The cache key becomes `(input_format, all_models, source, tuple(...))`, and `_load_structures(files, input_format, all_models, source)` calls `parse_sequences_from_structure(f, input_format, all_models=all_models, source=source)`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add modules/structures.py modules/parsers.py app.py tests/test_structures.py tests/test_parsers.py
git commit -m "feat: per-residue structure table and SEQRES/ATOM sequence source

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Mol\* viewer core and the Structure Viewer page

**Files:**
- Create: `modules/structure_view.py`
- Modify: `modules/navigation.py` (add "Structure Viewer"), `app.py` (dispatch)
- Test: `tests/test_structure_view.py`, `tests/test_navigation.py`

**Interfaces:**
- Consumes: `StructureFile`, `parse_structure`, `chain_residue_table`, `ResidueInfo`
- Produces:
  - `MOLSTAR_VERSION = "5.10.1"`, `CHAIN_COLORS: list[str]`, `PLDDT_BINS: list[tuple[float, str, str]]`
  - `ResidueRef(chain: str, number: int, icode: str = "")`, `Highlight(residue: ResidueRef, color: str, label: str = "")`, both frozen
  - `mvs_format(fmt) -> "pdb" | "mmcif"`; `archive_name(sf) -> "structure.pdb" | "structure.cif"`
  - `group_by_color(highlights) -> dict[str, list[ResidueRef]]`
  - `plddt_color(value) -> str`; `bfactor_highlights(table) -> list[Highlight]`
  - `build_view_state(sf, chain_ids, color_mode="chain", highlights=(), base_color="#d9d9d9", show_sticks=True) -> molviewspec Root builder`
  - `render_molstar(sf, builder, height=600)`
  - `structure_viewer_section(structures: dict[str, StructureFile])`

- [ ] **Step 1: Write the failing tests**

`tests/test_structure_view.py`:

```python
import json
import pathlib

from modules.structure_view import (
    Highlight, ResidueRef, archive_name, bfactor_highlights, build_view_state, group_by_color, mvs_format,
    plddt_color,
)
from modules.structures import StructureFile, chain_residue_table, parse_structure

DATA = pathlib.Path(__file__).parent / "data"
PDB_SF = StructureFile(name="my model.pdb", data=(DATA / "1crn.pdb").read_bytes(), fmt="PDB")
CIF_SF = StructureFile(name="1crn.cif", data=(DATA / "1crn.cif").read_bytes(), fmt="mmCIF")


def state_json(builder):
    return json.loads(builder.get_state().dumps())


def nodes(tree, kind):
    found = []
    stack = [tree["root"]]
    while stack:
        node = stack.pop()
        if node.get("kind") == kind:
            found.append(node)
        stack.extend(node.get("children", []))
    return found


def test_archive_name_and_format():
    assert archive_name(PDB_SF) == "structure.pdb"
    assert archive_name(CIF_SF) == "structure.cif"


def test_mmcif_format_name():
    assert mvs_format("mmCIF") == "mmcif"
    assert mvs_format("PDB") == "pdb"
    state = state_json(build_view_state(CIF_SF, ["A"]))
    assert nodes(state, "download")[0]["params"]["url"] == "structure.cif"
    assert nodes(state, "parse")[0]["params"]["format"] == "mmcif"


def test_chain_mode_colors_each_chain():
    colors = nodes(state_json(build_view_state(PDB_SF, ["A", "B"])), "color")
    assert [c["params"]["selector"]["auth_asym_id"] for c in colors] == ["A", "B"]


def test_highlights_grouped_and_tooltips():
    hl = [Highlight(ResidueRef("A", 5), "#ff0000", "K5R"), Highlight(ResidueRef("A", 9), "#ff0000", "A9W"),
          Highlight(ResidueRef("A", 12), "#0000ff")]
    assert {k: len(v) for k, v in group_by_color(hl).items()} == {"#ff0000": 2, "#0000ff": 1}
    state = state_json(build_view_state(PDB_SF, ["A"], color_mode="uniform", highlights=hl))
    tooltip_texts = sorted(n["params"]["text"] for n in nodes(state, "tooltip"))
    assert tooltip_texts == ["A9W", "K5R"]


def test_insertion_code_in_selector():
    hl = [Highlight(ResidueRef("H", 52, "A"), "#ff0000")]
    state = state_json(build_view_state(PDB_SF, ["H"], color_mode="uniform", highlights=hl))
    selectors = [n["params"].get("selector") for n in nodes(state, "color")]
    assert [{"auth_asym_id": "H", "auth_seq_id": 52, "pdbx_PDB_ins_code": "A"}] in selectors


def test_plddt_bins():
    assert plddt_color(95) == "#0053d6"
    assert plddt_color(80) == "#65cbf3"
    assert plddt_color(60) == "#ffdb13"
    assert plddt_color(10) == "#ff7d45"


def test_bfactor_highlights_cover_all_residues():
    table = chain_residue_table(parse_structure(PDB_SF))
    assert len(bfactor_highlights(table)) == 46
```

Append to `tests/test_navigation.py`:

```python
def test_structures_offer_viewer():
    assert "Structure Viewer" in available_analyses(True, True, "Protein")
    assert "Structure Viewer" not in available_analyses(True, False, "Protein")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_structure_view.py tests/test_navigation.py -v`
Expected: FAIL with `ModuleNotFoundError: modules.structure_view`.

- [ ] **Step 3: Implement `modules/structure_view.py`**

```python
"""Mol* (MolViewSpec) 3D views of structures with residue-level highlights."""
from collections import OrderedDict
from dataclasses import dataclass

import molviewspec as mvs
import streamlit as st
from molviewspec import ComponentExpression

from modules.structures import StructureFile, chain_residue_table, parse_structure

MOLSTAR_VERSION = "5.10.1"
CHAIN_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7",
                "#9c755f", "#bab0ac"]
# AlphaFold DB pLDDT color convention.
PLDDT_BINS = [(90.0, "#0053d6", "Very high (> 90)"), (70.0, "#65cbf3", "Confident (70–90)"),
              (50.0, "#ffdb13", "Low (50–70)"), (float("-inf"), "#ff7d45", "Very low (< 50)")]
MAX_TOOLTIPS = 300
MAX_STICKS = 300


@dataclass(frozen=True)
class ResidueRef:
    chain: str
    number: int
    icode: str = ""


@dataclass(frozen=True)
class Highlight:
    residue: ResidueRef
    color: str
    label: str = ""


def mvs_format(fmt):
    return "mmcif" if fmt == "mmCIF" else "pdb"


def archive_name(sf):
    """Fixed name inside the MVSX archive (uploaded file names may contain spaces)."""
    return "structure.cif" if sf.fmt == "mmCIF" else "structure.pdb"


def _selector(ref):
    params = {"auth_asym_id": ref.chain, "auth_seq_id": ref.number}
    if ref.icode.strip():
        params["pdbx_PDB_ins_code"] = ref.icode.strip()
    return ComponentExpression(**params)


def group_by_color(highlights):
    groups = OrderedDict()
    for h in highlights:
        groups.setdefault(h.color, []).append(h.residue)
    return groups


def plddt_color(value):
    for threshold, color, _label in PLDDT_BINS:
        if value > threshold:
            return color
    return PLDDT_BINS[-1][1]


def bfactor_highlights(table):
    """Color every residue by its CA B-factor using the pLDDT bins."""
    return [Highlight(ResidueRef(r.chain, r.number, r.icode), plddt_color(r.bfactor))
            for rows in table.values() for r in rows]


def build_view_state(sf, chain_ids, color_mode="chain", highlights=(), base_color="#d9d9d9", show_sticks=True):
    """MolViewSpec state: cartoon polymer colored by chain or uniformly, then per-residue highlights."""
    builder = mvs.create_builder()
    structure = builder.download(url=archive_name(sf)).parse(format=mvs_format(sf.fmt)).model_structure()
    cartoon = structure.component(selector="polymer").representation(type="cartoon")
    if color_mode == "chain":
        for i, chain_id in enumerate(chain_ids):
            cartoon.color(color=CHAIN_COLORS[i % len(CHAIN_COLORS)],
                          selector=ComponentExpression(auth_asym_id=chain_id))
    else:
        cartoon.color(color=base_color)
    for color, refs in group_by_color(highlights).items():
        cartoon.color(color=color, selector=[_selector(r) for r in refs])

    labelled = [h for h in highlights if h.label][:MAX_TOOLTIPS]
    if show_sticks and labelled:
        sticks = structure.component(selector=[_selector(h.residue) for h in labelled[:MAX_STICKS]])
        sticks_repr = sticks.representation(type="ball_and_stick")
        for color, refs in group_by_color(labelled[:MAX_STICKS]).items():
            sticks_repr.color(color=color, selector=[_selector(r) for r in refs])
    for h in labelled:
        structure.component(selector=_selector(h.residue)).tooltip(text=h.label)

    structure.component(selector="ligand").representation(type="ball_and_stick")
    return builder


def render_molstar(sf, builder, height=600):
    """Embed Mol* with the structure packed into an MVSX archive (no hosting needed)."""
    mvs.molstar_streamlit(builder, data={archive_name(sf): sf.data}, height=height,
                          molstar_version=MOLSTAR_VERSION)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_residue_table(data, fmt, name):
    return chain_residue_table(parse_structure(StructureFile(name=name, data=data, fmt=fmt)))


def structure_viewer_section(structures):
    st.header("🧊 Structure Viewer (Mol*)")
    if not structures:
        st.warning("Load a PDB/mmCIF file or fetch a structure in the sidebar first.")
        return
    name = st.selectbox("Structure", list(structures), key="viewer_structure")
    sf = structures[name]
    table = cached_residue_table(sf.data, sf.fmt, sf.name)
    mode = st.radio("Color by", ("Chain", "pLDDT / B-factor"), horizontal=True, key="viewer_color_mode",
                    help="AlphaFold models store pLDDT in the B-factor column.")
    if mode == "Chain":
        builder = build_view_state(sf, list(table), color_mode="chain")
    else:
        builder = build_view_state(sf, list(table), color_mode="uniform", highlights=bfactor_highlights(table),
                                   show_sticks=False)
        st.caption(" · ".join(f"<span style='color:{c}'>■</span> {label}" for _, c, label in PLDDT_BINS),
                   unsafe_allow_html=True)
    render_molstar(sf, builder)
    st.caption(f"{len(table)} protein chain(s): " + ", ".join(f"{c} ({len(r)} residues)" for c, r in table.items()))
```

In `modules/navigation.py`, append `"Structure Viewer"` to the structure options:

```python
    if has_structures:
        options += ["Structure Viewer", "PDB Residue Renumbering"]
```

Update `test_structures_add_extraction_and_renumbering` in `tests/test_navigation.py` so it still expects `options[-1] == "PDB Residue Renumbering"` (unchanged; the viewer comes just before it).

In `app.py`, import `structure_viewer_section` and dispatch:

```python
    elif alignment_mode == "Structure Viewer":
        structure_viewer_section(structures)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Manual check in a browser**

`streamlit run app.py`, then Fetch by ID → AlphaFold DB → `P69905` → Structure Viewer → "pLDDT / B-factor". The cartoon should be mostly dark blue, with orange/yellow termini. Switch to Fetch by ID → PDB → `1CRN` and color by chain.

- [ ] **Step 6: Commit**

```bash
git add modules/structure_view.py modules/navigation.py app.py tests/test_structure_view.py tests/test_navigation.py
git commit -m "feat: Mol* structure viewer via MolViewSpec with chain and pLDDT coloring

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: "Show on structure" for pairwise and MSA mutations

**Files:**
- Modify: `modules/structure_view.py` (mapping and panel)
- Modify: `modules/pairwise.py`, `modules/msa.py` (call the panel)
- Test: `tests/test_structure_view.py`

**Interfaces:**
- Consumes: `align_target_to_chain`, `get_mapping_from_alignment` (from `modules.pdb_renumber`), `fetch_pdb_structure`, `fetch_alphafold_structure`, `FetchError`, `clean_sequence`
- Produces:
  - `map_sequence_to_structure(ref_seq: str, residues: list[ResidueInfo]) -> tuple[dict[int, ResidueInfo], float]` (1-based ref position → residue; identity)
  - `rank_chains(ref_seq, table) -> list[tuple[str, float]]`, best first
  - `position_highlights(position_colors: dict[int, tuple[str, str]], pos_map) -> tuple[list[Highlight], list[int]]` (highlights, unmapped positions)
  - `available_structures() -> dict[str, StructureFile]` (input structures plus panel-added ones in `st.session_state.extra_structures`)
  - `structure_highlight_panel(ref_seq: str, position_colors: dict[int, tuple[str, str]], key: str, title="🧊 Show on structure (Mol*)")`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_structure_view.py`)

```python
from modules.structure_view import map_sequence_to_structure, position_highlights, rank_chains

CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def test_map_sequence_to_structure_with_mutation():
    table = chain_residue_table(parse_structure(PDB_SF))
    mutant = CRAMBIN[:8] + "W" + CRAMBIN[9:]
    pos_map, identity = map_sequence_to_structure(mutant, table["A"])
    assert pos_map[9].number == 9 and pos_map[9].aa == "A"
    assert identity > 0.95


def test_rank_chains():
    table = chain_residue_table(parse_structure(PDB_SF))
    assert rank_chains(CRAMBIN, table)[0][0] == "A"


def test_unmapped_positions_reported():
    table = chain_residue_table(parse_structure(PDB_SF))
    extended = "MKKLLL" + CRAMBIN          # 6 extra N-terminal residues absent from the structure
    pos_map, _ = map_sequence_to_structure(extended, table["A"])
    highlights, unmapped = position_highlights({2: ("#ff0000", "K2R"), 15: ("#ff0000", "A15W")}, pos_map)
    assert unmapped == [2]
    assert highlights[0].residue.number == 9 and highlights[0].label == "A15W"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_structure_view.py -v`
Expected: FAIL with `ImportError: map_sequence_to_structure`.

- [ ] **Step 3: Implement** (append to `modules/structure_view.py`)

```python
from modules.fetch import FetchError, fetch_alphafold_structure, fetch_pdb_structure
from modules.pdb_renumber import align_target_to_chain, get_mapping_from_alignment


def map_sequence_to_structure(ref_seq, residues):
    """Align a reference sequence to a chain; map 1-based reference positions to residues."""
    chain_seq = "".join(r.aa for r in residues)
    aligned_t, aligned_c, identity = align_target_to_chain(ref_seq, chain_seq)
    mapping = get_mapping_from_alignment(aligned_t, aligned_c)   # chain index -> ref position
    return {ref_pos: residues[idx] for idx, ref_pos in mapping.items()}, identity


def rank_chains(ref_seq, table):
    scores = [(chain_id, map_sequence_to_structure(ref_seq, rows)[1]) for chain_id, rows in table.items()]
    return sorted(scores, key=lambda item: item[1], reverse=True)


def position_highlights(position_colors, pos_map):
    highlights, unmapped = [], []
    for pos, (color, label) in sorted(position_colors.items()):
        res = pos_map.get(pos)
        if res is None:
            unmapped.append(pos)
        else:
            highlights.append(Highlight(ResidueRef(res.chain, res.number, res.icode), color, label))
    return highlights, unmapped


def available_structures():
    return {**st.session_state.get("structures", {}), **st.session_state.get("extra_structures", {})}


def _add_structure_controls(key):
    c1, c2 = st.columns(2)
    upload = c1.file_uploader("Add a structure", type=["pdb", "ent", "cif", "mmcif"], key=f"{key}_upload")
    if upload is not None:
        fmt = "mmCIF" if upload.name.lower().endswith(("cif", "mmcif")) else "PDB"
        sf = StructureFile(name=upload.name, data=upload.getvalue(), fmt=fmt)
        st.session_state.extra_structures = {**st.session_state.get("extra_structures", {}), sf.name: sf}
    ident = c2.text_input("…or fetch: PDB ID or UniProt accession (AlphaFold)", key=f"{key}_fetch_id")
    if c2.button("Fetch structure", key=f"{key}_fetch_btn") and ident.strip():
        ident = ident.strip().upper()
        try:
            sf = fetch_pdb_structure(ident) if len(ident) == 4 else fetch_alphafold_structure(ident)
            st.session_state.extra_structures = {**st.session_state.get("extra_structures", {}), sf.name: sf}
        except FetchError as e:
            st.error(str(e))


def structure_highlight_panel(ref_seq, position_colors, key, title="🧊 Show on structure (Mol*)"):
    """Color reference positions on a structure; `position_colors` maps position -> (hex color, label)."""
    with st.expander(title):
        _add_structure_controls(key)
        structures = available_structures()
        if not structures:
            st.caption("Load, upload or fetch a structure to view these positions in 3D.")
            return
        name = st.selectbox("Structure", list(structures), key=f"{key}_structure")
        sf = structures[name]
        table = cached_residue_table(sf.data, sf.fmt, sf.name)
        if not table:
            st.warning("No protein chains in this structure.")
            return
        ranked = rank_chains(ref_seq, table)
        chain_id = st.selectbox("Chain", [c for c, _ in ranked], key=f"{key}_chain",
                                format_func=lambda c: f"{c} ({dict(ranked)[c]:.1%} identity)")
        pos_map, identity = map_sequence_to_structure(ref_seq, table[chain_id])
        highlights, unmapped = position_highlights(position_colors, pos_map)
        st.caption(f"Reference aligned to chain {chain_id}: {identity:.1%} identity; "
                   f"{len(highlights)} of {len(position_colors)} positions shown.")
        if unmapped:
            st.caption(f"Not modelled in this chain: {', '.join(map(str, unmapped[:30]))}"
                       + (" …" if len(unmapped) > 30 else ""))
        render_molstar(sf, build_view_state(sf, list(table), color_mode="uniform", highlights=highlights))
```

Add `"extra_structures": {},` to the non-result defaults in `init_session_state`. These are user-added reference structures and survive dataset changes.

In `modules/pairwise.py`, add the following after the mutations block, only for protein. The import **must** be local to the function: `structure_view` imports `pdb_renumber`, which imports `pairwise`, so a module-level import would be circular.

```python
    if seq_type == "Protein" and result.mutations:
        from modules.structure_view import structure_highlight_panel  # local: avoids a circular import
        structure_highlight_panel(
            clean_sequence(seq1.seq, "Protein"),
            {int(re.search(r"\d+", m).group()): ("#d62728", m) for m in result.mutations},
            key="pairwise_3d",
        )
```

In `modules/msa.py`, import the panel and `clean_sequence`, and append inside the "Point Mutations" tab after the CSV download:

```python
            if seq_type == "Protein":
                mutated = [s for s in seq_ids if mutations.get(s)]
                if mutated:
                    show_id = st.selectbox("Sequence to show on structure", mutated, key="msa_3d_seq")
                    ref_seq = clean_sequence(next(s.seq for s in sequences if s.id == ref_seq_id), "Protein")
                    structure_highlight_panel(
                        ref_seq, {p: ("#d62728", f"{r}{p}{v}") for p, r, v in mutations[show_id]}, key="msa_3d")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Manual check**

Paste two globins (examples → Globins) and run Pairwise HBA_HUMAN vs HBA_MOUSE. Open "Show on structure", fetch `P69905`. The substitutions appear red with tooltips on hover.

- [ ] **Step 6: Commit**

```bash
git add modules/structure_view.py modules/pairwise.py modules/msa.py modules/utils.py tests/test_structure_view.py
git commit -m "feat: show pairwise and MSA substitutions on a structure in Mol*

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Conservation from the MSA, on the structure and as B-factors

**Files:**
- Create: `modules/conservation.py`
- Modify: `modules/msa.py` (new "Conservation" tab)
- Test: `tests/test_conservation.py`

**Interfaces:**
- Consumes: `structure_highlight_panel` pieces (`available_structures`, `cached_residue_table`, `rank_chains`, `map_sequence_to_structure`, `build_view_state`, `render_molstar`, `Highlight`, `ResidueRef`), `parse_structure`, `extract_header_records`
- Produces:
  - `CONSERVATION_COLORS: list[str]` (9 grades, variable → conserved; ConSurf-style palette)
  - `column_conservation(aligned: list[str], seq_type="Protein", max_gap_fraction=0.5) -> np.ndarray` (1 − normalized Shannon entropy; NaN for gappy columns)
  - `reference_conservation(aligned_pairs, reference_id, seq_type="Protein") -> dict[int, float]` (1-based reference position → score, skipping NaN)
  - `conservation_grade(score) -> int` (1–9)
  - `write_bfactors(sf, values: dict[tuple[str, int, str], float], default=0.0) -> str` (keys are `(chain, number, icode)`)

- [ ] **Step 1: Write the failing tests**

`tests/test_conservation.py`:

```python
import math
import pathlib
from io import StringIO

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from modules.conservation import (
    column_conservation, conservation_grade, reference_conservation, write_bfactors,
)
from modules.structures import StructureFile

DATA = pathlib.Path(__file__).parent / "data"


def test_column_conservation_protein():
    scores = column_conservation(["MKV", "MKL", "MRL"])
    assert scores[0] == 1.0
    h = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
    assert math.isclose(scores[1], 1 - h / math.log2(20))


def test_gappy_columns_are_nan():
    scores = column_conservation(["M-", "M-", "MK"])
    assert np.isnan(scores[1])


def test_dna_normalization():
    scores = column_conservation(["AC", "AG"], seq_type="DNA")
    assert math.isclose(scores[1], 1 - 1 / math.log2(4))


def test_reference_conservation_skips_reference_gaps():
    result = reference_conservation([("ref", "M-KV"), ("b", "MAKV"), ("c", "MAKL")], "ref")
    assert sorted(result) == [1, 2, 3]
    assert result[1] == 1.0


def test_conservation_grade():
    assert conservation_grade(0.0) == 1
    assert conservation_grade(1.0) == 9
    assert conservation_grade(0.5) == 5


def test_write_bfactors_pdb_roundtrip():
    sf = StructureFile(name="1crn.pdb", data=(DATA / "1crn.pdb").read_bytes(), fmt="PDB")
    text = write_bfactors(sf, {("A", 1, ""): 0.75})
    assert text.startswith("CRYST1")
    structure = PDBParser(QUIET=True).get_structure("x", StringIO(text))
    chain = structure[0]["A"]
    assert all(a.get_bfactor() == 0.75 for a in chain[1])
    assert all(a.get_bfactor() == 0.0 for a in chain[2])


def test_write_bfactors_mmcif():
    sf = StructureFile(name="1crn.cif", data=(DATA / "1crn.cif").read_bytes(), fmt="mmCIF")
    text = write_bfactors(sf, {("A", 2, ""): 0.5})
    structure = MMCIFParser(QUIET=True).get_structure("x", StringIO(text))
    assert all(a.get_bfactor() == 0.5 for a in structure[0]["A"][2])
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_conservation.py -v`
Expected: FAIL with `ModuleNotFoundError: modules.conservation`.

- [ ] **Step 3: Implement**

`modules/conservation.py`:

```python
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
```

In `modules/msa.py`, import `from modules.conservation import CONSERVATION_COLORS, column_conservation, conservation_grade, reference_conservation, write_bfactors` and `from modules.structure_view import Highlight, ResidueRef, available_structures, build_view_state, cached_residue_table, map_sequence_to_structure, rank_chains, render_molstar, structure_highlight_panel`. Insert `"Conservation"` into `tab_titles` just before `"Phylogenetic Tree"`, and add a tab body (index it via `tabs[tab_titles.index("Conservation")]`):

```python
    with tabs[tab_titles.index("Conservation")]:
        st.subheader("🎯 Conservation")
        ref_scores = reference_conservation(aligned, ref_seq_id, seq_type)
        st.caption(f"1 − normalized Shannon entropy per column (gaps excluded; columns with > 50% gaps "
                   f"are not scored). Positions are in {ref_seq_id} coordinates.")
        st.line_chart(pd.DataFrame({"Conservation": pd.Series(ref_scores)}), x_label="Reference position",
                      y_label="Conservation")
        if seq_type == "Protein":
            ref_seq = clean_sequence(next(s.seq for s in sequences if s.id == ref_seq_id), "Protein")
            with st.expander("🧊 Conservation on structure (Mol*) and B-factor export"):
                structures = available_structures()
                if not structures:
                    st.caption("Use 'Show on structure' in the Point Mutations tab to add a structure, "
                               "or load one via Fetch by ID.")
                else:
                    name = st.selectbox("Structure", list(structures), key="cons_structure")
                    sf = structures[name]
                    table = cached_residue_table(sf.data, sf.fmt, sf.name)
                    ranked = rank_chains(ref_seq, table)
                    chain_id = st.selectbox("Chain", [c for c, _ in ranked], key="cons_chain")
                    pos_map, identity = map_sequence_to_structure(ref_seq, table[chain_id])
                    highlights, values = [], {}
                    for pos, score in ref_scores.items():
                        res = pos_map.get(pos)
                        if res:
                            grade = conservation_grade(score)
                            highlights.append(Highlight(ResidueRef(res.chain, res.number, res.icode),
                                                        CONSERVATION_COLORS[grade - 1]))
                            values[(res.chain, res.number, res.icode)] = round(score, 3)
                    st.caption(f"{len(highlights)} residues colored ({identity:.1%} identity to chain {chain_id}); "
                               "teal = variable, maroon = conserved.")
                    render_molstar(sf, build_view_state(sf, list(table), color_mode="uniform",
                                                        highlights=highlights, show_sticks=False))
                    ext = "cif" if sf.fmt == "mmCIF" else "pdb"
                    st.download_button("📥 Download structure with conservation as B-factors",
                                       data=write_bfactors(sf, values), file_name=f"{sf.stem}_conservation.{ext}",
                                       mime="text/plain", on_click="ignore",
                                       help="Open in ChimeraX and use: color bfactor palette ^bluered")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest -q && python precommit_check.py`
Expected: all pass.

- [ ] **Step 5: Manual check**

Globins example → MSA → Conservation tab → fetch `P69905` in Point Mutations' panel, then return to Conservation. The heme pocket (proximal His87) should be maroon. Download the PDB and open it in ChimeraX: `color bfactor palette ^bluered`.

- [ ] **Step 6: Commit**

```bash
git add modules/conservation.py modules/msa.py tests/test_conservation.py
git commit -m "feat: MSA conservation on the structure in Mol* and B-factor export

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
