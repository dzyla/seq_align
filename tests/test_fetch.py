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
