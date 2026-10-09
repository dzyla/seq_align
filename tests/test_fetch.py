import pathlib

import pytest

import modules.fetch as fetch
from modules.fetch import FetchError, fetch_any, parse_id_list, validate_ids

DATA = pathlib.Path(__file__).parent / "data"


class FakeResponse:
    def __init__(self, status=200, text="", content=b"", json_data=None, headers=None):
        self.status_code = status
        self.text = text
        self.content = content or text.encode()
        self._json = json_data
        self.headers = headers or {}

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
    calls = install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/accessions":
                                  FakeResponse(text=">sp|P69905|HBA_HUMAN x\nMVLS\n")})
    assert fetch_any("UniProt", ["P69905"]).fasta.startswith(">sp|P69905")
    assert calls[0][1]["accessions"] == "P69905" and calls[0][1]["format"] == "fasta"


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


def test_ncbi_missing_id_reported(monkeypatch):
    install(monkeypatch, {"https://eutils.ncbi.nlm.nih.gov": FakeResponse(text=">NP_000549.1 x\nMVLS\n")})
    with pytest.raises(FetchError, match="NP_000518"):
        fetch_any("NCBI Protein", ["NP_000549.1", "NP_000518.1"])


def test_ncbi_version_and_case_insensitive_match(monkeypatch):
    install(monkeypatch, {"https://eutils.ncbi.nlm.nih.gov": FakeResponse(text=">NP_000549.2 x\nMVLS\n")})
    assert fetch_any("NCBI Protein", ["NP_000549"]).fasta.startswith(">NP_000549.2")


def test_ncbi_rate_limit_and_bad_request(monkeypatch):
    install(monkeypatch, {"https://eutils.ncbi.nlm.nih.gov": FakeResponse(status=429)})
    with pytest.raises(FetchError, match="rate limit"):
        fetch_any("NCBI Protein", ["NP_000549.1"])
    install(monkeypatch, {"https://eutils.ncbi.nlm.nih.gov": FakeResponse(status=400)})
    with pytest.raises(FetchError, match="database"):
        fetch_any("NCBI Protein", ["NP_000549.1"])


def test_uniprot_records_not_merged(monkeypatch):
    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/accessions":
                          FakeResponse(text=">sp|P69905|A x\nMVLS\n>sp|P68871|B y\nMVHL\n")})
    assert ">sp|P69905|A x\nMVLS\n>sp|P68871|B y\nMVHL" in fetch_any("UniProt", ["P69905", "P68871"]).fasta


# --- UniProt batch -------------------------------------------------------------------------------------------
TSV_HEADER = "Entry\tEntry Name\tProtein names\tGene Names (primary)\tOrganism\tLength\n"


def tsv(*rows):
    return TSV_HEADER + "".join("\t".join(map(str, r)) + "\n" for r in rows)


def test_extract_uniprot_ids_from_messy_text():
    from modules.fetch import extract_uniprot_ids
    acc, names = extract_uniprot_ids('P69905, hba_human; P68871\tfoo,"Q9Y6K9" P69905 123 P69905-2')
    assert acc == ["P69905", "P68871", "Q9Y6K9", "P69905-2"] and names == ["HBA_HUMAN"]


def test_batch_tsv_chunks_accessions_and_reports_missing(monkeypatch):
    from modules.fetch import uniprot_batch_tsv
    calls = install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/accessions":
                                  FakeResponse(text=tsv(("P69905", "HBA_HUMAN", "Hb alpha", "HBA1", "Human", 142)))})
    accessions = ["P69905"] + [f"Q{i:05d}" for i in range(1199)]
    frame, missing = uniprot_batch_tsv(accessions, [])
    assert len(calls) == 3 and [len(c[1]["accessions"].split(",")) for c in calls] == [500, 500, 200]
    assert list(frame["Accession"]) == ["P69905"] and frame["Length"].iloc[0] == 142
    assert len(missing) == 1199 and missing[0] == "Q00000"


def test_entry_name_lookup_keeps_exact_matches_only(monkeypatch):
    # UniProt's id: search is fuzzy: a request for NOPE_HUMAN returned IGDC4_HUMAN in a live probe
    from modules.fetch import uniprot_batch_tsv
    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/stream": FakeResponse(text=tsv(
        ("P69905", "HBA_HUMAN", "Hb alpha", "HBA1", "Human", 142),
        ("Q8TDY8", "IGDC4_HUMAN", "IGDC4", "IGDC4", "Human", 1250)))})
    frame, missing = uniprot_batch_tsv([], ["HBA_HUMAN", "NOPE_HUMAN"])
    assert list(frame["Entry name"]) == ["HBA_HUMAN"] and missing == ["NOPE_HUMAN"]


def test_gene_query_and_missing_genes(monkeypatch):
    from modules.fetch import gene_query, uniprot_gene_tsv
    assert gene_query(["HBB", "TP53"], 9606) == "(gene_exact:HBB OR gene_exact:TP53) AND organism_id:9606 AND reviewed:true"
    assert gene_query(["HBB"], 10090, reviewed=False) == "(gene_exact:HBB) AND organism_id:10090"
    calls = install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/stream": FakeResponse(text=tsv(
        ("P69905", "HBA_HUMAN", "Hb alpha", "HBA1; HBA2", "Human", 142)))})
    frame, missing = uniprot_gene_tsv(["hba2", "NOTAGENE"], 9606, isoforms=True)
    assert missing == ["NOTAGENE"] and calls[0][1]["includeIsoform"] == "true"


def test_query_count_reads_header(monkeypatch):
    from modules.fetch import uniprot_query_count
    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/search":
                          FakeResponse(text="Entry\n", headers={"x-total-results": "13"})})
    assert uniprot_query_count("family:globin") == 13


def test_uniprot_bad_query_and_rate_limit_messages(monkeypatch):
    from modules.fetch import uniprot_query_count
    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/search": FakeResponse(status=400)})
    with pytest.raises(FetchError, match="query syntax"):
        uniprot_query_count("family:(globin")
    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/search": FakeResponse(status=429)})
    with pytest.raises(FetchError, match="rate limit"):
        uniprot_query_count("family:globin")


def test_fasta_missing_accession_named(monkeypatch):
    install(monkeypatch, {"https://rest.uniprot.org/uniprotkb/accessions":
                          FakeResponse(text=">sp|P69905|HBA_HUMAN x\nMVLS\n")})
    with pytest.raises(FetchError, match="Q00000"):
        fetch.fetch_uniprot_fasta(["P69905", "Q00000"])


def test_extract_ids_with_nothing_valid():
    # Review Focus: an upload with no UniProt IDs is reported, not sent to UniProt
    from modules.fetch import extract_uniprot_ids
    assert extract_uniprot_ids("gene name, 12345, hello world") == ([], [])
