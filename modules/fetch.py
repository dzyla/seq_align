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
