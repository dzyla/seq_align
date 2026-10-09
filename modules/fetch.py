"""Fetch sequences and structures from public databases. All network access lives here."""
import hashlib
import re
from dataclasses import dataclass

import requests

from modules.structures import StructureFile

TIMEOUT = 30
HEADERS = {"User-Agent": "seq_align/1.0 (https://github.com/dzyla/seq_align)"}
FETCH_SOURCES = ("UniProt", "UniProt batch", "NCBI Protein", "NCBI Nucleotide", "PDB (structure)", "AlphaFold DB (structure)")
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


def _get(url, what, params=None, hints=None):
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
    except requests.RequestException as e:
        raise FetchError(f"Network error while fetching {what}: {e}") from e
    if response.status_code == 404:
        raise FetchError(f"{what} not found.")
    if hints and response.status_code in hints:
        raise FetchError(f"{what}: {hints[response.status_code]}")
    if response.status_code != 200:
        raise FetchError(f"Fetching {what} failed (HTTP {response.status_code}).")
    return response


UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
UNIPROT_FIELDS = "accession,id,protein_name,gene_primary,organism_name,length"
PREVIEW_COLUMNS = ["Accession", "Entry name", "Protein", "Gene", "Organism", "Length"]
UNIPROT_HINTS = {429: "UniProt rate limit reached; wait a few seconds and try again.",
                 400: "UniProt rejected the request; check the IDs or the query syntax."}
ORGANISMS = {"Human": 9606, "Mouse": 10090, "Rat": 10116, "Yeast (S. cerevisiae)": 559292, "E. coli K-12": 83333,
             "Zebrafish": 7955, "Fruit fly": 7227, "C. elegans": 6239, "Arabidopsis": 3702}
ACCESSION_CHUNK, NAME_CHUNK, GENE_CHUNK = 500, 100, 100
DEFAULT_QUERY_CAP, MAX_QUERY_CAP = 2000, 10000
_ENTRY_NAME = re.compile(r"^[A-Z0-9]{1,10}_[A-Z0-9]{1,5}$")


def _chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def _short(items, n=5):
    return ", ".join(items[:n]) + (f" and {len(items) - n} more" if len(items) > n else "")


def extract_uniprot_ids(text):
    """UniProt accessions and entry names found in free text or a CSV/TSV (any separator), upper-cased and
    de-duplicated in order: (accessions, entry_names)."""
    accessions, names, seen = [], [], set()
    for token in re.split(r"[\s,;|\t\"']+", text or ""):
        token = token.strip().upper()
        if not token or token in seen:
            continue
        if _UNIPROT.match(token):
            accessions.append(token)
        elif _ENTRY_NAME.match(token):
            names.append(token)
        else:
            continue
        seen.add(token)
    return accessions, names


def _tsv_frame(text):
    import pandas as pd
    from io import StringIO

    if not text.strip():
        return pd.DataFrame(columns=PREVIEW_COLUMNS)
    frame = pd.read_csv(StringIO(text), sep="\t", dtype=str).fillna("")
    frame.columns = PREVIEW_COLUMNS[:len(frame.columns)]
    if "Length" in frame:
        frame["Length"] = pd.to_numeric(frame["Length"], errors="coerce").astype("Int64")
    return frame


def _concat(frames):
    import pandas as pd

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=PREVIEW_COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates("Accession").reset_index(drop=True)


def uniprot_batch_tsv(accessions, entry_names=()):
    """Preview table for accessions (batched, <= 500 per request) and entry names (exact matches only; UniProt's
    id: search is fuzzy). Returns (frame, missing IDs in input order)."""
    frames = []
    for chunk in _chunks(list(accessions), ACCESSION_CHUNK):
        response = _get(f"{UNIPROT_REST}/accessions", f"UniProt {_short(chunk)}", hints=UNIPROT_HINTS,
                        params={"accessions": ",".join(chunk), "format": "tsv", "fields": UNIPROT_FIELDS})
        frames.append(_tsv_frame(response.text))
    for chunk in _chunks(list(entry_names), NAME_CHUNK):
        query = "(" + " OR ".join(f"id:{name}" for name in chunk) + ")"
        response = _get(f"{UNIPROT_REST}/stream", f"UniProt {_short(chunk)}", hints=UNIPROT_HINTS,
                        params={"query": query, "format": "tsv", "fields": UNIPROT_FIELDS})
        frame = _tsv_frame(response.text)
        frames.append(frame[frame["Entry name"].str.upper().isin(chunk)])
    frame = _concat(frames)
    found = set(frame["Accession"].str.upper()) | set(frame["Entry name"].str.upper())
    missing = [i for i in [*accessions, *entry_names] if i not in found]
    return frame, missing


def uniprot_query_count(query):
    """Number of UniProtKB entries matching a query (from the X-Total-Results header)."""
    response = _get(f"{UNIPROT_REST}/search", "UniProt search", hints=UNIPROT_HINTS,
                    params={"query": query, "size": 1, "format": "tsv", "fields": "accession"})
    return int(response.headers.get("x-total-results", 0))


def uniprot_query_tsv(query, isoforms=False):
    """Preview table for every entry matching a query (call uniprot_query_count first to respect the cap)."""
    response = _get(f"{UNIPROT_REST}/stream", "UniProt search", hints=UNIPROT_HINTS,
                    params={"query": query, "format": "tsv", "fields": UNIPROT_FIELDS,
                            "includeIsoform": "true" if isoforms else "false"})
    return _tsv_frame(response.text)


def gene_query(genes, taxon, reviewed=True):
    """UniProt query for exact gene names in one organism."""
    query = "(" + " OR ".join(f"gene_exact:{g}" for g in genes) + f") AND organism_id:{int(taxon)}"
    return query + (" AND reviewed:true" if reviewed else "")


def uniprot_gene_tsv(genes, taxon, reviewed=True, isoforms=False):
    """Preview table for gene names in one organism (batched, <= 100 genes per request).
    Returns (frame, genes with no entry)."""
    genes = list(dict.fromkeys(g.strip().upper() for g in genes if g.strip()))
    frame = _concat([uniprot_query_tsv(gene_query(chunk, taxon, reviewed), isoforms)
                     for chunk in _chunks(genes, GENE_CHUNK)])
    found = {g.strip().upper() for cell in frame["Gene"] for g in str(cell).split(";")}
    return frame, [g for g in genes if g not in found]


def fetch_uniprot_fasta(ids):
    """FASTA for UniProt accessions, batched (<= 500 per request). Raises FetchError naming any missing ID."""
    texts = []
    for chunk in _chunks(list(ids), ACCESSION_CHUNK):
        response = _get(f"{UNIPROT_REST}/accessions", f"UniProt {_short(chunk)}", hints=UNIPROT_HINTS,
                        params={"accessions": ",".join(chunk), "format": "fasta"})
        texts.append(response.text.strip())
    fasta = "\n".join(t for t in texts if t) + "\n"
    returned = {line.split("|")[1].upper() for line in fasta.splitlines() if line.startswith(">") and "|" in line}
    missing = [i for i in ids if i.upper() not in returned]
    if missing and len(missing) == len(ids):
        raise FetchError(f"UniProt returned no entry for {_short(list(ids))}.")
    if missing:
        raise FetchError(f"UniProt returned no entry for: {_short(missing)}")
    return fasta


def _accession_key(token):
    return token.split(".")[0].upper()


def fetch_ncbi_fasta(ids, db):
    params = {"db": db, "id": ",".join(ids), "rettype": "fasta", "retmode": "text"}
    hints = {
        400: "bad request; the database may be wrong (e.g. a nucleotide accession under NCBI Protein).",
        429: "NCBI rate limit reached; wait a few seconds and try again.",
    }
    text = _get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", f"NCBI {', '.join(ids)}", params,
                hints).text
    if not text.lstrip().startswith(">"):
        raise FetchError(f"NCBI returned no FASTA for {', '.join(ids)}.")
    returned = {_accession_key(line[1:].split()[0]) for line in text.splitlines()
                if line.startswith(">") and line[1:].split()}
    missing = [i for i in ids if _accession_key(i) not in returned]
    if missing:
        raise FetchError(f"NCBI returned no record for: {', '.join(missing)}")
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
