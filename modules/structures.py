"""Uploaded or fetched structure files kept as raw bytes; parsed on demand."""
import hashlib
from dataclasses import dataclass
from io import StringIO

from Bio.PDB import MMCIFParser, PDBParser

from modules.parsers import polymer_residues
from modules.utils import amino_acid_map

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
