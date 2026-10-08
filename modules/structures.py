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
