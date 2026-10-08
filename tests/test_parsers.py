import io

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.parsers import make_unique_ids, parse_sequences_from_structure


def pdb_lines(residues, chain="A"):
    """residues: list of (record, resname, number, icode)."""
    lines = []
    for serial, (record, resname, num, icode) in enumerate(residues, 1):
        lines.append(f"{record:<6}{serial:>5}  CA  {resname:>3} {chain}{num:>4}{icode:1}   "
                     f"{serial * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    return "\n".join(lines) + "\n"


class Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


PEPTIDE = [("ATOM", r, i, " ") for i, r in enumerate(["MET", "LYS", "VAL", "LEU", "ALA", "GLY"], 1)]


def test_nmr_first_model_only():
    text = "".join(f"MODEL        {m}\n" + pdb_lines(PEPTIDE) + "ENDMDL\n" for m in (1, 2)) + "END\n"
    records, error = parse_sequences_from_structure(Upload(text.encode(), "nmr.pdb"), "PDB")
    assert error is None
    assert [r.id for r in records] == ["nmr_A"]
    assert str(records[0].seq) == "MKVLAG"


def test_nmr_all_models():
    text = "".join(f"MODEL        {m}\n" + pdb_lines(PEPTIDE) + "ENDMDL\n" for m in (1, 2)) + "END\n"
    records, error = parse_sequences_from_structure(Upload(text.encode(), "nmr.pdb"), "PDB", all_models=True)
    assert error is None
    assert [r.id for r in records] == ["nmr_m1_A", "nmr_m2_A"]


def test_ligands_excluded_and_mse_included():
    residues = PEPTIDE + [("HETATM", "MSE", 7, " "), ("HETATM", "NAG", 101, " "), ("HETATM", "HOH", 201, " ")]
    records, error = parse_sequences_from_structure(Upload((pdb_lines(residues) + "END\n").encode(), "x.pdb"), "PDB")
    assert error is None
    assert str(records[0].seq) == "MKVLAGM"


def test_make_unique_ids():
    recs = [SeqRecord(Seq("A"), id="x_A"), SeqRecord(Seq("C"), id="x_A"), SeqRecord(Seq("D"), id="y_A")]
    assert [r.id for r in make_unique_ids(recs)] == ["x_A", "x_A_2", "y_A"]
