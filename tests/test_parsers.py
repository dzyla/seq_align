import io
import pathlib

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.parsers import make_unique_ids, parse_sequences_from_structure, parse_structure_text

DATA = pathlib.Path(__file__).parent / "data"


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


def _suffixes(records):
    return {r.id.rsplit("_", 1)[-1] for r in records}


def test_seqres_mmcif_uses_author_chains_and_skips_nucleic_acids():
    text = (DATA / "1a02.cif").read_text()
    atom, err = parse_structure_text(text, "1a02.cif", "mmCIF")
    assert err is None
    seqres, err = parse_structure_text(text, "1a02.cif", "mmCIF", source="seqres")
    assert err is None
    assert _suffixes(seqres) == _suffixes(atom)
    assert not any(set(str(r.seq)) <= {"X"} for r in seqres)


def test_seqres_dna_only_structure_has_no_protein():
    text = (DATA / "1bna.cif").read_text()
    records, err = parse_structure_text(text, "1bna.cif", "mmCIF", source="seqres")
    assert records is None and err
