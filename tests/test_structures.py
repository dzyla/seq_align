import pathlib

from modules.structures import ResidueInfo, StructureFile, chain_residue_table, parse_structure

DATA = pathlib.Path(__file__).parent / "data"
CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"

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


def test_chain_residue_table_1crn():
    sf = StructureFile(name="1crn.pdb", data=(DATA / "1crn.pdb").read_bytes(), fmt="PDB")
    table = chain_residue_table(parse_structure(sf))
    assert list(table) == ["A"]
    residues = table["A"]
    assert "".join(r.aa for r in residues) == CRAMBIN
    assert residues[0] == ResidueInfo("A", 1, "", "THR", "T", residues[0].bfactor)
    assert residues[0].bfactor > 0
