from modules.structures import StructureFile, parse_structure

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
