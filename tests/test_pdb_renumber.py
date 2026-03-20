import unittest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, StructureBuilder
from modules.pdb_renumber import prepare_renumbering, apply_renumbering

class TestPDBRenumber(unittest.TestCase):
    def test_renumber(self):
        # Create a tiny mock PDB file
        pdb_text = """ATOM      1  N   ALA A   1      11.104  12.204  13.304  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.104  12.204  13.304  1.00  0.00           C
ATOM      3  C   ALA A   1      11.104  12.204  13.304  1.00  0.00           C
ATOM      4  O   ALA A   1      11.104  12.204  13.304  1.00  0.00           O
ATOM      5  N   CYS A   2      11.104  12.204  13.304  1.00  0.00           N
ATOM      6  CA  CYS A   2      11.104  12.204  13.304  1.00  0.00           C
ATOM      7  C   CYS A   2      11.104  12.204  13.304  1.00  0.00           C
ATOM      8  O   CYS A   2      11.104  12.204  13.304  1.00  0.00           O
ATOM      9  N   GLY A   3      11.104  12.204  13.304  1.00  0.00           N
ATOM     10  CA  GLY A   3      11.104  12.204  13.304  1.00  0.00           C
ATOM     11  C   GLY A   3      11.104  12.204  13.304  1.00  0.00           C
ATOM     12  O   GLY A   3      11.104  12.204  13.304  1.00  0.00           O
TER      13      GLY A   3
END
"""
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w") as f:
            f.write(pdb_text)
            temp_path = f.name

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", temp_path)
        os.unlink(temp_path)

        # Target sequences: we have ACG in PDB, let's provide xACGxx where x is some other AA
        target = SeqRecord(Seq("MACGM"), id="target1")

        mappings, logs, alignments = prepare_renumbering(structure, [target])
        renumbered_structure = apply_renumbering(structure, mappings)

        # First residue (ALA) should map to position 2 (since 'A' is index 1, meaning 2nd char in MACGM)
        # MACGM
        #  ACG
        # indices for ACG in MACGM are 1, 2, 3 (0-based) -> 2, 3, 4 (1-based)

        chain = renumbered_structure[0]['A']
        res_list = list(chain.get_residues())
        self.assertEqual(res_list[0].id[1], 2)
        self.assertEqual(res_list[1].id[1], 3)
        self.assertEqual(res_list[2].id[1], 4)

if __name__ == '__main__':
    unittest.main()
