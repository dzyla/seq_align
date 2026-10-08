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

import io as _io

from Bio.PDB import PDBParser as _PDBParser

from modules.pdb_renumber import extract_header_records, plan_residue_numbers


def _pdb(residues, chain="A"):
    lines = []
    for serial, (record, resname, num, icode) in enumerate(residues, 1):
        lines.append(f"{record:<6}{serial:>5}  CA  {resname:>3} {chain}{num:>4}{icode:1}   "
                     f"{serial * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    return "\n".join(lines) + "\nEND\n"


def _structure(text):
    return _PDBParser(QUIET=True).get_structure("t", _io.StringIO(text))


class TestRenumberingPlan(unittest.TestCase):
    def test_no_duplicates_when_unmapped_tail_follows(self):
        self.assertEqual(plan_residue_numbers(6, {0: 4, 1: 5, 2: 6}),
                         [(4, ' '), (5, ' '), (6, ' '), (7, ' '), (8, ' '), (9, ' ')])

    def test_internal_insertion_gets_insertion_code(self):
        self.assertEqual(plan_residue_numbers(4, {0: 1, 1: 2, 3: 3}),
                         [(1, ' '), (2, ' '), (2, 'A'), (3, ' ')])

    def test_n_terminal_overhang_counts_down(self):
        self.assertEqual(plan_residue_numbers(4, {2: 1, 3: 2}),
                         [(-1, ' '), (0, ' '), (1, ' '), (2, ' ')])

    def test_insertion_codes_reset_and_assigned(self):
        # Chain numbered with Kabat-style insertion codes; target equals the chain sequence.
        residues = [("ATOM", "ALA", 52, " "), ("ATOM", "CYS", 52, "A"), ("ATOM", "GLY", 53, " ")]
        structure = _structure(_pdb(residues))
        mappings, _, _ = prepare_renumbering(structure, [SeqRecord(Seq("ACG"), id="t")])
        out = apply_renumbering(structure, mappings)
        self.assertEqual([r.id[1:] for r in out[0]["A"]], [(1, ' '), (2, ' '), (3, ' ')])

    def test_ligands_keep_their_ids(self):
        residues = [("ATOM", "ALA", 1, " "), ("ATOM", "CYS", 2, " "), ("HETATM", "NAG", 2, " ")]
        structure = _structure(_pdb(residues))
        mappings, _, _ = prepare_renumbering(structure, [SeqRecord(Seq("MACG"), id="t")])
        out = apply_renumbering(structure, mappings)
        ids = [r.id for r in out[0]["A"]]
        self.assertIn(("H_NAG", 2, " "), ids)
        self.assertIn((" ", 2, " "), ids)
        self.assertIn((" ", 3, " "), ids)

    def test_all_models_renumbered(self):
        body = _pdb([("ATOM", "ALA", 1, " "), ("ATOM", "CYS", 2, " ")]).replace("END\n", "")
        text = "MODEL        1\n" + body + "ENDMDL\nMODEL        2\n" + body + "ENDMDL\nEND\n"
        structure = _structure(text)
        mappings, _, _ = prepare_renumbering(structure, [SeqRecord(Seq("MAC"), id="t")])
        out = apply_renumbering(structure, mappings)
        for model in out:
            self.assertEqual([r.id[1] for r in model["A"]], [2, 3])

    def test_extract_header_records(self):
        text = "HEADER    X\nCRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\nATOM ...\n"
        self.assertEqual(extract_header_records(text).splitlines()[0][:6], "CRYST1")

if __name__ == '__main__':
    unittest.main()
