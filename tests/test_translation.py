import unittest
import sys
import os
from Bio.Seq import Seq

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dna import find_orfs

class TestTranslation(unittest.TestCase):

    def test_find_orfs_simple(self):
        # Frame 1: ATG CCC TGA (M P *)
        seq = "ATGCCCTGA"
        orfs = find_orfs(seq, min_len=1)
        self.assertEqual(len(orfs), 1)
        self.assertEqual(orfs[0]['Sequence'], "MP")
        self.assertEqual(orfs[0]['Frame'], "+1")

    def test_find_orfs_reverse(self):
        # Reverse complement of ATGCCCTGA is TCAGGGCAT
        # So providing TCAGGGCAT should find the same ORF in frame -1
        seq = "TCAGGGCAT"
        orfs = find_orfs(seq, min_len=1)
        self.assertEqual(len(orfs), 1)
        self.assertEqual(orfs[0]['Sequence'], "MP")
        self.assertEqual(orfs[0]['Frame'], "-1")

    def test_find_orfs_length_filter(self):
        seq = "ATGCCCTGA"
        orfs = find_orfs(seq, min_len=5)
        self.assertEqual(len(orfs), 0)

    def test_find_orfs_multiple(self):
        # M P * M A *
        # ATGCCCTGA ATGGCT TGA
        seq = "ATGCCCTGAATGGCTTGA"
        orfs = find_orfs(seq, min_len=1)
        self.assertEqual(len(orfs), 2)
        self.assertEqual(orfs[0]['Sequence'], "MP")
        self.assertEqual(orfs[1]['Sequence'], "MA")

if __name__ == '__main__':
    unittest.main()
