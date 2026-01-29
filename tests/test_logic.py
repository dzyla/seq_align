import unittest
import sys
import os

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from app import parse_sequences_from_text, perform_pairwise_alignment

class TestAppLogic(unittest.TestCase):

    def test_parse_sequences_from_text_valid(self):
        text = ">Seq1\nATGC\n>Seq2\nCGTA"
        sequences, error = parse_sequences_from_text(text)
        self.assertIsNone(error)
        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences[0].id, "Seq1")
        self.assertEqual(str(sequences[0].seq), "ATGC")

    def test_parse_sequences_from_text_invalid_format(self):
        text = "Seq1\nATGC"
        sequences, error = parse_sequences_from_text(text)
        self.assertIsNone(sequences)
        self.assertIn("FASTA format should start with '>'", error)

    def test_parse_sequences_from_text_duplicate_id(self):
        text = ">Seq1\nATGC\n>Seq1\nCGTA"
        sequences, error = parse_sequences_from_text(text)
        self.assertIsNone(sequences)
        self.assertIn("Duplicate sequence ID found", error)

    def test_perform_pairwise_alignment_robustness(self):
        seq1 = SeqRecord(Seq("ATGC"), id="Seq1")
        seq2 = SeqRecord(Seq("ATGG"), id="Seq2")

        # Use high gap penalty to force mismatch instead of gaps
        alignment_text, mutations = perform_pairwise_alignment(
            seq1, seq2, "DNA", open_gap_score=-10.0, extend_gap_score=-1.0
        )

        self.assertIn("Pairwise Alignment", alignment_text)
        self.assertTrue(len(mutations) > 0)
        self.assertEqual(mutations[0], "C4G")

    def test_perform_pairwise_alignment_empty(self):
        # Biopython pairwise aligner handles empty sequences gracefully usually, but let's check
        seq1 = SeqRecord(Seq(""), id="Seq1")
        seq2 = SeqRecord(Seq("ATGG"), id="Seq2")

        alignment_text, mutations = perform_pairwise_alignment(seq1, seq2, "DNA")
        # Should return error message string
        self.assertTrue(isinstance(alignment_text, str))
        self.assertIn("One or both sequences are empty", alignment_text)

if __name__ == '__main__':
    unittest.main()
