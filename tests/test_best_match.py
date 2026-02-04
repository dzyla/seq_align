import unittest
import sys
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import get_aligner, calculate_alignment_score, find_best_match

class TestBestMatchFinder(unittest.TestCase):

    def test_get_aligner(self):
        aligner = get_aligner("DNA", mode="global", open_gap_score=-2.0, extend_gap_score=-0.5)
        self.assertIsInstance(aligner, Align.PairwiseAligner)
        self.assertEqual(aligner.mode, "global")
        self.assertEqual(aligner.open_gap_score, -2.0)
        self.assertEqual(aligner.extend_gap_score, -0.5)

    def test_calculate_alignment_score(self):
        aligner = get_aligner("DNA")
        seq1 = SeqRecord(Seq("ATGC"), id="S1")
        seq2 = SeqRecord(Seq("ATGC"), id="S2")
        score = calculate_alignment_score(aligner, seq1, seq2, "DNA")
        # Score for match is usually 1 per base in standard NUC.4.4 if identity, but checking relative
        self.assertIsNotNone(score)
        self.assertTrue(score > 0)

        # Exact mismatch
        seq3 = SeqRecord(Seq("AAAA"), id="S3")
        seq4 = SeqRecord(Seq("TTTT"), id="S4")
        score_mismatch = calculate_alignment_score(aligner, seq3, seq4, "DNA")
        self.assertTrue(score > score_mismatch)

    def test_find_best_match(self):
        aligner = get_aligner("DNA")
        ref = SeqRecord(Seq("ATGCATGC"), id="Ref")
        s1 = SeqRecord(Seq("ATGCATGC"), id="S1") # Perfect match
        s2 = SeqRecord(Seq("ATGCAAAA"), id="S2") # Partial match
        s3 = SeqRecord(Seq("GGGGGGGG"), id="S3") # Mismatch

        sequences = [ref, s1, s2, s3]

        best_match, results = find_best_match(sequences, "Ref", aligner, "DNA")

        self.assertEqual(best_match.id, "S1")
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['Sequence ID'], "S1")
        self.assertEqual(results[0]['Rank'], 1)
        self.assertTrue(results[0]['Score'] > results[1]['Score'])

    def test_find_best_match_protein(self):
        aligner = get_aligner("Protein")
        ref = SeqRecord(Seq("MVLSPADKTN"), id="Ref")
        s1 = SeqRecord(Seq("MVLSPADKTN"), id="S1") # Perfect
        s2 = SeqRecord(Seq("MVLSPADK"), id="S2") # Truncated

        sequences = [ref, s1, s2]
        best_match, results = find_best_match(sequences, "Ref", aligner, "Protein")

        self.assertEqual(best_match.id, "S1")

if __name__ == '__main__':
    unittest.main()
