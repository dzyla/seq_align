import unittest
import numpy as np
from app import calculate_similarity_matrix, find_clusters
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

class TestClustering(unittest.TestCase):
    def setUp(self):
        self.seq1 = SeqRecord(Seq("AAAAA"), id="seq1")
        self.seq2 = SeqRecord(Seq("AAAAA"), id="seq2") # Identical to seq1
        self.seq3 = SeqRecord(Seq("TTTTT"), id="seq3") # Very different
        self.seq4 = SeqRecord(Seq("AAAAT"), id="seq4") # Close to seq1
        self.sequences = [self.seq1, self.seq2, self.seq3, self.seq4]

    def test_find_clusters(self):
        ids = ["seq1", "seq2", "seq3"]
        matrix = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        clusters = find_clusters(matrix, ids, threshold=0.9)
        self.assertEqual(len(clusters), 2)
        self.assertIn(["seq1", "seq2"], clusters) # Or similar structure
        self.assertIn(["seq3"], clusters)

    def test_pairwise_similarity(self):
        # Test the "Accurate (Pairwise)" path
        matrix, ids = calculate_similarity_matrix(self.sequences, "Accurate (Pairwise)", "DNA", "Global")

        self.assertIsNotNone(matrix)
        self.assertEqual(ids, ["seq1", "seq2", "seq3", "seq4"])

        # Check specific values
        # seq1 vs seq2 should be 1.0 (5/5 matches)
        self.assertEqual(matrix[0, 1], 1.0)

        # seq1 vs seq3 should be 0.0 (0/5 matches)
        self.assertEqual(matrix[0, 2], 0.0)

        # seq1 vs seq4 should be 0.8 (4/5 matches)
        self.assertAlmostEqual(matrix[0, 3], 0.8)

if __name__ == '__main__':
    unittest.main()
