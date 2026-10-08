import unittest
import numpy as np
from modules.clustering import calculate_similarity_matrix_pairwise, find_clusters
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
        matrix, ids = calculate_similarity_matrix_pairwise(self.sequences, "DNA", "Global")

        self.assertIsNotNone(matrix)
        self.assertEqual(ids, ["seq1", "seq2", "seq3", "seq4"])

        # Check specific values
        # seq1 vs seq2 should be 1.0 (5/5 matches)
        self.assertEqual(matrix[0, 1], 1.0)

        # seq1 vs seq3 should be 0.0 (0/5 matches)
        self.assertEqual(matrix[0, 2], 0.0)

        # seq1 vs seq4 should be 0.8 (4/5 matches)
        self.assertAlmostEqual(matrix[0, 3], 0.8)

    def test_msa_identity_ignores_double_gap_columns(self):
        from modules.clustering import msa_identity_matrix
        m = msa_identity_matrix(["MKV---LAA", "MKV---LAA", "MKVGGGLAA", "MKA------"])
        self.assertEqual(m[0, 1], 1.0)
        self.assertAlmostEqual(m[0, 2], 6 / 9)
        self.assertAlmostEqual(m[0, 3], 2 / 6)

    def test_local_identity_normalized_by_shorter_sequence(self):
        a = SeqRecord(Seq("W" * 8 + "H" * 32), id="a")
        b = SeqRecord(Seq("W" * 8 + "K" * 33), id="b")
        matrix, _ = calculate_similarity_matrix_pairwise([a, b], "Protein", "Local")
        self.assertAlmostEqual(matrix[0, 1], 8 / 40)

    def test_pairwise_matrix_lowercase(self):
        a = SeqRecord(Seq("mkvlaa"), id="a")
        b = SeqRecord(Seq("MKVLAA"), id="b")
        matrix, _ = calculate_similarity_matrix_pairwise([a, b], "Protein", "Global")
        self.assertEqual(matrix[0, 1], 1.0)

    def test_hierarchical_clusters(self):
        from modules.clustering import hierarchical_clusters
        sim = np.array([[1, .95, .2], [.95, 1, .2], [.2, .2, 1]])
        clusters, order = hierarchical_clusters(sim, ["a", "b", "c"], 0.9)
        self.assertEqual(sorted(map(sorted, clusters)), [["a", "b"], ["c"]])
        self.assertEqual(sorted(order), [0, 1, 2])

    def test_suggest_threshold_separates_modes(self):
        from modules.clustering import suggest_threshold
        rng = np.random.default_rng(0)
        values = np.clip(np.r_[rng.normal(0.3, 0.05, 200), rng.normal(0.9, 0.03, 50)], 0, 1)
        t = suggest_threshold(values)
        self.assertTrue(0.35 < t < 0.85, t)
        self.assertEqual(suggest_threshold(np.array([])), 0.8)

    def test_pcoa_preserves_neighbourhoods(self):
        from modules.clustering import perform_pcoa
        coords = perform_pcoa(np.array([[1, .9, .1], [.9, 1, .1], [.1, .1, 1]]))
        self.assertEqual(coords.shape, (3, 2))
        self.assertLess(np.linalg.norm(coords[0] - coords[1]), np.linalg.norm(coords[0] - coords[2]))

if __name__ == '__main__':
    unittest.main()
