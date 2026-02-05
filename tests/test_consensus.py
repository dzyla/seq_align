import unittest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from app import calculate_representative_sequence

class TestConsensusFix(unittest.TestCase):
    def test_calculate_representative_sequence(self):
        # Create a simple alignment
        # Seq1: A T G
        # Seq2: A T C
        # Seq3: A T G
        # Consensus (threshold 0.7): A T G
        # (Pos 1: A (3/3=1.0), Pos 2: T (3/3=1.0), Pos 3: G (2/3=0.66) -> X if threshold 0.7? No 2/3 is 0.66 < 0.7. So X)

        # Let's adjust threshold or sequences to be clear.
        # Seq1: A A A
        # Seq2: A A A
        # Seq3: A A T
        # Threshold 0.7:
        # Col 1: A (3/3=1.0) -> A
        # Col 2: A (3/3=1.0) -> A
        # Col 3: A (2/3=0.66) -> X (if strict > 0.7). Wait, my code is >= threshold. 2/3 is 0.666...
        # If threshold is 0.6, then A.

        seq1 = SeqRecord(Seq("AAA"), id="seq1")
        seq2 = SeqRecord(Seq("AAA"), id="seq2")
        seq3 = SeqRecord(Seq("AAT"), id="seq3")

        alignment = MultipleSeqAlignment([seq1, seq2, seq3])

        # Case 1: High threshold, last column ambiguous
        consensus, closest, min_diff, closest_id = calculate_representative_sequence(alignment, threshold=0.8)
        self.assertEqual(str(consensus.seq), "AAX")

        # Case 2: Lower threshold, last column included
        consensus, closest, min_diff, closest_id = calculate_representative_sequence(alignment, threshold=0.6)
        self.assertEqual(str(consensus.seq), "AAA")
        self.assertEqual(closest_id, "seq1") # seq1 or seq2 are both AAA, dist 0. seq3 is AAT, dist 1.

        # Case 3: Verify closest sequence logic
        # Ref: AAA
        # Seq1: AAA (dist 0)
        # Seq3: AAT (dist 1)
        self.assertEqual(min_diff, 0)
        self.assertIn(closest_id, ["seq1", "seq2"])

if __name__ == '__main__':
    unittest.main()
