from pyfamsa import Aligner, Sequence


def align(pairs, guide_tree="upgma"):
    """Align (id, sequence) pairs with pyFAMSA; runs in a child process because pyFAMSA can crash."""
    sequences = [Sequence(i.encode(), s.encode()) for i, s in pairs]
    msa = Aligner(guide_tree=guide_tree).align(sequences)
    return [(s.id.decode(), s.sequence.decode()) for s in msa]
