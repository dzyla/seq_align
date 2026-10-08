"""pyFAMSA worker, run as a separate Python process (by file path) because pyFAMSA can crash the interpreter.

Reads JSON {"pairs": [[id, seq], ...], "guide_tree": "upgma"} on stdin and writes the aligned pairs as JSON
on stdout. It imports nothing from the app, so it never re-runs the Streamlit page script.
"""
import json
import sys

from pyfamsa import Aligner, Sequence


def align(pairs, guide_tree="upgma"):
    """Align (id, sequence) pairs with pyFAMSA."""
    sequences = [Sequence(i.encode(), s.encode()) for i, s in pairs]
    msa = Aligner(guide_tree=guide_tree).align(sequences)
    return [(s.id.decode(), s.sequence.decode()) for s in msa]


if __name__ == "__main__":
    request = json.load(sys.stdin)
    json.dump(align(request["pairs"], request.get("guide_tree", "upgma")), sys.stdout)
