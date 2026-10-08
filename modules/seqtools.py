"""Sequence helpers shared by all analysis modules. No Streamlit calls."""

DNA_LETTERS = frozenset("ACGTNRYKMSWBDHV")
# Alphabet of BLOSUM62 as shipped with Biopython.
PROTEIN_LETTERS = frozenset("ARNDCQEGHILKMFPSTWYVBZX*")
# Valid FASTA residues absent from BLOSUM62, mapped to the closest standard residue.
PROTEIN_SUBSTITUTIONS = {"U": "C", "O": "K", "J": "X"}


def clean_sequence(seq, seq_type, keep_gaps=False):
    """Upper-case a sequence and reduce it to the scoring alphabet.

    DNA: U becomes T; anything outside IUPAC nucleotide codes is dropped.
    Protein: U->C, O->K, J->X; anything outside the BLOSUM62 alphabet is dropped.
    Gaps ('-') are dropped unless keep_gaps is True.
    """
    text = str(seq).upper()
    if seq_type == "DNA":
        text = text.replace("U", "T")
        allowed = DNA_LETTERS
    else:
        text = "".join(PROTEIN_SUBSTITUTIONS.get(c, c) for c in text)
        allowed = PROTEIN_LETTERS
    return "".join(c for c in text if c in allowed or (keep_gaps and c == "-"))
