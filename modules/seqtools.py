"""Sequence helpers shared by all analysis modules. No Streamlit calls."""
import hashlib

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


def fingerprint(records):
    """Order-sensitive SHA-1 over sequence IDs and residues; identifies a dataset."""
    digest = hashlib.sha1()
    for record in records:
        digest.update(record.id.encode())
        digest.update(b"\t")
        digest.update(str(record.seq).encode())
        digest.update(b"\n")
    return digest.hexdigest()


_NUCLEOTIDE_LETTERS = frozenset("ACGTUN")


def detect_seq_type(seqs, dna_fraction=0.9):
    """'DNA' if at least `dna_fraction` of all letters are A/C/G/T/U/N, else 'Protein'."""
    letters = [c for s in seqs for c in str(s).upper() if c.isalpha()]
    if not letters:
        return "Protein"
    nucleotide = sum(1 for c in letters if c in _NUCLEOTIDE_LETTERS)
    return "DNA" if nucleotide / len(letters) >= dna_fraction else "Protein"
