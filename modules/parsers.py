from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, MMCIFParser
from io import StringIO
import os
from modules.utils import amino_acid_map

def parse_sequences_from_text(text):
    """
    Parse sequences from pasted text in FASTA format.

    Parameters:
        text (str): FASTA formatted text with sequences

    Returns:
        tuple: (sequences, error) - List of sequence records or None, and error message or None
    """
    try:
        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [line.strip() for line in normalized_text.split('\n') if line.strip() != '']
        cleaned_text = '\n'.join(lines)
        if not cleaned_text.startswith('>'):
            return None, "FASTA format should start with '>'. Please check your input."
        seq_io = StringIO(cleaned_text)
        sequences = list(SeqIO.parse(seq_io, "fasta"))
        if not sequences:
            return None, "No valid FASTA sequences found. Please check your input."
        sequence_ids = set()
        for seq in sequences:
            if seq.id in sequence_ids:
                return None, f"Duplicate sequence ID found: '{seq.id}'. Each sequence must have a unique ID."
            sequence_ids.add(seq.id)
            if not str(seq.seq).strip():
                return None, f"Sequence '{seq.id}' is empty. Please provide valid sequences."
        return sequences, None
    except Exception as e:
        return None, f"An error occurred while parsing the text input: {e}"


def parse_sequences_from_file(file, format_name):
    """
    Parse sequences from an uploaded file based on the selected format.

    Parameters:
        file: Uploaded file object
        format_name (str): Format of the file

    Returns:
        tuple: (sequences, error) - List of sequence records or None, and error message or None
    """
    try:
        if format_name.lower() == "newick":
            return None, "Newick format is for phylogenetic trees, not sequences."
        file.seek(0)
        file_content = file.read().decode("utf-8")
        seq_io = StringIO(file_content)
        sequences = list(SeqIO.parse(seq_io, format_name.lower()))
        if not sequences:
            return None, f"No sequences found in the uploaded {format_name} file."

        sequence_ids = set()
        for seq in sequences:
            if seq.id in sequence_ids:
                return None, f"Duplicate sequence ID found: '{seq.id}'. Each sequence must have a unique ID."
            sequence_ids.add(seq.id)
            if not str(seq.seq).strip():
                return None, f"Sequence '{seq.id}' is empty. Please provide valid sequences."

        if format_name.lower() == 'clustal':
            for seq_record in sequences:
                seq_record.seq = seq_record.seq.ungap("-")

        return sequences, None
    except Exception as e:
        return None, f"Error parsing {format_name} file: {e}"


def polymer_residues(chain):
    """Residues of the polymer: standard residues plus modified amino acids such as MSE.

    Waters and ligands (e.g. NAG, HEM) are excluded.
    """
    return [res for res in chain
            if res.id[0] == " " or (res.id[0].startswith("H_") and res.resname.strip() in amino_acid_map)]


def make_unique_ids(records):
    """Return records with repeated IDs suffixed _2, _3, ... (first occurrence unchanged)."""
    seen = {}
    unique = []
    for record in records:
        count = seen.get(record.id, 0) + 1
        seen[record.id] = count
        if count > 1:
            record = SeqRecord(record.seq, id=f"{record.id}_{count}", name=record.name,
                               description=record.description, annotations=dict(record.annotations))
        unique.append(record)
    return unique


def parse_structure_text(text, name, format_name, all_models=False):
    """Extract one protein sequence per chain from PDB/mmCIF text. Returns (records, error)."""
    try:
        base = os.path.splitext(os.path.basename(name))[0]
        if format_name == "PDB":
            parser = PDBParser(QUIET=True)
        elif format_name == "mmCIF":
            parser = MMCIFParser(QUIET=True)
        else:
            return None, f"Unsupported format: {format_name}"
        structure = parser.get_structure(base, StringIO(text))
        models = list(structure) if all_models else list(structure)[:1]
        multi = len(models) > 1
        sequences = []
        for model_idx, model in enumerate(models, 1):
            for chain in model:
                residues = polymer_residues(chain)
                letters = [amino_acid_map.get(res.resname.strip(), "X") for res in residues]
                if not letters or all(letter == "X" for letter in letters):
                    continue  # empty, nucleic acid, or unknown polymer
                chain_label = chain.id.strip() or "_"
                seq_id = f"{base}_m{model_idx}_{chain_label}" if multi else f"{base}_{chain_label}"
                description = f"Chain {chain_label} from {name}" + (f", model {model_idx}" if multi else "")
                record = SeqRecord(Seq("".join(letters)), id=seq_id, description=description)
                record.annotations["structure"] = {"file": name, "chain": chain.id, "model": model_idx}
                sequences.append(record)
        if not sequences:
            return None, "No protein sequences could be extracted from the structure file."
        return make_unique_ids(sequences), None
    except Exception as e:
        return None, f"An error occurred while parsing the {format_name} file: {e}"


def parse_sequences_from_structure(file, format_name, all_models=False):
    """File-upload wrapper around parse_structure_text (first model only unless all_models)."""
    file.seek(0)
    raw = file.read()
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    return parse_structure_text(text, file.name, format_name, all_models)
