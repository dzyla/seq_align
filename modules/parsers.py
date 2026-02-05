import streamlit as st
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder
from io import StringIO
import os
import tempfile
import traceback
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


def parse_sequences_from_structure(file, format_name):
    """
    Parse sequences from PDB or mmCIF structure files.

    Parameters:
        file: Uploaded file object
        format_name (str): Format of the file ('PDB' or 'mmCIF')

    Returns:
        tuple: (sequences, error) - List of sequence records or None, and error message or None
    """
    try:
        file.seek(0)
        file_content = file.read()
        file_basename = os.path.splitext(os.path.basename(file.name))[0]

        # Use temporary files for reliable parsing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_name.lower()}") as temp_file:
            temp_file.write(file_content)
            temp_filepath = temp_file.name

        try:
            # Parse the structure
            if format_name == "PDB":
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(file_basename, temp_filepath)
            elif format_name == "mmCIF":
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure(file_basename, temp_filepath)
            else:
                return None, f"Unsupported format: {format_name}"

            # Extract sequences using multiple methods
            sequences = []
            ppb = PPBuilder()

            with st.sidebar.expander("Structure processing details"):
                for model_idx, model in enumerate(structure, 1):
                    st.write(f"Processing model {model_idx}...")

                    for chain in model:
                        chain_id = chain.id
                        if chain_id == " " or not list(chain.get_residues()):
                            continue

                        st.write(f"Extracting chain {chain_id}...")

                        # Method 1: Use PPBuilder
                        peptides = list(ppb.build_peptides(chain))
                        if peptides:
                            sequence = "".join(str(pp.get_sequence()) for pp in peptides)
                            if sequence:
                                seq_record = SeqRecord(
                                    Seq(sequence),
                                    id=f"{file_basename}_{chain_id}",
                                    description=f"Chain {chain_id} from {file.name}"
                                )
                                sequences.append(seq_record)
                                st.write(f"✅ Extracted {len(sequence)} residues")
                                continue

                        # Method 2: Manual extraction for problematic chains
                        st.write("Standard extraction failed, trying manual extraction...")
                        try:
                            aa_sequence = ""
                            valid_residues = 0

                            for res in chain:
                                # Skip non-standard residues and waters
                                if res.id[0] != " " and not res.id[0].startswith("H_"):
                                    continue

                                resname = res.resname.strip()
                                aa = amino_acid_map.get(resname, None)
                                if aa:
                                    aa_sequence += aa
                                    valid_residues += 1
                                elif len(resname) == 3 and resname[0].isalpha():
                                    aa_sequence += "X"
                                    valid_residues += 1

                            if valid_residues >= 5:  # At least 5 valid residues
                                seq_record = SeqRecord(
                                    Seq(aa_sequence),
                                    id=f"{file_basename}_{chain_id}_manual",
                                    description=f"Manually extracted from chain {chain_id}"
                                )
                                sequences.append(seq_record)
                                st.write(f"✅ Manually extracted {valid_residues} residues")
                        except Exception as chain_err:
                            st.write(f"❌ Failed manual extraction: {chain_err}")

            if not sequences:
                return None, "No protein sequences could be extracted from the structure file."

            # Check for duplicate sequence IDs
            sequence_ids = set()
            for seq in sequences:
                if seq.id in sequence_ids:
                    return None, f"Duplicate sequence ID found: '{seq.id}'. Each sequence must have a unique ID."
                sequence_ids.add(seq.id)

            return sequences, None
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_filepath)
            except:
                pass

    except Exception as e:
        traceback.print_exc()  # For debugging
        return None, f"An error occurred while parsing the {format_name} file: {str(e)}"
