import streamlit as st
import io
import os
import tempfile
import traceback
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pyfamsa import Aligner as PyFAMSAAligner, Sequence as PyFAMSASequence
from modules.utils import amino_acid_map

def align_with_pyfamsa(seq1_id, seq1_str, seq2_id, seq2_str):
    """
    Aligns two sequences using PyFAMSA and returns the aligned strings.
    """
    pyfamsa_sequences = [
        PyFAMSASequence(seq1_id.encode(), seq1_str.encode()),
        PyFAMSASequence(seq2_id.encode(), seq2_str.encode())
    ]
    aligner = PyFAMSAAligner(guide_tree="upgma")
    msa = aligner.align(pyfamsa_sequences)

    aligned_seq1 = ""
    aligned_seq2 = ""
    for seq in msa:
        if seq.id.decode() == seq1_id:
            aligned_seq1 = seq.sequence.decode()
        elif seq.id.decode() == seq2_id:
            aligned_seq2 = seq.sequence.decode()

    return aligned_seq1, aligned_seq2

def get_mapping_from_alignment(aligned_target, aligned_chain):
    """
    Given an alignment of target and chain, returns a dictionary mapping
    the 0-based index of the chain's un-gapped sequence to the
    1-based index of the target's un-gapped sequence.
    """
    mapping = {}
    target_pos = 1
    chain_pos = 0

    for t_char, c_char in zip(aligned_target, aligned_chain):
        if c_char != '-':
            if t_char != '-':
                mapping[chain_pos] = target_pos
            chain_pos += 1

        if t_char != '-':
            target_pos += 1

    return mapping

def pdb_renumber_section(target_sequences):
    """
    UI and workflow for PDB Residue Renumbering based on a target sequence.
    """
    st.header("🔢 PDB Residue Renumbering")
    st.info(
        "Upload a PDB file. We will match its chains to your uploaded target sequences (FASTA) "
        "and renumber the PDB residues to match the target sequences perfectly (1-based index). "
        "Gaps or unmodeled regions will be skipped."
    )

    if not target_sequences:
        st.warning("Please upload target sequences (FASTA) in the sidebar first.")
        return

    pdb_file = st.file_uploader(
        "Upload PDB File to Renumber",
        type=["pdb", "ent"],
        help="Upload the PDB file you want to renumber based on the target sequences"
    )

    if pdb_file:
        # Cache the alignment processing to avoid rerunning when UI updates
        if 'pdb_renumber_results' not in st.session_state or st.session_state.pdb_renumber_file != pdb_file.name:
            with st.spinner("Processing PDB and matching sequences with PyFAMSA..."):
                try:
                    pdb_file.seek(0)
                    parser = PDBParser(QUIET=True)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
                        temp_file.write(pdb_file.read())
                        temp_filepath = temp_file.name

                    try:
                        structure = parser.get_structure("renumber", temp_filepath)
                    finally:
                        os.unlink(temp_filepath)

                    # Compute alignments and proposed mappings
                    mappings, logs, alignment_visuals = prepare_renumbering(structure, target_sequences)

                    st.session_state.pdb_renumber_results = {
                        'structure': structure,
                        'mappings': mappings,
                        'logs': logs,
                        'visuals': alignment_visuals,
                    }
                    st.session_state.pdb_renumber_file = pdb_file.name
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.text(traceback.format_exc())
                    return

        # Display Interactive Review and Confirmation
        if 'pdb_renumber_results' in st.session_state:
            results = st.session_state.pdb_renumber_results
            st.subheader("Interactive Review & Confirmation")

            with st.expander("Review Sequence Alignments (PDB vs Target)", expanded=True):
                for chain_id, target_id, aln_text in results['visuals']:
                    st.markdown(f"**Chain {chain_id}** matched to **{target_id}**")
                    st.text(aln_text)

            with st.expander("Renumbering Logs"):
                for msg in results['logs']:
                    st.write(msg)

            st.warning("Please review the alignment above. If it looks correct, confirm the renumbering to generate the final PDB.")

            if st.button("Confirm Renumbering & Generate PDB"):
                with st.spinner("Renumbering PDB residues..."):
                    renumbered_structure = apply_renumbering(results['structure'], results['mappings'])

                    io_out = PDBIO()
                    io_out.set_structure(renumbered_structure)

                    out_stream = io.StringIO()
                    io_out.save(out_stream)
                    pdb_content = out_stream.getvalue()

                    st.success("Renumbering completed successfully!")
                    st.download_button(
                        label="📥 Download Renumbered PDB",
                        data=pdb_content,
                        file_name=f"{pdb_file.name.split('.')[0]}_renumbered.pdb",
                        mime="chemical/x-pdb"
                    )

def prepare_renumbering(structure: Structure, target_sequences: list):
    """
    Calculates the sequence mappings without modifying the structure.
    Returns mappings per chain, log messages, and alignment text.
    """
    log_messages = []
    alignment_visuals = []
    mappings = {}

    for model in structure:
        for chain in model:
            chain_id = chain.id

            chain_seq = ""
            residues = []
            for res in chain:
                if res.id[0] != " " and not res.id[0].startswith("H_"):
                    continue
                resname = res.resname.strip()
                aa = amino_acid_map.get(resname, "X")
                chain_seq += aa
                residues.append(res)

            if not chain_seq:
                log_messages.append(f"Chain {chain_id}: No standard residues found, skipping.")
                continue

            best_target = None
            best_score = -1
            best_aligned_t = None
            best_aligned_c = None

            for target in target_sequences:
                target_seq = str(target.seq).upper()

                # Use PyFAMSA
                aligned_t, aligned_c = align_with_pyfamsa(target.id, target_seq, f"chain_{chain_id}", chain_seq)

                # Simple score: count matches
                matches = sum(1 for a, b in zip(aligned_t, aligned_c) if a == b and a != '-')
                if matches > best_score:
                    best_score = matches
                    best_target = target
                    best_aligned_t = aligned_t
                    best_aligned_c = aligned_c

            if not best_target:
                log_messages.append(f"Chain {chain_id}: Could not find a matching target sequence.")
                continue

            mapping = get_mapping_from_alignment(best_aligned_t, best_aligned_c)
            mappings[chain_id] = mapping

            aln_text = f"Target ({best_target.id}): {best_aligned_t}\nChain  ({chain_id})       : {best_aligned_c}"
            alignment_visuals.append((chain_id, best_target.id, aln_text))
            log_messages.append(f"Chain {chain_id}: Best match is {best_target.id} with {best_score} matching residues.")

    return mappings, log_messages, alignment_visuals

def apply_renumbering(structure: Structure, mappings: dict):
    """
    Applies the mappings to the structure residues.
    """
    import copy
    renumbered_structure = copy.deepcopy(structure)

    for model in renumbered_structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in mappings:
                continue

            mapping = mappings[chain_id]

            # Extract standard residues that were mapped
            residues = []
            for res in chain:
                if res.id[0] == " " or res.id[0].startswith("H_"):
                    # Only map standard residues, skip waters/heteroatoms.
                    residues.append(res)

            # Keep track of original numbering before shifting
            original_ids = {i: res.id for i, res in enumerate(residues)}

            # Shift up to avoid unique id collision within a chain
            for i, res in enumerate(residues):
                res.id = (res.id[0], res.id[1] + 100000, res.id[2])

            # Assign real numbering
            for i, res in enumerate(residues):
                if i in mapping:
                    new_num = mapping[i]
                    res.id = (res.id[0], new_num, res.id[2])
                else:
                    # If unaligned, restore original ID
                    res.id = original_ids[i]

    return renumbered_structure
