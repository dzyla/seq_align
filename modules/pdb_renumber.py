import copy
import io
import string
import traceback

import streamlit as st
from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Structure import Structure

from modules.pairwise import get_aligner
from modules.parsers import parse_sequences_from_file, parse_sequences_from_text, polymer_residues
from modules.seqtools import clean_sequence
from modules.utils import amino_acid_map

HEADER_RECORDS = ("CRYST1", "SCALE", "ORIGX", "MTRIX")


def chain_sequence(residues):
    """One-letter sequence of polymer residues (length always equals len(residues))."""
    return "".join(amino_acid_map.get(res.resname.strip(), "X") for res in residues)


def align_target_to_chain(target_seq, chain_seq):
    """Semi-global (free end gaps) BLOSUM62 alignment of target vs chain.

    Returns (aligned_target, aligned_chain, identity over aligned columns).
    `chain_seq` must keep its length through cleaning, which holds for amino_acid_map letters.
    """
    aligner = get_aligner("Protein", mode="overlap")
    target = clean_sequence(target_seq, "Protein")
    chain = clean_sequence(chain_seq, "Protein")
    alignment = next(iter(aligner.align(target, chain)))
    aligned_t, aligned_c = str(alignment[0]), str(alignment[1])
    aligned_cols = sum(1 for a, b in zip(aligned_t, aligned_c) if a != '-' and b != '-')
    matches = sum(1 for a, b in zip(aligned_t, aligned_c) if a == b and a != '-')
    return aligned_t, aligned_c, (matches / aligned_cols if aligned_cols else 0.0)


def get_mapping_from_alignment(aligned_target, aligned_chain):
    """Map 0-based chain residue index -> 1-based target position for aligned columns."""
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


def plan_residue_numbers(n_residues, mapping):
    """Final (residue number, insertion code) for each polymer residue of a chain.

    Mapped residues get their target position with no insertion code. Unmapped residues
    between two mapped ones are insertions relative to the target and get codes A, B, C...
    after the preceding target position. N-terminal overhangs count down from the first
    mapped position; C-terminal overhangs count up from the last. Raises ValueError rather
    than produce duplicate IDs.
    """
    if not mapping:
        raise ValueError("No residue of this chain aligned to the target.")
    mapped = sorted(mapping)
    first, last = mapped[0], mapped[-1]
    numbers = [None] * n_residues
    for i in mapped:
        numbers[i] = (mapping[i], " ")
    for k, i in enumerate(range(first - 1, -1, -1), start=1):
        numbers[i] = (mapping[first] - k, " ")
    for k, i in enumerate(range(last + 1, n_residues), start=1):
        numbers[i] = (mapping[last] + k, " ")
    previous = mapping[first]
    run = 0
    for i in range(first, last + 1):
        if i in mapping:
            previous = mapping[i]
            run = 0
            continue
        if run >= len(string.ascii_uppercase):
            raise ValueError(f"More than 26 consecutive insertions after target position {previous}.")
        numbers[i] = (previous, string.ascii_uppercase[run])
        run += 1
    if len(set(numbers)) != len(numbers):
        raise ValueError("Renumbering would create duplicate residue IDs.")
    return numbers


def prepare_renumbering(structure: Structure, target_sequences: list):
    """Choose the best target for each chain of the first model and compute its mapping.

    Returns (mappings: {chain_id: {residue_index: target_pos}}, log messages, alignment visuals).
    """
    log_messages, alignment_visuals, mappings = [], [], {}
    model = next(iter(structure))
    for chain in model:
        residues = polymer_residues(chain)
        chain_seq = chain_sequence(residues)
        if not chain_seq or set(chain_seq) == {"X"}:
            log_messages.append(f"Chain {chain.id}: no protein residues found, skipping.")
            continue

        best = None
        for target in target_sequences:
            aligned_t, aligned_c, identity = align_target_to_chain(str(target.seq), chain_seq)
            matches = sum(1 for a, b in zip(aligned_t, aligned_c) if a == b and a != '-')
            if best is None or matches > best[0]:
                best = (matches, identity, target, aligned_t, aligned_c)
        if best is None:
            log_messages.append(f"Chain {chain.id}: no target sequences provided.")
            continue

        matches, identity, target, aligned_t, aligned_c = best
        mapping = get_mapping_from_alignment(aligned_t, aligned_c)
        if not mapping:
            log_messages.append(f"Chain {chain.id}: no residues aligned to {target.id}, skipping.")
            continue
        mappings[chain.id] = mapping
        alignment_visuals.append((chain.id, target.id,
                                  f"Target ({target.id}): {aligned_t}\nChain  ({chain.id})       : {aligned_c}"))
        message = (f"Chain {chain.id}: best match is {target.id} with {matches} identical residues "
                   f"({identity:.1%} identity over aligned columns).")
        if identity < 0.5:
            message += " ⚠️ Low identity: check that this is the right target."
        log_messages.append(message)
    return mappings, log_messages, alignment_visuals


def apply_renumbering(structure: Structure, mappings: dict):
    """Return a renumbered copy; every model uses the mapping computed on the first model."""
    renumbered = copy.deepcopy(structure)
    for model in renumbered:
        for chain in model:
            if chain.id not in mappings:
                continue
            residues = polymer_residues(chain)
            mapping = mappings[chain.id]
            if mapping and max(mapping) >= len(residues):
                raise ValueError(f"Model {model.id}, chain {chain.id}: residue count differs from the first model.")
            new_ids = plan_residue_numbers(len(residues), mapping)
            # Two passes so no intermediate ID collides with a residue not yet renumbered.
            for k, res in enumerate(residues):
                res.id = (res.id[0], -100000 - k, " ")
            for res, (number, icode) in zip(residues, new_ids):
                res.id = (res.id[0], number, icode)
    return renumbered


def extract_header_records(pdb_text):
    """Crystallographic records (CRYST1, SCALE, ORIGX, MTRIX) that PDBIO does not write."""
    return "".join(line + "\n" for line in pdb_text.splitlines() if line.startswith(HEADER_RECORDS))


def pdb_renumber_section(pdb_file, pdb_format):
    """UI and workflow for PDB Residue Renumbering based on a target sequence."""
    st.header("🔢 PDB Residue Renumbering")
    st.info(
        "Provide target sequences (FASTA). Each chain is matched to its best target and renumbered so "
        "residue numbers equal target positions (1-based). Insertions relative to the target get "
        "insertion codes (e.g. 52A); terminal overhangs continue the numbering outward. "
        "Ligands and waters keep their original numbers."
    )

    if not pdb_file:
        st.warning(f"Please upload a {pdb_format} file in the sidebar first.")
        return

    target_sequences = None
    target_fasta_text = st.text_area("Paste Target Sequences (FASTA format)")
    target_fasta_file = st.file_uploader("Or Upload Target FASTA File", type=["fasta", "fas", "fa", "txt"])
    if target_fasta_file:
        target_sequences, error = parse_sequences_from_file(target_fasta_file, "fasta")
        if error:
            st.error(error)
    elif target_fasta_text:
        target_sequences, error = parse_sequences_from_text(target_fasta_text)
        if error:
            st.error(error)
    if not target_sequences:
        return

    pdb_file.seek(0)
    raw_text = pdb_file.read().decode("utf-8", errors="replace")
    current_hash = hash((raw_text, tuple(str(s.seq) for s in target_sequences)))
    if st.session_state.get('pdb_renumber_hash') != current_hash:
        with st.spinner("Matching chains to target sequences..."):
            try:
                parser = MMCIFParser(QUIET=True) if pdb_format == "mmCIF" else PDBParser(QUIET=True)
                structure = parser.get_structure("renumber", io.StringIO(raw_text))
                mappings, logs, visuals = prepare_renumbering(structure, target_sequences)
                st.session_state.pdb_renumber_results = {
                    'structure': structure, 'mappings': mappings, 'logs': logs, 'visuals': visuals,
                }
                st.session_state.pdb_renumber_hash = current_hash
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.text(traceback.format_exc())
                return

    results = st.session_state.pdb_renumber_results
    st.subheader("Interactive Review & Confirmation")
    with st.expander("Review Sequence Alignments (PDB vs Target)", expanded=True):
        for chain_id, target_id, aln_text in results['visuals']:
            st.markdown(f"**Chain {chain_id}** matched to **{target_id}**")
            st.text(aln_text)
    with st.expander("Renumbering Logs", expanded=True):
        for msg in results['logs']:
            st.write(msg)

    if pdb_format == "mmCIF":
        st.caption("mmCIF output contains coordinates only; other mmCIF categories are not carried over.")
    st.warning("Please review the alignment above. If it looks correct, confirm the renumbering.")

    if st.button("Confirm Renumbering & Generate File"):
        try:
            renumbered = apply_renumbering(results['structure'], results['mappings'])
        except ValueError as e:
            st.error(f"Renumbering aborted: {e}")
            return
        writer = MMCIFIO() if pdb_format == "mmCIF" else PDBIO()
        writer.set_structure(renumbered)
        out_stream = io.StringIO()
        writer.save(out_stream)
        content = out_stream.getvalue()
        if pdb_format != "mmCIF":
            content = extract_header_records(raw_text) + content
        ext = ".cif" if pdb_format == "mmCIF" else ".pdb"
        st.success("Renumbering completed successfully!")
        st.download_button(
            label=f"📥 Download Renumbered {pdb_format}",
            data=content,
            file_name=f"{pdb_file.name.rsplit('.', 1)[0]}_renumbered{ext}",
            mime="chemical/x-mmcif" if pdb_format == "mmCIF" else "chemical/x-pdb",
        )
