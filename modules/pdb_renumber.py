import copy
import io
import string
import traceback

import streamlit as st
from Bio.PDB import PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Structure import Structure

from modules.pairwise import get_aligner
from modules.parsers import parse_sequences_from_file, parse_sequences_from_text, polymer_residues
from modules.seqtools import clean_sequence
from modules.structures import parse_structure
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


SKIP = "(keep original numbering)"


def score_targets(structure, target_sequences):
    """Align every protein chain of the first model to every target; candidates sorted best-first."""
    candidates = {}
    for chain in next(iter(structure)):
        chain_seq = chain_sequence(polymer_residues(chain))
        if not chain_seq or set(chain_seq) == {"X"}:
            continue
        rows = []
        for target in target_sequences:
            aligned_t, aligned_c, identity = align_target_to_chain(str(target.seq), chain_seq)
            matches = sum(1 for a, b in zip(aligned_t, aligned_c) if a == b and a != '-')
            rows.append({"target_id": target.id, "matches": matches, "identity": identity,
                         "aligned_t": aligned_t, "aligned_c": aligned_c})
        rows.sort(key=lambda r: (r["matches"], r["identity"]), reverse=True)
        candidates[chain.id] = rows
    return candidates


def build_mappings(candidates, overrides=None):
    """Mappings, log messages and alignment visuals from scored candidates and optional overrides."""
    overrides = overrides or {}
    mappings, logs, visuals = {}, [], []
    for chain_id, rows in candidates.items():
        if not rows:
            logs.append(f"Chain {chain_id}: no target sequences provided.")
            continue
        choice = overrides.get(chain_id)
        if choice == SKIP:
            logs.append(f"Chain {chain_id}: kept original numbering (excluded by user).")
            continue
        row = next((r for r in rows if r["target_id"] == choice), rows[0])
        mapping = get_mapping_from_alignment(row["aligned_t"], row["aligned_c"])
        if not mapping:
            logs.append(f"Chain {chain_id}: no residues aligned to {row['target_id']}, skipping.")
            continue
        mappings[chain_id] = mapping
        visuals.append((chain_id, row["target_id"],
                        f"Target ({row['target_id']}): {row['aligned_t']}\nChain  ({chain_id})       : {row['aligned_c']}"))
        message = (f"Chain {chain_id}: {'selected' if choice else 'best match is'} {row['target_id']} with "
                   f"{row['matches']} identical residues ({row['identity']:.1%} identity over aligned columns).")
        if row["identity"] < 0.5:
            message += " ⚠️ Low identity: check that this is the right target."
        logs.append(message)
    return mappings, logs, visuals


def prepare_renumbering(structure: Structure, target_sequences: list, overrides=None):
    """Backward-compatible: score all targets, then build mappings honouring overrides."""
    return build_mappings(score_targets(structure, target_sequences), overrides)


def apply_renumbering(structure: Structure, mappings: dict):
    """Return a renumbered copy; every model uses the mapping computed on the first model."""
    renumbered = copy.deepcopy(structure)
    for model_index, model in enumerate(renumbered):
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
            _move_clashing_nonpolymer(renumbered, chain, residues, record_notes=(model_index == 0))
    return renumbered


def _move_clashing_nonpolymer(structure, chain, polymer, record_notes):
    """Move waters/ligands whose (number, icode) equals a renumbered polymer residue's.

    Moved residues get numbers above the chain maximum (insertion code " "), keeping hetflag.
    """
    polymer_ids = {id(res) for res in polymer}
    taken = {res.id[1:] for res in polymer}
    clashing = [res for res in chain if id(res) not in polymer_ids and res.id[1:] in taken]
    if not clashing:
        return
    next_number = max(res.id[1] for res in chain) + 1
    moves = []
    for k, res in enumerate(clashing):
        moves.append((res, res.id[1], res.id[0], next_number + k))
        res.id = (res.id[0], -200000 - k, " ")
    for res, old_number, hetflag, new_number in moves:
        res.id = (hetflag, new_number, " ")
        if record_notes:
            structure.xtra.setdefault("renumber_notes", []).append(
                f"Chain {chain.id}: {res.resname.strip()} {old_number} renumbered to {new_number} "
                f"to avoid clashing with polymer residue {old_number}")


def extract_header_records(pdb_text):
    """Crystallographic records (CRYST1, SCALE, ORIGX, MTRIX) that PDBIO does not write."""
    return "".join(line + "\n" for line in pdb_text.splitlines() if line.startswith(HEADER_RECORDS))


def pdb_renumber_section(structures):
    """UI and workflow for PDB Residue Renumbering based on a target sequence."""
    st.header("🔢 PDB Residue Renumbering")
    st.info(
        "Provide target sequences (FASTA). Each chain is matched to its best target and renumbered so "
        "residue numbers equal target positions (1-based). Insertions relative to the target get "
        "insertion codes (e.g. 52A); terminal overhangs continue the numbering outward. "
        "Ligands and waters keep their original numbers."
    )
    if not structures:
        st.warning("Upload a PDB or mmCIF file in the sidebar first.")
        return

    name = st.selectbox("Structure to renumber", list(structures), key="renumber_structure")
    structure_file = structures[name]

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

    key = (structure_file.digest, tuple((s.id, str(s.seq)) for s in target_sequences))
    if st.session_state.get('pdb_renumber_hash') != key:
        with st.spinner("Matching chains to target sequences..."):
            try:
                st.session_state.pdb_renumber_results = {
                    "candidates": score_targets(parse_structure(structure_file), target_sequences)}
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.text(traceback.format_exc())
                return
            st.session_state.pdb_renumber_hash = key

    candidates = st.session_state.pdb_renumber_results["candidates"]
    st.subheader("Interactive Review & Confirmation")
    overrides = {}
    for chain_id, rows in candidates.items():
        labels = {r["target_id"]: f"{r['target_id']} — {r['identity']:.1%} identity, {r['matches']} identical"
                  for r in rows}
        options = [r["target_id"] for r in rows] + [SKIP]
        overrides[chain_id] = st.selectbox(f"Chain {chain_id} target", options,
                                           format_func=lambda v, labels=labels: labels.get(v, v),
                                           key=f"renumber_target_{chain_id}")
    mappings, logs, visuals = build_mappings(candidates, overrides)
    results = {"mappings": mappings}

    with st.expander("Review Sequence Alignments (PDB vs Target)", expanded=True):
        for chain_id, target_id, aln_text in visuals:
            st.markdown(f"**Chain {chain_id}** matched to **{target_id}**")
            st.text(aln_text)
    with st.expander("Renumbering Logs", expanded=True):
        for msg in logs:
            st.write(msg)

    is_cif = structure_file.fmt == "mmCIF"
    if is_cif:
        st.caption("mmCIF output contains coordinates only; other mmCIF categories are not carried over.")
    st.warning("Please review the alignment above. If it looks correct, confirm the renumbering.")

    if st.button("Confirm Renumbering & Generate File"):
        try:
            renumbered = apply_renumbering(parse_structure(structure_file), results['mappings'])
        except ValueError as e:
            st.error(f"Renumbering aborted: {e}")
            return
        for note in renumbered.xtra.get("renumber_notes", []):
            st.info(note)
        writer = MMCIFIO() if is_cif else PDBIO()
        writer.set_structure(renumbered)
        out_stream = io.StringIO()
        writer.save(out_stream)
        content = out_stream.getvalue()
        if not is_cif:
            content = extract_header_records(structure_file.text) + content
        ext = ".cif" if is_cif else ".pdb"
        st.success("Renumbering completed successfully!")
        st.download_button(
            label=f"📥 Download Renumbered {structure_file.fmt}",
            data=content,
            file_name=f"{structure_file.stem}_renumbered{ext}",
            mime="chemical/x-mmcif" if is_cif else "chemical/x-pdb", on_click="ignore")
