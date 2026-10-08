import os
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
import streamlit as st
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import json
import sys
from io import StringIO
from collections import Counter
import pandas as pd
from typing import List, Dict, Tuple
from modules.utils import have_params_changed, stale_notice
from modules.viz import alignment_to_codes, plot_msa
from modules.seqtools import clean_sequence, fingerprint
from modules.conservation import (
    CONSERVATION_COLORS, column_conservation, conservation_grade, identity_matrix_frame, logo_figure,
    logo_matrix, reference_conservation, write_bfactors,
)
from modules.structure_view import (
    Highlight, ResidueRef, available_structures, build_view_state, cached_residue_table,
    map_sequence_to_structure, rank_chains, render_molstar, structure_highlight_panel,
)
from modules.variants import variant_matrix, variants_long
from modules.phylogeny import (
    TREE_METHODS, TreeError, bootstrap_tree, build_tree_from_alignment, phylogenetic_tree_section)

def msa_section(sequences, seq_type):
    """Multiple Sequence Alignment workflow. The alignment is computed once per dataset."""
    st.header("📏 Multiple Sequence Alignment (MSA)")
    st.info("Multiple Sequence Alignment aligns three or more sequences together, "
            "identifying conserved regions and structural similarities.")
    if len(sequences) < 2:
        st.warning("Please upload at least two sequences for MSA.")
        return

    ref_seq_id = st.selectbox("Select Reference Sequence for Mutation Reporting", [s.id for s in sequences],
                              key="ref_seq_select",
                              help="Mutations in other sequences will be reported relative to this sequence")
    msa_output_format = st.selectbox("Select MSA Output Format", MSA_OUTPUT_FORMATS, key="msa_output_format",
                                     help="Format of the downloaded alignment. Changing it does not rerun the MSA.")
    plot_method = st.selectbox("Select Plotting Method", ("Plotly (Interactive)", "Matplotlib (Static)"),
                               help="Plotly provides interactive visualization. Matplotlib creates static images.")
    calculate_representative = st.checkbox(
        "Calculate Most Representative Sequence",
        help="Find the consensus sequence and sequence that best represents the entire alignment")
    consensus_threshold = 0.7
    if calculate_representative:
        consensus_threshold = st.slider(
            "Consensus Threshold", 0.5, 1.0, 0.7, 0.05,
            help="Minimum fraction of sequences that must share a residue for it to enter the consensus")

    method_options = [m for m in MSA_METHODS if m == "FAMSA" or mafft_available()]
    msa_method = st.selectbox("Aligner", method_options, key="msa_method",
                              index=method_options.index("MAFFT") if seq_type == "DNA" and "MAFFT" in method_options else 0,
                              help="FAMSA is fast for large protein sets. MAFFT (--auto) is recommended for DNA."
                                   + ("" if mafft_available() else " MAFFT is not installed on this server."))

    current_params = {"fingerprint": fingerprint(sequences), "seq_type": seq_type, "method": msa_method}
    params_changed = have_params_changed(current_params, "last_msa_params")

    if st.button("Run MSA"):
        with st.spinner("Performing MSA - this may take a moment for large datasets..."):
            try:
                st.session_state.msa_aligned = cached_run_msa(
                    tuple((s.id, str(s.seq)) for s in sequences), seq_type, msa_method)
            except MSAError as e:
                st.session_state.msa_aligned = None
                st.error(str(e))
                return
        st.session_state.last_msa_params = current_params
        st.session_state.consensus_data = None
        st.session_state.msa_tree = None
        params_changed = False

    aligned = st.session_state.get("msa_aligned")
    if not aligned:
        return
    if params_changed:
        stale_notice()
        return

    with st.expander("✂️ Alignment tools (trim columns / remove redundant sequences)"):
        max_gap = st.slider("Drop columns with gap fraction above", 0.0, 1.0, 1.0, 0.05, key="msa_trim",
                            help="1.0 keeps all columns. Trimming affects the heatmap, downloads and trees only; "
                                 "mutation, variant and conservation positions always use the full alignment.")
        max_id = st.slider("Drop sequences with identity at or above", 0.5, 1.0, 1.0, 0.01, key="msa_redundancy",
                           help="1.0 keeps all sequences. The reference sequence is always kept.")
    rows, view, removed = alignment_views(aligned, ref_seq_id, max_gap, max_id)
    if removed:
        st.caption(f"{len(removed)} redundant sequence(s) hidden: "
                   + ", ".join(f"{r} (≈ {k})" for r, k in list(removed.items())[:20]))

    view_empty = not view or len(view[0][1]) == 0
    if view_empty:
        st.warning("All alignment columns were removed by trimming; raise the gap threshold.")

    try:
        mutations = cached_mutations(tuple(rows), ref_seq_id)
    except MSAError as e:
        st.error(str(e))
        mutations = {}

    tab_titles = ["Alignment & Heatmap", "Point Mutations", "Variant Table", "Conservation", "Phylogenetic Tree"]
    if calculate_representative:
        tab_titles.insert(tab_titles.index("Variant Table") + 1, "Representative Sequence")
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader("📄 MSA Result")
        if not view_empty:
            st.write(f"Total sequences: {len(view)}. MSA length: {len(view[0][1])}")
            plot_msa(alignment_to_codes([s for _, s in view]), [i for i, _ in view], plot_method, seq_type)
            try:
                st.download_button(label="📥 Download MSA", data=serialize_alignment(view, msa_output_format),
                                   file_name=f"msa_alignment.{msa_output_format}", mime="text/plain",
                                   on_click="ignore")
            except MSAError as e:
                st.error(str(e))
            st.download_button("📥 Percent identity matrix (CSV)", identity_matrix_frame(view).to_csv(),
                               "percent_identity.csv", "text/csv", on_click="ignore")

    with tabs[tab_titles.index("Point Mutations")]:
        st.subheader("🔍 Point Mutations Relative to Reference")
        if any(mutations.values()):
            seq_ids = sorted(mutations)
            select_all = st.checkbox("Select all sequences", value=True, key="select_all_mutations")
            selected_seq_ids = seq_ids if select_all else st.multiselect(
                "Select sequences to view mutations for:", seq_ids,
                default=seq_ids[:min(5, len(seq_ids))], key="mutation_sequence_selector")
            for seq_id in selected_seq_ids:
                mut_list = mutations.get(seq_id, [])
                if mut_list:
                    with st.expander(f"Mutations for {seq_id} ({len(mut_list)} mutations)"):
                        st.dataframe(pd.DataFrame(mut_list, columns=["Position", "Reference", "Variant"]))
                        st.write(f"**Mutations:** {', '.join(f'{r}{p}{v}' for p, r, v in mut_list)}")
            csv_mutations = convert_msa_mutations_to_csv(mutations)
            if csv_mutations:
                st.download_button(label="📥 Download All Mutations (CSV)", data=csv_mutations,
                                   file_name="msa_mutations.csv", mime="text/csv", on_click="ignore")
            if seq_type == "Protein":
                mutated = [s for s in seq_ids if mutations.get(s)]
                if mutated:
                    show_id = st.selectbox("Sequence to show on structure", mutated, key="msa_3d_seq")
                    ref_seq = clean_sequence(next(s.seq for s in sequences if s.id == ref_seq_id), "Protein").replace("*", "")
                    structure_highlight_panel(
                        ref_seq, {p: ("#d62728", f"{r}{p}{v}") for p, r, v in mutations[show_id]}, key="msa_3d")
        else:
            st.write("No point mutations detected relative to the reference sequence.")

    with tabs[tab_titles.index("Variant Table")]:
        st.subheader("🧾 Variant Table (substitutions and indels)")
        matrix = variant_matrix(rows, ref_seq_id)
        if matrix.empty or matrix.shape[1] == 0:
            st.write("No variants relative to the reference.")
        else:
            st.caption(f"Rows: sequences. Columns: {ref_seq_id} positions. Indels use HGVS-like notation "
                       "(e.g. V3_L4insW, A5_A6del).")
            st.dataframe(matrix, width="stretch")
            st.download_button("📥 Variant matrix (CSV)", matrix.to_csv(), "variant_matrix.csv", "text/csv",
                               on_click="ignore")
            st.download_button("📥 Variants, one per row (CSV)", variants_long(rows, ref_seq_id).to_csv(index=False),
                               "variants_long.csv", "text/csv", on_click="ignore")

    if calculate_representative:
        with tabs[tab_titles.index("Representative Sequence")]:
            st.subheader("🧬 Representative Sequence Analysis")
            data = st.session_state.consensus_data
            consensus_key = (consensus_threshold, max_gap, max_id, ref_seq_id)
            if data is None or data.get('key') != consensus_key:
                alignment = aligned_to_msa(rows)
                consensus_record, closest_record, min_diff, closest_seq_id = calculate_representative_sequence(
                    alignment, threshold=consensus_threshold, seq_type=seq_type)
                data = st.session_state.consensus_data = {
                    'key': consensus_key, 'threshold': consensus_threshold, 'consensus_record': consensus_record,
                    'closest_record': closest_record, 'min_differences': min_diff,
                    'closest_seq_id': closest_seq_id, 'seq_count': len(alignment),
                }
            st.write(f"**Consensus threshold:** {data['threshold']}")
            st.write(f"**Number of sequences:** {data['seq_count']}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Consensus Sequence")
                st.code(f">{data['consensus_record'].id}\n{data['consensus_record'].seq}")
            with col2:
                st.markdown("#### Most Representative Sequence")
                st.write(f"**Sequence ID:** `{data['closest_seq_id']}`")
                st.write(f"**Differences from consensus:** {data['min_differences']}")
                st.code(f">{data['closest_record'].id}\n{data['closest_record'].seq}")
            st.download_button(
                label="📥 Download Consensus/Representative (FASTA)",
                data=(f">{data['consensus_record'].id}\n{data['consensus_record'].seq}\n"
                      f">{data['closest_record'].id}\n{data['closest_record'].seq}\n"),
                file_name="consensus_representative.fasta", mime="text/plain", on_click="ignore")

    with tabs[tab_titles.index("Conservation")]:
        st.subheader("🎯 Conservation")
        ref_scores = reference_conservation(rows, ref_seq_id, seq_type)
        st.caption(f"1 − normalized Shannon entropy per column (gaps excluded; columns with > 50% gaps "
                   f"are not scored). Positions are in {ref_seq_id} coordinates.")
        st.line_chart(pd.DataFrame({"Conservation": pd.Series(ref_scores)}), x_label="Reference position",
                      y_label="Conservation")
        if not view_empty:
            st.caption("Information content in bits against a uniform background, "
                       "scaled by each column's non-gap fraction.")
            length = len(view[0][1])
            if length < 2:
                st.caption("Logo needs at least 2 columns.")
            else:
                start, end = st.slider("Logo window (alignment columns)", 1, length, (1, min(length, 60)),
                                       key="logo_window")
                if end - start + 1 > 120:
                    st.warning("Showing the first 120 columns of the window.")
                    end = start + 119
                fig = logo_figure(logo_matrix([s for _, s in view], seq_type, start, end), seq_type)
                st.pyplot(fig)
                plt.close(fig)
        if seq_type == "Protein":
            ref_seq = clean_sequence(next(s.seq for s in sequences if s.id == ref_seq_id), "Protein").replace("*", "")
            with st.expander("🧊 Conservation on structure (Mol*) and B-factor export"):
                structures = available_structures()
                if not structures:
                    st.caption("Use 'Show on structure' in the Point Mutations tab to add a structure, "
                               "or load one via Fetch by ID.")
                else:
                    name = st.selectbox("Structure", list(structures), key="cons_structure")
                    sf = structures[name]
                    table = cached_residue_table(sf.data, sf.fmt, sf.name)
                    if not table:
                        st.warning("No protein chains in this structure.")
                    else:
                        ranked = rank_chains(ref_seq, table)
                        chain_id = st.selectbox("Chain", [c for c, _ in ranked], key="cons_chain")
                        pos_map, identity = map_sequence_to_structure(ref_seq, table[chain_id])
                        highlights, values = [], {}
                        for pos, score in ref_scores.items():
                            res = pos_map.get(pos)
                            if res:
                                grade = conservation_grade(score)
                                highlights.append(Highlight(ResidueRef(res.chain, res.number, res.icode),
                                                            CONSERVATION_COLORS[grade - 1]))
                                values[(res.chain, res.number, res.icode)] = round(score, 3)
                        st.caption(f"{len(highlights)} residues colored ({identity:.1%} identity to chain {chain_id}); "
                                   "teal = variable, maroon = conserved.")
                        render_molstar(sf, build_view_state(sf, list(table), color_mode="uniform",
                                                            highlights=highlights, show_sticks=False))
                        ext = "cif" if sf.fmt == "mmCIF" else "pdb"
                        st.download_button("📥 Download structure with conservation as B-factors",
                                           data=write_bfactors(sf, values), file_name=f"{sf.stem}_conservation.{ext}",
                                           mime="text/plain", on_click="ignore",
                                           help="Open in ChimeraX and use: color bfactor palette ^bluered. "
                                                "Unscored residues (gap-rich columns) and other chains are written as 0.0.")

    with tabs[tab_titles.index("Phylogenetic Tree")]:
        st.subheader("🌳 Phylogenetic Tree from MSA")
        c_m, c_c = st.columns(2)
        tree_method = c_m.selectbox("Tree method", list(TREE_METHODS), key="msa_tree_method")
        correction = c_c.checkbox("Correct for multiple substitutions", value=True, key="msa_tree_corr",
                                  help="Jukes–Cantor (DNA) or Kimura (protein) distance correction.")
        replicates = st.number_input("Bootstrap replicates", 0, 1000, 0, 50, key="msa_bootstrap",
                                     help="0 = no bootstrap. 100 is a common choice.")
        tree_params = {"method": tree_method, "correction": correction, "replicates": replicates,
                       "max_gap": max_gap, "max_id": max_id, "ref_seq_id": ref_seq_id}
        if st.button("Calculate Phylogenetic Tree from MSA", disabled=view_empty):
            with st.spinner("Building tree from MSA..."):
                try:
                    if replicates > 0:
                        tree = bootstrap_tree(view, seq_type, TREE_METHODS[tree_method], correction, replicates)
                    else:
                        tree = build_tree_from_alignment(
                            aligned_to_msa(view), seq_type, TREE_METHODS[tree_method], correction)
                    st.session_state.msa_tree = {"params": tree_params, "tree": tree}
                except TreeError as e:
                    st.error(str(e))
        stored = st.session_state.get('msa_tree')
        if stored:
            if stored["params"] != tree_params:
                stale_notice()
            else:
                phylogenetic_tree_section(stored["tree"], key_prefix="msa_tree")


class MSAError(RuntimeError):
    """Raised when the multiple sequence alignment cannot be computed or written."""


MSA_OUTPUT_FORMATS = ("fasta", "clustal", "phylip-relaxed", "stockholm")


MSA_METHODS = ("FAMSA", "MAFFT")


def mafft_available():
    return shutil.which("mafft") is not None


def run_mafft(pairs, seq_type):
    """Align with MAFFT --auto; input order, IDs and residue case are preserved."""
    if not mafft_available():
        raise MSAError("MAFFT is not installed on this server; use FAMSA.")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "input.fasta")
        with open(path, "w") as handle:
            handle.write("".join(f">{seq_id}\n{seq}\n" for seq_id, seq in pairs))
        try:
            proc = subprocess.run(
                ["mafft", "--auto", "--quiet", "--preservecase", "--inputorder", "--anysymbol", path],
                capture_output=True, text=True, timeout=600, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise MSAError(f"MAFFT failed: {e}") from e
    aligned = [(r.id, str(r.seq)) for r in SeqIO.parse(StringIO(proc.stdout), "fasta")]
    if [i for i, _ in aligned] != [i for i, _ in pairs]:
        raise MSAError("MAFFT output did not match the input sequences.")
    return aligned


def run_msa(pairs, seq_type, method="FAMSA"):
    """Align (id, sequence) pairs with FAMSA (default) or MAFFT and return [(id, aligned_sequence)]."""
    cleaned = [(seq_id, clean_sequence(seq, seq_type).replace("*", "")) for seq_id, seq in pairs]
    empty = [seq_id for seq_id, seq in cleaned if not seq]
    if empty:
        raise MSAError(f"Sequences empty after cleaning: {', '.join(empty)}")
    if len(cleaned) < 2:
        raise MSAError("At least two sequences are required for an MSA.")
    short = [seq_id for seq_id, seq in cleaned if len(seq) < 2]
    if short:  # pyFAMSA crashes the whole process on single-residue input
        raise MSAError(f"Sequences too short to align (need at least 2 residues): {', '.join(short)}")
    if method == "MAFFT":
        return run_mafft(cleaned, seq_type)
    return run_famsa(cleaned)


FAMSA_TIMEOUT = 600
FAMSA_CRASH_MESSAGE = ("FAMSA crashed on this input (this happens when a sequence shares no residues with "
                       "the others). Try MAFFT or check the input.")


FAMSA_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "famsa_worker.py")


def run_famsa(pairs):
    """Run pyFAMSA in a separate Python process: it can abort the interpreter, which try/except cannot catch.

    The worker is started by file path (not multiprocessing), so the Streamlit page script, which runs as an
    unguarded __main__, is never re-executed in the child.
    """
    request = json.dumps({"pairs": [list(p) for p in pairs]})
    try:
        done = subprocess.run([sys.executable, FAMSA_WORKER], input=request, capture_output=True,
                              text=True, timeout=FAMSA_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise MSAError(f"FAMSA did not finish within {FAMSA_TIMEOUT} s. Try MAFFT or fewer sequences.") from e
    if done.returncode < 0 or done.returncode in (134, 139):  # killed by a signal: native crash
        raise MSAError(FAMSA_CRASH_MESSAGE)
    if done.returncode != 0:
        detail = (done.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise MSAError(f"FAMSA failed: {detail}")
    return [tuple(p) for p in json.loads(done.stdout)]


def trim_columns(aligned_pairs, max_gap_fraction):
    """Drop columns whose gap fraction exceeds max_gap_fraction; returns (pairs, kept column indices)."""
    seqs = [s for _, s in aligned_pairs]
    n = len(seqs)
    kept = [c for c in range(len(seqs[0])) if sum(s[c] in "-." for s in seqs) / n <= max_gap_fraction]
    return [(i, "".join(s[c] for c in kept)) for i, s in aligned_pairs], kept


def remove_redundant(aligned_pairs, identity_threshold, keep_ids=()):
    """Greedy filter: keep `keep_ids` first, then longest sequences; drop any sequence whose identity
    to an already kept sequence is >= threshold. Returns (kept pairs, {removed_id: representative_id})."""
    from modules.clustering import msa_identity_matrix  # local: clustering imports msa

    ids = [i for i, _ in aligned_pairs]
    identity = msa_identity_matrix([s for _, s in aligned_pairs])
    ungapped = {i: len(s.replace("-", "")) for i, s in aligned_pairs}
    order = [i for i in keep_ids if i in ids] + sorted((i for i in ids if i not in keep_ids),
                                                       key=lambda i: -ungapped[i])
    index = {i: k for k, i in enumerate(ids)}
    kept, removed = [], {}
    for seq_id in order:
        rep = next((k for k in kept if identity[index[seq_id], index[k]] >= identity_threshold), None)
        if rep is None:
            kept.append(seq_id)
        else:
            removed[seq_id] = rep
    pairs_by_id = dict(aligned_pairs)
    return [(k, pairs_by_id[k]) for k in kept], removed


def alignment_views(aligned_pairs, reference_id, max_gap_fraction=1.0, max_identity=1.0):
    """(rows for coordinate-bearing analyses, trimmed view for display/export, removed map)."""
    rows, removed = aligned_pairs, {}
    if max_identity < 1.0:
        rows, removed = remove_redundant(aligned_pairs, max_identity, keep_ids=(reference_id,))
    view = trim_columns(rows, max_gap_fraction)[0] if max_gap_fraction < 1.0 else rows
    return rows, view, removed


def serialize_alignment(aligned_pairs, output_format):
    """Write aligned (id, sequence) pairs in an alignment format."""
    records = [SeqRecord(Seq(seq), id=seq_id, description="") for seq_id, seq in aligned_pairs]
    out = StringIO()
    try:
        SeqIO.write(records, out, output_format)
    except ValueError as e:
        raise MSAError(f"Could not write the alignment as {output_format}: {e}") from e
    return out.getvalue()


def mutations_from_alignment(aligned_pairs, reference_id):
    """Substitutions of every sequence relative to the reference, in reference coordinates."""
    aligned = dict(aligned_pairs)
    if reference_id not in aligned:
        raise MSAError(f"Reference sequence '{reference_id}' not found in the alignment.")
    ref = aligned[reference_id].upper()
    mutations = {}
    for seq_id, seq in aligned_pairs:
        if seq_id == reference_id:
            continue
        seq = seq.upper()
        seq_mutations = []
        ref_position = 0
        for ref_base, seq_base in zip(ref, seq):
            if ref_base != '-':
                ref_position += 1
            if ref_base != seq_base and ref_base != '-' and seq_base != '-':
                seq_mutations.append((ref_position, ref_base, seq_base))
        mutations[seq_id] = seq_mutations
    return mutations


def perform_msa(sequences, reference_id, seq_type, output_format):
    """
    Align SeqRecords with FAMSA and report mutations relative to the reference.

    Returns (msa_text, mutations). Raises MSAError on failure.
    """
    aligned = run_msa([(s.id, str(s.seq)) for s in sequences], seq_type)
    return serialize_alignment(aligned, output_format), mutations_from_alignment(aligned, reference_id)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_run_msa(pairs, seq_type, method="FAMSA"):
    """Alignment memoized on the exact (id, sequence) tuples, sequence type and method."""
    return run_msa(list(pairs), seq_type, method)


@st.cache_data(max_entries=16, show_spinner=False)
def cached_mutations(aligned, reference_id):
    return mutations_from_alignment(list(aligned), reference_id)


def aligned_to_msa(aligned_pairs):
    """Biopython MultipleSeqAlignment from (id, aligned_sequence) pairs."""
    return MultipleSeqAlignment([SeqRecord(Seq(seq), id=seq_id, description="") for seq_id, seq in aligned_pairs])


def calculate_representative_sequence(alignment, threshold=0.7, seq_type="Protein"):
    """
    Consensus sequence and the alignment member closest to it.

    Columns whose most frequent symbol is a gap are left out of the exported consensus.
    Columns below `threshold` get the ambiguity code (N for DNA, X for protein) and are not
    counted as differences when choosing the most representative sequence.

    Returns:
        tuple: (consensus_record, closest_record, min_differences, closest_seq_id)
    """
    ambiguous = "N" if seq_type == "DNA" else "X"
    seqs = [str(record.seq).upper() for record in alignment]
    n_seqs = len(seqs)

    consensus_aligned = []
    for i in range(alignment.get_alignment_length()):
        char, count = Counter(s[i] for s in seqs).most_common(1)[0]
        consensus_aligned.append(char if count / n_seqs >= threshold else ambiguous)
    consensus_aligned = "".join(consensus_aligned)

    def count_differences(seq):
        return sum(1 for c, s in zip(consensus_aligned, seq) if c != ambiguous and c != s)

    closest_index = min(range(n_seqs), key=lambda k: count_differences(seqs[k]))
    closest_record_src = alignment[closest_index]
    min_differences = count_differences(seqs[closest_index])

    consensus_record = SeqRecord(Seq(consensus_aligned.replace("-", "")), id="Consensus",
                                 description="Consensus sequence")
    closest_record = SeqRecord(Seq(str(closest_record_src.seq).replace("-", "")),
                               id=closest_record_src.id, description="Most representative sequence")
    return consensus_record, closest_record, min_differences, closest_record_src.id

def convert_msa_mutations_to_csv(mutations: Dict[str, List[Tuple[int, str, str]]]) -> str:
    """
    Converts a dictionary of MSA mutations to a CSV formatted string.
    """
    if not mutations:
        return ""

    all_mutations = []
    for seq_id, mut_list in mutations.items():
        for pos, ref, var in mut_list:
            all_mutations.append([seq_id, pos, ref, var])

    if not all_mutations:
        return ""

    df = pd.DataFrame(all_mutations, columns=["Sequence_ID", "Position", "Reference", "Variant"])
    output = StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()
