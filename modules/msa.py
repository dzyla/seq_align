import streamlit as st
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pyfamsa import Aligner as PyFAMSAAligner, Sequence as PyFAMSASequence
from io import StringIO
from collections import Counter
import pandas as pd
from typing import List, Dict, Tuple
from modules.utils import have_params_changed, stale_notice
from modules.viz import msa_to_image, plot_msa_image
from modules.seqtools import clean_sequence, fingerprint
from modules.phylogeny import TREE_METHODS, TreeError, build_tree_from_alignment, phylogenetic_tree_section

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

    current_params = {"fingerprint": fingerprint(sequences), "seq_type": seq_type}
    params_changed = have_params_changed(current_params, "last_msa_params")

    if st.button("Run MSA"):
        with st.spinner("Performing MSA - this may take a moment for large datasets..."):
            try:
                st.session_state.msa_aligned = cached_run_msa(
                    tuple((s.id, str(s.seq)) for s in sequences), seq_type)
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

    try:
        mutations = cached_mutations(tuple(aligned), ref_seq_id)
    except MSAError as e:
        st.error(str(e))
        mutations = {}

    tab_titles = ["Alignment & Heatmap", "Point Mutations", "Phylogenetic Tree"]
    if calculate_representative:
        tab_titles.insert(2, "Representative Sequence")
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader("📄 MSA Result")
        st.write(f"Total sequences: {len(aligned)}. MSA length: {len(aligned[0][1])}")
        msa_image, msa_letters = msa_to_image(serialize_alignment(aligned, "fasta"), "fasta")
        plot_msa_image(msa_image, msa_letters, plot_method)
        try:
            st.download_button(label="📥 Download MSA", data=serialize_alignment(aligned, msa_output_format),
                               file_name=f"msa_alignment.{msa_output_format}", mime="text/plain")
        except MSAError as e:
            st.error(str(e))

    with tabs[1]:
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
                                   file_name="msa_mutations.csv", mime="text/csv")
        else:
            st.write("No point mutations detected relative to the reference sequence.")

    if calculate_representative:
        with tabs[2]:
            st.subheader("🧬 Representative Sequence Analysis")
            data = st.session_state.consensus_data
            if data is None or data['threshold'] != consensus_threshold:
                alignment = aligned_to_msa(aligned)
                consensus_record, closest_record, min_diff, closest_seq_id = calculate_representative_sequence(
                    alignment, threshold=consensus_threshold, seq_type=seq_type)
                data = st.session_state.consensus_data = {
                    'threshold': consensus_threshold, 'consensus_record': consensus_record,
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
                file_name="consensus_representative.fasta", mime="text/plain")

    with tabs[-1]:
        st.subheader("🌳 Phylogenetic Tree from MSA")
        c_m, c_c = st.columns(2)
        tree_method = c_m.selectbox("Tree method", list(TREE_METHODS), key="msa_tree_method")
        correction = c_c.checkbox("Correct for multiple substitutions", value=True, key="msa_tree_corr",
                                  help="Jukes–Cantor (DNA) or Kimura (protein) distance correction.")
        if st.button("Calculate Phylogenetic Tree from MSA"):
            with st.spinner("Building tree from MSA..."):
                try:
                    st.session_state.msa_tree = build_tree_from_alignment(
                        aligned_to_msa(aligned), seq_type, TREE_METHODS[tree_method], correction)
                except TreeError as e:
                    st.error(str(e))
        if st.session_state.get('msa_tree'):
            phylogenetic_tree_section(st.session_state.msa_tree)


class MSAError(RuntimeError):
    """Raised when the multiple sequence alignment cannot be computed or written."""


MSA_OUTPUT_FORMATS = ("fasta", "clustal", "phylip-relaxed", "stockholm")


def run_msa(pairs, seq_type):
    """Align (id, sequence) pairs with FAMSA and return [(id, aligned_sequence)]."""
    cleaned = [(seq_id, clean_sequence(seq, seq_type).replace("*", "")) for seq_id, seq in pairs]
    empty = [seq_id for seq_id, seq in cleaned if not seq]
    if empty:
        raise MSAError(f"Sequences empty after cleaning: {', '.join(empty)}")
    if len(cleaned) < 2:
        raise MSAError("At least two sequences are required for an MSA.")
    famsa_input = [PyFAMSASequence(seq_id.encode(), seq.encode()) for seq_id, seq in cleaned]
    try:
        msa = PyFAMSAAligner(guide_tree="upgma").align(famsa_input)
    except Exception as e:
        raise MSAError(f"FAMSA failed: {e}") from e
    return [(s.id.decode(), s.sequence.decode()) for s in msa]


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
def cached_run_msa(pairs, seq_type):
    """FAMSA alignment memoized on the exact (id, sequence) tuples and sequence type."""
    return run_msa(list(pairs), seq_type)


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
