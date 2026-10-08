import streamlit as st
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pyfamsa import Aligner as PyFAMSAAligner, Sequence as PyFAMSASequence
from io import StringIO
from collections import Counter
import pandas as pd
from typing import List, Dict, Tuple
from modules.utils import have_params_changed
from modules.viz import msa_to_image, plot_msa_image
from modules.seqtools import clean_sequence
from modules.phylogeny import TREE_METHODS, TreeError, build_tree_from_alignment, phylogenetic_tree_section

def msa_section(sequences, seq_type):
    """
    Handles the Multiple Sequence Alignment (MSA) workflow including UI and computation.

    Parameters:
        sequences: List of sequence records to align
        seq_type (str): Type of sequences ('DNA' or 'Protein')
    """
    st.header("📏 Multiple Sequence Alignment (MSA)")
    st.info(
        "Multiple Sequence Alignment aligns three or more sequences together, "
        "identifying conserved regions and structural similarities."
    )

    if len(sequences) < 2:
        st.warning("Please upload at least two sequences for MSA.")
        return

    ref_seq_id = st.selectbox(
        "Select Reference Sequence for Mutation Reporting",
        [seq.id for seq in sequences],
        key="ref_seq_select",
        help="Mutations in other sequences will be reported relative to this sequence"
    )

    msa_output_format = st.selectbox(
        "Select MSA Output Format",
        MSA_OUTPUT_FORMATS,
        help="Choose the format for saving the alignment results"
    )

    plot_method = st.selectbox(
        "Select Plotting Method",
        ("Plotly (Interactive)", "Matplotlib (Static)"),
        help="Plotly provides interactive visualization. Matplotlib creates static images."
    )

    # Add option to calculate representative sequence
    calculate_representative = st.checkbox(
        "Calculate Most Representative Sequence",
        help="Find the consensus sequence and sequence that best represents the entire alignment"
    )

    if calculate_representative:
        consensus_threshold = st.slider(
            "Consensus Threshold",
            0.5, 1.0, 0.7, 0.05,
            help="Minimum fraction of sequences that must have the same residue at a position to be included in consensus"
        )

    # Create a unique parameter set to track MSA settings
    current_msa_params = {
        'sequences_ids': tuple(seq.id for seq in sequences),
        'ref_seq_id': ref_seq_id,
        'seq_type': seq_type,
        'msa_output_format': msa_output_format,
        'calculate_representative': calculate_representative
    }

    if calculate_representative:
        current_msa_params['consensus_threshold'] = consensus_threshold

    # Check if MSA parameters have changed
    msa_params_changed = have_params_changed(current_msa_params, 'last_msa_params')

    run_msa = st.button("Run MSA")

    # Use session state to store and retrieve MSA results
    if run_msa or (st.session_state.msa_result is not None and not msa_params_changed):
        # Only recalculate if parameters changed or explicitly requested
        if run_msa or msa_params_changed:
            with st.spinner("Performing MSA - this may take a moment for large datasets..."):
                try:
                    msa_result, mutations = perform_msa(sequences, ref_seq_id, seq_type, msa_output_format)
                except MSAError as e:
                    st.error(str(e))
                    msa_result, mutations = "", {}
                st.session_state.msa_result = msa_result
                st.session_state.mutations = mutations
                st.session_state.last_msa_params = current_msa_params

                # Reset consensus data when running new MSA
                st.session_state.consensus_data = None

                # Generate image data for plotting
                if msa_result:
                    try:
                        msa_image, msa_letters = msa_to_image(msa_result, msa_output_format)
                        st.session_state.msa_image = msa_image
                        st.session_state.msa_letters = msa_letters
                    except Exception as e:
                        st.error(f"Failed to generate MSA heatmap: {e}")

        if st.session_state.msa_result:
            tab_titles = ["Alignment & Heatmap", "Point Mutations", "Phylogenetic Tree"]
            if calculate_representative:
                tab_titles.insert(2, "Representative Sequence")

            tabs = st.tabs(tab_titles)

            with tabs[0]:
                st.subheader("📄 MSA Result")
                try:
                    _aln = AlignIO.read(StringIO(st.session_state.msa_result), msa_output_format)
                    st.write(f"Total sequences: {len(_aln)}. MSA length: {_aln.get_alignment_length()}")
                except ValueError:
                    pass

                if st.session_state.msa_image is not None and st.session_state.msa_letters is not None:
                    plot_msa_image(st.session_state.msa_image, st.session_state.msa_letters, plot_method)

                st.download_button(
                    label="📥 Download MSA",
                    data=st.session_state.msa_result,
                    file_name=f"msa_alignment.{msa_output_format}",
                    mime="text/plain"
                )

            with tabs[1]:
                st.subheader("🔍 Point Mutations Relative to Reference")
                if st.session_state.mutations:
                    seq_ids = sorted(list(st.session_state.mutations.keys()))
                    if seq_ids:
                        select_all = st.checkbox("Select all sequences", value=True, key="select_all_mutations")
                        selected_seq_ids = seq_ids if select_all else st.multiselect(
                            "Select sequences to view mutations for:",
                            seq_ids,
                            default=seq_ids[:min(5, len(seq_ids))],
                            key="mutation_sequence_selector"
                        )
                        for seq_id in selected_seq_ids:
                            mut_list = st.session_state.mutations.get(seq_id, [])
                            if mut_list:
                                with st.expander(f"Mutations for {seq_id} ({len(mut_list)} mutations)"):
                                    df = pd.DataFrame(mut_list, columns=["Position", "Reference", "Variant"])
                                    st.dataframe(df)
                                    string_mutations = ', '.join([f"{ref}{pos}{var}" for pos, ref, var in mut_list])
                                    st.write(f"**Mutations:** {string_mutations}")

                    csv_mutations = convert_msa_mutations_to_csv(st.session_state.mutations)
                    if csv_mutations:
                        st.download_button(
                            label="📥 Download All Mutations (CSV)",
                            data=csv_mutations,
                            file_name="msa_mutations.csv",
                            mime="text/csv"
                        )
                else:
                    st.write("No point mutations detected relative to the reference sequence.")

            if calculate_representative:
                with tabs[2]:
                    st.subheader("🧬 Representative Sequence Analysis")
                    recalculate = (st.session_state.consensus_data is None or
                                  'threshold' not in st.session_state.consensus_data or
                                  st.session_state.consensus_data['threshold'] != consensus_threshold)
                    if recalculate:
                        with st.spinner("Calculating consensus..."):
                            alignment = AlignIO.read(StringIO(st.session_state.msa_result), msa_output_format)
                            consensus_record, closest_record, min_diff, closest_seq_id = calculate_representative_sequence(alignment, threshold=consensus_threshold, seq_type=seq_type)
                            st.session_state.consensus_data = {
                                'threshold': consensus_threshold,
                                'consensus_record': consensus_record,
                                'closest_record': closest_record,
                                'min_differences': min_diff,
                                'closest_seq_id': closest_seq_id,
                                'alignment_length': alignment.get_alignment_length(),
                                'seq_count': len(alignment)
                            }

                    if st.session_state.consensus_data:
                        data = st.session_state.consensus_data
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

                        consensus_fasta = f">{data['consensus_record'].id}\n{data['consensus_record'].seq}\n>{data['closest_record'].id}\n{data['closest_record'].seq}"
                        st.download_button(
                            label="📥 Download Consensus/Representative (FASTA)",
                            data=consensus_fasta,
                            file_name="consensus_representative.fasta",
                            mime="text/plain"
                        )

            with tabs[-1]:
                st.subheader("🌳 Phylogenetic Tree from MSA")
                c_m, c_c = st.columns(2)
                tree_method = c_m.selectbox("Tree method", list(TREE_METHODS), key="msa_tree_method")
                correction = c_c.checkbox("Correct for multiple substitutions", value=True, key="msa_tree_corr",
                                          help="Jukes–Cantor (DNA) or Kimura (protein) distance correction.")
                if st.button("Calculate Phylogenetic Tree from MSA"):
                    with st.spinner("Building tree from MSA..."):
                        alignment = AlignIO.read(StringIO(st.session_state.msa_result), msa_output_format)
                        try:
                            st.session_state.msa_tree = build_tree_from_alignment(
                                alignment, seq_type, TREE_METHODS[tree_method], correction)
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
