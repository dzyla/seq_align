import streamlit as st
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pyfamsa import Aligner as PyFAMSAAligner, Sequence as PyFAMSASequence
from io import StringIO
import traceback
import pandas as pd
from typing import List, Dict, Tuple
from modules.utils import have_params_changed
from modules.viz import msa_to_image, plot_msa_image
from modules.parsers import parse_sequences_from_text
from modules.phylogeny import phylogenetic_tree_section, build_tree_from_alignment

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
        ("fasta", "clustal", "phylip", "stockholm"),
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
                msa_result, mutations = perform_msa(sequences, ref_seq_id, seq_type, msa_output_format)
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
                seqs_total, _ = parse_sequences_from_text(st.session_state.msa_result)
                if seqs_total:
                    st.write(f"Total sequences: {len(seqs_total)}. MSA length: {len(seqs_total[0].seq)}")

                if st.session_state.msa_image is not None and st.session_state.msa_letters is not None:
                    plot_msa_image(st.session_state.msa_image, st.session_state.msa_letters, plot_method)

                st.download_button(
                    label="📥 Download MSA",
                    data=st.session_state.msa_result,
                    file_name=f"msa_alignment.{msa_output_format}",
                    mime=f"text/{msa_output_format}"
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
                            consensus_record, closest_record, min_diff, closest_seq_id = calculate_representative_sequence(alignment, threshold=consensus_threshold)
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
                if st.button("Calculate Phylogenetic Tree from MSA"):
                    with st.spinner("Building tree from MSA..."):
                        alignment = AlignIO.read(StringIO(st.session_state.msa_result), msa_output_format)
                        tree = build_tree_from_alignment(alignment, seq_type)
                        if tree:
                            st.session_state.msa_tree = tree

                if 'msa_tree' in st.session_state and st.session_state.msa_tree:
                    phylogenetic_tree_section(st.session_state.msa_tree)

def perform_msa(sequences, reference_id, seq_type, output_format):
    """
    Perform Multiple Sequence Alignment using pyFAMSA and report mutations relative to the reference sequence.

    Parameters:
        sequences: List of sequence records to align
        reference_id (str): ID of the reference sequence for mutation detection
        seq_type (str): Type of sequences ('DNA' or 'Protein')
        output_format (str): Format for the MSA output

    Returns:
        tuple: (msa_text, mutations) - MSA text and dictionary of mutations per sequence
    """
    try:
        pyfamsa_sequences = [
            PyFAMSASequence(seq.id.encode(), str(seq.seq).encode()) for seq in sequences
        ]
        aligner = PyFAMSAAligner(guide_tree="upgma")
        msa = aligner.align(pyfamsa_sequences)
        aligned_seq_records = [
            SeqRecord(
                Seq(seq.sequence.decode()),
                id=seq.id.decode(),
                description=""
            )
            for seq in msa
        ]
        msa_io = StringIO()
        SeqIO.write(aligned_seq_records, msa_io, output_format)
        msa_text = msa_io.getvalue()
        # Get reference sequence from aligned records
        ref_seq = next((seq for seq in aligned_seq_records if seq.id == reference_id), None)
        if not ref_seq:
            st.error("Reference sequence not found in MSA results.")
            return msa_text, {}
        mutations = {}
        for seq in aligned_seq_records:
            if seq.id == reference_id:
                continue
            seq_mutations = []
            ref_position = 0
            for i, (ref_base, seq_base) in enumerate(zip(ref_seq.seq, seq.seq), start=1):
                if ref_base != '-':
                    ref_position += 1
                # Report mutation only if both residues are not gaps.
                if ref_base != seq_base and ref_base != '-' and seq_base != '-':
                    seq_mutations.append((ref_position, ref_base, seq_base))
            mutations[seq.id] = seq_mutations
        return msa_text, mutations
    except Exception as e:
        st.error(f"An error occurred during MSA: {e}")
        print(traceback.format_exc())
        return "", {}

def calculate_representative_sequence(alignment, threshold=0.7):
    """
    Calculate the consensus sequence and find the most representative sequence in the alignment.

    Parameters:
        alignment: Biopython Alignment object
        threshold (float): Consensus threshold (default: 0.7)

    Returns:
        tuple: (consensus_record, closest_record, min_differences, closest_seq_id)
    """
    # Calculate the consensus sequence
    consensus_seq_list = []
    length = alignment.get_alignment_length()
    for i in range(length):
        column = alignment[:, i]
        counts = {}
        for char in column:
            counts[char] = counts.get(char, 0) + 1

        max_count = 0
        most_freq = 'X'
        for char, count in counts.items():
            if count > max_count:
                max_count = count
                most_freq = char

        if max_count / len(column) >= threshold:
            consensus_seq_list.append(most_freq)
        else:
            consensus_seq_list.append('X')

    consensus_seq = "".join(consensus_seq_list)

    # Find the sequence closest to the consensus
    def count_differences(seq1, seq2):
        """Count the number of differences between two sequences."""
        return sum(1 for a, b in zip(seq1, seq2) if a != b)

    min_differences = None
    closest_sequence = None
    closest_seq_id = None

    for record in alignment:
        seq = str(record.seq)
        differences = count_differences(consensus_seq, seq)

        if (min_differences is None) or (differences < min_differences):
            min_differences = differences
            closest_sequence = seq
            closest_seq_id = record.id

    # Create SeqRecord objects
    consensus_record = SeqRecord(Seq(consensus_seq), id="Consensus", description="Consensus sequence")
    closest_record = SeqRecord(Seq(closest_sequence), id=closest_seq_id, description="Most representative sequence")

    return consensus_record, closest_record, min_differences, closest_seq_id

def save_msa_to_fasta(msa: List[PyFAMSASequence], output_path: str) -> bool:
    """
    Converts pyFAMSA MSA output to a FASTA file compatible with Biopython.

    Parameters:
        msa (List[PyFAMSASequence]): List of aligned sequences from pyFAMSA
        output_path (str): Path to save the FASTA file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        seq_records = []
        for seq in msa:
            seq_id = seq.id.decode('utf-8')
            sequence = seq.sequence.decode('utf-8')
            record = SeqRecord(Seq(sequence), id=seq_id, description="")
            seq_records.append(record)
        SeqIO.write(seq_records, output_path, "fasta")
        print(f"MSA successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"An error occurred while saving MSA to FASTA: {e}")
        return False


def get_msa_as_fasta(msa: List[PyFAMSASequence]) -> str:
    """
    Converts pyFAMSA MSA output to a FASTA-formatted string compatible with Biopython.

    Parameters:
        msa (List[PyFAMSASequence]): List of aligned sequences from pyFAMSA

    Returns:
        str: FASTA-formatted string
    """
    try:
        seq_records = []
        for seq in msa:
            seq_id = seq.id.decode('utf-8')
            sequence = seq.sequence.decode('utf-8')
            record = SeqRecord(Seq(sequence), id=seq_id, description="")
            seq_records.append(record)
        fasta_io = StringIO()
        SeqIO.write(seq_records, fasta_io, "fasta")
        fasta_content = fasta_io.getvalue()
        return fasta_content
    except Exception as e:
        st.error(f"An error occurred while converting MSA to FASTA: {e}")
        return ""

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
