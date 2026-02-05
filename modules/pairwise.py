import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from io import StringIO
import traceback
import re
from modules.utils import have_params_changed

def pairwise_alignment_section(sequences, seq_type):
    """
    Handles the pairwise alignment workflow including UI and computation.

    Parameters:
        sequences: List of sequence records to align
        seq_type (str): Type of sequences ('DNA' or 'Protein')
    """
    st.header("🔀 Pairwise Alignment")

    if len(sequences) < 2:
        st.warning("Please upload at least two sequences for pairwise alignment.")
        return

    align_mode = st.selectbox(
        "Select Alignment Mode",
        ("Global", "Local", "Overlap"),
        help="Global: Align entire sequences. Local: Find best local regions. Overlap: Allow end gaps."
    )

    seq_names = [seq.id for seq in sequences]

    # Replace multiselect with two separate selectboxes
    col1, col2 = st.columns(2)

    # Default to first sequence or previously selected
    default_ref_idx = 0
    if 'reference_id' in st.session_state and st.session_state.reference_id in seq_names:
        default_ref_idx = seq_names.index(st.session_state.reference_id)

    with col1:
        reference_id = st.selectbox(
            "Select Reference Sequence",
            seq_names,
            index=default_ref_idx,
            key="reference_seq_select",
            help="First sequence for alignment"
        )

    # Default to second sequence or previously selected
    default_target_idx = min(1, len(seq_names)-1)  # Default to second sequence if available
    if 'target_id' in st.session_state and st.session_state.target_id in seq_names:
        default_target_idx = seq_names.index(st.session_state.target_id)

    with col2:
        target_id = st.selectbox(
            "Select Target Sequence",
            seq_names,
            index=default_target_idx,
            key="target_seq_select",
            help="Second sequence for alignment"
        )

    # Store the current selections in session state
    st.session_state.reference_id = reference_id
    st.session_state.target_id = target_id

    # Check if selection has changed from last time
    selection_changed = ('last_selected_seqs' not in st.session_state or
                         st.session_state.last_selected_seqs != (reference_id, target_id))

    if selection_changed:
        # Reset alignment results when selection changes
        if 'alignment_text' in st.session_state:
            st.session_state.alignment_text = None
            st.session_state.pairwise_mutations = None
        # Reset the parameters tracking
        if 'last_pairwise_params' in st.session_state:
            st.session_state.last_pairwise_params = {}
        st.session_state.last_selected_seqs = (reference_id, target_id)

    c1, c2 = st.columns(2)
    open_gap_score = c1.number_input(
        "Open Gap Score",
        value=-0.5,
        step=0.1,
        help="Penalty for opening a gap in the alignment. More negative = higher penalty."
    )

    extend_gap_score = c2.number_input(
        "Extend Gap Score",
        value=-0.1,
        step=0.1,
        help="Penalty for extending an existing gap. More negative = higher penalty."
    )

    if reference_id == target_id:
        st.warning("Please select two different sequences for alignment.")
        return

    seq1 = next(seq for seq in sequences if seq.id == reference_id)
    seq2 = next(seq for seq in sequences if seq.id == target_id)

    # Create a unique parameter set to detect changes
    current_params = {
        'ref_id': reference_id,
        'target_id': target_id,
        'seq_type': seq_type,
        'align_mode': align_mode.lower(),
        'open_gap_score': open_gap_score,
        'extend_gap_score': extend_gap_score
    }

    # Check if parameters have changed
    params_changed = have_params_changed(current_params, 'last_pairwise_params')

    run_alignment = st.button("Run Pairwise Alignment")

    # Perform alignment if button is clicked or if we have results and parameters haven't changed
    if run_alignment or (st.session_state.get('alignment_text') is not None and not params_changed):
        # Only recalculate if parameters changed or explicitly requested
        if run_alignment or params_changed:
            with st.spinner("Performing pairwise alignment..."):
                alignment_text, mutations = perform_pairwise_alignment(
                    seq1, seq2, seq_type, align_mode.lower(), open_gap_score, extend_gap_score
                )
                st.session_state.alignment_text = alignment_text
                st.session_state.pairwise_mutations = mutations
                st.session_state.last_pairwise_params = current_params

        st.subheader("🧬 Alignment Result")
        st.code(st.session_state.alignment_text)

        st.subheader("🔍 Point Mutations Relative to Reference")
        if st.session_state.pairwise_mutations:
            st.write(f"**Reference Sequence:** `{reference_id}`")
            st.write(f"**Target Sequence:** `{target_id}`")
            mutations_str = ', '.join(st.session_state.pairwise_mutations)
            st.write(f"**Mutations:** {mutations_str}")

            mutation_positions = []
            mutation_labels = []
            for mut in st.session_state.pairwise_mutations:
                pos = ''.join(filter(str.isdigit, mut))
                mutation_positions.append(int(pos))
                mutation_labels.append(mut)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mutation_positions,
                y=[1]*len(mutation_positions),
                mode='markers',
                marker=dict(size=10, color='red'),
                text=mutation_labels,
                hoverinfo='text'
            ))
            fig.update_layout(
                title="Point Mutations Plot",
                xaxis_title="Position in Reference Sequence",
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            csv_mutations = convert_mutations_to_csv(st.session_state.pairwise_mutations)
            if csv_mutations:
                st.download_button(
                    label="📥 Download Mutations (CSV)",
                    data=csv_mutations,
                    file_name=f"pairwise_mutations_{reference_id}_vs_{target_id}.csv",
                    mime="text/csv"
                )
        else:
            st.write("No point mutations detected.")

        st.download_button(
            label="📥 Download Alignment",
            data=st.session_state.alignment_text,
            file_name=f"pairwise_alignment_{reference_id}_vs_{target_id}.txt",
            mime="text/plain"
        )

def perform_pairwise_alignment(seq1, seq2, seq_type, mode="global", open_gap_score=-0.5, extend_gap_score=-0.1):
    """
    Perform pairwise alignment using Biopython's Align.PairwiseAligner.

    Parameters:
        seq1: First sequence record
        seq2: Second sequence record
        seq_type (str): Type of sequences ('DNA' or 'Protein')
        mode (str): Alignment mode ('global', 'local', or 'overlap')
        open_gap_score (float): Score for opening a gap
        extend_gap_score (float): Score for extending a gap

    Returns:
        tuple: (alignment_text, mutations) - Formatted alignment text and list of mutations
    """
    try:
        # Clean sequences by converting to uppercase and removing invalid characters
        if seq_type == "DNA":
            valid_chars = set('ATGCNRYKMSWBDHV')
        else:  # Protein
            valid_chars = set('ACDEFGHIKLMNPQRSTVWYBXZJUO')

        # Clean the sequences by removing invalid characters
        seq1_str = ''.join(c.upper() for c in str(seq1.seq) if c.upper() in valid_chars or c == '-')
        seq2_str = ''.join(c.upper() for c in str(seq2.seq) if c.upper() in valid_chars or c == '-')

        if not seq1_str or not seq2_str:
             return "One or both sequences are empty after cleaning.", []

        # Create new Seq objects with the cleaned sequences
        cleaned_seq1 = Seq(seq1_str)
        cleaned_seq2 = Seq(seq2_str)

        # Get configured aligner
        aligner = get_aligner(seq_type, mode, open_gap_score, extend_gap_score)

        # Perform the alignment with cleaned sequences
        alignments = aligner.align(cleaned_seq1, cleaned_seq2)

        if not alignments:
            return "No alignments found. Sequences may be too dissimilar.", []

        alignment = next(alignments)

        # Get the alignment as strings with gaps
        aligned_seq1_str = str(alignment[0])
        aligned_seq2_str = str(alignment[1])

        # Generate match line
        match_line = generate_match_line(aligned_seq1_str, aligned_seq2_str, aligner.substitution_matrix)

        # Report mutations
        mutations = report_mutations(aligned_seq1_str, aligned_seq2_str)

        # Format alignment display
        alignment_text = format_alignment_display(seq1.id, aligned_seq1_str, match_line, seq2.id, aligned_seq2_str, alignment.score)

        return alignment_text, mutations
    except Exception as e:
        error_msg = f"An error occurred during pairwise alignment: {str(e)}"
        print(traceback.format_exc())
        return error_msg, []

def get_aligner(seq_type, mode="global", open_gap_score=-0.5, extend_gap_score=-0.1):
    """
    Configure and return a PairwiseAligner.

    Parameters:
        seq_type (str): Type of sequences ('DNA' or 'Protein')
        mode (str): Alignment mode ('global', 'local', or 'overlap')
        open_gap_score (float): Score for opening a gap
        extend_gap_score (float): Score for extending a gap

    Returns:
        Align.PairwiseAligner: Configured aligner object
    """
    aligner = Align.PairwiseAligner()

    if seq_type == "DNA":
        aligner.substitution_matrix = substitution_matrices.load("NUC.4.4")
    else:
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score

    if mode == "overlap":
        aligner.mode = "global"
        aligner.open_end_gap_score = 0.0
        aligner.extend_end_gap_score = 0.0
    else:
        aligner.mode = mode

    return aligner

def calculate_alignment_score(aligner, seq1, seq2, seq_type):
    """
    Calculate alignment score between two sequences.
    Handles basic cleaning of sequences.

    Parameters:
        aligner: Configured Bio.Align.PairwiseAligner
        seq1: First sequence (SeqRecord, Seq, or str)
        seq2: Second sequence (SeqRecord, Seq, or str)
        seq_type (str): Type of sequences ('DNA' or 'Protein')

    Returns:
        float: Alignment score or None if sequences are empty
    """
    if seq_type == "DNA":
        valid_chars = set('ATGCNRYKMSWBDHV')
    else:  # Protein
        valid_chars = set('ACDEFGHIKLMNPQRSTVWYBXZJUO')

    s1_str = str(seq1.seq) if hasattr(seq1, 'seq') else str(seq1)
    s2_str = str(seq2.seq) if hasattr(seq2, 'seq') else str(seq2)

    seq1_clean = ''.join(c.upper() for c in s1_str if c.upper() in valid_chars or c == '-')
    seq2_clean = ''.join(c.upper() for c in s2_str if c.upper() in valid_chars or c == '-')

    if not seq1_clean or not seq2_clean:
        return None

    return aligner.score(Seq(seq1_clean), Seq(seq2_clean))

def generate_match_line(aligned_seq1, aligned_seq2, substitution_matrix, threshold=1):
    """
    Generate a match line for two aligned sequences.

    Parameters:
        aligned_seq1 (str): First aligned sequence with gaps
        aligned_seq2 (str): Second aligned sequence with gaps
        substitution_matrix: Substitution matrix for scoring
        threshold (int): Threshold for similarity

    Returns:
        str: Match line with | for identical, : for similar, and space for different
    """
    match_line = []
    for a, b in zip(aligned_seq1, aligned_seq2):
        if a == b and a != '-':
            match_line.append('|')  # Identical
        elif a != '-' and b != '-' and are_similar(a, b, substitution_matrix, threshold):
            match_line.append(':')  # Similar
        else:
            match_line.append(' ')  # Different or gap

    return ''.join(match_line)


def are_similar(a, b, substitution_matrix, threshold=1):
    """
    Determine if two residues are similar based on the substitution matrix.

    Parameters:
        a (str): First residue
        b (str): Second residue
        substitution_matrix: Substitution matrix for scoring
        threshold (int): Threshold for similarity

    Returns:
        bool: True if residues are similar, False otherwise
    """
    try:
        if a == '-' or b == '-':
            return False
        score = substitution_matrix[a.upper()][b.upper()]
        return score >= threshold
    except KeyError:
        return False


def format_alignment_display(id1, aligned1, match_line, id2, aligned2, score, interval=10):
    """
    Create a formatted string for alignment display, including match lines and location markers.

    Parameters:
        id1 (str): ID of first sequence
        aligned1 (str): First aligned sequence with gaps
        match_line (str): Line showing matches between sequences
        id2 (str): ID of second sequence
        aligned2 (str): Second aligned sequence with gaps
        score (float): Alignment score
        interval (int): Interval for position markers

    Returns:
        str: Formatted alignment text
    """
    def generate_location_line(aligned_seq, interval=10):
        location = [' '] * len(aligned_seq)
        residue_count = 0
        next_mark = interval
        for i, char in enumerate(aligned_seq):
            if char != '-':
                residue_count += 1
                if residue_count == next_mark:
                    mark_str = str(next_mark)
                    mark_len = len(mark_str)
                    start_pos = i - mark_len + 1
                    if start_pos < 0:
                        start_pos = 0
                    for j, digit in enumerate(mark_str):
                        pos = start_pos + j
                        if pos < len(aligned_seq) and location[pos] == ' ':
                            location[pos] = digit
                    next_mark += interval
        return ''.join(location)

    max_label_length = max(len(id1), len("Match"), len(id2))
    id1_padded = id1.ljust(max_label_length)
    match_padded = "Match".ljust(max_label_length)
    id2_padded = id2.ljust(max_label_length)
    location1 = generate_location_line(aligned1, interval)
    location2 = generate_location_line(aligned2, interval)
    padding = ' ' * (max_label_length + 2)
    location1_padded = padding + location1
    location2_padded = padding + location2
    alignment_text = (
        "Pairwise Alignment:\n"
        f"{location1_padded}\n"
        f"{id1_padded}: {aligned1}\n"
        f"{match_padded}: {match_line}\n"
        f"{id2_padded}: {aligned2}\n"
        f"{location2_padded}\n"
        f"{'Score'.ljust(max_label_length)}: {score:.2f}\n"
    )
    return alignment_text


def report_mutations(aligned1, aligned2):
    """
    Report point mutations between two aligned sequences in the format 'S18I'.

    Parameters:
        aligned1 (str): First aligned sequence with gaps
        aligned2 (str): Second aligned sequence with gaps

    Returns:
        list: List of mutation strings (e.g., 'S18I')
    """
    mutations = []
    position = 0
    for i, (a, b) in enumerate(zip(aligned1, aligned2), start=1):
        if a != '-':
            position += 1
        if a != b and a != '-' and b != '-':
            mutations.append(f"{a}{position}{b}")
    return mutations

def convert_mutations_to_csv(mutations: list) -> str:
    """
    Converts a list of mutation strings to a CSV formatted string.
    """
    if not mutations:
        return ""

    mutation_data = []
    for mut in mutations:
        match = re.match(r"([A-Z-])(\d+)([A-Z-])", mut)
        if match:
            original, position, mutated = match.groups()
            mutation_data.append([original, int(position), mutated])

    if not mutation_data:
        return ""

    df = pd.DataFrame(mutation_data, columns=["Original", "Position", "Mutated"])
    output = StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()
