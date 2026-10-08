import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from io import StringIO
import logging
import re
from modules.utils import have_params_changed, stale_notice
from modules.seqtools import clean_sequence
from modules.variants import call_variants
from functools import lru_cache
from dataclasses import dataclass, field
from modules.ui import page_header, show_plotly

DEFAULT_OPEN_GAP = -10.0
DEFAULT_EXTEND_GAP = -0.5

logger = logging.getLogger(__name__)


LOLLIPOP_LABEL_LIMIT = 25   # more substitutions than this: no text labels (hover only)


@lru_cache(maxsize=None)
def load_matrix(name):
    """Load a substitution matrix once per process."""
    return substitution_matrices.load(name)

def pairwise_alignment_section(sequences, seq_type):
    """
    Handles the pairwise alignment workflow including UI and computation.

    Parameters:
        sequences: List of sequence records to align
        seq_type (str): Type of sequences ('DNA' or 'Protein')
    """
    page_header("Pairwise alignment", "Global, local or overlap alignment against a reference; mutations in reference coordinates.")

    if len(sequences) < 2:
        st.warning("Please upload at least two sequences for pairwise alignment.")
        return

    align_mode = st.selectbox(
        "Select Alignment Mode",
        ("Global", "Local", "Overlap"),
        key="pairwise_mode",
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

    is_circular_ref = False
    is_circular_target = False

    if seq_type == "DNA":
        c_ref, c_tgt = st.columns(2)
        with c_ref:
            is_circular_ref = st.checkbox(
                "Reference is Circular",
                help="Consider the reference DNA sequence as circular to find the best starting alignment point."
            )
        with c_tgt:
            is_circular_target = st.checkbox(
                "Target is Circular",
                help="Consider the target DNA sequence as circular to find the best starting alignment point."
            )

    # Store the current selections in session state
    st.session_state.reference_id = reference_id
    st.session_state.target_id = target_id

    # Check if selection has changed from last time
    selection_changed = ('last_selected_seqs' not in st.session_state or
                         st.session_state.last_selected_seqs != (reference_id, target_id))

    if selection_changed:
        # Reset alignment results when selection changes
        st.session_state.pairwise_result = None
        st.session_state.last_pairwise_params = {}
        st.session_state.last_selected_seqs = (reference_id, target_id)

    c1, c2 = st.columns(2)
    open_gap_score = c1.number_input(
        "Open Gap Score",
        value=DEFAULT_OPEN_GAP,
        step=0.1,
        help="Penalty for opening a gap in the alignment. More negative = higher penalty."
    )

    extend_gap_score = c2.number_input(
        "Extend Gap Score",
        value=DEFAULT_EXTEND_GAP,
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
        'extend_gap_score': extend_gap_score,
        'is_circular_ref': is_circular_ref,
        'is_circular_target': is_circular_target
    }

    # Check if parameters have changed
    params_changed = have_params_changed(current_params, 'last_pairwise_params')

    run_alignment = st.button("Run Pairwise Alignment")

    if run_alignment:
        with st.spinner("Performing pairwise alignment..."):
            try:
                st.session_state.pairwise_result = align_pair(
                    seq1, seq2, seq_type, align_mode.lower(), open_gap_score, extend_gap_score,
                    is_circular_ref, is_circular_target
                )
                st.session_state.last_pairwise_params = current_params
            except AlignmentError as e:
                st.session_state.pairwise_result = None
                st.session_state.last_pairwise_params = {}
                st.error(str(e))
                return

    result = st.session_state.get('pairwise_result')
    if result is None:
        return
    if params_changed and not run_alignment:
        stale_notice()
        return

    alignment_text = format_result_text(result)
    stats = compute_alignment_stats(result.aligned_ref, result.aligned_query, result.match_line)
    st.subheader("Alignment result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Identity", f"{stats['identity_pct']}%", f"{stats['identical']}/{stats['length']}", delta_color="off")
    c2.metric("Similarity", f"{stats['similarity_pct']}%", f"{stats['similar']}/{stats['length']}", delta_color="off")
    c3.metric("Gaps", f"{stats['gaps_pct']}%", f"{stats['gaps']}/{stats['length']}", delta_color="off")
    c4.metric("Score", f"{result.score:.1f}")
    st.code(format_alignment_blocks(result))
    for note in result.notes:
        st.caption(note)

    st.subheader("Point mutations relative to the reference")
    fig = mutation_lollipop(result.mutations, result.ref_length, seq_type)
    if fig is None:
        st.write("No point mutations detected.")
    else:
        st.write(f"**{len(result.mutations)} substitutions:** {', '.join(result.mutations)}")
        show_plotly(fig, width="stretch")
        st.download_button(label="Download Mutations (CSV)", data=convert_mutations_to_csv(result.mutations),
                           file_name=f"pairwise_mutations_{reference_id}_vs_{target_id}.csv", mime="text/csv",
                           on_click="ignore", icon=":material/download:")

    indels = [v.label for v in call_variants(result.aligned_ref, result.aligned_query, result.ref_start,
                                             result.ref_rotation, result.ref_length) if v.kind != "sub"]
    if indels:
        st.write(f"**Indels and terminal extensions ({len(indels)}):** {', '.join(indels)}")

    if seq_type == "Protein" and result.mutations:
        from modules.structure_view import structure_highlight_panel  # local: avoids a circular import
        structure_highlight_panel(
            clean_sequence(seq1.seq, "Protein"),
            {int(re.search(r"\d+", m).group()): ("#d62728", m) for m in result.mutations},
            key="pairwise_3d",
        )

    st.download_button(label="Download Alignment", data=alignment_text,
                       file_name=f"pairwise_alignment_{reference_id}_vs_{target_id}.txt", mime="text/plain",
                       on_click="ignore", icon=":material/download:")

def get_aligner(seq_type, mode="global", open_gap_score=DEFAULT_OPEN_GAP, extend_gap_score=DEFAULT_EXTEND_GAP):
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
    aligner.substitution_matrix = load_matrix("NUC.4.4" if seq_type == "DNA" else "BLOSUM62")
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score

    if mode == "overlap":
        # Global alignment with free end gaps (semi-global). Must be set after the
        # general gap scores, which also overwrite the end-gap scores.
        aligner.mode = "global"
        aligner.end_open_gap_score = 0.0
        aligner.end_extend_gap_score = 0.0
    else:
        aligner.mode = mode

    return aligner

class AlignmentError(ValueError):
    """Raised when two sequences cannot be aligned (e.g. empty after cleaning)."""


@dataclass
class PairwiseResult:
    ref_id: str
    query_id: str
    aligned_ref: str
    aligned_query: str
    match_line: str
    score: float
    ref_start: int          # reference residues before the first aligned column (0 unless local)
    query_start: int
    ref_rotation: int       # circular shift applied to the reference (0 if none)
    ref_length: int
    mutations: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    query_rotation: int = 0   # circular shift applied to the query (0 if none)
    query_length: int = 0


def _circular_rotations(s1, s2, seq_type, open_gap_score, extend_gap_score, circular1, circular2):
    """Find rotations that bring the best local match of the two sequences into register."""
    opt_aligner = get_aligner(seq_type, "local", open_gap_score, extend_gap_score)
    doubled1 = s1 + s1 if circular1 else s1
    doubled2 = s2 + s2 if circular2 else s2
    try:
        opt_aln = next(iter(opt_aligner.align(doubled1, doubled2)))
    except StopIteration:
        return 0, 0
    start1 = int(opt_aln.coordinates[0][0])
    start2 = int(opt_aln.coordinates[1][0])
    if circular2:
        return 0, (start2 - start1) % len(s2)
    return (start1 - start2) % len(s1), 0


def align_pair(seq1, seq2, seq_type, mode="global", open_gap_score=DEFAULT_OPEN_GAP,
               extend_gap_score=DEFAULT_EXTEND_GAP, is_circular1=False, is_circular2=False):
    """Align two SeqRecords. Mutation positions are always in original reference coordinates."""
    s1 = clean_sequence(seq1.seq, seq_type)
    s2 = clean_sequence(seq2.seq, seq_type)
    if not s1 or not s2:
        raise AlignmentError("One or both sequences are empty after cleaning.")

    notes = []
    rot1 = rot2 = 0
    if is_circular1 or is_circular2:
        rot1, rot2 = _circular_rotations(s1, s2, seq_type, open_gap_score, extend_gap_score,
                                         is_circular1, is_circular2)
        if rot1:
            s1 = s1[rot1:] + s1[:rot1]
            notes.append(f"Reference sequence was circularly shifted by {rot1} positions; "
                         f"reported positions refer to the original reference.")
        if rot2:
            s2 = s2[rot2:] + s2[:rot2]
            notes.append(f"Target sequence was circularly shifted by {rot2} positions to optimize alignment.")

    aligner = get_aligner(seq_type, mode, open_gap_score, extend_gap_score)
    try:
        alignment = next(iter(aligner.align(s1, s2)))
    except StopIteration:
        raise AlignmentError("No alignment found. Sequences may be too dissimilar.")

    aligned_ref = str(alignment[0])
    aligned_query = str(alignment[1])
    ref_start = int(alignment.coordinates[0][0])
    query_start = int(alignment.coordinates[1][0])

    return PairwiseResult(
        ref_id=seq1.id,
        query_id=seq2.id,
        aligned_ref=aligned_ref,
        aligned_query=aligned_query,
        match_line=generate_match_line(aligned_ref, aligned_query, aligner.substitution_matrix),
        score=float(alignment.score),
        ref_start=ref_start,
        query_start=query_start,
        ref_rotation=rot1,
        ref_length=len(s1),
        mutations=report_mutations(aligned_ref, aligned_query, ref_start=ref_start,
                                   ref_rotation=rot1, ref_length=len(s1)),
        notes=notes,
        query_rotation=rot2,
        query_length=len(s2),
    )


def compute_alignment_stats(aligned_ref, aligned_query, match_line):
    """EMBOSS-style statistics: identity and similarity over the full alignment length."""
    length = len(aligned_ref)
    identical = match_line.count("|")
    similar = identical + match_line.count(":")
    gaps = sum(1 for a, b in zip(aligned_ref, aligned_query) if a == "-" or b == "-")

    def pct(n):
        return round(100.0 * n / length, 1) if length else 0.0

    return {"length": length, "identical": identical, "similar": similar, "gaps": gaps,
            "identity_pct": pct(identical), "similarity_pct": pct(similar), "gaps_pct": pct(gaps)}


def format_alignment_blocks(result, width=60):
    """Alignment wrapped in blocks of `width` columns with start/end positions per line."""
    name_width = max(len(result.ref_id), len(result.query_id))

    def ref_label(pos):
        if result.ref_rotation and result.ref_length and pos > 0:
            return ((pos - 1 + result.ref_rotation) % result.ref_length) + 1
        return pos

    def query_label(pos):
        if result.query_rotation and result.query_length and pos > 0:
            return ((pos - 1 + result.query_rotation) % result.query_length) + 1
        return pos

    ref_count, query_count = result.ref_start, result.query_start
    lines = []
    for i in range(0, len(result.aligned_ref), width):
        r = result.aligned_ref[i:i + width]
        q = result.aligned_query[i:i + width]
        m = result.match_line[i:i + width]
        r_res = sum(c != "-" for c in r)
        q_res = sum(c != "-" for c in q)
        r_start, r_end = ref_count + (1 if r_res else 0), ref_count + r_res
        q_start, q_end = query_count + (1 if q_res else 0), query_count + q_res
        lines.append(f"{result.ref_id:<{name_width}} {ref_label(r_start):>6} {r} {ref_label(r_end)}")
        lines.append(f"{'':<{name_width}} {'':>6} {m}")
        lines.append(f"{result.query_id:<{name_width}} {query_label(q_start):>6} {q} {query_label(q_end)}")
        lines.append("")
        ref_count += r_res
        query_count += q_res
    return "\n".join(lines)


def format_result_text(result):
    """Plain-text report: header, statistics, wrapped alignment and notes."""
    s = compute_alignment_stats(result.aligned_ref, result.aligned_query, result.match_line)
    header = (
        "Pairwise Alignment\n"
        f"Reference: {result.ref_id}\nQuery:     {result.query_id}\n"
        f"Length: {s['length']}   Identity: {s['identical']}/{s['length']} ({s['identity_pct']}%)   "
        f"Similarity: {s['similar']}/{s['length']} ({s['similarity_pct']}%)   "
        f"Gaps: {s['gaps']}/{s['length']} ({s['gaps_pct']}%)   Score: {result.score:.2f}\n\n"
    )
    notes = "".join(f"Note: {n}\n" for n in result.notes)
    return header + format_alignment_blocks(result) + ("\n" + notes if notes else "")


def mutation_lollipop(mutations, ref_length, seq_type=None):
    """Stem-and-marker plot of substitutions along the reference; None if there are none.

    With `seq_type`, stem height is the substitution-matrix score of the change.
    """
    if not mutations:
        return None
    positions = [int(re.search(r"\d+", m).group()) for m in mutations]
    if seq_type:
        matrix_name = "NUC.4.4" if seq_type == "DNA" else "BLOSUM62"
        matrix = load_matrix(matrix_name)
        heights = []
        for m in mutations:
            ref_res, alt_res = m[0], m[-1]
            try:
                heights.append(float(matrix[ref_res][alt_res]))
            except (KeyError, IndexError):
                heights.append(0.0)
        y_axis = dict(title=f"Substitution score ({matrix_name})", zeroline=True, zerolinewidth=1,
                      zerolinecolor="#888888")
    else:
        heights = [1] * len(positions)
        y_axis = dict(visible=False, range=[0, 1.6])
    mode = "markers+text" if len(mutations) <= LOLLIPOP_LABEL_LIMIT else "markers"
    fig = go.Figure()
    fig.add_trace(go.Bar(x=positions, y=heights, width=max(1, ref_length / 400),
                         marker_color="#bbbbbb", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=positions, y=heights, mode=mode, text=mutations,
                             textposition="top center", marker=dict(size=10, color="#d62728"),
                             hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title="Substitutions along the reference", xaxis=dict(title="Reference position",
                      range=[0, ref_length + 1]), yaxis=y_axis,
                      height=260, showlegend=False, margin=dict(t=40, b=40))
    return fig


def perform_pairwise_alignment(seq1, seq2, seq_type, mode="global", open_gap_score=DEFAULT_OPEN_GAP,
                               extend_gap_score=DEFAULT_EXTEND_GAP, is_circular1=False, is_circular2=False):
    """Backward-compatible wrapper around align_pair returning (alignment_text, mutations)."""
    try:
        result = align_pair(seq1, seq2, seq_type, mode, open_gap_score, extend_gap_score,
                            is_circular1, is_circular2)
    except AlignmentError as e:
        return str(e), []
    except Exception as e:
        logger.exception("Pairwise alignment failed")
        return f"An error occurred during pairwise alignment: {e}", []
    return format_result_text(result), result.mutations


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
    s1_str = str(seq1.seq) if hasattr(seq1, 'seq') else str(seq1)
    s2_str = str(seq2.seq) if hasattr(seq2, 'seq') else str(seq2)

    seq1_clean = clean_sequence(s1_str, seq_type)
    seq2_clean = clean_sequence(s2_str, seq_type)

    if not seq1_clean or not seq2_clean:
        return None

    return aligner.score(seq1_clean, seq2_clean)

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


def report_mutations(aligned1, aligned2, ref_start=0, ref_rotation=0, ref_length=None):
    """
    Report substitutions between two aligned sequences as e.g. 'S18I'.

    Positions are in original reference coordinates: `ref_start` is the number of
    reference residues before the first aligned column (local alignments) and
    `ref_rotation` undoes a circular shift of the reference.
    """
    mutations = []
    position = ref_start
    for a, b in zip(aligned1, aligned2):
        if a != '-':
            position += 1
        if a != b and a != '-' and b != '-':
            pos = position
            if ref_rotation and ref_length:
                pos = ((position - 1 + ref_rotation) % ref_length) + 1
            mutations.append(f"{a}{pos}{b}")
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
