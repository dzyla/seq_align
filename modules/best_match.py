import streamlit as st
import re
import pandas as pd
import plotly.graph_objects as go
from modules.pairwise import (
    DEFAULT_EXTEND_GAP, DEFAULT_OPEN_GAP, AlignmentError, align_pair, format_result_text, get_aligner,
)
from modules.seqtools import clean_sequence
from modules.utils import stale_notice

SORT_OPTIONS = {
    "Relative Score": "Score as a fraction of the reference's self-alignment score (comparable across queries)",
    "Identity (%)": "Identical residues over aligned (non-gap) columns",
    "Coverage (%)": "Fraction of reference residues aligned to a query residue",
    "Score": "Raw alignment score",
    "Normalized Score": "Score divided by query length",
}

def best_match_finder_section(sequences, seq_type):
    """
    Handles the Best Match Finder workflow.

    Parameters:
        sequences: List of sequence records
        seq_type (str): Type of sequences ('DNA' or 'Protein')
    """
    st.header("🔍 Best Match Finder")
    st.info(
        "Identify the sequence that best matches a reference sequence from a set of input sequences. "
        "Perform pairwise alignments and analyze score distributions."
    )

    if len(sequences) < 2:
        st.warning("Please upload at least two sequences (1 reference + 1 query) for this analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        reference_id = st.selectbox(
            "Select Reference Sequence",
            [seq.id for seq in sequences],
            help="The sequence to compare all others against."
        )

    with col2:
        align_mode = st.selectbox(
            "Select Alignment Mode",
            ("Global", "Local", "Overlap"),
            index=0,
            help="Global: Align entire sequences. Local: Find best local regions. Overlap: Allow end gaps."
        )

    # Allow excluding sequences
    exclude_ids = st.multiselect(
        "Exclude Sequences from Search",
        [seq.id for seq in sequences if seq.id != reference_id],
        help="Select sequences to exclude from the best match search."
    )

    regex_exclude = st.text_input(
        "Exclude Sequences by Regex",
        help="Enter a regex pattern to exclude sequences (e.g., '^LX' to exclude IDs starting with LX)."
    )

    c1, c2 = st.columns(2)
    open_gap_score = c1.number_input(
        "Open Gap Score",
        value=DEFAULT_OPEN_GAP,
        step=0.1,
        help="Penalty for opening a gap."
    )
    extend_gap_score = c2.number_input(
        "Extend Gap Score",
        value=DEFAULT_EXTEND_GAP,
        step=0.1,
        help="Penalty for extending a gap."
    )

    sort_metric = st.selectbox(
        "Rank by",
        list(SORT_OPTIONS),
        help="\n".join(f"**{k}**: {v}" for k, v in SORT_OPTIONS.items()),
    )

    # State management for results
    if 'best_match_results' not in st.session_state:
        st.session_state.best_match_results = None
    if 'best_match_params' not in st.session_state:
        st.session_state.best_match_params = {}

    current_params = {
        'reference_id': reference_id,
        'align_mode': align_mode,
        'open_gap_score': open_gap_score,
        'extend_gap_score': extend_gap_score,
        'seq_count': len(sequences),
        'exclude_ids': tuple(sorted(exclude_ids)),
        'regex_exclude': regex_exclude,
    }

    run_analysis = st.button("Find Best Match")

    params_changed = st.session_state.best_match_params != current_params

    if st.session_state.best_match_results is not None and params_changed and not run_analysis:
        stale_notice()
        return

    if run_analysis or (st.session_state.best_match_results is not None and not params_changed):
        if run_analysis or params_changed:
            with st.spinner("Calculating pairwise alignments against reference..."):
                aligner = get_aligner(seq_type, align_mode.lower(), open_gap_score, extend_gap_score)

                # Filter sequences: keep reference and non-excluded sequences
                sequences_to_search = []
                for seq in sequences:
                    if seq.id == reference_id:
                        sequences_to_search.append(seq)
                        continue

                    if seq.id in exclude_ids:
                        continue

                    if regex_exclude:
                        try:
                            if re.search(regex_exclude, seq.id):
                                continue
                        except re.error:
                            pass  # Ignore invalid regex

                    sequences_to_search.append(seq)

                best_match, results = find_best_match(sequences_to_search, reference_id, aligner, seq_type, sort_key=sort_metric)

                st.session_state.best_match_results = (best_match, results)
                st.session_state.best_match_params = current_params

        # Retrieve results
        best_match, results = st.session_state.best_match_results
        results = sort_results([dict(r) for r in results], sort_metric)
        best_match = next(s for s in sequences if s.id == results[0]['Sequence ID']) if results else None

        if not results:
            st.error("No results found. Please check your sequences and parameters.")
            return

        # Display Best Match Summary
        st.success(f"🏆 Best Match: **{best_match.id}** — {sort_metric}: **{results[0][sort_metric]:.2f}**")

        # Detailed Alignment of Best Match
        with st.expander("Detailed Alignment", expanded=True):
            # Allow selecting any sequence from results
            result_ids = [res['Sequence ID'] for res in results]
            selected_match_id = st.selectbox(
                "Select sequence to view alignment against Reference:",
                result_ids,
                index=0,
                key="best_match_detail_selector"
            )

            selected_match_record = next(s for s in sequences if s.id == selected_match_id)
            ref_seq = next(s for s in sequences if s.id == reference_id)

            try:
                detail = align_pair(ref_seq, selected_match_record, seq_type, align_mode.lower(),
                                    open_gap_score, extend_gap_score)
                st.code(format_result_text(detail))
                if detail.mutations:
                    st.write(f"**Mutations ({len(detail.mutations)}):** {', '.join(detail.mutations)}")
            except AlignmentError as e:
                st.error(str(e))

        # Visualizations
        st.subheader("📊 Visualization")

        # Prepare data for plotting
        df_results = pd.DataFrame(results)

        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            # Histogram of Scores
            fig_hist = go.Figure(data=[go.Histogram(x=df_results['Score'], nbinsx=20)])
            fig_hist.update_layout(
                title="Distribution of Alignment Scores",
                xaxis_title="Alignment Score",
                yaxis_title="Count",
                bargap=0.1
            )
            st.plotly_chart(fig_hist, width="stretch")

        with col_plot2:
            # Scatter Plot: Score vs Length
            fig_scatter = go.Figure(data=go.Scatter(
                x=df_results['Length'],
                y=df_results['Score'],
                mode='markers',
                text=df_results['Sequence ID'],
                hovertemplate="<b>%{text}</b><br>Length: %{x}<br>Score: %{y:.2f}<extra></extra>"
            ))
            fig_scatter.update_layout(
                title="Score vs Sequence Length",
                xaxis_title="Sequence Length",
                yaxis_title="Alignment Score"
            )
            st.plotly_chart(fig_scatter, width="stretch")

        # Results Table
        st.subheader("📋 Results Table")

        # Reorder columns
        display_cols = ['Rank', 'Sequence ID', 'Relative Score', 'Identity (%)', 'Coverage (%)', 'Score', 'Length', 'Normalized Score']
        st.dataframe(df_results[display_cols], width="stretch")

        # Download results
        csv = df_results[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="best_match_results.csv",
            mime="text/csv", on_click="ignore")

def identity_and_coverage(aligned_ref, aligned_query, ref_length):
    """Percent identity over aligned columns and percent of the reference covered."""
    aligned_cols = sum(1 for a, b in zip(aligned_ref, aligned_query) if a != '-' and b != '-')
    matches = sum(1 for a, b in zip(aligned_ref, aligned_query) if a == b and a != '-')
    identity = matches / aligned_cols * 100 if aligned_cols else 0.0
    coverage = aligned_cols / ref_length * 100 if ref_length else 0.0
    return identity, coverage


def sort_results(results, sort_key):
    """Sort result rows by `sort_key` (descending) and renumber their Rank."""
    results.sort(key=lambda x: x.get(sort_key, x['Score']), reverse=True)
    for i, res in enumerate(results):
        res['Rank'] = i + 1
    return results


def find_best_match(sequences, reference_id, aligner, seq_type, sort_key="Score"):
    """
    Find the best matching sequence against a reference sequence.

    Returns:
        tuple: (best_match_record or None, results_list sorted by sort_key)
    """
    ref_seq = next((s for s in sequences if s.id == reference_id), None)
    if not ref_seq:
        return None, []
    ref_clean = clean_sequence(ref_seq.seq, seq_type)
    if not ref_clean:
        return None, []
    self_score = aligner.score(ref_clean, ref_clean)

    results = []
    for seq in sequences:
        if seq.id == reference_id:
            continue
        query_clean = clean_sequence(seq.seq, seq_type)
        if not query_clean:
            continue
        try:
            alignment = next(iter(aligner.align(ref_clean, query_clean)))
        except StopIteration:
            continue
        score = float(alignment.score)
        identity, coverage = identity_and_coverage(str(alignment[0]), str(alignment[1]), len(ref_clean))
        length = len(seq.seq)
        results.append({
            'Sequence ID': seq.id,
            'Score': score,
            'Length': length,
            'Normalized Score': score / length if length > 0 else 0,
            'Relative Score': score / self_score if self_score > 0 else 0.0,
            'Identity (%)': identity,
            'Coverage (%)': coverage,
        })

    if not results:
        return None, []

    sort_results(results, sort_key)
    best_match_record = next(s for s in sequences if s.id == results[0]['Sequence ID'])
    return best_match_record, results
