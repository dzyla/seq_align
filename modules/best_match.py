import streamlit as st
import re
import pandas as pd
import plotly.graph_objects as go
from modules.pairwise import get_aligner, perform_pairwise_alignment, calculate_alignment_score

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

    if len(sequences) < 3:
        st.warning("Please upload at least three sequences (1 reference + 2 queries) for this analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        reference_id = st.selectbox(
            "Select Reference Sequence",
            [seq.id for seq in sequences][::-1],
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
        value=-0.5,
        step=0.1,
        help="Penalty for opening a gap."
    )
    extend_gap_score = c2.number_input(
        "Extend Gap Score",
        value=-0.1,
        step=0.1,
        help="Penalty for extending a gap."
    )

    sort_metric = st.selectbox(
        "Rank by",
        ("Score", "Normalized Score"),
        help="Choose the metric to rank the results."
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
        'sort_metric': sort_metric
    }

    run_analysis = st.button("Find Best Match")

    params_changed = st.session_state.best_match_params != current_params

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

        if not results:
            st.error("No results found. Please check your sequences and parameters.")
            return

        # Display Best Match Summary
        st.success(f"🏆 Best Match: **{best_match.id}** with Score: **{results[0]['Score']:.2f}**")

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

            alignment_text, mutations = perform_pairwise_alignment(
                ref_seq, selected_match_record, seq_type,
                align_mode.lower(), open_gap_score, extend_gap_score
            )
            st.code(alignment_text)

            if mutations:
                 st.write(f"**Mutations ({len(mutations)}):** {', '.join(mutations)}")

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
            st.plotly_chart(fig_hist, use_container_width=True)

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
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Results Table
        st.subheader("📋 Results Table")

        # Reorder columns
        display_cols = ['Rank', 'Sequence ID', 'Score', 'Length', 'Normalized Score']
        st.dataframe(df_results[display_cols], use_container_width=True)

        # Download results
        csv = df_results[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="best_match_results.csv",
            mime="text/csv"
        )

def find_best_match(sequences, reference_id, aligner, seq_type, sort_key="Score"):
    """
    Find the best matching sequence against a reference sequence.

    Parameters:
        sequences (list): List of SeqRecord objects.
        reference_id (str): ID of the reference sequence.
        aligner (Bio.Align.PairwiseAligner): Configured aligner.
        seq_type (str): 'DNA' or 'Protein'.
        sort_key (str): Key to sort results by ('Score' or 'Normalized Score').

    Returns:
        tuple: (best_match_record, results_list)
            best_match_record: SeqRecord of the best match (or None if no others)
            results_list: List of dictionaries containing score, id, etc.
    """
    # Find reference sequence
    ref_seq = next((s for s in sequences if s.id == reference_id), None)
    if not ref_seq:
        return None, []

    results = []

    for seq in sequences:
        if seq.id == reference_id:
            continue

        score = calculate_alignment_score(aligner, ref_seq, seq, seq_type)
        if score is None:
            continue

        length = len(seq.seq)
        normalized_score = score / length if length > 0 else 0

        results.append({
            'Sequence ID': seq.id,
            'Score': score,
            'Length': length,
            'Normalized Score': normalized_score
            # 'sequence': seq # Store record if needed, but keeping it simple for dataframe
        })

    if not results:
        return None, []

    # Sort by sort_key descending
    results.sort(key=lambda x: x.get(sort_key, x['Score']), reverse=True)

    # Add Rank
    for i, res in enumerate(results):
        res['Rank'] = i + 1

    best_match_id = results[0]['Sequence ID']
    best_match_record = next(s for s in sequences if s.id == best_match_id)

    return best_match_record, results
