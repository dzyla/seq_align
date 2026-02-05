import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from Bio import AlignIO
from io import StringIO
from sklearn.decomposition import PCA
from modules.msa import perform_msa
from modules.pairwise import get_aligner, calculate_alignment_score

def sequence_clustering_section(sequences, seq_type):
    """
    Handles the Sequence Clustering workflow with enhanced features.
    """
    st.header("🧩 Sequence Clustering")
    st.info("Group similar sequences together based on pairwise similarity.")

    if len(sequences) < 2:
        st.warning("Need at least 2 sequences to perform clustering.")
        return

    # Sidebar controls or Main area controls? Let's use columns.
    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox(
            "Similarity Method",
            ("Fast (MSA-based)", "Accurate (Pairwise)"),
            help="Fast uses MSA to estimate similarity. Accurate performs N*N pairwise alignments."
        )

    align_mode = "global"
    if method == "Accurate (Pairwise)":
        with col2:
            align_mode = st.selectbox(
                "Pairwise Alignment Mode",
                ("Global", "Local"),
                index=0
            )

    # Session state initialization for clustering specific data
    if 'clustering_matrix' not in st.session_state:
        st.session_state.clustering_matrix = None
    if 'clustering_ids' not in st.session_state:
        st.session_state.clustering_ids = None
    if 'clustering_msa' not in st.session_state:
        st.session_state.clustering_msa = None

    # Step 1: MSA / Slicing (only for Fast method)
    sliced_matrix = None
    ids = [seq.id for seq in sequences]

    if method == "Fast (MSA-based)":
        if st.button("Run/Update MSA"):
            with st.spinner("Calculating MSA..."):
                msa_text, _ = perform_msa(sequences, sequences[0].id, seq_type, "fasta")
                if msa_text:
                    st.session_state.clustering_msa = msa_text

        if st.session_state.clustering_msa:
            st.success("MSA calculated.")
            # Parse MSA to get dimensions
            alignment = AlignIO.read(StringIO(st.session_state.clustering_msa), "fasta")
            align_len = alignment.get_alignment_length()

            st.subheader("MSA Slicing")
            st.write(f"MSA Length: {align_len}")

            slice_range = st.slider(
                "Select MSA Region to Analyze",
                min_value=1,
                max_value=align_len,
                value=(1, align_len)
            )

            start_pos, end_pos = slice_range

            if st.button("Calculate Similarity from Slice"):
                with st.spinner("Calculating similarity matrix..."):
                    # Slice the alignment
                    # Note: alignment objects can be sliced like lists/arrays: alignment[:, start:end]
                    # But index is 0-based. Slider is 1-based.
                    sliced_alignment = alignment[:, start_pos-1:end_pos]

                    # Calculate identity matrix
                    matrix, sorted_ids = calculate_similarity_matrix_from_msa(sliced_alignment, sequences)
                    st.session_state.clustering_matrix = matrix
                    st.session_state.clustering_ids = sorted_ids

    else: # Accurate Pairwise
        if st.button("Calculate Pairwise Similarity"):
            with st.spinner("Calculating pairwise similarity matrix (this may take time)..."):
                matrix, sorted_ids = calculate_similarity_matrix_pairwise(sequences, seq_type, align_mode)
                st.session_state.clustering_matrix = matrix
                st.session_state.clustering_ids = sorted_ids

    # Step 2: Visualization and Clustering (if matrix exists)
    if st.session_state.clustering_matrix is not None:
        matrix = st.session_state.clustering_matrix
        ids = st.session_state.clustering_ids

        st.divider()
        st.subheader("📊 Analysis")

        # Threshold Slider with fine control
        threshold = st.slider(
            "Similarity Threshold for Clustering",
            0.0, 1.0, 0.8, 0.01,
            format="%.2f",
            help="Sequences with similarity >= threshold will be grouped together."
        )

        # Calculate clusters
        clusters = find_clusters(matrix, ids, threshold)

        # Create a mapping of ID to Cluster
        id_to_cluster = {}
        for idx, cluster in enumerate(clusters):
            for seq_id in cluster:
                id_to_cluster[seq_id] = f"Cluster {idx+1}"

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["PCA & Clusters", "Similarity Histogram", "Matrix Heatmap"])

        with tab1:
            st.markdown("### PCA of Sequences")
            if len(ids) >= 3:
                pca_coords = perform_pca(matrix)

                # Create DataFrame for plotting
                df_pca = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'])
                df_pca['Sequence ID'] = ids
                df_pca['Cluster'] = [id_to_cluster.get(x, "Unknown") for x in ids]

                fig_pca = px.scatter(
                    df_pca, x='PC1', y='PC2',
                    color='Cluster',
                    hover_data=['Sequence ID'],
                    title=f"PCA of Sequences (Clustered at {threshold:.2f})"
                )
                fig_pca.update_traces(marker=dict(size=12))
                st.plotly_chart(fig_pca, use_container_width=True)
            else:
                st.info("Need at least 3 sequences for PCA.")

            st.markdown("### Cluster Details")
            st.write(f"Found {len(clusters)} clusters.")

            cluster_data = []
            for idx, cluster in enumerate(clusters, 1):
                cluster_data.append({
                    "Cluster ID": idx,
                    "Count": len(cluster),
                    "Sequences": ", ".join(cluster)
                })

            df_clusters = pd.DataFrame(cluster_data)
            st.dataframe(df_clusters, use_container_width=True)
            st.download_button(
                "📥 Download Clusters (CSV)",
                df_clusters.to_csv(index=False),
                "clusters.csv",
                "text/csv"
            )

        with tab2:
            st.markdown("### Similarity Distribution")
            # Extract upper triangle values excluding diagonal
            upper_tri_indices = np.triu_indices_from(matrix, k=1)
            similarities = matrix[upper_tri_indices]

            fig_hist = go.Figure(data=[go.Histogram(x=similarities, nbinsx=50)])
            fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
            fig_hist.update_layout(
                title="Distribution of Pairwise Similarities",
                xaxis_title="Similarity Score",
                yaxis_title="Count",
                bargap=0.1
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab3:
            st.markdown("### Similarity Matrix")
            fig_hm = go.Figure(data=go.Heatmap(
                z=matrix,
                x=ids,
                y=ids,
                colorscale="Viridis",
                zmin=0, zmax=1
            ))
            fig_hm.update_layout(
                title="Pairwise Similarity Matrix",
                xaxis_showticklabels=False,
                yaxis_showticklabels=False,
                height=600
            )
            st.plotly_chart(fig_hm, use_container_width=True)

def calculate_similarity_matrix_from_msa(alignment, sequences):
    """
    Calculate N*N similarity matrix from a sliced MSA.
    """
    # Create a map of id -> sequence string from alignment
    aln_dict = {rec.id: str(rec.seq) for rec in alignment}

    # Use the IDs from input sequences to maintain order
    ids = [seq.id for seq in sequences]
    # Ensure we only use IDs that are in the alignment (should be all)
    ids = [i for i in ids if i in aln_dict]

    aligned_seqs = [aln_dict[seq_id] for seq_id in ids]
    n = len(ids)
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            s1 = aligned_seqs[i]
            s2 = aligned_seqs[j]

            # Simple identity: matches / length of slice
            # If slice length is 0 (shouldn't happen with min_value=1), score is 0
            if len(s1) == 0:
                score = 0
            else:
                matches = sum(1 for a, b in zip(s1, s2) if a == b and a != '-')
                score = matches / len(s1)

            matrix[i, j] = matrix[j, i] = score

    return matrix, ids

def calculate_similarity_matrix_pairwise(sequences, seq_type, align_mode):
    """
    Calculate N*N similarity matrix using pairwise alignment.
    """
    n = len(sequences)
    ids = [seq.id for seq in sequences]
    matrix = np.zeros((n, n))

    aligner = get_aligner(seq_type, align_mode.lower(), -10.0, -0.5)

    progress_bar = st.progress(0)
    total_pairs = (n * (n - 1)) // 2
    current_pair = 0

    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            alignments = aligner.align(sequences[i], sequences[j])
            if not alignments:
                sim = 0
            else:
                aln = alignments[0]
                s1_aln = str(aln[0])
                # Count matches
                matches = sum(1 for a, b in zip(str(aln[0]), str(aln[1])) if a == b and a != '-')
                # Normalize by alignment length (including gaps)
                # or length of longer sequence?
                # Usually matches / alignment_length is good for identity
                sim = matches / len(s1_aln) if len(s1_aln) > 0 else 0

            matrix[i, j] = matrix[j, i] = sim

            current_pair += 1
            if total_pairs > 0 and current_pair % 10 == 0:
                progress_bar.progress(current_pair / total_pairs)

    progress_bar.empty()
    return matrix, ids

def perform_pca(matrix):
    """
    Perform PCA on the similarity matrix.
    Treats the similarity matrix row vectors as features for each sequence.
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)
    return coords

def find_clusters(similarity_matrix, ids, threshold):
    """
    Find connected components in the similarity graph.
    """
    n = len(ids)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if not visited[i]:
            # Start BFS/DFS
            component = []
            queue = [i]
            visited[i] = True

            while queue:
                curr = queue.pop(0)
                component.append(ids[curr])

                for j in range(n):
                    if not visited[j] and similarity_matrix[curr, j] >= threshold:
                        visited[j] = True
                        queue.append(j)

            clusters.append(component)

    return clusters
