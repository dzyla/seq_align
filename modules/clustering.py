import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from Bio import AlignIO
from io import StringIO
import zipfile
import io
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from modules.msa import perform_msa
from modules.pairwise import get_aligner, calculate_alignment_score
from modules.viz import msa_to_image, plot_msa_image

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

            # Show MSA Image
            msa_image, msa_letters = msa_to_image(st.session_state.clustering_msa, "fasta")
            if msa_image is not None:
                with st.expander("View MSA Image", expanded=True):
                    plot_msa_image(msa_image, msa_letters, "Plotly (Interactive)")

            # Sequence Coverage
            coverage = calculate_sequence_coverage(alignment)

            # Plot Coverage
            df_cov = pd.DataFrame({
                "Position": range(1, align_len + 1),
                "Coverage": coverage
            })

            fig_cov = px.area(df_cov, x="Position", y="Coverage", title="Sequence Coverage per Position")
            fig_cov.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_cov, use_container_width=True)

            st.subheader("MSA Slicing")
            st.write(f"MSA Length: {align_len}")

            # Suggest range
            s_start, s_end = suggest_msa_range(coverage)

            with st.form("msa_slicing_form"):
                slice_range = st.slider(
                    "Select MSA Region to Analyze",
                    min_value=1,
                    max_value=align_len,
                    value=(s_start, s_end),
                    help="Suggested range based on sequence coverage density."
                )

                start_pos, end_pos = slice_range

                submitted = st.form_submit_button("Calculate Similarity from Slice")

                if submitted:
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

        # Choose Clustering Method
        clustering_method = st.radio(
            "Clustering Method",
            ("Similarity Threshold", "PCA Coordinates (K-Means)"),
            horizontal=True
        )

        clusters = []

        if clustering_method == "Similarity Threshold":
            # Auto-detect threshold from histogram (last bin)
            upper_tri_indices = np.triu_indices_from(matrix, k=1)
            similarities = matrix[upper_tri_indices]
            suggested_threshold = 0.8

            if len(similarities) > 0:
                # Use 50 bins from 0 to 1
                hist, bin_edges = np.histogram(similarities, bins=50, range=(0, 1))
                # Find the last bin with counts
                for i in range(len(hist)-1, -1, -1):
                    if hist[i] > 0:
                        suggested_threshold = float(bin_edges[i])
                        break

            # Threshold Slider with fine control
            threshold = st.slider(
                "Similarity Threshold for Clustering",
                0.0, 1.0, suggested_threshold, 0.01,
                format="%.2f",
                help="Sequences with similarity >= threshold will be grouped together. Default is auto-detected from histogram."
            )

            # Calculate clusters
            clusters = find_clusters(matrix, ids, threshold)

        else:
            # PCA Clustering
            if len(ids) < 3:
                st.warning("Need at least 3 sequences for PCA clustering.")
                # Fallback to single cluster
                clusters = [ids]
            else:
                pca_coords = perform_pca(matrix)

                col_k1, col_k2 = st.columns([3, 2])

                # Session state for K
                if 'pca_k_val' not in st.session_state:
                    st.session_state.pca_k_val = min(3, len(ids)-1)

                # Ensure value is valid
                if st.session_state.pca_k_val >= len(ids):
                    st.session_state.pca_k_val = max(2, len(ids) - 1)
                if st.session_state.pca_k_val < 2:
                    st.session_state.pca_k_val = 2

                with col_k1:
                    n_clusters = st.number_input(
                        "Number of Clusters (k)",
                        min_value=2,
                        max_value=max(2, len(ids)-1),
                        key='pca_k_val'
                    )

                with col_k2:
                    st.write("") # Spacer
                    st.write("")
                    if st.button("Guess Best Number"):
                        # Guess best K using Silhouette Score
                        best_k = 2
                        best_score = -1
                        limit = min(len(ids), 10)

                        for k in range(2, limit):
                            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
                            labels = km.fit_predict(pca_coords)
                            if len(set(labels)) > 1:
                                score = silhouette_score(pca_coords, labels)
                                if score > best_score:
                                    best_score = score
                                    best_k = k

                        st.session_state.pca_k_val = best_k
                        st.rerun()

                # Perform K-Means
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                labels = km.fit_predict(pca_coords)

                # Group IDs by label
                cluster_map = {}
                for idx, label in enumerate(labels):
                    if label not in cluster_map:
                        cluster_map[label] = []
                    cluster_map[label].append(ids[idx])

                clusters = list(cluster_map.values())

        # Sort clusters by size (descending)
        clusters.sort(key=len, reverse=True)

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

                title_text = f"PCA of Sequences (Method: {clustering_method})"
                if clustering_method == "Similarity Threshold":
                     title_text = f"PCA of Sequences (Clustered at {threshold:.2f})"

                fig_pca = px.scatter(
                    df_pca, x='PC1', y='PC2',
                    color='Cluster',
                    hover_data=['Sequence ID'],
                    title=title_text
                )
                fig_pca.update_traces(marker=dict(size=12))
                # Fixed aspect ratio and size
                fig_pca.update_yaxes(scaleanchor="x", scaleratio=1)
                fig_pca.update_layout(width=700, height=700)
                st.plotly_chart(fig_pca, use_container_width=False)
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

            # ZIP Export for clusters with size >= 2
            valid_clusters = [c for c in clusters if len(c) >= 2]

            if valid_clusters:
                from Bio import SeqIO
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for idx, cluster_ids in enumerate(clusters, 1):
                        if len(cluster_ids) < 2:
                            continue

                        # Find sequence objects
                        cluster_records = [seq for seq in sequences if seq.id in cluster_ids]

                        # Create FASTA content
                        fasta_str = StringIO()
                        SeqIO.write(cluster_records, fasta_str, "fasta")

                        zip_file.writestr(f"cluster_{idx}.fasta", fasta_str.getvalue())

                st.download_button(
                    label="📥 Download Clusters (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="clusters.zip",
                    mime="application/zip",
                    help="Download clusters with 2 or more sequences as individual FASTA files."
                )
            else:
                 st.info("No clusters with 2 or more sequences to export.")

        with tab2:
            st.markdown("### Similarity Distribution")
            # Extract upper triangle values excluding diagonal
            upper_tri_indices = np.triu_indices_from(matrix, k=1)
            similarities = matrix[upper_tri_indices]

            fig_hist = go.Figure(data=[go.Histogram(x=similarities, nbinsx=50)])
            if clustering_method == "Similarity Threshold":
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
        # Optimization: Calculate only upper triangle (i < j) and mirror results
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

def calculate_sequence_coverage(alignment):
    """
    Calculate the sequence coverage (number of non-gap residues) per column in the MSA.
    """
    length = alignment.get_alignment_length()
    coverage = []

    for i in range(length):
        column = alignment[:, i]
        # Count non-gap characters
        count = sum(1 for char in column if char != '-')
        coverage.append(count)

    return coverage

def suggest_msa_range(coverage, threshold_ratio=0.5):
    """
    Suggest a start and end range based on coverage density.
    Finds the longest continuous region where coverage is >= threshold * max_coverage.
    """
    if not coverage:
        return 1, 1

    max_cov = max(coverage)
    threshold = max_cov * threshold_ratio

    # Identify positions meeting the threshold
    mask = [1 if c >= threshold else 0 for c in coverage]

    # Find longest sequence of 1s
    max_len = 0
    current_len = 0
    start_idx = 0
    best_start = 0

    for i, val in enumerate(mask):
        if val == 1:
            if current_len == 0:
                start_idx = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                best_start = start_idx
            current_len = 0

    # Check end of list
    if current_len > max_len:
        max_len = current_len
        best_start = start_idx

    # If no region found (shouldn't happen if max exists), return full range
    if max_len == 0:
        return 1, len(coverage)

    # Convert 0-based index to 1-based for UI
    suggested_start = best_start + 1
    suggested_end = best_start + max_len

    return suggested_start, suggested_end
