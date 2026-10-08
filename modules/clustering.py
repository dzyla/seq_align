import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import zipfile
import io
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from modules.msa import MSAError, aligned_to_msa, cached_run_msa
from modules.pairwise import get_aligner
from modules.viz import msa_to_image, plot_msa_image
from collections import deque
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from modules.seqtools import clean_sequence

HIERARCHICAL = "Hierarchical (average linkage)"
CONNECTED = "Connected components (single linkage)"
KMEANS = "K-Means on PCoA coordinates"
CLUSTERING_METHODS = (HIERARCHICAL, CONNECTED, KMEANS)

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
                try:
                    st.session_state.clustering_msa = cached_run_msa(
                        tuple((s.id, str(s.seq)) for s in sequences), seq_type)
                except MSAError as e:
                    st.error(str(e))

        aligned = st.session_state.clustering_msa
        if aligned:
            st.success("MSA calculated.")
            aligned_ids = [seq_id for seq_id, _ in aligned]
            aligned_seqs = [seq for _, seq in aligned]
            align_len = len(aligned_seqs[0])

            msa_image, msa_letters = msa_to_image(
                "".join(f">{i}\n{s}\n" for i, s in aligned), "fasta")
            if msa_image is not None:
                with st.expander("View MSA Image", expanded=True):
                    plot_msa_image(msa_image, msa_letters, "Plotly (Interactive)")

            coverage = calculate_sequence_coverage(aligned_seqs)
            fig_cov = px.area(pd.DataFrame({"Position": range(1, align_len + 1), "Coverage": coverage}),
                              x="Position", y="Coverage", title="Sequence Coverage per Position")
            fig_cov.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_cov, width="stretch")

            st.subheader("MSA Slicing")
            st.write(f"MSA Length: {align_len}")
            s_start, s_end = suggest_msa_range(coverage.tolist())
            with st.form("msa_slicing_form"):
                start_pos, end_pos = st.slider("Select MSA Region to Analyze", min_value=1, max_value=align_len,
                                               value=(s_start, s_end),
                                               help="Suggested range based on sequence coverage density.")
                if st.form_submit_button("Calculate Similarity from Slice"):
                    with st.spinner("Calculating similarity matrix..."):
                        st.session_state.clustering_matrix = msa_identity_matrix(
                            [s[start_pos - 1:end_pos] for s in aligned_seqs])
                        st.session_state.clustering_ids = aligned_ids

    else: # Accurate Pairwise
        if st.button("Calculate Pairwise Similarity"):
            with st.spinner("Calculating pairwise similarity matrix (this may take time)..."):
                progress_bar = st.progress(0.0)
                matrix, sorted_ids = calculate_similarity_matrix_pairwise(
                    sequences, seq_type, align_mode, progress_callback=progress_bar.progress)
                progress_bar.empty()
                st.session_state.clustering_matrix = matrix
                st.session_state.clustering_ids = sorted_ids

    # Step 2: Visualization and Clustering (if matrix exists)
    if st.session_state.clustering_matrix is not None:
        matrix = st.session_state.clustering_matrix
        ids = st.session_state.clustering_ids

        st.divider()
        st.subheader("📊 Analysis")

        clustering_method = st.radio(
            "Clustering Method",
            CLUSTERING_METHODS,
            horizontal=True,
            help="Average linkage avoids the chaining effect of single linkage. "
                 "K-Means runs on classical MDS (PCoA) coordinates of the distance matrix."
        )

        clusters = []
        threshold = None
        similarities = matrix[np.triu_indices_from(matrix, k=1)]

        if clustering_method in (HIERARCHICAL, CONNECTED):
            threshold = st.slider(
                "Similarity Threshold for Clustering",
                0.0, 1.0, suggest_threshold(similarities), 0.01,
                format="%.2f",
                help="Sequences are grouped when their similarity is at or above the threshold. "
                     "The default comes from Otsu's method on the similarity histogram."
            )
            if clustering_method == HIERARCHICAL:
                clusters, _ = hierarchical_clusters(matrix, ids, threshold)
            else:
                clusters = find_clusters(matrix, ids, threshold)
        elif len(ids) < 3:
            st.warning("Need at least 3 sequences for K-Means clustering.")
            clusters = [list(ids)]
        else:
            coords = perform_pcoa(matrix)
            col_k1, col_k2 = st.columns([3, 2])
            if 'pca_k_val' not in st.session_state:
                st.session_state.pca_k_val = min(3, len(ids) - 1)
            st.session_state.pca_k_val = min(max(2, st.session_state.pca_k_val), max(2, len(ids) - 1))
            with col_k1:
                n_clusters = st.number_input(
                    "Number of Clusters (k)", min_value=2, max_value=max(2, len(ids) - 1), key='pca_k_val'
                )
            with col_k2:
                st.write("")
                st.write("")
                if st.button("Guess Best Number"):
                    best_k, best_score = 2, -1
                    for k in range(2, min(len(ids), 10)):
                        labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(coords)
                        if len(set(labels)) > 1:
                            score = silhouette_score(coords, labels)
                            if score > best_score:
                                best_k, best_score = k, score
                    st.session_state.pca_k_val = best_k
                    st.rerun()
            labels = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit_predict(coords)
            cluster_map = {}
            for idx, label in enumerate(labels):
                cluster_map.setdefault(label, []).append(ids[idx])
            clusters = list(cluster_map.values())

        # Sort clusters by size (descending)
        clusters.sort(key=len, reverse=True)

        # Create a mapping of ID to Cluster
        id_to_cluster = {}
        for idx, cluster in enumerate(clusters):
            for seq_id in cluster:
                id_to_cluster[seq_id] = f"Cluster {idx+1}"

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["PCoA & Clusters", "Similarity Histogram", "Matrix Heatmap"])

        with tab1:
            st.markdown("### PCoA of Sequences")
            if len(ids) >= 3:
                df_pca = pd.DataFrame(perform_pcoa(matrix), columns=['PCoA 1', 'PCoA 2'])
                df_pca['Sequence ID'] = ids
                df_pca['Cluster'] = [id_to_cluster.get(x, "Unknown") for x in ids]
                title_text = f"PCoA of Sequences ({clustering_method})"
                if threshold is not None:
                    title_text += f", threshold {threshold:.2f}"
                fig_pca = px.scatter(df_pca, x='PCoA 1', y='PCoA 2', color='Cluster',
                                     hover_data=['Sequence ID'], title=title_text)
                fig_pca.update_traces(marker=dict(size=12))
                fig_pca.update_yaxes(scaleanchor="x", scaleratio=1)
                fig_pca.update_layout(width=700, height=700)
                st.plotly_chart(fig_pca, width="content")
            else:
                st.info("Need at least 3 sequences for PCoA.")

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
            st.dataframe(df_clusters, width="stretch")
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
            if threshold is not None:
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
            fig_hist.update_layout(
                title="Distribution of Pairwise Similarities",
                xaxis_title="Similarity Score",
                yaxis_title="Count",
                bargap=0.1
            )
            st.plotly_chart(fig_hist, width="stretch")

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
            st.plotly_chart(fig_hm, width="stretch")

def msa_identity_matrix(aligned):
    """Pairwise identity from aligned strings: identical residues / columns where either has a residue."""
    n = len(aligned)
    matrix = np.eye(n)
    if n == 0:
        return matrix
    arr = np.frombuffer("".join(s.upper() for s in aligned).encode(), dtype=np.uint8).reshape(n, -1)
    residue = arr != ord('-')
    for i in range(n - 1):
        same = (arr[i] == arr[i + 1:]) & residue[i]
        cols = residue[i] | residue[i + 1:]
        counts = cols.sum(axis=1)
        values = np.divide(same.sum(axis=1), counts, out=np.zeros(len(counts)), where=counts > 0)
        matrix[i, i + 1:] = values
        matrix[i + 1:, i] = values
    return matrix


def pairwise_identity(aligned1, aligned2, mode, len1, len2):
    """Identity of a pairwise alignment.

    Global/overlap: matches / alignment columns. Local: matches / length of the shorter
    sequence, so a short shared motif does not score as identical sequences.
    """
    matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
    if mode == "local":
        denominator = min(len1, len2)
    else:
        denominator = sum(1 for a, b in zip(aligned1, aligned2) if a != '-' or b != '-')
    return matches / denominator if denominator else 0.0


def calculate_similarity_matrix_pairwise(sequences, seq_type, align_mode, progress_callback=None):
    """N×N identity matrix from all-against-all pairwise alignments."""
    mode = align_mode.lower()
    ids = [seq.id for seq in sequences]
    cleaned = [clean_sequence(seq.seq, seq_type) for seq in sequences]
    n = len(sequences)
    matrix = np.eye(n)
    aligner = get_aligner(seq_type, mode)
    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = 0.0
            if cleaned[i] and cleaned[j]:
                try:
                    aln = next(iter(aligner.align(cleaned[i], cleaned[j])))
                    sim = pairwise_identity(str(aln[0]), str(aln[1]), mode, len(cleaned[i]), len(cleaned[j]))
                except StopIteration:
                    sim = 0.0
            matrix[i, j] = matrix[j, i] = sim
            done += 1
            if progress_callback and total_pairs and done % 10 == 0:
                progress_callback(done / total_pairs)
    return matrix, ids


def suggest_threshold(similarities, bins=50):
    """Otsu threshold on the similarity histogram; 0.8 when there is nothing to split."""
    values = np.asarray(similarities, dtype=float)
    if values.size < 2 or np.ptp(values) == 0:
        return float(values.mean()) if values.size else 0.8
    hist, edges = np.histogram(values, bins=bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = w0 * w1 * (m0 / w0 - (m0[-1] - m0) / w1) ** 2
    return float(round(edges[int(np.argmax(np.nan_to_num(between))) + 1], 2))


def hierarchical_clusters(similarity, ids, threshold, method="average"):
    """Cut an average-linkage tree at distance 1 - threshold.

    Returns (clusters as lists of IDs, leaf order of the dendrogram as row indices).
    """
    if len(ids) < 2:
        return [list(ids)], list(range(len(ids)))
    distance = np.clip(1.0 - np.asarray(similarity, dtype=float), 0, None)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method=method)
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")
    groups = {}
    for seq_id, label in zip(ids, labels):
        groups.setdefault(label, []).append(seq_id)
    return list(groups.values()), leaves_list(tree).tolist()


def perform_pcoa(matrix, n_components=2):
    """Classical multidimensional scaling (PCoA) of the distance 1 - similarity."""
    distance = 1.0 - np.asarray(matrix, dtype=float)
    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * centering @ (distance ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, order] * np.sqrt(np.clip(eigenvalues[order], 0, None))


def find_clusters(similarity_matrix, ids, threshold):
    """Connected components of the graph with edges where similarity >= threshold (single linkage)."""
    n = len(ids)
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        component = []
        queue = deque([i])
        visited[i] = True
        while queue:
            curr = queue.popleft()
            component.append(ids[curr])
            for j in range(n):
                if not visited[j] and similarity_matrix[curr, j] >= threshold:
                    visited[j] = True
                    queue.append(j)
        clusters.append(component)
    return clusters

def calculate_sequence_coverage(aligned):
    """Number of non-gap residues per alignment column."""
    if not aligned:
        return np.zeros(0, dtype=int)
    arr = np.frombuffer("".join(aligned).encode(), dtype=np.uint8).reshape(len(aligned), -1)
    return (arr != ord('-')).sum(axis=0)

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
