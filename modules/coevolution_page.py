"""Coevolution page: contacts from a deep MSA (current dataset or an uploaded A3M / Stockholm / FASTA)."""
import hashlib
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.coevolution import (CONTACT_CUTOFF, LIMITS, METHODS, PLM_ITERATIONS, contact_precision, mean_field_dca,
                                 parse_alignment, plm_dca, prepare_msa, sequence_weights, structure_distances,
                                 top_pairs)
from modules.ui import page_header, show_plotly

CLOSE, FAR, UNKNOWN = "#1b9e4b", "#d6312b", "#9e9e9e"


@st.cache_data(max_entries=4, show_spinner=False)
def cached_parse(data):
    return parse_alignment(data.decode("utf-8", "replace"))


@st.cache_data(max_entries=4, show_spinner=False)
def cached_contacts(rows, method, max_seqs, start, end, iters):
    """Prepared MSA summary and APC scores for one input and setting combination (cached)."""
    prepared = prepare_msa(list(rows), max_seqs=max_seqs)
    msa = prepared.msa[:, start - 1:end]
    t0 = time.time()
    w = sequence_weights(msa)
    scores = plm_dca(msa, w, iters=iters) if method == "plm" else mean_field_dca(msa, w)
    return {"scores": scores, "query": prepared.query_seq[start - 1:end], "offset": start - 1,
            "n_input": prepared.n_input, "n_used": len(msa), "neff": float(w.sum()),
            "dropped": (prepared.n_dropped_length, prepared.n_dropped_gappy, prepared.n_subsampled),
            "seconds": time.time() - t0}


@st.cache_data(max_entries=4, show_spinner=False)
def cached_distances(name, data, fmt, chain_id, query):
    from modules.structures import StructureFile

    return structure_distances(StructureFile(name=name, data=data, fmt=fmt), chain_id, query)


def contact_map_figure(result, n_show, distances=None, min_sep=6):
    """Predicted pairs above the diagonal (by score), structure contacts below (grey), query numbering."""
    L, off, q = len(result["query"]), result["offset"], result["query"]
    pairs = top_pairs(result["scores"], min_sep, n_show)
    fig = go.Figure()
    if distances is not None:
        ii, jj = np.where(np.triu(np.nan_to_num(distances, nan=99) < CONTACT_CUTOFF, min_sep))
        fig.add_trace(go.Scatter(x=ii + 1 + off, y=jj + 1 + off, mode="markers", name="Structure contact",
                                 marker=dict(size=4, color="#bdbdbd"), hoverinfo="skip"))
    if pairs:
        i, j, s = (np.array(v) for v in zip(*pairs))
        long_range = (j - i) >= 24
        smin, smax = float(s.min()), float(s.max())
        for mask, name, symbol in ((long_range, "Predicted (|i-j| >= 24)", "circle"),
                                   (~long_range, "Predicted (6 <= |i-j| < 24)", "diamond")):
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=j[mask] + 1 + off, y=i[mask] + 1 + off, mode="markers", name=name,
                    marker=dict(size=7, symbol=symbol, color=s[mask], colorscale="Viridis", showscale=False, cmin=smin, cmax=smax),
                    text=[f"{q[a]}{a + 1 + off} - {q[b]}{b + 1 + off}: {v:.3f}" for a, b, v in
                          zip(i[mask], j[mask], s[mask])], hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(height=640, xaxis=dict(title="Query position", range=[off, off + L + 1]),
                      yaxis=dict(title="Query position", range=[off + L + 1, off], scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=30))
    return fig


def _alignment_rows(sequences, seq_type, standalone):
    """(rows, label) from the chosen source, or (None, None) after telling the user what is missing."""
    from modules.msa import MSA_METHODS, MSAError, cached_run_msa, mafft_available

    sources = ["Upload an alignment"] if standalone or not sequences else ["Current dataset", "Upload an alignment"]
    source = st.radio("Alignment", sources, horizontal=True, key="coevo_source")
    if source == "Current dataset":
        if seq_type != "Protein":
            st.info("Coevolution needs protein sequences; switch the sequence type or upload a protein alignment.")
            return None, None
        if len(sequences) < 2:
            st.info("Load at least two sequences, or upload a deep alignment (e.g. a ColabFold .a3m).")
            return None, None
        method = st.selectbox("Aligner", [m for m in MSA_METHODS if m == "FAMSA" or mafft_available()],
                              key="coevo_aligner")
        try:
            with st.spinner("Aligning..."):
                rows = cached_run_msa(tuple((r.id, str(r.seq)) for r in sequences), "Protein", method)
        except MSAError as e:
            st.error(str(e))
            return None, None
        return rows, f"current dataset ({method})"
    upload = st.file_uploader("Alignment: ColabFold/HHblits .a3m, Stockholm .sto, or aligned FASTA",
                              type=["a3m", "sto", "stockholm", "fasta", "fa", "aln", "txt"], key="coevo_file")
    if upload is not None:
        st.session_state.coevo_alignment = (upload.name, upload.getvalue())
    stored = st.session_state.get("coevo_alignment")
    if not stored:
        st.info("Upload a deep alignment. The first sequence is the query (as in ColabFold A3M files).")
        return None, None
    rows = cached_parse(stored[1])
    if len(rows) < 2:
        st.error("The file contains fewer than two sequences.")
        return None, None
    return rows, stored[0]


def coevolution_section(sequences, seq_type, standalone=False):
    page_header("Coevolution contacts",
                "Residue pairs that co-vary across a deep alignment, predicted with plmDCA or mean-field DCA.")
    rows, label = _alignment_rows(sequences, seq_type, standalone)
    if rows is None:
        return
    c1, c2 = st.columns([2, 1])
    method_label = c1.segmented_control("Method", list(METHODS), default=list(METHODS)[0], required=True,
                                        key="coevo_method")
    method = METHODS[method_label]
    max_cols, max_seqs = LIMITS[method]
    iters = c2.slider("plmDCA iterations", 10, 100, PLM_ITERATIONS, 10, key="coevo_iters",
                      disabled=method != "plm")
    query = "".join(c for c in rows[0][1] if c not in "-.")
    L = len(query)
    if L > max_cols:
        st.warning(f"The query has {L} residues; this method handles up to {max_cols} columns on Streamlit Cloud. "
                   "Choose a region.")
        start, end = st.slider("Region (query numbering)", 1, L, (1, max_cols), key="coevo_region")
        if end - start + 1 > max_cols:
            st.error(f"Choose at most {max_cols} positions.")
            return
    else:
        start, end = 1, L
    st.caption(f"Limits on Streamlit Cloud for this method: {max_cols} columns, {max_seqs:,} sequences "
               "(larger alignments are subsampled with a fixed seed, the query always kept).")
    digest = hashlib.sha1("\n".join(f"{i}\t{s}" for i, s in rows).encode()).hexdigest()
    params = {"label": label, "n": len(rows), "digest": digest, "method": method, "start": start, "end": end, "iters": iters}
    if st.button("Compute contacts", key="coevo_run", type="primary", icon=":material/play_arrow:"):
        try:
            with st.spinner("Computing coevolution (seconds for small alignments, a few minutes at the size limits)..."):
                st.session_state.coevo_result = {"params": params, "result": cached_contacts(
                    tuple(rows), method, max_seqs, start, end, iters if method == "plm" else 0)}
        except (ValueError, MemoryError) as e:
            st.error(f"Could not compute contacts: {e}")
    stored = st.session_state.get("coevo_result")
    if not stored:
        return
    if stored["params"] != params:
        st.info("Settings changed: press Compute contacts to update.", icon=":material/refresh:")
        return
    _show_result(stored["result"])


def _show_result(result):
    from modules.structure_view import _add_structure_controls, available_structures, cached_residue_table, rank_chains

    L = len(result["query"])
    neff_l = result["neff"] / L
    cols = st.columns(5)
    cols[0].metric("Sequences used", f"{result['n_used']:,}", border=True)
    cols[1].metric("Effective sequences", f"{result['neff']:,.0f}", border=True)
    cols[2].metric("Neff / L", f"{neff_l:.1f}", border=True)
    cols[3].metric("Positions", L, border=True)
    cols[4].metric("Runtime", f"{result['seconds']:.1f} s", border=True)
    dl, dg, ds = result["dropped"]
    if dl or dg or ds:
        st.caption(f"Removed: {dl} of different length, {dg} with more than half gaps, {ds} by subsampling "
                   f"(of {result['n_input']:,} input sequences).")
    if neff_l < 1:
        st.error(f"Neff/L = {neff_l:.1f}: too few diverse sequences for reliable contacts. Use a deeper alignment "
                 "(e.g. a ColabFold .a3m).")
    elif neff_l < 5:
        st.warning(f"Neff/L = {neff_l:.1f}: contacts may be noisy.")

    distances = q_coords = sf = chain_ids = None
    with st.expander("Compare with a structure", expanded=bool(available_structures())):
        _add_structure_controls("coevo")
        structures = available_structures()
        if structures:
            c1, c2 = st.columns(2)
            name = c1.selectbox("Structure", list(structures), key="coevo_structure")
            sf = structures[name]
            table = cached_residue_table(sf.data, sf.fmt, sf.name)
            if table:
                ranked = rank_chains(result["query"], table)
                chain_id = c2.selectbox("Chain", [c for c, _ in ranked], key="coevo_chain",
                                        format_func=lambda c: f"{c} ({dict(ranked)[c]:.1%} identity)")
                distances, _, q_coords = cached_distances(sf.name, sf.data, sf.fmt, chain_id, result["query"])
                chain_ids = list(table)
    n_show = st.slider("Pairs shown", max(1, L // 5), max(2, 2 * L), L, key="coevo_n_show")
    show_plotly(contact_map_figure(result, n_show, distances), width="stretch")
    if distances is not None:
        rows = []
        for sep_label, sep in (("|i-j| >= 6", 6), ("|i-j| >= 24", 24)):
            rows.append([sep_label] + [contact_precision(result["scores"], distances, sep, max(1, L // k))
                                       for k in (5, 2, 1)])
        st.markdown("**Precision against the structure** (fraction of top pairs with CB-CB < 8 A)")
        st.dataframe(pd.DataFrame(rows, columns=["Pairs", "Top L/5", "Top L/2", "Top L"]).round(2), hide_index=True)
    off, q = result["offset"], result["query"]
    pairs = top_pairs(result["scores"], 6, n_show)
    table = pd.DataFrame([{"Rank": r + 1, "i": i + 1 + off, "Residue i": q[i], "j": j + 1 + off, "Residue j": q[j],
                           "Score": round(s, 4),
                           **({"Distance (A)": None if np.isnan(distances[i, j]) else round(float(distances[i, j]), 1)}
                              if distances is not None else {})}
                          for r, (i, j, s) in enumerate(pairs)])
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Top pairs (CSV)", table.to_csv(index=False), "coevolution_pairs.csv", "text/csv",
                       on_click="ignore", icon=":material/download:")
    if sf is not None and q_coords is not None and chain_ids:
        _show_links(sf, chain_ids, pairs, distances, q_coords, off, q, L)


def _show_links(sf, chain_ids, pairs, distances, q_coords, off, q, L):
    """Top pairs as tubes between CB atoms on the structure (green within 8 A, red otherwise)."""
    from modules.structure_view import build_view_state, render_molstar

    k = st.slider("Pairs drawn on the structure", 5, 100, min(100, max(5, L // 5)), key="coevo_n_links")
    builder = build_view_state(sf, chain_ids, color_mode="uniform")
    primitives = builder.primitives()
    drawn = skipped = 0
    for i, j, s in pairs[:k]:
        a, b = q_coords[i], q_coords[j]
        if np.isnan(a).any() or np.isnan(b).any():
            skipped += 1
            continue
        d = distances[i, j]
        color = UNKNOWN if np.isnan(d) else CLOSE if d < CONTACT_CUTOFF else FAR
        primitives.tube(start=[float(x) for x in a], end=[float(x) for x in b], radius=0.2, color=color,
                        tooltip=f"{q[i]}{i + 1 + off}-{q[j]}{j + 1 + off} (score {s:.3f}, {d:.1f} A)")
        drawn += 1
    st.caption(f"{drawn} pairs drawn (green: CB-CB < 8 A in this structure, red: farther)"
               + (f"; {skipped} not modelled in the structure." if skipped else "."))
    render_molstar(sf, builder)
