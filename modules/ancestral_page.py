"""Ancestral Reconstruction page: IQ-TREE marginal ancestral states, indels by parsimony, uncertainty and exports."""
import copy
import hashlib
import html
import io
import zipfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from modules.ancestral import (INSTALL_HINT, MAX_SEQUENCES, MIN_SEQUENCES, MODELS, ASRError, altall_sequence,
                               ancestor_options, ancestor_sequence, compare_to_extant, find_iqtree, mrca_node,
                               reconstruct_gaps, run_iqtree, uncertain_sites)
from modules.ui import page_header, show_plotly

CERTAIN, UNSURE, LOW = "#ffffff", "#fdd8a8", "#f59e42"
DIFFERS = "#2a5db0"


@st.cache_data(max_entries=4, show_spinner=False)
def cached_run_iqtree(rows, seq_type, model, bootstrap):
    return run_iqtree(list(rows), seq_type, model, bootstrap)


def _shade(p, threshold):
    return CERTAIN if p >= 0.9 else UNSURE if p >= threshold else LOW


def sequence_html(seq, posteriors, threshold, width=60):
    """Monospace ancestor with each residue shaded by its posterior (white >= 0.9, light < 0.9, orange < threshold)."""
    lines = []
    for start in range(0, len(seq), width):
        cells = "".join(f'<span style="background:{_shade(p, threshold)}" title="{start + k + 1}: {c} ({p:.2f})">'
                        f"{html.escape(c)}</span>"
                        for k, (c, p) in enumerate(zip(seq[start:start + width], posteriors[start:start + width])))
        lines.append(f'<span style="color:#888">{start + 1:>5} </span>{cells}')
    return ('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;line-height:1.6;color:#1c1b19;'
            'background:#fff;padding:8px;border-radius:6px">' + "<br>".join(lines) + "</div>")


def posterior_figure(posteriors, threshold):
    x = np.arange(1, len(posteriors) + 1)
    fig = go.Figure(go.Scatter(x=x, y=posteriors, mode="lines", line=dict(width=1.5),
                               hovertemplate="Position %{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#f59e42")
    fig.update_layout(height=240, xaxis_title="Ancestor position", yaxis=dict(title="Posterior", range=[0, 1.02]),
                      margin=dict(t=20, b=40))
    return fig


def ancestral_section(sequences, seq_type):
    from modules.msa import MSA_METHODS, MSAError, cached_run_msa, mafft_available
    from modules.phylogeny import TreeError
    from modules.tree_layout import prepare_tree

    page_header("Ancestral reconstruction",
                "Maximum-likelihood ancestors with a posterior for every residue (IQ-TREE), indels by parsimony.")
    found = find_iqtree()
    if found is None:
        st.error(INSTALL_HINT)
        return
    if not MIN_SEQUENCES <= len(sequences) <= MAX_SEQUENCES:
        st.info(f"Ancestral reconstruction needs {MIN_SEQUENCES}-{MAX_SEQUENCES} sequences "
                f"({len(sequences)} loaded).")
        return
    c1, c2, c3 = st.columns(3)
    aligner = c1.selectbox("Aligner", [m for m in MSA_METHODS if m == "FAMSA" or mafft_available()],
                           key="asr_aligner")
    model_label = c2.selectbox("Model", list(MODELS[seq_type]), key="asr_model")
    bootstrap = c3.toggle("Ultrafast bootstrap (1000)", key="asr_bootstrap")
    digest = hashlib.sha1("\n".join(f"{r.id}\t{r.seq}" for r in sequences).encode()).hexdigest()
    params = {"n": len(sequences), "digest": digest, "ids": tuple(r.id for r in sequences), "seq_type": seq_type, "aligner": aligner,
              "model": MODELS[seq_type][model_label], "bootstrap": bootstrap}
    if st.button("Reconstruct ancestors", key="asr_run", type="primary", icon=":material/play_arrow:"):
        try:
            with st.spinner("Aligning and running IQ-TREE (seconds to minutes)..."):
                rows = cached_run_msa(tuple((r.id, str(r.seq)) for r in sequences), seq_type, aligner)
                result = cached_run_iqtree(tuple(rows), seq_type, params["model"], 1000 if bootstrap else 0)
            st.session_state.asr_result = {"params": params, "rows": rows, "result": result}
        except (MSAError, ASRError) as e:
            st.error(str(e))
    stored = st.session_state.get("asr_result")
    if not stored:
        st.caption(f"Engine: {found[1]}")
        return
    if stored["params"] != params:
        st.info("Settings changed: press Reconstruct ancestors to update.", icon=":material/refresh:")
        return
    rows, result = stored["rows"], stored["result"]
    lnl = f"{result.loglik:.2f}" if result.loglik is not None else "n/a"
    st.caption(f"{result.version}; model {result.model}; log-likelihood {lnl}.")

    leaves = [r[0] for r in rows]
    c1, c2 = st.columns([1, 2])
    rooting = c1.radio("Root", ["Midpoint", "Outgroup"], key="asr_rooting")
    outgroup = c2.multiselect("Outgroup sequences", leaves, key="asr_outgroup") if rooting == "Outgroup" else []
    try:
        tree = prepare_tree(result.tree, "outgroup" if outgroup else "midpoint", outgroup)
    except TreeError as e:
        st.warning(f"{e} Using midpoint rooting instead.")
        tree = prepare_tree(result.tree, "midpoint")
    gaps = reconstruct_gaps(tree, rows)

    pick = st.radio("Choose an ancestor", ["Ancestor of chosen sequences", "From the list of nodes"],
                    horizontal=True, key="asr_pick")
    try:
        if pick == "Ancestor of chosen sequences":
            chosen = st.multiselect("Sequences", leaves, default=leaves[:2], key="asr_leaves")
            node = mrca_node(tree, chosen, result.states)
        else:
            options = ancestor_options(tree, result.states)
            labels = {o[2]: o[0] for o in options}
            node = labels[st.selectbox("Node", list(labels), key="asr_node")]
            st.caption("Node names come from IQ-TREE's unrooted tree; the sequences under each node follow the "
                       "rooting chosen above.")
    except ASRError as e:
        st.warning(str(e))
        return
    c1, c2 = st.columns(2)
    threshold = c1.slider("Uncertain below posterior", 0.5, 0.95, 0.8, 0.05, key="asr_threshold")
    min_alt = c2.slider("Show alternatives with posterior at least", 0.05, 0.5, 0.2, 0.05, key="asr_min_alt")
    seq, pp, anc_cols = ancestor_sequence(result, gaps, node)
    cols = st.columns(4)
    cols[0].metric("Ancestor", node, border=True)
    cols[1].metric("Length", len(seq), border=True)
    cols[2].metric("Mean posterior", f"{pp.mean():.2f}" if len(pp) else "n/a", border=True)
    cols[3].metric("Sites >= 0.9", f"{(pp >= 0.9).mean():.0%}" if len(pp) else "n/a", border=True)
    st.html(sequence_html(seq, pp, threshold))
    show_plotly(posterior_figure(pp, threshold), width="stretch")

    unsure = uncertain_sites(result, gaps, node, threshold, min_alt)
    st.markdown(f"**Uncertain sites** ({len(unsure)})")
    if unsure.empty:
        st.caption("Every site is above the threshold.")
    else:
        st.dataframe(unsure, hide_index=True, width="stretch")
    alt = altall_sequence(result, gaps, node, threshold, min_alt)
    header = f"{node} model={result.model} lnL={lnl}"
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("ML ancestor (FASTA)", f">{node} ML ancestor; {header}\n{seq}\n", f"{node}_ML.fasta",
                       "text/plain", on_click="ignore", icon=":material/download:")
    d2.download_button("AltAll (FASTA)", f">{node} AltAll; {header}\n{alt}\n", f"{node}_AltAll.fasta", "text/plain",
                       on_click="ignore", icon=":material/download:",
                       help="Every uncertain site switched to its best alternative: the usual robustness check.")
    d3.download_button("Uncertain sites (CSV)", unsure.to_csv(index=False), f"{node}_uncertain_sites.csv",
                       "text/csv", on_click="ignore", icon=":material/download:")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in result.files.items():
            zf.writestr(name, data)
    d4.download_button("IQ-TREE files (ZIP)", buffer.getvalue(), "iqtree_asr.zip", "application/zip",
                       on_click="ignore", icon=":material/download:")

    st.markdown("**Compare with a present-day sequence**")
    extant = st.selectbox("Sequence", leaves, key="asr_extant")
    subs, ins, dels, identity = compare_to_extant(result, gaps, node, rows, extant)
    st.caption(f"{len(subs)} substitutions, {ins} insertions and {dels} deletions in {extant} relative to {node}; "
               f"identity {identity:.1%} over shared positions.")
    if subs:
        st.code(", ".join(label for label, _ in subs), language=None)

    from modules.structure_view import structure_highlight_panel

    position = {int(c): k + 1 for k, c in enumerate(anc_cols)}       # alignment column -> ancestor position
    colors = {position[c]: (DIFFERS, f"{label} vs {extant}") for label, c in subs if c in position}
    colors.update({k + 1: (LOW if p < threshold else UNSURE, f"{seq[k]}{k + 1} ({p:.2f})")
                   for k, p in enumerate(pp) if p < 0.9})
    structure_highlight_panel(seq, colors, key="asr_3d",
                              title="Show confidence and differences on structure (Mol*)")
    st.caption(f"Structure colours: orange = posterior below {threshold:.2f}, light orange = below 0.9, "
               f"blue = differs from {extant}.")

    from modules.phylogeny import phylogenetic_tree_section

    display = copy.deepcopy(tree)
    for clade in display.get_nonterminals():
        clade.name = None          # IQ-TREE node names would otherwise merge with supports in the Newick export
    with st.expander("Tree"):
        phylogenetic_tree_section(display, key_prefix="asr_tree")
