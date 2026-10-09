"""MSA tabs: publication figure (ESPript style), back-to-consensus suggestions and motifs & sites."""
import streamlit as st

from modules.engineering import (FEW_SEQUENCES, MOTIF_LIBRARY, compile_motifs, consensus_suggestions, find_motifs,
                                 site_summary)
from modules.espript import (MAX_CELLS, STYLES, FigureOptions, cell_count, figure_caption, map_ss_to_alignment,
                             region_columns, render_alignment_figure, secondary_structure)
from modules.figures import export_bytes, preview_png

CONSENSUS_COLOUR, MOTIF_COLOUR = "#d6312b", "#2a5db0"


def consensus_settings(rows, ref_id):
    """Current Consensus-tab settings (also read by the Figure tab for its star markers)."""
    ids = [i for i, _ in rows]
    target = st.session_state.get("cons_target", ref_id)
    return (target if target in ids else ref_id), st.session_state.get("cons_min_freq", 0.6)


def motif_settings():
    """Current Motifs-tab settings: ({name: pattern}, syntax) for the chosen library motifs plus a custom one."""
    chosen = st.session_state.get("motif_select", ["N-glycosylation sequon"])
    motifs = {name: MOTIF_LIBRARY[name] for name in chosen if name in MOTIF_LIBRARY}
    custom = (st.session_state.get("motif_custom") or "").strip()
    syntax = "prosite" if st.session_state.get("motif_syntax", "PROSITE pattern") == "PROSITE pattern" else "regex"
    if custom:
        motifs["Custom motif"] = custom
    return motifs, syntax


def _compiled_motifs(motifs, syntax):
    """Library motifs are always PROSITE; only the custom motif follows the chosen syntax."""
    library = {k: v for k, v in motifs.items() if k != "Custom motif"}
    compiled = compile_motifs(library, "prosite")
    if "Custom motif" in motifs:
        compiled.update(compile_motifs({"Custom motif": motifs["Custom motif"]}, syntax))
    return compiled


@st.cache_data(max_entries=8, show_spinner=False)
def cached_secondary_structure(name, data, fmt, chain_id):
    from modules.structures import StructureFile

    return secondary_structure(StructureFile(name=name, data=data, fmt=fmt), chain_id)


@st.cache_data(max_entries=8, show_spinner=False)
def figure_outputs(rows, ref_id, ss_by_column, options, markers, ss_label, columns):
    """Preview PNG, SVG and PDF of the alignment figure (cached on every input)."""
    fig = render_alignment_figure(list(rows), ref_id, ss_by_column, FigureOptions(**dict(options)),
                                  {k: set(v) for k, v in markers}, ss_label, columns)
    return {"preview": preview_png(fig), "svg": export_bytes(fig, "svg"), "pdf": export_bytes(fig, "pdf")}


def _png_600(rows, ref_id, ss_by_column, options, markers, ss_label, columns):
    fig = render_alignment_figure(list(rows), ref_id, ss_by_column, FigureOptions(**dict(options)),
                                  {k: set(v) for k, v in markers}, ss_label, columns)
    return export_bytes(fig, "png")


def _ss_controls(ref_seq):
    """Secondary-structure source for the figure: a structure (loaded, uploaded, or fetched by PDB ID / UniProt
    accession for AlphaFold) and chain, or none. Returns (chain_seq, ss, source, label, reason)."""
    from modules.structure_view import _add_structure_controls, available_structures, cached_residue_table, rank_chains

    use = st.radio("Secondary structure", ["From a structure", "None"], horizontal=True, key="fig_ss_use",
                   index=0 if available_structures() else 1)
    if use == "None":
        return "", "", "none", None, ""
    _add_structure_controls("fig_ss")
    structures = available_structures()
    if not structures:
        st.caption("Load, upload or fetch a structure to show its secondary structure.")
        return "", "", "none", None, ""
    c1, c2 = st.columns(2)
    name = c1.selectbox("Structure", list(structures), key="fig_ss_structure")
    sf = structures[name]
    table = cached_residue_table(sf.data, sf.fmt, sf.name)
    if not table:
        st.warning("No protein chains in this structure.")
        return "", "", "none", None, ""
    ranked = rank_chains(ref_seq, table)
    chain_id = c2.selectbox("Chain", [c for c, _ in ranked], key="fig_ss_chain",
                            format_func=lambda c: f"{c} ({dict(ranked)[c]:.1%} identity)")
    seq, ss, source, reason = cached_secondary_structure(sf.name, sf.data, sf.fmt, chain_id)
    return seq, ss, source, f"{sf.stem} chain {chain_id}", reason


def figure_tab(rows, ref_id, seq_type):
    """Publication alignment figure in the ESPript style."""
    ids = [i for i, _ in rows]
    st.subheader("Publication figure")
    chosen = st.multiselect("Sequences (in this order)", ids, default=[ref_id] + [i for i in ids if i != ref_id],
                            key="fig_rows")
    if not chosen:
        st.info("Choose at least one sequence.")
        return
    shown = [(i, dict(rows)[i]) for i in chosen]
    shown_ref = ref_id if ref_id in chosen else chosen[0]
    shown_ref_aligned = dict(rows)[shown_ref]
    n_res = sum(c not in "-." for c in shown_ref_aligned)
    c1, c2, c3 = st.columns(3)
    start, end = c1.slider(f"Region ({shown_ref} numbering)", 1, max(n_res, 2), (1, max(n_res, 2)),
                           key="fig_region")
    style_label = c2.segmented_control("Style", list(STYLES), default="Classic ESPript", required=True,
                                       key="fig_style")
    block = c3.slider("Columns per block", 40, 120, 60, 10, key="fig_block")
    c1, c2, c3 = st.columns(3)
    font = c1.slider("Font size (pt)", 6, 12, 8, key="fig_font")
    threshold = c2.slider("Similarity threshold", 0.5, 1.0, 0.7, 0.05, key="fig_threshold")
    truncate = c3.number_input("Truncate names at (0 = off)", 0, 60, 0, key="fig_truncate")
    c1, c2, c3, c4, c5 = st.columns(5)
    consensus = c1.toggle("Consensus line", key="fig_consensus")
    conservation = c2.toggle("Conservation bar", value=True, key="fig_conservation")
    underscores = c3.toggle("Underscores as spaces", value=True, key="fig_underscores")
    mark_cons = c4.toggle("Mark consensus suggestions", value=True, key="fig_mark_cons")
    mark_motifs = c5.toggle("Mark motif sites", value=True, key="fig_mark_motifs")

    seq, ss, source, ss_label, reason = _ss_controls("".join(c for c in shown_ref_aligned if c not in "-."))
    if reason:
        st.caption(reason)
    ss_by_column = map_ss_to_alignment(shown_ref_aligned, seq, ss) if source != "none" else None

    markers = {}
    if mark_cons:
        target, min_freq = consensus_settings(rows, ref_id)
        suggestions = consensus_suggestions(rows, target, min_freq, seq_type=seq_type)
        if not suggestions.empty:
            markers["Consensus suggestion"] = tuple(sorted(suggestions["Column"]))
    if mark_motifs:
        motifs, syntax = motif_settings()
        try:
            hits = find_motifs(rows, _compiled_motifs(motifs, syntax)) if motifs else None
        except ValueError:
            hits = None
        if hits is not None and not hits.empty:
            markers["Motif site"] = tuple(sorted(set(hits["Column"])))

    c0, c1_ = region_columns(shown_ref_aligned, start, end)
    cells = cell_count(len(shown), c0, c1_)
    if cells > MAX_CELLS:
        st.warning(f"The figure would show {cells:,} residues (limit {MAX_CELLS:,}). Narrow the region or "
                   "choose fewer sequences.")
        return
    options = FigureOptions(style=STYLES[style_label], block=block, font=float(font), threshold=float(threshold),
                            consensus=consensus, conservation=conservation, truncate=int(truncate),
                            underscores=underscores)
    args = (tuple(shown), shown_ref, ss_by_column, tuple(vars(options).items()),
            tuple((k, tuple(v)) for k, v in markers.items()), ss_label, (c0, c1_))
    with st.spinner("Drawing the figure..."):
        out = figure_outputs(*args)
    st.image(out["preview"])
    st.caption(figure_caption(options, source, ss_label))
    d1, d2, d3 = st.columns(3)
    d1.download_button("SVG", out["svg"], "alignment_figure.svg", "image/svg+xml", icon=":material/download:",
                       on_click="ignore", width="stretch", key="fig_dl_svg")
    d2.download_button("PDF", out["pdf"], "alignment_figure.pdf", "application/pdf", icon=":material/download:",
                       on_click="ignore", width="stretch", key="fig_dl_pdf")
    d3.download_button("PNG (600 dpi)", lambda: _png_600(*args), "alignment_figure.png", "image/png",
                       icon=":material/download:", on_click="ignore", width="stretch", key="fig_dl_png")


def consensus_tab(rows, ref_id, seq_type):
    """Back-to-consensus mutation suggestions for one sequence."""
    from modules.structure_view import structure_highlight_panel

    ids = [i for i, _ in rows]
    st.subheader("Back-to-consensus suggestions")
    st.caption("Positions where the chosen sequence differs from a strongly conserved column consensus: a common "
               "source of stabilising mutations. Ranked by consensus frequency × conservation.")
    c1, c2 = st.columns(2)
    target = c1.selectbox("Sequence", ids, index=ids.index(ref_id), key="cons_target")
    min_freq = c2.slider("Minimum consensus frequency", 0.4, 0.95, 0.6, 0.05, key="cons_min_freq")
    if len(rows) < FEW_SEQUENCES:
        st.warning(f"Consensus from fewer than {FEW_SEQUENCES} sequences is unreliable "
                   f"({len(rows)} sequences in this alignment).")
    table = consensus_suggestions(rows, target, min_freq, seq_type=seq_type)
    if table.empty:
        st.info("No suggestions: the sequence matches the consensus at every qualifying column.")
        return
    st.dataframe(table.drop(columns=["Column"]), hide_index=True, width="stretch")
    st.download_button("Consensus suggestions (CSV)", table.drop(columns=["Column"]).to_csv(index=False),
                       "consensus_suggestions.csv", "text/csv", on_click="ignore", icon=":material/download:")
    target_seq = "".join(c for c in dict(rows)[target] if c not in "-.")
    colors = {int(r.Position): (CONSENSUS_COLOUR, r.Suggestion) for r in table.itertuples()}
    structure_highlight_panel(target_seq, colors, key="cons_3d", title="Show suggestions on structure (Mol*)")


def motifs_tab(rows, ref_id):
    """Motif and site map across the family."""
    from modules.structure_view import structure_highlight_panel

    st.subheader("Motifs and sites")
    st.multiselect("Built-in motifs", list(MOTIF_LIBRARY), default=["N-glycosylation sequon"], key="motif_select",
                   format_func=lambda k: f"{k}  ({MOTIF_LIBRARY[k]})")
    c1, c2 = st.columns([1, 3])
    c1.radio("Custom motif syntax", ["PROSITE pattern", "Regular expression"], key="motif_syntax")
    c2.text_input("Custom motif (optional)", key="motif_custom", placeholder="e.g. C-x(2)-C  or  C..C")
    motifs, syntax = motif_settings()
    if not motifs:
        st.info("Choose a built-in motif or enter a custom one.")
        return
    try:
        hits = find_motifs(rows, _compiled_motifs(motifs, syntax))
    except ValueError as e:
        st.error(f"Invalid motif: {e}")
        return
    if hits.empty:
        st.info("No matches in any sequence.")
        return
    summary = site_summary(hits, rows, ref_id)
    st.markdown("**Sites across the family**")
    st.dataframe(summary.drop(columns=["Column"]), hide_index=True, width="stretch")
    st.markdown("**Hits per sequence**")
    st.dataframe(hits.drop(columns=["Column"]), hide_index=True, width="stretch")
    d1, d2 = st.columns(2)
    d1.download_button("Sites (CSV)", summary.drop(columns=["Column"]).to_csv(index=False), "motif_sites.csv",
                       "text/csv", on_click="ignore", icon=":material/download:")
    d2.download_button("Hits (CSV)", hits.drop(columns=["Column"]).to_csv(index=False), "motif_hits.csv",
                       "text/csv", on_click="ignore", icon=":material/download:")
    ref_seq = "".join(c for c in dict(rows)[ref_id] if c not in "-.")
    colors = {int(r[1]): (MOTIF_COLOUR, f"{r[0]} ({r[2]})")
              for r in summary[["Motif", "Reference position", "Sequences with site"]].itertuples(index=False)
              if r[1] is not None and r[1] == r[1]}
    structure_highlight_panel(ref_seq, colors, key="motif_3d", title="Show sites on structure (Mol*)")
