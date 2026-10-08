"""Mol* (MolViewSpec) 3D views of structures with residue-level highlights."""
from collections import OrderedDict
from dataclasses import dataclass

import molviewspec as mvs
import streamlit as st
from molviewspec import ComponentExpression

from modules.fetch import FetchError, fetch_alphafold_structure, fetch_pdb_structure, validate_ids
from modules.pdb_renumber import align_target_to_chain, get_mapping_from_alignment
from modules.structures import StructureFile, chain_residue_table, parse_structure

MOLSTAR_VERSION = "5.10.1"
CHAIN_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7",
                "#9c755f", "#bab0ac"]
# AlphaFold DB pLDDT color convention.
PLDDT_BINS = [(90.0, "#0053d6", "Very high (> 90)"), (70.0, "#65cbf3", "Confident (70–90)"),
              (50.0, "#ffdb13", "Low (50–70)"), (float("-inf"), "#ff7d45", "Very low (< 50)")]
MAX_TOOLTIPS = 300
MAX_STICKS = 300


@dataclass(frozen=True)
class ResidueRef:
    chain: str
    number: int
    icode: str = ""


@dataclass(frozen=True)
class Highlight:
    residue: ResidueRef
    color: str
    label: str = ""


def mvs_format(fmt):
    return "mmcif" if fmt == "mmCIF" else "pdb"


def archive_name(sf):
    """Fixed name inside the MVSX archive (uploaded file names may contain spaces)."""
    return "structure.cif" if sf.fmt == "mmCIF" else "structure.pdb"


def _selector(ref):
    params = {"auth_asym_id": ref.chain, "auth_seq_id": ref.number}
    if ref.icode.strip():
        params["pdbx_PDB_ins_code"] = ref.icode.strip()
    return ComponentExpression(**params)


def group_by_color(highlights):
    groups = OrderedDict()
    for h in highlights:
        groups.setdefault(h.color, []).append(h.residue)
    return groups


def plddt_color(value):
    for threshold, color, _label in PLDDT_BINS:
        if value > threshold:
            return color
    return PLDDT_BINS[-1][1]


def bfactor_highlights(table):
    """Color every residue by its CA B-factor using the pLDDT bins."""
    return [Highlight(ResidueRef(r.chain, r.number, r.icode), plddt_color(r.bfactor))
            for rows in table.values() for r in rows]


def build_view_state(sf, chain_ids, color_mode="chain", highlights=(), base_color="#d9d9d9", show_sticks=True):
    """MolViewSpec state: cartoon polymer colored by chain or uniformly, then per-residue highlights."""
    builder = mvs.create_builder()
    structure = builder.download(url=archive_name(sf)).parse(format=mvs_format(sf.fmt)).model_structure()
    cartoon = structure.component(selector="polymer").representation(type="cartoon")
    if color_mode == "chain":
        for i, chain_id in enumerate(chain_ids):
            cartoon.color(color=CHAIN_COLORS[i % len(CHAIN_COLORS)],
                          selector=ComponentExpression(auth_asym_id=chain_id))
    else:
        cartoon.color(color=base_color)
    for color, refs in group_by_color(highlights).items():
        cartoon.color(color=color, selector=[_selector(r) for r in refs])

    labelled = [h for h in highlights if h.label][:MAX_TOOLTIPS]
    if show_sticks and labelled:
        sticks = structure.component(selector=[_selector(h.residue) for h in labelled[:MAX_STICKS]])
        sticks_repr = sticks.representation(type="ball_and_stick")
        for color, refs in group_by_color(labelled[:MAX_STICKS]).items():
            sticks_repr.color(color=color, selector=[_selector(r) for r in refs])
    for h in labelled:
        structure.component(selector=_selector(h.residue)).tooltip(text=h.label)

    structure.component(selector="ligand").representation(type="ball_and_stick")
    return builder


def render_molstar(sf, builder, height=600):
    """Embed Mol* with the structure packed into an MVSX archive (no hosting needed)."""
    mvs.molstar_streamlit(builder, data={archive_name(sf): sf.data}, height=height,
                          molstar_version=MOLSTAR_VERSION)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_residue_table(data, fmt, name):
    return chain_residue_table(parse_structure(StructureFile(name=name, data=data, fmt=fmt)))


def is_predicted_model(sf):
    """True for AlphaFold-style predictions (pLDDT in the B-factor column)."""
    return sf.name.startswith("AF-") or b"_ma_qa_metric" in sf.data


def structure_viewer_section(structures):
    st.header("🧊 Structure Viewer (Mol*)")
    if not structures:
        st.warning("Load a PDB/mmCIF file or fetch a structure in the sidebar first.")
        return
    name = st.selectbox("Structure", list(structures), key="viewer_structure")
    sf = structures[name]
    table = cached_residue_table(sf.data, sf.fmt, sf.name)
    mode = st.radio("Color by", ("Chain", "pLDDT / B-factor"), horizontal=True, key="viewer_color_mode",
                    help="AlphaFold models store pLDDT in the B-factor column.")
    if mode == "Chain":
        builder = build_view_state(sf, list(table), color_mode="chain")
    else:
        builder = build_view_state(sf, list(table), color_mode="uniform", highlights=bfactor_highlights(table),
                                   show_sticks=False)
        if is_predicted_model(sf):
            st.caption(" · ".join(f"<span style='color:{c}'>■</span> {label}" for _, c, label in PLDDT_BINS),
                       unsafe_allow_html=True)
        else:
            st.caption("Colored by B-factor using pLDDT bins; for experimental structures high values mean "
                       "flexible, not confident.")
    render_molstar(sf, builder)
    st.caption(f"{len(table)} protein chain(s): " + ", ".join(f"{c} ({len(r)} residues)" for c, r in table.items()))


def map_sequence_to_structure(ref_seq, residues):
    """Align a reference sequence to a chain; map 1-based reference positions to residues."""
    chain_seq = "".join(r.aa for r in residues)
    aligned_t, aligned_c, identity = align_target_to_chain(ref_seq, chain_seq)
    mapping = get_mapping_from_alignment(aligned_t, aligned_c)   # chain index -> ref position
    return {ref_pos: residues[idx] for idx, ref_pos in mapping.items()}, identity


def rank_chains(ref_seq, table):
    scores = [(chain_id, map_sequence_to_structure(ref_seq, rows)[1]) for chain_id, rows in table.items()]
    return sorted(scores, key=lambda item: item[1], reverse=True)


def position_highlights(position_colors, pos_map):
    highlights, unmapped = [], []
    for pos, (color, label) in sorted(position_colors.items()):
        res = pos_map.get(pos)
        if res is None:
            unmapped.append(pos)
        else:
            highlights.append(Highlight(ResidueRef(res.chain, res.number, res.icode), color, label))
    return highlights, unmapped


def available_structures():
    return {**st.session_state.get("structures", {}), **st.session_state.get("extra_structures", {})}


def _add_structure_controls(key):
    c1, c2 = st.columns(2)
    upload = c1.file_uploader("Add a structure", type=["pdb", "ent", "cif", "mmcif"], key=f"{key}_upload")
    if upload is not None:
        fmt = "mmCIF" if upload.name.lower().endswith(("cif", "mmcif")) else "PDB"
        sf = StructureFile(name=upload.name, data=upload.getvalue(), fmt=fmt)
        st.session_state.extra_structures = {**st.session_state.get("extra_structures", {}), sf.name: sf}
    ident = c2.text_input("…or fetch: PDB ID or UniProt accession (AlphaFold)", key=f"{key}_fetch_id")
    if c2.button("Fetch structure", key=f"{key}_fetch_btn") and ident.strip():
        ident = ident.strip().upper()
        try:
            source = "PDB (structure)" if len(ident) == 4 else "AlphaFold DB (structure)"
            validate_ids(source, [ident])
            with st.spinner(f"Fetching {ident}..."):
                sf = fetch_pdb_structure(ident) if len(ident) == 4 else fetch_alphafold_structure(ident)
            st.session_state.extra_structures = {**st.session_state.get("extra_structures", {}), sf.name: sf}
        except FetchError as e:
            st.error(str(e))


def structure_highlight_panel(ref_seq, position_colors, key, title="🧊 Show on structure (Mol*)"):
    """Color reference positions on a structure; `position_colors` maps position -> (hex color, label)."""
    with st.expander(title):
        _add_structure_controls(key)
        structures = available_structures()
        if not structures:
            st.caption("Load, upload or fetch a structure to view these positions in 3D.")
            return
        name = st.selectbox("Structure", list(structures), key=f"{key}_structure")
        sf = structures[name]
        table = cached_residue_table(sf.data, sf.fmt, sf.name)
        if not table:
            st.warning("No protein chains in this structure.")
            return
        ranked = rank_chains(ref_seq, table)
        chain_id = st.selectbox("Chain", [c for c, _ in ranked], key=f"{key}_chain",
                                format_func=lambda c: f"{c} ({dict(ranked)[c]:.1%} identity)")
        pos_map, identity = map_sequence_to_structure(ref_seq, table[chain_id])
        highlights, unmapped = position_highlights(position_colors, pos_map)
        st.caption(f"Reference aligned to chain {chain_id}: {identity:.1%} identity; "
                   f"{len(highlights)} of {len(position_colors)} positions shown.")
        if unmapped:
            st.caption(f"Not modelled in this chain: {', '.join(map(str, unmapped[:30]))}"
                       + (" …" if len(unmapped) > 30 else ""))
        render_molstar(sf, build_view_state(sf, list(table), color_mode="uniform", highlights=highlights))
