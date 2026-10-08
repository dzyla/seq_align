"""Mol* (MolViewSpec) 3D views of structures with residue-level highlights."""
from collections import OrderedDict
from dataclasses import dataclass

import molviewspec as mvs
import streamlit as st
from molviewspec import ComponentExpression

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
        st.caption(" · ".join(f"<span style='color:{c}'>■</span> {label}" for _, c, label in PLDDT_BINS),
                   unsafe_allow_html=True)
    render_molstar(sf, builder)
    st.caption(f"{len(table)} protein chain(s): " + ", ".join(f"{c} ({len(r)} residues)" for c, r in table.items()))
