import json
import pathlib

from modules.structure_view import (
    Highlight, ResidueRef, archive_name, bfactor_highlights, build_view_state, group_by_color, mvs_format,
    plddt_color,
)
from modules.structures import StructureFile, chain_residue_table, parse_structure

DATA = pathlib.Path(__file__).parent / "data"
PDB_SF = StructureFile(name="my model.pdb", data=(DATA / "1crn.pdb").read_bytes(), fmt="PDB")
CIF_SF = StructureFile(name="1crn.cif", data=(DATA / "1crn.cif").read_bytes(), fmt="mmCIF")


def state_json(builder):
    return json.loads(builder.get_state().dumps())


def nodes(tree, kind):
    found = []
    stack = [tree["root"]]
    while stack:
        node = stack.pop()
        if node.get("kind") == kind:
            found.append(node)
        stack.extend(reversed(node.get("children", [])))
    return found


def test_archive_name_and_format():
    assert archive_name(PDB_SF) == "structure.pdb"
    assert archive_name(CIF_SF) == "structure.cif"


def test_mmcif_format_name():
    assert mvs_format("mmCIF") == "mmcif"
    assert mvs_format("PDB") == "pdb"
    state = state_json(build_view_state(CIF_SF, ["A"]))
    assert nodes(state, "download")[0]["params"]["url"] == "structure.cif"
    assert nodes(state, "parse")[0]["params"]["format"] == "mmcif"


def test_chain_mode_colors_each_chain():
    colors = nodes(state_json(build_view_state(PDB_SF, ["A", "B"])), "color")
    assert [c["params"]["selector"]["auth_asym_id"] for c in colors] == ["A", "B"]


def test_highlights_grouped_and_tooltips():
    hl = [Highlight(ResidueRef("A", 5), "#ff0000", "K5R"), Highlight(ResidueRef("A", 9), "#ff0000", "A9W"),
          Highlight(ResidueRef("A", 12), "#0000ff")]
    assert {k: len(v) for k, v in group_by_color(hl).items()} == {"#ff0000": 2, "#0000ff": 1}
    state = state_json(build_view_state(PDB_SF, ["A"], color_mode="uniform", highlights=hl))
    tooltip_texts = sorted(n["params"]["text"] for n in nodes(state, "tooltip"))
    assert tooltip_texts == ["A9W", "K5R"]


def test_insertion_code_in_selector():
    hl = [Highlight(ResidueRef("H", 52, "A"), "#ff0000")]
    state = state_json(build_view_state(PDB_SF, ["H"], color_mode="uniform", highlights=hl))
    selectors = [n["params"].get("selector") for n in nodes(state, "color")]
    assert [{"auth_asym_id": "H", "auth_seq_id": 52, "pdbx_PDB_ins_code": "A"}] in selectors


def test_plddt_bins():
    assert plddt_color(95) == "#0053d6"
    assert plddt_color(80) == "#65cbf3"
    assert plddt_color(60) == "#ffdb13"
    assert plddt_color(10) == "#ff7d45"


def test_bfactor_highlights_cover_all_residues():
    table = chain_residue_table(parse_structure(PDB_SF))
    assert len(bfactor_highlights(table)) == 46


from modules.structure_view import map_sequence_to_structure, position_highlights, rank_chains

CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def test_map_sequence_to_structure_with_mutation():
    table = chain_residue_table(parse_structure(PDB_SF))
    mutant = CRAMBIN[:8] + "W" + CRAMBIN[9:]
    pos_map, identity = map_sequence_to_structure(mutant, table["A"])
    assert pos_map[9].number == 9 and pos_map[9].aa == "A"
    assert identity > 0.95


def test_rank_chains():
    table = chain_residue_table(parse_structure(PDB_SF))
    assert rank_chains(CRAMBIN, table)[0][0] == "A"


def test_unmapped_positions_reported():
    table = chain_residue_table(parse_structure(PDB_SF))
    extended = "MKKLLL" + CRAMBIN          # 6 extra N-terminal residues absent from the structure
    pos_map, _ = map_sequence_to_structure(extended, table["A"])
    highlights, unmapped = position_highlights({2: ("#ff0000", "K2R"), 15: ("#ff0000", "A15W")}, pos_map)
    assert unmapped == [2]
    assert highlights[0].residue.number == 9 and highlights[0].label == "A15W"


def test_is_predicted_model():
    from modules.structure_view import is_predicted_model
    assert not is_predicted_model(CIF_SF)
    assert is_predicted_model(StructureFile(name="AF-P69905-F1-model_v6.cif", data=b"x", fmt="mmCIF"))
    assert is_predicted_model(StructureFile(name="m.cif", data=b"loop_\n_ma_qa_metric.id\n", fmt="mmCIF"))
