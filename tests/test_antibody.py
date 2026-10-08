import shutil

import pytest

from modules.antibody import extract_regions, run_numbering

VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQ"
      "MNSLRAEDTAVYYCAKDRGYSSGWYFDYWGQGTLVTVSS")
VL = ("DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFA"
      "TYYCQQSYSTPLTFGQGTKVEIK")

needs_hmmer = pytest.mark.skipif(shutil.which("hmmscan") is None, reason="HMMER not installed")


def test_extract_regions_unknown_scheme_returns_empty():
    assert extract_regions([((1, ' '), 'E')], "aho", "H") == {}


@needs_hmmer
@pytest.mark.parametrize("scheme,expected_h,expected_l", [
    ("IMGT", ("GFTFSSYA", "ISGSGGST", "AKDRGYSSGWYFDY"), ("QSISSY", "AAS", "QQSYSTPLT")),
    ("Kabat", ("SYAMS", "AISGSGGSTYYADSVKG", "DRGYSSGWYFDY"), ("RASQSISSYLN", "AASSLQS", "QQSYSTPLT")),
    ("Chothia", ("GFTFSSY", "SGSGGS", "DRGYSSGWYFDY"), ("RASQSISSYLN", "AASSLQS", "QQSYSTPLT")),
])
def test_cdrs_match_reference_definitions(scheme, expected_h, expected_l):
    df = run_numbering([("vh", VH), ("vl", VL)], scheme, species=None, threshold=80)
    vh = df[df["Sequence ID"] == "vh"].iloc[0]
    vl = df[df["Sequence ID"] == "vl"].iloc[0]
    assert (vh["CDR1"], vh["CDR2"], vh["CDR3"]) == expected_h
    assert (vl["CDR1"], vl["CDR2"], vl["CDR3"]) == expected_l


from modules.antibody import public_columns, residue_regions


def test_residue_regions_labels_and_insertions():
    numbering = [((26, ' '), 'A'), ((27, ' '), 'G'), ((111, 'A'), 'Y'), ((112, ' '), '-')]
    assert residue_regions(numbering, "imgt", "H") == [("26", "FR1"), ("27", "CDR1"), ("111A", "CDR3")]


def test_public_columns_hide_internal():
    import pandas as pd
    assert public_columns(pd.DataFrame(columns=["A", "_labels", "B"])) == ["A", "B"]


@needs_hmmer
def test_germline_assignment():
    df = run_numbering([("vh", VH), ("vl", VL)], "IMGT", species=None, threshold=80)
    vh = df[df["Sequence ID"] == "vh"].iloc[0]
    vl = df[df["Sequence ID"] == "vl"].iloc[0]
    assert vh["V Gene"] == "IGHV3-23*04" and vh["V Identity (%)"] == 100.0
    assert vh["J Gene"] == "IGHJ4*01"
    assert vl["V Gene"] == "IGKV1-39*01"
    assert vh["Germline Species"] == "human"
    assert len(vh["_labels"]) == len(vh["Sequence"]) == len(vh["_regions"])


@needs_hmmer
def test_scfv_two_domains():
    df = run_numbering([("scfv", VH + "GGGGSGGGGSGGGGS" + VL)], "IMGT", species=None, threshold=80)
    assert sorted(df["Chain Type"]) == ["H", "K"]


@needs_hmmer
def test_germline_species_filter():
    df = run_numbering([("vh", VH)], "IMGT", species=["mouse"], threshold=80)
    assert df.iloc[0]["Germline Species"] in ("mouse", "")


from modules.antibody import find_liabilities


def test_liabilities_crafted_sequence():
    seq = "QNGTAMDPWC"
    labels = [str(i) for i in range(1, 11)]
    regions = ["FR1"] * 4 + ["CDR1"] * 6
    found = {(f["Liability"], f["Position"]) for f in find_liabilities(seq, labels, regions, "imgt", "H")}
    assert ("N-glycosylation", "2") in found
    assert ("Deamidation", "2") in found
    assert ("Oxidation", "6") in found          # M in CDR1
    assert ("Oxidation", "9") in found          # W in CDR1
    assert ("Fragmentation", "7") in found      # DP
    assert ("Unpaired cysteine", "10") in found


def test_oxidation_only_in_cdrs():
    found = find_liabilities("MW", ["1", "2"], ["FR1", "FR1"], "imgt", "H")
    assert not any(f["Liability"] == "Oxidation" for f in found)


def test_canonical_cysteines_not_flagged():
    found = find_liabilities("CC", ["23", "104"], ["FR1", "FR3"], "imgt", "H")
    assert not any(f["Liability"] == "Unpaired cysteine" for f in found)


def test_no_region_scheme():
    found = find_liabilities("NGTC", ["1", "2", "3", "4"], ["", "", "", ""], "aho", "H")
    assert {f["Liability"] for f in found} == {"N-glycosylation", "Deamidation"}


from modules.antibody import chain_stem, pair_domains


def _row(seq_id, idx, chain, cdr3="X", v="IGV"):
    return {"Sequence ID": seq_id, "Domain Index": idx, "Chain Type": chain, "CDR3": cdr3, "V Gene": v}


def test_chain_stem():
    assert chain_stem("mAb1_HC") == "mAb1"
    assert chain_stem("mAb1-light") == "mAb1"
    assert chain_stem("abc.VH") == "abc"
    assert chain_stem("plain") == "plain"
    assert chain_stem("seq_vel") == "seq_vel"


def test_scfv_paired_within_sequence():
    pairs, unpaired = pair_domains([_row("scfv", 1, "H"), _row("scfv", 2, "K")])
    assert pairs[0]["Format"] == "scFv / single chain" and unpaired == []


def test_separate_chains_paired_by_stem():
    pairs, _ = pair_domains([_row("mAb1_HC", 1, "H", "ARDY"), _row("mAb1_LC", 1, "K", "QQSY")])
    assert pairs[0]["Pair"] == "mAb1"
    assert (pairs[0]["Heavy CDR3"], pairs[0]["Light CDR3"]) == ("ARDY", "QQSY")


def test_unpaired_listed():
    pairs, unpaired = pair_domains([_row("lonely_H", 1, "H")])
    assert pairs == [] and unpaired == ["lonely_H (domain 1, H)"]
