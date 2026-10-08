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
