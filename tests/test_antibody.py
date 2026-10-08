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
