import pytest
import math
import pathlib
from io import StringIO

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from modules.conservation import (
    column_conservation, conservation_grade, reference_conservation, write_bfactors,
)
from modules.structures import StructureFile

DATA = pathlib.Path(__file__).parent / "data"


def test_column_conservation_protein():
    scores = column_conservation(["MKV", "MKL", "MRL"])
    assert scores[0] == 1.0
    h = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
    assert math.isclose(scores[1], 1 - h / math.log2(20))


def test_gappy_columns_are_nan():
    scores = column_conservation(["M-", "M-", "MK"])
    assert np.isnan(scores[1])


def test_dna_normalization():
    scores = column_conservation(["AC", "AG"], seq_type="DNA")
    assert math.isclose(scores[1], 1 - 1 / math.log2(4))


def test_reference_conservation_skips_reference_gaps():
    result = reference_conservation([("ref", "M-KV"), ("b", "MAKV"), ("c", "MAKL")], "ref")
    assert sorted(result) == [1, 2, 3]
    assert result[1] == 1.0


def test_conservation_grade():
    assert conservation_grade(0.0) == 1
    assert conservation_grade(1.0) == 9
    assert conservation_grade(0.5) == 5


def test_write_bfactors_pdb_roundtrip():
    sf = StructureFile(name="1crn.pdb", data=(DATA / "1crn.pdb").read_bytes(), fmt="PDB")
    text = write_bfactors(sf, {("A", 1, ""): 0.75})
    assert text.startswith("CRYST1")
    structure = PDBParser(QUIET=True).get_structure("x", StringIO(text))
    chain = structure[0]["A"]
    assert all(a.get_bfactor() == 0.75 for a in chain[1])
    assert all(a.get_bfactor() == 0.0 for a in chain[2])


def test_write_bfactors_mmcif():
    sf = StructureFile(name="1crn.cif", data=(DATA / "1crn.cif").read_bytes(), fmt="mmCIF")
    text = write_bfactors(sf, {("A", 2, ""): 0.5})
    structure = MMCIFParser(QUIET=True).get_structure("x", StringIO(text))
    assert all(a.get_bfactor() == 0.5 for a in structure[0]["A"][2])


from modules.conservation import identity_matrix_frame, logo_figure, logo_matrix  # noqa: E402


def test_logo_matrix_window_and_argmax():
    m = logo_matrix(["MKV-", "MKL-", "MRLA"])
    assert m.shape[0] == 4
    assert m.idxmax(axis=1).tolist() == ["M", "K", "L", "A"]
    window = logo_matrix(["MKV-", "MKL-", "MRLA"], start=2, end=3)
    assert window.index.tolist() == [2, 3]


def test_logo_figure_closes_cleanly():
    import matplotlib.pyplot as plt
    fig = logo_figure(logo_matrix(["ACGT", "ACGA"], seq_type="DNA"), "DNA")
    assert fig.axes
    plt.close(fig)


def test_identity_matrix_frame():
    df = identity_matrix_frame([("a", "MKV"), ("b", "MKL")])
    assert df.loc["a", "b"] == round(200 / 3, 1)
    assert df.loc["a", "a"] == 100.0


def test_logo_full_conservation_protein_is_log2_20():
    m = logo_matrix(["M", "M", "M", "M"])
    assert m.shape == (1, 20)
    assert m.sum(axis=1).iloc[0] == pytest.approx(math.log2(20), abs=1e-6)


def test_logo_full_conservation_dna_is_2_bits():
    m = logo_matrix(["A", "A", "A"], seq_type="DNA")
    assert m.shape == (1, 4)
    assert m.sum(axis=1).iloc[0] == pytest.approx(2.0, abs=1e-6)


def test_logo_all_gap_column_is_zero_not_nan():
    m = logo_matrix(["-", "-", "-"])
    assert not m.isna().any().any()
    assert m.sum(axis=1).iloc[0] == 0
