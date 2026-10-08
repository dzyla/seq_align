from modules.variants import call_variants, variant_matrix, variants_long


def labels(ref, qry, **kw):
    return [v.label for v in call_variants(ref, qry, **kw)]


def test_substitution_insertion_deletion():
    assert labels("MKV-LAAG", "MRVWL--G") == ["K2R", "V3_L4insW", "A5_A6del"]


def test_single_deletion():
    assert labels("MKVL", "MK-L") == ["V3del"]


def test_terminal_extensions():
    assert labels("--MKV", "AAMKV") == ["N-terminal +AA"]
    assert labels("MKV--", "MKVAA") == ["C-terminal +AA"]


def test_reference_offset_and_rotation():
    assert labels("MKV", "MRV", ref_start=10) == ["K12R"]
    assert labels("CDEAK", "CDEWK", ref_rotation=2, ref_length=5) == ["A1W"]


def test_double_gap_columns_ignored():
    assert labels("MK--V", "MK--L") == ["V3L"]


def test_variant_matrix_and_long():
    pairs = [("ref", "MKV-L"), ("a", "MRVWL"), ("b", "MKV-L")]
    matrix = variant_matrix(pairs, "ref")
    assert list(matrix.columns) == [2, 3]
    assert matrix.loc["a", 2] == "K2R" and matrix.loc["a", 3] == "V3_L4insW"
    assert matrix.loc["b", 2] == ""
    long = variants_long(pairs, "ref")
    assert list(long["Label"]) == ["K2R", "V3_L4insW"]
