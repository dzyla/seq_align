from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.properties import properties_table, protein_properties


def test_protein_properties_known_values():
    p = protein_properties("MKWVTFISLLLLFSSAYS")
    assert p["Length"] == 18
    assert round(p["MW (Da)"], 1) == 2106.5
    assert round(p["pI"], 2) == 8.34
    assert p["Ext. coeff. (reduced)"] == 6990
    assert round(p["GRAVY"], 3) == 1.289
    assert round(p["A280 0.1% (reduced)"], 3) == round(6990 / 2106.5, 3)


def test_properties_nonstandard():
    p = protein_properties("MKX*UBZW")
    assert p["Non-standard removed"] == 5
    assert p["Length"] == 3


def test_properties_table():
    df = properties_table([SeqRecord(Seq("MKW"), id="a"), SeqRecord(Seq("XX"), id="b")])
    assert list(df["Sequence ID"]) == ["a", "b"]
    assert df.loc[1, "Length"] == 0
