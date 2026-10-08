from modules.dna import codon_usage, restriction_map_figure, restriction_sites, reverse_complement


def test_reverse_complement_rna_and_case():
    assert reverse_complement("augc") == "GCAT"


def test_codon_usage():
    df = codon_usage("ATGAAAAAGTAA")
    row = df[df["Codon"] == "AAA"].iloc[0]
    assert row["AA"] == "K" and row["Count"] == 1 and row["Fraction of AA"] == 0.5


def test_restriction_sites_ecori():
    df = restriction_sites("AAGAATTCAA")
    eco = df[df["Enzyme"] == "EcoRI"].iloc[0]
    assert eco["Cuts"] == 1 and eco["Positions"] == [4]


def test_restriction_circular_origin():
    seq = "AATTCAAAAAAAAAG"      # GAATTC spans the origin when circular
    assert "EcoRI" not in set(restriction_sites(seq, circular=False)["Enzyme"])
    assert "EcoRI" in set(restriction_sites(seq, circular=True)["Enzyme"])


def test_restriction_map_figure():
    df = restriction_sites("AAGAATTCAAGGATCCAA")
    fig = restriction_map_figure(df, 18, ["EcoRI", "BamHI"])
    assert len(fig.data) == 2
