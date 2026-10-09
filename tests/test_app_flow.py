from streamlit.testing.v1 import AppTest

FASTA_A = ">a\nMKVLAAGIVG\n>b\nMKVLAGIVGA\n>c\nMRVLSAGLVA\n"
FASTA_B = ">a\nMKVLAAGIVG\n>b\nMKVLAGIVGA\n>c\nMRVLSAGLVW\n"


def load(text):
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.text_area(key="fasta_text").input(text).run()
    return at


def run_pairwise(at):
    at.radio(key="analysis_select").set_value("Pairwise").run()
    at.button[0].click().run()
    assert not at.exception
    return at


def test_changed_dataset_clears_results():
    at = run_pairwise(load(FASTA_A))
    assert at.session_state["pairwise_result"] is not None
    at.text_area(key="fasta_text").input(FASTA_B).run()
    assert at.session_state["pairwise_result"] is None


def test_parse_error_does_not_show_previous_dataset():
    at = load(FASTA_A)
    at.text_area(key="fasta_text").input("not fasta").run()
    assert len(at.sidebar.error) == 1
    assert any("Load sequences" in i.value for i in at.info)


def test_switching_input_format_clears_sequences():
    at = load(FASTA_A)
    at.selectbox(key="input_format").select("FASTA").run()
    assert any("Load sequences" in i.value for i in at.info)


def test_changing_msa_output_format_does_not_rerun():
    at = load(FASTA_A)
    at.radio(key="analysis_select").set_value("MSA").run()
    at.button[0].click().run()
    aligned = at.session_state["msa_aligned"]
    assert aligned
    at.selectbox(key="msa_output_format").select("clustal").run()
    assert not at.exception
    assert at.session_state["msa_aligned"] == aligned
    assert not any("Settings changed" in i.value for i in at.info)


def test_tree_built_only_on_button_and_reset_with_dataset():
    at = load(FASTA_A)
    at.radio(key="analysis_select").set_value("Phylogenetic Tree").run()
    assert at.session_state["tree_result"] is None
    at.button(key="build_tree").click().run()
    assert not at.exception
    assert at.session_state["tree_result"] is not None
    at.text_area(key="fasta_text").input(FASTA_B).run()
    assert at.session_state["tree_result"] is None


DNA = ">g\n" + "ATG" + "GCC" * 40 + "TAA" + "\n>h\nATGAAACCCGGGTTTTAA\n"


def test_orf_results_survive_rerun():
    at = load(DNA)
    at.selectbox(key="seq_type_select").select("DNA").run()
    at.radio(key="analysis_select").set_value("Translate DNA").run()
    before = len(at.dataframe)                 # the dataset overview table
    at.button(key="find_orfs").click().run()
    assert len(at.dataframe) == before + 1
    at.run()                                   # any rerun, e.g. from another widget
    assert len(at.dataframe) == before + 1


def test_pairwise_shows_stale_notice_after_param_change():
    at = run_pairwise(load(FASTA_A))
    at.selectbox(key="pairwise_mode").select("Local").run()
    assert any("Settings changed" in i.value for i in at.info)


def test_title_has_no_ability_wording():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    assert at.title[0].value == "seq_align"
    assert all("Abilit" not in m.value for m in at.markdown)


def test_dna_detected_automatically():
    at = load(">x\nATGCATGCATGCATGC\n>y\nATGCATGAATGCATGC\n")
    assert at.selectbox(key="seq_type_select").value == "DNA"


def test_load_example_button():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.selectbox(key="example_choice").select("Globins (protein)").run()
    at.button(key="load_example").click().run()
    assert not at.exception
    assert len(at.session_state["input_cache"][0]) == 6


def test_analysis_falls_back_when_unavailable():
    at = load(">x\nATGCATGCATGCATGC\n>y\nATGCATGAATGCATGC\n")
    at.radio(key="analysis_select").set_value("Translate DNA").run()
    at.selectbox(key="seq_type_select").select("Protein").run()
    assert not at.exception
    assert at.radio(key="analysis_select").value != "Translate DNA"


def test_fetch_pdb_offers_structure_tools(monkeypatch):
    import pathlib
    import modules.fetch as fetch

    data = (pathlib.Path(__file__).parent / "data" / "1crn.cif").read_bytes()

    class R:
        status_code = 200
        content = data
        text = data.decode()

    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: R())
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.selectbox(key="input_format").select("Fetch by ID").run()
    at.selectbox(key="fetch_source").select("PDB (structure)").run()
    at.text_input(key="fetch_ids").input("1CRN").run()
    at.button(key="fetch_button").click().run()
    assert not at.exception
    assert "PDB Residue Renumbering" in at.radio(key="analysis_select").options


def test_conservation_tab_without_protein_chains_warns():
    import pathlib
    from modules.structures import StructureFile
    data = (pathlib.Path(__file__).parent / "data" / "1bna.cif").read_bytes()
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.selectbox(key="example_choice").select("Globins (protein)").run()
    at.button(key="load_example").click().run()
    at.session_state["extra_structures"] = {"1bna.cif": StructureFile("1bna.cif", data, "mmCIF")}
    at.radio(key="analysis_select").set_value("MSA").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("No protein chains" in w.value for w in at.warning)


import shutil
import pytest


@pytest.mark.skipif(shutil.which("hmmscan") is None, reason="HMMER not installed")
def test_antibody_tabs_render():
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    at.selectbox(key="example_choice").select("Antibody VH/VL (protein)").run()
    at.button(key="load_example").click().run()
    at.radio(key="analysis_select").set_value("Antibody Prediction").run()
    at.button(key="predict_domains").click().run()
    assert not at.exception
    assert [t.label for t in at.tabs][:4] == ["Domains", "Germlines", "Liabilities", "Pairs"]
    frames = [d.value for d in at.dataframe]
    assert any("V Gene" in f.columns and "IGHV3-23*04" in set(f["V Gene"]) for f in frames)
    assert any("Pair" in f.columns and "example" in set(f["Pair"]) for f in frames)


def test_msa_logo_and_identity_export_render():
    at = load(FASTA_A)
    at.radio(key="analysis_select").set_value("MSA").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["logo_window"] == (1, 11)
    assert any(d.label == "Percent identity matrix (CSV)" for d in at.get("download_button"))


def test_msa_logo_guard_for_single_column(monkeypatch):
    # pyFAMSA segfaults on single-residue input, so stub the aligner to produce a 1-column MSA
    import modules.msa as msa_module
    monkeypatch.setattr(msa_module, "cached_run_msa", lambda pairs, seq_type, method="FAMSA": [("a", "M"), ("b", "K")])
    at = load(">a\nM\n>b\nK\n")
    at.radio(key="analysis_select").set_value("MSA").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Logo needs at least 2 columns" in c.value for c in at.caption)
    assert not any(s.key == "logo_window" for s in at.slider)


def _load_globins_msa():
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    at.selectbox(key="example_choice").select("Globins (protein)").run()
    at.button(key="load_example").click().run()
    at.radio(key="analysis_select").set_value("MSA").run()
    at.button[0].click().run()
    assert not at.exception
    return at


def test_msa_tree_goes_stale_when_redundancy_changes():
    at = _load_globins_msa()
    next(b for b in at.button if "Phylogenetic" in b.label).click().run()
    assert not at.exception
    assert at.session_state["msa_tree"] is not None
    assert not any("Settings changed" in i.value for i in at.info)
    at.slider(key="msa_redundancy").set_value(0.5).run()
    assert not at.exception
    assert any("Settings changed" in i.value for i in at.info)


def test_msa_all_columns_trimmed_shows_one_warning(monkeypatch):
    import modules.msa as msa_module
    monkeypatch.setattr(msa_module, "cached_run_msa", lambda pairs, seq_type, method="FAMSA": [
        ("a", "MK-"), ("b", "-LA"), ("c", "M--")])
    at = load(">a\nMKV\n>b\nLAV\n>c\nMLV\n")
    at.radio(key="analysis_select").set_value("MSA").run()
    at.button[0].click().run()
    assert not at.exception
    at.slider(key="msa_trim").set_value(0.0).run()
    assert not at.exception
    msgs = [w.value for w in at.warning if "All alignment columns were removed" in w.value]
    assert len(msgs) == 1


def test_similarity_matrix_colour_range_controls():
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    at.selectbox(key="example_choice").select("Globins (protein)").run()
    at.button(key="load_example").click().run()
    at.radio(key="analysis_select").set_value("Sequence Clustering").run()
    next(s for s in at.selectbox if s.label == "Similarity Method").set_value("Accurate (Pairwise)").run()
    next(b for b in at.button if "Calculate Pairwise Similarity" in b.label).click().run()
    assert not at.exception
    assert any("off-diagonal range" in c.value for c in at.caption)
    next(w for w in at.get("button_group") if w.key == "sim_range").set_value("Fixed 0–1").run()
    assert not at.exception
    assert any("0.000–1.000" in c.value for c in at.caption)


def test_tree_page_shows_figure_downloads():
    at = load(FASTA_A)
    at.radio(key="analysis_select").set_value("Phylogenetic Tree").run()
    at.button(key="build_tree").click().run()
    assert not at.exception
    assert "SVG" in [d.label for d in at.get("download_button")]


def test_shell_has_sidebar_groups_and_appearance():
    at = load(FASTA_A)
    assert not at.exception
    labels = [c.value for c in at.sidebar.caption]
    assert "**DATA**" in labels and "**ANALYSIS**" in labels
    assert at.session_state["appearance_palette"] == "Lab teal"
    assert [m.label for m in at.metric][:4] == ["Sequences", "Type", "Length range", "Median length"]


def test_uniprot_batch_preview_and_load(monkeypatch):
    import pandas as pd
    import modules.uniprot_batch as ub
    frame = pd.DataFrame({"Accession": ["P69905", "P68871"], "Entry name": ["HBA_HUMAN", "HBB_HUMAN"],
                          "Protein": ["Hb alpha", "Hb beta"], "Gene": ["HBA1", "HBB"],
                          "Organism": ["Human", "Human"], "Length": [142, 147]})
    monkeypatch.setattr(ub, "uniprot_batch_tsv", lambda acc, names: (frame, ["Q00000"]))
    monkeypatch.setattr(ub, "fetch_uniprot_fasta", lambda accs: "".join(
        f">sp|{a}|X_HUMAN x\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK\n" for a in accs))
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.selectbox(key="input_format").select("Fetch by ID").run()
    at.selectbox(key="fetch_source").select("UniProt batch").run()
    at.text_area(key="uniprot_ids").input("P69905 P68871 Q00000").run()
    at.button(key="uniprot_search").click().run()
    assert not at.exception
    assert any("Q00000" in w.value for w in at.warning)
    at.button(key="uniprot_load").click().run()
    assert not at.exception
    assert at.metric[0].value == "2"


def test_gene_preview_unticks_synonym_rows(monkeypatch):
    import pandas as pd
    import modules.uniprot_batch as ub
    frame = pd.DataFrame({"Accession": ["P02185", "P01900"], "Entry name": ["MYG_MOUSE", "DMB_MOUSE"],
                          "Protein": ["Myoglobin", "MHC class II"], "Gene": ["Mb", "H2-DMb1"],
                          "Organism": ["Mouse", "Mouse"], "Length": [154, 260], "Matched gene": ["MB", "MB"],
                          "Match": ["primary", "synonym"]})
    monkeypatch.setattr(ub, "uniprot_gene_tsv", lambda genes, taxon, reviewed, isoforms: (frame, []))
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.selectbox(key="input_format").select("Fetch by ID").run()
    at.selectbox(key="fetch_source").select("UniProt batch").run()
    at.radio(key="uniprot_mode").set_value("Genes + organism").run()
    at.text_area(key="uniprot_genes").input("MB").run()
    at.button(key="uniprot_search").click().run()
    assert not at.exception
    assert any("start unticked" in c.value for c in at.caption)


def test_safe_fasta_turns_failure_into_comment(monkeypatch):
    import modules.uniprot_batch as ub
    from modules.fetch import FetchError

    def boom(accessions):
        raise FetchError("UniProt returned no entry for: Q00000")
    monkeypatch.setattr(ub, "fetch_uniprot_fasta", boom)
    assert ub.safe_fasta(["Q00000"]) == "# UniProt download failed: UniProt returned no entry for: Q00000\n"


def test_msa_figure_consensus_and_motif_tabs():
    at = _load_globins_msa()
    seg = lambda key: next(w for w in at.get("button_group") if w.key == key)  # noqa: E731
    for style in ("Classic ESPript", "Lab notebook", "Greyscale"):
        seg("fig_style").set_value(style).run()
        assert not at.exception, (style, at.exception)
    labels = [d.label for d in at.get("download_button")]
    assert {"SVG", "PDF", "PNG (600 dpi)"} <= set(labels)
    assert any("Identical columns" in c.value for c in at.caption)
    assert "Consensus suggestions (CSV)" in labels and "Sites (CSV)" in labels
    at.text_input(key="motif_custom").input("C-x(2)-{P").run()
    assert not at.exception and any("Invalid motif" in e.value for e in at.error)


def test_msa_figure_with_structure_secondary_structure():
    import pathlib
    from modules.structures import StructureFile
    at = _load_globins_msa()
    data = (pathlib.Path(__file__).parent / "data" / "1crn.pdb").read_bytes()
    at.session_state["extra_structures"] = {"1crn.pdb": StructureFile(name="1crn.pdb", data=data, fmt="PDB")}
    at.run()
    at.radio(key="fig_ss_use").set_value("From a structure").run()
    assert not at.exception
    assert any("1crn chain A" in c.value for c in at.caption)


def test_msa_custom_regex_with_flags_and_backreference():
    at = _load_globins_msa()
    at.radio(key="motif_syntax").set_value("Regular expression").run()
    for pattern in ("(?i)n[^p][st]", r"(a)\1", "C(("):
        at.text_input(key="motif_custom").input(pattern).run()
        assert not at.exception, (pattern, at.exception)
    assert any("Invalid motif" in e.value for e in at.error)        # only the last one is invalid


def test_dna_msa_motifs_tab_points_to_custom_patterns():
    at = load(">a\nATGCATGCAATT\n>b\nATGCATGGAATT\n>c\nATGAATGCAATA\n")
    at.radio(key="analysis_select").set_value("MSA").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("built-in motifs are protein motifs" in c.value for c in at.caption)
    assert any("DNA: identity only" in c.value for c in at.caption)


def _coupled_a3m(n=300, L=40, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    msa = rng.integers(0, 20, size=(n, L))
    pick = np.array([[0, 5], [3, 9], [11, 2], [17, 14]])[rng.integers(0, 4, size=n)]
    msa[:, 5], msa[:, 30] = pick[:, 0], pick[:, 1]
    return "".join(f">s{k}\n" + "".join(aa[x] for x in row) + "\n" for k, row in enumerate(msa)).encode()


def test_coevolution_page_with_uploaded_alignment():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    at.button(key="coevo_standalone_btn").click().run()
    at.session_state["coevo_alignment"] = ("synthetic.a3m", _coupled_a3m())
    at.run()
    for method in ("plmDCA (accurate)", "Mean-field DCA (fast)"):
        next(w for w in at.get("button_group") if w.key == "coevo_method").set_value(method).run()
        at.button(key="coevo_run").click().run()
        assert not at.exception, (method, at.exception)
        assert any(d.label == "Top pairs (CSV)" for d in at.get("download_button"))
        assert any(m.label == "Neff / L" for m in at.metric)


def test_coevolution_long_query_needs_region_and_shallow_alignment_warns():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    at.button(key="coevo_standalone_btn").click().run()
    at.session_state["coevo_alignment"] = ("long.a3m", _coupled_a3m(n=40, L=250))
    at.run()
    assert any("Choose a region" in w.value for w in at.warning)          # 250 > plmDCA's 200-column limit
    at.button(key="coevo_run").click().run()
    assert not at.exception
    assert any("too few diverse sequences" in e.value for e in at.error)   # 40 random sequences: Neff/L < 1


def test_coevolution_result_invalidated_when_alignment_changes():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    at.button(key="coevo_standalone_btn").click().run()
    at.session_state["coevo_alignment"] = ("synthetic.a3m", _coupled_a3m(seed=0))
    at.run()
    at.button(key="coevo_run").click().run()
    assert any(m.label == "Neff / L" for m in at.metric)
    at.session_state["coevo_alignment"] = ("synthetic.a3m", _coupled_a3m(seed=1))
    at.run()
    assert any("Settings changed" in i.value for i in at.info)
    assert not any(m.label == "Neff / L" for m in at.metric)
