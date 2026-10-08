from streamlit.testing.v1 import AppTest

FASTA_A = ">a\nMKVLAAGIVG\n>b\nMKVLAGIVGA\n>c\nMRVLSAGLVA\n"
FASTA_B = ">a\nMKVLAAGIVG\n>b\nMKVLAGIVGA\n>c\nMRVLSAGLVW\n"


def load(text):
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.text_area(key="fasta_text").input(text).run()
    return at


def run_pairwise(at):
    at.selectbox(key="analysis_select").select("Pairwise").run()
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
    assert any("Please upload" in i.value for i in at.info)


def test_switching_input_format_clears_sequences():
    at = load(FASTA_A)
    at.selectbox(key="input_format").select("FASTA").run()
    assert any("Please upload" in i.value for i in at.info)


def test_changing_msa_output_format_does_not_rerun():
    at = load(FASTA_A)
    at.selectbox(key="analysis_select").select("MSA").run()
    at.button[0].click().run()
    aligned = at.session_state["msa_aligned"]
    assert aligned
    at.selectbox(key="msa_output_format").select("clustal").run()
    assert not at.exception
    assert at.session_state["msa_aligned"] == aligned
    assert not any("Settings changed" in i.value for i in at.info)


def test_tree_built_only_on_button_and_reset_with_dataset():
    at = load(FASTA_A)
    at.selectbox(key="analysis_select").select("Phylogenetic Tree").run()
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
    at.selectbox(key="analysis_select").select("Translate DNA").run()
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
    assert at.title[0].value == "🔬 Advanced Sequence Alignment Tool"
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
    at.selectbox(key="analysis_select").select("Translate DNA").run()
    at.selectbox(key="seq_type_select").select("Protein").run()
    assert not at.exception
    assert at.selectbox(key="analysis_select").value != "Translate DNA"


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
    assert "PDB Residue Renumbering" in at.selectbox(key="analysis_select").options


def test_conservation_tab_without_protein_chains_warns():
    import pathlib
    from modules.structures import StructureFile
    data = (pathlib.Path(__file__).parent / "data" / "1bna.cif").read_bytes()
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.selectbox(key="example_choice").select("Globins (protein)").run()
    at.button(key="load_example").click().run()
    at.session_state["extra_structures"] = {"1bna.cif": StructureFile("1bna.cif", data, "mmCIF")}
    at.selectbox(key="analysis_select").select("MSA").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("No protein chains" in w.value for w in at.warning)
