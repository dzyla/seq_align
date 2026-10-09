"""Batch UniProt download: look up many accessions, a search query, or gene names in one organism, preview the
entries, then load the ticked ones as sequences."""
import hashlib
import re

import streamlit as st

from modules.fetch import (DEFAULT_QUERY_CAP, MAX_QUERY_CAP, ORGANISMS, FetchError, FetchResult,
                           extract_uniprot_ids, fetch_uniprot_fasta, uniprot_batch_tsv, uniprot_gene_tsv,
                           uniprot_query_count, uniprot_query_tsv)

MODES = ("IDs", "Search query", "Genes + organism")
OTHER_TAXON = "Other taxon ID"


def _uploaded_text(key):
    upload = st.sidebar.file_uploader("…or upload a list (.txt, .csv, .tsv)", type=["txt", "csv", "tsv"], key=key)
    return upload.getvalue().decode("utf-8", "replace") if upload else ""


def _store(frame, missing=(), missing_label=""):
    st.session_state.uniprot_preview = {"frame": frame, "missing": list(missing), "missing_label": missing_label}


def uniprot_batch_sidebar():
    """Sidebar controls. 'Search UniProt' stores the preview in st.session_state.uniprot_preview."""
    mode = st.sidebar.radio("Look up by", MODES, key="uniprot_mode")
    try:
        if mode == "IDs":
            text = st.sidebar.text_area("UniProt accessions or entry names", key="uniprot_ids", height=120,
                                        placeholder="P69905, P68871\nHBA_HUMAN")
            text = f"{text}\n{_uploaded_text('uniprot_ids_file')}"
            if st.sidebar.button("Search UniProt", key="uniprot_search", icon=":material/search:"):
                accessions, names = extract_uniprot_ids(text)
                if not accessions and not names:
                    raise FetchError("No UniProt accessions or entry names found in the input.")
                with st.spinner(f"Looking up {len(accessions) + len(names)} IDs in UniProt..."):
                    frame, missing = uniprot_batch_tsv(accessions, names)
                _store(frame, missing, "IDs not found")
        elif mode == "Search query":
            query = st.sidebar.text_input("UniProt query", key="uniprot_query",
                                          placeholder="family:globin AND reviewed:true AND organism_id:9606")
            cap = st.sidebar.number_input("Maximum entries", 100, MAX_QUERY_CAP, DEFAULT_QUERY_CAP, 100,
                                          key="uniprot_cap")
            isoforms = st.sidebar.toggle("Include isoforms", key="uniprot_query_isoforms")
            if st.sidebar.button("Search UniProt", key="uniprot_search", icon=":material/search:"):
                if not query.strip():
                    raise FetchError("Enter a UniProt query.")
                with st.spinner("Counting matches..."):
                    count = uniprot_query_count(query, isoforms)
                if count == 0:
                    raise FetchError("The query matched no UniProt entries.")
                if count > cap:
                    raise FetchError(f"The query matches {count:,} entries, above the limit of {cap:,}. Narrow the "
                                     f"query or raise the limit (up to {MAX_QUERY_CAP:,}).")
                with st.spinner(f"Downloading {count:,} entries..."):
                    _store(uniprot_query_tsv(query, isoforms))
        else:
            text = st.sidebar.text_area("Gene names", key="uniprot_genes", height=100, placeholder="HBB, TP53, EGFR")
            text = f"{text}\n{_uploaded_text('uniprot_genes_file')}"
            organism = st.sidebar.selectbox("Organism", [*ORGANISMS, OTHER_TAXON], key="uniprot_organism")
            taxon = ORGANISMS.get(organism) or st.sidebar.number_input("Taxon ID", 1, 10 ** 8, 9606,
                                                                       key="uniprot_taxon")
            reviewed = st.sidebar.toggle("Reviewed (Swiss-Prot) only", value=True, key="uniprot_reviewed")
            isoforms = st.sidebar.toggle("Include isoforms", key="uniprot_gene_isoforms")
            if st.sidebar.button("Search UniProt", key="uniprot_search", icon=":material/search:"):
                genes = [g for g in re.split(r"[\s,;\t]+", text) if g]
                if not genes:
                    raise FetchError("Enter at least one gene name.")
                with st.spinner(f"Looking up {len(genes)} genes in UniProt..."):
                    frame, missing = uniprot_gene_tsv(genes, taxon, reviewed, isoforms)
                _store(frame, missing, "Genes with no entry")
    except FetchError as e:
        st.sidebar.error(str(e))


def safe_fasta(accessions):
    """FASTA for the lazy download button; a failed fetch becomes a comment line instead of an exception."""
    try:
        return fetch_uniprot_fasta(accessions)
    except FetchError as e:
        return f"# UniProt download failed: {e}\n"


def uniprot_batch_preview():
    """Main-area preview: tick entries, then load them (sets st.session_state.fetched for the normal pipeline)."""
    preview = st.session_state.get("uniprot_preview")
    if not preview:
        return
    frame, missing = preview["frame"], preview["missing"]
    with st.expander(f"UniProt results: {len(frame)} entries", expanded=st.session_state.get("fetched") is None):
        if missing:
            shown = ", ".join(missing[:50]) + (" …" if len(missing) > 50 else "")
            st.warning(f"{preview['missing_label']} ({len(missing)}): {shown}")
        if frame.empty:
            st.info("No entries to load.")
            return
        table = frame.copy()
        synonym = [m == "synonym" for m in frame["Match"]] if "Match" in frame else [False] * len(frame)
        table.insert(0, "Load", [not s for s in synonym])
        if any(synonym):
            st.caption("Rows matched only through a gene synonym start unticked.")
        digest = hashlib.sha1(frame.to_csv().encode()).hexdigest()[:10]
        edited = st.data_editor(table, hide_index=True, width="stretch",
                                disabled=[c for c in table.columns if c != "Load"], key=f"uniprot_editor_{digest}")
        chosen = edited.loc[edited["Load"], "Accession"].tolist()
        c1, c2, c3 = st.columns(3)
        if c1.button(f"Load {len(chosen)} selected", key="uniprot_load", type="primary", disabled=not chosen,
                     icon=":material/play_arrow:"):
            try:
                with st.spinner(f"Downloading {len(chosen)} sequences..."):
                    st.session_state.fetched = FetchResult(fasta=fetch_uniprot_fasta(chosen))
            except FetchError as e:
                st.error(str(e))
        c2.download_button("Table (TSV)", edited[edited["Load"]].drop(columns=["Load"]).to_csv(sep="\t", index=False),
                           "uniprot_entries.tsv", "text/tab-separated-values", on_click="ignore",
                           icon=":material/download:", disabled=not chosen)
        c3.download_button("FASTA", lambda accessions=chosen: safe_fasta(accessions), "uniprot_entries.fasta", "text/plain",
                           on_click="ignore", icon=":material/download:", disabled=not chosen)
