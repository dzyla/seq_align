import re

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from Bio.Restriction import Analysis, CommOnly, RestrictionBatch
from Bio.Seq import Seq

from modules.seqtools import clean_sequence
from modules.utils import stale_notice
from modules.ui import page_header, show_plotly

CODON_TABLES = {
    "1 Standard": 1,
    "2 Vertebrate mitochondrial": 2,
    "4 Mold/Mycoplasma": 4,
    "5 Invertebrate mitochondrial": 5,
    "11 Bacterial/Plastid": 11,
}

def dna_translation_section(sequences):
    """
    Handles the DNA Translation workflow.
    """
    page_header("DNA translation and ORFs", "Translate in all frames and find open reading frames.")
    st.info("Translate DNA sequences to protein and find Open Reading Frames (ORFs).")

    if not sequences:
        st.warning("Please upload DNA sequences.")
        return

    # Select sequence
    seq_ids = [seq.id for seq in sequences]
    selected_id = st.selectbox("Select Sequence", seq_ids)

    selected_seq = next(s for s in sequences if s.id == selected_id)

    # Display Original DNA
    with st.expander("Original DNA Sequence", expanded=False):
        st.code(str(selected_seq.seq))

    table_label = st.selectbox("Codon table", list(CODON_TABLES))
    table = CODON_TABLES[table_label]

    # Standard Translation (Frame +1)
    st.subheader("Translation (Frame +1)")

    # Clean sequence
    clean_seq = clean_sequence(selected_seq.seq, "DNA")

    # Translate
    remainder = len(clean_seq) % 3
    if remainder:
        clean_seq_tr = clean_seq[:-remainder]
    else:
        clean_seq_tr = clean_seq

    try:
        protein_seq = Seq(clean_seq_tr).translate(table=table)
        st.code(str(protein_seq))
    except Exception as e:
        st.error(f"Error during translation: {e}")

    st.subheader("ORF finder")
    min_len = st.number_input("Minimum ORF Length (AA)", value=30, min_value=10, step=5)
    params = {"id": selected_seq.id, "seq": str(selected_seq.seq), "min_len": min_len, "table": table}

    if st.button("Find ORFs", key="find_orfs"):
        st.session_state.orf_results = {
            "params": params, "orfs": find_orfs(str(selected_seq.seq), min_len=min_len, table=table)}

    stored = st.session_state.get("orf_results")
    if not stored:
        return
    if stored["params"] != params:
        stale_notice()
        return
    orfs = stored["orfs"]
    if not orfs:
        st.warning("No ORFs found with current settings.")
        return
    st.success(f"Found {len(orfs)} ORFs.")
    df = pd.DataFrame(orfs)
    st.dataframe(df)
    st.download_button("Download ORFs (CSV)", df.to_csv(index=False), "orfs.csv", "text/csv", on_click="ignore")
    st.subheader("ORF Sequences")
    for _, row in df.iterrows():
        with st.expander(f"Frame {row['Frame']}, {row['Start (nt)']}–{row['End (nt)']} nt, {row['Length (AA)']} aa"):
            st.code(row['Sequence'])


def find_orfs(sequence, min_len=30, table=1):
    """
    Find ORFs (first ATG to stop) in all six frames; one ORF per stop codon.

    Coordinates are 1-based, inclusive, on the input strand; End includes the stop codon.
    """
    seq = clean_sequence(sequence, "DNA")
    length = len(seq)
    orfs = []
    for strand, strand_seq in (("+", Seq(seq)), ("-", Seq(seq).reverse_complement())):
        for offset in range(3):
            frame_seq = strand_seq[offset:]
            frame_seq = frame_seq[:len(frame_seq) - len(frame_seq) % 3]
            if not frame_seq:
                continue
            protein = str(frame_seq.translate(table=table))
            segment_start = 0
            for match in re.finditer(r"\*", protein):
                stop = match.start()
                met = protein.find("M", segment_start, stop)
                if met != -1 and stop - met >= min_len:
                    start_nt = offset + 3 * met + 1
                    end_nt = offset + 3 * (stop + 1)
                    if strand == "-":
                        start_nt, end_nt = length - end_nt + 1, length - start_nt + 1
                    orfs.append({
                        "Frame": f"{strand}{offset + 1}",
                        "Strand": strand,
                        "Start (nt)": start_nt,
                        "End (nt)": end_nt,
                        "Length (AA)": stop - met,
                        "Sequence": protein[met:stop],
                    })
                segment_start = stop + 1
    return orfs


POPULAR_ENZYMES = ["AgeI", "ApaI", "AscI", "AvrII", "BamHI", "BglII", "BsaI", "BsmBI", "BspEI", "ClaI", "EcoRI",
                   "EcoRV", "HindIII", "KpnI", "MfeI", "MluI", "NcoI", "NdeI", "NheI", "NotI", "NsiI", "PacI",
                   "PstI", "SacI", "SalI", "SmaI", "SpeI", "SphI", "XbaI", "XhoI"]
ENZYME_SETS = {"Popular": POPULAR_ENZYMES, "All commercial": None}


def reverse_complement(seq):
    return str(Seq(clean_sequence(seq, "DNA")).reverse_complement())


def codon_usage(seq, frame=1, table=1):
    clean = clean_sequence(seq, "DNA")[frame - 1:]
    codons = [clean[i:i + 3] for i in range(0, len(clean) - len(clean) % 3, 3)]
    counts = pd.Series(codons).value_counts()
    rows = [{"Codon": codon, "AA": str(Seq(codon).translate(table=table)), "Count": int(count)}
            for codon, count in counts.items()]
    df = pd.DataFrame(rows, columns=["Codon", "AA", "Count"])
    df["Fraction of AA"] = df["Count"] / df.groupby("AA")["Count"].transform("sum")
    return df.sort_values(["AA", "Codon"]).reset_index(drop=True)


def restriction_sites(seq, circular=False, enzyme_set="Popular"):
    names = ENZYME_SETS[enzyme_set]
    batch = RestrictionBatch(CommOnly) if names is None else RestrictionBatch(names)
    hits = Analysis(batch, Seq(clean_sequence(seq, "DNA")), linear=not circular).full()
    rows = [{"Enzyme": str(enzyme), "Site": enzyme.site, "Cuts": len(positions), "Positions": sorted(positions)}
            for enzyme, positions in hits.items() if positions]
    df = pd.DataFrame(rows, columns=["Enzyme", "Site", "Cuts", "Positions"])
    return df.sort_values(["Cuts", "Enzyme"]).reset_index(drop=True)


def restriction_map_figure(sites_df, seq_length, enzymes):
    fig = go.Figure()
    chosen = sites_df[sites_df["Enzyme"].isin(enzymes)]
    for k, (_, row) in enumerate(chosen.iterrows()):
        y = (k % 6) + 1
        fig.add_trace(go.Scatter(x=row["Positions"], y=[y] * len(row["Positions"]), mode="markers+text",
                                 text=[row["Enzyme"]] * len(row["Positions"]), textposition="top center",
                                 marker=dict(symbol="line-ns-open", size=16), name=row["Enzyme"],
                                 hovertemplate=f"{row['Enzyme']} ({row['Site']}) cuts at %{{x}}<extra></extra>"))
    fig.add_shape(type="line", x0=1, x1=seq_length, y0=0, y1=0, line=dict(width=4, color="#555555"))
    fig.update_layout(height=320, showlegend=False, xaxis=dict(title="Position (nt)", range=[0, seq_length + 1]),
                      yaxis=dict(visible=False, range=[-0.5, 7.5]), margin=dict(t=30))
    return fig


def dna_tools_section(sequences):
    page_header("DNA tools", "Reverse complement, codon usage and restriction map.")
    selected_id = st.selectbox("Sequence", [s.id for s in sequences], key="dna_tools_seq")
    seq = str(next(s for s in sequences if s.id == selected_id).seq)
    tab_rc, tab_codon, tab_re = st.tabs(["Reverse complement", "Codon usage", "Restriction map"])

    with tab_rc:
        rc = reverse_complement(seq)
        st.code(rc)
        st.download_button("Reverse complement (FASTA)", f">{selected_id}_revcomp\n{rc}\n",
                           f"{selected_id}_revcomp.fasta", "text/plain", on_click="ignore", icon=":material/download:")

    with tab_codon:
        c1, c2 = st.columns(2)
        frame = c1.selectbox("Reading frame", (1, 2, 3), key="codon_frame")
        table = CODON_TABLES[c2.selectbox("Codon table", list(CODON_TABLES), key="codon_table")]
        usage = codon_usage(seq, frame, table)
        st.dataframe(usage.round(3), hide_index=True, width="stretch")
        st.download_button("Codon usage (CSV)", usage.to_csv(index=False), "codon_usage.csv", "text/csv",
                           on_click="ignore", icon=":material/download:")

    with tab_re:
        c1, c2 = st.columns(2)
        circular = c1.checkbox("Circular (plasmid)", key="re_circular")
        enzyme_set = c2.selectbox("Enzymes", list(ENZYME_SETS), key="re_set")
        sites = restriction_sites(seq, circular, enzyme_set)
        if sites.empty:
            st.write("No sites found.")
            return
        unique = sites[sites["Cuts"] == 1]["Enzyme"].tolist()
        shown = st.multiselect("Enzymes on map", sites["Enzyme"].tolist(), default=unique[:15], key="re_shown")
        show_plotly(restriction_map_figure(sites, len(clean_sequence(seq, "DNA")), shown), width="stretch")
        st.dataframe(sites, hide_index=True, width="stretch")
        st.caption(f"{len(unique)} unique cutter(s). Positions are the first nucleotide after the top-strand cut.")
        st.download_button("Restriction sites (CSV)", sites.to_csv(index=False), "restriction_sites.csv",
                           "text/csv", on_click="ignore", icon=":material/download:")
