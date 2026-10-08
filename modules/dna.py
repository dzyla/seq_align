import re

import streamlit as st
import pandas as pd
from Bio.Seq import Seq

from modules.seqtools import clean_sequence

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
    st.header("🧬 DNA Translation & ORF Finder")
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

    st.markdown("### 🕵️ ORF Finder")
    min_len = st.number_input("Minimum ORF Length (AA)", value=30, min_value=10, step=5)

    if st.button("Find ORFs"):
        orfs = find_orfs(str(selected_seq.seq), min_len=min_len, table=table)

        if orfs:
            st.success(f"Found {len(orfs)} ORFs.")
            df = pd.DataFrame(orfs)
            st.dataframe(df)

            # Download
            csv = df.to_csv(index=False)
            st.download_button("Download ORFs (CSV)", csv, "orfs.csv", "text/csv")

            # Detailed view
            st.subheader("ORF Sequences")
            for index, row in df.iterrows():
                with st.expander(f"Frame {row['Frame']}, {row['Start (nt)']}–{row['End (nt)']} nt, {row['Length (AA)']} aa"):
                    st.code(row['Sequence'])
        else:
            st.warning("No ORFs found with current settings.")


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
