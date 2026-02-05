import streamlit as st
import pandas as pd
from Bio.Seq import Seq

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

    # Standard Translation (Frame +1)
    st.subheader("Translation (Frame +1)")

    # Clean sequence
    dna_seq = str(selected_seq.seq).upper()
    valid_chars = set('ATGCNRYKMSWBDHV')
    clean_seq = ''.join(c for c in dna_seq if c in valid_chars)

    # Translate
    remainder = len(clean_seq) % 3
    if remainder:
        clean_seq_tr = clean_seq[:-remainder]
    else:
        clean_seq_tr = clean_seq

    try:
        protein_seq = Seq(clean_seq_tr).translate()
        st.code(str(protein_seq))
    except Exception as e:
        st.error(f"Error during translation: {e}")

    st.markdown("### 🕵️ ORF Finder")
    min_len = st.number_input("Minimum ORF Length (AA)", value=30, min_value=10, step=5)

    if st.button("Find ORFs"):
        orfs = find_orfs(str(selected_seq.seq).upper(), min_len=min_len)

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
                with st.expander(f"Frame {row['Frame']}, Start {row['Start (AA)']}, Length {row['Length (AA)']}"):
                    st.code(row['Sequence'])
        else:
            st.warning("No ORFs found with current settings.")


def find_orfs(sequence, min_len=30):
    """
    Find Open Reading Frames (ORFs) in a DNA sequence.
    Searches all 6 reading frames.

    Parameters:
        sequence (str): DNA sequence.
        min_len (int): Minimum length of the protein (in amino acids).

    Returns:
        list: List of dictionaries containing ORF details.
    """
    orfs = []
    seq_obj = Seq(sequence)

    frames = [1, 2, 3, -1, -2, -3]

    for frame in frames:
        if frame > 0:
            nucleotide_seq = seq_obj[frame - 1:]
            frame_label = f"+{frame}"
        else:
            nucleotide_seq = seq_obj.reverse_complement()[abs(frame) - 1:]
            frame_label = f"{frame}"

        # Translate
        n_len = len(nucleotide_seq)
        remainder = n_len % 3
        if remainder > 0:
            nucleotide_seq = nucleotide_seq[:-remainder]

        if len(nucleotide_seq) == 0:
            continue

        protein_seq = nucleotide_seq.translate(table=1)  # Standard code
        protein_str = str(protein_seq)

        aa_start = 0
        while aa_start < len(protein_str):
            # Find next Met
            met_index = protein_str.find('M', aa_start)
            if met_index == -1:
                break

            # Find next Stop after Met
            stop_index = protein_str.find('*', met_index)

            if stop_index != -1:
                # Found an ORF
                length_aa = stop_index - met_index
                if length_aa >= min_len:
                    orf_seq = protein_str[met_index:stop_index]

                    orfs.append({
                        "Frame": frame_label,
                        "Start (AA)": met_index + 1,
                        "End (AA)": stop_index + 1,
                        "Length (AA)": length_aa,
                        "Sequence": orf_seq
                    })

                # Move start to find nested ORFs
                aa_start = met_index + 1
            else:
                break

    return orfs
