import streamlit as st
import pandas as pd
import traceback
import anarci

def antibody_prediction_section(sequences):
    st.header("🛡️ Antibody Functional Domain Prediction")
    st.info("Predict functional domains (CDRs, Framework Regions) of antibodies using ANARCI.")

    if not sequences:
        st.error("No sequences available for prediction.")
        return

    # Options
    col1, col2 = st.columns(2)
    with col1:
        scheme = st.selectbox("Numbering Scheme", ["IMGT", "Chothia", "Kabat", "Martin", "AHo", "Wolfguy"], index=0)
    with col2:
        species = st.multiselect("Allowed Species", ["human", "mouse", "rat", "rabbit", "rhesus", "pig", "alpaca"], default=["human", "mouse"])

    threshold = st.slider("Bit Score Threshold", 0, 200, 80, help="Minimum score for HMM alignment")

    if st.button("Predict Domains"):
        with st.spinner("Running ANARCI..."):
             # Format sequences for ANARCI: list of (id, sequence)
             # Use the raw sequence string for ANARCI
             anarci_input = [(seq.id, str(seq.seq)) for seq in sequences]

             try:
                 # Run ANARCI
                 # Note: anarci.run_anarci takes a list of (id, seq) tuples
                 sequences_out, numbered, alignment_details, hit_tables = anarci.run_anarci(
                     anarci_input,
                     scheme=scheme.lower(),
                     allowed_species=species if species else None,
                     bit_score_threshold=threshold,
                     output=False
                 )

                 # Process results
                 results_data = []
                 for i, seq_res in enumerate(numbered):
                     if seq_res: # If domains found
                         for j, domain in enumerate(seq_res):
                             # domain is a tuple: (numbering, start, end)
                             numbering, start, end = domain
                             details = alignment_details[i][j]

                             domain_info = {
                                 "Sequence ID": sequences[i].id,
                                 "Domain Index": j+1,
                                 "Chain Type": details.get('chain_type', 'Unknown'),
                                 "Species": details.get('species', 'Unknown'),
                                 "E-value": float(details.get('evalue', 0)),
                                 "Score": float(details.get('bitscore', 0)),
                                 "Start": start,
                                 "End": end,
                             }

                             if scheme == "IMGT":
                                 regions = extract_cdr_imgt(numbering)
                                 domain_info.update(regions)

                             # Full Numbering String
                             seq_map = []
                             domain_sequence = ""
                             for (pos, ins), aa in numbering:
                                 pos_str = str(pos) + ins.strip()
                                 if aa != '-':
                                     seq_map.append(f"{pos_str}{aa}")
                                     domain_sequence += aa
                             domain_info["Numbering"] = " ".join(seq_map)
                             domain_info["Sequence"] = domain_sequence

                             results_data.append(domain_info)

                 if results_data:
                     df = pd.DataFrame(results_data)
                     st.success(f"Found {len(df)} domains.")

                     # Reorder columns for better display if IMGT
                     if scheme == "IMGT":
                         cols = ["Sequence ID", "Chain Type", "Species", "CDR1", "CDR2", "CDR3", "Score", "E-value", "Sequence"]
                         # Add other columns if they exist in df (handle errors if not)
                         display_cols = [c for c in cols if c in df.columns]
                         st.dataframe(df[display_cols])
                     else:
                         st.dataframe(df)

                     # Download CSV
                     csv = df.to_csv(index=False)
                     st.download_button("Download CSV", csv, "antibody_domains.csv", "text/csv")

                     # Detailed View
                     st.subheader("Detailed Domain View")
                     for index, row in df.iterrows():
                         with st.expander(f"{row['Sequence ID']} - Domain {row['Domain Index']} ({row['Chain Type']})"):
                             st.json(row.to_dict())

                             if scheme == "IMGT":
                                 # Visualize CDRs
                                 st.markdown("### Regions (IMGT)")
                                 # Create a visual representation
                                 regions_html = ""
                                 colors = {
                                     "FR1": "#e0e0e0", "CDR1": "#ffcccc",
                                     "FR2": "#e0e0e0", "CDR2": "#ccffcc",
                                     "FR3": "#e0e0e0", "CDR3": "#ccccff",
                                     "FR4": "#e0e0e0"
                                 }

                                 for region in ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]:
                                     seq_segment = row.get(region, "")
                                     if seq_segment:
                                         bg_color = colors.get(region, "#ffffff")
                                         regions_html += f"<span style='background-color: {bg_color}; padding: 2px; border-radius: 3px; margin-right: 2px;' title='{region}'><strong>{region}:</strong> {seq_segment}</span> "

                                 st.markdown(regions_html, unsafe_allow_html=True)

                 else:
                     st.warning("No domains found with the current settings.")

             except Exception as e:
                 st.error(f"Error running ANARCI: {e}")
                 st.text(traceback.format_exc())

def extract_cdr_imgt(numbering):
    """
    Extract CDRs and Framework regions based on IMGT numbering.
    """
    regions = {
        "FR1": [], "CDR1": [], "FR2": [], "CDR2": [], "FR3": [], "CDR3": [], "FR4": []
    }

    for (pos, ins), aa in numbering:
        if aa == '-': continue

        # IMGT definitions
        if 1 <= pos <= 26:
            regions["FR1"].append(aa)
        elif 27 <= pos <= 38:
            regions["CDR1"].append(aa)
        elif 39 <= pos <= 55:
            regions["FR2"].append(aa)
        elif 56 <= pos <= 65:
            regions["CDR2"].append(aa)
        elif 66 <= pos <= 104:
            regions["FR3"].append(aa)
        elif 105 <= pos <= 117:
            regions["CDR3"].append(aa)
        elif 118 <= pos <= 128:
            regions["FR4"].append(aa)

    return {k: "".join(v) for k, v in regions.items()}
