import traceback

import anarci
import pandas as pd
import streamlit as st

ALL_SPECIES = ["human", "mouse", "rat", "rabbit", "rhesus", "pig", "alpaca"]
SCHEMES = ["IMGT", "Chothia", "Kabat", "Martin", "AHo", "Wolfguy"]
REGION_ORDER = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]

_IMGT = [("FR1", 1, 26), ("CDR1", 27, 38), ("FR2", 39, 55), ("CDR2", 56, 65),
         ("FR3", 66, 104), ("CDR3", 105, 117), ("FR4", 118, 128)]
_KABAT_H = [("FR1", 1, 30), ("CDR1", 31, 35), ("FR2", 36, 49), ("CDR2", 50, 65),
            ("FR3", 66, 94), ("CDR3", 95, 102), ("FR4", 103, 113)]
_KABAT_L = [("FR1", 1, 23), ("CDR1", 24, 34), ("FR2", 35, 49), ("CDR2", 50, 56),
            ("FR3", 57, 88), ("CDR3", 89, 97), ("FR4", 98, 109)]
_CHOTHIA_H = [("FR1", 1, 25), ("CDR1", 26, 32), ("FR2", 33, 51), ("CDR2", 52, 56),
              ("FR3", 57, 94), ("CDR3", 95, 102), ("FR4", 103, 113)]

# Region boundaries are inclusive on the integer position; insertion codes (e.g. 35A, 100B)
# belong to their integer position. Martin numbering uses Chothia CDR definitions.
REGION_DEFINITIONS = {
    "imgt": {"H": _IMGT, "L": _IMGT},
    "kabat": {"H": _KABAT_H, "L": _KABAT_L},
    "chothia": {"H": _CHOTHIA_H, "L": _KABAT_L},
    "martin": {"H": _CHOTHIA_H, "L": _KABAT_L},
}
REGION_COLORS = {"FR1": "#e0e0e0", "CDR1": "#ffcccc", "FR2": "#e0e0e0", "CDR2": "#ccffcc",
                 "FR3": "#e0e0e0", "CDR3": "#ccccff", "FR4": "#e0e0e0"}


def extract_regions(numbering, scheme, chain_type):
    """Split an ANARCI numbering into framework and CDR sequences for supported schemes."""
    definitions = REGION_DEFINITIONS.get(scheme.lower())
    if not definitions:
        return {}
    if scheme.lower() == "imgt":
        bounds = definitions["H"]
    elif chain_type == "H":
        bounds = definitions["H"]
    elif chain_type in ("K", "L"):
        bounds = definitions["L"]
    else:
        return {}
    regions = {name: [] for name in REGION_ORDER}
    for (pos, _ins), aa in numbering:
        if aa == '-':
            continue
        for name, start, end in bounds:
            if start <= pos <= end:
                regions[name].append(aa)
                break
    return {k: "".join(v) for k, v in regions.items()}


def extract_cdr_imgt(numbering):
    """Backward-compatible IMGT region extraction."""
    return extract_regions(numbering, "imgt", "H")


def run_numbering(pairs, scheme, species, threshold):
    """Number (id, sequence) pairs with ANARCI and return one row per detected domain."""
    _, numbered, alignment_details, _ = anarci.run_anarci(
        list(pairs),
        scheme=scheme.lower(),
        allowed_species=species or None,
        bit_score_threshold=threshold,
        output=False,
    )
    rows = []
    for i, seq_res in enumerate(numbered):
        if not seq_res:
            continue
        for j, (numbering, start, end) in enumerate(seq_res):
            details = alignment_details[i][j]
            row = {
                "Sequence ID": pairs[i][0],
                "Domain Index": j + 1,
                "Chain Type": details.get('chain_type', 'Unknown'),
                "Species": details.get('species', 'Unknown'),
                "E-value": float(details.get('evalue', 0)),
                "Score": float(details.get('bitscore', 0)),
                "Start": start,
                "End": end,
            }
            row.update(extract_regions(numbering, scheme, row["Chain Type"]))
            row["Numbering"] = " ".join(f"{pos}{ins.strip()}{aa}" for (pos, ins), aa in numbering if aa != '-')
            row["Sequence"] = "".join(aa for _, aa in numbering if aa != '-')
            rows.append(row)
    return pd.DataFrame(rows)


def _regions_html(row):
    parts = []
    for region in REGION_ORDER:
        segment = row.get(region, "")
        if isinstance(segment, str) and segment:
            parts.append(
                f"<span style='background-color: {REGION_COLORS[region]}; padding: 2px; border-radius: 3px; "
                f"margin-right: 2px;' title='{region}'><strong>{region}:</strong> {segment}</span>"
            )
    return " ".join(parts)


def antibody_prediction_section(sequences):
    st.header("🛡️ Antibody Functional Domain Prediction")
    st.info("Predict functional domains (CDRs, Framework Regions) of antibodies using ANARCI.")

    if not sequences:
        st.error("No sequences available for prediction.")
        return

    col1, col2 = st.columns(2)
    with col1:
        scheme = st.selectbox("Numbering Scheme", SCHEMES, index=0)
    with col2:
        species = st.multiselect("Allowed Species", ALL_SPECIES, default=ALL_SPECIES,
                                 help="All species by default, so camelid VHH are not dropped.")
    threshold = st.slider("Bit Score Threshold", 0, 200, 80, help="Minimum score for HMM alignment")

    params = {"ids": tuple(s.id for s in sequences), "seqs": tuple(str(s.seq) for s in sequences),
              "scheme": scheme, "species": tuple(species), "threshold": threshold}

    if st.button("Predict Domains"):
        with st.spinner("Running ANARCI..."):
            try:
                df = run_numbering([(s.id, str(s.seq)) for s in sequences], scheme,
                                   None if set(species) == set(ALL_SPECIES) else species, threshold)
                st.session_state.antibody_results = {"params": params, "df": df}
            except Exception as e:
                st.session_state.antibody_results = None
                st.error(f"Error running ANARCI: {e}")
                st.text(traceback.format_exc())
                return

    stored = st.session_state.get("antibody_results")
    if not stored:
        return
    if stored["params"] != params:
        st.info("Settings changed since the last run. Click **Predict Domains** to update the results.")
        return

    df = stored["df"]
    if df.empty:
        st.warning("No domains found with the current settings.")
        return

    st.success(f"Found {len(df)} domains.")
    has_regions = "CDR1" in df.columns
    if has_regions:
        cols = ["Sequence ID", "Chain Type", "Species", "CDR1", "CDR2", "CDR3", "Score", "E-value", "Sequence"]
        st.dataframe(df[[c for c in cols if c in df.columns]])
    else:
        st.caption(f"CDR definitions are not available for the {scheme} scheme; showing numbering only.")
        st.dataframe(df)

    st.download_button("Download CSV", df.to_csv(index=False), "antibody_domains.csv", "text/csv", on_click="ignore")

    st.subheader("Detailed Domain View")
    for _, row in df.iterrows():
        with st.expander(f"{row['Sequence ID']} - Domain {row['Domain Index']} ({row['Chain Type']})"):
            st.json({k: (v if not hasattr(v, "item") else v.item()) for k, v in row.to_dict().items()})
            if has_regions:
                st.markdown(f"### Regions ({scheme})")
                st.markdown(_regions_html(row), unsafe_allow_html=True)
