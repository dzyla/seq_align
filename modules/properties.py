"""Protein physico-chemical properties (Biopython ProtParam)."""
import pandas as pd
import streamlit as st
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modules.ui import page_header

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
COLUMNS = ["Length", "MW (Da)", "pI", "Ext. coeff. (reduced)", "Ext. coeff. (cystines)",
           "A280 0.1% (reduced)", "A280 0.1% (cystines)", "GRAVY", "Instability index", "Aromaticity",
           "Non-standard removed"]


def protein_properties(seq):
    """ProtParam values on the standard residues only; non-standard residues are counted and removed."""
    text = str(seq).upper()
    clean = "".join(c for c in text if c in STANDARD_AA)
    removed = sum(1 for c in text if c.isalpha() or c == "*") - len(clean)
    if not clean:
        return {**dict.fromkeys(COLUMNS), "Length": 0, "Non-standard removed": removed}
    analysis = ProteinAnalysis(clean)
    mw = analysis.molecular_weight()
    reduced, cystines = analysis.molar_extinction_coefficient()
    return {
        "Length": len(clean),
        "MW (Da)": mw,
        "pI": analysis.isoelectric_point(),
        "Ext. coeff. (reduced)": reduced,
        "Ext. coeff. (cystines)": cystines,
        "A280 0.1% (reduced)": reduced / mw,
        "A280 0.1% (cystines)": cystines / mw,
        "GRAVY": analysis.gravy(),
        "Instability index": analysis.instability_index(),
        "Aromaticity": analysis.aromaticity(),
        "Non-standard removed": removed,
    }


def properties_table(records):
    return pd.DataFrame([{"Sequence ID": r.id, **protein_properties(r.seq)} for r in records])


def sequence_properties_section(sequences):
    page_header("Sequence properties", "Molecular weight, pI, extinction coefficients, GRAVY and more.")
    st.info("Molecular weight, pI, extinction coefficients (M⁻¹cm⁻¹ at 280 nm), A280 of a 1 mg/mL solution, "
            "GRAVY, instability index and aromaticity (Biopython ProtParam). Non-standard residues are excluded.")
    df = properties_table(sequences)
    st.dataframe(df.round(3), hide_index=True, width="stretch")
    st.download_button("Properties (CSV)", df.to_csv(index=False), "sequence_properties.csv", "text/csv",
                       on_click="ignore", icon=":material/download:")
    if (df["Non-standard removed"] > 0).any():
        st.caption("Some sequences contained non-standard residues (X, U, B, Z, *), which were excluded.")
