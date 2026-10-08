"""Dataset overview panel and bundled example datasets."""
import statistics
from pathlib import Path

import pandas as pd
import streamlit as st

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = {
    "Globins (protein)": EXAMPLE_DIR / "globins.fasta",
    "pUC19 plasmid (DNA)": EXAMPLE_DIR / "puc19.fasta",
    "Antibody VH/VL (protein)": EXAMPLE_DIR / "antibody_fv.fasta",
}


def dataset_summary(records):
    lengths = [len(r.seq) for r in records]
    return {
        "count": len(records),
        "min_length": min(lengths) if lengths else 0,
        "median_length": int(statistics.median(lengths)) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def summary_table(records, preview=40):
    rows = []
    for r in records:
        s = str(r.seq)
        rows.append({"ID": r.id, "Length": len(s), "Preview": s[:preview] + ("…" if len(s) > preview else "")})
    return pd.DataFrame(rows, columns=["ID", "Length", "Preview"])


def show_dataset_overview(records, seq_type, n_structures=0):
    """Bordered metric cards for the loaded dataset, with the sequence list in an expander."""
    summary = dataset_summary(records)
    cols = st.columns(5 if n_structures else 4)
    cols[0].metric("Sequences", summary["count"], border=True)
    cols[1].metric("Type", seq_type, border=True)
    cols[2].metric("Length range", f"{summary['min_length']}–{summary['max_length']}", border=True)
    cols[3].metric("Median length", summary["median_length"], border=True)
    if n_structures:
        cols[4].metric("Structures", n_structures, border=True)
    with st.expander("Sequence list"):
        st.dataframe(summary_table(records), hide_index=True, width="stretch")


def load_example_callback():
    """on_click callback: put the chosen example into the FASTA text box."""
    name = st.session_state.get("example_choice")
    if name in EXAMPLES:
        st.session_state.input_format = "Text (FASTA)"
        st.session_state.fasta_text = EXAMPLES[name].read_text()
