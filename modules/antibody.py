import re
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
# (name, regex, severity, report only inside CDRs)
LIABILITY_RULES = [
    ("N-glycosylation", r"N[^P][ST]", "High", False),
    ("Deamidation", r"NG", "High", False),
    ("Deamidation", r"N[ST]", "Medium", False),
    ("Isomerization", r"D[GS]", "Medium", False),
    ("Fragmentation", r"DP", "Medium", False),
    ("Oxidation", r"[MW]", "Medium", True),
]
# Conserved intradomain disulfide positions per numbering scheme.
CANONICAL_CYSTEINES = {
    "imgt": {"H": {23, 104}, "L": {23, 104}},
    "kabat": {"H": {22, 92}, "L": {23, 88}},
    "chothia": {"H": {22, 92}, "L": {23, 88}},
    "martin": {"H": {22, 92}, "L": {23, 88}},
}
REGION_COLORS = {"FR1": "#e0e0e0", "CDR1": "#ffcccc", "FR2": "#e0e0e0", "CDR2": "#ccffcc",
                 "FR3": "#e0e0e0", "CDR3": "#ccccff", "FR4": "#e0e0e0"}


def _region_bounds(scheme, chain_type):
    definitions = REGION_DEFINITIONS.get(scheme.lower())
    if not definitions:
        return None
    if scheme.lower() == "imgt" or chain_type == "H":
        return definitions["H"]
    if chain_type in ("K", "L"):
        return definitions["L"]
    return None


def residue_regions(numbering, scheme, chain_type):
    """(numbering label, region) for every non-gap residue; region is '' when undefined."""
    bounds = _region_bounds(scheme, chain_type)
    out = []
    for (pos, ins), aa in numbering:
        if aa == '-':
            continue
        region = ""
        if bounds:
            region = next((name for name, start, end in bounds if start <= pos <= end), "")
        out.append((f"{pos}{ins.strip()}", region))
    return out


def find_liabilities(sequence, labels, regions, scheme, chain_type):
    """Flag common sequence liabilities in one numbered domain (for review, not prediction)."""
    found = []
    for name, pattern, severity, cdr_only in LIABILITY_RULES:
        for match in re.finditer(f"(?=({pattern}))", sequence):
            i = match.start()
            if cdr_only and not regions[i].startswith("CDR"):
                continue
            found.append({"Liability": name, "Motif": match.group(1), "Position": labels[i],
                          "Region": regions[i], "Severity": severity})
    canonical = CANONICAL_CYSTEINES.get(scheme.lower(), {}).get("H" if chain_type == "H" else "L")
    if canonical is not None:
        for i, aa in enumerate(sequence):
            number = int(re.match(r"\d+", labels[i]).group())
            if aa == "C" and number not in canonical:
                found.append({"Liability": "Unpaired cysteine", "Motif": "C", "Position": labels[i],
                              "Region": regions[i], "Severity": "High"})
    return found


# A separator is required, so IDs that merely end in h/l/k (e.g. "seq_vel") are left alone.
_CHAIN_SUFFIX = re.compile(r"[_\-\s.|](heavy|light|hc|lc|vh|vl|vk|h|l|k)$", re.IGNORECASE)


def chain_stem(seq_id):
    stem = _CHAIN_SUFFIX.sub("", seq_id)
    return stem or seq_id


def _domain_label(row):
    return f"{row['Sequence ID']} (domain {row['Domain Index']}, {row['Chain Type']})"


def pair_domains(rows):
    """Pair heavy and light domains: within one sequence (scFv) first, then across IDs sharing a stem."""
    heavy = [r for r in rows if r["Chain Type"] == "H"]
    light = [r for r in rows if r["Chain Type"] in ("K", "L")]
    used, pairs = set(), []

    def add(h, l, fmt, name):
        used.update({id(h), id(l)})
        pairs.append({"Pair": name, "Format": fmt, "Heavy": _domain_label(h), "Light": _domain_label(l),
                      "Heavy CDR3": h.get("CDR3", ""), "Light CDR3": l.get("CDR3", ""),
                      "Heavy V": h.get("V Gene", ""), "Light V": l.get("V Gene", "")})

    for h in heavy:
        partner = next((l for l in light if id(l) not in used and l["Sequence ID"] == h["Sequence ID"]), None)
        if partner:
            add(h, partner, "scFv / single chain", h["Sequence ID"])
    for h in heavy:
        if id(h) in used:
            continue
        partner = next((l for l in light if id(l) not in used and l["Sequence ID"] != h["Sequence ID"]
                        and chain_stem(l["Sequence ID"]) == chain_stem(h["Sequence ID"])), None)
        if partner:
            add(h, partner, "Separate chains", chain_stem(h["Sequence ID"]))
    unpaired = [_domain_label(r) for r in heavy + light if id(r) not in used]
    return pairs, unpaired


def public_columns(df):
    return [c for c in df.columns if not str(c).startswith("_")]


def extract_regions(numbering, scheme, chain_type):
    """Split an ANARCI numbering into framework and CDR sequences for supported schemes."""
    if _region_bounds(scheme, chain_type) is None:
        return {}
    regions = {name: [] for name in REGION_ORDER}
    for (_label, region), aa in zip(residue_regions(numbering, scheme, chain_type),
                                    (aa for _, aa in numbering if aa != '-')):
        if region:
            regions[region].append(aa)
    return {k: "".join(v) for k, v in regions.items()}


def extract_cdr_imgt(numbering):
    """Backward-compatible IMGT region extraction."""
    return extract_regions(numbering, "imgt", "H")


def run_numbering(pairs, scheme, species, threshold, assign_germline=True):
    """Number (id, sequence) pairs with ANARCI; one row per domain, with germline genes."""
    _, numbered, alignment_details, _ = anarci.run_anarci(
        list(pairs),
        scheme=scheme.lower(),
        allowed_species=species or None,
        bit_score_threshold=threshold,
        assign_germline=assign_germline,
        output=False,
    )
    rows = []
    for i, seq_res in enumerate(numbered):
        if not seq_res:
            continue
        for j, (numbering, start, end) in enumerate(seq_res):
            details = alignment_details[i][j]
            chain_type = details.get('chain_type', 'Unknown')
            germlines = details.get("germlines") or {}
            v_hit, v_id = (germlines.get("v_gene") or [None, None])
            j_hit, j_id = (germlines.get("j_gene") or [None, None])
            labelled = residue_regions(numbering, scheme, chain_type)
            row = {
                "Sequence ID": pairs[i][0],
                "Domain Index": j + 1,
                "Chain Type": chain_type,
                "Species": details.get('species', 'Unknown'),
                "Germline Species": v_hit[0] if v_hit else "",
                "V Gene": v_hit[1] if v_hit else "",
                "V Identity (%)": round(100 * v_id, 1) if v_id is not None else None,
                "J Gene": j_hit[1] if j_hit else "",
                "J Identity (%)": round(100 * j_id, 1) if j_id is not None else None,
                "E-value": float(details.get('evalue', 0)),
                "Score": float(details.get('bitscore', 0)),
                "Start": start,
                "End": end,
            }
            row.update(extract_regions(numbering, scheme, chain_type))
            residues = [aa for _, aa in numbering if aa != '-']
            row["Numbering"] = " ".join(f"{label}{aa}" for (label, _), aa in zip(labelled, residues))
            row["Sequence"] = "".join(residues)
            row["_labels"] = [label for label, _ in labelled]
            row["_regions"] = [region for _, region in labelled]
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
        st.dataframe(df[public_columns(df)])

    st.download_button("Download CSV", df[public_columns(df)].to_csv(index=False), "antibody_domains.csv",
                       "text/csv", on_click="ignore")

    st.subheader("Detailed Domain View")
    for _, row in df.iterrows():
        with st.expander(f"{row['Sequence ID']} - Domain {row['Domain Index']} ({row['Chain Type']})"):
            st.json({k: (v if not hasattr(v, "item") else v.item()) for k, v in row.to_dict().items()
                     if not str(k).startswith("_")})
            if has_regions:
                st.markdown(f"### Regions ({scheme})")
                st.markdown(_regions_html(row), unsafe_allow_html=True)
