"""Bundle the current dataset, results and parameters into one ZIP."""
import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
from Bio import SeqIO

from modules.seqtools import fingerprint

PARAM_KEYS = ("last_pairwise_params", "last_msa_params", "best_match_params", "last_conversion_params")
RESULT_KEYS_EXPORTED = ("pairwise_result", "msa_aligned", "best_match_results", "clustering_matrix",
                        "antibody_results", "orf_results", "tree_result", "msa_tree", "converted_data")


def _filled(value):
    """True for non-empty results; safe for numpy arrays, which have no plain truth value."""
    if value is None:
        return False
    if hasattr(value, "__len__"):
        return len(value) > 0
    return True


def has_results(state):
    return any(_filled(state.get(k)) for k in RESULT_KEYS_EXPORTED)


def build_session_zip(state, sequences, seq_type):
    from modules.antibody import public_columns
    from modules.msa import serialize_alignment
    from modules.pairwise import format_result_text
    from modules.phylogeny import tree_to_newick

    files = {}
    fasta = io.StringIO()
    SeqIO.write(sequences, fasta, "fasta")
    files["sequences.fasta"] = fasta.getvalue()

    if _filled(state.get("pairwise_result")):
        files["pairwise_alignment.txt"] = format_result_text(state["pairwise_result"])
    if _filled(state.get("msa_aligned")):
        files["msa_alignment.fasta"] = serialize_alignment(state["msa_aligned"], "fasta")
    if _filled(state.get("best_match_results")):
        files["best_match.csv"] = pd.DataFrame(state["best_match_results"][1]).to_csv(index=False)
    if _filled(state.get("clustering_matrix")) and _filled(state.get("clustering_ids")):
        ids = state["clustering_ids"]
        files["clustering_identity_matrix.csv"] = pd.DataFrame(state["clustering_matrix"], index=ids,
                                                               columns=ids).to_csv()
    if _filled(state.get("antibody_results")):
        df = state["antibody_results"]["df"]
        files["antibody_domains.csv"] = df[public_columns(df)].to_csv(index=False)
    if _filled(state.get("orf_results")):
        files["orfs.csv"] = pd.DataFrame(state["orf_results"]["orfs"]).to_csv(index=False)
    if _filled(state.get("tree_result")):
        files["tree.nwk"] = tree_to_newick(state["tree_result"]["tree"])
    if _filled(state.get("msa_tree")):
        files["msa_tree.nwk"] = tree_to_newick(state["msa_tree"]["tree"])
    if _filled(state.get("converted_data")):
        files["converted_sequences.txt"] = state["converted_data"]

    meta = {
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seq_type": seq_type,
        "dataset_fingerprint": fingerprint(sequences),
        "sequence_count": len(sequences),
        "parameters": {k: state[k] for k in PARAM_KEYS if state.get(k)},
    }
    files["session.json"] = json.dumps(meta, indent=2, default=str)
    files["README.txt"] = ("Advanced Sequence Alignment Tool: session export\n\n"
                           + "\n".join(f"- {name}" for name in sorted(files)) + "\n")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()
