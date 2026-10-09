"""Ancestral sequence reconstruction with IQ-TREE (ML tree, marginal states with posteriors) plus Fitch parsimony
for indels, ancestor picking, uncertain sites, AltAll variants and comparison with extant sequences. No Streamlit."""
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from io import StringIO

import numpy as np
import pandas as pd

IQTREE_NAMES = ("iqtree2", "iqtree3", "iqtree")
IQTREE_TIMEOUT = 900
MIN_SEQUENCES, MAX_SEQUENCES = 4, 300
MODELS = {"Protein": {"LG+G4 (fast)": "LG+G4", "Best fit (ModelFinder, slower)": "MFP"},
          "DNA": {"GTR+G4 (fast)": "GTR+G4", "Best fit (ModelFinder, slower)": "MFP"}}
_INSTALL = "On Streamlit Cloud add `iqtree` to packages.txt; locally run `conda install -c bioconda iqtree`."
INSTALL_HINT = f"IQ-TREE is not installed. {_INSTALL}"
OLD_VERSION_HINT = "IQ-TREE 2 or newer is required (found: {line}). " + _INSTALL
GAP_CHARS = "-."
UNCERTAIN_COLUMNS = ["Ancestor position", "Alignment column", "ML residue", "Posterior", "Alternatives"]


class ASRError(RuntimeError):
    """Raised when IQ-TREE is missing, fails or times out, or an ancestor cannot be picked."""


@dataclass
class IQTreeResult:
    tree: object                         # Bio.Phylo tree, internal clades named NodeN (supports in .confidence)
    states: dict                         # node -> (ML states str, posteriors n_sites x n_alphabet)
    alphabet: str
    model: str
    loglik: float | None
    version: str
    files: dict = field(default_factory=dict)  # file name -> bytes (treefile, state, iqtree report)


def iqtree_major(version_line):
    """Major version from an IQ-TREE --version line ('IQ-TREE multicore version 2.0.3 ...' -> 2), else None."""
    m = re.search(r"version\s+(\d+)\.", version_line or "")
    return int(m.group(1)) if m else None


def iqtree_status():
    """((path, version line), None) for the first IQ-TREE 2+ binary on PATH whose --version output mentions
    IQ-TREE, else (None, message): not installed, or only IQ-TREE 1.x found (its -asr output differs)."""
    old = None
    for name in IQTREE_NAMES:
        path = shutil.which(name)
        path = os.path.abspath(path) if path else None
        if not path:
            continue
        try:
            out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        line = next((l.strip() for l in out.splitlines() if "IQ-TREE" in l), None)
        if not line:
            continue
        major = iqtree_major(line)
        if major is not None and major < 2:
            old = old or line
            continue
        return (path, line), None
    return None, (OLD_VERSION_HINT.format(line=old) if old else INSTALL_HINT)


def find_iqtree():
    """First IQ-TREE 2+ binary on PATH, as (path, version line), or None."""
    return iqtree_status()[0]


def missing_hint():
    """Why no usable IQ-TREE was found (install hint, or the version requirement)."""
    return iqtree_status()[1] or INSTALL_HINT


def _failure_lines(log, n=3):
    """The last n ERROR/WARNING lines of an IQ-TREE log, or its last non-empty line."""
    lines = [l.strip() for l in log.splitlines() if l.strip()]
    flagged = [l for l in lines if l.startswith(("ERROR", "WARNING"))]
    return "; ".join(flagged[-n:]) if flagged else (lines[-1] if lines else "unknown error")


def parse_state(text):
    """IQ-TREE .state file -> (alphabet, {node: (states, posteriors)}). Posteriors are float32 n_sites x n_states."""
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    header = lines[0].split("\t")
    alphabet = "".join(h[2:] for h in header[3:])
    rows = {}
    for line in lines[1:]:
        parts = line.split("\t")
        rows.setdefault(parts[0], []).append(parts)
    states = {}
    for node, parts in rows.items():
        parts.sort(key=lambda p: int(p[1]))
        states[node] = ("".join(p[2] for p in parts), np.array([[float(x) for x in p[3:]] for p in parts],
                                                                 dtype=np.float32))
    return alphabet, states


def parse_tree(text):
    """IQ-TREE .treefile -> Bio.Phylo tree; internal labels 'NodeN' or 'NodeN/support' are split into name and
    confidence (the support, when present)."""
    from Bio import Phylo

    tree = Phylo.read(StringIO(text), "newick")
    for clade in tree.find_clades():
        label = clade.name
        if clade.is_terminal() or not label:
            continue
        name, _, support = label.partition("/")
        clade.name = name
        clade.confidence = float(support) if support else clade.confidence
    return tree


def parse_report(text):
    """(model, log-likelihood) from an IQ-TREE .iqtree report."""
    model = None
    for key in ("Best-fit model according to BIC:", "Model of substitution:"):
        m = re.search(re.escape(key) + r"\s*(\S+)", text)
        if m:
            model = m.group(1)
            break
    m = re.search(r"Log-likelihood of the tree:\s*(-?[\d.]+)", text)
    return model, (float(m.group(1)) if m else None)


def run_iqtree(aligned_pairs, seq_type="Protein", model="LG+G4", bootstrap=0, timeout=IQTREE_TIMEOUT):
    """Run IQ-TREE with ancestral state reconstruction (-asr) on aligned (id, sequence) pairs."""
    found = find_iqtree()
    if found is None:
        raise ASRError(missing_hint())
    exe, version = found
    if not MIN_SEQUENCES <= len(aligned_pairs) <= MAX_SEQUENCES:
        raise ASRError(f"Ancestral reconstruction needs {MIN_SEQUENCES}-{MAX_SEQUENCES} sequences "
                       f"(got {len(aligned_pairs)}).")
    placeholders = {f"s{k:04d}": seq_id for k, (seq_id, _) in enumerate(aligned_pairs, 1)}
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "aln.fasta"), "w") as handle:
            for ph, (_, seq) in zip(placeholders, aligned_pairs):
                handle.write(f">{ph}\n{seq.replace('.', '-')}\n")
        cmd = [exe, "-s", "aln.fasta", "-st", "AA" if seq_type == "Protein" else "DNA", "-m", model, "-asr",
               "-T", "1", "--prefix", "run", "-quiet", "-redo"]
        if bootstrap:
            cmd += ["-B", str(int(bootstrap))]
        try:
            done = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise ASRError(f"IQ-TREE did not finish within {timeout // 60} min; use fewer sequences or "
                           "LG+G4 instead of model selection.") from e

        def read(name, mode="r"):
            path = os.path.join(tmp, name)
            if not os.path.exists(path):
                return None
            with open(path, mode) as handle:
                return handle.read()
        if done.returncode != 0 or read("run.state") is None:
            log = read("run.log") or done.stderr or done.stdout or ""
            raise ASRError(f"IQ-TREE failed: {_failure_lines(log)}")
        tree_text, state_text, report = read("run.treefile"), read("run.state"), read("run.iqtree") or ""
        files = {f"iqtree.{ext}": read(f"run.{ext}", "rb") for ext in ("treefile", "state", "iqtree", "log")
                 if read(f"run.{ext}", "rb") is not None}
    tree = parse_tree(tree_text)
    for clade in tree.get_terminals():
        clade.name = placeholders.get(clade.name, clade.name)
    alphabet, states = parse_state(state_text)
    check_site_count(states, len(aligned_pairs[0][1]))
    model_used, loglik = parse_report(report)
    return IQTreeResult(tree, states, alphabet, model_used or model, loglik, version, files)


def reconstruct_gaps(tree, aligned_pairs):
    """Fitch parsimony on residue presence per column over a rooted tree -> {internal node name: bool[n_sites]}.
    At the root a tie resolves to 'present' when most leaves have a residue in that column."""
    aligned = dict(aligned_pairs)
    present = {k: np.array([c not in GAP_CHARS for c in s]) for k, s in aligned.items()}
    majority = np.mean(list(present.values()), axis=0) >= 0.5
    sets = {}

    def up(clade):                                  # returns (can_absent, can_present)
        if clade.is_terminal():
            p = present[clade.name]
            sets[id(clade)] = (~p, p)
            return sets[id(clade)]
        children = [up(c) for c in clade.clades]
        a = np.logical_and.reduce([c[0] for c in children])
        b = np.logical_and.reduce([c[1] for c in children])
        empty = ~a & ~b
        a = np.where(empty, np.logical_or.reduce([c[0] for c in children]), a)
        b = np.where(empty, np.logical_or.reduce([c[1] for c in children]), b)
        sets[id(clade)] = (a, b)
        return sets[id(clade)]
    up(tree.root)
    out = {}

    def down(clade, parent_state):
        a, b = sets[id(clade)]
        if parent_state is None:
            state = np.where(a & b, majority, b)
        else:
            state = np.where(parent_state, b, ~a)     # keep the parent's state when allowed
        if not clade.is_terminal():
            if clade.name:
                out[clade.name] = state
            for child in clade.clades:
                down(child, state)
    down(tree.root, None)
    return out


def ancestor_options(tree, states):
    """[(node name, number of descendant sequences, label)] for every internal node that has states."""
    options = []
    for clade in tree.get_nonterminals():
        if clade.name in states:
            leaves = [t.name for t in clade.get_terminals()]
            more = f", +{len(leaves) - 3}" if len(leaves) > 3 else ""
            options.append((clade.name, len(leaves), f"{clade.name}: {len(leaves)} sequences "
                                                     f"({', '.join(leaves[:3])}{more})"))
    return sorted(options, key=lambda o: o[1])


def mrca_node(tree, leaf_names, states):
    """Name of the most recent common ancestor of the chosen leaves; a selection spanning the root raises."""
    if len(set(leaf_names)) < 2:
        raise ASRError("Choose at least two sequences: a single sequence is not an ancestor.")
    clade = tree.common_ancestor(list(leaf_names))
    if clade.name not in states:
        raise ASRError("The common ancestor of these sequences is the root of the displayed tree, which IQ-TREE "
                       "does not reconstruct (it reconstructs the internal nodes of the unrooted tree). Choose an "
                       "outgroup and select the remaining sequences, or pick a node from the list.")
    return clade.name


def check_site_count(states, n_columns):
    """Every node's state string must cover the alignment columns."""
    for name, (seq, _) in states.items():
        if len(seq) != n_columns:
            raise ASRError(f"IQ-TREE returned {len(seq)} sites for an alignment of {n_columns} columns; "
                           "cannot map ancestral states.")


def ancestor_sequence(result, gaps, node):
    """(sequence, posterior of each residue, alignment columns) of an ancestor, without absent columns."""
    states, post = result.states[node]
    keep = np.flatnonzero(gaps[node]) if node in gaps else np.arange(len(states))
    alphabet = result.alphabet
    seq = "".join(states[c] for c in keep)
    pp = np.array([post[c, alphabet.index(states[c])] if states[c] in alphabet else 0.0 for c in keep])
    return seq, pp, keep


def uncertain_sites(result, gaps, node, max_posterior=0.8, min_alternative=0.2):
    """Sites whose ML residue has posterior < max_posterior, with alternatives >= min_alternative."""
    seq, pp, cols = ancestor_sequence(result, gaps, node)
    post = result.states[node][1]
    rows = []
    for k, (c, p) in enumerate(zip(cols, pp)):
        if p >= max_posterior:
            continue
        alts = [(result.alphabet[i], float(post[c, i])) for i in np.argsort(-post[c])
                if result.alphabet[i] != seq[k] and post[c, i] >= min_alternative]
        rows.append([k + 1, int(c) + 1, seq[k], round(float(p), 3),
                     ", ".join(f"{a} ({v:.2f})" for a, v in alts)])
    return pd.DataFrame(rows, columns=UNCERTAIN_COLUMNS)


def altall_sequence(result, gaps, node, max_posterior=0.8, min_alternative=0.2):
    """AltAll: every uncertain site switched to its best alternative (>= min_alternative)."""
    seq, pp, cols = ancestor_sequence(result, gaps, node)
    post = result.states[node][1]
    out = list(seq)
    for k, (c, p) in enumerate(zip(cols, pp)):
        if p >= max_posterior:
            continue
        for i in np.argsort(-post[c]):
            a = result.alphabet[i]
            if a != seq[k] and post[c, i] >= min_alternative:
                out[k] = a
                break
    return "".join(out)


def compare_to_extant(result, gaps, node, aligned_pairs, extant_id):
    """Differences ancestor -> extant over alignment columns, in the extant's numbering:
    (substitutions [(label, column)], insertions in the extant, deletions in the extant, identity)."""
    states = result.states[node][0]
    present = gaps.get(node, np.ones(len(states), bool))
    extant = dict(aligned_pairs)[extant_id]
    letters = set(extant.upper())
    if "U" in letters and "T" not in letters:          # RNA input: IQ-TREE writes DNA states (T for U)
        states = states.replace("T", "U")
    subs, ins, dels, same, both, pos = [], 0, 0, 0, 0, 0
    for c, (a, e) in enumerate(zip(states, extant)):
        e_res = e not in GAP_CHARS
        pos += e_res
        if present[c] and e_res:
            both += 1
            if a == e.upper():
                same += 1
            else:
                subs.append((f"{a}{pos}{e.upper()}", c))
        elif e_res:
            ins += 1
        elif present[c]:
            dels += 1
    return subs, ins, dels, (same / both if both else 0.0)
