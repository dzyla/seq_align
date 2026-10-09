# Coevolution Contacts and Ancestral Reconstruction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two pages to seq_align, a **Coevolution** page and an **Ancestral Reconstruction** page, plus MSA colour schemes and a sequence viewer (Task 5). Everything must work on Streamlit Cloud.

**Coevolution page:**
- Computes plmDCA or mean-field DCA contacts from the current dataset or from an uploaded ColabFold `.a3m`, Stockholm or aligned-FASTA file.
- Shows a contact map, the precision against a structure, a top-pairs table and links drawn in Mol\*.

**Ancestral Reconstruction page:**
- Runs IQ-TREE marginal reconstruction with per-residue posteriors and reconstructs indels by Fitch parsimony.
- Shows uncertain sites, offers ML and AltAll FASTA, and compares an ancestor with a present-day sequence on a structure.

**Architecture:**
- **Pure modules:** `modules/coevolution.py` (parsing, reweighting, DCA, distances, precision) and `modules/ancestral.py` (IQ-TREE run and parsing, Fitch indels, ancestor picking, uncertainty, AltAll, comparison).
- **Streamlit pages:** `modules/coevolution_page.py` and `modules/ancestral_page.py`.
- **System binary:** IQ-TREE comes from the Debian `iqtree` package, through `packages.txt` on the cloud and the CI `apt-get` line. CI then exercises the same binary the cloud will run.

**Tech Stack:** numpy, scipy (`scipy.linalg.inv` in place), plotly, Biopython (Phylo), MolViewSpec primitives (`tube`), IQ-TREE 2/3 (binary `iqtree2`, `iqtree3` or `iqtree`), Streamlit 1.60.0.

**Spec:** `docs/superpowers/specs/2026-10-09-coevolution-and-ancestral-reconstruction-design.md`

**Prototype provenance:** every code block and test below comes from a prototype on a scratch branch.
- **Suite:** 476 passed and 2 skipped, the skips being the IQ-TREE tests without IQ-TREE on PATH. With IQ-TREE 2.0.3 on PATH and `CI=1`, every test passed. `precommit_check.py` passed.
- **Coevolution on real data:** the ColabFold A3M of human HBB (3,973 sequences), compared with 4HHB chain B.
  - plmDCA: P@L/5 = 0.66 in 7–10 s.
  - Mean-field DCA: P@L/5 = 0.41 in 0.4 s.
- **Cloud caps, measured:**
  - plmDCA with L = 200, N = 5,000: 0.98 GB peak.
  - Mean-field DCA with L = 400, N = 10,000: 0.71 GB peak, after switching to `scipy.linalg.inv(overwrite_a=True)`. `np.linalg.inv` peaked at 2.4 GB.
- **IQ-TREE:** 2.0.3 and 3.1.4 produce identical `.state` and `.treefile` formats.
- **Bugs the prototype caught (already fixed in the code below):**
  - A relative binary path broke runs in the temporary directory, so `find_iqtree` returns absolute paths.
  - Two widget-key collisions: the store for the uploaded alignment is `coevo_alignment`, because the structure controls own `coevo_upload`; and the ASR tree section uses the prefix `asr_tree`.

Transcribe the code exactly.

## Global Constraints

- Streamlit stays pinned at 1.60.0. The new system dependency is `iqtree` in `packages.txt` and the CI `apt-get` line. There are no new Python packages.
- Cloud caps live in code and are stated in the UI:
  - plmDCA: ≤ 200 columns, ≤ 5,000 sequences, 30 iterations by default.
  - Mean-field DCA: ≤ 400 columns, ≤ 10,000 sequences.
  - IQ-TREE: 4–300 sequences, `-T 1`, 15-minute timeout.
- Pure computation lives in functions without `st.*` and is unit-tested. Streamlit code lives in `*_section` functions.
- New figures use plotly through `modules.ui.show_plotly`. No pyplot.
- Tests never need the network. IQ-TREE and DSSP tests skip when the binary is absent. `test_ci_has_the_cloud_system_binaries` enforces both binaries in CI.
- No emoji characters in source (`tests/test_ui.py::test_no_emoji_in_app_source`).
- Tests run with `PATH=/home/dzyla/miniconda3/bin:$PATH /home/dzyla/seq_align/.venv/bin/python -m pytest -q`. Run `.venv/bin/python precommit_check.py` before every commit and never commit when it fails.
- Every commit message ends with `Co-Authored-By: <the model that wrote the code> <noreply@anthropic.com>`.
- Import cycles:
  - `structure_view → pdb_renumber → pairwise`.
  - `coevolution` imports `pdb_renumber` and `structures` inside functions.
  - The pages import `structure_view`, `msa`, `phylogeny` and `tree_layout` inside functions.

## Review Focus

Each of these risky inputs has a test in the named task:

1. **A3M with corrupted lines** (sequences of a different length): they are dropped and counted, never crash (Task 1).
2. **A query longer than the method's column cap:** the page asks for a region instead of running out of memory (Task 2).
3. **A shallow alignment** (a few sequences): the Neff/L error is shown, with no crash (Task 2).
4. **IQ-TREE missing on the machine:** the page shows the install hint (Tasks 3 and 4).
5. **An outgroup that cannot root the tree** (all leaves): the page falls back to midpoint rooting with a warning (Task 4).

---

### Task 1: Coevolution core (`modules/coevolution.py`)

**Files:**
- Create: `modules/coevolution.py`
- Test: `tests/test_coevolution.py`

**Interfaces:**
- Produces:
  - Constants:
    - `ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"`, `GAP = 20`, `Q = 21`
    - `MF_MAX_COLUMNS = 400`, `MF_MAX_SEQUENCES = 10000`, `PLM_MAX_COLUMNS = 200`, `PLM_MAX_SEQUENCES = 5000`, `PLM_ITERATIONS = 30`, `CONTACT_CUTOFF = 8.0`
    - `METHODS = {"plmDCA (accurate)": "plm", "Mean-field DCA (fast)": "mf"}`
    - `LIMITS = {"plm": (200, 5000), "mf": (400, 10000)}`
  - `PreparedMSA(msa, query_seq, n_input, n_dropped_length, n_dropped_gappy, n_subsampled)`
  - Alignment handling:
    - `parse_alignment(text) -> [(id, seq)]`
    - `prepare_msa(rows, query=0, max_seq_gap=0.5, max_seqs=MF_MAX_SEQUENCES, seed=0) -> PreparedMSA`
    - `sequence_weights(msa, theta=0.8, budget=150_000_000) -> np.ndarray`
  - Scoring:
    - `mean_field_dca(msa, w, pseudo=0.5) -> L×L`
    - `plm_dca(msa, w, iters=30, history=3) -> L×L`
    - `top_pairs(scores, min_sep=6, k=None) -> [(i, j, score)]`, 0-based
    - `contact_precision(scores, distances, min_sep=6, top_k=None, cutoff=8.0) -> float`
  - Structure mapping:
    - `query_to_chain(query_seq, chain_seq) -> {query index: chain index}`
    - `residue_coordinates(sf, chain_id)`
    - `structure_distances(sf, chain_id, query_seq) -> (L×L distances, mapping, L×3 query coordinates)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_coevolution.py`:

```python
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from modules.coevolution import (ALPHABET, MF_MAX_COLUMNS, PLM_MAX_COLUMNS, contact_precision, mean_field_dca,
                                 parse_alignment, plm_dca, prepare_msa, sequence_weights, structure_distances,
                                 top_pairs)
from modules.structures import StructureFile

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"


def test_a3m_insertions_removed_and_query_first():
    rows = parse_alignment(">query\nMKV-L\n>hit1 desc\nMkkKVAL\n>hit2\nM.KV-L\n")
    assert rows == [("query", "MKV-L"), ("hit1", "MKVAL"), ("hit2", "MKV-L")]


def test_stockholm_blocks_concatenate():
    text = "# STOCKHOLM 1.0\nseqA MKV\nseqB MRV\n#=GC SS_cons ...\n\nseqA LLE\nseqB LIE\n//\n"
    assert parse_alignment(text) == [("seqA", "MKVLLE"), ("seqB", "MRVLIE")]


def test_prepare_drops_query_gaps_length_mismatch_and_gappy_rows():
    rows = [("q", "M-KVL"), ("a", "MAKVL"), ("short", "MKV"), ("gappy", "--K--"), ("b", "MRKIL")]
    p = prepare_msa(rows)
    assert p.query_seq == "MKVL" and p.n_dropped_length == 1 and p.n_dropped_gappy == 1
    assert p.msa.shape == (3, 4) and "".join(ALPHABET[k] for k in p.msa[1]) == "MKVL"


def test_query_kept_when_gappy_and_subsampling_is_deterministic():
    rows = [("q", "M----")] + [(f"s{i}", "MKVLA") for i in range(50)]
    p1 = prepare_msa(rows, max_seqs=10, seed=1)
    p2 = prepare_msa(rows, max_seqs=10, seed=1)
    assert p1.query_seq == "M" and len(p1.msa) == 10 and p1.n_subsampled == 41
    assert np.array_equal(p1.msa, p2.msa)


def test_sequence_weights_by_hand():
    # a, b identical; c differs at 3 of 4 positions -> weights 1/2, 1/2, 1
    msa = prepare_msa([("a", "MKVL"), ("b", "MKVL"), ("c", "MRIA")]).msa
    assert sequence_weights(msa, theta=0.8).tolist() == [0.5, 0.5, 1.0]


def _coupled_msa(n=2000, L=30, i=5, j=20, seed=0):
    rng = np.random.default_rng(seed)
    msa = rng.integers(0, 20, size=(n, L)).astype(np.int8)
    pairs = np.array([[0, 5], [3, 9], [11, 2], [17, 14]], dtype=np.int8)
    pick = pairs[rng.integers(0, len(pairs), size=n)]
    msa[:, i], msa[:, j] = pick[:, 0], pick[:, 1]
    return msa


@pytest.mark.parametrize("method", [mean_field_dca, plm_dca])
def test_synthetic_coupling_ranked_first(method):
    msa = _coupled_msa()
    scores = method(msa, sequence_weights(msa))
    assert top_pairs(scores, min_sep=1, k=1)[0][:2] == (5, 20)


def test_contact_precision_on_toy_matrices():
    scores = np.zeros((10, 10))
    scores[0, 9] = scores[9, 0] = 3.0
    scores[1, 8] = scores[8, 1] = 2.0
    D = np.full((10, 10), 20.0)
    D[0, 9] = D[9, 0] = 5.0
    D[1, 8] = D[8, 1] = np.nan          # unmodelled pairs are skipped
    assert contact_precision(scores, D, min_sep=6, top_k=1) == 1.0
    assert contact_precision(scores, D, min_sep=6, top_k=2) == 0.5


def test_structure_distances_crambin_helix():
    sf = StructureFile(name="1crn.pdb", data=(DATA / "1crn.pdb").read_bytes(), fmt="PDB")
    query = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
    D, mapping, coords = structure_distances(sf, "A", query)
    assert D.shape == (46, 46) and len(mapping) == 46
    assert 4.0 < D[7, 11] < 7.5              # i, i+4 within the first helix (residues 7-19)
    assert np.allclose(np.diag(D), 0)


def test_limits_are_cloud_sized():
    assert PLM_MAX_COLUMNS == 200 and MF_MAX_COLUMNS == 400


@pytest.mark.parametrize("method,L,n,limit_gb", [("plm_dca", 200, 1000, 1.5), ("mean_field_dca", 400, 2000, 1.5)])
def test_memory_stays_within_cloud_budget(method, L, n, limit_gb):
    code = (f"import sys, resource, numpy as np; sys.path.insert(0, {str(ROOT)!r});"
            f"from modules.coevolution import {method}, sequence_weights;"
            f"msa = np.random.default_rng(0).integers(0, 21, size=({n}, {L})).astype(np.int8);"
            f"{method}(msa, sequence_weights(msa));"
            "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert int(done.stdout.strip().splitlines()[-1]) / 1e6 < limit_gb
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_coevolution.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.coevolution'`.

- [ ] **Step 3: Implement `modules/coevolution.py`**

```python
"""Coevolution contacts from a deep MSA: alignment parsing, reweighting, mean-field and pseudo-likelihood DCA,
structure distances and contact precision. Pure functions (no Streamlit); memory is bounded for Streamlit Cloud."""
import re
from dataclasses import dataclass

import numpy as np
import scipy.linalg

ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
GAP = 20
Q = 21
_INDEX = np.full(256, GAP, dtype=np.int8)  # anything unknown counts as a gap
for _k, _c in enumerate(ALPHABET):
    _INDEX[ord(_c)] = _k

# Streamlit Cloud caps (<= 2.7 GB RAM, ~2 shared cores); stated in the UI
MF_MAX_COLUMNS, MF_MAX_SEQUENCES = 400, 10000
PLM_MAX_COLUMNS, PLM_MAX_SEQUENCES = 200, 5000
PLM_ITERATIONS = 30
CONTACT_CUTOFF = 8.0
METHODS = {"plmDCA (accurate)": "plm", "Mean-field DCA (fast)": "mf"}
LIMITS = {"plm": (PLM_MAX_COLUMNS, PLM_MAX_SEQUENCES), "mf": (MF_MAX_COLUMNS, MF_MAX_SEQUENCES)}


@dataclass
class PreparedMSA:
    msa: np.ndarray          # int8, N x L (row 0 = query), states 0..20 (20 = gap)
    query_seq: str           # query residues of the kept columns
    n_input: int
    n_dropped_length: int
    n_dropped_gappy: int
    n_subsampled: int        # sequences removed by subsampling


def parse_alignment(text):
    """A3M (lowercase insertions and '.' removed), Stockholm, or aligned FASTA -> [(id, aligned sequence)].
    Sequences whose length differs from the first are kept here and dropped (and counted) by prepare_msa."""
    if text.lstrip().startswith("# STOCKHOLM"):
        seqs = {}
        for line in text.splitlines():
            if not line.strip() or line.startswith(("#", "//")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                seqs[parts[0]] = seqs.get(parts[0], "") + parts[1]
        rows = list(seqs.items())
    else:
        rows, name, buf = [], None, []
        for line in text.splitlines():
            if line.startswith(">"):
                if name is not None:
                    rows.append((name, "".join(buf)))
                header = line[1:].split()
                name, buf = (header[0] if header else f"seq{len(rows) + 1}"), []
            elif line.strip() and not line.startswith("#"):
                buf.append(line.strip())
        if name is not None:
            rows.append((name, "".join(buf)))
    return [(i, re.sub(r"[a-z.]", "", s)) for i, s in rows]


def prepare_msa(rows, query=0, max_seq_gap=0.5, max_seqs=MF_MAX_SEQUENCES, seed=0):
    """Encode, keep the query's residue columns, drop gappy sequences (query kept), subsample with a fixed seed."""
    if not rows:
        raise ValueError("The alignment is empty.")
    query_row = rows[query]
    L0 = len(query_row[1])
    same = [r for r in rows if len(r[1]) == L0 and r is not query_row]
    seqs = [query_row[1].upper()] + [s.upper() for _, s in same]
    n_input, n_dropped_length = len(rows), len(rows) - 1 - len(same)
    keep = [j for j, c in enumerate(seqs[0]) if c not in "-."]
    if not keep:
        raise ValueError("The query sequence has no residues.")
    arr = np.frombuffer("".join(seqs).encode("ascii", "replace"), dtype=np.uint8).reshape(len(seqs), L0)
    msa = _INDEX[arr][:, keep]
    gappy = (msa == GAP).mean(axis=1) > max_seq_gap
    gappy[0] = False
    n_dropped_gappy = int(gappy.sum())
    msa = msa[~gappy]
    n_subsampled = 0
    if len(msa) > max_seqs:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(np.arange(1, len(msa)), max_seqs - 1, replace=False))
        n_subsampled = len(msa) - max_seqs
        msa = msa[np.concatenate([[0], chosen])]
    query_seq = "".join(ALPHABET[k] for k in msa[0])
    return PreparedMSA(msa, query_seq, n_input, n_dropped_length, n_dropped_gappy, n_subsampled)


def sequence_weights(msa, theta=0.8, budget=150_000_000):
    """1 / number of sequences >= theta identical (over all L columns, gaps included). Rows are compared in chunks
    sized so one chunk's boolean comparison stays within `budget` bytes."""
    n, L = msa.shape
    chunk = max(1, int(budget // max(1, n * L)))
    counts = np.zeros(n)
    threshold = theta * L
    for a in range(0, n, chunk):
        same = (msa[a:a + chunk, None, :] == msa[None, :, :]).sum(axis=2, dtype=np.int32)
        counts[a:a + chunk] = (same >= threshold).sum(axis=1)
    return 1.0 / counts


def _one_hot(msa):
    n, L = msa.shape
    X = np.zeros((n, L * Q), dtype=np.float32)
    X[np.arange(n)[:, None], np.arange(L) * Q + msa] = 1.0
    return X


def _frobenius_apc(J):
    """J: L x 20 x L x 20 couplings (float32 ok) -> APC-corrected Frobenius norms. Zero-sum gauge in place (the
    three means are taken from the original J), then the norm via einsum so no squared copy is made."""
    m1 = J.mean(axis=1, keepdims=True)
    m3 = J.mean(axis=3, keepdims=True)
    m13 = J.mean(axis=(1, 3), keepdims=True)
    J -= m1
    J -= m3
    J += m13
    fn = np.sqrt(np.einsum("iajb,iajb->ij", J, J, dtype=np.float64))
    fn = (fn + fn.T) / 2
    np.fill_diagonal(fn, 0)
    apc = fn - np.outer(fn.mean(axis=0), fn.mean(axis=1)) / max(fn.mean(), 1e-12)
    np.fill_diagonal(apc, 0)
    return apc


def mean_field_dca(msa, w, pseudo=0.5):
    """Mean-field DCA (Morcos et al. 2011): couplings = -C^-1 of the reweighted, pseudocounted covariance.
    float32 and in-place steps keep the peak near 1.2 GB at the MF caps."""
    n, L = msa.shape
    X = _one_hot(msa)
    wf = (w / float(w.sum())).astype(np.float32)
    fi = wf @ X
    X *= np.sqrt(wf)[:, None]
    fij = X.T @ X
    del X
    fi = (1 - pseudo) * fi + pseudo / Q
    fij *= (1 - pseudo)
    fij += pseudo / Q ** 2
    for i in range(L):
        s = slice(i * Q, (i + 1) * Q)
        fij[s, s] = np.diag(fi[s])
    idx = (np.arange(L)[:, None] * Q + np.arange(Q - 1)[None, :]).ravel()   # drop the gap state
    C = fij[np.ix_(idx, idx)]
    del fij
    C -= np.outer(fi[idx], fi[idx])
    J = scipy.linalg.inv(C, overwrite_a=True, check_finite=False)   # in place: np.linalg.inv needs ~3x more
    del C
    J *= -1
    return _frobenius_apc(J.reshape(L, Q - 1, L, Q - 1))


def plm_dca(msa, w, iters=PLM_ITERATIONS, history=3, lam_h=0.01, lam_j_factor=0.2):
    """Symmetric pseudo-likelihood DCA (CCMpred-style regularisation), float32 upper-triangle couplings and a
    small L-BFGS with Armijo backtracking: memory stays under ~1.5 GB at the PLM caps."""
    n, L = msa.shape
    Lq = L * Q
    X = _one_hot(msa)
    w = w.astype(np.float32)
    lam_j = np.float32(lam_j_factor * (L - 1))
    block = np.repeat(np.arange(L, dtype=np.int32), Q)
    rows_, cols_ = np.triu_indices(Lq, 1)
    keep = block[rows_] != block[cols_]
    rows_, cols_ = rows_[keep].astype(np.int32), cols_[keep].astype(np.int32)
    del keep
    up = rows_.astype(np.int64) * Lq + cols_
    lo = cols_.astype(np.int64) * Lq + rows_
    del rows_, cols_
    nh = Lq
    W = np.zeros((Lq, Lq), dtype=np.float32)
    Wv = W.reshape(-1)

    def objective(theta):
        h = theta[:nh]
        Wv[up] = theta[nh:]
        Wv[lo] = theta[nh:]
        P = X @ W
        P += h
        P = P.reshape(n, L, Q)
        P -= P.max(axis=2, keepdims=True)
        np.exp(P, out=P)
        P /= P.sum(axis=2, keepdims=True)
        P = P.reshape(n, Lq)
        ll = np.log(np.maximum((P * X).reshape(n, L, Q).sum(axis=2), 1e-30)).sum(axis=1)
        value = -float((w * ll).sum()) + float(lam_h * (h ** 2).sum()) + float(lam_j * (theta[nh:] ** 2).sum())
        P -= X
        P *= w[:, None]                           # d(objective) / d(logits)
        gW = (X.T @ P).reshape(-1)
        grad = np.empty_like(theta)
        grad[:nh] = P.sum(axis=0) + 2 * lam_h * h
        grad[nh:] = gW[up] + gW[lo] + 2 * lam_j * theta[nh:]
        return value, grad

    theta = np.zeros(nh + up.size, dtype=np.float32)
    f, g = objective(theta)
    S, Y = [], []
    for _ in range(iters):
        q = g.copy()
        alphas = []
        for s, y in reversed(list(zip(S, Y))):
            a = float(s @ q) / float(y @ s)
            alphas.append(a)
            q -= a * y
        if S:
            q *= float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])
        else:
            q /= max(1.0, float(np.abs(g).max()))
        for (s, y), a in zip(zip(S, Y), reversed(alphas)):
            q += (a - float(y @ q) / float(y @ s)) * s
        d = -q
        slope = float(g @ d)
        if slope >= 0:                            # not a descent direction: restart from the gradient
            S, Y = [], []
            d = -g / max(1.0, float(np.abs(g).max()))
            slope = float(g @ d)
        step = 1.0
        for _ in range(20):
            trial = theta + step * d
            f2, g2 = objective(trial)
            if f2 <= f + 1e-4 * step * slope:
                break
            step *= 0.5
        s, y = trial - theta, g2 - g
        if float(s @ y) > 1e-10:
            S.append(s)
            Y.append(y)
            if len(S) > history:
                S.pop(0)
                Y.pop(0)
        theta, f, g = trial, f2, g2
    Wv[up] = theta[nh:]
    Wv[lo] = theta[nh:]
    J = np.ascontiguousarray(W.reshape(L, Q, L, Q)[:, :Q - 1, :, :Q - 1])
    del W
    return _frobenius_apc(J)


def top_pairs(scores, min_sep=6, k=None):
    """[(i, j, score)] for i < j with j - i >= min_sep, best first (0-based query positions)."""
    L = scores.shape[0]
    i, j = np.triu_indices(L, min_sep)
    order = np.argsort(-scores[i, j], kind="stable")
    if k is not None:
        order = order[:k]
    return [(int(i[o]), int(j[o]), float(scores[i[o], j[o]])) for o in order]


def contact_precision(scores, distances, min_sep=6, top_k=None, cutoff=CONTACT_CUTOFF):
    """Fraction of the top_k pairs (|i-j| >= min_sep, both positions modelled) closer than cutoff (A)."""
    L = scores.shape[0]
    pairs = [(i, j) for i, j, _ in top_pairs(scores, min_sep) if not np.isnan(distances[i, j])]
    pairs = pairs[:top_k or L]
    if not pairs:
        return float("nan")
    return sum(distances[i, j] < cutoff for i, j in pairs) / len(pairs)


def query_to_chain(query_seq, chain_seq):
    """Map 0-based query positions to 0-based chain residue indices via a semi-global alignment."""
    from modules.pdb_renumber import align_target_to_chain  # local: pdb_renumber imports pairwise

    aligned_q, aligned_c, _ = align_target_to_chain(query_seq.replace("-", "X"), chain_seq)
    mapping, qi, ci = {}, 0, 0
    for a, b in zip(aligned_q, aligned_c):
        if a != "-" and b != "-":
            mapping[qi] = ci
        qi += a != "-"
        ci += b != "-"
    return mapping


def residue_coordinates(sf, chain_id):
    """(chain sequence, residues, N x 3 coordinates of CB, CA for Gly, NaN if missing) for one protein chain."""
    from modules.structures import chain_residue_table, parse_structure

    structure = parse_structure(sf)
    residues = chain_residue_table(structure)[chain_id]
    chain = list(structure)[0][chain_id]
    coords = np.full((len(residues), 3), np.nan)
    for k, r in enumerate(residues):
        key = (" ", r.number, r.icode or " ")
        res = chain[key] if key in chain else None
        if res is None:
            hetero = [x for x in chain if x.id[1] == r.number and (x.id[2].strip() == r.icode)]
            res = hetero[0] if hetero else None
        if res is not None:
            atom = res["CB"] if "CB" in res else (res["CA"] if "CA" in res else None)
            if atom is not None:
                coords[k] = atom.coord
    return "".join(r.aa for r in residues), residues, coords


def structure_distances(sf, chain_id, query_seq):
    """L x L CB-CB distances (CA for Gly) between query positions mapped onto a structure chain; NaN if unmapped.
    Also returns the query -> chain index map and the coordinates for drawing links."""
    chain_seq, residues, coords = residue_coordinates(sf, chain_id)
    mapping = query_to_chain(query_seq, chain_seq)
    L = len(query_seq)
    q_coords = np.full((L, 3), np.nan)
    for qi, ci in mapping.items():
        q_coords[qi] = coords[ci]
    diff = q_coords[:, None, :] - q_coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2)), mapping, q_coords
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_coevolution.py -q`
Expected: 12 passed. The two memory tests run subprocesses and take about 30 s. Then run the full suite and precommit: green.

- [ ] **Step 5: Commit**

```bash
git add modules/coevolution.py tests/test_coevolution.py
git commit -m "feat: coevolution core (A3M/Stockholm/FASTA parsing, reweighting, plmDCA and mean-field DCA within cloud memory)"
```

---

### Task 2: Coevolution page

**Files:**
- Create: `modules/coevolution_page.py`
- Modify: `modules/navigation.py`, `app.py`, `tests/test_navigation.py`
- Test: `tests/test_app_flow.py` (append)

**Interfaces:**
- Consumes: everything from Task 1. Also, imported inside functions: `modules.msa` (`MSA_METHODS`, `MSAError`, `cached_run_msa`, `mafft_available`), `modules.structure_view` (`_add_structure_controls`, `available_structures`, `cached_residue_table`, `rank_chains`, `build_view_state`, `render_molstar`) and `modules.ui` (`page_header`, `show_plotly`).
- Produces: `coevolution_section(sequences, seq_type, standalone=False)`.
- Session keys:
  - `coevo_alignment` holds the uploaded file as `(name, bytes)`. **Do not name it `coevo_upload`:** the structure controls own that key.
  - `coevo_result` holds `{"params", "result"}`.
  - `coevo_standalone` is a flag.
- Widget keys: `coevo_source`, `coevo_aligner`, `coevo_file`, `coevo_method`, `coevo_iters`, `coevo_region`, `coevo_run`, `coevo_structure`, `coevo_chain`, `coevo_n_show`, `coevo_n_links`, `coevo_standalone_btn`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_navigation.py`:
1. In `test_protein_sequences`, insert `"Coevolution",` after `"Phylogenetic Tree",` in the expected list, so it reads `..., "Phylogenetic Tree", "Coevolution", "Antibody Prediction", ...`.
2. In `test_single_sequence_omits_multi_sequence_analyses`, change the expected list to `["Coevolution", "Translate DNA", "Convert Formats"]`.

Append to `tests/test_app_flow.py`:

```python
def _coupled_a3m(n=300, L=40, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    msa = rng.integers(0, 20, size=(n, L))
    pick = np.array([[0, 5], [3, 9], [11, 2], [17, 14]])[rng.integers(0, 4, size=n)]
    msa[:, 5], msa[:, 30] = pick[:, 0], pick[:, 1]
    return "".join(f">s{k}\n" + "".join(aa[x] for x in row) + "\n" for k, row in enumerate(msa)).encode()


def test_coevolution_page_with_uploaded_alignment():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    at.button(key="coevo_standalone_btn").click().run()
    at.session_state["coevo_alignment"] = ("synthetic.a3m", _coupled_a3m())
    at.run()
    for method in ("plmDCA (accurate)", "Mean-field DCA (fast)"):
        next(w for w in at.get("button_group") if w.key == "coevo_method").set_value(method).run()
        at.button(key="coevo_run").click().run()
        assert not at.exception, (method, at.exception)
        assert any(d.label == "Top pairs (CSV)" for d in at.get("download_button"))
        assert any(m.label == "Neff / L" for m in at.metric)


def test_coevolution_long_query_needs_region_and_shallow_alignment_warns():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    at.button(key="coevo_standalone_btn").click().run()
    at.session_state["coevo_alignment"] = ("long.a3m", _coupled_a3m(n=40, L=250))
    at.run()
    assert any("Choose a region" in w.value for w in at.warning)          # 250 > plmDCA's 200-column limit
    at.button(key="coevo_run").click().run()
    assert not at.exception
    assert any("too few diverse sequences" in e.value for e in at.error)   # 40 random sequences: Neff/L < 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_navigation.py tests/test_app_flow.py -k "coevolution or protein_sequences or single_sequence" -q`
Expected: FAIL. "Coevolution" is not in the menu, and there is no `coevo_standalone_btn`.

- [ ] **Step 3: Create `modules/coevolution_page.py`**

```python
"""Coevolution page: contacts from a deep MSA (current dataset or an uploaded A3M / Stockholm / FASTA)."""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.coevolution import (CONTACT_CUTOFF, LIMITS, METHODS, PLM_ITERATIONS, contact_precision, mean_field_dca,
                                 parse_alignment, plm_dca, prepare_msa, sequence_weights, structure_distances,
                                 top_pairs)
from modules.ui import page_header, show_plotly

CLOSE, FAR, UNKNOWN = "#1b9e4b", "#d6312b", "#9e9e9e"


@st.cache_data(max_entries=4, show_spinner=False)
def cached_parse(data):
    return parse_alignment(data.decode("utf-8", "replace"))


@st.cache_data(max_entries=4, show_spinner=False)
def cached_contacts(rows, method, max_seqs, start, end, iters):
    """Prepared MSA summary and APC scores for one input and setting combination (cached)."""
    prepared = prepare_msa(list(rows), max_seqs=max_seqs)
    msa = prepared.msa[:, start - 1:end]
    t0 = time.time()
    w = sequence_weights(msa)
    scores = plm_dca(msa, w, iters=iters) if method == "plm" else mean_field_dca(msa, w)
    return {"scores": scores, "query": prepared.query_seq[start - 1:end], "offset": start - 1,
            "n_input": prepared.n_input, "n_used": len(msa), "neff": float(w.sum()),
            "dropped": (prepared.n_dropped_length, prepared.n_dropped_gappy, prepared.n_subsampled),
            "seconds": time.time() - t0}


@st.cache_data(max_entries=4, show_spinner=False)
def cached_distances(name, data, fmt, chain_id, query):
    from modules.structures import StructureFile

    return structure_distances(StructureFile(name=name, data=data, fmt=fmt), chain_id, query)


def contact_map_figure(result, n_show, distances=None, min_sep=6):
    """Predicted pairs above the diagonal (by score), structure contacts below (grey), query numbering."""
    L, off, q = len(result["query"]), result["offset"], result["query"]
    pairs = top_pairs(result["scores"], min_sep, n_show)
    fig = go.Figure()
    if distances is not None:
        ii, jj = np.where(np.triu(np.nan_to_num(distances, nan=99) < CONTACT_CUTOFF, min_sep))
        fig.add_trace(go.Scatter(x=ii + 1 + off, y=jj + 1 + off, mode="markers", name="Structure contact",
                                 marker=dict(size=4, color="#bdbdbd"), hoverinfo="skip"))
    if pairs:
        i, j, s = (np.array(v) for v in zip(*pairs))
        long_range = (j - i) >= 24
        for mask, name, symbol in ((long_range, "Predicted (|i-j| >= 24)", "circle"),
                                   (~long_range, "Predicted (6 <= |i-j| < 24)", "diamond")):
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=j[mask] + 1 + off, y=i[mask] + 1 + off, mode="markers", name=name,
                    marker=dict(size=7, symbol=symbol, color=s[mask], colorscale="Viridis", showscale=False),
                    text=[f"{q[a]}{a + 1 + off} - {q[b]}{b + 1 + off}: {v:.3f}" for a, b, v in
                          zip(i[mask], j[mask], s[mask])], hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(height=640, xaxis=dict(title="Query position", range=[off, off + L + 1]),
                      yaxis=dict(title="Query position", range=[off + L + 1, off], scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=30))
    return fig


def _alignment_rows(sequences, seq_type, standalone):
    """(rows, label) from the chosen source, or (None, None) after telling the user what is missing."""
    from modules.msa import MSA_METHODS, MSAError, cached_run_msa, mafft_available

    sources = ["Upload an alignment"] if standalone or not sequences else ["Current dataset", "Upload an alignment"]
    source = st.radio("Alignment", sources, horizontal=True, key="coevo_source")
    if source == "Current dataset":
        if seq_type != "Protein":
            st.info("Coevolution needs protein sequences; switch the sequence type or upload a protein alignment.")
            return None, None
        if len(sequences) < 2:
            st.info("Load at least two sequences, or upload a deep alignment (e.g. a ColabFold .a3m).")
            return None, None
        method = st.selectbox("Aligner", [m for m in MSA_METHODS if m == "FAMSA" or mafft_available()],
                              key="coevo_aligner")
        try:
            with st.spinner("Aligning..."):
                rows = cached_run_msa(tuple((r.id, str(r.seq)) for r in sequences), "Protein", method)
        except MSAError as e:
            st.error(str(e))
            return None, None
        return rows, f"current dataset ({method})"
    upload = st.file_uploader("Alignment: ColabFold/HHblits .a3m, Stockholm .sto, or aligned FASTA",
                              type=["a3m", "sto", "stockholm", "fasta", "fa", "aln", "txt"], key="coevo_file")
    if upload is not None:
        st.session_state.coevo_alignment = (upload.name, upload.getvalue())
    stored = st.session_state.get("coevo_alignment")
    if not stored:
        st.info("Upload a deep alignment. The first sequence is the query (as in ColabFold A3M files).")
        return None, None
    rows = cached_parse(stored[1])
    if len(rows) < 2:
        st.error("The file contains fewer than two sequences.")
        return None, None
    return rows, stored[0]


def coevolution_section(sequences, seq_type, standalone=False):
    page_header("Coevolution contacts",
                "Residue pairs that co-vary across a deep alignment, predicted with plmDCA or mean-field DCA.")
    rows, label = _alignment_rows(sequences, seq_type, standalone)
    if rows is None:
        return
    c1, c2 = st.columns([2, 1])
    method_label = c1.segmented_control("Method", list(METHODS), default=list(METHODS)[0], required=True,
                                        key="coevo_method")
    method = METHODS[method_label]
    max_cols, max_seqs = LIMITS[method]
    iters = c2.slider("plmDCA iterations", 10, 100, PLM_ITERATIONS, 10, key="coevo_iters",
                      disabled=method != "plm")
    query = "".join(c for c in rows[0][1] if c not in "-.")
    L = len(query)
    if L > max_cols:
        st.warning(f"The query has {L} residues; this method handles up to {max_cols} columns on Streamlit Cloud. "
                   "Choose a region.")
        start, end = st.slider("Region (query numbering)", 1, L, (1, max_cols), key="coevo_region")
        if end - start + 1 > max_cols:
            st.error(f"Choose at most {max_cols} positions.")
            return
    else:
        start, end = 1, L
    st.caption(f"Limits on Streamlit Cloud for this method: {max_cols} columns, {max_seqs:,} sequences "
               "(larger alignments are subsampled with a fixed seed, the query always kept).")
    params = {"label": label, "n": len(rows), "method": method, "start": start, "end": end, "iters": iters}
    if st.button("Compute contacts", key="coevo_run", type="primary", icon=":material/play_arrow:"):
        try:
            with st.spinner("Computing coevolution (this can take a minute for deep alignments)..."):
                st.session_state.coevo_result = {"params": params, "result": cached_contacts(
                    tuple(rows), method, max_seqs, start, end, iters if method == "plm" else 0)}
        except (ValueError, MemoryError) as e:
            st.error(f"Could not compute contacts: {e}")
    stored = st.session_state.get("coevo_result")
    if not stored:
        return
    if stored["params"] != params:
        st.info("Settings changed: press Compute contacts to update.", icon=":material/refresh:")
        return
    _show_result(stored["result"])


def _show_result(result):
    from modules.structure_view import _add_structure_controls, available_structures, cached_residue_table, rank_chains

    L = len(result["query"])
    neff_l = result["neff"] / L
    cols = st.columns(5)
    cols[0].metric("Sequences used", f"{result['n_used']:,}", border=True)
    cols[1].metric("Effective sequences", f"{result['neff']:,.0f}", border=True)
    cols[2].metric("Neff / L", f"{neff_l:.1f}", border=True)
    cols[3].metric("Positions", L, border=True)
    cols[4].metric("Runtime", f"{result['seconds']:.1f} s", border=True)
    dl, dg, ds = result["dropped"]
    if dl or dg or ds:
        st.caption(f"Removed: {dl} of different length, {dg} with more than half gaps, {ds} by subsampling "
                   f"(of {result['n_input']:,} input sequences).")
    if neff_l < 1:
        st.error(f"Neff/L = {neff_l:.1f}: too few diverse sequences for reliable contacts. Use a deeper alignment "
                 "(e.g. a ColabFold .a3m).")
    elif neff_l < 5:
        st.warning(f"Neff/L = {neff_l:.1f}: contacts may be noisy.")

    distances = q_coords = sf = chain_ids = None
    with st.expander("Compare with a structure", expanded=bool(available_structures())):
        _add_structure_controls("coevo")
        structures = available_structures()
        if structures:
            c1, c2 = st.columns(2)
            name = c1.selectbox("Structure", list(structures), key="coevo_structure")
            sf = structures[name]
            table = cached_residue_table(sf.data, sf.fmt, sf.name)
            if table:
                ranked = rank_chains(result["query"], table)
                chain_id = c2.selectbox("Chain", [c for c, _ in ranked], key="coevo_chain",
                                        format_func=lambda c: f"{c} ({dict(ranked)[c]:.1%} identity)")
                distances, _, q_coords = cached_distances(sf.name, sf.data, sf.fmt, chain_id, result["query"])
                chain_ids = list(table)
    n_show = st.slider("Pairs shown", max(1, L // 5), max(2, 2 * L), L, key="coevo_n_show")
    show_plotly(contact_map_figure(result, n_show, distances), width="stretch")
    if distances is not None:
        rows = []
        for sep_label, sep in (("|i-j| >= 6", 6), ("|i-j| >= 24", 24)):
            rows.append([sep_label] + [contact_precision(result["scores"], distances, sep, max(1, L // k))
                                       for k in (5, 2, 1)])
        st.markdown("**Precision against the structure** (fraction of top pairs with CB-CB < 8 A)")
        st.dataframe(pd.DataFrame(rows, columns=["Pairs", "Top L/5", "Top L/2", "Top L"]).round(2), hide_index=True)
    off, q = result["offset"], result["query"]
    pairs = top_pairs(result["scores"], 6, n_show)
    table = pd.DataFrame([{"Rank": r + 1, "i": i + 1 + off, "Residue i": q[i], "j": j + 1 + off, "Residue j": q[j],
                           "Score": round(s, 4),
                           **({"Distance (A)": None if np.isnan(distances[i, j]) else round(float(distances[i, j]), 1)}
                              if distances is not None else {})}
                          for r, (i, j, s) in enumerate(pairs)])
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Top pairs (CSV)", table.to_csv(index=False), "coevolution_pairs.csv", "text/csv",
                       on_click="ignore", icon=":material/download:")
    if sf is not None and q_coords is not None and chain_ids:
        _show_links(sf, chain_ids, pairs, distances, q_coords, off, q, L)


def _show_links(sf, chain_ids, pairs, distances, q_coords, off, q, L):
    """Top pairs as tubes between CB atoms on the structure (green within 8 A, red otherwise)."""
    from modules.structure_view import build_view_state, render_molstar

    k = st.slider("Pairs drawn on the structure", 5, 100, min(100, max(5, L // 5)), key="coevo_n_links")
    builder = build_view_state(sf, chain_ids, color_mode="uniform")
    primitives = builder.primitives()
    drawn = skipped = 0
    for i, j, s in pairs[:k]:
        a, b = q_coords[i], q_coords[j]
        if np.isnan(a).any() or np.isnan(b).any():
            skipped += 1
            continue
        d = distances[i, j]
        color = UNKNOWN if np.isnan(d) else CLOSE if d < CONTACT_CUTOFF else FAR
        primitives.tube(start=[float(x) for x in a], end=[float(x) for x in b], radius=0.2, color=color,
                        tooltip=f"{q[i]}{i + 1 + off}-{q[j]}{j + 1 + off} (score {s:.3f}, {d:.1f} A)")
        drawn += 1
    st.caption(f"{drawn} pairs drawn (green: CB-CB < 8 A in this structure, red: farther)"
               + (f"; {skipped} not modelled in the structure." if skipped else "."))
    render_molstar(sf, builder)
```

- [ ] **Step 4: Navigation and app wiring**

In `modules/navigation.py`, inside `available_analyses` and right after the lines
```python
        if n_sequences != 1:
            options += list(NEEDS_MULTIPLE)
```
add:
```python
        options.append("Coevolution")               # its upload source works with any dataset
```

In `app.py`:
1. Add the import `from modules.coevolution_page import coevolution_section`, next to the other `modules.*` imports.
2. In `main()`, replace the no-dataset block
   ```python
       if not sequences:
           st.info("Load sequences in the sidebar to begin: paste FASTA, upload a file, fetch by ID or pick an example.")
           _sidebar_footer()
           return
   ```
   with:
   ```python
       if not sequences:
           st.info("Load sequences in the sidebar to begin: paste FASTA, upload a file, fetch by ID or pick an example.")
           if st.button("Coevolution from an uploaded alignment (A3M / Stockholm / FASTA)", key="coevo_standalone_btn",
                        icon=":material/hub:"):
               st.session_state.coevo_standalone = True
           if st.session_state.get("coevo_standalone"):
               coevolution_section(None, None, standalone=True)
           _sidebar_footer()
           return
   ```
3. In the analysis dispatch, add this immediately before `elif alignment_mode == "Convert Formats":`:
   ```python
       elif alignment_mode == "Coevolution":
           coevolution_section(sequences, seq_type)
   ```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest -q`, then `.venv/bin/python precommit_check.py`.
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add modules/coevolution_page.py modules/navigation.py app.py tests/test_navigation.py tests/test_app_flow.py
git commit -m "feat: Coevolution page (current dataset or uploaded A3M/Stockholm/FASTA; contact map, precision, Mol* links)"
```

---

### Task 3: Ancestral reconstruction core (`modules/ancestral.py`)

**Files:**
- Create: `modules/ancestral.py`
- Create, by copying the committed prototype fixtures: `tests/data/asr/globins_aln.fasta`, `tests/data/asr/globins.state`, `tests/data/asr/globins.treefile`, `tests/data/asr/globins.iqtree`
- Test: `tests/test_ancestral.py`

**Interfaces:**
- Produces:
  - Constants:
    - `IQTREE_NAMES = ("iqtree2", "iqtree3", "iqtree")`, `IQTREE_TIMEOUT = 900`, `MIN_SEQUENCES = 4`, `MAX_SEQUENCES = 300`
    - `MODELS = {"Protein": {"LG+G4 (fast)": "LG+G4", "Best fit (ModelFinder, slower)": "MFP"}, "DNA": {"GTR+G4 (fast)": "GTR+G4", "Best fit (ModelFinder, slower)": "MFP"}}`
    - `INSTALL_HINT`
  - `ASRError`
  - `IQTreeResult(tree, states, alphabet, model, loglik, version, files)`
  - Running IQ-TREE:
    - `find_iqtree() -> (absolute path, version line) | None`
    - `run_iqtree(aligned_pairs, seq_type="Protein", model="LG+G4", bootstrap=0, timeout=900) -> IQTreeResult`
  - Parsing:
    - `parse_state(text) -> (alphabet, {node: (states, posteriors)})`
    - `parse_tree(text)`, which splits `NodeN/support`
    - `parse_report(text) -> (model, loglik)`
  - Reconstruction:
    - `reconstruct_gaps(tree, aligned_pairs) -> {node: bool[n_sites]}`
    - `ancestor_options(tree, states) -> [(name, n_leaves, label)]`
    - `mrca_node(tree, leaf_names, states) -> name`
    - `ancestor_sequence(result, gaps, node) -> (seq, posteriors, columns)`
    - `uncertain_sites(result, gaps, node, max_posterior=0.8, min_alternative=0.2) -> DataFrame`
    - `altall_sequence(result, gaps, node, max_posterior=0.8, min_alternative=0.2) -> str`
    - `compare_to_extant(result, gaps, node, aligned_pairs, extant_id) -> (substitutions [(label, column)], insertions, deletions, identity)`

- [ ] **Step 1: Add the fixtures and write the failing tests**

The fixtures are a real IQ-TREE 2.0.3 run (`-m LG+G4 -asr -B 1000`) on the 6-globin FAMSA alignment. Copy them in:

```bash
mkdir -p tests/data/asr
cp /tmp/claude-1000/-home-dzyla-seq-align/2a284106-e858-4c33-b684-30e5ab79d2de/scratchpad/plan09_fixtures/asr/* tests/data/asr/
ls tests/data/asr   # globins.iqtree  globins.state  globins.treefile  globins_aln.fasta
```

Create `tests/test_ancestral.py`:

```python
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import modules.ancestral as anc
from modules.ancestral import (ASRError, IQTreeResult, altall_sequence, ancestor_options, ancestor_sequence,
                               compare_to_extant, mrca_node, parse_report, parse_state, parse_tree,
                               reconstruct_gaps, uncertain_sites)
from modules.tree_layout import prepare_tree

DATA = Path(__file__).parent / "data" / "asr"


def read_fasta(path):
    rows, name = [], None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
            rows.append([name, ""])
        elif line.strip():
            rows[-1][1] += line.strip()
    return [tuple(r) for r in rows]


def fixture_result():
    alphabet, states = parse_state((DATA / "globins.state").read_text())
    model, loglik = parse_report((DATA / "globins.iqtree").read_text())
    return IQTreeResult(parse_tree((DATA / "globins.treefile").read_text()), states, alphabet, model, loglik,
                        "IQ-TREE multicore version 2.0.3", {})


ROWS = read_fasta(DATA / "globins_aln.fasta")


def test_parse_state_report_and_tree_labels():
    r = fixture_result()
    assert r.alphabet == "ARNDCQEGHILKMFPSTWYV" and sorted(r.states) == ["Node1", "Node2", "Node3", "Node4"]
    states, post = r.states["Node4"]
    assert len(states) == len(ROWS[0][1]) == post.shape[0] and post.shape[1] == 20
    assert np.allclose(post.sum(axis=1), 1, atol=1e-3)
    assert (r.model, round(r.loglik, 2)) == ("LG+G4", -1478.58)
    node4 = next(c for c in r.tree.get_nonterminals() if c.name == "Node4")
    assert node4.confidence == 48.0                     # 'Node4/48' split into name + support


def test_mrca_and_single_leaf_rejected():
    r = fixture_result()
    tree = prepare_tree(r.tree, "midpoint")
    assert mrca_node(tree, ["HBB_HUMAN", "HBB1_MOUSE"], r.states) == "Node4"
    with pytest.raises(ASRError, match="at least two"):
        mrca_node(tree, ["HBB_HUMAN"], r.states)
    labels = [o[2] for o in ancestor_options(tree, r.states)]
    assert any(l.startswith("Node4: 2 sequences") for l in labels)


def test_reconstruct_gaps_rules():
    from io import StringIO
    from Bio import Phylo
    tree = Phylo.read(StringIO("(((a,b)N2,c)N1,d)N0;"), "newick")
    rows = [("a", "M-K"), ("b", "M-K"), ("c", "MRK"), ("d", "M-R")]
    gaps = reconstruct_gaps(tree, rows)
    assert gaps["N2"].tolist() == [True, False, True]      # gap shared by the a+b clade -> absent
    assert gaps["N1"][1] == False                           # noqa: E712 - c alone has the residue
    rows2 = [("a", "MRK"), ("b", "M-K"), ("c", "MRK"), ("d", "MRK")]
    assert reconstruct_gaps(tree, rows2)["N2"].tolist() == [True, True, True]   # single-leaf gap -> present


def test_uncertain_sites_altall_and_compare():
    r = fixture_result()
    tree = prepare_tree(r.tree, "midpoint")
    gaps = reconstruct_gaps(tree, ROWS)
    seq, pp, cols = ancestor_sequence(r, gaps, "Node4")
    assert len(seq) == len(pp) == len(cols) and 0 <= pp.min() and pp.max() <= 1
    table = uncertain_sites(r, gaps, "Node4", 0.8, 0.2)
    assert (table["Posterior"] < 0.8).all() and len(table) == int((pp < 0.8).sum())
    alt = altall_sequence(r, gaps, "Node4", 0.8, 0.2)
    assert len(alt) == len(seq) and alt != seq
    changed = [k for k, (a, b) in enumerate(zip(seq, alt)) if a != b]
    assert all(pp[k] < 0.8 for k in changed)
    subs, ins, dels, identity = compare_to_extant(r, gaps, "Node4", ROWS, "HBB_HUMAN")
    assert 0.8 < identity < 1.0 and all(label[0] != label[-1] for label, _ in subs)


def test_missing_binary_and_timeout(monkeypatch):
    monkeypatch.setattr(anc, "find_iqtree", lambda: None)
    with pytest.raises(ASRError, match="not installed"):
        anc.run_iqtree(ROWS, "Protein")
    monkeypatch.setattr(anc, "find_iqtree", lambda: ("/bin/iqtree", "IQ-TREE 2"))

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="iqtree", timeout=1)
    monkeypatch.setattr(anc.subprocess, "run", slow)
    with pytest.raises(ASRError, match="did not finish"):
        anc.run_iqtree(ROWS, "Protein", timeout=1)


def test_too_few_sequences_rejected(monkeypatch):
    monkeypatch.setattr(anc, "find_iqtree", lambda: ("/bin/iqtree", "IQ-TREE 2"))
    with pytest.raises(ASRError, match="4-300"):
        anc.run_iqtree(ROWS[:3], "Protein")


@pytest.mark.skipif(anc.find_iqtree() is None, reason="IQ-TREE not installed")
def test_end_to_end_iqtree_run_keeps_original_ids():
    r = anc.run_iqtree(ROWS, "Protein", "LG+G4")
    assert sorted(t.name for t in r.tree.get_terminals()) == sorted(i for i, _ in ROWS)
    assert len(r.states) == 4
    for states, post in r.states.values():
        assert np.allclose(post.sum(axis=1), 1, atol=1e-3)
    assert "IQ-TREE" in r.version and {"iqtree.treefile", "iqtree.state"} <= set(r.files)


@pytest.mark.skipif(not __import__("os").environ.get("CI"), reason="only enforced in CI (GitHub Actions sets CI)")
def test_ci_has_the_cloud_system_binaries():
    # packages.txt installs these on Streamlit Cloud; CI installs the same Debian/Ubuntu packages, so a missing
    # binary must fail here instead of silently skipping the IQ-TREE and DSSP tests
    from modules.espript import find_mkdssp
    assert anc.find_iqtree() is not None, "iqtree package not found (iqtree2/iqtree3/iqtree)"
    assert find_mkdssp() is not None, "dssp package not found (mkdssp)"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_ancestral.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.ancestral'`.

- [ ] **Step 3: Implement `modules/ancestral.py`**

```python
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
INSTALL_HINT = ("IQ-TREE is not installed. On Streamlit Cloud add `iqtree` to packages.txt; locally run "
                "`conda install -c bioconda iqtree`.")
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


def find_iqtree():
    """First IQ-TREE binary on PATH whose --version output mentions IQ-TREE, as (path, version line)."""
    for name in IQTREE_NAMES:
        path = shutil.which(name)
        path = os.path.abspath(path) if path else None
        if not path:
            continue
        try:
            out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        line = next((l for l in out.splitlines() if "IQ-TREE" in l), None)
        if line:
            return path, line.strip()
    return None


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
        raise ASRError(INSTALL_HINT)
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
            last = next((l for l in reversed(log.splitlines()) if l.strip()), "unknown error")
            raise ASRError(f"IQ-TREE failed: {last}")
        tree_text, state_text, report = read("run.treefile"), read("run.state"), read("run.iqtree") or ""
        files = {f"iqtree.{ext}": read(f"run.{ext}", "rb") for ext in ("treefile", "state", "iqtree", "log")
                 if read(f"run.{ext}", "rb") is not None}
    tree = parse_tree(tree_text)
    for clade in tree.get_terminals():
        clade.name = placeholders.get(clade.name, clade.name)
    alphabet, states = parse_state(state_text)
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
    """Name of the most recent common ancestor of the chosen leaves that has reconstructed states."""
    if len(set(leaf_names)) < 2:
        raise ASRError("Choose at least two sequences: a single sequence is not an ancestor.")
    clade = tree.common_ancestor(list(leaf_names))
    while clade is not None and clade.name not in states:   # e.g. an unnamed midpoint root
        clade = tree.get_path(clade)[-2] if len(tree.get_path(clade)) > 1 else None
        if clade is None:
            break
    if clade is None or clade.name not in states:
        children = [c for c in tree.common_ancestor(list(leaf_names)).clades if c.name in states]
        if not children:
            raise ASRError("No reconstructed ancestor for these sequences.")
        clade = children[0]
    return clade.name


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_ancestral.py -q`
Expected: 6 passed and 2 skipped. The skips are the end-to-end IQ-TREE run and the CI binary check, because IQ-TREE is not installed locally; both run in CI. Then run the full suite and precommit: green.

- [ ] **Step 5: Commit**

```bash
git add modules/ancestral.py tests/test_ancestral.py tests/data/asr
git commit -m "feat: ancestral reconstruction core (IQ-TREE marginal states, Fitch indels, MRCA picking, uncertain sites, AltAll)"
```

---

### Task 4: Ancestral Reconstruction page, cloud wiring and README

**Files:**
- Create: `modules/ancestral_page.py`
- Modify: `modules/navigation.py`, `app.py`, `tests/test_navigation.py`, `packages.txt`, `.github/workflows/tests.yml`, `README.md`
- Test: `tests/test_app_flow.py` (append)

**Interfaces:**
- Consumes:
  - Task 3: everything.
  - Imported inside functions: `modules.msa` (cached alignment), `modules.phylogeny` (`TreeError`, `phylogenetic_tree_section`), `modules.tree_layout.prepare_tree`, `modules.structure_view.structure_highlight_panel`.
  - `modules.ui`: `page_header`, `show_plotly`.
- Produces: `ancestral_section(sequences, seq_type)`.
- Session key: `asr_result` holds `{"params", "rows", "result"}`.
- Widget keys: `asr_aligner`, `asr_model`, `asr_bootstrap`, `asr_run`, `asr_rooting`, `asr_outgroup`, `asr_pick`, `asr_leaves`, `asr_node`, `asr_threshold`, `asr_min_alt`, `asr_extant`, `asr_3d`.
- The embedded tree section uses the prefix `asr_tree`, because `asr` would collide with `asr_rooting`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_navigation.py`:

```python
def test_ancestral_reconstruction_needs_four_sequences():
    assert "Ancestral Reconstruction" not in available_analyses(True, False, "Protein", n_sequences=3)
    assert "Ancestral Reconstruction" in available_analyses(True, False, "Protein", n_sequences=4)
    assert "Coevolution" in available_analyses(True, False, "Protein", n_sequences=1)
```

Append to `tests/test_app_flow.py`:

```python
def test_ancestral_page_with_fixture(monkeypatch):
    from pathlib import Path
    import modules.ancestral_page as page
    from modules.ancestral import IQTreeResult, parse_report, parse_state, parse_tree
    data = Path(__file__).parent / "data" / "asr"
    alphabet, states = parse_state((data / "globins.state").read_text())
    model, loglik = parse_report((data / "globins.iqtree").read_text())
    text = (data / "globins_aln.fasta").read_text().split(">")[1:]
    rows = [(chunk.splitlines()[0], "".join(chunk.splitlines()[1:])) for chunk in text]
    result = IQTreeResult(parse_tree((data / "globins.treefile").read_text()), states, alphabet, model, loglik,
                          "IQ-TREE multicore version 2.0.3", {"iqtree.state": b"x"})
    monkeypatch.setattr(page, "find_iqtree", lambda: ("/usr/bin/iqtree2", "IQ-TREE 2.0.3"))
    monkeypatch.setattr(page, "cached_run_iqtree", lambda r, s, m, b: result)
    import modules.msa as msa_module
    monkeypatch.setattr(msa_module, "cached_run_msa", lambda pairs, seq_type, method="FAMSA": rows)
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    at.selectbox(key="example_choice").select("Globins (protein)").run()
    at.button(key="load_example").click().run()
    at.radio(key="analysis_select").set_value("Ancestral Reconstruction").run()
    at.button(key="asr_run").click().run()
    assert not at.exception, at.exception
    labels = [d.label for d in at.get("download_button")]
    assert {"ML ancestor (FASTA)", "AltAll (FASTA)", "IQ-TREE files (ZIP)"} <= set(labels)
    at.radio(key="asr_pick").set_value("From the list of nodes").run()
    assert not at.exception
    at.radio(key="asr_rooting").set_value("Outgroup").run()
    at.multiselect(key="asr_outgroup").set_value([r[0] for r in rows]).run()     # all leaves cannot root
    assert not at.exception and any("midpoint rooting instead" in w.value for w in at.warning)


def test_ancestral_page_without_iqtree_and_bad_outgroup(monkeypatch):
    import modules.ancestral_page as page
    monkeypatch.setattr(page, "find_iqtree", lambda: None)
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    at.selectbox(key="example_choice").select("Globins (protein)").run()
    at.button(key="load_example").click().run()
    at.radio(key="analysis_select").set_value("Ancestral Reconstruction").run()
    assert not at.exception and any("conda install -c bioconda iqtree" in e.value for e in at.error)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_navigation.py tests/test_app_flow.py -k "ancestral" -q`
Expected: FAIL. "Ancestral Reconstruction" is not in the menu, and `modules.ancestral_page` cannot be imported.

- [ ] **Step 3: Create `modules/ancestral_page.py`**

```python
"""Ancestral Reconstruction page: IQ-TREE marginal ancestral states, indels by parsimony, uncertainty and exports."""
import html
import io
import zipfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from modules.ancestral import (INSTALL_HINT, MAX_SEQUENCES, MIN_SEQUENCES, MODELS, ASRError, altall_sequence,
                               ancestor_options, ancestor_sequence, compare_to_extant, find_iqtree, mrca_node,
                               reconstruct_gaps, run_iqtree, uncertain_sites)
from modules.ui import page_header, show_plotly

CERTAIN, UNSURE, LOW = "#ffffff", "#fdd8a8", "#f59e42"
DIFFERS = "#2a5db0"


@st.cache_data(max_entries=4, show_spinner=False)
def cached_run_iqtree(rows, seq_type, model, bootstrap):
    return run_iqtree(list(rows), seq_type, model, bootstrap)


def _shade(p, threshold):
    return CERTAIN if p >= 0.9 else UNSURE if p >= threshold else LOW


def sequence_html(seq, posteriors, threshold, width=60):
    """Monospace ancestor with each residue shaded by its posterior (white >= 0.9, light < 0.9, orange < threshold)."""
    lines = []
    for start in range(0, len(seq), width):
        cells = "".join(f'<span style="background:{_shade(p, threshold)}" title="{start + k + 1}: {c} ({p:.2f})">'
                        f"{html.escape(c)}</span>"
                        for k, (c, p) in enumerate(zip(seq[start:start + width], posteriors[start:start + width])))
        lines.append(f'<span style="color:#888">{start + 1:>5} </span>{cells}')
    return ('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;line-height:1.6;color:#1c1b19;'
            'background:#fff;padding:8px;border-radius:6px">' + "<br>".join(lines) + "</div>")


def posterior_figure(posteriors, threshold):
    x = np.arange(1, len(posteriors) + 1)
    fig = go.Figure(go.Scatter(x=x, y=posteriors, mode="lines", line=dict(width=1.5),
                               hovertemplate="Position %{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#f59e42")
    fig.update_layout(height=240, xaxis_title="Ancestor position", yaxis=dict(title="Posterior", range=[0, 1.02]),
                      margin=dict(t=20, b=40))
    return fig


def ancestral_section(sequences, seq_type):
    from modules.msa import MSA_METHODS, MSAError, cached_run_msa, mafft_available
    from modules.phylogeny import TreeError
    from modules.tree_layout import prepare_tree

    page_header("Ancestral reconstruction",
                "Maximum-likelihood ancestors with a posterior for every residue (IQ-TREE), indels by parsimony.")
    found = find_iqtree()
    if found is None:
        st.error(INSTALL_HINT)
        return
    if not MIN_SEQUENCES <= len(sequences) <= MAX_SEQUENCES:
        st.info(f"Ancestral reconstruction needs {MIN_SEQUENCES}-{MAX_SEQUENCES} sequences "
                f"({len(sequences)} loaded).")
        return
    c1, c2, c3 = st.columns(3)
    aligner = c1.selectbox("Aligner", [m for m in MSA_METHODS if m == "FAMSA" or mafft_available()],
                           key="asr_aligner")
    model_label = c2.selectbox("Model", list(MODELS[seq_type]), key="asr_model")
    bootstrap = c3.toggle("Ultrafast bootstrap (1000)", key="asr_bootstrap")
    params = {"n": len(sequences), "ids": tuple(r.id for r in sequences), "seq_type": seq_type, "aligner": aligner,
              "model": MODELS[seq_type][model_label], "bootstrap": bootstrap}
    if st.button("Reconstruct ancestors", key="asr_run", type="primary", icon=":material/play_arrow:"):
        try:
            with st.spinner("Aligning and running IQ-TREE (seconds to minutes)..."):
                rows = cached_run_msa(tuple((r.id, str(r.seq)) for r in sequences), seq_type, aligner)
                result = cached_run_iqtree(tuple(rows), seq_type, params["model"], 1000 if bootstrap else 0)
            st.session_state.asr_result = {"params": params, "rows": rows, "result": result}
        except (MSAError, ASRError) as e:
            st.error(str(e))
    stored = st.session_state.get("asr_result")
    if not stored:
        st.caption(f"Engine: {found[1]}")
        return
    if stored["params"] != params:
        st.info("Settings changed: press Reconstruct ancestors to update.", icon=":material/refresh:")
        return
    rows, result = stored["rows"], stored["result"]
    lnl = f"{result.loglik:.2f}" if result.loglik is not None else "n/a"
    st.caption(f"{result.version}; model {result.model}; log-likelihood {lnl}.")

    leaves = [r[0] for r in rows]
    c1, c2 = st.columns([1, 2])
    rooting = c1.radio("Root", ["Midpoint", "Outgroup"], key="asr_rooting")
    outgroup = c2.multiselect("Outgroup sequences", leaves, key="asr_outgroup") if rooting == "Outgroup" else []
    try:
        tree = prepare_tree(result.tree, "outgroup" if outgroup else "midpoint", outgroup)
    except TreeError as e:
        st.warning(f"{e} Using midpoint rooting instead.")
        tree = prepare_tree(result.tree, "midpoint")
    gaps = reconstruct_gaps(tree, rows)

    pick = st.radio("Choose an ancestor", ["Ancestor of chosen sequences", "From the list of nodes"],
                    horizontal=True, key="asr_pick")
    try:
        if pick == "Ancestor of chosen sequences":
            chosen = st.multiselect("Sequences", leaves, default=leaves[:2], key="asr_leaves")
            node = mrca_node(tree, chosen, result.states)
        else:
            options = ancestor_options(tree, result.states)
            labels = {o[2]: o[0] for o in options}
            node = labels[st.selectbox("Node", list(labels), key="asr_node")]
    except ASRError as e:
        st.warning(str(e))
        return
    c1, c2 = st.columns(2)
    threshold = c1.slider("Uncertain below posterior", 0.5, 0.95, 0.8, 0.05, key="asr_threshold")
    min_alt = c2.slider("Show alternatives with posterior at least", 0.05, 0.5, 0.2, 0.05, key="asr_min_alt")
    seq, pp, anc_cols = ancestor_sequence(result, gaps, node)
    cols = st.columns(4)
    cols[0].metric("Ancestor", node, border=True)
    cols[1].metric("Length", len(seq), border=True)
    cols[2].metric("Mean posterior", f"{pp.mean():.2f}" if len(pp) else "n/a", border=True)
    cols[3].metric("Sites >= 0.9", f"{(pp >= 0.9).mean():.0%}" if len(pp) else "n/a", border=True)
    st.html(sequence_html(seq, pp, threshold))
    show_plotly(posterior_figure(pp, threshold), width="stretch")

    unsure = uncertain_sites(result, gaps, node, threshold, min_alt)
    st.markdown(f"**Uncertain sites** ({len(unsure)})")
    if unsure.empty:
        st.caption("Every site is above the threshold.")
    else:
        st.dataframe(unsure, hide_index=True, width="stretch")
    alt = altall_sequence(result, gaps, node, threshold, min_alt)
    header = f"{node} model={result.model} lnL={lnl}"
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("ML ancestor (FASTA)", f">{node} ML ancestor; {header}\n{seq}\n", f"{node}_ML.fasta",
                       "text/plain", on_click="ignore", icon=":material/download:")
    d2.download_button("AltAll (FASTA)", f">{node} AltAll; {header}\n{alt}\n", f"{node}_AltAll.fasta", "text/plain",
                       on_click="ignore", icon=":material/download:",
                       help="Every uncertain site switched to its best alternative: the usual robustness check.")
    d3.download_button("Uncertain sites (CSV)", unsure.to_csv(index=False), f"{node}_uncertain_sites.csv",
                       "text/csv", on_click="ignore", icon=":material/download:")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in result.files.items():
            zf.writestr(name, data)
    d4.download_button("IQ-TREE files (ZIP)", buffer.getvalue(), "iqtree_asr.zip", "application/zip",
                       on_click="ignore", icon=":material/download:")

    st.markdown("**Compare with a present-day sequence**")
    extant = st.selectbox("Sequence", leaves, key="asr_extant")
    subs, ins, dels, identity = compare_to_extant(result, gaps, node, rows, extant)
    st.caption(f"{len(subs)} substitutions, {ins} insertions and {dels} deletions in {extant} relative to {node}; "
               f"identity {identity:.1%} over shared positions.")
    if subs:
        st.code(", ".join(label for label, _ in subs), language=None)

    from modules.structure_view import structure_highlight_panel

    position = {int(c): k + 1 for k, c in enumerate(anc_cols)}       # alignment column -> ancestor position
    colors = {position[c]: (DIFFERS, f"{label} vs {extant}") for label, c in subs if c in position}
    colors.update({k + 1: (LOW if p < threshold else UNSURE, f"{seq[k]}{k + 1} ({p:.2f})")
                   for k, p in enumerate(pp) if p < 0.9})
    structure_highlight_panel(seq, colors, key="asr_3d",
                              title="Show confidence and differences on structure (Mol*)")
    st.caption(f"Structure colours: orange = posterior below {threshold:.2f}, light orange = below 0.9, "
               f"blue = differs from {extant}.")

    from modules.phylogeny import phylogenetic_tree_section

    import copy

    display = copy.deepcopy(tree)
    for clade in display.get_nonterminals():
        clade.name = None          # IQ-TREE node names would otherwise merge with supports in the Newick export
    with st.expander("Tree"):
        phylogenetic_tree_section(display, key_prefix="asr_tree")
```

- [ ] **Step 4: Navigation, app, cloud and README**

In `modules/navigation.py`:
1. Under `NEEDS_MULTIPLE = (...)`, add the line `MIN_ANCESTRAL_SEQUENCES = 4`.
2. Directly after the `options.append("Coevolution")` line from Task 2, add:
   ```python
           if n_sequences is not None and n_sequences >= MIN_ANCESTRAL_SEQUENCES:
               options.append("Ancestral Reconstruction")
   ```

In `app.py`:
1. Add the import `from modules.ancestral_page import ancestral_section`.
2. In the dispatch, after the Coevolution branch, add:
   ```python
       elif alignment_mode == "Ancestral Reconstruction":
           ancestral_section(sequences, seq_type)
   ```
3. In `GUIDE`, replace the line `Quick calculators (restriction maps, codon usage, primers, cloning) live in [Bio-Bench](%s).` with these three lines (keep the `% BIO_BENCH_URL` formatting after the closing quotes):
   ```
   **Coevolution** predicts residue contacts from a deep alignment (upload a ColabFold .a3m); **Ancestral
   Reconstruction** infers ancestors with IQ-TREE.

   Quick calculators (restriction maps, codon usage, primers, cloning) live in [Bio-Bench](%s).
   ```

`packages.txt` becomes:
```
hmmer
mafft
dssp
iqtree
```

In `.github/workflows/tests.yml`, change the install step to:
```yaml
      - name: Install HMMER, MAFFT, DSSP and IQ-TREE (same Debian/Ubuntu packages as packages.txt)
        run: sudo apt-get update && sudo apt-get install -y hmmer mafft dssp iqtree
```

In `README.md`, add these two rows directly after the `| UniProt batch |` row:
```markdown
| Coevolution | Residue contacts from a deep MSA (current dataset, or an uploaded ColabFold/HHblits .a3m, Stockholm or aligned FASTA): plmDCA (accurate) or mean-field DCA (fast) with sequence reweighting and APC; Neff/L depth check; contact map, precision against a structure, top pairs (CSV) and links drawn in Mol*. Cloud limits: plmDCA 200 columns / 5,000 sequences, mean-field 400 / 10,000 |
| Ancestral reconstruction | IQ-TREE maximum-likelihood tree (LG+G4, GTR+G4 or ModelFinder) with marginal ancestral states and per-residue posteriors, indels by Fitch parsimony, ancestor of chosen sequences, uncertain sites with alternatives, ML and AltAll FASTA, comparison with present-day sequences, confidence on a structure. Needs `iqtree` (packages.txt on the cloud; `conda install -c bioconda iqtree` locally) |
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest -q`, then `.venv/bin/python precommit_check.py`.
Expected: all green, about 476 passed and 2 skipped.

- [ ] **Step 6: Commit**

```bash
git add modules/ancestral_page.py modules/navigation.py app.py tests/test_navigation.py tests/test_app_flow.py packages.txt .github/workflows/tests.yml README.md
git commit -m "feat: Ancestral Reconstruction page; iqtree for Streamlit Cloud and CI; README"
```

---

### Task 5: MSA colour schemes and sequence viewer

Added on 2026-10-09 at the owner's request.
- "Colour schemes" means Jalview-style ClustalX, Zappo, Taylor, hydrophobicity, percent identity, conservation and similarity to the reference.
- "Sequence viewer" means a column window with letters, a conservation track, a consensus row and a reference ruler, built on Plotly so no JavaScript component is needed.
- Prototype-verified: 459 passed and precommit passed on this branch's base. The static renders of ClustalX, percent identity, conservation and similarity were inspected.

**Files:**
- Create: `modules/msa_colors.py`
- Modify: `modules/viz.py` (colouring support, viewer, `alignment_view`), `modules/msa.py` (the first tab uses `alignment_view`)
- Test: `tests/test_msa_colors.py`, `tests/test_app_flow.py` (append)

**Interfaces:**
- Consumes:
  - `modules.conservation.column_conservation`
  - `modules.espript.column_classes` and `consensus_line` (Plan 08)
  - `modules.pairwise.load_matrix`
  - `modules.ui.show_plotly`
- Produces:
  - In `modules/msa_colors.py`:
    - Constants: `SCHEMES` (Protein: Residue, ClustalX, Zappo, Taylor, Hydrophobicity, Percent identity, Conservation, Similarity to reference; DNA: Nucleotide, Percent identity, Conservation, Similarity to reference), `DESCRIPTIONS`, `CLUSTALX_COLOURS`, `IDENTITY_SHADES`, `CONSERVATION_RAMP`, `HYDRO_RAMP`, `WHITE`
    - `Coloring(index, colors, legend)`
    - `color_alignment(seqs, scheme, seq_type="Protein", reference=0) -> Coloring`
  - In `modules/viz.py`:
    - `msa_figure_plotly(..., coloring=None)` and `msa_figure_matplotlib(..., coloring=None)`
    - `render_msa_png(codes, ids, seq_type, scheme=None, reference=0)`
    - `plot_msa(codes, ids, method, seq_type, scheme=None, reference=0)`, which stays backward compatible (clustering still calls it without a scheme)
    - `reference_numbering(ref_aligned)`
    - `msa_viewer_figure(codes, ids, coloring, start, width, ref_numbers, conservation, consensus)`
    - `alignment_view(pairs, seq_type, plot_method, reference_id)`
- Widget keys: `msa_view_mode`, `msa_color_scheme`, `msa_view_width`, `msa_view_start`, `msa_view_jump`, `msa_view_row`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_msa_colors.py`:

```python
import numpy as np
import pytest

from modules.msa_colors import (CLUSTALX_COLOURS, CONSERVATION_RAMP, HYDRO_RAMP, IDENTITY_SHADES, SCHEMES, WHITE,
                                color_alignment)
from modules.viz import msa_viewer_figure, reference_numbering


def colour_of(coloring, row, col):
    return coloring.colors[coloring.index[row, col]]


def column(seqs_column):
    """Rows of a one-column alignment."""
    return list(seqs_column)


def test_clustalx_rules():
    c = color_alignment(column("GGGG"), "ClustalX")
    assert colour_of(c, 0, 0) == CLUSTALX_COLOURS["ORANGE"]               # glycine always orange
    c = color_alignment(column("LLLLLLLKKK"), "ClustalX")
    assert colour_of(c, 0, 0) == CLUSTALX_COLOURS["BLUE"]                 # 70% hydrophobic
    assert colour_of(c, 9, 0) == WHITE                                    # K: KR only 30%
    assert colour_of(color_alignment(column("CCCC"), "ClustalX"), 0, 0) == CLUSTALX_COLOURS["PINK"]
    c = color_alignment(column("KKKKKKKAAA"), "ClustalX")
    assert colour_of(c, 0, 0) == CLUSTALX_COLOURS["RED"] and colour_of(c, 9, 0) == WHITE
    c = color_alignment(column("NNNNNNDDDD"), "ClustalX")
    assert colour_of(c, 0, 0) == CLUSTALX_COLOURS["GREEN"]                # N > 50%
    assert colour_of(color_alignment(column("--"), "ClustalX"), 0, 0) == WHITE


def test_percent_identity_shades():
    for seqs, expected in (("AAAAAAAAAL", IDENTITY_SHADES[0][1]), ("AAAAAAALLV", IDENTITY_SHADES[1][1]),
                           ("AAAAALLVVW", IDENTITY_SHADES[2][1]), ("AAAALLLVVW", WHITE)):
        c = color_alignment(column(seqs), "Percent identity")
        assert colour_of(c, 0, 0) == expected, seqs
    c = color_alignment(column("AAAAAAAAAL"), "Percent identity")
    assert colour_of(c, 9, 0) == WHITE                                    # non-consensus residue stays white


def test_conservation_and_similarity():
    c = color_alignment(["MK-", "MKA", "MRA", "MRG"], "Conservation")
    assert colour_of(c, 0, 0) == CONSERVATION_RAMP[-1]                    # invariant column
    assert colour_of(c, 0, 2) == WHITE                                    # gap
    c = color_alignment(["LLD-", "ILDA", "DLDA"], "Similarity to reference", reference=1)
    assert [colour_of(c, 1, j) for j in range(3)] == [c.colors[1]] * 3    # reference identical to itself
    assert colour_of(c, 0, 0) == c.colors[2] and colour_of(c, 2, 0) == c.colors[3]   # L~I similar, D vs I not
    assert colour_of(c, 0, 3) == WHITE                                    # gap


def test_hydrophobicity_and_schemes_list():
    c = color_alignment(["IR"], "Hydrophobicity")
    assert colour_of(c, 0, 0) == HYDRO_RAMP[-1] and colour_of(c, 0, 1) == HYDRO_RAMP[0]
    for seq_type, schemes in SCHEMES.items():
        for scheme in schemes:
            c = color_alignment(["ACGT-", "ACGA-"] if seq_type == "DNA" else ["MKV-L", "MRVAL"], scheme, seq_type)
            assert c.index.shape == (2, 5) and c.index.max() < len(c.colors)
    with pytest.raises(ValueError):
        color_alignment(["A"], "Rainbow")


def test_viewer_window_and_reference_ruler():
    seqs = ["M-KVLAAGIVGLLAQWERTYASDFGKLCV" * 2, "MKKVLAAGIVGLLAQWERTYASDFGKLCV" * 2]
    from modules.viz import alignment_to_codes
    ref_numbers = reference_numbering(seqs[0])
    assert ref_numbers[:3] == [1, None, 2]
    coloring = color_alignment(seqs, "ClustalX")
    fig = msa_viewer_figure(alignment_to_codes(seqs), ["a", "b"], coloring, start=5, width=40, ref_numbers=ref_numbers,
                            conservation=np.ones(len(seqs[0])), consensus="x" * len(seqs[0]))
    heat = next(t for t in fig.data if t.type == "heatmap")
    assert np.array(heat.z).shape == (3, 40) and list(heat.y) == ["consensus", "a", "b"]
    assert heat.texttemplate == "%{text}"
    ticks = fig.layout.xaxis2.ticktext
    assert all(int(t) % 10 == 0 for t in ticks)
```

Append to `tests/test_app_flow.py`:

```python
def test_msa_colour_schemes_and_sequence_viewer():
    at = _load_globins_msa()
    for scheme in ("ClustalX", "Percent identity", "Conservation", "Similarity to reference", "Zappo"):
        at.selectbox(key="msa_color_scheme").set_value(scheme).run()
        assert not at.exception, (scheme, at.exception)
    next(w for w in at.get("button_group") if w.key == "msa_view_mode").set_value("Sequence viewer").run()
    assert not at.exception
    at.number_input(key="msa_view_jump").set_value(60).run()
    assert not at.exception
    at.slider(key="msa_view_width").set_value(120).run()
    assert not at.exception
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest tests/test_msa_colors.py tests/test_app_flow.py::test_msa_colour_schemes_and_sequence_viewer -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.msa_colors'`, and there is no `msa_color_scheme` widget.

- [ ] **Step 3: Create `modules/msa_colors.py`**

```python
"""MSA colour schemes (Jalview-style): each maps an alignment to a per-cell colour index, a colour list and a
legend. Pure functions, shared by the overview heatmap, the sequence viewer and the static PNG."""
from dataclasses import dataclass

import numpy as np

GAP_CHARS = "-."
WHITE = "#ffffff"
SCHEMES = {
    "Protein": ("Residue", "ClustalX", "Zappo", "Taylor", "Hydrophobicity", "Percent identity", "Conservation",
                "Similarity to reference"),
    "DNA": ("Nucleotide", "Percent identity", "Conservation", "Similarity to reference"),
}
DESCRIPTIONS = {
    "Residue": "Every residue type in its usual Clustal colour, regardless of the column.",
    "ClustalX": "Jalview's ClustalX rules: a residue is coloured only when its column's composition supports it.",
    "Zappo": "Physicochemical groups.",
    "Taylor": "A distinct colour for every amino acid (Taylor 1997).",
    "Hydrophobicity": "Kyte-Doolittle scale: blue hydrophilic to red hydrophobic.",
    "Percent identity": "Residues matching their column's most common residue, shaded by how many sequences share it.",
    "Conservation": "Every residue shaded by its column's conservation (1 - entropy); gap-rich columns blank.",
    "Similarity to reference": "Identical to the reference, similar (BLOSUM62 > 0) or different.",
    "Nucleotide": "A, C, G and T/U in their usual colours.",
}

RESIDUE = {**dict.fromkeys("AILMFWV", "#80a0f0"), "C": "#f08080", **dict.fromkeys("KR", "#f01505"),
           **dict.fromkeys("DE", "#c048c0"), **dict.fromkeys("NQST", "#15c015"), "G": "#f09048", "P": "#c0c000",
           **dict.fromkeys("HY", "#15a4a4")}
NUCLEOTIDE = {"A": "#64f73f", "C": "#ffb340", "G": "#eb413c", "T": "#3c88ee", "U": "#3c88ee"}
ZAPPO = {**dict.fromkeys("ILVAM", "#ffafaf"), **dict.fromkeys("FWY", "#ffc800"), **dict.fromkeys("KRH", "#6464ff"),
         **dict.fromkeys("DE", "#ff0000"), **dict.fromkeys("STNQ", "#00ff00"), **dict.fromkeys("PG", "#ff00ff"),
         "C": "#ffff00"}
ZAPPO_LEGEND = [("Aliphatic ILVAM", "#ffafaf"), ("Aromatic FWY", "#ffc800"), ("Positive KRH", "#6464ff"),
                ("Negative DE", "#ff0000"), ("Hydrophilic STNQ", "#00ff00"), ("Conformational PG", "#ff00ff"),
                ("Cysteine C", "#ffff00")]
TAYLOR = {"A": "#ccff00", "R": "#0000ff", "N": "#cc00ff", "D": "#ff0000", "C": "#ffff00", "Q": "#ff00cc",
          "E": "#ff0066", "G": "#ff9900", "H": "#0066ff", "I": "#66ff00", "L": "#33ff00", "K": "#6600ff",
          "M": "#00ff00", "F": "#00ff66", "P": "#ffcc00", "S": "#ff3300", "T": "#ff6600", "W": "#00ccff",
          "Y": "#00ffcc", "V": "#99ff00"}
KYTE_DOOLITTLE = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
                  "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5,
                  "N": -3.5, "K": -3.9, "R": -4.5}
HYDRO_RAMP = ("#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b")
IDENTITY_SHADES = ((0.8, "#6464ff", "> 80%"), (0.6, "#9999ff", "> 60%"), (0.4, "#ccccff", "> 40%"))
CONSERVATION_RAMP = ("#f0fdfa", "#ccfbf1", "#99f6e4", "#5eead4", "#2dd4bf", "#14b8a6", "#0d9488", "#0f766e",
                     "#115e59", "#134e4a")
SIMILARITY = (("Identical", "#5fb8a8"), ("Similar (BLOSUM62 > 0)", "#cde9e3"), ("Different", "#f6c9c4"))

# Jalview ClustalX colours and rules: (colour, residues coloured, [(threshold, group, combined?)])
# A "combined" group means the summed fraction of its residues; otherwise any single residue of the group.
CLUSTALX_COLOURS = {"BLUE": "#80a0f0", "RED": "#f01505", "MAGENTA": "#c048c0", "GREEN": "#15c015",
                    "PINK": "#f08080", "ORANGE": "#f09048", "YELLOW": "#c0c000", "CYAN": "#15a4a4"}
HYDROPHOBIC = "WLVIMAFCHP"
CLUSTALX_RULES = [
    ("PINK", "C", [(0.85, "C", True)]),
    ("BLUE", "AILMFWVC", [(0.60, HYDROPHOBIC, True)]),
    ("RED", "KR", [(0.60, "KR", True), (0.80, "KRQ", False)]),
    ("MAGENTA", "E", [(0.60, "KR", True), (0.50, "QE", True), (0.50, "ED", True), (0.85, "EQD", False)]),
    ("MAGENTA", "D", [(0.60, "KR", True), (0.85, "KRQ", False), (0.50, "ED", True)]),
    ("GREEN", "N", [(0.50, "N", True), (0.85, "NY", False)]),
    ("GREEN", "Q", [(0.60, "KR", True), (0.50, "QE", True), (0.85, "QEKR", False)]),
    ("GREEN", "ST", [(0.60, HYDROPHOBIC, True), (0.50, "TS", True), (0.85, "ST", False)]),
    ("ORANGE", "G", [(0.0, "G", True)]),
    ("YELLOW", "P", [(0.0, "P", True)]),
    ("CYAN", "HY", [(0.60, HYDROPHOBIC, True), (0.85, "WYACPQFHILMV", False)]),
]
CLUSTALX_LEGEND = [("Hydrophobic", "#80a0f0"), ("Positive", "#f01505"), ("Negative", "#c048c0"),
                   ("Polar", "#15c015"), ("Cysteine", "#f08080"), ("Glycine", "#f09048"), ("Proline", "#c0c000"),
                   ("Aromatic", "#15a4a4"), ("Not conserved", WHITE)]


@dataclass
class Coloring:
    index: np.ndarray        # uint8 N x L: index into colors (0 = white)
    colors: list             # hex colours; colors[0] is white
    legend: list             # [(label, hex)]


def _letters(seqs):
    return np.array([list(s.upper()) for s in seqs]) if seqs else np.zeros((0, 0), dtype="<U1")


def _from_lookup(letters, lookup, legend):
    colors = [WHITE] + sorted(set(lookup.values()) - {WHITE})
    pos = {c: k for k, c in enumerate(colors)}
    table = np.zeros(256, dtype=np.uint8)
    for residue, color in lookup.items():
        table[ord(residue)] = pos[color]
    codes = np.frombuffer("".join("".join(r) for r in letters).encode("ascii", "replace"), dtype=np.uint8)
    return Coloring(table[codes].reshape(letters.shape), colors, legend)


def _rule_met(fraction, threshold, group, combined):
    if combined:
        return sum(fraction.get(r, 0.0) for r in group) > threshold
    return any(fraction.get(r, 0.0) > threshold for r in group)


def clustalx_colors(letters):
    n, L = letters.shape
    colors = [WHITE] + list(dict.fromkeys(CLUSTALX_COLOURS.values()))
    pos = {c: k for k, c in enumerate(colors)}
    index = np.zeros((n, L), dtype=np.uint8)
    for j in range(L):
        column = letters[:, j]
        residues, counts = np.unique(column, return_counts=True)
        fraction = {r: c / n for r, c in zip(residues, counts) if r not in GAP_CHARS}
        for residue in fraction:
            color = None
            for name, applies_to, conditions in CLUSTALX_RULES:
                if residue in applies_to and any(_rule_met(fraction, t, g, comb) for t, g, comb in conditions):
                    color = CLUSTALX_COLOURS[name]
                    break
            if color:
                index[column == residue, j] = pos[color]
    return Coloring(index, colors, CLUSTALX_LEGEND)


def percent_identity_colors(letters):
    n, L = letters.shape
    colors = [WHITE] + [c for _, c, _ in IDENTITY_SHADES]
    index = np.zeros((n, L), dtype=np.uint8)
    for j in range(L):
        column = letters[:, j]
        residues = [r for r in column if r not in GAP_CHARS]
        if not residues:
            continue
        values, counts = np.unique(residues, return_counts=True)
        top = values[np.argmax(counts)]
        frac = counts.max() / n
        shade = next((k + 1 for k, (t, _, _) in enumerate(IDENTITY_SHADES) if frac > t), 0)
        index[column == top, j] = shade
    return Coloring(index, colors, [(label, c) for _, c, label in IDENTITY_SHADES])


def conservation_colors(letters, seq_type):
    from modules.conservation import column_conservation  # local: keeps viz importable without the MSA stack

    scores = column_conservation(["".join(r) for r in letters], seq_type)
    colors = [WHITE] + list(CONSERVATION_RAMP)
    bins = np.where(np.isnan(scores), -1, np.minimum((np.nan_to_num(scores) * 10).astype(int), 9))
    index = np.zeros(letters.shape, dtype=np.uint8)
    residue = ~np.isin(letters, list(GAP_CHARS))
    for j, b in enumerate(bins):
        if b >= 0:
            index[residue[:, j], j] = b + 1
    legend = [("Low conservation", CONSERVATION_RAMP[0]), ("High conservation", CONSERVATION_RAMP[-1])]
    return Coloring(index, colors, legend)


def similarity_colors(letters, reference_row, seq_type):
    from modules.pairwise import load_matrix

    colors = [WHITE] + [c for _, c in SIMILARITY]
    codes = np.frombuffer("".join("".join(r) for r in letters).encode("ascii", "replace"),
                          dtype=np.uint8).reshape(letters.shape)
    ref = codes[reference_row][None, :]
    gap = np.isin(codes, [ord(c) for c in GAP_CHARS]) | np.isin(ref, [ord(c) for c in GAP_CHARS])
    similar = np.zeros((256, 256), dtype=bool)
    if seq_type == "Protein":
        blosum = load_matrix("BLOSUM62")
        for a in blosum.alphabet:
            for b in blosum.alphabet:
                if a != "*" and b != "*" and blosum[a][b] > 0:
                    similar[ord(a), ord(b)] = True
    index = np.where(codes == ref, 1, np.where(similar[codes, np.broadcast_to(ref, codes.shape)], 2, 3))
    index[gap] = 0
    return Coloring(index.astype(np.uint8), colors, list(SIMILARITY))


def hydrophobicity_colors(letters):
    lookup = {r: HYDRO_RAMP[min(8, int((v + 4.5) / 9.0 * 9))] for r, v in KYTE_DOOLITTLE.items()}
    return _from_lookup(letters, lookup, [("Hydrophilic", HYDRO_RAMP[0]), ("Hydrophobic", HYDRO_RAMP[-1])])


def color_alignment(seqs, scheme, seq_type="Protein", reference=0):
    """Coloring for aligned sequences under a scheme (see SCHEMES); `reference` is a row index."""
    letters = _letters(seqs)
    if letters.size == 0:
        return Coloring(np.zeros((0, 0), dtype=np.uint8), [WHITE], [])
    if scheme == "Residue":
        return _from_lookup(letters, RESIDUE, CLUSTALX_LEGEND[:-1])
    if scheme == "Nucleotide":
        return _from_lookup(letters, NUCLEOTIDE, [("A", NUCLEOTIDE["A"]), ("C", NUCLEOTIDE["C"]),
                                                  ("G", NUCLEOTIDE["G"]), ("T/U", NUCLEOTIDE["T"])])
    if scheme == "ClustalX":
        return clustalx_colors(letters)
    if scheme == "Zappo":
        return _from_lookup(letters, ZAPPO, ZAPPO_LEGEND)
    if scheme == "Taylor":
        return _from_lookup(letters, TAYLOR, [])
    if scheme == "Hydrophobicity":
        return hydrophobicity_colors(letters)
    if scheme == "Percent identity":
        return percent_identity_colors(letters)
    if scheme == "Conservation":
        return conservation_colors(letters, seq_type)
    if scheme == "Similarity to reference":
        return similarity_colors(letters, reference, seq_type)
    raise ValueError(f"Unknown colour scheme {scheme!r}")
```

- [ ] **Step 4: Viewer and colouring in `modules/viz.py`; use it in `modules/msa.py`**

Save each diff to a file and `git apply` it from the repository root. Tasks 1–4 do not touch these files, so the diffs apply cleanly.

```diff
diff --git a/modules/viz.py b/modules/viz.py
index e5861bb..3e57c42 100644
--- a/modules/viz.py
+++ b/modules/viz.py
@@ -6,6 +6,7 @@ import numpy as np
 import plotly.graph_objects as go
 import streamlit as st
 from matplotlib.colors import ListedColormap
+from modules.msa_colors import DESCRIPTIONS, SCHEMES, color_alignment
 from modules.ui import show_plotly
 
 ALPHABET = "-ACDEFGHIKLMNPQRSTVWYXBZJOU*"
@@ -66,13 +67,20 @@ def render_mode(n_rows, n_cols, method):
     return "interactive_hover" if cells <= HOVER_LIMIT else "interactive"
 
 
-def msa_figure_plotly(codes, ids, seq_type, hover=True):
-    colors = palette(seq_type)
+def _discrete_colorscale(colors):
     k = len(colors)
-    colorscale = []
+    scale = []
     for i, color in enumerate(colors):
-        colorscale += [[i / k, color], [(i + 1) / k, color]]
-    heatmap = dict(z=codes, x=np.arange(1, codes.shape[1] + 1), y=list(ids), colorscale=colorscale,
+        scale += [[i / k, color], [(i + 1) / k, color]]
+    return scale
+
+
+def msa_figure_plotly(codes, ids, seq_type, hover=True, coloring=None):
+    """Overview heatmap; `coloring` (msa_colors.Coloring) overrides the per-residue palette."""
+    colors = coloring.colors if coloring is not None else palette(seq_type)
+    z = coloring.index if coloring is not None else codes
+    k = len(colors)
+    heatmap = dict(z=z, x=np.arange(1, codes.shape[1] + 1), y=list(ids), colorscale=_discrete_colorscale(colors),
                    zmin=-0.5, zmax=k - 0.5, showscale=False)
     if hover:
         heatmap["text"] = codes_to_letters(codes)
@@ -93,13 +101,14 @@ def msa_figure_plotly(codes, ids, seq_type, hover=True):
     return fig
 
 
-def msa_figure_matplotlib(codes, ids, seq_type):
-    colors = palette(seq_type)
+def msa_figure_matplotlib(codes, ids, seq_type, coloring=None):
+    colors = coloring.colors if coloring is not None else palette(seq_type)
+    z = coloring.index if coloring is not None else codes
     k = len(colors)
     n_rows, n_cols = codes.shape
     fig, ax = plt.subplots(figsize=(20, max(3.5, min(30, 0.25 * len(ids) + 2))))
     # extent makes column k (1-based) sit at x=k, matching the Plotly view.
-    ax.imshow(codes, cmap=ListedColormap(colors), vmin=-0.5, vmax=k - 0.5, aspect="auto",
+    ax.imshow(z, cmap=ListedColormap(colors), vmin=-0.5, vmax=k - 0.5, aspect="auto",
               interpolation="nearest", extent=(0.5, n_cols + 0.5, n_rows - 0.5, -0.5))
     ax.set_title("Multiple Sequence Alignment", fontsize=14)
     ax.set_xlabel("Alignment position", fontsize=12)
@@ -108,21 +117,27 @@ def msa_figure_matplotlib(codes, ids, seq_type):
         ax.set_yticklabels(ids, fontsize=8)
     else:
         ax.set_ylabel("Sequence", fontsize=12)
-    # One legend entry per distinct color, labelled with the residues that share it.
-    groups = {}
-    for c in np.unique(codes):
-        groups.setdefault(colors[c], []).append(ALPHABET[c])
-    handles = [mpatches.Patch(facecolor=color, edgecolor="#999999",
-                              label="gap" if letters == ["-"] else "".join(letters))
-               for color, letters in groups.items()]
-    ax.legend(handles=handles, title="Residues", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
+    if coloring is not None and coloring.legend:
+        handles = [mpatches.Patch(facecolor=color, edgecolor="#999999", label=label)
+                   for label, color in coloring.legend]
+    else:  # one legend entry per distinct color, labelled with the residues that share it
+        groups = {}
+        for c in np.unique(codes):
+            groups.setdefault(colors[c], []).append(ALPHABET[c])
+        handles = [mpatches.Patch(facecolor=color, edgecolor="#999999",
+                                  label="gap" if letters == ["-"] else "".join(letters))
+                   for color, letters in groups.items()]
+    if handles:
+        ax.legend(handles=handles, title="Colours", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
     fig.tight_layout()
     return fig
 
 
 @st.cache_data(max_entries=4, show_spinner=False)
-def render_msa_png(codes, ids, seq_type):
-    fig = msa_figure_matplotlib(codes, list(ids), seq_type)
+def render_msa_png(codes, ids, seq_type, scheme=None, reference=0):
+    coloring = color_alignment(["".join(r) for r in codes_to_letters(codes)], scheme, seq_type, reference) \
+        if scheme else None
+    fig = msa_figure_matplotlib(codes, list(ids), seq_type, coloring)
     buf = BytesIO()
     try:
         fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
@@ -131,18 +146,112 @@ def render_msa_png(codes, ids, seq_type):
     return buf.getvalue()
 
 
-def plot_msa(codes, ids, method="Plotly (Interactive)", seq_type="Protein"):
-    """Show an MSA heatmap sized to the alignment, plus a PNG download."""
+def plot_msa(codes, ids, method="Plotly (Interactive)", seq_type="Protein", scheme=None, reference=0):
+    """Show an MSA heatmap sized to the alignment, plus a PNG download. `scheme` is an msa_colors scheme name."""
     if codes.size == 0:
         st.error("No alignment to plot.")
         return
     mode = render_mode(codes.shape[0], codes.shape[1], method)
-    png = render_msa_png(codes, tuple(ids), seq_type)
+    coloring = color_alignment(["".join(r) for r in codes_to_letters(codes)], scheme, seq_type, reference) \
+        if scheme else None
+    png = render_msa_png(codes, tuple(ids), seq_type, scheme, reference)
     if mode == "static":
         if not method.startswith("Matplotlib"):
             st.info("Alignment is too large for the interactive view; showing a static image.")
         st.image(png, width="stretch")
     else:
-        show_plotly(msa_figure_plotly(codes, ids, seq_type, hover=mode == "interactive_hover"), width="stretch")
+        show_plotly(msa_figure_plotly(codes, ids, seq_type, hover=mode == "interactive_hover", coloring=coloring),
+                    width="stretch")
     st.download_button("Download MSA Heatmap (PNG)", data=png, file_name="msa_heatmap.png",
                        mime="image/png", on_click="ignore", icon=":material/download:")
+
+
+VIEWER_WIDTHS = (40, 150, 80)      # min, max, default columns in the sequence viewer window
+VIEWER_MAX_ROWS = 120              # rows shown at once; more sequences get a row window
+
+
+def reference_numbering(ref_aligned):
+    """Reference residue number for every alignment column (None where the reference has a gap)."""
+    numbers, k = [], 0
+    for c in ref_aligned:
+        k += c not in "-."
+        numbers.append(k if c not in "-." else None)
+    return numbers
+
+
+def msa_viewer_figure(codes, ids, coloring, start, width, ref_numbers, conservation, consensus):
+    """Sequence viewer: a window of columns with letters in every cell, a conservation bar and a consensus row
+    above, the x axis labelled in reference numbering every 10 residues."""
+    from plotly.subplots import make_subplots
+
+    end = min(codes.shape[1], start + width)
+    cols = np.arange(start, end)
+    letters = codes_to_letters(codes[:, start:end])
+    z = coloring.index[:, start:end]
+    k = len(coloring.colors)
+    rows = ["consensus"] + list(ids)
+    z = np.vstack([np.zeros((1, len(cols)), dtype=z.dtype), z])
+    text = np.vstack([np.array(list(consensus[start:end]))[None, :], letters])
+    refpos = np.array([[("" if ref_numbers[c] is None else str(ref_numbers[c])) for c in cols]] * len(rows))
+    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.12, 0.88], vertical_spacing=0.01)
+    cons = np.nan_to_num(conservation[start:end])
+    fig.add_trace(go.Bar(x=cols + 1, y=cons, marker_color="#0f766e", showlegend=False,
+                         hovertemplate="Column %{x}: conservation %{y:.2f}<extra></extra>"), row=1, col=1)
+    fig.add_trace(go.Heatmap(z=z, x=cols + 1, y=rows, colorscale=_discrete_colorscale(coloring.colors),
+                             zmin=-0.5, zmax=k - 0.5, showscale=False, text=text, customdata=refpos,
+                             texttemplate="%{text}", textfont=dict(size=11, color="#1a1a1a",
+                                                                   family="IBM Plex Mono, monospace"),
+                             xgap=1, ygap=1,
+                             hovertemplate="%{y}<br>Column %{x} (reference %{customdata})<br>%{text}<extra></extra>"),
+                  row=2, col=1)
+    ticks = [c + 1 for c in cols if ref_numbers[c] is not None and ref_numbers[c] % 10 == 0]
+    fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=[str(ref_numbers[t - 1]) for t in ticks],
+                     row=2, col=1, title_text="Reference position")
+    fig.update_yaxes(range=[0, 1], showticklabels=False, row=1, col=1)
+    fig.update_yaxes(type="category", autorange="reversed", row=2, col=1)
+    fig.update_layout(height=min(4000, 120 + 22 * len(rows)), margin=dict(l=10, r=10, t=10, b=40),
+                      plot_bgcolor="white", bargap=0.1)
+    return fig
+
+
+def alignment_view(pairs, seq_type, plot_method, reference_id):
+    """MSA tab view: colour scheme + overview heatmap or sequence viewer."""
+    from modules.conservation import column_conservation
+    from modules.espript import column_classes, consensus_line
+
+    ids = [i for i, _ in pairs]
+    seqs = [s for _, s in pairs]
+    c1, c2 = st.columns([1, 1])
+    mode = c1.segmented_control("View", ["Overview heatmap", "Sequence viewer"], default="Overview heatmap",
+                                required=True, key="msa_view_mode")
+    schemes = SCHEMES["DNA" if seq_type == "DNA" else "Protein"]
+    scheme = c2.selectbox("Colour scheme", schemes, key="msa_color_scheme")
+    st.caption(DESCRIPTIONS[scheme])
+    reference = ids.index(reference_id) if reference_id in ids else 0
+    codes = alignment_to_codes(seqs)
+    if mode == "Overview heatmap":
+        plot_msa(codes, ids, plot_method, seq_type, scheme, reference)
+        return
+    coloring = color_alignment(seqs, scheme, seq_type, reference)
+    ref_numbers = reference_numbering(seqs[reference])
+    lo, hi, default = VIEWER_WIDTHS
+    c1, c2, c3 = st.columns([1, 2, 1])
+    width = c1.slider("Columns shown", lo, hi, default, 10, key="msa_view_width")
+    n_cols = codes.shape[1]
+    jump = c3.number_input(f"Jump to {ids[reference]} residue", 0, max(n for n in ref_numbers if n) if any(ref_numbers)
+                           else 0, 0, key="msa_view_jump", help="0 keeps the slider position.")
+    if jump:
+        target = next((c for c, n in enumerate(ref_numbers) if n == jump), 0)
+        st.session_state.msa_view_start = max(1, min(n_cols, target + 1 - width // 2))
+    start = c2.slider("Start column", 1, max(1, n_cols - width + 1) if n_cols > width else 1,
+                      key="msa_view_start") if n_cols > width else 1
+    rows = np.arange(len(ids))
+    if len(ids) > VIEWER_MAX_ROWS:
+        first = st.slider("First sequence", 1, len(ids) - VIEWER_MAX_ROWS + 1, 1, key="msa_view_row")
+        rows = rows[first - 1:first - 1 + VIEWER_MAX_ROWS]
+    conservation = column_conservation(seqs, seq_type)
+    consensus = consensus_line(seqs, column_classes(seqs, 0.7, seq_type))
+    # column statistics come from the whole alignment; a row window only slices the colouring
+    sub = type(coloring)(coloring.index[rows], coloring.colors, coloring.legend)
+    show_plotly(msa_viewer_figure(codes[rows], [ids[r] for r in rows], sub, start - 1, width, ref_numbers,
+                                  conservation, consensus), width="stretch")
```

```diff
diff --git a/modules/msa.py b/modules/msa.py
index b0a1fec..81d4de5 100644
--- a/modules/msa.py
+++ b/modules/msa.py
@@ -17,7 +17,7 @@ import pandas as pd
 from typing import List, Dict, Tuple
 from modules.alignment_tabs import consensus_tab, figure_tab, motifs_tab
 from modules.utils import have_params_changed, stale_notice
-from modules.viz import alignment_to_codes, plot_msa
+from modules.viz import alignment_to_codes, alignment_view
 from modules.seqtools import clean_sequence, fingerprint
 from modules.conservation import (
     CONSERVATION_COLORS, column_conservation, conservation_grade, identity_matrix_frame, logo_figure,
@@ -118,7 +118,7 @@ def msa_section(sequences, seq_type):
         st.subheader("MSA result")
         if not view_empty:
             st.write(f"Total sequences: {len(view)}. MSA length: {len(view[0][1])}")
-            plot_msa(alignment_to_codes([s for _, s in view]), [i for i, _ in view], plot_method, seq_type)
+            alignment_view(view, seq_type, plot_method, ref_seq_id)
             try:
                 st.download_button(label="Download MSA", data=serialize_alignment(view, msa_output_format),
                                    file_name=f"msa_alignment.{msa_output_format}", mime="text/plain",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PATH=/home/dzyla/miniconda3/bin:$PATH .venv/bin/python -m pytest -q`, then `.venv/bin/python precommit_check.py`.
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add modules/msa_colors.py modules/viz.py modules/msa.py tests/test_msa_colors.py tests/test_app_flow.py
git commit -m "feat: MSA colour schemes (ClustalX, Zappo, Taylor, hydrophobicity, identity, conservation, similarity) and a sequence viewer"
```

