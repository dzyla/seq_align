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


CLUSTAL_HEADERS = ("CLUSTAL", "PROBCONS", "MUSCLE", "MSAPROBS", "Kalign", "Biopython")   # Bio.AlignIO's
ANNOTATION_ROW = re.compile(r"^(ss_|sa_)|^consensus$", re.IGNORECASE)   # HH-suite rows, dropped as reformat.pl does


def parse_alignment(text):
    """A3M (lowercase insertions and '.' removed), Stockholm, Clustal, or aligned FASTA -> [(id, aligned sequence)].
    HH-suite annotation rows (ss_pred, ss_conf, sa_..., Consensus) are dropped. Sequences whose length differs
    from the first are kept here and dropped (and counted) by prepare_msa."""
    stripped = text.lstrip()
    if stripped.startswith(CLUSTAL_HEADERS):
        from io import StringIO

        from Bio import AlignIO

        rows = [(r.id, str(r.seq)) for r in AlignIO.read(StringIO(stripped), "clustal")]
    elif stripped.startswith("# STOCKHOLM"):
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
    rows = [(i, s) for i, s in rows if not ANNOTATION_ROW.search(i)]
    if len({len(s) for _, s in rows}) > 1:                      # A3M: lowercase and '.' are insertions
        return [(i, re.sub(r"[a-z.]", "", s)) for i, s in rows]
    return [(i, s.upper().replace(".", "-")) for i, s in rows]   # aligned FASTA / Stockholm


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
    flat = np.int32 if Lq * Lq < 2 ** 31 else np.int64     # int32 at the PLM caps halves the index memory
    up = rows_.astype(flat) * flat(Lq) + cols_
    lo = cols_.astype(flat) * flat(Lq) + rows_
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
        else:                                     # no acceptable step: keep theta, restart from the gradient
            S, Y = [], []
            continue
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
