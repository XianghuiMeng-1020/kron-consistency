"""Kron-equivalent series refinement of original undirected edges."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from kron_consistency.config import FRACS, HARM_PROBES
from kron_consistency.operators import canonicalize_sparse, maxabs, undirected_edges


def refinement_counts(m: int, fracs=FRACS) -> list:
    counts = []
    prev = 0
    for f in fracs:
        if f <= 0.0:
            k = 0
        else:
            k = int(round(f * m))
            if m >= 1:
                k = max(1, k)
            k = min(int(m), k)
        k = max(k, prev)
        counts.append(int(k))
        prev = k
    return counts


def refine_graph(n: int, ei, ej, ew, split_idx: np.ndarray):
    split = set(int(x) for x in split_idx.tolist())
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    inserted = []
    for k in range(int(ei.size)):
        i = int(ei[k])
        j = int(ej[k])
        w = float(ew[k])
        if k in split:
            inserted.append((i, j, w))
        else:
            rows.extend((i, j))
            cols.extend((j, i))
            data.extend((w, w))
    n_new = len(inserted)
    n_full = n + n_new
    for t, (i, j, w) in enumerate(inserted):
        z = n + t
        ww = 2.0 * w
        rows.extend((i, z, j, z))
        cols.extend((z, i, z, j))
        data.extend((ww, ww, ww, ww))
    W = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_full, n_full),
        dtype=np.float64,
    )
    return canonicalize_sparse(W), inserted


def build_E(n: int, inserted) -> sparse.csr_matrix:
    n_full = n + len(inserted)
    rows = list(range(n))
    cols = list(range(n))
    data = [1.0] * n
    for t, (i, j, _w) in enumerate(inserted):
        z = n + t
        rows.extend((z, z))
        cols.extend((i, j))
        data.extend((0.5, 0.5))
    E = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_full, n),
        dtype=np.float64,
    )
    return canonicalize_sparse(E)


def schur_onto_original(L: sparse.csr_matrix, n_b: int) -> sparse.csr_matrix:
    """Kron / DtN on the original nodes. Uses the diagonal interior block. No ridge."""
    L = canonicalize_sparse(L)
    if L.shape[0] == n_b:
        return L
    L_bb = L[:n_b, :n_b]
    L_bi = L[:n_b, n_b:]
    L_ib = L[n_b:, :n_b]
    L_ii = L[n_b:, n_b:]
    off = L_ii.copy()
    off.setdiag(0.0)
    off.eliminate_zeros()
    if off.nnz != 0:
        raise RuntimeError(
            f"L_II is not diagonal: nnz_off={off.nnz}, maxabs={maxabs(off)}"
        )
    diag = np.asarray(L_ii.diagonal()).ravel()
    if np.any(diag <= 0):
        raise RuntimeError("L_II diagonal is not strictly positive")
    inv = 1.0 / diag
    corr = L_bi @ sparse.diags(inv) @ L_ib
    return canonicalize_sparse(L_bb - corr)


def collapse_to_original(n: int, W_full: sparse.csr_matrix, inserted) -> sparse.csr_matrix:
    W_b = canonicalize_sparse(W_full[:n, :n])
    if not inserted:
        return W_b
    rows = []
    cols = []
    data = []
    for i, j, w in inserted:
        rows.extend((i, j))
        cols.extend((j, i))
        data.extend((float(w), float(w)))
    extra = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n, n),
        dtype=np.float64,
    )
    return canonicalize_sparse(W_b + extra)


def W_from_laplacian(L: sparse.csr_matrix) -> sparse.csr_matrix:
    W = canonicalize_sparse(-L)
    W.setdiag(0.0)
    return canonicalize_sparse(W)


def harmonicity(L: sparse.csr_matrix, E: sparse.csr_matrix, n: int, rng: np.random.Generator) -> dict:
    n_i = L.shape[0] - n
    if n_i <= 0:
        return {"n_probes": 0, "maxabs_interior": 0.0, "rms_interior": 0.0}
    worst = 0.0
    rss = 0.0
    n_ent = 0
    for _ in range(HARM_PROBES):
        x = rng.normal(size=n)
        r = L @ (E @ x)
        interior = np.asarray(r).ravel()[n:]
        worst = max(worst, float(np.max(np.abs(interior))) if interior.size else 0.0)
        rss += float(np.sum(interior * interior))
        n_ent += interior.size
    return {
        "n_probes": HARM_PROBES,
        "maxabs_interior": worst,
        "rms_interior": float(np.sqrt(rss / max(n_ent, 1))),
    }


def graph_stats(name: str, W: sparse.csr_matrix, source: dict) -> dict:
    n = W.shape[0]
    ei, ej, ew = undirected_edges(W)
    d = np.asarray(W.sum(axis=1)).ravel()
    n_comp = int(sparse.csgraph.connected_components(W, directed=False, return_labels=False))
    return {
        "name": name,
        "n_nodes": int(n),
        "n_undirected_edges": int(ei.size),
        "n_components": n_comp,
        "min_degree": float(d.min()) if n else 0.0,
        "max_degree": float(d.max()) if n else 0.0,
        "n_zero_degree": int(np.sum(d <= 0.0)),
        "weight_min": float(ew.min()) if ei.size else 0.0,
        "weight_max": float(ew.max()) if ei.size else 0.0,
        "symmetric": bool((W - W.T).nnz == 0 or maxabs(W - W.T) == 0.0),
        "diag_zero": bool(np.all(W.diagonal() == 0.0)),
        "source": source,
        "refinement_counts": {
            f"{int(f * 100) if f else 0}%": k
            for f, k in zip(FRACS, refinement_counts(int(ei.size)))
        },
    }
