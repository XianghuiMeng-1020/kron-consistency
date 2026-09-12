"""Frozen processors and sparse linear-algebra primitives."""
from __future__ import annotations

import platform

import numpy as np
from scipy import sparse

from kron_consistency.config import ALPHA_AFF, ALPHA_QUAD, CHEB_C, SPECTRAL_N_MAX


def software_versions() -> dict:
    import scipy

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def canonicalize_sparse(A) -> sparse.csr_matrix:
    if not sparse.issparse(A):
        A = sparse.csr_matrix(np.asarray(A, dtype=np.float64))
    else:
        A = A.tocsr().astype(np.float64, copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    return A


def to_csr_adj(W) -> sparse.csr_matrix:
    A = canonicalize_sparse(W)
    A.setdiag(0.0)
    A.eliminate_zeros()
    A = 0.5 * (A + A.T)
    return canonicalize_sparse(A)


def undirected_edges(W: sparse.csr_matrix):
    A = sparse.triu(W, k=1).tocoo()
    A.sum_duplicates()
    order = np.lexsort((A.col, A.row))
    i = A.row[order].astype(np.int64)
    j = A.col[order].astype(np.int64)
    w = A.data[order].astype(np.float64)
    return i, j, w


def laplacian(W: sparse.csr_matrix) -> sparse.csr_matrix:
    W = canonicalize_sparse(W)
    d = np.asarray(W.sum(axis=1)).ravel()
    return canonicalize_sparse(sparse.diags(d) - W)


def fro_norm(A) -> float:
    if sparse.issparse(A):
        A = A.tocsr()
        A.sum_duplicates()
        if A.nnz == 0:
            return 0.0
        return float(np.linalg.norm(A.data))
    return float(np.linalg.norm(np.asarray(A), "fro"))


def maxabs(A) -> float:
    if sparse.issparse(A):
        A = A.tocsr()
        A.sum_duplicates()
        if A.nnz == 0:
            return 0.0
        return float(np.max(np.abs(A.data)))
    a = np.asarray(A)
    return float(np.max(np.abs(a))) if a.size else 0.0


def spectral_rel(D, T, n: int):
    if n > SPECTRAL_N_MAX:
        return None
    Dd = D.toarray() if sparse.issparse(D) else np.asarray(D)
    Td = T.toarray() if sparse.issparse(T) else np.asarray(T)
    n2_t = float(np.linalg.norm(Td, 2))
    n2_d = float(np.linalg.norm(Dd, 2))
    return {
        "abs_2": n2_d,
        "delta_2": (n2_d / n2_t) if n2_t > 0 else float("nan"),
    }


def residual_pair(A: sparse.spmatrix, B: sparse.spmatrix) -> dict:
    D = canonicalize_sparse(A) - canonicalize_sparse(B)
    D = D.tocsr()
    D.sum_duplicates()
    return {
        "frobenius": fro_norm(D),
        "maxabs": maxabs(D),
        "nnz": int(D.nnz),
        "exactly_zero": bool(D.nnz == 0 or maxabs(D) == 0.0),
    }


def random_walk(W: sparse.csr_matrix) -> sparse.csr_matrix:
    d = np.asarray(W.sum(axis=1)).ravel()
    invd = np.zeros_like(d)
    nz = d > 0.0
    invd[nz] = 1.0 / d[nz]
    return canonicalize_sparse(sparse.diags(invd) @ W)


def ahat(W: sparse.csr_matrix) -> sparse.csr_matrix:
    n = W.shape[0]
    Wt = canonicalize_sparse(W + sparse.eye(n, dtype=np.float64, format="csr"))
    d = np.asarray(Wt.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(d, 0.0))
    return canonicalize_sparse(sparse.diags(dinv) @ Wt @ sparse.diags(dinv))


def ltilde(W: sparse.csr_matrix) -> sparse.csr_matrix:
    """L_sym - I with lambda_max = 2 fixed, i.e. -D^{-1/2} W D^{-1/2}."""
    d = np.asarray(W.sum(axis=1)).ravel()
    dinv = np.zeros_like(d)
    nz = d > 0.0
    dinv[nz] = 1.0 / np.sqrt(d[nz])
    nrm = sparse.diags(dinv) @ W @ sparse.diags(dinv)
    return canonicalize_sparse(-nrm)


def processor_matrix(name: str, W: sparse.csr_matrix) -> sparse.csr_matrix:
    n = W.shape[0]
    I = sparse.eye(n, dtype=np.float64, format="csr")
    if name == "affine":
        L = laplacian(W)
        return canonicalize_sparse(I - ALPHA_AFF * L)
    if name == "quadratic":
        L = laplacian(W)
        return canonicalize_sparse(I - ALPHA_QUAD * L + 0.5 * (ALPHA_QUAD ** 2) * (L @ L))
    if name == "rw2":
        P = random_walk(W)
        return canonicalize_sparse(P @ P)
    if name == "norm2":
        A = ahat(W)
        return canonicalize_sparse(A @ A)
    if name == "cheb2":
        Lt = ltilde(W)
        t0 = I
        t1 = Lt
        t2 = canonicalize_sparse(2.0 * (Lt @ Lt) - t0)
        c0, c1, c2 = CHEB_C
        return canonicalize_sparse(c0 * t0 + c1 * t1 + c2 * t2)
    raise KeyError(name)


def processor_defs() -> dict:
    return {
        "affine": {
            "formula": "I - 0.15 L",
            "role": "negative_control",
            "class": "affine_combinatorial_laplacian",
        },
        "quadratic": {
            "formula": "I - 0.08 L + 0.5*(0.08)^2 L^2",
            "role": "primary",
            "class": "quadratic_combinatorial_laplacian",
        },
        "rw2": {
            "formula": "P^2, P = D^{-1} W; zero-degree rows of P are 0",
            "role": "primary",
            "class": "two_hop_random_walk",
        },
        "norm2": {
            "formula": "Ahat^2, Ahat = Dtilde^{-1/2}(W+I)Dtilde^{-1/2}",
            "role": "primary",
            "label": "two-hop GCN normalization / fixed two-hop normalized propagation",
            "class": "two_hop_normalized_adjacency",
            "not": "trained_GCN",
        },
        "cheb2": {
            "formula": "0.5 T0 + 0.35 T1 + 0.15 T2 on L_tilde = L_sym - I, lambda_max=2 fixed",
            "role": "optional_fixed",
            "class": "fixed_chebyshev2_propagation",
            "lambda_max": 2.0,
        },
    }


def summarize(vals) -> dict:
    a = np.asarray(list(vals), dtype=np.float64)
    if a.size == 0:
        return {"median": None, "mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n": int(a.size),
    }


def iqr_summary(vals) -> dict:
    a = np.asarray(list(vals), dtype=np.float64)
    q1, q3 = np.percentile(a, [25.0, 75.0])
    s = summarize(a)
    s.update({
        "iqr": float(q3 - q1),
        "q25": float(q1),
        "q75": float(q3),
        "graph_medians": [float(x) for x in a],
    })
    return s


def defect_record(T_r, T_o, n: int) -> dict:
    if sparse.issparse(T_r) or sparse.issparse(T_o):
        D = canonicalize_sparse(T_r) - canonicalize_sparse(T_o)
        D = D.tocsr()
        D.sum_duplicates()
        nF = fro_norm(T_o)
        dF = fro_norm(D)
        rec = {
            "delta_F": (dF / nF) if nF > 0 else float("nan"),
            "abs_F": dF,
            "maxabs": maxabs(D),
            "nnz_D": int(D.nnz),
            "norm_F_T": nF,
        }
        spec = spectral_rel(D, T_o, n)
    else:
        D = np.asarray(T_r) - np.asarray(T_o)
        nF = float(np.linalg.norm(np.asarray(T_o), "fro"))
        dF = float(np.linalg.norm(D, "fro"))
        rec = {
            "delta_F": (dF / nF) if nF > 0 else float("nan"),
            "abs_F": dF,
            "maxabs": float(np.max(np.abs(D))) if D.size else 0.0,
            "norm_F_T": nF,
        }
        spec = spectral_rel(D, T_o, n)
    if spec is not None:
        rec.update(spec)
    return rec
