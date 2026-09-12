"""Exact Kron elimination of original vertices (not inserted degree-2 nodes)."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu

from kron_consistency.config import FRACS
from kron_consistency.operators import canonicalize_sparse


def component_anchors(W: sparse.csr_matrix):
    n_comp, labels = sparse.csgraph.connected_components(W, directed=False, return_labels=True)
    anchors = []
    for c in range(int(n_comp)):
        members = np.where(labels == c)[0]
        anchors.append(int(members.min()))
    return int(n_comp), labels.astype(np.int64), np.asarray(anchors, dtype=np.int64)


def elimination_counts(n: int, n_eligible: int, fracs=FRACS) -> list:
    counts = []
    prev = 0
    for f in fracs:
        if f <= 0.0:
            k = 0
        else:
            k = int(round(f * n))
            if n_eligible >= 1:
                k = max(1, k)
            k = min(int(k), int(n_eligible))
        k = max(k, prev)
        counts.append(int(k))
        prev = k
    return counts


def lambda_min_spd(A: sparse.csr_matrix) -> float:
    n = A.shape[0]
    if n == 0:
        return float("nan")
    if n == 1:
        return float(A[0, 0])
    if n <= 250:
        ev = np.linalg.eigvalsh(A.toarray())
        return float(ev.min())
    ev = eigsh(A.tocsr(), k=1, which="SA", return_eigenvectors=False)
    return float(ev[0])


def kron_reduce(L: sparse.csr_matrix, B: np.ndarray, I: np.ndarray):
    """Exact Schur complement. No ridge. Returns L_K, E (original order), diagnostics."""
    n = L.shape[0]
    n_b = int(B.size)
    n_i = int(I.size)
    if n_i == 0:
        E = np.eye(n, dtype=np.float64)
        L_K = L.toarray()
        return L_K, E, {
            "lambda_min_LII": None,
            "lu_min_abs_U": None,
            "n_I": 0,
            "n_B": n_b,
        }

    L_BB = L[B][:, B]
    L_BI = L[B][:, I]
    L_IB = L[I][:, B]
    L_II = canonicalize_sparse(L[I][:, I])
    lam = lambda_min_spd(L_II)
    if not np.isfinite(lam) or lam <= 0.0:
        raise RuntimeError(
            f"L_II not numerically SPD: n_I={n_i}, lambda_min={lam}"
        )

    lu = splu(L_II.tocsc())
    u_diag = np.abs(lu.U.diagonal())
    if u_diag.size == 0 or float(u_diag.min()) == 0.0:
        raise RuntimeError(
            f"SuperLU U-diagonal vanishes: min={float(u_diag.min()) if u_diag.size else 0.0}"
        )
    rhs = L_IB.toarray() if sparse.issparse(L_IB) else np.asarray(L_IB, dtype=np.float64)
    X = lu.solve(rhs)
    L_BB_d = L_BB.toarray() if sparse.issparse(L_BB) else np.asarray(L_BB, dtype=np.float64)
    L_BI_d = L_BI.toarray() if sparse.issparse(L_BI) else np.asarray(L_BI, dtype=np.float64)
    L_K = L_BB_d - L_BI_d @ X

    E = np.zeros((n, n_b), dtype=np.float64)
    E[B, :] = np.eye(n_b, dtype=np.float64)
    E[I, :] = -X
    return L_K, E, {
        "lambda_min_LII": float(lam),
        "lu_min_abs_U": float(u_diag.min()),
        "n_I": n_i,
        "n_B": n_b,
    }


def W_from_LK(L_K: np.ndarray, tol: float):
    off = L_K.copy()
    np.fill_diagonal(off, 0.0)
    pos = off[off > tol]
    n_pos = int(pos.size)
    max_pos = float(pos.max()) if n_pos else 0.0
    W = -L_K.copy()
    np.fill_diagonal(W, 0.0)
    n_clip = int(np.sum(np.abs(W) < tol))
    W[np.abs(W) < tol] = 0.0
    n_neg_weight = int(np.sum(W < -tol))
    return canonicalize_sparse(W), {
        "fp_tol": tol,
        "n_clipped_abs_lt_tol": n_clip,
        "n_L_offdiag_gt_tol": n_pos,
        "max_L_offdiag_positive": max_pos,
        "n_W_meaningfully_negative": n_neg_weight,
    }


def kron_closure(L_K: np.ndarray, L: sparse.csr_matrix, E: np.ndarray, B: np.ndarray):
    n_b = L_K.shape[0]
    sym = float(np.max(np.abs(L_K - L_K.T)))
    row = float(np.max(np.abs(L_K.sum(axis=1))))
    off = L_K.copy()
    np.fill_diagonal(off, 0.0)
    max_pos_off = float(np.max(off)) if off.size else 0.0
    min_off = float(np.min(off)) if off.size else 0.0
    ones = np.ones(n_b)
    quad_ones = float(ones @ L_K @ ones)
    LE = L @ E
    sle = LE[B, :]
    sle_res = sle - L_K
    lam_min = None
    if 0 < n_b <= 400:
        lam_min = float(np.linalg.eigvalsh(0.5 * (L_K + L_K.T)).min())
    return {
        "symmetry_maxabs": sym,
        "row_sum_maxabs": row,
        "offdiag_max": max_pos_off,
        "offdiag_min": min_off,
        "ones_quadratic": quad_ones,
        "lambda_min": lam_min,
        "SLE_minus_LK_fro": float(np.linalg.norm(sle_res, "fro")),
        "SLE_minus_LK_maxabs": float(np.max(np.abs(sle_res))) if sle_res.size else 0.0,
    }


def harmonic_residuals(L: sparse.csr_matrix, E: np.ndarray, B: np.ndarray, I: np.ndarray):
    se = E[B, :]
    se_res = se - np.eye(B.size, dtype=np.float64)
    LE = L @ E
    interior = LE[I, :] if I.size else np.zeros((0, B.size))
    return {
        "SE_minus_I_maxabs": float(np.max(np.abs(se_res))) if se_res.size else 0.0,
        "SE_minus_I_fro": float(np.linalg.norm(se_res, "fro")),
        "LE_I_maxabs": float(np.max(np.abs(interior))) if interior.size else 0.0,
        "LE_I_fro": float(np.linalg.norm(interior, "fro")),
    }


def every_component_has_B(labels: np.ndarray, B: np.ndarray) -> bool:
    Bset = set(int(x) for x in B.tolist())
    for c in range(int(labels.max()) + 1 if labels.size else 0):
        members = np.where(labels == c)[0]
        if not any(int(v) in Bset for v in members.tolist()):
            return False
    return True
