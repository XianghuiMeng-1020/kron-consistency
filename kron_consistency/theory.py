"""Exact Kron / series algebra: proofs as executable identities."""
from __future__ import annotations

import numpy as np

EPS = 1e-10


def laplacian(W: np.ndarray) -> np.ndarray:
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return np.diag(W.sum(axis=1)) - W


def schur(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    mask = np.ones(L.shape[0], dtype=bool)
    mask[B] = False
    I = np.where(mask)[0]
    LBB = L[np.ix_(B, B)]
    if I.size == 0:
        return LBB.copy()
    LBI = L[np.ix_(B, I)]
    LII = L[np.ix_(I, I)]
    return LBB - LBI @ np.linalg.solve(LII + 1e-15 * np.eye(I.size), LBI.T)


def harmonic_lift(L: np.ndarray, B: np.ndarray, vB: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    x = np.zeros(n)
    x[B] = vB
    mask = np.ones(n, dtype=bool)
    mask[B] = False
    I = np.where(mask)[0]
    if I.size:
        x[I] = -np.linalg.solve(L[np.ix_(I, I)] + 1e-15 * np.eye(I.size),
                                L[np.ix_(I, B)] @ vB)
    return x


def S_g_E(L: np.ndarray, B: np.ndarray, apply) -> np.ndarray:
    """Restricted action S g(L) E as a |B|x|B| matrix."""
    m = len(B)
    cols = []
    for j in range(m):
        e = np.zeros(m)
        e[j] = 1.0
        x = harmonic_lift(L, B, e)
        y = apply(L, x)
        cols.append(y[B])
    return np.column_stack(cols)


def poly_apply(coeffs):
    def apply(L, x):
        acc = np.zeros_like(x)
        p = x.copy()
        for k, a in enumerate(coeffs):
            if k:
                p = L @ p
            acc = acc + a * p
        return acc
    return apply


def series_split_one(W: np.ndarray, i: int, j: int, m: int = 2) -> np.ndarray:
    """Replace edge ij of weight w by m series edges of weight m*w."""
    w = W[i, j]
    assert w > 0
    n = W.shape[0]
    extra = m - 1
    W2 = np.zeros((n + extra, n + extra))
    W2[:n, :n] = W
    W2[i, j] = W2[j, i] = 0.0
    ww = m * w
    chain = [i] + list(range(n, n + extra)) + [j]
    for a, b in zip(chain[:-1], chain[1:]):
        W2[a, b] = W2[b, a] = ww
    return W2


def two_terminal_path(w: float = 1.0):
    W = np.array([[0.0, w], [w, 0.0]])
    return W, np.array([0, 1])


def triangle(w: float = 1.0):
    W = np.array([[0, w, w], [w, 0, w], [w, w, 0]], dtype=float)
    return W, np.array([0, 1, 2])


def triangle_plus_pendant(w: float = 1.0, wp: float = 2.0):
    W = np.zeros((4, 4))
    W[0, 1] = W[1, 0] = w
    W[1, 2] = W[2, 1] = w
    W[2, 0] = W[0, 2] = w
    W[0, 3] = W[3, 0] = wp
    return W, np.array([0, 1, 2])


def obstruction(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    LK = schur(L, B)
    mask = np.ones(L.shape[0], dtype=bool)
    mask[B] = False
    I = np.where(mask)[0]
    if I.size == 0:
        return np.zeros_like(LK)
    LBI = L[np.ix_(B, I)]
    LII = L[np.ix_(I, I)]
    M = LBI @ np.linalg.solve(LII + 1e-15 * np.eye(I.size), LBI.T)
    return M @ LK


def gcn_mat(W: np.ndarray) -> np.ndarray:
    Wt = W + np.eye(W.shape[0])
    d = np.maximum(Wt.sum(1), 1e-15)
    s = 1.0 / np.sqrt(d)
    return (s[:, None] * Wt) * s[None, :]


def rw_mat(W: np.ndarray) -> np.ndarray:
    d = np.maximum(W.sum(1), 1e-15)
    return W / d[:, None]


def lsym(W: np.ndarray) -> np.ndarray:
    d = np.maximum(W.sum(1), 1e-15)
    s = 1.0 / np.sqrt(d)
    return np.eye(W.shape[0]) - (s[:, None] * W) * s[None, :]


def cheb_apply(W, x, c0=0.5, c1=0.35, c2=0.15):
    Ls = lsym(W)
    ev = np.linalg.eigvalsh(Ls)
    lmax = max(float(ev[-1]), 1e-8)
    Lt = 2.0 * Ls / lmax - np.eye(W.shape[0])
    T0, T1 = x, Lt @ x
    T2 = 2.0 * (Lt @ T1) - T0
    return c0 * T0 + c1 * T1 + c2 * T2
