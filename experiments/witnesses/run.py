"""Print the analytic witnesses: two-terminal split and triangle series.

The reported numbers are Frobenius gaps
    || S Phi_G E - Phi_{G_K} ||_F
under harmonic lifting. Cheb-2 uses lambda_max = 2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kron_consistency.theory import (  # noqa: E402
    S_g_E,
    gcn_mat,
    laplacian,
    poly_apply,
    rw_mat,
    schur,
    series_split_one,
    triangle,
    two_terminal_path,
)


def fro(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, "fro"))


def apply_matrix(M: np.ndarray):
    def apply(_L, x):
        return M @ x

    return apply


def cheb_matrix(W: np.ndarray, coeffs=(0.5, 0.35, 0.15)) -> np.ndarray:
    d = np.maximum(W.sum(axis=1), 1e-15)
    scale = 1.0 / np.sqrt(d)
    lt = -(scale[:, None] * W) * scale[None, :]
    eye = np.eye(W.shape[0])
    t0, t1 = eye, lt
    t2 = 2.0 * (lt @ t1) - t0
    c0, c1, c2 = coeffs
    return c0 * t0 + c1 * t1 + c2 * t2


def gaps(W0: np.ndarray, W: np.ndarray, B: np.ndarray) -> dict:
    L = laplacian(W)
    LK = schur(L, B)
    l2 = fro(S_g_E(L, B, poly_apply([0.0, 0.0, 1.0])) - LK @ LK)
    p2 = fro(S_g_E(L, B, apply_matrix(rw_mat(W) @ rw_mat(W))) - rw_mat(W0) @ rw_mat(W0))
    a2 = fro(S_g_E(L, B, apply_matrix(gcn_mat(W) @ gcn_mat(W))) - gcn_mat(W0) @ gcn_mat(W0))
    c2 = fro(S_g_E(L, B, apply_matrix(cheb_matrix(W))) - cheb_matrix(W0))
    return {"L2": l2, "P2": p2, "Ah2": a2, "Cheb2": c2}


def main() -> None:
    W0, B = two_terminal_path(1.0)
    split = gaps(W0, series_split_one(W0, 0, 1, 2), B)

    W0, B = triangle(1.0)
    tri = gaps(W0, series_split_one(W0, 0, 1, 2), B)

    print()
    print(f"{'processor':<10} {'two-terminal split':>20} {'triangle series':>18}")
    print("-" * 50)
    print(f"{'L^2':<10} {split['L2']:>20.2e} {tri['L2']:>18.2f}")
    print(f"{'P^2':<10} {'—':>20} {tri['P2']:>18.2f}")
    print(f"{'Ahat^2':<10} {'—':>20} {tri['Ah2']:>18.2f}")
    print(f"{'Cheb-2':<10} {'—':>20} {tri['Cheb2']:>18.2f}")
    print()


if __name__ == "__main__":
    main()
