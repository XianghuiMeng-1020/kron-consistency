"""Machine-precision tests for Kron / series identities."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kron_consistency.theory import (  # noqa: E402
    S_g_E, cheb_apply, gcn_mat, harmonic_lift, laplacian, lsym, obstruction,
    poly_apply, rw_mat, schur, series_split_one, triangle, triangle_plus_pendant,
    two_terminal_path,
)

TOL = 1e-10
rng = np.random.default_rng(0)


def assert_small(name, A, tol=TOL):
    nrm = float(np.linalg.norm(A))
    ok = nrm < tol
    print(f"{'PASS' if ok else 'FAIL':4s}  {name:62s}  ||.||={nrm:.3e}")
    if not ok:
        raise AssertionError(f"{name}: {nrm}")


def assert_large(name, A, floor=1e-6):
    nrm = float(np.linalg.norm(A))
    ok = nrm > floor
    print(f"{'PASS' if ok else 'FAIL':4s}  {name:62s}  ||.||={nrm:.3e}")
    if not ok:
        raise AssertionError(f"{name} unexpectedly small: {nrm}")


def test_affine_universal():
    graphs = []
    W, B = two_terminal_path()
    graphs.append(("path2", W, B))
    W, B = triangle()
    graphs.append(("triangle", W, B))
    W, B = triangle_plus_pendant()
    graphs.append(("tri+pendant", W, B))
    W, B = triangle()
    graphs.append(("tri-series", series_split_one(W, 0, 1, 2), B))
    for name, W, B in graphs:
        L = laplacian(W)
        LK = schur(L, B)
        for a0, a1 in [(1.0, 0.0), (1.0, -0.2), (0.0, 1.0), (2.0, 3.5)]:
            left = S_g_E(L, B, poly_apply([a0, a1]))
            right = a0 * np.eye(len(B)) + a1 * LK
            assert_small(f"affine ({a0},{a1}) on {name}", left - right)


def test_two_terminal_NOT_L2_counterexample():
    W, B = two_terminal_path(1.7)
    W2 = series_split_one(W, 0, 1, 2)
    L2 = laplacian(W2)
    LK = schur(L2, B)
    left = S_g_E(L2, B, poly_apply([0.0, 0.0, 1.0]))
    obst = obstruction(L2, B)
    assert_small("2-terminal series: S L^2 E - L_K^2", left - LK @ LK)
    assert_small("2-terminal series: obstruction M L_K", obst)


def test_triangle_series_IS_L2_counterexample():
    W, B = triangle(1.0)
    W2 = series_split_one(W, 0, 1, 2)
    L = laplacian(W2)
    LK = schur(L, B)
    L0 = laplacian(W)
    assert_small("triangle series preserves L_K", LK - L0)
    left = S_g_E(L, B, poly_apply([0.0, 0.0, 1.0]))
    gap = left - LK @ LK
    assert_large("triangle series: L^2 Kron gap", gap, floor=1e-4)
    assert_small("triangle series: obstruction identity", gap - obstruction(L, B))


def test_pendant_L2_counterexample():
    W, B = triangle_plus_pendant()
    L = laplacian(W)
    LK = schur(L, B)
    left = S_g_E(L, B, poly_apply([0.0, 0.0, 1.0]))
    assert_large("tri+pendant: L^2 Kron gap", left - LK @ LK, floor=1e-4)


def test_scaling_characterizes_degree():
    """If p has a term of degree >= 2, scaled triangle-series fails for generic t."""
    W0, B = triangle(1.0)
    W2 = series_split_one(W0, 0, 1, 2)
    L1 = laplacian(W2)
    # scale weights by t: L(t)=t L(1), E scale-invariant
    coeffs = [0.4, -1.1, 0.7, -0.2]  # deg 3

    def gap_at(t):
        L = t * L1
        LK = schur(L, B)
        left = S_g_E(L, B, poly_apply(coeffs))
        right = np.zeros_like(LK)
        P = np.eye(len(B))
        for a in coeffs:
            right = right + a * P
            P = LK @ P if False else P
        # p(LK)
        acc = np.zeros_like(LK)
        Pk = np.eye(len(B))
        for a in coeffs:
            acc = acc + a * Pk
            Pk = LK @ Pk
        return left - acc

    vals = [np.linalg.norm(gap_at(t)) for t in (0.3, 1.0, 2.5, 7.0)]
    print(f"INFO  scaled deg-3 gaps: {['%.3e' % v for v in vals]}")
    assert max(vals) > 1e-3, "deg>=2 polynomial unexpectedly Kron-consistent on scaled triangle-series"


def test_series_midpoint_first_order_arbitrary_signal():
    W, B = triangle()
    W2 = series_split_one(W, 0, 1, 2)
    x0 = rng.normal(size=3)
    x2 = np.zeros(4)
    x2[:3] = x0
    x2[3] = 0.5 * (x0[0] + x0[1])
    L0, L2 = laplacian(W), laplacian(W2)
    assert_small("series midpoint: Lx on original nodes", (L0 @ x0) - (L2 @ x2)[:3])


def test_normalized_counterexamples():
    W, B = triangle()
    W2 = series_split_one(W, 0, 1, 2)
    vB = rng.normal(size=3)
    x0 = vB
    x2 = harmonic_lift(laplacian(W2), B, vB)
    # RW
    y0 = rw_mat(W) @ x0
    y2 = rw_mat(W2) @ x2
    assert_large("RW series gap on B", y0 - y2[B], floor=1e-4)
    # GCN
    y0 = gcn_mat(W) @ x0
    y2 = gcn_mat(W2) @ x2
    assert_large("GCN series gap on B", y0 - y2[B], floor=1e-4)
    # Cheb
    y0 = cheb_apply(W, x0)
    y2 = cheb_apply(W2, x2)
    assert_large("Cheb series gap on B", y0 - y2[B], floor=1e-4)
    # weighted degree of endpoints changes
    assert abs(W2.sum(1)[0] - W.sum(1)[0]) > 1e-12


def main():
    test_affine_universal()
    test_two_terminal_NOT_L2_counterexample()
    test_triangle_series_IS_L2_counterexample()
    test_pendant_L2_counterexample()
    test_scaling_characterizes_degree()
    test_series_midpoint_first_order_arbitrary_signal()
    test_normalized_counterexamples()
    print("ALL THEORY TESTS PASSED")


if __name__ == "__main__":
    main()
