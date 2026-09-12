"""Exact Kron elimination of original nodes: S Phi_G E versus Phi_{G_K}.

Primary object (signal-independent):

    D = S Phi_G E - Phi_{G_K}
    delta_F^Kron = ||D||_F / ||Phi_{G_K}||_F

I is a nested subset of ORIGINAL vertices (not inserted degree-2 nodes).
B = V \\ I. One protected anchor per connected component is never eliminated.

Processor coefficients are frozen copies of the series-refinement experiment.
Do not edit them after seeing results.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from graphs import load_graphs  # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "results" / (Path(__file__).stem + ".json")

SEEDS = [0, 1, 2, 3, 4]
FRACS = [0.00, 0.01, 0.05, 0.10, 0.25]
ALPHA_AFF = 0.15
ALPHA_QUAD = 0.08
CHEB_C = (0.5, 0.35, 0.15)
AFFINE_ATOL = 1e-8
WK_FP_TOL = 1e-14
SPECTRAL_N_MAX = 400
PROCESSORS = ("affine", "quadratic", "rw2", "norm2", "cheb2")
NON_AFFINE = ("quadratic", "rw2", "norm2", "cheb2")


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


def maxabs_arr(A) -> float:
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


def component_anchors(W: sparse.csr_matrix):
    n_comp, labels = sparse.csgraph.connected_components(W, directed=False, return_labels=True)
    anchors = []
    for c in range(int(n_comp)):
        members = np.where(labels == c)[0]
        anchors.append(int(members.min()))
    return int(n_comp), labels.astype(np.int64), np.asarray(anchors, dtype=np.int64)


def elimination_counts(n: int, n_eligible: int) -> list:
    counts = []
    prev = 0
    for f in FRACS:
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
    X = lu.solve(rhs)  # |I| x |B|  = L_II^{-1} L_IB
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
    # Do not clip meaningful negatives (would come from L_K_ij > tol).
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


def defect_record(T_full, T_kron, n_b: int) -> dict:
    D = np.asarray(T_full if not sparse.issparse(T_full) else T_full.toarray()) - (
        np.asarray(T_kron if not sparse.issparse(T_kron) else T_kron.toarray())
    )
    nF = float(np.linalg.norm(np.asarray(T_kron if not sparse.issparse(T_kron) else T_kron.toarray()), "fro"))
    dF = float(np.linalg.norm(D, "fro"))
    rec = {
        "delta_F": (dF / nF) if nF > 0 else float("nan"),
        "abs_F": dF,
        "maxabs": float(np.max(np.abs(D))) if D.size else 0.0,
        "norm_F_T": nF,
    }
    spec = spectral_rel(D, T_kron, n_b)
    if spec is not None:
        rec.update(spec)
    return rec


def every_component_has_B(labels: np.ndarray, B: np.ndarray) -> bool:
    Bset = set(int(x) for x in B.tolist())
    for c in range(int(labels.max()) + 1 if labels.size else 0):
        members = np.where(labels == c)[0]
        if not any(int(v) in Bset for v in members.tolist()):
            return False
    return True


def main() -> dict:
    t0 = time.perf_counter()
    graphs, prov = load_graphs()
    if len(graphs) != 8:
        raise RuntimeError(f"expected 8 graphs, got {len(graphs)}")
    provenance = {g["name"]: g for g in prov.get("graphs", [])}

    preprocessing = {
        "note": "Identical to the frozen operator-level series-refinement experiment.",
        "ieee": (
            "Official MATPOWER case*.m. In-service branches only. "
            "Weight = DC series susceptance |1/x|; if |x|<1e-12, 1/max(|r|,1e-6). "
            "Parallel branches summed. Then connected_undirected: "
            "W <- (W+W^T)/2, diagonal 0, negatives clipped, isolates (if any) "
            "joined to the next index with weight 1."
        ),
        "traffic": (
            "DCRNN / Zenodo pickle adj if present; else Gaussian distance kernel, "
            "threshold 0.1. Then same connected_undirected. Signals unused."
        ),
        "cora": (
            "Planetoid Cora via torch_geometric. Directed edge_index as 0/1, then "
            "connected_undirected. Features / labels unused."
        ),
        "signals_used": False,
    }

    packed = []
    for g in graphs:
        W = to_csr_adj(g.W)
        n_comp, labels, anchors = component_anchors(W)
        n = int(W.shape[0])
        eligible = np.setdiff1d(np.arange(n, dtype=np.int64), anchors, assume_unique=False)
        ei, ej, ew = undirected_edges(W)
        d = np.asarray(W.sum(axis=1)).ravel()
        packed.append({
            "name": g.name,
            "W": W,
            "L": laplacian(W),
            "n": n,
            "n_comp": n_comp,
            "labels": labels,
            "anchors": anchors,
            "eligible": eligible,
            "counts": elimination_counts(n, int(eligible.size)),
            "Phi": {p: processor_matrix(p, W) for p in PROCESSORS},
            "stats": {
                "name": g.name,
                "n_nodes": n,
                "n_undirected_edges": int(ei.size),
                "n_components": n_comp,
                "n_protected_anchors": int(anchors.size),
                "n_eligible": int(eligible.size),
                "protected_anchors": [int(x) for x in anchors.tolist()],
                "min_degree": float(d.min()),
                "max_degree": float(d.max()),
                "n_zero_degree": int(np.sum(d <= 0.0)),
                "weight_min": float(ew.min()) if ei.size else 0.0,
                "weight_max": float(ew.max()) if ei.size else 0.0,
                "source": provenance.get(g.name, {}),
                "nominal_elimination_counts": {
                    f"{int(f * 100) if f else 0}%": k
                    for f, k in zip(FRACS, elimination_counts(n, int(eligible.size)))
                },
                "achieved_fractions": {
                    f"{int(f * 100) if f else 0}%": (
                        elimination_counts(n, int(eligible.size))[i] / n if n else 0.0
                    )
                    for i, f in enumerate(FRACS)
                },
            },
        })

    records = []
    aff_all = []
    harm_all = []
    clos_all = []
    lii_all = []
    dense_notes = []

    for g in packed:
        name, n = g["name"], g["n"]
        print(f"== {name} n={n} eligible={g['eligible'].size} "
              f"anchors={g['anchors'].size} counts={g['counts']}", flush=True)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(g["eligible"])
            for frac, k in zip(FRACS, g["counts"]):
                I = np.sort(perm[:k].astype(np.int64))
                mask = np.ones(n, dtype=bool)
                mask[I] = False
                B = np.where(mask)[0].astype(np.int64)
                if not every_component_has_B(g["labels"], B):
                    raise RuntimeError(f"{name} seed={seed} frac={frac}: a component lost all of B")
                if np.intersect1d(I, g["anchors"]).size:
                    raise RuntimeError(f"{name} seed={seed}: protected anchor eliminated")

                L_K, E, diag = kron_reduce(g["L"], B, I)
                if diag["lambda_min_LII"] is not None:
                    lii_all.append(diag["lambda_min_LII"])
                harm = harmonic_residuals(g["L"], E, B, I)
                clos = kron_closure(L_K, g["L"], E, B)
                W_K, winfo = W_from_LK(L_K, WK_FP_TOL)
                harm_all.append(harm)
                clos_all.append(clos)

                n_b = int(B.size)
                nnz_WK = int(W_K.nnz)
                dens_WK = nnz_WK / max(n_b * n_b, 1)
                if dens_WK > 0.25 and n_b >= 400:
                    dense_notes.append({
                        "graph": name, "seed": seed, "frac": frac,
                        "n_B": n_b, "W_K_density": dens_WK,
                    })

                rec = {
                    "graph": name,
                    "seed": int(seed),
                    "nominal_frac": float(frac),
                    "n_I": int(I.size),
                    "n_B": n_b,
                    "achieved_frac": float(I.size / n) if n else 0.0,
                    "I": [int(x) for x in I.tolist()],
                    "admissible": True,
                    "L_II": diag,
                    "harmonicity": harm,
                    "kron_closure": clos,
                    "W_K": winfo,
                    "W_K_nnz": nnz_WK,
                    "W_K_density": dens_WK,
                    "processors": {},
                }

                for p in PROCESSORS:
                    Phi_E = g["Phi"][p] @ E
                    T_full = np.asarray(Phi_E[B, :])
                    T_kron = processor_matrix(p, W_K)
                    T_kron_d = T_kron.toarray()
                    drec = defect_record(T_full, T_kron_d, n_b)
                    rec["processors"][p] = drec
                    if p == "affine":
                        aff_all.append(drec["delta_F"])

                records.append(rec)
                print(
                    f"  seed={seed} frac={frac:.2f} |I|={I.size} |B|={n_b} "
                    f"ach={rec['achieved_frac']:.4f} "
                    f"LII_min={diag['lambda_min_LII']} "
                    f"aff={rec['processors']['affine']['delta_F']:.3e} "
                    f"quad={rec['processors']['quadratic']['delta_F']:.3e}",
                    flush=True,
                )

    worst_aff = max(abs(x) for x in aff_all) if aff_all else 0.0
    if worst_aff > AFFINE_ATOL:
        raise RuntimeError(f"affine control failed: worst delta_F={worst_aff}")

    by_gpf = defaultdict(list)
    by_gpf_abs = defaultdict(list)
    by_gpf_max = defaultdict(list)
    by_gpf_d2 = defaultdict(list)
    for rec in records:
        # key by nominal frac; also store achieved
        for p, d in rec["processors"].items():
            key = (rec["graph"], p, rec["nominal_frac"])
            by_gpf[key].append(d["delta_F"])
            by_gpf_abs[key].append(d["abs_F"])
            by_gpf_max[key].append(d["maxabs"])
            if "delta_2" in d:
                by_gpf_d2[key].append(d["delta_2"])

    graph_level = {}
    for g in packed:
        name = g["name"]
        graph_level[name] = {}
        for p in PROCESSORS:
            graph_level[name][p] = {}
            for frac in FRACS:
                key = (name, p, frac)
                graph_level[name][p][f"{frac:.2f}"] = {
                    "n_I": g["counts"][FRACS.index(frac)],
                    "achieved_frac": g["counts"][FRACS.index(frac)] / g["n"],
                    "delta_F": summarize(by_gpf[key]),
                    "abs_F": summarize(by_gpf_abs[key]),
                    "maxabs": summarize(by_gpf_max[key]),
                    **({"delta_2": summarize(by_gpf_d2[key])} if by_gpf_d2[key] else {}),
                }

    cross = {}
    for p in PROCESSORS:
        cross[p] = {}
        for frac in FRACS:
            meds = [graph_level[g["name"]][p][f"{frac:.2f}"]["delta_F"]["median"] for g in packed]
            abs_meds = [graph_level[g["name"]][p][f"{frac:.2f}"]["abs_F"]["median"] for g in packed]
            max_meds = [graph_level[g["name"]][p][f"{frac:.2f}"]["maxabs"]["median"] for g in packed]
            cross[p][f"{frac:.2f}"] = {
                "delta_F": iqr_summary(meds),
                "abs_F": iqr_summary(abs_meds),
                "maxabs": iqr_summary(max_meds),
            }

    mono = {
        "rule": "weakly nondecreasing graph-level median delta_F over 0,1,5,10,25%",
        "n_graphs": 8,
        "by_processor": {},
    }
    for p in NON_AFFINE:
        n_ok = 0
        violations = []
        for g in packed:
            name = g["name"]
            vals = [graph_level[name][p][f"{f:.2f}"]["delta_F"]["median"] for f in FRACS]
            ok = all(vals[i] <= vals[i + 1] + 1e-18 for i in range(len(vals) - 1))
            if ok:
                n_ok += 1
            else:
                drops = []
                for i in range(len(vals) - 1):
                    if vals[i] > vals[i + 1]:
                        drops.append({
                            "from_frac": FRACS[i],
                            "to_frac": FRACS[i + 1],
                            "from": vals[i],
                            "to": vals[i + 1],
                            "drop": vals[i] - vals[i + 1],
                        })
                violations.append({"graph": name, "values": vals, "drops": drops})
        mono["by_processor"][p] = {
            "n_monotone": n_ok,
            "n_graphs": 8,
            "violations": violations,
        }

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment": "kron_elimination_original_nodes",
        "primary_metric": "delta_F^Kron = ||S Phi_G E - Phi_{G_K}||_F / ||Phi_{G_K}||_F",
        "signal_dependent": False,
        "graph_preprocessing": preprocessing,
        "graphs": [g["stats"] for g in packed],
        "sampling": {
            "seeds": SEEDS,
            "nominal_percentages": FRACS,
            "rule": (
                "Protect the lowest-index vertex of every connected component. "
                "Permute remaining eligible vertices with the seed. Nested prefixes. "
                "Counts are round(frac * n_original), at least 1 if eligible, "
                "capped at n_eligible, nested nondecreasing."
            ),
            "W_K_fp_tol": WK_FP_TOL,
        },
        "processor_definitions": processor_defs(),
        "affine_control": {
            "worst_delta_F": worst_aff,
            "median_delta_F": float(np.median(aff_all)),
            "mean_delta_F": float(np.mean(aff_all)),
            "worst_maxabs": max(rec["processors"]["affine"]["maxabs"] for rec in records),
            "n": len(aff_all),
        },
        "harmonicity_worst": {
            "SE_minus_I_maxabs": max(x["SE_minus_I_maxabs"] for x in harm_all),
            "LE_I_maxabs": max(x["LE_I_maxabs"] for x in harm_all),
        },
        "kron_closure_worst": {
            "symmetry_maxabs": max(x["symmetry_maxabs"] for x in clos_all),
            "row_sum_maxabs": max(x["row_sum_maxabs"] for x in clos_all),
            "offdiag_max": max(x["offdiag_max"] for x in clos_all),
            "SLE_minus_LK_maxabs": max(x["SLE_minus_LK_maxabs"] for x in clos_all),
            "SLE_minus_LK_fro": max(x["SLE_minus_LK_fro"] for x in clos_all),
        },
        "L_II_lambda_min": {
            "min": (min(lii_all) if lii_all else None),
            "max": (max(lii_all) if lii_all else None),
            "n": len(lii_all),
        },
        "density_notes": dense_notes,
        "per_record": records,
        "graph_level": graph_level,
        "cross_graph": cross,
        "monotonicity": mono,
        "library_versions": software_versions(),
        "runtime_sec": elapsed,
        "reproduction": "python3 kron_elimination.py",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON} in {elapsed:.1f}s", flush=True)
    return payload


if __name__ == "__main__":
    main()
