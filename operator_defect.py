"""Operator-level boundary-processor defect under Kron-equivalent series refinement.

Primary object (signal-independent):

    D_Phi(G, G') = S_{G'} Phi_{G'} E_{G'} - Phi_G
    delta_F      = ||D_Phi||_F / ||Phi_G||_F

B = V(G) for every original graph, so T_Phi(G) = Phi_G.
Refinements are nested prefixes of a seed-wise permutation of ORIGINAL
undirected edges. Each selected edge of conductance w is replaced by a
unique degree-2 node with conductances 2w, 2w.

Do not edit this script's coefficients after seeing results.
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from graphs import load_graphs  # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "results" / (Path(__file__).stem + ".json")

SEEDS = [0, 1, 2, 3, 4]
FRACS = [0.00, 0.01, 0.05, 0.10, 0.25]
ALPHA_AFF = 0.15
ALPHA_QUAD = 0.08
CHEB_C = (0.5, 0.35, 0.15)
KRON_ATOL = 1e-8
SPECTRAL_N_MAX = 400
HARM_PROBES = 4
HARM_SEED = 20260911

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
    L = sparse.diags(d) - W
    return canonicalize_sparse(L)


def fro_norm(A: sparse.spmatrix) -> float:
    A = A.tocsr()
    A.sum_duplicates()
    if A.nnz == 0:
        return 0.0
    return float(np.linalg.norm(A.data))


def maxabs(A: sparse.spmatrix) -> float:
    A = A.tocsr()
    A.sum_duplicates()
    if A.nnz == 0:
        return 0.0
    return float(np.max(np.abs(A.data)))


def spectral_rel(D: sparse.spmatrix, T: sparse.spmatrix, n: int):
    if n > SPECTRAL_N_MAX:
        return None
    Dd = D.toarray()
    Td = T.toarray()
    n2_t = float(np.linalg.norm(Td, 2))
    n2_d = float(np.linalg.norm(Dd, 2))
    return {
        "abs_2": n2_d,
        "delta_2": (n2_d / n2_t) if n2_t > 0 else float("nan"),
    }


def refinement_counts(m: int) -> list:
    counts = []
    prev = 0
    for f in FRACS:
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


def defect_record(T_r: sparse.csr_matrix, T_o: sparse.csr_matrix, n: int) -> dict:
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
    if spec is not None:
        rec.update(spec)
    return rec


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


def main() -> dict:
    t0 = time.perf_counter()
    graphs, prov = load_graphs()
    if len(graphs) != 8:
        raise RuntimeError(f"expected 8 graphs, got {len(graphs)}")

    preprocessing = {
        "ieee": (
            "Official MATPOWER case*.m. In-service branches only. "
            "Weight = DC series susceptance |1/x|; if |x|<1e-12, 1/max(|r|,1e-6). "
            "Parallel branches summed. Then connected_undirected: "
            "W <- (W+W^T)/2, diagonal 0, negatives clipped, isolates (if any) "
            "joined to the next index with weight 1."
        ),
        "traffic": (
            "DCRNN / Zenodo pickle adj [ids, id2ind, adj] if present; else Gaussian "
            "distance kernel, threshold 0.1. Then same connected_undirected. "
            "Signals are loaded by the shared loader but are unused here."
        ),
        "cora": (
            "Planetoid Cora via torch_geometric. Directed edge_index written as a "
            "0/1 adjacency, then connected_undirected (so bidirectional citations "
            "remain weight 1). Isolates, if any, receive a weight-1 edge to i+1. "
            "Node features / labels / heat fields are unused."
        ),
        "boundary": "B = V(G) for every original graph; E_G = S_G = I.",
        "signals_used": False,
    }

    provenance = {g["name"]: g for g in prov.get("graphs", [])}
    packed = []
    for g in graphs:
        W = to_csr_adj(g.W)
        ei, ej, ew = undirected_edges(W)
        packed.append({
            "name": g.name,
            "W": W,
            "n": int(W.shape[0]),
            "ei": ei,
            "ej": ej,
            "ew": ew,
            "m": int(ei.size),
            "stats": graph_stats(g.name, W, provenance.get(g.name, {})),
        })

    T_orig = {}
    for g in packed:
        T_orig[g["name"]] = {p: processor_matrix(p, g["W"]) for p in PROCESSORS}

    records = []
    kron_all = []
    sle_all = []
    aff_all = []
    harm_all = []
    canon_all = []
    lii_off_all = []

    rng_h = np.random.default_rng(HARM_SEED)

    for g in packed:
        name, n, m = g["name"], g["n"], g["m"]
        counts = refinement_counts(m)
        print(f"== {name} n={n} m={m} counts={counts}", flush=True)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(m)
            for frac, k in zip(FRACS, counts):
                split = perm[:k]
                W_r, inserted = refine_graph(n, g["ei"], g["ej"], g["ew"], split)
                E = build_E(n, inserted)
                L_r = laplacian(W_r)
                L_o = laplacian(g["W"])

                L_ii = L_r[n:, n:] if W_r.shape[0] > n else None
                if L_ii is not None and L_ii.shape[0]:
                    off = L_ii.copy()
                    off.setdiag(0.0)
                    off.eliminate_zeros()
                    lii_off_all.append(maxabs(off) if off.nnz else 0.0)

                L_k = schur_onto_original(L_r, n)
                kron = residual_pair(L_k, L_o)
                SLE = canonicalize_sparse((L_r @ E)[:n, :])
                sle = residual_pair(SLE, L_o)
                harm = harmonicity(L_r, E, n, rng_h)
                kron_all.append(kron)
                sle_all.append(sle)
                harm_all.append(harm)

                if k > 0 and kron["maxabs"] > KRON_ATOL:
                    raise RuntimeError(
                        f"Kron residual failed on {name} seed={seed} frac={frac}: {kron}"
                    )

                W_canon = collapse_to_original(n, W_r, inserted)
                W_kron = W_from_laplacian(L_k)
                rec = {
                    "graph": name,
                    "seed": int(seed),
                    "frac": float(frac),
                    "n_refined_edges": int(k),
                    "n_full": int(W_r.shape[0]),
                    "kron": kron,
                    "sle": sle,
                    "harmonicity": harm,
                    "n_zero_degree_refined": int(
                        np.sum(np.asarray(W_r.sum(axis=1)).ravel() <= 0.0)
                    ),
                    "processors": {},
                    "canon_W_vs_orig": residual_pair(W_canon, g["W"]),
                    "kron_W_vs_orig": residual_pair(W_kron, g["W"]),
                    "canon_processors": {},
                }

                for p in PROCESSORS:
                    Phi_r = processor_matrix(p, W_r)
                    T_r = canonicalize_sparse((Phi_r @ E)[:n, :])
                    drec = defect_record(T_r, T_orig[name][p], n)
                    rec["processors"][p] = drec
                    if p == "affine":
                        aff_all.append(drec["delta_F"])

                    Phi_c = processor_matrix(p, W_canon)
                    crec = defect_record(Phi_c, T_orig[name][p], n)
                    rec["canon_processors"][p] = crec
                    canon_all.append({"graph": name, "seed": seed, "frac": frac,
                                      "processor": p, **crec})

                records.append(rec)
                print(
                    f"  seed={seed} frac={frac:.2f} k={k} "
                    f"kron_max={kron['maxabs']:.3e} "
                    f"aff={rec['processors']['affine']['delta_F']:.3e} "
                    f"quad={rec['processors']['quadratic']['delta_F']:.3e}",
                    flush=True,
                )

    worst_kron_f = max(x["frobenius"] for x in kron_all)
    worst_kron_a = max(x["maxabs"] for x in kron_all)
    worst_sle_a = max(x["maxabs"] for x in sle_all)
    worst_aff = max(abs(x) for x in aff_all)
    worst_harm = max(x["maxabs_interior"] for x in harm_all)
    if worst_kron_a > KRON_ATOL:
        raise RuntimeError(f"worst Kron maxabs {worst_kron_a} exceeds {KRON_ATOL}")
    if worst_aff > 1e-8:
        raise RuntimeError(f"affine control failed: worst delta_F={worst_aff}")

    # graph x processor x frac aggregates across seeds
    by_gpf = defaultdict(list)
    by_gpf_abs = defaultdict(list)
    by_gpf_max = defaultdict(list)
    by_gpf_d2 = defaultdict(list)
    for rec in records:
        for p, d in rec["processors"].items():
            key = (rec["graph"], p, rec["frac"])
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
                    "delta_F": summarize(by_gpf[key]),
                    "abs_F": summarize(by_gpf_abs[key]),
                    "maxabs": summarize(by_gpf_max[key]),
                    **({"delta_2": summarize(by_gpf_d2[key])} if by_gpf_d2[key] else {}),
                }

    cross = {}
    for p in PROCESSORS:
        cross[p] = {}
        for frac in FRACS:
            meds = [
                graph_level[g["name"]][p][f"{frac:.2f}"]["delta_F"]["median"]
                for g in packed
            ]
            abs_meds = [
                graph_level[g["name"]][p][f"{frac:.2f}"]["abs_F"]["median"]
                for g in packed
            ]
            max_meds = [
                graph_level[g["name"]][p][f"{frac:.2f}"]["maxabs"]["median"]
                for g in packed
            ]
            cross[p][f"{frac:.2f}"] = {
                "delta_F": iqr_summary(meds),
                "abs_F": iqr_summary(abs_meds),
                "maxabs": iqr_summary(max_meds),
            }

    mono = {"rule": "weakly nondecreasing median delta_F over 0,1,5,10,25%",
            "n_graphs": 8, "by_processor": {}}
    for p in NON_AFFINE:
        n_ok = 0
        violations = []
        for g in packed:
            name = g["name"]
            vals = [graph_level[name][p][f"{f:.2f}"]["delta_F"]["median"] for f in FRACS]
            ok = all(vals[i] <= vals[i + 1] + 0.0 for i in range(len(vals) - 1))
            # allow exact fp jitter at machine scale on a vanishing baseline
            ok_loose = all(vals[i] <= vals[i + 1] + 1e-18 for i in range(len(vals) - 1))
            if ok or ok_loose:
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

    worst_canon_delta = 0.0
    worst_canon_item = None
    for item in canon_all:
        if item["delta_F"] > worst_canon_delta:
            worst_canon_delta = item["delta_F"]
            worst_canon_item = {
                "graph": item["graph"],
                "seed": item["seed"],
                "frac": item["frac"],
                "processor": item["processor"],
                "delta_F": item["delta_F"],
                "abs_F": item["abs_F"],
                "maxabs": item["maxabs"],
            }

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment": "operator_defect_kron_series",
        "primary_metric": "delta_F = ||S Phi(G') E - Phi(G)||_F / ||Phi(G)||_F",
        "signal_dependent": False,
        "graph_preprocessing": preprocessing,
        "graphs": [g["stats"] for g in packed],
        "refinement": {
            "seeds": SEEDS,
            "percentages": FRACS,
            "rule": (
                "One RNG permutation of original undirected edges per seed. "
                "Nested prefixes. 0% = 0 edges. Nonzero % rounds to nearest "
                "integer and is at least 1 when m>=1. Inserted edges are never re-refined. "
                "Selected {i,j} of weight w is replaced by i--z--j with weights 2w, 2w."
            ),
            "counts": {g["name"]: g["stats"]["refinement_counts"] for g in packed},
        },
        "E_construction": {
            "original_rows": "identity",
            "inserted_row": "1/2 at the two original endpoints",
            "S": "select original-node rows",
        },
        "processor_definitions": processor_defs(),
        "kron_verification": {
            "ridge": None,
            "method": "L_K = L_BB - L_BI diag(L_II)^{-1} L_IB",
            "worst_frobenius": worst_kron_f,
            "worst_maxabs": worst_kron_a,
            "worst_SLE_maxabs": worst_sle_a,
            "n_checks": len(kron_all),
            "n_exactly_zero": int(sum(1 for x in kron_all if x["exactly_zero"])),
            "all_exactly_zero": bool(all(x["exactly_zero"] for x in kron_all)),
            "L_II_offdiag_worst_maxabs": max(lii_off_all) if lii_off_all else 0.0,
            "gate": KRON_ATOL,
            "passed": bool(worst_kron_a <= KRON_ATOL),
        },
        "harmonicity": {
            "n_probes_per_refined_graph": HARM_PROBES,
            "worst_maxabs_interior": worst_harm,
            "worst_rms": max(x["rms_interior"] for x in harm_all),
        },
        "affine_control": {
            "worst_delta_F": worst_aff,
            "median_delta_F": float(np.median(aff_all)),
            "mean_delta_F": float(np.mean(aff_all)),
            "n": len(aff_all),
            "expected": "numerical zero / machine precision",
        },
        "per_record": records,
        "graph_level": graph_level,
        "cross_graph": cross,
        "monotonicity": mono,
        "canonicalization": {
            "method": "collapse each inserted degree-2 node; restore original edge weight w",
            "worst_delta_F": worst_canon_delta,
            "worst": worst_canon_item,
            "n_checks": len(canon_all),
        },
        "library_versions": software_versions(),
        "runtime_sec": elapsed,
        "reproduction": "python3 operator_defect.py",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON} in {elapsed:.1f}s", flush=True)
    return payload


if __name__ == "__main__":
    main()
