"""Operator-level boundary-processor defect under Kron-equivalent series refinement.

Primary object (signal-independent):

    D_Phi(G, G') = S_{G'} Phi_{G'} E_{G'} - Phi_G
    delta_F      = ||D_Phi||_F / ||Phi_G||_F

B = V(G) for every original graph, so T_Phi(G) = Phi_G.
Refinements are nested prefixes of a seed-wise permutation of ORIGINAL
undirected edges. Each selected edge of conductance w is replaced by a
unique degree-2 node with conductances 2w, 2w.

Do not edit the frozen coefficients in kron_consistency.config after seeing results.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kron_consistency.config import (  # noqa: E402
    FRACS,
    HARM_SEED,
    KRON_ATOL,
    NON_AFFINE,
    PROCESSORS,
    SEEDS,
)
from kron_consistency.graphs import load_graphs  # noqa: E402
from kron_consistency.operators import (  # noqa: E402
    canonicalize_sparse,
    defect_record,
    iqr_summary,
    laplacian,
    maxabs,
    processor_defs,
    processor_matrix,
    residual_pair,
    software_versions,
    summarize,
    to_csr_adj,
    undirected_edges,
)
from kron_consistency.paths import RESULTS_DIR  # noqa: E402
from kron_consistency.series import (  # noqa: E402
    W_from_laplacian,
    build_E,
    collapse_to_original,
    graph_stats,
    harmonicity,
    refine_graph,
    refinement_counts,
    schur_onto_original,
)

OUT_JSON = RESULTS_DIR / "operator_defect.json"


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
        "reproduction": "python experiments/operator_defect/run.py",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON} in {elapsed:.1f}s", flush=True)
    return payload


if __name__ == "__main__":
    main()
