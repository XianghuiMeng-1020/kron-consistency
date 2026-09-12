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
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kron_consistency.config import (  # noqa: E402
    AFFINE_ATOL,
    FRACS,
    NON_AFFINE,
    PROCESSORS,
    SEEDS,
    WK_FP_TOL,
)
from kron_consistency.graphs import load_graphs  # noqa: E402
from kron_consistency.kron import (  # noqa: E402
    W_from_LK,
    component_anchors,
    elimination_counts,
    every_component_has_B,
    harmonic_residuals,
    kron_closure,
    kron_reduce,
)
from kron_consistency.operators import (  # noqa: E402
    defect_record,
    iqr_summary,
    laplacian,
    processor_defs,
    processor_matrix,
    software_versions,
    summarize,
    to_csr_adj,
    undirected_edges,
)
from kron_consistency.paths import RESULTS_DIR  # noqa: E402

OUT_JSON = RESULTS_DIR / "kron_elimination.json"


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
        "reproduction": "python experiments/kron_elimination/run.py",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON} in {elapsed:.1f}s", flush=True)
    return payload


if __name__ == "__main__":
    main()
