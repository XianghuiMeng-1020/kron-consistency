"""Frozen trained GCN on Cora under exactly Kron-equivalent series refinements.

Train a two-layer Kipf–Welling GCN once on the original Planetoid Cora graph.
Freeze weights. At test time, replace the graph by nested series refinements
that preserve the combinatorial Laplacian on the original nodes, harmonically
extend features to inserted degree-2 vertices, and evaluate the same model on
the original Planetoid test nodes.

This is a test-time representation-invariance experiment. No retraining on G'.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.stats import wilcoxon
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kron_consistency.paths import PLANETOID_DIR, RESULTS_DIR  # noqa: E402

DATA_ROOT = PLANETOID_DIR
OUT_JSON = RESULTS_DIR / "gcn_illustration.json"

SEEDS = list(range(10))
FRACS = (0.00, 0.01, 0.05, 0.10, 0.25)
HIDDEN = 16
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 200
BOOT_SEED = 20260911
N_BOOT = 10_000
KRON_ATOL = 1e-8


def software_versions() -> dict:
    import torch_geometric

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": "cpu",
    }


def load_cora():
    data = Planetoid(str(DATA_ROOT), name="Cora")[0]
    x = data.x.cpu().numpy().astype(np.float64)
    y = data.y.cpu().numpy().astype(np.int64)
    train = data.train_mask.cpu().numpy().astype(bool)
    val = data.val_mask.cpu().numpy().astype(bool)
    test = data.test_mask.cpu().numpy().astype(bool)
    ei = data.edge_index.cpu().numpy()
    pairs = set()
    n_self = 0
    n_raw_directed = int(ei.shape[1])
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        u, v = int(u), int(v)
        if u == v:
            n_self += 1
            continue
        if u > v:
            u, v = v, u
        pairs.add((u, v))
    edges = np.array(sorted(pairs), dtype=np.int64)
    n = int(x.shape[0])
    W = sparse.coo_matrix(
        (np.ones(len(edges), dtype=np.float64), (edges[:, 0], edges[:, 1])),
        shape=(n, n),
    )
    W = (W + W.T).tocsr()
    n_comp = sparse.csgraph.connected_components(W, directed=False, return_labels=False)
    return {
        "x": x,
        "y": y,
        "train": train,
        "val": val,
        "test": test,
        "edges": edges,
        "n": n,
        "n_undirected": int(edges.shape[0]),
        "n_raw_directed": n_raw_directed,
        "n_self_loops_raw": n_self,
        "n_features": int(x.shape[1]),
        "n_classes": int(y.max() + 1),
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "n_components": int(n_comp),
        "W": W,
    }


def undirected_to_mp(edges: np.ndarray, weights: np.ndarray):
    """One undirected edge of weight w -> two directed MP entries of weight w."""
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    w = np.concatenate([weights, weights])
    edge_index = torch.tensor(np.stack([src, dst], 0), dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    return edge_index, edge_weight


def combinatorial_laplacian(n: int, edges: np.ndarray, weights: np.ndarray) -> sparse.csr_matrix:
    W = sparse.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))
    W = (W + W.T).tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    return sparse.diags(deg) - W


def schur_onto_original(
    n_orig: int,
    base_edges: np.ndarray,
    split_idx: np.ndarray,
) -> np.ndarray:
    """L_K of the refined graph onto original nodes. L_II is exactly 4 I."""
    split = set(int(i) for i in split_idx)
    keep = [k for k in range(len(base_edges)) if k not in split]
    if keep:
        keep_e = base_edges[keep]
        keep_w = np.ones(len(keep), dtype=np.float64)
        L_BB = combinatorial_laplacian(n_orig, keep_e, keep_w).tolil()
    else:
        L_BB = sparse.lil_matrix((n_orig, n_orig), dtype=np.float64)
    if split_idx.size == 0:
        return L_BB.tocsr().toarray()
    se = base_edges[split_idx]
    # Each inserted z: edges (i,z) and (z,j) of weight 2.
    # L_BB gains +2 on the two endpoint diagonals (degree contribution of i-z, j-z).
    for i, j in se:
        i, j = int(i), int(j)
        L_BB[i, i] += 2.0
        L_BB[j, j] += 2.0
    L_BB = L_BB.tocsr()
    n_I = int(se.shape[0])
    rows = np.concatenate([se[:, 0], se[:, 1]])
    cols = np.concatenate([np.arange(n_I), np.arange(n_I)])
    vals = np.full(2 * n_I, -2.0, dtype=np.float64)
    L_BI = sparse.coo_matrix((vals, (rows, cols)), shape=(n_orig, n_I)).tocsr()
    # L_II = 4 I, so L_BI L_II^{-1} L_IB = (1/4) L_BI L_BI^T
    M = (L_BI @ L_BI.T) * 0.25
    return (L_BB - M).toarray()


def refine_graph(cora: dict, split_idx: np.ndarray):
    n = cora["n"]
    edges = cora["edges"]
    x = cora["x"]
    split = np.asarray(split_idx, dtype=np.int64)
    keep_mask = np.ones(len(edges), dtype=bool)
    keep_mask[split] = False
    kept = edges[keep_mask]
    n_new = int(split.size)
    n_full = n + n_new
    if n_new == 0:
        mp_e = kept
        mp_w = np.ones(len(mp_e), dtype=np.float64)
        return {
            "n_full": n,
            "x_full": x.copy(),
            "mp_edges": mp_e,
            "mp_weights": mp_w,
            "n_refined_edges": 0,
            "n_new_nodes": 0,
            "inserted": [],
        }
    se = edges[split]
    z = n + np.arange(n_new)
    new_e = np.vstack(
        [
            np.stack([se[:, 0], z], 1),
            np.stack([se[:, 1], z], 1),
        ]
    )
    new_w = np.full(2 * n_new, 2.0, dtype=np.float64)
    mp_e = np.vstack([kept, new_e]) if len(kept) else new_e
    mp_w = np.concatenate([np.ones(len(kept), dtype=np.float64), new_w])
    x_full = np.zeros((n_full, x.shape[1]), dtype=np.float64)
    x_full[:n] = x
    x_full[z] = 0.5 * (x[se[:, 0]] + x[se[:, 1]])
    inserted = [
        {"z": int(zi), "i": int(i), "j": int(j)}
        for zi, (i, j) in zip(z.tolist(), se.tolist())
    ]
    return {
        "n_full": n_full,
        "x_full": x_full,
        "mp_edges": mp_e,
        "mp_weights": mp_w,
        "n_refined_edges": n_new,
        "n_new_nodes": n_new,
        "inserted": inserted,
    }


def kron_errors(cora: dict, split_idx: np.ndarray) -> dict:
    n = cora["n"]
    L_orig = combinatorial_laplacian(n, cora["edges"], np.ones(len(cora["edges"]))).toarray()
    L_k = schur_onto_original(n, cora["edges"], split_idx)
    diff = L_k - L_orig
    return {
        "frobenius": float(np.linalg.norm(diff, ord="fro")),
        "maxabs": float(np.max(np.abs(diff))),
    }


class GCN(torch.nn.Module):
    """Two-layer Kipf–Welling GCN. Normalization is GCNConv's standard
    D̃^{-1/2}(A+I)D̃^{-1/2} with self-loop weight 1. cached=False."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(
            in_dim, hidden, improved=False, cached=False,
            add_self_loops=True, normalize=True,
        )
        self.conv2 = GCNConv(
            hidden, out_dim, improved=False, cached=False,
            add_self_loops=True, normalize=True,
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def parameter_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_gcn(cora: dict, seed: int) -> dict:
    set_seed(seed)
    device = torch.device("cpu")
    model = GCN(cora["n_features"], HIDDEN, cora["n_classes"], DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    edge_index, edge_weight = undirected_to_mp(
        cora["edges"], np.ones(len(cora["edges"]), dtype=np.float64)
    )
    x = torch.tensor(cora["x"], dtype=torch.float32)
    y = torch.tensor(cora["y"], dtype=torch.long)
    train, val = cora["train"], cora["val"]
    best = {
        "epoch": -1,
        "val_acc": -1.0,
        "val_nll": float("inf"),
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    history = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index, edge_weight)
        loss = F.cross_entropy(logits[train], y[train])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index, edge_weight)
            val_nll = float(F.nll_loss(F.log_softmax(val_logits[val], dim=1), y[val]))
            val_acc = float((val_logits[val].argmax(1) == y[val]).float().mean())
        history.append({"epoch": epoch, "val_acc": val_acc, "val_nll": val_nll})
        better = (val_acc > best["val_acc"] + 1e-12) or (
            abs(val_acc - best["val_acc"]) <= 1e-12 and val_nll < best["val_nll"]
        )
        if better:
            best = {
                "epoch": epoch,
                "val_acc": val_acc,
                "val_nll": val_nll,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
    model.load_state_dict(best["state"])
    model.eval()
    return {
        "model": model,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "best_epoch": int(best["epoch"]),
        "best_val_acc": float(best["val_acc"]),
        "best_val_nll": float(best["val_nll"]),
        "param_hash": parameter_hash(model),
        "n_history": len(history),
    }


@torch.no_grad()
def evaluate(model, x_np, edges, weights, y, mask, n_orig: int):
    model.eval()
    edge_index, edge_weight = undirected_to_mp(edges, weights)
    x = torch.tensor(x_np, dtype=torch.float32)
    logits = model(x, edge_index, edge_weight)
    logits_b = logits[:n_orig]
    logp = F.log_softmax(logits_b, dim=1)
    prob = logp.exp()
    pred = logits_b.argmax(1).numpy()
    y_t = torch.tensor(y, dtype=torch.long)
    acc = float((pred[mask] == y[mask]).mean())
    nll = float(F.nll_loss(logp[mask], y_t[mask]))
    return {
        "acc": acc,
        "nll": nll,
        "pred": pred,
        "prob": prob.numpy(),
        "logits": logits_b.numpy(),
    }


def kl_mean(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def bootstrap_mean_ci(vals: np.ndarray, seed: int, n_boot: int) -> list:
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    boots = np.empty(n_boot, dtype=float)
    m = len(vals)
    for b in range(n_boot):
        boots[b] = vals[rng.integers(0, m, size=m)].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return [float(lo), float(hi)]


def canonicalize_from_schur(cora: dict, split_idx: np.ndarray):
    L_k = schur_onto_original(cora["n"], cora["edges"], split_idx)
    L_orig = combinatorial_laplacian(
        cora["n"], cora["edges"], np.ones(len(cora["edges"]))
    ).toarray()
    W_k = -L_k.copy()
    np.fill_diagonal(W_k, 0.0)
    W_o = -L_orig.copy()
    np.fill_diagonal(W_o, 0.0)
    # Recover unique undirected edges with weight 1 from W_k (should be 0/1).
    iu, ju = np.triu_indices(cora["n"], k=1)
    keep = np.abs(W_k[iu, ju]) > 0.5
    rec_e = np.stack([iu[keep], ju[keep]], 1).astype(np.int64)
    rec_w = np.ones(len(rec_e), dtype=np.float64)
    return {
        "maxabs_L": float(np.max(np.abs(L_k - L_orig))),
        "maxabs_W": float(np.max(np.abs(W_k - W_o))),
        "edges": rec_e,
        "weights": rec_w,
    }


def run():
    t0 = time.perf_counter()
    cora = load_cora()
    m = cora["n_undirected"]
    counts = {f: int(round(f * m)) for f in FRACS}
    # Nested prefixes require monotone counts.
    prev = 0
    for f in FRACS:
        counts[f] = max(counts[f], prev)
        prev = counts[f]
    dataset_stats = {
        "n_original_nodes": cora["n"],
        "n_undirected_original_edges": cora["n_undirected"],
        "n_raw_directed_edges": cora["n_raw_directed"],
        "n_self_loops_in_raw_edge_index": cora["n_self_loops_raw"],
        "feature_dim": cora["n_features"],
        "n_classes": cora["n_classes"],
        "n_train": cora["n_train"],
        "n_val": cora["n_val"],
        "n_test": cora["n_test"],
        "n_connected_components": cora["n_components"],
        "raw_edge_weight": "Planetoid Cora has no edge_attr; combinatorial weight is 1 per undirected pair",
        "preprocessing": (
            "unique undirected pairs from Planetoid edge_index, drop self-loops, "
            "weight 1, then bidirectional message-passing copies of that weight"
        ),
        "refinement_edge_counts": {str(int(100 * f)): counts[f] for f in FRACS},
    }
    print("dataset", json.dumps(dataset_stats, indent=2), flush=True)

    L_orig = combinatorial_laplacian(
        cora["n"], cora["edges"], np.ones(len(cora["edges"]))
    ).toarray()
    print("orig L row-sum maxabs", float(np.max(np.abs(L_orig.sum(1)))), flush=True)

    per_seed = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        trained = train_gcn(cora, seed)
        print(
            f"  trained epoch={trained['best_epoch']} "
            f"val_acc={trained['best_val_acc']:.4f} hash={trained['param_hash'][:12]}",
            flush=True,
        )
        base = evaluate(
            trained["model"],
            cora["x"],
            cora["edges"],
            np.ones(len(cora["edges"])),
            cora["y"],
            cora["test"],
            cora["n"],
        )
        rng = np.random.default_rng(seed)
        perm = rng.permutation(m)
        rec = {
            "seed": seed,
            "best_epoch": trained["best_epoch"],
            "best_val_acc": trained["best_val_acc"],
            "best_val_nll": trained["best_val_nll"],
            "param_hash": trained["param_hash"],
            "baseline": {
                "test_acc": base["acc"],
                "test_nll": base["nll"],
            },
            "refinements": {},
        }
        for f in FRACS:
            k = counts[f]
            split_idx = perm[:k]
            refined = refine_graph(cora, split_idx)
            kron = kron_errors(cora, split_idx)
            if kron["maxabs"] > KRON_ATOL:
                raise RuntimeError(
                    f"Kron residual too large at seed={seed} frac={f}: {kron}"
                )
            # Feature-extension sample check
            feat_err = 0.0
            for item in refined["inserted"][:8]:
                z, i, j = item["z"], item["i"], item["j"]
                feat_err = max(
                    feat_err,
                    float(
                        np.max(
                            np.abs(
                                refined["x_full"][z]
                                - 0.5 * (cora["x"][i] + cora["x"][j])
                            )
                        )
                    ),
                )
            # Weight check: inserted edges must be 2, kept edges 1
            w_unique = sorted(set(np.round(refined["mp_weights"], 8).tolist()))
            ev = evaluate(
                trained["model"],
                refined["x_full"],
                refined["mp_edges"],
                refined["mp_weights"],
                cora["y"],
                cora["test"],
                cora["n"],
            )
            h2 = parameter_hash(trained["model"])
            if h2 != trained["param_hash"]:
                raise RuntimeError("model parameters changed during evaluation")
            flip = float((ev["pred"][cora["test"]] != base["pred"][cora["test"]]).mean())
            kl = kl_mean(base["prob"][cora["test"]], ev["prob"][cora["test"]])
            logit_l2 = float(
                np.linalg.norm(ev["logits"][cora["test"]] - base["logits"][cora["test"]])
                / (np.linalg.norm(base["logits"][cora["test"]]) + 1e-12)
            )
            canon = canonicalize_from_schur(cora, split_idx)
            cev = evaluate(
                trained["model"],
                cora["x"],
                canon["edges"],
                canon["weights"],
                cora["y"],
                cora["test"],
                cora["n"],
            )
            rec["refinements"][str(int(100 * f))] = {
                "n_refined_edges": refined["n_refined_edges"],
                "n_new_nodes": refined["n_new_nodes"],
                "n_full": refined["n_full"],
                "edge_weights_present": w_unique,
                "feature_extension_maxabs_sample": feat_err,
                "kron_frobenius": kron["frobenius"],
                "kron_maxabs": kron["maxabs"],
                "test_acc": ev["acc"],
                "test_nll": ev["nll"],
                "delta_acc": ev["acc"] - base["acc"],
                "delta_nll": ev["nll"] - base["nll"],
                "flip_rate": flip,
                "mean_kl": kl,
                "logit_rel_l2_test": logit_l2,
                "param_hash_match": h2 == trained["param_hash"],
                "canon": {
                    "maxabs_L": canon["maxabs_L"],
                    "maxabs_W": canon["maxabs_W"],
                    "maxabs_logit": float(
                        np.max(np.abs(cev["logits"] - base["logits"]))
                    ),
                    "pred_all_match": bool(np.array_equal(cev["pred"], base["pred"])),
                    "delta_acc": cev["acc"] - base["acc"],
                    "test_acc": cev["acc"],
                },
            }
            print(
                f"  frac={int(100*f):2d}% k={k:4d} acc={ev['acc']:.4f} "
                f"dAcc={ev['acc']-base['acc']:+.4f} flip={flip:.4f} "
                f"kronF={kron['frobenius']:.3e} kronM={kron['maxabs']:.3e}",
                flush=True,
            )
        per_seed.append(rec)

    # Aggregate across seeds (skip the 0% row for deltas vs baseline;
    # 0% is identity and is reported as a control).
    agg = {}
    for f in FRACS:
        key = str(int(100 * f))
        base_acc = np.array([r["baseline"]["test_acc"] for r in per_seed])
        ref_acc = np.array([r["refinements"][key]["test_acc"] for r in per_seed])
        dacc = np.array([r["refinements"][key]["delta_acc"] for r in per_seed])
        dnll = np.array([r["refinements"][key]["delta_nll"] for r in per_seed])
        flip = np.array([r["refinements"][key]["flip_rate"] for r in per_seed])
        kl = np.array([r["refinements"][key]["mean_kl"] for r in per_seed])
        kron_f = np.array([r["refinements"][key]["kron_frobenius"] for r in per_seed])
        kron_m = np.array([r["refinements"][key]["kron_maxabs"] for r in per_seed])
        stat = None
        if np.max(np.abs(dacc)) > 0:
            try:
                stat = wilcoxon(dacc, alternative="two-sided", zero_method="wilcox")
                stat = {"statistic": float(stat.statistic), "pvalue": float(stat.pvalue)}
            except ValueError:
                stat = {"statistic": None, "pvalue": None, "note": "wilcoxon undefined"}
        else:
            stat = {"statistic": None, "pvalue": None, "note": "all deltas zero"}
        n_pos = int(np.sum(dacc > 0))
        n_neg = int(np.sum(dacc < 0))
        n_zero = int(np.sum(dacc == 0))
        agg[key] = {
            "n_refined_edges": counts[f],
            "n_new_nodes": counts[f],
            "mean_baseline_acc": float(base_acc.mean()),
            "mean_refined_acc": float(ref_acc.mean()),
            "mean_delta_acc_pp": float(100.0 * dacc.mean()),
            "std_delta_acc_pp": float(100.0 * dacc.std(ddof=1)),
            "median_delta_acc_pp": float(100.0 * np.median(dacc)),
            "ci95_mean_delta_acc_pp": [
                100.0 * v for v in bootstrap_mean_ci(dacc, BOOT_SEED, N_BOOT)
            ],
            "mean_delta_nll": float(dnll.mean()),
            "mean_flip_rate": float(flip.mean()),
            "mean_kl": float(kl.mean()),
            "worst_kron_frobenius": float(kron_f.max()),
            "worst_kron_maxabs": float(kron_m.max()),
            "n_seeds_delta_pos": n_pos,
            "n_seeds_delta_neg": n_neg,
            "n_seeds_delta_zero": n_zero,
            "wilcoxon": stat,
            "per_seed_delta_acc": [float(v) for v in dacc],
            "per_seed_acc_refined": [float(v) for v in ref_acc],
        }

    payload = {
        "experiment": "frozen_gcn_cora_series_refinement",
        "software": software_versions(),
        "hyperparameters": {
            "architecture": "two-layer GCNConv",
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "optimizer": "Adam",
            "checkpoint": "max val_acc on original graph, tie-break min val_nll, earliest if still tied",
            "normalization": (
                "PyG GCNConv: add self-loops of weight 1, then "
                "Dtilde^{-1/2} (A+I) Dtilde^{-1/2}; improved=False; cached=False"
            ),
            "self_loops": "processor-only, not in the combinatorial Laplacian used for Kron checks",
            "seeds": SEEDS,
            "fractions": list(FRACS),
            "bootstrap_seed": BOOT_SEED,
            "n_bootstrap": N_BOOT,
        },
        "dataset": dataset_stats,
        "implementation_checks": {
            "original_undirected_weight": 1,
            "refined_series_weights": [2, 2],
            "message_passing": "each undirected edge becomes two opposite directed entries with the same weight",
            "gcn_self_loops": "added inside GCNConv with fill weight 1; not part of Kron graph",
            "harmonic_extension": "X_z = (X_i + X_j) / 2 on the full 1433-d feature",
            "masks": "Planetoid train/val/test; metrics on original test nodes only",
            "checkpoint_uses_test": False,
            "retrain_on_refined": False,
        },
        "per_seed": per_seed,
        "aggregate": agg,
        "runtime_sec": float(time.perf_counter() - t0),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print("\nwrote", OUT_JSON, "runtime_sec", payload["runtime_sec"], flush=True)
    print("\n=== AGGREGATE ===", flush=True)
    for key, a in agg.items():
        print(
            f"  {key:>3s}%  base={a['mean_baseline_acc']:.4f} "
            f"ref={a['mean_refined_acc']:.4f} "
            f"dAcc={a['mean_delta_acc_pp']:+.3f}pp "
            f"CI={a['ci95_mean_delta_acc_pp']} "
            f"flip={a['mean_flip_rate']:.4f} "
            f"dNLL={a['mean_delta_nll']:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    run()
