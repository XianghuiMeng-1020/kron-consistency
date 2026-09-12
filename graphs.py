"""Load the eight public graphs used in the operator-level checks.

Signals are unused: every original vertex is treated as a boundary vertex,
and the reported statistic compares processed maps, not chosen inputs.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
IEEE_DIR = ROOT / "data" / "ieee"
TRAFFIC_DIR = ROOT / "data" / "traffic"
PLANETOID = ROOT / "data" / "planetoid"
EPS = 1e-12


@dataclass
class Graph:
    W: np.ndarray
    name: str

    @property
    def n(self) -> int:
        return int(self.W.shape[0])


def _symmetrize(W: np.ndarray) -> np.ndarray:
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    W[W < 0] = 0.0
    return W


def connected_undirected(W: np.ndarray) -> np.ndarray:
    W = _symmetrize(W)
    n = W.shape[0]
    isolates = np.where(W.sum(axis=1) <= EPS)[0]
    for i in isolates:
        j = (i + 1) % n
        W[i, j] = W[j, i] = 1.0
    return W


def _parse_matlab_matrix(text: str, key: str) -> np.ndarray:
    token = f"mpc.{key}"
    start = text.find(token)
    if start < 0:
        raise KeyError(key)
    lb = text.find("[", start)
    rb = text.find("];", lb)
    rows = []
    for line in text[lb + 1 : rb].splitlines():
        line = line.split("%", 1)[0].strip().replace(";", " ")
        parts = [p for p in line.replace("\t", " ").split(" ") if p]
        if parts:
            rows.append([float(p) for p in parts])
    if not rows:
        return np.zeros((0, 0))
    width = max(len(r) for r in rows)
    arr = np.zeros((len(rows), width))
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return arr


def load_matpower_case(path: Path) -> Graph:
    text = path.read_text()
    bus = _parse_matlab_matrix(text, "bus")
    branch = _parse_matlab_matrix(text, "branch")
    bus_ids = bus[:, 0].astype(int)
    id2i = {int(b): i for i, b in enumerate(bus_ids)}
    n = len(bus_ids)
    W = np.zeros((n, n))
    for row in branch:
        if row.shape[0] < 4:
            continue
        status = row[10] if row.shape[0] > 10 else 1.0
        if status <= 0:
            continue
        i, j = id2i[int(row[0])], id2i[int(row[1])]
        x = float(row[3])
        if abs(x) < 1e-12:
            w = 1.0 / max(abs(float(row[2])), 1e-6)
        else:
            w = abs(1.0 / x)
        W[i, j] += w
        W[j, i] += w
    return Graph(W=connected_undirected(W), name=path.stem.replace("case", "IEEE-"))


def _try_pickle_adj(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        adj = np.asarray(obj[2], float)
    elif isinstance(obj, np.ndarray):
        adj = np.asarray(obj, float)
    else:
        return None
    np.fill_diagonal(adj, 0.0)
    adj[adj < 0] = 0.0
    return connected_undirected(0.5 * (adj + adj.T))


def load_cora() -> Graph:
    from torch_geometric.datasets import Planetoid

    data = Planetoid(root=str(PLANETOID), name="Cora")[0]
    n = int(data.num_nodes)
    ei = data.edge_index.cpu().numpy()
    W = np.zeros((n, n))
    W[ei[0], ei[1]] = 1.0
    return Graph(W=connected_undirected(W), name="Cora")


def load_graphs() -> Tuple[List[Graph], dict]:
    graphs = [load_matpower_case(IEEE_DIR / f"{c}.m") for c in ("case14", "case30", "case57", "case118", "case300")]
    for fname, label in (
        ("adj_mx_METR-LA.pkl", "METR-LA"),
        ("adj_mx_PEMS-BAY.pkl", "PEMS-BAY"),
    ):
        W = _try_pickle_adj(TRAFFIC_DIR / fname)
        if W is None:
            raise FileNotFoundError(TRAFFIC_DIR / fname)
        graphs.append(Graph(W=W, name=label))
    graphs.append(load_cora())
    prov = {
        "graphs": [
            {
                "name": g.name,
                "n": g.n,
                "n_edges": int((g.W > 0).sum() // 2),
            }
            for g in graphs
        ],
        "warnings": [],
    }
    return graphs, prov
