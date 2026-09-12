# Kron Consistency of Graph Filters under Exact Boundary Equivalence

Xianghui Meng and Jionghao Lin

Finite weighted graphs may induce the same Kron (Dirichlet-to-Neumann) operator on a declared node set \(B\). That exact representation equivalence does **not** determine the effective boundary map of a processor applied on the unreduced graphs.

The object of study is
\[
T_\Phi(G,B)=S\Phi_G E,
\]
the restriction, after harmonic lifting, of a processor \(\Phi_G\) acting on the combinatorial structure of \(G\).

**Theorem.** For the same real polynomial \(p\) applied directly to the unreduced combinatorial Laplacian, \(Sp(L)E=p(L_K)\) holds for every admissible pair \((G,B)\) if and only if \(p\) is affine. The same characterization extends to entire functions of \(L\).

A map \(q(L_K)\) applied *after* reduction is consistent by construction. The theorem concerns the direct-processing convention \(T_\Phi=S\Phi E\).

---

## Contents

| File | Role |
| :--- | :--- |
| `theory.py` | Kron / series algebra used in the identities |
| `witnesses.py` | Analytic witnesses (two-terminal split; triangle series) |
| `operator_defect.py` | Signal-independent map defect \(\delta_F\) under series refinement |
| `kron_elimination.py` | Exact Kron elimination of original vertices |
| `gcn_illustration.py` | Frozen two-layer GCN on Cora at test time |
| `graphs.py` | Loaders for the eight public graphs |
| `data/` | MATPOWER cases and DCRNN traffic adjacencies |

Cora is downloaded on first use via Planetoid. Generated JSON is written to `results/` and is not part of the repository.

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`operator_defect.py` and `kron_elimination.py` need PyTorch Geometric only to load Cora. `gcn_illustration.py` trains on CPU.

---

## Reproduction

```bash
python witnesses.py
python operator_defect.py
python kron_elimination.py
python gcn_illustration.py
```

Seeds, fractions, and processor coefficients are frozen in the scripts. Do not edit them after seeing output.

### Processors

| Name | Map | Role |
| :--- | :--- | :--- |
| affine | \(I-0.15L\) | Positive control (in-theorem) |
| quadratic | \(I-0.08L+\tfrac12(0.08)^2L^2\) | In-theorem failure |
| \(P^2\) | two-hop random walk | Outside the theorem |
| \(\hat A^2\) | two-hop normalized adjacency | Outside the theorem |
| Cheb-2 | \(0.5T_0+0.35T_1+0.15T_2\) on \(\tilde L=L_{\mathrm{sym}}-I\) | Outside the theorem; \(\lambda_{\max}=2\) |

Series refinement replaces a selected original edge of conductance \(w\) by a degree-2 node with conductances \(2w,2w\). That operation preserves the Kron map on the original vertices. For the operator-level checks, every original vertex is a boundary vertex, so \(T_\Phi(G)=\Phi_G\). The reported statistic
\[
\delta_F=\lVert S\Phi_{G'}E-\Phi_G\rVert_F\big/\lVert\Phi_G\rVert_F
\]
does not use a chosen signal.

---

## Expected output

**Analytic witnesses** (Frobenius gap; Cheb-2 uses \(\lambda_{\max}=2\))

|  | Two-terminal split | Triangle series |
| :--- | ---: | ---: |
| \(L^2\) | \(0\) | \(3.46\) |
| \(P^2\) | — | \(0.54\) |
| \(\hat A^2\) | — | \(0.30\) |
| Cheb-2 | — | \(0.27\) |

The two-terminal split annihilates the \(L^2\) defect and cannot prove necessity. The triangle series is the scaled-witness seed.

**Series refinement** on eight graphs (cross-graph median \(\delta_F\), five seeds)

| Processor | \(1\%\) | \(5\%\) | \(10\%\) | \(25\%\) |
| :--- | ---: | ---: | ---: | ---: |
| affine | \(3.65\times10^{-17}\) | \(5.68\times10^{-17}\) | \(8.18\times10^{-17}\) | \(1.18\times10^{-16}\) |
| quadratic | \(0.0105\) | \(0.0273\) | \(0.0754\) | \(0.149\) |
| \(P^2\) | \(0.159\) | \(0.300\) | \(0.399\) | \(0.595\) |
| \(\hat A^2\) | \(0.121\) | \(0.242\) | \(0.342\) | \(0.522\) |
| Cheb-2 | \(0.0483\) | \(0.108\) | \(0.154\) | \(0.269\) |

Worst affine defect \(1.57\times10^{-15}\). Kron residual on the refined Laplacian \(\lVert L_K-L\rVert_F=7.1\times10^{-13}\) on the large IEEE weights.

**Kron elimination** of original vertices (cross-graph median \(\delta_F\) at \(25\%\)): affine \(1.3\times10^{-16}\); quadratic \(0.084\); \(P^2\) \(0.437\); \(\hat A^2\) \(0.400\); Cheb-2 \(0.179\).

**Frozen Cora GCN** (ten seeds; Planetoid split; width \(16\); Adam, learning rate \(0.01\), weight decay \(5\times10^{-4}\), dropout \(0.5\); at most \(200\) epochs with validation checkpoint). At \(25\%\) series refinement the same frozen weights change \(3.50\%\) of test-node predictions and lose \(0.89\) percentage points of accuracy on \(9/10\) seeds. Features on inserted nodes are the midpoint \((x_i+x_j)/2\). This illustration lies outside the theorem.

Machine-precision affine residuals may differ in the last digits across BLAS implementations; they should remain at floating-point scale.

---

## Graphs

| Graph | Source | Weight |
| :--- | :--- | :--- |
| IEEE-14/30/57/118/300 | MATPOWER `case*.m` | DC series susceptance \(\lvert 1/x\rvert\) |
| METR-LA, PEMS-BAY | DCRNN sensor adjacency | Gaussian distance kernel, threshold \(0.1\) |
| Cora | Planetoid | Unweighted citation edges |

Isolated vertices, if any, receive a weight-\(1\) edge to the next index so that every graph is connected and undirected. Traffic speed time series are not used.

---

## Citation

```bibtex
@article{meng2026kron,
  title   = {Kron Consistency of Graph Filters under Exact Boundary Equivalence},
  author  = {Meng, Xianghui and Lin, Jionghao},
  year    = {2026}
}
```
