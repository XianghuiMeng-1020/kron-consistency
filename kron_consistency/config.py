"""Frozen coefficients shared by the operator-level experiments.

Do not edit after seeing results.
"""

SEEDS = [0, 1, 2, 3, 4]
FRACS = [0.00, 0.01, 0.05, 0.10, 0.25]
ALPHA_AFF = 0.15
ALPHA_QUAD = 0.08
CHEB_C = (0.5, 0.35, 0.15)
KRON_ATOL = 1e-8
AFFINE_ATOL = 1e-8
WK_FP_TOL = 1e-14
SPECTRAL_N_MAX = 400
HARM_PROBES = 4
HARM_SEED = 20260911
PROCESSORS = ("affine", "quadratic", "rw2", "norm2", "cheb2")
NON_AFFINE = ("quadratic", "rw2", "norm2", "cheb2")
