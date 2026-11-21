import numpy as np

from lossytoric.fss import fit_threshold, model, pairwise_crossings


def test_fit_recovers_known_threshold():
    """Synthetic data drawn from the FSS ansatz with q_c = 0.1, nu = 1.5."""
    rng = np.random.default_rng(0)
    true = np.array([0.10, 1.5, 0.25, 1.2, 0.5])
    q = np.tile(np.linspace(0.09, 0.11, 9), 4)
    L = np.repeat([8, 12, 16, 24], 9)
    shots = np.full(q.shape, 200_000)
    P = np.clip(model(q, L, true), 1e-4, 1 - 1e-4)
    failures = rng.binomial(shots, P)
    fit = fit_threshold(q, L, shots, failures, q0=0.098, n_boot=50)
    assert abs(fit.qc - 0.10) < 3 * fit.qc_err + 1e-4
    assert abs(fit.nu - 1.5) < 0.2
    xs = [x for _, _, x in pairwise_crossings(q, L, shots, failures)]
    assert len(xs) == 3 and all(abs(x - 0.10) < 2e-3 for x in xs)
