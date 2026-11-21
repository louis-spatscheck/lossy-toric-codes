"""Threshold extraction by finite-size scaling.

Near threshold the logical failure rate is assumed to be a smooth function of
the scaling variable  x = (q - q_c) L^{1/nu}, with q the swept noise parameter
(p at fixed e, or e at fixed p):

    P_L(q) = A + B x + C x^2

Fit parameters: q_c, nu, A, B, C (weighted least squares, binomial errors).
Uncertainties: parametric bootstrap (failure counts resampled from
Binomial(shots, observed rate), or y from Gaussians), refit, and half the
16-84% range of the refitted values (robust against runaway refits).

Caveat: this quadratic ansatz has no corrections to scaling. Use data close
to the crossing and check stability against dropping the smallest L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def _model(X, qc, nu, A, B, C):
    q, L = X
    x = (q - qc) * L ** (1.0 / nu)
    return A + B * x + C * x * x


@dataclass
class FSSFit:
    qc: float
    qc_err: float
    nu: float
    nu_err: float
    params: np.ndarray
    chi2_dof: float
    sys: float = float("nan")   # finite-size systematic, set by the caller (see pairwise_crossings)


def _fit_once(q, L, P, sigma, q0, nu0):
    p0 = [q0, nu0, float(np.median(P)), 1.0, 0.0]
    popt, _ = curve_fit(_model, (q, L), P, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=20000)
    return popt


def fit_threshold(q, L, shots, failures, q0=None, nu0=1.5, n_boot=200, seed=0) -> FSSFit:
    """FSS fit of logical failure rates (binomial counts)."""
    shots, failures = np.asarray(shots, float), np.asarray(failures, float)
    P = failures / shots
    sigma = np.sqrt(np.clip(P * (1 - P), 1.0 / shots, None) / shots)
    rng = np.random.default_rng(seed)
    resample = lambda: rng.binomial(shots.astype(int), P) / shots
    return _fit_with_bootstrap(q, L, P, sigma, resample, q0, nu0, n_boot)


def fit_threshold_y(q, L, y, sigma, q0=None, nu0=1.5, n_boot=200, seed=0) -> FSSFit:
    """FSS fit of any dimensionless observable y (e.g. xi/L) with Gaussian errors sigma."""
    y, sigma = np.asarray(y, float), np.asarray(sigma, float)
    rng = np.random.default_rng(seed)
    resample = lambda: y + sigma * rng.standard_normal(len(y))
    return _fit_with_bootstrap(q, L, y, sigma, resample, q0, nu0, n_boot)


def _fit_with_bootstrap(q, L, y, sigma, resample, q0, nu0, n_boot) -> FSSFit:
    q, L = np.asarray(q, float), np.asarray(L, float)
    q0 = float(np.mean(q)) if q0 is None else q0
    popt = _fit_once(q, L, y, sigma, q0, nu0)
    resid = (y - _model((q, L), *popt)) / sigma
    chi2_dof = float(resid @ resid / max(len(y) - 5, 1))
    boots = []
    for _ in range(n_boot):
        try:
            boots.append(_fit_once(q, L, resample(), sigma, popt[0], popt[1]))
        except RuntimeError:
            continue
    if not boots:
        return FSSFit(popt[0], float("nan"), popt[1], float("nan"), popt, chi2_dof)
    boots = np.array(boots)
    # robust spread: half the 16-84% range (a few runaway refits must not dominate the error)
    half68 = lambda x: float(np.subtract(*np.percentile(x, [84, 16])) / 2)
    return FSSFit(popt[0], half68(boots[:, 0]), popt[1], half68(boots[:, 1]), popt, chi2_dof)


def pairwise_crossings(q, L, shots, failures=None) -> list[tuple[int, int, float]]:
    """Linear-interpolated crossing of P_L(q) for consecutive system sizes.

    Their spread is a cheap, honest estimate of the finite-size systematic that the
    quadratic FSS ansatz (no corrections to scaling) does not capture.
    """
    q, L = np.asarray(q, float), np.asarray(L)
    # called either with (shots, failures) or with a precomputed observable y as 3rd argument
    P = np.asarray(shots, float) if failures is None else np.asarray(failures, float) / np.asarray(shots, float)
    Ls = sorted(set(L.tolist()))
    out = []
    for a, b in zip(Ls[:-1], Ls[1:]):
        qa, Pa = q[L == a], P[L == a]
        qb, Pb = q[L == b], P[L == b]
        common, ia, ib = np.intersect1d(qa, qb, return_indices=True)
        d = Pb[ib] - Pa[ia]
        idx = np.where(np.diff(np.sign(d)) != 0)[0]
        if len(idx):
            i = idx[0]
            x = common[i] - d[i] * (common[i + 1] - common[i]) / (d[i + 1] - d[i])
            out.append((a, b, float(x)))
    return out


def model(q, L, params):
    return _model((np.asarray(q, float), np.asarray(L, float)), *params)
