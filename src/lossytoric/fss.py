"""Threshold extraction by finite-size scaling.

Near threshold the logical failure rate is assumed to be a smooth function of
the scaling variable  x = (q - q_c) L^{1/nu}, with q the swept noise parameter
(p at fixed e, or e at fixed p):

    P_L(q) = A + B x + C x^2

Fit parameters: q_c, nu, A, B, C (weighted least squares, binomial errors).
Uncertainties: parametric bootstrap, resampling every failure count from
Binomial(shots, observed rate) and refitting.

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
    q, L = np.asarray(q, float), np.asarray(L, float)
    shots, failures = np.asarray(shots, float), np.asarray(failures, float)
    P = failures / shots
    sigma = np.sqrt(np.clip(P * (1 - P), 1.0 / shots, None) / shots)
    q0 = float(np.mean(q)) if q0 is None else q0
    popt = _fit_once(q, L, P, sigma, q0, nu0)
    resid = (P - _model((q, L), *popt)) / sigma
    chi2_dof = float(resid @ resid / max(len(P) - 5, 1))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        Pb = rng.binomial(shots.astype(int), P) / shots
        try:
            boots.append(_fit_once(q, L, Pb, sigma, popt[0], popt[1]))
        except RuntimeError:
            continue
    if not boots:
        return FSSFit(popt[0], float("nan"), popt[1], float("nan"), popt, chi2_dof)
    boots = np.array(boots)
    return FSSFit(popt[0], float(boots[:, 0].std()), popt[1], float(boots[:, 1].std()), popt, chi2_dof)


def pairwise_crossings(q, L, shots, failures) -> list[tuple[int, int, float]]:
    """Linear-interpolated crossing of P_L(q) for consecutive system sizes.

    Their spread is a cheap, honest estimate of the finite-size systematic that the
    quadratic FSS ansatz (no corrections to scaling) does not capture.
    """
    q, L = np.asarray(q, float), np.asarray(L)
    P = np.asarray(failures, float) / np.asarray(shots, float)
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
