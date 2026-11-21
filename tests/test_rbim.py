import itertools

import numpy as np
import pytest

from lossytoric.rbim import (bond_energy, jackknife, local_field, nishimori_beta, sample_couplings,
                             simulate, xi_over_L)


def exact_averages(Jh, Jv, beta):
    """Brute-force <m^2> and <sum J s s> for one 4x4 coupling set (2^16 states)."""
    L = Jh.shape[0]
    N = L * L
    confs = np.array(list(itertools.product([-1, 1], repeat=N)), dtype=np.int8).reshape(-1, L, L)
    E = bond_energy(confs, Jh[None], Jv[None]).astype(float)
    w = np.exp(beta * (E - E.max()))
    w /= w.sum()
    m = confs.sum(axis=(1, 2)) / N
    return float(w @ m**2), float(w @ E)


def test_local_field_matches_energy_difference():
    rng = np.random.default_rng(0)
    Jh, Jv = sample_couplings(6, 0.2, 0.2, 3, rng)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(3, 6, 6))
    h = local_field(s, Jh, Jv)
    for (r, i, j) in [(0, 0, 0), (1, 3, 5), (2, 5, 2)]:
        t = s.copy()
        t[r, i, j] *= -1
        dE = bond_energy(t, Jh, Jv)[r] - bond_energy(s, Jh, Jv)[r]
        assert dE == -2 * s[r, i, j] * h[r, i, j]


@pytest.mark.parametrize("seed", [1, 2])
def test_mc_matches_exact_enumeration(seed):
    """Fixed random couplings (with frustration and dilution) on a 4x4 torus."""
    rng = np.random.default_rng(seed)
    Jh1, Jv1 = sample_couplings(4, 0.25, 0.2, 1, rng)
    beta = 0.8
    m2_ex, E_ex = exact_averages(Jh1[0], Jv1[0], beta)
    R = 400                                   # independent chains on the same couplings
    Jh, Jv = np.repeat(Jh1, R, axis=0), np.repeat(Jv1, R, axis=0)
    out = simulate(4, 0.1, 0.0, R, n_therm=200, n_meas=400, seed=seed, start="hot", Jh=Jh, Jv=Jv, beta=beta)
    m2, m2_err = out["m2"].mean(), out["m2"].std() / np.sqrt(R)
    E, E_err = out["ebond"].mean(), out["ebond"].std() / np.sqrt(R)
    assert abs(m2 - m2_ex) < 4 * m2_err + 1e-3
    assert abs(E - E_ex) < 4 * E_err + 1e-2


@pytest.mark.parametrize("p,e", [(0.08, 0.0), (0.11, 0.0), (0.06, 0.25)])
def test_nishimori_energy_identity(p, e):
    """[<sum_b J s s>] = N_b (1-e)(1-2p) exactly on the Nishimori line."""
    L, R = 8, 400
    out = simulate(L, p, e, R, n_therm=100, n_meas=200, seed=7)
    per_bond = out["ebond"] / (2 * L * L)
    mean, err = per_bond.mean(), per_bond.std() / np.sqrt(R)
    assert abs(mean - (1 - e) * (1 - 2 * p)) < 4 * err


def test_planted_and_hot_start_agree():
    """Equilibration check with parallel tempering: both starts give the same xi/L near p_c.
    (Without PT this fails: at L = 12 the starts still disagree after 10^4 sweeps.)"""
    L, R, p = 12, 200, 0.105
    res = {}
    for start in ("planted", "hot"):
        out = simulate(L, p, 0.0, R, n_therm=300, n_meas=300, seed=11, start=start, n_temps=8, beta_min=0.6)
        res[start] = jackknife(lambda a, b: xi_over_L(a, b, L), out["m2"], out["mk2"])
    (a, da), (b, db) = res["planted"], res["hot"]
    assert abs(a - b) < 4 * np.hypot(da, db)


def test_nishimori_beta():
    assert np.isclose(np.tanh(nishimori_beta(0.1)), 0.8)
