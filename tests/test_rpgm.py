import itertools

import numpy as np
import pytest

from lossytoric.rpgm import (PLANES, _polyakov_moments, action, gauge_transform, jackknife, local_field,
                             nishimori_beta, plaquettes, polyakov, sample_couplings, simulate, xi_over_L)


def random_links(shape, rng):
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=shape)


def test_sample_couplings_rates():
    rng = np.random.default_rng(0)
    p, q, e = 0.1, 0.2, 0.3
    J = sample_couplings(6, p, q, e, 500, rng)
    assert set(np.unique(J[:, 0])) == {-1, 1}                      # no erasure on spacelike plaquettes
    assert abs((J[:, 0] == -1).mean() - q) < 0.005
    Jt = J[:, 1:]
    assert abs((Jt == 0).mean() - e) < 0.005
    assert abs((Jt == -1).sum() / (Jt != 0).sum() - p) < 0.005


def test_gauge_invariance():
    """U_mu(s) -> g(s) U_mu(s) g(s+mu) leaves every plaquette and every Polyakov loop unchanged."""
    rng = np.random.default_rng(1)
    U = random_links((3, 3, 4, 4, 4), rng)
    g = random_links((3, 4, 4, 4), rng)
    V = gauge_transform(U, g)
    assert (V != U).any()
    assert np.array_equal(plaquettes(V), plaquettes(U))
    for d in range(3):
        assert np.array_equal(polyakov(V, d), polyakov(U, d))


def test_single_link_flip_hits_four_plaquettes_and_two_polyakov_lines():
    U = np.ones((3, 4, 4, 4), dtype=np.int8)
    U[2, 1, 2, 3] = -1                                             # one time link
    assert (plaquettes(U) == -1).sum() == 4
    assert (polyakov(U, 2) == -1).sum() == 1                       # the time line through it
    assert (polyakov(U, 0) == -1).sum() == 0                       # space lines don't contain it


def test_local_field_matches_action_difference():
    """Flipping link (mu, s) changes sum K_P U_P by exactly -2 U g."""
    rng = np.random.default_rng(2)
    L = 4
    U = random_links((2, 3, L, L, L), rng)
    K = rng.normal(size=(2, 3, L, L, L))                           # arbitrary anisotropic couplings
    A0 = action(U, K)
    for r, mu, x, y, t in [(0, 0, 0, 0, 0), (1, 1, 3, 2, 1), (0, 2, 1, 3, 3), (1, 2, 0, 0, 3)]:
        g = local_field(U, K, mu)
        V = U.copy()
        V[r, mu, x, y, t] *= -1
        dA = action(V, K)[r] - A0[r]
        assert np.isclose(dA, -2 * U[r, mu, x, y, t] * g[r, x, y, t])


def tree_gauge_configs(L=2):
    """All link configurations on the L=2 box up to gauge: the 7 links of a maximal tree are
    fixed to +1 (time links at t=0, y links at y=t=0, the x link at the origin), the other 17
    take all values. Each gauge orbit appears exactly once, so averages of gauge-invariant
    observables are exact."""
    assert L == 2
    tree = {(2, x, y, 0) for x in range(2) for y in range(2)} | {(1, x, 0, 0) for x in range(2)} | {(0, 0, 0, 0)}
    links = [(mu, x, y, t) for mu in range(3) for x in range(2) for y in range(2) for t in range(2)]
    free = [l for l in links if l not in tree]
    assert len(free) == 17
    confs = np.array(list(itertools.product([-1, 1], repeat=len(free))), dtype=np.int8)
    U = np.ones((len(confs), 3, 2, 2, 2), dtype=np.int8)
    for i, (mu, x, y, t) in enumerate(free):
        U[:, mu, x, y, t] = confs[:, i]
    return U


@pytest.mark.parametrize("seed", [3, 4])
def test_mc_matches_exact_enumeration(seed):
    """Fixed couplings with frustration, erasure and different space/time temperatures on the
    2x2x2 box: Monte Carlo averages = exact gauge-fixed sums."""
    rng = np.random.default_rng(seed)
    J1 = sample_couplings(2, 0.25, 0.15, 0.2, 1, rng)
    bp, bq = 0.6, 0.45
    Kb = np.array([bq, bp, bp])[:, None, None, None] * J1[0]
    U = tree_gauge_configs()
    A = action(U, Kb[None])
    w = np.exp(A - A.max())
    w /= w.sum()
    JP = J1[0][None] * plaquettes(U)
    phase = np.exp(2j * np.pi * np.arange(2) / 2)
    ex = {"e_space": w @ JP[:, 0].sum(axis=(1, 2, 3)),
          "e_time": w @ JP[:, 1:].sum(axis=(1, 2, 3, 4)),
          "pt2": w @ _polyakov_moments(U, 2, phase)[0],
          "ps2": w @ (0.5 * (_polyakov_moments(U, 0, phase)[0] + _polyakov_moments(U, 1, phase)[0]))}
    R = 400                                                    # independent chains, same couplings
    out = simulate(2, 0.1, R, n_therm=200, n_meas=400, seed=seed, start="hot",
                   J=np.repeat(J1, R, axis=0), beta=(bp, bq))
    for k, v in ex.items():
        mc, err = out[k].mean(), out[k].std() / np.sqrt(R)
        assert abs(mc - v) < 4 * err + 1e-3, (k, mc, v, err)


@pytest.mark.parametrize("p,q,e", [(0.03, 0.03, 0.0), (0.05, 0.02, 0.0), (0.02, 0.04, 0.25)])
def test_nishimori_energy_identity(p, q, e):
    """[<J_P U_P>] = 1-2p per non-erased timelike and 1-2q per spacelike plaquette, exactly."""
    L, R = 6, 200
    out = simulate(L, p, R, n_therm=50, n_meas=100, q=q, e=e, seed=7)
    N = L ** 3
    t = out["e_time"] / (2 * N)
    s = out["e_space"] / N
    assert abs(t.mean() - (1 - e) * (1 - 2 * p)) < 4 * t.std() / np.sqrt(R)
    assert abs(s.mean() - (1 - 2 * q)) < 4 * s.std() / np.sqrt(R)


def test_planted_and_hot_start_agree():
    """Polyakov loops are not invariant under the Nishimori gauge symmetry, so the planted start
    is biased until the chain decorrelates: both starts must give the same xi_P/L near p_c."""
    L, R, p = 6, 150, 0.033
    res = {}
    for start in ("planted", "hot"):
        out = simulate(L, p, R, n_therm=300, n_meas=300, seed=11, start=start, n_temps=6, beta_min=0.55)
        res[start] = jackknife(lambda a, b: xi_over_L(a, b, L), out["ps2"], out["psk2"])
    (a, da), (b, db) = res["planted"], res["hot"]
    assert abs(a - b) < 4 * np.hypot(da, db), res


def test_planes_and_nishimori_beta():
    assert PLANES == ((0, 1), (0, 2), (1, 2))
    assert np.isclose(np.tanh(nishimori_beta(0.03)), 0.94)
