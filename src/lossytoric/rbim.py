"""Random-bond Ising model (RBIM) on the Nishimori line: the stat-mech dual of
optimal (maximum-likelihood) decoding of the toric code.

Mapping (Dennis, Kitaev, Landahl, Preskill 2002)
------------------------------------------------
* Ising spins live on the dual lattice (plaquettes of the toric code); there is
  one bond per qubit. For an L x L torus the dual lattice is again L x L.
* A bit flip on a qubit makes its bond antiferromagnetic: J = -1 (else +1).
* An erased qubit carries a uniformly random flip, so it gives no information:
  its bond is removed, J = 0 ("dilution"). This is the erasure version of the
  mapping (cf. Stace & Barrett 2010; Colmenarez, Kim, Mueller 2025).
* The temperature is fixed by the noise, not chosen freely: on the Nishimori
  line  beta = (1/2) ln((1-p)/p),  i.e.  tanh(beta) = 1 - 2p.
* The optimal threshold p_c(e) is where this disordered magnet loses
  ferromagnetic order. For e = 0 this is the Nishimori point, p_c ~= 0.109.

Exact facts used as checks
-------------------------
1. Nishimori energy identity: [<sum_b J_b s s>] = N_b (1-e)(1-2p) exactly.
2. Planted start: disorder is drawn in the "ferromagnetic gauge", where the
   all-up configuration is an exact equilibrium sample *for gauge-invariant
   observables* (e.g. the energy), so those need no equilibration. The
   magnetisation is NOT gauge invariant: in this gauge it is the overlap with
   the starting configuration, so it is biased upwards until the chain has
   decorrelated, while a hot (random) start is biased downwards. Equilibration
   check: the two starts must agree.

Updates: checkerboard heat bath plus parallel tempering (PT). Each disorder
sample is simulated at a ladder of inverse temperatures beta_N = b_0 > b_1 >
... > b_min, and neighbouring temperatures swap configurations with the
Metropolis probability min(1, exp[(b_t - b_{t+1})(E_t - E_{t+1})]). Only the
coldest copy (the Nishimori temperature) is measured; hotter copies exist to
decorrelate it. Without PT the magnetisation equilibrates far too slowly near
the transition (planted and hot starts still disagree after 10^4 sweeps at
L = 12); Swendsen-Wang cluster moves were tried and did not help here.
Everything is vectorised over R disorder samples and T temperatures (arrays of
shape (R, T, L, L)). L must be even.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def nishimori_beta(p: float) -> float:
    if not 0 < p < 0.5:
        raise ValueError("Nishimori line needs 0 < p < 1/2")
    return 0.5 * np.log((1 - p) / p)


def sample_couplings(L: int, p: float, e: float, R: int, rng: np.random.Generator):
    """Bond signs for R disorder samples. Jh[r,i,j]: (i,j)-(i,j+1); Jv[r,i,j]: (i,j)-(i+1,j).
    Values: +1 (no error), -1 (bit flip), 0 (erased)."""
    def draw():
        J = np.where(rng.random((R, L, L)) < p, -1, 1).astype(np.int8)
        J[rng.random((R, L, L)) < e] = 0
        return J
    return draw(), draw()


def local_field(s, Jh, Jv):
    """h_x = sum over the 4 neighbours y of J_xy s_y (integer in -4..4); lattice = last two axes."""
    return (Jh * np.roll(s, -1, axis=-1) + np.roll(Jh * s, 1, axis=-1)
            + Jv * np.roll(s, -1, axis=-2) + np.roll(Jv * s, 1, axis=-2))


def bond_energy(s, Jh, Jv):
    """sum_b J_b s_i s_j (= -H), summed over the last two (lattice) axes."""
    return ((Jh * s * np.roll(s, -1, axis=-1)).sum(axis=(-2, -1))
            + (Jv * s * np.roll(s, -1, axis=-2)).sum(axis=(-2, -1)))


def beta_ladder(beta0: float, n_temps: int, beta_min: float = 0.35) -> np.ndarray:
    """Geometric ladder from beta0 (coldest, measured) down to beta_min."""
    if n_temps == 1:
        return np.array([beta0])
    return beta0 * (beta_min / beta0) ** (np.arange(n_temps) / (n_temps - 1))


@dataclass
class Accumulator:
    """Per-sample thermal averages, accumulated over measurement sweeps."""
    R: int
    n: int = 0
    m2: np.ndarray = field(default=None)
    m4: np.ndarray = field(default=None)
    mk2: np.ndarray = field(default=None)
    ebond: np.ndarray = field(default=None)

    def __post_init__(self):
        for name in ("m2", "m4", "mk2", "ebond"):
            setattr(self, name, np.zeros(self.R))

    def finalize(self) -> dict:
        return {k: getattr(self, k) / self.n for k in ("m2", "m4", "mk2", "ebond")}


def simulate(L: int, p: float, e: float, R: int, n_therm: int, n_meas: int,
             seed: int = 0, start: str = "planted", Jh=None, Jv=None, beta: float | None = None,
             n_temps: int = 1, beta_min: float = 0.35) -> dict:
    """Heat bath (+ parallel tempering if n_temps > 1) for R disorder samples.

    Returns per-sample thermal averages at the coldest temperature (beta, by default
    the Nishimori value): m2 = <m^2>, m4 = <m^4>, mk2 = <|m(k_min)|^2> (mean of the
    x and y directions), ebond = <sum_b J s s>, with m = (1/N) sum_x s_x; plus
    'swap_rate', the PT acceptance per neighbouring temperature pair.
    """
    if L % 2:
        raise ValueError("checkerboard update needs even L")
    rng = np.random.default_rng(seed)
    if Jh is None:
        Jh, Jv = sample_couplings(L, p, e, R, rng)
    R = Jh.shape[0]
    beta = nishimori_beta(p) if beta is None else beta
    betas = beta_ladder(beta, n_temps, beta_min)
    T = len(betas)
    Jh4, Jv4 = Jh[:, None], Jv[:, None]                        # broadcast over temperatures
    # heat bath: P(s = +1 | h) = 1 / (1 + exp(-2 beta h)), lookup table over h = -4..4
    p_up = 1.0 / (1.0 + np.exp(-2.0 * betas[:, None] * np.arange(-4, 5)[None, :]))
    t_idx = np.arange(T)[None, :, None, None]
    if start == "planted":
        s = np.ones((R, T, L, L), dtype=np.int8)
    elif start == "hot":
        s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, T, L, L))
    else:
        raise ValueError(start)
    ii, jj = np.indices((L, L))
    masks = [((ii + jj) % 2 == c) for c in (0, 1)]
    N = L * L
    phase = np.exp(2j * np.pi * np.arange(L) / L)
    acc = Accumulator(R)
    swaps_acc = np.zeros(max(T - 1, 1))
    swaps_try = np.zeros(max(T - 1, 1))

    for sweep in range(n_therm + n_meas):
        for mask in masks:
            h = local_field(s, Jh4, Jv4)
            up = rng.random(s.shape) < p_up[t_idx, h + 4]
            s = np.where(mask, np.where(up, 1, -1).astype(np.int8), s)
        if T > 1:                                           # replica-exchange moves
            eb = bond_energy(s, Jh4, Jv4).astype(float)    # (R, T)
            for t in range(sweep % 2, T - 1, 2):
                # swap configs at t, t+1 with prob min(1, exp(-(b_t - b_{t+1})(eb_t - eb_{t+1})))
                log_a = -(betas[t] - betas[t + 1]) * (eb[:, t] - eb[:, t + 1])
                acc_mask = np.log(rng.random(R)) < log_a
                if acc_mask.any():
                    tmp = s[acc_mask, t].copy()
                    s[acc_mask, t] = s[acc_mask, t + 1]
                    s[acc_mask, t + 1] = tmp
                swaps_acc[t] += acc_mask.sum()
                swaps_try[t] += R
        if sweep >= n_therm:
            s0 = s[:, 0]
            m = s0.sum(axis=(1, 2)) / N
            mkx = (s0.sum(axis=1) @ phase) / N              # k = (2pi/L) along j
            mky = (s0.sum(axis=2) @ phase) / N              # k = (2pi/L) along i
            acc.m2 += m * m
            acc.m4 += m ** 4
            acc.mk2 += 0.5 * (np.abs(mkx) ** 2 + np.abs(mky) ** 2)
            acc.ebond += bond_energy(s0, Jh, Jv)
            acc.n += 1
    out = acc.finalize()
    out["swap_rate"] = swaps_acc / np.maximum(swaps_try, 1)
    return out


# ---------- disorder averages with jackknife errors ----------

def xi_over_L(m2, mk2, L):
    """Second-moment correlation length / L from disorder-averaged <m^2>, <|m_k|^2>."""
    ratio = np.mean(m2) / np.mean(mk2)
    return np.sqrt(max(ratio - 1.0, 0.0)) / (2 * np.sin(np.pi / L)) / L


def binder(m2, m4):
    return 1.5 - 0.5 * np.mean(m4) / np.mean(m2) ** 2


def jackknife(estimator, *arrays, n_blocks: int = 20):
    """Jackknife mean and error of a (non-linear) function of sample means."""
    n = len(arrays[0])
    n_blocks = min(n_blocks, n)
    idx = np.array_split(np.arange(n), n_blocks)
    full = estimator(*arrays)
    loo = np.array([estimator(*(a[np.setdiff1d(np.arange(n), b)] for a in arrays)) for b in idx])
    err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((loo - loo.mean()) ** 2))
    return full, err
