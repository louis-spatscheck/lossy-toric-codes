"""Random-plaquette Z2 gauge model (RPGM) on the Nishimori line: the stat-mech dual of
optimal decoding of the toric code with faulty syndrome measurements.

Mapping (Dennis, Kitaev, Landahl, Preskill 2002; Wang, Harrington, Preskill 2003)
-------------------------------------------------------------------------------
* Repeat the syndrome measurement for T rounds. Errors live on the edges of a
  spacetime lattice: a *space* edge is a qubit flipped in one round, a *time*
  edge is a wrong readout of one check between two rounds.
* Two error histories are equivalent iff they differ by a set of elementary
  squares of that lattice (a stabilizer applied in one round, or a qubit that
  flips and flips back while both of its checks lie). One Z2 variable per
  square -> on the dual lattice (a point at each cube centre) the variables
  live on LINKS, and each error edge pierces one dual PLAQUETTE.
* The weight of a history is a product over plaquettes of exp(beta_P J_P U_P),
  U_P = product of the 4 links of P: a Z2 lattice gauge theory with random
  signs. J_P = -1 where the reference history has an error, +1 elsewhere,
  0 for an erased qubit (no information, "dilution").
* Which plaquette carries which noise: a qubit (space) edge along x pierces the
  dual plaquette in the y-t plane, a readout (time) edge pierces one in the
  x-y plane. So
      TIMELIKE plaquettes (planes xt, yt) <- qubit flips p and erasures e,
      SPACELIKE plaquettes (plane xy)     <- readout errors q.
  Invisible at p = q, e = 0; it matters as soon as the model is anisotropic.
* Nishimori line: tanh(beta_p) = 1 - 2p on timelike, tanh(beta_q) = 1 - 2q on
  spacelike plaquettes. The optimal threshold is the confinement transition
  along this line; for p = q, e = 0 it is p_c ~ 0.033 (Ohno, Arakawa,
  Ichinose, Matsui 2004).

Conventions
-----------
* Periodic L x L x L box (T = L), L even. Directions 0, 1 = space (x, y),
  2 = time; the lattice is always the last three array axes, in that order.
* Links U[..., mu, x, y, t] = +-1 (int8): the link from site s to s + e_mu.
* Plaquettes and couplings J[..., plane, x, y, t] with plane index
  0 = (x, y) spacelike, 1 = (x, t) and 2 = (y, t) timelike. The plaquette at
  site s in plane (a, b) is U_a(s) U_b(s + a) U_a(s + b) U_b(s).

Symmetries used in the tests
---------------------------
* Lattice gauge symmetry: U_mu(s) -> g(s) U_mu(s) g(s + mu) for any g = +-1 on
  sites changes no plaquette. Hence (Elitzur) only gauge-invariant products,
  closed loops, have non-zero averages: plaquettes, Wilson loops, and
  Polyakov loops (straight lines wrapping a periodic direction).
* Nishimori gauge symmetry: flipping links AND the couplings of the plaquettes
  they touch. Disorder is drawn in the "ferromagnetic gauge", where all
  U = +1 is an exact equilibrium sample for observables invariant under it
  (the energy), but NOT for Polyakov loops, which involve U only. So the
  planted start needs the same equilibration check as in rbim.py.
* Nishimori energy identity: [<J_P U_P>] = (1 - 2p) on each non-erased
  timelike and (1 - 2q) on each spacelike plaquette, exactly.

Updates: heat bath on links in 6 sublattices (direction mu x parity of
x + y + t; two links of the same direction share a plaquette only if their
base sites are neighbours) plus parallel tempering over a ladder of overall
coupling scales f_0 = 1 > f_1 > ... > f_min. Everything is vectorised over
R disorder samples and T temperatures: arrays of shape (R, T, 3, L, L, L).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rbim import binder, jackknife, nishimori_beta, xi_over_L   # noqa: F401  (re-exported)

PLANES = ((0, 1), (0, 2), (1, 2))        # plane index -> the two directions spanning it
SPACE_PLANES = (0,)                      # readout errors q
TIME_PLANES = (1, 2)                     # qubit flips p, erasures e


def _ax(d: int) -> int:
    """Array axis of lattice direction d (the lattice is the last three axes)."""
    return d - 3


def _planes_with(mu: int):
    """(plane index, other direction) for the two planes that contain direction mu."""
    return [(i, b if a == mu else a) for i, (a, b) in enumerate(PLANES) if mu in (a, b)]


# ---------- disorder ----------

def sample_couplings(L: int, p: float, q: float, e: float, R: int, rng: np.random.Generator) -> np.ndarray:
    """Plaquette signs J[r, plane, x, y, t] for R disorder samples: +1 (no error), -1 (error), 0 (erased).

    Timelike planes: erased with prob e (independently in every round, i.e. a lost qubit is
    replaced before the next round), otherwise flipped with prob p. Spacelike plane: flipped
    with prob q (readout errors; checks are assumed measurable even next to a lost qubit).
    """
    J = np.empty((R, 3, L, L, L), dtype=np.int8)
    for i in SPACE_PLANES:
        J[:, i] = np.where(rng.random((R, L, L, L)) < q, -1, 1)
    for i in TIME_PLANES:
        J[:, i] = np.where(rng.random((R, L, L, L)) < p, -1, 1)
        J[:, i][rng.random((R, L, L, L)) < e] = 0
    return J


# ---------- gauge field ----------

def plaquette(U: np.ndarray, plane: int) -> np.ndarray:
    """U_a(s) U_b(s+a) U_a(s+b) U_b(s) for one plane (a, b); shape (..., L, L, L)."""
    a, b = PLANES[plane]
    Ua, Ub = U[..., a, :, :, :], U[..., b, :, :, :]
    return Ua * np.roll(Ub, -1, axis=_ax(a)) * np.roll(Ua, -1, axis=_ax(b)) * Ub


def plaquettes(U: np.ndarray) -> np.ndarray:
    """All plaquettes, shape (..., 3, L, L, L)."""
    return np.stack([plaquette(U, i) for i in range(3)], axis=-4)


def action(U: np.ndarray, K: np.ndarray) -> np.ndarray:
    """sum_P K_P U_P (= -beta H), summed over planes and lattice: shape (...)."""
    return (K * plaquettes(U)).sum(axis=(-4, -3, -2, -1))


def local_field(U: np.ndarray, K: np.ndarray, mu: int) -> np.ndarray:
    """g_mu(s) = sum over the 4 plaquettes P containing link (mu, s) of K_P * (product of the
    other three links of P). The action depends on U_mu(s) only through U_mu(s) * g_mu(s), so
    heat bath gives P(U_mu(s) = +1) = 1 / (1 + exp(-2 g)). Shape (..., L, L, L).

    The link (mu, s) lies in the plaquette at s and in the one at s - e_nu, for each of the
    two planes (mu, nu); since U = +-1, (other three) = U_P * U_mu(s).
    """
    g = 0.0
    for i, nu in _planes_with(mu):
        KP = K[..., i, :, :, :] * plaquette(U, i)
        g = g + KP + np.roll(KP, 1, axis=_ax(nu))
    return U[..., mu, :, :, :] * g


def gauge_transform(U: np.ndarray, g: np.ndarray) -> np.ndarray:
    """U_mu(s) -> g(s) U_mu(s) g(s + mu); g has shape (..., L, L, L)."""
    out = U.copy()
    for mu in range(3):
        out[..., mu, :, :, :] = g * U[..., mu, :, :, :] * np.roll(g, -1, axis=_ax(mu))
    return out


def polyakov(U: np.ndarray, d: int) -> np.ndarray:
    """Polyakov loops along direction d: product of the L links on each straight line that
    wraps the box in direction d. Shape (..., L, L) over the two transverse directions."""
    return np.prod(U[..., d, :, :, :], axis=_ax(d), dtype=np.int8)


# ---------- Monte Carlo ----------

def coupling_ladder(n_temps: int, f_min: float) -> np.ndarray:
    """Overall coupling scales f_0 = 1 (the measured, Nishimori temperature) > ... > f_min."""
    if n_temps == 1:
        return np.array([1.0])
    return f_min ** (np.arange(n_temps) / (n_temps - 1))


@dataclass
class Accumulator:
    """Per-sample thermal averages, accumulated over measurement sweeps.

    Polyakov observables use Pbar_d = mean over transverse positions of the loops along d,
    and its lowest Fourier mode Pk_d (k = 2 pi / L along each transverse direction):
      ps2, ps4, psk2: <Pbar^2>, <Pbar^4>, <|Pk|^2> averaged over the two SPACE directions
      pt2, pt4, ptk2: the same for loops along TIME
      e_space, e_time: <sum J_P U_P> over spacelike / timelike plaquettes
    """
    R: int
    n: int = 0
    names: tuple = ("ps2", "ps4", "psk2", "pt2", "pt4", "ptk2", "e_space", "e_time")
    data: dict = field(default=None)

    def __post_init__(self):
        self.data = {k: np.zeros(self.R) for k in self.names}

    def add(self, **vals):
        for k, v in vals.items():
            self.data[k] += v
        self.n += 1

    def finalize(self) -> dict:
        return {k: v / self.n for k, v in self.data.items()}


def _polyakov_moments(U0: np.ndarray, d: int, phase: np.ndarray):
    """Pbar^2, Pbar^4 and the mean of the two |Pk|^2 for loops along d; each shape (R,)."""
    P = polyakov(U0, d).astype(float)                       # (R, L, L)
    N = P.shape[-1] * P.shape[-2]
    m = P.sum(axis=(-2, -1)) / N
    k1 = (P.sum(axis=-1) @ phase) / N
    k2 = (P.sum(axis=-2) @ phase) / N
    return m * m, m ** 4, 0.5 * (np.abs(k1) ** 2 + np.abs(k2) ** 2)


def simulate(L: int, p: float, R: int, n_therm: int, n_meas: int, q: float | None = None,
             e: float = 0.0, seed: int = 0, start: str = "planted", J: np.ndarray | None = None,
             beta: float | tuple[float, float] | None = None, n_temps: int = 1,
             beta_min: float = 0.5) -> dict:
    """Heat bath (+ parallel tempering if n_temps > 1) for R disorder samples.

    beta: None -> Nishimori line, (beta_p, beta_q) = (beta_N(p), beta_N(q)); a float sets both;
          a pair sets (timelike, spacelike) separately.
    q:    readout error rate, default q = p.
    J:    optional fixed couplings (R, 3, L, L, L); otherwise drawn with sample_couplings.
    beta_min: the hottest timelike coupling of the PT ladder (all couplings are scaled
          together). The clean 3D Z2 gauge transition is at beta ~ 0.76, so keep it below that.

    Returns per-sample thermal averages at the coldest temperature (see Accumulator) and
    'swap_rate', the PT acceptance per neighbouring pair.
    """
    if L % 2:
        raise ValueError("sublattice update needs even L")
    q = p if q is None else q
    rng = np.random.default_rng(seed)
    if J is None:
        J = sample_couplings(L, p, q, e, R, rng)
    R = J.shape[0]
    if beta is None:
        bp, bq = nishimori_beta(p), nishimori_beta(q)
    elif np.isscalar(beta):
        bp = bq = float(beta)
    else:
        bp, bq = map(float, beta)
    bplane = np.array([bq, bp, bp], dtype=np.float32)        # plane 0 spacelike, 1-2 timelike
    f = coupling_ladder(n_temps, min(1.0, beta_min / bp)).astype(np.float32)
    T = len(f)
    Kbase = bplane[:, None, None, None] * J.astype(np.float32)          # (R, 3, L, L, L)
    K = f[None, :, None, None, None, None] * Kbase[:, None]             # (R, T, 3, L, L, L)

    if start == "planted":
        U = np.ones((R, T, 3, L, L, L), dtype=np.int8)
    elif start == "hot":
        U = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, T, 3, L, L, L))
    else:
        raise ValueError(start)

    x, y, t = np.indices((L, L, L))
    masks = [((x + y + t) % 2 == c) for c in (0, 1)]
    phase = np.exp(2j * np.pi * np.arange(L) / L)
    acc = Accumulator(R)
    swaps_acc = np.zeros(max(T - 1, 1))
    swaps_try = np.zeros(max(T - 1, 1))

    for sweep in range(n_therm + n_meas):
        for mu in range(3):
            for mask in masks:
                g = local_field(U, K, mu)
                up = rng.random(g.shape, dtype=np.float32) * (1.0 + np.exp(-2.0 * g)) < 1.0
                U[:, :, mu] = np.where(mask, np.where(up, 1, -1).astype(np.int8), U[:, :, mu])
        if T > 1:                                                         # replica exchange
            A = action(U, Kbase[:, None]).astype(float)                   # (R, T), unscaled
            for k in range(sweep % 2, T - 1, 2):
                log_a = (f[k] - f[k + 1]) * (A[:, k + 1] - A[:, k])
                ok = np.log(rng.random(R)) < log_a
                if ok.any():
                    tmp = U[ok, k].copy()
                    U[ok, k] = U[ok, k + 1]
                    U[ok, k + 1] = tmp
                    A[ok, k], A[ok, k + 1] = A[ok, k + 1], A[ok, k]
                swaps_acc[k] += ok.sum()
                swaps_try[k] += R
        if sweep >= n_therm:
            U0 = U[:, 0]
            JP = J * plaquettes(U0)
            vals = {"e_space": JP[:, list(SPACE_PLANES)].sum(axis=(1, 2, 3, 4)),
                    "e_time": JP[:, list(TIME_PLANES)].sum(axis=(1, 2, 3, 4))}
            sx = _polyakov_moments(U0, 0, phase)
            sy = _polyakov_moments(U0, 1, phase)
            st = _polyakov_moments(U0, 2, phase)
            vals.update(ps2=0.5 * (sx[0] + sy[0]), ps4=0.5 * (sx[1] + sy[1]), psk2=0.5 * (sx[2] + sy[2]),
                        pt2=st[0], pt4=st[1], ptk2=st[2])
            acc.add(**vals)
    out = acc.finalize()
    out["swap_rate"] = swaps_acc / np.maximum(swaps_try, 1)
    return out
