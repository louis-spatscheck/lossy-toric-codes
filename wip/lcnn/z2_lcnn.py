"""Minimal gauge-covariant neural decoder for the toric code with erasure (Z2, code capacity).

Dictionary (conventions of src/lossytoric/toric.py)
--------------------------------------------------
* An X-error chain C on the primal edges is a Z2 link field U = (-1)^C on the DUAL lattice.
* Adding an X stabilizer (plaquette_x) flips the 4 dual links at one dual site: a gauge
  transformation. Equivalent errors = gauge-equivalent link fields.
* Dual plaquettes = primal vertices: the plaquette products of U are the SYNDROME.
* Dual Polyakov loops are the logical-Z supports: P0(j) = (-1)^{sum_i C[h(i,j)]} (one line per
  column j; row 0 of logical_z() is the line j = 0), P1(i) = (-1)^{sum_j C[v(i,j)]}.
  Adding a logical X (an error loop winding the torus) flips every P of one family.
* Erasure is a gauge-invariant scalar on links (a mask).

What survives for an abelian group
----------------------------------
In an L-CNN, site features W(x) transform as W -> Omega(x) W Omega(x)^dagger. For Z2 this
action is trivial, so every closed-loop feature is just a gauge-invariant number, i.e. a
product of syndrome bits. The only gauge-invariant information in U beyond the syndrome is the
two holonomies. Any gauge-invariant decoder is therefore f(syndrome, erasure, P0, P1).

The optimal decoder has one more exact symmetry: with Delta F_x(C) = ln Z(C) / Z(C + logical_x)
(the domain-wall free energy, Phase 2a), Delta F_x(C + logical_x) = -Delta F_x(C). This model
builds that in with a Polyakov readout:
    logit_x = mean over lines l of  P_x(l) * G_x(l; syndrome, erasure),
where G_x is a translation-equivariant CNN (circular padding) averaged along each line. So the
output is (i) exactly gauge invariant, (ii) exactly odd under adding a logical operator to the
reference chain, (iii) translation equivariant. logit_x > 0 means "the reference chain's class
is right in direction x"; the exact optimum would be logit_x = Delta F_x.

The reference chain C can be any chain with the observed syndrome; here MWPM's correction, so
the network learns when to flip MWPM's class. Because of (i)-(ii), handing it a different
representative (more stabilizers, or a logical) changes nothing or flips the output exactly.

Training uses BCE on the two binary labels t_x = 1 if (error + C) does not wind in direction x.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lossytoric.decoder import ErasureMWPM
from lossytoric.noise import sample_errors
from lossytoric.toric import ToricCode, h_edge, v_edge


# ---------------- data: chains -> gauge-invariant inputs ----------------

def logical_x(L: int) -> np.ndarray:
    """2 x n primal loops winding the torus: row 0 is detected by P0 (h(0, j) for all j),
    row 1 by P1 (v(i, 0) for all i)."""
    lx = np.zeros((2, 2 * L * L), dtype=np.uint8)
    for j in range(L):
        lx[0, h_edge(0, j, L)] = 1
    for i in range(L):
        lx[1, v_edge(i, 0, L)] = 1
    return lx


def features(C: np.ndarray, erased: np.ndarray, H, L: int):
    """Chains (B, 2L^2) -> local invariants x (B, L, L, 3) = [syndrome, erased h, erased v]
    and Polyakov signs P0 (B, L) over columns j, P1 (B, L) over rows i."""
    B = C.shape[0]
    C = C.astype(np.int64)                                  # unsigned ints would wrap in 1 - 2*parity
    erased = erased.astype(np.int64)
    s = (C @ H.T.toarray() % 2).reshape(B, L, L)           # vertex (i, j) -> [i, j]
    Ch = C[:, : L * L].reshape(B, L, L)                     # h(i, j) -> [i, j]
    Cv = C[:, L * L:].reshape(B, L, L)                      # v(i, j) -> [i, j]
    P0 = 1 - 2 * (Ch.sum(axis=1) % 2)                       # line j: h(i, j) for all i
    P1 = 1 - 2 * (Cv.sum(axis=2) % 2)                       # line i: v(i, j) for all j
    eh = erased[:, : L * L].reshape(B, L, L)
    ev = erased[:, L * L:].reshape(B, L, L)
    x = np.stack([s, eh, ev], axis=-1).astype(np.float32)
    return x, P0.astype(np.float32), P1.astype(np.float32)


def make_data(L: int, p: float, e: float, shots: int, seed: int):
    """Sample errors, take MWPM's correction as the reference chain, and label each direction
    by whether error + correction winds (t = 1: MWPM's class is right)."""
    code = ToricCode(L)
    H = code.check_matrix()
    rng = np.random.default_rng(seed)
    erased, err = sample_errors(code.n_qubits, p, e, shots, rng)
    syn = (err.astype(np.int64) @ H.T.toarray()) % 2
    C = ErasureMWPM(H).decode_batch(syn, erased).astype(np.uint8)
    wind = ((err ^ C) @ code.logical_z().T) % 2             # (shots, 2): 1 = logical error
    x, P0, P1 = features(C, erased, H, L)
    return dict(x=x, P0=P0, P1=P1, t=(1 - wind).astype(np.float32), C=C, erased=erased, H=H)


# ---------------- model ----------------

def init_params(key, width: int = 32, depth: int = 6, c_in: int = 3, residual: bool = False):
    """residual=True adds two output channels for a gauge-invariant bias B_x (see logits)."""
    params = []
    c = c_in
    for d in range(depth):
        key, k = jax.random.split(key)
        c_out = (4 if residual else 2) if d == depth - 1 else width
        w = jax.random.normal(k, (3, 3, c, c_out)) * jnp.sqrt(2.0 / (9 * c))
        params.append((w, jnp.zeros(c_out)))
        c = c_out
    return params


def conv_circular(x, w, b):
    xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")     # periodic: the torus
    y = jax.lax.conv_general_dilated(xp, w, (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + b


def logits(params, x, P0, P1):
    """(B, 2) logits; logit_x > 0: keep the reference class in direction x.

    Covariant mode (2 output channels): logit_x = mean_l P_x(l) G_x(l), exactly odd under adding
    a logical to the reference, so the answer does not depend on which representative is given;
    the price is that the net has to decode from the syndrome alone.
    Residual mode (4 channels): logit_x = B_x + mean_l P_x(l) G_x(l), with B_x the global mean of a
    gauge-invariant map. Still exactly gauge invariant, but no longer odd: it can express "trust
    the reference (MWPM) unless the syndrome says otherwise"."""
    h = x
    for w, b in params[:-1]:
        h = jax.nn.gelu(conv_circular(h, w, b))
    g = conv_circular(h, *params[-1])                       # (B, L, L, 2 or 4)
    G0 = g[..., 0].mean(axis=1)                             # average along line j (over i): (B, L)
    G1 = g[..., 1].mean(axis=2)                             # average along line i (over j): (B, L)
    z = jnp.stack([(P0 * G0).mean(axis=1), (P1 * G1).mean(axis=1)], axis=1)
    if g.shape[-1] == 4:
        z = z + g[..., 2:].mean(axis=(1, 2))
    return z


def loss_fn(params, batch):
    z = logits(params, batch["x"], batch["P0"], batch["P1"])
    return optax.sigmoid_binary_cross_entropy(z, batch["t"]).mean()


# ---------------- training / evaluation ----------------

def train(L=5, p=0.10, e=0.1, steps=6000, batch=256, lr=2e-3, seed=0, log=500, residual=False,
          refresh=100, n_train=None):
    """Adam + cosine schedule on BCE. Data are streamed: a fresh pool of refresh*batch shots is
    sampled every `refresh` steps, so each shot is seen about once (a fixed training set of a few
    1e5 shots overfits: train accuracy beats MWPM, fresh-data accuracy does not).
    n_train: if given, use a fixed training set of that size instead (for comparison)."""
    t0 = time.time()
    rng = np.random.default_rng(seed + 1)
    keys = ("x", "P0", "P1", "t")
    params = init_params(jax.random.PRNGKey(seed), residual=residual)
    sched = optax.cosine_decay_schedule(lr, steps)
    opt = optax.adam(sched)
    state = opt.init(params)

    @jax.jit
    def step(params, state, b):
        l, g = jax.value_and_grad(loss_fn)(params, b)
        u, state = opt.update(g, state, params)
        return optax.apply_updates(params, u), state, l

    fixed = make_data(L, p, e, n_train, seed) if n_train else None
    pool, n_shots, running = fixed, 0, []
    for k in range(steps):
        if fixed is None and k % refresh == 0:
            pool = make_data(L, p, e, refresh * batch, seed + 1000 + k)
            n_shots += refresh * batch
            perm = rng.permutation(refresh * batch)
        if fixed is None:
            idx = perm[(k % refresh) * batch:(k % refresh + 1) * batch]
        else:
            idx = rng.integers(0, n_train, batch)
        params, state, l = step(params, state, {n: jnp.asarray(pool[n][idx]) for n in keys})
        running.append(float(l))
        if k % log == 0 or k == steps - 1:
            print(f"step {k:5d}  loss (mean of last {len(running)}) {np.mean(running):.4f}  "
                  f"shots seen {n_shots or n_train}  {time.time() - t0:.0f}s", flush=True)
            running = []
    return params


def evaluate(params, L=5, p=0.10, e=0.1, shots=50_000, seed=99):
    d = make_data(L, p, e, shots, seed)
    f = jax.jit(logits)
    z = np.concatenate([np.asarray(f(params, jnp.asarray(d["x"][k:k + 4096]), jnp.asarray(d["P0"][k:k + 4096]),
                                     jnp.asarray(d["P1"][k:k + 4096]))) for k in range(0, shots, 4096)])
    keep = z > 0                                            # network: keep or flip MWPM's class
    mwpm_fail = (d["t"].min(axis=1) == 0).mean()
    net_fail = ((keep == (d["t"] == 1)).min(axis=1) == 0).mean()
    se = lambda r: np.sqrt(r * (1 - r) / shots)
    print(f"L={L} p={p} e={e}: logical failure  MWPM {mwpm_fail:.4f} +- {se(mwpm_fail):.4f}   "
          f"MWPM + gauge-covariant net {net_fail:.4f} +- {se(net_fail):.4f}")
    return mwpm_fail, net_fail


if __name__ == "__main__":
    params = train()
    evaluate(params)
