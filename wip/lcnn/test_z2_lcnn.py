"""Exact symmetry checks of the minimal gauge-covariant decoder (random, untrained weights)."""
import jax
import jax.numpy as jnp
import numpy as np
from z2_lcnn import features, init_params, logical_x, logits, make_data

from lossytoric.toric import ToricCode


def _out(params, C, erased, H, L):
    x, P0, P1 = features(C, erased, H, L)
    return np.asarray(logits(params, jnp.asarray(x), jnp.asarray(P0), jnp.asarray(P1)))


def test_symmetries():
    L = 5
    d = make_data(L, 0.12, 0.1, 64, seed=3)
    params = init_params(jax.random.PRNGKey(0))
    C, er, H = d["C"], d["erased"], d["H"]
    z = _out(params, C, er, H, L)
    rng = np.random.default_rng(4)
    S = ToricCode(L).plaquette_x()

    # 1) gauge invariance: add random stabilizers to the reference chain
    C2 = (C + (rng.integers(0, 2, (len(C), len(S))) @ S)) % 2
    assert (C2 != C).any()
    assert np.allclose(_out(params, C2.astype(np.uint8), er, H, L), z, atol=1e-6)

    # 2) logical covariance: adding the winding loop of direction x flips logit_x only
    lx = logical_x(L)
    for k in range(2):
        z2 = _out(params, ((C + lx[k]) % 2).astype(np.uint8), er, H, L)
        assert np.allclose(z2[:, k], -z[:, k], atol=1e-6)
        assert np.allclose(z2[:, 1 - k], z[:, 1 - k], atol=1e-6)

    # 3) translation equivariance: shifting the whole configuration leaves the output unchanged
    def shift(v):                                    # (B, 2L^2) chain/mask, shift i -> i + 2
        h, w = v[:, : L * L].reshape(-1, L, L), v[:, L * L:].reshape(-1, L, L)
        return np.concatenate([np.roll(h, 2, 1).reshape(-1, L * L), np.roll(w, 2, 1).reshape(-1, L * L)], 1)
    assert np.allclose(_out(params, shift(C), shift(er), H, L), z, atol=1e-5)


def test_residual_mode_is_gauge_invariant_but_not_odd():
    L = 5
    d = make_data(L, 0.12, 0.1, 32, seed=6)
    params = init_params(jax.random.PRNGKey(1), residual=True)
    C, er, H = d["C"], d["erased"], d["H"]
    z = _out(params, C, er, H, L)
    S = ToricCode(L).plaquette_x()
    C2 = (C + (np.random.default_rng(7).integers(0, 2, (len(C), len(S))) @ S)) % 2
    assert np.allclose(_out(params, C2.astype(np.uint8), er, H, L), z, atol=1e-6)
    z3 = _out(params, ((C + logical_x(L)[0]) % 2).astype(np.uint8), er, H, L)
    assert not np.allclose(z3[:, 0], -z[:, 0], atol=1e-3)


def test_polyakov_lines_match_logical_z():
    """P0 at column 0 and P1 at row 0 are exactly the repo's logical_z() rows."""
    L = 5
    d = make_data(L, 0.12, 0.0, 32, seed=5)
    code = ToricCode(L)
    _, P0, P1 = features(d["C"], d["erased"], d["H"], L)
    lz = (d["C"].astype(int) @ code.logical_z().T) % 2
    assert np.array_equal(P0[:, 0], 1 - 2 * lz[:, 0])
    assert np.array_equal(P1[:, 0], 1 - 2 * lz[:, 1])
