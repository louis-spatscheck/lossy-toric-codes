"""Noise for the Z_d toric code. Only erasure is done.

Open question (notes.md #1): which shift-error model? For qubits "bit flip with
prob p" is unambiguous. For qudits I need to choose, and the threshold depends on it.
"""
import numpy as np


def sample_erasure(n, e, rng):
    """Boolean mask of erased qudits. An erased qudit gets a uniform shift in Z_d."""
    return rng.random(n) < e


def sample_shifts(n, p, d, rng, model="symmetric"):
    """Shift errors X^a on non-erased qudits.

    model="symmetric": a uniform over {1..d-1} with total prob p. Standard in
        Duclos-Cianci & Poulin; makes the stat-mech model a d-state Potts model
        on the Nishimori line.
    model="adjacent":  a = +-1 only. Closer to a clock model. Which model fits
        photonic qudit noise is open (notes.md #1).
    """
    if model == "symmetric":
        hit = rng.random(n) < p
        return np.where(hit, rng.integers(1, d, n), 0)
    raise NotImplementedError(model)


def decode(syndrome, erased, d):
    """Placeholder. PyMatching is qubit-only: for d = 3 two +1 charges cannot
    annihilate each other, but three can (1+1+1 = 0 mod 3), so this is no longer
    a matching problem. Options in notes.md #2."""
    raise NotImplementedError
