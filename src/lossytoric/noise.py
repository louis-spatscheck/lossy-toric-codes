"""Code-capacity noise: bit flips at rate p plus erasures (heralded loss) at rate e.

An erased qubit is replaced by a maximally mixed state. In the X sector this
means an X (or Y) error with probability 1/2, and the decoder knows *where*
the erasure happened (that is the whole point of erasure: the loss is heralded,
e.g. by a missing photon detection).
"""
from __future__ import annotations

import numpy as np


def sample_errors(n_qubits: int, p: float, e: float, shots: int, rng: np.random.Generator):
    """Return (erased, x_err), both bool arrays of shape (shots, n_qubits)."""
    if not (0.0 <= p <= 1.0 and 0.0 <= e <= 1.0):
        raise ValueError("p and e must be probabilities")
    erased = rng.random((shots, n_qubits)) < e
    flip_prob = np.where(erased, 0.5, p)
    x_err = rng.random((shots, n_qubits)) < flip_prob
    return erased, x_err
