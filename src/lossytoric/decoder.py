"""Minimum-weight perfect matching (PyMatching) with erasure information.

Edge weights are log-likelihood ratios w = log((1 - q)/q) with q the flip
probability of that qubit:
  * non-erased qubit: q = p      -> w > 0
  * erased qubit:     q = 1/2    -> w = 0  (an erased edge is 'free' to use)

With only these two classes, MWPM depends on p only through an overall scale
of the non-zero weights, so we use weight 1 for non-erased edges. Without
erasures a single Matching object is reused and shots are decoded in batch.
With erasures the weights change every shot, so a Matching is rebuilt per shot
(PyMatching has no per-shot weight API); this is the main cost of the code.
"""
from __future__ import annotations

import numpy as np
import pymatching
import scipy.sparse as sp


class ErasureMWPM:
    def __init__(self, H: sp.csc_matrix):
        self.H = sp.csc_matrix(H)
        self.n = self.H.shape[1]
        self._static = pymatching.Matching.from_check_matrix(self.H, weights=np.ones(self.n))

    def decode_batch(self, syndromes: np.ndarray, erased: np.ndarray | None = None) -> np.ndarray:
        """Return corrections (shots x n_qubits, uint8)."""
        if erased is None or not erased.any():
            return self._static.decode_batch(syndromes.astype(np.uint8))
        out = np.empty((syndromes.shape[0], self.n), dtype=np.uint8)
        for k in range(syndromes.shape[0]):
            if not erased[k].any():
                out[k] = self._static.decode(syndromes[k])
                continue
            w = np.where(erased[k], 0.0, 1.0)
            m = pymatching.Matching.from_check_matrix(self.H, weights=w)
            out[k] = m.decode(syndromes[k])
        return out
