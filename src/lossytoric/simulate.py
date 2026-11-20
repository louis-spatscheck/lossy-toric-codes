"""Monte Carlo estimate of the logical failure rate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decoder import ErasureMWPM
from .noise import sample_errors
from .toric import ToricCode


@dataclass
class Result:
    L: int
    p: float
    e: float
    shots: int
    failures: int

    @property
    def rate(self) -> float:
        return self.failures / self.shots

    def wilson(self, z: float = 1.0) -> tuple[float, float]:
        """Wilson score interval (default z=1, i.e. ~68%)."""
        n, k = self.shots, self.failures
        phat = k / n
        denom = 1 + z * z / n
        centre = (phat + z * z / (2 * n)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
        return centre - half, centre + half


def syndrome(H, x_err: np.ndarray) -> np.ndarray:
    """s = H e mod 2 for a batch of errors (shots x n_qubits) -> (shots x n_checks)."""
    s = H @ x_err.T.astype(np.int32)
    return (np.asarray(s).T % 2).astype(np.uint8)


def logical_failures(code: ToricCode, x_err: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """Bool per shot: residual error winds around the torus (either direction)."""
    residual = (x_err.astype(np.uint8) ^ correction.astype(np.uint8))
    parities = (residual @ code.logical_z().T) % 2
    return parities.any(axis=1)


def run(L: int, p: float, e: float, shots: int, seed: int = 0, batch: int = 2000) -> Result:
    code = ToricCode(L)
    H = code.check_matrix()
    dec = ErasureMWPM(H)
    rng = np.random.default_rng(seed)
    fails = 0
    done = 0
    while done < shots:
        b = min(batch, shots - done)
        erased, x_err = sample_errors(code.n_qubits, p, e, b, rng)
        syn = syndrome(H, x_err)
        corr = dec.decode_batch(syn, erased if e > 0 else None)
        fails += int(logical_failures(code, x_err, corr).sum())
        done += b
    return Result(L, p, e, shots, fails)
