"""Toric code on an L x L periodic square lattice: the bit-flip (X-error) sector.

Conventions
-----------
* Qubits live on edges, Z-type checks on vertices (each check is Z on the 4
  incident edges), so an X error on an edge flips the two vertex checks at its
  endpoints. Every column of H has exactly two ones, i.e. H is the incidence
  matrix of a graph: decoding is a matching problem.
* Vertex (i, j) has index i*L + j, with i the row and j the column.
* Horizontal edge h(i, j) joins (i, j) -- (i, j+1); index  i*L + j.
* Vertical   edge v(i, j) joins (i, j) -- (i+1, j); index  L*L + i*L + j.
* An undetectable X error is a cycle on the primal lattice. It is a logical
  error iff it winds around the torus. Winding is detected by two logical Z
  operators supported on dual non-contractible loops:
    - row 0 of Lz: h(i, 0) for all i  -> detects winding in the j-direction
    - row 1 of Lz: v(0, j) for all j  -> detects winding in the i-direction

The phase-flip sector is identical on the dual lattice, so for code-capacity
noise with independent X and Z errors it is enough to simulate one sector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def h_edge(i: int, j: int, L: int) -> int:
    return (i % L) * L + (j % L)


def v_edge(i: int, j: int, L: int) -> int:
    return L * L + (i % L) * L + (j % L)


def vertex(i: int, j: int, L: int) -> int:
    return (i % L) * L + (j % L)


@dataclass(frozen=True)
class ToricCode:
    L: int

    @property
    def n_qubits(self) -> int:
        return 2 * self.L * self.L

    @property
    def n_checks(self) -> int:
        return self.L * self.L

    def check_matrix(self) -> sp.csc_matrix:
        """Vertex-edge incidence matrix H (n_checks x n_qubits) over GF(2)."""
        L = self.L
        rows, cols = [], []
        for i in range(L):
            for j in range(L):
                e = h_edge(i, j, L)
                rows += [vertex(i, j, L), vertex(i, j + 1, L)]
                cols += [e, e]
                e = v_edge(i, j, L)
                rows += [vertex(i, j, L), vertex(i + 1, j, L)]
                cols += [e, e]
        data = np.ones(len(rows), dtype=np.uint8)
        return sp.csc_matrix((data, (rows, cols)), shape=(self.n_checks, self.n_qubits))

    def logical_z(self) -> np.ndarray:
        """2 x n_qubits matrix of logical-Z supports (rows detect the two windings)."""
        L = self.L
        Lz = np.zeros((2, self.n_qubits), dtype=np.uint8)
        for i in range(L):
            Lz[0, h_edge(i, 0, L)] = 1
        for j in range(L):
            Lz[1, v_edge(0, j, L)] = 1
        return Lz

    def plaquette_x(self) -> np.ndarray:
        """X-type plaquette stabilizers (L*L x n_qubits): trivial cycles. Used in tests."""
        L = self.L
        P = np.zeros((L * L, self.n_qubits), dtype=np.uint8)
        for i in range(L):
            for j in range(L):
                r = i * L + j
                for e in (h_edge(i, j, L), h_edge(i + 1, j, L), v_edge(i, j, L), v_edge(i, j + 1, L)):
                    P[r, e] = 1
        return P
