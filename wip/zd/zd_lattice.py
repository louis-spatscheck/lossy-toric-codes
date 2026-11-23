"""Z_d toric code on an L x L torus, X-type (shift) sector only for now.

Same edge/vertex indexing as src/lossytoric/toric.py, but edges are now
*oriented*, because for d > 2 the sign matters:
    h(i, j): (i, j) -> (i, j+1)
    v(i, j): (i, j) -> (i+1, j)
An X^a error on an edge adds +a to the check at its head and -a at its tail
(mod d). For d = 2 this reduces to the qubit incidence matrix.

TODO: Z-type sector on the dual lattice (needed for the full failure rate).
"""
import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from lossytoric.toric import h_edge, v_edge, vertex  # reuse the qubit indexing


def incidence(L, d):
    """Signed vertex-edge incidence D (L^2 x 2L^2), entries mod d."""
    D = np.zeros((L * L, 2 * L * L), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            e = h_edge(i, j, L)
            D[vertex(i, j + 1, L), e] += 1
            D[vertex(i, j, L), e] -= 1
            e = v_edge(i, j, L)
            D[vertex(i + 1, j, L), e] += 1
            D[vertex(i, j, L), e] -= 1
    return D % d


def winding(L):
    """2 x 2L^2. Row 0 = net flux through the cut at column 0 (winding in j),
    row 1 = net flux through the cut at row 0 (winding in i). Same supports as
    logical_z() in the qubit code, all signs + because of the orientation."""
    W = np.zeros((2, 2 * L * L), dtype=np.int64)
    for i in range(L):
        W[0, h_edge(i, 0, L)] = 1
    for j in range(L):
        W[1, v_edge(0, j, L)] = 1
    return W


def plaquettes(L, d):
    """Signed plaquette cycles (counter-clockwise), for sanity checks: D @ P.T = 0."""
    P = np.zeros((L * L, 2 * L * L), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            r = i * L + j
            P[r, h_edge(i, j, L)] += 1      # (i,j) -> (i,j+1)
            P[r, v_edge(i, j + 1, L)] += 1  # (i,j+1) -> (i+1,j+1)
            P[r, h_edge(i + 1, j, L)] -= 1  # back along the top
            P[r, v_edge(i, j, L)] -= 1      # back down
    return P % d
