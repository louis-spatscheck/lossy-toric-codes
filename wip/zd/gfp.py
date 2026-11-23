"""Small linear algebra over GF(p), p prime. Dense numpy, fine for L <= ~24.

Only works for prime d. For Z_4 (not a field) I'd need Smith normal form
instead of Gaussian elimination -- not done yet, see notes.md.
"""
import numpy as np


def row_reduce(A, p):
    """Reduced row echelon form of A mod p. Returns (R, pivot_columns)."""
    R = np.array(A, dtype=np.int64) % p
    m, n = R.shape
    pivots = []
    r = 0
    for c in range(n):
        if r == m:
            break
        nz = np.nonzero(R[r:, c])[0]
        if nz.size == 0:
            continue
        k = r + nz[0]
        R[[r, k]] = R[[k, r]]
        R[r] = (R[r] * pow(int(R[r, c]), -1, p)) % p
        col = R[:, c].copy()
        col[r] = 0
        R = (R - np.outer(col, R[r])) % p
        pivots.append(c)
        r += 1
    return R[:r], pivots


def rank(A, p):
    if A.size == 0:
        return 0
    return len(row_reduce(A, p)[1])


def nullspace(A, p):
    """Basis of {x : A x = 0 mod p}, as rows."""
    m, n = A.shape
    if m == 0:
        return np.eye(n, dtype=np.int64)
    R, piv = row_reduce(A, p)
    free = [c for c in range(n) if c not in set(piv)]
    N = np.zeros((len(free), n), dtype=np.int64)
    for t, f in enumerate(free):
        N[t, f] = 1
        for i, pc in enumerate(piv):
            N[t, pc] = (-R[i, f]) % p
    return N
