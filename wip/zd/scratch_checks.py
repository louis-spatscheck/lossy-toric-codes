"""Sanity checks for the Z_d lattice; to move into tests/ once the decoder exists.
Run: python wip/zd/scratch_checks.py"""
import numpy as np
from gfp import nullspace, rank
from zd_lattice import incidence, winding, plaquettes
from lossytoric.toric import h_edge, v_edge

for d in (2, 3, 5):
    for L in (3, 4, 6):
        D, P, W = incidence(L, d), plaquettes(L, d), winding(L)
        assert np.all((D @ P.T) % d == 0), "plaquettes must be cycles"
        assert np.all((W @ P.T) % d == 0), "plaquettes must not wind"
        # cycle space of the torus graph: dim = E - V + 1 = L^2 + 1 (connected graph)
        assert nullspace(D, d).shape[0] == L * L + 1
        # homology: cycles mod plaquettes = Z_d^2
        assert rank(W @ nullspace(D, d).T, d) == 2
print("lattice checks ok")

# Is the erasure rank k ever d-dependent? Candidate: two cycles with windings
# (1,0) and (1,2) span Z^2 over Q but only a line over Z_2, suggesting k = 1 for
# d = 2 and k = 2 for d = 3. It fails: where the loops cross they share vertices,
# and the pieces form a third cycle with winding (0,1). Output: k = 2 for both.
L = 6
row = [h_edge(0, j, L) for j in range(L)]                      # winding (1, 0)
stair = []                                                     # winding (1, 2): 1 right, 2 up, repeat
i = j = 0
for _ in range(L):
    stair.append(h_edge(i, j, L)); j += 1
    stair.append(v_edge(i, j, L)); i += 1
    stair.append(v_edge(i, j, L)); i += 1
E = sorted(set(row) | set(stair))
for d in (2, 3):
    D, W = incidence(L, d), winding(L)
    N = nullspace(D[:, E], d)
    print(f"d={d}: k = {rank(W[:, E] @ N.T, d)}")
# -> k = 2 for both. Conjecture: for subgraphs of the torus the image of the cycle
#    space in H_1 is always saturated, so k is the same for every prime d.
#    Not proven -- erasure_only.py counts k_2 != k_3 on random samples as a check.
