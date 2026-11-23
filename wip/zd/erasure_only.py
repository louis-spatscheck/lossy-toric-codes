"""Z_d toric code, pure erasure, optimal decoding (X sector).

Given the erased set E, an erased qudit carries a uniformly random shift, so the
optimal decoder can only guess the logical class within the k-dimensional space
of homology classes supported on E:  P(fail | E) = 1 - d^(-k),
k = rank_d of the windings of the cycles inside E.

So instead of sampling errors I average 1 - d^-k over erasure sets
(conditional Monte Carlo: same mean, lower variance). k is computed for d=2 and d=3 on the *same* E, which also checks
whether k is ever d-dependent (scratch_checks.py).

Expectation: threshold at e = 1/2 for every d (bond percolation), but above
threshold P_fail grows with d. Cross-check: d=2 must match Phase 1 MWPM data
(data/erasure_only.csv), since for pure erasure every consistent correction
is equally good.

Run: python wip/zd/erasure_only.py   (~ a few min on 2 cores)
"""
import argparse, pathlib, time
import numpy as np
from gfp import nullspace, rank
from zd_lattice import incidence, winding

ROOT = pathlib.Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, nargs="+", default=[8, 12, 16])
ap.add_argument("--e", type=float, nargs=3, default=[0.44, 0.56, 7])
ap.add_argument("--samples", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("erasure_only_zd.csv")))
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
es = np.linspace(args.e[0], args.e[1], int(args.e[2]))
rows = []
for L in args.L:
    D = {d: incidence(L, d) for d in (2, 3)}
    W = winding(L)
    t0 = time.time()
    for e in es:
        k = {2: [], 3: []}
        for _ in range(args.samples):
            E = np.nonzero(rng.random(2 * L * L) < e)[0]
            for d in (2, 3):
                N = nullspace(D[d][:, E], d)
                k[d].append(rank(W[:, E] @ N.T, d) if len(N) else 0)
        k2, k3 = np.array(k[2]), np.array(k[3])
        for d, kd in ((2, k2), (3, k3)):
            pf = 1 - float(d) ** (-kd)
            rows.append((L, d, e, args.samples, pf.mean(), pf.std(ddof=1) / np.sqrt(len(pf)),
                         int((k2 != k3).sum())))
    print(f"L={L}: {time.time() - t0:.0f} s")

with open(args.out, "w") as f:
    f.write("L,d,e,samples,p_fail,err,k_mismatch\n")
    for r in rows:
        f.write("%d,%d,%.4f,%d,%.5f,%.5f,%d\n" % r)
print("wrote", args.out)
