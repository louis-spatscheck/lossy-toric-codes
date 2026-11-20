"""Sweep logical failure rates over a (L, p, e) grid and append to a CSV.

Examples
--------
# bit-flip only, around the MWPM threshold
python scripts/sweep.py --L 8 12 16 24 --p 0.09 0.11 9 --e 0 --shots 20000 --out data/e0.csv
# erasure only (p = 0), around the 50% percolation threshold
python scripts/sweep.py --L 8 12 16 24 --p 0 --e 0.44 0.56 7 --shots 10000 --out data/p0.csv

A range is given as  START STOP NUM  (like numpy.linspace); a single number is a fixed value.
Rows are appended, so re-running with a new seed adds statistics (the fit script sums them).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lossytoric import run


def parse_range(vals: list[float]) -> list[float]:
    if len(vals) == 1:
        return [float(vals[0])]
    if len(vals) == 3:
        return [round(float(x), 6) for x in np.linspace(vals[0], vals[1], int(vals[2]))]
    raise SystemExit("give either one value or START STOP NUM")


def job(args):
    L, p, e, shots, seed = args
    t = time.time()
    r = run(L, p, e, shots, seed=seed)
    return L, p, e, r.shots, r.failures, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", required=True)
    ap.add_argument("--p", type=float, nargs="+", required=True)
    ap.add_argument("--e", type=float, nargs="+", required=True)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ps, es = parse_range(a.p), parse_range(a.e)
    jobs = []
    for L in a.L:
        for p in ps:
            for e in es:
                # distinct, reproducible seed per grid point and per --seed
                seed = hash((L, round(p * 1e6), round(e * 1e6), a.seed)) % (2**32)
                jobs.append((L, p, e, a.shots, seed))
    # largest (slowest) jobs first for better load balance
    jobs.sort(key=lambda j: -j[0])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    with open(a.out, "a", newline="") as f, ProcessPoolExecutor(a.workers) as ex:
        w = csv.writer(f)
        if new:
            w.writerow(["L", "p", "e", "shots", "failures"])
        for L, p, e, n, k, dt in ex.map(job, jobs):
            w.writerow([L, p, e, n, k])
            f.flush()
            print(f"L={L:3d} p={p:.4f} e={e:.4f}  P_L={k / n:.4f}  ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
