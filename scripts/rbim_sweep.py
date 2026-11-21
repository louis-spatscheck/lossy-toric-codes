"""Monte Carlo of the random-bond Ising model on the Nishimori line over a (L, p, e) grid.

Each job simulates R_chunk disorder samples with heat bath + parallel tempering and
saves the per-sample thermal averages to data/rbim/L{L}_p{p}_e{e}_s{seed}.npz, so
errors can later be jackknifed over disorder samples. Re-running with new --seed
values adds samples.

Example (optimal threshold without erasure, about 25 min on 2 cores):
python scripts/rbim_sweep.py --L 8 12 16 20 --p 0.102 0.117 6 --e 0 --samples 1000

Equilibration: the number of thermalisation sweeps per L (THERM below) was chosen
so that planted (all-up) and hot (random) starts agree within errors at p ~ p_c;
check again with scripts/rbim_eqcheck.py before trusting new L or e.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lossytoric.rbim import simulate

ROOT = Path(__file__).resolve().parents[1]
THERM = {8: 300, 12: 500, 16: 1000, 20: 1500, 24: 3000}   # thermalisation sweeps (checked)
MEAS = {8: 400, 12: 500, 16: 500, 20: 500, 24: 600}       # measurement sweeps
N_TEMPS, BETA_MIN = 8, 0.6


def parse_range(vals):
    if len(vals) == 1:
        return [float(vals[0])]
    return [round(float(x), 6) for x in np.linspace(vals[0], vals[1], int(vals[2]))]


def job(args):
    L, p, e, R, seed, outdir = args
    t = time.time()
    out = simulate(L, p, e, R, n_therm=THERM[L], n_meas=MEAS[L], seed=seed,
                   n_temps=N_TEMPS, beta_min=BETA_MIN)
    f = Path(outdir) / f"L{L}_p{p:.5f}_e{e:.3f}_s{seed}.npz"
    np.savez(f, L=L, p=p, e=e, **out)
    return L, p, e, R, time.time() - t, float(out["swap_rate"].min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", required=True)
    ap.add_argument("--p", type=float, nargs="+", required=True)
    ap.add_argument("--e", type=float, nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=1000, help="disorder samples per grid point")
    ap.add_argument("--chunk", type=int, default=250, help="samples per job (memory/parallelism)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "rbim")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for L in a.L:
        if L not in THERM:
            raise SystemExit(f"no checked thermalisation length for L={L}; add it to THERM")
        for p in parse_range(a.p):
            for e in parse_range(a.e):
                for c in range(a.samples // a.chunk):
                    seed = hash((L, round(p * 1e6), round(e * 1e6), a.seed, c)) % (2**32)
                    jobs.append((L, p, e, a.chunk, seed, str(a.out)))
    jobs.sort(key=lambda j: -j[0])          # big lattices first for load balance
    with ProcessPoolExecutor(a.workers) as ex:
        for L, p, e, R, dt, smin in ex.map(job, jobs):
            print(f"L={L:3d} p={p:.4f} e={e:.3f} R={R}  {dt:6.1f}s  min PT swap rate {smin:.2f}", flush=True)


if __name__ == "__main__":
    main()
