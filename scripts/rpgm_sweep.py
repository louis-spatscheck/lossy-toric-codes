"""Monte Carlo of the random-plaquette Z2 gauge model on the Nishimori line over a (L, p, e) grid.

Optimal decoding with faulty syndrome measurements (phenomenological noise, q = p by default).
Each job simulates R_chunk disorder samples with heat bath + parallel tempering and saves the
per-sample thermal averages to data/rpgm/L{L}_p{p}_q{q}_e{e}_s{seed}.npz, so errors can later
be jackknifed over disorder samples. Re-running with new --seed values adds samples.

Example (known optimal threshold p_c ~ 0.033 at e = 0):
python scripts/rpgm_sweep.py --L 4 6 8 --p 0.026 0.040 8 --e 0 --samples 400

Equilibration: THERM below must be checked with scripts/rpgm_eqcheck.py (planted and hot
starts must agree) before trusting a new L, p range or e. 3D is much slower than the 2D RBIM.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lossytoric.rpgm import simulate

ROOT = Path(__file__).resolve().parents[1]
# thermalisation / measurement sweeps and PT ladder per L (see rpgm_eqcheck.py).
# Checked at p = 0.033, e = 0: at L = 8 planted and hot starts disagree by 10-15 sigma after
# 150 sweeps, by ~1.5 sigma after 500, and agree (0.1 sigma) after 1500 with 14 temperatures.
# Larger L or e > 0 need their own check.
THERM = {4: 500, 6: 1000, 8: 1500}
MEAS = {4: 500, 6: 500, 8: 500}
N_TEMPS = {4: 8, 6: 10, 8: 14}
BETA_MIN = 0.6           # hottest coupling; below the clean transition (beta ~ 0.76)


def parse_range(vals):
    if len(vals) == 1:
        return [float(vals[0])]
    return [round(float(x), 6) for x in np.linspace(vals[0], vals[1], int(vals[2]))]


def job(args):
    L, p, q, e, R, seed, outdir = args
    t = time.time()
    out = simulate(L, p, R, n_therm=THERM[L], n_meas=MEAS[L], q=q, e=e, seed=seed,
                   n_temps=N_TEMPS[L], beta_min=BETA_MIN)
    f = Path(outdir) / f"L{L}_p{p:.5f}_q{q:.5f}_e{e:.3f}_s{seed}.npz"
    np.savez(f, L=L, p=p, q=q, e=e, **out)
    return L, p, e, R, time.time() - t, float(out["swap_rate"].min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", required=True)
    ap.add_argument("--p", type=float, nargs="+", required=True, help="value, or start stop n")
    ap.add_argument("--e", type=float, nargs="+", default=[0.0])
    ap.add_argument("--q-over-p", type=float, default=1.0, help="readout error rate q = ratio * p")
    ap.add_argument("--samples", type=int, default=400, help="disorder samples per grid point")
    ap.add_argument("--chunk", type=int, default=50, help="samples per job (memory/parallelism)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "rpgm")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for L in a.L:
        if L not in THERM:
            raise SystemExit(f"no checked thermalisation length for L={L}; add it to THERM")
        for p in parse_range(a.p):
            q = round(a.q_over_p * p, 6)
            for e in parse_range(a.e):
                for c in range(a.samples // a.chunk):
                    seed = hash((L, round(p * 1e6), round(q * 1e6), round(e * 1e6), a.seed, c)) % (2**32)
                    jobs.append((L, p, q, e, a.chunk, seed, str(a.out)))
    jobs.sort(key=lambda j: -j[0])          # big lattices first for load balance
    with ProcessPoolExecutor(a.workers) as ex:
        for L, p, e, R, dt, smin in ex.map(job, jobs):
            print(f"L={L:3d} p={p:.4f} e={e:.3f} R={R}  {dt:6.1f}s  min PT swap rate {smin:.2f}", flush=True)


if __name__ == "__main__":
    main()
