"""Equilibration check for the RBIM Monte Carlo: planted (all-up) vs hot (random) start.

The planted start over-estimates xi/L until the chain has decorrelated, the hot start
under-estimates it, so agreement within errors is a two-sided check.

python scripts/rbim_eqcheck.py --L 16 --p 0.108 --e 0 --samples 150
"""
from __future__ import annotations

import argparse

from rbim_sweep import BETA_MIN, MEAS, N_TEMPS, THERM

from lossytoric.rbim import jackknife, simulate, xi_over_L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--e", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--therm", type=int, default=None, help="default: THERM[L] from rbim_sweep.py")
    a = ap.parse_args()
    nt = a.therm or THERM[a.L]
    res = {}
    for start in ("planted", "hot"):
        out = simulate(a.L, a.p, a.e, a.samples, n_therm=nt, n_meas=MEAS.get(a.L, 500), seed=5,
                       start=start, n_temps=N_TEMPS, beta_min=BETA_MIN)
        res[start] = jackknife(lambda m2, mk2: xi_over_L(m2, mk2, a.L), out["m2"], out["mk2"])
        print(f"{start:8s} xi/L = {res[start][0]:.3f} +- {res[start][1]:.3f}")
    (x, dx), (y, dy) = res["planted"], res["hot"]
    z = abs(x - y) / (dx * dx + dy * dy) ** 0.5
    print(f"difference = {z:.1f} sigma -> {'OK' if z < 2 else 'NOT EQUILIBRATED'} (L={a.L}, {nt} sweeps)")


if __name__ == "__main__":
    main()
