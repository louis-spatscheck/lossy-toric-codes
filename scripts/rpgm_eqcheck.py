"""Equilibration check for the gauge-model Monte Carlo: planted (all links +1) vs hot (random) start.

Polyakov loops are not invariant under the Nishimori gauge symmetry, so the planted start is
biased upwards (too ordered) and the hot start downwards until the chain has decorrelated:
agreement of xi_P/L within errors is a two-sided check. Run it before trusting a new L,
p range or erasure rate.

python scripts/rpgm_eqcheck.py --L 6 --p 0.033 --samples 100
python scripts/rpgm_eqcheck.py --L 8 --p 0.033 --therm 1000 2000 4000   # scan the length
"""
from __future__ import annotations

import argparse

import numpy as np
from rpgm_sweep import BETA_MIN, MEAS, N_TEMPS, THERM

from lossytoric.rpgm import jackknife, simulate, xi_over_L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--q", type=float, default=None, help="default q = p")
    ap.add_argument("--e", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--therm", type=int, nargs="+", default=None, help="default: THERM[L] from rpgm_sweep.py")
    ap.add_argument("--temps", type=int, default=None, help="default: N_TEMPS[L]")
    ap.add_argument("--beta-min", type=float, default=BETA_MIN)
    a = ap.parse_args()
    nts = a.therm or [THERM[a.L]]
    temps = a.temps or N_TEMPS.get(a.L, 10)
    for nt in nts:
        res = {}
        for start in ("planted", "hot"):
            out = simulate(a.L, a.p, a.samples, n_therm=nt, n_meas=MEAS.get(a.L, 1000), q=a.q, e=a.e,
                           seed=5, start=start, n_temps=temps, beta_min=a.beta_min)
            res[start] = jackknife(lambda m2, mk2: xi_over_L(m2, mk2, a.L), out["ps2"], out["psk2"])
            print(f"  {start:8s} xi_P/L = {res[start][0]:.3f} +- {res[start][1]:.3f}   "
                  f"PT swap rates {np.round(out['swap_rate'], 2).tolist()}")
        (x, dx), (y, dy) = res["planted"], res["hot"]
        z = abs(x - y) / np.hypot(dx, dy)
        print(f"L={a.L} p={a.p} e={a.e} therm={nt}: difference {z:.1f} sigma -> "
              f"{'OK' if z < 2 else 'NOT EQUILIBRATED'}", flush=True)


if __name__ == "__main__":
    main()
