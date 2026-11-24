"""Clean limit (p = q = e = 0) of the gauge-model code: the pure 3D Z2 lattice gauge theory.

Its deconfinement transition is exactly dual to the 3D Ising model:
beta_c(gauge) = -1/2 ln tanh(beta_c(Ising)), with beta_c(Ising) = 0.2216546 -> 0.7614.
The Polyakov correlation length xi_P/L for different L should cross there. No disorder, so
the R samples are just independent chains.

python scripts/rpgm_clean_check.py
"""
from __future__ import annotations

import numpy as np

from lossytoric.fss import pairwise_crossings
from lossytoric.rpgm import jackknife, simulate, xi_over_L

BETA_C = -0.5 * np.log(np.tanh(0.2216546))


def main():
    betas = np.round(np.linspace(0.73, 0.79, 5), 4)
    Ls, rows = (4, 6, 8), []
    for L in Ls:
        for b in betas:
            out = simulate(L, 0.0, 24, n_therm=400, n_meas=800, beta=float(b), seed=int(1000 * b) + L)
            v, dv = jackknife(lambda a, c: xi_over_L(a, c, L), out["ps2"], out["psk2"])
            rows.append((L, b, v, dv))
            print(f"L={L} beta={b:.4f}  xi_P/L = {v:.3f} +- {dv:.3f}", flush=True)
    L, b, y, _ = map(np.array, zip(*rows))
    xs = pairwise_crossings(b, L, y)
    print("pairwise crossings:", [(a, c, round(x, 4)) for a, c, x in xs], f" exact beta_c = {BETA_C:.4f}")


if __name__ == "__main__":
    main()
