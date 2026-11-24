"""Optimal thresholds with faulty measurements: xi_P/L crossings of the random-plaquette gauge model.

python scripts/rpgm_analyze.py
Reads data/rpgm/*.npz (from rpgm_sweep.py), checks the Nishimori energy identity at every grid
point, fits the crossing of the Polyakov-loop correlation length xi_P/L per (q/p, e), and writes
results/rpgm_thresholds.csv and figures/rpgm_crossings.png.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from lossytoric.fss import fit_threshold_y, model, pairwise_crossings
from lossytoric.rpgm import jackknife, xi_over_L

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, RES = ROOT / "data" / "rpgm", ROOT / "figures", ROOT / "results"
L_COLORS = {4: "#86b6ef", 6: "#3987e5", 8: "#1c5cab", 10: "#0d366b", 12: "#08244a"}
INK, MUTED, GRID, SURFACE, LIT = "#1f1f1e", "#6b6a64", "#e6e5df", "#fcfcfb", "#c2410c"
P_OHNO = 0.033      # Ohno, Arakawa, Ichinose, Matsui, Nucl. Phys. B 697, 462 (2004): p = q, e = 0

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
})
KEYS = ("ps2", "psk2", "ps4", "pt2", "ptk2", "pt4", "e_space", "e_time")


def load():
    """{(L, p, q, e): dict of concatenated per-sample arrays}"""
    groups = defaultdict(lambda: defaultdict(list))
    for f in sorted(DATA.glob("*.npz")):
        d = np.load(f)
        key = (int(d["L"]), round(float(d["p"]), 6), round(float(d["q"]), 6), round(float(d["e"]), 6))
        for k in KEYS:
            groups[key][k].append(d[k])
    return {k: {n: np.concatenate(v) for n, v in g.items()} for k, g in groups.items()}


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    data = load()
    if not data:
        raise SystemExit(f"no data in {DATA}; run scripts/rpgm_sweep.py first")

    # 1) exact check on every grid point: Nishimori energy identity, timelike and spacelike
    worst = 0.0
    for (L, p, q, e), d in data.items():
        N = L ** 3
        for obs, expect in ((d["e_time"] / (2 * N), (1 - e) * (1 - 2 * p)), (d["e_space"] / N, 1 - 2 * q)):
            worst = max(worst, abs(obs.mean() - expect) / (obs.std() / np.sqrt(len(obs))))
    print(f"Nishimori energy identity: worst deviation over {len(data)} grid points = {worst:.1f} sigma")

    # 2) xi_P/L (space-direction Polyakov loops) per grid point, FSS fit per (q/p, e)
    table = defaultdict(list)          # (q/p, e) -> rows (L, p, xi/L, err, n)
    for (L, p, q, e), d in sorted(data.items()):
        v, dv = jackknife(lambda a, b: xi_over_L(a, b, L), d["ps2"], d["psk2"])
        table[(round(q / p, 3), e)].append((L, p, v, dv, len(d["ps2"])))

    results = []
    for (r, e), rows in sorted(table.items()):
        L, p, y, dy, n = map(np.array, zip(*rows))
        xs = [x for *_, x in pairwise_crossings(p, L, y)]
        if len(set(L)) < 3:
            print(f"q/p={r}, e={e}: fewer than 3 sizes, pairwise crossings only: {np.round(xs, 4).tolist()}")
            continue
        fit = fit_threshold_y(p, L, y, dy, q0=float(np.median(p)), nu0=1.0)
        fit.sys = (max(xs) - min(xs)) / 2 if len(xs) >= 2 else float("nan")
        results.append(((r, e), fit, (L, p, y, dy), xs))
        print(f"q/p={r}, e={e:.2f}: p_c = {fit.qc:.4f} +- {fit.qc_err:.4f} (stat) +- {fit.sys:.4f} (sys), "
              f"nu = {fit.nu:.2f} +- {fit.nu_err:.2f}, chi2/dof = {fit.chi2_dof:.2f}, "
              f"pairwise crossings {np.round(xs, 4).tolist()}, samples/point {int(n.min())}")

    with open(RES / "rpgm_thresholds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["q_over_p", "e", "p_c_optimal", "stat_err", "sys_err", "nu", "nu_err", "chi2_dof"])
        for (r, e), fit, *_ in results:
            w.writerow([r, e, f"{fit.qc:.5f}", f"{fit.qc_err:.5f}", f"{fit.sys:.5f}",
                        f"{fit.nu:.3f}", f"{fit.nu_err:.3f}", f"{fit.chi2_dof:.2f}"])

    panels = results or [(key, None, tuple(map(np.array, zip(*rows)))[:4], []) for key, rows in sorted(table.items())]
    n = len(panels)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(3.9 * ncol, 6.4), 3.3 * nrow + 0.6), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, ((r, e), fit, (L, p, y, dy), _) in zip(axes.flat, panels):
        pp = np.linspace(p.min(), p.max(), 200)
        for Lv in sorted(set(L)):
            m = L == Lv
            c = L_COLORS.get(int(Lv), INK)
            ax.errorbar(p[m], y[m], yerr=dy[m], fmt="o", ms=4, color=c, mec=SURFACE, mew=1,
                        elinewidth=1, capsize=0, label=f"L = {Lv}", zorder=3)
            if fit is not None:
                ax.plot(pp, model(pp, np.full_like(pp, Lv), fit.params), color=c, lw=1.5, zorder=2)
            else:
                ax.plot(p[m], y[m], color=c, lw=1, zorder=2)
        title = f"q/p = {r:g}, e = {e:g}"
        if fit is not None:
            ax.axvline(fit.qc, color=MUTED, lw=1, ls="--")
            err = np.hypot(fit.qc_err, fit.sys) if np.isfinite(fit.sys) else fit.qc_err
            title += f":  p_c = {fit.qc:.4f} ± {err:.4f}"
        if r == 1 and e == 0:
            ax.axvline(P_OHNO, color=LIT, lw=1, ls=":")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("error rate p (Nishimori line)")
        ax.set_title(title, loc="left", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Polyakov correlation length $\xi_P/L$")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Random-plaquette gauge model on the Nishimori line: xi_P/L crossings\n"
                 "dashed: fitted p_c;  dotted (q = p, e = 0): Ohno et al. 2004, p_c ~ 0.033",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "rpgm_crossings.png", dpi=160)
    print("figure written")


if __name__ == "__main__":
    main()
