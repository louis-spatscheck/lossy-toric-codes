"""Optimal thresholds from the RBIM Monte Carlo: xi/L crossings on the Nishimori line.

python scripts/rbim_analyze.py
Reads data/rbim/*.npz (from rbim_sweep.py), writes results/rbim_thresholds.csv,
figures/rbim_crossings.png and figures/tradeoff_optimal_vs_mwpm.png.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from lossytoric.fss import fit_threshold_y, model, pairwise_crossings
from lossytoric.rbim import jackknife, xi_over_L

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, RES = ROOT / "data" / "rbim", ROOT / "figures", ROOT / "results"
L_COLORS = {8: "#86b6ef", 12: "#3987e5", 16: "#1c5cab", 20: "#0d366b", 24: "#08244a"}
MWPM_COLOR, OPT_COLOR = "#1c5cab", "#c2410c"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5df", "#fcfcfb"
# Ohzeki, PRA 85, 060301(R) (2012), Table I (independent X/Z errors = our bit-flip sector):
# optimal thresholds from a duality analysis, and matching-decoder values of Stace & Barrett [PRA 81, 022317]
OHZEKI = {"e": [0.0, 0.1, 0.2, 0.3, 0.4], "optimal": [0.11003, 0.09240, 0.07245, 0.04984, 0.02462],
          "matching": [0.10486, 0.08816, 0.06997, 0.04836, 0.02561]}
P_NISHIMORI = 0.10919   # Hasenbusch, Parisen Toldin, Pelissetto, Vicari, PRE 77, 051115 (2008): 1 - 0.89081(7)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
})


def load():
    """{(L, p, e): dict of concatenated per-sample arrays}"""
    groups = defaultdict(lambda: defaultdict(list))
    for f in sorted(DATA.glob("*.npz")):
        d = np.load(f)
        key = (int(d["L"]), round(float(d["p"]), 6), round(float(d["e"]), 6))
        for k in ("m2", "mk2", "m4", "ebond"):
            groups[key][k].append(d[k])
    return {k: {n: np.concatenate(v) for n, v in g.items()} for k, g in groups.items()}


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    data = load()

    # 1) exact check on every grid point: Nishimori energy identity
    worst = 0.0
    for (L, p, e), d in data.items():
        per_bond = d["ebond"] / (2 * L * L)
        z = (per_bond.mean() - (1 - e) * (1 - 2 * p)) / (per_bond.std() / np.sqrt(len(per_bond)))
        worst = max(worst, abs(z))
    print(f"Nishimori energy identity: worst deviation over {len(data)} grid points = {worst:.1f} sigma")

    # 2) xi/L per grid point (jackknife over disorder samples), FSS fit per erasure rate
    table = defaultdict(list)          # e -> rows (L, p, xi/L, err, n_samples)
    for (L, p, e), d in sorted(data.items()):
        v, dv = jackknife(lambda a, b: xi_over_L(a, b, L), d["m2"], d["mk2"])
        table[e].append((L, p, v, dv, len(d["m2"])))

    results = []
    for e, rows in sorted(table.items()):
        L, p, y, dy, n = map(np.array, zip(*rows))
        if len(set(L)) < 3:
            print(f"e={e}: fewer than 3 sizes, skipped")
            continue
        fit = fit_threshold_y(p, L, y, dy, q0=float(np.median(p)), nu0=1.5)
        xs = [x for *_, x in pairwise_crossings(p, L, y)]
        fit.sys = (max(xs) - min(xs)) / 2 if len(xs) >= 2 else float("nan")
        results.append((e, fit, (L, p, y, dy), xs))
        print(f"e={e:.2f}: p_c = {fit.qc:.4f} +- {fit.qc_err:.4f} (stat) +- {fit.sys:.4f} (sys), "
              f"nu = {fit.nu:.2f} +- {fit.nu_err:.2f}, chi2/dof = {fit.chi2_dof:.2f}, "
              f"pairwise crossings {np.round(xs, 4).tolist()}, samples/point {int(n.min())}")

    with open(RES / "rbim_thresholds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["e", "p_c_optimal", "stat_err", "sys_err", "nu", "nu_err", "chi2_dof"])
        for e, fit, *_ in results:
            w.writerow([e, f"{fit.qc:.5f}", f"{fit.qc_err:.5f}", f"{fit.sys:.5f}",
                        f"{fit.nu:.3f}", f"{fit.nu_err:.3f}", f"{fit.chi2_dof:.2f}"])

    # Figure: xi/L crossings, one panel per erasure rate
    n = len(results)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.3 * nrow), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (e, fit, (L, p, y, dy), _) in zip(axes.flat, results):
        pp = np.linspace(p.min(), p.max(), 200)
        for Lv in sorted(set(L)):
            m = L == Lv
            c = L_COLORS[Lv]
            ax.errorbar(p[m], y[m], yerr=dy[m], fmt="o", ms=4, color=c, mec=SURFACE, mew=1,
                        elinewidth=1, capsize=0, label=f"L = {Lv}", zorder=3)
            ax.plot(pp, model(pp, np.full_like(pp, Lv), fit.params), color=c, lw=1.5, zorder=2)
        ax.axvline(fit.qc, color=MUTED, lw=1, ls="--")
        if e == 0:
            ax.axvline(P_NISHIMORI, color=OPT_COLOR, lw=1, ls=":")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("bit-flip rate p (Nishimori line)")
        ax.set_title(f"e = {e:g}:  p_c = {fit.qc:.4f} ± {np.hypot(fit.qc_err, fit.sys):.4f}", loc="left", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"correlation length $\xi/L$")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("RBIM on the Nishimori line: xi/L crossings give the optimal threshold\n"
                 "dashed: fitted p_c;  dotted (e = 0): literature Nishimori point 0.1092",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "rbim_crossings.png", dpi=160)

    # Figure: optimal vs MWPM trade-off
    mw = list(csv.DictReader(open(RES / "thresholds.csv")))
    mw_pts = sorted([(float(r["fixed_value"]), float(r["threshold"]),
                      float(np.hypot(float(r["stat_err"]), float(r["sys_err"])))) for r in mw if r["swept"] == "p"])
    mw_pts.append((0.5, 0.0, 0.0))
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    ex, py, pe = map(np.array, zip(*mw_pts))
    ax.plot(ex, py, color=MWPM_COLOR, lw=2, zorder=2)
    ax.errorbar(ex, py, yerr=pe, fmt="o", ms=5, color=MWPM_COLOR, mec=SURFACE, mew=1.2, capsize=0, zorder=3,
                label="MWPM, this work")
    oe = np.array([r[0] for r in results] + [0.5])
    op = np.array([r[1].qc for r in results] + [0.0])
    oerr = np.array([np.hypot(r[1].qc_err, r[1].sys) for r in results] + [0.0])
    ax.plot(oe, op, color=OPT_COLOR, lw=2, zorder=2)
    ax.errorbar(oe, op, yerr=oerr, fmt="s", ms=5, color=OPT_COLOR, mec=SURFACE, mew=1.2, capsize=0, zorder=3,
                label="optimal, this work (RBIM Monte Carlo)")
    ax.plot(OHZEKI["e"], OHZEKI["matching"], "o", ms=8, mfc="none", mec=MWPM_COLOR, mew=1.2, zorder=4,
            label="matching, Stace & Barrett 2010 (via Ohzeki)")
    ax.plot(OHZEKI["e"], OHZEKI["optimal"], "D", ms=8, mfc="none", mec=OPT_COLOR, mew=1.2, zorder=4,
            label="optimal, Ohzeki 2012 (duality estimate)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_xlim(-0.01, 0.53)
    ax.set_ylim(0, 0.135)
    ax.set_xlabel("erasure (loss) rate e")
    ax.set_ylabel("bit-flip threshold $p_c(e)$")
    ax.set_title("Toric code: what MWPM achieves vs what is possible\n"
                 "optimal = ferromagnetic transition of the diluted RBIM on the Nishimori line",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff_optimal_vs_mwpm.png", dpi=160)
    print("figures written")


if __name__ == "__main__":
    main()
