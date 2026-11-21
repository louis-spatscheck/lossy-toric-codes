"""Fit thresholds by finite-size scaling and make the README figures.

python scripts/analyze.py            # reads data/fine.csv and data/erasure_only.csv
Writes figures/*.png and results/thresholds.csv
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from lossytoric.fss import fit_threshold, model, pairwise_crossings

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, RES = ROOT / "data", ROOT / "figures", ROOT / "results"

# ordinal single-hue ramp (light -> dark = small -> large L), validated light-mode
L_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5df", "#fcfcfb"
ERASURE_WINDOW = 0.025   # half-width of the erasure-only fit window (see main)
MWPM_REF = 0.103   # MWPM bit-flip threshold, toric code (Wang, Harrington, Preskill 2003)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
})


def load(path: Path):
    """Sum repeated rows (re-runs with new seeds) -> {(L, p, e): [shots, failures]}."""
    agg = defaultdict(lambda: [0, 0])
    if not path.exists():
        return agg
    for r in csv.DictReader(open(path)):
        k = (int(r["L"]), float(r["p"]), float(r["e"]))
        agg[k][0] += int(r["shots"])
        agg[k][1] += int(r["failures"])
    return agg


def sweep_arrays(agg, fixed: str, value: float):
    """Rows with p (or e) fixed; returns swept q, L, shots, failures."""
    idx_fixed, idx_q = (2, 1) if fixed == "e" else (1, 2)
    rows = sorted((k[0], k[idx_q], n, f) for k, (n, f) in agg.items() if abs(k[idx_fixed] - value) < 1e-9)
    if not rows:
        return None
    L, q, n, f = map(np.array, zip(*rows))
    return q, L, n, f


def plot_sweep(ax, q, L, n, f, fit, xlabel, title):
    Ls = sorted(set(L))
    qq = np.linspace(q.min(), q.max(), 200)
    for c, Lv in zip(L_COLORS[-len(Ls):] if len(Ls) < 4 else L_COLORS, Ls):
        m = L == Lv
        P = f[m] / n[m]
        err = np.sqrt(P * (1 - P) / n[m])
        ax.errorbar(q[m], P, yerr=err, fmt="o", ms=4, color=c, mec=SURFACE, mew=1, elinewidth=1, capsize=0,
                    label=f"L = {Lv}", zorder=3)
        ax.plot(qq, model(qq, np.full_like(qq, Lv), fit.params), color=c, lw=1.5, zorder=2)
    ax.axvline(fit.qc, color=MUTED, lw=1, ls="--", zorder=1)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10, color=INK)


def systematic(arr) -> float:
    """Half the spread of pairwise crossings (finite-size systematic)."""
    xs = [x for _, _, x in pairwise_crossings(*arr)]
    return (max(xs) - min(xs)) / 2 if len(xs) >= 2 else float("nan")


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    fine, eo = load(DATA / "fine.csv"), load(DATA / "erasure_only.csv")
    es = sorted({k[2] for k in fine})
    results = []
    sweeps = []
    for e in es:
        arr = sweep_arrays(fine, "e", e)
        if len(set(arr[1])) < 3 or len(arr[0]) < 12:   # incomplete sweep: skip
            print(f"skipping e = {e:g}: incomplete sweep ({len(arr[0])} points)")
            continue
        fit = fit_threshold(*arr)
        fit.sys = systematic(arr)
        results.append(("p", e, fit))
        sweeps.append((arr, fit, "bit-flip rate p", f"e = {e:g}:  p_c = {fit.qc:.4f} ± {fit.qc_err:.4f} ± {fit.sys:.4f}"))
    arr = sweep_arrays(eo, "p", 0.0)
    if arr is not None:
        # The quadratic ansatz only holds close to the crossing. The erasure sweep spans +-0.06,
        # too wide (chi2/dof ~ 37), so refit inside a window centred on a first-pass estimate.
        # The window is centred on the data, not on the known answer 0.5.
        first = fit_threshold(*arr, n_boot=0)
        m = np.abs(arr[0] - first.qc) <= ERASURE_WINDOW + 1e-9
        arr = tuple(a[m] for a in arr)
        fit = fit_threshold(*arr)
        fit.sys = systematic(arr)
        results.append(("e", 0.0, fit))
        sweeps.append((arr, fit, "erasure rate e", f"p = 0:  e_c = {fit.qc:.4f} ± {fit.qc_err:.4f} ± {fit.sys:.4f}"))

    with open(RES / "thresholds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["swept", "fixed_value", "threshold", "stat_err", "sys_err", "nu", "nu_err", "chi2_dof"])
        for swept, val, fit in results:
            w.writerow([swept, val, f"{fit.qc:.5f}", f"{fit.qc_err:.5f}", f"{fit.sys:.5f}", f"{fit.nu:.3f}", f"{fit.nu_err:.3f}",
                        f"{fit.chi2_dof:.2f}"])
            print(f"{swept} swept, other={val:g}: threshold {fit.qc:.4f} ± {fit.qc_err:.4f} (stat) ± {fit.sys:.4f} (sys), "
                  f"nu = {fit.nu:.2f} ± {fit.nu_err:.2f}, chi2/dof = {fit.chi2_dof:.2f}")

    # Figure 1: all crossings, small multiples
    ncol = 3
    nrow = int(np.ceil(len(sweeps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.3 * nrow), squeeze=False)
    for ax, (arr, fit, xl, title) in zip(axes.flat, sweeps):
        plot_sweep(ax, *arr, fit, xl, title)
    for ax in axes.flat[len(sweeps):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("logical failure rate $P_L$")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Toric code + MWPM: logical failure rate vs noise, curves = finite-size-scaling fit",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "crossings.png", dpi=160)

    # Figure 2: threshold trade-off diagram
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    tot = lambda f: float(np.hypot(f.qc_err, np.nan_to_num(f.sys)))
    pts = [(val, fit.qc, tot(fit), 0.0) for s, val, fit in results if s == "p"]
    pts += [(fit.qc, 0.0, 0.0, tot(fit)) for s, val, fit in results if s == "e"]
    pts.sort()
    ex, py, pye, exe = map(np.array, zip(*pts))
    ax.fill_between(ex, 0, py, color=L_COLORS[0], alpha=0.18, lw=0, zorder=1)
    ax.plot(ex, py, color=L_COLORS[2], lw=2, zorder=2)
    ax.errorbar(ex, py, yerr=pye, xerr=exe, fmt="o", ms=6, color=L_COLORS[2], mec=SURFACE, mew=1.5,
                elinewidth=1.2, capsize=0, zorder=3)
    ax.plot([0, 0.5], [MWPM_REF, 0], color=MUTED, lw=1, ls=":", zorder=1)
    ax.annotate("known: 10.3% (MWPM, e = 0)", (0, MWPM_REF), xytext=(0.06, 0.106), fontsize=8.5, color=MUTED)
    ax.annotate("known: 50% (bond percolation)", (0.5, 0), xytext=(0.255, 0.006), fontsize=8.5, color=MUTED)
    ax.text(0.09, 0.03, "correctable", fontsize=10, color=L_COLORS[3])
    ax.set_xlim(-0.01, 0.53)
    ax.set_ylim(0, 0.115)
    ax.set_xlabel("erasure (loss) rate e")
    ax.set_ylabel("bit-flip threshold $p_c(e)$")
    ax.set_title("Loss/error threshold trade-off (toric code, MWPM)\ndotted: straight line between the known endpoints",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.png", dpi=160)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
