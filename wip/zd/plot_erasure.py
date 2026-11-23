"""Plot erasure_only_zd.csv, with the Phase 1 MWPM qubit data on top as a cross-check."""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = pathlib.Path(__file__).resolve().parent
root = here.parents[1]
z = np.genfromtxt(here / "erasure_only_zd.csv", delimiter=",", names=True)
q = np.genfromtxt(root / "data" / "erasure_only.csv", delimiter=",", names=True)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, d in zip(axes, (2, 3)):
    for c, L in zip(("C0", "C1", "C2"), sorted(set(z["L"].astype(int)))):
        m = (z["d"] == d) & (z["L"] == L)
        ax.errorbar(z["e"][m], z["p_fail"][m], z["err"][m], fmt="o-", ms=3, color=c, label=f"L={L}")
        if d == 2:
            mq = (q["L"] == L)
            if mq.any():
                # Phase 1 may have several rows per e (appended runs) -> pool them
                es = np.unique(q["e"][mq])
                pf = [q["failures"][mq & (q["e"] == e)].sum() / q["shots"][mq & (q["e"] == e)].sum() for e in es]
                ax.plot(es, pf, "x", color=c, ms=6)
    ax.axvline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_title(f"$d={d}$" + ("   (x = Phase 1 MWPM)" if d == 2 else ""))
    ax.set_xlabel("erasure rate e")
axes[0].set_ylabel("P_fail, X sector, optimal")
axes[0].legend(frameon=False)
fig.tight_layout()
out = here / "erasure_only_zd.png"
fig.savefig(out, dpi=130)
print("saved", out)
