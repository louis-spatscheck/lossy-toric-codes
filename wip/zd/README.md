# wip/zd: towards Z_d (qudit) toric codes

**Work in progress.** Code here is not packaged or tested like `src/`. `notes.md` has the to-do list and open questions.

Goal: the loss / shift-error trade-off for qudit toric codes, first for $d = 3$, with a
decoder curve and an optimal curve as for qubits in the main README.

First step done: **pure erasure** with optimal decoding. For any $d$, the code can only fail
when the erased qudits contain a non-contractible cycle, so the threshold is the bond-percolation
value $e_c = 1/2$ for every $d$. What does change with $d$ is the failure rate above threshold:
given the erased set, $P(\text{fail}) = 1 - d^{-k}$, with $k$ the number of independent
logical classes supported on it.

```bash
python wip/zd/scratch_checks.py      # lattice sanity checks
python wip/zd/erasure_only.py        # ~10 min for L = 8-16 (slow, see notes.md)
python wip/zd/plot_erasure.py
```
