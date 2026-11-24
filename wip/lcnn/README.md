# Minimal gauge-covariant neural decoder (Z2, code capacity, with erasure)

Work in progress; not tested and documented like `src/`. Needs `jax` and `optax`
(`pip install jax optax`); everything else comes from `lossytoric`.

## Idea

The X-error chain is a Z2 link field on the dual lattice: adding a stabilizer is a gauge
transformation, the syndrome is its plaquettes, and the rows of `ToricCode.logical_z()` are its
Polyakov loops. For an abelian group the L-CNN machinery collapses (site features transform
trivially, Omega W Omega^dagger = W), so every gauge-invariant function of the chain is a function
of (syndrome, erasure mask, holonomies). The model is

    logit_x = [B_x] + mean over lines l of  P_x(l) * G_x(l)

with G_x and B_x from a translation-equivariant CNN on (syndrome, erasure) with periodic padding,
and P_x(l) the Polyakov loop of the reference chain along line l.

* **covariant** (no B): exactly gauge invariant and exactly odd under adding a logical operator to
  the reference chain, like the domain-wall free energy Delta F = ln Z(C)/Z(C + logical). The
  answer does not depend on which representative it is given, so it must decode from the syndrome.
* **residual** (with B): still exactly gauge invariant, not odd; can express "trust MWPM unless
  the syndrome says otherwise".

`test_z2_lcnn.py` checks these symmetries exactly with random weights (stabilizers added to the
reference, logicals added, lattice translations) and that the Polyakov lines equal `logical_z()`.

An interactive, step-by-step introduction to these ideas, with the exact optimal decoder at L = 4 as a
reference, is in `notebooks/gauge_covariant_decoding.ipynb`.

## Exact optimum at L = 4 (from the notebook)

At L = 4 the stabilizer group has only 2^15 elements, so the optimal decoder is computed by brute force.
p = 0.06, e = 0.1, 20 000 test shots: MWPM 0.1601, covariant network 0.1544 (paired change
-0.0057 +- 0.0020), exact optimum 0.1489. The network closes about half of the gap, and its logits
track the exact Delta F with correlation 0.98, although it was trained on right/wrong labels only.

## First results (L = 5, p = 0.06, e = 0.1; reference chain = MWPM correction)

Width 32, depth 6, 3x3 kernels; Adam, cosine schedule from 2e-3, batch 512, 8000 steps.
Evaluated on 200 000 fresh shots, paired against MWPM on the same shots:

| training data | covariant | residual |
|---|---|---|
| fixed set, 3e5 shots (~14 passes) | **worse** than MWPM: 0.1446 vs 0.1349 | worse: 0.1421 |
| streamed, 4.1e6 fresh shots (each seen once) | 0.1309: -2.9 % (7 sigma) | 0.1292: -4.3 % (12 sigma) |

MWPM logical failure rate: 0.1349 +- 0.0008. The fixed-set runs overfit: on their training data
they beat MWPM (per-direction accuracy 0.939 vs 0.923), on fresh data they lose. Streaming fixes it.
An earlier run at p = 0.10 (above threshold, MWPM fails 34 %) with a fixed set was also worse
than MWPM.

Reproduce one streamed run (about 20 min on 2 CPU cores, data generation dominates):

```python
from z2_lcnn import train, evaluate
params = train(L=5, p=0.06, e=0.1, steps=8000, batch=512, residual=True, log=1000)
evaluate(params, L=5, p=0.06, e=0.1, shots=200_000)
```

## Open

* How far from optimal? The gain over MWPM is 3-4 %; the optimal decoder's failure rate at L = 5
  is not computed yet. Per-sample Delta F (the free-energy route) would give it, and could serve
  as a regression target instead of binary labels.
* Larger L (receptive field: depth 6 covers L = 5 only; dilations or a U-Net for more), other
  (p, e), and whether one trained net transfers across L (it is translation equivariant and
  size-agnostic by construction).
* 3D (faulty measurements): the error field becomes a 2-form on the dual lattice, gauge
  transformations live on links, and the local invariants are the detection events.
