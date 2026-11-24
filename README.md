# lossy-toric-codes

[![tests](https://github.com/louis-spatscheck/lossy-toric-codes/actions/workflows/tests.yml/badge.svg)](https://github.com/louis-spatscheck/lossy-toric-codes/actions/workflows/tests.yml)

Thresholds of the toric code under **bit-flip errors plus erasures**,
computed in two independent ways:

* **decoder:**minimum-weight perfect matching (MWPM).
* **optimal:** stat-mech mapping to the
  random-bond Ising model (RBIM) on the Nishimori line, by Monte Carlo with parallel tempering.

Thresholds are extracted by finite-size scaling (FSS).

**Status.** Every number below reproduces a published value within errors, for the matching decoder and
for the optimal one. That makes it a tested baseline for the next step: qudit $\mathbb{Z}_d$ codes, where
the loss/error trade-off is much less mapped out (see *Roadmap*).

![Optimal vs MWPM trade-off](figures/tradeoff_optimal_vs_mwpm.png)

## Phase 1: MWPM thresholds

Code-capacity noise on an $L\times L$ torus: each qubit is erased with probability $e$. Each non-erased qubit gets an X error with probability $p$. MWPM uses edge weight 0
on erased qubits and 1 elsewhere. System sizes are $L = 8, 12, 16, 24$, with $10^4$–$4\times10^4$ shots per point.

| noise | threshold (this work) | known value | FSS exponent $\nu$ |
|---|---|---|---|
| bit-flip only ($e=0$) | $p_c = 0.1042 \pm 0.0002_{\rm stat} \pm 0.0008_{\rm sys}$ | 0.103 (MWPM) [2] | $1.53 \pm 0.04$ |
| $e = 0.1$ | $p_c = 0.0887 \pm 0.0003 \pm 0.0013$ | | $1.38 \pm 0.08$ |
| $e = 0.2$ | $p_c = 0.0691 \pm 0.0003 \pm 0.0021$ | | $1.54 \pm 0.10$ |
| $e = 0.3$ | $p_c = 0.0475 \pm 0.0003 \pm 0.0012$ | | $1.60 \pm 0.11$ |
| $e = 0.4$ | $p_c = 0.0247 \pm 0.0003 \pm 0.0021$ | | $1.65 \pm 0.09$ |
| erasure only ($p=0$) | $e_c = 0.4989 \pm 0.0009 \pm 0.0013$ | 1/2 (bond percolation) [3] | $1.33 \pm 0.07$ (percolation: 4/3) |

* **Validation.** Both endpoints agree with the literature. For erasure only, the fitted
  exponent $\nu = 1.33 \pm 0.07$ is consistent with the 2D percolation value $4/3$, but it depends on
  the fit window (see *Limitations*), so treat it as a consistency check. The reason to expect
  percolation: with erasures only, the decoder can only fail when the erased qubits contain a loop around the torus.
  The bit-flip value sits about 1.5σ (combined error) above 10.3%. The pairwise crossings
  of consecutive sizes range from 0.1024 to 0.1041, so the remaining offset is consistent with
  finite-size corrections at $L \le 24$.
* **Trade-off.** $p_c(e)$ decreases smoothly from 10.4% to 0 at $e = 1/2$. It lies slightly
  above the straight line between the two endpoints. Erasure is much cheaper to correct than
  unknown errors, which is why heralded loss is the "good" error for photonics.
* **Error bars.** Statistical errors come from a parametric bootstrap of the FSS fit (half the 16–84%
  range of the refitted values). The systematic
  error is half the spread of pairwise crossings of consecutive $L$, a crude estimate of the
  finite-size corrections that the quadratic ansatz $P_L = A + Bx + Cx^2$,
  $x = (q-q_c)L^{1/\nu}$, ignores.

### Limitations

* **Code capacity**: perfect syndrome measurements. No measurement errors and no circuit-level noise.
* **MWPM is not optimal.** 
* Independent X/Z noise, so one sector is simulated. A toric rather than a planar geometry.


## Phase 2a: optimal thresholds from the stat-mech mapping

**Mapping** [1]. Put Ising spins on the dual lattice, one bond per qubit. A bit flip makes its bond
antiferromagnetic ($J=-1$). An erased qubit carries a uniformly random flip, i.e. no information, so its
bond is removed ($J=0$, dilution) [4,5]. The temperature is not a free parameter: Bayes-optimal
decoding puts the model on the **Nishimori line** [11], $\tanh\beta = 1-2p$. The optimal decoder
succeeds for large $L$ exactly when this disordered magnet is ferromagnetically ordered, so the optimal
threshold $p_c(e)$ is the ferro/para transition along the Nishimori line.

**Method.** Monte Carlo of $10^3$ disorder samples per point, for $L = 8$–$20$ ($e = 0$) and $L = 8$–$16$
($e > 0$). Checkerboard heat bath plus parallel tempering [12] (8 temperatures from $\beta_N$ to
$\beta = 0.6$), vectorised over samples and temperatures in numpy. Observable: the second-moment
correlation length $\xi/L$, which is scale invariant at the transition, so curves for different $L$ cross at
$p_c$. Errors are jackknifed over disorder samples, then fitted with the same FSS ansatz as Phase 1.

| erasure $e$ | optimal $p_c$ (this work) | optimal, Ohzeki 2012 [10] | MWPM (Phase 1) | matching, Stace & Barrett [4] (via [10]) |
|---|---|---|---|---|
| 0   | $0.1106 \pm 0.0008 \pm 0.0030$ | 0.1100 (precise MC: 0.1092 [13]) | 0.1042 | 0.1049 |
| 0.1 | $0.0930 \pm 0.0013 \pm 0.0011$ | 0.0924 | 0.0887 | 0.0882 |
| 0.2 | $0.0716 \pm 0.0018 \pm 0.0024$ | 0.0725 | 0.0691 | 0.0700 |
| 0.3 | $0.0499 \pm 0.0015 \pm 0.0000^{*}$ | 0.0498 | 0.0475 | 0.0484 |
| 0.4 | $0.0277 \pm 0.0011 \pm 0.0019$ | 0.0246 | 0.0247 | 0.0256 |

Ohzeki's values are read from Table I of [10]; check the table for the approximation level before quoting them.

$^{*}$ With three sizes there are only two pairwise crossings; at $e=0.3$ they coincide by chance,
so the systematic error is underestimated there.

* **Agreement.** Every optimal value agrees with Ohzeki's duality estimates [10] within errors, and
  $e=0$ agrees with the high-precision Nishimori point $p_c = 0.10919(7)$ [13] at 0.5σ. Optimal lies
  above MWPM at every $e$, by about 0.003–0.006: MWPM is close to optimal for this code and noise.
  Whether that stays true for qudits, where matching is no longer exact, is the first question for $\mathbb{Z}_3$.
* **One detail at $e = 0.4$.** The duality estimate (0.0246) lies *below* the matching value (0.0256),
  which an optimal threshold cannot do; the duality method is an approximation, and [10] notes
  finite-size effects from lost qubits at large loss. Our Monte Carlo keeps the correct ordering
  (0.0277 > 0.0247), although it is only 1.4σ from 0.0246, so this is a consistency check, not a claim.
* **Exact checks.** (i) Brute-force enumeration of all $2^{16}$ states of a 4×4 lattice with frustrated
  and diluted couplings. (ii) The Nishimori energy identity $[\langle\sum_b J_b s_i s_j\rangle] =
  N_b(1-e)(1-2p)$ holds at all 96 grid points (z-scores: mean 0.3, standard deviation 0.9).
  (iii) Equilibration: a planted (all-up) start over-estimates $\xi/L$ until the chain decorrelates, a
  random start under-estimates it; the two agree within errors at every $L$ and $e$ used
  (`scripts/rbim_eqcheck.py`).

![RBIM crossings](figures/rbim_crossings.png)

### What did not work (and why the sizes stop at 16–20)

* **Single-spin updates alone:** at $L=12$ near $p_c$ the planted and random starts still disagree
  after $10^4$ sweeps. Disorder pins domain walls, so equilibration is very slow.
* **Swendsen–Wang cluster moves:** valid for frustrated couplings but hardly faster here; the clusters
  span the system in the ordered phase.
* **Parallel tempering** fixes it, but the time to equilibrate grows steeply with $L$: $L=20$ needs
  about 1500 sweeps; $L=24$ was still not equilibrated after 1000 and would need about 3000. With 2 CPU
  cores, the run stops at $L = 20$. That limits precision ($\pm 0.003$ vs $\pm 0.0001$ in [13]) and
  leaves finite-size corrections visible in the scatter of pairwise crossings.
* The **planted start is not a free equilibration** for the magnetisation (a mistake in my first
  version): it is an exact equilibrium sample only for gauge-invariant observables such as the energy.

## Phase 2b (started): faulty measurements, random-plaquette gauge model

**Why.** Everything above assumes perfect syndrome measurements (`syn = syndrome(H, x_err)` in
`simulate.run`). With readout errors (rate $q$) the syndrome is measured for $T$ rounds, errors
become chains in a 3D spacetime lattice (space edges: qubit flips; time edges: wrong readouts),
and the decoder pairs *detection events*, i.e. changes between rounds.

**Mapping** [1,2]. Equivalent error histories differ by elementary spacetime squares (a
stabilizer in one round, or a flip and flip-back hidden by two wrong readouts). One Z2 variable
per square lives on a link of the dual lattice; each error edge pierces one dual plaquette. The
result is a Z2 lattice gauge theory with random plaquette signs (RPGM) on the Nishimori line.
Qubit flips $p$ and erasures sit on **timelike** plaquettes, readout errors $q$ on **spacelike**
ones. The local gauge symmetry kills local order parameters (Elitzur), so the code measures
Polyakov loops. A logical error is a $Z_2$ vortex wrapping space; the optimal threshold is the
confinement transition, $p_c \simeq 0.033$ at $p = q$, $e = 0$ [15].

**Code.** `src/lossytoric/rpgm.py` (heat bath on links in 6 sublattices + parallel tempering,
vectorised like `rbim.py`); erasure is fresh every round (a lost qubit is replaced); `scripts/rpgm_*`.

**Validated so far.**
* Exact enumeration on the $2^3$ box (gauge-fixed on a maximal tree: $2^{17}$ configurations), with
  frustration, erasure and different space/time couplings: Monte Carlo = exact.
* Nishimori energy identity on timelike and spacelike plaquettes, separately: over 36 checks the
  z-scores have mean 0.02 and standard deviation 1.08.
* Clean limit (`scripts/rpgm_clean_check.py`): $\xi_P/L$ crossings at $\beta = 0.782$ ($L$ = 4/6)
  and 0.767 (6/8), approaching the exact 3D-Ising-dual value 0.7614.
* Equilibration at $L = 8$, $p = 0.033$: planted and hot starts are 10σ apart after 150 sweeps,
  1.5σ after 500, and agree (0.1σ) after 1500 with 14 temperatures.

**Threshold: not measured yet.** A first sweep ($L$ = 4, 6, 8; 100 disorder samples per point,
`figures/rpgm_crossings.png`) behaves as expected for $p \ge 0.036$ ($\xi_P/L$ falls with $L$), but
below that the errors are too large to resolve a crossing; the fit (0.029 ± 0.002, $\chi^2$/dof 2.2)
is not a result. It needs ≥ 400 samples per point and $L \ge 10$: a cluster job.

## Roadmap

1. ~~Optimal qubit thresholds via the RBIM on the Nishimori line, with dilution~~ (done, Phase 2a).
   * **Faulty measurements** (Phase 2b, started): reproduce $p_c \simeq 0.033$, then the loss/error
     trade-off $p_c(e)$ with readout errors; then a 3D matching decoder for comparison.
2. **Domain-wall free energy.** The optimal decoder's failure probability is set by ratios of
   partition functions with twisted boundary conditions, a free-energy-difference estimation
   problem (variance reduction, signal-to-noise). It gives a decoder-level observable directly
   comparable to Phase 1.
3. **$\mathbb{Z}_3$ (qutrit) toric code** with erasure. A self-written decoder (PyMatching is qubit-only).
4. **Diluted random 3-state Potts/clock model** → optimal loss/error trade-off for qutrits.
   Stay at $d \le 4$: for $d \ge 5$ the clean clock model has BKT transitions.
5. *Optional:* $\mathbb{Z}_4$ vs $\mathbb{Z}_2\times\mathbb{Z}_2$ (same local dimension, different group):
   a first look at group dependence. Group surface codes, the quantum double models of finite groups,
   give non-Clifford gates for the qubit surface code [14]; how much loss they tolerate is open territory
   for this kind of pipeline.

First steps towards item 3 (pure erasure for $d = 2, 3$, lattice and stubs) are in `wip/zd/`; not yet tested like `src/`.

Prior work this builds on (so nothing gets reinvented): qubit loss + error thresholds [3,4];
exact stat-mech mappings with erasure + computational errors for qubits [5]; qudit toric-code
decoders and thresholds without erasure [6,7]; random-bond clock models for decohered
$\mathbb{Z}_N$ toric codes [8].



## Layout

```
src/lossytoric/
  toric.py      lattice, check matrix H, logical operators (conventions in the docstring)
  noise.py      bit-flip + erasure sampling
  decoder.py    PyMatching MWPM with erasure weights
  simulate.py   Monte Carlo of the logical failure rate (Wilson intervals)
  fss.py        finite-size-scaling fit, bootstrap errors, pairwise crossings
  rbim.py       random-bond Ising model on the Nishimori line: heat bath + parallel tempering, jackknife
  rpgm.py       random-plaquette Z2 gauge model (faulty measurements): link heat bath + PT, Polyakov loops
scripts/        sweep.py, analyze.py                      (Phase 1: MWPM)
                rbim_sweep.py, rbim_eqcheck.py, rbim_analyze.py   (Phase 2a: stat-mech MC)
                rpgm_sweep.py, rpgm_eqcheck.py, rpgm_analyze.py, rpgm_clean_check.py   (Phase 2b)
tests/          lattice algebra, decoder sanity, erasure use, FSS fit on synthetic data,
                RBIM vs exact enumeration, Nishimori identity, planted vs random start,
                gauge model: gauge invariance, exact enumeration (gauge-fixed), Nishimori identity
notebooks/      gauge_covariant_decoding.ipynb: interactive introduction to gauge-covariant neural decoding
wip/            zd/ (qudit erasure), lcnn/ (gauge-covariant neural decoder, needs jax + optax)
data/ results/ figures/
```

## References

1. E. Dennis, A. Kitaev, A. Landahl, J. Preskill, *Topological quantum memory*, J. Math. Phys. 43, 4452 (2002).
2. C. Wang, J. Harrington, J. Preskill, *Confinement-Higgs transition in a disordered gauge theory and the accuracy threshold for quantum memory*, Ann. Phys. 303, 31 (2003).
3. T. M. Stace, S. D. Barrett, A. C. Doherty, *Thresholds for topological codes in the presence of loss*, PRL 102, 200501 (2009).
4. T. M. Stace, S. D. Barrett, *Error correction and degeneracy in surface codes suffering loss*, PRA 81, 022317 (2010).
5. L. Colmenarez, S. Kim, M. Müller, *Fundamental thresholds for computational and erasure errors via the coherent information*, PRX Quantum 6, 040327 (2025).
6. G. Duclos-Cianci, D. Poulin, *Kitaev's Z_d-codes threshold estimates*, arXiv:1302.3638.
7. H. Anwar, B. J. Brown, E. T. Campbell, D. E. Browne, *Fast decoders for qudit topological codes*, NJP 16, 063038 (2014).
8. A. Vijay, J. Y. Lee, *Information critical phases under decoherence*, arXiv:2512.22121.
9. O. Higgott, C. Gidney, *Sparse Blossom: correcting a million errors per core second with minimum-weight matching* (PyMatching v2), Quantum 9, 1600 (2025).
10. M. Ohzeki, *Error threshold estimates for surface code with loss of qubits*, PRA 85, 060301(R) (2012).
11. H. Nishimori, *Internal energy, specific heat and correlation function of the bond-random Ising model*, Prog. Theor. Phys. 66, 1169 (1981).
12. K. Hukushima, K. Nemoto, *Exchange Monte Carlo method and application to spin glass simulations*, J. Phys. Soc. Jpn. 65, 1604 (1996).
13. M. Hasenbusch, F. Parisen Toldin, A. Pelissetto, E. Vicari, *Multicritical Nishimori point in the phase diagram of the ±J Ising model on a square lattice*, PRE 77, 051115 (2008).
14. N. Manjunath, V. Mattei, A. Tiwari, T. D. Ellison, *Universal quantum computation with group surface codes*, arXiv:2603.05502 (2026).
15. T. Ohno, G. Arakawa, I. Ichinose, T. Matsui, *Phase structure of the random-plaquette Z2 gauge model: accuracy threshold for a toric quantum memory*, Nucl. Phys. B 697, 462 (2004), arXiv:quant-ph/0401101.



## Use of AI tools

I wrote this repository with help from an AI coding assistant. I set the research
questions, the methods and the order of the work. Claude wrote most of the code. I
take full responsibility for everything here, including any errors.




## Licence

MIT. Questions or corrections are welcome: open an issue or email louisspatscheck@gmail.com.
