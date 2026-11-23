# Z_d: status and open questions

## Where things are
- [x] oriented Z_d lattice, signed incidence, windings (`zd_lattice.py`), checks in `scratch_checks.py`
- [x] GF(p) nullspace / rank (`gfp.py`). Prime d only.
- [x] pure erasure, optimal: P(fail | E) = 1 - d^-k, averaged over E (`erasure_only.py`)
- [ ] Z sector on the dual lattice (right now: X sector only)
- [ ] shift errors + erasure: noise model (#1), decoder (#2)
- [ ] optimal boundary for d = 3: random Potts / clock model on the Nishimori line (#3)
- [ ] speed: dense elimination is O(n^3), L = 16 takes minutes, L = 24 doesn't finish.
      k only needs the cycle space of E -> union-find / spanning forest of E should give it
      in ~linear time (count non-tree edges, track windings along tree paths).

## Open questions
1. **Noise model.** Symmetric (Potts-like) vs adjacent shifts (clock-like). For photonic qudits
   (time-bin, path, OAM, ...) the form of the dominant non-loss error is not settled here.
2. **Decoder for d = 3.** Charges +-1 mod 3; three equal charges fuse to vacuum, so plain MWPM
   is not exact. Candidates: (a) MWPM on charge pairs + handling of triples (Anwar et al. 2014),
   (b) RG / clustering decoder (Duclos-Cianci & Poulin), (c) union-find generalisation with erasure
   as pre-grown clusters -- probably the natural one with loss.
3. **Optimal threshold for d = 3 with erasure.** Erasure = dilution again. Nishimori condition:
   beta J(a) = log P(a). For the symmetric model this is the random-bond 3-state Potts model.
   Can reuse `rbim.py` structure (heat bath + parallel tempering), but the observable / FSS
   need checking (xi/L for Potts order parameter). d = 3 Potts in 2D: first-order? No, continuous
   for q <= 4 -- fine. Avoid d >= 5 for the clock model (BKT).
4. **Is k ever d-dependent?** Conjecture: not for prime d on the torus. A candidate counterexample
   fails (scratch_checks.py); erasure_only.py counts mismatches between d = 2 and d = 3.
5. Z_4 vs Z_2 x Z_2: needs non-field arithmetic (Smith normal form). Later.

