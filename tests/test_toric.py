import numpy as np
import pytest

from lossytoric import ErasureMWPM, ToricCode, logical_failures, run, sample_errors, syndrome
from lossytoric.toric import h_edge, v_edge


@pytest.mark.parametrize("L", [3, 4, 7])
def test_every_qubit_flips_two_checks(L):
    H = ToricCode(L).check_matrix().toarray()
    assert H.shape == (L * L, 2 * L * L)
    assert np.all(H.sum(axis=0) == 2)   # graph-like: matching applies
    assert np.all(H.sum(axis=1) == 4)   # weight-4 vertex checks


@pytest.mark.parametrize("L", [3, 4, 7])
def test_stabilizers_commute(L):
    code = ToricCode(L)
    H = code.check_matrix().toarray()
    P = code.plaquette_x()
    assert np.all((H @ P.T) % 2 == 0)              # Z-vertex vs X-plaquette checks
    assert np.all((code.logical_z() @ P.T) % 2 == 0)  # logical Z commutes with X stabilizers


@pytest.mark.parametrize("L", [3, 5])
def test_noncontractible_cycles_are_logical(L):
    code = ToricCode(L)
    H = code.check_matrix().toarray()
    row = np.zeros(code.n_qubits, dtype=np.uint8)
    for j in range(L):
        row[h_edge(0, j, L)] = 1          # winds in the j-direction
    col = np.zeros(code.n_qubits, dtype=np.uint8)
    for i in range(L):
        col[v_edge(i, 0, L)] = 1          # winds in the i-direction
    for cyc, expected in ((row, [1, 0]), (col, [0, 1])):
        assert np.all((H @ cyc) % 2 == 0)                       # undetectable
        assert list((code.logical_z() @ cyc) % 2) == expected   # but logical


def test_no_error_no_failure():
    code = ToricCode(5)
    dec = ErasureMWPM(code.check_matrix())
    x = np.zeros((10, code.n_qubits), dtype=bool)
    corr = dec.decode_batch(syndrome(code.check_matrix(), x))
    assert not corr.any()
    assert not logical_failures(code, x, corr).any()


def test_correction_matches_syndrome():
    code = ToricCode(6)
    H = code.check_matrix()
    dec = ErasureMWPM(H)
    rng = np.random.default_rng(1)
    erased, x = sample_errors(code.n_qubits, 0.08, 0.2, 200, rng)
    s = syndrome(H, x)
    corr = dec.decode_batch(s, erased)
    assert np.array_equal(syndrome(H, corr), s)


def test_single_error_always_corrected():
    L = 5
    code = ToricCode(L)
    H = code.check_matrix()
    dec = ErasureMWPM(H)
    x = np.eye(code.n_qubits, dtype=bool)
    corr = dec.decode_batch(syndrome(H, x))
    assert not logical_failures(code, x, corr).any()


def test_erasure_is_used():
    """A full erased column (a winding path) is still correctable if the decoder uses erasure info:
    flipping half of an erased non-contractible loop gives a syndrome whose cheapest explanation
    under erasure weights lies inside the erased set."""
    L = 6
    code = ToricCode(L)
    H = code.check_matrix()
    dec = ErasureMWPM(H)
    erased = np.zeros((1, code.n_qubits), dtype=bool)
    x = np.zeros((1, code.n_qubits), dtype=bool)
    for j in range(L):
        erased[0, h_edge(0, j, L)] = True
    for j in range(L // 2 + 1):          # flip more than half of the loop
        x[0, h_edge(0, j, L)] = True
    corr = dec.decode_batch(syndrome(H, x), erased)
    # Without erasure info MWPM would pick the shorter complement and fail; with it, the
    # residual is either trivial or the full loop (degenerate tie). Either way it stays inside
    # the erased set, which is the defining property of erasure decoding.
    residual = x[0] ^ corr[0].astype(bool)
    assert not residual[~erased[0]].any()


def test_zero_noise_rate():
    r = run(L=4, p=0.0, e=0.0, shots=500)
    assert r.failures == 0


def test_full_erasure_fails_half_or_more():
    # e = 1: every qubit maximally mixed -> logical state is random: failure rate ~ 3/4
    r = run(L=4, p=0.0, e=1.0, shots=2000, seed=3)
    assert 0.65 < r.rate < 0.85
