"""
The qubit switch witness certifies the switch construct for every target dimension d.

Argument: the qubit switch is the d-dimensional switch with fixed local operations
in A's and B's labs (compress the target into span{|0>,|1>}, act, embed it back).
Local operations preserve causal separability, so pulling the qubit witness back
through those operations gives a valid witness for the d-dimensional construct,
with the same value -0.1694 on the switch (target |0>).

In lab terms: A and B run the channel N_U (project onto the qubit subspace, send
the rest to |0>, then apply U (+) 1) instead of the qubit unitary U.

Checked here: N_U is a channel; d = 2 reproduces the shipped witness; the value on
the d-dimensional switch is the predicted one (d = 3, 4); the pulled-back witness
is >= 0 on random causally ordered processes with a d-dimensional target.
"""

import numpy as np
import pytest

from qrl.switch_witness import (
    QUBIT_STATES,
    WITNESS_UNITARIES,
    switch_process,
    switch_witness,
)

PREDICTED = -0.16940767627468822
rng = np.random.default_rng(2026)
_NAMES = list(WITNESS_UNITARIES)


def _haar(n):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _compress_kraus(U, d):
    """Kraus operators of N_U on C^d: project onto span{|0>,|1>}, send the
    rest to |0>, then apply U (+) 1."""
    Ut = np.eye(d, dtype=complex)
    Ut[:2, :2] = U
    P = np.zeros((d, d))
    P[0, 0] = P[1, 1] = 1
    e = np.eye(d)
    return [Ut @ P] + [Ut @ np.outer(e[0], e[k]) for k in range(2, d)]


def _choi(kraus):
    """Transposed Choi (input, output), as in qrl.switch_witness: conj(sum |K>><<K|)."""
    d = kraus[0].shape[1]
    C = 0
    for K in kraus:
        v = sum(np.kron(np.eye(d)[i], K[:, i]) for i in range(d))
        C = C + np.outer(v, v.conj())
    return np.asarray(C).conj()


def _pulled_back_value(W, d):
    """sum_terms weight * Tr[(conj|p><p| x Choi(N_a) x Choi(N_b) x |s><s|) W]."""
    n = d * d
    chois = np.stack([_choi(_compress_kraus(WITNESS_UNITARIES[u], d)) for u in _NAMES])
    T = W.reshape(2, n, n, 2, 2, n, n, 2)            # P A B F ; P' A' B' F'
    cache = {}
    value = 0.0
    for p, a, b, s, c in switch_witness().terms:
        if (p, s) not in cache:
            prep = np.outer(QUBIT_STATES[p], QUBIT_STATES[p].conj()).conj()
            eff = np.outer(QUBIT_STATES[s], QUBIT_STATES[s].conj())
            R = np.einsum("ij,kl,jabkicdl->abcd", prep, eff, T)    # (A B ; A' B')
            cache[(p, s)] = np.einsum("xij,ykl,jlik->xy", chois, chois, R)
        value += c * cache[(p, s)][_NAMES.index(a), _NAMES.index(b)].real
    return value


def _random_ordered_process(first, d, m=2):
    """Random process with order P < first < second < F and target dimension d.

    Same construction as the qubit helper in test_switch_witness.py, with
    memory dimension m: P's output enters an isometry into (first's input, M);
    a unitary takes (first's output, M) to (second's input, M'); another takes
    (second's output, M') to (F, junk).
    """
    V0 = _haar(d * m)[:, :2].reshape(d, m, 2)             # [x_in, M, P_O]
    V1 = _haar(d * m).reshape(d, m, d, m)                 # [y_in, M', x_out, M]
    V2 = _haar(d * m).reshape(2, d * m // 2, d, m)        # [F, junk, y_out, M']
    w = np.einsum("amp,qnbm,fjcn->pabqcfj", V0, V1, V2)  # P_O x_in x_out y_in y_out F junk
    if first == "B":
        w = w.transpose(0, 3, 4, 1, 2, 5, 6)
    w = w.reshape(4 * d ** 4, -1)
    return w @ w.conj().T


@pytest.mark.parametrize("d", [3, 4])
def test_compress_channel_is_trace_preserving(d):
    for U in WITNESS_UNITARIES.values():
        Ks = _compress_kraus(U, d)
        assert np.allclose(sum(K.conj().T @ K for K in Ks), np.eye(d), atol=1e-12)


def test_d2_reduces_to_shipped_witness():
    W = switch_process(2)
    assert np.isclose(_pulled_back_value(W.W, 2), switch_witness().value(W), atol=1e-10)


@pytest.mark.parametrize("d", [3, 4])
def test_pulled_back_witness_on_qudit_switch(d):
    assert np.isclose(_pulled_back_value(switch_process(d).W, d), PREDICTED, atol=1e-9)


@pytest.mark.parametrize("d", [3, 4])
def test_random_ordered_qudit_processes_are_valid(d):
    W = _random_ordered_process("A", d)
    assert np.allclose(W, W.conj().T)
    assert np.linalg.eigvalsh(W).min() > -1e-10
    # same normalisation as the switch construct: Tr W = d_{P_O} d_{A_O} d_{B_O}
    assert np.isclose(np.trace(W).real, np.trace(switch_process(d).W).real)
    assert np.isclose(np.trace(W).real, 2 * d * d)


@pytest.mark.parametrize("d", [3, 4])
def test_pulled_back_witness_nonnegative_on_random_ordered_qudit_processes(d):
    values = []
    for _ in range(15):
        values.append(_pulled_back_value(_random_ordered_process("A", d), d))
        values.append(_pulled_back_value(_random_ordered_process("B", d), d))
        q = rng.random()
        W = q * _random_ordered_process("A", d) + (1 - q) * _random_ordered_process("B", d)
        values.append(_pulled_back_value(W, d))
    assert min(values) >= -1e-9
