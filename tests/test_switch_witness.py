"""
Tests for qrl.switch_witness — the quantum switch construct as a four-party
process (P, A, B, F) and its measurable causal witness.

Reference: Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner (2015).
Witnessing causal nonseparability. New Journal of Physics 17, 102001.
arXiv:1506.03776.

Covers:
  - switch_process(): shape, validity, normalisation, purity per target branch
  - outcome probabilities match a direct simulation of the experiment,
    including complex targets, controls and unitaries (Choi convention)
  - witness value -0.1694 on the construct; real operator
  - witness >= 0 on random causally ordered processes (P<A<B<F, P<B<A<F, mixtures)
  - classical switch (dephased control wire) is not certified
  - noise tolerance, evaluate() from frequencies
"""

import numpy as np
import pytest

from qrl.causal import QuantumSwitch, cptp_from_unitary
from qrl.switch_witness import (
    QUBIT_STATES,
    WITNESS_UNITARIES,
    SwitchWitness,
    switch_process,
    switch_witness,
)

PREDICTED = -0.16940767627468822
rng = np.random.default_rng(2026)


def _haar(n):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _random_ordered_process(first="A"):
    """A random process with definite order P < first < second < F.

    P's output enters an isometry into (first's input, memory); a random
    unitary takes (first's output, memory) to (second's input, memory'); another
    takes (second's output, memory') to (F, discarded system).
    """
    V0 = _haar(4)[:, :2].reshape(2, 2, 2)            # [x_in, M, P_O]
    V1 = _haar(4).reshape(2, 2, 2, 2)                # [y_in, M', x_out, M]
    V2 = _haar(4).reshape(2, 2, 2, 2)                # [F, junk, y_out, M']
    w = np.einsum("amp,qnbm,fjcn->pabqcfj", V0, V1, V2)   # P_O x_in x_out y_in y_out F junk
    if first == "B":
        w = w.transpose(0, 3, 4, 1, 2, 5, 6)
    w = w.reshape(64, 2)
    return w @ w.conj().T


def _direct_probability(control, UA, UB, outcome, target):
    """P prepares `control`; the switch applies U_B U_A (control 0) or U_A U_B
    (control 1) to `target`; F measures the control; the target is discarded."""
    out = (control[0] * np.kron([1, 0], UB @ UA @ target)
           + control[1] * np.kron([0, 1], UA @ UB @ target))
    proj = np.kron(np.outer(outcome, outcome.conj()), np.eye(2))
    return float(np.vdot(out, proj @ out).real)


def _setting_probability(W, control, UA, UB, outcome):
    """Same probability from the process matrix (transposed-Choi convention)."""
    def choi_u(U):
        v = sum(np.kron(np.eye(2)[i], U[:, i]) for i in range(2))
        return np.outer(v, v.conj()).conj()
    M = np.kron(np.kron(np.kron(np.outer(control, control.conj()).conj(), choi_u(UA)),
                        choi_u(UB)), np.outer(outcome, outcome.conj()))
    return float(np.trace(M @ W).real)


# ------------------------------------------------------------------ #
# The process                                                          #
# ------------------------------------------------------------------ #

def test_switch_process_shape_and_parties():
    pm = switch_process()
    assert pm.W.shape == (64, 64)
    assert pm.parties == ["P", "A", "B", "F"]
    assert pm.input_dims == [1, 2, 2, 2]
    assert pm.output_dims == [2, 2, 2, 1]


def test_switch_process_is_valid():
    pm = switch_process()
    assert pm.is_valid()
    assert np.isclose(np.trace(pm.W).real, 8.0)


def test_switch_process_is_real_for_target_zero():
    assert np.abs(switch_process().W.imag).max() < 1e-15


def test_probabilities_match_direct_simulation_for_random_settings():
    """Checks the Choi convention with complex targets, controls and unitaries."""
    for _ in range(30):
        target = _haar(2)[:, 0]
        W = switch_process(target=target).W
        control, outcome = _haar(2)[:, 0], _haar(2)[:, 0]
        UA, UB = _haar(2), _haar(2)
        assert np.isclose(_setting_probability(W, control, UA, UB, outcome),
                          _direct_probability(control, UA, UB, outcome, target), atol=1e-12)


def test_probabilities_normalised_for_every_setting():
    probs = switch_witness().probabilities(switch_process())
    for p, a, b in switch_witness().settings:
        assert np.isclose(probs[(p, a, b, "+")] + probs[(p, a, b, "-")], 1.0, atol=1e-10)


# ------------------------------------------------------------------ #
# The witness on the switch construct                                  #
# ------------------------------------------------------------------ #

def test_shipped_witness_metadata():
    w = switch_witness()
    assert isinstance(w, SwitchWitness)
    assert len(w.terms) == 1200
    assert len(w.settings) == 600
    assert {t[0] for t in w.terms} == {"0", "1", "+", "-", "+i", "-i"}
    assert {t[1] for t in w.terms} == set(WITNESS_UNITARIES)
    assert {t[3] for t in w.terms} == {"+", "-"}
    assert set(QUBIT_STATES) >= {t[0] for t in w.terms}
    assert w.separable_bound == 0.0
    assert np.isclose(w.noise_tolerance, -PREDICTED)


def test_witness_value_on_switch_matches_prediction():
    w = switch_witness()
    assert np.isclose(w.value(switch_process()), PREDICTED, atol=1e-10)
    assert w.certifies(switch_process())


def test_witness_operator_is_real_and_agrees_with_probabilities():
    w = switch_witness()
    S = w.operator()
    W = switch_process().W
    assert np.abs(S.imag).max() < 1e-12
    assert np.allclose(S, S.conj().T)
    assert np.isclose(np.trace(S @ W).real, w.value(W), atol=1e-10)


def test_evaluate_from_frequencies():
    w = switch_witness()
    freqs = w.probabilities(switch_process())
    assert np.isclose(w.evaluate(freqs), PREDICTED, atol=1e-10)
    freqs.pop(next(iter(freqs)))
    with pytest.raises(KeyError):
        w.evaluate(freqs)


def test_witness_tolerates_white_noise_up_to_about_29_percent():
    w = switch_witness()
    W = switch_process().W
    noise = np.eye(64) * 8 / 64                    # white noise, trace 8
    assert w.value(0.75 * W + 0.25 * noise) < 0
    assert w.value(0.67 * W + 0.33 * noise) > 0


# ------------------------------------------------------------------ #
# The witness on causally separable processes                          #
# ------------------------------------------------------------------ #

def test_classical_switch_is_not_certified():
    """Dephase the P -> F control wire: a classical mixture of the two orders."""
    W = switch_process().W.reshape([2] * 12)
    W = np.einsum("pabcdfqghijk,pq->pabcdfqghijk", W, np.eye(2)).reshape(64, 64)
    assert switch_witness().value(W) >= 0


@pytest.mark.parametrize("first", ["A", "B"])
def test_random_ordered_processes_are_valid(first):
    W = _random_ordered_process(first)
    assert np.isclose(np.trace(W).real, 8.0)
    probs = switch_witness().probabilities(W)
    for p, a, b in switch_witness().settings[:20]:
        assert np.isclose(probs[(p, a, b, "+")] + probs[(p, a, b, "-")], 1.0, atol=1e-10)


def test_witness_nonnegative_on_random_ordered_processes():
    S = switch_witness().operator()
    for _ in range(300):
        q = rng.uniform()
        W = q * _random_ordered_process("A") + (1 - q) * _random_ordered_process("B")
        assert np.trace(S @ W).real >= -1e-9


# ------------------------------------------------------------------ #
# QuantumSwitch integration                                            #
# ------------------------------------------------------------------ #

def test_quantum_switch_certificate_is_the_shipped_witness():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    qs = QuantumSwitch(channel_A=cptp_from_unitary(X), channel_B=cptp_from_unitary(X))
    assert qs.witness_certificate() is switch_witness()
    assert np.isclose(qs.witness_value(), PREDICTED, atol=1e-10)
    assert qs.is_causally_separable() is False


def test_certificate_not_implemented_for_qutrit_target():
    U = np.eye(3, dtype=complex)
    qs = QuantumSwitch(channel_A=cptp_from_unitary(U), channel_B=cptp_from_unitary(U))
    with pytest.raises(NotImplementedError):
        qs.witness_certificate()
