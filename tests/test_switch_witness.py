"""
Tests for qrl.switch_witness — the quantum switch as a tripartite process and
its measurable causal witness.

Reference: Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner (2015).
Witnessing causal nonseparability. New Journal of Physics 17, 102001.
arXiv:1506.03776, Sections IV-V.

Covers:
  - switch_process(): shape, validity, normalisation, dependence on control
  - outcome probabilities match a direct simulation of the switch
  - witness value -0.1507 on the switch, for any target state
  - witness >= 0 on random causally ordered processes (A<B<F, B<A<F, mixtures)
  - noise tolerance, evaluate() from frequencies, the witness operator
"""

import itertools

import numpy as np
import pytest

from qrl.causal import QuantumSwitch, cptp_from_unitary
from qrl.switch_witness import (
    CONTROL_OUTCOMES,
    WITNESS_UNITARIES,
    SwitchWitness,
    switch_process,
    switch_witness,
)

PREDICTED = -0.1507112794795265
rng = np.random.default_rng(2026)


def _haar(n):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _random_ordered_process(first="A"):
    """A random pure process with definite order first < second < F.

    The target (with a qubit memory M) enters the first party; a random
    unitary takes (first's output, M) to (second's input, M'); another takes
    (second's output, M') to F's 4-dimensional input.
    """
    psi = _haar(4)[:, 0].reshape(2, 2)               # [in_1, M]
    V1 = _haar(4).reshape(2, 2, 2, 2)                # [in_2, M', out_1, M]
    V2 = _haar(4).reshape(4, 2, 2)                   # [F, out_2, M']
    # w[in1, out1, in2, out2, F]
    w = np.einsum("am,pqbm,fcq->abpcf", psi, V1, V2)
    if first == "B":                                  # relabel: B acts first
        w = w.transpose(2, 3, 0, 1, 4)
    w = w.reshape(-1)
    return np.outer(w, w.conj())


# ------------------------------------------------------------------ #
# The process                                                          #
# ------------------------------------------------------------------ #

def test_switch_process_shape_and_parties():
    pm = switch_process()
    assert pm.W.shape == (64, 64)
    assert pm.parties == ["A", "B", "F"]
    assert pm.input_dims == [2, 2, 4]
    assert pm.output_dims == [2, 2, 1]


def test_switch_process_is_valid_and_pure():
    pm = switch_process()
    assert pm.is_valid()
    eigs = pm.eigenvalues()
    assert np.sum(eigs > 1e-8) == 1


def test_probabilities_normalised_for_every_setting():
    probs = switch_witness().probabilities(switch_process())
    for a, b in switch_witness().settings:
        assert np.isclose(probs[(a, b, "+")] + probs[(a, b, "-")], 1.0, atol=1e-10)


@pytest.mark.parametrize("a,b", [("X", "Z"), ("(X+Y)/√2", "Y"), ("(Y-Z)/√2", "(X+Z)/√2")])
def test_probabilities_match_direct_simulation(a, b):
    """p(+) = (1 + Re<0|(U_B U_A)† U_A U_B|0>) / 2 for control |+>, target |0>."""
    UA, UB = WITNESS_UNITARIES[a], WITNESS_UNITARIES[b]
    zero = np.array([1, 0], dtype=complex)
    direct = (1 + np.real(np.vdot(UB @ UA @ zero, UA @ UB @ zero))) / 2
    p = switch_witness().probabilities(switch_process())[(a, b, "+")]
    assert np.isclose(p, direct, atol=1e-10)


# ------------------------------------------------------------------ #
# The witness on the switch                                            #
# ------------------------------------------------------------------ #

def test_shipped_witness_metadata():
    w = switch_witness()
    assert isinstance(w, SwitchWitness)
    assert len(w.terms) == 200
    assert len(w.settings) == 100
    assert {t[2] for t in w.terms} == set(CONTROL_OUTCOMES)
    assert w.separable_bound == 0.0
    assert np.isclose(w.noise_tolerance, -PREDICTED)


def test_witness_value_on_switch_matches_prediction():
    w = switch_witness()
    assert np.isclose(w.value(switch_process()), PREDICTED, atol=1e-10)
    assert w.certifies(switch_process())


def test_witness_value_independent_of_target_state():
    for _ in range(5):
        target = _haar(2)[:, 0]
        assert np.isclose(switch_witness().value(switch_process(target=target)),
                          PREDICTED, atol=1e-10)


def test_operator_agrees_with_probabilities():
    w = switch_witness()
    W = switch_process().W
    assert np.isclose(np.trace(w.operator() @ W).real, w.value(W), atol=1e-10)


def test_evaluate_from_frequencies():
    w = switch_witness()
    freqs = w.probabilities(switch_process())
    assert np.isclose(w.evaluate(freqs), PREDICTED, atol=1e-10)
    freqs.pop(next(iter(freqs)))
    with pytest.raises(KeyError):
        w.evaluate(freqs)


def test_witness_tolerates_white_noise_up_to_about_a_quarter():
    w = switch_witness()
    W = switch_process().W
    noise = np.eye(64) / 16                      # 1/(d_AI d_BI d_FI), trace 4
    assert w.value(0.8 * W + 0.2 * noise) < 0
    assert w.value(0.7 * W + 0.3 * noise) > 0


# ------------------------------------------------------------------ #
# The witness on causally separable processes                          #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("control", [[1, 0], [0, 1], np.eye(2) / 2])
def test_definite_control_is_not_certified(control):
    """No coherence in the control: definite order, witness >= 0."""
    assert switch_witness().value(switch_process(control=control)) >= -1e-10


@pytest.mark.parametrize("first", ["A", "B"])
def test_random_ordered_processes_are_valid(first):
    W = _random_ordered_process(first)
    probs = switch_witness().probabilities(W)
    assert np.isclose(np.trace(W).real, 4.0)
    for a, b in itertools.islice(switch_witness().settings, 10):
        assert np.isclose(probs[(a, b, "+")] + probs[(a, b, "-")], 1.0, atol=1e-10)


def test_witness_nonnegative_on_random_ordered_processes():
    w = switch_witness()
    for _ in range(40):
        q = rng.uniform()
        W = q * _random_ordered_process("A") + (1 - q) * _random_ordered_process("B")
        assert w.value(W) >= -1e-9


# ------------------------------------------------------------------ #
# QuantumSwitch integration                                            #
# ------------------------------------------------------------------ #

def test_quantum_switch_certificate_is_the_shipped_witness():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    qs = QuantumSwitch(channel_A=cptp_from_unitary(X), channel_B=cptp_from_unitary(X))
    assert qs.witness_certificate() is switch_witness()
    assert qs.is_causally_separable() is False


def test_certificate_not_implemented_for_qutrit_target():
    U = np.eye(3, dtype=complex)
    qs = QuantumSwitch(channel_A=cptp_from_unitary(U), channel_B=cptp_from_unitary(U))
    with pytest.raises(NotImplementedError):
        qs.witness_certificate()
