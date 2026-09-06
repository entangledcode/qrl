"""Tests for the QRL surface-language interpreter (session A: basic + relational).

Covers E-Bell / E-GHZ / E-Ask / T-Tensor and the run / distribution API.
"""

import numpy as np
import pytest

from qrl.lang import run, distribution
from qrl.lang.interp import Outcome, Pair, QState
from qrl.lang.errors import QRLRuntimeError


# ---- basic stratum ----------------------------------------------------

def test_ket_evaluates_to_density_matrix():
    v = run("|0>")
    assert isinstance(v, QState) and v.n == 1
    assert np.allclose(v.rho, [[1, 0], [0, 0]])


def test_ket_plus():
    v = run("|+>")
    assert np.allclose(v.rho, [[0.5, 0.5], [0.5, 0.5]])


def test_let_binding():
    v = run("let x = |1> in x")
    assert np.allclose(v.rho, [[0, 0], [0, 1]])


# ---- relational stratum --------------------------------------------

def test_entangle_produces_bell_state_ignoring_arguments():
    # E-Bell: the argument states are not used
    a = run("entangle(|0>, |0>)")
    b = run("entangle(|1>, |+>)")
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    ref = np.outer(bell, bell.conj())
    assert np.allclose(a.rho, ref)
    assert np.allclose(b.rho, ref)


def test_entangle_ghz():
    v = run("let a = |0> in entangle(a, a, a)")
    assert v.n == 3
    ghz = np.zeros(8, dtype=complex)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    assert np.allclose(v.rho, np.outer(ghz, ghz.conj()))


def test_ask_returns_outcome_and_relation():
    v = run("ask(entangle(|0>, |1>), Z)", seed=0)
    assert isinstance(v, Pair)
    assert isinstance(v.fst, Outcome) and v.fst.m in (0, 1)
    assert isinstance(v.snd, QState) and v.snd.n == 2


def test_ask_collapses_bell_to_correlated_outcome():
    # measuring qubit 0 of a Bell pair in Z leaves |00> or |11>
    for seed in range(8):
        v = run("ask(entangle(|0>, |1>), Z)", seed=seed)
        m = v.fst.m
        expected = np.zeros(4, dtype=complex)
        expected[0 if m == 0 else 3] = 1.0
        assert np.allclose(v.snd.rho, np.outer(expected, expected))


def test_tensor_of_states_is_kron():
    v = run("|0> * |1>")
    assert v.n == 2
    assert np.allclose(np.diag(v.rho), [0, 1, 0, 0])


# ---- run / distribution API -------------------------------------

def test_seed_makes_runs_reproducible():
    src = "ask(entangle(|0>, |1>), Z)"
    a = [r.fst.m for r in run(src, seed=99, shots=20)]
    b = [r.fst.m for r in run(src, seed=99, shots=20)]
    assert a == b


def test_shots_returns_a_list():
    out = run("ask(entangle(|0>, |1>), Z)", seed=1, shots=5)
    assert isinstance(out, list) and len(out) == 5


def test_shots_one_returns_single_value():
    out = run("|0>", shots=1)
    assert isinstance(out, QState)


def test_distribution_is_exact_and_sums_to_one():
    d = distribution("ask(entangle(|0>, |1>), Z)")
    assert set(d) == {0, 1}
    assert abs(d[0] - 0.5) < 1e-12 and abs(sum(d.values()) - 1.0) < 1e-12


def test_distribution_through_let_bindings():
    d = distribution("let a = |0> in let b = |1> in ask(entangle(a, b), X)")
    assert abs(d[0] - 0.5) < 1e-12


def test_distribution_rejects_non_ask_program():
    with pytest.raises(QRLRuntimeError):
        distribution("|0>")


def test_sampled_frequencies_track_the_distribution():
    src = "ask(entangle(|0>, |1>), Z)"
    n = 4000
    ones = sum(r.fst.m for r in run(src, seed=2024, shots=n))
    assert abs(ones / n - 0.5) < 0.05


# ---- causal stratum, unitary path (session B) ----------------------

from qrl.lang.interp import Channel, Process


def test_cptp_of_unitary_is_a_unitary_channel():
    v = run("cptp([[0, 1], [1, 0]])")
    assert isinstance(v, Channel) and v.is_unitary and v.cptp.input_dim == 2


def test_cptp_of_kraus_map_is_a_nonunitary_channel():
    v = run("cptp([[1, 0], [0, 0]], [[0, 0], [0, 1]])")
    assert isinstance(v, Channel) and not v.is_unitary


def test_seq_composes_channels():
    v = run("let X = cptp([[0, 1], [1, 0]]) in X ; X")   # X ; X == identity
    assert isinstance(v, Channel)
    out = v.cptp.apply(np.array([[1, 0], [0, 0]], dtype=complex))
    assert np.allclose(out, [[1, 0], [0, 0]])


def test_switch_reproduces_ocb_causal_game_advantage():
    v = run(open("examples/lang/switch.qrl").read())
    assert isinstance(v, Process)
    assert abs(v.p_win - (2 + np.sqrt(2)) / 4) < 1e-12
    assert abs(v.robustness - (np.sqrt(2) - 1)) < 1e-9


def test_switch_matches_the_python_api_differentially():
    from qrl.causal import QuantumSwitch, cptp_from_unitary
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    qs = QuantumSwitch(channel_A=cptp_from_unitary(X), channel_B=cptp_from_unitary(Z))
    src = ("let X = cptp([[0, 1], [1, 0]]) in "
           "let Z = cptp([[1, 0], [0, -1]]) in switch(X, Z, |+>)")
    v = run(src)
    assert np.isclose(v.p_win, qs.causal_inequality_value())
    assert np.allclose(v.pm.W, qs.process_matrix().W)


def test_switch_is_causally_nonseparable():
    v = run("let U = cptp([[1, 0], [0, -1]]) in switch(U, U, |0>)")
    assert v.robustness > 0  # r* > 0  <=>  causally nonseparable


# ---- boundaries ---------------------------------------------------

def test_remaining_causal_forms_not_yet_implemented():
    with pytest.raises(QRLRuntimeError):
        run("dag([A, B], [[A, B]], [[A, B, [[1, 0], [0, 1]]]])")


def test_ill_typed_program_is_rejected_before_running():
    from qrl.lang.errors import QRLTypeError
    with pytest.raises(QRLTypeError):
        run("ask(|0>, Z)")
