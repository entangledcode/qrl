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


# ---- session-A boundary --------------------------------------------

def test_causal_stratum_not_yet_implemented():
    with pytest.raises(QRLRuntimeError):
        run("let U = cptp([[0, 1], [1, 0]]) in switch(U, U, |+>)")


def test_ill_typed_program_is_rejected_before_running():
    from qrl.lang.errors import QRLTypeError
    with pytest.raises(QRLTypeError):
        run("ask(|0>, Z)")
