"""
Exact certificate for the shipped switch witness (qrl.switch_witness.verify_exact_certificate).

The check proves, in exact rational arithmetic, Tr[S W] >= 8 TAU (TAU = -2^-20)
for every causally separable process W, i.e. every process where P's output
classically controls the order of A and B.  These tests check that the exact
witness is the shipped one, that the check passes, and that it can fail.
"""

from fractions import Fraction

import numpy as np
import pytest

import qrl._switch_witness_cert as cert
from qrl.switch_witness import (
    _PAULI,
    _bareiss_positive_definite,
    _exact_witness_coeffs,
    switch_process,
    switch_witness,
    verify_exact_certificate,
)


def _matrix(coeffs):
    M = np.zeros((64, 64), dtype=complex)
    for k, v in coeffs.items():
        P = _PAULI[k[0]]
        for j in k[1:]:
            P = np.kron(P, _PAULI[j])
        M += float(v) * P
    return M


def test_exact_witness_is_the_shipped_operator():
    S_exact = _matrix(_exact_witness_coeffs())
    assert np.allclose(S_exact, switch_witness().operator(), atol=1e-12)


def test_exact_witness_is_real():
    # only Pauli strings with an even number of Y's (real matrices)
    assert all(sum(j == 2 for j in k) % 2 == 0 for k in _exact_witness_coeffs())


def test_exact_certificate_is_valid():
    r = verify_exact_certificate()
    assert r["valid"] and r["branches"] == [True, True]
    assert r["separable_bound"] == Fraction(-8, 2 ** 20)
    # the switch is far below the certified separable bound
    assert switch_witness().value(switch_process()) < float(r["separable_bound"]) - 0.16


def test_certificate_fails_for_a_bound_it_cannot_support(monkeypatch):
    # The witness's true minimum over separable processes is 0, so no certificate
    # can prove Tr[S W] >= 8 * 2^-10 > 0. The check must say so.
    monkeypatch.setattr(cert, "TAU_NUM", 1)
    monkeypatch.setattr(cert, "TAU_DEN", 2 ** 10)
    assert verify_exact_certificate()["valid"] is False


def test_certificate_fails_without_the_branch_operators(monkeypatch):
    monkeypatch.setattr(cert, "Q0", ())
    monkeypatch.setattr(cert, "Q1", ())
    assert verify_exact_certificate()["valid"] is False


def test_bareiss_positive_definite():
    assert _bareiss_positive_definite([[2, 1], [1, 2]])
    assert not _bareiss_positive_definite([[1, 2], [2, 1]])      # indefinite
    assert not _bareiss_positive_definite([[1, 1], [1, 1]])      # singular
    assert not _bareiss_positive_definite([[0, 0], [0, 1]])
    rng = np.random.default_rng(11)
    for _ in range(20):
        A = rng.integers(-5, 6, size=(6, 6))
        M = A @ A.T + np.eye(6, dtype=int)
        assert _bareiss_positive_definite(M.tolist())
        assert not _bareiss_positive_definite((M - 1000 * np.eye(6, dtype=int)).tolist())
