"""Tests for the QRL static type checker.

One or more tests per typing rule of quantum-causal-structure.tex Section 2.2,
plus the negative cases the rules are designed to reject.
"""

import numpy as np
import pytest

from qrl.lang import check_source, types as T
from qrl.lang.errors import QRLTypeError


def ty(src):
    return check_source(src)


# ---- basic stratum ----------------------------------------------------

def test_t_qubit():
    assert ty("|0>") == T.Qubit()


def test_t_var_and_let():
    assert ty("let x = |+> in x") == T.Qubit()


def test_t_var_unbound_rejected():
    with pytest.raises(QRLTypeError):
        ty("entangle(a, |0>)")


def test_t_let_annotation_mismatch_rejected():
    with pytest.raises(QRLTypeError):
        ty("let x : Switch(2) = |0> in x")


def test_t_let_annotation_via_subtype_ok():
    # UniProc(2) <: Proc(2, 2)
    assert ty("let p : Proc(2, 2) = cptp([[0, 1], [1, 0]]) in |0>") == T.Qubit()


# ---- relational stratum --------------------------------------------

def test_t_entangle_2():
    assert ty("entangle(|0>, |1>)") == T.Rel(2)


def test_t_entangle_n():
    assert ty("let a = |0> in entangle(a, a, a, a)") == T.Rel(4)


def test_t_entangle_non_qubit_rejected():
    with pytest.raises(QRLTypeError):
        ty("entangle(|0>, cptp([[0, 1], [1, 0]]))")


def test_t_ask():
    assert ty("ask(entangle(|0>, |1>), Z)") == T.Tensor(T.Outcome(), T.Rel(2))


def test_t_ask_requires_relation():
    with pytest.raises(QRLTypeError):
        ty("ask(|0>, Z)")


def test_t_ask_requires_question():
    with pytest.raises(QRLTypeError):
        ty("ask(entangle(|0>, |1>), |0>)")


def test_t_tensor():
    assert ty("|0> * |1>") == T.Tensor(T.Qubit(), T.Qubit())


# ---- causal stratum ----------------------------------------------

def test_t_uniproc_from_unitary():
    assert ty("cptp([[0, 1], [1, 0]])") == T.UniProc(2)


def test_t_uniproc_rejects_non_unitary_single_kraus():
    # single non-unitary Kraus operator is not trace-preserving on its own
    with pytest.raises(QRLTypeError):
        ty("cptp([[1, 0], [0, 0]])")


def test_cptp_map_types_as_proc():
    # two Kraus operators forming a valid CPTP map (dephasing)
    src = "cptp([[1, 0], [0, 0]], [[0, 0], [0, 1]])"
    assert ty(src) == T.Proc(2, 2)


def test_t_switch_coherent():
    src = """
        let X = cptp([[0, 1], [1, 0]]) in
        let Z = cptp([[1, 0], [0, -1]]) in
        switch(X, Z, |+>)
    """
    assert ty(src) == T.Switch(2)


def test_t_switch_rejects_non_unitary_channel():
    src = """
        let K = cptp([[1, 0], [0, 0]], [[0, 0], [0, 1]]) in
        switch(K, K, |+>)
    """
    with pytest.raises(QRLTypeError):
        ty(src)


def test_t_switch_rejects_dim_mismatch():
    src = """
        let X = cptp([[0, 1], [1, 0]]) in
        let Y = cptp([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) in
        switch(X, Y, |+>)
    """
    with pytest.raises(QRLTypeError):
        ty(src)


def test_t_switch_rejects_non_qubit_control():
    src = """
        let X = cptp([[0, 1], [1, 0]]) in
        switch(X, X, entangle(|0>, |1>))
    """
    with pytest.raises(QRLTypeError):
        ty(src)


def test_t_seq_composes_dims():
    src = "let X = cptp([[0, 1], [1, 0]]) in X ; X"
    assert ty(src) == T.Proc(2, 2)


def test_t_seq_dim_mismatch_rejected():
    src = """
        let X = cptp([[0, 1], [1, 0]]) in
        let Y = cptp([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) in
        X ; Y
    """
    with pytest.raises(QRLTypeError):
        ty(src)


def _pm_identity_src(n=2, d=2):
    dim = d ** (2 * n)
    w = np.eye(dim) * (d ** n / dim)
    rows = "[" + ",".join(
        "[" + ",".join(repr(float(x)) for x in row) + "]" for row in w.tolist()
    ) + "]"
    return f"pm({rows}, [0.5, 0.5])"


def test_t_pm_valid_process_matrix():
    assert ty(_pm_identity_src()) == T.PM(2, 2)


def test_t_pm_rejects_non_psd():
    # Hermitian but indefinite
    src = "pm([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [1])"
    with pytest.raises(QRLTypeError):
        ty(src)


def test_t_dag_acyclic():
    src = "dag([A, B, C], [[A, B], [C, B]], [[A, B, [[1, 0], [0, 1]]], [C, B, [[1, 0], [0, 1]]]])"
    assert ty(src) == T.CausalDAG(3)


def test_t_dag_rejects_cycle():
    with pytest.raises(QRLTypeError):
        ty("dag([A, B], [[A, B], [B, A]], [])")


def test_t_dag_rejects_missing_phi():
    with pytest.raises(QRLTypeError):
        ty("dag([A, B], [[A, B]], [])")


def test_t_do():
    src = """
        let g = dag([A, B], [[A, B]], [[A, B, [[1, 0], [0, 1]]]]) in
        do(g, B, [[1, 0], [0, 0]])
    """
    assert ty(src) == T.Outcome()


def test_t_do_rejects_non_vertex():
    src = """
        let g = dag([A, B], [[A, B]], [[A, B, [[1, 0], [0, 1]]]]) in
        do(g, Z, [[1, 0], [0, 0]])
    """
    with pytest.raises(QRLTypeError):
        ty(src)


def test_t_do_rejects_non_density_matrix():
    src = """
        let g = dag([A, B], [[A, B]], [[A, B, [[1, 0], [0, 1]]]]) in
        do(g, B, [[1, 0], [0, 1]])
    """
    with pytest.raises(QRLTypeError):
        ty(src)


# ---- soundness-relevant end-to-end ---------------------------------

def test_switch_soundness_boundary():
    """The type system's whole job: unitary -> Switch(2); CPTP -> not."""
    good = "let U = cptp([[1, 0], [0, -1]]) in switch(U, U, |0>)"
    bad = ("let U = cptp([[1, 0], [0, 0]], [[0, 1], [0, 0]]) in "
           "switch(U, U, |0>)")
    assert ty(good) == T.Switch(2)
    with pytest.raises(QRLTypeError):
        ty(bad)


def test_error_message_has_source_caret():
    try:
        check_source("switch(|0>, |0>, |+>)")
    except QRLTypeError as e:
        assert "^" in str(e)
