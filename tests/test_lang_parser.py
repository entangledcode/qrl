"""Tests for the QRL surface-language parser."""

import pytest

from qrl.lang import ast
from qrl.lang.parser import parse, parse_type
from qrl.lang.errors import QRLSyntaxError


def test_ket_atom():
    t = parse("|+>")
    assert isinstance(t, ast.Ket) and t.value == "+"


def test_let_with_annotation():
    t = parse("let x : Qubit = |0> in x")
    assert isinstance(t, ast.Let)
    assert t.name == "x"
    assert isinstance(t.annot, ast.TyName) and t.annot.name == "Qubit"
    assert isinstance(t.body, ast.Var)


def test_entangle_nary():
    t = parse("entangle(|0>, |1>, |+>)")
    assert isinstance(t, ast.Entangle) and len(t.args) == 3


def test_ask_shape():
    t = parse("ask(entangle(|0>, |1>), Z)")
    assert isinstance(t, ast.Ask)
    assert isinstance(t.relation, ast.Entangle)
    assert isinstance(t.question, ast.Var) and t.question.name == "Z"


def test_seq_is_left_associative():
    t = parse("a ; b ; c")
    assert isinstance(t, ast.Seq)
    assert isinstance(t.left, ast.Seq)          # (a ; b) ; c
    assert isinstance(t.right, ast.Var) and t.right.name == "c"


def test_tensor_binds_tighter_than_seq():
    t = parse("a * b ; c")
    assert isinstance(t, ast.Seq)
    assert isinstance(t.left, ast.Tensor)


def test_cptp_matrix_literal():
    t = parse("cptp([[0, 1], [1, 0]])")
    assert isinstance(t, ast.Cptp)
    assert t.kraus[0].rows == [[0j, 1 + 0j], [1 + 0j, 0j]]


def test_switch_three_args():
    t = parse("switch(f, g, |0>)")
    assert isinstance(t, ast.Switch)
    assert isinstance(t.control, ast.Ket)


def test_dag_structure():
    t = parse("dag([A, B], [[A, B]], [[A, B, [[1, 0], [0, 1]]]])")
    assert isinstance(t, ast.DAG)
    assert t.vertices == ["A", "B"]
    assert t.edges == [("A", "B")]
    assert t.phi[0][0:2] == ("A", "B")


def test_dag_empty_edge_and_phi_lists():
    t = parse("dag([A], [], [])")
    assert t.edges == [] and t.phi == []


def test_do_form():
    t = parse("do(g, B, [[1, 0], [0, 0]])")
    assert isinstance(t, ast.Do) and t.var == "B"


def test_parens_group():
    t = parse("(a ; b)")
    assert isinstance(t, ast.Seq)


def test_type_expr_tensor():
    te = parse_type("Outcome * Rel(2)")
    assert isinstance(te, ast.TyTensor)


def test_type_proc_two_args():
    te = parse_type("Proc(2, 3)")
    assert isinstance(te, ast.TyProc) and (te.d_in, te.d_out) == (2, 3)


def test_trailing_junk_rejected():
    with pytest.raises(QRLSyntaxError):
        parse("|0> |1>")


def test_missing_in_rejected():
    with pytest.raises(QRLSyntaxError):
        parse("let x = |0>")


def test_error_has_caret():
    try:
        parse("let x = |0>")
    except QRLSyntaxError as e:
        assert "^" in str(e)
