import pytest

from qrl.lang.errors import QRLError, QRLSyntaxError, QRLTypeError, QRLRuntimeError


def test_qrl_error_construction():
    err = QRLError("test message", 1, 1)
    assert err.message == "test message"
    assert err.line == 1
    assert err.col == 1
    assert err.src is None


def test_qrl_error_kind():
    assert QRLError("test", 1, 1).kind == "error"


def test_qrl_error_render_no_src():
    err = QRLError("test message", 1, 1)
    assert err._render() == "error: test message (line 1, col 1)"


def test_qrl_error_render_with_src():
    src = "line1\nline2"
    err = QRLError("test message", 2, 3, src)
    expected = "error: test message (line 2, col 3)\n   2 | line2\n         ^"
    assert err._render() == expected


def test_qrl_error_render_out_of_range_line():
    src = "line1\nline2"
    err = QRLError("test message", 3, 1, src)
    assert err._render() == "error: test message (line 3, col 1)"


def test_qrl_error_with_source():
    err = QRLError("test message", 1, 1)
    new_err = err.with_source("new source")
    assert isinstance(new_err, QRLError)
    assert new_err.src == "new source"


def test_qrl_syntax_error_kind():
    assert QRLSyntaxError("test", 1, 1).kind == "syntax error"


def test_qrl_type_error_kind():
    assert QRLTypeError("test", 1, 1).kind == "type error"


def test_qrl_runtime_error_kind():
    assert QRLRuntimeError("test", 1, 1).kind == "runtime error"
