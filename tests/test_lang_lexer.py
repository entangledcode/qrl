"""Tests for the QRL surface-language lexer."""

import pytest

from qrl.lang.lexer import tokenize
from qrl.lang.tokens import TokenKind as K
from qrl.lang.errors import QRLSyntaxError


def kinds(src):
    return [t.kind for t in tokenize(src) if t.kind is not K.EOF]


def test_kets():
    assert kinds("|0> |1> |+>") == [K.KET, K.KET, K.KET]


def test_line_comment_is_skipped():
    toks = tokenize("-- a comment\n|0>")
    assert [t.kind for t in toks] == [K.KET, K.EOF]
    assert toks[0].line == 2


def test_keywords_vs_idents():
    ks = kinds("let x in y")
    assert ks == [K.LET, K.IDENT, K.IN, K.IDENT]


def test_numbers_including_complex():
    toks = tokenize("1 -0.5 0.7071 1j 0.5+0.5j")
    nums = [t.text for t in toks if t.kind is K.NUMBER]
    assert nums == ["1", "-0.5", "0.7071", "1j", "0.5+0.5j"]


def test_minus_minus_not_treated_as_number():
    # "--" starts a comment even when digits follow on the next construct
    assert kinds("--3\n|1>") == [K.KET]


def test_positions_are_one_based():
    tok = tokenize("  |0>")[0]
    assert (tok.line, tok.col) == (1, 3)


def test_bad_ket_raises():
    with pytest.raises(QRLSyntaxError):
        tokenize("|9>")


def test_unexpected_char_raises():
    with pytest.raises(QRLSyntaxError):
        tokenize("@")
