"""QRL surface language: lexer, parser, and static type checker.

Public API:

    from qrl.lang import parse, check, check_source

    ty = check_source('''
        let U = cptp([[0, 1], [1, 0]]) in
        let V = cptp([[1, 0], [0, -1]]) in
        switch(U, V, |+>)
    ''')
    #  -> Switch(2)

The thirteen typing rules implemented here are exactly those of
`papers/qpl-2026/quantum-causal-structure.tex`, Section 2.2.
"""

from __future__ import annotations

from . import ast, types
from .errors import QRLError, QRLSyntaxError, QRLTypeError
from .lexer import tokenize
from .parser import parse, parse_type
from .typecheck import check_program as check

__all__ = [
    "ast",
    "types",
    "tokenize",
    "parse",
    "parse_type",
    "check",
    "check_source",
    "QRLError",
    "QRLSyntaxError",
    "QRLTypeError",
]


def check_source(src: str):
    """Parse and type-check QRL source, returning its `qrl.lang.types.Type`.

    On failure, raises a `QRLError` whose message includes a source caret.
    """
    try:
        return check(parse(src))
    except QRLError as e:
        raise e.with_source(src) from None
