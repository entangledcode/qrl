"""Token definitions for the QRL surface language.

The lexer (`qrl.lang.lexer`) turns source text into a flat list of these.
Every token carries a 1-based line/column so the parser and type checker can
point a caret at the offending span (`qrl.lang.errors`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenKind(Enum):
    # literals / names
    IDENT = auto()        # x, U, Z, Qubit, Switch, ...
    NUMBER = auto()       # 1, -0.5, 0.7071, 1j, 0.5+0.5j  (lexed whole)
    KET = auto()          # |0>  |1>  |+>

    # keywords
    LET = auto()
    IN = auto()

    # punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    SEMI = auto()          # ; sequential composition
    STAR = auto()          # * tensor
    EQUALS = auto()        # =
    COLON = auto()         # : type annotation

    EOF = auto()


KEYWORDS = {
    "let": TokenKind.LET,
    "in": TokenKind.IN,
}


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    line: int
    col: int

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Token({self.kind.name}, {self.text!r}, {self.line}:{self.col})"
