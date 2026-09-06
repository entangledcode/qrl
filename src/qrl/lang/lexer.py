"""Hand-rolled tokenizer for the QRL surface language.

No external dependency (the project keeps its dependency set minimal). The
grammar is small enough that a scanner in ~120 lines covers it with room for
good diagnostics.
"""

from __future__ import annotations

from .errors import QRLSyntaxError
from .tokens import KEYWORDS, Token, TokenKind

_SIMPLE = {
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    ",": TokenKind.COMMA,
    ";": TokenKind.SEMI,
    "*": TokenKind.STAR,
    "=": TokenKind.EQUALS,
    ":": TokenKind.COLON,
}

_KET_BODIES = {"0", "1", "+"}


def tokenize(src: str) -> list[Token]:
    toks: list[Token] = []
    i = 0
    line = 1
    col = 1
    n = len(src)

    def adv(k: int = 1) -> None:
        nonlocal i, line, col
        for _ in range(k):
            if i < n and src[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        c = src[i]

        # whitespace
        if c in " \t\r\n":
            adv()
            continue

        # line comment  -- ...
        if c == "-" and i + 1 < n and src[i + 1] == "-":
            while i < n and src[i] != "\n":
                adv()
            continue

        start_line, start_col = line, col

        # ket literal  |0>  |1>  |+>
        if c == "|":
            if i + 2 < n and src[i + 1] in _KET_BODIES and src[i + 2] == ">":
                text = src[i:i + 3]
                adv(3)
                toks.append(Token(TokenKind.KET, text, start_line, start_col))
                continue
            raise QRLSyntaxError(
                "expected |0>, |1> or |+>", start_line, start_col, src
            )

        # number: optional sign, digits, '.', exponent, trailing 'j',
        # and an optional  +imag j  tail so 0.5+0.5j lexes as one token.
        if c.isdigit() or (c == "-" and i + 1 < n and src[i + 1].isdigit()) \
                or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i + 1
            while j < n and (src[j].isdigit() or src[j] in ".eE"):
                if src[j] in "eE" and j + 1 < n and src[j + 1] in "+-":
                    j += 1
                j += 1
            if j < n and src[j] == "j":
                j += 1
            # complex tail:  <real>(+|-)<imag>j
            if j < n and src[j] in "+-":
                k = j + 1
                saw_digit = False
                while k < n and (src[k].isdigit() or src[k] in ".eE"):
                    saw_digit = saw_digit or src[k].isdigit()
                    k += 1
                if saw_digit and k < n and src[k] == "j":
                    j = k + 1
            text = src[i:j]
            adv(j - i)
            toks.append(Token(TokenKind.NUMBER, text, start_line, start_col))
            continue

        # identifier / keyword
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            text = src[i:j]
            adv(j - i)
            kind = KEYWORDS.get(text, TokenKind.IDENT)
            toks.append(Token(kind, text, start_line, start_col))
            continue

        # simple punctuation
        if c in _SIMPLE:
            adv()
            toks.append(Token(_SIMPLE[c], c, start_line, start_col))
            continue

        raise QRLSyntaxError(f"unexpected character {c!r}", start_line, start_col, src)

    toks.append(Token(TokenKind.EOF, "", line, col))
    return toks
