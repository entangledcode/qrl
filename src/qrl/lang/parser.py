"""Recursive-descent parser for the QRL surface language.

    program    := term EOF
    term       := "let" IDENT [":" typeexpr] "=" term "in" term  |  seq
    seq        := tensor (";" tensor)*
    tensor     := atom ("*" atom)*
    atom       := KET | IDENT | "(" term ")" | <builtin> "(" ... ")"
    builtin    := entangle | ask | cptp | switch | pm | dag | do

Matrix / list sub-forms are parsed by dedicated helpers.
"""

from __future__ import annotations

from typing import List, Optional

from . import ast
from .errors import QRLSyntaxError
from .lexer import tokenize
from .tokens import Token, TokenKind as K

_BUILTINS = {"entangle", "ask", "cptp", "switch", "pm", "dag", "do"}
_TYPE_NAMES = {"Qubit", "Ques", "Outcome"}
_TYPE_CTORS = {"Rel", "Switch", "UniProc", "Proc", "PM", "CausalDAG"}


def parse(src: str) -> ast.Term:
    return _Parser(src).parse_program()


def parse_type(src: str) -> ast.TypeExpr:
    p = _Parser(src)
    t = p._type_expr()
    p._expect(K.EOF)
    return t


class _Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks: List[Token] = tokenize(src)
        self.pos = 0

    # -- token helpers ----------------------------------------------------
    @property
    def cur(self) -> Token:
        return self.toks[self.pos]

    def _at(self, kind: K) -> bool:
        return self.cur.kind == kind

    def _at_ident(self, text: str) -> bool:
        return self.cur.kind == K.IDENT and self.cur.text == text

    def _advance(self) -> Token:
        t = self.cur
        if t.kind != K.EOF:
            self.pos += 1
        return t

    def _expect(self, kind: K) -> Token:
        if self.cur.kind != kind:
            raise QRLSyntaxError(
                f"expected {kind.name}, got {self.cur.kind.name} ({self.cur.text!r})",
                self.cur.line, self.cur.col, self.src,
            )
        return self._advance()

    def _err(self, msg: str) -> QRLSyntaxError:
        return QRLSyntaxError(msg, self.cur.line, self.cur.col, self.src)

    # -- entry ----------------------------------------------------------
    def parse_program(self) -> ast.Term:
        t = self._term()
        self._expect(K.EOF)
        return t

    # -- terms --------------------------------------------------------
    def _term(self) -> ast.Term:
        if self._at(K.LET):
            return self._let()
        return self._seq()

    def _let(self) -> ast.Term:
        kw = self._expect(K.LET)
        name = self._expect(K.IDENT).text
        annot: Optional[ast.TypeExpr] = None
        if self._at(K.COLON):
            self._advance()
            annot = self._type_expr()
        self._expect(K.EQUALS)
        value = self._term()
        if not self._at(K.IN):
            raise self._err("expected 'in' to close 'let'")
        self._advance()
        body = self._term()
        return ast.Let(kw.line, kw.col, name, annot, value, body)

    def _seq(self) -> ast.Term:
        node = self._tensor()
        while self._at(K.SEMI):
            op = self._advance()
            rhs = self._tensor()
            node = ast.Seq(op.line, op.col, node, rhs)
        return node

    def _tensor(self) -> ast.Term:
        node = self._atom()
        while self._at(K.STAR):
            op = self._advance()
            rhs = self._atom()
            node = ast.Tensor(op.line, op.col, node, rhs)
        return node

    def _atom(self) -> ast.Term:
        t = self.cur
        if t.kind == K.KET:
            self._advance()
            return ast.Ket(t.line, t.col, t.text[1])  # the 0/1/+ char
        if t.kind == K.LPAREN:
            self._advance()
            inner = self._term()
            self._expect(K.RPAREN)
            return inner
        if t.kind == K.IDENT:
            if t.text in _BUILTINS and self.toks[self.pos + 1].kind == K.LPAREN:
                return self._builtin(t.text)
            self._advance()
            return ast.Var(t.line, t.col, t.text)
        raise self._err(f"unexpected token {t.kind.name} ({t.text!r})")

    def _builtin(self, name: str) -> ast.Term:
        kw = self._advance()          # the builtin ident
        self._expect(K.LPAREN)
        if name == "entangle":
            args = [self._term()]
            while self._at(K.COMMA):
                self._advance()
                args.append(self._term())
            self._expect(K.RPAREN)
            return ast.Entangle(kw.line, kw.col, args)
        if name == "ask":
            rel = self._term()
            self._expect(K.COMMA)
            q = self._term()
            self._expect(K.RPAREN)
            return ast.Ask(kw.line, kw.col, rel, q)
        if name == "cptp":
            mats = [self._matrix()]
            while self._at(K.COMMA):
                self._advance()
                mats.append(self._matrix())
            self._expect(K.RPAREN)
            return ast.Cptp(kw.line, kw.col, mats)
        if name == "switch":
            f = self._term()
            self._expect(K.COMMA)
            g = self._term()
            self._expect(K.COMMA)
            c = self._term()
            self._expect(K.RPAREN)
            return ast.Switch(kw.line, kw.col, f, g, c)
        if name == "pm":
            w = self._matrix()
            self._expect(K.COMMA)
            probs = self._number_list()
            self._expect(K.RPAREN)
            return ast.PM(kw.line, kw.col, w, probs)
        if name == "dag":
            verts = self._ident_list()
            self._expect(K.COMMA)
            edges = self._edge_list()
            self._expect(K.COMMA)
            phi = self._phi_list()
            self._expect(K.RPAREN)
            return ast.DAG(kw.line, kw.col, verts, edges, phi)
        if name == "do":
            g = self._term()
            self._expect(K.COMMA)
            var = self._expect(K.IDENT).text
            self._expect(K.COMMA)
            rho = self._matrix()
            self._expect(K.RPAREN)
            return ast.Do(kw.line, kw.col, g, var, rho)
        raise self._err(f"unknown builtin {name!r}")  # pragma: no cover

    # -- literal sub-forms --------------------------------------------
    def _number(self) -> complex:
        t = self._expect(K.NUMBER)
        txt = t.text.replace("j", "j")
        try:
            return complex(txt)
        except ValueError:
            raise QRLSyntaxError(f"bad number literal {t.text!r}", t.line, t.col, self.src)

    def _number_list(self) -> List[complex]:
        self._expect(K.LBRACKET)
        out: List[complex] = []
        if not self._at(K.RBRACKET):
            out.append(self._number())
            while self._at(K.COMMA):
                self._advance()
                out.append(self._number())
        self._expect(K.RBRACKET)
        return out

    def _matrix(self) -> ast.MatrixLit:
        lb = self._expect(K.LBRACKET)
        rows: List[List[complex]] = []
        if not self._at(K.RBRACKET):
            rows.append(self._number_list())
            while self._at(K.COMMA):
                self._advance()
                rows.append(self._number_list())
        self._expect(K.RBRACKET)
        return ast.MatrixLit(lb.line, lb.col, rows)

    def _ident_list(self) -> List[str]:
        self._expect(K.LBRACKET)
        out: List[str] = []
        if not self._at(K.RBRACKET):
            out.append(self._expect(K.IDENT).text)
            while self._at(K.COMMA):
                self._advance()
                out.append(self._expect(K.IDENT).text)
        self._expect(K.RBRACKET)
        return out

    def _edge_list(self):
        self._expect(K.LBRACKET)
        out = []
        if not self._at(K.RBRACKET):
            out.append(self._pair())
            while self._at(K.COMMA):
                self._advance()
                out.append(self._pair())
        self._expect(K.RBRACKET)
        return out

    def _pair(self):
        self._expect(K.LBRACKET)
        a = self._expect(K.IDENT).text
        self._expect(K.COMMA)
        b = self._expect(K.IDENT).text
        self._expect(K.RBRACKET)
        return (a, b)

    def _phi_list(self):
        self._expect(K.LBRACKET)
        out = []
        if not self._at(K.RBRACKET):
            out.append(self._phi_entry())
            while self._at(K.COMMA):
                self._advance()
                out.append(self._phi_entry())
        self._expect(K.RBRACKET)
        return out

    def _phi_entry(self):
        self._expect(K.LBRACKET)
        a = self._expect(K.IDENT).text
        self._expect(K.COMMA)
        b = self._expect(K.IDENT).text
        self._expect(K.COMMA)
        m = self._matrix()
        self._expect(K.RBRACKET)
        return (a, b, m)

    # -- type expressions -------------------------------------------
    def _type_expr(self) -> ast.TypeExpr:
        node = self._type_atom()
        while self._at(K.STAR):
            op = self._advance()
            rhs = self._type_atom()
            node = ast.TyTensor(op.line, op.col, node, rhs)
        return node

    def _type_atom(self) -> ast.TypeExpr:
        if self._at(K.LPAREN):
            self._advance()
            inner = self._type_expr()
            self._expect(K.RPAREN)
            return inner
        t = self._expect(K.IDENT)
        if t.text in _TYPE_NAMES:
            return ast.TyName(t.line, t.col, t.text)
        if t.text in _TYPE_CTORS:
            self._expect(K.LPAREN)
            nums = [int(self._number().real)]
            while self._at(K.COMMA):
                self._advance()
                nums.append(int(self._number().real))
            self._expect(K.RPAREN)
            return self._mk_type_ctor(t, nums)
        raise QRLSyntaxError(f"unknown type {t.text!r}", t.line, t.col, self.src)

    def _mk_type_ctor(self, t: Token, nums: List[int]) -> ast.TypeExpr:
        name = t.text
        if name == "Rel":
            return ast.TyRel(t.line, t.col, nums[0])
        if name == "Switch":
            return ast.TySwitch(t.line, t.col, nums[0])
        if name == "UniProc":
            return ast.TyUniProc(t.line, t.col, nums[0])
        if name == "CausalDAG":
            return ast.TyCausalDAG(t.line, t.col, nums[0])
        if name == "Proc":
            if len(nums) != 2:
                raise QRLSyntaxError("Proc takes 2 args", t.line, t.col, self.src)
            return ast.TyProc(t.line, t.col, nums[0], nums[1])
        if name == "PM":
            if len(nums) != 2:
                raise QRLSyntaxError("PM takes 2 args", t.line, t.col, self.src)
            return ast.TyPM(t.line, t.col, nums[0], nums[1])
        raise QRLSyntaxError(f"unknown type {name!r}", t.line, t.col, self.src)  # pragma: no cover
