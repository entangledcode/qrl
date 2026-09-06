"""Semantic types for the QRL surface language (Definition 1 of the paper).

These are the objects the type checker computes and compares.  They are distinct
from the `TypeExpr` AST in `ast.py`, which is only the parsed surface annotation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Type:
    def __str__(self) -> str:  # pragma: no cover - overridden
        return self.__class__.__name__


@dataclass(frozen=True)
class Qubit(Type):
    def __str__(self) -> str:
        return "Qubit"


@dataclass(frozen=True)
class Ques(Type):
    def __str__(self) -> str:
        return "Ques"


@dataclass(frozen=True)
class Outcome(Type):
    def __str__(self) -> str:
        return "Outcome"


@dataclass(frozen=True)
class Rel(Type):
    n: int

    def __str__(self) -> str:
        return f"Rel({self.n})"


@dataclass(frozen=True)
class Tensor(Type):
    left: Type
    right: Type

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass(frozen=True)
class Proc(Type):
    d_in: int
    d_out: int

    def __str__(self) -> str:
        return f"Proc({self.d_in}, {self.d_out})"


@dataclass(frozen=True)
class UniProc(Type):
    d: int

    def __str__(self) -> str:
        return f"UniProc({self.d})"


@dataclass(frozen=True)
class PM(Type):
    n: int
    d: int

    def __str__(self) -> str:
        return f"PM({self.n}, {self.d})"


@dataclass(frozen=True)
class Switch(Type):
    d: int

    def __str__(self) -> str:
        return f"Switch({self.d})"


@dataclass(frozen=True)
class CausalDAG(Type):
    n: int

    def __str__(self) -> str:
        return f"CausalDAG({self.n})"


def is_subtype(sub: Type, sup: Type) -> bool:
    """Subtyping.  The only non-reflexive rule in the paper is

        UniProc(d)  <=  Proc(d, d)

    (a unitary channel is a channel).  Tensor is covariant.
    """
    if sub == sup:
        return True
    if isinstance(sub, UniProc) and isinstance(sup, Proc):
        return sup.d_in == sub.d and sup.d_out == sub.d
    if isinstance(sub, Tensor) and isinstance(sup, Tensor):
        return is_subtype(sub.left, sup.left) and is_subtype(sub.right, sup.right)
    return False
