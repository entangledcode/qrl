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


@dataclass(frozen=True)
class CausalClass(Type):
    """A refinement of two-slot process matrices (slots of dimension d):

        QCFO(d) <= QCCC(d) <= QCQC(d) <= PM(2, d)    (Wechs et al. 2021)
        Noncausal(d) <= PM(2, d)                     (outside QC-QC)

    A value gets one of these types only with a certificate that checks
    (qrl.causal_classes)."""
    kind: str          # "QCFO" | "QCCC" | "QCQC" | "Noncausal"
    d: int

    def __str__(self) -> str:
        return f"{self.kind}({self.d})"


_CHAIN = ("QCFO", "QCCC", "QCQC")


def is_subtype(sub: Type, sup: Type) -> bool:
    """Subtyping:

        UniProc(d)  <=  Proc(d, d)                 (a unitary channel is a channel)
        QCFO(d) <= QCCC(d) <= QCQC(d) <= PM(2, d)  (certificates convert upward)
        Noncausal(d)  <=  PM(2, d)
        Switch(d)  <=  QCQC(d)                     (closed-form certificate)

    Tensor is covariant.
    """
    if sub == sup:
        return True
    if isinstance(sub, UniProc) and isinstance(sup, Proc):
        return sup.d_in == sub.d and sup.d_out == sub.d
    if isinstance(sub, CausalClass) and isinstance(sup, PM):
        return sup.n == 2 and sup.d == sub.d
    if isinstance(sub, CausalClass) and isinstance(sup, CausalClass):
        return (sub.d == sup.d and sub.kind in _CHAIN and sup.kind in _CHAIN
                and _CHAIN.index(sub.kind) <= _CHAIN.index(sup.kind))
    if isinstance(sub, Switch) and isinstance(sup, CausalClass):
        return sup.kind == "QCQC" and sup.d == sub.d
    if isinstance(sub, Tensor) and isinstance(sup, Tensor):
        return is_subtype(sub.left, sup.left) and is_subtype(sub.right, sup.right)
    return False
