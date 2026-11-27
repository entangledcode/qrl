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
    """A causal class of two-slot processes (slots A, B of dimension d) with a
    global past P and a global future F, as in Wechs et al. 2021:

        QCFO(d) <= QCCC(d) <= QCQC(d)                (certificates convert upward)
        Noncausal(d)                                 (outside QC-QC)
        Switch(d) <= QCQC(d)

    P and F may be trivial (a pm literal) or not (the switch: P, A, B, F with
    input dims [1, d, d, 2]), so a class type is NOT a subtype of PM(2, d),
    whose values have trivial P and F.  A class value enters from PM(2, d)
    only through T-Refine, which checks a certificate (qrl.causal_classes)."""
    kind: str          # "QCFO" | "QCCC" | "QCQC" | "Noncausal"
    d: int

    def __str__(self) -> str:
        return f"{self.kind}({self.d})"


_CHAIN = ("QCFO", "QCCC", "QCQC")


def _direct_supertypes(t: Type):
    """One-step subtyping edges.  is_subtype is their reflexive-transitive
    closure (plus covariant Tensor), so it is transitive by construction."""
    if isinstance(t, UniProc):
        yield Proc(t.d, t.d)
    elif isinstance(t, Switch):
        yield CausalClass("QCQC", t.d)
    elif isinstance(t, CausalClass) and t.kind in _CHAIN:
        k = _CHAIN.index(t.kind)
        if k + 1 < len(_CHAIN):
            yield CausalClass(_CHAIN[k + 1], t.d)


def is_subtype(sub: Type, sup: Type) -> bool:
    """Subtyping:

        UniProc(d)  <=  Proc(d, d)                 (a unitary channel is a channel)
        QCFO(d) <= QCCC(d) <= QCQC(d)              (certificates convert upward)
        Switch(d)  <=  QCQC(d)                     (closed-form certificate)

    No class type is a subtype of PM(2, d): see CausalClass.  Tensor is
    covariant.  The relation is the reflexive-transitive closure of
    _direct_supertypes, hence transitive.
    """
    if isinstance(sub, Tensor) and isinstance(sup, Tensor):
        return is_subtype(sub.left, sup.left) and is_subtype(sub.right, sup.right)
    seen, todo = set(), [sub]
    while todo:
        t = todo.pop()
        if t == sup:
            return True
        if t in seen:
            continue
        seen.add(t)
        todo.extend(_direct_supertypes(t))
    return False
