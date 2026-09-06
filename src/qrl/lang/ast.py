"""Abstract syntax for the QRL surface language.

Term nodes correspond 1:1 to Definition 2 of
`papers/qpl-2026/quantum-causal-structure.tex`.  Type-annotation nodes
correspond to Definition 1.  Every node keeps `(line, col)` for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------
# Type-annotation AST (surface `: T` forms; Definition 1)
# --------------------------------------------------------------------------

@dataclass
class TypeExpr:
    line: int
    col: int


@dataclass
class TyName(TypeExpr):
    """Qubit | Ques | Outcome"""
    name: str = ""


@dataclass
class TyRel(TypeExpr):
    n: int = 0


@dataclass
class TySwitch(TypeExpr):
    d: int = 0


@dataclass
class TyUniProc(TypeExpr):
    d: int = 0


@dataclass
class TyProc(TypeExpr):
    d_in: int = 0
    d_out: int = 0


@dataclass
class TyPM(TypeExpr):
    n: int = 0
    d: int = 0


@dataclass
class TyCausalDAG(TypeExpr):
    n: int = 0


@dataclass
class TyTensor(TypeExpr):
    left: TypeExpr = None
    right: TypeExpr = None


# --------------------------------------------------------------------------
# Term AST (Definition 2)
# --------------------------------------------------------------------------

@dataclass
class Term:
    line: int
    col: int


@dataclass
class Var(Term):
    name: str = ""


@dataclass
class Ket(Term):
    """|0> |1> |+>"""
    value: str = ""


@dataclass
class Entangle(Term):
    args: List[Term] = field(default_factory=list)


@dataclass
class Ask(Term):
    relation: Term = None
    question: Term = None


@dataclass
class Let(Term):
    name: str = ""
    annot: Optional[TypeExpr] = None
    value: Term = None
    body: Term = None


@dataclass
class Cptp(Term):
    """cptp(K1, ..., Km) — Kraus operators as matrix literals."""
    kraus: List["MatrixLit"] = field(default_factory=list)


@dataclass
class Seq(Term):
    left: Term = None
    right: Term = None


@dataclass
class Tensor(Term):
    left: Term = None
    right: Term = None


@dataclass
class Switch(Term):
    f: Term = None
    g: Term = None
    control: Term = None


@dataclass
class PM(Term):
    """pm(W, p) — W a matrix literal, p a bracketed list of numbers."""
    w: "MatrixLit" = None
    probs: List[complex] = field(default_factory=list)


@dataclass
class DAG(Term):
    """dag(V, E, Phi):
      V  = [name, ...]
      E  = [[src, dst], ...]
      Phi = [[src, dst, <matrix>], ...]   (edge -> CPTP map)
    """
    vertices: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    phi: List[Tuple[str, str, "MatrixLit"]] = field(default_factory=list)


@dataclass
class Do(Term):
    graph: Term = None
    var: str = ""
    rho: "MatrixLit" = None


@dataclass
class MatrixLit(Term):
    """[[a, b], [c, d]] — rows of complex entries."""
    rows: List[List[complex]] = field(default_factory=list)
