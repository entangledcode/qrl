"""Static type checker for the QRL surface language.

Implements the thirteen typing rules of
`papers/qpl-2026/quantum-causal-structure.tex` (Section 2.2):

    basic       T-Var  T-Qubit  T-Let
    relational  T-Entangle-2  T-Entangle-N  T-Ask  T-Tensor
    causal      T-UniProc  T-Switch  T-Seq  T-PM  T-DAG  T-Do

The numeric side-conditions (U†U = I for T-UniProc, W >= 0 for T-PM,
acyclicity for T-DAG) are checked with numpy at tolerance 1e-9.  The full
Araujo structural-projector validity of a process matrix (Tr_out W = I_in as an
operator identity) is delegated to `qrl.causal.ProcessMatrix` at run time; the
checker verifies Hermiticity, positive-semidefiniteness and the process-matrix
trace normalisation Tr[W] = d**n.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import ast
from . import types as T
from .errors import QRLTypeError

_TOL = 1e-9

# Observable names available in the initial environment (typed Ques).
_OBSERVABLES = {"Z", "X", "Y", "H", "S", "T"}


class Env:
    def __init__(self, parent: Optional["Env"] = None):
        self.vars: Dict[str, T.Type] = {}
        # vertex sets for CausalDAG-typed bindings, for T-Do membership checks
        self.dag_vertices: Dict[str, List[str]] = {}
        self.parent = parent

    def lookup(self, name: str) -> Optional[T.Type]:
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def lookup_vertices(self, name: str) -> Optional[List[str]]:
        if name in self.dag_vertices:
            return self.dag_vertices[name]
        if self.parent:
            return self.parent.lookup_vertices(name)
        return None

    def child(self) -> "Env":
        return Env(self)


def check_program(term: ast.Term) -> T.Type:
    env = Env()
    for name in _OBSERVABLES:
        env.vars[name] = T.Ques()
    return _check(term, env)


# --------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------

def _matrix(node: ast.MatrixLit, where: str) -> np.ndarray:
    if not node.rows or any(len(r) != len(node.rows[0]) for r in node.rows):
        raise QRLTypeError(f"{where}: matrix rows have unequal length",
                           node.line, node.col)
    return np.array(node.rows, dtype=complex)


def _is_square(m: np.ndarray) -> bool:
    return m.ndim == 2 and m.shape[0] == m.shape[1]


def _is_unitary(m: np.ndarray) -> bool:
    if not _is_square(m):
        return False
    d = m.shape[0]
    return np.allclose(m.conj().T @ m, np.eye(d), atol=1e-6)


def _is_hermitian(m: np.ndarray) -> bool:
    return _is_square(m) and np.allclose(m, m.conj().T, atol=_TOL)


def _is_psd(m: np.ndarray) -> bool:
    if not _is_hermitian(m):
        return False
    w = np.linalg.eigvalsh((m + m.conj().T) / 2)
    return bool(np.all(w >= -_TOL))


def _trace_preserving(kraus: List[np.ndarray]) -> bool:
    d = kraus[0].shape[1]
    acc = np.zeros((d, d), dtype=complex)
    for k in kraus:
        acc += k.conj().T @ k
    # looser tolerance: hand-written Kraus matrices carry truncated decimals
    return np.allclose(acc, np.eye(d), atol=1e-6)


# --------------------------------------------------------------------------
# the rules
# --------------------------------------------------------------------------

def _check(t: ast.Term, env: Env) -> T.Type:
    m = getattr(_Rules, type(t).__name__, None)
    if m is None:  # pragma: no cover
        raise QRLTypeError(f"no typing rule for {type(t).__name__}", t.line, t.col)
    return m(t, env)


class _Rules:
    # ---- basic stratum ------------------------------------------------
    @staticmethod
    def Var(t: ast.Var, env: Env) -> T.Type:                 # T-Var
        ty = env.lookup(t.name)
        if ty is None:
            raise QRLTypeError(f"unbound variable {t.name!r}", t.line, t.col)
        return ty

    @staticmethod
    def Ket(t: ast.Ket, env: Env) -> T.Type:                 # T-Qubit
        return T.Qubit()

    @staticmethod
    def Let(t: ast.Let, env: Env) -> T.Type:                 # T-Let
        vty = _check(t.value, env)
        if t.annot is not None:
            want = _resolve_type(t.annot)
            if not T.is_subtype(vty, want):
                raise QRLTypeError(
                    f"annotation mismatch: {t.name!r} declared {want}, "
                    f"bound value has type {vty}",
                    t.annot.line, t.annot.col,
                )
            vty = want
        inner = env.child()
        inner.vars[t.name] = vty
        if isinstance(t.value, ast.DAG):
            inner.dag_vertices[t.name] = list(t.value.vertices)
        return _check(t.body, inner)

    # ---- relational stratum ----------------------------------------
    @staticmethod
    def Entangle(t: ast.Entangle, env: Env) -> T.Type:       # T-Entangle-2 / -N
        n = len(t.args)
        if n < 2:
            raise QRLTypeError("entangle needs at least 2 subsystems",
                               t.line, t.col)
        for i, a in enumerate(t.args):
            aty = _check(a, env)
            if not isinstance(aty, T.Qubit):
                raise QRLTypeError(
                    f"entangle argument {i + 1} has type {aty}, expected Qubit",
                    a.line, a.col,
                )
        return T.Rel(n)

    @staticmethod
    def Ask(t: ast.Ask, env: Env) -> T.Type:                 # T-Ask
        rty = _check(t.relation, env)
        if not isinstance(rty, T.Rel):
            raise QRLTypeError(f"ask expects a relation, got {rty}",
                               t.relation.line, t.relation.col)
        qty = _check(t.question, env)
        if not isinstance(qty, T.Ques):
            raise QRLTypeError(f"ask expects a question (Ques), got {qty}",
                               t.question.line, t.question.col)
        return T.Tensor(T.Outcome(), T.Rel(rty.n))

    @staticmethod
    def Tensor(t: ast.Tensor, env: Env) -> T.Type:           # T-Tensor
        return T.Tensor(_check(t.left, env), _check(t.right, env))

    # ---- causal stratum ------------------------------------------
    @staticmethod
    def Cptp(t: ast.Cptp, env: Env) -> T.Type:               # T-UniProc (+ Proc)
        mats = [_matrix(k, "cptp") for k in t.kraus]
        d0 = mats[0].shape
        for k in mats:
            if not _is_square(k) or k.shape != d0:
                raise QRLTypeError("cptp: all Kraus operators must be square "
                                   "and the same dimension", t.line, t.col)
        d = d0[0]
        if len(mats) == 1 and _is_unitary(mats[0]):
            return T.UniProc(d)                              # T-UniProc
        if not _trace_preserving(mats):
            raise QRLTypeError(
                "cptp: not trace-preserving (sum_i K_i† K_i != I) — "
                "cannot type as a channel", t.line, t.col,
            )
        return T.Proc(d, d)

    @staticmethod
    def Switch(t: ast.Switch, env: Env) -> T.Type:           # T-Switch
        fty = _check(t.f, env)
        gty = _check(t.g, env)
        for label, ty, node in (("first", fty, t.f), ("second", gty, t.g)):
            if isinstance(ty, T.UniProc):
                continue
            if isinstance(ty, T.Proc):
                raise QRLTypeError(
                    f"switch: {label} channel has type {ty}, but T-Switch "
                    f"requires UniProc(d) — a switch over a non-unitary channel "
                    f"reduces to an incoherent mixture of causal orders "
                    f"(causally separable) and cannot inhabit Switch(d)",
                    node.line, node.col,
                )
            raise QRLTypeError(
                f"switch: {label} channel has type {ty}, expected UniProc(d)",
                node.line, node.col,
            )
        if fty.d != gty.d:
            raise QRLTypeError(
                f"switch: channel dimensions differ ({fty.d} vs {gty.d})",
                t.line, t.col,
            )
        cty = _check(t.control, env)
        if not isinstance(cty, T.Qubit):
            raise QRLTypeError(f"switch: control has type {cty}, expected Qubit",
                               t.control.line, t.control.col)
        return T.Switch(fty.d)

    @staticmethod
    def Seq(t: ast.Seq, env: Env) -> T.Type:                 # T-Seq
        lty = _as_proc(_check(t.left, env), t.left)
        rty = _as_proc(_check(t.right, env), t.right)
        if lty.d_out != rty.d_in:
            raise QRLTypeError(
                f"sequential composition: {lty} then {rty} — "
                f"output dim {lty.d_out} != input dim {rty.d_in}",
                t.line, t.col,
            )
        return T.Proc(lty.d_in, rty.d_out)

    @staticmethod
    def PM(t: ast.PM, env: Env) -> T.Type:                   # T-PM
        w = _matrix(t.w, "pm")
        if not _is_square(w):
            raise QRLTypeError("pm: W must be square", t.w.line, t.w.col)
        if not _is_hermitian(w):
            raise QRLTypeError("pm: W is not Hermitian", t.w.line, t.w.col)
        if not _is_psd(w):
            raise QRLTypeError("pm: W is not positive semidefinite (W >= 0 fails)",
                               t.w.line, t.w.col)
        n, d = _pm_shape(w.shape[0])
        if n is None:
            raise QRLTypeError(
                f"pm: W dimension {w.shape[0]} is not d**(2n) for integer d, n",
                t.w.line, t.w.col,
            )
        if not np.isclose(np.trace(w).real, d ** n, atol=1e-6):
            raise QRLTypeError(
                f"pm: Tr[W] = {np.trace(w).real:.4f}, expected d**n = {d ** n} "
                f"for a valid {n}-party process matrix",
                t.w.line, t.w.col,
            )
        return T.PM(n, d)

    @staticmethod
    def DAG(t: ast.DAG, env: Env) -> T.Type:                 # T-DAG
        vset = set(t.vertices)
        if len(vset) != len(t.vertices):
            raise QRLTypeError("dag: duplicate vertex name", t.line, t.col)
        for (a, b) in t.edges:
            if a not in vset or b not in vset:
                raise QRLTypeError(f"dag: edge ({a}, {b}) references unknown vertex",
                                   t.line, t.col)
        if _has_cycle(t.vertices, t.edges):
            raise QRLTypeError("dag: (V, E) is not acyclic", t.line, t.col)
        edge_set = set(t.edges)
        phi_edges = set()
        for (a, b, mnode) in t.phi:
            if (a, b) not in edge_set:
                raise QRLTypeError(f"dag: Phi maps non-edge ({a}, {b})",
                                   mnode.line, mnode.col)
            phi_edges.add((a, b))
            mat = _matrix(mnode, "dag Phi")
            if not _is_square(mat):
                raise QRLTypeError(f"dag: Phi({a},{b}) is not a square CPTP matrix",
                                   mnode.line, mnode.col)
        missing = edge_set - phi_edges
        if missing:
            raise QRLTypeError(f"dag: Phi undefined on edges {sorted(missing)}",
                               t.line, t.col)
        return T.CausalDAG(len(t.vertices))

    @staticmethod
    def Do(t: ast.Do, env: Env) -> T.Type:                   # T-Do
        gty = _check(t.graph, env)
        if not isinstance(gty, T.CausalDAG):
            raise QRLTypeError(f"do: expected CausalDAG, got {gty}",
                               t.graph.line, t.graph.col)
        verts = None
        if isinstance(t.graph, ast.DAG):
            verts = t.graph.vertices
        elif isinstance(t.graph, ast.Var):
            verts = env.lookup_vertices(t.graph.name)
        if verts is not None and t.var not in verts:
            raise QRLTypeError(
                f"do: {t.var!r} is not a vertex of the graph {sorted(verts)}",
                t.line, t.col,
            )
        rho = _matrix(t.rho, "do")
        if not _is_psd(rho) or not np.isclose(np.trace(rho).real, 1.0, atol=1e-6):
            raise QRLTypeError("do: rho is not a density matrix (PSD, trace 1)",
                               t.rho.line, t.rho.col)
        return T.Outcome()

    @staticmethod
    def MatrixLit(t: ast.MatrixLit, env: Env) -> T.Type:  # pragma: no cover
        raise QRLTypeError("a matrix literal is not a term here", t.line, t.col)


# --------------------------------------------------------------------------
# support
# --------------------------------------------------------------------------

def _as_proc(ty: T.Type, node: ast.Term) -> T.Proc:
    if isinstance(ty, T.Proc):
        return ty
    if isinstance(ty, T.UniProc):
        return T.Proc(ty.d, ty.d)          # UniProc(d) <: Proc(d, d)
    raise QRLTypeError(f"expected a process, got {ty}", node.line, node.col)


def _pm_shape(dim: int) -> Tuple[Optional[int], Optional[int]]:
    for d in (2, 3, 4, 5):
        n = 0
        v = 1
        while v < dim:
            v *= d * d
            n += 1
        if v == dim and n >= 1:
            return n, d
    return None, None


def _has_cycle(vertices: List[str], edges: List[Tuple[str, str]]) -> bool:
    adj: Dict[str, List[str]] = {v: [] for v in vertices}
    for a, b in edges:
        adj[a].append(b)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in vertices}

    def visit(u: str) -> bool:
        color[u] = GRAY
        for w in adj[u]:
            if color[w] == GRAY:
                return True
            if color[w] == WHITE and visit(w):
                return True
        color[u] = BLACK
        return False

    return any(color[v] == WHITE and visit(v) for v in vertices)


def _resolve_type(te: ast.TypeExpr) -> T.Type:
    if isinstance(te, ast.TyName):
        return {"Qubit": T.Qubit(), "Ques": T.Ques(), "Outcome": T.Outcome()}[te.name]
    if isinstance(te, ast.TyRel):
        return T.Rel(te.n)
    if isinstance(te, ast.TySwitch):
        return T.Switch(te.d)
    if isinstance(te, ast.TyUniProc):
        return T.UniProc(te.d)
    if isinstance(te, ast.TyProc):
        return T.Proc(te.d_in, te.d_out)
    if isinstance(te, ast.TyPM):
        return T.PM(te.n, te.d)
    if isinstance(te, ast.TyCausalDAG):
        return T.CausalDAG(te.n)
    if isinstance(te, ast.TyTensor):
        return T.Tensor(_resolve_type(te.left), _resolve_type(te.right))
    raise QRLTypeError("unrecognised type annotation", te.line, te.col)  # pragma: no cover
