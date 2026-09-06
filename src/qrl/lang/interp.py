"""Interpreter for the QRL surface language.

Evaluates a type-checked AST following the small-step operational semantics of
`papers/qpl-2026/quantum-causal-structure.tex` Section 2.3.  State is carried as
density matrices throughout (the paper's "values are density matrices rho ...").

Sessions A and B cover the basic, relational, and unitary-causal strata:

    T-Qubit / T-Var / T-Let        |0> |1> |+>, variables, let-binding
    E-Bell / E-GHZ                  entangle(...) -> canonical maximally
                                    entangled state, *ignoring* its arguments
    E-Ask                          Born-rule partial measurement of subsystem 0
    T-Tensor                       t1 * t2 on states (Kronecker product)
    T-UniProc / cptp               unitary and general CPTP channels
    T-Seq                          t1 ; t2  sequential channel composition
    E-Switch-Coherent              switch(f, g, c) -> quantum-switch process
    T-PM / pm                      process matrix from a raw W
    T-DAG / dag                    quantum causal DAG (single-parent nodes)
    T-Do / do                      do-intervention, sampled outcome on the sink

E-Switch-Incoherent is intentionally not implemented: no well-typed surface
program can reach it (switch requires UniProc channels; a switch over a general
CPTP map has type PM(2,d), not Switch(d)).

Public entry points (re-exported from `qrl.lang`):

    run(src, *, seed=None, shots=1)   -> Value | list[Value]
    distribution(src)                 -> dict[int, float]     (exact, no sampling)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from . import ast
from .errors import QRLError, QRLRuntimeError
from .parser import parse
from .typecheck import check_program

_TOL = 1e-9


# --------------------------------------------------------------------------
# runtime values  (paper Section 2.3)
# --------------------------------------------------------------------------

class Value:
    pass


@dataclass
class QState(Value):
    """An n-qubit quantum state as a density matrix (2**n x 2**n)."""
    rho: np.ndarray

    @property
    def n(self) -> int:
        return int(round(np.log2(self.rho.shape[0])))

    def __str__(self) -> str:
        return f"QState(n={self.n}{_name_state(self.rho)})"


@dataclass
class Outcome(Value):
    m: int

    def __str__(self) -> str:
        return f"Outcome({self.m})"


@dataclass
class Pair(Value):
    fst: Value
    snd: Value

    def __str__(self) -> str:
        return f"({self.fst}, {self.snd})"


@dataclass
class Channel(Value):
    """A CPTP channel — `qrl.causal.CPTPMap`.  Unitary iff a single unitary Kraus."""
    cptp: object

    @property
    def is_unitary(self) -> bool:
        return self.cptp.is_unitary()

    def __str__(self) -> str:
        kind = "unitary" if self.is_unitary else "CPTP"
        return f"Channel({kind}, d={self.cptp.input_dim})"


@dataclass
class Process(Value):
    """A process matrix — `qrl.causal.ProcessMatrix`.

    `source` is the originating `QuantumSwitch` when this came from `switch(...)`,
    which is where the readable P_win / robustness accessors come from.
    """
    pm: object
    source: object = None
    control: object = None

    @property
    def p_win(self):
        return self.pm.causal_inequality_value()

    @property
    def robustness(self):
        return self.pm.causal_nonseparability_robustness()

    def __str__(self) -> str:
        pw = self.p_win
        r = self.robustness
        extra = ""
        if pw is not None:
            extra = f", P_win={pw:.4f}"
            if r is not None:
                extra += f", robustness={r:.4f}"
        return f"Process({'x'.join(map(str, self.pm.input_dims))}{extra})"


@dataclass
class Dag(Value):
    """A quantum causal DAG — `qrl.causal.QuantumCausalDAG`."""
    dag: object
    vertices: tuple = ()
    edges: tuple = ()

    def __str__(self) -> str:
        return f"Dag({len(self.vertices)} nodes, {len(self.edges)} edges)"


# --------------------------------------------------------------------------
# single-qubit data
# --------------------------------------------------------------------------

_KET_VEC = {
    "0": np.array([1.0, 0.0], dtype=complex),
    "1": np.array([0.0, 1.0], dtype=complex),
    "+": np.array([1.0, 1.0], dtype=complex) / np.sqrt(2),
}


def _ketbra(v: np.ndarray) -> np.ndarray:
    return np.outer(v, v.conj())


def _is_unitary(m: np.ndarray) -> bool:
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    return np.allclose(m.conj().T @ m, np.eye(m.shape[0]), atol=1e-6)


# observable name -> list of (outcome_label, projector) in the single-qubit space
def _proj(v: np.ndarray) -> np.ndarray:
    return _ketbra(v)


_MINUS = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
_IPLUS = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2)
_IMINUS = np.array([1.0, -1.0j], dtype=complex) / np.sqrt(2)

_OBSERVABLES: Dict[str, List[np.ndarray]] = {
    # outcome 0 first, then outcome 1
    "Z": [_proj(_KET_VEC["0"]), _proj(_KET_VEC["1"])],
    "X": [_proj(_KET_VEC["+"]), _proj(_MINUS)],
    "Y": [_proj(_IPLUS), _proj(_IMINUS)],
}
# H, S, T are seeded as Ques by the checker but are gates, not observables here;
# measuring in "the H basis" is the X basis.
_OBSERVABLES["H"] = _OBSERVABLES["X"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _name_state(rho: np.ndarray) -> str:
    """Recognise a few canonical states for readable output."""
    n = int(round(np.log2(rho.shape[0])))
    candidates = {}
    if n == 2:
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        candidates[" = Bell"] = _ketbra(bell)
    if n >= 2:
        ghz = np.zeros(2 ** n, dtype=complex)
        ghz[0] = ghz[-1] = 1 / np.sqrt(2)
        candidates[f" = GHZ{n}"] = _ketbra(ghz)
    for label, ref in candidates.items():
        if rho.shape == ref.shape and np.allclose(rho, ref, atol=1e-6):
            return label
    return ""


def _embed_projector(proj1: np.ndarray, k: int, n: int) -> np.ndarray:
    """I ⊗ ... ⊗ proj1 (position k) ⊗ ... ⊗ I  on n qubits."""
    ops = [np.eye(2, dtype=complex)] * n
    ops[k] = proj1
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _topo_order(vertices: List[str], edges: List[tuple]) -> List[str]:
    """Kahn's algorithm; input is already known acyclic (T-DAG)."""
    indeg = {v: 0 for v in vertices}
    adj: Dict[str, List[str]] = {v: [] for v in vertices}
    for a, b in edges:
        indeg[b] += 1
        adj[a].append(b)
    queue = [v for v in vertices if indeg[v] == 0]
    out: List[str] = []
    while queue:
        u = queue.pop(0)
        out.append(u)
        for w in adj[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    return out


def _canonical_entangled(n: int) -> np.ndarray:
    """(|0...0> + |1...1>) / sqrt(2) as a density matrix — E-Bell / E-GHZ."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = psi[-1] = 1.0 / np.sqrt(2)
    return _ketbra(psi)


def _measure_probs(rho: np.ndarray, observable: str, k: int) -> List[float]:
    n = int(round(np.log2(rho.shape[0])))
    projs = _OBSERVABLES[observable]
    out = []
    for p1 in projs:
        P = _embed_projector(p1, k, n)
        pm = np.trace(P @ rho).real
        out.append(max(pm, 0.0))
    s = sum(out)
    if s < _TOL:
        raise QRLRuntimeError("measurement has zero total probability", 0, 0)
    return [p / s for p in out]


def _post_state(rho: np.ndarray, observable: str, k: int, m: int) -> np.ndarray:
    n = int(round(np.log2(rho.shape[0])))
    P = _embed_projector(_OBSERVABLES[observable][m], k, n)
    new = P @ rho @ P
    tr = np.trace(new).real
    return new / tr


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

class _Env:
    def __init__(self, parent: Optional["_Env"] = None):
        self.vars: Dict[str, Value] = {}
        self.parent = parent

    def lookup(self, name: str) -> Optional[Value]:
        if name in self.vars:
            return self.vars[name]
        return self.parent.lookup(name) if self.parent else None

    def child(self) -> "_Env":
        return _Env(self)


# --------------------------------------------------------------------------
# interpreter
# --------------------------------------------------------------------------

class Interp:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def eval(self, t: ast.Term, env: _Env) -> Value:
        method = getattr(self, f"_ev_{type(t).__name__}", None)
        if method is None:
            raise QRLRuntimeError(
                f"cannot evaluate {type(t).__name__} as a term", t.line, t.col
            )
        return method(t, env)

    # ---- basic stratum ----------------------------------------------
    def _ev_Ket(self, t: ast.Ket, env: _Env) -> Value:
        return QState(_ketbra(_KET_VEC[t.value]))

    def _ev_Var(self, t: ast.Var, env: _Env) -> Value:
        v = env.lookup(t.name)
        if v is None:
            raise QRLRuntimeError(f"unbound variable {t.name!r} at run time",
                                  t.line, t.col)
        return v

    def _ev_Let(self, t: ast.Let, env: _Env) -> Value:
        val = self.eval(t.value, env)
        inner = env.child()
        inner.vars[t.name] = val
        return self.eval(t.body, inner)

    # ---- relational stratum --------------------------------------
    def _ev_Entangle(self, t: ast.Entangle, env: _Env) -> Value:
        # E-Bell / E-GHZ: arguments are declarations, not inputs.  We still
        # evaluate them (they must be well-formed) but discard the results.
        for a in t.args:
            self.eval(a, env)
        return QState(_canonical_entangled(len(t.args)))

    def _ev_Ask(self, t: ast.Ask, env: _Env) -> Value:
        rel = self.eval(t.relation, env)
        if not isinstance(rel, QState):
            raise QRLRuntimeError("ask: argument did not evaluate to a state",
                                  t.relation.line, t.relation.col)
        obs = self._observable_name(t.question)
        probs = _measure_probs(rel.rho, obs, k=0)
        m = int(self.rng.choice(len(probs), p=probs))
        post = _post_state(rel.rho, obs, k=0, m=m)
        return Pair(Outcome(m), QState(post))

    def _ev_Tensor(self, t: ast.Tensor, env: _Env) -> Value:
        a = self.eval(t.left, env)
        b = self.eval(t.right, env)
        if isinstance(a, QState) and isinstance(b, QState):
            return QState(np.kron(a.rho, b.rho))
        raise QRLRuntimeError(
            "tensor is currently supported only for states (QState * QState)",
            t.line, t.col,
        )

    # ---- causal stratum (unitary path) -------------------------
    def _ev_Cptp(self, t: ast.Cptp, env: _Env) -> Value:
        from qrl.causal import CPTPMap, cptp_from_unitary

        mats = [np.array(k.rows, dtype=complex) for k in t.kraus]
        d = mats[0].shape[0]
        if len(mats) == 1 and _is_unitary(mats[0]):
            return Channel(cptp_from_unitary(mats[0]))
        try:
            return Channel(CPTPMap(kraus_ops=mats, input_dim=d, output_dim=d))
        except ValueError as e:
            raise QRLRuntimeError(f"cptp: {e}", t.line, t.col) from None

    def _ev_Seq(self, t: ast.Seq, env: _Env) -> Value:
        left = self._as_channel(self.eval(t.left, env), t.left)
        right = self._as_channel(self.eval(t.right, env), t.right)
        # t1 ; t2  =  apply t1 first, then t2  =  t2 ∘ t1
        try:
            return Channel(right.cptp.compose(left.cptp))
        except ValueError as e:
            raise QRLRuntimeError(f"sequential composition: {e}", t.line, t.col) from None

    def _ev_Switch(self, t: ast.Switch, env: _Env) -> Value:
        from qrl.causal import QuantumSwitch

        f = self._as_channel(self.eval(t.f, env), t.f)
        g = self._as_channel(self.eval(t.g, env), t.g)
        control = self.eval(t.control, env)
        if not isinstance(control, QState) or control.n != 1:
            raise QRLRuntimeError("switch: control must be a single qubit",
                                  t.control.line, t.control.col)
        if not (f.is_unitary and g.is_unitary):
            # T-Switch should have prevented this; belt and braces
            raise QRLRuntimeError(
                "switch: both channels must be unitary (E-Switch-Coherent)",
                t.line, t.col,
            )
        qs = QuantumSwitch(channel_A=f.cptp, channel_B=g.cptp)
        return Process(qs.process_matrix(), source=qs, control=control.rho)

    def _ev_PM(self, t: ast.PM, env: _Env) -> Value:
        from qrl.causal import ProcessMatrix
        from .typecheck import _pm_shape

        W = np.array(t.w.rows, dtype=complex)
        n, d = _pm_shape(W.shape[0])
        parties = [chr(ord("A") + i) for i in range(n)]
        try:
            pm = ProcessMatrix(W=W, parties=parties,
                               input_dims=[d] * n, output_dims=[d] * n)
        except ValueError as e:
            raise QRLRuntimeError(f"pm: {e}", t.w.line, t.w.col) from None
        return Process(pm)

    def _ev_DAG(self, t: ast.DAG, env: _Env) -> Value:
        from qrl.causal import QuantumCausalDAG, CPTPMap

        parents: Dict[str, list] = {v: [] for v in t.vertices}
        phi_mat: Dict[tuple, np.ndarray] = {}
        for (a, b, mnode) in t.phi:
            parents[b].append(a)
            phi_mat[(a, b)] = np.array(mnode.rows, dtype=complex)
        for b, ps in parents.items():
            if len(ps) > 1:
                raise QRLRuntimeError(
                    f"dag: node {b!r} has {len(ps)} parents; the interpreter "
                    f"currently handles single-parent nodes only (use the "
                    f"Python API for joint mechanisms)",
                    t.line, t.col,
                )

        # infer a dimension for every vertex from the Phi matrices
        dim: Dict[str, int] = {}
        for (a, b), M in phi_mat.items():
            dim[b] = M.shape[0]
            dim.setdefault(a, M.shape[1])
        for v in t.vertices:
            dim.setdefault(v, 2)

        g = QuantumCausalDAG(description="qrl.lang dag")
        order = _topo_order(list(t.vertices), list(t.edges))
        for v in order:
            is_root = not parents[v]
            prior = None
            if is_root:
                prior = np.zeros((dim[v], dim[v]), dtype=complex)
                prior[0, 0] = 1.0  # |0><0|
            g.add_node(v, dim=dim[v], prior=prior)
        for (a, b), M in phi_mat.items():
            try:
                g.add_channel(a, b, CPTPMap(kraus_ops=[M], input_dim=M.shape[1],
                                            output_dim=M.shape[0]))
            except ValueError as e:
                raise QRLRuntimeError(f"dag: edge ({a}, {b}): {e}",
                                      t.line, t.col) from None
        return Dag(g, vertices=tuple(t.vertices), edges=tuple(t.edges))

    def _ev_Do(self, t: ast.Do, env: _Env) -> Value:
        g = self.eval(t.graph, env)
        if not isinstance(g, Dag):
            raise QRLRuntimeError("do: first argument is not a causal DAG",
                                  t.graph.line, t.graph.col)
        rho = np.array(t.rho.rows, dtype=complex)
        sinks = [v for v in g.vertices
                 if not any(a == v for (a, b) in g.edges)]
        if len(sinks) != 1:
            raise QRLRuntimeError(
                f"do: the DAG has {len(sinks)} sink nodes {sinks}; the "
                f"interpreter samples the outcome on a unique sink",
                t.line, t.col,
            )
        try:
            out_rho = g.dag.interventional_state(sinks[0], {t.var: rho})
        except (ValueError, KeyError) as e:
            raise QRLRuntimeError(f"do: {e}", t.line, t.col) from None
        probs = np.clip(np.diag(out_rho).real, 0.0, None)
        probs = probs / probs.sum()
        return Outcome(int(self.rng.choice(len(probs), p=probs)))

    # ---- helpers --------------------------------------------------
    @staticmethod
    def _as_channel(v: Value, node: ast.Term) -> "Channel":
        if isinstance(v, Channel):
            return v
        raise QRLRuntimeError(
            f"expected a channel, got {type(v).__name__}", node.line, node.col
        )

    # ---- helpers --------------------------------------------------
    @staticmethod
    def _observable_name(q: ast.Term) -> str:
        if isinstance(q, ast.Var) and q.name in _OBSERVABLES:
            return q.name
        name = getattr(q, "name", type(q).__name__)
        raise QRLRuntimeError(
            f"ask: {name!r} is not a measurable observable "
            f"(known: {', '.join(sorted(_OBSERVABLES))})",
            q.line, q.col,
        )


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def run(src: str, *, seed: Optional[int] = None, shots: int = 1
        ) -> Union[Value, List[Value]]:
    """Parse, type-check, and execute QRL source.

    `shots == 1` returns a single `Value`; `shots > 1` returns a list (re-running
    the whole program, so probabilistic `ask` outcomes vary).  `seed` makes the
    run reproducible.
    """
    tree = _parse_and_check(src)
    if shots < 1:
        raise ValueError("shots must be >= 1")
    interp = Interp(seed)
    results = [interp.eval(tree, _Env()) for _ in range(shots)]
    return results[0] if shots == 1 else results


def distribution(src: str) -> Dict[int, float]:
    """Exact outcome distribution for a program whose result is an outcome.

    Supported when the top-level term is `ask(t, q)` (result type
    `Outcome * Rel(n)`).  No sampling.
    """
    tree = _parse_and_check(src)
    interp = Interp()
    env = _Env()
    # peel outer let-bindings so `let x = ... in ask(...)` is supported
    while isinstance(tree, ast.Let):
        val = interp.eval(tree.value, env)
        env = env.child()
        env.vars[tree.name] = val
        tree = tree.body
    if not isinstance(tree, ast.Ask):
        raise QRLRuntimeError(
            "distribution() is only defined when the program's result comes "
            "from a top-level `ask(...)` (possibly under let-bindings)",
            tree.line, tree.col,
        )
    rel = interp.eval(tree.relation, env)
    if not isinstance(rel, QState):
        raise QRLRuntimeError("ask: argument did not evaluate to a state",
                              tree.relation.line, tree.relation.col)
    obs = Interp._observable_name(tree.question)
    probs = _measure_probs(rel.rho, obs, k=0)
    return {m: float(p) for m, p in enumerate(probs)}


def _parse_and_check(src: str) -> ast.Term:
    try:
        tree = parse(src)
        check_program(tree)
        return tree
    except QRLError as e:
        raise e.with_source(src) from None
