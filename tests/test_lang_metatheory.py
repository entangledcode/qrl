"""Executable check of Subject Reduction and Progress for the core calculus.

The paper (quantum-causal-structure.tex, Section 2.5) proves, for the core calculus
(everything except dag/do):

  * Progress: a closed well-typed term is a value or steps, with step
    probabilities summing to 1 (so evaluation never gets stuck);
  * Subject Reduction: every step preserves the type;
  * Termination: every run has at most as many steps as the term has
    let/entangle/ask/;/switch nodes.

This file checks the implementation against those statements: it generates
well-typed closed programs, runs each with several seeds, and asserts that the
interpreter never raises and that the value it returns has the checked type
(the value-typing rules V-*).  It also holds the regression tests for the four
well-typed programs that used to crash the interpreter (review 2026-11-27).
"""

import random

import numpy as np
import pytest

from qrl.lang import check_source, run, types as T
from qrl.lang.interp import Channel, Observable, Outcome, Pair, Process, QState


# ---- value typing (the V-* rules, read on interpreter values) ----------

def _state_qubits(ty):
    """Number of qubits if `ty` is a state type (its values are QStates), else None."""
    if isinstance(ty, T.Qubit):
        return 1
    if isinstance(ty, T.Rel):
        return ty.n
    if isinstance(ty, T.Tensor):
        a, b = _state_qubits(ty.left), _state_qubits(ty.right)
        return None if a is None or b is None else a + b
    return None


def _is_state(v, n):
    if not isinstance(v, QState) or v.rho.shape != (2 ** n, 2 ** n):
        return False
    rho = v.rho
    return (np.allclose(rho, rho.conj().T, atol=1e-9)
            and np.isclose(np.trace(rho).real, 1.0, atol=1e-9)
            and np.linalg.eigvalsh(rho).min() > -1e-9)


def has_type(v, ty) -> bool:
    n = _state_qubits(ty)
    if n is not None:
        return _is_state(v, n)
    if isinstance(ty, T.Outcome):
        return isinstance(v, Outcome)
    if isinstance(ty, T.Ques):
        return isinstance(v, Observable)
    if isinstance(ty, T.Tensor):
        return isinstance(v, Pair) and has_type(v.fst, ty.left) and has_type(v.snd, ty.right)
    if isinstance(ty, T.UniProc):
        return isinstance(v, Channel) and v.is_unitary and v.cptp.input_dim == ty.d
    if isinstance(ty, T.Proc):
        return isinstance(v, Channel) and v.cptp.input_dim == ty.d_in
    if isinstance(ty, T.Switch):
        return (isinstance(v, Process) and v.source is not None
                and list(v.pm.input_dims) == [1, ty.d, ty.d, 2])
    if isinstance(ty, (T.PM, T.CausalClass)):
        return isinstance(v, Process)
    return False


# ---- a generator of well-typed closed core programs --------------------

_CHANNELS = {
    "X": "cptp([[0, 1], [1, 0]])",
    "Z": "cptp([[1, 0], [0, -1]])",
    "I": "cptp([[1, 0], [0, 1]])",
    "Hd": "cptp([[0.7071067811865476, 0.7071067811865476], "
          "[0.7071067811865476, -0.7071067811865476]])",
    "AD": "cptp([[1, 0], [0, 0.7071067811865476]], [[0, 0.7071067811865476], [0, 0]])",
}
_OBS = ["Z", "X", "Y", "H"]

QUBIT, OUT, QUES, PROC, UNI, SW = (T.Qubit(), T.Outcome(), T.Ques(),
                                   T.Proc(2, 2), T.UniProc(2), T.Switch(2))


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.k = 0

    def fresh(self):
        self.k += 1
        return f"v{self.k}"

    def var(self, env, ty):
        names = [x for x, t in env if T.is_subtype(t, ty)]
        return self.r.choice(names) if names else None

    def term(self, ty, env, depth):
        """A source term with checked type <: ty, closed under env."""
        r = self.r
        if depth > 0 and r.random() < 0.25:                   # let, maybe annotated
            sigma = r.choice([QUBIT, PROC, UNI, T.Rel(2), SW, QUES,
                              T.Tensor(OUT, T.Rel(2))])
            x = self.fresh()
            rhs = self.term(sigma, env, depth - 1)
            annot = ""
            if sigma == UNI and r.random() < 0.5:
                annot, sigma = " : Proc(2, 2)", PROC
            elif sigma == SW and r.random() < 0.5:
                annot, sigma = " : QCQC(2)", T.CausalClass("QCQC", 2)
            body = self.term(ty, env + [(x, sigma)], depth - 1)
            return f"(let {x}{annot} = {rhs} in {body})"
        v = self.var(env, ty)
        if v is not None and r.random() < 0.3:
            return v
        if ty == QUBIT:
            return r.choice(["|0>", "|1>", "|+>"])
        if ty == QUES:
            return r.choice(_OBS)
        if isinstance(ty, T.Rel):
            args = ", ".join(self.term(QUBIT, env, depth - 1) for _ in range(ty.n))
            return f"entangle({args})"
        if ty == UNI:
            return _CHANNELS[r.choice(["X", "Z", "I", "Hd"])]
        if ty == PROC:
            if depth > 0 and r.random() < 0.4:
                return f"({self.term(PROC, env, depth - 1)} ; {self.term(PROC, env, depth - 1)})"
            return _CHANNELS[r.choice(list(_CHANNELS))]
        if ty == SW:
            return (f"switch({self.term(PROC, env, depth - 1)}, "
                    f"{self.term(PROC, env, depth - 1)}, {self.term(QUBIT, env, depth - 1)})")
        if isinstance(ty, T.Tensor) and ty.left == OUT and isinstance(ty.right, T.Rel):
            return f"ask({self.term(ty.right, env, depth - 1)}, {self.term(QUES, env, depth - 1)})"
        if isinstance(ty, T.Tensor):
            return f"({self.term(ty.left, env, depth - 1)} * {self.term(ty.right, env, depth - 1)})"
        if isinstance(ty, T.CausalClass) and ty.kind == "QCQC":
            return self.term(SW, env, depth)
        raise AssertionError(f"no generator for {ty}")

    def program(self):
        ty = self.r.choice([QUBIT, T.Rel(2), T.Rel(3), T.Tensor(OUT, T.Rel(2)), PROC, UNI,
                            SW, T.Tensor(PROC, QUBIT), T.Tensor(SW, T.Tensor(OUT, T.Rel(2))),
                            T.Tensor(UNI, UNI)])
        return self.term(ty, [], depth=3), ty


def _programs(n, seed=20261127):
    g = Gen(seed)
    return [g.program() for _ in range(n)]


@pytest.mark.parametrize("chunk", range(10))
def test_generated_programs_preserve_types_and_never_get_stuck(chunk):
    for src, want in _programs(300, seed=20261127 + chunk):
        ty = check_source(src)                       # generator builds well-typed terms
        assert T.is_subtype(ty, want), (src, ty, want)
        for seed in (0, 1, 2):
            v = run(src, seed=seed)                  # Progress: never raises
            assert has_type(v, ty), (src, ty, v)     # Subject Reduction (end to end)


# ---- the four well-typed programs that used to get stuck ---------------

def test_tensor_of_channels_is_a_pair():
    src = "let A = cptp([[1, 0], [0, 1]]) in A * A"
    v = run(src, seed=0)
    assert isinstance(v, Pair) and has_type(v, check_source(src))


def test_observable_through_a_variable():
    src = "let q = Z in ask(entangle(|0>, |0>), q)"
    assert has_type(run(src, seed=0), check_source(src))


def test_observable_can_be_shadowed():
    # Z rebound to a qubit is no longer a question: rejected statically.
    from qrl.lang.errors import QRLTypeError
    with pytest.raises(QRLTypeError):
        check_source("let Z = |0> in ask(entangle(|0>, |0>), Z)")


@pytest.mark.parametrize("name", ["S", "T"])
def test_gates_are_not_questions(name):
    # S and T used to be typed Ques but had no measurement at run time.
    from qrl.lang.errors import QRLTypeError
    with pytest.raises(QRLTypeError):
        check_source(f"ask(entangle(|0>, |0>), {name})")
