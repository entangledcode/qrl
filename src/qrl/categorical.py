"""
Categorical quantum mechanics — Coecke & Kissinger (2017).

Processes are morphisms A → B in a †-symmetric monoidal category.
Composition is sequential (.then / .compose) or parallel (.tensor).
The dagger gives adjoints. Spiders are the Frobenius algebra elements
of the ZX-calculus.

This module makes QRL's compositional structure explicit and provides
the vocabulary for the formal semantics (Phase 2).

Reference: B. Coecke and A. Kissinger, "Picturing Quantum Processes",
Cambridge University Press, 2017.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Process:
    """
    A morphism A → B in the category of quantum processes.

    matrix:       numpy array, shape (output_dim, input_dim)
    input_type:   tuple of ints — dimensions of each input wire
    output_type:  tuple of ints — dimensions of each output wire
    name:         optional label

    The scalar (empty system) has type ().  A single qubit has type (2,).
    Two qubits have type (2, 2), with index ordering |i,j⟩ ↔ 2i+j.

    Sequential composition:  p.then(q)   or  q.compose(p)   or  q @ p
    Parallel composition:    p.tensor(q) or  p * q
    Adjoint:                 p.dagger()
    """

    matrix: np.ndarray
    input_type: tuple
    output_type: tuple
    name: str = ""

    def __post_init__(self) -> None:
        self.input_type = tuple(self.input_type)
        self.output_type = tuple(self.output_type)
        exp_in = int(np.prod(self.input_type)) if self.input_type else 1
        exp_out = int(np.prod(self.output_type)) if self.output_type else 1
        m = np.atleast_2d(np.asarray(self.matrix, dtype=complex))
        if m.shape != (exp_out, exp_in):
            raise ValueError(
                f"Matrix shape {m.shape} inconsistent with types: "
                f"input {self.input_type} (dim {exp_in}), "
                f"output {self.output_type} (dim {exp_out})"
            )
        self.matrix = m

    @property
    def input_dim(self) -> int:
        return int(np.prod(self.input_type)) if self.input_type else 1

    @property
    def output_dim(self) -> int:
        return int(np.prod(self.output_type)) if self.output_type else 1

    def then(self, other: Process) -> Process:
        """Sequential composition: self then other (other ∘ self). self: A→B, other: B→C."""
        if self.output_type != other.input_type:
            raise ValueError(
                f"Type mismatch in sequential composition: "
                f"output {self.output_type} ≠ input {other.input_type}"
            )
        name = f"({self.name}; {other.name})" if self.name and other.name else ""
        return Process(
            matrix=other.matrix @ self.matrix,
            input_type=self.input_type,
            output_type=other.output_type,
            name=name,
        )

    def compose(self, other: Process) -> Process:
        """Sequential composition: other then self (self ∘ other). self: B→C, other: A→B."""
        return other.then(self)

    def tensor(self, other: Process) -> Process:
        """Parallel composition: self ⊗ other. Types concatenate."""
        name = f"({self.name} ⊗ {other.name})" if self.name and other.name else ""
        return Process(
            matrix=np.kron(self.matrix, other.matrix),
            input_type=self.input_type + other.input_type,
            output_type=self.output_type + other.output_type,
            name=name,
        )

    def dagger(self) -> Process:
        """Adjoint: conjugate transpose, input/output types swapped."""
        name = f"{self.name}†" if self.name else ""
        return Process(
            matrix=self.matrix.conj().T,
            input_type=self.output_type,
            output_type=self.input_type,
            name=name,
        )

    def is_unitary(self, tol: float = 1e-10) -> bool:
        if self.input_type != self.output_type:
            return False
        I = np.eye(self.input_dim)
        return np.allclose(self.matrix @ self.matrix.conj().T, I, atol=tol) and np.allclose(
            self.matrix.conj().T @ self.matrix, I, atol=tol
        )

    def is_isometry(self, tol: float = 1e-10) -> bool:
        """P†P = I_input (encoding a pure state or isometric embedding)."""
        return np.allclose(
            self.matrix.conj().T @ self.matrix, np.eye(self.input_dim), atol=tol
        )

    def is_causal(self, tol: float = 1e-10) -> bool:
        """A pure process is causal iff it is an isometry (trace-preserving on pure states)."""
        return self.is_isometry(tol)

    def approx_equal(self, other: Process, tol: float = 1e-10) -> bool:
        """Matrix equality, allowing a global scalar phase."""
        if self.input_type != other.input_type or self.output_type != other.output_type:
            return False
        a = self.matrix.ravel()
        b = other.matrix.ravel()
        nz = np.nonzero(np.abs(a) > tol)[0]
        if len(nz) == 0:
            return np.allclose(b, 0, atol=tol)
        scale = b[nz[0]] / a[nz[0]]
        return np.allclose(a * scale, b, atol=tol)

    def __matmul__(self, other: Process) -> Process:
        """p @ q  ≡  q.then(p)  (standard math composition order)."""
        return self.compose(other)

    def __mul__(self, other: Process) -> Process:
        """p * q  ≡  p.tensor(q)."""
        return self.tensor(other)

    def __repr__(self) -> str:
        label = f'"{self.name}" ' if self.name else ""
        return f"Process {label}{self.input_type!r} → {self.output_type!r}"


# ---------------------------------------------------------------------------
# Elementary processes
# ---------------------------------------------------------------------------

def identity_process(d: int = 2) -> Process:
    """Identity morphism on a single wire of dimension d."""
    return Process(matrix=np.eye(d, dtype=complex), input_type=(d,), output_type=(d,), name=f"id_{d}")


def scalar_process(alpha: complex) -> Process:
    """Global scalar: a morphism () → () with value alpha."""
    return Process(
        matrix=np.array([[alpha]], dtype=complex), input_type=(), output_type=(), name=f"scalar({alpha:.3g})"
    )


def hadamard_process() -> Process:
    """Hadamard gate H: (2,) → (2,)."""
    H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    return Process(matrix=H, input_type=(2,), output_type=(2,), name="H")


def swap_process(d: int = 2) -> Process:
    """SWAP gate: (d, d) → (d, d).  |i,j⟩ ↦ |j,i⟩."""
    S = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[j * d + i, i * d + j] = 1.0
    return Process(matrix=S, input_type=(d, d), output_type=(d, d), name="SWAP")


# ---------------------------------------------------------------------------
# Spiders (Coecke & Kissinger §5-6)
# ---------------------------------------------------------------------------

def z_spider(n_in: int, n_out: int, phase: float = 0.0) -> Process:
    """
    Z-spider (green) with n_in inputs, n_out outputs, phase α.

        Z^α_{n,m} = |0...0⟩⟨0...0| + e^{iα}|1...1⟩⟨1...1|

    Key instances (unnormalised):
      z_spider(0, 1, 0)  →  |0⟩ + |1⟩          Z-basis "unit" state
      z_spider(0, 1, π)  →  |0⟩ − |1⟩
      z_spider(0, 2, 0)  →  |00⟩ + |11⟩        Bell pair / cup
      z_spider(1, 0, 0)  →  ⟨0| + ⟨1|          Z-basis effect (⟨+|√2)
      z_spider(1, 1, α)  →  diag(1, e^{iα})    Z-rotation gate Rz(α)
      z_spider(1, 2, 0)  →  |0⟩⟨0|⊗|0⟩ + |1⟩⟨1|⊗|1⟩  copy map
      z_spider(2, 1, 0)  →  |0⟩⟨00| + |1⟩⟨11|  merge/addition map
    """
    in_dim = 2**n_in if n_in > 0 else 1
    out_dim = 2**n_out if n_out > 0 else 1
    M = np.zeros((out_dim, in_dim), dtype=complex)
    M[0, 0] = 1.0
    M[out_dim - 1, in_dim - 1] += np.exp(1j * phase)
    return Process(
        matrix=M,
        input_type=(2,) * n_in,
        output_type=(2,) * n_out,
        name=f"Z({phase:.4g},{n_in}→{n_out})",
    )


def x_spider(n_in: int, n_out: int, phase: float = 0.0) -> Process:
    """
    X-spider (red) with n_in inputs, n_out outputs, phase α.

        X^α_{n,m} = H^{⊗n_out} · Z^α_{n,m} · H^{⊗n_in}

    Key instances (unnormalised):
      x_spider(0, 1, 0)  →  |0⟩√2              X-basis unit state
      x_spider(1, 1, π)  →  Pauli X gate
      x_spider(1, 2, 0)  →  copy in X basis
    """
    z = z_spider(n_in, n_out, phase)
    H = hadamard_process()

    def _h_n(n: int) -> Optional[Process]:
        if n == 0:
            return None
        p = H
        for _ in range(n - 1):
            p = p.tensor(H)
        return p

    H_in = _h_n(n_in)
    H_out = _h_n(n_out)

    result = z
    if H_in is not None:
        result = H_in.then(result)
    if H_out is not None:
        result = result.then(H_out)
    result.name = f"X({phase:.4g},{n_in}→{n_out})"
    return result


# ---------------------------------------------------------------------------
# Compact structure: cups and caps (C&K §6)
# ---------------------------------------------------------------------------

def cup(d: int = 2) -> Process:
    """
    Bell pair as a process () → (d, d).

        cup = Σ_i |ii⟩   (unnormalised maximally entangled state)

    This is the unit of the compact structure: every entangled state
    factors through a cup.  In QRL's relations-first view, the cup
    *is* the primitive entangled relation — all correlation flows from it.

    Note: cup(2) = z_spider(0, 2, 0).
    """
    M = np.zeros((d * d, 1), dtype=complex)
    for i in range(d):
        M[i * d + i, 0] = 1.0
    return Process(matrix=M, input_type=(), output_type=(d, d), name=f"cup_{d}")


def cap(d: int = 2) -> Process:
    """
    Bell effect as a process (d, d) → ().

        cap = Σ_i ⟨ii|   (unnormalised Bell measurement effect)

    cap = cup(d)†.
    """
    c = cup(d)
    c2 = c.dagger()
    c2.name = f"cap_{d}"
    return c2


# ---------------------------------------------------------------------------
# ZX-calculus rules (C&K §8-9)
# ---------------------------------------------------------------------------

def spider_fusion(p: Process, q: Process) -> Process:
    """
    Spider fusion: same-colour spiders connected by k wires fuse into one.

        Z^α_{k→m} ∘ Z^β_{n→k} = Z^{α+β}_{n→m}
        X^α_{k→m} ∘ X^β_{n→k} = X^{α+β}_{n→m}

    This is a thin wrapper over .then() — the fusion theorem guarantees the
    result is itself a spider with summed phases.  Verified in tests.
    """
    return p.then(q)


def colour_change(p: Process) -> Process:
    """
    Colour change rule: conjugating a Z-spider by Hadamards gives an X-spider.

        H^{⊗n_out} · Z^α_{n,m} · H^{⊗n_in} = X^α_{n,m}

    Works for any Process, not just named spiders.
    """
    H = hadamard_process()
    n_in = len(p.input_type)
    n_out = len(p.output_type)

    def _h_n(n: int) -> Optional[Process]:
        if n == 0:
            return None
        q = H
        for _ in range(n - 1):
            q = q.tensor(H)
        return q

    result = p
    H_in = _h_n(n_in)
    H_out = _h_n(n_out)
    if H_in is not None:
        result = H_in.then(result)
    if H_out is not None:
        result = result.then(H_out)
    result.name = f"colour_change({p.name})"
    return result


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def are_complementary(z_state: Process, x_state: Process, tol: float = 1e-10) -> bool:
    """
    Z and X are complementary (mutually unbiased bases) iff:

        |⟨z_state|x_state⟩|² = 1 / d

    where d = output dimension.  Takes single-output state processes
    (input_type=(), output_type=(d,)), e.g. z_spider(0,1,0) and x_spider(0,1,0).
    """
    if z_state.input_type != () or x_state.input_type != ():
        return False
    if z_state.output_type != x_state.output_type or len(z_state.output_type) != 1:
        return False
    d = z_state.output_type[0]
    z_vec = z_state.matrix
    x_vec = x_state.matrix
    z_norm = float(np.linalg.norm(z_vec))
    x_norm = float(np.linalg.norm(x_vec))
    if z_norm < tol or x_norm < tol:
        return False
    inner = float(abs(complex((z_vec.conj().T @ x_vec)[0, 0])))
    overlap_sq = (inner / (z_norm * x_norm)) ** 2
    return bool(np.isclose(overlap_sq, 1.0 / d, atol=tol))


# ---------------------------------------------------------------------------
# Derived processes
# ---------------------------------------------------------------------------

def cnot_from_spiders() -> Process:
    """
    CNOT gate as a composition of Z-copy and X-merge.

        CNOT = (id ⊗ X-merge) ∘ (Z-copy ⊗ id)

    The result is proportional to the standard CNOT (global scalar 1/√2
    in the unnormalised spider convention).

    Input/output type: (2, 2) — (control, target).
    """
    id2 = identity_process(2)
    z_copy = z_spider(1, 2, 0)   # (2,) → (2, 2)
    x_merge = x_spider(2, 1, 0)  # (2, 2) → (2,)
    result = z_copy.tensor(id2).then(id2.tensor(x_merge))
    result.name = "CNOT"
    return result


def bell_state() -> Process:
    """
    Bell state |Φ+⟩ as a process () → (2, 2).

    Equivalent to cup(2).  In QRL's relations-first view, the Bell pair is
    the primitive entangled relation from which all CHSH, GHZ, and loophole
    analyses derive.
    """
    p = cup(2)
    p.name = "Bell|Φ+⟩"
    return p


def controlled_z_process() -> Process:
    """CZ gate: (2, 2) → (2, 2).  Diagonal with entries (1,1,1,-1)."""
    CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
    return Process(matrix=CZ, input_type=(2, 2), output_type=(2, 2), name="CZ")


# ---------------------------------------------------------------------------
# Bridges to existing QRL
# ---------------------------------------------------------------------------

def process_from_unitary(U: np.ndarray, name: str = "") -> Process:
    """
    Wrap a unitary matrix as a Process on n-qubit space.
    Assumes all wires are qubits (dimension 2).
    """
    U = np.asarray(U, dtype=complex)
    n = U.shape[0]
    n_qubits = int(round(math.log2(n)))
    if 2**n_qubits != n:
        raise ValueError(f"Matrix dimension {n} is not a power of 2")
    t = (2,) * n_qubits
    return Process(matrix=U, input_type=t, output_type=t, name=name)


def process_from_kraus(kraus_ops: List[np.ndarray], name: str = "") -> Process:
    """
    Convert a list of Kraus operators {K_i} to a Process via the superoperator.

    The superoperator S acts on vectorised density matrices:

        vec(Λ(ρ)) = S · vec(ρ),   S = Σ_i K_i ⊗ K_i*

    Input/output type: (d²,) where d = K_i.shape[0] (= shape[1] for trace-preserving maps).
    This represents the map on the space of density matrices viewed as a Hilbert space.
    """
    K0 = np.asarray(kraus_ops[0], dtype=complex)
    d_out, d_in = K0.shape
    S = sum(np.kron(K.astype(complex), K.conj()) for K in kraus_ops)
    d2_in = d_in * d_in
    d2_out = d_out * d_out
    return Process(
        matrix=S,
        input_type=(d2_in,),
        output_type=(d2_out,),
        name=name or f"channel({len(kraus_ops)} Kraus ops)",
    )
