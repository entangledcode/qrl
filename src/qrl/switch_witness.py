"""
Quantum switch as a tripartite process, and a measurable causal witness for it.

The quantum switch is causally nonseparable, but it cannot violate a causal
inequality: its statistics in any device-independent game (such as the OCB
game) can be reproduced by a causally ordered process.  Its nonseparability is
therefore certified device-dependently, with a causal witness S:

    Tr[S W_sep] >= 0   for every causally separable process W_sep
    Tr[S W_switch] < 0

Following Araújo et al. (2015), the switch is modelled with three parties:
A (input A_I, output A_O), B (input B_I, output B_O) and a final party F with
input F_I = control ⊗ target and no output.  The target enters the first party
to act in state |psi>; the control state decides the order.

The witness shipped here is restricted to what a lab can measure: A and B each
apply one of ten unitaries, F measures the control in the |+>/|-> basis.  Its
weights were computed by SDP and verified (see `_switch_witness_data`):

    witness value on the switch (target |0>, control |+>)  = -0.1507
    witness value on any causally separable process       >= 0
    worst-case noise tolerated (generalised robustness)    = 0.1507
    optimal witness without restrictions                   = -0.5454

References:
    Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner (2015). Witnessing
    causal nonseparability. New Journal of Physics 17, 102001.
    arXiv:1506.03776, Sections IV-V (Eqs. 66, 71, 78, 85-87).
    Chiribella, D'Ariano, Perinotti, Valiron (2013). Quantum computations
    without definite causal structure. Physical Review A 88, 022318.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from qrl.causal import ProcessMatrix
from qrl._switch_witness_data import PREDICTED_VALUE, TERMS

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I = np.eye(2, dtype=complex)
_R2 = np.sqrt(2.0)

#: The ten unitaries of Araújo et al. (2015), Eq. 85.
WITNESS_UNITARIES: Dict[str, np.ndarray] = {
    "I": _I, "X": _X, "Y": _Y, "Z": _Z,
    "(X+Y)/√2": (_X + _Y) / _R2, "(X-Y)/√2": (_X - _Y) / _R2,
    "(X+Z)/√2": (_X + _Z) / _R2, "(X-Z)/√2": (_X - _Z) / _R2,
    "(Y+Z)/√2": (_Y + _Z) / _R2, "(Y-Z)/√2": (_Y - _Z) / _R2,
}

#: F's measurement outcomes on the control qubit.
CONTROL_OUTCOMES: Dict[str, np.ndarray] = {
    "+": np.array([1, 1], dtype=complex) / _R2,
    "-": np.array([1, -1], dtype=complex) / _R2,
}


def _as_density(state: Union[np.ndarray, Sequence[complex]]) -> np.ndarray:
    s = np.asarray(state, dtype=complex)
    if s.ndim == 1:
        s = s / np.linalg.norm(s)
        return np.outer(s, s.conj())
    return s


def switch_process(
    d: int = 2,
    control: Optional[Union[np.ndarray, Sequence[complex]]] = None,
    target: Optional[Union[np.ndarray, Sequence[complex]]] = None,
) -> ProcessMatrix:
    """
    Process matrix of the quantum switch with parties A, B and F.

    Subsystem order: A_I, A_O, B_I, B_O, F_c (control), F_t (target).
    For pure control a|0> + b|1> and target |psi>, W = |w><w| with

        |w> = a |psi>_{A_I} |1>>_{A_O B_I} |1>>_{B_O F_t} |0>_{F_c}
            + b |psi>_{B_I} |1>>_{B_O A_I} |1>>_{A_O F_t} |1>_{F_c}

    (Araújo et al. 2015, Eq. 66).  Mixed control or target states are
    handled linearly.  The process does not depend on the operations A and B
    perform: it is the switch itself, into which A and B plug their
    operations.

    Args:
        d:       Target dimension (default 2).
        control: Control state, vector or density matrix (default |+>).
        target:  Target input state, vector or density matrix (default |0>).

    Returns:
        ProcessMatrix with parties ['A', 'B', 'F'], input_dims [d, d, 2d],
        output_dims [d, d, 1].
    """
    rho_c = _as_density([1, 1] if control is None else control)
    rho_t = _as_density(np.eye(d)[0] if target is None else target)
    if rho_c.shape != (2, 2):
        raise ValueError("control must be a qubit state")
    if rho_t.shape != (d, d):
        raise ValueError(f"target must be a {d}-dimensional state")

    D = d ** 5 * 2
    W = np.zeros((D, D), dtype=complex)
    evals, evecs = np.linalg.eigh(rho_t)
    for lam, psi in zip(evals, evecs.T):
        if lam < 1e-14:
            continue
        # branch[k]: vector for "A first" (k=0) and "B first" (k=1), F_c excluded
        branch = np.zeros((2, d, d, d, d, d), dtype=complex)  # k, AI AO BI BO Ft
        for p in range(d):
            for i in range(d):
                for j in range(d):
                    branch[0, p, i, i, j, j] += psi[p]   # A_I=psi, A_O->B_I, B_O->F_t
                    branch[1, i, j, p, i, j] += psi[p]   # B_I=psi, B_O->A_I, A_O->F_t
        for k in range(2):
            for l in range(2):
                if abs(rho_c[k, l]) < 1e-15:
                    continue
                wk = np.zeros((d, d, d, d, 2, d), dtype=complex)
                wk[:, :, :, :, k, :] = branch[k]
                wl = np.zeros((d, d, d, d, 2, d), dtype=complex)
                wl[:, :, :, :, l, :] = branch[l]
                W += lam * rho_c[k, l] * np.outer(wk.reshape(-1), wl.reshape(-1).conj())
    return ProcessMatrix(
        W=W,
        parties=["A", "B", "F"],
        input_dims=[d, d, 2 * d],
        output_dims=[d, d, 1],
        description=f"Quantum switch (A, B, F; target d={d})",
    )


def _choi(U: np.ndarray) -> np.ndarray:
    """|U>><<U| on (input, output), with |U>> = sum_i |i> ⊗ U|i>."""
    d = U.shape[0]
    v = sum(np.kron(np.eye(d)[i], U[:, i]) for i in range(d))
    return np.outer(v, v.conj())


@dataclass(frozen=True)
class SwitchWitness:
    """
    A causal witness for the qubit quantum switch that a lab can measure.

    Each term is (U_A, U_B, outcome, weight): A applies U_A, B applies U_B,
    F measures the control and gets `outcome`.  From the outcome
    probabilities p(outcome | U_A, U_B) the witness value is

        value = sum(weight * p(outcome | U_A, U_B))

    `value >= separable_bound` for every causally separable process; a value
    below it certifies causal nonseparability (device-dependently: it assumes
    A and B really apply the stated unitaries and F really measures in the
    stated basis).

    Attributes:
        terms:            (U_A name, U_B name, F outcome, weight) tuples.
        predicted_value:  Value for the ideal switch (target |0>, control |+>).
        separable_bound:  Lower bound over all causally separable processes.
        noise_tolerance:  Worst-case noise the witness tolerates on the switch.
    """

    terms: Tuple[Tuple[str, str, str, float], ...]
    predicted_value: float
    separable_bound: float = 0.0
    noise_tolerance: float = 0.0

    @property
    def settings(self) -> Tuple[Tuple[str, str], ...]:
        """The distinct (U_A, U_B) configurations to run."""
        return tuple(sorted({(a, b) for a, b, _, _ in self.terms}))

    def operator(self, d: int = 2) -> np.ndarray:
        """The witness operator S on A_I A_O B_I B_O F_c F_t (identity on F_t)."""
        if d != 2:
            raise NotImplementedError("the shipped witness is for a qubit target")
        S = np.zeros((32, 32), dtype=complex)
        for a, b, s, c in self.terms:
            ket = CONTROL_OUTCOMES[s]
            S += c * np.kron(np.kron(_choi(WITNESS_UNITARIES[a]),
                                     _choi(WITNESS_UNITARIES[b])),
                             np.outer(ket, ket.conj()))
        return np.kron(S, np.eye(2))

    def probabilities(self, process: Union[ProcessMatrix, np.ndarray]
                      ) -> Dict[Tuple[str, str, str], float]:
        """Outcome probabilities p(outcome | U_A, U_B) this process predicts."""
        W = process.W if isinstance(process, ProcessMatrix) else np.asarray(process)
        if W.shape != (64, 64):
            raise ValueError("expected a qubit switch process (A, B, F), shape (64, 64)")
        out = {}
        for a, b, s, _ in self.terms:
            ket = CONTROL_OUTCOMES[s]
            M = np.kron(np.kron(np.kron(_choi(WITNESS_UNITARIES[a]),
                                        _choi(WITNESS_UNITARIES[b])),
                                np.outer(ket, ket.conj())), np.eye(2))
            out[(a, b, s)] = float(np.trace(M @ W).real)
        return out

    def evaluate(self, frequencies: Mapping[Tuple[str, str, str], float]) -> float:
        """Witness value from outcome frequencies, e.g. measured in a lab."""
        missing = [(a, b, s) for a, b, s, _ in self.terms if (a, b, s) not in frequencies]
        if missing:
            raise KeyError(f"missing frequencies for {len(missing)} terms, e.g. {missing[0]}")
        return float(sum(c * frequencies[(a, b, s)] for a, b, s, c in self.terms))

    def value(self, process: Union[ProcessMatrix, np.ndarray]) -> float:
        """Witness value Tr[S W] of a process."""
        return self.evaluate(self.probabilities(process))

    def certifies(self, process: Union[ProcessMatrix, np.ndarray],
                  tol: float = 1e-9) -> bool:
        """True if the process's witness value is below the separable bound."""
        return self.value(process) < self.separable_bound - tol

    def __str__(self) -> str:
        return (f"SwitchWitness({len(self.settings)} settings, "
                f"predicted={self.predicted_value:+.4f}, "
                f"separable bound={self.separable_bound:g})")


@lru_cache(maxsize=1)
def switch_witness() -> SwitchWitness:
    """The precomputed ten-unitary witness for the qubit quantum switch."""
    return SwitchWitness(
        terms=tuple(TERMS),
        predicted_value=PREDICTED_VALUE,
        separable_bound=0.0,
        noise_tolerance=-PREDICTED_VALUE,
    )
