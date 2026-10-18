"""
The quantum switch construct as a process, and a measurable causal witness for it.

The quantum switch is causally nonseparable, but it cannot violate a causal
inequality: its statistics in any device-independent game (such as the OCB
game) can be reproduced by a causally ordered process.  Its nonseparability is
therefore certified device-dependently, with a causal witness S:

    Tr[S W_sep] >= 0   for every causally separable process W_sep
    Tr[S W_switch] < 0

QRL's type Switch(d) describes the switch *construct*, with the control as an
input.  Following Araújo et al. (2015), the construct is a four-party process:

    P   past party, no input; prepares the control (output P_O, a qubit)
    A   first slot  (input A_I, output A_O)
    B   second slot (input B_I, output B_O)
    F   future party, no output; receives the control (input F, a qubit)

The target enters the first slot to act in state |0> and is discarded after
the second.  Subsystem order: P_O, A_I, A_O, B_I, B_O, F.

The witness shipped here is restricted to what a lab can run: P prepares one of
six control states, A and B each apply one of ten unitaries, F measures the
control in the |+>/|-> basis.  Its weights were computed by SDP and verified
(see `_switch_witness_data`):

    witness value on the switch construct            = -0.1694
    witness value on any causally separable process  >= 0
    optimal witness without lab restrictions         = -7/9 (worst-case noise)

The certificate is device-dependent: it assumes A and B really apply the stated
unitaries, P really prepares the stated states and F really measures in the
stated basis.

References:
    Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner (2015). Witnessing
    causal nonseparability. New Journal of Physics 17, 102001.
    arXiv:1506.03776, Sections II, IV and V.
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

#: Single-qubit states: P's control preparations and F's outcomes.
QUBIT_STATES: Dict[str, np.ndarray] = {
    "0": np.array([1, 0], dtype=complex),
    "1": np.array([0, 1], dtype=complex),
    "+": np.array([1, 1], dtype=complex) / _R2,
    "-": np.array([1, -1], dtype=complex) / _R2,
    "+i": np.array([1, 1j], dtype=complex) / _R2,
    "-i": np.array([1, -1j], dtype=complex) / _R2,
}

_DIM = 64  # P_O A_I A_O B_I B_O F, one qubit each


def _as_density(state: Union[np.ndarray, Sequence[complex]]) -> np.ndarray:
    s = np.asarray(state, dtype=complex)
    if s.ndim == 1:
        s = s / np.linalg.norm(s)
        return np.outer(s, s.conj())
    return s


def switch_process(
    d: int = 2,
    target: Optional[Union[np.ndarray, Sequence[complex]]] = None,
) -> ProcessMatrix:
    """
    Process matrix of the quantum switch construct, parties P, A, B, F.

    For target |psi>, W = sum_t |w_t><w_t| (t = the discarded target output) with

        |w_t> = |0>_{P_O} |0>_F  |psi>_{A_I} |1>>_{A_O B_I} |t>_{B_O}
              + |1>_{P_O} |1>_F  |psi>_{B_I} |1>>_{B_O A_I} |t>_{A_O}

    P's output decides the order and reaches F.  The process does not depend
    on the operations A and B perform, nor on the control state: P chooses the
    control when the construct is run.

    Args:
        d:      Target dimension (default 2).
        target: Target input state, vector or density matrix (default |0>).

    Returns:
        ProcessMatrix with parties ['P', 'A', 'B', 'F'],
        input_dims [1, d, d, 2], output_dims [2, d, d, 1].
    """
    rho_t = _as_density(np.eye(d)[0] if target is None else target)
    if rho_t.shape != (d, d):
        raise ValueError(f"target must be a {d}-dimensional state")

    D = 4 * d ** 4
    W = np.zeros((D, D), dtype=complex)
    evals, evecs = np.linalg.eigh(rho_t)
    for lam, psi in zip(evals, evecs.T):
        if lam < 1e-14:
            continue
        for t in range(d):
            w = np.zeros((2, d, d, d, d, 2), dtype=complex)   # P_O A_I A_O B_I B_O F
            for p in range(d):
                for i in range(d):
                    w[0, p, i, i, t, 0] += psi[p]   # A first: A_I = psi, A_O -> B_I
                    w[1, i, t, p, i, 1] += psi[p]   # B first: B_I = psi, B_O -> A_I
            v = w.reshape(-1)
            W += lam * np.outer(v, v.conj())
    return ProcessMatrix(
        W=W,
        parties=["P", "A", "B", "F"],
        input_dims=[1, d, d, 2],
        output_dims=[2, d, d, 1],
        description=f"Quantum switch construct (P, A, B, F; target d={d})",
    )


def _choi_unitary(U: np.ndarray) -> np.ndarray:
    """Choi operator of rho -> U rho U^dag on (input, output), transposed
    (Araújo et al. 2015 convention): |U*>><<U*| with |U>> = sum_i |i> (x) U|i>."""
    d = U.shape[0]
    v = sum(np.kron(np.eye(d)[i], U[:, i]) for i in range(d))
    return np.outer(v, v.conj()).conj()


def _choi_state(ket: np.ndarray) -> np.ndarray:
    """Choi operator of preparing |ket> (no input), transposed: conj(|ket><ket|)."""
    return np.outer(ket, ket.conj()).conj()


def _setting_operator(p: str, a: str, b: str, s: str) -> np.ndarray:
    ket = QUBIT_STATES[s]
    return np.kron(np.kron(np.kron(_choi_state(QUBIT_STATES[p]),
                                   _choi_unitary(WITNESS_UNITARIES[a])),
                           _choi_unitary(WITNESS_UNITARIES[b])),
                   np.outer(ket, ket.conj()))


@dataclass(frozen=True)
class SwitchWitness:
    """
    A causal witness for the qubit quantum switch construct that a lab can run.

    Each term is (P preparation, U_A, U_B, outcome, weight): P prepares the
    control, A applies U_A, B applies U_B, F measures the control and gets
    `outcome`.  From the outcome probabilities the witness value is

        value = sum(weight * p(outcome | P preparation, U_A, U_B))

    `value >= separable_bound` for every causally separable process; a value
    below it certifies causal nonseparability (device-dependently).

    Attributes:
        terms:            (P prep, U_A, U_B, F outcome, weight) tuples.
        predicted_value:  Value for the ideal switch construct (target |0>).
        separable_bound:  Lower bound over all causally separable processes.
        noise_tolerance:  Worst-case noise the witness tolerates on the switch.
    """

    terms: Tuple[Tuple[str, str, str, str, float], ...]
    predicted_value: float
    separable_bound: float = 0.0
    noise_tolerance: float = 0.0

    @property
    def settings(self) -> Tuple[Tuple[str, str, str], ...]:
        """The distinct (P preparation, U_A, U_B) configurations to run."""
        return tuple(sorted({(p, a, b) for p, a, b, _, _ in self.terms}))

    def operator(self) -> np.ndarray:
        """The witness operator S on P_O A_I A_O B_I B_O F."""
        S = np.zeros((_DIM, _DIM), dtype=complex)
        for p, a, b, s, c in self.terms:
            S += c * _setting_operator(p, a, b, s)
        return S

    def probabilities(self, process: Union[ProcessMatrix, np.ndarray]
                      ) -> Dict[Tuple[str, str, str, str], float]:
        """Outcome probabilities this process predicts for every term."""
        W = process.W if isinstance(process, ProcessMatrix) else np.asarray(process)
        if W.shape != (_DIM, _DIM):
            raise ValueError("expected a qubit switch process (P, A, B, F), shape (64, 64)")
        return {(p, a, b, s): float(np.trace(_setting_operator(p, a, b, s) @ W).real)
                for p, a, b, s, _ in self.terms}

    def evaluate(self, frequencies: Mapping[Tuple[str, str, str, str], float]) -> float:
        """Witness value from outcome frequencies, e.g. measured in a lab."""
        missing = [t[:4] for t in self.terms if t[:4] not in frequencies]
        if missing:
            raise KeyError(f"missing frequencies for {len(missing)} terms, e.g. {missing[0]}")
        return float(sum(c * frequencies[(p, a, b, s)] for p, a, b, s, c in self.terms))

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
    """The precomputed lab witness for the qubit quantum switch construct."""
    return SwitchWitness(
        terms=tuple(TERMS),
        predicted_value=PREDICTED_VALUE,
        separable_bound=0.0,
        noise_tolerance=-PREDICTED_VALUE,
    )
