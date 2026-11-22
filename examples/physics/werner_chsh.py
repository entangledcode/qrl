"""
Werner state CHSH violation, computed from the state.

Build |Φ+⟩⟨Φ+| as a density matrix, apply qrl.causal.depolarizing_channel(p) to
ONE qubit, and compute the CHSH parameter S = Tr[ρ B_CHSH] directly from the
resulting state. Compare with the closed form in qrl.physics.bell.chsh_noisy.

Depolarizing one qubit of |Φ+⟩ gives the Werner state

    ρ_W(p) = (1 - p) |Φ+⟩⟨Φ+| + p · I/4,

with visibility v = 1 - p, so at the optimal angles S = 2√2 (1 - p).
CHSH is violated (S > 2) iff v > 1/√2, i.e. p < 1 - 1/√2 ≈ 0.29289.

Reference:
    Werner (1989), "Quantum states with Einstein-Podolsky-Rosen correlations
    admitting a hidden-variable model", Phys. Rev. A 40, 4277.

Run:
    .venv/bin/python examples/physics/werner_chsh.py
"""

from typing import Dict, Optional, Tuple

import numpy as np

from qrl.causal import depolarizing_channel
from qrl.physics.bell import chsh_noisy, optimal_chsh_angles

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
P_CRITICAL = 1.0 - 1.0 / np.sqrt(2.0)


def phi_plus_density() -> np.ndarray:
    """|Φ+⟩⟨Φ+| for |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    phi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2.0)
    return np.outer(phi, phi.conj())


def werner_state(p: float) -> np.ndarray:
    """Depolarize qubit 0 of |Φ+⟩ with probability p."""
    return depolarizing_channel(p).apply_to_subsystem(phi_plus_density(), 0, 2)


def observable(angle: float) -> np.ndarray:
    """σ_θ = cos θ σ_z + sin θ σ_x (angle from the Z axis, in the XZ plane)."""
    return np.cos(angle) * SIGMA_Z + np.sin(angle) * SIGMA_X


def correlation(rho: np.ndarray, angle_a: float, angle_b: float) -> float:
    """E(a, b) = Tr[ρ (σ_a ⊗ σ_b)]."""
    return float(np.trace(rho @ np.kron(observable(angle_a), observable(angle_b))).real)


def chsh_from_state(rho: np.ndarray,
                    angles: Optional[Dict[str, Tuple[float, float]]] = None) -> float:
    """S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|, optimal angles by default."""
    angles = angles or optimal_chsh_angles()
    a, a2 = angles["alice"]
    b, b2 = angles["bob"]
    return abs(correlation(rho, a, b) - correlation(rho, a, b2)
               + correlation(rho, a2, b) + correlation(rho, a2, b2))


def main() -> None:
    print("Werner state CHSH: S from the state vs chsh_noisy (Werner 1989)\n")
    print(f"{'p':>9} | {'S (state)':>10} | {'S (formula)':>11} | violates")
    print("-" * 48)

    p_values = [0.0, 0.1, 0.2, 0.25, P_CRITICAL, 0.3, 0.5, 1.0]
    worst = 0.0
    for p in p_values:
        rho = werner_state(p)
        assert np.isclose(np.trace(rho).real, 1.0) and np.allclose(rho, rho.conj().T)
        s_state = chsh_from_state(rho)
        s_formula = chsh_noisy(p)["S"]
        worst = max(worst, abs(s_state - 2 * np.sqrt(2) * (1 - p)))
        label = "threshold" if p == P_CRITICAL else ("yes" if s_state > 2.0 + 1e-9 else "no")
        print(f"{p:9.5f} | {s_state:10.6f} | {s_formula:11.6f} | {label}")

    print(f"\nLargest |S(state) - 2√2(1-p)| over the table: {worst:.1e}")
    print(f"Violation iff p < 1 - 1/√2 = {P_CRITICAL:.5f}; "
          f"Tsirelson bound 2√2 = {2 * np.sqrt(2):.6f}, classical bound 2.")


if __name__ == "__main__":
    main()
