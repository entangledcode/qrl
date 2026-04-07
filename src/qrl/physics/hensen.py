"""
Hensen et al. (2015) — Loophole-Free Bell Test

Implements the physics from:
    Hensen B. et al. (2015). Loophole-free Bell inequality violation
    using electron spins separated by 1.3 kilometres.
    Nature, 526, 682-686. doi:10.1038/nature15759

Three physical models:

1. Detection efficiency and the detection loophole
   - Critical threshold eta_crit = 2/(1+sqrt(2)) ~ 0.8284
   - Effective CHSH in loophole-free (no-click = +1) analysis

2. Heralded entanglement (event-ready Bell pairs)
   - Werner state parametrised by fidelity F
   - CHSH as a function of fidelity

3. Statistical significance
   - p-value for Bell violation (Normal approximation)
   - Required trials for a target significance level

Author: David Coldeira (dcoldeira@gmail.com)
License: MIT
"""

import numpy as np
from math import erfc, sqrt as _sqrt, ceil, log
from dataclasses import dataclass, field
from typing import Optional

_TSIRELSON = 2.0 * np.sqrt(2)    # ~2.8284
_CLASSICAL_LIMIT = 2.0


# =============================================================================
# Detection Loophole
# =============================================================================

def detection_loophole_threshold() -> float:
    """
    Minimum detection efficiency to close the detection loophole for CHSH.

    For a maximally entangled Bell state and symmetric detection efficiency
    eta on both sides, a local hidden variable (LHV) model can reproduce any
    observed correlations when eta <= eta_crit.  Above this threshold, the
    loophole-free effective S exceeds the classical bound.

    Derived from Garg and Mermin (1987):
        eta_crit = 2 / (1 + sqrt(2)) ~ 0.8284

    Returns:
        Critical detection efficiency (dimensionless).

    Example:
        >>> eta_crit = detection_loophole_threshold()
        >>> print(f"eta_crit = {eta_crit:.4f}")   # 0.8284
    """
    return 2.0 / (1.0 + np.sqrt(2))


def effective_chsh_loophole_free(eta: float, S_ideal: float = None) -> dict:
    """
    Effective CHSH parameter in a loophole-free (no-click = +1) analysis.

    When non-detection events are assigned outcome +1 (worst case for LHV),
    the effective S for a maximally entangled state is:

        S_eff(eta) = eta^2 * S_ideal + 2 * (1 - eta)^2

    Derivation: the marginals of |Phi+> are zero, so only the coincidence
    term and the double-no-click term survive.  S_eff reduces to 2.0 at
    eta=0 (classical) and S_ideal at eta=1 (perfect detectors).

    The detection loophole is closed only when S_eff > 2.

    Args:
        eta: Detection efficiency, 0 <= eta <= 1.
        S_ideal: Ideal CHSH value (no detector loss). Defaults to Tsirelson
                 bound 2*sqrt(2).

    Returns:
        dict with keys: eta, S_ideal, S_loophole_free, classical_bound,
        loophole_closed, eta_crit, note.

    Example:
        >>> r = effective_chsh_loophole_free(0.90)
        >>> r["loophole_closed"]   # True  (0.90 > eta_crit ~ 0.828)
        >>> r = effective_chsh_loophole_free(0.70)
        >>> r["loophole_closed"]   # False
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    if S_ideal is None:
        S_ideal = _TSIRELSON

    S_eff = eta**2 * S_ideal + 2.0 * (1.0 - eta)**2
    eta_crit = detection_loophole_threshold()

    return {
        "eta": eta,
        "S_ideal": round(float(S_ideal), 6),
        "S_loophole_free": round(S_eff, 6),
        "classical_bound": _CLASSICAL_LIMIT,
        "loophole_closed": bool(S_eff > _CLASSICAL_LIMIT),
        "eta_crit": round(eta_crit, 6),
        "note": (
            f"Loophole-free S at eta={eta:.3f} (no-click = +1 model). "
            f"eta_crit = {eta_crit:.4f}. "
            + ("Detection loophole CLOSED."
               if S_eff > _CLASSICAL_LIMIT
               else "Detection loophole OPEN — LHV not ruled out.")
        ),
    }


def detection_efficiency_sweep(eta_values=None) -> list:
    """
    Loophole-free CHSH S as a function of detection efficiency.

    Args:
        eta_values: List of eta values. Defaults to 0.0..1.0 in steps of 0.1.

    Returns:
        List of dicts, each with: eta, S_loophole_free, loophole_closed.

    Example:
        >>> sweep = detection_efficiency_sweep()
        >>> for r in sweep:
        ...     print(f"eta={r['eta']:.1f}  S={r['S_loophole_free']:.4f}")
    """
    if eta_values is None:
        eta_values = [round(i * 0.1, 1) for i in range(11)]
    return [
        {
            "eta": eta,
            "S_loophole_free": round(eta**2 * _TSIRELSON + 2.0 * (1.0 - eta)**2, 6),
            "loophole_closed": bool(
                eta**2 * _TSIRELSON + 2.0 * (1.0 - eta)**2 > _CLASSICAL_LIMIT
            ),
        }
        for eta in eta_values
    ]


# =============================================================================
# Heralded Entanglement
# =============================================================================

def heralded_bell_pair(fidelity: float = 1.0) -> np.ndarray:
    """
    Density matrix of a heralded Bell pair as a function of state fidelity.

    In Hensen et al. (2015), entanglement is generated via a photon herald:
    each NV centre emits a photon, and simultaneous detection at a central
    beam-splitter heralds a successful Bell pair.  Memory dephasing and
    optical losses reduce the fidelity of the heralded state below 1.

    The heralded state is modelled as a Werner state:

        rho = v * |Phi+><Phi+| + (1-v)/4 * I_4

    where visibility v = (4F - 1) / 3 and F = <Phi+|rho|Phi+> is the
    fidelity with |Phi+> = (|00> + |11>) / sqrt(2).

    Args:
        fidelity: State fidelity F, 0.25 <= F <= 1.
                  F = 1    -> ideal Bell pair (v = 1)
                  F = 0.25 -> maximally mixed (v = 0)

    Returns:
        4x4 density matrix (complex128 numpy array).

    Example:
        >>> rho = heralded_bell_pair(fidelity=0.92)   # ~Hensen et al. value
        >>> np.trace(rho)   # 1.0
    """
    if not (0.25 <= fidelity <= 1.0):
        raise ValueError(f"fidelity must be in [0.25, 1], got {fidelity}")

    v = (4.0 * fidelity - 1.0) / 3.0

    # |Phi+><Phi+| in {|00>, |01>, |10>, |11>} basis
    phi_plus_dm = 0.5 * np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
    ], dtype=complex)

    rho = v * phi_plus_dm + (1.0 - v) / 4.0 * np.eye(4, dtype=complex)
    return rho


def chsh_heralded(fidelity: float = 1.0) -> dict:
    """
    CHSH parameter for a heralded Bell pair as a function of state fidelity.

    For a Werner state with visibility v = (4F - 1) / 3:

        S(F) = 2*sqrt(2) * v = 2*sqrt(2) * (4F - 1) / 3

    CHSH violation (S > 2) requires F > (1 + 3/sqrt(2)) / 4 ~ 0.780.
    Hensen et al. (2015) achieved F ~ 0.92, giving S ~ 2.53 (model) vs
    S = 2.42 measured (additional experimental imperfections).

    Args:
        fidelity: State fidelity F, 0.25 <= F <= 1.

    Returns:
        dict with: fidelity, visibility, S, violated, fidelity_threshold,
        classical_bound, quantum_max, note.

    Example:
        >>> r = chsh_heralded(0.92)
        >>> r["S"]        # ~2.53
        >>> r["violated"] # True
    """
    if not (0.25 <= fidelity <= 1.0):
        raise ValueError(f"fidelity must be in [0.25, 1], got {fidelity}")

    v = (4.0 * fidelity - 1.0) / 3.0
    S = _TSIRELSON * v
    F_threshold = fidelity_threshold_chsh()

    return {
        "fidelity": fidelity,
        "visibility": round(v, 6),
        "S": round(S, 6),
        "violated": bool(S > _CLASSICAL_LIMIT),
        "classical_bound": _CLASSICAL_LIMIT,
        "quantum_max": round(_TSIRELSON, 6),
        "fidelity_threshold": round(F_threshold, 6),
        "note": (
            f"Werner state with visibility v={v:.4f}. "
            + ("CHSH violated — entanglement confirmed."
               if S > _CLASSICAL_LIMIT
               else f"No CHSH violation. Need F > {F_threshold:.3f}.")
        ),
    }


def fidelity_threshold_chsh() -> float:
    """
    Minimum Bell pair fidelity F for CHSH violation in a Werner state.

    Derived from S(F) = 2*sqrt(2)*(4F-1)/3 > 2:
        F_min = (1 + 3/sqrt(2)) / 4 ~ 0.7803

    Returns:
        Minimum fidelity for CHSH violation.

    Example:
        >>> F_min = fidelity_threshold_chsh()
        >>> print(f"Need F > {F_min:.4f} for Bell violation")
    """
    return (1.0 + 3.0 / np.sqrt(2)) / 4.0


# =============================================================================
# Statistical Significance
# =============================================================================

def _normal_sf(z: float) -> float:
    """One-sided survival function P(Z >= z) for standard Normal."""
    return 0.5 * erfc(z / _sqrt(2))


def _probit_upper(alpha: float, tol: float = 1e-10) -> float:
    """
    z such that P(Z >= z) = alpha (one-sided upper normal quantile).
    Uses binary search — dependency-free and exact to tol.
    """
    lo, hi = 0.0, 40.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _normal_sf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bell_p_value(N_per_setting: int, S_observed: float) -> dict:
    """
    p-value for a Bell violation under the LHV null hypothesis.

    Uses a Normal approximation for the CHSH estimator.  Under LHV:
        E[S] <= 2.0
        SE(S) = 2 / sqrt(N)    (four correlations each with SE 1/sqrt(N))

    One-sided p-value:
        p = P(S >= S_observed | LHV) ~ Phi_bar((S_observed - 2) * sqrt(N) / 2)

    Hensen et al. (2015) used the prediction-based ratio martingale test
    (Elkouss and Wehner), which is stricter but gives similar results for
    N >= 50.

    Args:
        N_per_setting: Trials per measurement setting (>= 1).
        S_observed: Observed CHSH parameter.

    Returns:
        dict with: N_per_setting, S_observed, SE, z_score, p_value,
        significant_5pct, significant_1pct, note.

    Example:
        >>> r = bell_p_value(61, 2.42)   # 245 total / 4 settings ~ Hensen
        >>> r["p_value"]                  # ~ 0.05 (Normal approx of 0.039)
        >>> r = bell_p_value(1000, 2.42)
        >>> r["p_value"]                  # << 0.001
    """
    if N_per_setting < 1:
        raise ValueError("N_per_setting must be >= 1")

    SE = 2.0 / _sqrt(N_per_setting)
    z = (S_observed - _CLASSICAL_LIMIT) / SE
    p = _normal_sf(z)

    return {
        "N_per_setting": N_per_setting,
        "S_observed": S_observed,
        "SE": round(SE, 6),
        "z_score": round(z, 4),
        "p_value": round(p, 6),
        "significant_5pct": bool(p < 0.05),
        "significant_1pct": bool(p < 0.01),
        "note": (
            f"Normal approx. SE(S) = {SE:.4f} with N={N_per_setting}/setting. "
            f"z = {z:.2f}, p = {p:.4f}."
        ),
    }


def trials_needed(target_p_value: float, S_expected: float) -> dict:
    """
    Trials per setting required to achieve a target p-value for a Bell test.

    Inverts the Normal approximation:
        N = ceil((2 * z_alpha / (S_expected - 2))^2)

    where z_alpha is the Normal quantile for significance level alpha.

    Args:
        target_p_value: Desired p-value, e.g. 0.05 for 5% significance.
        S_expected: Expected CHSH parameter under the quantum hypothesis.

    Returns:
        dict with: target_p_value, S_expected, z_required, N_per_setting,
        N_total (all 4 settings combined), note.

    Example:
        >>> r = trials_needed(0.05, 2.42)
        >>> r["N_per_setting"]   # trials per setting
        >>> r["N_total"]         # total across all 4 settings
    """
    if not (0.0 < target_p_value < 1.0):
        raise ValueError("target_p_value must be in (0, 1)")
    if S_expected <= _CLASSICAL_LIMIT:
        raise ValueError(
            f"S_expected must be > {_CLASSICAL_LIMIT} for any Bell violation"
        )

    z_required = _probit_upper(target_p_value)
    N = ceil((2.0 * z_required / (S_expected - _CLASSICAL_LIMIT)) ** 2)

    return {
        "target_p_value": target_p_value,
        "S_expected": S_expected,
        "z_required": round(z_required, 4),
        "N_per_setting": N,
        "N_total": 4 * N,
        "note": (
            f"Need {N} trials/setting ({4 * N} total) for p < {target_p_value} "
            f"with S_expected = {S_expected:.3f}."
        ),
    }


# =============================================================================
# High-Level Result and Test
# =============================================================================

@dataclass
class HensenResult:
    """Result of a loophole-free Bell test analysis."""
    eta: float
    fidelity: float
    p_herald: float
    S_loophole_free: float
    S_heralded: float
    loophole_closed: bool
    chsh_violated: bool
    eta_crit: float = field(default_factory=detection_loophole_threshold)
    note: str = ""

    @property
    def is_loophole_free_violation(self) -> bool:
        """True if detection loophole is closed AND CHSH is violated."""
        return self.loophole_closed and self.chsh_violated

    def __str__(self) -> str:
        lines = [
            "Loophole-Free Bell Test (Hensen et al. 2015)",
            "=" * 50,
            f"Detection efficiency : eta = {self.eta:.4f}",
            f"Critical threshold   : eta_crit = {self.eta_crit:.4f}",
            f"Detection loophole   : {'CLOSED' if self.loophole_closed else 'OPEN'}",
            "",
            f"State fidelity       : F = {self.fidelity:.4f}",
            f"Herald probability   : p_herald = {self.p_herald:.2e}",
            "",
            f"S (loophole-free)    : {self.S_loophole_free:.4f}",
            f"S (heralded state)   : {self.S_heralded:.4f}",
            f"Classical limit      : 2.0000",
            f"Tsirelson bound      : {_TSIRELSON:.4f}",
            "",
            f"CHSH violated        : {'YES' if self.chsh_violated else 'NO'}",
            f"Loophole-free result : {'YES' if self.is_loophole_free_violation else 'NO'}",
        ]
        if self.note:
            lines += ["", self.note]
        return "\n".join(lines)


def loophole_free_bell_test(
    eta: float = 0.92,
    fidelity: float = 0.92,
    p_herald: float = 2.4e-9,
) -> HensenResult:
    """
    Full loophole-free Bell test analysis (Hensen et al. 2015 model).

    Default parameters approximate the Hensen et al. (2015) NV-centre
    experiment at Delft (1.3 km separation):
        eta = 0.92    detection efficiency (both NV centres)
        F   = 0.92    heralded Bell pair fidelity
        p_herald = 2.4e-9   herald probability per entanglement attempt

    Args:
        eta: Detection efficiency for both parties, 0 < eta <= 1.
        fidelity: Bell pair fidelity F, 0.25 <= F <= 1.
        p_herald: Probability of successful herald per attempt (determines
                  rate, not loophole status).

    Returns:
        HensenResult with full loophole analysis.

    Example:
        >>> result = loophole_free_bell_test()
        >>> print(result)
        >>> result.is_loophole_free_violation   # True
    """
    eff = effective_chsh_loophole_free(eta)
    her = chsh_heralded(fidelity)
    eta_crit = detection_loophole_threshold()

    return HensenResult(
        eta=eta,
        fidelity=fidelity,
        p_herald=p_herald,
        S_loophole_free=eff["S_loophole_free"],
        S_heralded=her["S"],
        loophole_closed=eff["loophole_closed"],
        chsh_violated=her["violated"],
        eta_crit=eta_crit,
        note=(
            "NV-centre loophole-free Bell test. "
            f"Herald rate ~{p_herald:.1e}/attempt. "
            "Hensen et al. (2015): 245 trials, p=0.039, S=2.42."
        ),
    )


def hensen_experimental_params() -> dict:
    """
    Experimental parameters from Hensen et al. (2015).

    Returns the key numbers from the original loophole-free Bell test at Delft
    for reference and comparison with model predictions.

    Reference:
        Hensen B. et al. (2015). Loophole-free Bell inequality violation
        using electron spins separated by 1.3 kilometres.
        Nature, 526, 682-686. doi:10.1038/nature15759

    Returns:
        dict with: separation_km, qubit_type, herald_mechanism,
        p_herald_per_attempt, fidelity, detection_efficiency,
        N_trials, p_value, S_observed, year, doi, note.

    Example:
        >>> params = hensen_experimental_params()
        >>> params["separation_km"]   # 1.3
        >>> params["N_trials"]        # 245
        >>> params["p_value"]         # 0.039
    """
    return {
        "separation_km": 1.3,
        "qubit_type": "NV centre electron spin",
        "herald_mechanism": "photon entanglement swapping (beam-splitter)",
        "p_herald_per_attempt": 2.4e-9,
        "fidelity": 0.92,
        "detection_efficiency": 0.92,
        "N_trials": 245,
        "p_value": 0.039,
        "S_observed": 2.42,
        "classical_bound": 2.0,
        "tsirelson_bound": round(_TSIRELSON, 6),
        "eta_crit": round(detection_loophole_threshold(), 6),
        "year": 2015,
        "doi": "10.1038/nature15759",
        "note": (
            "First loophole-free Bell test. Closed both detection loophole "
            "(eta > eta_crit via heralded entanglement) and locality loophole "
            "(spacelike separation, fast random number generators). "
            "245 total trials."
        ),
    }
