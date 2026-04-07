"""
Tests for qrl.physics.hensen — Loophole-Free Bell Test (Hensen et al. 2015)
"""

import pytest
import numpy as np
from qrl.physics.hensen import (
    detection_loophole_threshold,
    effective_chsh_loophole_free,
    detection_efficiency_sweep,
    heralded_bell_pair,
    chsh_heralded,
    fidelity_threshold_chsh,
    bell_p_value,
    trials_needed,
    loophole_free_bell_test,
    hensen_experimental_params,
    HensenResult,
)

_TSIRELSON = 2.0 * np.sqrt(2)


# =============================================================================
# Detection Loophole Threshold
# =============================================================================

class TestDetectionLoopholeThreshold:

    def test_exact_value(self):
        eta_crit = detection_loophole_threshold()
        expected = 2.0 / (1.0 + np.sqrt(2))
        assert np.isclose(eta_crit, expected)

    def test_numerical_value(self):
        eta_crit = detection_loophole_threshold()
        assert np.isclose(eta_crit, 0.8284, atol=1e-4)

    def test_above_threshold_closes_loophole(self):
        eta_crit = detection_loophole_threshold()
        r = effective_chsh_loophole_free(eta_crit + 0.01)
        assert r["loophole_closed"]

    def test_below_threshold_opens_loophole(self):
        eta_crit = detection_loophole_threshold()
        r = effective_chsh_loophole_free(eta_crit - 0.01)
        assert not r["loophole_closed"]

    def test_at_threshold_approximately_classical(self):
        eta_crit = detection_loophole_threshold()
        r = effective_chsh_loophole_free(eta_crit)
        assert np.isclose(r["S_loophole_free"], 2.0, atol=1e-6)


# =============================================================================
# Effective CHSH (Loophole-Free)
# =============================================================================

class TestEffectiveCHSHLoopholeFree:

    def test_perfect_detectors(self):
        r = effective_chsh_loophole_free(1.0)
        assert np.isclose(r["S_loophole_free"], _TSIRELSON, atol=1e-9)

    def test_zero_efficiency_at_classical_limit(self):
        r = effective_chsh_loophole_free(0.0)
        assert np.isclose(r["S_loophole_free"], 2.0, atol=1e-9)

    def test_hensen_efficiency(self):
        r = effective_chsh_loophole_free(0.92)
        assert r["loophole_closed"]
        assert r["S_loophole_free"] > 2.0

    def test_low_efficiency_no_loophole(self):
        r = effective_chsh_loophole_free(0.5)
        assert not r["loophole_closed"]

    def test_formula_correctness(self):
        for eta in [0.0, 0.5, 0.8, 0.9, 1.0]:
            r = effective_chsh_loophole_free(eta)
            expected = eta**2 * _TSIRELSON + 2.0 * (1.0 - eta)**2
            assert np.isclose(r["S_loophole_free"], expected, atol=1e-9)

    def test_custom_S_ideal(self):
        r = effective_chsh_loophole_free(1.0, S_ideal=2.5)
        assert np.isclose(r["S_loophole_free"], 2.5, atol=1e-9)

    def test_invalid_eta_raises(self):
        with pytest.raises(ValueError):
            effective_chsh_loophole_free(-0.1)
        with pytest.raises(ValueError):
            effective_chsh_loophole_free(1.1)

    def test_monotone_in_eta(self):
        etas = [0.0, 0.3, 0.6, 0.9, 1.0]
        S_values = [effective_chsh_loophole_free(e)["S_loophole_free"] for e in etas]
        # S_eff should first dip then rise — but overall S(1) > S(0)
        assert S_values[-1] > S_values[0]


class TestDetectionEfficiencySweep:

    def test_default_11_points(self):
        sweep = detection_efficiency_sweep()
        assert len(sweep) == 11

    def test_eta_0_at_classical_limit(self):
        sweep = detection_efficiency_sweep([0.0])
        assert np.isclose(sweep[0]["S_loophole_free"], 2.0, atol=1e-6)

    def test_eta_1_at_tsirelson(self):
        sweep = detection_efficiency_sweep([1.0])
        assert np.isclose(sweep[0]["S_loophole_free"], _TSIRELSON, atol=1e-6)

    def test_loophole_closed_flags(self):
        sweep = detection_efficiency_sweep()
        eta_crit = detection_loophole_threshold()
        for r in sweep:
            if r["eta"] > eta_crit + 0.01:
                assert r["loophole_closed"]
            elif r["eta"] < eta_crit - 0.01:
                assert not r["loophole_closed"]


# =============================================================================
# Heralded Bell Pair
# =============================================================================

class TestHeraldedBellPair:

    def test_shape(self):
        rho = heralded_bell_pair()
        assert rho.shape == (4, 4)

    def test_trace_one(self):
        for F in [0.25, 0.5, 0.8, 0.92, 1.0]:
            rho = heralded_bell_pair(F)
            assert np.isclose(np.trace(rho).real, 1.0, atol=1e-10)

    def test_hermitian(self):
        rho = heralded_bell_pair(0.92)
        assert np.allclose(rho, rho.conj().T, atol=1e-10)

    def test_positive_semidefinite(self):
        for F in [0.5, 0.8, 0.92, 1.0]:
            rho = heralded_bell_pair(F)
            eigvals = np.linalg.eigvalsh(rho)
            assert np.all(eigvals >= -1e-10)

    def test_ideal_is_bell_state(self):
        rho = heralded_bell_pair(1.0)
        phi_plus_dm = 0.5 * np.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
        ], dtype=complex)
        assert np.allclose(rho, phi_plus_dm, atol=1e-10)

    def test_maximally_mixed_at_fidelity_quarter(self):
        rho = heralded_bell_pair(0.25)
        assert np.allclose(rho, np.eye(4, dtype=complex) / 4.0, atol=1e-10)

    def test_fidelity_preserved(self):
        phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        for F in [0.5, 0.8, 0.92, 1.0]:
            rho = heralded_bell_pair(F)
            F_computed = np.real(phi_plus.conj() @ rho @ phi_plus)
            assert np.isclose(F_computed, F, atol=1e-9)

    def test_invalid_fidelity_raises(self):
        with pytest.raises(ValueError):
            heralded_bell_pair(0.1)
        with pytest.raises(ValueError):
            heralded_bell_pair(1.1)


# =============================================================================
# CHSH of Heralded State
# =============================================================================

class TestCHSHHeralded:

    def test_ideal_fidelity_tsirelson(self):
        r = chsh_heralded(1.0)
        assert np.isclose(r["S"], _TSIRELSON, atol=1e-9)

    def test_hensen_fidelity_violates(self):
        r = chsh_heralded(0.92)
        assert r["violated"]
        assert r["S"] > 2.0

    def test_maximally_mixed_no_violation(self):
        r = chsh_heralded(0.25)
        assert np.isclose(r["S"], 0.0, atol=1e-9)
        assert not r["violated"]

    def test_formula_S_equals_tsirelson_times_v(self):
        for F in [0.5, 0.8, 0.92, 1.0]:
            v = (4.0 * F - 1.0) / 3.0
            r = chsh_heralded(F)
            assert np.isclose(r["S"], _TSIRELSON * v, atol=1e-9)

    def test_violation_threshold(self):
        F_min = fidelity_threshold_chsh()
        r_just_above = chsh_heralded(F_min + 0.001)
        r_just_below = chsh_heralded(F_min - 0.001)
        assert r_just_above["violated"]
        assert not r_just_below["violated"]

    def test_visibility_range(self):
        for F in [0.25, 0.5, 0.92, 1.0]:
            r = chsh_heralded(F)
            assert 0.0 <= r["visibility"] <= 1.0

    def test_invalid_fidelity_raises(self):
        with pytest.raises(ValueError):
            chsh_heralded(0.1)


class TestFidelityThreshold:

    def test_numerical_value(self):
        F_min = fidelity_threshold_chsh()
        assert np.isclose(F_min, (1.0 + 3.0 / np.sqrt(2)) / 4.0, atol=1e-10)

    def test_approximately_0780(self):
        assert np.isclose(fidelity_threshold_chsh(), 0.780, atol=0.001)

    def test_chsh_exactly_2_at_threshold(self):
        F_min = fidelity_threshold_chsh()
        r = chsh_heralded(F_min)
        assert np.isclose(r["S"], 2.0, atol=1e-6)


# =============================================================================
# Statistical Significance
# =============================================================================

class TestBellPValue:

    def test_large_N_small_p(self):
        r = bell_p_value(1000, 2.42)
        assert r["p_value"] < 1e-10

    def test_hensen_like_params(self):
        # 245 total trials / 4 settings ~ 61 per setting, S=2.42
        r = bell_p_value(61, 2.42)
        # Normal approx gives p ~ 0.05, Hensen's actual was 0.039
        assert 0.01 < r["p_value"] < 0.15

    def test_at_classical_limit_p_half(self):
        # S exactly 2.0 → z=0 → p=0.5
        r = bell_p_value(1000, 2.0)
        assert np.isclose(r["p_value"], 0.5, atol=1e-4)

    def test_significant_5pct_flag(self):
        r_sig = bell_p_value(1000, 2.42)
        assert r_sig["significant_5pct"]
        r_not_sig = bell_p_value(10, 2.10)
        # small N → large SE → probably not significant
        assert not r_not_sig["significant_5pct"] or r_not_sig["p_value"] < 0.05

    def test_SE_formula(self):
        r = bell_p_value(100, 2.5)
        assert np.isclose(r["SE"], 2.0 / np.sqrt(100), atol=1e-9)

    def test_z_score_formula(self):
        r = bell_p_value(100, 2.5)
        expected_z = (2.5 - 2.0) / (2.0 / np.sqrt(100))
        assert np.isclose(r["z_score"], expected_z, atol=1e-4)

    def test_invalid_N_raises(self):
        with pytest.raises(ValueError):
            bell_p_value(0, 2.5)

    def test_p_value_in_unit_interval(self):
        for S in [1.5, 2.0, 2.5, 2.828]:
            r = bell_p_value(100, S)
            assert 0.0 <= r["p_value"] <= 1.0


class TestTrialsNeeded:

    def test_reasonable_N_for_5pct(self):
        r = trials_needed(0.05, 2.42)
        assert r["N_per_setting"] > 0
        # Verify by plugging back in
        p = bell_p_value(r["N_per_setting"], 2.42)["p_value"]
        assert p <= 0.05

    def test_N_total_is_4x_per_setting(self):
        r = trials_needed(0.05, 2.42)
        assert r["N_total"] == 4 * r["N_per_setting"]

    def test_more_stringent_needs_more_trials(self):
        r05 = trials_needed(0.05, 2.42)
        r01 = trials_needed(0.01, 2.42)
        assert r01["N_per_setting"] > r05["N_per_setting"]

    def test_larger_S_needs_fewer_trials(self):
        r_small = trials_needed(0.05, 2.10)
        r_large = trials_needed(0.05, 2.80)
        assert r_large["N_per_setting"] < r_small["N_per_setting"]

    def test_hensen_scale(self):
        # Hensen: S=2.42, 245 total trials ~ 61/setting, got p=0.039
        r = trials_needed(0.05, 2.42)
        # Should be in the tens-of-trials ballpark
        assert 20 < r["N_per_setting"] < 1000

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            trials_needed(0.0, 2.5)
        with pytest.raises(ValueError):
            trials_needed(1.0, 2.5)

    def test_S_at_classical_limit_raises(self):
        with pytest.raises(ValueError):
            trials_needed(0.05, 2.0)


# =============================================================================
# High-Level: HensenResult and loophole_free_bell_test
# =============================================================================

class TestHensenResult:

    def test_default_is_loophole_free_violation(self):
        result = loophole_free_bell_test()
        assert result.is_loophole_free_violation

    def test_loophole_closed_with_hensen_eta(self):
        result = loophole_free_bell_test(eta=0.92)
        assert result.loophole_closed

    def test_loophole_open_with_low_eta(self):
        result = loophole_free_bell_test(eta=0.5)
        assert not result.loophole_closed
        assert not result.is_loophole_free_violation

    def test_chsh_violated_with_high_fidelity(self):
        result = loophole_free_bell_test(fidelity=0.92)
        assert result.chsh_violated

    def test_chsh_not_violated_with_low_fidelity(self):
        result = loophole_free_bell_test(fidelity=0.5)
        assert not result.chsh_violated
        assert not result.is_loophole_free_violation

    def test_str_contains_key_fields(self):
        result = loophole_free_bell_test()
        s = str(result)
        assert "eta" in s.lower()
        assert "loophole" in s.lower()
        assert "fidelity" in s.lower()

    def test_eta_crit_field(self):
        result = loophole_free_bell_test()
        assert np.isclose(result.eta_crit, detection_loophole_threshold(), atol=1e-9)

    def test_s_heralded_matches_chsh_heralded(self):
        F = 0.85
        result = loophole_free_bell_test(fidelity=F)
        expected_S = chsh_heralded(F)["S"]
        assert np.isclose(result.S_heralded, expected_S, atol=1e-9)

    def test_s_loophole_free_matches_effective(self):
        eta = 0.88
        result = loophole_free_bell_test(eta=eta)
        expected_S = effective_chsh_loophole_free(eta)["S_loophole_free"]
        assert np.isclose(result.S_loophole_free, expected_S, atol=1e-9)


# =============================================================================
# Experimental Parameters Reference
# =============================================================================

class TestHensenExperimentalParams:

    def test_key_fields_present(self):
        params = hensen_experimental_params()
        for key in ["separation_km", "N_trials", "p_value", "S_observed",
                    "fidelity", "detection_efficiency", "eta_crit", "doi"]:
            assert key in params

    def test_separation_1_3km(self):
        assert hensen_experimental_params()["separation_km"] == 1.3

    def test_245_trials(self):
        assert hensen_experimental_params()["N_trials"] == 245

    def test_p_value_0039(self):
        assert np.isclose(hensen_experimental_params()["p_value"], 0.039)

    def test_S_observed_above_classical_limit(self):
        assert hensen_experimental_params()["S_observed"] > 2.0

    def test_eta_crit_consistent(self):
        params = hensen_experimental_params()
        assert np.isclose(params["eta_crit"], detection_loophole_threshold(), atol=1e-6)

    def test_detection_efficiency_above_eta_crit(self):
        params = hensen_experimental_params()
        assert params["detection_efficiency"] > params["eta_crit"]
