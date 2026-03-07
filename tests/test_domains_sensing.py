"""Tests for qrl.domains.sensing — Quantum Sensing domain module."""

import numpy as np
import pytest

from qrl.domains.sensing import (
    QuantumSensor,
    atomic_clock_stability,
    cramer_rao_bound,
    heisenberg_limit,
    mach_zehnder,
    quantum_advantage_factor,
    quantum_fisher_information,
    ramsey_interferometry,
    spin_squeezing,
    standard_quantum_limit,
)


# ------------------------------------------------------------------ #
# quantum_fisher_information                                           #
# ------------------------------------------------------------------ #

class TestQuantumFisherInformation:
    def _qubit_rho(self, pure=True):
        """Single qubit |+> state."""
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        return np.outer(psi, psi.conj())

    def _sigma_z(self):
        return np.array([[1, 0], [0, -1]], dtype=complex)

    def test_pure_state_ghz_n4(self):
        # GHZ state with collective Jz: QFI = n² = 16
        sensor = QuantumSensor("test", n_probes=4)
        sensor.set_state("ghz")
        sensor.set_generator("Jz")
        assert sensor.qfi() == pytest.approx(16.0, rel=1e-4)

    def test_pure_state_ghz_n2(self):
        # n=2 GHZ: QFI = 4
        sensor = QuantumSensor("test", n_probes=2)
        sensor.set_state("ghz")
        sensor.set_generator("Jz")
        assert sensor.qfi() == pytest.approx(4.0, rel=1e-4)

    def test_product_state_gives_sql_scaling(self):
        # Product state |+>^n: QFI = n (SQL)
        sensor = QuantumSensor("test", n_probes=4)
        sensor.set_state("product")
        sensor.set_generator("Jz")
        assert sensor.qfi() == pytest.approx(4.0, rel=1e-4)

    def test_ghz_better_than_product(self):
        n = 6
        ghz = QuantumSensor("ghz", n_probes=n)
        ghz.set_state("ghz").set_generator("Jz")
        prod = QuantumSensor("prod", n_probes=n)
        prod.set_state("product").set_generator("Jz")
        assert ghz.qfi() > prod.qfi()

    def test_non_negative(self):
        rho = self._qubit_rho()
        H = self._sigma_z()
        assert quantum_fisher_information(rho, H) >= 0.0

    def test_mixed_state_lower_qfi(self):
        # Fully mixed state → QFI = 0
        rho_mixed = np.eye(2, dtype=complex) / 2
        H = self._sigma_z()
        assert quantum_fisher_information(rho_mixed, H) == pytest.approx(0.0, abs=1e-10)

    def test_custom_generator(self):
        rho = self._qubit_rho()
        H = self._sigma_z()
        qfi = quantum_fisher_information(rho, H)
        assert qfi > 0


# ------------------------------------------------------------------ #
# Limit functions                                                      #
# ------------------------------------------------------------------ #

class TestLimits:
    def test_heisenberg_limit_formula(self):
        assert heisenberg_limit(4) == pytest.approx(0.25)
        assert heisenberg_limit(10) == pytest.approx(0.1)

    def test_sql_formula(self):
        assert standard_quantum_limit(4) == pytest.approx(0.5)
        assert standard_quantum_limit(100) == pytest.approx(0.1)

    def test_heisenberg_better_than_sql(self):
        for n in [2, 5, 10, 100]:
            assert heisenberg_limit(n) < standard_quantum_limit(n)

    def test_cramer_rao_positive(self):
        assert cramer_rao_bound(16.0) > 0

    def test_cramer_rao_zero_qfi(self):
        assert cramer_rao_bound(0.0) == float("inf")

    def test_cramer_rao_improves_with_measurements(self):
        qfi = 4.0
        b1 = cramer_rao_bound(qfi, 1)
        b10 = cramer_rao_bound(qfi, 10)
        assert b10 < b1

    def test_quantum_advantage_factor_ghz(self):
        # GHZ n=4: QFI=16, SQL_QFI=4 → advantage = √(16/4) = 2
        adv = quantum_advantage_factor(16.0, 4)
        assert adv == pytest.approx(2.0, rel=1e-6)

    def test_quantum_advantage_separable(self):
        # Product state: QFI=n → advantage = 1
        n = 8
        adv = quantum_advantage_factor(float(n), n)
        assert adv == pytest.approx(1.0, rel=1e-6)


# ------------------------------------------------------------------ #
# QuantumSensor class                                                  #
# ------------------------------------------------------------------ #

class TestQuantumSensor:
    def test_ghz_at_heisenberg_limit(self):
        sensor = QuantumSensor("test", n_probes=4)
        sensor.set_state("ghz").set_generator("Jz")
        prec = sensor.precision()
        hl = sensor.heisenberg_limit()
        assert prec == pytest.approx(hl, rel=1e-4)

    def test_product_at_sql(self):
        sensor = QuantumSensor("test", n_probes=4)
        sensor.set_state("product").set_generator("Jz")
        prec = sensor.precision()
        sql = sensor.standard_quantum_limit()
        assert prec == pytest.approx(sql, rel=1e-4)

    def test_quantum_advantage_ghz_n4(self):
        sensor = QuantumSensor("test", n_probes=4)
        sensor.set_state("ghz").set_generator("Jz")
        assert sensor.quantum_advantage() == pytest.approx(2.0, rel=1e-4)

    def test_dephasing_reduces_qfi(self):
        sensor = QuantumSensor("test", n_probes=4)
        sensor.set_state("ghz").set_generator("Jz")
        qfi_clean = sensor.qfi()
        sensor.add_dephasing(0.5)
        qfi_noisy = sensor.qfi()
        assert qfi_noisy < qfi_clean

    def test_summary_keys(self):
        sensor = QuantumSensor("mag", n_probes=4)
        sensor.set_state("ghz").set_generator("Jz")
        s = sensor.summary()
        for key in ["name", "n_probes", "qfi", "precision_qcrb",
                    "heisenberg_limit", "standard_quantum_limit",
                    "quantum_advantage", "at_heisenberg_limit"]:
            assert key in s

    def test_summary_at_heisenberg_limit_true_for_ghz(self):
        sensor = QuantumSensor("mag", n_probes=4)
        sensor.set_state("ghz").set_generator("Jz")
        assert sensor.summary()["at_heisenberg_limit"] is True

    def test_error_no_state(self):
        sensor = QuantumSensor("test", n_probes=4)
        with pytest.raises(RuntimeError):
            sensor.qfi()

    def test_custom_state(self):
        n = 2
        d = 2 ** n
        psi = np.zeros(d, dtype=complex)
        psi[0] = 1.0
        rho = np.outer(psi, psi.conj())
        sensor = QuantumSensor("test", n_probes=n)
        sensor.set_state(rho).set_generator("Jz")
        assert sensor.qfi() >= 0


# ------------------------------------------------------------------ #
# ramsey_interferometry                                                #
# ------------------------------------------------------------------ #

class TestRamseyInterferometry:
    def test_returns_expected_keys(self):
        result = ramsey_interferometry(1000, 100.0)
        for key in ["n_atoms", "t_us", "T2_us", "qfi", "precision_rad",
                    "decoherence_factor", "quantum_advantage"]:
            assert key in result

    def test_no_decoherence(self):
        result = ramsey_interferometry(1000, 100.0, T2_us=float("inf"))
        assert result["decoherence_factor"] == pytest.approx(1.0)

    def test_decoherence_reduces_qfi(self):
        r_clean = ramsey_interferometry(100, 10.0, T2_us=float("inf"))
        r_noisy = ramsey_interferometry(100, 10.0, T2_us=20.0)
        assert r_noisy["qfi"] < r_clean["qfi"]

    def test_more_atoms_better_precision(self):
        r10 = ramsey_interferometry(10, 100.0)
        r100 = ramsey_interferometry(100, 100.0)
        assert r100["precision_rad"] < r10["precision_rad"]

    def test_longer_time_better_precision(self):
        r_short = ramsey_interferometry(100, 10.0, T2_us=float("inf"))
        r_long = ramsey_interferometry(100, 100.0, T2_us=float("inf"))
        assert r_long["precision_rad"] < r_short["precision_rad"]

    def test_quantum_advantage_positive(self):
        result = ramsey_interferometry(100, 10.0)
        assert result["quantum_advantage"] > 0


# ------------------------------------------------------------------ #
# mach_zehnder                                                         #
# ------------------------------------------------------------------ #

class TestMachZehnder:
    def test_noon_at_heisenberg_limit(self):
        n = 10
        result = mach_zehnder(n, state="noon", loss=0.0)
        assert result["at_heisenberg_limit"] is True
        assert result["precision_rad"] == pytest.approx(heisenberg_limit(n), rel=1e-6)

    def test_coherent_at_sql(self):
        n = 10
        result = mach_zehnder(n, state="coherent", loss=0.0)
        assert result["precision_rad"] == pytest.approx(standard_quantum_limit(n), rel=1e-6)

    def test_noon_better_than_coherent(self):
        n = 10
        noon = mach_zehnder(n, state="noon", loss=0.0)
        coh = mach_zehnder(n, state="coherent", loss=0.0)
        assert noon["precision_rad"] < coh["precision_rad"]

    def test_loss_degrades_noon(self):
        n = 5
        no_loss = mach_zehnder(n, state="noon", loss=0.0)
        with_loss = mach_zehnder(n, state="noon", loss=0.1)
        assert with_loss["precision_rad"] > no_loss["precision_rad"]

    def test_returns_keys(self):
        result = mach_zehnder(4)
        for key in ["n_photons", "state", "precision_rad",
                    "heisenberg_limit_rad", "sql_rad", "quantum_advantage"]:
            assert key in result

    def test_invalid_state(self):
        with pytest.raises(ValueError):
            mach_zehnder(4, state="fock")


# ------------------------------------------------------------------ #
# spin_squeezing                                                        #
# ------------------------------------------------------------------ #

class TestSpinSqueezing:
    def test_squeezed_xi_less_than_one(self):
        result = spin_squeezing(0.5, n_atoms=100)
        assert result["is_squeezed"] is True

    def test_unsqueezed_xi_equals_one(self):
        result = spin_squeezing(1.0, n_atoms=100)
        assert result["is_squeezed"] is False

    def test_metrological_gain_db_positive_for_squeezed(self):
        result = spin_squeezing(0.1, n_atoms=100)
        assert result["metrological_gain_dB"] > 0

    def test_precision_better_than_sql(self):
        result = spin_squeezing(0.25, n_atoms=100)
        assert result["precision_rad"] < result["sql_precision_rad"]

    def test_heisenberg_limit_detection(self):
        n = 10
        result = spin_squeezing(1.0 / n, n_atoms=n)
        assert result["at_heisenberg_limit"] is True


# ------------------------------------------------------------------ #
# atomic_clock_stability                                               #
# ------------------------------------------------------------------ #

class TestAtomicClockStability:
    def test_returns_expected_keys(self):
        result = atomic_clock_stability(1000, T_cycle_s=1.0, T2_s=0.5)
        for key in ["allan_deviation", "sql_allan_deviation",
                    "heisenberg_allan_deviation", "decoherence_factor"]:
            assert key in result

    def test_more_atoms_lower_allan_deviation(self):
        r10 = atomic_clock_stability(10, 1.0, 0.5)
        r1000 = atomic_clock_stability(1000, 1.0, 0.5)
        assert r1000["allan_deviation"] < r10["allan_deviation"]

    def test_longer_averaging_lower_instability(self):
        r1 = atomic_clock_stability(100, 1.0, 0.5, tau_s=1.0)
        r100 = atomic_clock_stability(100, 1.0, 0.5, tau_s=100.0)
        assert r100["allan_deviation"] < r1["allan_deviation"]

    def test_heisenberg_allan_lower_than_sql(self):
        result = atomic_clock_stability(100, 1.0, 0.5)
        assert result["heisenberg_allan_deviation"] < result["sql_allan_deviation"]

    def test_optimal_interrogation_half_T2(self):
        T2 = 0.4
        result = atomic_clock_stability(100, T_cycle_s=1.0, T2_s=T2)
        assert result["optimal_T_interrogation_s"] == pytest.approx(T2 / 2, rel=1e-6)
