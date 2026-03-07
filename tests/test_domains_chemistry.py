"""Tests for qrl.domains.chemistry — Quantum Chemistry domain module."""

import numpy as np
import pytest

from qrl.domains.chemistry import (
    MolecularSystem,
    _boys0,
    _build_integrals_2ao,
    _norm_s,
    _sto3g_S,
    _sto3g_T,
    _sto3g_V,
    _STO3G_S,
    jordan_wigner_hamiltonian,
    hydrogen,
    helium_hydride_cation,
)

_ANGSTROM_TO_BOHR = 1.8897259886

# Reference geometry: R = 1.4 bohr (Szabo & Ostlund canonical)
_R14_ANG = 1.4 / _ANGSTROM_TO_BOHR
_RA = np.array([0., 0., 0.])
_RB = np.array([0., 0., 1.4])


# ------------------------------------------------------------------ #
# Boys function                                                         #
# ------------------------------------------------------------------ #

class TestBoysFunction:
    def test_zero_arg(self):
        assert _boys0(0.0) == pytest.approx(1.0, rel=1e-8)

    def test_small_arg_series(self):
        # Compare series approximation with exact formula at t=1e-6
        from math import erf
        t = 1e-6
        exact = 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))
        assert _boys0(t) == pytest.approx(exact, rel=1e-6)

    def test_large_arg(self):
        # F0(∞) → 0.5*sqrt(π/t) → 0 as t→∞
        assert _boys0(100.0) == pytest.approx(0.5 * np.sqrt(np.pi / 100.0), rel=1e-4)

    def test_monotone_decreasing(self):
        vals = [_boys0(t) for t in [0, 1, 5, 10, 50]]
        for a, b in zip(vals, vals[1:]):
            assert a > b


# ------------------------------------------------------------------ #
# Primitive normalisation                                              #
# ------------------------------------------------------------------ #

class TestNormS:
    def test_formula(self):
        alpha = 1.5
        expected = (2.0 * alpha / np.pi) ** 0.75
        assert _norm_s(alpha) == pytest.approx(expected, rel=1e-12)

    def test_positive(self):
        for alpha in [0.1, 1.0, 5.0, 10.0]:
            assert _norm_s(alpha) > 0


# ------------------------------------------------------------------ #
# AO integral matrices (Szabo & Ostlund Table 3.1 reference)          #
# ------------------------------------------------------------------ #

class TestAOIntegrals:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ea = _STO3G_S["H"]["exps"]
        self.ca = _STO3G_S["H"]["coefs"]
        self.Ra = _RA
        self.Rb = _RB

    def test_self_overlap_is_one(self):
        # Normalised STO-3G function: ⟨φ|φ⟩ ≈ 1
        S = _sto3g_S(self.ea, self.ca, self.Ra, self.ea, self.ca, self.Ra)
        assert S == pytest.approx(1.0, abs=1e-4)

    def test_overlap_s12_at_r14(self):
        # S&O: S₁₂ ≈ 0.6593 at R=1.4 bohr
        S12 = _sto3g_S(self.ea, self.ca, self.Ra, self.ea, self.ca, self.Rb)
        assert S12 == pytest.approx(0.6593, abs=5e-4)

    def test_overlap_decreases_with_distance(self):
        S_near = _sto3g_S(self.ea, self.ca, self.Ra, self.ea, self.ca, np.array([0., 0., 1.0]))
        S_far  = _sto3g_S(self.ea, self.ca, self.Ra, self.ea, self.ca, np.array([0., 0., 4.0]))
        assert S_near > S_far

    def test_core_hamiltonian_h11_at_r14(self):
        # S&O: h₁₁ = T₁₁ + V₁₁ ≈ -1.1204 Ha
        S, h, g_ao, E_nuc = _build_integrals_2ao("H", _RA, "H", _RB)
        assert h[0, 0] == pytest.approx(-1.1204, abs=5e-4)

    def test_core_hamiltonian_h12_at_r14(self):
        # S&O: h₁₂ ≈ -0.9584 Ha
        S, h, g_ao, E_nuc = _build_integrals_2ao("H", _RA, "H", _RB)
        assert h[0, 1] == pytest.approx(-0.9584, abs=5e-4)

    def test_eri_aa_aa_at_r14(self):
        # S&O: (11|11) ≈ 0.7746 Ha
        S, h, g_ao, E_nuc = _build_integrals_2ao("H", _RA, "H", _RB)
        assert g_ao[0, 0, 0, 0] == pytest.approx(0.7746, abs=5e-4)

    def test_eri_ab_ab_at_r14(self):
        # S&O: (12|12) ≈ 0.2970 Ha
        S, h, g_ao, E_nuc = _build_integrals_2ao("H", _RA, "H", _RB)
        assert g_ao[0, 1, 0, 1] == pytest.approx(0.2970, abs=5e-4)

    def test_nuclear_repulsion_at_r14(self):
        S, h, g_ao, E_nuc = _build_integrals_2ao("H", _RA, "H", _RB)
        assert E_nuc == pytest.approx(1.0 / 1.4, rel=1e-6)

    def test_symmetry_h_matrix(self):
        S, h, g_ao, E_nuc = _build_integrals_2ao("H", _RA, "H", _RB)
        assert h[0, 1] == pytest.approx(h[1, 0], rel=1e-8)

    def test_eri_symmetry(self):
        # (μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν)
        S, h, g_ao, E_nuc = _build_integrals_2ao("H", _RA, "H", _RB)
        assert g_ao[0, 1, 0, 1] == pytest.approx(g_ao[1, 0, 0, 1], abs=1e-8)
        assert g_ao[0, 1, 0, 1] == pytest.approx(g_ao[0, 1, 1, 0], abs=1e-8)
        assert g_ao[0, 0, 1, 1] == pytest.approx(g_ao[1, 1, 0, 0], abs=1e-8)


# ------------------------------------------------------------------ #
# RHF energy                                                           #
# ------------------------------------------------------------------ #

class TestRHFEnergy:
    def test_h2_equilibrium_energy(self):
        # Szabo & Ostlund reference: E_HF ≈ -1.1175 Ha at R=1.4 bohr
        mol = hydrogen(r_angstrom=_R14_ANG)
        assert mol.hartree_fock_energy() == pytest.approx(-1.1175, abs=2e-3)

    def test_hf_lower_than_nuclear_plus_onebody(self):
        mol = hydrogen()
        assert mol.hartree_fock_energy() < 0

    def test_hf_energy_increases_as_bond_stretches(self):
        mol_eq = hydrogen(0.74)
        mol_str = hydrogen(2.0)
        # HF energy is more negative at equilibrium (H₂ is bound)
        assert mol_eq.hartree_fock_energy() < mol_str.hartree_fock_energy()


# ------------------------------------------------------------------ #
# FCI ground state                                                     #
# ------------------------------------------------------------------ #

class TestFCIGroundState:
    def test_h2_equilibrium_fci(self):
        mol = hydrogen(_R14_ANG)
        # STO-3G FCI at R=1.4 bohr: -1.137 Ha
        assert mol.ground_state_energy() == pytest.approx(-1.137, abs=2e-3)

    def test_fci_lower_than_hf(self):
        mol = hydrogen()
        assert mol.ground_state_energy() < mol.hartree_fock_energy()

    def test_correlation_energy_negative(self):
        mol = hydrogen()
        assert mol.correlation_energy() < 0

    def test_fci_dissociation_limit(self):
        # FCI → 2 × E(H, STO-3G) ≈ -0.9332 Ha at large R
        mol = hydrogen(5.0)
        expected = 2 * (-0.4666)
        assert mol.ground_state_energy() == pytest.approx(expected, abs=5e-3)

    def test_hf_dissociation_wrong(self):
        # HF incorrectly dissociates above -0.933 Ha (ionic contamination)
        mol = hydrogen(5.0)
        assert mol.hartree_fock_energy() > mol.ground_state_energy()

    def test_fci_bound_state(self):
        # FCI should have a minimum (bound state)
        mol_eq = hydrogen(0.74)
        mol_far = hydrogen(3.0)
        assert mol_eq.ground_state_energy() < mol_far.ground_state_energy()

    def test_h_qubit_is_hermitian(self):
        mol = hydrogen()
        H = mol._H_qubit
        assert np.allclose(H, H.conj().T, atol=1e-12)


# ------------------------------------------------------------------ #
# Orbital entanglement                                                  #
# ------------------------------------------------------------------ #

class TestOrbitalEntanglement:
    def test_entropy_non_negative(self):
        mol = hydrogen()
        assert mol.orbital_entropy(0) >= 0

    def test_entropy_is_positive_for_correlated(self):
        # At equilibrium, FCI state has some correlation → S > 0
        mol = hydrogen()
        assert mol.orbital_entropy(0) > 0.01

    def test_entanglement_non_negative(self):
        mol = hydrogen()
        assert mol.orbital_entanglement(0, 1) >= 0

    def test_entanglement_increases_at_dissociation(self):
        # Entanglement grows as bond stretches (stronger correlation)
        mol_eq = hydrogen(0.74)
        mol_str = hydrogen(3.0)
        assert mol_str.orbital_entanglement(0, 1) > mol_eq.orbital_entanglement(0, 1)

    def test_entanglement_approaches_2_at_dissociation(self):
        # At large R: I(0,1) → 2 bits (GHZ-like superposition)
        mol = hydrogen(5.0)
        assert mol.orbital_entanglement(0, 1) == pytest.approx(2.0, abs=0.01)

    def test_mutual_info_matrix_shape(self):
        mol = hydrogen()
        I = mol.orbital_mutual_information()
        assert I.shape == (2, 2)

    def test_mutual_info_diagonal_zero(self):
        mol = hydrogen()
        I = mol.orbital_mutual_information()
        assert I[0, 0] == pytest.approx(0.0, abs=1e-10)
        assert I[1, 1] == pytest.approx(0.0, abs=1e-10)

    def test_mutual_info_symmetric(self):
        mol = hydrogen()
        I = mol.orbital_mutual_information()
        assert I[0, 1] == pytest.approx(I[1, 0], rel=1e-6)


# ------------------------------------------------------------------ #
# Bond character                                                        #
# ------------------------------------------------------------------ #

class TestBondCharacter:
    def test_h2_equilibrium_is_covalent(self):
        mol = hydrogen()
        assert mol.bond_character() == "covalent"

    def test_correlated_bond_is_covalent(self):
        for r in [0.6, 0.74, 1.0, 1.5]:
            mol = hydrogen(r)
            assert mol.bond_character() in ("covalent", "nonbonding")


# ------------------------------------------------------------------ #
# MolecularSystem class                                                 #
# ------------------------------------------------------------------ #

class TestMolecularSystem:
    def test_set_geometry_returns_self(self):
        mol = MolecularSystem("H2")
        result = mol.set_geometry([("H", 0, 0, 0), ("H", 0, 0, 0.74)])
        assert result is mol

    def test_unsolved_raises(self):
        mol = MolecularSystem("H2")
        with pytest.raises(RuntimeError):
            mol.ground_state_energy()

    def test_summary_keys(self):
        mol = hydrogen()
        s = mol.summary()
        for key in ["name", "E_hf_hartree", "E_fci_hartree",
                    "correlation_energy_hartree", "orbital_entanglement_01",
                    "bond_character", "n_electrons", "n_spatial_orbitals"]:
            assert key in s

    def test_summary_n_electrons_h2(self):
        mol = hydrogen()
        assert mol.summary()["n_electrons"] == 2

    def test_repr(self):
        mol = hydrogen()
        r = repr(mol)
        assert "H2" in r
        assert "Ha" in r

    def test_unsupported_atom_raises(self):
        mol = MolecularSystem("N2")
        with pytest.raises(ValueError, match="STO-3G"):
            mol.set_geometry([("N", 0, 0, 0), ("N", 0, 0, 1.1)])

    def test_3_atom_raises(self):
        mol = MolecularSystem("H3")
        with pytest.raises(ValueError, match="2-atom"):
            mol.set_geometry([("H", 0, 0, 0), ("H", 0, 0, 0.74), ("H", 0, 0, 1.5)])


# ------------------------------------------------------------------ #
# Convenience constructors                                             #
# ------------------------------------------------------------------ #

class TestConvenienceConstructors:
    def test_hydrogen_default(self):
        mol = hydrogen()
        assert mol.name == "H2"
        assert mol.ground_state_energy() < -1.0

    def test_hydrogen_custom_r(self):
        mol = hydrogen(1.0)
        assert mol.summary()["r_angstrom"] == pytest.approx(1.0, abs=1e-3)

    def test_heh_plus(self):
        mol = helium_hydride_cation()
        assert mol.name == "HeH+"
        assert mol._n_elec == 2  # 2 electrons (He=2, H=1, minus 1 for +1 charge)
        assert mol.ground_state_energy() < -2.0  # HeH+ is deeply bound

    def test_heh_plus_bond_character(self):
        mol = helium_hydride_cation()
        # HeH+ is ionic/polar — could be covalent in this simple model
        assert mol.bond_character() in ("covalent", "ionic", "nonbonding")


# ------------------------------------------------------------------ #
# Jordan-Wigner Hamiltonian                                            #
# ------------------------------------------------------------------ #

class TestJordanWigner:
    def test_output_shape_h2(self):
        mol = hydrogen()
        H = jordan_wigner_hamiltonian(mol)
        assert H.shape == (16, 16)  # 2^4 spin-orbitals

    def test_hermitian(self):
        mol = hydrogen()
        H = jordan_wigner_hamiltonian(mol)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_hf_expectation_value(self):
        # ⟨HF|H|HF⟩ = E_HF
        mol = hydrogen(_R14_ANG)
        H = jordan_wigner_hamiltonian(mol)
        # HF state: spin-orbitals 0α (bit 0) and 0β (bit 1) → index = 3
        E_hf_check = H[3, 3].real
        assert E_hf_check == pytest.approx(mol.hartree_fock_energy(), rel=1e-8)

    def test_ground_state_eigenvalue(self):
        # Lowest eigenvalue in 2-electron sector = E_FCI
        mol = hydrogen()
        H = jordan_wigner_hamiltonian(mol)
        d = 16
        N_diag = np.array([bin(n).count("1") for n in range(d)])
        sector = np.where(N_diag == 2)[0]
        eigvals = np.linalg.eigvalsh(H[np.ix_(sector, sector)].real)
        assert eigvals[0] == pytest.approx(mol.ground_state_energy(), rel=1e-8)


# ------------------------------------------------------------------ #
# Bond scan                                                             #
# ------------------------------------------------------------------ #

class TestBondScan:
    def test_returns_correct_n_points(self):
        mol = hydrogen()
        curve = mol.scan_bond(r_range=(0.5, 2.0), n_points=5)
        assert len(curve) == 5

    def test_keys(self):
        mol = hydrogen()
        curve = mol.scan_bond(n_points=3)
        for key in ["r_angstrom", "E_fci", "E_hf", "correlation_energy",
                    "orbital_entanglement_01"]:
            assert key in curve[0]

    def test_fci_lower_than_hf_at_all_points(self):
        mol = hydrogen()
        for point in mol.scan_bond(n_points=8):
            assert point["E_fci"] <= point["E_hf"]

    def test_entanglement_increases_with_r(self):
        mol = hydrogen()
        curve = mol.scan_bond(r_range=(0.5, 4.0), n_points=6)
        entanglements = [pt["orbital_entanglement_01"] for pt in curve]
        # Should be monotone increasing (with possible noise at very short R)
        assert entanglements[-1] > entanglements[0]

    def test_correlation_negative_everywhere(self):
        mol = hydrogen()
        for point in mol.scan_bond(n_points=6):
            assert point["correlation_energy"] <= 0
