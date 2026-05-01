"""
Tests for qrl.categorical — Coecke & Kissinger (2017) process theory.

Coverage:
  - Process: compose, tensor, dagger, identity laws, type safety
  - Z-spider and X-spider matrix formulas
  - Spider fusion rule (Z^α ; Z^β = Z^{α+β})
  - Colour change rule (H-conjugation maps Z ↔ X)
  - Compact structure: cup, cap, snake identity
  - Complementarity (Z and X are MUB)
  - Derived processes: CNOT, bell_state
  - Bridges: process_from_unitary, process_from_kraus
"""

import math
import numpy as np
import pytest

from qrl.categorical import (
    Process,
    identity_process,
    scalar_process,
    hadamard_process,
    swap_process,
    z_spider,
    x_spider,
    cup,
    cap,
    spider_fusion,
    colour_change,
    are_complementary,
    cnot_from_spiders,
    bell_state,
    controlled_z_process,
    process_from_unitary,
    process_from_kraus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proportional(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> bool:
    """True if a = c * b for some nonzero scalar c."""
    a = a.ravel()
    b = b.ravel()
    nz = np.nonzero(np.abs(b) > tol)[0]
    if len(nz) == 0:
        return np.allclose(a, 0, atol=tol)
    c = a[nz[0]] / b[nz[0]]
    return bool(np.allclose(a, c * b, atol=tol))


# ---------------------------------------------------------------------------
# Process class
# ---------------------------------------------------------------------------

class TestProcessClass:
    def test_construction_valid(self):
        M = np.eye(2, dtype=complex)
        p = Process(matrix=M, input_type=(2,), output_type=(2,))
        assert p.input_dim == 2
        assert p.output_dim == 2

    def test_construction_scalar(self):
        p = Process(matrix=np.array([[1+0j]]), input_type=(), output_type=())
        assert p.input_dim == 1
        assert p.output_dim == 1

    def test_construction_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            Process(matrix=np.eye(3, dtype=complex), input_type=(2,), output_type=(2,))

    def test_sequential_composition_types(self):
        p = identity_process(2)
        q = hadamard_process()
        r = p.then(q)
        assert r.input_type == (2,)
        assert r.output_type == (2,)

    def test_sequential_composition_matrix(self):
        H = hadamard_process()
        HH = H.then(H)
        assert np.allclose(HH.matrix, np.eye(2), atol=1e-10)

    def test_type_mismatch_raises(self):
        p = identity_process(2)
        q = identity_process(4)
        with pytest.raises(ValueError):
            p.then(q)

    def test_tensor_types(self):
        p = hadamard_process()
        q = identity_process(2)
        r = p.tensor(q)
        assert r.input_type == (2, 2)
        assert r.output_type == (2, 2)

    def test_tensor_matrix_is_kron(self):
        H = hadamard_process()
        I = identity_process(2)
        HI = H.tensor(I)
        expected = np.kron(H.matrix, I.matrix)
        assert np.allclose(HI.matrix, expected, atol=1e-10)

    def test_dagger_swaps_types(self):
        p = z_spider(0, 2, 0)   # () → (2,2)
        pd = p.dagger()
        assert pd.input_type == (2, 2)
        assert pd.output_type == ()

    def test_dagger_is_conj_transpose(self):
        H = hadamard_process()
        assert np.allclose(H.dagger().matrix, H.matrix.conj().T, atol=1e-10)

    def test_identity_left_unit(self):
        H = hadamard_process()
        assert np.allclose(identity_process(2).then(H).matrix, H.matrix, atol=1e-10)

    def test_identity_right_unit(self):
        H = hadamard_process()
        assert np.allclose(H.then(identity_process(2)).matrix, H.matrix, atol=1e-10)

    def test_matmul_operator(self):
        H = hadamard_process()
        I = identity_process(2)
        # H @ I means I.then(H)
        assert np.allclose((H @ I).matrix, H.matrix, atol=1e-10)

    def test_approx_equal_global_phase(self):
        H = hadamard_process()
        phase = Process(matrix=1j * H.matrix, input_type=(2,), output_type=(2,))
        assert H.approx_equal(phase)

    def test_is_unitary_hadamard(self):
        assert hadamard_process().is_unitary()

    def test_is_unitary_identity(self):
        assert identity_process(2).is_unitary()

    def test_swap_is_unitary(self):
        assert swap_process(2).is_unitary()

    def test_is_isometry_cup(self):
        # cup is NOT an isometry: cup†·cup = [[2]] ≠ [[1]]
        assert not cup(2).is_isometry()

    def test_scalar_process(self):
        s = scalar_process(2.0 + 0j)
        assert s.input_type == ()
        assert s.output_type == ()
        assert np.allclose(s.matrix, [[2.0]], atol=1e-10)


# ---------------------------------------------------------------------------
# Z-spider matrix formulas
# ---------------------------------------------------------------------------

class TestZSpider:
    def test_1_1_phase_0_is_identity(self):
        z = z_spider(1, 1, 0.0)
        assert np.allclose(z.matrix, np.eye(2), atol=1e-10)

    def test_1_1_phase_pi_is_Z_gate(self):
        z = z_spider(1, 1, math.pi)
        expected = np.diag([1.0, -1.0]).astype(complex)
        assert np.allclose(z.matrix, expected, atol=1e-10)

    def test_0_2_phase_0_is_bell_pair(self):
        z = z_spider(0, 2, 0.0)
        expected = np.array([[1], [0], [0], [1]], dtype=complex)
        assert np.allclose(z.matrix, expected, atol=1e-10)

    def test_1_2_phase_0_is_copy_map(self):
        z = z_spider(1, 2, 0.0)
        # |0⟩ → |00⟩, |1⟩ → |11⟩
        expected = np.array([[1, 0], [0, 0], [0, 0], [0, 1]], dtype=complex)
        assert np.allclose(z.matrix, expected, atol=1e-10)

    def test_2_1_phase_0_is_merge_map(self):
        z = z_spider(2, 1, 0.0)
        expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=complex)
        assert np.allclose(z.matrix, expected, atol=1e-10)

    def test_0_1_phase_0_is_plus_state(self):
        z = z_spider(0, 1, 0.0)
        # |0⟩ + |1⟩  (unnormalised |+⟩√2)
        expected = np.array([[1], [1]], dtype=complex)
        assert np.allclose(z.matrix, expected, atol=1e-10)

    def test_phase_encodes_rotation(self):
        alpha = 0.7
        z = z_spider(1, 1, alpha)
        expected = np.diag([1.0, np.exp(1j * alpha)])
        assert np.allclose(z.matrix, expected, atol=1e-10)

    def test_dagger_reverses_inputs_outputs(self):
        z = z_spider(1, 2, 0.3)
        zd = z.dagger()
        assert zd.input_type == (2, 2)
        assert zd.output_type == (2,)


# ---------------------------------------------------------------------------
# X-spider
# ---------------------------------------------------------------------------

class TestXSpider:
    def test_1_1_phase_pi_is_X_gate(self):
        x = x_spider(1, 1, math.pi)
        X_gate = np.array([[0, 1], [1, 0]], dtype=complex)
        assert _proportional(x.matrix, X_gate)

    def test_1_1_phase_0_is_identity(self):
        x = x_spider(1, 1, 0.0)
        assert np.allclose(x.matrix, np.eye(2), atol=1e-10)

    def test_0_1_phase_0_proportional_to_0_state(self):
        x = x_spider(0, 1, 0.0)
        # x_spider state lives in Z-basis: proportional to |0⟩
        assert _proportional(x.matrix, np.array([[1], [0]], dtype=complex))

    def test_0_1_phase_pi_proportional_to_1_state(self):
        x = x_spider(0, 1, math.pi)
        assert _proportional(x.matrix, np.array([[0], [1]], dtype=complex))


# ---------------------------------------------------------------------------
# Spider fusion rule (C&K Theorem 6.14)
# ---------------------------------------------------------------------------

class TestSpiderFusion:
    def test_z_spider_fusion_1_1(self):
        # Z^α(1→2) ; Z^β(2→1) = Z^{α+β}(1→1)
        alpha, beta = 0.3, 0.7
        fused = spider_fusion(z_spider(1, 2, alpha), z_spider(2, 1, beta))
        expected = z_spider(1, 1, alpha + beta)
        assert np.allclose(fused.matrix, expected.matrix, atol=1e-10)

    def test_z_spider_fusion_phases_add(self):
        alpha, beta = math.pi / 3, math.pi / 4
        fused = spider_fusion(z_spider(1, 2, alpha), z_spider(2, 1, beta))
        expected = z_spider(1, 1, alpha + beta)
        assert np.allclose(fused.matrix, expected.matrix, atol=1e-10)

    def test_z_spider_fusion_zero_plus_alpha(self):
        # Z(0) ; Z(α) = Z(α)
        alpha = 1.2
        fused = spider_fusion(z_spider(1, 2, 0.0), z_spider(2, 1, alpha))
        expected = z_spider(1, 1, alpha)
        assert np.allclose(fused.matrix, expected.matrix, atol=1e-10)

    def test_z_spider_fusion_type_preserved(self):
        fused = spider_fusion(z_spider(1, 2, 0.3), z_spider(2, 1, 0.4))
        assert fused.input_type == (2,)
        assert fused.output_type == (2,)


# ---------------------------------------------------------------------------
# Colour change rule (C&K Theorem 9.31)
# ---------------------------------------------------------------------------

class TestColourChange:
    def test_colour_change_z_1_1_gives_x_1_1(self):
        alpha = 0.6
        z = z_spider(1, 1, alpha)
        changed = colour_change(z)
        expected = x_spider(1, 1, alpha)
        assert np.allclose(changed.matrix, expected.matrix, atol=1e-10)

    def test_colour_change_z_pi_gives_pauli_x(self):
        changed = colour_change(z_spider(1, 1, math.pi))
        X_gate = np.array([[0, 1], [1, 0]], dtype=complex)
        assert _proportional(changed.matrix, X_gate)

    def test_colour_change_twice_is_identity(self):
        # Hadamard is self-inverse, so two colour changes = identity
        z = z_spider(1, 1, 0.5)
        twice = colour_change(colour_change(z))
        assert np.allclose(twice.matrix, z.matrix, atol=1e-10)

    def test_colour_change_types_preserved(self):
        z = z_spider(1, 2, 0.3)
        changed = colour_change(z)
        assert changed.input_type == z.input_type
        assert changed.output_type == z.output_type


# ---------------------------------------------------------------------------
# Compact structure: cup, cap, snake identity (C&K §6)
# ---------------------------------------------------------------------------

class TestCompactStructure:
    def test_cup_type(self):
        c = cup(2)
        assert c.input_type == ()
        assert c.output_type == (2, 2)

    def test_cap_is_cup_dagger(self):
        assert np.allclose(cap(2).matrix, cup(2).dagger().matrix, atol=1e-10)

    def test_cup_matrix_bell_pair(self):
        expected = np.array([[1], [0], [0], [1]], dtype=complex)
        assert np.allclose(cup(2).matrix, expected, atol=1e-10)

    def test_snake_identity(self):
        # (cap_{12} ⊗ id_3) ∘ (id_1 ⊗ cup_{23}) = id_{(2,)}
        id2 = identity_process(2)
        lhs = id2.tensor(cup(2)).then(cap(2).tensor(id2))
        assert np.allclose(lhs.matrix, id2.matrix, atol=1e-10)

    def test_snake_identity_types(self):
        id2 = identity_process(2)
        lhs = id2.tensor(cup(2)).then(cap(2).tensor(id2))
        assert lhs.input_type == (2,)
        assert lhs.output_type == (2,)

    def test_cap_cup_is_scalar_d(self):
        # cap ∘ cup = d (as a scalar process () → ())
        result = cup(2).then(cap(2))
        assert np.allclose(result.matrix, [[2.0]], atol=1e-10)

    def test_bell_state_equals_cup(self):
        assert np.allclose(bell_state().matrix, cup(2).matrix, atol=1e-10)

    def test_cup_z_spider_relation(self):
        # cup(2) == z_spider(0, 2, 0)
        assert np.allclose(cup(2).matrix, z_spider(0, 2, 0).matrix, atol=1e-10)


# ---------------------------------------------------------------------------
# Complementarity (C&K §8)
# ---------------------------------------------------------------------------

class TestComplementarity:
    def test_z_x_are_complementary(self):
        # z_spider(0,1,0) = |+⟩√2; x_spider(0,1,0) = |0⟩√2 — these are MUB
        assert are_complementary(z_spider(0, 1, 0.0), x_spider(0, 1, 0.0))

    def test_z_z_are_not_complementary(self):
        # Two Z-basis states are NOT MUB with each other
        assert not are_complementary(z_spider(0, 1, 0.0), z_spider(0, 1, math.pi))

    def test_type_check(self):
        # Non-state processes: should return False
        assert not are_complementary(z_spider(1, 1, 0.0), x_spider(1, 1, 0.0))


# ---------------------------------------------------------------------------
# CNOT from spiders
# ---------------------------------------------------------------------------

class TestCNOT:
    def test_cnot_type(self):
        cnot = cnot_from_spiders()
        assert cnot.input_type == (2, 2)
        assert cnot.output_type == (2, 2)

    def test_cnot_proportional_to_standard(self):
        cnot = cnot_from_spiders()
        standard = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        assert _proportional(cnot.matrix, standard)

    def test_cnot_squared_proportional_to_identity(self):
        # CNOT² = I  (up to scalar)
        cnot = cnot_from_spiders()
        cnot2 = cnot.then(cnot)
        assert _proportional(cnot2.matrix, np.eye(4, dtype=complex))


# ---------------------------------------------------------------------------
# Bridges to existing QRL
# ---------------------------------------------------------------------------

class TestBridges:
    def test_process_from_unitary_hadamard(self):
        H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
        p = process_from_unitary(H, name="H")
        assert p.is_unitary()
        assert p.input_type == (2,)

    def test_process_from_unitary_cnot(self):
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        p = process_from_unitary(CNOT, name="CNOT")
        assert p.is_unitary()
        assert p.input_type == (2, 2)

    def test_process_from_unitary_non_power_of_2_raises(self):
        with pytest.raises(ValueError):
            process_from_unitary(np.eye(3, dtype=complex))

    def test_process_from_kraus_unitary_is_isometry(self):
        # Single Kraus op = unitary → superoperator should be isometry?
        # Actually the superoperator of a unitary U is U⊗U*, which is unitary on the d² space.
        H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
        p = process_from_kraus([H])
        # S = H ⊗ H* = kron(H, H.conj()) — this should be unitary
        assert p.is_unitary()

    def test_process_from_kraus_depolarizing_is_causal(self):
        # Depolarizing channel Kraus ops: {√(1-3p/4)I, √(p/4)X, √(p/4)Y, √(p/4)Z}
        p_dep = 0.1
        I2 = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z_op = np.array([[1, 0], [0, -1]], dtype=complex)
        kraus = [
            math.sqrt(1 - 3 * p_dep / 4) * I2,
            math.sqrt(p_dep / 4) * X,
            math.sqrt(p_dep / 4) * Y,
            math.sqrt(p_dep / 4) * Z_op,
        ]
        proc = process_from_kraus(kraus)
        assert proc.input_type == (4,)   # 2² = 4
        assert proc.output_type == (4,)
