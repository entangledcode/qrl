"""
Tests for Araújo et al. (2015) — causal nonseparability witness.

Implements: Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner (2015).
Witnessing causal nonseparability. New Journal of Physics, 17, 102001.
arXiv:1506.03776.

Covers:
  - is_causally_ordered_AB / is_causally_ordered_BA structural tests
  - project_to_AB_order / project_to_BA_order
  - witness_value(Omega) and causal_nonseparability_witness() factory
  - causal_nonseparability_robustness()
"""

import numpy as np
import pytest

from qrl.causal import (
    ProcessMatrix,
    identity_process,
    definite_order_process,
    quantum_switch_process_matrix,
    causal_nonseparability_witness,
    _causal_project_AB,
    _causal_project_BA,
)


# ------------------------------------------------------------------ #
# Structural causal order tests                                        #
# ------------------------------------------------------------------ #

def test_definite_AB_is_causally_ordered_AB():
    """definite_order_process('AB') lies in the A≺B causal subspace."""
    W = definite_order_process(order='AB')
    assert W.is_causally_ordered_AB()


def test_definite_AB_is_not_causally_ordered_BA():
    """definite_order_process('AB') does not lie in the B≺A causal subspace."""
    W = definite_order_process(order='AB')
    assert not W.is_causally_ordered_BA()


def test_identity_is_causally_ordered_AB():
    """The maximally mixed process is in both causal subspaces (no causal structure)."""
    W = identity_process(n_parties=2)
    assert W.is_causally_ordered_AB()


def test_identity_is_causally_ordered_BA():
    """The maximally mixed process is also in the B≺A subspace."""
    W = identity_process(n_parties=2)
    assert W.is_causally_ordered_BA()


def test_single_party_ordered_returns_false():
    """Non-bipartite processes return False for both structural tests."""
    W = identity_process(n_parties=1)
    assert not W.is_causally_ordered_AB()
    assert not W.is_causally_ordered_BA()


def test_wrong_dims_ordered_returns_false():
    """Non-qubit 2-party process returns False."""
    W = identity_process(n_parties=2, d_in=4, d_out=4)
    assert not W.is_causally_ordered_AB()
    assert not W.is_causally_ordered_BA()


# ------------------------------------------------------------------ #
# Causal order projections                                             #
# ------------------------------------------------------------------ #

def test_project_AB_on_AB_process_is_identity():
    """Projecting an A≺B process onto A≺B is a no-op."""
    W = definite_order_process(order='AB')
    assert np.allclose(W.W, W.project_to_AB_order().W, atol=1e-10)


def test_project_AB_preserves_trace():
    """Φ_{A≺B} preserves the trace of the process matrix."""
    W = identity_process(n_parties=2)
    W_proj = W.project_to_AB_order()
    assert np.isclose(np.trace(W_proj.W).real, np.trace(W.W).real, atol=1e-10)


def test_project_BA_preserves_trace():
    """Φ_{B≺A} preserves the trace of the process matrix."""
    W = definite_order_process(order='AB')
    W_proj = W.project_to_BA_order()
    assert np.isclose(np.trace(W_proj.W).real, np.trace(W.W).real, atol=1e-10)


def test_project_AB_idempotent():
    """Applying Φ_{A≺B} twice gives the same result (projector is idempotent)."""
    W = identity_process(n_parties=2)
    W_proj1 = W.project_to_AB_order()
    W_proj2 = W_proj1.project_to_AB_order()
    assert np.allclose(W_proj1.W, W_proj2.W, atol=1e-10)


def test_project_BA_idempotent():
    """Applying Φ_{B≺A} twice gives the same result."""
    W = identity_process(n_parties=2)
    W_proj1 = W.project_to_BA_order()
    W_proj2 = W_proj1.project_to_BA_order()
    assert np.allclose(W_proj1.W, W_proj2.W, atol=1e-10)


def test_project_AB_result_is_in_AB_subspace():
    """The Φ_{A≺B} projection always lands in the A≺B subspace."""
    W = identity_process(n_parties=2)
    assert W.project_to_AB_order().is_causally_ordered_AB()


def test_project_BA_result_is_in_BA_subspace():
    """The Φ_{B≺A} projection always lands in the B≺A subspace."""
    W = definite_order_process(order='AB')
    assert W.project_to_BA_order().is_causally_ordered_BA()


def test_project_BA_on_AB_gives_identity():
    """Φ_{B≺A} applied to the definite A→B process gives the maximally mixed process."""
    W_AB = definite_order_process('AB')
    W_proj = W_AB.project_to_BA_order()
    W_id = identity_process(n_parties=2)
    assert np.allclose(W_proj.W, W_id.W, atol=1e-10)


def test_project_wrong_dims_raises():
    """project_to_AB_order raises ValueError for non-qubit processes."""
    W = identity_process(n_parties=1)
    with pytest.raises(ValueError, match="2-party qubit"):
        W.project_to_AB_order()
    with pytest.raises(ValueError, match="2-party qubit"):
        W.project_to_BA_order()


# ------------------------------------------------------------------ #
# Causal nonseparability witness                                       #
# ------------------------------------------------------------------ #

def test_witness_is_hermitian():
    """The witness Ω is Hermitian (Ω = Ω†)."""
    Omega = causal_nonseparability_witness()
    assert np.allclose(Omega, Omega.conj().T, atol=1e-10)


def test_witness_shape():
    """Witness has shape (16, 16) for d=2."""
    Omega = causal_nonseparability_witness()
    assert Omega.shape == (16, 16)


def test_witness_not_implemented_for_d4():
    """Only d=2 is supported."""
    with pytest.raises(NotImplementedError):
        causal_nonseparability_witness(d=4)


def test_witness_separable_AB_nonnegative():
    """Tr[Ω · W_{A→B}] ≥ 0 (definite A→B is causally separable)."""
    Omega = causal_nonseparability_witness()
    val = definite_order_process('AB').witness_value(Omega)
    assert val >= -1e-10


def test_witness_identity_nonnegative():
    """Tr[Ω · W_{id}] ≥ 0 (maximally mixed is causally separable)."""
    Omega = causal_nonseparability_witness()
    val = identity_process(n_parties=2).witness_value(Omega)
    assert val >= -1e-10


def test_witness_value_equals_formula():
    """Tr[Ω · W] = 3 − 4·P_win for any 2-party qubit process."""
    Omega = causal_nonseparability_witness()
    W = definite_order_process('AB')
    p_win = W.causal_inequality_value()
    expected = 3.0 - 4.0 * p_win
    assert np.isclose(W.witness_value(Omega), expected, atol=1e-8)


def test_witness_value_identity_process():
    """Tr[Ω · W_{id}] = 3 − 4·(1/2) = 1 for the maximally mixed process."""
    Omega = causal_nonseparability_witness()
    val = identity_process(n_parties=2).witness_value(Omega)
    assert np.isclose(val, 1.0, atol=1e-8)


# ------------------------------------------------------------------ #
# Causal nonseparability robustness                                    #
# ------------------------------------------------------------------ #

def test_robustness_separable_is_zero():
    """Robustness is zero for causally separable processes."""
    assert definite_order_process('AB').causal_nonseparability_robustness() == 0.0


def test_robustness_identity_is_zero():
    """Robustness is zero for the maximally mixed process."""
    assert identity_process(n_parties=2).causal_nonseparability_robustness() == 0.0


def test_robustness_switch_is_sqrt2_minus_1():
    """Quantum switch robustness = √2 − 1 ≈ 0.414 (Araújo et al. result)."""
    r = quantum_switch_process_matrix().causal_nonseparability_robustness()
    assert np.isclose(r, np.sqrt(2) - 1.0, atol=1e-8)


def test_robustness_nonnegative_all_cases():
    """Robustness is always ≥ 0."""
    for W in [
        identity_process(n_parties=2),
        definite_order_process('AB'),
        quantum_switch_process_matrix(),
    ]:
        assert W.causal_nonseparability_robustness() >= 0.0


def test_robustness_none_for_inapplicable():
    """Robustness returns None for processes where P_win is not applicable."""
    W = identity_process(n_parties=1)
    assert W.causal_nonseparability_robustness() is None


def test_robustness_equals_witness_magnitude():
    """For nonseparable W: robustness = |witness value| = -(3 − 4·P_win)."""
    W_switch = quantum_switch_process_matrix()
    r = W_switch.causal_nonseparability_robustness()
    p_win = W_switch.causal_inequality_value()
    assert np.isclose(r, -(3.0 - 4.0 * p_win), atol=1e-8)


# ------------------------------------------------------------------ #
# Private projection helpers (numerical verification)                  #
# ------------------------------------------------------------------ #

def test_causal_project_AB_on_identity():
    """Φ_{A≺B}(I/4) = I/4 (identity is in both causal subspaces)."""
    I16 = np.eye(16, dtype=complex) / 4.0
    proj = _causal_project_AB(I16)
    assert np.allclose(proj, I16, atol=1e-10)


def test_causal_project_BA_on_identity():
    """Φ_{B≺A}(I/4) = I/4."""
    I16 = np.eye(16, dtype=complex) / 4.0
    proj = _causal_project_BA(I16)
    assert np.allclose(proj, I16, atol=1e-10)


def test_causal_project_AB_on_AB_process():
    """Φ_{A≺B}(W_{A→B}) = W_{A→B} (definite A→B is already in A≺B subspace)."""
    W_AB = definite_order_process('AB').W
    proj = _causal_project_AB(W_AB)
    assert np.allclose(proj, W_AB, atol=1e-10)


def test_causal_project_BA_on_AB_process():
    """Φ_{B≺A}(W_{A→B}) = I/4 (the A→B process projects to the identity under B≺A)."""
    W_AB = definite_order_process('AB').W
    proj = _causal_project_BA(W_AB)
    I16 = np.eye(16, dtype=complex) / 4.0
    assert np.allclose(proj, I16, atol=1e-10)
