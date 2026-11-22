"""
Refinement classes QC-FO ⊂ QC-CC ⊂ QC-QC of two-slot processes, with
certificates (qrl.causal_classes; Wechs, Dourdent, Abbott, Branciard 2021).

Checks need only numpy; the SDP search tests are skipped without cvxpy.
"""

import numpy as np
import pytest

from qrl.causal import ProcessMatrix, definite_order_process
from qrl.causal_classes import (
    CLASSICAL_CONTROL,
    QUANTUM_CONTROL,
    ClassicalControlCertificate,
    FixedOrderCertificate,
    QuantumControlCertificate,
    find_certificate,
    fixed_order_certificate,
    switch_quantum_control_certificate,
)
from qrl.switch_witness import switch_process, verify_exact_certificate

rng = np.random.default_rng(2026)


def _haar(n):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _branch(first):
    """Random comb on A_I A_O B_I B_O F with order first < second < F."""
    V0 = _haar(4)[:, :1].reshape(2, 2)
    V1 = _haar(4).reshape(2, 2, 2, 2)
    V2 = _haar(4).reshape(2, 2, 2, 2)
    w = np.einsum("am,qnbm,fjcn->abqcfj", V0, V1, V2)
    if first == "B":
        w = w.transpose(2, 3, 0, 1, 4, 5)
    w = w.reshape(32, 2)
    return w @ w.conj().T


def _p_controlled():
    """P's output, measured in a random basis, chooses the order: QC-CC, not QC-FO.
    Returns the process and its two branches."""
    B = _haar(2)
    parts = [np.kron(np.outer(B[:, x], B[:, x].conj()).conj(), _branch(f))
             for x, f in ((0, "A"), (1, "B"))]
    pm = ProcessMatrix(W=parts[0] + parts[1], parties=["P", "A", "B", "F"],
                       input_dims=[1, 2, 2, 2], output_dims=[2, 2, 2, 1])
    return pm, parts


# ------------------------------------------------------------------ #
# QC-FO                                                                #
# ------------------------------------------------------------------ #

def test_definite_order_has_its_order_only():
    pm = definite_order_process("AB")
    assert fixed_order_certificate(pm) == FixedOrderCertificate("AB")
    assert not FixedOrderCertificate("BA").check(pm)
    assert fixed_order_certificate(definite_order_process("BA")) == FixedOrderCertificate("BA")


def test_fixed_order_converts_up_the_chain():
    pm = definite_order_process("BA")
    cc = FixedOrderCertificate("BA").to_classical_control(pm)
    assert cc.check(pm)
    assert cc.to_quantum_control(pm).check(pm)


# ------------------------------------------------------------------ #
# QC-CC                                                                #
# ------------------------------------------------------------------ #

def test_p_controlled_process_is_classical_control_not_fixed_order():
    pm, parts = _p_controlled()
    assert pm.is_valid()
    assert fixed_order_certificate(pm) is None
    cc = ClassicalControlCertificate(parts[0], parts[1])
    assert cc.check(pm)
    assert cc.to_quantum_control(pm).check(pm)


def test_classical_control_certificate_rejects_wrong_split():
    pm, parts = _p_controlled()
    assert not ClassicalControlCertificate(parts[1], parts[0]).check(pm)      # orders swapped
    half = (parts[0] + parts[1]) / 2
    assert not ClassicalControlCertificate(half, half).check(pm)             # not combs
    assert not ClassicalControlCertificate(parts[0], parts[1] + 0.01).check(pm)  # does not sum to W


# ------------------------------------------------------------------ #
# QC-QC: the switch                                                    #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("d", [2, 3])
def test_switch_is_quantum_control(d):
    pm = switch_process(d)
    assert switch_quantum_control_certificate(d).check(pm)
    assert fixed_order_certificate(pm) is None


def test_switch_is_not_classical_control():
    # The non-membership certificate is the switch's exact causal witness:
    # Tr[S W] >= -2^-17 on QC-CC (= causally separable), -0.1694 on the switch.
    assert verify_exact_certificate()["valid"]


def test_quantum_control_certificate_rejects_tampering():
    pm = switch_process(2)
    c = switch_quantum_control_certificate(2)
    bad = [
        QuantumControlCertificate(c.W0A, c.W0B, 2 * c.WAB, c.WBA),           # wrong scale
        QuantumControlCertificate(c.W0B, c.W0A, c.WAB, c.WBA),               # first slots swapped
        QuantumControlCertificate(c.W0A, c.W0B, c.WAB, np.zeros_like(c.WBA)),  # branch missing
        QuantumControlCertificate(c.W0A, c.W0B, -c.WAB, c.WBA),              # not PSD
    ]
    assert all(not b.check(pm) for b in bad)


def test_certificates_fail_on_a_causal_loop():
    v = np.zeros(16, dtype=complex)
    for i in range(2):              # A_O -> B_I and B_O -> A_I identity wires
        for j in range(2):
            v[(j * 2 + i) * 4 + i * 2 + j] = 1
    loop = ProcessMatrix(W=np.outer(v, v.conj()), parties=["A", "B"],
                         input_dims=[2, 2], output_dims=[2, 2])
    assert not loop.is_valid()
    assert fixed_order_certificate(loop) is None


# ------------------------------------------------------------------ #
# SDP search (needs cvxpy)                                             #
# ------------------------------------------------------------------ #

def test_search_finds_the_right_classes():
    pytest.importorskip("cvxpy")
    pm, _ = _p_controlled()
    assert isinstance(find_certificate(pm, CLASSICAL_CONTROL), ClassicalControlCertificate)
    sw = switch_process(2)
    assert find_certificate(sw, CLASSICAL_CONTROL) is None
    assert isinstance(find_certificate(sw, QUANTUM_CONTROL), QuantumControlCertificate)
