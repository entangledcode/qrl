"""
Refinement classes QC-FO ⊂ QC-CC ⊂ QC-QC of two-slot processes, with
certificates (qrl.causal_classes; Wechs, Dourdent, Abbott, Branciard 2021).

Checks need only numpy; the SDP search tests are skipped without cvxpy.
"""

import numpy as np
import pytest

from qrl.causal import (
    ProcessMatrix,
    causal_nonseparability_witness,
    definite_order_process,
    identity_process,
    ocb_process,
)
from qrl.causal_classes import (
    CLASSICAL_CONTROL,
    CausalInequalityCertificate,
    ocb_certificate,
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
# Outside QC-QC: the OCB causal inequality                             #
# ------------------------------------------------------------------ #

def test_ocb_process_values_match_the_literature():
    W = ocb_process()
    assert W.is_valid()
    assert np.isclose(W.causal_inequality_value(), (2 + np.sqrt(2)) / 4)
    assert np.isclose(W.witness_value(causal_nonseparability_witness()), 1 - np.sqrt(2))
    assert np.isclose(identity_process(n_parties=2).causal_inequality_value(), 0.5)
    assert np.isclose(definite_order_process("AB").causal_inequality_value(), 0.75)


def test_ocb_process_is_certified_outside_quantum_control():
    c = ocb_certificate()
    assert c.check(ocb_process())
    assert fixed_order_certificate(ocb_process()) is None


def test_causal_inequality_certificate_fails_inside_quantum_control():
    c = ocb_certificate()
    assert not c.check(definite_order_process("AB"))
    assert not c.check(switch_process(2))                 # QC-QC: at most 3/4
    pm, _ = _p_controlled()
    assert not c.check(pm)


def test_causal_inequality_certificate_rejects_invalid_instruments():
    c = ocb_certificate()
    A = dict(c.instruments_A)
    A[(0, 0)] = 2 * A[(0, 0)]                              # not trace preserving
    assert not CausalInequalityCertificate(tuple(A.items()), c.instruments_B).check(ocb_process())


def test_ocb_game_never_beats_three_quarters_on_causally_ordered_processes():
    c = ocb_certificate()
    for _ in range(50):
        for first in "AB":
            w = _branch(first)                            # A_I A_O B_I B_O F
            W = w.reshape(16, 2, 16, 2).trace(axis1=1, axis2=3)   # discard F
            pm = ProcessMatrix(W=W, parties=["A", "B"], input_dims=[2, 2], output_dims=[2, 2])
            assert c.value(pm) <= 0.75 + 1e-9


# ------------------------------------------------------------------ #
# SDP search (needs cvxpy)                                             #
# ------------------------------------------------------------------ #

def test_ocb_bound_holds_over_all_of_quantum_control():
    """With OCB's instruments, no QC-QC process beats 3/4 (Wechs et al. Prop. 8)."""
    cp = pytest.importorskip("cvxpy")
    from qrl.causal import ocb_score_operator
    from qrl.causal_classes import _cp_kron_id, _cp_ptrace
    dims = {"P": 1, "AI": 2, "AO": 2, "BI": 2, "BO": 2, "F": 1}
    n = lambda nm: int(np.prod([dims[q] for q in nm]))
    V = {k: cp.Variable((n(nm), n(nm)), hermitian=True) for k, nm in
         (("0A", ("P", "AI")), ("0B", ("P", "BI")),
          ("AB", ("P", "AI", "AO", "BI")), ("BA", ("P", "AI", "BI", "BO")))}
    t0A, _ = _cp_ptrace(V["0A"], ("P", "AI"), ["AI"], dims)
    t0B, _ = _cp_ptrace(V["0B"], ("P", "BI"), ["BI"], dims)
    lAB, _ = _cp_ptrace(V["AB"], ("P", "AI", "AO", "BI"), ["BI"], dims)
    rAB, _ = _cp_kron_id(V["0A"], ("P", "AI"), "AO", dims)
    lBA, _ = _cp_ptrace(V["BA"], ("P", "AI", "BI", "BO"), ["AI"], dims)
    rBA, _ = _cp_kron_id(V["0B"], ("P", "BI"), "BO", dims)
    xAB, _ = _cp_kron_id(V["AB"], ("P", "AI", "AO", "BI"), "BO", dims)
    xBA, _ = _cp_kron_id(V["BA"], ("P", "AI", "BI", "BO"), "AO", dims)
    W = xAB + xBA                                          # F trivial: W = Tr_F W
    cons = [v >> 0 for v in V.values()] + [t0A + t0B == np.eye(1), lAB == rAB, lBA == rBA]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(W @ ocb_score_operator())) / 4), cons)
    prob.solve(solver="SCS", eps=1e-8)
    assert prob.value <= 0.75 + 1e-6


def test_search_finds_no_quantum_control_certificate_for_ocb():
    pytest.importorskip("cvxpy")
    assert find_certificate(ocb_process(), QUANTUM_CONTROL) is None

def test_search_finds_the_right_classes():
    pytest.importorskip("cvxpy")
    pm, _ = _p_controlled()
    assert isinstance(find_certificate(pm, CLASSICAL_CONTROL), ClassicalControlCertificate)
    sw = switch_process(2)
    assert find_certificate(sw, CLASSICAL_CONTROL) is None
    assert isinstance(find_certificate(sw, QUANTUM_CONTROL), QuantumControlCertificate)
