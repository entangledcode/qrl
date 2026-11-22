"""
Refinement classes of process matrices, with certificates.

For processes with two slots A and B, a global past P and a global future F
(the switch scenario; P or F may be trivial), Wechs, Dourdent, Abbott and
Branciard (PRX Quantum 2, 030335, 2021) characterise three nested classes:

    QC-FO  quantum circuits with a fixed order          (Prop. 2)
    QC-CC  quantum circuits with classical control of   (Prop. 5)
           the order  (= causally separable processes; Wechs, Abbott,
           Branciard 2026; Wei, Pang 2026)
    QC-QC  quantum circuits with quantum control of     (Prop. 7)
           the order  (= processes realisable under closed labs;
           Salzger, Vilasini 2026)

    QC-FO  ⊂  QC-CC  ⊂  QC-QC  ⊂  valid process matrices

Membership in each class is witnessed by a certificate that is checked with
linear algebra and eigenvalues only:

    FixedOrderCertificate(order)              the order, 'AB' or 'BA'
    ClassicalControlCertificate(W_A, W_B)     W = W_A + W_B, one PSD part per order
    QuantumControlCertificate(...)            Wechs's four matrices W_(K, k)

and non-membership in QC-QC by

    CausalInequalityCertificate(instruments)  OCB game won with probability > 3/4

Certificates convert along the inclusions (fixed order -> classical control ->
quantum control), which is how the refinement types QC-FO <: QC-CC <: QC-QC
are realised.  find_certificate() searches for one by semidefinite
programming; it needs cvxpy, but checking a certificate does not.

Conventions: certificates use the subsystem order P, A_I, A_O, B_I, B_O, F
(trivial dimensions dropped); a ProcessMatrix is matched to it by party name,
so its parties may come in any order.  Operators of a certificate live on the
named subsystems listed in that order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from qrl.causal import ProcessMatrix

SYS = ("P", "AI", "AO", "BI", "BO", "F")


# ------------------------------------------------------------------ #
# Scenario and named operators                                         #
# ------------------------------------------------------------------ #

def _canonical(pm: ProcessMatrix) -> Tuple[np.ndarray, Dict[str, int]]:
    """W permuted into the subsystem order P, A_I, A_O, B_I, B_O, F, matched by
    party name (parties {A, B} or {P, A, B, F}, in any order), and the dims."""
    names = list(pm.parties)
    if sorted(names) not in (["A", "B"], ["A", "B", "F", "P"]):
        raise ValueError("expected parties named A, B (and optionally P, F)")
    info = {n: (i, o) for n, i, o in zip(names, pm.input_dims, pm.output_dims)}
    if "P" in info and info["P"][0] != 1:
        raise ValueError("party P must have no input")
    if "F" in info and info["F"][1] != 1:
        raise ValueError("party F must have no output")
    dims = {"P": info.get("P", (1, 1))[1], "AI": info["A"][0], "AO": info["A"][1],
            "BI": info["B"][0], "BO": info["B"][1], "F": info.get("F", (1, 1))[0]}
    # subsystems of pm.W, in its own order: each party's (input, output)
    src = []
    for n in names:
        src += [(n, "I"), (n, "O")]
    label = {("P", "O"): "P", ("A", "I"): "AI", ("A", "O"): "AO",
             ("B", "I"): "BI", ("B", "O"): "BO", ("F", "I"): "F"}
    src_dims = [info[n][0] if io == "I" else info[n][1] for n, io in src]
    src_names = [label.get(x) for x in src]          # None for trivial P_I, F_O
    k = len(src)
    T = np.asarray(pm.W, dtype=complex).reshape(src_dims + src_dims)
    perm = [src_names.index(x) for x in SYS if x in src_names]
    rest = [i for i in range(k) if i not in perm]      # trivial (dim 1) axes last
    T = T.transpose(perm + rest + [p + k for p in perm + rest])
    D = int(np.prod(src_dims))
    return T.reshape(D, D), dims


def scenario_dims(pm: ProcessMatrix) -> Dict[str, int]:
    """Subsystem dimensions of a two-slot process (parties A, B and optionally P, F)."""
    return _canonical(pm)[1]


def _order(names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(names, key=SYS.index))


@dataclass(frozen=True)
class NamedOp:
    """An operator on the subsystems `names` (in SYS order)."""
    mat: np.ndarray
    names: Tuple[str, ...]

    def _dims(self, dims):
        return [dims[n] for n in self.names]

    def trace(self, out: Sequence[str], dims: Dict[str, int]) -> "NamedOp":
        """Partial trace over the subsystems `out`."""
        ds = self._dims(dims)
        k = len(ds)
        T = self.mat.reshape(ds + ds)
        for name in sorted(out, key=lambda n: -self.names.index(n)):
            i = self.names.index(name)
            T = np.trace(T, axis1=i, axis2=i + T.ndim // 2)
        keep = tuple(n for n in self.names if n not in out)
        dk = int(np.prod([dims[n] for n in keep])) if keep else 1
        return NamedOp(T.reshape(dk, dk), keep)

    def with_identity(self, name: str, dims: Dict[str, int]) -> "NamedOp":
        """self ⊗ 1_name, with the subsystems re-sorted into SYS order."""
        names = self.names + (name,)
        M = np.kron(self.mat, np.eye(dims[name]))
        target = _order(names)
        ds = [dims[n] for n in names]
        perm = [names.index(n) for n in target]
        k = len(ds)
        T = M.reshape(ds + ds).transpose(perm + [p + k for p in perm])
        d = M.shape[0]
        return NamedOp(T.reshape(d, d), target)

    def close(self, other: "NamedOp", tol: float) -> bool:
        return self.names == other.names and np.allclose(self.mat, other.mat, atol=tol)


def _full(pm: ProcessMatrix) -> NamedOp:
    return NamedOp(_canonical(pm)[0], SYS)


def _psd(M: np.ndarray, tol: float) -> bool:
    H = (M + M.conj().T) / 2
    return bool(np.allclose(M, H, atol=tol) and np.linalg.eigvalsh(H).min() >= -tol)


def _other(k: str) -> str:
    return "B" if k == "A" else "A"


# ------------------------------------------------------------------ #
# Certificates                                                         #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class FixedOrderCertificate:
    """W is a quantum comb with the order P < first < second < F (QC-FO)."""
    order: str                                   # 'AB' or 'BA'

    def check(self, pm: ProcessMatrix, tol: float = 1e-8) -> bool:
        dims = scenario_dims(pm)
        W = _full(pm)
        if not _psd(W.mat, tol):
            return False
        return _comb_ok(W, self.order[0], dims, tol, top=True)

    def to_classical_control(self, pm: ProcessMatrix) -> "ClassicalControlCertificate":
        W = _canonical(pm)[0]
        Z = np.zeros_like(W)
        return ClassicalControlCertificate(W, Z) if self.order == "AB" \
            else ClassicalControlCertificate(Z, W)


@dataclass(frozen=True)
class ClassicalControlCertificate:
    """W = W_A + W_B with W_k >= 0 a comb for the order starting with k, the
    normalisation at P imposed on the sum (QC-CC, Wechs et al. Prop. 5)."""
    W_A: np.ndarray
    W_B: np.ndarray

    def check(self, pm: ProcessMatrix, tol: float = 1e-8) -> bool:
        dims = scenario_dims(pm)
        if not np.allclose(self.W_A + self.W_B, _canonical(pm)[0], atol=tol):
            return False
        if not (_psd(self.W_A, tol) and _psd(self.W_B, tol)):
            return False
        parts = {"A": NamedOp(self.W_A, SYS), "B": NamedOp(self.W_B, SYS)}
        if not all(_comb_ok(parts[k], k, dims, tol, top=False) for k in "AB"):
            return False
        firsts = [_first_level(parts[k], k, dims) for k in "AB"]
        tops = [f.trace([k + "I"], dims) for f, k in zip(firsts, "AB")]
        one_P = NamedOp(np.eye(dims["P"], dtype=complex), ("P",))
        return NamedOp(tops[0].mat + tops[1].mat, ("P",)).close(one_P, tol)

    def to_quantum_control(self, pm: ProcessMatrix) -> "QuantumControlCertificate":
        dims = scenario_dims(pm)
        mats = {}
        for k in "AB":
            l = _other(k)
            Wk = NamedOp(self.W_A if k == "A" else self.W_B, SYS)
            second = Wk.trace(["F", l + "O"], dims)                  # P kI kO lI
            second = NamedOp(second.mat / dims[l + "O"], second.names)
            first = second.trace([l + "I"], dims)                    # P kI kO
            first = NamedOp(first.trace([k + "O"], dims).mat / dims[k + "O"], ("P", k + "I"))
            mats[("", k)] = first.mat
            mats[(k, l)] = second.mat
        return QuantumControlCertificate(mats[("", "A")], mats[("", "B")],
                                         mats[("A", "B")], mats[("B", "A")])


@dataclass(frozen=True)
class QuantumControlCertificate:
    """Wechs et al. Prop. 7 for two slots: PSD matrices
        W0A on P A_I,  W0B on P B_I,  WAB on P A_I A_O B_I,  WBA on P A_I B_I B_O
    with  Tr_AI W0A + Tr_BI W0B = 1_P,
          Tr_BI WAB = W0A ⊗ 1_AO,   Tr_AI WBA = W0B ⊗ 1_BO,
          Tr_F W = WAB ⊗ 1_BO + WBA ⊗ 1_AO   (QC-QC)."""
    W0A: np.ndarray
    W0B: np.ndarray
    WAB: np.ndarray
    WBA: np.ndarray

    def check(self, pm: ProcessMatrix, tol: float = 1e-8) -> bool:
        dims = scenario_dims(pm)
        if not _psd(_canonical(pm)[0], tol):
            return False
        ops = {
            ("", "A"): NamedOp(self.W0A, _order(["P", "AI"])),
            ("", "B"): NamedOp(self.W0B, _order(["P", "BI"])),
            ("A", "B"): NamedOp(self.WAB, _order(["P", "AI", "AO", "BI"])),
            ("B", "A"): NamedOp(self.WBA, _order(["P", "AI", "BI", "BO"])),
        }
        for (K, k), op in ops.items():
            d = int(np.prod([dims[n] for n in op.names]))
            if op.mat.shape != (d, d) or not _psd(op.mat, tol):
                return False
        one_P = NamedOp(np.eye(dims["P"], dtype=complex), ("P",))
        top = ops[("", "A")].trace(["AI"], dims).mat + ops[("", "B")].trace(["BI"], dims).mat
        if not NamedOp(top, ("P",)).close(one_P, tol):
            return False
        for k in "AB":
            l = _other(k)
            lhs = ops[(k, l)].trace([l + "I"], dims)
            rhs = ops[("", k)].with_identity(k + "O", dims)
            if not lhs.close(rhs, tol):
                return False
        trF = _full(pm).trace(["F"], dims)
        rhs = ops[("A", "B")].with_identity("BO", dims).mat + ops[("B", "A")].with_identity("AO", dims).mat
        return trF.close(NamedOp(rhs, trF.names), tol)


def _first_level(Wk: NamedOp, k: str, dims) -> NamedOp:
    """W_(k) on P k_I implied by a comb Wk whose first slot is k."""
    l = _other(k)
    second = Wk.trace(["F", l + "O"], dims)
    first = second.trace([l + "I"], dims)
    return NamedOp(first.trace([k + "O"], dims).mat / (dims[l + "O"] * dims[k + "O"]),
                   ("P", k + "I"))


def _comb_ok(Wk: NamedOp, k: str, dims, tol: float, top: bool) -> bool:
    """Comb conditions for the order P < k < l < F (Wechs et al. Props. 2, 5):
    Tr_F Wk = W_(k,l) ⊗ 1_lO,  Tr_lI W_(k,l) = W_(k) ⊗ 1_kO,  and, if top,
    Tr_kI W_(k) = 1_P."""
    l = _other(k)
    trF = Wk.trace(["F"], dims)
    second = Wk.trace(["F", l + "O"], dims)
    second = NamedOp(second.mat / dims[l + "O"], second.names)
    if not trF.close(second.with_identity(l + "O", dims), tol):
        return False
    first = second.trace([l + "I"], dims)
    first_red = NamedOp(first.trace([k + "O"], dims).mat / dims[k + "O"], ("P", k + "I"))
    if not first.close(first_red.with_identity(k + "O", dims), tol):
        return False
    if top:
        one_P = np.eye(dims["P"], dtype=complex)
        return bool(np.allclose(first_red.trace([k + "I"], dims).mat, one_P, atol=tol))
    return True


@dataclass(frozen=True)
class CausalInequalityCertificate:
    """W is NOT a quantum circuit with quantum control of causal order (QC-QC):
    with the given local instruments, W wins the OCB causal game with
    probability above 3/4.  Every QC-QC process, and so every causally
    separable or fixed-order one, gives at most 3/4 (Wechs et al. 2021,
    Prop. 8; Oreshkov, Costa, Brukner 2012).

    instruments_A[(x, a)] and instruments_B[((y, b'), b)] are transposed Choi
    operators on each party's (input, output).  For processes with parties
    P and F, P prepares p_state (transposed, i.e. conj(rho)) and F is
    discarded."""
    instruments_A: Tuple[Tuple[Tuple[int, int], np.ndarray], ...]
    instruments_B: Tuple[Tuple[Tuple[Tuple[int, int], int], np.ndarray], ...]
    p_state: Optional[np.ndarray] = None

    BOUND = 0.75

    def _valid_instruments(self, dims, tol) -> bool:
        A, B = dict(self.instruments_A), dict(self.instruments_B)
        for inst, k, settings in ((A, "A", [(x,) for x in (0, 1)]),
                                  (B, "B", [((y, bp),) for y in (0, 1) for bp in (0, 1)])):
            dI, dO = dims[k + "I"], dims[k + "O"]
            for (setting,) in settings:
                total = np.zeros((dI * dO, dI * dO), dtype=complex)
                for o in (0, 1):
                    J = inst.get((setting, o))
                    if J is None or J.shape != (dI * dO, dI * dO) or not _psd(J, tol):
                        return False
                    total = total + J
                red = NamedOp(total, (k + "I", k + "O")).trace([k + "O"], dims)
                if not np.allclose(red.mat, np.eye(dI), atol=tol):     # trace preserving
                    return False
        return True

    def value(self, pm: ProcessMatrix) -> float:
        """OCB game winning probability of pm with these instruments."""
        from qrl.causal import ocb_wins
        W, dims = _canonical(pm)
        rho = self.p_state if self.p_state is not None else np.eye(dims["P"]) / dims["P"]
        # contract P with its preparation and discard F: W_AB on A_I A_O B_I B_O
        T = NamedOp(np.kron(rho, np.eye(W.shape[0] // dims["P"])) @ W, SYS)
        WAB = T.trace(["P", "F"], dims).mat
        A, B = dict(self.instruments_A), dict(self.instruments_B)
        p = 0.0
        for x in (0, 1):
            for y in (0, 1):
                for bp in (0, 1):
                    for a in (0, 1):
                        for b in (0, 1):
                            if ocb_wins(x, y, bp, a, b):
                                M = np.kron(A[(x, a)], B[((y, bp), b)])
                                p += np.trace(M @ WAB).real / 8
        return float(p)

    def check(self, pm: ProcessMatrix, tol: float = 1e-8) -> bool:
        dims = scenario_dims(pm)
        if not pm.is_valid():
            return False
        if self.p_state is not None and not _psd(self.p_state, tol):
            return False
        if not self._valid_instruments(dims, tol):
            return False
        return self.value(pm) > self.BOUND + tol


def ocb_certificate() -> CausalInequalityCertificate:
    """OCB's strategy as a certificate that a qubit process is outside QC-QC."""
    from qrl.causal import ocb_strategy
    A, B = ocb_strategy()
    return CausalInequalityCertificate(tuple(A.items()), tuple(B.items()))


# ------------------------------------------------------------------ #
# Classes                                                              #
# ------------------------------------------------------------------ #

FIXED_ORDER = "QC-FO"
CLASSICAL_CONTROL = "QC-CC"
QUANTUM_CONTROL = "QC-QC"


def fixed_order_certificate(pm: ProcessMatrix, tol: float = 1e-8) -> Optional[FixedOrderCertificate]:
    """QC-FO is decided directly (its conditions are linear): the order, or None."""
    for order in ("AB", "BA"):
        c = FixedOrderCertificate(order)
        if c.check(pm, tol):
            return c
    return None


def switch_quantum_control_certificate(d: int = 2) -> QuantumControlCertificate:
    """QC-QC certificate of the switch construct (qrl.switch_witness.switch_process),
    read off its two branches: tracing out F removes the coherence between them.
    With control |c> from P and target |0>, order A-then-B when c = 0."""
    P0 = np.diag([1.0, 0.0]).astype(complex)
    P1 = np.diag([0.0, 1.0]).astype(complex)
    t0 = np.zeros((d, d), dtype=complex); t0[0, 0] = 1
    wire = np.zeros(d * d, dtype=complex)
    for i in range(d):
        wire[i * d + i] = 1
    W1 = np.outer(wire, wire)                                    # |1>><<1|
    W0A = np.kron(P0, t0)                                        # P AI
    W0B = np.kron(P1, t0)                                        # P BI
    WAB = np.kron(np.kron(P0, t0), W1)                           # P AI (AO BI)
    WBA = _switch_WBA(d)                                         # P AI BI BO
    return QuantumControlCertificate(W0A, W0B, WAB, WBA)


def _switch_WBA(d: int) -> np.ndarray:
    """|1><1|_P ⊗ |0><0|_BI ⊗ |1>><<1|_{BO, AI}, as an operator on P AI BI BO."""
    v = np.zeros((2, d, d, d), dtype=complex)                   # P AI BI BO
    for i in range(d):
        v[1, i, 0, i] = 1
    v = v.reshape(-1)
    return np.outer(v, v)


def find_certificate(pm: ProcessMatrix, cls: str, tol: float = 1e-7):
    """
    Search for a certificate that pm is in class `cls` (QC-FO, QC-CC or QC-QC).
    QC-FO is decided directly; QC-CC and QC-QC by a semidefinite feasibility
    problem (needs cvxpy).  Returns a certificate that passes check(), or None
    if the problem is infeasible or the solution does not check.
    """
    if cls == FIXED_ORDER:
        return fixed_order_certificate(pm, tol)
    try:
        import cvxpy as cp
    except ImportError as e:                                       # pragma: no cover
        raise ImportError("find_certificate needs cvxpy for QC-CC and QC-QC") from e
    W, dims = _canonical(pm)
    D = W.shape[0]
    if cls == CLASSICAL_CONTROL:
        WA = cp.Variable((D, D), hermitian=True)
        cons = [WA >> 0, W - WA >> 0]
        cons += _cp_comb(WA, "A", dims) + _cp_comb(W - WA, "B", dims)
        cons += _cp_top_cc(WA, W - WA, dims)
        _solve(cp.Problem(cp.Minimize(0), cons))
        if WA.value is None:
            return None
        cert = ClassicalControlCertificate(WA.value, W - WA.value)
    elif cls == QUANTUM_CONTROL:
        n = lambda names: int(np.prod([dims[x] for x in names]))
        V = {k: cp.Variable((n(names), n(names)), hermitian=True) for k, names in
             (("0A", ("P", "AI")), ("0B", ("P", "BI")),
              ("AB", ("P", "AI", "AO", "BI")), ("BA", ("P", "AI", "BI", "BO")))}
        cons = [v >> 0 for v in V.values()] + _cp_qcqc(V, W, dims)
        _solve(cp.Problem(cp.Minimize(0), cons), small=True)
        if any(v.value is None for v in V.values()):
            return None
        cert = QuantumControlCertificate(V["0A"].value, V["0B"].value,
                                         V["AB"].value, V["BA"].value)
    else:
        raise ValueError(f"unknown class {cls!r}")
    return cert if cert.check(pm, tol=1e-5) else None


# --- cvxpy helpers: the same conditions as the checks, as linear maps ---------

def _solve(prob, small: bool = False):
    """Solve accurately: SCS at tight tolerance; Clarabel as a fallback for
    small problems (it needs much more memory)."""
    import cvxpy as cp
    order = [("SCS", {"eps": 1e-9, "max_iters": 200000})]
    if small:
        order.append(("CLARABEL", {}))
    for solver, opts in order:
        if solver in cp.installed_solvers():
            try:
                prob.solve(solver=solver, **opts)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    return
            except cp.error.SolverError:
                continue

def _cp_ptrace(X, names, out, dims):
    import cvxpy as cp
    names = list(names)
    for name in out:
        ds = [dims[n] for n in names]
        X = cp.partial_trace(X, ds, axis=names.index(name))
        names.remove(name)
    return X, tuple(names)


def _cp_kron_id(X, names, name, dims):
    """X ⊗ 1_name, permuted into SYS order (as a cvxpy expression)."""
    import cvxpy as cp
    names = tuple(names) + (name,)
    Y = cp.kron(X, np.eye(dims[name]))
    target = _order(names)
    if target == names:
        return Y, target
    ds = [dims[n] for n in names]
    perm = [names.index(n) for n in target]
    P = _perm_matrix(ds, perm)
    return P @ Y @ P.T, target


def _perm_matrix(ds, perm):
    d = int(np.prod(ds))
    idx = np.arange(d).reshape(ds).transpose(perm).reshape(-1)
    P = np.zeros((d, d))
    P[np.arange(d), idx] = 1
    return P


def _cp_comb(Wk, k, dims):
    l = _other(k)
    trF, n1 = _cp_ptrace(Wk, SYS, ["F"], dims)
    sec, n2 = _cp_ptrace(Wk, SYS, ["F", l + "O"], dims)
    sec = sec / dims[l + "O"]
    rhs, _ = _cp_kron_id(sec, n2, l + "O", dims)
    first, n3 = _cp_ptrace(sec, n2, [l + "I"], dims)
    red, n4 = _cp_ptrace(first, n3, [k + "O"], dims)
    rhs2, _ = _cp_kron_id(red / dims[k + "O"], n4, k + "O", dims)
    return [trF == rhs, first == rhs2]


def _cp_top_cc(WA, WB, dims):
    tops = []
    for Wk, k in ((WA, "A"), (WB, "B")):
        l = _other(k)
        X, names = _cp_ptrace(Wk, SYS, ["F", l + "O", l + "I", k + "O", k + "I"], dims)
        tops.append(X / (dims[l + "O"] * dims[k + "O"]))
    return [tops[0] + tops[1] == np.eye(dims["P"])]


def _cp_qcqc(V, W, dims):
    t0A, _ = _cp_ptrace(V["0A"], ("P", "AI"), ["AI"], dims)
    t0B, _ = _cp_ptrace(V["0B"], ("P", "BI"), ["BI"], dims)
    cons = [t0A + t0B == np.eye(dims["P"])]
    lAB, _ = _cp_ptrace(V["AB"], ("P", "AI", "AO", "BI"), ["BI"], dims)
    rAB, _ = _cp_kron_id(V["0A"], ("P", "AI"), "AO", dims)
    lBA, _ = _cp_ptrace(V["BA"], ("P", "AI", "BI", "BO"), ["AI"], dims)
    rBA, _ = _cp_kron_id(V["0B"], ("P", "BI"), "BO", dims)
    cons += [lAB == rAB, lBA == rBA]
    trF = NamedOp(W, SYS).trace(["F"], dims).mat
    xAB, _ = _cp_kron_id(V["AB"], ("P", "AI", "AO", "BI"), "BO", dims)
    xBA, _ = _cp_kron_id(V["BA"], ("P", "AI", "BI", "BO"), "AO", dims)
    cons += [xAB + xBA == trF]
    return cons
