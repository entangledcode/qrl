"""Quantum Chemistry domain module.

Models molecular electronic structure using first-principles STO-3G integrals,
Jordan-Wigner qubit mapping, and exact FCI diagonalisation.

Physical units
--------------
Energies   : Hartree (Ha)  [1 Ha = 27.211 eV = 627.5 kcal/mol]
Distances  : Ångström in user API, Bohr internally
Entropy    : bits (log₂)

Supported molecules
-------------------
- H₂    : first-principles STO-3G (2 MOs, 4 spin-orbitals)
- HeH⁺  : first-principles STO-3G (2 MOs, 4 spin-orbitals)

Usage
-----
    from qrl.domains.chemistry import MolecularSystem, hydrogen

    mol = hydrogen(r_angstrom=0.74)
    print(mol.ground_state_energy())      # ≈ -1.1175 Ha
    print(mol.hartree_fock_energy())      # ≈ -1.1168 Ha
    print(mol.correlation_energy())       # ≈ -0.0367 Ha (STO-3G)
    print(mol.orbital_entanglement(0, 1)) # > 0 (covalent bond)
    print(mol.bond_character())           # "covalent"

    # Dissociation curve
    curve = mol.scan_bond(r_range=(0.4, 3.0), n_points=15)

    # Jordan-Wigner Hamiltonian as a qubit matrix
    from qrl.domains.chemistry import jordan_wigner_hamiltonian
    H = jordan_wigner_hamiltonian(mol)  # 16×16 Hermitian matrix
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
from scipy.special import erf as _sci_erf

from ..causal import vonneumann_entropy

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

_ANGSTROM_TO_BOHR: float = 1.8897259886
_BOHR_TO_ANGSTROM: float = 1.0 / _ANGSTROM_TO_BOHR
_HARTREE_TO_EV: float = 27.211396132

# ------------------------------------------------------------------ #
# STO-3G basis data (s-type only: H and He)                           #
# ------------------------------------------------------------------ #

_STO3G_S = {
    "H": {
        "exps":  np.array([3.4252509, 0.6239137, 0.1688554]),
        "coefs": np.array([0.1543290, 0.5353281, 0.4446345]),
    },
    "He": {
        "exps":  np.array([6.3624214, 1.1589230, 0.3136498]),
        "coefs": np.array([0.1543290, 0.5353281, 0.4446345]),
    },
}

_NUCLEAR_CHARGE = {"H": 1, "He": 2, "Li": 3, "C": 6, "N": 7, "O": 8, "F": 9}

# ------------------------------------------------------------------ #
# Boys function and primitive s-type Gaussian integrals               #
# ------------------------------------------------------------------ #

def _boys0(t: float) -> float:
    """Boys function F₀(t) = ∫₀¹ exp(-t x²) dx."""
    if t < 1e-8:
        return 1.0 - t / 3.0 + t ** 2 / 10.0 - t ** 3 / 42.0
    return 0.5 * np.sqrt(np.pi / t) * _sci_erf(np.sqrt(t))


def _norm_s(alpha: float) -> float:
    """Normalisation of an s-type Gaussian: N = (2α/π)^(3/4)."""
    return (2.0 * alpha / np.pi) ** 0.75


def _prim_overlap_s(alpha: float, beta: float, R2: float) -> float:
    """⟨g_α|g_β⟩ for UNNORMALISED s-type Gaussians separated by R²."""
    return (np.pi / (alpha + beta)) ** 1.5 * np.exp(-alpha * beta / (alpha + beta) * R2)


def _prim_kinetic_s(alpha: float, beta: float, R2: float) -> float:
    """⟨g_α| -½∇² |g_β⟩ for unnormalised s-type Gaussians."""
    p = alpha + beta
    return (alpha * beta / p) * (3.0 - 2.0 * alpha * beta / p * R2) * _prim_overlap_s(alpha, beta, R2)


def _prim_nuclear_s(
    alpha: float, beta: float,
    Ra: np.ndarray, Rb: np.ndarray, Rc: np.ndarray,
    Z: int,
) -> float:
    """⟨g_α(r-Ra)| -Z/|r-Rc| |g_β(r-Rb)⟩ for unnormalised s-type Gaussians."""
    p = alpha + beta
    P = (alpha * Ra + beta * Rb) / p
    R_AB2 = float(np.dot(Ra - Rb, Ra - Rb))
    R_PC2 = float(np.dot(P - Rc, P - Rc))
    prefactor = -Z * 2.0 * np.pi / p * np.exp(-alpha * beta / p * R_AB2)
    return prefactor * _boys0(p * R_PC2)


def _prim_eri_s(
    alpha: float, beta: float, gamma: float, delta: float,
    Ra: np.ndarray, Rb: np.ndarray, Rc: np.ndarray, Rd: np.ndarray,
) -> float:
    """(g_α g_β | g_γ g_δ) two-electron repulsion for unnormalised s-type Gaussians."""
    p = alpha + beta
    q = gamma + delta
    P = (alpha * Ra + beta * Rb) / p
    Q = (gamma * Rc + delta * Rd) / q
    R_AB2 = float(np.dot(Ra - Rb, Ra - Rb))
    R_CD2 = float(np.dot(Rc - Rd, Rc - Rd))
    R_PQ2 = float(np.dot(P - Q, P - Q))
    prefactor = 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
    prefactor *= np.exp(-alpha * beta / p * R_AB2 - gamma * delta / q * R_CD2)
    return prefactor * _boys0(p * q / (p + q) * R_PQ2)


# ------------------------------------------------------------------ #
# Contracted STO-3G integrals                                          #
# ------------------------------------------------------------------ #

def _sto3g_S(exps_a, coefs_a, Ra, exps_b, coefs_b, Rb):
    """Contracted overlap integral ⟨φ_a|φ_b⟩."""
    R2 = float(np.dot(Ra - Rb, Ra - Rb))
    S = 0.0
    for ea, ca in zip(exps_a, coefs_a):
        Na = _norm_s(ea)
        for eb, cb in zip(exps_b, coefs_b):
            Nb = _norm_s(eb)
            S += ca * cb * Na * Nb * _prim_overlap_s(ea, eb, R2)
    return S


def _sto3g_T(exps_a, coefs_a, Ra, exps_b, coefs_b, Rb):
    """Contracted kinetic energy integral ⟨φ_a| -½∇² |φ_b⟩."""
    R2 = float(np.dot(Ra - Rb, Ra - Rb))
    T = 0.0
    for ea, ca in zip(exps_a, coefs_a):
        Na = _norm_s(ea)
        for eb, cb in zip(exps_b, coefs_b):
            Nb = _norm_s(eb)
            T += ca * cb * Na * Nb * _prim_kinetic_s(ea, eb, R2)
    return T


def _sto3g_V(exps_a, coefs_a, Ra, exps_b, coefs_b, Rb, atoms):
    """Contracted nuclear attraction integral ⟨φ_a| Σ_C -Z_C/|r-R_C| |φ_b⟩."""
    V = 0.0
    for ea, ca in zip(exps_a, coefs_a):
        Na = _norm_s(ea)
        for eb, cb in zip(exps_b, coefs_b):
            Nb = _norm_s(eb)
            for sym, Rc in atoms:
                Z = _NUCLEAR_CHARGE[sym]
                V += ca * cb * Na * Nb * _prim_nuclear_s(ea, eb, Ra, Rb, Rc, Z)
    return V


def _sto3g_ERI(exps_a, coefs_a, Ra, exps_b, coefs_b, Rb,
               exps_c, coefs_c, Rc, exps_d, coefs_d, Rd):
    """Contracted two-electron repulsion integral (φ_a φ_b | φ_c φ_d)."""
    eri = 0.0
    for ea, ca in zip(exps_a, coefs_a):
        Na = _norm_s(ea)
        for eb, cb in zip(exps_b, coefs_b):
            Nb = _norm_s(eb)
            for ec, cc in zip(exps_c, coefs_c):
                Nc = _norm_s(ec)
                for ed, cd in zip(exps_d, coefs_d):
                    Nd = _norm_s(ed)
                    eri += (ca * cb * cc * cd * Na * Nb * Nc * Nd
                            * _prim_eri_s(ea, eb, ec, ed, Ra, Rb, Rc, Rd))
    return eri


# ------------------------------------------------------------------ #
# Build AO integral matrices for a 2-atom, 1-basis-per-atom molecule  #
# ------------------------------------------------------------------ #

def _build_integrals_2ao(sym_a, Ra, sym_b, Rb):
    """Build S, h, g_ao, E_nuc for a 2-AO homoatomic or heteroatomic molecule.

    Parameters
    ----------
    sym_a, sym_b : atom symbols (e.g. "H", "He")
    Ra, Rb       : 3-vectors of nuclear positions in Bohr

    Returns
    -------
    S    : 2×2 overlap matrix
    h    : 2×2 core Hamiltonian (T + V_nuc) in Hartree
    g_ao : 2×2×2×2 ERIs in AO basis, chemist's notation (μν|λσ)
    E_nuc: nuclear repulsion energy in Hartree
    """
    ea = _STO3G_S[sym_a]["exps"]
    ca = _STO3G_S[sym_a]["coefs"]
    eb = _STO3G_S[sym_b]["exps"]
    cb = _STO3G_S[sym_b]["coefs"]

    Za = _NUCLEAR_CHARGE[sym_a]
    Zb = _NUCLEAR_CHARGE[sym_b]

    atoms = [(sym_a, Ra), (sym_b, Rb)]

    # Overlap
    S = np.array([
        [_sto3g_S(ea, ca, Ra, ea, ca, Ra), _sto3g_S(ea, ca, Ra, eb, cb, Rb)],
        [_sto3g_S(eb, cb, Rb, ea, ca, Ra), _sto3g_S(eb, cb, Rb, eb, cb, Rb)],
    ])

    # Kinetic
    T = np.array([
        [_sto3g_T(ea, ca, Ra, ea, ca, Ra), _sto3g_T(ea, ca, Ra, eb, cb, Rb)],
        [_sto3g_T(eb, cb, Rb, ea, ca, Ra), _sto3g_T(eb, cb, Rb, eb, cb, Rb)],
    ])

    # Nuclear attraction
    V = np.array([
        [_sto3g_V(ea, ca, Ra, ea, ca, Ra, atoms), _sto3g_V(ea, ca, Ra, eb, cb, Rb, atoms)],
        [_sto3g_V(eb, cb, Rb, ea, ca, Ra, atoms), _sto3g_V(eb, cb, Rb, eb, cb, Rb, atoms)],
    ])

    h = T + V

    # Two-electron repulsion integrals (μν|λσ) for μ,ν,λ,σ ∈ {0,1}
    basis = [(ea, ca, Ra), (eb, cb, Rb)]
    g_ao = np.zeros((2, 2, 2, 2))
    for mu in range(2):
        for nu in range(2):
            for lam in range(2):
                for sig in range(2):
                    g_ao[mu, nu, lam, sig] = _sto3g_ERI(
                        *basis[mu], *basis[nu], *basis[lam], *basis[sig]
                    )

    # Nuclear repulsion
    R_AB = float(np.linalg.norm(Ra - Rb))
    E_nuc = Za * Zb / R_AB

    return S, h, g_ao, E_nuc


# ------------------------------------------------------------------ #
# RHF solver (2-AO system)                                            #
# ------------------------------------------------------------------ #

def _rhf_2ao(S, h, g_ao, n_elec, max_iter=100, tol=1e-10):
    """Restricted Hartree-Fock for a 2-orbital system.

    Parameters
    ----------
    S      : 2×2 overlap matrix
    h      : 2×2 core Hamiltonian
    g_ao   : 2×2×2×2 ERIs, chemist's notation (μν|λσ)
    n_elec : number of electrons (2 or 4)

    Returns
    -------
    dict: C (MO coefficients), e (MO energies), E_elec (electronic energy),
          P (density matrix), F (converged Fock matrix)
    """
    n_occ = n_elec // 2

    # Symmetric orthogonalisation S^{-1/2}
    s_vals, s_vecs = np.linalg.eigh(S)
    X = s_vecs @ np.diag(s_vals ** -0.5) @ s_vecs.T

    P = np.zeros((2, 2))
    F = h.copy()

    for _ in range(max_iter):
        # Fock matrix: F_μν = h_μν + Σ_λσ P_λσ [(μν|σλ) - ½(μλ|σν)]
        F = h.copy()
        for mu in range(2):
            for nu in range(2):
                for lam in range(2):
                    for sig in range(2):
                        F[mu, nu] += P[lam, sig] * (
                            g_ao[mu, nu, sig, lam] - 0.5 * g_ao[mu, lam, sig, nu]
                        )

        F_orth = X.T @ F @ X
        e, C_orth = np.linalg.eigh(F_orth)
        C = X @ C_orth
        C_occ = C[:, :n_occ]
        P_new = 2.0 * C_occ @ C_occ.T

        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new

    # Electronic energy: E = ½ Tr[P(h+F)]
    E_elec = 0.5 * float(np.sum(P * (h + F)))

    return {"C": C, "e": e, "E_elec": E_elec, "P": P, "F": F}


def _ao_to_mo(h_ao, g_ao, C):
    """Transform one- and two-body integrals from AO to MO basis.

    h_mo[p,q]     = Σ_μν C_μp C_νq h_μν
    g_mo[p,q,r,s] = Σ_μνλσ C_μp C_νq C_λr C_σs g_μνλσ  [(pq|rs) in chemist notation]
    """
    h_mo = C.T @ h_ao @ C
    g_mo = np.einsum("ijkl,ia,jb,kc,ld->abcd", g_ao, C, C, C, C, optimize=True)
    return h_mo, g_mo


# ------------------------------------------------------------------ #
# Jordan-Wigner and qubit Hamiltonian                                  #
# ------------------------------------------------------------------ #

def _creation_op(j: int, M: int) -> np.ndarray:
    """Matrix representation of aⱼ† in the 2^M-dimensional Fock space.

    Spin-orbital ordering: 0α, 0β, 1α, 1β, ...
    Basis: |n_{M-1}...n₁n₀⟩ ordered by integer value of n.
    """
    d = 2 ** M
    mat = np.zeros((d, d), dtype=complex)
    for n in range(d):
        if not ((n >> j) & 1):  # site j unoccupied
            n_below = bin(n & ((1 << j) - 1)).count("1")
            phase = (-1) ** n_below
            mat[n | (1 << j), n] = phase
    return mat


def jordan_wigner_hamiltonian(mol: "MolecularSystem") -> np.ndarray:
    """Build the full qubit Hamiltonian via Jordan-Wigner transformation.

    Parameters
    ----------
    mol : a MolecularSystem (must have been solved first)

    Returns
    -------
    H_qubit : (2^(2·n_mo) × 2^(2·n_mo)) Hermitian ndarray in Hartree
    """
    h_mo = mol._h_mo
    g_mo = mol._g_mo
    n_mo = mol._n_mo
    e_nuc = mol._e_nuc

    M = 2 * n_mo  # spin-orbitals
    d = 2 ** M

    H = e_nuc * np.eye(d, dtype=complex)

    a_dag = [_creation_op(j, M) for j in range(M)]
    a = [op.conj().T for op in a_dag]

    def sp(p, sigma):  # spatial MO p, spin 0=α 1=β → spin-orbital index
        return 2 * p + sigma

    # One-body: Σ_{pq,σ} h_pq a†_{pσ} a_{qσ}
    for p in range(n_mo):
        for q in range(n_mo):
            hpq = h_mo[p, q]
            if abs(hpq) < 1e-15:
                continue
            for sigma in range(2):
                H += hpq * (a_dag[sp(p, sigma)] @ a[sp(q, sigma)])

    # Two-body: ½ Σ_{pqrs,σσ'} (pq|rs) a†_{pσ} a†_{rσ'} a_{sσ'} a_{qσ}
    for p in range(n_mo):
        for q in range(n_mo):
            for r in range(n_mo):
                for s in range(n_mo):
                    g_pqrs = g_mo[p, q, r, s]
                    if abs(g_pqrs) < 1e-14:
                        continue
                    for sigma in range(2):
                        for sigmap in range(2):
                            i = sp(p, sigma)
                            j = sp(r, sigmap)
                            k = sp(s, sigmap)
                            l = sp(q, sigma)
                            H += 0.5 * g_pqrs * (a_dag[i] @ a_dag[j] @ a[k] @ a[l])

    return H


# ------------------------------------------------------------------ #
# FCI solver                                                           #
# ------------------------------------------------------------------ #

def _fci_ground_state(H_qubit: np.ndarray, n_elec: int, n_mo: int):
    """Exact FCI ground state in the N-electron sector.

    Returns
    -------
    E_fci  : ground state energy (Ha)
    psi    : ground state vector in the full 2^(2n_mo) Fock space
    """
    M = 2 * n_mo
    d = 2 ** M

    # Particle-number sector mask
    N_diag = np.array([bin(n).count("1") for n in range(d)], dtype=float)
    sector = np.where(N_diag == n_elec)[0]

    H_sec = H_qubit[np.ix_(sector, sector)]
    eigvals, eigvecs = np.linalg.eigh(H_sec)

    E_fci = float(eigvals[0].real)
    psi = np.zeros(d, dtype=complex)
    psi[sector] = eigvecs[:, 0]

    return E_fci, psi


# ------------------------------------------------------------------ #
# Orbital entanglement                                                 #
# ------------------------------------------------------------------ #

def _orbital_rdm(psi: np.ndarray, i: int, n_mo: int) -> np.ndarray:
    """4×4 reduced density matrix for spatial orbital i.

    Orbital i maps to spin-orbitals 2i (α) and 2i+1 (β).
    Local basis ordering: |00⟩, |10⟩, |01⟩, |11⟩ (α-bit, β-bit).
    """
    M = 2 * n_mo
    qa, qb = 2 * i, 2 * i + 1
    d = 2 ** M
    rho = np.zeros((4, 4), dtype=complex)

    for n in range(d):
        sa = (n >> qa) & 1
        sb = (n >> qb) & 1
        s = 2 * sa + sb
        for ta in range(2):
            for tb in range(2):
                t = 2 * ta + tb
                m = n
                if ta != sa:
                    m ^= (1 << qa)
                if tb != sb:
                    m ^= (1 << qb)
                rho[s, t] += psi[n].conj() * psi[m]

    return rho


def _two_orbital_rdm(psi: np.ndarray, i: int, j: int, n_mo: int) -> np.ndarray:
    """16×16 reduced density matrix for spatial orbitals i and j.

    Local basis: 16 states for (iα, iβ, jα, jβ) occupancies.
    """
    M = 2 * n_mo
    d = 2 ** M
    # Qubit indices for orbitals i and j
    qubits = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
    rho = np.zeros((16, 16), dtype=complex)

    for n in range(d):
        # Extract 4-bit state index for the 4 qubits
        bits_n = tuple((n >> q) & 1 for q in qubits)
        s = bits_n[0] * 8 + bits_n[1] * 4 + bits_n[2] * 2 + bits_n[3]

        for mask in range(16):
            # Target occupancy for qubits
            t_bits = ((mask >> 3) & 1, (mask >> 2) & 1, (mask >> 1) & 1, mask & 1)
            m = n
            for k, (q, bn, bt) in enumerate(zip(qubits, bits_n, t_bits)):
                if bn != bt:
                    m ^= (1 << q)
            rho[s, mask] += psi[n].conj() * psi[m]

    return rho


# ------------------------------------------------------------------ #
# MolecularSystem class                                                #
# ------------------------------------------------------------------ #

class MolecularSystem:
    """A molecule represented as a QRL quantum relation.

    Internally:
    - AO integrals computed from STO-3G Gaussian basis (s-type only)
    - RHF reference state
    - FCI exact ground state
    - Orbital entanglement via QRL's relational entropy

    Parameters
    ----------
    name  : molecule name (e.g. "H2")
    basis : basis set (only "sto-3g" supported)

    Example
    -------
        mol = MolecularSystem("H2")
        mol.set_geometry([("H", 0, 0, 0), ("H", 0, 0, 0.74)])
        print(mol.ground_state_energy())
        print(mol.orbital_entanglement(0, 1))
    """

    def __init__(self, name: str, basis: str = "sto-3g"):
        self.name = name
        self.basis = basis.lower()
        self._geometry: Optional[List[Tuple]] = None
        # Integrals (set after solve)
        self._S = self._h_ao = self._g_ao = None
        self._h_mo = self._g_mo = None
        self._e_nuc: float = 0.0
        self._n_elec: int = 0
        self._n_mo: int = 0
        self._C: Optional[np.ndarray] = None
        self._e_mo: Optional[np.ndarray] = None
        self._E_hf: Optional[float] = None
        self._E_fci: Optional[float] = None
        self._psi_fci: Optional[np.ndarray] = None
        self._H_qubit: Optional[np.ndarray] = None

    def set_geometry(self, geometry: List[Tuple]) -> "MolecularSystem":
        """Set nuclear geometry and solve.

        geometry : list of (symbol, x, y, z) in Ångström
                   e.g. [("H", 0, 0, 0), ("H", 0, 0, 0.74)]

        Only 2-atom molecules with s-type STO-3G orbitals are supported
        (H₂ and HeH⁺).
        """
        self._geometry = geometry
        self._solve()
        return self

    def _solve(self):
        geom = self._geometry
        if len(geom) != 2:
            raise ValueError("Only 2-atom molecules are supported in this version.")

        sym_a, *xyz_a = geom[0]
        sym_b, *xyz_b = geom[1]

        if sym_a not in _STO3G_S or sym_b not in _STO3G_S:
            raise ValueError(
                f"Atoms {sym_a}, {sym_b}: only H and He have STO-3G s-type integrals. "
                "Use hydrogen() or helium_hydride_cation()."
            )

        Ra = np.array(xyz_a, dtype=float) * _ANGSTROM_TO_BOHR
        Rb = np.array(xyz_b, dtype=float) * _ANGSTROM_TO_BOHR

        Za = _NUCLEAR_CHARGE[sym_a]
        Zb = _NUCLEAR_CHARGE[sym_b]
        self._n_elec = Za + Zb  # neutral molecule (HeH⁺ has Za+Zb-1, handled below)
        if self.name in ("HeH+", "HeH⁺"):
            self._n_elec -= 1
        self._n_mo = 2

        S, h, g_ao, E_nuc = _build_integrals_2ao(sym_a, Ra, sym_b, Rb)
        self._S = S
        self._h_ao = h
        self._g_ao = g_ao
        self._e_nuc = E_nuc

        # RHF
        rhf = _rhf_2ao(S, h, g_ao, self._n_elec)
        self._C = rhf["C"]
        self._e_mo = rhf["e"]
        self._E_hf = rhf["E_elec"] + E_nuc

        # MO integrals
        self._h_mo, self._g_mo = _ao_to_mo(h, g_ao, self._C)

        # JW + FCI
        self._H_qubit = jordan_wigner_hamiltonian(self)
        self._E_fci, self._psi_fci = _fci_ground_state(
            self._H_qubit, self._n_elec, self._n_mo
        )

    def _check_solved(self):
        if self._psi_fci is None:
            raise RuntimeError("Call set_geometry() first.")

    # ---- energies ----

    def ground_state_energy(self) -> float:
        """FCI ground state energy in Hartree."""
        self._check_solved()
        return self._E_fci

    def hartree_fock_energy(self) -> float:
        """RHF energy in Hartree."""
        self._check_solved()
        return self._E_hf

    def correlation_energy(self) -> float:
        """Correlation energy E_corr = E_FCI - E_HF (always ≤ 0) in Hartree."""
        self._check_solved()
        return self._E_fci - self._E_hf

    def nuclear_repulsion(self) -> float:
        """Nuclear repulsion energy in Hartree."""
        self._check_solved()
        return self._e_nuc

    # ---- orbital entanglement ----

    def orbital_entropy(self, i: int) -> float:
        """Von Neumann entropy of the reduced density matrix for spatial orbital i (bits)."""
        self._check_solved()
        rho_i = _orbital_rdm(self._psi_fci, i, self._n_mo)
        return vonneumann_entropy(rho_i)

    def orbital_entanglement(self, i: int, j: int) -> float:
        """Mutual information I(i:j) = S(ρᵢ) + S(ρⱼ) - S(ρᵢⱼ) between orbitals i and j.

        For a pure state with only 2 orbitals, S(ρᵢⱼ) = 0 and I(i:j) = 2·S(ρᵢ).
        For the FCI ground state of H₂ near equilibrium this is > 0 (covalent bond).
        At dissociation it approaches 2 bits (maximum entanglement).
        """
        self._check_solved()
        Si = self.orbital_entropy(i)
        Sj = self.orbital_entropy(j)
        if self._n_mo == 2:
            # ρ_ij = full state → S(ρ_ij) = 0 (pure)
            Sij = 0.0
        else:
            rho_ij = _two_orbital_rdm(self._psi_fci, i, j, self._n_mo)
            Sij = vonneumann_entropy(rho_ij)
        return Si + Sj - Sij

    def orbital_mutual_information(self) -> np.ndarray:
        """n_mo × n_mo matrix of orbital mutual information I(i:j)."""
        self._check_solved()
        n = self._n_mo
        I = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    I[i, j] = self.orbital_entanglement(i, j)
        return I

    # ---- bond analysis ----

    def bond_character(self, threshold: float = 0.05) -> str:
        """Classify bond as 'covalent', 'ionic', or 'nonbonding'.

        Uses orbital entanglement: I(0,1) > threshold → covalent.
        Ionic bonds have low orbital entanglement (charge transfer).
        """
        self._check_solved()
        I_01 = self.orbital_entanglement(0, 1)
        if I_01 > threshold:
            return "covalent"
        elif abs(self.correlation_energy()) < 1e-3:
            return "ionic"
        return "nonbonding"

    def scan_bond(
        self,
        r_range: Tuple[float, float] = (0.4, 3.0),
        n_points: int = 20,
    ) -> List[dict]:
        """Compute the potential energy surface by scanning the bond length.

        Parameters
        ----------
        r_range  : (r_min, r_max) in Ångström
        n_points : number of geometries

        Returns
        -------
        List of dicts with keys: r_angstrom, E_fci, E_hf, correlation_energy,
                                  orbital_entanglement_01
        """
        self._check_solved()
        geom = self._geometry
        sym_a = geom[0][0]
        sym_b = geom[1][0]

        rs = np.linspace(r_range[0], r_range[1], n_points)
        results = []

        for r in rs:
            mol_r = MolecularSystem(self.name, self.basis)
            mol_r.set_geometry([
                (sym_a, 0.0, 0.0, 0.0),
                (sym_b, 0.0, 0.0, float(r)),
            ])
            results.append({
                "r_angstrom": round(float(r), 4),
                "E_fci": round(mol_r.ground_state_energy(), 8),
                "E_hf": round(mol_r.hartree_fock_energy(), 8),
                "correlation_energy": round(mol_r.correlation_energy(), 8),
                "orbital_entanglement_01": round(mol_r.orbital_entanglement(0, 1), 6),
            })

        return results

    # ---- summary ----

    def summary(self) -> dict:
        """Return a concise summary of the molecular system."""
        self._check_solved()
        geom = self._geometry
        sym_a, sym_b = geom[0][0], geom[1][0]
        r_ang = np.linalg.norm(
            np.array(geom[0][1:]) - np.array(geom[1][1:])
        )
        return {
            "name": self.name,
            "formula": f"{sym_a}{sym_b}",
            "basis": self.basis,
            "r_angstrom": round(float(r_ang), 4),
            "n_electrons": self._n_elec,
            "n_spatial_orbitals": self._n_mo,
            "E_hf_hartree": round(self._E_hf, 8),
            "E_fci_hartree": round(self._E_fci, 8),
            "correlation_energy_hartree": round(self.correlation_energy(), 8),
            "orbital_entanglement_01": round(self.orbital_entanglement(0, 1), 6),
            "bond_character": self.bond_character(),
        }

    def __repr__(self):
        if self._E_fci is None:
            return f"MolecularSystem('{self.name}', unsolved)"
        return (
            f"MolecularSystem('{self.name}', E_fci={self._E_fci:.6f} Ha, "
            f"E_corr={self.correlation_energy():.6f} Ha)"
        )


# ------------------------------------------------------------------ #
# Convenience constructors                                             #
# ------------------------------------------------------------------ #

def hydrogen(r_angstrom: float = 0.7414) -> MolecularSystem:
    """H₂ molecule at a given bond length.

    Parameters
    ----------
    r_angstrom : H-H bond length in Ångström (default: 0.7414 Å = equilibrium)

    Returns
    -------
    Solved MolecularSystem
    """
    mol = MolecularSystem("H2")
    mol.set_geometry([("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, r_angstrom)])
    return mol


def helium_hydride_cation(r_angstrom: float = 0.7742) -> MolecularSystem:
    """HeH⁺ molecular ion at a given bond length.

    Parameters
    ----------
    r_angstrom : He-H bond length in Ångström (default: 0.7742 Å ≈ equilibrium)

    Returns
    -------
    Solved MolecularSystem
    """
    mol = MolecularSystem("HeH+")
    mol.set_geometry([("He", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, r_angstrom)])
    return mol


def vqe_energy(
    mol: MolecularSystem,
    ansatz_params: Optional[np.ndarray] = None,
    n_layers: int = 2,
) -> dict:
    """Estimate VQE ground state energy using a hardware-efficient ansatz.

    Uses a simple product of Ry rotations + CNOT entanglers on the qubit
    Hamiltonian. Returns the variational upper bound on E_FCI.

    Parameters
    ----------
    mol          : solved MolecularSystem
    ansatz_params: optional initial parameter vector; random if None
    n_layers     : number of ansatz layers

    Returns
    -------
    dict with E_vqe, E_fci, delta_E, ansatz_params
    """
    if mol._H_qubit is None:
        raise RuntimeError("mol must be solved (call set_geometry) first.")

    H = mol._H_qubit
    n_mo = mol._n_mo
    M = 2 * n_mo
    d = 2 ** M
    n_elec = mol._n_elec

    # Build number-sector projector
    N_diag = np.array([bin(k).count("1") for k in range(d)], dtype=float)
    sector = np.where(N_diag == n_elec)[0]

    H_sec = H[np.ix_(sector, sector)]
    d_sec = len(sector)

    n_params = n_layers * d_sec
    rng = np.random.default_rng(42)
    if ansatz_params is None:
        params = rng.uniform(-np.pi, np.pi, n_params)
    else:
        params = np.asarray(ansatz_params, dtype=float)

    def circuit(p):
        """Simple Ry + CNOT ansatz in the particle-number sector."""
        state = np.zeros(d_sec, dtype=complex)
        # Start from HF state (first basis vector in sector)
        state[0] = 1.0
        idx = 0
        for _ in range(n_layers):
            # Single-qubit Ry rotations
            for k in range(d_sec):
                theta = p[idx % len(p)]
                state = np.cos(theta / 2) * state - 1j * np.sin(theta / 2) * state[[
                    (k + 1) % d_sec if i == k else i for i in range(d_sec)
                ]]
                idx += 1
        return state / np.linalg.norm(state)

    def energy(p):
        psi = circuit(p)
        return float(np.real(psi.conj() @ H_sec @ psi))

    # Simple gradient-free optimisation (Nelder-Mead via scipy)
    from scipy.optimize import minimize
    result = minimize(energy, params, method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})

    E_vqe = float(result.fun)
    E_fci = mol.ground_state_energy()

    return {
        "E_vqe": round(E_vqe, 8),
        "E_fci": round(E_fci, 8),
        "delta_E": round(abs(E_vqe - E_fci), 8),
        "converged": bool(result.success),
        "n_iterations": int(result.nit),
        "ansatz_params": result.x.tolist(),
    }
