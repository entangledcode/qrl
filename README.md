# Quantum Relational Language (QRL)

**A relations-first approach to quantum computing: describe the correlations, derive the predictions.**

*Formerly known as QPL (Quantum Process Language) — renamed January 2026 to avoid conflict with Selinger's QPL (2004)*

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18292199-blue)](https://doi.org/10.5281/zenodo.18292199)
[![Tests](https://img.shields.io/badge/Tests-595%20passing-brightgreen)](tests/)
[![Lines](https://img.shields.io/badge/Code-~9500%20lines-blue)](src/)
[![Photonic](https://img.shields.io/badge/Photonic-Verified-purple)](examples/quandela/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## Vision

**QRL is a physics modeling tool first, quantum programming framework second.**

Most quantum frameworks ask: *"How do I run this circuit on hardware?"*

QRL asks: *"Given these physical relations, what does quantum mechanics predict?"*

This flips the question. Instead of programming a computer, you're **modeling physics**. The relational formalism matches how quantum mechanics actually works—correlations between subsystems are the fundamental reality, not gates acting on states.

> **The hypothesis:** Because QRL's relational approach aligns with the structure of quantum physics, it may reveal insights that gate-centric formalisms obscure. This is a research proposition we're actively exploring.

## What is QRL?

QRL treats **entanglement as a first-class primitive** and compiles directly to **Measurement-Based Quantum Computing (MBQC)** patterns—without intermediate gate decomposition.

Unlike gate-based languages (Qiskit, Cirq, Q#), QRL expresses quantum programs as relationships between systems, which map naturally to:
- The cluster states and measurement patterns that power **photonic quantum computers**
- The correlations that define **Bell tests** and foundational quantum experiments
- The relational structure of **quantum networks** and protocols

## Why Relations First?

**The insight:** Quantum mechanics is fundamentally about correlations between subsystems. Bell's theorem, GHZ paradox, teleportation—these aren't about gates, they're about *relations*.

```
Gate-based thinking:  "Apply CNOT, then Hadamard, then measure"
Relational thinking:  "A and B are maximally correlated—what do measurements reveal?"
```

QRL lets you describe the correlations directly. The compilation to hardware follows from the physics.

```
Traditional: Gates → Circuit → Decompose → MBQC patterns → Hardware
QRL:         Relations → Graph extraction → MBQC patterns → Hardware
```

## Quick Start

```python
from qrl import QRLProgram, create_question, QuestionType
from qrl.mbqc import extract_graph, generate_pattern_from_relation

# Create entangled quantum systems
program = QRLProgram("Bell State Demo")
qubit_a = program.create_system()
qubit_b = program.create_system()
bell_pair = program.entangle(qubit_a, qubit_b)

# Extract MBQC graph structure
graph = extract_graph(bell_pair)
print(f"Cluster state: {graph.number_of_nodes()} qubits, {graph.number_of_edges()} edges")

# Generate measurement pattern
pattern = generate_pattern_from_relation(bell_pair)
print(f"Pattern: {pattern.description}")

# Measure with explicit context
alice = program.add_perspective("alice")
question = create_question(QuestionType.SPIN_Z)
result = program.ask(bell_pair, question, perspective="alice")
print(f"Measurement result: {result}")
```

## Installation

```bash
git clone https://github.com/entangledcode/qrl.git
cd qrl

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# Install QRL (editable mode for development)
pip install -e .
```

**Requirements:** Python 3.8+, NumPy, NetworkX

**Optional backends** (install as needed):
```bash
pip install pennylane            # PennyLane backend (Xanadu / simulation)
pip install perceval-quandela   # Photonic compilation + Quandela Cloud
pip install graphix              # Graph-state MBQC backend
```

## CLI

Installing QRL provides the `qrl` command. Run `qrl help` for the full usage guide.

```bash
# Run experiments
qrl run bell --shots 500 -v         # Bell/CHSH inequality test
qrl run ghz --qubits 4              # GHZ/Mermin inequality test
qrl run demo --quick                # Interactive physics demo

# Inspect compilation artifacts
qrl inspect graph bell              # Graph state topology
qrl inspect pattern ghz             # MBQC measurement pattern

# Compile to backends
qrl compile bell --target perceval  # Compile to Perceval circuit
qrl compile ghz --target graphix    # Compile to graphix pattern

# Cloud execution (requires QUANDELA_TOKEN)
qrl cloud status                    # Check platform availability
qrl cloud run bell                  # Run on Quandela sim:belenos

# Tools
qrl info                            # Version, dependencies, source stats
qrl shell                           # Interactive REPL
```

### Interactive REPL

`qrl shell` starts an interactive session with tab completion:

```
$ qrl shell
qrl> entangle mybell 2
Created 'mybell': 2-qubit bell relation
qrl> graph mybell
Graph for 'mybell':
  Nodes: [0, 1]
  Edges: [(0, 1)]
qrl> compile mybell
qrl> chsh --shots 500
S = 2.8200  (limit 2.0)  VIOLATED
qrl> quit
```

---

## Implementation Status

**~9,500 lines of code | 595 tests passing | Full photonic pipeline verified**

### Stage 0-3: Core Language & MBQC Compiler (Complete)

| Component | Status | Description |
|-----------|--------|-------------|
| QuantumRelation | ✅ | Entanglement as first-class citizen |
| n-qubit States | ✅ | GHZ states (tested to 5 qubits), W states, Bell pairs |
| Graph Extraction | ✅ | `extract_graph()` — Relations → cluster state topology |
| Pattern Generation | ✅ | Bell, GHZ, H/X/Z/S/T gates, CNOT, CZ, rotations |
| Adaptive Corrections | ✅ | Pauli corrections based on measurement outcomes |
| Teleportation | ✅ | Full protocol with fidelity = 1.0 |

### Stage 4: Photonic Integration (Complete)

QRL compiles to photonic platforms via [Perceval](https://github.com/Quandela/Perceval)/[Quandela Cloud](https://cloud.quandela.com/) and [PennyLane](https://pennylane.ai/).

| Component | Status | Description |
|-----------|--------|-------------|
| QRL → Perceval | ✅ | Direct path-encoded circuit generation |
| QRL → PennyLane | ✅ | Mid-circuit measurements + adaptive corrections |
| Local Simulation | ✅ | SLOS backend + PennyLane default.qubit |
| Cloud Connection | ✅ | Quandela `sim:belenos` verified |
| Full Pipeline | ✅ | QRL → MBQC → Backend → Results |

```
Pipeline: QRL Relations → MBQC Pattern ─┬─→ Perceval → Quandela
                                         └─→ PennyLane → Simulation
```

**Validated on hardware:** Bell state confirmed on `qpu:belenos` (Quandela's photonic QPU) — 423/1000 shots yielded valid dual-rail events (42.3% yield), 57.7% HOM-bunched as expected from linear optics.

### Stage 5: Domain Modules (Complete)

QRL includes domain-specific modules that apply relational quantum mechanics to real scientific problems.

#### `qrl.physics` — Foundational Layer

The physics primitives everything else builds on. Hardware-verified S = 2.61 ± 0.08 on `qpu:belenos`.

```python
from qrl.physics import BellTest, GHZTest

test = BellTest()
print(test.compare(trials=2000))
# S parameter: Theory 2.8284, Observed 2.8340 — VIOLATED
```

| Module | Description |
|--------|-------------|
| `qrl.physics.bell` | CHSH inequality, BellTest |
| `qrl.physics.ghz` | GHZ paradox, Mermin inequality |

#### `qrl.domains.biology` — Quantum Biology

```python
from qrl.domains.biology import fmo_complex, QuantumBioNetwork

fmo = fmo_complex()  # Fenna-Matthews-Olson complex
network = QuantumBioNetwork(fmo)
print(network.coherence_lifetime())
```

| Function | Description |
|----------|-------------|
| `fmo_complex()` | FMO photosynthesis complex (7-site) |
| `RadicalPair` | Avian magnetoreception model |
| `lindblad_evolve()` | Lindblad master equation evolution |
| `decoherence_rate()`, `coherence_lifetime()` | Environmental noise metrics |
| `phonon_bath()` | Vibrational environment coupling |
| `ENAQT` | Environment-Assisted Quantum Transport |

#### `qrl.domains.sensing` — Quantum Sensing

```python
from qrl.domains.sensing import QuantumSensor, heisenberg_limit, ramsey_interferometry

sensor = QuantumSensor(n_qubits=10)
print(f"Fisher info: {sensor.quantum_fisher_information():.4f}")
print(f"Heisenberg limit: {heisenberg_limit(10):.6f}")
print(f"Advantage: {sensor.quantum_advantage_factor():.2f}x")
```

| Function | Description |
|----------|-------------|
| `QuantumSensor` | Entanglement-enhanced sensor |
| `quantum_fisher_information()` | QFI for parameter estimation |
| `cramer_rao_bound()` | Quantum Cramér-Rao bound |
| `heisenberg_limit()` | 1/N scaling limit |
| `ramsey_interferometry()` | Ramsey protocol simulation |
| `mach_zehnder()` | Mach-Zehnder interferometer |
| `spin_squeezing()` | Spin-squeezed state sensing |
| `atomic_clock_stability()` | Allan deviation model |

#### `qrl.domains.chemistry` — Quantum Chemistry

```python
from qrl.domains.chemistry import hydrogen, MolecularSystem

mol = hydrogen(bond_length=0.74)
print(f"HF energy:  {mol.hf_energy:.4f} Ha")
print(f"FCI energy: {mol.fci_energy:.4f} Ha")
print(f"Correlation energy: {mol.correlation_energy:.4f} Ha")
```

| Function | Description |
|----------|-------------|
| `hydrogen()` | H₂ molecule (STO-3G, full VQE) |
| `helium_hydride_cation()` | HeH⁺ (first molecule in universe) |
| `MolecularSystem` | General molecular Hamiltonian |
| `jordan_wigner_hamiltonian()` | Fermion-to-qubit mapping |
| `vqe_energy()` | Variational Quantum Eigensolver |

H₂ benchmark: E_HF = −1.1167 Ha, E_FCI = −1.1373 Ha; entanglement reaches 2 bits at dissociation.

---

### Validation

```bash
python -m pytest tests/ -v
```

- **595 tests passing**
- **Bell correlations** verified (CHSH violation S = 2.83)
- **GHZ paradox** demonstrated (Mermin inequality M = 4, classical limit 2)
- **Teleportation fidelity = 1.0**
- **Photonic pipeline** validated locally and on `qpu:belenos`

---

## MBQC Compilation Pipeline

```python
from qrl import QRLProgram
from qrl.mbqc import (
    extract_graph,
    generate_pattern_from_relation,
    generate_teleportation_pattern,
    simulate_teleportation
)

# 1. Create quantum relation
program = QRLProgram("GHZ State")
qubits = [program.create_system() for _ in range(3)]
ghz = program.entangle(*qubits)

# 2. Extract graph state structure
graph = extract_graph(ghz)
# GHZ₃ → star graph (3 nodes, 2 edges)

# 3. Generate measurement pattern
pattern = generate_pattern_from_relation(ghz)

# 4. Teleportation with adaptive corrections
import numpy as np
input_state = np.array([0.6, 0.8])  # |ψ⟩ = 0.6|0⟩ + 0.8|1⟩
output, outcomes, corrections = simulate_teleportation(input_state)
# Fidelity = 1.0 (perfect teleportation)
```

---

## Key Features

### Relations-First Programming
```python
# Instead of gates, work with relationships
bell = program.entangle(qubit_a, qubit_b)  # Creates QuantumRelation
ghz = program.entangle(q0, q1, q2)          # 3-qubit GHZ state
```

### Contextual Measurement
```python
# Measurements are questions asked from a perspective
question = create_question(QuestionType.SPIN_X)  # X-basis measurement
result = program.ask(relation, question, perspective="alice")
```

### Automatic Graph Extraction
```python
# QRL automatically determines cluster state topology
graph = extract_graph(relation)
# Bell state → edge graph
# GHZ state → star graph
# W state → ring topology
```

### Adaptive Pauli Corrections
```python
# MBQC requires corrections based on measurement outcomes
pattern = generate_teleportation_pattern()
# Automatically includes X/Z corrections conditioned on Bell measurement results
```

---

## Project Structure

```
qrl/
├── src/qrl/
│   ├── cli.py               # CLI entry point (qrl command)
│   ├── core.py              # QuantumRelation, QuantumQuestion, Perspective
│   ├── measurement.py       # Measurement and basis transformations
│   ├── tensor_utils.py      # n-qubit tensor operations
│   ├── causal.py            # CPTPMap, QuantumSwitch, QuantumCausalDAG, do-calculus
│   ├── mbqc/                # MBQC compiler
│   │   ├── graph_extraction.py
│   │   ├── pattern_generation.py
│   │   ├── adaptive_corrections.py
│   │   └── measurement_pattern.py
│   ├── backends/            # Hardware backends
│   │   ├── pennylane_adapter.py
│   │   ├── perceval_path_adapter.py
│   │   └── graphix_adapter.py
│   ├── domains/             # Scientific domain modules
│   │   ├── biology.py       # FMO, RadicalPair, ENAQT
│   │   ├── sensing.py       # QuantumSensor, Ramsey, Fisher info
│   │   ├── chemistry.py     # H₂, HeH⁺, VQE, Jordan-Wigner
│   │   └── networks.py      # QuantumNetwork, repeaters
│   └── physics/             # Foundational layer
│       ├── bell.py          # CHSH inequality, BellTest
│       └── ghz.py           # GHZ paradox, Mermin inequality
├── tests/                   # 595 tests
├── examples/
│   ├── pennylane/
│   └── quandela/
├── docs/
└── requirements.txt
```

---

## Research Direction

### Core Thesis

QRL explores whether a **relations-first formalism**—where correlations are primitives, not derived properties—offers genuine advantages for:

1. **Understanding quantum physics:** Does describing correlations directly reveal structure that gate-centric approaches obscure?
2. **MBQC compilation:** Can relational specifications compile more naturally to measurement-based patterns?
3. **Photonic hardware:** Does the relational model align better with linear optical quantum computing?

### Theoretical Foundations

| Concept | Connection to QRL |
|---------|-------------------|
| **Relational QM** (Rovelli) | Properties exist only relative to other systems—QRL models this directly |
| **Bell's Theorem** | Correlations without local hidden variables—relations ARE the reality |
| **MBQC** (Raussendorf-Briegel) | Computation via measurements on entangled states—natural fit for relations |

#### Example: Bell Test in QRL

```python
from qrl.physics import BellTest

test = BellTest()
print(test.predict())
# -> Predicted CHSH parameter: S = 2.8284
# -> Classical limit: 2.0

print(test.compare(trials=2000))
# -> S parameter: Theory 2.8284, Observed 2.8340
# -> Violated: YES
```

#### Running the Interactive Demo

```bash
qrl run demo          # Full demo (~5 minutes)
qrl run demo --quick  # Quick mode (~1 minute)
qrl run demo --section 3  # GHZ paradox only
```

---

## Documentation

- **[Technical Blog](https://dcoldeira.github.io/)** — Development journey, deep dives, research notes
- **[Published Paper](https://doi.org/10.5281/zenodo.18292199)** — "QRL: A Relations-First Programming Language for Measurement-Based Quantum Computing" (Zenodo, January 2026)
- **[Photonic Examples](examples/quandela/)** — Working examples for Quandela Cloud integration
- **[PennyLane Examples](examples/pennylane/)** — Cross-platform MBQC via PennyLane

## Related

- **[Bell Platform](https://entangledcode.dev)** — Natural-language QRL interface (web app) built on top of this library
- **[Quantum Advantage Advisor](https://github.com/dcoldeira/quantum-advantage-advisor)** — Reality-check tool for quantum computing applicability

## Contributing

QRL is an active research project exploring relations-first quantum computing. Contributions welcome from researchers interested in:

- **Foundations of quantum mechanics** — Relational QM, Bell inequalities, contextuality
- **MBQC theory and compilation** — Measurement patterns, graph states, flow conditions
- **Photonic quantum computing** — Linear optics, path encoding, Perceval/Quandela
- **Quantum programming languages** — Type systems, compilation, formal verification

## Contact

**David Coldeira**
- Email: dcoldeira@gmail.com
- GitHub: [@dcoldeira](https://github.com/dcoldeira)
- Blog: [dcoldeira.github.io](https://dcoldeira.github.io)
- Platform: [entangledcode.dev](https://entangledcode.dev)

## License

MIT License — see [LICENSE](LICENSE)
