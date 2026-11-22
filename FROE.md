# Working on QRL

QRL (Quantum Relational Language) is a Python library for quantum physics that
starts from relations and causal structure rather than circuits.

## Running code

Always use the project venv. System `python3` cannot import `qrl`.

- Run a script: `.venv/bin/python path/to/script.py`
- Quick check: `.venv/bin/python -c "from qrl.physics.bell import theoretical_chsh; print(theoretical_chsh())"`
- Tests for one file: `.venv/bin/python -m pytest -q tests/test_bell_physics.py`
- Whole suite (about 20s): `.venv/bin/python -m pytest -q tests`

## Where things are (`src/qrl/`)

- `physics/`: `bell.py` (CHSH, `chsh_noisy`), `ghz.py` (GHZ, Mermin), `hensen.py` (loophole-free Bell)
- `causal.py`: `ProcessMatrix`, `CPTPMap`, `QuantumSwitch`, `QuantumCausalDAG`, `causal_nonseparability_witness` (Araujo, generic)
- `switch_witness.py`: the quantum switch as a four-party process (`switch_process(d)`, parties P, A, B, F) and its
  lab witness (`switch_witness()`, value -0.1694 on the switch, >= 0 on causally separable processes).
  Use this, not `causal_nonseparability_witness`, for anything about certifying the switch.
- `categorical.py`: `Process`, `z_spider`, `x_spider`, `cup`, `cap`
- `core.py`, `measurement.py`: states and measurement
- `domains/`: biology, sensing, chemistry, built on `physics/`
- `lang/`: the `.qrl` surface language. Examples are in `examples/lang/`.

## Rules

- States are density matrices throughout. Do not add a statevector path.
- Check physics against a known value (an analytic result or a published
  number) and print both. A number with no reference is not a result.
- Put new physics scripts in `examples/physics/` (create it if missing), and
  new tests in `tests/`.
- Never read or edit `papers/`, `archive/` or `vision/`.
- Do not "fix" these; they are deliberate: `DAG` and `Do` take single-parent
  nodes only, and `physics/` vs `domains/` are separate on purpose.
