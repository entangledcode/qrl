# QRL surface language — interpreter (`interp.py`)

**Goal:** lower a type-checked surface AST to execution, following the small-step
operational semantics of `papers/qpl-2026/quantum-causal-structure.tex` §2.3
(E-Bell, E-GHZ, E-Ask, E-Switch-Coherent, E-Switch-Incoherent). After this,
`qrl run FILE.qrl` executes surface programs directly — the last limitation in
§6 of the paper ("checked surface programs are not yet lowered to execution")
goes away.

**Not in scope:** new physics. The interpreter is glue: it evaluates AST nodes
by calling the existing `qrl.core` / `qrl.causal` implementations. If a behaviour
isn't already in those modules, it's a separate task, not part of this.

## Design

New file `src/qrl/lang/interp.py`. Entry points on `qrl.lang`:

```python
run(src, *, seed=None, shots=1)   -> Value | list[Value]   # samples E-Ask
distribution(src)                 -> dict[outcome, prob]     # exact, no sampling
```

### Runtime values (paper §2.3: "density matrices ρ, naturals n, CPTP maps Λ,
### process matrices W, pairs (v1, v2)")

| `Value` variant | payload | produced by |
|---|---|---|
| `QState`   | density matrix `np.ndarray` (n-qubit) | `|k>`, `entangle`, `ask` post-state, `switch` |
| `Outcome`  | `int` | `ask`, `do` |
| `Channel`  | `qrl.causal.CPTPMap` | `cptp`, `;` |
| `Process`  | `qrl.causal.ProcessMatrix` | `pm`, `switch` (as process) |
| `Dag`      | `qrl.causal.QuantumCausalDAG` | `dag` |
| `Pair`     | `(Value, Value)` | `ask` (`Outcome * Rel(n)`), `*` |

### AST node -> evaluation

| node | rule | implementation |
|---|---|---|
| `Ket`      | T-Qubit    | `QState(|0><0|)` etc. |
| `Var`      | —          | env lookup |
| `Let`      | —          | eval value, bind, eval body |
| `Entangle` | E-Bell / E-GHZ | **construct the canonical state directly, ignoring the argument values** — this is the paper's central design choice. `(|0..0> + |1..1>)/√2`. |
| `Ask`      | E-Ask      | Born rule against the named observable's spectral projectors; sample with the run's RNG; return `Pair(Outcome, QState(post))`. `distribution()` returns the exact `p_m`. Reuse `qrl.core` measurement helpers where they fit. |
| `Tensor`   | T-Tensor   | Kron for `QState`; `CPTPMap`/`ProcessMatrix` tensor helpers |
| `Cptp`     | —          | `cptp_from_unitary(U)` (1 unitary Kraus) or `CPTPMap(kraus)` |
| `Seq`      | —          | `CPTPMap.compose` (note order: `t1 ; t2` = apply t1 then t2) |
| `Switch`   | E-Switch-Coherent / -Incoherent | `QuantumSwitch(channel_A=f, channel_B=g)`; control state from `c`; `.apply(...)`. Coherent iff both channels unitary (matches type: a `Switch(d)`-typed term always hits the coherent path; the incoherent path is only reachable for API-level `PM(2,d)` terms). |
| `PM`       | —          | `ProcessMatrix(W, parties=['A','B'], ...)` — derive dims from `n, d` computed by the checker |
| `DAG`      | —          | `QuantumCausalDAG`; `add_node` for each vertex (root vs internal from in-degree), `add_channel` for each Φ edge |
| `Do`       | T-Do       | `dag.do(var, rho)` then reduce to an `Outcome` (interventional_state -> sample) |
| `MatrixLit`| —          | not a term; only consumed by `Cptp`/`PM`/`DAG`/`Do` |

### Observables

`ask`'s question is a `Var` naming an observable. Seed the interpreter's
observable table with `Z, X, Y` (Pauli) and `H, S, T` -> spectral decomposition.
Same set the type checker seeds as `Ques`. Single-qubit only for v1; a partial
measurement on subsystem k of an `Rel(n)` state measures qubit k (paper: "only
subsystem k collapses").

### Runtime errors

The checker catches dimension mismatches for `;`, `switch`, `entangle`. What's
left for runtime: an observable name with no spectral table entry, a `pm`/`dag`
whose numeric data the checker accepted but `qrl.causal` rejects at construction
(e.g. full Araújo projector validity). Wrap these as `QRLRuntimeError` with the
source span.

## Phasing (one session each unless noted)

- **A — basic + relational. DONE (Sept 6 2026).** `interp.py`: `Value` types
  (density-matrix `QState`, `Outcome`, `Pair`), seeded RNG, `Ket`/`Var`/`Let`/
  `Entangle` (E-Bell/E-GHZ, args discarded)/`Ask` (E-Ask, measures subsystem 0)/
  `Tensor` on states. `run(src, seed=, shots=)` + `distribution(src)` (peels
  outer `let`s, exact probs). CLI: `qrl exec FILE.qrl [--shots N] [--seed S]
  [--dist]` — note: `exec`, not `run`, because `qrl run` is the experiment
  runner (bell/ghz/demo). `examples/lang/bell.qrl` executes. 17 tests in
  `tests/test_lang_interp.py` (73 lang tests total, all pass under pytest).
- **B — causal, unitary path. DONE (Sept 6 2026).** `Cptp` (→ `Channel`, unitary
  or CPTP), `Seq` (`t1 ; t2` = `t2.compose(t1)`), `Switch` (E-Switch-Coherent →
  `Process` wrapping `QuantumSwitch.process_matrix()`, with `.p_win` /
  `.robustness` accessors). `examples/lang/switch.qrl` executes:
  `Process(2x2, P_win=0.8536, robustness=0.4142)`. Differential test vs the
  Python API (`v.pm.W == qs.process_matrix().W`, p_win match). 6 new tests
  (79 lang tests total).
  NOTE for session D: the paper's §3 listing calls `sw.causal_game_advantage()`
  and `W.causal_nonseparability_robustness()` — the first method name does not
  exist (`causal_inequality_value()` does). Fix the listing in session D.
- **C — remaining causal. DONE (Sept 6 2026).** `PM` (→ `Process` from raw W,
  parties A..; dims [d]*n), `DAG` (→ `Dag`; node dims inferred from Φ matrices;
  **single-parent nodes only**, multi-parent raises a runtime error), `Do`
  (intervene, propagate, sample Z-basis outcome on the unique sink).
  `E-Switch-Incoherent` deliberately skipped — unreachable from well-typed
  surface programs. `examples/lang/causal_dag.qrl` added. 6 new tests
  (84 lang tests total). Every term form now evaluates.
- **D — polish + paper.** `qrl exec -` from stdin already works. Runtime error
  messages with carets (done for session A; extend to B/C). Update `docs/surface-syntax.md`
  ("Not yet covered" section shrinks). Paper edits: §6 remove the "execution of
  surface programs" limitation; §2.5 add one sentence that `qrl run` executes the
  operational semantics; §7 conclusion. Recompile.
- **E — optional, property tests.** Well-typed-doesn't-get-stuck (progress, as a
  fuzz test not a proof); round-trip `parse -> check -> run` on all `examples/lang/`.

## Decisions (David, Sept 6 2026)

1. **`qrl run` samples** — one outcome, or N with `--shots`, `--seed` for
   reproducibility (mirrors `qrl run bell`). `distribution(src)` is a
   library-only function returning exact `p_m`.
2. **Density matrix throughout.** `QState` wraps a `2^n × 2^n` ρ. One code path;
   no statevector↔ρ conversions. `switch`/`pm`/`do` consume ρ directly.

## Minor, decide during session A

- `qrl run` output format: type line + summary. For a `QState`, recognise and
  name common states (Bell, GHZ, |k>), else print a rounded ρ. TBD.
- `distribution()` scope: only defined for results of type `Outcome` or
  `Outcome * Rel(n)`; raise a clear error otherwise.
