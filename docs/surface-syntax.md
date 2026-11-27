# QRL surface syntax

A concrete syntax for QRL-Core, the calculus of
`papers/qpl-2026/quantum-causal-structure.tex`. The parser and the static type
checker live in `src/qrl/lang/`.

```
qrl check FILE.qrl              # parse + type-check, print  ok  FILE : <type>
qrl parse FILE.qrl              # dump the AST
qrl exec  FILE.qrl [--shots N] [--seed S] [--dist]   # type-check then run
```

or programmatically:

```python
from qrl.lang import check_source, run, distribution
check_source("ask(entangle(|0>, |1>), Z)")     # -> (Outcome * Rel(2))
run("ask(entangle(|0>, |1>), Z)", seed=0)      # -> (Outcome(1), QState(n=2))
distribution("ask(entangle(|0>, |1>), Z)")     # -> {0: 0.5, 1: 0.5}  (exact)
```

`run` samples the probabilistic `ask` rule (`--seed` / `shots=` for
reproducibility); `distribution` computes exact outcome probabilities and is
defined when the program's result comes from a top-level `ask`.

## Grammar

```
program    ::= term

term       ::= "let" ident [":" type] "=" term "in" term
             | seq
seq        ::= tensor (";" tensor)*                  -- sequential composition
tensor     ::= atom ("*" atom)*                      -- tensor product
atom       ::= ket
             | ident                                  -- variable / observable
             | "(" term ")"
             | "entangle" "(" term ("," term)+ ")"
             | "ask" "(" term "," term ")"
             | "cptp" "(" matrix ("," matrix)* ")"
             | "switch" "(" term "," term "," term ")"
             | "pm" "(" matrix "," numlist ")"
             | "dag" "(" identlist "," edgelist "," philist ")"
             | "do" "(" term "," ident "," matrix ")"

ket        ::= "|0>" | "|1>" | "|+>"
matrix     ::= "[" [ numlist ("," numlist)* ] "]"     -- list of rows
numlist    ::= "[" [ number ("," number)* ] "]"
identlist  ::= "[" [ ident ("," ident)* ] "]"
edgelist   ::= "[" [ "[" ident "," ident "]" (",""[" ident "," ident "]")* ] "]"
philist    ::= "[" [ "[" ident "," ident "," matrix "]" ... ] "]"
number     ::= real | complex                         -- 1  -0.5  0.7071  1j  0.5+0.5j

type       ::= typeatom ("*" typeatom)*
typeatom   ::= "Qubit" | "Ques" | "Outcome"
             | "Rel" "(" n ")"       | "Switch" "(" d ")"
             | "UniProc" "(" d ")"   | "CausalDAG" "(" n ")"
             | "Proc" "(" d "," d ")" | "PM" "(" n "," d ")"
             | "(" type ")"

comment    ::= "--" ... end-of-line
```

## Typing (the thirteen rules)

| Rule | Form | Result |
|------|------|--------|
| T-Var        | `x`                          | `Γ(x)` |
| T-Qubit      | `\|0>` `\|1>` `\|+>`         | `Qubit` |
| T-Let        | `let x = t1 in t2`           | type of `t2` |
| T-Entangle-2 | `entangle(a, b)`             | `Rel(2)` |
| T-Entangle-N | `entangle(a, ..., n)` (n≥3) | `Rel(n)` |
| T-Ask        | `ask(t, q)`                  | `Outcome * Rel(n)` |
| T-Tensor     | `t1 * t2`                    | `τ1 * τ2` |
| T-UniProc    | `cptp(U)`, `U†U = I`         | `UniProc(d)` |
| T-Switch     | `switch(f, g, c)`, `f,g : Proc(d, d)`, `c : Qubit` | `Switch(d)` |
| T-Seq        | `t1 ; t2`                    | `Proc(d1, d3)` |
| T-PM         | `pm(W, p)`, `W ≥ 0`, `Tr[W] = dⁿ`, `L_V(W) = W` | `PM(n, d)` |
| T-Refine     | `let x : C(d) = pm(W, p)`, a certificate for W ∈ C checks | `C(d)`, C ∈ {QCFO, QCCC, QCQC, Noncausal} |
| T-DAG        | `dag(V, E, Φ)`, `(V,E)` acyclic | `CausalDAG(\|V\|)` |
| T-Do         | `do(g, x, ρ)`, `x ∈ V(g)`    | `Outcome` |

Subtyping: `UniProc(d) <: Proc(d, d)` (a unitary channel is a channel);
`QCFO(d) <: QCCC(d) <: QCQC(d)` (causal classes of two-slot processes with a
global past and future, Wechs et al. 2021; certificates convert upward);
`Switch(d) <: QCQC(d)` (closed-form certificate); tensor is covariant. The
relation is transitive. Class types are **not** subtypes of `PM(2, d)`: a
`PM(2, d)` value has trivial past and future (16×16 at d = 2), while the switch
is a four-party process (P, A, B, F; 64×64 at d = 2). A class value comes from
`PM(2, d)` only through T-Refine.

T-Refine finds and checks a certificate (`qrl.causal_classes`). QCFO is decided
directly; a fixed-order certificate converts up to QCCC and QCQC without a
solver; otherwise QCCC and QCQC need an SDP solver (cvxpy). Noncausal (d = 2)
is certified by a violation of the OCB causal inequality, which also refutes
QCCC and QCQC without a solver. Examples: `examples/lang/classes.qrl`,
`examples/lang/noncausal.qrl`. Nothing coerces *into* `Switch(d)` — that is what makes Theorem 1
(Switch Soundness) enforceable at the type level.

### Why `cptp` splits into two rules

`cptp(U)` with a single **unitary** Kraus operator types as `UniProc(d)`.
`cptp(K1, ..., Km)` forming a trace-preserving map types as `Proc(d, d)`.
A non-trace-preserving argument is a type error.

`switch` accepts any channels of type `Proc(d, d)`. `Switch(d)` denotes the
switch construct (parties P, A, B, F), which does not depend on the channels
plugged into it, so a switch over a non-unitary channel is still certified
(`examples/lang/switch_channels.qrl`). Running it uses the Kraus form of the
switch; coherence between the orders survives in general (Ebler, Salek,
Chiribella 2018). `examples/lang/pm_loop.qrl` is the canonical rejected
program: a causal loop fails T-PM.

## Numeric side-conditions

Checked with numpy at tolerance 1e-6: `U†U = I` (T-UniProc); for T-PM,
Hermiticity, positive-semidefiniteness, `Tr[W] = dⁿ` and the no-causal-loop
condition `L_V(W) = W` (Araújo et al. 2015, Sec. II B), via
`qrl.causal.ProcessMatrix.satisfies_no_loop_condition`; density-matrix
conditions (T-Do).

## Execution status

`interp.py` runs **every term form**, carrying state as density matrices:

- **basic + relational** — `|k>`, `let`, `entangle` (E-Bell / E-GHZ, arguments
  discarded), `ask` (E-Ask; measures **subsystem 0** in the named basis `Z`/`X`/`Y`,
  returns `(outcome, post-measurement relation)`), `*` on states.
- **unitary causal** — `cptp` (unitary → unitary `Channel`; Kraus list → CPTP
  `Channel`), `;` (channel composition, `t1 ; t2` applies `t1` first),
  `switch(f, g, c)` (E-Switch-Coherent → a `Process` wrapping the quantum-switch
  process matrix, with `.p_win` and `.robustness`).
- **`pm`** → `Process` from a raw `W`.
- **`dag`** → `Dag` (`qrl.causal.QuantumCausalDAG`). Node dimensions are inferred
  from the Φ matrices. **Single-parent nodes only** — a multi-parent node raises a
  runtime error (use the Python API for joint mechanisms).
- **`do(g, x, ρ)`** → intervene `x := ρ`, propagate through the DAG, sample a
  Z-basis outcome on the unique sink node.

`switch(f, g, c)` evaluates to the switch construct with its channels and
control (rule E-Switch); the construct's process matrix does not depend on them.

## Not yet covered

- No module system / top-level definitions — one `term` per file.
- `pm` validity is the trace-normalisation check, not the full projector identity.
- `ask` has no subsystem index in the surface syntax — it always measures
  qubit 0.
- `dag` interpretation handles single-parent nodes only.
