# QRL Surface Syntax + Static Type Checker

**Goal:** turn QRL from "Python API with a formal model" into "a language with a
parser and a static type checker that enforces the 13 typing rules from
`papers/qpl-2026/quantum-causal-structure.tex`."

**Why:** QPL 2026 reviewers B and C said the paper "reads as early-stage" and the
Limitations section concedes "QRL is currently a Python API, not a language with
a parser." A concrete syntax + checker is the single highest-leverage change for
the 2027 resubmission, and becomes a new contribution to name in the abstract.

**Scope discipline:** syntax + parser + checker ONLY. No metatheory proofs, no
NCPSU layer, no n-party ICO, no hardware compilation of causal types. Those are
separate papers and would slip the resubmission.

## Design

New package: `src/qrl/lang/`

| File | Contents |
|------|----------|
| `tokens.py`   | `Token`, `TokenKind`, source positions |
| `lexer.py`    | hand-rolled tokenizer (no external dep — matches project ethos) |
| `ast.py`      | term + type AST nodes (dataclasses), 1:1 with paper BNF |
| `parser.py`   | recursive-descent parser → AST |
| `types.py`    | semantic type objects + subtyping (`UniProc(d) <= Proc(d,d)`) |
| `typecheck.py`| the 13 rules, `Gamma |- t : tau` |
| `errors.py`   | `QRLSyntaxError`, `QRLTypeError` with carets |
| `interp.py`   | (later) lower checked AST onto existing `qrl.core` / `qrl.causal` |

### Concrete syntax (aligned with Definitions 1 & 2)

```
-- Bell pair, then measure
let a = |0> in
let b = |0> in
ask(entangle(a, b), Z)

-- Quantum switch (causal stratum)
let U = cptp([[0, 1], [1, 0]]) in
let V = cptp([[1, 0], [0, -1]]) in
switch(U, V, |+>)
```

- comments: `--` to end of line
- qubit literals: `|0>` `|1>` `|+>`
- terms: `entangle(...)`, `ask(t, q)`, `let x = t in t`, `cptp(M, ...)`,
  `t ; t` (seq), `t * t` (tensor), `switch(f, g, c)`, `pm(W, p)`,
  `dag(V, E, Phi)`, `do(g, x, rho)`
- types (for optional annotations `let x : T = ...`): `Qubit`, `Rel(n)`,
  `Ques`, `Outcome`, `T * T`, `Proc(d1, d2)`, `UniProc(d)`, `PM(n, d)`,
  `Switch(d)`, `CausalDAG(n)`
- matrix literal: `[[a, b], [c, d]]`, entries are real or complex
  (`1`, `-0.5`, `0.7071`, `1j`, `0.5+0.5j`)
- observables: bare identifiers `Z`, `X`, ... typed as `Ques`

### The 13 rules (from the paper)

Basic:      T-Var, T-Qubit, T-Let
Relational: T-Entangle-2, T-Entangle-N, T-Ask, T-Tensor
Causal:     T-UniProc, T-Switch, T-Seq, T-PM, T-DAG, T-Do

Key checks:
- T-UniProc: parsed matrix must satisfy `U† U = I` (numeric, tol 1e-9)
- T-Switch: both channel args must have type `UniProc(d)` (NOT `Proc`) — this is
  the rule that makes the Soundness Theorem bite. `cptp` of a non-unitary matrix
  types as `Proc(d,d)` and `switch` on it is a type error.
- T-PM: `W >= 0` and `Tr_out(W) = I_in`
- T-DAG: `(V, E)` acyclic

## CLI

- `qrl check FILE.qrl`   — parse + typecheck, print `t : tau` or errors
- `qrl parse FILE.qrl`   — dump AST
- extend `qrl run` later once `interp.py` lands

## Status

- [x] tokens + lexer  (`src/qrl/lang/tokens.py`, `lexer.py`)
- [x] ast  (`src/qrl/lang/ast.py`)
- [x] parser  (`src/qrl/lang/parser.py`) — recursive descent, no external dep
- [x] types + subtyping  (`src/qrl/lang/types.py`) — `UniProc(d) <: Proc(d,d)`
- [x] typecheck (13 rules)  (`src/qrl/lang/typecheck.py`)
- [x] errors with source carets  (`src/qrl/lang/errors.py`)
- [x] tests — `tests/test_lang_lexer.py`, `_parser.py`, `_typecheck.py` (56 tests, all pass)
- [x] CLI `check` / `parse`  (`src/qrl/cli.py`)
- [x] example programs — `examples/lang/{bell,switch,switch_rejected}.qrl`
- [x] grammar reference — `docs/surface-syntax.md`
- [ ] paper: update Limitations + add a syntax figure + contributions bullet
      — DRAFTED, awaiting go-ahead (reopens "paper is done" from May 12)
- [ ] interp.py (lowering onto `qrl.core` / `qrl.causal`) — stretch, not required
      for resubmission

## Notes for the paper edit (when ready)

- §6 "Surface syntax" paragraph: change from "future engineering work" to
  "QRL now has a concrete surface syntax and a static type checker enforcing all
  thirteen rules (§2.2); see `docs/surface-syntax.md`. It does not yet lower to
  execution."
- §1 contributions: add a bullet — "A concrete surface syntax and a static type
  checker (~900 lines, 56 tests) that enforces the thirteen typing rules,
  including the T-UniProc / T-Switch separation on which Theorem 1 rests."
- §7 conclusion: drop "a concrete surface syntax with a type-checker" from the
  remaining-work list; keep metatheory + NCPSU.
- Consider a `figure` with `examples/lang/switch.qrl` beside the rejected
  `switch_rejected.qrl` and its type error — shows the theorem biting.
