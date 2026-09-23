# Froe on QRL

*As of 2026-09-20.*

Plan and reference for using Froe as a coding agent on the QRL codebase,
across multiple sessions. Living doc — also tracked at
https://claude.ai/artifact/MjGExna4jUnwzvxgpyjtaF (source of truth for
comments/collaborative edits; sync changes back here periodically).

## Guardrails

An earlier unattended, unsupervised Froe run strayed outside its intended
scope. No lasting harm resulted, but it's the reason for the rules below.

**Non-negotiables going forward:**

1. **No unattended `-yolo` on QRL or Bell.** Every `froe do` run is
   interactive — a human approves each shell command — until Froe has a
   real track record on this codebase specifically.
2. **Use Mistral, not the local model, for real work.** `froe doctor` must
   show Mistral as `key set` before starting a session
   (`froe do -model mistral ...`). The local `bonsai-27b` fallback is what
   caused the incident; it stays a last resort, not the default.
3. **Scope every session with `-root`.** Point Froe at the specific
   subdirectory it needs (e.g. `qrl-public/`), never the whole `qrl/` tree
   — no SSH access, no server `.env` files, no reason to leave the repo
   it's working in.
4. **Never hand Froe the paper (`.tex`) files.** Finishing the paper is the
   current top priority (see `CLAUDE.md`) and not something to risk on an
   unproven agent. Revisit only after Froe has a track record on
   lower-stakes QRL work.
5. **Review every diff before `froe commit -p`.** Froe drafts commits; a
   human reads the diff before it pushes, same as any other change to
   these repos.

These guardrails loosen only as trust is earned — see the phased rollout
below.

## QRL reference

Repo: `qrl-public/` (public, `entangledcode/qrl`). Two ways to write QRL:
the Python library, and the surface-syntax `.qrl` language on top of it.

**Python library modules** (`src/qrl/`):

| Module | Purpose |
| --- | --- |
| `core.py` | Base quantum state / gate machinery |
| `causal.py` | `ProcessMatrix`, `CPTPMap`, `QuantumCausalDAG` — the causal-structure layer |
| `categorical.py` | `Process`, `z_spider`, `x_spider`, `cup`/`cap` — categorical QM (Coecke & Kissinger) |
| `measurement.py` | Measurement / observable machinery |
| `physics/` | Foundational: `bell.py` (CHSH), `ghz.py` (Mermin), `hensen.py` (loophole-free Bell), Araújo witness |
| `domains/` | Biology, sensing, chemistry — built on `physics/`, not the focus area |
| `mbqc/`, `backends/`, `compiler/` | Lower-level / hardware-facing, less relevant to language work |

**Surface language** (`src/qrl/lang/`, ~900 lines, 84 tests) — a small
typed language that compiles down to the Python library:

| File | Role |
| --- | --- |
| `lexer.py` / `tokens.py` | Tokenizer |
| `parser.py` / `ast.py` | Parser → AST |
| `types.py` / `typecheck.py` | Type system — 13 typing rules (T-Var, T-Qubit, T-Let, T-Entangle-N, T-Tensor, T-Seq, T-PM, T-DAG, T-Do, …), see `INTERP_PLAN.md` |
| `interp.py` | Interpreter — every term form executes (Sessions A–D, complete as of Sept 6, 2026) |
| `errors.py` | `QRLRuntimeError` etc. |

CLI: `qrl check\|parse\|exec FILE.qrl` (`qrl exec --shots N --seed S
--dist`). Example programs in `examples/lang/`: `bell.qrl`, `switch.qrl`,
`causal_dag.qrl`, `switch_rejected.qrl` — read these first, they're the
clearest "how do I write QRL" reference.

**Known constraints Froe should respect, not "fix":**
- Density matrices throughout (no statevector path) — deliberate, one code
  path.
- `DAG`/`Do` only support single-parent nodes; multi-parent raises a
  runtime error by design (E-Switch-Incoherent is deliberately unreachable
  from well-typed surface programs).
- `qrl.physics` vs `qrl.domains` asymmetry is intentional — physics is the
  foundational layer domains build on.

**Dev environment note:** this machine has no pytest/networkx/pip, so
`import qrl` fails here — `qrl.lang` was tested via a standalone harness
during development. Froe should check what's actually runnable before
assuming `pytest` works.

**Docs to read before writing QRL code:** `LANG_PLAN.md`, `INTERP_PLAN.md`
(surface language + interpreter design, phased history),
`docs/surface-syntax.md`.

## Phased rollout

Froe's autonomy on QRL grows in steps, gated on a track record, not a
calendar date.

| Phase | Scope | Autonomy | Exit criteria |
| --- | --- | --- | --- |
| **0 — Setup** (done Sept 20) | Environment only | — | Mistral key live for Froe (`froe doctor` shows `key set`), guardrails written down |
| **1 — Supervised, mechanical** | Small, contained QRL tasks (below) | Interactive `froe do -model mistral`, human approves every command, reviews every diff | 3–5 tasks landed cleanly, no surprising/off-script behavior |
| **2 — Supervised, larger** | Multi-file features, new tests, a QRL Book chapter | Same interactive mode, longer sessions | Track record holds on bigger diffs; Froe's commit messages and code style need little correction |
| **3 — Semi-autonomous** | Well-scoped `-root`-limited tasks | `-accept-edits` (auto-approve file edits, still prompts for shell) considered; `-yolo` still off | Explicit re-evaluation — not automatic |
| **Never (for now)** | The paper (`.tex`), server/deploy access, unattended cron | — | Revisit only if Phase 3 is solid and the user explicitly reopens it |

Current status: **Phase 1, not yet started.** Next session should pick one
task from the backlog below and run it interactively.

## Task backlog

| Task | Phase | Source | Notes |
| --- | --- | --- | --- |
| Runtime error messages with carets, extended to phases B/C (Cptp/Seq/Switch, PM/DAG/Do) | 1 | `INTERP_PLAN.md` D-polish | Session A already has this; mechanical extension |
| Update `docs/surface-syntax.md` "Not yet covered" section | 1 | `INTERP_PLAN.md` D-polish | Doc-only, zero code risk |
| Property tests: well-typed-doesn't-get-stuck (fuzz, not proof) | 1–2 | `INTERP_PLAN.md` Session E (optional) | Contained to `tests/` |
| Property tests: round-trip `parse → check → run` on all `examples/lang/` | 1–2 | `INTERP_PLAN.md` Session E (optional) | Contained to `tests/` |
| QRL Book Ch 15 — QuantumSwitch Soundness (still placeholder) | 2 | `project_qrl_book` memory | Larger, needs the Soundness Theorem explained correctly — review closely |
| Additional tests for existing `qrl.lang` modules | 1 | This session | Good first task — no new surface area, just coverage |
| **QRL as a framework for physics** — explore using QRL's causal-structure/relational-correlation primitives to model physical systems directly (not just language-internals work), leaning into what makes QRL different from circuit-first SDKs | 2–3 | User, this session | Open-ended and exploratory by nature — bigger and harder to review at a glance than the other backlog items, so needs a track record first. No concrete task shape yet; first step next session is probably scoping *what* to model, not writing code |

Pick the next task here before starting a session; update this row (or add
a Session log entry) with what actually happened.

## Session log

| Date | What happened |
| --- | --- |
| 2026-09-20 | Doc created. Phase 0 done: Mistral key rotated (after a leaked-secret incident during a Bell unattended-cron test, see Guardrails) and confirmed live for Froe. Phase 1 not yet started — no QRL task run yet. |
| 2026-09-23 | Phase 1 started. **Task 1/5** — verify `docs/surface-syntax.md` "Not yet covered" section against code (`codestral-latest`). Ran clean, read-only, correctly found the section already accurate — no diff, nothing to commit. **Task 2/5** — write `tests/test_lang_errors.py` for `src/qrl/lang/errors.py` (same model). Two findings: (1) driving `froe do` through Claude's own Bash tool gives Froe no TTY to prompt for write approval — the `write_file` call was declined, and Froe then falsely reported "Created tests/test_lang_errors.py" anyway (file didn't exist; environment-specific, not reproduced when run directly in a real terminal). (2) `codestral-latest` computed the caret-offset assertion in `_render()` wrong **twice in a row**, even when explicitly told to compute (not guess) from the gutter format — got 6 spaces both times, actual is 9. Verified by hand-executing `errors.py`, fixed the one-line assertion, ran all 9 tests directly against the module (all pass), then committed via `froe commit -a -m "test: add unit tests for qrl.lang error types" -p` → `bc1d348`, pushed, attribution clean (`David Coldeira <dcoldeira@gmail.com>`, no co-author lines). **Takeaway: always run `froe do` in a real terminal (not proxied through another agent's shell tool), and hand-verify any Froe-generated assertion involving computed offsets/positions before committing.** 2 of 5 Phase 1 tasks done (1 clean, 1 needed a human fix) — continue with backlog next session. |
