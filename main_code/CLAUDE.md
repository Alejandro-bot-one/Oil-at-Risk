# CLAUDE.md — Oil-at-Risk TFM

Project context for Claude. **At the start of every session, read the `context/` folder before
doing any work.** These files are the curated memory of this project; keep them current.

## Read these first (in `main_code/context/`)

1. **`context/architecture.md`** — how the code is structured: Python backend (`auxi/` modules +
   `diagnostics/` subpackage) and Jupyter frontend (notebooks), with the data flow and
   dependency graph.
2. **`context/conventions.md`** — code style, naming, the layered `compute_*`/`plot_*`/
   orchestrator pattern, statistical conventions, tests. Academic project: choices are tied to
   papers in `../references/`.
3. **`context/known_errors.md`** — mistakes already made (lookahead, stale hardcoded paths,
   quantile crossing, misaligned `dropna`, …) and how to avoid repeating them.
4. **`context/decisions.md`** — settled design decisions, so we don't rebuild or relitigate
   them. Includes what is explicitly out of scope (YAGNI).
5. **`context/workflow.md`** — the change process: spec → plan → TDD (red/green) → commit per
   task → git tags → notebook smoke test → verification.
6. **`context/glossary.md`** — how Alejandro communicates, plus the domain/data/code vocabulary.

## Standing rules

- Address the user as **Alejandro** in every reply.
- Don't answer unless highly confident; think twice, look for flaws, corroborate with more than
  one source; say so when you don't know.
- The **code is ground truth**; specs in `docs/superpowers/` explain intent and may be stale.
- Other instructions live in `.claude/` (e.g. `settings.local.json`).

## Keep context fresh

After any non-trivial change, update the relevant `context/` file in the same session
(new module → architecture; new pattern → conventions; new bug → known_errors; new design call
→ decisions; new term → glossary). A context file that lies is worse than none — re-verify
against the code when in doubt.
