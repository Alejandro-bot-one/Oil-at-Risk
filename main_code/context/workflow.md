# Workflow — Oil-at-Risk TFM

> Context file for Claude. The process to follow when changing or extending this codebase.
> It is the process the project has actually used (spec → plan → TDD → commit-per-task →
> notebook smoke test), reconstructed from `docs/superpowers/` and the git history.

## Project working rules (from Alejandro)

- Address Alejandro by name in every reply.
- Don't answer if you don't know; only answer when highly confident. Think twice and actively
  look for flaws in your own answer before giving it.
- Don't trust a single source — corroborate with at least a second (a second paper, the code
  itself, a test, or a reference in `references/`).
- The **code is ground truth**; specs and plans explain intent and may be stale.

## The change lifecycle

For anything beyond a trivial one-liner, follow these stages in order. Small fixes may collapse
stages 1–2, but never skip the test and verification stages.

### 1. Design spec (the "why" and "what")

Write or update a short spec in `docs/superpowers/specs/` named
`YYYY-MM-DD-<topic>-design.md`. It states context/motivation, the **explicit decisions** taken,
the final architecture (function inventory tables, a small mermaid diagram if useful), known
limitations, and an explicit **out-of-scope / YAGNI** list. Get agreement before coding. Mirror
the structure of the existing specs (caviar-indicator, backend-reorg, rolling-window-entropy).

### 2. Implementation plan (the "how", task by task)

Write a plan in `docs/superpowers/plans/` named `YYYY-MM-DD-<topic>.md`, broken into numbered
**tasks**, each with `- [ ]` checkbox steps. Every task lists the files it creates/modifies/
deletes, the exact test to write, the command to run, and the expected output (red, then green).
End with a final **verification** task. This is the pattern both existing plans follow.

### 3. TDD: red → green, one unit at a time

1. Write the failing test first (in `tests/`, reusing the `synthetic_panel` fixture from
   `conftest.py`). Run it and **confirm it fails for the expected reason** (`ImportError`,
   `AttributeError`, wrong value).
2. Write the minimal code to pass it. Run the test; confirm green.
3. Run the **existing** suite as a regression check (`python -m pytest tests/ -v`) before moving
   on — adding to `qreg`/`caviar` must not break their consumers.

Test the numeric core with value asserts; test plotters with an `Agg`-backend smoke test that
asserts no exception and the right return object. Always guard the standing invariants: input
panel not mutated, no lookahead, public surface resolves, degenerate inputs raise.

### 4. Commit per task

One focused commit per completed task, using the project's message convention:

```
feat: …        new capability
fix: …         bug fix
refactor: …    move/restructure, no behaviour change
test: …        tests only
docs: …        documentation / context files
chore: …       tooling, gitignore, cleanup
```

(See `git log` for live examples.) For a multi-task refactor, **tag the pre- and post- states**
(`git tag pre-<topic>-YYYY-MM-DD`) so `git reset --hard <tag>` is a clean rollback. Verify
`git status` is clean and `__pycache__` / `.ipynb_checkpoints` / `.pytest_cache` are gitignored
before committing.

### 5. Propagate to the notebooks (frontend integration)

If a public symbol moved, was renamed, or was deleted, update every call site:

1. Grep the notebooks **and** `auxi/` for the old name (`import auxi.X`, `X.func`).
2. Rewrite each call. If a function was removed, either delete the cell or comment it with a
   dated `# DEPRECATED YYYY-MM-DD: <name> removed in <change>` note explaining the replacement.
3. **Restart Kernel → Run All** on every affected notebook. A notebook that runs top-to-bottom
   without error is the integration test. Expect only intentionally-deprecated cells to no-op.

### 6. Final verification

- Smoke-import every module/subpackage in one line and confirm `OK`.
- Re-check the public surface with `dir()` against the pre-change snapshot: nothing intended to
  survive should have disappeared; nothing intended to be deleted should remain.
- Full `pytest` green.
- For statistical changes, sanity-check the numbers (e.g. boosted vs. readable risk metrics
  match to ~1e-11; OOS results respect the cutoff; bounds don't cross en masse).
- For high-stakes correctness (anything touching lookahead, OOS, or a thesis result),
  double-check independently — re-derive the cutoff logic, or have a subagent review the diff.

## When NOT to follow the full ceremony

A typo, a docstring tweak, a one-line bugfix with an obvious test, or editing these context
files doesn't need a spec + plan. Use judgment: the heavier the change (new estimator, refactor
across modules, anything affecting a reported result), the more of stages 1–6 apply.

## Keeping context current

These `context/` files are living documents. After a non-trivial change, update the relevant
one in the same session: new module → `architecture.md`; new pattern → `conventions.md`; new
bug → `known_errors.md`; new design call → `decisions.md`; new vocabulary → `glossary.md`. A
context file that lies is worse than none — when in doubt, re-verify against the code.
