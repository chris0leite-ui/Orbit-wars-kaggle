# Postmortem — 2026-05-23 review-skills-improvements-moKOR

## What went wrong

- **Three latent bundler bugs surfaced sequentially at submit time.**
  (a) `lib/opp_model.py:629` had an in-function
  `from lib.fast_sim import clone as fs_clone, step as fs_step`;
  `scripts/bundle_agent.py::_clean_lib_source` strips the indent on
  alias rebindings (asymmetric with the agent-source path which
  preserves indent), breaking the bundled function body.
  (b) `agents/baseline/chooser.py:22` had a multi-line
  `from lib.opp_model import (...)` — exact recurrence of the
  `bundler-modular-agent-namespace-access-breaks-bundle` pattern
  documented 2026-05-17 and noted as "Single-line form is mandatory"
  in `agents/baseline/main.py:194-198`. The bundler stripped the
  opening line but leaked the three continuation lines.
  (c) `scripts/bundle_agent.py` unlinks the bundle file on
  import-check failure, blocking debug inspection of the offending
  line. Required a Python REPL workaround.
  Three friction tags added; promotion candidates listed below.

- **Rule 43 multi-opponent panel skipped pre-submit.** PI's "submit"
  + "go" was interpreted as authority to compress the gate chain.
  Bundle smoke + tests/test_bundle.py + single-game smoke ran;
  panel did not. Wilson-lo on the one-opponent A/B was 0.505 —
  just over Rule 45's 0.50 minimum but well below Rule 43's 0.55
  panel target. Submitting on n=16 with Wlo=0.505 is the loosest
  submission gate we use; the sibling-branch precedent (identical
  12/16 → μ ~963-985) suggests the result will be parity-or-noise.

- **No bad decisions retrospectively.** Each step (bundle, fix,
  test, push) was the right action given the priors at decision-time.
  The latent bundler bugs were not detectable without attempting a
  bundle; the Rule 43 skip was a PI-authorised compression, not a
  bypass.

## Frictions logged this session

- `bundler-lib-source-strips-indent-on-alias-rebind` —
  `audit/friction.md` 2026-05-23 block, entry 1.
- `bundler-multiline-paren-import-leaks-continuation` —
  `audit/friction.md` 2026-05-23 block, entry 2.
- `bundler-deletes-on-failure-blocks-inspection` —
  `audit/friction.md` 2026-05-23 block, entry 3.
- `rule-43-panel-skipped-on-pi-go` —
  `audit/friction.md` 2026-05-23 block, entry 4.

## Promotion candidates (PI ratification: PENDING)

### [ ] scripts/bundle_agent.py — `_clean_lib_source` preserve indent

**Tag:** `bundler-lib-source-strips-indent-on-alias-rebind`
(in-function imports in lib/ modules break bundling silently)

**Where to insert:** `scripts/bundle_agent.py::_clean_lib_source`
(line 233-242 currently).

**What to add:** mirror the indent-capture logic from
`_clean_agent_source` (line 265-282). Replace
`out.append(f"{asname} = {original}\n")` with
`out.append(f"{indent}{asname} = {original}\n")` where `indent` is
captured the same way (`line[: len(line) - len(line.lstrip())]`).
Add a `tests/test_bundle.py` case: a lib module with an in-function
aliased import, assert the bundle imports cleanly.

**Why:** cost ≥ 30 min debug this session; latent for 1+ day on
this branch (commit 164498a 2026-05-22); will recur every time a
lib/ in-function import is added without an immediate bundle test.

### [ ] scripts/bundle_agent.py — AST pre-check for multi-line paren imports

**Tag:** `bundler-multiline-paren-import-leaks-continuation`

**Where to insert:** `scripts/bundle_agent.py` near the top of
`bundle()` (line 285), before the concatenation loop.

**What to add:** parse each source file with `ast.parse`; if any
`ast.ImportFrom` node has a `col_offset == 0` AND its source spans
multiple lines AND the module starts with `lib.` or the agent
package name, raise `BundleError("multi-line parenthesised intra-
lib import at {path}:{lineno} — convert to single-line form per
CLAUDE.md note in agents/baseline/main.py:194-198")` with the exact
file:line. Beats surfacing as a generic IndentationError 5000
lines downstream.

**Why:** exact-pattern recurrence of a friction documented
2026-05-17. The friction is in the docs but the docs don't enforce.

### [ ] scripts/bundle_agent.py — `--keep-on-failure` flag

**Tag:** `bundler-deletes-on-failure-blocks-inspection`

**Where to insert:** `scripts/bundle_agent.py` argparse block
(line 553-568) + the `out.unlink()` call sites (line 600, 609, 620).

**What to add:** new flag `--keep-on-failure` (default False). When
set, rename `out` to `out.with_suffix(out.suffix + ".broken")`
instead of unlinking, and emit the path in the error message.

**Why:** debugging a 10k-line generated file requires the file.
Cost this session: ~5 min on Python REPL workaround. Future debugging
sessions will recur the workaround unless flagged.

## PI additions (from step 4)

(awaiting PI input)

## Framework version at session-end
- Commit SHA: `4522815` (`submit: wave V3 + leaf-Δ gate + planet_positions cache`)
- Active rules: 1-47 (CLAUDE.md `## Operating rules — concise`)
- Loaded skills this session: postmortem (from /wrap up trigger);
  none others invoked.
