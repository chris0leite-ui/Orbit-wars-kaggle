# 2026-05-22 — test-infra env-var leak diagnosis

## Diagnosis

Local A/B test infrastructure has a fundamental env-var pollution bug
that corrupts variant-vs-variant tests sharing the same code paths.

### Mechanism

1. `agents/baseline_*/main.py` shims and `submissions/*.py` bundles
   call `os.environ.setdefault(KEY, VALUE)` at module load time.
2. `agents/baseline/main.py` constants like `STAGNANT_DRAIN_ENABLED =
   os.environ.get("BASELINE_STAGNANT_DRAIN", "0") == "1"` are
   evaluated at import time and cached as module-level globals.
3. When `fast.py` (or any in-process A/B) loads two agents in the same
   Python worker:
   - First-loaded agent's `setdefault` sets env keys.
   - Second-loaded agent's `setdefault` is a no-op for already-set
     keys.
   - Module-level constants (e.g., `JOINT_TOP_K_PER_TARGET`) inside
     `agents.baseline.chooser_trajectory` are cached on first import
     and never re-read.

### Empirical confirmation

```python
m1 = load('agents/baseline_joint_aggr_consolidated_topk8/main.py', ...)
# m1 sets BASELINE_JOINT_TOP_K=8
m2 = load('submissions/baseline_joint_aggr_consolidated.py', ...)
# m2's setdefault BASELINE_JOINT_TOP_K=5 is no-op (already 8)
# m2's bundle code reads env → TOP_K=8 (not 5!)
```

Both agents end up with TOP_K=8.

## Consequences

- The previous session's small-n A/B variants (drain, sniper,
  orbital, combat_stack) tested in fast.py against
  `iter_baseline.py` (= older bundle without those env-gated
  features) may have been clean because the older bundle didn't
  read the new env vars.
- Tests sharing env vars (e.g., topk5 vs topk8, MILP-on vs MILP-off)
  through the standard fast.py path produce **invalid** results —
  both agents end up with the same effective config.
- The previous session's "4P null hypothesis is 25%" misread in
  `audit/friction.md` 2026-05-21 may have ALSO been a misread of
  CORRECT 2P results — i.e., drain @ 6/16 = 37.5% is a -12.5pp
  REGRESSION against the 50% null, not a +12.5pp "lift".

## Workarounds

### For variant-vs-bundle tests (different codebases)
Use `fast.py eval <shim_dir> --vs <old_bundle.py>` directly. Safe
because the old bundle doesn't read the new env vars.

### For variant-vs-variant tests (same codebase, different env)
Hardcode constants inline in two separate bundles:
```python
# variant_topk5_locked.py — copy of consolidated with this swap:
JOINT_TOP_K_PER_TARGET: int = 5  # was: int(os.environ.get(...))
JOINT_MAX_PAIRS: int = 60

# variant_topk8_locked.py — same but:
JOINT_TOP_K_PER_TARGET: int = 8
JOINT_MAX_PAIRS: int = 100
```

Each bundle is loaded as a unique module via
`spec_from_file_location`; their hardcoded constants live in
separate namespaces, unaffected by `os.environ`.

### True per-game isolation
`scripts/clean_ab.py` (this session) spawns one subprocess per game.
Each subprocess has fresh env at start, but **env leak between the
two agents loaded inside the subprocess still applies**. Useful
only for variant-vs-bundle (different code).

True per-agent process isolation (each agent its own subprocess
per game) is not supported by `kaggle_environments.env.run()` —
would require custom plumbing.

## Hardcoded variants prepared this session

- `submissions/variant_topk5_locked.py` — TOP_K=5, MAX_PAIRS=60
- `submissions/variant_topk8_locked.py` — TOP_K=8, MAX_PAIRS=100
- `submissions/variant_milp_on.py` — OPENING_MILP_ENABLED=True
- `submissions/variant_milp_off.py` — OPENING_MILP_ENABLED=False
- `submissions/variant_wallclock800.py` — 800ms budget
- `submissions/variant_wallclock600.py` — 600ms budget (control)

Each isolates exactly one structural axis.
