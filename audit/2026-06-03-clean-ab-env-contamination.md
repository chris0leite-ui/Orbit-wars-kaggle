# 2026-06-03 — `clean_ab` env-var contamination between A/B agents

Branch: `claude/state-k-arrival-aware`. Found while building the
orbital-lead state-K A/B.

## The bug

`scripts/clean_ab.py` runs each game as `env.run([focal_path, opp_path])`
inside a per-game subprocess. **Both agents load into that ONE process**
(orbit_wars / `kaggle_environments.core` has no per-agent subprocess
isolation — `core.py` uses `multiprocessing.Pool` only for parallel
*episodes*, and the game process has no child processes per agent).

Agent config is applied at import via `os.environ.setdefault(...)`, and the
agent reads those keys **at turn time**. `os.environ` is process-global, so:

- whichever agent imports first sets the shared keys;
- the second agent's `setdefault` is a no-op for any key the first already
  set;
- both agents then run with the **first-loaded agent's** value for every
  shared key.

## Demonstrated directly

Two wrapper agents, one setting `BASELINE_STATE_K_ORBITAL_LEAD=1`, one
leaving it unset (expects 0). One game via `env.run`:

```
ON-wrapper  sees lead=True   pid=14970
OFF-wrapper sees lead=True   pid=14970     <-- contaminated (should be False)
```

Same pid, OFF saw the ON value. Confirmed in-process; confirmed `clean_ab`
uses the identical `env.run([p0,p1])` call.

## Scope of the damage (UNVERIFIED, needs follow-up)

Any A/B where focal and opp **read the same config key with different
intended values** is contaminated — i.e. most **single-variable** A/Bs run
through `clean_ab` (variant-ON vs variant-OFF of the same baseline). This is
exactly the shape of many `mechanism-ledger` "dead/falsified" verdicts and
the shipped state-K's `24/32=75%` lever A/B. **A contaminated single-variable
A/B compares X-vs-X → pure noise**, which would read as parity/null.

This is a strong candidate root cause for the chronic
`local-AB-not-calibrated-to-live-ladder` gap (Rule 43's origin): some
"falsified" mechanisms may have been falsified on contaminated evidence, i.e.
**working ideas may have been discarded**.

Cross-architecture A/Bs (e.g. v15-dir vs baseline-dir reading *disjoint*
keys) are NOT affected — only shared-key comparisons.

## The workaround (used for the orbital-lead A/B)

Set-each-turn wrapper: each agent writes ITS OWN value to the env key
immediately before delegating to the real agent. Agents are called
sequentially within a turn, so each computes with the correct value:

```python
_MINE = "1"  # or "0" for the opponent
def agent(obs):
    os.environ["BASELINE_STATE_K_ORBITAL_LEAD"] = _MINE
    return _real(obs)
```

Verified clean: ON sees True, OFF sees False in the same game.

## Recommended fixes (not yet done)

1. **Harden `clean_ab`**: either (a) bake config as module-level CONSTANTS
   into fully-inlined bundles that don't read shared `os.environ`, or
   (b) adopt the set-each-turn wrapper convention, or (c) drive each agent
   from a genuinely separate process via the agent protocol.
2. **Re-audit** 1–2 high-value "falsified" single-variable A/Bs with a
   contamination-proof harness before trusting their null verdicts.
3. Add a regression check: a probe asserting two same-key wrappers see
   their own values in one game.

Origin friction tag: `clean-ab-shared-environ-contaminates-single-variable-ab`.
