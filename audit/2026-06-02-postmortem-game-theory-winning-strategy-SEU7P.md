# Postmortem — 2026-06-02 game-theory-winning-strategy-SEU7P

PI-prompted speed-audit session that pivoted into a structural fix of the
kinematic-table singleton. One refactor landed (commit `40f2614`); two
audit docs and one new diagnostic tool added; one repeated pattern
(substrate claims that needed PI correction to land at a realistic
distribution).

## What went wrong

Decision-quality flags, evaluated against priors that existed at
decision-time (per `knowledge-base/concepts/decision-quality-vs-outcome-quality.md`).

- **Trusted the microbench's headline as a substrate-health summary.**
  `scripts/bench_fast_sim.py` reports `0.107 ms/step` and "215×
  speedup" — true for an idle reused Snapshot, misleading for production
  load. I framed the morning around it ("sim is fast, look elsewhere")
  before writing the realistic-load probe. I had the priors to know the
  bench was idle-shaped — its docstring says so — and chose to read the
  headline anyway.
- **Reported a strategic claim from a single-seed probe.** First
  production-cost probe ran one game (seed 42, baseline-vs-baseline,
  ended at turn 365 with one side dominated). Returned "agent uses 81 ms
  / 950 ms, 869 ms headroom." Took PI pushback ("we go over budget,
  check realistic") to expand to 5 seeds vs v7_0 → median 286 ms / p95
  701 ms — 3× different. I should have run multi-seed + real opponent
  before sending the headroom claim, especially given the empty late
  bucket was visible in the seed-42 output.
- **Misread `swept_pair_hit` as fleet-vs-fleet without reading the call
  site.** Claimed O(fleets²) all morning. PI asked "fleets cannot collide
  in air. is that what we check?" — verified at
  `lib/game/interpreter.py:743-768` that it's fleet-vs-PLANET, O(fleets
  × planets), enforcing the game rule that fleet–fleet "combat" only
  resolves at arrival on a shared planet. The grep took 10 s when finally
  done.

## PI overrides

Three corrective interventions this session:

1. **"check a realistic setting"** — recovered the realistic per-turn
   cost distribution (p95 = 701 ms, vs claimed median 81 ms). Without
   this pull, the speed audit would have shipped an inverted finding.
2. **"fleets cannot collide in air. is that what we check?"** —
   surfaced the function-shape misread; closed a class of optimisation
   thinking that was based on the wrong cost shape.
3. **"do option 2 in this branch so others can pick it up"** —
   directed the kinematic-table refactor onto the right axis
   (structural fix attaching state to `World`, not procedural
   mandate). Resulted in commit `40f2614`.

All three were calibration data-points where the agent's claim was
3-10× wrong on first pass.

## Rule-bypass failures

- **Rule 38 (fix-verification reproduces failure state)** — the new
  test `tests/test_kinematic_table_per_world_isolation.py` does
  reproduce the original 2026-05-29 contamination scenario; this rule
  was followed on the refactor side. **Bypassed on the speed-audit
  side:** I did not reproduce the "we go over budget" condition before
  reporting the morning headroom claim. The probe was a NEW
  diagnostic, not a reproduction of the existing condition. Rule 38
  applies more broadly than its title suggests — "before reporting a
  diagnostic conclusion, reproduce the state being diagnosed."
- **Rule 41 (confound-sweep before correlational conclusion)** —
  "single-seed mirror-match" vs "5-seed-vs-real-opponent" is exactly
  a confound difference (opponent strength, seed dispersion, game
  length). The morning probe didn't sweep either confound before
  drawing a strategic conclusion.

## Rule-gap failures

No new rule-gap surfaces today. The friction patterns are recurrences
of known classes (`microbench-headline-overclaims-vs-production-cost`
echoes earlier "headline-metric-misreports-substrate" patterns;
`function-name-misread-as-fleet-fleet-collision` is a sub-clause of
the general "read the call site" discipline).

## Frictions logged this session

Cross-link to `audit/friction.md ## 2026-06-02`:

- `microbench-headline-overclaims-vs-production-cost`
- `single-seed-probe-misleads-headroom-claim`
- `function-name-misread-as-fleet-fleet-collision`
- `kinematic-table-singleton-cross-seat-contamination` (refactor landed
  in `40f2614`)

## Promotion candidates (PI ratified: no)

PI response to "anything you'd add to the postmortem? promote these
candidates?": **"Nothing to add or to promote."**

The three candidates drafted in chat (multi-seed-probe-requirement,
substrate-bench-distribution-reporting, and the still-pending Rule 50
KT-singleton-mandate) all remain UNRATIFIED.

- Rule 50 (procedural mandate for `scripts/clean_ab.py`) continues to
  sit in the 2026-05-29 postmortem's pending queue. Today's structural
  fix (`40f2614`) does not retire it.
- The two new candidates from today (multi-seed + bench-distribution)
  do not promote this cycle. If either pattern recurs they re-enter the
  promotion queue with stronger cost evidence per the postmortem skill
  step 3 criteria.

## PI additions (from step 4)

None. PI declined to add frictions, decisions, or promotion candidates
beyond what the agent surfaced.

## Framework version at session-end

- Commit SHA: `89e6d5291438eae6fcc999916ed66c7551a6b506`
- Branch: `claude/game-theory-winning-strategy-SEU7P` (ahead 53 of `origin/main`)
- Active rules: 1..48 (per CLAUDE.md `## Operating rules — concise`).
  Rule 50 promotion candidate from 2026-05-29 remains pending.
- Loaded skills this session: `postmortem`, `kaggle-comp`, `update-config`.

## What ships from today

- `lib/kinematic_table.py` — per-World attachment via `attach()` /
  `for_world()`. Sibling branches can cherry-pick.
- `lib/trajectory.py` — `_table_window_or_none` cut over to
  `for_world(world)` with no singleton fallback. Load-bearing safety
  cut.
- `tests/test_kinematic_table_per_world_isolation.py` — 7 assertions
  reproducing the contamination scenario + locking the transitional
  contract.
- `scripts/production_cost_probe.py` — new diagnostic for realistic
  per-turn cost distributions; takes `--seeds` and `--opp`.
- `audit/2026-06-01-fast-sim-bench.md` and
  `audit/2026-06-01-production-cost-probe*.md` — recorded numbers,
  reproducible.

## Open thread for next session

`knowledge-base/thoughts/2026-06-02-speed-audit-and-kt-refactor.md`
records the unfinished diagnostic: the 138 ms median non-sim per-turn
cost has no breakdown yet. Without that breakdown we cannot tell
whether sim-step cost or proposer-cost is the bigger lever for getting
under the 920 ms hard cap. The `production_cost_probe.py` infrastructure
is in place to add it.
