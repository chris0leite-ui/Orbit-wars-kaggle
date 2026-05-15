# 2026-05-15 — Replay-seed regression testing (PI strategy)

> "note this strategy of analysis of games of us and using their seeds
> for testing."

## The pattern

Each Kaggle submission is evaluated across ~80-130 random-seeded
episodes. The replays are downloadable via:

```
kaggle competitions episodes <submission_id> -v
kaggle competitions replay <episode_id>
```

Each replay JSON has `info.seed` (the configuration seed) and
`info.TeamNames` (opponent identity). Loss replays give us a SEED
where our agent has demonstrably failed against a specific opponent
class.

**Use those seeds as a TARGETED REGRESSION TEST SET for the next
iteration.** Concretely:

```
python fast.py play agents/<new_version> --vs <opponent> --seed <loss_seed>
```

This is more informative than uniform random seeds because:
1. It REPRODUCES known failure patterns. If new code still loses on
   the same loss seed, the iteration didn't fix the failure mode.
2. It DIVERSE-CHECKS structural changes. A v10 that beats v9 on the
   board where v8 won but loses to v8 on Felipe's seed reveals
   something subtle that aggregate A/Bs would average out.
3. It SHORT-CIRCUITS sample noise. n=1 deterministic on a known-bad
   board says more than n=32 random.

## When this is most valuable

- After a live regression (v9 dropped −208μ at n=10) — pick seeds
  from the new submission's LOSSES, test next iteration against
  those seeds specifically.
- After a structural change — run the prior submission's full loss
  set as a "we didn't break what already worked" gate.
- When debugging "won locally lost live" calibration gaps — find
  which OPPONENT CLASS we lose to live, test new code on those
  same opponents/seeds.

## Cost-benefit

- Cheap: a seed test is one game (~5-30s on 1 core).
- Diagnostic: each seed reveals a CONCRETE failure scenario, not an
  aggregate statistic.
- Direction-setting: a single high-information seed test can change
  the iteration plan more than an n=64 aggregate A/B.

## Felipe-Ferreira example (2026-05-15)

v8 lost live game 76655989 (2P vs Felipe Ferreira) at seed=1492346051.
v9 came in at 890.7 (sparse-sample noise). Felipe-seed local test
of v10:
- vs v8: 0/2 (v10 loses both sides)
- vs v9: 2/2 (v10 wins both sides)

This is INSTANT diagnostic: v10's wait-then-fire mechanism beats v9
on this board but loses to v8. The v9 base's PV-discount + sum-of-opps
likely make the chooser too patient on this specific layout, and
even adding wait-then-fire on top can't fully compensate. The single
seed reveals more than the aggregate.

## Operational checklist

1. Download fresh replays after each submission: `kaggle competitions
   episodes <sub_id> -v` → `kaggle competitions replay <ep_id>`.
2. Classify wins/losses (e.g. `scripts/diag_outcomes.py`).
3. Pick 3-5 loss seeds covering different opponent classes (2P weak,
   2P strong, 4P with leader, 4P with parity).
4. Add them to a regression-seed set kept under
   `audit/<date>-<sub_id>-loss-seeds.md`.
5. Run new versions vs the regression set BEFORE submitting.

## What this protects against

The "local-overpredicts-live" pattern (v3.5.1, geo, iter v1/v2, v9):
local panel vs synthetic baselines (v7_0/v4_planner/v3.5.1) doesn't
catch the failure modes the live ladder exposes. Loss-seed testing
brings the LIVE distribution into the local evaluation loop.
