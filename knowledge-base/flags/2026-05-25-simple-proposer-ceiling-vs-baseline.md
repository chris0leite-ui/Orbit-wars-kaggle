# 2026-05-25 — simple-proposer family hits a hard ceiling vs `agents/baseline`

## Claim

Any 1-step greedy proposer of the
`realize(intents, obs, mechanisms=DEFAULT_MECHANISMS)` shape — even
with production-first target selection, defensive reinforcement,
enemy-denial weighting, and synchronized salvo via cross-turn
ledger — **loses 0/8 to `agents/baseline`** on the standard 8-game
no-swap A/B harness.

Evidence: V2 (greedy + defense), V3 (V2 + enemy_multiplier when
behind), V4 (V3 + synchronized salvo + cross-turn wait ledger) all
shipped at 0/8 vs baseline. Commits 2f37c3e, 8580ce7, c75f524 on
branch `claude/strategy-battlefield-game-6v82d`.

## Why this is a flag, not a friction

The 0/8 outcome isn't a bug; it's the **structural** signature of a
complexity-class gap. Baseline runs K-step rollout with opp modeling
on top of the same mechanism pipeline; we ran 1-step greedy on top
of the same pipeline. No knob within the greedy family closes that.

The flag is load-bearing for **future agent design choices**:

- If a new agent wants to "improve on momentum_strike", adding more
  knobs to the greedy proposer won't lift vs baseline. Don't waste
  iterations testing that hypothesis again.
- The next useful direction from this branch's design family is
  structural: K-step rollout, joint-action enumeration over multiple
  sources, or wrapping baseline directly.
- For agents that DO want to use momentum_strike as a calibration
  probe (PI's stated intent), the 27/32 simple-panel + 0/8 baseline
  signature is the published behaviour to benchmark against — don't
  re-test the baseline gap, it's known.

## When to revisit

Drop this flag if:
- A simple-proposer-family agent on this branch lift to ≥ 3/8 vs
  `agents/baseline`. Would invalidate the "structural" claim.
- The baseline μ on the live ladder drops materially (e.g. baseline
  evicted by a weaker rolling-pair partner) and "0/8 vs current
  baseline" no longer maps to "predicted μ << floor."

## Cross-links

- `audit/2026-05-25-postmortem-strategy-battlefield-game-6v82d.md` —
  full session postmortem.
- `audit/friction.md` 2026-05-25 — `structural-gap-not-knob-tunable`
  friction entry.
- `knowledge-base/thoughts/2026-05-25-momentum-strike-as-calibration-probe.md`
  — what the artifact is FOR, given the ceiling.
