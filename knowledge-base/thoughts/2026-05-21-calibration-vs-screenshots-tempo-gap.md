# Calibration vs PI screenshots — the LP runs at half-tempo of strong opps

**Date**: 2026-05-21 evening
**Origin**: PI flagged three failure modes in sub 52894340 (FND) via
ladder replay screenshots. This entry records the empirical findings
from the first calibration cut.

## What was compared

Three live-ladder games of FND (`submissions/_phase4_step1_FND.py`,
sub 52894340) vs strong opponents:

| Screenshot | Episode | Seed | Opp (μ snapshot) | Result | Steps |
|---|---|---|---|---|---|
| S1 step 14 | 77321232 | 1250638780 | Mille Initiate (~1028) | LOSS 62-75 | 158 |
| S2 step 12 | 77320686 | 1085160712 | KoshinM (~1122) | LOSS 53-53 tiebreak | 121 |
| S3 step 44 | 77323008 | 669336863 | Aidan P5 (~1232) | LOSS 326-339 | 108 |

Side-by-side per-step action diff produced via
`scripts/replay_compare_screenshots.py` + per-side aggregate stats via
`scripts/replay_stats_screenshots.py`.

## Empirical findings — same pattern in all 3 games

| Behavioral metric | OUR (3-game avg) | OPP | Direction |
|---|---:|---:|---|
| Tempo (% of steps with ≥1 launch, n=100) | 33-49% | **48-72%** | OPP fires more often |
| Median delay first-launch-from-newly-captured planet | **9-10 ticks** | **4-6 ticks** | OPP exploits captures **2x faster** |
| Mass-burst count per game (≥3 launches same src+angle) | 2-3 | **0** | OPP never bursts; we do |
| Mean ships per launch (n=60) | 22.9-29.1 | 14.6-36.8 | mixed |
| Distinct sources used (n=60) | 10-11 | 8-11 | similar |
| Max same-source streak (consecutive launches from one src) | 5-11 | 3 | We over-rely on a few sources |

## The smoking gun

At all three PI-flagged steps (14, 12, 44), **we did not fire** while OPP
did. Cross-game pattern:

- S1 step 13: OPP fired (their 9th launch from initial P0). Step 14:
  silent on both sides. Step 15: OPP fired again. **We had a 5-step gap
  (steps 13-17) of zero launches.**
- S2 step 12: OPP fired from P19 — a planet they captured at step 9, so
  3 ticks after capture. We didn't fire from our P16 (captured step 9)
  until step 16 — **7 ticks delay**.
- S3 step 44: We fired 10 ships (small); OPP fired 20+. The LP picked a
  wait_N>0 variant accumulating for later, OPP fired immediately.

## Root cause hypothesis (most testable)

The LP's `wait_N` selection in `lib/joint_solver/lp_outcome.py` is
**systematically too patient**. For each candidate (src, tgt) pair, the
proposer emits multiple wait_N variants; the LP picks one. When the
"accumulate ships, fire later" variant has higher `prod_stream_me`
value (because more ships → more decisive capture → higher Wald-passing
margin), the LP picks it. MPC re-derivation each turn pushes the
fire_step further into the future indefinitely.

This is the **same architectural failure mode** the lighthouse plan
flagged in Phase 4 v1:

> 77% of "fires" were wait_N>0 deferrals that MPC re-solved away the
> next turn. Phase 4 attempt 1, audit 2026-05-20.

Phase 5C (outcome-table-aware LP) addressed it for defensive cases —
"defense emerges from math when prod_stream_opp is large on empty
subset" — but did not address the OFFENSIVE side. We still wait too
long to attack.

## Connection to PI's framing

PI: "what are we solving? It should give us expansion and production
compounding together with good tactical positions."

The tempo gap IS the missing compounding. Each tick we don't fire from
a new planet:
- We don't acquire its next-tier neighbor → expansion suffers
- The neighbor's production accrues to OPP or stays neutral
- Compounding curve shifts right by the delay

The math: a planet with prod=3 captured 5 ticks early gives 15 extra
production ships over the game. Multiply by 10-20 planets and the gap
is hundreds of ships.

## Mass-burst pattern (secondary finding)

We fire 5-7 identical-src-and-angle launches in a single step:

- S1 step 49: 5× src=24 angle=+1.80, 150 ships total
- S2 step 39: 5× src=17 angle=+2.60, 129 ships total
- S2 step 45: 6× src=17 angle=+1.80, 177 ships total
- S3 step 35: 4× src=16 angle=+2.60, 176 ships total
- S3 step 37: 7× src=12 angle=-2.80, 196 ships total
- S3 step 39: 5× src=8 angle=-0.80, 219 ships total

OPP NEVER does this. We're stacking spec-min-cap candidates against the
same target when ONE larger fleet would do. The proposer's
`enumerate_ship_counts` emits multiple ship-count variants per (src,
tgt); the LP picks "all of them" because each variant scores positive
value individually with the LP unaware that they're redundant. This
points to a **candidate-deduplication or LP subset-pricing bug**.

## Most testable next step

Add `WAIT_N_PENALTY` term to `_value_for_outcome` in lp_outcome.py:

```
value -= WAIT_N_PENALTY * col.wait_N
```

Initial `WAIT_N_PENALTY = 0.5` per tick. Rationale: typical
prod_stream values are 200-400; a 5-tick wait costs 2.5 — small enough
to lose to a clearly better target, large enough to break ties in
favor of fire-now.

Rule 38 pin test: synthetic scenario where two columns have equal
prod_stream but different wait_N. Pre-fix: LP indifferent (or picks
larger wait_N per tie-break). Post-fix: LP picks wait_N=0.

Then re-run the screenshot diagnostic; tempo% should rise from 40-50%
to 55-65% (closer to OPP's range). If yes, expand to n=4 A/B vs FND.

## Open questions

1. Is the burst pattern a different mechanism (candidate dedup) or
   another symptom of LP not seeing that variants are redundant?
2. Does the delay-to-first-launch-from-new-planet have an additional
   cause — e.g., MIN_LAUNCH_GARRISON=8 blocks new planets with garrison
   <8, but production catches up in ~3 ticks at prod=3?
3. Top-1-3 ladder players (3Comets 1684, bowwowforeach 1632, Vadasz
   1612) — do they have even higher tempo? We don't have replays of
   their games since FND didn't play them. Pulling their submissions'
   episodes would give upper-bound references.

## What to commit

- `scripts/replay_compare_screenshots.py` — per-step action diff
- `scripts/replay_stats_screenshots.py` — per-side aggregate stats
- `audit/live-episodes/52894340/summary.json` — auto-generated by
  `live_episode_summary.py`, kept per .gitignore exception
- `audit/replays/screenshot-stats.json` + `.md` are gitignored;
  regenerate via the scripts after pulling replays.
- This thoughts entry (permanent, append-only per Rule 35).
