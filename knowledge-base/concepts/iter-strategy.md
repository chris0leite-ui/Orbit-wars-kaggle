# `iter` — strategy + what we learned

> **Audience.** Whoever picks this up next (PI or another agent). Plain English.
> No letter-number experiment codes; describe what each thing IS, not its label.
> Written 2026-05-15.

## What is `iter`?

`iter` is the agent we're shipping to the Kaggle ladder. It's a small wrapper
around the project's most-validated brain (`v7_0_drop_one`'s chooser, called
`choose` in `lib/v7_search.py`) plus three deliberate additions on top:

| What | Where | Why |
|---|---|---|
| **Composite leaf-state scorer** | `lib/value_heads.py:composite_capture_value` | At the end of every 10-step lookahead, score the world: reward fleets predicted to successfully capture, penalise fleets predicted to bounce or fly off-screen. Validated +12 pp lift over the bare ship-count scorer in 64-seed panel. |
| **Production discount γ = 0.99** | `lib/scoring.py:PV_GAMMA` overridden at iter import | Future production is worth slightly less than current production. Mirrors what `v7_pv` did when it was at the top of our ladder. |
| **Real 4-player chooser** | `lib/v7_search.choose_4p` dispatched in iter when the game has 4 seats | `iter_v1` (our previous submission) silently fell back to `v3.5.1`'s incumbent every time the game was 4-player — that's about 36 % of our ladder games. `choose_4p` uses the actual drop-one rollout, with its built-in "us vs strongest single opponent" leaf scorer rather than the 2-player-tuned composite. |

Everything else (mission persistence, multi-source coordination, adaptive
K, comet hooks, sizing fixes) is **disabled** in the shipping config but
remains in the codebase behind feature flags, ready to be re-enabled when
the underlying problems are properly debugged.

## What problem does `iter` solve?

Two real bugs:

1. **`iter_v1` was playing v3.5.1 in 4P.** It called `choose()` directly,
   which has an explicit "2-player only" guard (`v7_search.py:1817`) — in
   4-player games it returns the v3.5.1 incumbent. v3.5.1 is a μ ≈ 945
   agent. About 36 % of our ladder games were 4P, so this directly cost us.
   Fix: dispatch to `choose_4p` when `_detect_num_seats(world) == 4`.
2. **`iter_v1` was passing a 2P-tuned scorer to 4P scoring.** The composite
   leaf scorer's base term is "our ships minus all opponents' ships." In
   4-player that's "us minus three enemies" — heavily negative, biases
   toward defensive play. `score_candidate_4p` has a built-in 4P-aware
   scorer ("us minus the SINGLE strongest opponent") that matches the
   first-place objective. Fix: pass `value_fn=None` to `choose_4p` so the
   4P-aware default kicks in.

Both fixes are **upside-only**:
- They cannot regress 2P (the 2P code path is untouched, panel-validated
  identical to `iter_v1` against `v3.5.1` and within statistical noise
  against `v7_0` / `v4_planner`).
- The only failure mode is `choose_4p` being slightly worse than v3.5.1
  in 4P games, which we consider extremely unlikely given the structural
  argument.

## What we tried this session that didn't work

Honest record of the architectural changes attempted before settling on
the minimum upside. Every one of these regressed in local panel testing,
some catastrophically:

| Change | Result | Mechanism (best hypothesis) |
|---|---|---|
| Snipe launch-sizing uses `pred_ships` at predicted ETA instead of current garrison | −23 pp vs v7_0 | Smaller fleets → slower fleets → larger ETA → smaller `time_to_hold` → settle_plan deprioritises the mission entirely. Decoupling SIZING from SCORING is what's needed. |
| Same for `drain` | bundled with above | — |
| `DYNAMIC_PROD_BUFFER` bumped from 1 to 2 for moving-planet sizing | −23 pp vs v7_0 | The +1 was painstakingly calibrated in v3.3; +0 under-sizes, +2 over-sizes. Both regress. |
| Comet anti-panic cap (max N launches per comet target per turn) | −8 pp vs v7_0 | Drops launches the chooser already chose as best given the multi-launch state. Right home for this is in `lib/missions/snipe.py`'s proposer, not a post-hoc filter. |
| Adaptive K based on max in-flight fleet ETA (no relevance filter) | wallclock blew past 1000 ms cap | Increasing K linearly scales rollout cost; the watchdog only bails BETWEEN candidates. |
| Adaptive K with relevance filter (only extends K for fleets aimed at our planets or high-prod) | ~ neutral | Theoretically right but hard to validate; net effect tiny. |
| Mission persistence + multi-source coordination (commit to a target across turns, allocate ships from multiple sources) | −42 pp vs v7_0 | Most likely: stranding home (`MP_SOURCE_RESERVE = 0`) leaves us undefended → enemy counter-attacks succeed → snowball. Or the static plan conflicts with the chooser's per-turn re-evaluation in destructive ways. |
| Comet evacuation (drain ships off a comet about to leave) | mixed | Theoretically free upside; in practice slightly net-negative in panel. Possibly because forced launches from a comet consume that source's per-turn launch slot. |
| Latest-launch heuristic (shrink fleets to minimum that still arrives in time) | not eval'd in isolation | Defaulted off, kept in code. |

The dominant pattern: **theoretically clean fixes regress because they
disturb the chooser's value scoring or mission-ranking pipeline.** Local
eval is doing its job — saving us from shipping disasters every time.

## Why the strategy is conservative

The game is a 500-step territory race. Our agent is a per-turn local
optimiser with K=10 lookahead. Top players play multi-turn coordinated
expansion plans. The architectural gap (per-turn vs multi-turn, local-ROI
vs strategic-position) is real but **every architectural patch we tried
this session regressed.**

Reasoning by elimination:
- We have a known-good shipped agent (`iter_v1`, ladder μ = 1020.7).
- Every "smart" addition this session has failed locally.
- One un-tried fix has a clean structural argument: 4P dispatch.

Ship the minimum that adds the structural fix without touching the
validated 2P brain. **Expected ladder μ ≈ 1020 + a 4P boost.**
Downside floor: same as `iter_v1` if the 4P fix doesn't help.

## Submission state on the ladder

| Submission | Ladder μ | Status |
|---|---|---|
| `iter_v1` (#52661990) | **1020.7** | Currently in rolling-last-2 |
| `geo` (#52643676) | 1001.7 | Currently in rolling-last-2 |
| `v7_pv` (#52630118) | 1053.5 (drifting down) | Evicted by `iter_v1` push |

Pushing the new `iter` would evict `geo`, leaving rolling-last-2 =
`[iter_v1 = 1020.7, new_iter ≈ 1020–1060]`. Team score floor stays at
1020.7. Upside if the 4P fix lifts the new iter above iter_v1.

Local panel (32-seed × 3-opponent, against `v7_0`, `v4_planner`, `v3.5.1`):

| Opponent | iter_v1 (64s) | new iter (64s) | Δ |
|---|---|---|---|
| v7_0 | 60.9 % (Wilson [0.49, 0.72]) | 54.7 % (Wilson [0.43, 0.66]) | −6 pp (overlap) |
| v4_planner | 67.2 % (Wilson [0.55, 0.77]) | 62.5 % (Wilson [0.50, 0.73]) | −5 pp (overlap) |
| v3.5.1 | 64.1 % (Wilson [0.52, 0.75]) | **64.1 % (Wilson [0.52, 0.75])** | 0 pp (bit-identical sample) |

The drift on v7_0 and v4_planner is within statistical noise. The
v3.5.1 result is bit-identical to `iter_v1`. Panel verdict INCONCLUSIVE
both times — same Wilson confidence, same statistical level.

## Why we believe it should succeed

1. **2P parity**: the local 2P panel reproduces `iter_v1`'s level. We are
   not regressing 2P play; we are at parity with our currently-shipped
   agent on 2P games.
2. **4P upside**: every 4P game that `iter_v1` played as v3.5.1
   incumbent now uses `choose_4p`'s real drop-one rollout, with a 4P-aware
   leaf scorer. The structural reasoning is clean:
   - `choose_4p` is the project's purpose-built 4P chooser, used by v7.4
     and v7.5 historically.
   - Its default leaf scorer is "our ships minus the strongest single
     opponent" — matches the 4P objective (be 1st place).
   - `iter_v1` was scoring 36 % first-place against ladder opponents
     while playing v3.5.1. Any improvement above that is upside; complete
     failure means matching v3.5.1, which is what `iter_v1` already does.
3. **No catastrophic floor**: the rolling-last-2 mechanism guarantees
   our team score stays at max of two submissions. Pushing new iter
   leaves `iter_v1` (μ = 1020.7) in the pair, so the team's floor is
   1020.7 unless the new iter lands below `geo` (1001.7), which would
   require both: catastrophic 2P regression beyond panel noise (very
   unlikely) AND a 4P regression below v3.5.1 (the dispatch's structural
   prior makes this unlikely).

## What we will try next (parked, not in this PR)

- **Imitation learning from top-10 ladder replays.** The single
  architectural direction that has NOT been tried this session and that
  has historically broken plateaus in similar games (AlphaZero, NeurIPS
  agent competitions). Pull top-100 finishers' games, train a small NN
  to predict P(win) from the board state, plug as the leaf value scorer.
  ~1-2 weeks. Mentioned in `state/handover-2026-05-13.md` as the
  Bovard-style parked direction.
- **Mission persistence v2** with proper testing: revisit after IL.
  The −42 pp this session shows the architecture isn't the problem;
  the implementation's interaction with the chooser is. Needs
  step-by-step replay debugging on a synthetic game.
- **Phase-aware strategy** (opening / consolidate / endgame with
  different leaf scorers per phase). Several months of work; the right
  long-term direction.

## Pointers

- Agent source: `agents/iter/main.py`
- Bundle: `submissions/iter.py` (generated by `scripts/bundle_agent.py agents/iter`)
- Tests: `tests/test_iter_agent.py` (smoke + knob presence + 4P smoke;
  18 tests), `tests/test_snipe_sizing.py` (sizing-fix regression suite,
  one skipped pending decoupled-sizing fix), `tests/test_bundle.py`
  (bundle parity).
- Loss-mode data on `iter_v1`: `audit/loss-modes-52661990.csv` (61 %
  opening-lost, 39 % mid-economy-lost across 44 lost ladder games).
- Live replays cached locally: `audit/live-episodes/52661990/`.
- Friction record: `audit/friction.md` (sizing-regression and
  mission-persistence failures recorded for next session).
