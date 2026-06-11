# HANDOVER.md — next-session brief

## Live state (check freshness: `kaggle competitions submissions orbit-wars`)

Rolling pair: **sub 53558897** `ledger_v1_2` + **sub 53556728** `ledger_v1`
(both climbing; v1 was ~1060, v1_2 ~905 mid-warm-up at last read).
The 1300.9 producer_plus agent was evicted by PI order for live feedback.

## Unshipped improvements (in `agents/ledger/main.py` = `submissions/ledger_v1_4.py`)

1. **4-player leader objective** — attack only the projected winner;
   brawling non-leaders discounts to 0.2. Measured: **9/16 first places
   vs three strong agents (parity = 4/16, so 2.25x)**. 4P is ~32% of
   ladder games; this is the biggest unshipped lift.
2. **Stalemate-gated endgame gambit** — projection-behind + frozen board
   => admit any plan, bypass the veto (from live loss ep 79496718).
3. Coalition rescue + shopping-commitment scaling (v1_1/v1_2 — v1_2 is
   live; v1 lacks both).

Submitting v1_4 evicts ledger_v1 (the older pair half). PI sign-off
required per submission (Rule 1).

## Plan remainder (designs agreed, not yet built)

- **Race modeling on contested neutrals**: replace the flat RACE_DISCOUNT
  with a ledger walk that injects the opponent's hypothetical
  garrison+1 launch at their best ETA and prices arriving second (let
  them pay the sink, snipe the surplus). All machinery exists
  (`_walk_planet` extra_list).
- **Wave merge**: when several planned targets cluster, merge budgets
  into one overwhelming synchronized wave (top-1600 agents launch half
  as often at 2-4x mass).
- **Opponent profiling**: NEGATIVE RESULT as built (fleet-launch
  classifier misreads garrison-holding defenders as passive; both
  calibrations degraded pools). If revisited: classify by *garrison
  growth at threatened planets*, not launches. Code in git history at
  e82fed2^.
- **Loss loop routine**: `python scripts/live_episode_summary.py <sub> --pull`
  then read the losses. Both live submissions accumulate episodes.

## Verification benchmarks (paired pools, current build)

- live-1300.9 bundle rebuild (`/tmp/latest_live_sub.py`, rebuild recipe in
  audit 2026-06-10 + sibling branch commit d849637): seeds 600-615 -> 15-16/16
- v7_0 12-seed pool (700,702,505,508 + 8 winners): 8-9/12
- 4P panel seeds 1000-1015 vs v7_0/v4_planner/v3.5.1: 9/16
- `tests/test_ledger_forecast.py` must stay green (engine exactness).

## Pointers

- `audit/2026-06-10-ledger-agent-from-first-principles.md` — the night.
- `agents/ledger/main.py` — the agent (heavily commented header).
- `tests/test_ledger_forecast.py` — the exactness gate; keep it green.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42) before any submit.
- `state/STRATEGY.md` — previous strategy (baseline_adaptive_k), not yet
  superseded on paper; PI decision pending.
