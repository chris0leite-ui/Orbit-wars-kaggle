# 2026-05-25 postmortem — K1+Z v2 ship + proposer-tightening axis exhaustion

Session: `claude/agent-design-exploration-Q0q9T`. Commits 843cc35..6627420 (8 commits, 2 reverts).

## Scoreboard

| Outcome | Detail |
|---|---|
| Shipped | sub 53018599 (K1 + Layer Z v2 bundle, commit 603f45f, sha256 `2f58cad1042862d3`). Status PENDING; settle 4-12h. |
| Local lift (vs phi1_only) | K1 alone: 45.3% n=64 / K1+Z v2: 56.2% n=64 / K1+Z v2 + Fix A+B: 80% n=5 |
| Local regression (vs joint_aggr) | K1+Z v2+Fix A+B: 20% n=5 / K1+Z v2+Fix A: 0% n=5 — Fix B reverted then Fix A reverted |
| Wallclock | p95 1101 ms → 887 ms; production max 651 → 718 ms (well under 1000 ms budget) |
| Branch HEAD | `6627420` — matches sub 53018599's source state + new `scripts/ab_quick.py` |

## What went right

1. **K1 (`c0035ff`) was a clean ship.** Discovered a built-and-parity-gated caching layer (`lib/kinematic_table.py`) extracted to origin/main from a sibling branch (`72fe45a`) but never wired into our agent. Two file edits (10 LOC each) + one conftest fixture for test isolation. `predict_relative` cumulative time -90% across a 219-turn game. Bit-parity by construction (21/21 parity tests + the original 564 byte-identical FleetFate assertions from sibling-branch `c48e143`). The wallclock saving was exactly within the documented 47-114 ms/step range.

2. **Z v2 (`603f45f`) was a real bug fix.** The original Layer Z formula `ships - prod·eta` omitted the `pred_ships` subtraction. Defended targets got hidden ship-credit equal to their predicted garrison. Fix was 5 LOC + dropping a redundant opening_planner site. n=64 A/B vs phi1_only: +10.9pp lift over K1-alone. Reproducible at n=5 (80%). The diagnosis (cheap-recapture asymmetry shown in seed-2020490432 screenshot) was correctly identified.

3. **New A/B standard tool (`0f2a23d`).** `scripts/ab_quick.py` — 5 games × 250-step cap × no seat swap × 3-opp panel. ~30 min wallclock vs ~30 min for the previous n=64 single-opp eval, but provides PER-OPP signal across the live ladder cohort. Used immediately to find the regression that single-opp eval missed.

4. **Two reverts ran cleanly.** Cache attempt `9870575` reverted to `52e771c`; Fix B/A reverted in two commits (`e277c53`, `6627420`). No accumulation of dead code in the branch HEAD.

## What went wrong

1. **Cache-the-MILP attempt (`9870575 → 52e771c`).** Tried to cache `opening_plan` output once per game based on a guess that the MILP was eating 1.7 s/call. Local A/B regressed 34.4% → 9.4%. Root cause: re-deriving each turn was a strategic FEATURE (the MILP responds to mid-opening state change), not a wallclock bug. Should have profiled FIRST. cProfile showed each `opening_plan` call was ~5 ms; the variance came from `predict_relative` × candidates, not from the MILP itself. Friction tag: `cache-attempt-falsified-by-rederive-feature`.

2. **Fix A+B (`03cb25b`) and Fix A alone (`e277c53`) regressed vs joint_aggr.** Three consecutive proposer-tightening attempts (Z v2, Fix A+B, Fix A alone) showed the same pattern: wins vs quiet opp (phi1_only), losses vs aggressive opp (joint_aggr, the live μ=1120.1 rolling-pair half). Rule 37 cap reached. The proposer pre-filter axis is now CLOSED for this opp cohort. The diagnosis (cheap-recapture asymmetry) is real but fixing it via pre-filters over-restricts. Next axis = chooser-side opp model, not proposer-side thresholds. Friction tag: `proposer-tightening-axis-exhausted-vs-joint-aggr`.

3. **HANDOVER Priority 2 recipe was mathematically wrong.** HANDOVER claimed `eta = d/√n for n>4` as the Z v2 formula basis. Env `lib/fleet.py:speed` actually uses `(log n / log 1000)^1.5`, not √n. Existing `aim_and_eta` already computed eta via `fleet_speed(ships)`, so "make eta fleet-speed-aware" was a no-op. The real Z v2 bug (missing `pred_ships` subtraction) was different from what HANDOVER described. Friction tag: `handover-recipe-mismath-vs-env`.

4. **n=5 panel results read as more definitive than warranted.** 0/5 vs joint_aggr has Wilson CI [0, 0.434] — point estimate is alarming but the true winrate could be 30%. Triage signal, not falsification. Friction tag: `n5-too-noisy-for-falsification`.

## Promotion candidates (per postmortem step 4b)

Surface these to `.claude/skills/kaggle-comp/improvements.md` if they recur. Today is first occurrence for all four; promote on 3rd recurrence per Rule 37 ledger discipline.

- **`cache-attempt-falsified-by-rederive-feature`** — when considering caching an expensive recomputation, profile the cost FIRST. Don't guess that "this is the bottleneck" from a high-variance metric (p95 turn time spread across heterogeneous game states).
- **`handover-recipe-mismath-vs-env`** — Phase-1 Explore agents must verify formula recipes from HANDOVER against the env's actual code. The HANDOVER author may have been working from an older / different env. Update HANDOVER.md's P2 entry in next session to reference the real bug.
- **`proposer-tightening-axis-exhausted-vs-joint-aggr`** — three falsifications on "tighten proposer pre-filter" within one session triggered Rule 37. The pattern (win-vs-quiet-opp / lose-vs-aggressive-opp) suggests pre-filter axis is asymmetrically lossy. Future filter changes should be A/B'd vs joint_aggr (or whichever strong opp anchors the rolling pair) BEFORE shipping.
- **`n5-too-noisy-for-falsification`** — the new 5×250×no-swap standard is fast triage but n=5 column results have wide Wilson CIs. Use n=5 for direction, n=16-32 for falsification.

## Predicted-vs-actual calibration (Rule 26 snapshot)

Tracking only sub 53018599 from today (the only push). Predicted band 1100-1180; actual will land 4-12h from submit (2026-05-25 11:54 UTC). To be back-filled in tomorrow's session.

Past three days' calibration:
- sub 53000996 (phi1_only): predicted 1050-1200, actual 1115.2 — WITHIN BAND.
- sub 52993021 (concentration): predicted 1100-1180, actual 1117.9 — WITHIN BAND.
- sub 52968889 (buildup_planner bundler-fix): predicted 1080-1120, actual 1142.4 — OVER (4 μ above band; predicted too low, agent was stronger than n=16 suggested).

3/3 within or above band over the last 3 pushes. No `pi-stamp-risk` trigger yet.

## Next-session first-action ranking

1. **Wait 4-12h for sub 53018599 to settle.** Per Rule 48. Don't iterate until live μ is known.
2. **If μ ≥ 1130:** K1+Z v2 confirmed; pivot to HANDOVER P0 (replay scout). The chooser-side opp model is the next axis. Consider a behaviour-cloned policy from top-50 ladder replays.
3. **If μ 1080-1130:** parity-band on the ladder; K1 alone was probably the right call. Un-stack and A/B Z v2 in isolation at n=32 against joint_aggr (NOT phi1_only) to confirm.
4. **If μ < 1080:** Z v2 was the regression source. Revert Z v2 too, leaving K1 alone. Re-submit K1-only bundle.

## Artifact pointers

- `audit/2026-05-25-consolidation-profile.md` — pre-K1 cProfile.
- `audit/2026-05-25-consolidation-review.md` — K1 finding + review.
- `audit/2026-05-25-consolidation-profile-post-K1.md` — post-K1 cProfile (-52% p95).
- `knowledge-base/thoughts/2026-05-25-k1-zv2-ship-and-axis-exhaust.md` — narrative + open questions.
- `scripts/ab_quick.py` — new A/B standard tool.
