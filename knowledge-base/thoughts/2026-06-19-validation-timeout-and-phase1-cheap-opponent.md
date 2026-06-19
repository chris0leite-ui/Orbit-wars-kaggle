# 2026-06-19 — the depth-3 validation timeout, and Phase 1 (cheap rollout opponent)

## The observation
Sub **53836276** (`lr_depth3.tar.gz`) came back **ERROR** on Kaggle. Pulling the
real error field from the API (`competition_submissions(...).error_description`):

> **"Validation Episode failed."**

Distinct from the earlier ERROR on 53768768, whose field was **"Internal scoring
error"** — that one was an infra flake (its content-identical resubmit 53772947
COMPLETEd at 1098.1). "Validation Episode failed" is a *genuine agent failure*:
Kaggle runs one **self-vs-self validation game** (4 copies of our agent) before
admitting a submission to the ladder; per-turn cap is **1 s**, drawing on a **60 s
overage pool**, and once that's drained a slow turn **errors the agent out**.

## The diagnosis — a per-turn timeout, not a crash
The deep-search call in `agent()` is wrapped in `try/except → greedy fallback`, so
a code exception can't produce this error. The only uncaught failure is the
wall-clock. `_deep_pick` re-runs the **producer mirror** (`_producer_move_obs`,
torch, ~10-50 ms/node) at every seat × every rollout turn × every candidate, and
its time guard is checked *before* each rollout (the first rollout always runs), so
a turn costs ≈ `budget (700 ms) + one full depth-3 rollout`. Self-vs-self games are
balanced → long → crowded late boards → the most expensive rollouts; on the slower
validation slot those tip past 1 s, drain the overage pool, and the agent errors.

This is **exactly the wall `state/DROPOUT_NATIVE_DESIGN.md` predicted**: "depth is
capped by the 1000 ms wall because each node re-runs the producer mirror."

## Two process lessons
1. **torch was missing locally** (not in `requirements.txt`; this container is
   fresh). Without it the agent's `_ORBIT_OK` block falls back to a pure-Python
   path AND `agent()` gates the whole deep search on `orbit is not None` — so the
   deep search never even runs. My first "reproduction" was therefore the degraded
   fallback, not the real agent. **Fix:** installed CPU torch and folded the
   install into `bootstrap.sh` (idempotent, non-fatal) so it stops recurring.
2. **The Rule-46 smoke under-tested the failure mode.** It played `--vs
   v7_0_drop_one` (a cheap, short game) on fast multi-threaded hardware. The real
   validator is a 4-seat self-vs-self *full* game on a slower slot. The smoke now
   reproduces that scenario (single-BLAS-thread self-vs-self, assert max turn well
   under the wall).

## The fix = the strategy's Phase 1 (cheap opponent), not a budget cut
A smaller time budget would fight the strategy's thesis ("search depth converts
compute→strength"). The right move — and the plan's own next step — is a *cheaper
per-node opponent*: knob **`LR_DEEP_OPP`** in `agents/least_resistance/main.py`.
`0` = producer mirror (byte-identical default); `1` = the existing cheap
`lite_greedy_policy` (`lib/opp_model.py`, ~1-2 ms/node, models expansion). The
strong torch leaf scorer `_project_value` is unchanged — Phase 1 swaps only the
opponent model. A cheaper node makes even the first rollout fit under the wall AND
is what lets depth grow to 5-6. **One mechanism fixes the timeout and advances the
strategy.**

Landed this session: the knob (`_deep_opp` + `_deep_opp_move`, wired into both
opponent call sites of `_deep_pick`, mode read once per turn), a torch-agnostic
unit test (`tests/test_deep_opp_dispatch.py`), and the timing smoke. **No submit.**

## Timing smoke (real torch path, 4-seat self-vs-self, bank off, seed 1492346051)
Per-turn ms over a capped game (multi-thread fast box — treat as a lower bound;
apply a slow-slot factor, but the margins are enormous):

| config | mean | p95 | MAX | turns >1 s |
|---|---|---|---|---|
| mirror depth-3 (early game, steps <16 only) | 192 | 239 | **293** | 0 — but grows late-game → ~739 ms dev / >1 s on Kaggle's slot = the failure |
| lite depth-3 (Phase 1) | 28 | 42 | **48** | 0 |
| lite depth-5 | 29 | 43 | **50** | 0 |
| lite depth-6 | 29 | 47 | **80** | 0 |

The cheap opponent is ~6–7× cheaper *even in the early game*, and — unlike the
mirror — its cost stays **flat from depth 3 to 6** (the torch leaf scorer
dominates; the opponent is nearly free). Phase 1 removes the timeout and opens the
depth headroom the strategy wanted. (Engineering note: single-BLAS-thread + full
self-vs-self games were too heavy/slow for this box's background runner — they got
killed; capped multi-thread games gave the clean read above.)

## Open question for the kill-gate (next session)
`lib/opp_model.py` warns lite_greedy is "too attack-biased" (in a *different*
consumer — the baseline chooser's ME-defense baseline). As the *opponent* model in
LR's rollout that may be fine or may distort the search. The kill-gate settles it:
`LR_DEEP_OPP=1` × depth {3,4,5,6} vs Producer V2 (`scripts/eval_panel.py`), triage
n=16 → confirm n≥32 (Rule 45); **pass = cheaper-deeper ≥ mirror depth-3 (17/28).**
Only on a pass do we submit (header bakes `LR_DEEP_OPP=1` + winning depth, bank off,
Rule 42 claim, PI sign-off).
