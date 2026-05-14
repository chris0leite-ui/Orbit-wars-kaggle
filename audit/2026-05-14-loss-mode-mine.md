# Loss-mode mine — v7_pv wins vs losses

2026-05-14

## TL;DR

Pulled v7_pv's complete recent episode set (30 wins + 42 losses, sub
52630118). Ran the same Mine-4 metric on win and loss subsets. The
"don't throttle late" hypothesis from `audit/2026-05-14-eda-rollup.md`
is **largely inapplicable to our ladder** — median episode length is
180-184 steps, so the "last 100 turns of a long endgame" regime that
Mine 4 was built on doesn't actually occur in our matches.

The real gap is in the **opening 100 turns**: by t=100, our wins
already lead our losses by **+30 percentage points of ship share**
(54.7% vs 24.3%). And v7_pv's launch rate in *its own wins* is 0.44 /
turn — still **37% below** top-10 winners' 0.70 / turn. We don't
under-throttle late; we under-launch in the opening, win or lose.

This shifts Tier 1 priority: the cluster-conditional opening
templates (Mine 1 finding) are no longer just "nice headroom" — they
are the principal lever for closing the v7_pv → top-10 gap.

## Corpus

| corpus | n | source |
|---|---|---|
| v7_pv WINS | 30 | sub 52630118 episodes where v7_pv.reward = +1 |
| v7_pv LOSSES | 42 | sub 52630118 episodes where v7_pv.reward = −1 |
| top-10 WINS (reference) | 60 | `audit/2026-05-11-top-performer-fingerprints.json` |

Replays at `audit/eda-2026-05-14/own_replays/{win,loss}/ep-*.json`
(gitignored). Feature extraction in
`audit/eda-2026-05-14/mine_wins_vs_losses.py`; per-replay rows at
`audit/eda-2026-05-14/own_replays/v7pv_features.json`.

## Findings

### F1 — Games end at t≈180, not t=500

Episode-length distribution from our 72-game corpus:

| bucket | n | median | p25 | p75 | min | max |
|---|---|---|---|---|---|---|
| v7_pv WINS | 30 | 180 | 139 | 216 | 88 | 500 |
| v7_pv LOSSES | 42 | 184 | 155 | 252 | 108 | 500 |

Only 3/30 wins survive past t=400, and only 3/42 losses do.
The Mine-4 "compare last-100-turn ship share Δ" framing collapses
because for the typical game, the "last 100 turns" *is* the entire
mid-game; there is no separate endgame to throttle in.

### F2 — Games are decided by t=100

Ship-share trajectory (median):

| step | WIN | LOSS | delta |
|---|---|---|---|
| t=0   | 0.016 | 0.014 | +0.001 |
| t=100 | **0.547** | **0.243** | **+0.304** |
| t=200 | 0.941 | 0.080 | +0.861 |
| final | 1.000 | 0.000 | +1.000 |

By turn 100 the winner already holds 55% of mobile ships and the
loser 24%. Once the gap is established the runaway feedback (more
planets → more production → more ships) finishes the game inside
80-100 further turns.

### F3 — Opening launch rate is the behavioural gap

| metric | v7_pv WINS | v7_pv LOSSES | top-10 WINS |
|---|---|---|---|
| launches / turn (first 100) | **0.44** | **0.29** | **0.70** |
| opening_first_launch_turn | 4.0 | 5.0 | – |
| opening_launches first 30 | 5.0 | 4.5 | – |
| mean target distance (first 100) | 48.6 | 48.3 | **34.9** |
| mean target production (first 100) | 3.27 | 3.17 | 2.91 |

Two behavioural signatures of v7_pv vs top-10:

1. **Half the launch cadence.** Even in *our wins* (0.44/turn) we
   launch at 63% of top-10's pace (0.70/turn). In our losses it
   drops to 0.29/turn — 41% of top-10.
2. **Longer arm, higher production.** Top-10 fires shorter (35 vs
   our 48 mean distance) at slightly lower-production targets
   (2.91 vs our 3.17). The "patient long-arm" profile is a
   v7_pv-family signature; top-10's profile is "aggressive close-arm."

### F4 — Opponent-strength sanity check

TrueSkill matches us against stronger opponents when we're losing
streaks. Mean opponent rating: **WINS 1009, LOSSES 1107 (+98)**.
About half of the W/L ship-share gap is explainable by opponent
strength. But in the matched-strength band (opponents within ±100
of our μ=1064), we're **20W / 28L = 42% winrate**, so a real
behavioural shortfall remains.

Critically, **the opening-launch-rate gap (F3) is independent of
opponent strength**: v7_pv's *wins* themselves under-launch by 37%
vs top-10's wins. That's a pure agent profile gap, not a matchmaking
artifact.

## Implications for the Tier-1 plan

The original `audit/2026-05-14-eda-rollup.md` queued three Tier-1
items:

1. **Cluster-conditional opening templates** — still the right
   call, now with stronger backing. The lever is opening launch
   rate and target geometry; Mines 1/3 already mapped 4 clusters
   to template intensities. Promoting from "queue" to "next slot."
2. **Value-head don't-throttle-late instrumentation** — running in
   background (32 v7_0 self-play games, 3/32 done at write time).
   *Demoted* from headline diagnostic to sanity-check. The
   late-game phase doesn't exist in our games; whatever it shows
   will not change the Tier-1 priority order.
3. **Loss corpus** — done by this document (using our own losses,
   not top-10's; better signal since same agent under both
   outcomes).

## Panel hardening (Tier 0c)

Wired a named preset into `scripts/strategy_panel.py`:

    python -m scripts.strategy_panel --panel hardened --seeds 32

`hardened` = `[v7_0_drop_one, v3.5.1, roi, baseline]` — four
opponent classes (v7-search, v3-lookahead, aggressive simple,
comp reference). Required minimum for pre-submit calibration
per `audit/2026-05-14-postmortem-geo-session.md` (was: "panel
MUST include ≥3 opponent classes"). Resolver gained
`agents/v7_ablations/<name>/main.py` lookup so `v7_0_drop_one`
resolves cleanly.

Not built this cycle: full state-similarity bowwow-ghost
(replay-as-policy) agent. Needs a state-feature → action-template
layer that's ~2-3 hr of new code. Queued for next session if
opening-templates work outpaces the gate budget.

## Compute spent

- 72-replay pull: ~10 s (kaggle API)
- Mine 4 re-run on v7_pv corpus: ~25 s
- Opponent-rating cross-check: ~2 s
- Panel-preset smoke: ~3 min (1 seed, no self-play)
- Self-play diagnostic (background): ~7 min in (3/32), will
  finish without further attention
- Total active: ~15 min

## Open questions

- How much of the opening launch-rate gap is "v7_pv's enumerator
  doesn't *propose* enough launches" vs "v7_pv's value head
  *rejects* aggressive bundles"? Either points back to the
  enumerator/value-fn split — same code, different lever.
- Does top-10's launch cadence taper after t=100, or stay flat?
  The fingerprint corpus only computes features over the first
  100 turns, so we don't know yet. If the cadence drops, "aggressive
  close-arm" is an opening-only signature and the rest of the game
  may converge.
- Cluster-conditional templates (Mine 1) measured 4 board types;
  does each cluster have a different optimal launch *cadence*, or
  only different target geometry? Worth a follow-up mine before
  building the template overlay.
