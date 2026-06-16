# Inverse-producer study — short (200-step) games

> ## ⚠️ CORRECTION (2026-06-16) — the head-to-head results below are INVALID
> A contamination bug invalidated every inverse-vs-static head-to-head in this
> doc (the 200-step "tie", the full-game "collapse"). The producer_plus bundles
> set their flags via `os.environ.setdefault` at import; `os.environ` is
> process-global and read per-turn, so loading the inverse focal (sets
> `OPP_PROJECTION=1`) and the "static" opponent in ONE process leaked the flag
> into the opponent — **both seats ran opp_projection**. So "inverse vs static"
> was really inverse-vs-inverse → forced mirror draws.
>
> **Fixed** with per-turn env isolation (`scripts/iso_game.py`). Corrected
> result: inverse **crushes** a genuinely static producer (seed 0: 807-15;
> seed 1: 673-0, static eliminated by step 102) — the opponent model is hugely
> valuable, as expected. The static-vs-static control, the decision-diff (run in
> separate processes), and the v7_0 comparison (v7_0 ignores `PRODUCER_PLUS_*`)
> were NOT contaminated and stand. See the 2026-06-16 kb entry for the corrected
> write-up.

**Date:** 2026-06-15 (night session, branch `claude/affectionate-newton-19kqrp`)
**PI ask:** iterate on the "inverse producer" — like our producer, but instead
of assuming a static opponent it assumes a *producer* opponent that maximises
its own ships over 18 steps (assuming us static), then we maximise against that
set of actions. Test only in short 200-step games, 4 seeds.

## What the inverse producer already is

The mechanism already exists: `PRODUCER_PLUS_OPP_PROJECTION=1`
(`agents/producer/orbit_lite/opp_projection.py`, wired in
`agents/producer_plus/main.py`). With the single flag on (the bundler's
`opp_proj` variant), each turn it runs *our own* `plan_lite_waves` from each
opponent seat with `background=None` (opponent assumes we do nothing) over the
producer's native horizon (18 in 2P), and injects the opponent's predicted
launches as mixed-owner background into our per-candidate flow scorer — so every
candidate is scored against "I do X **and** the opponent does their predicted
thing." This is exactly the PI's spec. It is also already part of the live 1280
champion (`vetorf4p_seq_strength`), composed with veto / reactive-floor /
reply-seq / FFA.

"Our producer" (static-opponent) = producer_plus with all flags OFF, which is
action-stream-identical to the vendored `agents/producer/` (verified, 60 turns).

## Method

- `scripts/short_margin_ab.py`: caps the episode at 200 steps and scores each
  truncated game by the competition metric read from the step-200 observation
  (ships on owned planets + ships in owned fleets). Reports binary outcome AND
  the continuous margin. Every seed played at both seats (seat-bias control).
- Matched A/B: `bare` vs `opp_proj` (and variants) bundles from the SAME
  `producer_plus/main.py` — the only behavioural difference is the opponent
  model. New `bare` variant added to `scripts/bundle_producer_plus.py`.
- `scripts/inv_decision_diff.py`: replays one shared observation trajectory
  through the producer with opp_projection OFF vs ON and counts how often / when
  the chosen action actually changes (isolates the decision-level effect from
  trajectory divergence).

## Results (focal vs matched static producer, 200-step, seeds 0-3 x 2 seats = 8 games)

| variant | wins | mean margin | median | notes |
|---|---|---|---|---|
| control: static vs static | 2/8 | +0.0 | 0 | seeds 0,1 perfect draws; 2,3 seat-1 decided |
| inverse_producer (opp_proj, K=1, H18) | 2/8 | +0.0 | 0 | **identical pattern to control** — no edge |
| inverse_multisize (multi_size + opp_proj) | 0/8 | **-32.2** | 0 | **regression** — max margin +0 (never ahead, even at favourable seat) |
| inverse_h24 (opp_proj, shared horizon 24) | 0/8 | **-3.2** | 0 | mild **regression** — max margin +0 |
| inverse_denial (opp_proj + denial w0.01) | 1/8 | +0.0 | 0 | also ties; turned seed 2 into a perfect draw, amplified seed 3's seat skew |

(inverse_k2 — opponent plans 2 ticks — was started but killed before completing;
not re-run because the instrument is structurally insensitive here, see below.)

Per-seed structure for control & inverse_producer is **exactly seat-antisymmetric**
(P0 margin = -P1 margin) and **no agent wins both seats of any seed**: seeds 0,1
are perfect producer-vs-producer mirror draws through 200 steps; seeds 2,3 are
decided by seat geometry regardless of agent. The inverse producer *changes the
trajectories* (seed 2 became an early-term blowout; seed 3 went blowout→close)
but never flips who wins.

## Diagnosis — why K=1 inverse ≈ static

Decision-diff (opp_projection ON vs OFF on one shared observation stream):
**seed 0 (draw): action differed on 22/199 turns (11%), first at turn 48.
seed 3 (contested): 35/191 turns (18%), first at turn 41.** Contest raises the
change rate and pulls the first change a little earlier, but the opponent model
is a literal no-op through the entire opening (~first 40 turns) and changes the
chosen action only ~1-in-6 turns even when territories collide. Against a mirror producer on a (4-fold-)symmetric board the
opponent's predicted launches are almost all on *their own side* — uncontestable
(we are farther from their neutrals) — and the midline is symmetric, so a
near-perfect opponent model unlocks **no profitable deviation**. Best-responding
is almost always identical to ignoring the opponent.

The instrument is not blind: it cleanly flagged `inverse_multisize` as a
regression (the agents differ enough that board symmetry no longer cancels). It
is specifically that K=1 opp_projection is *too close* to the static producer to
register.

## The deeper point

Modeling the opponent as a producer is *exactly right* against a producer — but
against the mirror on symmetric seeds that correctness yields no edge (symmetry),
and against a non-producer the model is *wrong*. The opponent model only pays off
in the narrow regime where the opponent is producer-like yet exploitably
asymmetric. This is consistent with opp_projection living in the champion (the
real ladder is producer-like AND asymmetric) while the bare mechanism shows no
edge in symmetric self-play.

## Does exploitation help? No.

`inverse_denial` (credit captures of targets the opponent is predicted to grab —
race for contested neutrals) also ties: 1/8, mean +0.0. It stabilised seed 2 into
a perfect draw and amplified seed 3's seat skew — net zero. So even an explicit
"exploit the prediction" term does not break the mirror in 200 steps.

## The measurement is structurally insensitive (the real lesson)

The competition boards are 4-fold symmetric by design (fairness). In symmetric
self-play, head-to-head A-vs-B at both seats is forced toward antisymmetric
margins (P0 = -P1), so the mean is ~0 unless one agent is strong enough to
overturn seat geometry within the truncation window — which 200 steps does not
allow (seeds 0,1 are still balanced draws at step 200). So a 200-step
inverse-vs-mirror-producer A/B **cannot** surface the opp-model's edge even if it
exists; it can only catch large regressions (which it did: multisize, h24).

This is consistent with opp_projection being a real contributor on the live
ladder (it is baked into the 1280 champion `vetorf4p_seq_strength`, and the
`multi_opp_def` stack measured 24/32 = 75% vs producer in **full** games) while
showing **no edge** in short symmetric self-play. The mechanism's value is a
slow-burn against asymmetric opponents, not a short-game mirror advantage.

## Phase 3 — vs a non-mirror opponent (v7_0), 200-step, 8 games

| | wins | mean margin | median |
|---|---|---|---|
| static producer vs v7_0 | 8/8 | +2393.8 | +1577.5 |
| inverse producer vs v7_0 | 8/8 | **+3250.2** | **+1750.5** |

vs a non-mirror opponent the focal wins BOTH seats (no antisymmetry — strength,
not seat, decides), so the metric is sensitive here. The inverse producer beats
v7_0 by a larger margin (+857 mean / +173 median). BUT: both crush v7_0 8/8
(elimination by ~step 100), the margins are blowout-dominated and noisy, and
per-seed it is mixed (inverse higher on seeds 0 & 3-P0, lower on seed 1). A faint
positive hint, not a clean signal — and confounded by v7_0 not being a producer,
so the opponent model is technically MIS-specified yet still nets positive.

## Bottom line + recommendation

The inverse producer is faithfully implemented (it provably changes 11-18% of
decisions) but **200-step games cannot measure its marginal value**:
- vs the mirror producer → symmetric board forces a tie (null instrument);
- vs anything weaker (v7_0) → ceiling, both win 8/8 by elimination.
There is no available 200-step instrument where the edge is both present and
cleanly measurable. The historical evidence for the mechanism's value
(`multi_opp_def` 24/32 vs producer; opp_projection in the live 1280 champion) is
all from FULL games, where the slow-burn edge compounds past the symmetric
opening.

Options to actually evaluate it: (a) full-length games vs the producer (the
validated instrument — but the PI constrained this study to short games), or
(b) a strength-matched non-mirror opponent for short games (hard to source; the
producer is near the top of the field). Recommend surfacing to the PI before
spending more compute on 200-step mirror A/Bs.

## Latency note (flag)

opp_projection roughly doubles per-turn cost; mid-game max turns hit 1.8-2.6 s
locally (over the 1 s Kaggle cap) under CPU contention. The live champion carries
opp_projection within budget on Kaggle hardware, but multi-tick K>1 would be
riskier. Watch for ladder timeouts on any multi-tick variant.
