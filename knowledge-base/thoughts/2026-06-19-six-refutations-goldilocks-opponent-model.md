# 2026-06-19 — six refutations, and the Goldilocks opponent model

**Branch:** claude/kaggle-dropout-strategy-2hk9wr
**Benchmark:** Producer V2, 2P, seeds 6000–6031 (n=32). OFF baseline (shipped
`least_resistance`, 2-ply + simultaneous-producer opponent model): **20/32,
margin +319, worst game −3402.**

## The arc

Observation-driven iteration to push past the local optimum vs V2. Six mechanisms,
each built default-OFF, smoked (bundle + parity + timing), A/B'd at n=32, then
compared to the recorded OFF baseline. **Every one refuted.**

| mechanism | what it changed | result vs V2 (n=32) |
|---|---|---|
| dropout (prior session) | leaf penalty for exposure | refuted earlier |
| hold-search (tuned) | extra decisive-hold candidate plans | 19/32, margin −142, tail −9387 |
| win-equity leaf | control share `(ours−theirs)/(ours+theirs)` not raw margin | 19/32, +278, tail −3849 |
| robust multi-reply search | worst-case over {producer, lite_greedy} replies | 17/32, −286, tail −6994 |
| V2 as the search opponent | model the *actual* opponent | 13/32, −802, tail −5981 |
| best-response adversary | producer reacts to my plan (Stackelberg, tougher) | 13/32, −959, tail −5469 |

## The finding: the opponent model is a calibrated Goldilocks point

The three opponent-model experiments line up cleanly and **non-monotonically**:

- **Too weak** (model the real, sparse V2): 13/32. The search predicts an idle
  opponent on most turns (V2 only launches on a high-ROI affordable target), sees
  no punishment in the look-ahead, and picks **complacent, over-extended** plans
  that V2's eventual sparse launches punish.
- **Calibrated** (simultaneous producer — a denser, stronger sparring partner than
  reality): **20/32**. Steady predicted pressure forces robust, defensible plans
  that generalize and beat the weaker real V2.
- **Too tough** (best-responding/clairvoyant producer): 13/32. Every plan looks
  punished, so the search can't discriminate, plays **over-cautiously**, and gets
  ground down.

So the lever is **not** "model the true opponent" (worse) and **not** "model the
strongest possible adversary" (also worse). It is a calibration sweet-spot, and
the shipped agent already sits on it. Too weak → complacency; too tough →
paralysis; both regress to ~13/32 from opposite ends.

Correcting the mid-session mistake: I first read the producer-beats-V2-model gap
as "strength of the adversary is what matters" and recommended a *stronger*
adversary. The best-response result refuted that — it's a sweet-spot, not a
monotone. Lesson recorded in the postmortem.

## What this says about the agent

Six diverse perturbations — three to leaf scoring (dropout, hold-search,
win-leaf), three to the opponent model (worst-case set, weaker, tougher) — all
regress or no-op. That is strong evidence the shipped **2-ply + simultaneous-
producer + margin-leaf** agent is a **robust local optimum vs V2**, not a
way-station. We already beat V2 (~20/32 ≈ the "+7 in 2P" STRATEGY figure); the
neighborhood around that config is mapped and downhill in every direction tried.

## Open questions for next session

- Every refuted lever stayed *inside* the 2-ply + producer-eval frame. A move off
  this local optimum likely needs a **different frame**, not another bolt-on:
  deeper search that actually pays for itself, a learned/value-head leaf, or a
  genuinely different candidate-plan generator — not leaf reweighting or opponent
  swaps.
- 4P is unmeasured for all six (only 2P A/B'd). The win-leaf 4P caveat still
  stands: by default the 4P leaf scores share-vs-the-field-sum, which is not the
  4P win condition; the correct framing needs `LR_LEADER_RELATIVE_4P=1`
  (share/margin vs the single strongest rival). If 4P is ever the lever, fix that
  first.
- All six knobs remain in the code, default-OFF and documented, as a mapped
  refuted-lever neighborhood. Don't re-test them without a new reason.

## Side outputs

- Fixed pre-existing bundler drift: `lib.kinematic_table` was missing from the
  inliner's `DEFAULT_LIB_ORDER` (trajectory.py lazy-imports it). `test_bundle.py`
  4 passed/6 errors → 11 passed.
- New tool `scripts/render_game.py`: one game → self-contained HTML replay viewer,
  2P or 4P (`--agents` comma list), agent env-knobs applied before load.
