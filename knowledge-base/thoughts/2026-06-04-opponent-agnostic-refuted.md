# Opponent-agnostic scoring is refuted FOR OUR CHOOSER (2026-06-04)

The "be like Producer — drop the opponent model, score opponent-agnostically"
direction is **dead for our agent**, settled by n=32 clean A/Bs this session.

## What we tried

Two gated, default-OFF mechanisms, both composing the public Producer agent's
opponent-passive value lens into our own `score_candidate_v4` rollout:

- `BASELINE_OPP_PASSIVE=1` — opponents launch nothing in the rollout (they still
  produce + resolve in-flight fleets). Producer's frozen-opponent assumption.
- `BASELINE_VALUE_HEAD=net_swing` — leaf = `my_ships − Σ_opp ships` (on-planet +
  in-flight), horizon pinned to Producer's 18/13, favor-tuned post-leaf stack
  bypassed. With the passive flag this is algebraically Producer's
  `competitive_score` (proven by a conservation unit test), but computed through
  the REAL engine.

## The numbers (process-isolated `clean_ab`, n=32)

- net-swing lens vs **Producer**: 21.9% — bought nothing vs Producer.
- net-swing lens vs **our champion**: 37.5% — worse than `favor`.
- **passive-opponent ALONE vs champion: 31.2%** — the disambiguator.

## What it means

The **opponent-agnostic assumption is the culprit, not the leaf.** Passive-alone
already loses (31.2%); the net-swing leaf on top is marginally better (37.5%) but
still loses. Our rollout's reactive `lite_greedy` opponent provides a
**recapture penalty** that is load-bearing — it stops the chooser committing
ships to captures that get retaken. Drop it and we overextend. (This is exactly
the risk flagged when the spike landed.)

Producer wins *despite* being opponent-passive — because of its candidate
generation / selection / hold-reserve machinery, NOT because opponent-agnostic
scoring is itself better. The 3%-vs-78% `producer_lite` result identified "the
scorer" as the edge, but that was producer_lite with NO rollout at all; our
chooser already forward-simulates, so transplanting only the lens does not
transfer Producer's edge.

## Standing implication — DO NOT re-open

- The opponent model in our rollout (`lib.opp_model.lite_greedy_policy` via
  `opp_actions_for_snap`) is **load-bearing — keep it.** The earlier plan to
  "delete `lib/opp_model` on success" is void.
- `BASELINE_OPP_PASSIVE` and `BASELINE_VALUE_HEAD=net_swing` remain in the tree,
  default-OFF (champion byte-identical) — confirmed-negative levers, kept only as
  reproducible evidence. Do not bake either into a bundle.
- If Producer's edge is ever revisited, the hypothesis to test is its
  **candidate generation / greedy selection / hold-reserve**, NOT its value lens.

## Corrected during the session (worth keeping)

The earlier doc claim that our champion is "a static one-shot scorer that never
simulates forward" was **wrong** — the live champion (`score_candidate_v4`)
already rolls K ticks through the real engine with a reactive opponent and a
`favor` leaf. That correction stands regardless of the negative result.
