# 2026-06-12 — Shot-success MLP: offline counterfactual on our live launches

**Context.** The May shot-validator plan (37k labeled examples, MLP never
trained) was resurrected on PI sign-off and modernized to the Producer era:
286,618 launches labeled from 634 live-ladder episodes of 7 recent
submissions (all seats), 24-feature encoder unchanged (FEATURE_VERSION 1),
fleet-speed curve fixed to match the engine (1-ship floor, 1000-ship cap —
the May copy over-sped >1000-ship fleets, skewing eta features and labels).

**Model.** 24-32-16-8-1 MLP, ReLU hidden, sigmoid head. Grouped split by
episode (no leakage): val AUC **0.871** on 43,778 held-out examples.
Decile calibration near-diagonal (lowest decile predicted ≤0.06 →
empirical 0.044; top decile ≥0.96 → 0.974). numpy-vs-torch round-trip
max diff 1.2e-7. Weights baked into `agents/producer_plus/shot_mlp.py`.

## The counterfactual

Score every launch OUR agents actually made in the downloaded episodes
(`focal_team == ChrisLeiteScha`, n = 107,762 launches):

| sub | n launches | base success | P<0.30 share | success when P<0.30 |
|---|---|---|---|---|
| 53324164 (old champion line) | 11,639 | 0.705 | 16.4% | 0.292 |
| 53384340 (multi_opp_def, μ1285) | 27,545 | 0.604 | 27.0% | 0.132 |
| 53527125 (ffa_uniform) | 10,204 | 0.585 | 27.6% | 0.116 |
| 53542171 (veto2p_ffa) | 13,576 | 0.571 | 31.3% | 0.145 |
| 53547475 (vetorf2p_ffa, μ1292 peak) | 12,782 | 0.619 | 23.5% | 0.131 |
| 53564198 (vetorf4p_seq_strength) | 16,592 | 0.631 | 23.2% | 0.135 |
| 53577315 (vetorf4p_sync, live pair) | 15,424 | 0.633 | 23.7% | 0.136 |

Pooled: launches with model P < 0.15 are **16.9%** of everything we
launch and succeed **7.9%** of the time (kept: 72.9%). P < 0.30 is
**25.0%** of launches succeeding **14.5%** (kept: 77.7%).

**Read.** A quarter of our real ladder launches are near-certain failures
by a well-calibrated model of "target still ours 10 turns after arrival."
This matches the 2026-06-10 live mining (30% of capture-sized attacks fail;
65% of failures die to in-flight reinforcement). Caveat: "failed hold"
is not identical to "bad launch" — some low-P launches are deliberate
tempo/denial taps; whether vetoing them wins games is what the A/B
measures, not this table.

## Mechanism + measurement notes

- In-agent veto `PRODUCER_PLUS_SHOT_MLP=<threshold>`: reject-only on
  ATTACK waves, runs after the response veto / redirect, default OFF,
  byte-identical when unset.
- Bundle threshold is **hardcoded into the gate function at build time**,
  not baked as an env var: baked `setdefault` env vars are process-global
  and leak into a same-process opponent bundle (the 2026-05-22 env-leak
  friction) — with an env-var gate, candidate-vs-control silently becomes
  mirror-vs-mirror. First two "A/B" games of this session were exactly
  that before the hardcoding fix.
- Bundler prelude modules are exec'd into a synthetic module object
  (namespaced), NOT flattened: shot_mlp's float `fleet_speed` shadowed
  orbit_lite's tensor `fleet_speed` in the flat bundle namespace and
  crashed every game at step 1 (both bundles, since the prelude inlines
  for every variant).
- Vendor pin: producer_plus copied from `awesome-clarke` @ `be71c97`
  (the live garval submission's build commit), so the A/B control
  byte-matches the ladder agent.

## A/B (running at write time)

n=32 each, seat-balanced, subprocess workers, single-thread torch:
- ARM 1: `pp_sync_shotmlp_on` (threshold 0.30) vs vanilla `producer`
- ARM 2: `pp_sync_control` (live vetorf4p_sync config) vs vanilla `producer`

Referee choice: vanilla producer is the branch-standard referee for all
producer_plus mechanism A/Bs (immune to env leakage, non-family enough
that the response veto's mirror is imperfect). Results to be appended.

## Results (appended)

**A/B verdict: parity, and provably uninformative locally.**

| arm | result vs vanilla producer (n=32, seeds 0-15 × both seats) |
|---|---|
| candidate (threshold 0.30) | 21/32 = 65.6%, Wilson [0.483, 0.796] |
| control (live config) | 22/32 = 68.8%, Wilson [0.514, 0.820] |

Instrumented firing counts explain the parity — the filter is near-inert
in the local ecosystem:

- full game vs vanilla producer: 68 attack waves scored, **0 dropped**
- full game vs the old champion (rebuilt byte-identical to live pin
  `6c0419dc20`) + one 4P game vs 3× producer: 90 waves scored, **1
  dropped** (p=0.26); candidate won both games

Locally, the planner + response veto (whose 1-ply mirror models
family/local opponents accurately) never emits sub-0.30 attacks. On the
live ladder it demonstrably does: **49.5% of our real attack launches
score P<0.30 and succeed 13.9%** (attack base success 40.1%). Split by
format: 2P 45.5% low-P (succeed 11.8%), 4P 54.7% (succeed 16.1%) — not
a 4P artifact. By class: 93% of low-P waste is attacks (the veto's
target); doomed-garrison reinforcement donations are only 6.8%.

**Conclusion.** The mechanism is built, calibrated, leak-proofed, cheap
(~15 ms p50 turn overhead), and harmless locally. The behavior it
targets exists only against the live field — no local referee in our
zoo reproduces the conditions (real opponents' baits / reinforcement
walls / multi-front sieges). The only informative test is a live probe,
which costs a submission slot and PI sign-off (Rules 1/12/42). If
probed live, threshold **0.15** is the conservative first experiment
(vetoes only the 7.9%-success tail, ~17% of all launches) — 0.30 would
alter roughly half of all live attack behavior in one step.

Rule 46 state for the candidate bundle (`pp_sync_shotmlp_on.py`):
bundler GREEN, `tests/test_shot_mlp.py` 3/3 GREEN, full-game smoke
GREEN (turn p50 84 ms, max 381 ms < 1000 ms).

## Live probe (appended)

Submitted as **sub 53595717** (2026-06-12 08:43 UTC, threshold 0.15,
sha256 03ce7fe729ba64a6…, 410 225 B) on PI sign-off, re-confirmed after
the board change (evicts garval 53588922 mid-warmup; the μ≈1240
coalition sub was already evicted by oracle_rw 53594710 at 08:12).

**How to read the verdict (next session):**
1. Settled μ vs the ≈1258 anchor of the identical-base sub 53577315
   (TrueSkill warm-up: ~600 → settles over ~24 h; do not read early).
2. The direct mechanism read: pull this sub's episodes
   (`python -m scripts.live_episode_summary 53595717 --pull`), relabel,
   and compare against the 53577315 baseline — low-P(<0.15) attack share
   should collapse from ~16% toward ~0 (vetoed pre-launch), overall
   attack success should rise from 0.42, and ships-wasted-per-game
   should drop. That measurement is independent of ladder noise.

## VERDICT — 2026-06-13 (~26 h settled, n=119 field episodes)

**Settled μ:** probe 53595717 = **1263.0**; no-filter anchor 53577315 =
**1241.6** (+21.4, within ~1σ — not attributable to the filter).

**Mechanism read (the outcome-independent metric):** the filter changed
nothing about our live launch distribution.

| metric | PROBE 53595717 | BASE 53577315 |
|---|---|---|
| field episodes (self-matches excluded) | 119 | 92 |
| attack launches / episode | **70.7** | 67.7 |
| attack share model-P<0.15 | **33.4%** | 34.8% |
| attack success rate | 42.1% | 41.0% |
| ships lost to failed attacks / ep | 1959 | 1681 |

The intended signature (low-P share collapsing toward 0) did **not**
happen, and attacks/episode went **up**, not down. (Yesterday's n=1 peek
showing 3.6% / 87.5% was a single won game — noise, as flagged.)

**Root cause — delay, not prevention.** A reject-only filter in front of
the producer planner does not remove a bad launch; the planner is
memoryless across turns, so a wave dropped at turn T is re-proposed at
T+1, T+2 … until the board drifts enough that the *same* launch clears
0.15, then it fires (a few turns later, often against a target that has
moved — hence wasted-ships slightly *worse*). Net behaviour: the low-P
attacks still happen, just later. This is the same architectural lesson
as the mechanism-ledger graveyard: stacking a veto in front of a
re-proposing planner is absorbed by re-proposal.

**Encoder fidelity (ruled out as the cause):**
`scripts/diag_shot_target_attribution.py` — the offline ray-cast target
agrees with the planner's true target **84.4%** of the time (45 waves,
local). The offline metric is mostly faithful; the flat read is real,
not a measurement artifact. Reconfirmed referee blindness: **0%** of
locally-scored attacks are below 0.15, so no local A/B can ever move
this mechanism.

**Disposition:** shot-validator is a **null** as a standalone reject-only
filter — local AND live. Do **not** escalate to 0.30 (it would delay
*more* attacks, not prevent them). The probe (1263) is a harmless null
and is fine to leave in the rolling pair as the high half.

**The only path that could realize the offline counterfactual's promise:**
pair the MLP veto with **redirect** (`PRODUCER_PLUS_REDIRECT`) so freed
ships are re-aimed at a target the model likes, instead of re-proposing
the same bad shot — turning "delay" into "substitute." Unproven, costs a
live slot (referee blindness), and counter-indicated by the ledger's
history of front-of-planner stacking. PI decision, not auto-pursued.
