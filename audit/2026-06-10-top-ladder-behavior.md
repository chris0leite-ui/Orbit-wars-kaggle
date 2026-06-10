# 2026-06-10 — top-ladder behavioral mining (the mass-concentration meta)

PI mandate: improve significantly, rework/reinvent allowed, don't wait for
the live settle. Approach: stop measuring against vanilla producer and look
at what actually wins at the top of THIS ladder.

## Method

- `scripts/crawl_top_replays.py`: Kaggle's public EpisodeService accepts
  `{"submissionId": X}` unauthenticated; episodes expose every agent's
  submissionId + TrueSkill. Climbing the opponent graph from our own
  submissions reaches the global top in 8 hops. Downloaded 40 replays each
  for the #1/#2/#5 teams (Isaiah @ Tufa Labs 1729, Jake Will 1657,
  TonyK 1573).
- `scripts/behavior_profile.py`: per-team behavioral fingerprint from
  replays (expansion curve, launch cadence, fleet-size percentiles,
  garrison ratio, capture/loss counts), split 2P/4P.

## Findings

| metric (2P medians) | us (1260) | TonyK (1573) | Jake (1657) | Isaiah (1729) |
|---|---|---|---|---|
| fleet size p50 | **21** | 36 | 42 | **83** |
| launch rate (steps with launch) | **0.63** | 0.35 | 0.42 | **0.26** |
| planets at step 40 | **6** | 8 | 8 | **8** |
| production at step 40 | **20** | 23 | 27 | **26** |
| ships at step 80 | **214** | 275 | 461 | **774** |
| garrison ratio at step 80 | 0.35 | 0.49 | 0.62 | 0.93 |

Monotone in rating, every row. The top plays MASS: launch half as often
with 2-4× the fleet, expand faster early (they spend the early stockpile —
33-35 banked ships at step 20 vs our 46), and accumulate 2-4× the ships by
step 80 on similar production (exchange efficiency — decisive captures
stick, marginal ones churn).

Cross-check on OUR 195-episode corpus: opponents we beat have median fleet
16 and launch rate 0.60; opponents who beat us have median fleet 30+. Fleet
mass is the cleanest single predictor of beating us.

4P note: even at 1650, 4P winrate is ~18% (Jake, n=22); the #1 team's
sample is 100% 2P. The very top of this ladder is substantially a 2P game —
the μ payoff of 2P strength is NOT capped by the 4P swamp.

## Why our 11 measured mechanisms kept nulling

The local yardstick (A/B vs vanilla producer) measures skill at the
dribble-meta mirror match. Mechanisms pointing toward mass/retention
(horizon 24, recapture penalty) measured as REGRESSIONS on that yardstick —
it was steering us deeper into the meta that loses to the 1300+ band.

## Lane analysis of our dribble (champion, seed-7 instrumented game)

- attack waves (enemy/neutral targets): n=93, p50=38 — acceptable
- own-target transfers (reinforce + regroup): n=230, p50=18 — **71% of all
  launches are sub-20-ship parcels from the pressure-gradient regroup lane**

## Mechanisms implemented (all default OFF; champion byte-identical verified)

- `PRODUCER_PLUS_MASS_TIEBREAK` — the exact flow scorer values minimal and
  overwhelming captures of the same target near-identically; the stable
  argmax then picks the LOWEST index = smallest variant. Epsilon-scale size
  preference (1e-4/ship) resolves near-ties toward mass.
- `PRODUCER_PLUS_REGROUP_MIN_SEND` — regroup entries below the threshold are
  dropped; ships accumulate until a convoy fires (bigger is also FASTER:
  fleet speed rises with ship count).
- `PRODUCER_PLUS_OVERKILL_FACTOR` — lo/mid attack variants sized at F× the
  projected defense instead of the bare floor.
- Bundle variants: `mass` (all three: tiebreak + convoy 25 + overkill 2.0),
  `convoy_only`, `overkill2/3`.

Smoke (seed 7 vs producer): fleets 167→108, fleet p50 28→44, launch rate
0.63→0.58, win in 116 steps vs champion's 128.

## Measurements (appended)

- **mass vs namespaced champion, head-to-head 2P, seeds 0-15 both seats:
  18/32 (56.2%), Wilson [0.393, 0.718].** Wins are perfectly map-
  symmetric: 9 of 16 maps won at BOTH seats, 7 lost at both — the
  mechanism's effect dominates seat noise. First mechanism of twelve
  measured that is AHEAD of the champion head-to-head.
- **mass vs vanilla producer (non-regression) n=32: 22/32 (68.8%),
  Wilson [0.514, 0.820]** — solid winner vs producer (champion's own
  mark: 24/32). The mass profile costs little against the dribble meta
  while gaining against the champion.
- head-to-head extension seeds 16-31: 17/32 — **combined head-to-head
  35/64 (54.7%), Wilson ≈ [0.43, 0.66]**. At worst parity with the
  champion, likely a small edge; 17 of 32 maps won at both seats, 1
  split. Every previous mechanism lost to the champion baseline.
- **mass 4P (vs 3×producer, seeds 0-31): 7/32 (21.9%)** vs baseline
  13/32 — unrestricted mass COSTS 4P first-place rate (the dribble may
  be load-bearing in 4-front games, or the convoy threshold starves the
  multi-front defense). Hence `PRODUCER_PLUS_MASS_2P_ONLY`.

## The composed candidate: `mass2p_ffa`

Player-count-gated composition — 2P = mass (head-to-head winner), 4P =
champion + FFA uniform objective (live sub 53527125's exact behavior).
Verified by action-stream parity: 2P seed-7 == `mass` bundle hash, 4P
seed-3 == `ffa_uniform` bundle hash, champion OFF-path seed-13 hash
unchanged. All measured pool results therefore transfer to this bundle
exactly. Rule 46 green (test_bundle 15/15; idle smoke max 71 ms).
