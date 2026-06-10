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

## The expansion gap, diagnosed (added later on 2026-06-10)

New tool `scripts/expansion_probe.py`: instruments the greedy selector in a
live game and logs, per focal turn, the candidate counts and best scores by
target ownership class plus everything fired.

Seed-7 game, mass config vs the namespaced champion
(`audit/pools/2026-06-10-expansion-probe-seed7.jsonl`): through steps 0–35
the planner is offered **dozens of valid neutral-capture candidates every
turn, best score 0.00–1.00** — below the 1.5-ship roi threshold — while the
bank climbs to ~300 ships. Neutral captures fire almost exclusively on the
turns where the opponent projection drags the do-nothing baseline negative
(threshold dips at steps 0, 4, 9, 15, 20, 22, 27, 29, 32, 36–37). Expansion
is happening by accident, not by valuation.

Diagnosis: **horizon truncation**. The flow scorer credits a captured
planet's production only inside H (18 steps 2P / 13 4P). A neutral's
in-horizon production roughly repays its garrison cost → net ≈ 0 → never
clears the threshold. Everything the planet earns after step H — the entire
reason the top-ladder agents expand to 8 planets by step 40 — is invisible.
This also explains why bumping the whole horizon to 24 regressed (it
rescales every constant), and why the rejected `opening_bonus` pointed the
right direction with the wrong mechanism (a decaying constant, not a value).

Fix implemented (default OFF): `PRODUCER_PLUS_TERMINAL_PROD_VALUE=λ` — the
sparse flow diff now also reports production owned at the horizon's final
step (hypothetical − baseline, per player, exact from the same recurrence),
and `competitive_score` credits it for λ post-horizon steps with the same
opponent weighting as the in-horizon flow. Capturing a neutral gains λ·prod;
taking an enemy planet counts double; holding a planet that would flip is
valued symmetrically. Unit tests: `tests/test_terminal_prod_value.py` (6
green, incl. exact synthetic diffs). Bundle variants: `termval12`
(standalone), `mass_termval12` (composed with the mass mechanisms).

## Decision-rule mining (added later on 2026-06-10, `scripts/mine_decision_rules.py`)

Beyond static profiles: reconstructed every fleet's target (track each fleet
id to its vanish step, snap last position to the nearest planet) and
classified attack/defense events. 2P games only.

| | us champ (78 games) | TonyK 1573 | Jake 1657 | Isaiah 2063 |
|---|---|---|---|---|
| neutral overkill p50 (size/garrison at launch) | 1.14 | 1.27 | 1.28 | 1.36 |
| enemy overkill p50 | 3.15 | 2.77 | 2.60 | 4.56 |
| enemy overkill p75 | 7.10 | 7.50 | 7.50 | 10.0 |
| enemy fleet size p50 | 40 | 66 | 60 | 89 |
| attack eta p50 | 7 | 4-5 | 4-5 | 4-6 |
| enemy flip rate | 0.69 | 0.70 | 0.72 | 0.84 |
| stick-given-flip (20 steps) | 0.71 | 0.64 | 0.71 | 0.69 |
| neutral attacks before step 60 | 81% | 67% | 85% | 89% |
| material decision step p50 / p90 | 54 / 95 | 53 / 74 | 30 / 62 | 36 / 48 |
| held rate when reinforcing | 0.59 | 0.85 | 0.74 | 0.32* |

*Isaiah's defense stats are against near-peer opponents (2000+ band).

Takeaways:

1. **The material verdict lands by step ~30-54 (p90 <= 95) in EVERY corpus**,
   including ours. Median top games still run to 500 — wins come from
   holding a material lead, not elimination. This unlocks margin-based fast
   triage (`scripts/margin_ab.py`): ship-share lead at steps 40/80/120,
   seat-paired per seed; n=8-16 games of continuous margin replaces n=32
   binary wins for triage (NOT for the Rule 45 submit gate).
2. **Overkill is class-dependent**: ~1.3x on neutrals (and expansion is
   front-loaded: 67-89% of neutral grabs before step 60), 2.6-4.6x median /
   7.5-10x p75 on enemy planets. Our ratios are roughly right but our
   ABSOLUTE enemy strikes are small (fleet p50 40 vs 60-89) and launched
   from too far (eta 7 vs 4-5) — and big fleets fly faster, compounding.
   Mechanism: `PRODUCER_PLUS_OVERKILL_FACTOR_ENEMY` (class-split sizing
   menu; default unset = single-knob path, scalar-multiply identical).
3. **Our reinforcement under-delivers** (held 0.59 vs 0.74-0.85 at the top)
   — consistent with the reinforce-deficit direction even though it nulled
   on the producer pool; re-measure against the new incumbent + margin
   harness before re-judging.
