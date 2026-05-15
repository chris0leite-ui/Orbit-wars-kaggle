# 2026-05-15 — Additive-candidate + hand-designed-leaf falsifications

Session: `claude/competitive-programmer-setup-LHyoP`. Three falsifications
under a clean experimental harness. iter_v2 (#52678866) still PENDING on
ladder during these probes; no live submissions.

## 1. Multi-step plan ROI scorer (MSP) — DORMANT

Commit `089431b`. Built 4 templates (saturation_strike, near_chain,
high_prod_chain, cluster_complete) + analytical scorer (extends
`_score_phase1_analytical` to support future-turn launches via
`simulate_planet_timeline`) + orchestrator that emits the first-turn
action of the best-scoring plan as ONE candidate in
`_choose_two_phase`.

**Outcome:** orchestrator returned None in 0/40 turns of self-play.

**Root cause:** plan templates' first-turn action either (a) is empty
(chain templates schedule launches at future fire_turns) or (b) scores
identical to incumbent (saturation_strike at multi-planet states
converges to incumbent's single-source launch because one source
usually has enough ships to solo-capture). MSP never produces an
action that beats incumbent's analytical Phase-1 score.

**Architectural read:** the multi-turn VALUE of a plan lives in
turns 1..K; emitting only step-0 launches into a K=10 chooser that
plays its own fixed follow-up policy LOSES the multi-turn signal.
Same fundamental issue as `choose_depth2` (v7_2 −31pp) and mission
persistence (−42pp).

Knob: `MULTI_STEP_PLAN_ENABLED = False`. Code remains as scaffolding.

## 2. Geo allocator candidate — REGRESSION

Commit `d3641fb`. Wired `lib.geo.allocator.allocate_greedy_multi` as
ONE additional candidate in iter's drop-one list. Generates joint
multi-launch actions under posture-aware reserves (OPENING/EXPAND=0,
DEFEND=threat_budget per-planet).

**8-seed A/B vs v7_0 (paired seeds):**
| Config | Winrate | Per-seed |
|---|---|---|
| baseline (geo OFF, TP OFF) | 6/8 = 75.0% | `[1,1,-1,-1,1,1,1,1]` |
| TP-only (geo OFF, TP ON) | 6/8 = 75.0% | `[1,1,-1,-1,1,1,1,1]` |
| both ON | 5/8 = 62.5% | `[1,1,-1,-1,1,-1,1,1]` |

**Decomposition:** TWO_PHASE alone is byte-identical to baseline. The
seed=5 flip (+1 → −1) is caused entirely by the geo candidate. Real
signal, not noise.

**Root cause:** chooser's K=10 leaf scorer (composite_capture_value)
misranks the geo candidate. Pure-additive doesn't save you when the
gating function (leaf score) is itself noisy. The geo allocator
proposes a multi-source over-extension; chooser sees the immediate
capture, scores high, picks it; deeper consequences play out badly.

**Audit observation that contradicts the PI's framing pre-A/B:** the
PI hypothesised "we start too late, garrison too high in opening."
Live-game audit of iter_v2 disproved this — median first launch at
step 3.3, 12.9 ships vs opponents' 11.5, 1.7 ships left at home in
steps 0–5 (7.5× ratio launched-to-kept). The geo candidate's failure
is NOT about garrison sizing; it's about over-extension into the front.

Knob: `GEO_ALLOCATOR_CANDIDATE_ENABLED = False`. Code remains as
scaffolding.

## 3. Cluster-aware leaf head — REGRESSION

Commit `a5bf9b9`. Added `cluster_value` and `composite_plus_cluster`
to `lib/value_heads.py`. Per-cluster (not per-planet) production-time
weighted by weakest-link defensibility (min threat_eta across cluster
members), plus frontier_discount on `front_pids` planets.

**8-seed A/B vs v7_0 (paired seeds):**
| Config | Winrate | Per-seed |
|---|---|---|
| composite (baseline) | 6/8 = 75.0% | `[1,1,-1,-1,1,1,1,1]` |
| composite+cluster (fd=0.5) | 4/8 = 50.0% | `[1,-1,-1,-1,1,1,1,-1]` |
| composite+cluster (fd=1.0) | 4/8 = 50.0% | `[1,1,-1,-1,1,-1,1,-1]` |

**Decomposition:** Both frontier discount values regress identically
in winrate. fd=1.0 (no penalty) regresses just as much as fd=0.5. The
cluster cohesion term itself is the regressor, not the frontier
discount.

**Magnitude check:** at +30 turns of self-play, cluster_value's
increment over base = 164, composite's increment = 53.6. Cluster
dominates composite by 3× in the layered head — but this is partial
confound, not the full cause (fd=1.0 isolated the cohesion term and
still regressed).

**Root cause:** cluster head biases toward HOLDING (rewards
defensible clusters, penalises frontier planets). iter's edge is
aggressive expansion; cohesion-rewarding heads tell it to stop
expanding. Same failure mode as `territory_value` (−19pp) and the
defensibility variants. **Hand-designed leaf heads that emphasize
holding over capturing under-perform composite in this game.**

Knob: `VALUE_FN = "composite"` (default; iter_v2 unchanged). Code
remains gated as alternate `VALUE_FN` choices.

## Rule 37 status — caps fired

Two axes have now hit the consecutive-falsification cap:

- **"Additive candidate" axis** (3/3 fail): MSP dormant, geo allocator
  −12.5pp, prior mission-persistence −42pp. The K=10 chooser cannot
  reliably gate new candidates regardless of how they're proposed.

- **"Hand-designed leaf head" axis** (2/3 fail since composite WIN):
  territory −19pp, cluster −25pp. Hand-designed scorers that
  emphasize HOLDING regress in this game's aggressive-expansion
  metagame. Composite remains the only successful leaf head.

## Implication for next session

The structurally correct next move is **imitation learning leaf
head** (H24 in `state/hypothesis-board.md`). 50 top-10 replays
already cached at `audit/external/replays/`. Effort ~1 week:

- Data pipeline: parse replays into (state, P(win)) pairs.
- Feature extraction: ~24-dim board features per state.
- MLP training: PyTorch, small model (~3 layers), early stopping.
- Inference shim: wrap as `value_fn` for `lib.v7_search.score_candidate`.

This is the only direction that escapes "human-designed leaf scorer"
entirely. AlphaZero pattern: stable learned value head unlocks deeper
search (multi-step planning becomes safe again).

## Honest end-of-session state

- iter_v2 (#52678866) ladder PENDING; no submission decisions until
  it settles (~12-24h).
- 3 commits this session (MSP, geo, cluster), all default OFF.
- iter behavior unchanged from session start.
- Diagnostic value: two failure-mode axes now decisively exhausted;
  the IL direction is no longer optional, it's the only structurally
  unexplored path.
