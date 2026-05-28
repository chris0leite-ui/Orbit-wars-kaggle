# 2026-05-28 PM3 — H14 recipe locked from konbu17 + cross-comp synthesis

Status: research synthesis only. No new code; H14 implementation deferred to next session with the recipe now mechanically specified.

Companion to:
- `knowledge-base/thoughts/2026-05-28-pm-distillation-action-rank-collapse.md` (Phase A failure diagnosis)
- `knowledge-base/thoughts/2026-05-28-pm2-feature-sufficiency-probe.md` (Stage 1-3 probes)

## What this note adds

PM2's recommendation pointed at "Direction 1 — Per-emit MLP filter (konbu17 architecture, +19 pp evidence)" as the highest-EV next step. PM3 does the depth research on the actual konbu17 notebooks plus the cross-comp simulation winners (Halite III/IV, Lux S1/S2, Planet Wars 2010, Hungry Geese) so the recipe is no longer "konbu17-shaped" but mechanically pinned. The scaffold for the recipe is already on this branch (H14 in `top-performer-strategies.md`).

## konbu17 recipe — mechanically specified

Sources: `train-submit-v4-ml-validator-topk2-tutorial.ipynb` cells 8/11/12/14/16 + `orbit-wars-rule-base-ml-shot-validator-hybrid.ipynb` cells 0-4 (the earlier single-model version with the 5-failed-ML-direction postmortem).

### Features — 24-d, per shot (NOT per state)

Exact schema in `data/shot_validator/schema.json` on this branch — matches konbu17 cell 8 lines 348-385 verbatim. All values normalised to ≈ [0, 1].

| idx | feature | formula (BOARD=100, MAX_SPEED=6) |
|---|---|---|
| 0–2 | source ships / prod / radius | `sships/100, sprod/5, sr/4` |
| 3–5 | target ships / prod / radius | `tships/100, tprod/5, tr/4` |
| 6–8 | target owner one-hot | `(own_self, own_neutral, own_enemy)` |
| 9 | ships sent | `ships_sent/100` |
| 10 | ship fraction | `ships_sent / max(sships,1)` |
| 11 | distance | `max(hypot(tx-sx,ty-sy)-sr-tr, 0)/100` |
| 12 | ETA | `dist / max(speed, 0.5) / 60` |
| 13 | fleet speed | `(1 + 5·(log(ships)/log(1000))^1.5) / 6` |
| 14–17 | in-flight allied n/ships, enemy n/ships | `/10, /100, /10, /100` |
| 18 | turn | `step/500` |
| 19–22 | my/enemy totals, ship diff, my planet count | `/200, /200, /200, /20` |
| 23 | enemy planet count | `/20` |

Per-shot features 9–13 vary across candidate moves from the same state ⇒ a chooser using this **can** distinguish siblings. This is the structural difference from Phase A's pooled-state 40-d setup.

### Label — local-success binary

```python
arrival_turn = step_idx + ceil(dist / speed)
label = 1 iff side owns target_id at any turn in [arrival_turn, arrival_turn + 10]
       else 0
```

10-turn buffer filters fast-flip noise (capture then immediately lose).

**Critical filter:** drop self-reinforcement shots (target already owned by side) ENTIRELY before training. Including them pushes `pos_rate ≈ 0.96` and the BCE `pos_weight = (1 − p)/p` collapses → validator outputs uniformly low → threshold rejects everything → topk1 sends nothing. Filtered pos_rate target: 0.50–0.75. konbu17 cell 12 line 4084 prints a warning if `pos_rate > 0.85` and explicitly suggests adding stronger opponents or more self-play.

### Architecture — 3-layer MLP, ~3.5 k params

```python
nn.Sequential(
    nn.Linear(24, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
loss = nn.BCEWithLogitsLoss(pos_weight = (1 - p_rate) / p_rate)
opt  = Adam(lr=1e-3); epochs = 40; batch = 512
```

Ensemble: 3 models with seeds [42, 100, 7]. Sigmoid-averaged at inference. konbu17 reports **+5 pp overall, +31 pp on obn_v4** vs single model.

Inference path: pure-numpy `@ + np.maximum + manual sigmoid`. Parity check against torch ensemble: max abs diff < 1e-6 (cell 14 lines 614-627).

### Training corpus — mixed opponents, ~140 games, ~20 k labels

konbu17 v4 mix:
- 5 inline opponents (`nearest_sniper`, `weakest_first`, `production_first`, `defender`, `random_play`) × 6 seeds × 2 sides = 60 games
- v4-vs-v4 self-play × 30 seeds × 2 sides = 60 games
- Total ≈ 120-140 games, both sides labeled = ≈ 20 k shots

**Game-level 80/20 split** (NOT row-level). Shots within a game are temporally correlated; row-level split leaks 15-20 pp val accuracy.

### Inference path — strict-improvement FILTER

1. Rule-based proposer returns its candidate move list (e.g. `agents/baseline` proposer).
2. For each candidate move, ray-cast `(src_xy, angle)` to identify target; skip self-reinforcement (never filtered out by the validator).
3. Encode 24-d features per move; batch through 3-model ensemble forward; average sigmoid.
4. **Drop moves with avg P < 0.30** (ensemble threshold; single-model uses 0.40).
5. **topk1**: among survivors, keep only the largest-ship move per turn (`max(moves, key=lambda m: int(m[2]))`).

This is a FILTER, not a re-ranker. Cannot regress vs proposer alone — only rejects. Strict Pareto improvement.

### Embedding mechanics

Two patterns in the konbu17 lineage; both pure-Python at inference:

- **Train v4 (production):** `weights.npz` next to `main.py`; bundle tar.gz contains both. Loader checks `/kaggle_simulations/agent/weights.npz`, cwd, `__file__`-relative.
- **Earlier hybrid:** base64-encoded npz blob in `decode_weights.py` (~15 KB), decoded at submit time.

For our bundler (`scripts/bundle_agent.py`), the base64 pattern fits cleanly — no extra files in the tar.

### Known gotchas — these matter

- **`kaggle_environments < 1.28` ships without `orbit_wars` env, AND comp-linked notebooks have internet disabled.** Bundle the wheel in a Kaggle Dataset, or define `Planet`/`Fleet` namedtuples locally to mirror the env schema. AidenSong does the latter; konbu17 does the former.
- **`re.sub(r"^def agent\(obs, config=None\):", "def _v4_agent_internal(...)", v4_source, count=1, flags=re.MULTILINE)`** — konbu17's rewriter expects this exact signature when wrapping the rule-base inside the validator agent. Trivial to skip; flagged here so the bundler doesn't break the rule-base import.
- **No PyTorch / sklearn at submit time.** Train in torch, export to numpy.
- **topk2 vs topk1 depends on validator quality.** Tutorial v2 tried topk2 with the self-trained val_acc ≈ 0.76 validator and regressed −85 LB. topk1 is the safer default; topk2 needs the LB-1450+ replay-trained validator.

## Cross-comp synthesis (Halite, Planet Wars, Lux, Hungry Geese)

The deep-dive across competition winners is consistent and unanimous on the architectural question.

### Pattern 1 — per-candidate scoring + assignment (Halite IV)

ttvand (1st), 0Zeta (4th): per-ship × per-action **score matrix** (6 actions × N ships). Each role (mining / hunting / guarding) gets its own scoring formula over per-(ship, action) features (target halite, distance, inspiration, friendly/enemy proximity, dominance / control field, halite-near, etc.). **Hungarian / linear sum assignment** over the matrix gives conflict-free actions for all ships simultaneously.

Direct analogue for Orbit Wars: each (source-planet, candidate-target) pair scored independently with per-pair features; assignment if you want multi-source coordination.

### Pattern 2 — closed-form per-candidate value (Halite III)

teccles (1st), mlomb, TheDuck314: every (ship, cell) pair gets a closed-form score with explicit time/cost decomposition. teccles' miner formula:
```
score = x · RETURN_TIME + (TRAVEL_TIME + MINING_TIME) · (MAX_HALITE − SHIP_HALITE) / (HALITE_MINED − HALITE_BURNED)
```
TRAVEL_TIME from Dijkstra per ship. Rank-agreement is automatic because each candidate has its own feature row.

Direct analogue: istinetz's Orbit Wars discussion-thread formula `pv = production · (γ^arrival − γ^horizon) / (1 − γ)` with γ=0.99 (LB ≈ 1000 "with just this + trajectory calc") is the same pattern.

### Pattern 3 — per-cell action map (Lux S1, Hungry Geese)

Toad Brigade (Lux S1 1st) and GeeseZero (Hungry Geese 5th): full-board ResNet (24 residual blocks × 128 channels for Toad Brigade) outputting a per-cell action map. The **policy head** does action selection, the scalar value head is used only as IMPALA / AlphaZero baseline. **None of the winners used scalar-value-over-candidates as their action selection mechanism.**

This pattern doesn't map cleanly to Orbit Wars because the action space is not a per-cell map — but the lesson does: when you use a learned value head for action selection, the value head's output structure must match the action's, or you need a substrate (forward-sim, per-candidate features) to bridge the gap.

### Pattern 4 — TheDuck314's learned per-action classifier

Halite III. Small NN trained on 4-player collision-prediction. **"basically no impact on mu"** — documented failure case of learned per-candidate scoring being indistinguishable from rules. The lift on these competitions tends to come from STRATEGIC restructuring (multi-source coordination, opening recipes, label calibration), not from per-action net quality.

### The unanimous lesson

> **When the rule-base ceiling and the ML floor don't intersect, let ML EDIT the rule-base, don't make it stand alone.** (konbu17, validator notebook cell 12 lines 4076-77)

konbu17 documented five pure-ML failures (PPO scratch, PPO+curriculum, smoother curriculum, SFT single-teacher, SFT multi-teacher) before the hybrid worked. Lux S1 Toad Brigade needed teacher-KL regularisation and reward-shaped curriculum to stabilise pure RL. Halite III/IV winners didn't use ML at all. Pure-ML wins this competition family only when the substrate is right; hybrid filter/edit wins consistently.

## Implications for this branch's next session

### Re-affirmed (PM3 doesn't change PM2's direction)

- The "Direction 1 — per-emit MLP filter" recommendation from PM2 is the right next move.
- It's strict-improvement (filter, not re-ranker), low-risk for the rolling pair coordination.
- Existing scaffold on this branch (24-d schema + label_shot_outcomes.py + README) covers ~70 % of the build.

### Newly mechanically specified (PM3 adds)

- Architecture: 3-MLP ensemble seeds [42, 100, 7], `nn.Linear(24,64)→ReLU→Linear(64,32)→ReLU→Linear(32,1)`, BCEWithLogitsLoss, Adam lr=1e-3, 40 epochs, batch 512.
- Pos_rate calibration: exclude self-reinforcement first, then mix opponent strengths to keep pos_rate in 0.50–0.75. The model is tiny; **data filtering, not architecture, is the load-bearing decision**.
- Game-level 80/20 split (not row-level — leaks 15-20 pp val accuracy).
- Inference threshold 0.30 with ensemble (0.40 with single model). topk1 (largest ship survivor only) as the default; topk2 needs LB-1450+ replays.
- Embedding: pure-numpy forward at submit time; base64 npz weights inside the bundle.

### Concrete sequencing

1. **Validate the existing label pipeline end-to-end on tiny corpus.** 4-8 games × 2 sides, smoke through `label_shot_outcomes.py`, sanity-check label distribution (pos_rate should be 0.4-0.8 with self-reinforcement filter on; pure-mainline 0.95+ without it). Requires either a populated `audit/external/replays/` or a wallclock-capped self-play run.
2. **Generate the corpus.** ~140 games with the konbu17 opponent mix adapted to our agent set:
   - vs `agents/baseline_favor` (weakest)
   - vs `agents/baseline_full` (moderate)
   - vs `agents/baseline_joint_aggr_consolidated_orbitfix` (strong, sibling branch)
   - vs `agents/baseline_pv_eta` (live rolling champ, sibling branch)
   - self-vs-self (`agents/baseline` × `agents/baseline`)
   Tune the ratio to keep filtered pos_rate ≈ 0.6.
3. **Train the ensemble.** 3 seeds × 40 epochs × Adam lr=1e-3. Save weights as base64 npz.
4. **Embed via `scripts/bundle_agent.py`.** Validate with `pytest tests/test_bundle.py` + `python fast.py play <bundle>` (Rule 46 gate).
5. **A/B vs production stack** (`agents/baseline` no validator vs `agents/baseline` with validator) at n=32, BASELINE_WALLCLOCK_MS=100. Wilson-lo ≥ 0.50 gate.
6. **A/B vs live rolling champ** (`baseline_pv_eta` at μ=1154.8) at n=32 if step 5 cleared. Wilson-lo ≥ 0.50 to gate the submission.
7. **Push (Rule 1, 42, 43, 46, 47 checklist).**

### Compute estimate

- Corpus gen: 140 games × ~150 ms/turn × 200 turns × 2 seats ≈ 4 hours single-core; ≈ 30 min on 8 workers (need `BASELINE_WALLCLOCK_MS=100` cap to bound per-turn cost).
- Labeling: minutes.
- Training: seconds.
- Bundle + parity: 5 minutes.
- A/B n=32: 30-60 minutes serial.

**Total: ~1 session.** The H14 "15 days" estimate in `top-performer-strategies.md` predates the existing scaffold + the konbu17 recipe being mechanically pinned. With both in hand, this fits in an overnight + morning shift.

## Falsifications worth promoting (over and above PM2)

- **"Phase B-1's CRN-paired advantage labels with the existing 40-pool features will recover rank order."** Cross-comp says no — when winners used scalar value heads (AidenSong, GBC) they always evaluated on forward-simulated terminal states. Direct-argmax over pooled state never appears as a winning architecture. The CRN-advantage idea is preserved for Direction 2 (per-candidate score head), not Phase B-1 as originally framed.
- **"Pure RL / pure IL will save us if we throw enough compute."** konbu17 burned 5 attempts before pivoting to hybrid filter; Lux S1 needed teacher-KL + curriculum to stabilise; Halite winners didn't use ML at all. Pure-ML in this competition family is a long-tail bet against a hybrid that's mechanically specified.

## Pointers

- `data/shot_validator/{README.md, schema.json}` — 24-d feature spec already aligned to konbu17.
- `scripts/label_shot_outcomes.py` — labeling pipeline, ready to run on populated `audit/external/replays/`.
- `scripts/generate_selfplay_replays.py` — self-play replay generator, needs wallclock cap for batch runs.
- `knowledge-base/concepts/top-performer-strategies.md` §H14 — original strategy hypothesis (May, "15 days").
- `state/MULTI_BRANCH.md` push board — Rule 42 coordination required before submit.

## Sources

- **Orbit Wars Kaggle notebooks** (full code extracted):
  - konbu17 v4 validator: https://www.kaggle.com/code/konbu17/train-submit-v4-ml-validator-topk2-tutorial
  - konbu17 single-model: https://www.kaggle.com/code/konbu17/orbit-wars-rule-base-ml-shot-validator-hybrid
  - AidenSong search + value: https://www.kaggle.com/code/aidensong123/lb-highest-1000-search-learned-value-function
  - Pilkwang structured baseline: https://www.kaggle.com/code/pilkwang/orbit-wars-structured-baseline
- **Cross-comp winners**:
  - Toad Brigade Lux S1: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
  - ryandy Lux S2: https://github.com/ryandy/Lux-S2-public
  - ttvand Halite IV: https://github.com/ttvand/Halite
  - 0Zeta Halite IV: https://github.com/0Zeta/HaliteIV-Bot
  - teccles Halite III: https://github.com/teccles-halite/halite3-bot
  - mlomb Halite III: https://mlomb.dev/blog/halite-iii-postmortem
  - TheDuck314 Halite III: https://github.com/TheDuck314/halite2018
  - Planet Wars archive: http://satirist.org/ai/planetwars/
  - zvold Planet Wars: http://zvold.blogspot.com/2010/12/two-bots-for-planet-wars-ai-challenge.html
  - Hungry Geese GeeseZero: https://www.kaggle.com/competitions/hungry-geese/writeups/takedarts-5th-place-solution-geesezero
