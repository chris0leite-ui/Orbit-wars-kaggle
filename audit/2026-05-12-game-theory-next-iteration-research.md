# 2026-05-12 — Game-theoretic next-iteration research (7-step analysis)

PI requested research-only documentation; no implementation in this
turn. Captures the structured analysis of where to go next from a
game-theoretic perspective.

## TL;DR

Heuristic iteration is plateaued at μ ≈ 1040-1100 across all our work
(v3.4 → v3.5.1 → σ-equiv-v1 → v7_minimax all cluster in this band).
The next *game-theoretic* iteration is **PSRO (Policy-Space Response
Oracle) meta-agent over our existing zoo**. After PSRO, the only
proven path to top-10 (μ=1447) is self-play RL.

This document is research/strategy. Implementation deferred.

---

## 1. Define the problem

Maximize μ at deadline 2026-06-23 (42 days out).

| Anchor point | μ |
|---|---|
| Our v7_minimax (#52568317) | PENDING (predicted 1040-1090) |
| Our σ-equiv-v1 peak | 1041.8 (evicted) |
| Top-10 cliff (ShunkiKyoya) | 1447 |
| LB #1 (bowwowforeach) | 1697 |

Gap to top-10: ≈ +400μ over 42 days.

Game-theoretic framing: Orbit Wars 2P is a finite symmetric zero-sum
stochastic game. Nash equilibrium exists with value 0 by von
Neumann's minimax theorem. Distance from Nash IS the μ-gap; closing
it is the actual problem.

## 2. Disaggregate (Δ-to-Nash decomposition)

```
Δ_to_Nash = Δ_action_quality       (per-turn pick quality)
          + Δ_lookahead_depth      (multi-turn planning)
          + Δ_opp_class_coverage   (robustness across diverse opps)
          + Δ_randomization        (exploitability of pure strategies)
          + Δ_compute_efficiency   (search throughput per actTimeout)
```

Where we stand:

| Component | Current state | Gap |
|---|---|---|
| Action quality | v3 + σ-equiv (in-class Nash via tie-break fix) | Small in-class; large cross-class |
| Lookahead depth | v7 K=3 maximin | Compute-bound; K=5+ infeasible |
| Opp class coverage | v7 M=2 (v3 + drop-one v3) | Narrow — no precision/RL/v2 model |
| Randomization | None (pure-strategy v7) | Mixed Nash unexplored |
| Compute efficiency | v3 rollout ~30-50ms/call bottleneck | Cheaper rollout policy could 3-5× K |

## 3. Prioritize

By expected μ-leverage per build day, ranked:

| Approach | Build days | Est. μ-lift | GT rigor |
|---|---|---|---|
| **PSRO meta-agent** over existing zoo | 1-2 | +10-40μ | **High** — Nash over policy space |
| Wider opp class in v7 (M=4-6) | 1 | +10-30μ | Medium — wider maximin |
| Mixed strategy on v7 candidates | 0.5 | +5-15μ | High — mixed minimax |
| Recapture missions | 2-3 | +30-50μ | Low — heuristic strength |
| Wave bundling | 3-5 | +30-50μ | Low — heuristic strength |
| Self-play RL (PPO) | 10-14 + GPU | +100-300μ | **Highest** — true Nash |
| CFR + state abstraction | 7-10 | +30-100μ | High — provable Nash in abstraction |

Game-theoretic rigor lens: PSRO + mixed strategies + RL + CFR are
canonical game theory; recapture/wave/strike-window are strength
upgrades that don't change Nash-approximation level.

## 4. Workplan — PSRO build (primary recommendation)

**Why PSRO**: proven Nash-approximation method short of RL (AlphaStar
pipeline). All inputs exist in our repo. Uniquely addresses
Δ_opp_class_coverage AND Δ_randomization simultaneously.

**Concrete steps** (1-2 day build):

1. **Policy pool** (~30 min):
   ```
   Π = {v7_minimax, v3.4-σ-equiv, v3.5.1, v2, roi,
        baseline-nearest-sniper, v4_mirror_t0, v4_hybrid,
        v4_endgame, precision_v3 (gitignored)}
   ```
   ~10 distinct policies spanning ~5 classes.

2. **All-pairs payoff matrix** (6-10h wallclock):
   - 10×10 − 10 = 90 ordered pairs × 8 seeds = 720 games
   - Both-sides → 1440 game-results
   - Use existing `scripts/tournament.py`
   - Output: P[i][j] = wins_i − wins_j across all games where i vs j

3. **Solve mixed Nash** (2-4h):
   - `pip install nashpy` → `nashpy.Game(P).support_enumeration()` (~50 LOC)
   - OR write LP via `scipy.optimize.linprog` (~300 LOC)
   - Output: probability distribution p* over Π

4. **Build meta-agent** (~3-5h):
   - `agents/v8_psro_meta/main.py`: at game start, sample policy_idx
     from p* using seed = hash(initial_planets). Per-game determinism.
   - Delegate each turn to chosen policy's agent function.
   - Bundle all policies (~200-400KB total).

5. **Verification gates**:
   - Self-play 16 seeds: expect ≥ 80% draws (seed-deterministic
     sampling → both sides usually pick same policy → mirror play)
   - PSRO vs each pool member: PSRO ≥ each in expectation by Nash
   - Bundle smoke-test
   - Per-turn time well under 1000ms

## 5. Risks + alternatives ruled out

### PSRO risks

| Risk | Probability | Mitigation |
|---|---|---|
| Nash degenerate (single policy dominates) | Med | Then submit that policy; PSRO confirms which |
| Pool not diverse enough → Nash = v7 | Med | Add CFR-trained or hand-crafted "anti-v7" policy |
| Empirical Nash noisy (8 seeds insufficient) | Med | Wilson CI on cells; iterate to 32 seeds if needed |
| Bundle size exceeds Kaggle limit | Low | tar.gz allows MBs |
| Multi-policy delegate fails validation | Low | Smoke-test bundle before push |

### Alternatives explicitly NOT recommended this iteration

- **Wider opp class in v7 (M=4-6)**: incremental tuning, doesn't change
  framework. Lower μ-leverage than PSRO.
- **Recapture / wave bundling**: heuristic strength, not GT. Adds
  capability but stays in v3-class.
- **Self-play RL**: 2-week + GPU commitment. Cannot fit current sprint.
  Right answer eventually but not first.
- **CFR**: requires state-abstraction design (~5 days just for that).
  Higher activation energy than PSRO. Defer.
- **Strike-window timing**: heuristic; tested empirically in precision_v3
  which is at 1009 — similar μ to our σ-equiv. Doesn't break the
  heuristic plateau.

## 6. Synthesize — recommendation

**Next iteration: v8_psro_meta over existing zoo.**

Justification:
1. **Provable property**: against any policy in Π's strategic span,
   mixed Nash expected payoff ≥ pool-game-value. For symmetric
   zero-sum pool this means cannot-lose against pool span.
2. **Empirical robustness**: even if ladder opponents are outside
   our pool, mixture means no single-weakness exploit can land
   consistently.
3. **Leverages prior work**: every iteration this session becomes a
   component of the meta-strategy. σ-equiv-v3's lock, v7's maximin,
   precision's wave-bundling, mirror's structural property — all
   contributing.
4. **Reasonable expected μ**: 1050-1090 (correcting for my
   pessimism-calibration noted in the friction log: probably 1060-1110
   realistic).
5. **What it doesn't deliver**: top-10 (1447). Heuristic ceiling
   appears to be ≈ 1040-1100 across all variants tried by our team
   AND parallel branches. RL is required for further.

**Two-stage roadmap**:

- **Week 1 (Days 1-2)**: build + submit PSRO meta. Measure live μ.
- **Week 2-3 (Days 3-14)**: if PSRO confirms heuristic ceiling at
  ~1080-1120, start self-play RL on Kaggle GPU notebook. PPO with
  PSRO pool as warm-start (RPO seeding). Target μ: 1100-1200.
- **Week 4+ (Days 15+)**: refine RL with self-play league /
  population-based training (PBT).

## 7. Communicate — concrete next session plan

PSRO build. **Definition of done**:

- ✅ All 10 pool policies load + run cleanly (some require cherry-picks
  from gitignored branches: precision_v3 from origin/merge-precision-to-main)
- ✅ Payoff matrix P[10][10] computed over 720 games (~6h tournament)
- ✅ Mixed Nash p* computed via nashpy or scipy LP
- ✅ Meta-agent samples per-game via seed-deterministic index
- ✅ Self-play 16 seeds verifies near-draws property
- ✅ Local probe: PSRO vs v7 ≥ 50% (sanity check)
- ✅ Bundle smoke-test
- ✅ Live submission

**Decision tree post-PSRO live μ**:

```
PSRO live μ < 1041 (regression):
  → pool diversity issue OR bug in policy delegation
  → investigate before further iteration

PSRO live μ ∈ [1050, 1090] (expected):
  → confirms heuristic ceiling
  → pivot to self-play RL training (Week 2-3 roadmap)

PSRO live μ > 1100 (positive surprise):
  → PSRO compounds nicely
  → iterate (add CFR-trained or RL-trained policy to pool,
     re-run PSRO with expanded Π)
```

## Open questions deferred

- Should PSRO's per-game sampling be ε-deterministic (small randomness
  via per-game seed) or fully deterministic? Affects whether self-play
  produces variance that's useful for ladder-μ measurement.
- Once self-play RL is in scope, do we use pure PPO or PSRO-with-PPO
  (where each PSRO oracle is trained via PPO best-response)? The
  latter is closer to AlphaStar but ~2× compute.
- Should we cherry-pick precision_v3 from `origin/merge-precision-to-main`
  into our branch for PSRO purposes? Currently gitignored. Would
  expand pool diversity.

## What this document is NOT

Implementation. PSRO is queued as "next iteration if PI green-lights;"
this document captures the research, not the build.

## Calibration ledger note

Earlier this session I predicted σ-equiv-v1 μ ≈ 995 (off by +47).
Adjusting for that miscalibration, the predictions in this document
should probably be read as +10-20μ higher than written. PSRO realistic
range: 1070-1110, not 1050-1090.
