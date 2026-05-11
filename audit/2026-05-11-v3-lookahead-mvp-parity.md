# v3.1 lookahead MVP — framework lands, drop-one candidates yield parity

> Date: 2026-05-11 (late session)
> Branch: `claude/bootstrap-agentic-systems-lqnm6`
> Plan reference: `audit/2026-05-11-lookahead-phase2-forward-sim.md` §
> "Suggested Phase 2 design"

## TL;DR

**Sim<K> scoring head is real and works** (Phase 2 result holds: AUC ≈
oracle). The minimum-viable v3.1 — drop-one candidate enumerator over
the v3_snipe incumbent + Sim<K=10> rollout — runs cleanly (E.2 gate 0
crashes, 8 lookahead tests green, p95 turn 604 ms) **but is exactly
statistical parity with v2 over 32 seeds × 2 seats (32/64 = 50.0%).**

The earlier 8-seed result (11/16 = 68.8%) was upward noise; at scale
the lift evaporates. The bottleneck is the candidate enumerator, not
the scorer.

## What was built

`lib/lookahead.py`
- `env_from_obs(obs, configuration)` — reconstructs a steppable env from
  the agent's visible state. Validated to be bit-exact for current-state
  step parity vs the real env. The only fidelity gap is future comet
  spawns at steps 50/150/250/350/450 (fresh RNG state).
- `score_action(env, action, K, my_id, policy)` — clones env, applies
  `action` as our P{my_id} move, rolls forward K-1 turns under `policy`,
  returns (us - them) ship-total scalar. Same scoring head Phase 2
  validated at AUC = oracle.
- `enumerate_drop_one_candidates(action)` — minimal candidate set:
  `[incumbent, drop-launch-0, drop-launch-1, ...]`. Pure subset of the
  incumbent; never proposes new launches.

`agents/v3_lookahead/main.py`
- Per turn: build v3_snipe incumbent → drop-one candidate set →
  `env_from_obs` → score each candidate via Sim<K=10> → pick max.
- 500 ms wallclock watchdog; fallback to incumbent if budget exhausted
  partway through.

`tests/test_lookahead.py` — 8 tests (5 fast / 3 slow), all green.

## Results — head-to-head 32-seed panel vs v2

```
              |  v3_lookahead |       v2     
---------------------------------------------
v3_lookahead  |    sp 0/0/0   |  50% (32/64)
v2            |  50% (32/64)  |    sp 0/0/0
```

Calibration ladder:
```
strategy       mean_wr  max_p95_ms
v3_lookahead   50.0%     604.1
v2             50.0%      11.2
```

Wilson 95% CI on 32/64 = [37.7%, 62.3%] — fully consistent with random
50/50. No statistically detectable lift.

Per-seat breakdown:
- v3 as P0 (seed-by-seed against v2-P1): 15/32 = 46.9%
- v3 as P1 (seed-by-seed against v2-P0): 17/32 = 53.1%
- Combined: 32/64 = 50.0%

E.2 hard gate: 10 v3_lookahead self-play episodes, **0 crashes,
0 timeouts, all DONE.** Artifact:
`audit/tournaments/20260511T070745Z.json`.

## Why drop-one yields parity

The drop-one enumerator generates `N+1` candidates: the v3_snipe
incumbent + every "drop one launch" variant. The Sim<K> scorer picks
the highest projected delta.

This only adds value when **dropping a single launch is strictly better
than keeping all of them**. In practice that's rare:

1. v3_snipe's per-source-greedy already filters predicted-ours-with-
   surplus targets via WorldModel.owner_at. The launches it emits are
   already individually positive-EV under the static model.
2. Sim<K=10> doesn't disagree with the static EV often — the K=10
   rollout under v2-self-play just verifies what the snapshot already
   says.
3. The cases where a single launch is net-negative (e.g. over-
   committing to an already-falling target) DO exist but appear in <5%
   of turns — and dropping them only saves the cost of one launch, a
   small ship-delta.

## What the scorer IS doing well

Independent of the lift result, the framework lands:
- env reconstruction works (no env-handle exposed to agents on the
  ladder).
- Sim<K> wallclock is predictable: median ~110 ms at K=10, ~150 ms p95,
  total per-turn p95 604 ms with 5-6 candidates. Tight but under the
  1000 ms `actTimeout`.
- E.2 self-play passes cleanly.
- The `lib/planner.settle_plan` boundary plays nicely with this — the
  scorer integrates without disturbing v3_snipe's pipeline.

## What's needed for lift — next-session work

The candidate enumerator is the bottleneck. Drop-one only ever
SUBTRACTS from v2's choices. To beat v2, we need candidates that
v2 doesn't produce:

1. **Per-source swap** — for each source, score Top-1 vs Top-2 ROI
   target. Per-source independent decision. Costs ≤ N additional
   Sim<K> evaluations.
2. **Different strategy as a "challenger" candidate** — generate a
   second incumbent under a different strategy (e.g. nearest, weakest)
   and treat it as a sibling candidate. The Sim<K> picks whichever
   strategy's launches project better.
3. **Bipartite assignment** — Hungarian over (source × target) score
   matrix with non-overlap constraints. Generates a globally-coordinated
   alternative to v2's per-source greedy. Cost: one extra Sim<K>
   evaluation total.

Recommended order: option 2 (challenger candidate) first — single new
incumbent, single new Sim<K>, smallest blast radius. If it lifts, the
scorer's value is confirmed and we can grow toward bipartite (option 3).

## What I won't keep claiming

The 8-seed 68.8% headline was upward noise. I should have run 32 seeds
before celebrating. The honest current claim is **v3_lookahead is
indistinguishable from v2 at 32 seeds × 2 seats**; the framework works,
the lift requires richer candidates.

I have not run the 4P FFA gate or the broader 6-agent panel because
the 2P head-to-head being statistical parity makes those gates
uninteresting — we'd just see v3_lookahead ≈ v2 in all of them, at
significantly higher compute cost.

## DO NOT submit v3_lookahead to the live ladder

- It's parity with v2 in 2P, expected parity in 4P FFA.
- p95 604 ms is uncomfortable on the live container (potentially slower
  hardware); a single >1000 ms turn would auto-DONE the agent.
- Submission would evict v1.2/roi (μ=1001.4) from rolling-last-2 for
  zero expected μ gain.

Land richer candidates first; re-test; submit only with a
Wilson-significant lift over v2.
