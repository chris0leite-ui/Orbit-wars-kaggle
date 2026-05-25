# Postmortem — 2026-05-25 strategy-battlefield-game-6v82d

Session goal: build a "strategy battlefield master" agent per PI
directive — expand counterclockwise, build momentum, launch synchronized
mass-attack when upper hand achieved, predictable launches, large-planet
focus, no idle resources. Three iterations shipped (V2 → V3 → V4),
diagnostic methodology was diagnose-first + 8-game A/B (no swap, 250-turn
elim cap) per PI override.

End state: `agents/momentum_strike/` shipped as a calibration-tier
agent — beats 4 simple-strategy opponents 27/32 with 27/32 early-elim
by turn 250; loses 0/8 to `agents/baseline` (μ=1078). PI reframed at
wrap: "we will use this strategy to benchmark other of our strategies"
— momentum_strike is repositioned as a known-quantity calibration probe,
not a ladder candidate.

## What went wrong (decision quality)

- **V1 bypassed `DEFAULT_MECHANISMS` in favor of hand-rolled aim + fate
  gate.** Pre-decision priors already showed `agents/simple/production`
  and `nearest` use `realize(intents, obs, mechanisms=DEFAULT_MECHANISMS)`
  and win their seats. I chose to bypass for "control" and ate the
  bundle-relevant safety stack (`lead_aim`, `arrival_size`, `sun_avoid`,
  `path_clears_other_planets`, `oob_guard`). Cost: ~1.5h rebuild as V2.
  V2 jumped from 1/8 to 7/8 vs `nearest` solely by switching to the
  realize pipeline.

- **V3 picked a knob (POST_CAPTURE_GARRISON) on a single diagnostic
  signal.** The "captures lost within 5 turns = 10" trace pointed to
  garrison thickness; I picked `+5` over-commit. A LATER emission-rate
  diagnostic showed baseline emits 2× more fleets per game (100 vs 51)
  — emission VOLUME, not garrison thickness, was the dominant gap.
  Should have measured both before the knob choice. Cost: 1 A/B
  iteration wasted.

- **Continued knob-tuning after 2 failed attempts to budge 0/8 vs
  baseline.** POST_CAPTURE (V3a), ENEMY_MULTIPLIER (V3b), synchronized
  salvo (V4) — all 0/8 unchanged. The gap is structural (K-step
  rollout vs greedy proposer), not knob-tunable. Earlier termination
  would have saved compute. Complements Rule 37 (axis cap at 3 variants
  within an axis); this is "the whole knob ladder is wrong axis."

## Frictions logged this session

Cross-link: `audit/friction.md` under `## 2026-05-25` heading. Eight
entries:

- `mechanism-pipeline-bypassed-in-v1`
- `knob-tuned-without-dominant-mode-measured`
- `sequential-env-step-diverged-from-play-one`
- `bundle-multi-line-import-broke-bundle` (Rule 38 recurrence)
- `bundle-default-lib-order-stale-kinematic-table`
- `salvo-reserve-defensive-not-offensive`
- `long-wait-salvo-starved-expansion`
- `structural-gap-not-knob-tunable`

## PI-overrides

- After V1 over-engineering (7/8 loss to nearest): "find a simple
  approach that works. it is the simple things that move us forward."
  Resulted in V2 strip-down to realize+DEFAULT_MECHANISMS — the
  single most effective decision of the session.
- After V3's "what next?" report: "3" (= add the synchronized salvo,
  the originally-requested novel mechanism).
- At session end: "wrap up. we will use this strategy to benchmark
  other of our strategies" — reframes the artifact. Not a failed
  submission attempt; a calibration probe for future agents.

## Promotion candidates (pending PI ratification)

A. **Diagnose ≥2 failure modes before picking a knob** — to
   `.claude/skills/kaggle-comp/improvements.md`. Tag:
   `knob-tuned-without-dominant-mode-measured`. Single-axis diagnose
   missed the dominant emission gap; one knob iteration wasted.

B. **Authoritative A/B uses `play_one`/`env.run`, not manual
   `env.step`** — to `improvements.md` / testing methodology section.
   Tag: `sequential-env-step-diverged-from-play-one`. ~30 min × 2
   chasing irreproducible diagnostic wins.

C. **Structural-gap stop-rule: declare structural after 2 failed
   knobs** — to CLAUDE.md (would be Rule 48) or `improvements.md`.
   Tag: `structural-gap-not-knob-tunable`. Complements Rule 37
   (axis-internal cap); this is the cross-axis cap.

**Not yet ratified.** PI was given the postmortem draft and chose
`commit and push` (skipped the ratification dialogue). Promotions
NOT applied this commit; carry forward to next session for
explicit ratification.

## PI additions (from step 4)

None — PI elected to skip the additions step and commit directly. The
"benchmark for other strategies" reframe IS the most material PI
input; folded into the postmortem context above.

## Framework version at session-end

- Branch: `claude/strategy-battlefield-game-6v82d`
- HEAD commits (newest first):
  - `c75f524` — V4 synchronized salvo + cross-turn ledger, gated OFF
  - `8580ce7` — V3 enemy-multiplier when behind
  - `2f37c3e` — V2 production-first expansion + defense + CCW tie-break
- Active rules: CLAUDE.md `## Operating rules` Rule 0-47 (no
  promotions ratified this session).
- Loaded skills: `postmortem` (this skill). No others invoked.

## What V2/V3/V4 produced (artifact summary)

- `agents/momentum_strike/` — 4-file modular agent (main.py +
  proposer.py + __init__.py; salvo support code lives in lib/).
- `lib/polar.py` — CCW polar-angle helpers (used as 3rd-tier
  tie-break).
- `lib/salvo.py` — synchronized-arrival planner with parallel
  `wait_Ns: list[int]` field and parametrized `reserve_fn`.
- `scripts/momentum_strike_ab.py` — 8-game P0-only A/B driver
  matching PI methodology (no swap, 250-turn elim cap).
- `scripts/bundle_agent.py` — `DEFAULT_LIB_ORDER` extended
  (`kinematic_table`, `polar`, `salvo`).

### A/B baseline against simple panel (V3-default, V4-gated-off)

```
vs agents/simple/nearest      : 7/8 wins, 7/8 early-elim ≤250
vs agents/simple/weakest      : 8/8 wins, 8/8 early-elim
vs agents/simple/production   : 4/8 wins, 4/8 early-elim
vs agents/simple/enemy_first  : 8/8 wins, 8/8 early-elim
                       TOTAL  : 27/32 wins (84%), 27/32 early-elim (84%)
vs agents/baseline (μ=1078)   : 0/8 — structural gap, not knob-tunable
```

These numbers define momentum_strike's place in the panel for
benchmarking future agents: a moderate, deterministic, fast-to-converge
opponent between trivial (random/nearest) and production (baseline).
