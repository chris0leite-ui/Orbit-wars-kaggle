# 2026-05-29 — Launch-discipline rules (Rule A neutral, Rule B opponent-K)

Branch: `claude/champion-strategy-rules-00JzI` (merged the champion
`baseline_pv_eta_anchor_1163`, μ≈1163, from `game-theory-winning-strategy-SEU7P`).

## What the PI asked for

Two rules, *guaranteed* not merely encouraged:

- **Rule A — neutral discipline.** Never send ships to a NEUTRAL planet
  unless they capture it. Capture by fleets arriving the **same tick**
  from different planets is allowed; staggered pokes are not.
- **Rule B — opponent predictability ceiling.** Only commit to capturing
  an OPPONENT planet if the fleet arrives within **K turns** (K=10).

PI decisions this session:
- Rule B = **ceiling** (drop opponent captures arriving *later* than K).
- Rule A = **drop** non-capturing neutral launches (no active coalition
  synchronization).
- **Scope = all launches** (incl. sniper, the three rear-drains,
  reinforcements).

## Where the champion leaked (pre-fix)

- Live scorer `score_candidate_v4` ranks by favor-delta — no capture
  check. The bounce penalty lives only in the dead v2 scorer.
- Proposer trajectory filter checks *reachability*, not *capture*.
- `drain_*` / `emit_sniper_strikes` inject launches past every proposer
  filter.
- Combat sums only same-tick arrivals (`lib/combat.py`), so a solo
  under-sized neutral launch is consumed for nothing.
- Nothing bounded opponent-capture arrival vs K.

## What shipped

- **`agents/baseline/launch_rules.py`** — post-emit validator.
  - `resolve_launch_target` ray-casts each bare `[src, angle, ships]`
    move via `lib.trajectory.predict_fleet_fate` (sentinel target id) to
    get the first planet struck + arrival tick.
  - `enforce_launch_rules` — reinforcements/sun/oob kept; opponents kept
    iff `arrival <= K`; neutrals grouped by `(planet, tick)` and kept iff
    `lib.world_model.predict_garrison_at(ledger + our same-tick legs)`
    resolves to us (atomic group drop otherwise).
  - Env-gated `BASELINE_LAUNCH_RULES` (default `0`), K via
    `BASELINE_CAPTURE_HORIZON_K` (default `10`).
- **`main.py`** — `enforce_launch_rules` wired as the LAST step before
  every `return` in `agent()` (opening-MILP, trajectory, roi, composite).
- **`proposer.py`** — cheap gated prune of fire-now opponent candidates
  with `eta > K` (efficiency only; neutrals untouched so coalitions reach
  the post-pass).
- **`tests/test_launch_rules.py`** — 13 logic tests (fake fate, real
  combat) + a Rule-38 reproduce/confirm integration cycle on full games.

## Verification

- 13/13 logic tests green (neutral solo bounce/capture, same-tick
  coalition kept, staggered coalition dropped, atomic partial-coalition
  drop, ledger same-tick enemy, opponent in/out of horizon, configurable
  K, reinforcement exempt, sun out-of-scope, gate-off no-op, mixed).
- **Rules bind in practice** (not a null): with the gate OFF the champion
  emits neutral non-captures / opponent captures beyond K over a game
  (reproduce test passes). With the gate ON, every emitted launch on every
  turn of 3 full games has zero violations (guarantee test passes).
- Bundle builds with correct inlining (`launch_rules` inlined, 0 active
  `from agents.baseline` imports); cold-load succeeds (`fast.py play`
  reaches the match loop). NB: the bundler's self-play **parity gate**
  fails in this environment on a pre-existing `agents`-namespace import
  collision (breaks at `main.py:173` opening_planner→proposer, *before*
  the new import) — unrelated to this change; cold-load + the integration
  tests cover the behavioral check.

## Status / next

Default OFF — no behavioral change to the live champion yet. Pending
before any default-flip or submission (Rules 12/43/45, PI sign-off):
`fast.py eval --vs baseline_pv_eta_anchor_1163` n≥32, `--vs-panel`,
`--geometry-panel --by-archetype`. Known risk to watch: the K=10 ceiling
may curtail long-range sniper strikes — surface to PI before tuning K
(Rule 40).
