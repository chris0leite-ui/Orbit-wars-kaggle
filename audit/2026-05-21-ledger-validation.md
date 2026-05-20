# 2026-05-21 — Ledger Validation: FAILED

**Branch**: `claude/audit-workflow-performance-btjeK`
**Predecessor**: `audit/2026-05-20-ledger-design.md` (the implementation)
**Plan**: `/root/.claude/plans/so-now-research-and-zany-widget.md`

## Verdict

**No-ship.** Both soft-mode and hard-mode ledger variants lose
decisively to the current production behavior in head-to-head play.
The wait_N>0 reserve-without-emit pattern we called a bug is actually
doing useful defensive work (ship hoarding for big counter-launches).
What-if rollout overstated the lift because opp's recorded actions
became static after divergence; in true reactive h2h the ledger
regresses.

## Phase results

### Phase A — close-inspection (PASS)

`audit/2026-05-21-ledger-close-inspection.md`. Ledger telemetry on the
6 what-if games looked clean: 75-100% emit success, all drops
legitimate (tgt_now_ours / planet_missing), zero duplicate-src turns,
~3% tiny-launch tail. No anomalies that warranted stopping.

### Phase B — A/B wrappers (PASS)

`agents/_ledger_on/main.py`, `agents/_ledger_off/main.py`,
`agents/_ledger_hard/main.py`, `agents/_mpc/main.py` all built and
smoke-tested. Empty-obs call returns `[]`.

### Phase C — h2h gates (FAIL at n=8)

```
ledger_on (soft) vs ledger_off at n=8:  2/16 wins (12.5%)  Wilson-LB=0.035
ledger_on (hard) vs ledger_off at n=8:  0/16 wins (0.0%)   Wilson-LB=0.000
```

Both well below the n=8 gate floor of Wilson-LB ≥ 0.40. Did not
escalate to n=16/32/64.

### Phases D-G — not run

Gates failed at C; plan branches to "no-ship + failure analysis".

## Replay-level diagnosis (4 captured games, n=2 seeds × 2 directions)

`audit/replays/20260520T073756Z/`:

```
                       launches  ships  idle%  final-planets-total
led_on  (across 4):    321       14578  72%    0
led_off (across 4):    416       23041  66%    89
```

led_off (current production with the "wait_N>0 reserve" bug) emits
**29% more launches** and **58% more ships** than led_on. In every
game led_on was eliminated (0 final planets); led_off held 20-24
planets each.

### Mechanism

The ledger's soft mode lets the chooser fire-now from a source with
a pending wait commit. Those fire-nows DRAIN the source. When the
wait commit's `wait_remaining` reaches 0, the source has fewer ships
than originally planned. `_tick_ledger` emits
`min(ships_planned, available)` — a smaller-than-planned launch that
**lands as a bounce** against the target's defenders.

In the meantime, the source's ship pool is depleted and **cannot
mount a big counter-launch when the opponent attacks**. The opponent
captures the source, eliminates us.

Hard mode (reserves the source through the wait window) avoids the
drain, but in the chooser's idle-baseline scoring the "src is busy"
state means the chooser doesn't surface fire-now alternatives — net
fewer emits than soft mode AND net fewer than baseline.

The current production's wait_N>0 reservation has the side effect of
keeping ships co-located on the planet. Those reserved ships are
available for the agent's defensive reactive policy when threats land
(via `me_defends_policy` in the chooser's rollout). The reserved ships
also accumulate over time and produce bigger counter-attack launches
when the chooser eventually does fire from that source.

## Why what-if was misleading

The what-if harness drives opp actions from the recorded replay
verbatim. Once our state diverges from the recording (turn ~10-25),
opp's actions target wrong planets, miss, or no-op. Our agent then
faces an **effectively passive opponent**. Both modes (ledger and
baseline) capture lots of planets because nothing is really fighting
back.

In real h2h, the opponent plays REACTIVELY against our current state.
Bigger fleets become important for both offense AND defense. The
ledger's tendency to emit smaller (drained) fleets translates to
bouncing offense + collapsing defense.

This generalises a friction tag from
`audit/2026-05-17-fleet-efficiency-negative-result.md`:
**"launch-rate-is-symptom-not-cause"** — chasing the under-emission
metric by adding emit logic regresses the agent because the metric
isn't the root cause. v15's chooser is co-tuned end-to-end; isolated
component changes (here: chooser-emit re-architecture) break
calibration in adversarial play.

## What the failure tells us about the original sary diagnosis

The 2026-05-20 audit (`audit/2026-05-20-filter-rejection-trace.md`)
correctly identified that 49% of sary-game turns were idle and that
wait_N>0 candidates were the top-scored. That observation is true.
But the diagnosis "this is a bug that under-emits" was incomplete.
The reservation is part of a defensive scheme: hoard ships now,
spend them at the right moment.

Sary won the sary game (and we lost it) not because we under-emitted
in the static sense but because sary's strategy emits aggressively in
a way that pulls our agent into a bad-trade exchange. Our defensive
reserve was correctly maintained but the WHOLE STRATEGY against a
sary-class opponent needs to be different — not just more emits.

## Out of scope for further iteration

Per Rule 37 (consecutive-falsification cap): two same-axis variants
have failed (soft + hard). A third attempt on the chooser-emit axis
would lock the cap. MPC was tested in what-if and presumably has the
same h2h issue (drains srcs without commitment). The ledger axis is
saturated.

## What stays in the codebase (gated off)

- `agents/baseline/main.py` — `_PENDING_LAUNCHES` dict + `_tick_ledger`
  function. Gated on `BASELINE_LEDGER=on` (default OFF). Production
  behaviour unchanged.
- `agents/baseline/chooser_trajectory.py` — `(moves, commits)` return,
  `reserved_srcs` / `reserved_for_new_commits` args. With the default
  empty sets, behaviour is identical to pre-change. Defensive logic
  for joint-candidate reservations is preserved.
- `agents/baseline/chooser.py` — same shape for composite chooser.
- `agents/_ledger_*` wrappers + `agents/_mpc` — kept for future
  re-investigation if the diagnosis is revisited.
- `scripts/whatif_postmortem.py`, `scripts/_ledger_ab_driver.py` —
  diagnostic infrastructure.

## Next-session brainstorm (Rule 7, 5 untried mechanisms)

1. **Better opp model in the rollout.** Replace
   `lib.opp_model.lite_greedy_policy` with a learned policy trained
   on top-LB replays. The chooser's plan scoring is only as good as
   its opp model; a sary-class opp model would price wait commits
   correctly (or expose them as bad). Effort: 2-4 days.
2. **Score the defensive value of a reserve.** Add a leaf-favor term
   that rewards "ships sitting on a planet near an enemy" — i.e.,
   pricing the defensive option. Would explicitly justify wait_N>0
   reservation rather than treating it as accidental. Effort: 1 day.
3. **Multi-step planning (MCTS or beam search) over candidate
   sequences.** Replace the greedy chooser with depth-K planning that
   considers `wait_N=0 now` AND `wait_N=k then capture` as full
   trajectories. Properly priced wait plans without the
   statelessness bug. Effort: 4-7 days.
4. **Sary-class opponent in the local panel.** Build a fast emitter
   that mimics sary's launch cadence (1.7+/turn). Use it as a panel
   anchor. Today the panel is too soft to surface this regression
   class. Effort: 1-2 days.
5. **Direct LB-replay imitation learning.** Pull top-LB-team replays,
   train an imitation policy. Use either as the agent itself OR as
   the rollout's opp model. Effort: 3-5 days.

Recommendation: PI direction needed. Option 4 (sary-class panel) is
the cheapest enabler — every other axis benefits from a real local
opponent that reproduces the live failure mode.

## Provenance

- h2h n=8 soft: stdout in
  `/tmp/claude-0/.../tasks/bfeo6ztca.output` (snapshotted in writeup).
- h2h n=8 hard: stdout in same dir under `b0v85p3hq.output`.
- Captured replays: `audit/replays/20260520T073756Z/*.json.gz` (4 games).
- A/B driver: `scripts/_ledger_ab_driver.py`.
- Wrappers: `agents/_ledger_{on,off,hard}/main.py`, `agents/_mpc/main.py`.
