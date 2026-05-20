# 2026-05-20 — Stateful Commit Ledger Design + Cross-Game Validation

**Branch**: `claude/audit-workflow-performance-btjeK`
**Audit predecessor**: `audit/2026-05-20-filter-rejection-trace.md`
**Plan**: `/root/.claude/plans/so-now-research-and-zany-widget.md`

## TL;DR

The trajectory chooser's `wait_N > 0 reserve-without-emit` rule
(`chooser_trajectory.py:856`, `chooser.py:179`) caused 49% idle turns in
the sary loss because the agent is stateless across turns and the wait
commit was never honoured. The proper fix is a **stateful commit ledger**:
the agent remembers wait commitments across turns and emits them on
schedule, with re-aim against current src/tgt geometry. Default mode
"soft" lets the chooser still fire-now from a src that has an in-flight
commit (the commit drops at emit time if not enough ships remain).

What-if rollout harness validated the design on 6 recent live episodes
(4 losses, 2 wins; 4×2P + 2×4P). Aggregate result: **ledger_soft wins on
final-planet count in 6/6 games** (total: 15 → 118 final planets, 7.9×
improvement). Idle rate drops by 5pp+ in 4/6 games (up to 17pp in
77133549). Launch volume rises +28% across the corpus. Wallclock p95
DOWN slightly (367ms → 327ms — the ledger tick is cheap; the chooser
sees fewer candidates and finishes faster).

## Architecture

### The ledger

Module-level dict in `agents/baseline/main.py`:

```python
_PENDING_LAUNCHES: dict[int, list[dict]] = {}  # keyed by player_id
```

Entry shape:

```python
{
    "src_id": int, "tgt_id": int,
    "ships_planned": int, "angle_original": float,
    "wait_remaining": int, "commit_step": int,
    # populated by validation:
    "fired_at_step": int,  # on successful emit
    "drop_reason": str,    # on drop (src_lost / tgt_now_ours / ...)
}
```

### Per-turn lifecycle (in `agent()`)

1. **New-match detection** — `obs.step == 0` clears `_PENDING_LAUNCHES[me]`.
2. **Tick + emit due commits** — `_tick_ledger(me, world, model, omega)`:
   - Decrement every entry's `wait_remaining` by 1.
   - If `wait_remaining > 0`: keep alive (survivors).
   - If `wait_remaining <= 0`: validate + emit OR drop:
     - Drop if `src_lost` (we don't own src), `tgt_now_ours` (already
       captured), `src_empty`, `planet_missing`, `aim_failed`.
     - Otherwise re-aim with `proposer.aim_and_eta(src, tgt, ships,
       omega, wait_N=0)` — essential because planets orbit during the
       wait — and emit `[src_id, angle, min(ships_planned, src.ships)]`.
3. **Build reservation sets** for the chooser:
   - `reserved_for_new_commits = firing_srcs ∪ pending_srcs` — block
     stacking a second commit on a src that already has one.
   - `reserved_srcs` (blocks fire-now emits from src):
     - Hard mode: `firing_srcs ∪ pending_srcs` — preserve ship reserve.
     - **Soft mode (default)**: `firing_srcs` only — chooser can
       fire-now from a src with a pending commit; commit drops at
       emit time if not enough ships remain.
4. **Run chooser** — `choose_trajectory(..., reserved_srcs,
   reserved_for_new_commits)`:
   - Scoring loop: skip candidates blocked by the appropriate set.
   - Emit loop: wait_N==0 → moves; wait_N>0 → commits (the new return).
   - Joint enumeration also honours `reserved_srcs`.
5. **Merge** — `_PENDING_LAUNCHES[me] = surviving + new_commits`; return
   `due_moves + chooser_moves`.

### Why "soft" beats "hard" empirically

Hard mode reserves the src for the entire wait window (5-10 turns). The
src ACCUMULATES ships during the wait. But while accumulating, the src
can't fire-now at smaller-but-positive opportunities. Empirically, on
the sary game (and 4 others), the cost of those missed fire-nows
exceeds the value of the bigger planned capture.

Soft mode keeps the src available for fire-now. At emit time:
- If ≥ `ships_planned` remain (production accrual minus what the
  chooser fired meanwhile): emit the commit. Big capture lands.
- If fewer ships remain: still emit `min(planned, available)`.
  Smaller-than-planned capture; may bounce or partial-capture.

Either way the src isn't wasted. The empirical 6-game data agrees.

## What-if rollout harness — `scripts/whatif_postmortem.py`

Simulates a recorded live episode from turn 0 forward, driving fast_sim
with:
- OUR seat's action — produced by `agents.baseline.main.agent` under a
  configurable chooser policy.
- OPP seats' actions — replayed verbatim from the recording. Once our
  state diverges from the recording, opp's actions may target unexpected
  state; the engine drops illegal ones. That's the closest we can get
  to a counter-factual without an opp model.

Supported policies (env-var driven):
- `baseline` — current chooser, no change.
- `mpc` — drop wait_N>0 from chooser's scored list entirely (pure
  receding-horizon).
- `ledger` — hard ledger (reserves src for the wait window).
- `ledger_soft` — soft ledger (default; the shipped variant).

Per-policy output: per-turn action stream, divergence point, ledger
emit/drop telemetry, final state diff vs recorded.

## Cross-game validation (6 episodes)

```
episode      size  base_idle  led_idle  Δpp   base_L  led_L  +%    base_FP  led_FP
77133549     2P    134(79%)   105(62%)  -17   42      71    +69%   0       23
77135140     2P    108(70%)   91(59%)   -11   57      80    +40%   0       27
77140674     2P    58(48%)    46(38%)   -10   101     112   +11%   9       22
77137480     2P    93(59%)    98(62%)   +3    90      86    -4%    6       12
77150441     4P    153(91%)   128(76%)  -15   15      47    +213%  0       13
77158235     4P    128(66%)   121(62%)  -4    79      96    +22%   0       21

aggregate            674 idle  589 idle        384 L   492 L         15      118
ledger_soft wins idle (≥5pp drop): 4 of 6
ledger_soft wins final-planets:    6 of 6
```

Caveat: opp's recorded actions become stale after divergence; final
planet count under-counts the regression risk in adversarial play. But
the launch-volume + idle-rate signals are unambiguous and pre-divergence
checks confirm the ledger is firing genuinely better actions early.

## Wallclock

```
baseline      p50=78ms  p95=367ms  p99=497ms  max=615ms  (n=966)
ledger_soft   p50=114ms p95=327ms  p99=451ms  max=553ms  (n=957)
```

p50 +36ms (the ledger tick); p95/p99/max all lower. Well within the
1000ms env budget. No perf regression.

## Code changes

Production code:
- `agents/baseline/main.py` — `_PENDING_LAUNCHES` dict, `_tick_ledger`
  function, both trajectory and composite chooser callers wire the
  ledger + reservation sets; ledger gated on `BASELINE_LEDGER=on`
  (default `off`). Mode `BASELINE_LEDGER_MODE` ∈ `{hard, soft}` (default
  `hard`; harness sets `soft` for the validated policy).
- `agents/baseline/chooser_trajectory.py` — `choose_trajectory` returns
  `(moves, commits)`. New args `reserved_srcs`, `reserved_for_new_commits`.
  Scoring loop and joint enumeration both honour the reservations.
- `agents/baseline/chooser.py` — same shape for the composite chooser
  (parallel implementation; default chooser is trajectory).

Test updates:
- `tests/test_chooser_trajectory.py` — unpacked tuple return.
- `tests/test_baseline_chooser.py` — empty-prerank returns `([], [])`.

Diagnostic / harness:
- `scripts/whatif_postmortem.py` — new what-if rollout harness with
  4 policies; ledger telemetry (commits, drops by reason, emit
  success rate).

## Default behavior (production)

`BASELINE_LEDGER=off` by default. Bundled submission behaviour unchanged
unless explicitly enabled (e.g., by setting `BASELINE_LEDGER=on
BASELINE_LEDGER_MODE=soft` in the bundle's setdefault block).

**Next-session decision (Phase 6)**: flip the default to `on,soft` AND
re-bundle. Local A/B vs champion 52827111 (n=64) + regression check
on v15-era seeds (Forrest, 213tubo). Ship if Wlo ≥ 0.55 and panel
clears.

## Open questions for execution

1. **Should ledger entries expire after N turns?** Currently entries
   stay alive until wait_remaining hits 0 or src/tgt invalidates. If
   the chooser commits wait_N=15 and the world shifts dramatically over
   15 turns, the commit may be stale. Cross-game data shows 90% emit
   success rate — most commits do execute meaningfully — so an expiry
   is not urgent.

2. **Should commits be re-validated each tick (cancel if no longer
   optimal)?** Soft mode partially addresses this by letting fire-now
   override. Hard re-validation would mean re-scoring each pending
   entry against current state. Higher cost; likely small benefit.

3. **4P-specific tuning?** 4P games show smaller improvements
   (-4pp idle, +22% launches in 77158235) than 2P. The joint candidate
   path is 2P-only, so ledger interacts differently. Worth a 4P-specific
   audit if Phase 6 shows 4P regression.

4. **Wait-N max cap?** Current proposer enumerates wait_N up to
   `MAX_HORIZON-eta-SIM_SETTLE_TURNS`. Capping at e.g., `wait_N <= 10`
   would prevent extreme long-wait commits. Cross-game data doesn't
   show this as an issue yet.

## Provenance

- Diagnostic data: `audit/whatif/52827111/{77133549,77135140,77137480,77140674,77150441,77158235}/`.
- Predecessor audit (filter-rejection trace): `audit/2026-05-20-filter-rejection-trace.md`.
- Replay-driven scoring trace harness: `scripts/baseline_postmortem.py`.
- What-if rollout harness: `scripts/whatif_postmortem.py` (this session).
