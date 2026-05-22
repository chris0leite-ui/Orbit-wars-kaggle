# 2026-05-22 — Physics substrate extraction from `claude/strategy-axis-decision-3437`

**Branch:** `claude/extract-physics-trajectory-Vjaz9`
**Commit:** `72fe45a` — "feat(physics): extract kinematic table substrate from strategy-axis branch"
**Purpose:** Hand the next agent build a clean physics/trajectory base
without copying any strategy or chooser/proposer code from the
sibling Phase η branch.

## What landed

| Path                                        | Status | Lines | Notes                                                                                |
|---------------------------------------------|--------|------:|--------------------------------------------------------------------------------------|
| `lib/kinematic_table.py`                    | NEW    | 436   | Per-turn precompute of planet positions (static/orbital/comet). Singleton + fingerprint rebuild. |
| `lib/orbit.py`                              | +37    | -     | Added `predict_relative_cached(planet, ω, lead, *, table=None)`.                     |
| `lib/trajectory.py`                         | +47    | -     | Gate behind `KINEMATIC_TABLE_ENABLED=1`. Falls back to inline build on any miss.     |
| `tests/test_kinematic_table_parity.py`      | NEW    | 621   | `==` parity pins (static/orbital/comet/off-board-sentinel). No tolerance.            |

## What was deliberately NOT extracted

- `lib/joint_solver/trajectory_matrix.py` — couples to
  `agents.baseline.proposer.aim_and_eta`; that is agent-layer infra,
  not pure physics.
- `tests/test_kinematic_table_full_game_parity.py` — imports specific
  agents (`analytical_phase_c`, `baseline`).
- All strategy code from the sibling branch: pipeline/decision/
  joint_solver/missions/chooser/proposer/value_heads/opp_model.

## How to use it (for the next agent build)

1. **Default OFF.** Existing call sites of `predict_fleet_fate` are
   unchanged. The env-var gate keeps the old inline path live until
   the new agent explicitly opts in.

2. **Opt-in (per-process):**
   ```bash
   export KINEMATIC_TABLE_ENABLED=1
   ```
   Plus the agent must call `lib.kinematic_table.begin_turn(world)` at
   the top of every turn (cheap; idempotent within a turn).

3. **Single-call lookup:**
   ```python
   from lib.kinematic_table import lookup_relative, window, get_default
   xy = lookup_relative(pid, lead)                      # bit-identical to predict_relative
   pos_by_pid = window(pids, start_offset=wait_N, length=max_steps + 1)
   ```

4. **Slow-path safety:** `predict_relative_cached(planet, ω, lead,
   table=None)` falls through to `predict_relative` for synthetic /
   hypothetical-future planet states (fixed-point loops, opp-model
   projections). NEVER hand the table synthetic data.

## Verification

- Unit tests: 39 / 39 green
  (`tests/test_kinematic_table_parity.py`, `test_orbit.py`,
  `test_trajectory.py`).
- Broader sweep: 80 passed, 1 skipped, 0 failed
  (geometry/orbital-safety/proposer/snipe).
- End-to-end parity smoke (2-planet world, ω=0.05, fleet → sun):
  `FleetFate` identical with table primed vs cold (slow-path fallback).

## Where to find the source

The full Phase α/β/γ plan and bit-parity rationale lives in the
file's module docstring (`lib/kinematic_table.py:1-35`). The sibling
branch's Phase η (opening trajectory matrix) is the next layer up and
was skipped here because it imports `agents.baseline.proposer`.
