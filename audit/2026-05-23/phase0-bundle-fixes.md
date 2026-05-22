# Phase 0 — bundle infra fixes and topology env-var threading

> Session: 2026-05-22 evening, branch `claude/strategy-axis-decision-3437`.
> Plan ref: `/root/.claude/plans/composed-noodling-riddle.md`.
> Sister session ran in parallel on the same branch — see commits
> `7f0b607` (Rules 44-47 + clean_ab), `ef613eb` (multi-line imports +
> indent-preserving alias rebind), `96ca45f` (W calibration), `62c6429`
> (bundle module order). My commits this session: `109c1d3` and `bd431b5`.

## Outcome

Phase 0 GREEN. Both bundles produced, loadable with callable `agent`,
parity test passes, topology threading verified live.

| Gate | Result |
|---|---|
| `submissions/analytical_phase_c.py` loadable | ✅ (854,139 bytes, agent callable) |
| `submissions/_phase4_step1_FND.py` loadable | ✅ (854,139 bytes, agent callable, topology OFF) |
| `tests/test_bundle_analytical_phase_c_parity.py` | ✅ 3 passed |
| Topology env-var threading (LP_TOPOLOGY_FEATURES) | ✅ ON-bundle reports `_topology_features_enabled() = True`; OFF-bundle reports `False` |
| `fast.py play` smoke seed=42 | ✅ 500 steps, valid game (vs default v7_0) |
| `env.run([analytical_phase_c.py, _phase4_step1_FND.py], debug=True)` seed=42 | ✅ 205 steps, rewards (-1, 1), no exception |
| `check_fleet_outcomes` seed=42 | ✅ 102 emits, 100% target rate, 0 sun, 0 oob |

## Bugs found and fixed

### 1. Bundler indent-preservation on function-local imports

**Symptom**: `IndentationError: expected an indented block after 'try' statement`
on the first re-bundle attempt.

**Root cause**: Both `scripts/bundle_agent.py::_clean_lib_source` (and
`_clean_agent_source`) and `scripts/bundle_analytical_phase_c.py::_strip_imports`
emitted alias rebindings at column 0 regardless of the original import's
indent. Function-local imports inside `try:` / `if cond:` blocks ended
up with the rebinding outside the enclosing block AND the block body
empty — `try: <empty>` is an IndentationError.

**Fix**: preserve `_leading_ws(line)` on every emitted rebinding line.
For imports with no aliases (`from lib.X import Y`), emit
`{indent}pass  # inlined import stripped` to keep the block non-empty.
Same pattern applied to BOTH bundlers (we have two — `bundle_agent.py`
for the baseline agent, `bundle_analytical_phase_c.py` for the analytical
agent).

Note: sister session's commit `ef613eb` made a similar fix in
`bundle_agent.py`. After rebase the changes are complementary —
`bundle_agent.py` has both fixes; mine added `bundle_analytical_phase_c.py`.

### 2. `lib/kinematic_table.py` missing from `DEFAULT_LIB_ORDER`

**Symptom**: `bundler: lib import(s) without a corresponding module in --lib order:
agents/baseline:224: from lib.kinematic_table import ...`.

**Fix**: added `"kinematic_table"` to `scripts/bundle_agent.py::DEFAULT_LIB_ORDER`
between `"orbit"` and `"aim"` (depends on geometry + orbit; consumed by
trajectory.py's hot-path lookup).

### 3. `lib/joint_solver/trajectory_matrix.py` + `opening_search.py` missing from `JOINT_SOLVER_ORDER`

**Symptom**: `NameError: name 'opening_search_enabled' is not defined`
when running the bundled analytical agent.

**Fix**: added both modules to `scripts/bundle_analytical_phase_c.py::JOINT_SOLVER_ORDER`
in dependency order (trajectory_matrix → opening_search, after opening_planner).
Also added pipeline dependencies of `decision_depth2_search` to
`PIPELINE_ORDER`: `opp_mirror_analytical`, `portfolio_enum`,
`portfolio_enum_lp_seeded`, `decision_outcome_aware_discounted`,
`decision_depth2_search`.

### 4. Module-level name collisions in flat bundle

The bundle concatenates all lib + pipeline + agent files into a single
file; module-level names with the same identifier collide and last-write
wins. This caused multiple silent failures (`debug=False` swallows the
exception, game ends at step=2 with rewards=None).

Found and renamed 13 colliding privates:

| Collision | Files | Resolution |
|---|---|---|
| `_DEFAULT` | kinematic_table, trajectory_matrix, pending_schedule | kinematic_table keeps `_DEFAULT`; trajectory_matrix→`_TM_DEFAULT`; pending_schedule→`_PS_DEFAULT` |
| `get_default` | kinematic_table, trajectory_matrix | →`get_default_table` / `get_default_matrix` |
| `clear` | kinematic_table, trajectory_matrix, pending_schedule | →`clear_table` / `clear_matrix`; pending_schedule keeps `clear` (only one left) |
| `_as_dict`, `_build_columns`, `_num_seats` | mpc, perception (and prerank) | mpc → `_mpc_*` prefix |
| `_kinematic_table_enabled` | trajectory, perception | perception → `_perception_*` |
| `_solve_milp`, `_greedy_fallback`, `_build_candidates` | opening_planner, opening_search | opening_search → `_opening_search_*` |
| `_source_inventory`, `_greedy_fallback` | lp, lp_outcome, portfolio_enum | lp_outcome → `_lp_outcome_*`; portfolio_enum → `_portfolio_enum_*` |
| `_ships_to_capture` | portfolio, opening_planner | portfolio → `_portfolio_*` |

External callers updated where the name was imported with the old
spelling (mostly tests using `from lib.X import get_default` style —
remapped to `from lib.X import get_default_table as get_default`).

### 5. Topology env-var setdefault threading

**Symptom**: bundle reports `_TOPOLOGY_FEATURES_ENABLED = False` even
when `LP_TOPOLOGY_FEATURES=1` is set via `os.environ.setdefault` in
the agent.

**Root cause**: `lp_outcome.py:148-159` evaluated `_TOPOLOGY_FEATURES_ENABLED`
at module load (a constant). In the bundle, the inlined lp_outcome.py
section runs BEFORE the agent's setdefault — so the constant locks to
False regardless of what the agent later sets.

**Fix**: converted the four flags to lazy functions
(`_topology_features_enabled()`, `_reach_bonus_enabled()`,
`_defense_bonus_enabled()`, `_front_penalty_enabled()`) that read
`os.environ.get(...)` at call time. Updated all 4 call sites
(`lp_outcome.py:544, 561, 578, 846`). Pin tests in
`tests/test_lp_topology_features.py` updated to use `monkeypatch.setenv`
instead of `monkeypatch.setattr` (15/15 green).

Verification post-fix:

```
analytical_phase_c.py: LP_TOPOLOGY_FEATURES env=1, _topology_features_enabled() = True
_phase4_step1_FND.py:  LP_TOPOLOGY_FEATURES env=0, _topology_features_enabled() = False
```

## Next steps

Phase 0.5 (physics sanity probe) is running. Phase β (topology A/B) is
the queued next experiment if 0.5 passes.

Sister session's W-calibration result (`audit/2026-05-23/calibrate_W_results.json`):
- (b1) Pearson r(W, focal_reward) = 0.545 (below 0.6 gate, marginal).
- (b2) ΔW spread healthy.
- (b3) ΔW-per-action vs outcome-shift r = 0.044 (very weak).

Implication: Phase α (smooth ΔW) should use conservative λ_W ≤ 0.3,
not 1.0. PI decision pending per sister session's commit message.
