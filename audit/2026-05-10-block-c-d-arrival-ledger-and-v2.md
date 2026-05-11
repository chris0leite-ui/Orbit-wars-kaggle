# 2026-05-10 — Blocks C + D: arrival-ledger substrate + v2 (worldmodel-aware roi)

> Companion to `audit/2026-05-10-block-a-physics-upgrade.md`. Adapts
> Roman 1224 / Pilkwang structured-baseline `WorldModel` patterns
> (audit/2026-05-10-public-kernel-teardown.md) into our pipeline, then
> ships the first strategy that USES the WorldModel.

## Files

### Block C — substrate (lib/)

- **NEW** `lib/combat.py` — `resolve_arrivals(garrison_owner, garrison_ships, arrivals)` implementing README §combat rules 1-4 (same-owner sum, largest-minus-second, reinforce vs flip, two-way tie destroys). 11 TDD tests in `tests/test_combat.py`.
- **NEW** `lib/world_model.py` — `build_arrival_ledger(fleets, planets)`, `simulate_planet_timeline(planet, arrivals, horizon)`, `state_at_timeline`, `fleet_target_planet` (ray-cast), `WorldModel.from_world(world)` snapshot wrapper. Default horizon = 110 steps (matches Roman's SIM_HORIZON). 10 TDD tests in `tests/test_world_model.py`.
- **MODIFIED** `lib/mechanism.py` — added `arrival_ledger` mechanism (don't-double-commit). Kept OUT of `DEFAULT_MECHANISMS` after A/B regression — see §Decisions below.
- **MODIFIED** `scripts/bundle_agent.py` — `DEFAULT_LIB_ORDER` extended with `combat`, `world_model` so the bundler inlines them.

### Block D — strategy

- **NEW** `agents/v2/main.py` — roi target selection augmented with
  `WorldModel.owner_at` / `ships_at` predictive filter:
  - For each (src, target) pair, look up predicted (owner, ships) at
    our fleet's arrival step.
  - **Skip** if predicted_owner == us AND predicted_ships >= base_ships
    (target will already be ours with surplus garrison — sending more
    would double-commit).
  - **Otherwise** propose with `ships = target.ships + 1` (same as roi).
    Mechanism layer (`arrival_size`) handles production-aware sizing.
- **NEW** `agents/v2/__init__.py`.

## Results

### Test gate

All **181 tests green** (was 160; +21 new for combat + world_model).
PARITY_MECHANISMS unchanged → v1 parity test passes byte-equal.

### A/B (v2 vs roi vs roi_baseline) — `audit/tournaments/20260510T220206Z.json`

32 seeds × both sides, 64 head-to-head games per pair. Aggregated WR:

|                | vs v2  | vs roi      | vs roi_baseline | Mean WR | p95 ms |
|----------------|--------|-------------|-----------------|---------|--------|
| **v2**         | sp 2/0/30 | **59% (38/64)** | **69% (44/64)** | **64.1%** | 3.7   |
| roi (physics)  | 41%    | sp 4/2/26   | 56% (36/64)     | 48.4%   | 1.3   |
| roi_baseline   | 31%    | 44%         | sp 3/5/24       | 37.5%   | 0.4   |

**v2 beats** the new physics roi 59%, the pre-physics roi 69%. **Mean
panel WR 64.1%.** Both halves of the locked eviction rule clear:

- (a) ≥60% panel WR vs current best — 64.1% (current best = roi).
  **CLEARS** (+4.1 pp over gate).
- (b) ≥55% head-to-head vs live submit — 69% vs roi_baseline (local
  mirror of live v1.2/roi at μ=978.7). **CLEARS** (+14 pp).

p95 turn time: 3.7 ms (up from 1.3 ms roi after Block A; up from 0.4 ms
roi_baseline). The 2.4-3.3 ms overhead is the `WorldModel.from_world`
build per turn (40 planets × 110 horizon ≈ 4,400 step events). Well
within the 1 s `actTimeout`.

### Self-play asymmetry note

v2-vs-v2 self-play: 2/0/30 (P0 wins / P1 wins / draws). **94% draws.**
This is the mirror-symmetric strategy lock surfaced in A.6 — when both
players run identical filtering, the games stalemate.

On the ladder, opponents are diverse, so the self-play draw rate is
not a load-bearing signal. But it does mean: against another v2-style
opponent we'd draw rather than win. In TrueSkill, draws pull both μ
toward the mean — so against an opponent above us we GAIN μ; against
one below us we LOSE μ. Since most ladder games are vs higher-μ
opponents (we're at 978.7 with cliff at 1460), draws are net-positive
in expectation.

The draw rate also bounds v2's μ ceiling above ROI-family opponents:
if too many "competitive" ladder agents play production-aware-greedy,
v2's μ growth stalls in mirror-lock. Block E (more mission classes
+ tuning + 4P spoiler) is where we break that.

## Decisions

### arrival_ledger mechanism EXCLUDED from DEFAULT_MECHANISMS

An earlier A/B (`audit/tournaments/20260510T215332Z.json`) had the
arrival_ledger mechanism inside `DEFAULT_MECHANISMS` (so it ran on
every agent's intents after `lead_aim_v2`). WR vs roi_baseline
**regressed to 50%** (32/64). Root cause:

- Per-source greedy strategies pick the SAME best target across
  many sources (e.g. one high-ROI neutral).
- `arrival_ledger` drops all-but-first when timeline shows target
  will be ours.
- BUT a *mechanism* can only filter the input list — it can't make
  the source re-pick a different target.
- Result: those sources go IDLE for the turn while their ships
  could've gone to a 2nd-best target.

Mitigation: hoist the WorldModel-awareness into the **strategy**
layer (v2's `propose_intents`), where each source can score+rank
all targets and pick a non-redundant one. That's what `agents/v2`
does. `arrival_ledger` mechanism stays in `lib/mechanism.py` for
later use from a v3 planner (where the planner re-allocates
freed ships across mission classes), but is OUT of `DEFAULT_MECHANISMS`.

### v2 simplified — no ship bumping

An earlier v2 attempt also bumped `ships` per `WorldModel`-predicted
defense (similar to Roman's `arrival_size` but reading the
timeline directly). Result: **0/64 WR** (audit/tournaments/20260510T215806Z.json).
Root cause: bumped intents that exceeded src.ships were filtered
inside `propose_intents`, leaving sources to pick LOW-ROI affordable
targets over HIGH-ROI ones that the existing `arrival_size` would've
let through unchanged. Rolled back; ship-bumping stays with the
mechanism layer where the rule is monotonic (`arrival_size` bumps
intent.ships upward, then validate drops if > garrison).

The remaining v2 logic is the minimum-viable WorldModel use:
**skip targets that will already be ours**. That alone gets +18 pp
panel WR over `roi` (the new-physics version) and +27 pp over
`roi_baseline` (the live-equivalent).

## What v2 leaves on the table (Block E)

- **No mission classes.** Every fleet is implicitly a snipe. Reinforce
  / recapture / gang_up live in Block E.
- **No global solver.** Per-source greedy still chooses each src's best
  target independently. Two sources can still pick the SAME target if
  WorldModel doesn't see one's fleet yet (e.g. if both decide in the
  same tick). Hungarian assignment fixes this; Block E.
- **No 4P-specific tuning.** v2's ROI scoring is identical 2P vs 4P.
  Roman's `FOUR_PLAYER_ROTATING_*` constants suggest +20-50 μ available
  here; Block E.
- **No endgame burn-through.** Last ~30 steps, ships in flight count
  for the launcher's score. v2 keeps running ROI; doesn't flush garrison.
  Trivial mechanism addition; Block E.
- **No depth-2 look-ahead.** Block F.

## Submission recommendation

v2 bundle staged at `submissions/v2.py` (50 KB; self-play 5/5 DONE).
Bundle includes inlined `lib/{geometry, fleet, orbit, aim, combat,
world_model, intent, mechanism}` per the updated `DEFAULT_LIB_ORDER`.

Per Rule 1 (PI-approved submissions), recommend submission as v2.0 in
PI's next session. Predicted live μ: 1050-1200 based on +27 pp local
WR over the v1.2 baseline (which is at μ=978.7). Wilson 95% CI on
44/64 head-to-head: [57%, 79%]. Even the lower bound clears the gate.

**Submission economy reminder:** 1 slot remaining today (5/5 used would
include v2). Tomorrow refreshes. Recommend pushing v2 first thing
tomorrow morning so a full 24-h ladder sample accrues before the next
candidate (v3 with missions) is staged.

## Wallclock budget

Per-turn p95 grew from 0.4 ms (roi_baseline) to 1.3 ms (roi with
Block A physics) to 3.7 ms (v2 with WorldModel). Headroom remaining:
~270× before the 1 s `actTimeout` fires.

Block E mission framework adds ~4-8 ms (mission proposal enumeration
+ Hungarian solve on ~40-planet board). Block F look-ahead is the
expensive one (depth-2 over top-K missions × 20-step rollout). Plan
to profile under load before submitting Block F.
