# state/STRATEGY.md — current main strategy

> **READ FIRST.** This is the canonical "what are we running" doc. Everything
> in `CLAUDE.md` and `HANDOVER.md` points back here.

## The strategy: `baseline_adaptive_k`

**Source of truth:** commit `0025c67` on branch `claude/champion-ml-graft-majestic-storm` (adds the compute_by_ships lever on top of the adaptive_k baseline).
**Reproducible build:** `scripts/_build_compute_by_ships_bundle.sh` (uses the bundler's `DEFAULT_LIB_ORDER`; mirrors the `_build_adaptive_k_bundle.sh` template).
**Live submission #1 (most-recent):** `champ_computeByShips_on.py`, sub **53332500** (2026-06-03 15:11 UTC), bundle sha256 `53bf813b...`, 697 927 B. Adaptive K + per-source compute_by_ships lever both baked ON.
**Live submission #2 (backstop):** `champ_adaptiveK_on.py`, sub **53324164** (2026-06-03 10:37 UTC), live **μ = 1185.2** — anchor / safety net.
**Local A/B of compute_by_ships:** 16-game CRN vs same-source lever-off sibling = 7/16 wins, Wilson [0.231, 0.668] — INCONCLUSIVE (parity ± noise). Predicted live μ ≈ 1170.

### Next mechanism — large-idle-fleet spend-down

PI observation 2026-06-03: live games show planets accumulating > 200 ships and
sitting idle when no opponent is within K-eta reach. Even a "wasted" launch
from a 200-ship planet has positive expected value: it forces opponent
defense, may capture, and the ships are otherwise contributing zero.

The mechanism we're going to build next (after compute_by_ships's live result
lands): **threshold-triggered forced launch**. Any planet whose ship count
exceeds a threshold (PI mentioned 200) MUST emit at least one launch this
turn, targeted at the nearest opponent-owned planet, with the K-eta cap
BYPASSED for this specific launch class. Single launch per source per turn
(chooser's existing rule already enforces this); self-regulating (planet
fires until below threshold). No reposition to ally → no back-and-forth-loop
pathology. Compose-clean with adaptive K and compute_by_ships.

Not implemented yet. Spec-only until compute_by_ships's live μ comes in
(~24 h from 2026-06-03 15:11 UTC TrueSkill warm-up).

### What it is in one sentence

The all-time-champion config (`launch_rules_universal`, 12 env vars: joint-aggressive
multi-source coalitions / reinforcement / neutral-bonus / orbital-safety /
present-value-discount / universal K=10 launch-discipline ceiling) with **one** new
lever ON: an **adaptive horizon K** that lifts the launch-arrival ceiling in the
predictable opening (K_OPEN = 20) and decays linearly to the disciplined champion
floor (K = 10) by step 30.

`K(step) = max(10, round(20 - (20-10) · step / 30))`

### Why adaptive K

Static K = 10 hid ~75 % of the opening expansion map (median neutral ETA = 22 turns).
The opening is genuinely predictable — few in-flight fleets, planets at known
positions — so far launches are safe there. Midgame, predictability collapses and
the static K = 10 floor is the right discipline. See
`audit/2026-06-01-adaptive-horizon-k-investigation.md`.

### The single lever

`agents/baseline/launch_rules.capture_horizon_k(step)` — read by:
1. the launch-discipline gate (`enforce_launch_rules`),
2. the proposer's far-candidate prune (`agents/baseline/proposer.py`),
3. the sync-coalition cap (chooser path).

All three readers see the same step-dependent K → phase awareness propagates
consistently. The value function already evaluates at horizon ~40 so far captures
are valued once admitted.

### TrueSkill warm-up — DO NOT panic at early μ

Kaggle's TrueSkill starts every new submission at **μ = 600** and climbs as games
accumulate. Sub 53324164 will show ~600 immediately after submission and rise over
~24 h toward the predicted ~1170. Do not draw any conclusion from the first few
hours of leaderboard data.

### Iteration protocol — observation-driven

1. **PI observes** something concrete — a single-game replay, a specific loss
   pattern, a leaderboard move, a turn-by-turn trace, an opponent behaviour.
2. **PI reports** the observation in plain English. (Per CLAUDE.md Rule 0.)
3. **AI diagnoses** — minimal investigation, surface the modeling cause (per
   CLAUDE.md Rule 40: model the right thing; do NOT bump a constant).
4. **AI proposes** the smallest change that addresses the cause, gated behind a
   default-OFF env var so the champion bundle stays byte-identical until proven.
5. **PI signs off** on the proposal.
6. **AI implements** in `agents/baseline/`; smoke gates per CLAUDE.md Rule 46.
7. **AI submits** per CLAUDE.md Rules 1 / 12 / 42; appends a row to
   `state/MULTI_BRANCH.md` push-claim board.
8. Wait for the next observation. Go to step 1.

No multi-axis exploration. No speculative ports. One observation → one mechanism
→ one push. The PI is the observation source; the AI is the mechanism builder.

## How to bundle, smoke, and submit

```
# 1. Build (env-vars baked in; outputs to submissions/champ_adaptiveK_on.py).
bash scripts/_build_adaptive_k_bundle.sh

# 2. Rule 46 smoke (required before every submit).
python -m pytest tests/test_bundle.py -q                              # 15/15 expected
python fast.py play submissions/champ_adaptiveK_on.py \
       --vs submissions/v7_0_drop_one.py --seed 7                     # max turn < 1000 ms

# 3. Rule 42 gate (check evicted-μ).
kaggle competitions submissions orbit-wars | head -3                  # see rolling pair
#   append a claim row to state/MULTI_BRANCH.md before pushing

# 4. Submit with explicit PI sign-off.
kaggle competitions submit -c orbit-wars \
    -f submissions/champ_adaptiveK_on.py \
    -m "<plain-English description + sub-id of evicted submission + sha256 of new bundle>"
```

## How to modify

A new mechanism lives in `agents/baseline/` (a new module or a new function in an
existing one). Gate behind `BASELINE_<MECHANISM>` env var, default OFF →
byte-identical champion when the var is unset. The build script bakes the var ON
in the bundle header. Pattern: see `scripts/_build_adaptive_k_bundle.sh` and the
env-header block at the top of `submissions/champ_adaptiveK_on.py`.

Tests for the new mechanism go in `tests/`. The Rule 46 bundle test
(`tests/test_bundle.py`) re-runs automatically and protects byte-parity of the
default-OFF code path.

## Pointers

- `CLAUDE.md` — process rules (kept lean).
- `HANDOVER.md` — next-session brief (kept lean, points here).
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42); historical track registry trimmed.
- `state/TOOLS.md` — tools registry (A/B harnesses, diagnostics, validation, bundler).
- `comp-context.md` — settled-once competition facts (env spec, deadline, gate clearance).
- `audit/` — append-only audit trail (postmortems, investigations, replays).
- `knowledge-base/` — PI second-brain (`thoughts/`, `concepts/`, `flags/`, `questions/`).
