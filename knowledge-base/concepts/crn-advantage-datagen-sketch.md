# CRN-paired advantage data-gen — design sketch for Phase B-1 (merged with B-3)

Status: SKETCH, not yet implemented. Companion to
`knowledge-base/thoughts/2026-05-28-pm-distillation-action-rank-collapse.md`.

## What this script generates

A training corpus of `(features, advantage)` pairs where
`advantage = margin_action_at_game_end − margin_idle_at_game_end`,
computed by two paired rollouts from the same state with the same
opponent RNG. Replaces Phase A's `(features, favor_hybrid_scalar)`
corpus.

## Why we can't reuse `scripts/gen_value_training_data.py` as-is

The Phase A script labels each state with `favor_hybrid(state)`, a
deterministic function of state alone. The CRN advantage label is a
function of `(state, action, rollout_policy, opp_RNG)`, and requires
the script to:

1. Pick an action `a` for the focal seat at the sampled turn.
2. Branch the game state.
3. Replay forward from the branch point with locked opponent RNG and
   chooser policy — twice (once doing `a`, once doing nothing).
4. Record the difference in final margin.

`kaggle_environments` does not expose clean state cloning, and its
internal RNG isn't snapshot/restore-able from the agent side. The
sketch below uses `lib/fast_sim.py` which is byte-exact to the Kaggle
engine and has `clone()` + deterministic `step()`.

## Concrete sketch (pseudocode)

```python
# scripts/gen_value_training_data_crn.py  (new file)

from lib.fast_sim import from_obs, clone, step, rollout, ship_totals
from lib.value_features import extract_features
from agents.baseline.main import agent as our_agent   # production chooser
# Strong opp: baseline_pv_eta lives on sibling branch.
# Copy submissions/baseline_pv_eta.py into a path the script can load.
OPP_PATH = "submissions/baseline_pv_eta.py"   # extracted at session start

SEATS = 2
SAMPLES_PER_GAME = 8        # subsampled, NOT every turn
GAME_MAX_TURNS = 250         # hard cap on rollout depth
MIN_SAMPLE_TURN = 5          # avoid pre-game-trivial states


def _opp_callable():
    """Load baseline_pv_eta as a kaggle-style (obs, config) -> emits."""
    spec = importlib.util.spec_from_file_location("_opp", OPP_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m.agent


def _focal_margin(snap, focal_seat):
    """Signed ship-margin from focal seat's POV at terminal state."""
    by_owner = ship_totals(snap)
    me = by_owner.get(focal_seat, 0.0)
    other = sum(v for k, v in by_owner.items() if k != focal_seat)
    return float(me - other)


def _rollout_to_end(snap, our_policy, opp_policy, max_turns):
    """Step forward until terminal or max_turns hit, using fixed policies."""
    while not snap.done and snap.turn < max_turns:
        # Both seats decide from the SAME snap.observation;
        # snap.step() applies their joint action.
        a_us = our_policy(snap.obs_for(0), snap.config)
        a_opp = opp_policy(snap.obs_for(1), snap.config)
        snap = step(snap, [a_us, a_opp], in_place=True)
    return snap


def _one_game_samples(seed, our_policy, opp_policy, samples_per_game):
    """Play one game, then back-fill paired rollouts at sampled turns."""
    # 1) Play the canonical game and record the turn-by-turn history.
    snap = from_obs.fresh(seed=seed, num_seats=SEATS)
    history = []   # list of (snap_clone, focal_action) per turn
    while not snap.done and snap.turn < GAME_MAX_TURNS:
        snap_t = clone(snap)   # for later branch points
        a_us = our_policy(snap.obs_for(0), snap.config)
        a_opp = opp_policy(snap.obs_for(1), snap.config)
        history.append((snap_t, a_us))
        snap = step(snap, [a_us, a_opp], in_place=True)

    if len(history) < MIN_SAMPLE_TURN + 1:
        return None   # too short to sample

    # 2) Subsample turns to label.
    rng = random.Random(seed + 999983)
    pool = list(range(MIN_SAMPLE_TURN, len(history)))
    sample_turns = sorted(rng.sample(pool, min(samples_per_game, len(pool))))

    # 3) For each sampled turn, paired rollouts.
    rows = []
    for t in sample_turns:
        snap_t, a_us_t = history[t]
        focal_obs = snap_t.obs_for(0)
        feats = extract_features(focal_obs, me=0, num_seats=SEATS)

        # Leg A: actual action a_us_t.
        snap_a = clone(snap_t)
        a_opp_t = opp_policy(snap_t.obs_for(1), snap_t.config)
        snap_a = step(snap_a, [a_us_t, a_opp_t], in_place=True)
        snap_a = _rollout_to_end(snap_a, our_policy, opp_policy, GAME_MAX_TURNS)
        margin_a = _focal_margin(snap_a, focal_seat=0)

        # Leg B: idle (empty emits). SAME opp_action at the divergent step.
        snap_b = clone(snap_t)
        snap_b = step(snap_b, [[], a_opp_t], in_place=True)
        snap_b = _rollout_to_end(snap_b, our_policy, opp_policy, GAME_MAX_TURNS)
        margin_b = _focal_margin(snap_b, focal_seat=0)

        advantage = margin_a - margin_b
        rows.append((feats, advantage))

    return rows
```

(Real script: add `multiprocessing.Pool`, chunked `.npz` writes, the
existing `--pairing` / `--merge` plumbing, and the focal-seat-swap
trick from the Phase A script.)

## Locked-opp-RNG mechanics

The CRN trick only cancels variance if both legs see the **same** opp
action at the divergent step. Two guarantees we need:

1. **Same `snap_t` observation feeding the opp policy.** Achieved by
   computing `a_opp_t = opp_policy(snap_t.obs_for(1), ...)` ONCE and
   reusing it for both legs at the divergent step.
2. **Determinism downstream.** If `our_policy` and `opp_policy` are
   deterministic given `(obs, config)`, the remaining rollout is
   deterministic too. `baseline_pv_eta` is bundle-deterministic
   (env-var-gated, no `random` calls in its hot path — verify before
   shipping). The trajectory chooser uses no stochasticity either.
3. **No `np.random` global state pollution.** Workers each get their
   own seed; no shared global RNG.

If we ever introduce stochastic agents (Thompson tie-breaking,
softmax-sampling search), the CRN trick generalises by snapshotting
that agent's RNG seed at the divergent step and replaying both legs
with the same seed. Out-of-scope for B-1.

## What the focal-seat side runs during rollout

Two reasonable choices, with different bias / cost tradeoffs:

| Choice | Cost | Bias |
|---|---|---|
| Same chooser as during data-gen (`baseline` w/ `favor_hybrid` head, BASELINE_WALLCLOCK_MS=100) | High (full chooser per turn × 200 turns × 2 legs × samples) | Unbiased estimate of "this action's advantage under our actual play" |
| Cheap rollout policy (e.g. `favor` argmax of top-K candidates only) | ~10× cheaper | Biased toward cheap-policy strengths; head learns "what's good under cheap-policy continuation" |

**Default: same chooser.** The advantage label has to reflect what
our chooser will actually do; biasing the rollout undermines the
target. If cost is prohibitive, drop `SAMPLES_PER_GAME` from 8 to 4
or reduce `BASELINE_WALLCLOCK_MS` for rollout legs (but not for the
canonical game).

## Compute budget (back-of-envelope, refinement #2)

Per game:
- Canonical play: 200 turns × 150 ms × 2 seats = 60 s
- 8 sampled turns × 2 legs × (200 − sample_turn) × 150 ms × 2 seats
  averaging at half-game depth ≈ 8 × 2 × 100 × 150 ms × 2 = 480 s
- ≈ 540 s ≈ 9 min per game

Corpus targets:
| Corpus size | Wallclock single-core | 8 workers | Practical path |
|---|---|---|---|
| 500 games (4000 examples) | 75 h | 9 h | overnight local |
| 1500 games (12000 examples) | 225 h | 28 h | Kaggle GPU kernel |
| 5000 games (40000 examples) | 750 h | 94 h | multi-kernel chain |

Recommendation: **start with 500 games local 8-worker overnight as
diagnostic**; only scale to 1500+ on Kaggle if the 500-game candidate
shows directional lift (Spearman-τ on held-out + Wilson-lo near 0.45
vs `pv_eta`).

## Training-side changes

`scripts/kaggle_value_head_kernel/train.py` works as-is on (X, y).
Two additions:

1. **Rank-correlation diagnostic.** After each epoch, compute
   Spearman-τ between predicted-advantage and held-out true-advantage,
   not just RMSE / R². Mandatory gate per the Phase A failure mode.
2. **Per-state action-Δ diagnostic (optional but worth it).** Group
   examples by `(state)` — when multiple actions were sampled from
   the same state (we'd need to sample 2-3 candidates per state for
   this to be meaningful), compute pairwise rank-agreement between
   predicted and true. This is the direct chooser-relevance metric;
   the Spearman across all examples is a cheaper proxy.

If Spearman-τ < 0.3 on held-out, do NOT bundle / A/B. The training
signal didn't transfer; debug the gen / label step before sinking
A/B compute.

## Known unknowns / things to verify before code

- [ ] **fast_sim API ergonomics.** `from_obs.fresh(seed=...)` may not
      be the exact constructor; verify `lib/fast_sim.py:172` and the
      `obs_for(seat)` accessor. The sketch above uses the names the
      `__doc__` suggests; real code needs to use the actual API.
- [ ] **`baseline_pv_eta` determinism audit.** Search the bundle for
      `random.`, `np.random.`, `time.`. None expected, but a single
      `random.choice` for tie-breaking would silently break CRN.
- [ ] **Focal-seat swap.** Phase A's script ran both seats; the CRN
      version probably should too (label twice per state, once per
      seat) for data efficiency.
- [ ] **Action representation.** `a_us_t` is the chooser's emit list
      (list of `[src, tgt, ships]` triples). The "idle" baseline is
      `[]`. Are there cases where the chooser ALWAYS emits something
      and `[]` never appears in normal play? Yes — many states.
      That's fine; `[]` is a legal action under the game rules and
      the advantage is meaningful relative to it.
- [ ] **Termination handling.** If a leg ends earlier than the other
      (one side eliminated before max_turns), the final margins are
      still well-defined. Confirm `_focal_margin` reads `ship_totals`
      from a terminal snapshot correctly.

## Sequencing if PI greenlights

1. Verify the four "known unknowns" above (≤30 min).
2. Implement `scripts/gen_value_training_data_crn.py` (~half day,
   following the sketch).
3. Smoke run: 4 games × 4 samples, single-core, verify the .npz
   structure and label scale (advantage typical magnitude should be
   in the same range as final margins, maybe ~half).
4. 500-game overnight local run (8 workers).
5. Train + Spearman-τ gate.
6. Bundle + A/B vs `baseline_pv_eta` n=32 (NOT vs `favor_hybrid`).
7. Decision: lift → continue to 1500-game GPU corpus. No lift →
   Option B (structural rethink).

## Pointers

- `knowledge-base/thoughts/2026-05-28-pm-distillation-action-rank-collapse.md`
  — diagnostic results + interpretation that triggered this sketch.
- `scripts/gen_value_training_data.py` — Phase A scalar-label gen
  (kept as reference, NOT modified).
- `lib/fast_sim.py` — the deterministic clone/step substrate this
  script depends on.
- `submissions/baseline_pv_eta.py` (sibling branch
  `claude/kaggle-submission-review-gZsCu`) — strong opp; extract via
  `git show <branch>:submissions/baseline_pv_eta.py > <local_path>`.
