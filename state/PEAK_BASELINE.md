# PEAK_BASELINE — single source of truth for "build on top cleanly"

**Read this before proposing any change to the baseline agent.** The peak is
the strongest historical bundle we have ever submitted. Every recent
attempt to build on top of it has regressed; this document fixes the
common starting point and the protocol so the next iteration doesn't.

---

## What "the peak" is (2026-05-29 update — supersedes the peak-1165 line below)

| field | value |
|---|---|
| Git commit | `0d71aa6` (`bundle: regenerate baseline.py + add baseline_pv_eta.py wrapper`) |
| Bundle SHA-256 (prefix) | `7964bfa4` (full: `7964bfa4b0ceaef7942c515179fbd549687aec2db1faf1baedb7016a23e6dfff`) |
| Frozen anchor in tree | [`submissions/baseline_pv_eta_anchor_1163.py`](../submissions/baseline_pv_eta_anchor_1163.py) |
| Tracked bundle (same bytes) | [`submissions/baseline_pv_eta.py`](../submissions/baseline_pv_eta.py) |
| Submission | [`53111837`](https://www.kaggle.com/competitions/orbit-wars/submissions) (2026-05-28, μ=**1163.5**) — **NEW PEAK** above prior 1144-1165 band |
| Status on ladder | EVICTED 2026-05-28 23:22 (sibling branch pushed validator on top) — **resubmit-eligible**, byte-identical bundle frozen |

**Use this anchor for ALL local A/Bs on this branch from 2026-05-29 onward.**

### Why this is the peak (not leaf_pv_2p, not peak-1165)

The original `peak-1165` bundle (μ=1144-1165, sub 52912707 / resubmit 53013786) was
superseded by `baseline_pv_eta.py` (sub 53111837, μ=1163.5). PV_ETA adds a
modeling-correct `γ^(wait_N + eta)` present-value pull-back on candidate Δ
(supersedes the SHIP_TURN_KAPPA band-aid disaster of sub 53099001).

The follow-up `baseline_leaf_pv_2p.py` (sub 53117942, μ=1101.9) was layered on
top of PV_ETA but **never A/B'd against PV_ETA alone** — the local n=10 7-3
result that justified the submit was vs `baseline_peak_1165_anchor` (pre-PV_ETA),
so the marginal effect of LEAF_PV_2P on top of PV_ETA was assumed additive
but never measured. The ladder confirmed it's -62μ (1101.9 vs 1163.5).

Subsequent "fix the opp model" experiments (K-cap 6/16, MLP-validator 12/32)
all A/B'd against the regressed leaf_pv_2p — false-negative risk on the
new mechanisms. **Stop comparing to leaf_pv_2p. Compare to PV_ETA.**

### Historical peak-1165 (kept for cross-comp comparison; do NOT A/B against)

| field | value |
|---|---|
| Git tag | `peak-1165` |
| Git commit | `458f663` (`fix: complete orbital-safety modeling pass (B1-B7)`) |
| Bundle SHA-256 | `9ec3af835a2aefcc91afa9fd586ca75246fc884cac035e3a00e83e5cbbcc6512` |
| Frozen anchor in tree | [`submissions/baseline_peak_1165_anchor.py`](../submissions/baseline_peak_1165_anchor.py) |
| Original submission | [`52912707`](https://www.kaggle.com/competitions/orbit-wars/submissions) (2026-05-22, μ=**1165.4**) |
| Byte-identical resubmit | `53013786` (2026-05-25, μ=**1144.6**) — same bundle, ~20μ rolling-pair noise |
| Status | superseded by PV_ETA above; kept for sanity reference, NOT the build-on-top anchor |

---

## What the agent does (plain English)

This agent plays a "patient capture" game. Each turn it looks at every
planet it owns and asks "what's the cheapest planet near me worth
grabbing or defending?" — generating a few hundred candidate launches
with three sizes each (just enough to capture, double that, or send
everything). Before doing anything expensive, it filters out launches
that would obviously fail: paths that crash into the sun, run out of
bounds, hit the wrong planet, drain a planet that's already under
attack, capture a target the opponent can easily steal back, or let
the opponent grab the same planet for cheaper. The survivors get
rolled forward in a fast simulator for roughly 25-40 ticks each, with
all opponents reacting greedily, to estimate "how much better off am
I after this move than if I'd done nothing?" The best-scoring moves
are emitted, with one important twist: it actively looks for pairs
of nearby planets that should attack the same target together (joint
launches) — single planets often can't capture a contested target
alone, but two coordinated launches usually can. After the main
decision, a post-pass scans for own planets about to fall and
patches in defensive reinforcements. In 4-player games it biases
toward attacking the strongest opponent and toward eliminating the
weakest one when it can finish them off cheaply. It's not searching
deep, it's not learning — it's a careful one-step-lookahead engine
with very good filters that throw away wasteful moves before they
ever get scored.

---

## What is ACTIVE at peak (config that drives decisions)

The wrapper preamble sets 9 env vars; `agents/baseline/main.py` adds
a few setdefaults; many proposer/chooser knobs default-on. The full
live surface:

| Env var | Set in | Value | Live call-site |
|---|---|---|---|
| `BASELINE_JOINT_AGGR` | wrapper | `1` | `chooser_trajectory.py:246` (`JOINT_LIFT_USED_TGTS`); used at `:894`, `:974`, `:988` to drop `used_tgts` lock in emit |
| `BASELINE_JOINT_TOP_K` | wrapper | `5` | `chooser_trajectory.py:240` → cap at `:919` |
| `BASELINE_JOINT_MAX_PAIRS` | wrapper | `60` | `chooser_trajectory.py:243` → cap at `:921`, `:926` |
| `BASELINE_REINFORCE_EMIT` | wrapper | `1` | `baseline/main.py:72` → gates `emit_threat_reinforcements` body at `:319`, `:340` |
| `BASELINE_REINFORCE_ANTICIPATE` | wrapper | `1` | `baseline/main.py:82` → gates anticipated-threat branch at `:373` |
| `BASELINE_ORBITAL_SAFETY` | wrapper | `1` | `proposer.py:443` (`cheap_marginal_value`), `:569` (`_target_holdable_after_capture`), `:683` (`_target_cost_parity_ok`); also threaded into `lib/world_model.time_to_enemy_threat`, `lib/scoring.expected_hold` |
| `BASELINE_VALUE_HEAD` | `baseline/main.py:28` setdefault | `hybrid` | `value.py:224` → `select_favor_fn` returns `favor_hybrid` (composite in 2P, A2-favor in 4P) |
| `BASELINE_CHOOSER` | `baseline/main.py:38` setdefault | `trajectory` | `baseline/main.py:903` → dispatches `choose_trajectory` |
| `BASELINE_JOINT` | `baseline/main.py:46` setdefault | `1` | `chooser_trajectory.py:898` → enables joint enumeration |
| `BASELINE_GAMMA` | unset → default | `0.99` | `baseline/main.py:220` (`_gamma()`) — threaded everywhere |
| `BASELINE_WALLCLOCK_MS` | unset → default | `600` | `baseline/main.py:213` (`_wallclock_ms()`) |
| `PROPOSER_TRAJECTORY_FILTER` | unset → default-on | on | `proposer.py:993` |
| `PROPOSER_DRAIN_FILTER` | unset → default-on | on | `proposer.py:1022` |
| `PROPOSER_HOLD_FEASIBILITY` | unset → default-on | on | `proposer.py:1040` |
| `PROPOSER_COST_PARITY` | unset → default-on | on | `proposer.py:1059` |
| `PROPOSER_REACTOR_CANDIDATES` | unset → default-on | on | `proposer.py:956` |
| `BASELINE_COMET_AIM` | unset → default-on | on | `proposer.py:123` (`aim_and_eta`) |
| `BASELINE_WAIT_GRID` | unset → default | `backward` | `proposer.py:53`, branched at `:378` |
| `COST_PARITY_MARGIN` | unset → default | `0.7` | `proposer.py:635` |

## What is DORMANT at peak (declared / module-read but inert)

| Env var | Where declared | Why inert |
|---|---|---|
| `BASELINE_NEUTRAL_BONUS=2.0` | wrapper line 30 | Read at `chooser_trajectory.py:71` (`NEUTRAL_BONUS_WEIGHT`); used ONLY in the dead v2 scorer `score_candidate` (`:309-312`). Live `score_candidate_v4` (called from `:865`) doesn't touch it. **Recommended live home:** inside `score_candidate_v4` after the leaf Δ, gated on `tgt.owner == -1`. |
| `BASELINE_NEUTRAL_EARLY_HORIZON=50` | wrapper line 32 | Same — feeds dead v2 (`:72`, used at `:311`). |
| `BASELINE_NEUTRAL_EARLY_EXTRA=1.5` | wrapper line 31 | Same — feeds dead v2 (`:73`, used at `:312`). |
| `BASELINE_LEADER_FOCUS` (unset → 1.0) | not set in wrapper | `chooser_trajectory.py:61` → only `score_candidate` v2 reads it. Even if set, no effect on peak. |
| `BASELINE_ME_REACTS=0`, `BASELINE_ME_DEFENDS=0` | not set | Module-level flags at `:139`, `:170`. Gated branches at `:493`, `:565`, `:581`, `:646`, `:659` never fire. Reproducibility scaffolding only. |
| `BASELINE_LEDGER` (unset → off) | not set | `baseline/main.py:140`, `:844`. With it off, `_PENDING_LAUNCHES` is dead, `choose_trajectory`'s `commits` return value is discarded, and `wait_N > 0` winners emit nothing this turn AND nothing next turn — wait-grid candidates are effectively pruned. |
| `BASELINE_LEDGER_MODE=hard` | not set | Only meaningful with `BASELINE_LEDGER=on`. Dead. |
| `BASELINE_OPENING_MILP=0` | not set | `baseline/main.py:155`, gated branch at `:881` never runs. |
| `BASELINE_IDLE_DRAIN`, `BASELINE_STAGNANT_DRAIN`, `BASELINE_COMBAT_STACK`, `BASELINE_SNIPER` (and their `*_RESERVE/*_MAX/*_MIN` knobs — ~28 total) | not set | All four post-chooser drain/sniper passes (`baseline/main.py:966-969`) return moves unchanged when `*_ENABLED` is false. |
| `TRAJECTORY_SKIP_ADMISSIBILITY` | not set | `chooser_trajectory.py:793`. Debug ablation only. |
| `BASELINE_JOINT_4P` | not set | `chooser_trajectory.py:895`. AGGR=1 already lifts the 4P gate via `JOINT_LIFT_USED_TGTS`, so this is redundant at peak. |
| `BASELINE_SPATIAL_WEIGHT`, `BASELINE_SPATIAL_DECAY` | not set | `value.py:58-59` → only `favor_hybrid_spatial` uses them; selected only when `BASELINE_VALUE_HEAD=hybrid_spatial`. Peak uses `hybrid`. |

**Critical lesson (2026-05-27).** Sub 53083109 "fixed" the NEUTRAL_BONUS
family by wiring it into `score_candidate_v4` + `_v4_joint`. That
"fix" coincided with a ~20μ regression on the ladder (REVERT 53088099
landed at 1125.2 vs peak 1144.6). **An env var that looks dead may be
load-bearing precisely because it's dead.** Do not "wire up" dormant
env vars without an isolated n=32 A/B against the peak anchor first.

---

## Top 5 fragility risks (likelihood × severity)

1. **Silent wait-grid pruning via ledger-off.** `baseline/main.py:140`
   (`LEDGER_ENABLED=False`) + `chooser_trajectory.py:998-1005`. Any
   change that increases the proposer's `wait_N > 0` share (e.g.
   `BASELINE_WAIT_GRID=forward`, or tightening the backward-grid
   filter) gets the wait winners SILENTLY DISCARDED — they enter
   `commits`, but `commits` is dropped because the ledger is off.
   Symptom: chooser emits 0 launches while logs show 200 candidates
   scored. **Mitigation:** assert at startup that
   `LEDGER_ENABLED == True` whenever any `wait_N > 0` reaches the
   chooser, OR in the trajectory branch promote `commits` to
   `due_moves` next turn unconditionally.

2. **NEUTRAL_BONUS family fakes a "neutral-attack tilt" without
   applying it.** `agents/baseline_joint_aggr_consolidated_orbitfix/main.py:30-32`
   + `chooser_trajectory.py:309-312`. The wrapper bundle name and
   docstring imply neutral-targeting bias; in reality
   `score_candidate_v4` ignores it. A future iteration that "tunes"
   the constants will see zero LB movement and conclude neutrals
   don't matter — wrong conclusion from a dead variable.
   **Mitigation:** either delete the three setdefaults from the
   wrapper (zero behavior change), or wire the bonus into
   `score_candidate_v4` after the leaf Δ (real behavior change —
   must re-A/B at n=32 panel).

3. **`favor` leaf double-counts production via `pv_horizon`.**
   `value.py:127` + `lib/scoring.py:118-122`. `pv` is computed at
   `eta=0` so the production term is weighted by `(1-γ^h)/(1-γ)`. At
   step=0 that's ≈63 (γ=0.99, h=500); at step=400 it's ≈63→39. Ship
   term is O(100s). One production point ≈ 39-63 ship-equivalents in
   late game vs ≈63 early — so `ELIMINATION_BONUS=+55` is one
   production point's worth of credit, not a strategic threshold.
   **Mitigation:** none required at peak (calibrated empirically),
   but any change to `γ`, `t_total`, or `ELIMINATION_BONUS` without
   re-tuning the others will silently re-weight ship vs production
   EV. Add a unit test pinning `favor()` outputs at three
   representative game states.

4. **Asymmetric ME-reacts/defends scaffolding.** `chooser_trajectory.py:481-498`
   (baseline) vs `:568-583` (candidate rollout). Both legs are
   ME-idle at peak (`_ME_DEFENDS`/`_ME_REACTS` both off), so
   `Δ = leaf − baseline` is well-defined. Any future toggle of
   `BASELINE_ME_DEFENDS=1` flips the candidate leg to inject
   defensive launches WITHOUT flipping the baseline — Δ will
   silently change meaning and previously-positive candidates may
   flip negative. **Mitigation:** if `_ME_DEFENDS` is ever enabled,
   mirror the change in `build_trajectory_baseline`. Add a contract
   test.

5. **`_target_holdable_after_capture` and `_target_cost_parity_ok`
   gate on `omega != 0.0`.** `proposer.py:571`, `:685`. If the
   engine ever returns `omega=0.0` for a stationary scenario (e.g.
   comet-only seeds, or a future map variant), all the
   orbital-safety geometry collapses to current-position math —
   silently reverting to the pre-`BASELINE_ORBITAL_SAFETY=1`
   behavior PI flagged as broken. **Mitigation:** assert
   `world.omega` is non-zero in `WorldModel.from_world`, or treat
   the `omega==0.0` branch as an explicit different-physics regime
   rather than a silent fallback.

---

## Build-on-top protocol (mandatory checklist)

Every change layered on the peak goes through these gates. Skipping any
gate to "ship for early feedback" has, in our recorded history,
delivered a μ regression every single time.

### 0. Start from a clean tree

```bash
git checkout claude/<your-branch>
git diff peak-1165..HEAD agents/baseline/ lib/  # know what's already different
```

If the diff against `peak-1165` is non-empty, you are NOT building on the
peak — you are building on whatever else has been layered. Decide
explicitly whether to keep those layers.

### 1. Single env-var-gated change, default OFF

The change MUST be gated behind a `BASELINE_<NAME>` env var. The default
value MUST produce byte-identical behavior to the peak. Verify with a
unit test that imports the chooser/proposer at the default and asserts
identical scores against a synthetic fixture.

Why: the env-var gate is the only path that lets us submit the unchanged
bundle as a true control while testing the variant locally, and the only
back-out that doesn't require a code revert.

### 2. Local A/B against the peak anchor, NOT v7_0

Use `submissions/baseline_peak_1165_anchor.py` as the opponent. The
ladder evidence is unambiguous: winrate vs v7_0 does not predict winrate
vs peer-anchor (sub 53083109 was 48/64=75% vs v7_0 but 2/32=6% vs
peer-anchor, settled μ=921). v7_0 is too weak a sparring partner.

```bash
BASELINE_<NAME>=<value> python fast.py eval \
    <focal_agent> --vs submissions/baseline_peak_1165_anchor.py --n 32
```

### 3. Rule 45 gate — n ≥ 32 minimum

`n=8` does not predict ladder behavior. We have two days of evidence:

| sub | local A/B | live μ |
|---|---|---:|
| 53083109 | 48/64=75% vs v7_0 | **921** |
| 53099001 | 6/8=75% vs head_anchor | **680** |

Wilson 95%-lower-bound at n=8 is too wide; n=32 minimum, n=64 preferred.

### 4. Rule 43 gate — multi-opponent panel

```bash
python fast.py eval <focal_agent> \
    --vs-panel --require-h2h submissions/baseline_peak_1165_anchor.py \
    --geometry-panel --by-archetype --n 32
```

Pass criterion: per-opponent Wilson-lo ≥ 0.55 (the gate exists because
A/B loops — A beats B beats C beats A — are common in this game).

### 5. Bundle + Rule 46 parity smoke

```bash
python scripts/bundle_agent.py agents/baseline --force
cp submissions/baseline.py submissions/<your-bundle-name>.py
# prepend orbitfix preamble + your new env-var setdefault (see existing pattern)
python -m pytest tests/test_bundle.py -q
python fast.py play submissions/<your-bundle-name>.py --vs v7_0 --seed 7
```

### 6. Rule 42 push-coordination

```bash
kaggle competitions submissions orbit-wars | head -5
# Read what the rolling pair currently is; identify which slot will be evicted.
# If evicted-μ > predicted-μ → BLOCKED until explicit PI signoff.
```

Append a row to the push-claim board in `state/MULTI_BRANCH.md` with
branch, agent, predicted μ band, evicted (sub_id, μ), and PI signoff.

### 7. Submit (Rule 1)

One submission per approved go. Wait for it to settle. Update the
push-claim row with the settled μ.

---

## Anti-patterns to avoid

These are the recorded failure modes from the past two days:

- **"Just submit at n=8 for early feedback."** Sub 53099001 cost a μ=680
  rolling-pair slot. Sub 53083109 cost a μ=921 slot. Both were "early
  feedback" pushes that skipped Rule 45.
- **Wiring a dormant env var without isolation.** The NEUTRAL_BONUS-into-v4
  plumbing change (sub 53083109) is suspected to carry ~20μ regression.
  The change SHOULD have been gated independently from the other 3 fixes
  it was bundled with.
- **Stacking 3+ changes in one submit.** Sub 53083109 stacked Fix 1
  (NEUTRAL_BONUS), Fix 2a (holdability v2), Fix 3 (source-drain), Fix 4
  (follow-on). When it regressed, we couldn't attribute. The REVERT
  isolated 3 of the 4 (made them opt-in) but kept Fix 1 active —
  hence we're still ~20μ below peak.
- **Trusting the bundler at face value.** The orbitfix wrapper pattern
  (cross-agent import) is NOT directly bundled by `bundle_agent.py
  agents/baseline_joint_aggr_consolidated_orbitfix`; it requires
  bundling `agents/baseline` + hand-prepending the env-var preamble.
  See `scripts/bundle_agent.py` line 368 (`if agent_dir.is_dir():`) and
  the friction note in commit 458f663's message.

---

## How to use the peak anchor in scripts

```python
# Local A/B reference
PEAK_ANCHOR = "submissions/baseline_peak_1165_anchor.py"

# Frozen bundle SHA-256 — assert before A/B
EXPECTED_SHA = "9ec3af835a2aefcc91afa9fd586ca75246fc884cac035e3a00e83e5cbbcc6512"
```

If the anchor's SHA ever drifts from the expected value, somebody
modified the frozen bundle. Stop and restore from
`git show peak-1165:submissions/baseline_joint_aggr_consolidated_orbitfix.py`.

---

## Open questions worth answering before the next build-on-top

1. **What does removing the NEUTRAL_BONUS-into-v4 plumbing actually
   recover?** The REVERT (1125.2) kept that plumbing active; the peak
   anchor doesn't have it. The peak-restore submission `53099429`
   (pending) is the cleanest test we have run. If 53099429 settles
   ≥1140, we've confirmed the ~20μ gap was the plumbing — and we know
   to never add it back without isolated panel-clearance.
2. **Are any of the dormant env vars worth wiring up correctly?**
   `BASELINE_LEADER_FOCUS` (declared at peak, value 1.0 = inert) and
   the 3 NEUTRAL_BONUS knobs were designed as tilts. They may help if
   plumbed AND tuned at panel scale. The "fix-and-tune" path needs
   Rule 43 evidence before submit.
3. **Why does v7_0 winrate fail to predict peer-anchor winrate?** This
   is the single most important calibration question. The likely
   answer: v7_0 is a weak generalist; the peer-anchor's behavior is
   distinct enough that our local fixture (which often targets v7_0's
   specific weaknesses) generalizes poorly. Worth a dedicated
   investigation when ladder budget allows.
