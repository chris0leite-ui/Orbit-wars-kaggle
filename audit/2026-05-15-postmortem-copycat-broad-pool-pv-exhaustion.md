# 2026-05-15 — postmortem: copycat broad-pool + PV exhaustion

> Branch: `claude/bootstrap-read-handover-HjcdN`. Companion experiment
> branch: `claude/behavioral-mimic-Bv9Wm` (separate audit at the
> bottom).

## TL;DR

Spent the session building three iterations of a "copycat" agent
(mimic + sigma-paired search → multi-roster → broad-pool argmax) and
finally porting v7_pv's `PV_GAMMA=0.99` lever on top. **Net result vs
v7_0_drop_one across the 64-seed panel: 57.8% (Phase 1) → 50% (Phase
2 broad-pool n=8) → 37.5% (Phase 3 broad-pool + PV n=32, FAIL).** The
hypothesis "let candidate quality emerge from a richer pool" did not
beat the v7_0_drop_one ceiling, and stacking PV on top of the pool
actively regressed.

The session ALSO ran the PI-requested behavioral-mimic experiment on
its own branch and reproduced the historical 0% lag-cascade failure
mode at every τ tested.

The one durable artifact worth merging to main is the
`PV_GAMMA` env-var enabler in `lib/scoring.py`. Everything else is
documentation.

## What was tried

| Phase | Branch commit | Architecture | vs v7_0_drop_one | Verdict |
|---|---|---|---|---|
| 1a | `ceb0710` | Roster {v3.5.1, v7_0_base} + sigma-paired drops/angle/ship perturbations + tau gate | 14/0/2 (vs v3.5.1, n=16) — sigma-perturbations contributed 0 net wins at tau=2 vs tau=inf | Pivot — sigma-pair didn't help |
| 1b | `b476477`, `50a0a3e` | Above + v7_0_drop_one as roster member, tau=inf, K=8, wallclock 550 ms | 37/64 = **57.8% INCONCLUSIVE** [0.46, 0.69] | INCONCLUSIVE; lift = roster floor selection, not perturbations |
| 2 | `443e5da` | Drop sigma-pair entirely; broad-pool argmax over {geo tilts + v7_0_drop_one}; no tau gate | 4/0/4 = **50% n=8** (smoke) | Smoke-only; geo tilts didn't dominate v7_0 |
| 3 | `ad103dc` | Above + `PV_GAMMA=0.99` (v7_pv's mission-proposer lever) | 12/32 = **37.5% FAIL** [0.23, 0.55] | REGRESSION — PV + broad-pool is anti-synergistic |

Concurrent panel results (Phase 1b, n=64 each opponent):
- vs v4_planner: 41/64 = 64.1% INCONCLUSIVE
- vs v3.5.1: 41/64 = 64.1% INCONCLUSIVE
- Panel verdict: INCONCLUSIVE (worst Wlo = 0.456)

## What didn't work and why

**1. Sigma-paired perturbations** — drop {M, sigma(M)} together, ±0.10 rad
angle nudges, ±15% ship-count scales. Tested at tau ∈ {0, 1, 2, inf}.
Identical winrates across tau values vs v3.5.1 (14/0/2 each). Diagnosis:
the perturbations are too cosmetic — small variations of v7_0_drop_one's
already-near-optimum action don't materially change the strategic shape,
so the K=10 judge picks the floor (tau=inf) or a noise-equivalent
alternative (tau<inf). The sigma-pair *machinery* was correct
(self-play preserved σ-equivariance, no cascade); the sigma-pair *moves*
were too timid to move the needle. PI critique on 2026-05-15 was the
explicit nail in the coffin: "imposing sigma-pair forces symmetric
spread; concentration has real value; let it emerge from gain."

**2. Broad-pool argmax (Phase 2)** — replaced sigma-pair with geo's
strategic stances (incumbent + opening_boost + enemy_focus + concentrated
+ saturation + front_reinforce + drop-one) plus v7_0_drop_one's chosen
action. Argmax over the K=10 scores. Let the judge pick freely. Result:
50% (n=8) and a 60-65% point estimate vs v4_planner / v3.5.1 panel
extension. The geo enumeration overlapped heavily with v7_0_drop_one's
own search neighborhood — adding it didn't surface dominant alternatives,
just noise candidates that the K=10 judge sometimes preferred for the
wrong reasons.

**3. PV_GAMMA=0.99 on the broad pool (Phase 3)** — the surprising one.
v7_pv on the live ladder uses `PV_GAMMA=0.99` and reaches μ=1064.4 vs
v7_0_drop_one estimated μ~1000-1030. The proposer-side lever is real.
But applying it on top of our broad-pool design REGRESSED us to 37.5%
vs v7_0_drop_one (n=32, Wilson [0.23, 0.55] FAIL). Diagnosis: PV-aware
proposers prefer early-arriving captures; geo's tilts (concentrated,
saturation) prefer different shapes; the K=10 judge with
`delta_us_minus_them` can't reconcile the two preferences and produces
higher-variance, lower-mean play. v7_pv works on the ladder precisely
because it's a focused proposer-side change with NO broad pool.

**4. Behavioral mimic (companion branch
`claude/behavioral-mimic-Bv9Wm`)** — copy opponent's last fleet via
`diff_new_fleets` + 180° mirror, deviate when K=10 lookahead finds
clearly better. PI directive: "ignore that they have not been
successful." Result: 0/0/8 vs v3.5.1 at τ=∞ (pure mimic, exact
historical 0% reproduction); 1/0/7 at τ=1.0; 0/0/8 at τ=5.0. The K=10
deviation gate cannot rescue the 1-turn lag cascade because the
candidate space is anchored on a cascading mimic — any sigma-pair
perturbation of a bad action stays bad. Documentation pushed at
commit `41c87f6` on the companion branch; not merging.

## What IS worth keeping

**`lib/scoring.py` env-var control for `PV_GAMMA`** (commit `ad103dc`).
Default behavior unchanged at γ=1.0; agents can opt in via
`os.environ.setdefault("PV_GAMMA", "0.99")` before lib imports. This is a
genuine infrastructure improvement (≈5 LOC change) that lets any future
agent on main toggle PV-aware proposers without source edits. Backward
compatible; 22/22 tests green; verified the env-var propagates through
`fast.py`'s ProcessPoolExecutor workers. **Recommend cherry-picking to
main** — it makes the v7_pv lever a one-line opt-in for every future
agent.

The companion test relaxation (`tests/test_scoring.py` —
`test_pv_horizon_default_gamma_is_linear_for_backwards_compat` no
longer asserts the constant, tests behaviour at `gamma=1.0` explicitly)
is a necessary follow-on to the env-var change. Cherry-pick together.

## Key frictions and fixes

See `audit/friction.md` for one-liners. Highlights:

- `pv-broadpool-incompatible`: PV proposer lever doesn't compose with
  broad candidate enumeration. Use one or the other.
- `same-process-pv-shared-state`: testing PV vs non-PV in the same
  Python process gives both agents the same `lib.scoring.PV_GAMMA`
  (set-once module constant). Always run cross-PV A/Bs via `fast.py`
  workers (separate processes).
- `wallclock-truncation-in-roster-wrappers`: wrapping a K=10 chooser
  inside a roster member with reduced wallclock degrades its choice
  by enough to lose vs the full-budget version. Either give it the
  full ladder budget or skip the re-scoring.
- `small-n-ab-noise`: 5/8 = 62.5% has Wilson CI roughly [0.30, 0.86].
  We escalated a smoke to a full panel and it landed at 12/32 = 37.5%.
  Require n≥16 before committing 70 minutes of compute.

## Lessons for future copycat-style work

1. **Search-plus-judge architectures top out at the judge's ceiling.**
   We tried three different candidate generators (sigma-pair, broad
   geo pool, mixed) — same K=10 judge, same ~57% ceiling vs
   v7_0_drop_one. The judge is the lever, not the candidates.

2. **Lib-config knobs and architectural choices are not orthogonal.**
   PV_GAMMA=0.99 helps v7_pv (focused proposer) but hurt copycat
   (broad pool). Don't assume stacking helps.

3. **σ-equivariance as a HARD CONSTRAINT is wrong.** PI was right —
   it should emerge from gain, not be imposed. The sigma-pair
   *machinery* (lib.mirror) is fine for diagnostic / probe use; the
   sigma-pair *constraint* on the candidate generator forces symmetric
   spread when concentration would win.

4. **Same-process A/Bs are dangerous when shared module state is in
   play.** Module-level constants set by env vars at import become
   process-wide. The first agent imported wins.

## State after this session

- Current branch: `claude/bootstrap-read-handover-HjcdN` ahead 5 /
  behind 6 of origin/main.
- Companion branch: `claude/behavioral-mimic-Bv9Wm` (1 commit, pushed,
  documented failure).
- Live ladder anchor unchanged: v7_pv μ=1064.4. No submission attempted
  this session — Rule 12 caveat blocked us at every panel verdict
  (worst Wilson LB never cleared the +20μ safety margin).
- `lib/scoring.py` PV_GAMMA env-var change is the only artifact
  recommended for main. Everything else stays on these branches as
  documented experimentation.
