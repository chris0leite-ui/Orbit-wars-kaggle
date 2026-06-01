# Contest-aware conversion — integrated design

> Written 2026-06-01 PM (`champion-strategy-rules-00JzI`). Integrates four
> improvement ideas into one architecture. Grounded in
> `audit/2026-06-01-loss-mode-diagnosis.md` (the loss mode is conversion,
> not hoarding) and the PI's steering (state-driven horizon; fleets don't
> die in flight; opening tempo real; beware selection bias).
> Status: **design, not built.** Adaptive-K v1 (the crude step-schedule
> precursor of Lever 1) is shipped (sub 53265480, settling).

## 1. The problem, stated precisely

**Fleets cannot die in flight** (no air collisions; the only flight deaths
are sun/OOB, already filtered). Therefore *every* launch that fails to
capture did one of exactly three things:

  (a) arrived **under-strength** — the target was reinforced by arrival;
  (b) arrived **too late** — someone else captured it first;
  (c) arrived at a planet that **flipped en route** — neutral→enemy.

All three are the same error: **we mispredicted the target's state at our
arrival tick, or committed at the wrong time.** The diagnosis shows this as
"launch a lot, capture little" in the midgame of losses. So the lever is not
*more* launching (flat-expand-credit proved that, −124μ) and not naive
sizing (size-balance regressed) — it is **predicting the contested arrival
state correctly and committing the right force at the right time.**

## 2. The unifying primitive: `predict_arrival_contest(src, tgt, fire_step)`

One function answers, for a candidate launch, *what will this target look
like when my fleet arrives, given the opponent will react?* It returns:

- `my_arrival_tick` — when our fleet lands (existing aim + eta);
- `predicted_garrison` — target ships at arrival, including its own
  production, all **in-flight** fleets (both sides, already modelled), **and
  the opponent's LIKELY new reinforcement** (the missing piece — §Lever 3);
- `predicted_owner` at arrival;
- `opp_earliest_contest_tick` — the soonest the opponent could land enough
  force to take or hold this target from us.

Everything below is **derived from this one prediction.** That is the
integration: not four bolt-ons, but four faces of a single contest model.
Reuses (Rule 47, no new physics): `lib.aim`, `world_model.predict_garrison_at`,
`world_model.time_to_enemy_threat`, `joint_solver/opp_projection`.

## 3. The four derived levers

### Lever 1 — Horizon becomes a *derived* quantity (state-driven K)

The launch-discipline ceiling K stops being a schedule and **becomes the
predictability of the specific target**:

  `K_target = clamp(K_floor=10, K_ceil≈30, opp_earliest_contest_tick)`

If nobody can contest a target for 25 ticks, a launch arriving in 22 is
safe → admit it (high K). If the opponent can contest in 6, don't commit a
fleet that arrives in 15 → it lands in a changed world (low K). This is the
principled replacement for the shipped step-schedule v1 (which fakes
predictability with the clock). Same single chokepoint
(`launch_rules.capture_horizon_k`), now fed target+state instead of `step`.

### Lever 2 — Urgency / race class (contest-aware target value)

Compare `my_arrival_tick` to `opp_earliest_contest_tick`:

- **race-win** (`my_arrival + hold_margin < opp_contest`): we take it *and*
  hold the counter → **prioritise, now.** High value multiplier.
- **bankable** (opp cannot contest at all): grab when convenient →
  **defer**, low urgency (don't spend tempo on what can't be lost).
- **race-loss** (`my_arrival ≥ opp_contest`): we'd arrive at a contested or
  flipped planet → **suppress the launch entirely.** These are the wasted
  fleets inflating our launch count in losses.

Front-loads the captures we actually win; deletes the (b)/(c) wasted
launches. Opponent-conditional, so distinct from the falsified flat-expand-
credit ("expand more" regardless of contest).

### Lever 3 — Opponent reinforcement in the arrival garrison (the root)

The piece Levers 1-2 depend on, and the fix for failure (a). Today's
`predict_garrison_at` sees in-flight fleets but not the opponent's *future*
reinforcement of a contested target. Add: the opponent's nearest viable
source + tempo → the force they will most likely land here by each tick.
Fold it into the arrival-garrison so sizing/timing account for the defence
that *will* be there, not what's there now. A Rule-40 modelling fix — it is
*why* the naive size-balance failed (it sized against a present/naive
garrison, not the contested-arrival one).

### Lever 4 — Forward staging (tempo form of "move ships to the front")

When no current source yields a race-win for a high-value target, check a
one-hop **forward redeploy** (own→own) that creates a source which *does*.
Value the redeploy by the **race-win capture it unlocks next**, not its
(zero) immediate gain. Shortens our effective ETA to contested targets →
we win more races → the same launches convert. (The paired positioning
snapshot was null, so this is the *tempo/rate* form the PI's "we fail to
move them to the front" survives as, not a static-position fix.)

## 4. Why this is one design, not four

The agent currently values a capture by production × arrival-discount, with
the opponent present only as static in-flight fleets. This design makes the
agent reason about a contested capture the way a player does: **who can get
there first, with how much force, and where should I stage to win the
race.** `predict_arrival_contest` is the shared brain; horizon, urgency,
sizing and staging are its outputs. It targets the conversion gap at its
root (prediction + timing) rather than its symptoms (volume, raw size).

## 5. Build sequence (incremental, each default-OFF + independently A/B-able)

Never ship the four together blind (the flat-credit lesson). Each lever is a
separate env gate and its own A/B; the shared predictor lands first.

1. **`predict_arrival_contest` primitive** + a single-game trace (Rule 47):
   confirm `opp_earliest_contest_tick` is sane on real games before any
   chooser wiring. Cheap; no behaviour change.
2. **Lever 2 (urgency)** first — biggest direct hit on the wasted-launch
   symptom, cheapest to test. Replay pre-check: of targets the opponent
   captured, how many did we *also* launch at and lose the race to? That is
   the waste it removes, measured directly.
3. **Lever 1 (state-driven K)** — replaces the shipped step-schedule; A/B
   per-target-K vs per-clock-K, plus the multi-opp panel the v1 still owes.
4. **Lever 3 (opp reinforcement in garrison)** — highest effort; sharpen
   only once 1-2 prove the contest model moves win-rate. (The analytical
   track stalled on opponent modelling — keep it cheap, per-candidate.)
5. **Lever 4 (forward staging)** — last; gated on a paired launch-ETA
   diagnostic showing ours is longer than the opponent's to contested
   targets.

## 6. Gates / risks

- **Cost (Rule 2):** `predict_arrival_contest` runs per candidate — the
  opponent-contest estimate must be O(nearest-source), not a search. Profile
  before wiring; the shipped adaptive-K stayed at p95≈283ms, keep that.
- **Selection bias (Rule 41):** validate every lever with *paired* or
  *vs-aggressive-opponent* A/Bs, never the champion mirror alone (it can't
  see contest value — the same blind spot that made flat-credit look
  plausible).
- **Falsification (Rules 21/37):** each lever gets ≥3 hyperparameter
  variants; cap at 3 per axis. If the contest model itself (Lever 3) shows
  no signal across variants, the conversion-via-opponent-awareness thesis is
  wrong and we pivot.
- **Submit gates (Rules 43/45):** champion h2h n≥64 Wilson-lo ≥ 0.50 +
  multi-opp panel Wilson-lo ≥ 0.55 each, before any submission.

## 7. Relationship to existing work

- **Replaces/subsumes:** the step-schedule adaptive-K v1 (Lever 1 is its
  principled form); the naive size-balance (Lever 3 is the modelling-correct
  version of what it attempted).
- **Distinct from closed tracks:** not the falsified flat-expand-credit
  (opponent-conditional, not "expand more"); not the analytical-chooser
  replacement (this augments the existing chooser's value, doesn't replace
  the rollout); not the reach-frontier prescription.
- **Reuses:** `opp_projection`, `predict_garrison_at`, `time_to_enemy_threat`,
  `aim`, and the redeploy-2hop sketch (`knowledge-base/concepts/redeploy-
  2hop-capture-design.md`) for Lever 4.
