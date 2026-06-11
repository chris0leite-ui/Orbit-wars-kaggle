# 2026-06-12 — Concentration rebuild vs the Producer archetype (head-on)

> **CORRECTION (added end of session): the "fortress freeze win
> condition" below is a MEASUREMENT ARTIFACT.** The 8/10 battery ran 9
> games concurrently; the Producer runs torch with multi-threaded
> inference, and the thread-pool thrashing pushed its turns past the
> engine's 1-second act timeout — its 220-tick "deterred passivity"
> was timed-out no-op turns. Proof, two experiments at honest compute:
> (1) the Producer needs only 13-21 ms/turn solo (so the freeze was
> never its value function "going quiet" under normal conditions, but
> under 9-way contention torch degrades far worse than pure-python
> agents); (2) forcing our agent defense-only from t40 (recreating the
> exact fat-garrison board) does NOT freeze the full-compute Producer —
> it eats the board even faster (91-97 steps, 3/3). Total-launch
> liveness asserts cannot catch MID-GAME degradation; see the new
> binding lesson at the end. Read every "fortress/freeze/deterrence"
> claim below through this correction.

PI directive: "Continue the head-on Producer hunt with the ledger — the
concentration rebuild." This doc records the measured diagnosis chain and
each mechanism added. All Producer games run with liveness asserts
(opponent launches > 0, steps > 30) per the dead-opponent correction.

## Tooling added

- `scripts/trace_duel.py` — per-tick economy/launch trace of one 2P game:
  planet/production/garrison/in-flight curves per side, capture events
  with sink costs, every launch, and fleet-death classification
  (sun / out-of-bounds / landed-on-planet, with the planet's owner).
  Liveness asserts built in.
- `LEDGER_DEBUG=<file>` env hook in the agent: per-turn decision log
  (budgets+reserves, candidate order, buys with coalition makeup and
  arrival ticks, banking/gather events, veto keeps, redeploys). Off in
  production (kaggle runs a clean env).

## Measured diagnosis chain (seed 300 family, alive Producer)

1. **No physics waste**: zero sun/out-of-bounds deaths either side.
   All losses are landings.
2. **The stick-rate gap (the headline number)**: the Producer ends
   98-99% of its landed tonnage on planets it owns after resolution;
   the old build managed ~50% — 700-1300 ships per game died landing on
   planets they did not take or hold.
3. **Opening expansion was NOT the problem** (sinks paid comparable,
   capture counts even at t14). The earlier "cheap-garrison-first"
   kill-chain story was wrong in this respect.
4. **Min-sizing was a root cause**: the chooser only ever bought the
   minimum-sized mission (`take = min(spare, n_need)`); captures landed
   1-5 surplus, were re-sniped in 1-3 ticks by 50-148-ship stacks, and
   follow-ups died into those stacks (the ping-pong logs).
5. **Tempo holes**: distance-ordered coalitions admitted 1-ship members
   whose slow flight dragged whole missions to arrival t+24..30 (the
   Producer lands in 5-8); the response model priced the enemy of TODAY,
   so 28-tick missions to deep targets looked unanswerable while really
   landing into a 3x bigger future army.
6. **Worst-case response pricing overshoots**: making every neutral
   grab pay for the enemy's full feasible response priced expansion to
   zero — the opponent expands freely while you cower (t20-30 planet
   counts 4->6 vs 4->9). Defense of owned planets is near-certain
   (1:1, sink-free) but racing your neutral grab is speculative; the
   response model must distinguish them.
7. **Reserve freeze**: reserving a garrison that STILL cannot survive
   the worst feasible wave both loses the planet and freezes the
   capital. In this game in-flight ships are invulnerable — unsavable
   mass belongs in the air.
8. **Redeploy churn**: evacuating doomed garrisons to destinations that
   flip before arrival is a steady 2-30-ship donation stream; the
   ledger only knows launched fleets, not the stack's next hop.

## Mechanisms added (this session, in order)

a. Size axis in plan_attack: price the full-spare variant against the
   minimum; keep the higher-valued (bigger = faster + holds longer +
   forward-bases surplus).
b. Robust sizing: requirement includes the enemy's feasible pre-arrival
   reinforcement (launched after seeing my fleet), shared-response
   scaled; neutral targets subtract the sink the racer pays.
c. Gather-to-strike banking: unaffordable best plans ship member spares
   to the member nearest the target (hub) instead of freezing in place;
   hub verified to stay mine through every arrival.
d. Time-correct response curve: responders bank production until their
   launch tick; defense-of-owned grows, races-at-neutrals use current
   mass only.
e. Coalition members ordered by arrival tick (fastest first), not
   distance.
f. Reserve release: if the worst wave beats the full garrison anyway,
   reserve 0 (mission capital), not everything.
g. Last-resort redeploy of doomed garrisons (nearest surviving friendly
   with non-negative defense margin, else capturable neutral), with an
   anti-dribble gate (lump >= 8 or final 2 ticks).
h. Arrive-second race pricing on contested neutrals: size to beat the
   racer's landed surplus + regrowth; the blunt RACE_DISCOUNT no longer
   double-charges exactly-priced races.
i. Prefix veto: rollout variants now include value-ordered prefixes
   (top-1..top-k) so the veto chooses the portfolio SIZE — drop-one
   could never express "drop the worst five". (From the load-truncation
   discovery; alone it did NOT reproduce the fortress: the 18-tick
   reactive rollout under-punishes expansion because its opponent model
   races nothing and deterrence lives at the 100-tick scale.)
j. Ledger-based response sources: _response_curve and
   _pre_arrival_response read post_owner/post_ships at the implied
   launch tick instead of the current snapshot — prices the rebound (a
   stack landing at P makes P a source from the next tick), stops
   counting planets my known fleets will take.
k. Rolling-archetype response scale: an opponent whose mass is in
   FLIGHT is not shopping-committed (a landed stack relaunches within a
   tick); resp_scale now recovers toward 1.0 with their
   inflight/ground ratio. The old scale halved the modeled response
   exactly when the Producer's stack dominated.
l. Fortress constraint (2P): per-turn offense spending capped so home
   mass never drops below the enemy's deliverable wave, where the wave
   is read off the ledger per my planet (enemy garrisons at their
   launch ticks within a 16-tick window — must cover a stack rotation —
   plus known hostile arrivals). Gambit/stalemate bypass.
m. Admit-floor cap: the negative allowance is a constant (-6 in 2P,
   -16 in gambit), no longer scaled by fleet size — the per-ship floor
   predates the size axis and admitted 130-ship plans at value -20.
n. Fortress consolidation: when the cap binds, unspent spares mass at
   the planet where the enemy wave lands hardest (ledger-verified
   friendly merge, 6-tick reach, anti-dribble).

## Seed-300 solo trajectory across builds (steps survived, all losses)

baseline 163 -> +a 111 -> +b 101 -> +c 93 -> +d/e 151 -> +f 163 ->
+g/h 177 -> +i/j 124 -> +k/l/m 163 -> +n 151. Mid-game now reaches
full parity (t60-70: 13 planets/45 prod vs 13/49-52, garrison 851)
but the multi-stack wave still breaks through at t70-90 on this seed.
Seed 300 solo remains unbeaten; the clean battery decides.

## Next planned mechanism (not yet implemented)

Response sources should read the LEDGER, not the snapshot: a stack
landing at planet P at tick T makes P a response source with that
garrison from T+1 (the rebound that kills missions one hop after it
lands). post_owner/post_ships already hold these timelines per planet.
Same margin logic then serves rescue feasibility (don't rescue what the
rebound retakes) and redeploy destinations.

## Battery (this build, results pending at write time)

- vs Producer, seeds 300-323 (n=24, single seat — deterministic agents
  mirror on seat swap)
- vs v7_0, 12-seed regression pool
- vs live-1300.9 bundle rebuild, seeds 600-607

## MID-BATTERY DISCOVERY: load-dependent strength inversion

Seed 300 lost at 177 steps in a SOLO spot-check, then WON at 500 steps
inside the 9-way-concurrent battery — same code, same seed. The agent's
TIME_BUDGET (0.70 s wall-clock) truncates the buy loop earlier under
CPU contention: fewer missions bought per turn, and that played BETTER
vs the Producer. Two conclusions:

1. A/B results under heavy parallelism are NOT production behavior;
   batteries must run at low worker counts (<= 3-4) for valid reads.
2. Over-shopping is real and the rollout veto cannot see it: its
   variants are {all, drop-ONE, defense-only} — "drop five of twelve"
   is unrepresentable. Fix: add value-ordered PREFIX variants
   (top-1, top-2, ... all) so the veto chooses the portfolio SIZE.
   This also makes behavior robust to wherever the clock truncates.

## CLEAN VERIFICATION (3 workers, the committed 14-mechanism state)

vs Producer, seeds 300-323: **0/24** — eliminated in 97-169 steps.
The load-confounded 8/10 does NOT reproduce at honest compute. Given
full time, the buy loop purchases "individually profitable" missions
that are collectively suicidal vs the stack archetype.

## Post-battery gate experiments (all reverted — each made solo worse)

Six attempts to encode the truncation policy deterministically, tested
on solo seeds 300/310/316 (committed state survives 151 on s300):

1. Solo-kill wave filter + 1.5x deterrence margin + production-flow
   spend bound -> 110-119 steps (throttled the OPENING: in the early
   game every planet "solo-kills" every other).
2. Safe-mission cap exemption (target self-defends vs ledger response
   for 6+ ticks) -> 97-112 (exemptions leak the fortress back into the
   war).
3. Hard phase gate (offense fully OFF under a live wave) + gather
   gated to freeze-in-place -> 101-118 (still throttled openings).
4. Concentration filter (source must hold >= 30% of its owner's army)
   -> 103-119.
5. Landing-classified rolling correction (neutral-bound enemy fleets
   are committed shopping, not responders) — a genuine model fix to
   mechanism k, but did not unblock the opening: 103-123.
6. (During 5: World.__slots__ missed _aggr_fly -> crash guard returned
   [] silently every turn; liveness asserts caught it. Lesson: the
   crash guard hides breakage — the parity test + liveness asserts are
   the only tells.)

Root issue left UNRESOLVED: in the committed build the opening buy
cadence is already throttled (~15 buys/game vs 60+ in older builds);
values collapse between the order pass and the buy pass (banked
sources shrink later coalitions). The truncation policy that won 8/10
remains unreproduced. Next session: BISECT from c42c9fc — measure each
mechanism's solo contribution instead of stacking gates on top.

## Designed next (after prefix veto + ledger response sources)

Opponent-adaptive response propensity: the World already re-derives
every enemy fleet's exact landing at launch; attribute each enemy
launch to target type (retake-mine / neutral / own-consolidation) and
keep a per-owner EWMA. Scale response curves by observed retake share
(Producer ~ max, v7_0 ~ low) instead of charging every opponent the
same feasible-response pessimism. Module-level memory, reset on
step-regression (new-episode detection). This addresses the structural
either/or: worst-case response pricing loses to expanders, optimistic
pricing loses to re-snipers — only a measured opponent model serves
both. (Earlier profiling failure misclassified by LAUNCH counts; the
exact-landing attribution removes that failure mode.)

## FINAL SESSION VERDICT (clean, liveness-checked, 3 workers)

The committed 14-mechanism build (c42c9fc) at honest compute:

| Opponent | c42c9fc | ledger_v1_4 reference |
|---|---|---|
| Producer (seeds 300-323) | 0/24 | 0/16 (known) |
| v7_0 (12-seed pool) | 5/12 | 8-9/12 |
| live-1300.9 bundle (600-607) | 1/8 | ~75-85% band |

The v7_0/bundle losses are mostly FULL-LENGTH games lost on score:
the build under-expands against everyone — the same response-pricing
pessimism that was supposed to stop waste throttles ordinary
expansion. The session's response-model changes (robust sizing,
time-correct + ledger-based curves) need per-mechanism bisection, not
joint shipping.

**Action taken: agents/ledger/main.py restored to the ledger_v1_4
state (byte-identical to submissions/ledger_v1_4.py, parity green).**
The 14-mechanism state remains at c42c9fc for bisection.

What survives the session unconditionally:
- scripts/trace_duel.py (curves, captures, launches, death
  classification, liveness asserts);
- LEDGER_DEBUG decision introspection hook (in c42c9fc; re-add on
  demand — restored v1_4 predates it);
- the measured doctrine map of the Producer matchup (this document);
- two binding methodology lessons: (1) A/B under CPU contention is not
  production behavior — wall-clock TIME_BUDGET makes strength
  load-dependent; cap workers at 3 and spot-check solo; (2) the crash
  guard (return []) silently hides breakage — liveness asserts and the
  parity test are the only tells (the __slots__ incident).

## Post-restoration coda: the freeze-gate hunt and the artifact proof

After restoring v1_4, the PI asked whether the fortress finding could
be turned into a winning strategy. Nine encodings of the phase switch
were built and measured on solo seeds 300/310/316 (v1_4 baseline:
163/100/113 steps survived): stack-wave gates with concentration
filters and maturity guards, realized-wave triggers, airborne-mass
reserve channels with three landing-classification variants, and a
hard time-gate. Every one made the seeds worse (85-123) — each either
throttled the opening (early garrisons/fleets are proportionally as
"dangerous" as late stacks; only phase separates them) or fired after
planets were already doomed.

The tenth experiment killed the premise instead: see the CORRECTION at
the top. The freeze was the opponent timing out, not being deterred.
Producer matchup record at honest compute after the full campaign:
0 wins in ~45 games across all builds.

**New binding A/B lesson: per-phase opponent liveness.** Total-launch
asserts pass on an opponent that degrades mid-game. Batteries must
either run with torch thread limits + low worker counts AND assert
opponent launches in every ~100-tick window, or log per-turn agent
wall-times. (The 8/10 battery passed total-liveness while the opponent
no-opped from ~t80 on.)
