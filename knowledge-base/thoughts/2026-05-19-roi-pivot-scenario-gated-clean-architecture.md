# ROI pivot: clean architecture + observation-grounded scenario gate

**Date**: 2026-05-19
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Context**: End of Phase E session. Phase 3 compound-weight sweep
finalised; PI gave the strategic pivot live (not in writing first this
time — voice-dump captured in this turn's chat). Plan written and
approved at `/root/.claude/plans/no-go-forward-test-fluttering-token.md`.

## The pivot, as PI articulated it

> Currently, what we have now, we have this architecture. It should be
> precise and fast. Right now we're iterating on what would the enemy
> do and so on. Actually, the first strategy that turned out pretty
> well was a return on investment strategy, and it scored about one
> thousand. We also observed that in live games we have many
> inconsistencies — we waste a lot of ships.
>
> So I would suggest to pivot entirely, to drop everything but the
> architecture that we have, and to test a simple return on investment
> strategy with our clean architecture, and to improve that return on
> investment strategy by making sure that it does make the right
> decisions in some synthetic obvious scenarios.
>
> First, develop the ROI strategy — really simple, trajectory-native.
> Second, tweak it to comply with obvious scenarios. We develop a large
> set of synthetic scenarios that are obvious, and we need to make it
> pass all these scenarios. First, draft these scenarios and make sure
> they're good (no sun in the way). Then create the ROI, A/B against
> our current best, and iterate ROI until it complies with all the
> scenarios.

A second voice-dump added two more failure modes and the
"no-hotfixes" architectural constraint (failure modes (d) split-
majority and (e) distant-idleness; ROI architecture must extend
naturally, not by stacking `if` patches).

## What the architecture is, concretely

**Keep:** `lib/{fast_sim, trajectory_layer, world_model, trajectory,
opp_model}.py`. These are the precise+fast substrate.

**Drop:** `agents/bundle/`'s entire decision stack — chooser, scorer
coefficients, candidate enumeration, all the `BUNDLE_*` env vars.
Also: NOT porting baseline's `MissionPanel`.

**Build:** `agents/trajectory_roi/` with six first-class primitives.
Multi-source bundling is in `enumerate()` as a candidate TYPE, not
a coefficient.

## Why this is a real pivot, not a re-skin

Bundle's failure mode wasn't "wrong coefficient." It was an
architectural axis exhaustion (chooser+scorer+opp-model knobs all
exhausted), masking a structural issue: bundle's enumeration only
produces single-source candidates. Joint coordination has to emerge
implicitly via beam search; it almost never does because solo
launches that would bounce get pruned BEFORE their partner is
discovered. Phase 1's joint-bonus coefficient tried to compensate
for this by promoting bouncing-but-joint-promising candidates —
nulled in production.

ROI's `enumerate()` produces multi-source candidates as
first-class types, scored as a unit. The 100+100-vs-50 scenario
(failure mode (d)) is solved by construction.

## Why scenarios are now the gate, not just A/Bs

The calibration gap (-20 to -30 pp local-vs-live over three recent
submissions) is a louder signal than any of our A/B winrates. It
means the local evaluator is over-fit — most likely to v7_0-style
play, since that's our most-A/B'd opponent. We've been polishing
against a noisy outcome metric, and a biased one at that.

Synthetic scenarios bypass both noise and bias:
- Each scenario has a deterministic correct answer.
- A scenario's pass/fail is independent of opponent strength
  distribution on the local pool.
- A scenario describes a MODELING claim ("the agent should know
  this") not an outcome claim ("the agent should win at this
  rate").

This is consistent with Rule 40 (modeling-correctness over
restriction-tuning) and Rule 38 (fix-verification reproduces failure).
A scenario IS a fix-verification rig.

## The "no hotfixes" constraint

PI's explicit framing: "It should comply with these scenarios in a
natural way and not by imposing more and more hotfixes."

Operational meaning: when a new scenario fails, we don't add an `if
scenario_pattern_X: do_Y` patch to the scorer. We extend one of the
six primitives:

| Failure pattern | Primitive to extend |
|---|---|
| Multi-source bundling needed | `enumerate()` |
| Recapture mis-prediction | `predict_arrival()` |
| Ray-cast / drift unaccounted | `reachable()` |
| Source-exposure unaccounted | `score()` (source-cost component) |
| Opp counter-play missed | `refine_via_rollout()` |
| Same-target dogpile / dedup wrong | `select()` |

If a fix doesn't fit any of these, the architecture is wrong and we
revisit the primitives — not graft an exception.

## What I learned about my own behaviour this session

PI's strategic redirect was written on 2026-05-18. The redirect said:
pivot off chooser-axis, go objective-first. I executed three more
chooser-axis variants (Phase 1 joint coordination, Phase 2 bounce
penalty, Phase 3 compound-ROI). All justified by the older Phase E
plan, written BEFORE the redirect.

This is Rule 38 territory (fix-verification reproduces failure) and
also Rule 37 territory (axis exhaustion). I'd been told the axis was
exhausted; I kept iterating on it because the existing plan said to.
Plan documents have inertia. Strategic redirects don't propagate
automatically — they need a plan rewrite to bind.

Anti-pattern to log: **executing a plan document that pre-dates the
latest strategic redirect**. Mitigation: before any phase start,
check `knowledge-base/thoughts/` for strategic-redirect entries
newer than the plan doc.

## Open questions for next session

1. **Live μ of sub 52744856 (composite+A2 hybrid).** Was PENDING at
   last session log; status unknown today. Determines whether the live
   champion line moved and whether bundle's gap to live is the same
   gap we measured locally.

2. **Does failure mode (c) garrison-counter actually show up in
   composite+A2 / v15 replays?** Or is it a mental-model failure that
   we'd see in bundle but not in the live agents? Replay mining
   (Phase 1a) answers this.

3. **Does failure mode (e) distant-idleness show up in live? Or is it
   bundle-specific?** Same answer source: replay mining.

If (c) or (e) don't show up in live replays, the priority order of
scenarios changes — they become lower-priority than whatever IS
showing up in live failures.

## Connection to prior thoughts

- `2026-05-18-strategic-redirect-from-tactical-mechanics.md` — the
  redirect this session was supposed to honour and didn't (for three
  more phases) before finally pivoting.
- `2026-05-17-substrate-reframe.md` — earlier framing of "stop
  iterating leaves, build substrate." Same shape; this is the third
  re-articulation.
- `2026-05-16-chooser-family-saturation.md` — Rule 37 origin on the
  bundle chooser axis.
