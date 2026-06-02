# jsr-line vs champion — the architectural wall

**Date:** 2026-06-01 (UTC: session straddled into 2026-06-02 AM)
**Branch:** claude/competition-objective-alignment-hqNVM
**Author:** Claude (Opus 4.7)

## TL;DR

Today's session attempted 5+ architectural mods to make jsr beat the
rolling-pair champion (`baseline_launch_rules_universal_local`, live
μ=1183.7). All failed at the local-A/B-vs-champion gate (0/16). The
strongest mod (v7_search add_one aggression-mode handoff) DID lift
jsr by +22pp over its baseline (68.8% vs jsr at n=16) — but still
loses 0/16 to champion. The jsr-line architecture appears to be
structurally bounded below champion's level, by some load-bearing
mechanism that no jsr-internal tweak addresses.

## What we tried (in chronological order)

1. **bisect (no joint_sync)** — sub-component A/B vs full jsr. Negative.
2. **bisect (no composite — strip distilled opp + B.3 head)** — negative.
3. **bisect (no size_balance)** — negative.
4. **ROI mode-switch v1** — buggy resolver (counted outgoing in-flight
   as my_ships). 5/16 vs jsr (31%).
5. **ROI mode-switch v2** — fixed resolver. 7/16 vs jsr (44%). Wallclock
   max=1445ms (over 1000ms cap). Closed-form ROI structurally over
   commits because per-turn greedy scoring is myopic about cross-turn
   cumulative drain. Smoke seed=0 collapsed from ratio=1.54 at turn 56
   to 0.53 at turn 65 (9 turns), opponent counter-landing on emptied
   planets.
6. **v7_search add_one (drop_one falsified at exploration as too passive)**
   — first 2 attempts (v1 with decorators, v2 without) BOTH 0/16, masked
   by bundle name-collision bugs (`choose` and `score_candidate`).
7. **v7_search add_one v5** (all bundle-name fixes) — **11/16 vs jsr
   (68.8%, Wilson [0.444, 0.858])**, wallclock max=929ms under cap.
   Direction was strong but didn't clear Rule 45's 0.75 bump-to-n=32
   threshold. **0/16 vs champion** — same as all other jsr-line mods.

## What the data tells us

The pattern is consistent across every jsr-line mod today: **wins vs
jsr can grow** (parity → +22pp), **but vs champion stays at 0%**.

Implications:
- jsr's diagnosed weakness (failure to convert advantage) is REAL — the
  +22pp local lift from v7's K=10 rollout add_one confirms aggression
  conversion was the rate-limiting step for jsr-vs-jsr-style matchups.
- BUT champion has a structural advantage that the conversion fix
  doesn't reach. Hypotheses worth exploring next session:
  - Champion's launch_rules universal validator may resolve a
    different failure mode (defensive timing) than jsr's value-head
    veto (offensive conversion). Improving conversion doesn't help
    when the lossy step is something else.
  - Champion's prerank may surface different candidates than jsr's
    even before the chooser stage. Adding conversion to jsr's
    candidate pool doesn't help if the right candidates aren't there.
  - Champion may exploit a counter-strategy specific to jsr-line
    (e.g., reading jsr's predictable buildup phase and timing a
    counter-attack to the exact moment of conversion). Architectural
    mismatch, not parameter tuning.

## Bundle-collision lessons (load-bearing for future submissions)

The catastrophic 0/16 vs jsr happened TWICE before we found the real
cause. Both times the cause was bundle-inlining creating name
collisions in the global namespace.

The bundler uses naive substitution:
- `from lib.v7_search import X` → `X = X` (after inlining the lib)
- This works when X is unique, but when X is duplicated across libs
  inlined LATER, the later def-statement rebinds the bare name.
- Python late-binds, so even `X` REFERENCES inside an earlier-inlined
  function body get resolved to the LATER def at call time.

**Two fixes are needed for each collision:**
1. Public entry points: define a module-level alias at the END of
   the early-inlined lib (`v7_search_choose = choose`). The alias
   captures the function object, not the name — survives later
   rebind.
2. Internal helpers: rename inside the early-inlined lib so the
   function body's references resolve to a uniquified name
   (`_v7_score_candidate`).

The full collision check across v7_search.py vs chooser{,_trajectory,_roi}.py
turned up only 2 conflicts (`choose`, `score_candidate`). Worth
running this check at bundle time, raise on any duplicate top-level
def.

## What to keep, what to throw away

**KEEP (committed, useful even outside the mode-switch design):**
- `v7_search_choose` alias (lib/v7_search.py end) — bundle-safe
  access to v7_search.choose under a unique name. Useful for any
  agent that wants to call v7_search from within agents/baseline/.
- `_v7_score_candidate` rename — fixes the same class of bug. May
  be needed by any future bundling that inlines both v7_search.py
  and chooser_trajectory.py.
- `_resolve_chooser_for_turn` helper in agents/baseline/main.py —
  parameterized by `BASELINE_AGGRESSION_CHOOSER`. The aggression
  handoff dispatch is wired and tested (10/10 bundle parity, 12/12
  resolver unit tests). Future axes can reuse this without rewiring.
- `v7_add_one` dispatch branch in agents/baseline/main.py — works,
  honored by Rule 46 parity. Doesn't activate unless explicitly
  enabled by env var. Available for future agents.

**DEFER (in repo, not active):**
- The `_modeswitch` wrapper (ROI handoff) — superseded by `_addone`
  but the closed-form ROI dispatch may be useful for a different
  context (e.g., applied to champion-line where the failure mode is
  different).
- The `_addone` wrapper — keep for archival. Its 0/16 vs champion
  rules it out for submission via THIS path; doesn't rule it out
  for a different host agent.

**THROW AWAY (none — all today's work compiles and tests):**
- nothing to delete

## What I'd do next session (PI to override)

The PI signaled "lock champion as rolling-pair anchor + defer." That
implies tomorrow's first priority is to figure out the structural
question — WHY can't jsr-line beat champion? Three concrete next moves
ranked by EV:

1. **Apply add_one handoff to CHAMPION (not jsr).** Different axis
   (Rule 37 — N resets when the load-bearing component changes).
   Champion + aggression-conversion is the SYMMETRIC bet to today's
   jsr + aggression-conversion. If champion is already converting
   well, this won't help. If champion misses some conversions, this
   could be the breakout. ~1 hour to implement + A/B vs both
   current rolling-pair entries.

2. **Public-notebook scan (Rule 22 at plateau).** What is the field
   doing that we aren't? Top 5 notebooks with ≥10 votes. May surface
   a structural insight we can apply.

3. **Replay analysis of champion vs jsr.** Pick a seed where champion
   beats addone-v5 decisively. Walk through the game turn-by-turn,
   identify the LOSING DECISION class (defense timing, prerank
   mismatch, counter-attack response). Inform whether move 1 or 2
   is more likely to land.

## Loose ends

- `_resolve_chooser_for_turn` debug log file overhead — fine for
  smoke (we instrument BASELINE_ROI_SWITCH_DEBUG=/tmp/path), but
  the per-turn fopen+write+close is non-trivial in eval. Default
  is no debug, so production-clean. Just don't ship the debug var
  enabled.
- v7's K=10 rollout produced empty moves on 43/144 turns in the
  smoke — i.e., 30% of the time it decided "incumbent already
  optimal, don't add." That's the parity floor working as designed,
  but it means the lift comes from the OTHER 70% where it added 1-7
  launches. If we want more conversion, lower the K (more turns
  where the rollout sees positive lift) — at the cost of more
  greedy decisions. PI may want to A/B K=8 vs K=10 in a follow-up.
