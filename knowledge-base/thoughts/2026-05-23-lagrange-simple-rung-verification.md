# 2026-05-23 — Rung-verification on `agents/lagrange_simple` (branch `claude/session-EqJuT`)

## TL;DR

Re-ran the three rungs from the PI direction note (`01f7544`) against
the current EqJuT HEAD `ce70160`. Result:

| Rung | Opp        | n  | Wins | ELIMs | Losses | Verdict             |
|------|------------|----|------|-------|--------|---------------------|
| 1    | `random`   | 16 | 16   | **9** | 0      | FAIL 100%-ELIM gate |
| 2    | `starter`  | 16 | 15   | **9** | 1      | FAIL 100%-ELIM gate |
| 3    | `baseline` | 16 | **0** | 0    | **16** (15 by-elim against us) | FAR FAIL — direction reversed |

Artifacts: `audit/probe-results/2026-05-23-rung{1,2,3}-*-n16.out`.

## Two surprises vs prior in-tree claims

1. **HEAD commit `ce70160` claims "14/16 ELIM at default rng-seed=2026"
   for rung 1; actual re-run produced 9/16 ELIM.** Same script, same
   default seed (2026), no source change in tree. Either the commit
   self-report was inaccurate at write-time, or some non-tracked state
   (kinematic_table on-disk cache, scoring code path under
   `BASELINE_ORBITAL_SAFETY=1`) drifted the outcome distribution. Most
   plausible reading: the original 14/16 came from a stale-cache run
   that warmed before-Phase-B and the post-Phase-B re-evaluation was
   not actually executed under the published commit.
2. **The session-end PI direction note in
   `knowledge-base/questions/2026-05-23-multi-source-dogpile-next.md`
   recommends rung 3 = "~50 LOC dogpile, iterate until pass". That
   recommendation was filed BEFORE the within-session Phase C dogpile
   experiments — all 6 of which regressed, closing the axis under
   Rule 37 (per the body of `ce70160`).** The note still recommends
   the now-falsified mechanism. The axis closure invalidates the
   recommended rung-3 path.

## Why 100%-ELIM-vs-baseline is structurally not reachable here

(Decision-quality framing: even with retroactive knowledge of the gate
result, the *prior at decision-time* that the note was written from is
itself contradicted by within-session evidence. The conclusion below
holds against the priors actually in tree at HEAD.)

1. **Direction reversed.** The gate asks us to drive baseline to 0
   planets. The empirical state is baseline drives US to 0 by turn
   ~170 median. We don't fail the ELIM bar — we fail the survive-
   to-midgame bar.
2. **Single-source-per-target is load-bearing in `score.py:233`
   (`ships > gar_at_arr`).** Baseline consolidates; final-pocket
   garrisons exceed any single source we have. The dogpile axis (the
   only way to deliver multi-source volleys) is closed permanently
   under Rule 37 after 6 same-session variants regressed.
3. **Phase B `_source_defensive_ok` over-prunes against strong
   opponents.** When opp's projected counter is heavy on every
   source, the filter rejects almost every solo — the agent goes
   passive and baseline overruns the map.
4. **B1 hold filter is calibrated for orbitfix midgame, relaxed only
   at `n_my ≥ 3*n_opp`.** Against baseline we never reach
   dominant-endgame, so the relaxation gate that fixed the random
   seed 14514 cannot fire — every attack whose target opp could
   recapture is filtered, which against baseline is essentially
   every attack.
5. **No opening theory / non-reactive launch policy.** The agent is
   purely reactive: enumerate `(src, tgt, launch_tick)`, score under
   Lagrangian. Against baseline's drain+sniper opener this is too
   slow to react before we lose force concentration.

## What this implies for next steps

- The rung-3 target (100% ELIM vs baseline) is not extensible from
  `lagrange_simple` along the dogpile axis. To make rung-3 reachable
  the agent needs all three of: cross-source defensive coordination
  (Phase D — a coalition-of-rears check that doesn't reduce to
  per-source checks), an opening / force-projection theory (not
  pure-reaction enumeration), and game-phase-aware filter tuning.
  That is a different agent design, not an extension.
- Even rung 1 (random) doesn't currently clear at default seed.
  Whatever stale-cache drift moved 14/16 → 9/16 between commit
  `ce70160` and the verification run needs to be reproduced before
  any further work on this agent — otherwise the gate isn't a
  trustworthy signal.

## Cross-references

- HEAD commit: `ce70160` ("opp-projection + rear-defense check
  (13/16 → 14/16 ELIM)").
- PI direction note (now stale): `knowledge-base/questions/2026-05-23-multi-source-dogpile-next.md` "PI direction (2026-05-23 session-end)".
- Closed axis evidence: `agents/lagrange_simple/dual.py:30-44`
  (6 dogpile variants, all regressed).
- Earlier postmortem: `audit/2026-05-23-postmortem-session-EqJuT.md`.
- Friction tags appended this verification session:
  `gate-score-claim-not-reproducible-from-head`,
  `pi-direction-note-overtaken-by-same-session-evidence`,
  `rung3-empirically-out-of-reach-for-lagrange-simple`.
