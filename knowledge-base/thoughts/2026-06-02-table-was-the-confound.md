# 2026-06-02 — The kinematics table was a hidden confound on a whole month of A/Bs

## The core realization

The kinematics position-cache was only *actively wired* for ~2 days (05-28 19:06 →
05-30 19:56), and our all-time live peak (μ=1183.7) sits inside that window. Almost
every idea we judged across the comp was therefore measured under one of two
distortions:

1. **Table OFF** (before 05-28, after 05-30): the search is time-adaptive, so fewer
   candidates get scored per turn → compute-hungry features (coalitions, opening
   optimization, richer value heads) had less room to show value, and several were
   filed as "null/parity".
2. **Table ON but singleton-buggy** (05-28→05-30): the module-global cache was
   shared across both seats in in-process A/Bs → corrupted win-rates (the
   "perf-chain confound" / "flat 37.5%").

Both are now removed: the table is de-singletonized (per-turn `world._kt`) and
re-enabled. So the right move isn't only "what's new" — it's **re-running the
shelved ideas with a now-correctly-measured, full-strength agent.**

## Why this matters beyond this comp

A performance substrate (a cache, a vectorization, a budget) is not neutral when the
agent is compute-bound under a per-turn deadline: it changes *which moves get made*,
so it silently confounds every strategy A/B run on top of it. When such a substrate
is added/removed/bugged, **the strategy ledger above it is suspect and should be
re-validated**, not trusted. We nearly threw away ~20 μ by reading the table as
"behaviorally neutral" (true given infinite compute; false under a deadline).

## The PI's instinct led this

The PI insisted "the live champion used the table and performed way better" against
my initial "behaviorally neutral, moot" read. The data backed the PI: the table
flips from net-negative in sparse 2P (overhead > savings) to net-positive in dense
late-game (it kept us under the 950 ms deadline while the table-off agent timed out
at 1003–1208 ms and dropped turns). Lesson reinforced: a single offline average
(the n=32 2P parity A/B) can hide a regime-dependent effect; cut by regime.

## Loss mode (clean signal, vs weaker opponents)

Losing to agents we beat 94–100% removes the "we met a better agent" confound. Those
losses are pure tempo: identical fleet-outcome mix to wins, but we fail to build the
planet lead in steps 50–100 (wins +9 by step 100, losses −5) and then get snowballed
to zero. The two top re-test candidates (team-up coalitions, MILP opening) target
exactly this window.

## Open question for the PI

When we stack the re-enabled levers, do we expect them to compound or to overlap
(team-up + opening planner both fight for the same early-expansion tempo)? Re-A/B
the stack, not just each solo — a solo winner can regress once another lever already
captures the same value.
