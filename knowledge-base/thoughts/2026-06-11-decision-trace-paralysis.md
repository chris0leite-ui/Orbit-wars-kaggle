# The paralysis trace — what the agent considers, rejects, and should see

Date: 2026-06-11. Trigger: PI screenshot of the Gregor Lied loss ("we fail
to mobilize our ships early") + new tool scripts/decision_trace.py (replays
a live episode step through the planner from our seat and dumps shortlist,
floors, scores, veto).

Three planning turns (16/22/27) of the live loss, verbatim from the
planner's internals: all six planets ARE sources (176 drainable ships, no
safety reserve binds), 13 of 22 targets ARE shortlisted — and EVERY capture
candidate scores +0.0 against the +1.5 fire threshold. Zero launches,
three turns running, while the opponent streams 16-ship captures.

Causes by target class:
1. In-horizon truncation: a +5-production neutral reachable at eta 18
   (= horizon edge) earns production for 0 in-window turns -> score 0.
   Banking looks free; expansion looks worthless.
2. 62-garrison corners: floor 63 > best single drain 59 -> structurally
   invalid for any single source (the coalition capability now live in
   sub 53577315 addresses exactly this).
3. Mid-board neutrals near the enemy: reactive-floor margin inflates
   floors to 70-108 — honest lost races, not defects.

The obvious fix (flat terminal production credit, λ=12) is REFUTED on
both local instruments: the invested capital gets punished faster than it
pays back, even by the mild old champion (4/8 +20% vs control 6/8 +51%).
The diagnosis stands but the fix must price COUNTER-SAFETY: terminal
credit per target = production x expected holding time given the
opponent's feasible retake (the ledger branch's capture pricing). That is
the next foundation build.

Meta-lesson: the decision trace turns "why didn't it act" from speculation
into a table. Use it on every PI replay observation before proposing
mechanisms.
