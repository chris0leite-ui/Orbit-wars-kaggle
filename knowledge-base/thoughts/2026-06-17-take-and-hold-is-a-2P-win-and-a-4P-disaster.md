# Take-and-hold is a 2P win and a 4P disaster — the μ cap is 4P

**Date:** 2026-06-17. **Trigger:** PI asked to revise the last ladder games,
look at losses directly, and rethink weaknesses + planned improvements —
reminding me that scores do not settle (we keep playing, competitors improve).

## The score correction (warm-up, not regression)

Last session I called take-and-hold a "224 μ regression" off an **890** reading.
That was TrueSkill warm-up. Today the same submission (53772947) is at **1075.8
and still climbing**. Lesson, again: never read a verdict off a warming score.

But it IS structurally capped — and pulling the real ladder replays for both
agents shows exactly where.

## The finding — split by game type (real ladder, not self-play)

| agent | 2P win% | 4P win% (fair share 25%) |
|---|---|---|
| lr-fixed (μ1116) | 60.9% (28/46) | **31.2% (10/32)** — above fair |
| take-and-hold (μ1075) | **70.4% (19/27)** | **10.5% (2/19)** — bottom-tier |

**Take-and-hold traded +9.5pp in 2P for −20.7pp in 4P.** 4P is ~40–60% of
games, so the 4P collapse outweighs the 2P gain — that is the entire μ cap.
It is NOT the dribble and NOT warm-up. And lr-fixed was *good* at 4P (31% >
fair share); take-and-hold's concentrate-and-hold actively wrecked it.

This also explains why the local n=32 panel missed it: "4P parity" at n=32 was
too noisy / too favorable a panel to catch a −20pp ladder collapse. The ladder
is the truth.

## CORRECTION (same day, from PI replay observations): it is RECAPTURE CHURN, not under-expansion

The PI watched 4P replays and pushed back: *"we are not expanding too slowly, we
are rather too aggressive too early"* and *"we capture a neutral and the opponent
recaptures it — where is our thinking?"* Measured directly, the PI is right and
my "under-expand" reading was a symptom, not the cause:

- **~⅓ of opening (steps 0–30) ship-mass is aimed at ENEMY planets** (31%
  take-and-hold, 32% lr-fixed) instead of uncontested neutrals — genuinely
  aggressive for a 4P FFA opening, and present in BOTH agents.
- **The killer metric — recapture churn:** across the 17 take-and-hold 4P
  losses, of **227 planets captured, 98 (43%) were recaptured within 8 steps**,
  and **90 of those 98 (92%) had an enemy fleet already inbound with more ships
  than the garrison we left (median garrison 18).** We capture straight into a
  *visible* recapture.

So we are not passive — we are HYPERACTIVE and ineffective: we grab tons of
planets, ~half evaporate immediately, which only *looks* like under-expansion in
a step-30 planet snapshot. The "collapse 3.8→2.9" above is this churn.

**Modeling cause (the real one):** capture sizing is minimum force (`garrison+1`)
and the value function credits a captured planet's production *as if we keep it*
— it is **blind to enemy fleets already in flight toward that planet.** The 2-ply
lookahead / garrison-flow leaf does not subtract the visible recapture, so a
planet we will lose in 3 turns scores like one we will hold.

**This supersedes the player-count gate as the fix.** The gate reverts 4P to
minimum-force captures — i.e. exactly the behavior that churns — so it does not
address the root cause. Built + bundled (sha `39679a9a`) as a possible interim,
but shelved.

**The fundamental fix (proposed, awaiting sign-off): recapture-aware capture
valuation.** Size every capture (neutral AND enemy) to survive the enemy mass
*already inbound to that specific planet*; if we cannot afford to hold it, do not
take it (redirect the ships to a holdable target). The "dynamic margin" done
right — margin = the real visible incoming threat per planet, not a flat 0.5 and
not a 2P/4P gate. Kills the churn, tempers early aggression, and spends the
compute headroom (per-candidate threat evaluation). **Verifiable without trusting
the ladder:** reproduce the 43%/92% recapture rate locally, apply the fix,
confirm the foreseeable-recapture rate drops (a behavioral check independent of
local-μ-predicts-ladder-μ).

## (Original, now-superseded framing) under-expand + collapse

- **Under-expand:** take-and-hold holds **3.2 planets @step30** vs lr-fixed
  **3.8** — fewer opening captures (6.5 vs 7.8 launches in steps 0–30).
- **Then collapse:** take-and-hold falls 3.8 planets @60 → **2.9 @90**;
  lr-fixed holds steady (4.3 → 4.0). (Re-read above as recapture churn.)

**Mechanism in one sentence:** concentrate-and-hold is a *duel* strategy. 2P is
one front — concentrate, overwhelm, hold against the single rival → wins (70%).
4P is three fronts — can't hold everywhere, over-committing to one hold gets you
eaten by the other two, breadth/tempo beats concentration → loses (10%).

## The fix (proposed, awaiting PI sign-off): player-count gate

Run take-and-hold in 2P, fall back to lr-fixed's breadth-first minimum-force in
4P. Smallest change; modeling-correct (a duel and an FFA are different games,
so the concentration model's validity genuinely depends on opponent count — not
a band-aid cap, Rule 40 OK).

**Why this is unusually safe — and answers the session's "local panels lie"
problem:** no local A/B needed. Each half is *already ladder-measured* — 2P
take-and-hold = 70%, 4P lr-fixed = 31%. The spliced agent is byte-identical to
take-and-hold in 2P and byte-identical to lr-fixed in 4P. It is not a new bet;
it is the ladder-winning behavior in each regime. Expected ≈ 70% 2P + 31% 4P →
beats BOTH current agents, should settle above 1116.

## Reframes the queued "threat-aware dynamic margin"

The margin should not track threat *magnitude* — it should track number of
*fronts* (opponents). 2 fronts → hold (margin 0.5). 3 fronts → breadth
(margin 0). The 2P/4P gate is the cleanest first cut of exactly that idea. If a
continuous version is ever wanted, scale the margin down by opponent count;
but the ladder is 2P/4P only, so the gate captures it.

## Weaknesses, ranked

1. **4P concentration-collapse — the whole μ cap.** Fix = the player-count gate.
2. **Dribble (size-4 on long tracks)** — real, but a 2P micro-inefficiency and
   we win 70% of 2P. Parked behind #1.
3. **Latent slow turn (1441 ms 4P-vs-field)** — still unbounded; separate risk.

## Repro

`python -m scripts.live_episode_summary <sub_id> --pull` then the two scratch
tracers (`/tmp/fourp_trace.py`, `/tmp/launch_profile.py` this session) over
`audit/live-episodes/<sub_id>/`. Raw replays are gitignored (re-pullable);
`summary.json` + `episodes.csv` are committed.
