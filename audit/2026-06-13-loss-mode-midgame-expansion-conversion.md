# 2026-06-13 — Loss mode: mid-game expansion-conversion failure (step 50–100)

**Source:** real ladder episodes of sub 53595717 (our live agent;
`scripts/diag_live_loss_phase.py`). 2P: 30 W / 20 L. 4P: 30 W / 39 L.
Self-matches excluded. Corroborated by the un-blinded local 4P panel
(40.6% first-place vs a strong-peer pool, 71.9% vs weak lineage).

## The finding

We do **not** lose in the opening, and **not** in the endgame. We lose in
the **step 50–100 expansion window**. Focal planet count (absolute):

| step | 2P WIN | 2P LOSS | 4P WIN | 4P LOSS |
|---|---|---|---|---|
| 25 | 4.2 | 3.5 | 2.7 | **3.0** (we start ahead!) |
| 50 | 9.2 | 6.5 | 4.6 | 3.7 |
| 100 | 12.5 | **6.5 (stalled)** | 9.3 | **3.5 (stalled)** |
| 150 | 14.2 | 4.8 (declining) | 9.7 | 2.6 (declining) |

At step 25 the win/loss games are even (4P losses even start with *more*
planets). Then in losses our planet count **stalls** at ~6.5 (2P) / ~3.5
(4P) through step 50→100 while winners roughly double. By step 100 the
game is effectively decided.

## It is NOT hoarding and NOT under-launching — it is non-conversion

Garrison vs in-flight in the step 50→100 stall window of LOSS games:

| | 2P LOSS 50→100 | 4P LOSS 50→100 |
|---|---|---|
| planets | 6.5 → 6.5 (flat) | 3.7 → 3.5 (flat) |
| garrison ships | 175 → 156 (eroding) | 130 → 155 (flat) |
| **in-flight ships** | **206 → 197** | **123 → 112** |

We have ~120 (4P) / ~200 (2P) ships **in flight** the whole window — we
ARE launching — but planet count does not grow and garrison erodes. The
fleets are not converting into captured-and-held planets. Winners over the
same window launch MORE (in-flight 124→249 in 4P) AND convert (planets
4.6→9.3). So:

**Loss mode = our mid-game launches fail to convert to net planet gains
against a strong opponent, while the strong opponent's launches do.**

Plausible mechanisms (not yet isolated — next diagnostic):
1. We target contested/defended planets and lose the combats (no net gain).
2. Capture-and-immediate-loss churn (take a planet, lose it back within turns).
3. Launches diverted to reinforcement/defense rather than neutral expansion.
4. Reach: after nearby neutrals are taken (~step 50) the planner stops
   reaching for farther neutrals the winners still grab (echoes the
   historic "static K=10 hid 75% of the expansion map" finding).

Note: this is distinct from the shot-validator's "low-P attack" framing.
Removing doomed attacks (reject-only) was a null because the problem is
not that we launch bad attacks — it is that our launches don't CONVERT.
The fix must improve conversion (targeting / sizing / reach), not just
veto.

## Why this matters now

- It is the dominant loss mode on the real ladder, in BOTH formats, and it
  is exactly where we trail a strong ProducerLite peer (the un-blinded
  panel's 40.6% pool).
- It is now testable locally: the heterogeneous 4P panel reproduces it
  (`producer,v7_0,nearest`), so a candidate fix can be A/B'd before a live
  slot (the referee-blindness fix's payoff).

## Proposed next step (PI to steer — observation-driven loop)

Diagnose mechanism #1–4 above by tracing ONE step-50–100 loss game (what
are the ~120 in-flight ships targeting, and what happens to them?), then
propose the smallest existing-knob fix that lifts step-50–100 conversion.
Candidate knobs already in the stack: neutral-expansion reach/horizon,
capture-floor sizing vs contested neutrals, regroup/convoy thresholds.
No submission until a fix clears the un-blinded 4P panel.

## Mechanism isolated (2026-06-13) — capture-and-lose, not reach/under-trying

`scripts/diag_midgame_launches.py` over the step 40–110 window, our
launches by target owner + reach + net planet change:

| window 40–110 | neutral | enemy | own | near | far | ships→neutral | net Δplanets |
|---|---|---|---|---|---|---|---|
| 2P WIN | 20.7 | 31.0 | 64.4 | 72.4 | 43.7 | 688 | **+7.7** |
| 2P LOSS | **23.1** | 24.4 | 54.9 | 61.8 | 40.6 | **675** | **+0.1** |
| 4P WIN | 11.0 | 22.3 | 35.5 | 47.7 | 21.1 | 513 | +6.8 |
| 4P LOSS | 8.1 | 19.4 | 32.0 | 43.9 | 15.6 | 306 | +0.1 |

**2P is the smoking gun:** in losses we make MORE neutral-expansion
launches (23.1 vs 20.7) and send the SAME ships to neutrals (675 vs 688),
yet net +0.1 planets vs the winner's +7.7. Reach is comparable (far 40.6
vs 43.7). So the loss is NOT under-expanding and NOT a reach ceiling — it
is **non-conversion**: our mid-game captures do not stick. Combined with
the earlier signal (garrison erodes 175→156 while in-flight holds ~200),
the mode is **capture-and-lose churn** — we take neutrals and lose them
(or lose the capture race), throwing the same tonnage as winners for zero
net gain.

This reframes the fix (Rule 40 — model the right thing): improve mid-game
capture **holdability**, not launch volume. The launches the shot-MLP
flagged as "won't hold" were real, but rejecting them was a null because
the fix is to make captures STICK (size to survive the reachable counter;
prefer neutrals we can defend), not to stop launching.

Candidate existing knobs (PI to steer; A/B on the un-blinded 4P panel, no
submission until it clears):
- `PRODUCER_PLUS_REACTIVE_FLOOR` (on at 0.5 live) — sizes captures vs
  reachable enemy reinforcement; mid-game holds may need it higher, or it
  may not cover neutral captures. **Most directly implied.**
- capture-floor sizing vs contested neutrals; hold-feasibility target
  prefilter (the btjeK Phase-B idea, never solo-validated).

Next diagnostic if needed: finer trace of a single loss — do captured
neutrals flip back (churn) or do our fleets lose the arrival race
(opponent captures first)? That picks sizing-to-hold vs target-selection.

## Reactive-floor A/B (2026-06-13) — holdability is a SYMPTOM, not the win-lever

Tested the most-directly-implied fix: bump `PRODUCER_PLUS_REACTIVE_FLOOR`
(sizes captures to beat garrison + reachable enemy reinforcement; confirmed
applied to neutral captures at main.py:2172). Un-blinded 4P panel
(`producer,v7_0,nearest`), 8 seeds × 4 seats = 32 games per setting,
thread-pinned, separate processes (env-leak-safe):

| RF | first-place | material share @100 (win / loss) | @150 |
|---|---|---|---|
| 0.5 (live) | 13/32 (40.6%) | 0.487 / 0.160 | 0.718 / 0.089 |
| 1.0 | 13/32 (40.6%) | **0.588 / 0.190** | 0.790 / 0.149 |
| 1.5 | 10/32 (31.2%) | 0.558 / 0.182 | 0.707 / 0.112 |

RF=1.0 raised the mid-game material share (the conversion metric) in BOTH
wins and losses — captures do stick more — yet first-place is **identical**
(13/32 → 13/32). RF=1.5 over-sizes and regresses. So:

**Holdability sizing is refuted as a win-lever.** Making captures stickier
moves the mechanism metric but not the outcome — the same "metric up,
wins flat" pattern as the shot-MLP. We do NOT lose because captures fail to
stick; capture-and-lose churn and losing are both downstream of a deeper
cause. The reactive-floor axis is closed (1.0 flat, 1.5 negative); no
strong-pool regression check needed (nothing to protect).

**Re-pointed diagnosis:** the remaining candidate is target *selection* —
WHICH neutrals we contest against a strong peer, and whether we lose the
arrival RACE (opponent captures first) rather than losing the hold. That
is a proposer/scorer change, not a knob, and should be PI-steered.

Next cheap diagnostic (PI to greenlight): finer trace of loss games — for
our mid-game neutral launches, did the target end up OURS, ENEMY (lost the
race), or did we capture-then-lose? Splits target-selection vs the (now
refuted) sizing story and points to the real lever.
