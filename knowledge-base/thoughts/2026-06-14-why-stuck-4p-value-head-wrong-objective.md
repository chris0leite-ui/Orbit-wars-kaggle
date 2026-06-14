# Why we're stuck: the 4-player value head optimizes the wrong objective

*2026-06-14. The PI asked: why are we stuck at a local optimum, what are we
missing, and — looking at the game through different representations — what are
the next steps. This is the rethink. It supersedes the "4P axis is exhausted,
consolidate" conclusion from earlier today: that conclusion was right about the
**sub-axis** we were on and wrong about the axis existing at all.*

## TL;DR

Every 4-player lever we have built and rejected this session optimized
**survival and material** — defend better, waste less force, disengage. But the
competition only rewards **finishing 1st** (TrueSkill is margin-agnostic; in a
4-player game 2nd, 3rd and 4th are identical losses). We have been hill-climbing
the coefficients of an agent whose **4-player value function literally maximizes
material-against-the-field**, not probability-of-finishing-first. No coefficient
tweak escapes that basin, because the basin is defined by the objective itself.

The single highest-leverage finding, **verified in the live code**: our 2-player
value head already does the right thing and our 4-player head does the wrong
thing, in exactly the way that turns us into a safe-2nd-place bot.

## The smoking gun (verified, `agents/baseline/value.py:68-127`)

The chooser picks moves to maximize a leaf "favor" value. Its opponent term:

- **2-player (line 100-101):** `opp = max over opponents`. Value =
  `my_strength − strongest_opponent`. This is **gap-to-the-leader** — and in a
  two-player winner-take-all game the leader *is* the one opponent you must
  beat. Correct. This is why 2-player is our strength (μ ≈ 1271-1291).
- **4-player (line 102-120):** `opp = Σ over all opponents` with the weakest
  weighted ×1.5. Value = `my_strength − weakness-weighted total of the whole
  field`. This is **gap-to-the-field** — a material-hoarding / survival score.

To make the 4-player value large you accumulate material and chip the weakest
rival. You are **not** rewarded for being the single strongest — only for being
far ahead of the *sum*. The state that maximizes this objective is a comfortable
2nd-3rd place with lots of ships. TrueSkill pays that **zero**. The 4-player
weakness-bias is documented in the file as borrowed from a peak-leaderboard bot
(romantamrazov, μ=1224), i.e. a copied heuristic, not something derived from the
winner-take-all structure of the game.

**The asymmetry in the value head is the asymmetry in our performance.** Strong
2-player, plateaued 4-player — because the 2-player objective is winner-take-all-
correct and the 4-player objective is a material proxy.

## Why this is precisely the local-optimum trap

Our whole iteration loop has been: PI observes a 4-player loss → we add a
default-off lever that adjusts how the chooser *values* or *filters* a send →
A/B it → it's null or negative → shelve. Five times this session and before:
reinforcement floor, decline-captures term, threat-window extension,
disengagement brake, offensive re-weighting. **Every one was a refinement of a
material/survival objective.** Re-read through the right lens, their failure was
*inevitable*, not informative:

- The **disengagement brake** (conceding doomed planets) lost material-share, so
  we called it falsified. But material-share is the wrong scoreboard. What it
  *actually* did — and what we buried — is it dropped first-place finishes
  19/63 → 17/63 → 15/63. It hurt the **right** metric too, but for a reason we
  mis-stated: conceding makes you the soft target. Fine. The point is we were
  reading the wrong number as the headline the entire time.
- The **threat-window** and **reinforcement floor** were survival mechanisms —
  they make you a better 2nd place. Even if they had "worked" on share/rank,
  they would not have moved win-rate, because they don't make you the leader.

We weren't unlucky with five levers. We were optimizing the gradient of the
wrong function. You cannot reach "win more" by descending "lose less material."

## The same conclusion from seven representations of the game

The PI asked to look at the problem in different representations. Each one,
independently, points at the same missing piece — **optimize P(1st), expressed
as gap-to-the-leader, and accept variance to get there.**

1. **Game-theoretic (winner-take-all, margin-agnostic).** Only 1st scores; 2nd =
   4th. Therefore a line that wins 30% / comes last 70% beats a line that
   reliably comes 2nd and wins ~0%. The optimal policy is **variance-seeking
   when the safe line doesn't finish 1st.** Every lever we built *reduced*
   variance. We have been doing the opposite of what the payoff structure
   rewards.

2. **Value-function (what the agent literally maximizes).** Confirmed above:
   4-player favor = gap-to-the-field = material/survival. The objective is baked
   wrong in code, not just in our measurement.

3. **Optimization-landscape (our own iteration).** ~20 `BASELINE_*` coefficient
   flags + ten post-chooser "drain" passes (idle, stagnant, combat-stack,
   sniper, reinforce…). The drains are band-aids that exist *because* the greedy
   chooser leaves value on the table. We are deep in a coefficient basin. Escape
   requires changing the **objective** or the **architecture**, not another
   coefficient.

4. **Combat-math (`interpreter.py:817-832`).** Only the largest and 2nd-largest
   forces at a planet are scored; a 3rd player's fleet is mathematically
   ignored; equal forces annihilate. This *rewards concentration and decisive
   overtake* and *punishes spreading* — exactly the behavior a gap-to-the-leader
   objective produces and a gap-to-the-field objective discourages.

5. **Temporal / phase.** The opening (steps 0-~30) is predictable — few fleets,
   known positions — so it is *searchable*, and it is where the eventual leader
   is decided. We have **no** opening strategy (direction is emergent from
   nearest-8 scoring; the documented adaptive-opening lever isn't even shipped).
   This is the phase where both "become the leader" and "search is cheap"
   coincide — the highest-leverage place to plan rather than greedily score.

6. **Architecture (greedy sequential + shallow opponent model).** The agent
   K-step-simulates each candidate but then **emits greedily, one move at a
   time, assuming opponents play one-turn-reactive greedy.** It cannot express
   "sacrifice value now to set up the winning blow in three turns," and it
   cannot see "all three will gang up on me." Multi-step planning and a deeper
   opponent model are the structural unlocks — but they're expensive, and the
   value-head fix is far cheaper and likely captures most of the 4-player gain
   first.

7. **Seat-geometry (fixed map, random seat).** Bad seats (adjacent to the strong
   rival, weak rival unreachable) are real and unchangeable. But the correct
   response to a bad seat in a winner-take-all game is **not** "optimize a safe
   2nd" — it's "take the variance line that occasionally steals 1st." Our
   "forced collapse" narrative was really "we folded to 2nd instead of gambling
   for 1st." The geometry caps our *safe* equity, not our *variance* equity.

## What we're missing, in one sentence

A **win-equity objective**: value the gap to the *strongest* opponent (the one
you must overtake to finish 1st), which automatically produces concentration
(combat math) and variance-seeking-when-behind (game theory) — instead of the
gap-to-the-field material score that makes us a comfortable, unrewarded 2nd.

## Next steps, ranked by leverage / cost

1. **Fix the scoreboard first (cheap, unblocks everything).** Make first-place
   rate the headline metric in every 4-player A/B; demote material-share and
   mean-rank to diagnostics. Re-score the shelved levers on win-rate — did we
   reject a winner or keep a loser? This is re-analysis + a calibration probe,
   not a new lift claim (Rule 45 exception), but any new claim still needs
   n ≥ 32, Wilson-lo ≥ 0.50.

2. **The modeling fix — leader-relative 4-player value head (Rule 40, highest
   leverage).** Change the 4-player branch of `favor` from gap-to-the-
   weakness-weighted-sum to **gap-to-the-strongest-opponent** (mirror the 2P
   `max-of-opps`, possibly softened to `max + small·rest`). Default-OFF gated,
   byte-identical when off, A/B on **win-rate**. Hypothesis: it converts safe
   2nds into overtake attempts — material-share and mean-rank may *fall* while
   first-place rate *rises*. That divergence is the whole point and is exactly
   why our old metrics would have hidden it.

3. **Opening as a small search problem (architecture, where it's tractable).**
   In the predictable opening, plan the expansion to win the race-to-leader
   rather than greedily scoring nearest neutrals. Medium effort; targets the
   phase that most determines who finishes 1st.

4. **(Longer, defer) deeper / coalition-aware opponent model.** Predict gang-ups
   and multi-turn opponent plans. Expensive; only if (2) and (3) show the
   bottleneck has moved here.

## Open reconciliation (resolve before implementing)

State drift to settle first: `STRATEGY.md` names `baseline_adaptive_k` as the
champion and describes an adaptive-opening lever that is **not shipped** (live
code uses fixed horizons). Meanwhile this session's work (reinforcement floor,
threat-window, brake) was on a **`producer_plus`** stack living in a separate
worktree — which does **not** exist in the main repo, where the live value head
is `agents/baseline/value.py`. Before touching the value head we must confirm
**which stack is actually the live submission**, so the fix lands on the agent
that plays the ladder. This is the first thing to nail down.
