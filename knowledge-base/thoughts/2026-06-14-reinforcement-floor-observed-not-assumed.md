# The 4-player reinforcement floor: observed, not assumed

*2026-06-14. Observation-driven iteration on the 4-player loss mode.*

## The observation we started from

In 4-player games our agent keeps losing the "whoever reinforces more wins"
way: it pours ships into capturing planets that the field then reinforces
mid-flight, and loses the contested planet plus the ships it sent. Our own
reactive-reinforcement floor — the mechanism that would make us *decline* a
capture the enemy can reinforce — is gated to 2-player only, so in 4-player the
capture floor carries no reinforcement term at all.

The public "Producer V2" agent (currently the strongest public bot, and the
yardstick we now measure against) closes exactly this gap with a simple,
always-on term: it inflates the capture floor by an amount proportional to the
enemy ship-mass parked within reach of the target, times a smooth timing ramp.
So I ported that term verbatim into our agent, gated default-off and 4-player
only, and measured it.

## What we found: the term is a matchup trade, not a win

Two clean 32-game paired pools (same seeds, same seats; only our agent's term
differs):

| Opponent field | term OFF | term ON | effect |
|---|---|---|---|
| 3 × Producer V2 (disciplined) | 22% first place | **38%** | **+16 points** (7 seeds flipped to a win, 2 the other way) |
| 3 × Producer V1 (greedy)       | 44% first place | **34%** | **−9 points** (2 flipped to a win, 5 the other way) |

The flips are consistent in both directions — this is signal, not noise. The
term helps a lot against a disciplined field and hurts against a greedy one.
This also explains why every earlier 4-player reinforcement attempt measured
"null / mild regression": those were averaged over mixed pools where the help
and the harm cancel.

## The root cause

The term scores **potential** reinforcement as if it were **certain**
reinforcement. It reads stationary enemy ships within reach and assumes a fixed
fraction of them (the constant we copied from V2, 2.2) will actually come
defend the planet. That constant is a baked-in assumption about how aggressively
the opponent converts nearby ships into defense.

- Producer V2 really does play at roughly that rate, so the prediction is
  accurate, and declining the doomed capture is correct.
- Greedy bots play at a rate near zero — they never reinforce, they just keep
  expanding — so the prediction is a fantasy. We decline captures we'd actually
  win and hand over free territory.

One fixed number cannot be right for both. **The matchup-dependence isn't really
about the opponent's identity — it's a fixed prior colliding with two different
realities.**

## The fix that needs no opponent-routing

The tempting reaction is "route on the opponent." But you *can't* — in the real
competition the opponent is an anonymous black box; the agent never sees "this
is V1" or "this is V2," only planets and fleets. So the only thing you can ever
condition on is the opponent's **observed behaviour this game**: how much of
their nearby mass have they actually been sending to defend contested planets?

And the moment the routing signal is something you can observe, the routing
*dissolves into* the correct model. There is no "if opponent == X" branch — just
a continuous control law that reads the board and replaces the assumed constant
with the measured one. "Route on observed reinforcement" and "model the game
correctly" turn out to be the same thing. That's the tell that it's right.

## What made this cheap: the infrastructure was already there

The team had already built a reply-trust estimate — an exponential moving
average of how reliably the opponent's *actual* fleet launches match the
launches our opponent-model *predicts*. It runs high against a disciplined,
predictable replier and collapses to its floor against a greedy bot that never
makes the defensive replies we'd model. That is exactly the observed
reinforcement-rate signal the fix needs, already computed online with
cross-turn memory.

So the proper solution is one change: scale the reinforcement term's strength by
that observed trust (remapped so the trust-floor means "term fully off"). The
fixed constant becomes a measured quantity. No opponent identification — which we
couldn't do anyway.

Care taken: the new path computes the trust into its *own* variable and does not
touch the existing reply-trust consumers (background pricing, response veto), so
2-player play stays byte-identical; the term itself is still 4-player only.

## Status

Built as the `vetorf2p_ffa_v2rtrust4p` bundle variant, default-off gates, 2P
byte-identical (parity smoke green), confirmed live in 4-player (it diverges from
the fixed-beta version on a shared seed). Both-pool measurement running at
write-time; the bar for "it worked" is: keep roughly the +16 against the
disciplined field **and** erase the −9 against the greedy field, in one config.

## The transferable lesson

When a mechanism's sign flips with the opponent, don't reach for a per-opponent
switch. Find the **constant that's standing in for an observable quantity**, and
replace it with the observation. The opponent-dependence was never the problem;
the fixed prior was.
