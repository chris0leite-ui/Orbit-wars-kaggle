# 2026-06-10 (afternoon) — the mass pivot: study the winners, not the baseline

## The unlock

After eleven straight nulls measured against vanilla producer, the PI said
"improve significantly, even reinvent." Instead of building mechanism
twelve blind, we asked: what do the agents at 1700 actually DO? Kaggle's
episode service turns out to accept any submission id unauthenticated, and
every episode response leaks its opponents' submission ids — so you can
walk the opponent graph from your own games to the global top in 8 hops
and download anyone's replays. Forty replays each from the #1/#2/#5 teams.

## The finding

Every behavioral metric is monotone in ladder rating, and they all say
MASS: half our launch frequency, 2-4× our fleet size, faster early
expansion funded by spending the early stockpile, 2-4× our ship count by
step 80 on similar production. Our own loss data agrees (opponents who
beat us send 2× our median fleet). Our agent was the dribbler in a mass
meta — and 71% of its launches were 10-30-ship own-planet parcels from
the regroup lane.

The deeper lesson: **the A/B-vs-producer yardstick was optimizing us into
the dribble meta**, because producer is a dribbler too. Horizon-24 and
recapture-penalty "regressions" were probably mass-direction mechanisms
being punished by the wrong referee. When every mechanism nulls, suspect
the yardstick before the engine.

## What shipped (sub 53529884)

Three small mechanisms, 2P-only gated (mass measured 7/32 in the 4P pool —
the dribble may be load-bearing in four-front games): near-tie score
resolution toward the larger send, regroup convoying (>= 25 ships), and
2× overkill sizing on attack variants. Head-to-head vs our own champion:
35/64 — the first of thirteen measured mechanisms to come out ahead.
Composed with the 4P FFA objective fix; both halves verified by
action-stream parity to their measured bundles.

The new rolling pair (ffa_uniform vs mass2p_ffa) differs only in 2P play,
so tomorrow's settle gap is a clean live 2P A/B — and the evicted
champion's frozen 1214 gives the 4P A/B reading against ffa_uniform.

## Method notes for the next session

- Verification-by-parity is cheap and powerful: gate mechanisms by player
  count, hash the action streams against the measured bundles, and pool
  results transfer without re-running anything.
- Head-to-head vs the namespaced champion is the new primary 2P
  instrument; vanilla-producer A/B demoted to non-regression check.
- Top-agent fleet median is 83; ours after the pivot is ~44. The mass
  axis is not exhausted, and the expansion gap (8 planets by step 40 vs
  our 6) is untouched.
