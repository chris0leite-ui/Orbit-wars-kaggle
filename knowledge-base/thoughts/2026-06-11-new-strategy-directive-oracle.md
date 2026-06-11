# 2026-06-11 — PI directive: an entirely new strategy (the oracle track)

PI (verbatim intent): "Now with everything you have learned, research and
create an entirely new strategy on your own, that should be able to beat
all our agents and perform well on the leaderboard. You may use machine
learning and reinforcement learning. Inform yourself through research
first. We have kaggle GPU available. You may as well go through a no ML
path again. Think for yourself and work autonomously, but create
something new."

## What the AI chose and why

Learning the top ladder's decisions directly (behavior cloning from
scraped replays of 1500-1750-rated games) on top of our exact-physics
ledger, instead of (a) more hand-tuned pricing — the documented recurring
failure across 30+ sessions — or (b) from-scratch self-play RL — highest
ceiling but a research project in a 12-day window; the survey of past
Kaggle sim comps says well-engineered rules/search agents usually win,
with IL-from-top-replays the proven ML recipe (Lux S1).

The decisive enabler: Kaggle's episode service lets anyone walk the match
graph (submission -> episodes -> opponents' submissions, with ratings) and
download any replay. 9 hops from our submissions reached the #1 team.
~66.9k episodes catalogued, ~1.1k top replays on disk in an afternoon.

## The day's hard lessons (each cost hours; each is general)

1. **Outcome models cannot rank actions.** AUC 0.999 for "who wins" and
   the model still preferred null over the expert's actual move 75% of
   the time. If you want a chooser, learn the choice.
2. **Calibration beats thresholds only when the conditioning is right.**
   Per-pair fire probabilities collapsed on cold boards because the mass
   rode on fleets-already-in-flight features; the per-state initiation
   head fixed what no threshold tuning could.
3. **The impossible number is the best debugger.** Fire rate of exactly
   0.000 over 40,665 cold states is not a small effect — it is a wiring
   proof. The replay action at index t belongs to obs[t-1]; we paired it
   with obs[t]. One index, three artifacts (aftermath learning, fake
   far-launch gap, inflated size labels), one fix.
4. The dead-agent failure mode from the 06-12 audits bit AGAIN (loader
   exec without __file__). The liveness columns in the new battery
   harness caught it in game one.

## Standing thought

The expert-mimicry layer makes the agent only as good as the population's
best habits. The exact engine (sizes snapped to capture floors at true
arrival ticks, verified flights) should make it slightly better than its
teachers in execution; whether that nets out above 1500 is the open
empirical question for tomorrow.
