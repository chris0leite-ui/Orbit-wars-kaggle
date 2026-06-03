# Play-against-the-board — research synthesis (2026-06-03)

> Built on branch `claude/champion-strategy-rules-00JzI` via a 5-angle
> deep-research fan-out (Planet Wars 2010 / Halite+Lux / combat+territory
> math / decision+optimization math / per-turn compute budgeting).
> Purpose: ground the PI-committed "play against the board, not the
> player" design — an opponent-agnostic agent whose value function is the
> production-time integral, with worst-case-reach as the only opponent
> term. Each claim carries its source; thin spots are flagged.

## 0. Headline

The research **strongly validates** the opponent-agnostic direction and,
more usefully, **sharpens how**. The single most relevant data point: the
winner of the direct ancestor game (Google AI Challenge 2010 Planet Wars)
was opponent-agnostic and shallow, and its author **tried deeper search
and opponent modeling and both underperformed** the robust evaluation.
The genre base rate (Halite I/II/III, Lux S2) is heuristic value-of-
position + global assignment + influence maps; RL self-play won this genre
exactly once (Lux S1) and only with heavy stabilization.

The key refinement to our prior (falsified) reach-frontier work: the
proven winner used worst-case hold as a **binary GATE on what counts as
"mine"** (don't credit a planet unless I'd survive the opponent's *full*
attack), not as a continuous hold-time **SCORE**. Our closed reach-frontier
track scored captures by `ρ_opp − ρ_me`; the winner gated them. Gate, not
score.

---

## 1. Prior art — what won, and whether it modeled the opponent

### Google AI Challenge 2010 "Planet Wars" (direct ancestor)

- **Winner bocsimacko (Gábor Melis) was opponent-agnostic.** For most of
  the contest it was a **1-ply search with no opponent modeling**
  ("for the longest time, it was a 1-ply search. Opponent moves were never
  considered"). Source:
  [quotenil.com post-mortem](http://quotenil.com/Planet-Wars-Post-Mortem.html).
- **Its value function = the "full-attack future."** State score = sum of
  per-planet scores; a planet is credited to you **only if you'd hold it
  when the opponent sends *all* ships at it**. The theorem: if for every
  planet of player 1, player 2 cannot take it even sending everything, then
  player 2 cannot take *any* of player 1's planets under any attack pattern.
  This is the formal "don't attack/credit a planet you can't hold." Source:
  [quotenil.com](http://quotenil.com/Planet-Wars-Post-Mortem.html).
- **Small positional-pressure term:** a slight penalty per simulated turn
  per enemy ship → biases toward sitting near the enemy and threatening
  multiple planets at once. Source: quotenil.com.
- **Deeper search underperformed.** Alpha-beta was added late and only for
  the opening (4-ply until the 3rd planet captured), and even that was
  "too greedy"; deeper alpha-beta "fell way short of expectations" live.
  Evaluation robustness > depth. Source: quotenil.com.
- **Move generation = greedy step-combination, not full enumeration.**
  Score candidate steps, sort descending, greedily combine from the top —
  keeps branching tractable inside the 1 s/turn limit. Source: quotenil.com.
- **Expansion gated by "safe-to-take":** dynamic horizon (constant 30, later
  extended to the earliest break-even turns of safe neutrals); a neutral is
  safe only if from investment to break-even **no friendly planet could be
  lost in a full-attack future**. Source: quotenil.com.
- **Surplus / sniping-aware:** "surplus" = ships sendable without ever
  causing the source to be lost later given fleets already in flight;
  arrival constraints make the eval sniping-aware. Source: quotenil.com.
- **zvold's own A/B confirms it:** his **non-modeling heuristic** bot
  (≈#350, knapsack expand + snipe + reacquire, ~50–100 ms/turn) outranked
  his **2-ply minimax-with-pessimistic-opponent-model** bot (≈#700). Source:
  [zvold.blogspot.com](http://zvold.blogspot.com/2010/12/two-bots-for-planet-wars-ai-challenge.html).
- **oddshrimp** (strong search bot) used iterative-deepening alpha-beta to
  8-ply, "indirect wealth" (planets worth more next to high-growth
  neighbors), a movement-distance penalty, and explicit **safe/contested/
  unsafe** neutral classification with anti-snipe force sizing. Source:
  [satirist.org](https://satirist.org/ai/planetwars/).
- Cross-bot lesson: "credit/attack only what you can hold" via reaction
  time; balanced aggression beat opponent-prediction. Source:
  [satirist.org strategy](http://satirist.org/ai/planetwars/strategy.html).

### Halite I/II/III and Lux AI S1/S2

- **Halite III winner teccles: pure heuristics, no ML.** Value-of-cell =
  closed-form **production-per-time** (`gain/(travelTurns+miningTurns)`),
  Dijkstra distances, greedy ship→target assignment, ROI-gated dropoffs.
  Source: [teccles repo](https://github.com/teccles-halite/halite3-bot).
- **Halite III top-tier also used Hungarian assignment + influence/density
  maps** (dzou): ships→cells as max-weight bipartite matching; summed
  "halite density" grids. Source:
  [dzou repo](https://github.com/dzou/halite3).
- **Lux S1 winner Toad Brigade: deep RL self-play** — but only with a
  **frozen teacher + KL loss** to kill strategic cycles and **20M steps of
  reward shaping**; naive sparse-reward self-play did not work. IMPALA+UPGO+
  TD(λ), ResNet ~20M params, per-unit action masking. Source:
  [Toad Brigade repo](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/README.md).
- **Lux S2 winner ry-andy + top-5 mostly rule-based** (one RL); larger
  action space made RL sample-inefficiency bite. Source:
  [ry-andy repo](https://github.com/ryandy/Lux-S2-public) (exact internals
  not extracted — MEDIUM confidence on composition).
- **Base rate:** heuristic value-of-position + assignment + influence wins
  this genre; RL won once and only with heavy scaffolding. Matches repo
  Rule 6.

---

## 2. The math, mapped to our pieces

### Combat sizing — Lanchester LINEAR law (this game)

- Orbit Wars combat is "larger force minus smaller force survives" → the
  **linear law** (loss rate ∝ product of forces; with equal lethality,
  survivors = `A − B`). Source:
  [Lanchester's laws — Wikipedia](https://en.wikipedia.org/wiki/Lanchester%27s_laws),
  [doolanshire](https://www.doolanshire.net/2017/08/30/lanchesters-laws-of-combat/).
- **Minimum to capture a garrison G:** `A > G` → send `G + 1`.
- **Capture AND hold against a worst-case follow-on force F within the
  horizon:** `A ≥ G + F + 1`. **This is the defensive-sufficiency formula,
  derived** — and it is exactly bocsimacko's full-attack-future gate
  expressed as a ship count.
- **No square-law concentration bonus.** Under the linear law, strength is
  linear in N, so massing one mega-fleet on a target is *no stronger in
  combat* than splitting the surplus across several capturable garrisons.
  The lever is target **allocation + timing**, not raw mass. Source:
  Wikipedia (regime distinction).
- **Game-specific nuance (from the Orbit Wars spec, not the source):** fleet
  *speed rises with size* (`speed = 1 + (maxSpeed−1)·(ln k/ln 1000)^1.5`).
  So concentration still helps — via **faster arrival → win the reach-race
  → longer hold**, not via combat. Size fleets for reach/timing.

### Value — production-time integral = finite-horizon return

- Owning a producing asset for the rest of the game = the finite-horizon
  undiscounted return of a constant reward stream: `production × (T −
  capture_turn)`. Earlier capture ⇒ more terms ⇒ strictly larger — the
  "early compounds" statement. Source:
  [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).
- **Caveat (load-bearing):** the `(T − t)` form assumes you hold to the end.
  If recapturable, the integral truncates at the loss turn → use
  `production × hold_time`, where `hold_time` is bounded by the worst-case
  reach of the nearest threat. This is the reach-frontier value *and* the
  full-attack-future gate, unified.

### Margin-agnostic → maximize expected gap, but add variance when behind

- Maximizing `E[my − opp]` is a sound monotone proxy for `P(finish first)`
  **under a location-shift assumption** (shifting the gap's mean up gives
  first-order stochastic dominance, so `P(gap>0)` rises). Source:
  [Stochastic dominance — Wikipedia](https://en.wikipedia.org/wiki/Stochastic_dominance).
- **But when two lines have equal `E[gap]` and different variance, P(win) is
  NOT the mean.** Trailing → prefer **higher variance**; ahead/even →
  risk-neutral `E[gap]`. So the value head should tilt toward variance when
  behind. Actionable, novel for us.

### Robust decision — maximin is too passive; use minimax-regret

- Pure worst-case (maximin) is documented as **over-conservative / too
  passive** — it plans against an adversary that may never materialize.
  **Minimax-regret** (minimize worst-case gap to the best-possible-vs-that-
  scenario) is the standard **less-conservative** alternative. Source:
  [Minimax-regret robust planning (arXiv 2012.04626)](https://arxiv.org/abs/2012.04626),
  [adjustable regret (arXiv 2105.05536)](https://arxiv.org/html/2105.05536).
- **This is the theoretical reason `v7_minimax` and the reactive-opponent
  rollouts went passive and lost.** Two fixes, both already in our design:
  (a) the **asymmetric posture** — worst-case on *defense only*, optimistic
  on offense — and (b) aggregate the Rule-43 panel by **per-opponent regret**,
  not pooled worst case.

### Target selection — submodular greedy, with the teamwork caveat

- Monotone-submodular value under a cardinality cap: greedy gets `(1−1/e)≈
  0.63` of optimum (Nemhauser-Wolsey-Fisher), and **lazy-greedy / CELF**
  gives the same with ~700× fewer evaluations. Sources:
  [Krause-Golovin survey](https://viterbi-web.usc.edu/~shanghua/teaching/Fall2023-670/krause12survey.pdf),
  [CELF (Leskovec)](https://www.cs.cmu.edu/~jure/pubs/detect-kdd07.pdf).
- **Breaks under (a) complementarity** — fleets that only win together are
  *super*-modular (the exact thing that made our joint-coordination greedy
  under-commit) → seed coalitions as **atomic candidates**; **(b) non-
  monotonicity** — a launch that over-extends *reduces* value → voids the
  guarantee. Test the marginal-gain inequality empirically before trusting
  the bound.
- **Assignment (Hungarian, O(n³)) / min-cost-flow** for routing sources→
  targets is exact **only when value is separable across arcs**; use it as a
  fast inner solver *after* the coalition/subset decision, never as the
  top-level chooser. Source:
  [Assignment problem — Wikipedia](https://en.wikipedia.org/wiki/Assignment_problem).

### Territory by reach (opponent-agnostic frontier)

- A speed-weighted **reach-time Voronoi** (multiplicatively-weighted; cells
  bounded by Apollonius arcs when speeds differ) partitions planets by who
  arrives first. Cheap grid approximation = an **influence map**: friendly
  positive, enemy negative, summed per cell; the **contested frontier is the
  zero-crossing**. Sources:
  [Weighted Voronoi — Wikipedia](https://en.wikipedia.org/wiki/Weighted_Voronoi_diagram),
  [influence maps](https://grant.tuxinator.net/post/influence-maps-part-1/).
- Use: expansion = take planets on/just across my side of the frontier;
  defense = reinforce my planets near the zero-crossing; offense = enemy
  planets with weak influence.

---

## 3. Compute — how to spend ~1 s/turn with an exact simulator

The literature converges on one architecture:

1. **Anytime iterative-deepening shell.** Pass 1 = cheap heuristic produces
   a guaranteed-legal launch set immediately; later passes overwrite only if
   better. Soft clock between passes + hard clock every N sim calls. Spend
   more passes when the chosen set keeps flipping (unclear position).
   Source: [chessprogramming Time Management](https://www.chessprogramming.org/Time_Management).
2. **Cheap heuristic scores ALL candidates → keep top-K.** Cheapest big win.
   **Seed coalition atoms BEFORE the top-K cut** or they never survive
   (Rule 41). Source:
   [heuristic move pruning in MCTS](https://www.researchgate.net/publication/286604336).
3. **Truncated rollout + closed-form tail = "simulate until it settles,
   then close the integral analytically."** This is playout-truncation with
   a bootstrapped terminal value (AlphaZero/MuZero eliminate rollouts and
   bootstrap a depth-limited leaf). `predict_fleet_fate` is the natural
   settle-detector; the reach-frontier production share is the tail. Sources:
   [MCTS-minimax hybrids (JAIR)](https://www.jair.org/index.php/jair/article/download/11208/26419/20772),
   [MuZero](https://xlnwel.github.io/blog/reinforcement%20learning/MuZero/).
   **Verify the analytic tail matches a full-sim tail on sampled settled
   states before trusting it** (Rule 47-style).
4. **Shallow-eval-many / deep-confirm-few.** Heuristic-score all → shallow
   truncated rollout on top-K → exact deep-confirm top 2–3. With an *exact*
   simulator the deep-confirm marginal gains are exact. Source:
   [implicit minimax backups (arXiv 1406.0486)](https://arxiv.org/pdf/1406.0486).
   Caveat (Catch-the-Lion): a static heuristic sometimes beats expensive
   search — confirm the deep tier adds μ over the cheap heuristic on the
   panel, don't assume.
5. **Three time knobs:** beam width / top-K / truncation depth — the soft
   clock turns these down when the turn is expensive.
6. **numpy-batched rollouts** (leading batch axis = the K candidate sets)
   make the deep-confirm tier affordable; caveat: branch-heavy combat
   resolution may not vectorize cleanly — measure vs a tight scalar loop.
   Source: [Pgx (arXiv 2303.17503)](https://arxiv.org/pdf/2303.17503).

---

## 4. The agent this implies (one simple idea)

> **Grab the production you can hold.** Each turn, value every planet by
> `production × hold_time`, where `hold_time` runs until the nearest
> worst-case threat could reach it. Greedily take the best planets I can
> both **capture** (`garrison + 1`) and **hold** (`garrison + worst-case
> follow-on + 1`). Never drain a planet below its own worst-case-hold
> floor. Re-solve from scratch every turn.

This is bocsimacko's full-attack-future **gate**, on top of the production-
integral **value**, with per-turn re-solve for adaptivity — the recipe that
won the ancestor game and matches the Halite/Lux base rate. Every quality
the PI listed (cheap recapture, never-sacrifice, early committed expansion,
4P attack-weakest) emerges from this with no special-case rule.

Compute path: anytime shell → cheap heuristic ranks all candidates →
top-K shallow truncated rollout (settle + closed-form tail) → exact
deep-confirm of the best 2–3.

---

## 5. Risks / where evidence is thin

- **This is adjacent to a CLOSED track** (reach-frontier closed-form,
  falsified 0/20, 0/32, 4/32). The research says the difference that
  matters is **gate-not-score + exact-sim settle tail + asymmetric posture**.
  Frame any build as the falsifiable question "does gate+exact-tail beat the
  old symmetric closed-form?" — not a blanket reopen (Rule 44).
- **Local-vs-ladder calibration** remains the dominant historical failure
  here; curriculum wins are a correctness signal only (PI-confirmed). Real
  gate = Rule 43 multi-opp panel (aggregated by *regret*) + Rule 45 n≥32 +
  champion h2h.
- **The closed-form tail being a faithful substitute for full simulation**
  is our own modeling bet (the within-band/between-band open question in
  `evaluation-metrics.md`) — verify before trusting.
- **Lux S2 #1 internals** and a few secondary rankings are MEDIUM confidence
  (JS-rendered writeups not fully extracted).
- **numpy vectorization of branch-heavy combat** may not pay off — measure.

## 6. Source list

Planet Wars: quotenil.com post-mortem; github.com/melisgl/planet-wars;
satirist.org/ai/planetwars (+ strategy, playing-styles); zvold.blogspot.com.
Halite/Lux: teccles-halite/halite3-bot; dzou/halite3;
IsaiahPressman/Kaggle_Lux_AI_2021; ryandy/Lux-S2-public.
Math: Wikipedia (Lanchester's laws, Weighted Voronoi, Stochastic dominance,
Assignment problem, Hungarian algorithm); OpenAI Spinning Up; Krause-Golovin
submodular survey; Leskovec CELF; arXiv 2012.04626 / 2105.05536 (regret).
Compute: chessprogramming.org (Time Management, Iterative Deepening); JAIR
MCTS-minimax hybrids; MuZero notes; arXiv 1406.0486; arXiv 2303.17503 (Pgx);
influence-map articles (grant.tuxinator.net, andrewshunt.com, Game AI Pro).
