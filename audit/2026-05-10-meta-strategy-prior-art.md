# 2026-05-10 — meta-strategy + opponent-classification: prior-art survey

> Companion to `/root/.claude/plans/read-the-handover-next-imperative-whisper.md`
> Phase 1+. Lands at the start of Phase 1 so the design lineage is in-tree
> rather than only in the planning chat.

## Why this exists

The framework we are building has clear academic precedent:

1. Build a population of distinct heuristic strategies (a "zoo").
2. Capture short behavioural replays from self-play between zoo members.
3. Extract a small-dimensional fingerprint from each replay.
4. Train a classifier `fingerprint → strategy class`.
5. At game-time, classify the opponent and switch to a best-response
   strategy from a pre-computed table.

This survey grounds each piece in published work, names the load-bearing
risk for each piece, and points to one *recommended-first-read* per area.

## 1. Opponent modeling in multi-agent RL

- **Albrecht & Stone (Artificial Intelligence 258, 2018)**, *"Autonomous
  agents modelling other agents: a comprehensive survey and open
  problems."* Taxonomises agent-modelling methods (policy reconstruction,
  type-based, plan recognition, classification). "Type-based" — exactly
  our fingerprint-classifier-with-best-response design — is named as a
  mature subfield. Worth reading once for the vocabulary alone.
- **He, Boyd-Graber, Kwok & Daume III (ICML 2016)**, *"Opponent Modeling
  in Deep Reinforcement Learning" (DRON).* Two architectures map onto our
  design: DRON-Concat (learn an end-to-end opponent embedding fed
  alongside state) and DRON-MoE (per-opponent expert Q-net gated by
  inferred type). **Our design is closer to DRON-MoE**: discrete
  strategies as experts, classifier as the gate.
- **Foerster et al. (AAMAS 2018)**, *"Learning with Opponent-Learning
  Awareness" (LOLA).* Models the opponent's *learning step*. Useful only
  as a contrast: our opponents are static heuristic agents (or static
  RL policies once submitted), so the type-classification level is the
  right level of abstraction; LOLA's machinery would be overkill.

## 2. Empirical Game-Theoretic Analysis (EGTA) / PSRO

- **Lanctot, Zambaldi, Gruslys, Lazaridou, Tuyls, Perolat, Silver &
  Graepel (NeurIPS 2017)**, *"A Unified Game-Theoretic Approach to
  Multiagent Reinforcement Learning" (PSRO).* Defines the population
  best-response loop. Introduces **joint policy correlation** —
  quantifies the failure mode our best-response table is most exposed
  to: a BR overfit to a specific zoo subpopulation that fails to
  generalise.
- **Bighashdel, Wang, McAleer, Savani & Oliehoek (IJCAI 2024)**,
  *"Policy Space Response Oracles: A Survey."* Catalogues newer
  variants: Diverse PSRO, BD&RD-PSRO, JPSRO (correlated-equilibrium
  meta-solver), Conflux-PSRO. Future reference if/when we want to grow
  the zoo by best-response oracles instead of hand-coded heuristics.
- **OpenSpiel** (Lanctot et al., DeepMind tech report 2019). Reference
  PSRO/EGTA implementation; tooling we can lift if we need a meta-game
  Nash solver for `BR_robust` ranking.

## 3. Strategy embedding / low-dim policy manifold

- **Grover, Al-Shedivat, Gupta, Burda & Edwards (ICML 2018)**,
  *"Learning Policy Representations in Multiagent Systems."* **The
  closest published precedent for our exact pipeline:** short
  interaction trajectory → low-dim agent embedding via discriminative
  / triplet loss → few-shot identification + downstream best-response.
  **Recommended read #1.**
- **Vinyals et al. (Nature 2019)**, *AlphaStar.* The League result is
  the empirical evidence that a real RTS strategy space is **not**
  spanned by a single mode and needs a finite multi-modal population.
  Implication for our framework: PCA on a "smooth low-dim subspace" is
  the wrong primary diagnostic. Strategies cluster in **discrete
  basins**; use kNN / random forest / learned embedding, not linear
  PCA. The "small-dim manifold" hypothesis is fine; the "*linear*
  small-dim manifold" hypothesis is the one to falsify first.
- **Hu, Lerer, Peysakhovich & Foerster (ICML 2020)**, *"'Other-Play'
  for Zero-Shot Coordination."* Quotients self-play policies by
  environment symmetries. Relevant warning: naive self-play picks an
  arbitrary point on a high-dim equivalence class, so our fingerprint
  must be *invariant* to such symmetries (in Orbit Wars: 4-fold
  rotational mirror around the sun) or the classifier learns label
  noise.

## 4. Online opponent classification from short observation

- **Brown & Sandholm (Science 2019)**, *"Superhuman AI for multiplayer
  poker (Pluribus)."* Explicit design quote: Pluribus plays an
  approximate-equilibrium fixed strategy and "does not adapt to the
  observed tendencies of the opponents." So Pluribus is *evidence
  against* online exploitation being the right answer at the very top
  — but the paper notes exploitation needs sample sizes humans can't
  supply, which is the bottleneck our short fingerprint is designed to
  solve.
- **Ganzfried & Sandholm (2015 / 2016)**, *"Bayesian Opponent
  Exploitation in Imperfect-Information Games."* Closer to our design:
  prior over opponent types + Bayesian update from observed actions +
  *safe* best-response (capped exploitation that bounds the regret
  against an adversary who is faking type). The "safe" framing is what
  our `ABSTAIN_BELOW` confidence threshold + epsilon-randomised switch
  to a robust default operationalises.
- **Southey et al. (UAI 2005)**, *"Bayes' Bluff: Opponent Modelling in
  Poker."* The original "infer type from short play, then exploit"
  template. Small scale, but the math is what we'd implement if we
  preferred a Bayesian classifier over a random forest.

## 5. Practical failure modes (what bites this design)

| Failure                                          | Source signal                                 | Mitigation in our plan                                                                |
| ------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------- |
| Out-of-distribution opponents (Roman, RL bots)   | Pluribus warning; DRON Sec. 6                 | Mahalanobis-distance / energy-score abstain → fall back to ROI; spoofers in zoo.      |
| Joint-policy correlation overfit                 | Lanctot 2017                                  | Leave-one-family-out CV; `BR_robust = argmax_a min_b WR(a,b)` column.                 |
| Manifold non-linearity / PCA fails               | AlphaStar League                              | kNN / random-forest / Grover-style embedding as primary; PCA only as 2D viz.          |
| K-turn classifier latency > game length          | Domain constraint (500-step games)            | Plot accuracy vs K early; gate on K ≤ 100 (20% of game length).                       |
| Meta-counter (opponent spoofs fingerprint)       | LOLA-style adversarial counter                | ε-randomise the switch; spoofer agents in training zoo; monitor live μ for dips.      |
| Bundle size / classifier serialisation bugs      | Rules §2.12 (no ingress at eval)              | Inline classifier weights as base64; mirror v1 parity-test gate at Phase 4.           |

## Recommended-first-reads, by priority

1. **Grover et al. 2018** — the closest precedent; replicate the
   protocol on our zoo before believing or disbelieving the manifold
   hypothesis.
2. **Albrecht & Stone 2018** — the survey; orients vocabulary and
   places the design in the broader literature.
3. **DRON 2016** — when we need the gate-and-experts implementation
   detail.
4. **Pluribus 2019 §"Comparison with prior work"** — short read; sets
   the priors on when *not* to exploit.

## What we are NOT doing (vs the prior art)

- **No iterative best-response oracle** (PSRO's "compute new best
  response, add to population, repeat"). Our zoo is hand-coded; the
  oracle step is deferred to Phase 5 (RL fallback).
- **No hidden-information abstraction** (Pluribus / Brown-Sandholm).
  Orbit Wars is full-information at every turn; we don't need card
  abstractions.
- **No theoretically-safe-exploitation** bound (Ganzfried-Sandholm).
  Our framework just abstains under uncertainty; the formal regret
  bound is left as a future-work item.
- **No learned end-to-end embedding in Phase 1.** Hand-designed
  features first; if Phase 1 fails, switch to Grover's discriminative
  embedding in Phase 2 — but only after the cheap baseline has had a
  fair test.
