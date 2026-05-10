# 2026-05-10 — PI voice-dump: senior-engineer / game-theory perspectives on Orbit Wars heuristics

> Captured per CLAUDE Rule 35 (PI thoughts are append-only, permanent,
> never overwritten or archived).
>
> Structured analysis lives in `docs/strategies/heuristics-research.md`.
> This file is the verbatim source.

## What the PI asked for

Take a senior-software-engineer + games + game-theory perspective.
Problem-solve and research a set of rules / heuristics that could help
**any** strategy. Document — do not implement.

## Topics (PI's phrasing, lightly cleaned of fillers)

1. **Backwards reasoning — deterministic-win detection.**
   Are there scenarios where we already know we have won? Think
   backwards from the win condition. When is the game effectively
   decided so we can stop optimising offense?

2. **Planet prioritization — which planets to send ships to.**
   Heuristics to evaluate which targets to attack first.

3. **End-steps-ahead production maximization.**
   Looking N steps ahead, what is the action that maximises our ship
   production by the end of the game?

4. **Ship bundling for speed.**
   To send ships faster, bundle them. Planets close by could send all
   their ships to one nearby planet so the combined pile travels as one
   larger, faster fleet to the real target.

5. **Open brainstorm — which other heuristic ideas can there be?**

6. **Compete-relative vs absolute objective.**
   Not all agents should optimise for their own ships. One agent family
   might optimise own production. Another might think "I actually win
   if I'm better than the competitor" — for them, what matters is
   whether the opponent's production / capabilities are lower than
   ours, not whether ours are absolutely high.

## PI's framing

- Document these ideas as markdown so we can build upon them later.
- Provide multiple perspectives.
- Research only — no implementation.
- First, really try to understand what we're doing (the game, the
  current code) before generating heuristics.
