# 2026-05-11 — cannot-lose final finding: v3 IS the floor

## TL;DR

After 7 iterations (v4_mirror Tier 0-2, v4_hybrid, v4_endgame, v5_psp, v6_steady)
trying to ADD a cannot-lose layer on top of v3_snipe, the empirical
evidence is decisive:

**The cannot-lose strategy at our μ-bracket is v3_snipe itself, with
no overlays.** Every overlay we tried (mirror, end-scenarios, Sim<K>
filter) BREAKS the natural v3-vs-v3 draw lock and degrades performance.

This is the answer to PI's question "what are we missing?" — we were
missing the realization that the cannot-lose property is INTRINSIC to
being at-or-near Nash equilibrium for a finite symmetric zero-sum
game, not a property that can be added structurally.

## The decisive empirical evidence

`agents/v3_snipe/main.py` vs `agents/v3_snipe/main.py` (loaded as two
separate module instances), 16 seeds × 500 steps each:

```
TOTAL v3-vs-v3: draws=13/16, P0wins=2, P1wins=1
```

13/16 = **81% draws**. The 3 non-draws are all eliminations (one
player wiped out before step 500: seeds 10, 13, 15). When considered
across both seat assignments (we'd play 32 games against v3, half
as P0, half as P1):

```
Expected W/D for "agent that plays v3" vs v3:
  - 13 mutual draws × 2 seats = 26 draw outcomes
  - 3 decided games × 2 seats = 3 wins + 3 losses
  - Total: 26 + 3 W = 29/32 ≈ 90.6% W/D
```

**That's the cannot-lose floor.** Just play v3. No overlays.

## What all the overlays broke, and how

| Iteration | Approach | W/D vs v3 (16 games) | Failure mode |
|---|---|---|---|
| Tier 0 pure mirror | reactive 180° rotation | 0% | 1-turn lag + cascade after first lost planet |
| Tier 1 + lag-comp | + arithmetic lag bump | 0% | cascade unresolved |
| Tier 2 + sun-veto | + crash-bound veto | 0% | v3 doesn't make those errors |
| Hybrid v3+mirror | v3 + mirror residual | 12.5% | mirror residuals fire at v3-defended targets |
| v4_endgame | v3 + W1/W4/D1 | 37.5% | W1 fires on ANY production lead; we coast, opp captures, we lose |
| v5_psp | v4_endgame + Sim<K> | 12.5% | rollout policy-mismatch picks weaker ROI over v3 |
| v6_steady | v3 + W1_SAFE | 25% | even W1_SAFE (rarely-firing) breaks the symmetric lock |
| **(pure v3, control)** | **no overlay** | **~90%** | **none** |

Every overlay creates ASYMMETRIC behavior: the overlay-side
occasionally diverges from v3's decisions (returns [], or picks an
alternative candidate, or skips a mirror) while v3-side keeps playing
normally. This single divergence cascades through 500 turns of combat
and one side ends up ahead by a winning margin.

## Why the cannot-lose theorem holds and yet overlays fail

The symmetric-game value theorem says: a finite symmetric 2P zero-sum
game has a Nash equilibrium with value 0. By symmetry, the strategy
that achieves it is symmetric (both players use it). The strategy
produces draws in self-play and expected payoff ≥ 0 against any
opponent.

v3_snipe approximates this Nash for its strategy-class (heuristic
ROI + missions + same-turn ledger). The 81% empirical draw rate is
how close v3 is to true Nash within its class. The 19% non-draws are
the gap to true Nash — combat ties, orbital RNG, etc.

Our overlays tried to ADD cannot-lose structurally on top of v3. But
v3 ALREADY has the property within its class. The overlays only added
asymmetry — they took us OUT of the Nash basin.

**Analogy**: it's like trying to make a stable bicycle "more stable"
by attaching unbalanced weights to one side. The bike is already
balanced; the weights only tip it over.

## To rise ABOVE the cannot-lose floor at our μ-bracket

Cannot-lose at level X = play near-Nash at level X. To climb to
level X+1, build a strictly stronger near-Nash agent. Two paths:

1. **Heuristic improvements** (1-3 days each, additive):
   - **Recapture missions** (Roman's playbook, HANDOVER §3). When
     our planet flips to enemy, score "can we retake before they
     fortify?" Adds a mission class to v3. Est +50μ.
   - **Gang-up timing**. Coordinate multi-source attacks on a single
     target with synchronized arrival. Adds another mission class.
     Est +30-50μ.
   - **Sibling-strategy Sim<K>** (PSP_v2 with v3 as rollout policy,
     K=5). Sim<K=5> with v3 is ~600ms — feasible. Resolves the
     policy-mismatch problem from v5_psp. May add lift if Sim<K> can
     identify v3's near-tied target choices.

2. **Self-play RL** (1-3 weeks, transformative):
   - PPO or A3C with v3 as warm-start
   - Train against population pool {v3, v3+recap, v3+gang, ...}
   - Converges to true NE in the limit
   - Only proven path to STRICTLY-CANNOT-LOSE against any opponent
   - Time-intensive; out of budget unless prioritized

The "cannot-lose" property comes free with each: a strictly-stronger
v3 in self-play has a NEW draw lock at a HIGHER μ-level.

## Critical realization for next session

The temptation will be to keep adding "safety overlays" to v3. Resist
it. Every overlay has been counterproductive. The right work is:

  ✗ Add cannot-lose overlay
  ✓ Build stronger v3 base
  ✓ Train via self-play RL

For the next iteration, the recommended attack vector is **recapture
missions** (HANDOVER #3). It's a known piece of Roman's portfolio,
estimated +50μ, ~2-3 day build. Adds a mission class; doesn't break
v3's structure.

## State of the branch

`claude/game-theory-strategy-analysis-0oH4N`, 11 commits:
- bcff311 Iter 0 mirror pure
- 1de635e Iter 1 lag-comp
- 79439aa Iter 2 + sun-veto
- 4f01b16 Iter 3 hybrid
- f4999f6 Iter 4 v4_endgame
- 1d844e6 tournament comparison
- 3f0df28 Iter 5 v5_psp wip
- bda6ff8 v5_psp tuning
- 6a66f0a v5_psp falsification audit
- fc024b6 Iter 6 v6_steady (still breaks lock)
- (this audit)

**v3_snipe** is — and was already — the closest practical
cannot-lose strategy in our reach. Submitted as #52544634. No further
overlay work needed; pursue stronger base for ceiling raises.
