# Head-to-head: v7_0_drop_one vs v7_minimax (2026-05-12)

> Direct local 2P A/B between the two v7 candidates. v7_minimax
> is the agent the PI already submitted (#52568317, converged μ=1063);
> v7_0_drop_one is the bundle this session built on
> `claude/game-ai-lookahead-3ucqH`. The PI wants to know which is better.

## Result

```
24 games, both sides, 12 unique seeds
v7_0_drop_one : 19 wins (79.2%)
v7_minimax    :  5 wins
draws         :  0
Wilson 95% lo for v7_0: 59.5%  →  PASS (gate ≥ 55%)
elapsed       : 270s on 4 workers
artifact      : audit/tournaments/20260512T134627Z.json
```

p95 turn ms:
- v7_0_drop_one: 746 (P0) / 751 (P1)
- v7_minimax:     35 (P0) /  33 (P1)

v7_minimax's p95 is ~20× faster — its K=3 + adaptive K=2 keeps
rollouts cheap, while v7_0 burns more budget on K=10 rollouts.
Both stay under the 800 ms safety gate locally.

## Per-game pattern

v7_0 was P0 in seeds 0–11; v7_minimax was P0 in seeds 0–11 (the
mirror). Final 5 games (v7_minimax as P0):

```
seed=5  v7_minimax wins (+2486 ships)
seed=9  v7_minimax wins (+4196)
seed=8  v7_0 wins      (-6576 = our +6576)
seed=11 v7_0 wins      (-2276)
seed=10 v7_0 wins      (-4407)
```

When v7_0 wins it wins decisively (mean ship delta ≈ −4400 = our
+4400). When v7_minimax wins it wins by similar margins. The split
is 5 v7_minimax wins / 19 v7_0 wins. No correlation between starting
position and winner — both seats are competitive for both agents.

## Why v7_0 wins (mechanistic, from the structural-weaknesses audit)

The audit `audit/2026-05-12-v7-minimax-structural-weaknesses.md`
lists 10 structural weaknesses of v7_minimax. v7_0 fixes 4 of them:

| Weakness (audit §2) | v7_minimax | v7_0_drop_one |
|---|---|---|
| **c. K=3 too shallow** | K=3 (adaptive 2) | **K=10** (3.3× deeper) |
| **e. rollout policy is v3_snipe (weak)** | v3_snipe (aggressive=False) | **v3.5.1 (aggressive=True)** |
| **g. drop-smallest only** | Single dropped launch | **Drop-each-launch** (N+1 candidates) |
| **h. 4P falls back to v3_snipe** | Falls back to v3_snipe | **Falls back to v3.5.1** |

v7_minimax has 1 advantage v7_0 lacks:
- σ-equivariance patches (deterministic tie-break + sym_hypot +
  score rounding) and `score_joint_action_symmetric` (cancels env's
  P1 tie-break bias). The audit notes σ-equiv alone scored 1041.4
  live; the maximin overlay added only +22 μ over that.

Combining these into one story: **the lift comes from the depth +
rollout policy, not from real maximin over an opp class.** v7_0's
K=10 + v3.5.1-mirror rollout discriminates candidates more reliably
than v7_minimax's K=3 + v3_snipe-mirror, even though v7_minimax has
a "real" 2×2 minimax structure.

## Predicted live μ for v7_0_drop_one

v7_minimax converged at μ=1063. v7_0 beats it 79.2% locally
(Wilson lo 59.5%).

Rough TrueSkill math: at σ≈25 (post-many-games), 79.2% winrate
corresponds to a μ gap of about +20-35. So:

- **Central estimate: v7_0 lands at μ ≈ 1085-1100.**
- **Best case: ≈ 1110** (if the local 79.2% holds against the
  full ladder distribution).
- **Worst case: ≈ 1060** (parity with v7_minimax if the local
  result was variance-favorable).

For context: top-10 cliff μ = 1447.6. We're still ~+350 μ short of
top-10 in any scenario.

## Caveats

1. **n=24 is the minimum sample for the Wilson 55% gate.** Wilson
   lo 59.5% is robust to a small Wilson regression at n=64, but the
   point estimate (79.2%) could tighten to 70-75% at scale.
2. **v3.5.1 is still PENDING (#52565976).** We don't know its μ.
   v7_0 is built on v3.5.1, so its actual ceiling depends on
   v3.5.1's μ + a translation factor. The h2h vs v7_minimax tells
   us v7_0 outperforms v7_minimax — but we still don't know v3.5.1
   alone's μ in case the v3.5.1 pipeline lifts more than the
   rollout-veto overlay does on top of it.
3. **p95 746 ms on v7_0 is tight.** Live containers may run
   slower. The 700 ms watchdog inside `lib.v7_search.choose()`
   guarantees we never breach 1000 ms actTimeout, but worst-case
   the watchdog truncates the rollout early and v7_0 falls back to
   v3.5.1 incumbent on some turns.
4. **Both agents use Sim<K> with v3-family rollouts.** Both are
   blind to the same structural weaknesses: shallow horizon (still
   3% of a 500-step game even at K=10), aggression-biased scoring
   head, no opponent intent modeling, 4P fallback to a non-search
   agent. The h2h tells us which is the better v7-family agent —
   not whether a v8 different paradigm would win.

## Recommendation for the PI

**Submit v7_0_drop_one.**

Reasoning:
- We have strong local evidence v7_0 > v7_minimax (79.2% h2h,
  Wilson lo 59.5%).
- v7_minimax has a known live μ (1063). v7_0 is predicted to land
  at 1080-1100 — a +17 to +37 μ ladder gain.
- Rolling-last-2 after the push becomes
  `[v7_minimax (1063), v7_0_drop_one (PENDING)]`. **Worst case:
  v7_0 disappoints to 1060, and v7_minimax stays as the floor —
  no regression.** Best case: +20-40 μ lift.
- The eviction is **v3.5.1 (PENDING)**, not a known peak.
  v3.5.1's predicted μ was 1090-1100 — possibly higher than
  v7_minimax. We DO lose the chance to measure v3.5.1 directly.
  Trade-off acceptable because:
  - v7_0 is built on v3.5.1 — if v3.5.1's pipeline carries lift,
    v7_0 inherits it AND adds the rollout-veto overlay on top.
  - We already have v7_minimax (1063) and v3_snipe (1055.5) as
    reference points; v3.5.1's measurement is not load-bearing
    for any next decision.
- Slot usage: 1 of remaining 5 daily slots. Plenty of room.

## If the PI prefers caution

Alternatives that preserve v3.5.1's pending measurement:
- **Wait 12-24 h for v3.5.1 μ to converge**, then decide between
  v7_0 (build on the best known base) and a v8 redesign.
- **Local n=64 confirmation of v7_0 vs v7_minimax** before
  submitting — tightens the Wilson estimate. ~12 min CPU.

The h2h result is strong enough that submitting NOW is defensible.
Per Rule 1 the actual `kaggle competitions submit` requires the
PI's explicit single-shot authorisation.
