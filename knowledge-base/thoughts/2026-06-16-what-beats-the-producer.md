# 2026-06-16 — What beats the producer? (and is the banking fixable?)

PI question: "Which agent would you build that *definitely* beats the producer?"
Then, after the banking/idle replay observation: "What is the fundamental fix?"

Answer, after a deep investigation: **nothing definitely beats it (~67% ceiling),
and the banking is NOT fixable with production value — that's a trap.** Every one
of my hypotheses was refuted by the data.

## Method (trustworthy this time)

All vs the **producer** (= iso_game `static` config = bare engine, action-identical
to `agents/producer`), env-**isolated** + single-thread **deterministic**
(scripts/iso_game.py), 6 seeds × 2 seats = 12 games per config, 2P (+ some 4P).
The two bugs that made earlier numbers lie (env-flag contamination, parallel-torch
FP nondeterminism) are fixed — see 2026-06-16-inverse-producer-investigation.md.

## Results — wins vs the producer (2P, /12)

| config | wins | note |
|---|---|---|
| champion (full stack) | 8/12 (67%) | already beats it; old "81% loss" was wrong |
| champion_strongest | 8/12 | 2P == champion |
| multi_opp (opp-model + multi-size, lean) | 8/12 | ties champion; the offensive lever is multi-size |
| inverse (opp-model only) | 7/12 | modest |
| multi_opp_snipe (+ denial/snipe, no prod) | 8/12 | no help; ≤ multi_opp |
| inverse_prod (+ HOLD_VALUE) | 0/12 | **production trap** |
| multi_opp_prod (+ HOLD_VALUE) | 0/12 | trap |
| lean_snipe (+ HOLD_VALUE) | 0/12 | trap |
| champion_holdval (+ HOLD_VALUE, SHIPPED 53734450) | 2/12 | trap (veto only softens) |
| multi_opp_keep (+ KEEP_VALUE) | 0/12 | "fundamental fix" — **failed** |
| champion_keep (+ KEEP_VALUE) | 2/12 | == holdval; failed |

4P (vs 3 producers): champion 3/6; champion_holdval 0/6 (eliminated every game).

## The answer

**No agent definitely beats the producer.** The ceiling is ~67% (champion /
multi_opp) — a real edge, never a certainty (symmetric boards + the producer is
genuinely strong). It IS beatable, though; it's not the wall the stale "beats us
81%" number implied.

## Why the producer is beatable at all (the 67% edge)

`safe_drain` lets it drain sources to ~0 and `capture_floor` sizes captures thin,
both against the **do-nothing (static-opp) projection** — so it persistently
leaves thin sources/captures, assuming we won't attack. Our opp-model predicts
exactly which planets it thins; multi-size lets us take them decisively. That's
the edge. But it caps at ~67% because the producer replans every turn and defends
visible threats, so the blind spot is a one-turn lag, not a free win — and on a
symmetric board it has the same edge against us.

## The big finding: production value is a TRAP, and the banking is not a pricing bug

"Maximize production + ships" loses to the producer in **every** form: flat
HOLD_VALUE (0/12 lean, 2/12 champion), and the holdability-discounted /
concentration-speed **KEEP_VALUE** "fundamental fix" (0/12, 2/12 — identical).
The over-expansion is **strategic over-extension** (production value grabs too
much, spreads the agent thin everywhere, and it loses the concentrated brawl),
**not** a per-capture keepability bug — so no holdability check fixes it. The
banking/idle the PI saw is partly *correct caution*: forcing expansion via a
production bonus is punished by a strong opponent.

`KEEP_VALUE` (PRODUCER_PLUS_KEEP_VALUE) is committed (gated default-off): it prices
the holdability threat at the opponent's concentration-speed recapture reply. It's
the right *idea* (opponent's reply, not current mass) but empirically inert here —
the over-extension is upstream of the per-capture credit.

## Refuted hypotheses (the data won, repeatedly)

- "strip the veto" → veto is ~neutral vs the producer (champion 8/12 == multi_opp 8/12).
- "lean beats the champion" → they tie.
- "production value helps" → trap (0-2/12).
- "keepable production is the fundamental fix" → failed (0-2/12).
- "snipe/deny exploit" → no lift (≤ multi_opp).

## Constructive takeaways

- The **champion** (already shipped) is the best anti-producer we have (~67%); the
  leaner **multi_opp** matches it more cheaply.
- Don't pursue production value as a champion lever — it's a producer-loser, and
  the ladder is producer-heavy.
- The only plausible remaining lever is **context-dependent expansion** (expand
  vs weak opponents, consolidate vs strong) — a different, harder mechanism, with
  no evidence yet that it helps. A flat "value production" bonus is the wrong tool.
