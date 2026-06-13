# The 4-player kingmaker tax — confirmed, but the obvious lever is a dud

Date: 2026-06-13. Corpus: live 4P episodes from subs 53588922 (garrison-value)
and 53577315 (coalition). Builds on `audit/2026-06-12-garval-first-live-read.md`
(garrison-value fires too late) and the war-ledger law "whoever reinforces
more wins."

## The observation: we fight the strongest; winners feed the weakest

`scripts/mine_conquest.py` records, for every planet that changes hands to a
human player, the **ship-rank of the prior owner at that moment** (rank 1 =
strongest of the four). Across the live 4P corpus:

| | enemy captures, mean prior-owner rank | rank-1 (strongest) share |
|---|---|---|
| **our LOSSES, by us** | **1.76** | 118 / 221 = 53% |
| our losses, by the eventual WINNER | 2.66 | 23 / 487 = 5% |
| **our WINS, by us** | 2.47 | 4 / 163 = 2% |

The pattern is unambiguous and it flips with the outcome. When we **lose**, we
are aimed at the **strongest** rival (rank 1.76) — we spend our offense
cracking the leader's expensive, well-defended planets. The player who **wins
that same game** almost never touches the leader (5%); they feed on the weak
(rank 2.66). And when **we** win, our own profile matches the winner's (2.47):
we were feeding the weak. The coalition sub shows the identical split
(losses 1.83, winner 2.40).

This is the classic free-for-all **kingmaker tax**: the player who polices the
leader pays for it in ships while two bystanders quietly compound past both
fighters. The economy miner corroborates the cost — by step 60 the eventual
winner holds 6 enemy planets to our 2, and production snowballs 18→38 while
ours plateaus at ~8-12.

## The mechanism is in our scorer — literally

`competitive_score = Δnet_me − Σ_opp w_opp · Δnet_opp`. Our live config sets
`PRODUCER_PLUS_FFA_WEIGHTS=strength`, so `w_opp ∝ rival strength`. Damaging
the **strongest** rival therefore scores **highest** — the objective pays us
to attack the leader's planets, which are exactly the most expensive to crack.
We built strength-weighting to answer the PI's "we don't attack the strongest
opponent" observation. The fix worked — and walked us straight into the
kingmaker tax.

## The obvious fix — re-weight toward the weak — does almost nothing

Added `PRODUCER_PLUS_FFA_WEIGHTS=weakness` (gated, default-off; the default
`strength` and existing `uniform` paths stay byte-identical; 2P untouched —
weights are only built at player_count ≥ 3). Weakness uses a bounded
complement, `w_opp ∝ (Σ_living strength − own strength)`, so the weakest
living rival earns the most weight, with a guard that puts full weight on the
lone survivor in an endgame duel.

Then measured the redirect on the live 4P loss replays (Rule 38 — reproduce
the failure state, apply the lever, see if the failure moves):

| config | enemy-aimed launches, mean owner rank | rank-1 share |
|---|---|---|
| strength (live) | 1.74 | 45 / 80 = 56% |
| weakness (complement) | 1.82 | 44 / 84 = 52% |
| steep inverse (`w ∝ 1/strength`) | 1.84 | 88 / 170 = 52% |

The knob changes *something* on 23% of decision steps, but it barely moves our
**aim**: even a steep `1/strength` inverse still points us at the leader on
half our enemy launches (88 of 170). **Re-weighting the opponent term cannot
fix the kingmaker tax.**

## Why: the leader-fights are positionally forced, not chosen

The reason is now clear. We do not attack the strongest rival because the
*offensive valuation* rewards it — we attack the strongest rival because the
strongest rival is our **neighbour and is attacking us**. The fight is forced
by the board geometry, and the offensive weight term rides on top of an
already-committed border war. You cannot re-price your way out of a war you
are already in. This is a clean, high-value **negative result**: it removes
target-selection re-weighting from the candidate-fix list for 4P.

## What this reframes

The 4P loss is **not** a target-CHOICE problem. It is a **positional /
midgame-collapse** problem: we get pinned in a war with our strongest
neighbour while the other two compound, then get carved by ≥2 rivals after our
~step-60 peak (see the carve signatures, unchanged across both live subs). The
two threads — "whoever reinforces more wins" and "we lose the enemy-conquest
race" — are the same collapse seen from two sides.

The fix has to be **structural**, and the supported candidate is already on
the table from today's first garrison-value read:

- **Extend the reinforcement threat WINDOW beyond our send horizon.** Today's
  read showed garrison-value is mute during the collapse because the
  half-weight deficit only turns positive once the enemy mass is already too
  close to rescue. Computing the deficit over a longer lookahead `W > K`
  (while reinforcement candidates stay at `K`) rings the alarm while rescue is
  still feasible — the modeling-correctness fix (Rule 40), with magnitude
  unchanged so mirror exposure is plausibly small. This addresses the
  *survival* side of the collapse directly, where re-targeting could not.

A second, larger idea — **disengagement** from a losing forced border war
(stop reinforcing it; redirect force to expand into weak/neutral territory) —
matches the winner's behaviour but is a much bigger design change and needs PI
discussion before any build.

## Disposition

- `weakness` mode kept in the engine, gated and default-off, with unit tests
  (`tests/test_ffa_score.py`, 13 green) — consistent with the codebase norm of
  preserving falsified-but-gated mechanisms (cf. rotation-aware margins). It is
  **not** recommended for a live slot: it barely changes behaviour.
- No new submission. Rolling pair unchanged: 53577315 (1241.6) + 53588922
  (1197.6, warming). Daily budget untouched.
- Recommended next mechanism, pending PI sign-off: the threat-window
  extension.
