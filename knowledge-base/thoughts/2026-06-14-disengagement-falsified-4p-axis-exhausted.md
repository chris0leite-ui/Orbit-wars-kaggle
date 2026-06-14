# Disengagement tested and falsified — the 4P axis is exhausted

*2026-06-14, closing the disengagement thread the PI green-lit. Continues
`2026-06-14-4p-defense-levers-and-the-disengagement-pointer.md`.*

The PI chose to design disengagement together, then chose the **passive brake**
as the smallest first cut: penalize defensive sends to CLEARLY-DOOMED own
planets (balance-of-force deficit ≥ MARGIN × our biggest feasible reinforcement
wave, so no feasible defense saves it), freeing that force to the scorer's
next-best candidate. Built gated (`PRODUCER_PLUS_FUTILE_BRAKE` / `_MARGIN`,
default-off, byte-identical; commit f83b7b0).

## It fires, and it robustly HURTS

The brake engages (91 doomed-planet sends penalized over 2 games — the agent
*was* trying to reinforce planets it can't save). But measured on `margin_ffa`
(fast 4P triage, material share + rank + eliminations, n=63 vs 3×V2):

| config | share @120 | share @final | elim |
|---|---|---|---|
| baseline garval | 23.8% | 24.8% | 33/63 |
| brake 2.0 / margin 2.0 | 20.8% | 21.1% | 37/63 |
| brake 1.0 / margin 3.0 (conservative) | 21.3% | 20.6% | 36/63 |

Both configs lose material and get eliminated MORE, and the harm **compounds**
(tied at step 40, diverging by 120). Conservative tuning doesn't rescue it — so
it is the **idea**, not the calibration.

## Why — the insight that kills passive disengagement in 4P

In a free-for-all **carve**, conceding a doomed planet makes you the **soft
target**: rivals take it cheaply and snowball off you faster. Not-contesting is
not "saving force" — it is "dying faster." Contesting a planet you will lose
still taxes the attacker and delays the carve. The winner does **not** concede;
they contest *and* feed the weak. So "stop reinforcing the lost border" is
exactly backwards for our seat: the lost border is where rivals carve us, and
making it cheap accelerates the collapse the mechanism was meant to prevent.

This also kills the **active-redirect v2** in advance: it would still concede
the doomed planets (same fatal flaw) *and* it needs a reachable winnable front
to redirect to — which the forced-war geometry denies (the war is forced
precisely because the weak rival is unreachable past our strong neighbour).

## The 4P axis is comprehensively exhausted

Tallying this session against the team's prior work, every buildable 4P lever
is now accounted for:

- **Re-aim offense (re-weight strong→weak):** refuted — can't re-price out of a
  forced war (kingmaker-tax audit).
- **Neutral expansion / quota:** refuted — not the separator; lane shelved 10-10.
- **Decline contested captures (reinforcement floor term):** misaligned —
  raises floors most on the dense leader, amplifying 4P dilution.
- **Defend earlier (threat-window extension):** structurally cornered (detection
  gated at K, sizing inflated); faint +2/+1, impractical to confirm.
- **Disengage / concede the lost border (this brake):** falsified — conceding
  feeds the carve.

The 4P loss is a **positionally forced, seat-draw collapse**. The winner's edge
("feed the weak, rank 2.66") is substantially a *geometry draw* — they are
adjacent to a weak rival; we are adjacent to the strong one — not a strategy we
can adopt from our seat. No floor/deficit/redirection mechanism reverses it,
and the one that tried to *minimize* the loss (disengagement) makes it worse.

## Recommendation

Stop spending on 4P mechanisms. The remaining ~9 days are better spent
protecting and sharpening the **2P strength** (~1291) and managing the
rolling-pair endgame (Rule 12: Kaggle keeps the rolling last two). If the
ladder mix is 2P-heavy, the strong 2P stack carries; if it is 4P-heavy, we are
near the forced-collapse floor and no available lever moves it. Consolidation,
not another 4P bet, is the positive-expected-value play from here.
