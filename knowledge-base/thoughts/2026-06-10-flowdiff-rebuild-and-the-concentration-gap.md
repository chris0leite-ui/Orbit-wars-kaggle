# The single-currency rebuild, and what it revealed: the Producer plays concentration

Date: 2026-06-10. Probe work on `agents/protoflow/main.py`, fast single-game iteration
(per PI instruction: no waiting on big panels; decision-level reads).

## What was built (all default-off, gated, calibrated, pushed)

1. **The single-currency evaluator** (`FLOWDIFF_VALUE`). Root cause measured on the panel:
   the old values priced what a move gains but never the ships it spends — attrition was
   charged nowhere, value could not go negative, the agent always found something to do and
   over-launched into collapse. New currency: signed terminal wealth (garrison at the window
   edge, mine positive / enemy negative / neutral zero), injected minus baseline, minus newly
   sent ships at par. Pyrrhic captures go negative; protect-nets-zero-when-safe, no-bleed and
   doomed-planet write-offs all emerge as value facts instead of code.
2. **Ownership continuation** (`FLOWDIFF_TAIL`). Pure window readout refused any neutral whose
   garrison exceeds in-window repayment (we under-expanded and even HOARDED — seed 2: 395
   ships vs 68 and still lost). Fix: the owner at the readout keeps producing; credit that
   stream discounted per turn (the discount prices the unseen retake).
3. **Defender reaction** (`FLOWDIFF_REACTION`). Measured: only 42–53% of our offense waves
   even captured at landing — the defender watches our fleet approach and reinforces; the
   flip tier was the dying cohort (flip-off pair: capture rate 0.84). Fix in the model:
   inject the target's standing counter into the injected rollout only (the reaction is
   caused by our wave; the do-nothing baseline correctly lacks it). Result: capture rate
   0.87/0.77, hold rate 0.92/0.76, wave count halved. NOTE: this same idea was inert under
   the old swing-integral evaluator — the retake landed past the horizon. The terminal-wealth
   form sees it. A dead-end under one evaluator can be live under another.

Field panel (12 seeds): vs v7 5/12 with end-planets 9.2 (was 3/12 / 6.2); nearest/production
held; Producer still 0/12.

## The new instruments (the real session yield)

- `scripts/protoflow_game_trace.py` — one-game decision narrative + wave outcomes
  (captured at landing+2? still ours at +15?). One game now answers what a panel used to.
- Defense-outcome and killing-wave scripts (in-session): for every planet we lose, was any
  defense sent, what wave killed it, what wealth did we have in reach?

## The finding that should drive the next session

Vs the Producer, with clean offense economics (87% capture / 92% hold), we still lose 0/12.
The defense measurement says why: **24 of 25 lost planets had zero defensive response, and
the killing waves were 165–421 ships against garrisons of 18–398, with our ships-within-25
at ZERO in 10 of 12 falls** (while our total wealth was 200–800). The write-offs were
correct — those planets were unsavable. The asymmetry is positional: the Producer masses
wealth into one bank and swings one hammer; our wealth dies dispersed. A per-planet
evaluator prices a ship identically anywhere — position is the unpriced dimension, and the
heuristic regroup pass (evaluator-blind) does not fix it (regroup-off pair: identical
losses).

Candidate directions (PI decision pending):
- **Consolidation in the currency**: relocation cells valued by the same terminal-wealth
  diff across the two touched planets — evacuation of doomed wealth and pooling into
  defensible stacks would emerge. Needs the doomed test kept honest (in-flight committed
  threat, not the hypothetical standing maximum).
- **Sequential-commit assembler** (plan phase 2) — correct marginal accounting, but does NOT
  obviously price position either.
- Accept the Producer gap for now: the live field is not all Producers, and the rebuild
  already gains vs the v7 lineage.

## Continuation (same session, after PI's "think for yourself, iterate")

Four more mechanisms tested at single-game granularity:

4. **Committed-wave defense** (`FLOWDIFF_COMMITTED_DEFENSE`) — NEGATIVE. Physics said
   defense is possible (warning windows 9–14 turns, 7/13 falls had a capable ally in
   reach), and after fixing a re-buy bug (399 ships poured into one planet over four
   turns because in-flight reinforcements weren't credited) the funded planets DID hold —
   but total losses were unchanged: the Producer's larger bank just takes a different
   planet. Whack-a-mole. Defense spend at this scale is a losing allocation.
5. **Adaptive reaction** (`FLOWDIFF_REACTION_ADAPTIVE`) — KEEP (panel-backed). The fixed
   reaction term was opponent-dependent: right vs the Producer (pools defense), ruinous
   vs v7 (doesn't punish; 5/12 → 1/12, idle 0.75). Fear is an observable opponent
   property: score each landing against the do-nothing prediction stored at launch,
   EMA the reinforcement rate, scale the injected counter by it. CRITICAL bug class
   found: score one turn after the REAL landing turn at actual fleet size — scoring at
   the planned arrival reads our own unresolved wave as phantom reinforcement (spurious
   fear 0.49–0.69 vs v7; after the fix v7 converges to ~0.0 and part of the Producer's
   apparent pooling also turned out to be artifact).
6. **Low fear prior (0.15)** — no gain on a 6-game discriminator; reverted to the
   panel-backed 0.5. Prior tuning needs panel-grade evidence.

Final panel standings (12 seeds): adaptive composite vs Producer 1/12 / end 2.6 (best yet);
vs v7 3/12 / 6.9. Plain flowdiff+tail remains v7-best (5/12 / 9.2). The keep-set decision
between them is a field-mix bet — the adaptive arm is architecturally right (it converges
to plain-tail behavior against passive opponents) but pays a residual convergence tax.

Two ship-accounting theorems derived and used (worth remembering):
- **Combat is 1:1, so garrisons always trade fair**: keeping wealth on a doomed planet
  bleeds the attacker exactly as much as evacuating saves — planets, not ships, are the
  only real prize. Evacuation mechanisms are currency-neutral; don't build them.
- **Deterrence-by-garrison cannot work against the Producer**: garrison losses cancel in
  its competitive score; only fundability gates its captures. Don't build garrison
  deterrence either.

## Process notes

- Single-game iteration with decision-level instruments was the right speed: five
  mechanisms tested in one session (flowdiff, tail, regroup-off, flip-off, reaction), each
  with a clean verdict, no panel-waiting.
- The anticipatory drain (previous push) was confirmed net-negative on the panel and stays
  default-off; superseded conceptually by the flowdiff drain pricing.
