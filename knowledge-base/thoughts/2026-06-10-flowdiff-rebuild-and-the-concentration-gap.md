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

## Process notes

- Single-game iteration with decision-level instruments was the right speed: five
  mechanisms tested in one session (flowdiff, tail, regroup-off, flip-off, reaction), each
  with a clean verdict, no panel-waiting.
- The anticipatory drain (previous push) was confirmed net-negative on the panel and stays
  default-off; superseded conceptually by the flowdiff drain pricing.
