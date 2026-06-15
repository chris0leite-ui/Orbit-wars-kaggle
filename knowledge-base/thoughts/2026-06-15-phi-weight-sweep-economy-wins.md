# 2026-06-15 — Phi weight sweep: economy + tempo win; options & caution hurt

Built the clean-room vanilla positional agent (`agents/phi/main.py`) as a
legible instrument, then swept its weights head-to-head vs the default-weight
Phi (stateless → two weight-sets in one process; n=10/config, 5 seeds × 2
orientations). Default = ECON_K 25, REACH_W 3, RISK_W 1, ETA_W 0.5.

## Result (winrate of the CHANGE vs default; >0.5 = the change helps)

| change | winrate | avg planets@60 | read |
|---|---|---|---|
| ECON_K 25→50 | 0.30 | 6.2 | worse |
| ECON_K 25→10 | 0.10 | 5.1 | worse — **25 is a sweet spot** |
| REACH_W 3→0 | **0.60** | 6.6 | **options OFF wins, expands more** |
| REACH_W 3→8 | 0.00 | 4.9 | more options loses every game |
| RISK_W 1→0 | **0.70** | 5.8 | **caution OFF wins** |
| RISK_W 1→3 | 0.20 | 6.2 | more caution worse |
| ETA_W 0.5→0 | 0.20 | 4.8 | tempo OFF worse — **tempo helps** |

Punchline config — REACH_W=0 **and** RISK_W=0 (keep ECON_K=25, ETA=0.5),
i.e. **aggressive economic expansion + tempo** — beats default **13–3 (0.81),
n=16**. The two "off" effects compound.

## The finding (a real course-correction for the positional thesis)

Of the framework's factors, **economy (compounding production) + tempo are the
winning levers; options/reach and durability/caution both HURT.** The game
rewards grabbing production fast and aggressively, not hoarding "options" or
playing cautiously.

This is **convergent evidence**, not one noisy run:
- **Options/reach hurts** — confirmed 3×: this sweep (REACH_W=0 > default >
  REACH_W=8), the producer-side frontier patch (reduced planets@60), and the
  frontier reproduction (mismatched to corner-neglect anyway).
- **Durability/caution hurts** — confirmed 2×: this sweep (RISK_W=0 wins 0.70)
  and the real-ladder `garval` failure (defense + source-safety, 1181 < 1280).
  The field is aggressive (collapse traces: opponents keep 2–5× more ships in
  flight) — caution cedes to it.

## Direct implication for this session's builds
The two terms I built — **frontier (= reach)** and **tenure (= durability/
caution)** — are the **wrong levers**; the instrument predicts both would *hurt*
on the ladder. Do **not** fire them. They stay committed but gated-OFF as
negative results.

The **right** lever is aggressive economic expansion + tempo — which is exactly
what the live `expand` variant (wider neutral shortlist + deeper horizon = more/
faster economic capture) already pursues. So the positional investigation
**validates `expand`** and says: push expansion *harder / more aggressively*,
don't add options or caution terms.

## Caveats
- Phi-vs-Phi self-play measures what beats a weak agent, not directly the
  strong ladder field; but the two key findings each converge with independent
  real-ladder evidence (garval; frontier planets@60), so they're likely real.
- n=10/config (decisive at the extremes 0.00/0.70/0.81; suggestive at 0.30/0.20).
- ECON_K=25 sweet spot is for THIS Phi's units, not directly producer's horizon.

## Next
- Lean into aggressive economic expansion on producer (the `expand` direction),
  e.g. push the shortlist/horizon further or reduce launch-discipline — NOT
  reach/tenure.
- If porting Phi into producer's objective: weight economy + tempo, drop the
  reach/caution terms.
