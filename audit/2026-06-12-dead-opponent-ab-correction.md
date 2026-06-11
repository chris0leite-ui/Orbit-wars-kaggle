# 2026-06-12 — CORRECTION: dead-opponent contamination of local A/B results

**Trigger: PI question — "Sure that there isn't a bug in our A/B test
against the producer?" The answer is yes, there was. Two of them.**

## The two dead-opponent mechanisms

1. **Producer adapter `__file__` bug.** `agents/producer/producer_agent.py`
   located its directory via `__file__`, which kaggle_environments' agent
   loader does not define when exec'ing an agent file from a path string.
   `fast.py` (importlib loading) defines it. Consequence: in EVERY custom
   harness run (`env.run(['...', 'agents/producer/producer_agent.py'])`),
   the Producer errored at load, the episode ended immediately, and the
   harness counted `reward == 1` as a win. Fixed 2026-06-12 (loader-
   independent directory discovery).
2. **torch missing after container restarts.** Both the Producer and the
   producer_plus bundles import torch at module load. torch was installed
   manually in the first session and was never in requirements; across
   some session restarts it vanished, killing torch-dependent opponents
   at load in those sessions' runs. Fixed: torch added to requirements.

A dead opponent sweeps exactly like a dominated one. None of the
contaminated runs logged game length or opponent liveness, so nothing
flagged. The live ladder (37.5% two-player win rate at ~1020 settle) was
the honest measurement all along.

## What is VOID

- ALL Producer results from custom harnesses: 8/8, 32/32 (n=32 battery),
  16/16 fresh seeds, 6/6, 4/4 regression cells. The headline
  "beats the Producer 48/48" is fiction.
- The live-1300.9 bundle pool results measured in torch-less sessions:
  31/32 (post-submit battery), 15/16 and 16/16 paired pools (seeds
  600-615). Proven by era-artifact replay: `ledger_v1_2` loses seeds 600
  and 603 against the alive bundle with the same step counts as the
  current build.

## What STANDS (opponents verified alive or torch-free)

- v7_0 results (all sessions): real fights, ~70-87% across pools.
- 4-player panel results (v7_0 / v4_planner / v3.5.1): real; the leader
  objective's 9/16 vs 4/16 parity baseline stands.
- The first 4-game live-bundle comparison (2026-06-11 session, torch
  present): real 500-step fights, 3/4 after the banking fix.
- All live-ladder data.

## TRUE standings (measured 2026-06-12 with liveness assertions)

- **vs Producer (alive): 0/16.** Eliminated in 90-170 steps every game.
  The ledger has never beaten a live Producer.
- vs live-1300.9 bundle (alive): mixed (3/4 on seeds 200-203 from the
  valid 06-11 session; 0/4 on seeds 600/603; full n=12 re-measurement in
  flight at write time — see addendum below).

## Operational lessons (binding for future A/Bs)

1. **Liveness assertion in every A/B**: opponent total launches > 0 AND
   game length > 30 steps, else the run aborts loudly.
2. **Torch (and any opponent dependency) lives in requirements**, not in
   session memory.
3. Log game lengths in every A/B row; sweeps with missing lengths are
   unaudited claims.

## Addendum: true live-bundle record + the non-transitivity read

Liveness-guarded n=12 (seeds 200-211): **11/12 wins**, mostly full
500-step score fights. Seeds 600/603: 0/2. Combined true rate vs the
producer_plus vetorf bundle: roughly 70-85%, seed-family dependent.

So the genuine standings are non-transitive:
- vanilla Producer **beats ledger 16/0** (relentless early pressure
  cracks the thin spread economy by step ~90-170);
- ledger beats the veto'd producer_plus derivative ~75% (its veto drops
  attacks; against an economist, withheld pressure = long games = the
  better economy wins on score);
- producer_plus beats vanilla Producer 24/32 (per its own audits).

The ladder population is Producer-heavy (it is public), which is why the
live 2-player record (37.5%) tracks the Producer matchup, not the
bundle matchup. The #1 strategic axis is now the vanilla-Producer loss
mechanism: all 16 losses are early-mid eliminations with the same
signature as the live mid-tier losses (behind in banked ships by t~50,
then finished by concentration).
