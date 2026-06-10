# HANDOVER.md — next-session brief

## What changed tonight (2026-06-10)

PI directed a from-scratch rethink ("simplest, yet feasible most
powerful"). The result is a new agent, **`agents/ledger/main.py`** — a
single self-contained file (it IS the submission, no bundler) built on
one mechanism: an engine-exact forecast of every planet's future, with
all actions priced off it and an event-driven rollout veto. Full story:
`audit/2026-06-10-ledger-agent-from-first-principles.md`.

**Measured (n=32, balanced seats, local):**
- vs Producer (public agent that beats our live champion 81%): **32/32**, Wilson-lo 0.893
- vs v7_0 (production baseline): **28/32 (87.5%)**, Wilson-lo 0.719
- timing p95 46 ms / max 74 ms against the 1000 ms budget
- self-play validation clean; forecast-parity test green

## The candidate

`submissions/ledger_v1.py` — ready to submit, **awaiting PI sign-off**
(CLAUDE.md Rule 1). Suggested submit gate for this artifact (replaces the
champion-bundle-specific Rule 46 steps):

```
python -m pytest tests/test_ledger_forecast.py -q          # parity green
python fast.py play submissions/ledger_v1.py --vs v7_0 --seed 7   # smoke
python fast.py bench submissions/ledger_v1.py              # p95 < 800ms
kaggle competitions submissions orbit-wars | head -5       # Rule 42 board
```

## Open questions for the PI

1. Submit `ledger_v1` now, or run more validation first (geometry panel
   128 seeds / more fresh seeds)?
2. STRATEGY.md still describes `baseline_adaptive_k`. Replace it with the
   ledger agent as the main strategy, or run both live first?
3. 4-player posture is rough (FFA fix applied, lightly tested). How much
   do 4P games weigh on the ladder? (`scripts/live_episode_summary.py`
   on a live submission id can answer.)

## Mode

Observation-driven iteration continues to apply: one observation → one
mechanism → one push (CLAUDE.md). The ledger agent's iteration log in the
audit file shows the loop working — every fix tonight came from reading a
specific losing game.

## Pointers

- `audit/2026-06-10-ledger-agent-from-first-principles.md` — the night.
- `agents/ledger/main.py` — the agent (heavily commented header).
- `tests/test_ledger_forecast.py` — the exactness gate; keep it green.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42) before any submit.
- `state/STRATEGY.md` — previous strategy (baseline_adaptive_k), not yet
  superseded on paper; PI decision pending.
