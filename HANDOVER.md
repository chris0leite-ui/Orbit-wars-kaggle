# HANDOVER.md — next-session brief

## Live state (check freshness: `kaggle competitions submissions orbit-wars`)

Rolling pair: **sub 53558897** `ledger_v1_2` (~870, climbing) + **sub
53556728** `ledger_v1` (~1000, settled). Both losing the 2P Producer-
style matchups, winning 4P at above-parity — the live mix tracks local
measurements.

## The agent (agents/ledger/main.py)

RESTORED to the **ledger_v1_4** state (byte-identical to
`submissions/ledger_v1_4.py`, forecast parity green) after the
2026-06-12 concentration-rebuild session regressed clean panels
(audit/2026-06-12-concentration-rebuild.md has the full arc). v1_4 =
v1_2 (live) + 4-player leader objective (9/16 first places vs parity
4/16) + stalemate-gated endgame gambit. Submitting v1_4 evicts
ledger_v1; PI sign-off required (Rule 1).

## The Producer matchup (head-on hunt, PI-mandated)

Status: 0-for-everything at honest compute, BUT the win condition was
found and measured once (load-truncated build went 8/10): fortress
freeze -> Producer goes passive at score parity -> stalemate gambit
dismantles it (it has no freeze-breaking logic). Six deterministic
encodings of that policy failed (gates either throttle the opening or
leak the fortress). Key measured facts in the audit: its landed-
tonnage stick rate 98-99% vs our ~50%; opening economics at parity;
collapse window t50-80 via 2-3 parallel solo-kill stacks.

**Next concrete step: bisect c42c9fc (14 mechanisms) one at a time**
against a fixed panel {Producer x8, v7_0 x8, bundle x8}, keep only
mechanisms that don't regress the panel, then revisit the
opponent-adaptive response propensity design (exact landing
attribution — design in the audit).

## Tooling (new this session, keep)

- `scripts/trace_duel.py` — per-tick duel tracer: economy curves,
  captures with sinks, launch sizes, fleet-death classification, with
  liveness asserts built in. The workhorse.
- `LEDGER_DEBUG=<file>` decision introspection (in c42c9fc, not in the
  restored v1_4 — cherry-pick on demand).
- Analyzer scratch: /tmp/analyze_trace.py (rebuild from audit if gone).

## Binding methodology lessons (audit-backed)

1. **CPU contention changes agent behavior** (wall-clock TIME_BUDGET):
   batteries at >3 workers measure a different agent; always solo
   spot-check the headline seeds.
2. **The crash guard hides breakage**: a World.__slots__ typo made the
   agent return [] every turn silently — liveness asserts + the parity
   test are the only tells.
3. Editing the agent file mid-battery contaminates the battery
   (workers load at game start).

## Verification benchmarks (the restored v1_4)

- v7_0 12-seed pool: 8-9/12 expected
- live-1300.9 bundle rebuild (/tmp/latest_live_sub.py, rebuild recipe
  in audit 2026-06-10): ~75-85% on seeds 600-615
- 4P panel seeds 1000-1015 vs v7_0/v4_planner/v3.5.1: 9/16
- `tests/test_ledger_forecast.py` green (engine exactness)
- restoration sanity on bundle seeds 600-607 was running at handover
  write time (/tmp/restore_bun_*.json)

## Plan remainder (unbuilt designs, see audit + git history)

- Race modeling refinements (arrive-second pricing exists in c42c9fc).
- Wave-merge offense; opponent profiling by garrison growth (negative
  result as launch-classifier; landing-attribution design ready).
- Loss loop: `python scripts/live_episode_summary.py <sub> --pull`.
