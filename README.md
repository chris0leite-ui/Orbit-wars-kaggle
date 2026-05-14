# Orbit Wars — agent competition

Repo for the Kaggle **Orbit Wars** code/agent competition
(https://www.kaggle.com/competitions/orbit-wars). Submissions are
Python agents evaluated by tournament play (TrueSkill / Elo).

**State (2026-05-14):** workspace re-bootstrapped from scratch. Prior
strategy code (~6 KLoC `lib/`, 25 historical agents, 31 scripts, 67
tests, `fast.py`) is preserved on `main` but removed from this branch
so we can build a clean strategy. Workflow infra (CLAUDE.md rules,
audit/, knowledge-base/, state/) is preserved.

## Quick start

```bash
bash bootstrap.sh                          # one-time: data + deps + smoke
pytest tests/ -q                           # < 30 s; env + agent wiring
python eval.py --smoke                     # 4 games vs random
python eval.py --vs nearest -n 24          # vs the comp's shipped baseline
python eval.py --panel  -n 24              # 3-opponent calibration panel
./submit.sh "v0: nearest sniper baseline"  # PI-approved single-shot submit
```

## Layout

```
main.py            our agent (kaggle-submittable as-is)
eval.py            local A/B harness; Wilson CI; parallel; 2P/4P
submit.sh          kaggle submit wrapper
baselines/         opponents for local A/B
  nearest.py         comp's shipped baseline (~μ303)
  v7_0.py            historical strong baseline (~μ1081, drop-one chooser)
  v4_planner.py      architecturally different strong baseline (~μ1038)
tests/test_smoke.py  env import + agent-vs-random smoke
data/                comp spec (README.md, agents.md, main.py)
```

Process docs (read on trigger, not every session):
| Path | Role |
|---|---|
| `CLAUDE.md` | 39 operating rules + pointers |
| `SETUP.md` | Day-1 onboarding |
| `WRAPUP.md` | Wrap-up / prepare-handover procedure |
| `HANDOVER.md` | Next-session brief |
| `ISSUES.md` | Live problem-tree / claim board |
| `comp-context.md` | Settled-once facts |
| `state/` | Mutable session state (calibration, ladder, ledger) |
| `audit/` | Dated probes + friction notes |
| `knowledge-base/` | PI second-brain (permanent — Rules 35-36) |
| `.claude/skills/` | `kaggle-comp` (process loops) and `postmortem` |

## Competition essentials

- **Board:** 100x100 continuous; sun at (50,50) r=10. 500 turns. 2P or 4P.
- **Submission cap:** 5/day. Kaggle auto-keeps your rolling-last-2 for
  final evaluation (Rule 12: never push speculative variants after a
  known-good submit).
- **Deadline:** 2026-06-23 23:59 UTC.
- **Action format:** `[[from_planet_id, angle_rad, num_ships], ...]`
  per turn; entrypoint is `agent(obs)` at module level.

Full game spec in `data/README.md`. Workflow rules in `CLAUDE.md`.
