# HANDOVER.md — next-session brief

> Last written: 2026-05-27 by `claude/agent-design-exploration-Q0q9T`.
> Older sections archived to `audit/archive-2026-05-24-handover.md` and
> `audit/archive-2026-05-26-handover.md`.

## PI directive (2026-05-27): no μ values in tracked docs

Kaggle's μ rating drifts continuously as ladder games keep coming in.
Every snapshot we write becomes stale within hours and misleads the
next session. **Always query Kaggle live; never read μ from a docs
file. Don't write μ values into HANDOVER, MULTI_BRANCH, or commit
messages.**

```bash
kaggle competitions submissions orbit-wars | head -10
```

That's the single source of truth for live state.

## Read order (Rule 44 — mandatory)

1. **`kaggle competitions submissions orbit-wars`** — live rolling pair + recent submits + status. **Do this first.**
2. **`state/MULTI_BRANCH.md`** — track registry, closed-axes list, push claim board (no μ values).
3. **`state/TOOLS.md`** — A/B harnesses, diagnostics, bundle/validation.
4. **`CLAUDE.md`** — rules 1-48.
5. **This file.**

## Where we are (2026-05-27)

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC.
- **Daily submits today:** check `kaggle competitions submissions orbit-wars` and count today's dated rows.
- **Rolling pair (auto-kept by Kaggle):** check live. Top 2 dated rows = current rolling pair (most recent = "weak half" by Kaggle's terminology, older = "strong half" if it was strong).
- **Recent submits, this branch:**
  - sub **53065150** (2026-05-26 23:58) — full stack: K1+Zv2+Phi1+per-ship-sort. Settled low (a destructive composition).
  - sub **53018599** (2026-05-25 11:54) — K1+Zv2.
  - sub **53000996** (2026-05-24 22:38) — Phi-1 leaf only.
  - sub **52993021** (2026-05-24 16:10) — concentration.
  - sub **52968889** (2026-05-23 23:59) — plain phase-dispatch + FINISHER (bundler-trailer fix). Branch peak per memory at submit time.
  - sub **52968305** (2026-05-23 23:17) — bundler entrypoint ERROR (root cause `eb1653a`).

## Today's work (2026-05-27)

1. **Cherry-picked sibling per-ship-sort flag + bundle-order fix** (commit `a5a83da`). Moved `SORT_BY_EV_PER_SHIP` from module-level cache to call-time read inside `choose_trajectory`. Wired `BASELINE_SORT_BY_EV_PER_SHIP=1` in `agents/buildup_planner/main.py`. Submitted as sub `53065150` (one-minute PI submit, no full panel A/B).
2. **Sub 53065150 settled badly** — the 4-way composition (K1 + Zv2 + Phi1 + per-ship-sort) destructively interfered. Query Kaggle live for the number; the key fact is that stacking 4 levers without per-step A/B regressed below the simpler 52968889 lineage.
3. **Speed-discipline filters** (commit `3359fe2`): PI observed self-play with small fleets crawling at log-speed. Added `BASELINE_MAX_ETA` (default = MAX_HORIZON, backward compat) and `BASELINE_MIN_SHIPS_LAUNCH` (default = MIN_FLEET_SIZE) env-gated filters in `agents/baseline/proposer.py`. Both call-time reads (bundle-order safe). `agents/buildup_planner/main.py` sets MAX_ETA=20 and MIN_SHIPS_LAUNCH=10.
4. **Speed-filter smoke results:**
   - vs `random` (87 steps, WIN): 57 launches; min=7, median=46, max=338; <10 ships 2/57 (3.5%, was 15%).
   - vs `submissions/baseline_joint_aggr_consolidated_orbitfix.py` (293 steps, decisive WIN 28-0): 297 launches; min=10, median=32, max=291; zero launches <10 ships.
   - **Not yet A/B'd at n=16 against joint_aggr.** The decisive vs-joint_aggr single game is a positive triage signal but not a Rule 45 gate.
5. **Doc cleanup** (commit `bdf604e`): PI directive to never document μ. Stripped `state/MULTI_BRANCH.md`, this file, `state/calibration-ladder.md`, `state/mechanism-ledger.md`, `audit/INDEX.md`, `audit/friction.md`. Push claim board trimmed to stable fields only.

## Current state (decision needed next session)

The speed-filter is **committed but unshipped**. The big question is whether to:

- **Path A:** Strip `agents/buildup_planner/main.py` back to the 52968889 config (no Phi-1, no K1, no Zv2, no per-ship-sort), add ONLY the speed-filter, A/B at n=16 vs joint_aggr. Cleanest signal; preserves the known-good lineage as the base.
- **Path B:** Keep the current 4-way stack and add speed-filter on top. Lower information value (regression source stays unidentified) but a single shippable artifact.

PI tone at end of session: stacking-without-A/B has burned trust. Path A is the disciplined choice.

## Falsified or dead axes

- **Proposer pre-filter tightening (general).** Wins vs quiet opp, loses vs aggressive opp. Rule 37 cap. Future filter changes MUST A/B vs `submissions/baseline_joint_aggr_consolidated_orbitfix.py` BEFORE shipping. The speed-discipline filter (3359fe2) is conceptually adjacent — needs the vs-joint_aggr gate before claiming lift.
- **Commit-and-execute MILP caching.** Re-derive each turn is a feature (mid-opening adaptation), not a bug. `predict_relative` was the real hot path, fixed by K1 cache.
- **4-way composition stacking without per-step A/B.** Sub 53065150 (K1+Zv2+Phi1+per-ship-sort) vs the simpler 52968889 lineage is the warning shot. Each "improvement" we added since 52968889 has regressed us per live μ. Stack ONE lever at a time; A/B each addition; never compose without per-step calibration.

## Next-session first actions (ranked)

### Priority 0 — Sync, then check sub 53065150's settled state live

```bash
git fetch origin && git pull --ff-only origin claude/agent-design-exploration-Q0q9T
kaggle competitions submissions orbit-wars | head -10
```

Find sub 53065150's row. If still PENDING, wait. If COMPLETE, note
the number but don't write it anywhere — it's already locked into
ladder history.

### Priority 1 — Strip back to 52968889 config, then add speed-filter only

The 4-way composition (K1+Zv2+Phi1+per-ship-sort) is the suspect.
Walk back `agents/buildup_planner/main.py`'s top-of-file env sets
that diverge from the 52968889 source. Specifically:

  - REMOVE: `os.environ["BASELINE_VALUE_HEAD"] = "phi"` (Phi-1)
  - REMOVE: `os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")` (K1)
  - REMOVE: `os.environ.setdefault("BASELINE_EFFECTIVE_LANDING_PRUNE", "1")` (Z v2)
  - REMOVE: `os.environ.setdefault("BASELINE_SORT_BY_EV_PER_SHIP", "1")` (per-ship-sort)
  - KEEP: `os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")` (was in 52968889)
  - KEEP the new speed filters: `BASELINE_MAX_ETA=20` + `BASELINE_MIN_SHIPS_LAUNCH=10` (commit 3359fe2).

Bundle, parity smoke (skip-parity-gate due to known lux_ai_s3
shadow), then A/B at n=16 vs `submissions/baseline_joint_aggr_consolidated_orbitfix.py`.
If Wilson-lo ≥ 0.55, this is a defensible submit. If parity at
n=16, escalate to n=32 before deciding.

### Priority 2 — If P1 doesn't lift, replay-scout

Pull 5-10 top-of-ladder replays. Catalog opening + midgame patterns.
Especially: how do top agents handle "opp closer to the midline
neutral than we are"? Cheap, high info value.

### Priority 2 — Replay-scout (deferred since 2026-05-24)

Pull 5-10 top-50 ladder replays. Catalog opening + midgame patterns.
Especially: how do top agents handle "opp closer to the midline
neutral than we are"? Cheap, high info value.

### Priority 3 — Better rollout opp model

The cheap-recapture diagnosis was real. The proposer pre-filter axis
is closed (Rule 37). The right fix is structurally better opp model
inside `score_candidate_v4`'s rollout: priority-based projection
(opp targets highest-prod nearest-K), or cap rollout opp launches at
K=2.

### Priority 4 — Φ refactor Stages 2-5

Plan: `/root/.claude/plans/go-also-checknfor-similar-purring-flute.md`.
Don't start until P0-P3 resolved.

## Submission discipline

- Submits require **n=32 Wilson-lo ≥ 0.55** OR clear PI override.
- **No μ values in commit messages or PR bodies** — they go stale.
  Reference sub IDs only.
- Pre-submit Rule 42 check: compare predicted candidate vs evicted
  candidate LIVE against `kaggle competitions submissions orbit-wars`,
  not against a docs snapshot.

## Pointers

- `state/MULTI_BRANCH.md` — track registry, closed-axes list, push claim board.
- `state/TOOLS.md` — A/B harnesses, bundle/validation.
- `scripts/ab_quick.py` — 5-game / 250-step / no-swap A/B standard.
- `audit/2026-05-25-postmortem-k1-zv2-axis-exhaustion.md` — proposer-tightening closure.
- `audit/2026-05-25-consolidation-profile.md` — pre-K1 cProfile.
- `audit/2026-05-25-consolidation-review.md` — K1 finding + Rule-44 cross-reference.
- `audit/2026-05-25-consolidation-profile-post-K1.md` — post-K1 cProfile.
