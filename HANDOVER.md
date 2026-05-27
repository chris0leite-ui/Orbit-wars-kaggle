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

## Current open work

- **Speed-discipline filters** (commit `3359fe2`, 2026-05-27): `BASELINE_MAX_ETA=20` + `BASELINE_MIN_SHIPS_LAUNCH=10` env-gated in `agents/baseline/proposer.py`. Wired ON in `agents/buildup_planner/main.py`. Local smoke: median ship count 30 → 46; share of <10-ship launches 15% → 3.5%; game length vs random 114 → 87 steps. **NOT YET A/B'd** on the broken composite stack or on the known-good 52968889 baseline.
- **Open question:** the full stack (K1+Zv2+Phi1+per-ship-sort) destructively interfered (sub 53065150 settled low). Diagnose by stripping back to the 52968889 config, then add ONE lever at a time. Speed-filter is the cleanest first candidate.

## Falsified or dead axes

- **Proposer pre-filter tightening (general).** Wins vs quiet opp, loses vs aggressive opp. Rule 37 cap. Future filter changes MUST A/B vs `submissions/baseline_joint_aggr_consolidated_orbitfix.py` BEFORE shipping. The speed-discipline filter (3359fe2) is conceptually adjacent — needs the vs-joint_aggr gate before claiming lift.
- **Commit-and-execute MILP caching.** Re-derive each turn is a feature (mid-opening adaptation), not a bug. `predict_relative` was the real hot path, fixed by K1 cache.
- **4-way composition stacking without per-step A/B.** sub 53065150 vs sub 52968889 lineage is the warning shot. Stack one lever at a time; A/B each addition.

## Next-session first actions (ranked)

### Priority 0 — Diagnose the destructive composition

Strip `agents/buildup_planner/main.py` env-sets back to the 52968889
config (just plain phase-dispatch + FINISHER, no Phi-1 hard-set, no
K1, no Z v2, no per-ship-sort). Bundle, smoke, and A/B vs joint_aggr
at n=16. If parity-or-better, we know the additions are the
regression source. Then add ONE lever (speed-filter is the cleanest
candidate) and A/B again. Slow is smooth.

### Priority 1 — Validate the speed-discipline filters on the known-good baseline

After P0 establishes the baseline, add `BASELINE_MAX_ETA=20` +
`BASELINE_MIN_SHIPS_LAUNCH=10` only (commit 3359fe2's env-sets in
buildup_planner). A/B at n=16 vs joint_aggr. If clears Rule 45
Wilson-lo ≥ 0.55, ship it as a submit.

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
