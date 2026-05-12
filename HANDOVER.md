# HANDOVER.md — next-session brief

> Last written: 2026-05-12 by `claude/simplify-codebase-p39Hm`. Format
> budget ≤150 lines.

## Where we are

- **Comp:** Orbit Wars (slug `orbit-wars`). Deadline 2026-06-23 23:59 UTC
  → **42 days remaining.**
- **Best live submission:** **σ-equivariance v1 (#52565034) at μ=1063.2**,
  pulled fresh from the Kaggle API. Beat v3.5.1 aggressive (#52565976,
  μ=994.7) by +68.5μ. Rolling-last-2 = `[v3.5.1 (#52565976, 994.7),
  σ-equiv v1 (#52565034, 1063.2)]`.
- **State files were off by 60-100μ** in both directions before today's
  check — see friction `state-files-out-of-date-vs-live-mu`. Next session
  should pull live μ at start.
- **Branch HEAD:** `7c1e078`. Local-only; not yet a PR.
- **Daily submission budget:** 0/5 used today.
- **Test suite:** 156/156 green in 77 s.

## Today's progress

Three load-bearing changes on `claude/simplify-codebase-p39Hm`:

1. **Codebase collapse (commit `1f69039`).** Deleted `lib/`, `agents/`,
   `submissions/`, 22 scripts, 15 tests. Single-file source-of-truth
   `agent.py` (1010 LOC, submit directly — no bundler). 156 tests cover
   only live code paths. Bit-identical to v3.5.1 modulo unreachable
   branches; self-play 16/16 draws confirmed determinism.

2. **σ-equivariance merge (commit `294da5e`).** 3 surgical patches from
   `claude/game-theory-strategy-analysis-0oH4N` (`6c12b9f`, `7b60938`,
   `24bae06`): `sym_hypot` order-independent hypot in `propose_snipe` +
   `propose_reinforce` distance calcs; σ-equivariant tie-break
   `(-kx, -ky, target_id)` in `settle_plan`; score rounded to 6 dp
   before tie-break. +50 LOC. Self-play preserved 16/16 draws.
   A/B vs `opponents/v3_snipe_frozen.py`: 18/32 = 56.2% Wilson
   [39.3%, 71.8%] (was 15/32 pre-merge). **This is the variant most
   likely to match σ-equiv v1's μ=1063.2 on the ladder.**

3. **v7_minimax merge (commit `9114f31`, +500 LOC) — STATUS: BROKEN
   σ-EQUIV.** Pulled K-step maximin from same upstream branch: 2 our
   candidates × 2 opp candidates payoff matrix, seat-symmetric Sim<K>
   scoring, v3 as rollout policy, 4P fallback to v3. 16 unit tests
   added. **Self-play dropped 16/16 → 8/16 draws** (50% non-draws).
   **A/B vs frozen: 17/32 = 53.1%, *worse* than σ-equiv-only's 18/32.**
   Root cause is algorithmic (max-min vs min-max diverge when the
   payoff matrix has no pure saddle point), NOT RNG drift — see
   `audit/2026-05-12-postmortem-simplify-codebase-p39Hm.md` for the
   misdiagnosis story. The `7c1e078` "deterministic seed fix" turned
   out to be a no-op (env was already deterministic via tournament
   harness).

## Falsified or dead this session

- **v7_minimax-as-σ-equiv-superset claim** (game-theory branch commit
  `cbed49e`): false on mixed-strategy turns. Local A/B 53% vs frozen
  is below σ-equiv-only's 56%. Don't submit v7 without resolving this
  first.
- **`cfg["seed"] = obs.step` fix for self-play asymmetry** (commit
  `7c1e078`): no-op. Env was already deterministic. Kept on branch
  as harmless defensive code; could be reverted.
- **State-file μ predictions** as decision priors: v3.5.1 was state-
  predicted at μ=1090-1100, actually 994.7; σ-equiv state-claimed
  976.3, actually 1063.2. Stop quoting expected-μ from local-A/B
  winrates — quote only Wilson CI and let live games settle it.

## Next-session first-action

Ranked. EV-priority. PI-authorised for submits (Rule 1).

1. **Decide v7 disposition.** Three options pending from end of this
   session: (a) revert v7 entirely, keep σ-equiv-only HEAD; (b) try
   adding σ-equivariant tie-break to `_maximin_pick` to restore the
   cannot-lose property; (c) submit v7 anyway and let live μ decide.
   Recommended (a) — local A/B already shows σ-equiv-only is stronger
   and live data shows σ-equiv is +68μ over the best-aggressive bet.
2. **PI submit decision** on whatever lands from (1). If (a) chosen,
   submit `agent.py` at `294da5e` (revert `7c1e078` + `9114f31`).
   Hypothesis: σ-equiv-on-aggressive-sizing recovers or improves on
   σ-equiv v1's μ=1063.2.
3. **Aggressive-sizing audit.** Live data hints aggressive sizing
   regressed (v3.5.1 at 994.7 vs no-aggressive v3_snipe at 1005.7).
   Worth A/B-ing a σ-equiv-without-aggressive variant if a submission
   slot opens up — but defer until v7 disposition is settled.
4. **Refresh `tests/fixtures/sample_live_replay.json.gz`** + re-pair
   `opponents/v3_snipe_frozen.py` so `test_replay_parity` can re-
   tighten its 0.9 floor → 1.0.

## Pointers (added this session)

- `agent.py` — single-file source of truth (~1100 LOC after all merges).
- `opponents/v3_snipe_frozen.py` — A/B baseline; bundled from `agents/v3_snipe`
  + `lib/` snapshot before deletion; sha256 `a4ac85b1cbf30838`.
- `scripts/kaggle_submit.py` — pre-submit self-play smoke + Kaggle CLI
  wrapper (replaces deleted bundler).
- `tests/test_v7_minimax.py` — 16 unit tests for the maximin picker +
  obs-swap path; passing.
- `tests/test_replay_parity.py` — now a ≥90% smoke gate; loads from
  `opponents/v3_snipe_frozen.py`.
- `audit/2026-05-12-postmortem-simplify-codebase-p39Hm.md` — today's
  postmortem (4 decisions reviewed, 5 frictions logged, 2 promotion
  candidates drafted).

## PR status

No PR. Branch is local-only via push to origin. PI to review before
opening one.
