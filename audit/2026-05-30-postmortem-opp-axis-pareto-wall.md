# Postmortem — 2026-05-30 kaggle-submission-review-gZsCu

Session arc: read the PV_ETA-anchor handover → traced the chooser's
internal scoring (`audit/2026-05-29-pm-pv-eta-noop-emit-analysis.md`)
→ inventoried agent capabilities → ran four asymmetric paired-seed
A/Bs probing the opp-model axis (nearest, top_tier_mirror, v7_0,
no-launch as opp policy) → mapped the cost/quality Pareto wall at
the 1000ms turn cap → flagged for future sessions.

## What went wrong

- **Anchor-self-play env-var routing was structurally symmetric** —
  designed the first nearest A/B as "set BASELINE_OPP_MODEL=nearest
  in parent shell, anchor self-play." Both `env.run([p0, p1])` seats
  inherit os.environ → both run nearest → game is structurally
  symmetric → expected winrate 50% by construction. PI caught it
  with "how are you comparing?" before n=5 finished. ~12 min compute
  averted; mental model corrected.

- **Mirror 5-0 was almost reported as a strategy null** — drew the
  "top_tier_mirror is too attack-biased" conclusion from the
  prior-session code comments, without instrumenting. PI prompted
  "diagnose closely to make sure there is no bug." Confound check
  (`scripts/verify_mirror_bake.py`) revealed mirror is 5-10× slower
  per call → chooser's affordable_validate_cap shrinks per-turn
  evals to ~80 vs lite_greedy's ~1,100 → P1 is search-starved,
  not strategy-mispredict. Without PI prompt, the wrong interpretation
  would have shipped into the audit. Rule 41 (confound-sweep before
  correlational conclusion) was directly applicable; not applied
  proactively.

- **v7_0-as-opp clarification took two pings** — PI asked "what
  about v7 drop one as opponent?" I launched it as the *external
  opponent* in a new A/B. PI clarified "i mesnt v7 drop one as
  opponent model instead of light greedy." Cost: ~10 min of false
  start, including a partial A/B run. The original question used
  "opponent" / "opponent model" ambiguously; I picked the literal
  reading. Lesson: when PI's term is ambiguous between two
  experiments and one is cheaper to verify, ask before launching.

## Frictions logged this session

See `audit/friction.md::2026-05-30` for full one-liners:

- `anchor-self-play-symmetric-when-env-var-shared` — structural
  symmetric A/B design.
- `opp-model-confound-search-starvation` — mirror 5-0 was budget,
  not strategy.
- `skip-the-bench-when-cost-is-the-answer` — v7_0 A/B avoided by
  pre-bench.
- `stale-monitor-spam` — cosmetic, long-armed monitors emitted
  timeout notifications hours after their tasks finished.

## Promotion candidates (PI ratified: no)

Three rule drafts presented to PI for promotion to
`.claude/skills/kaggle-comp/improvements.md`:

1. **Asymmetric A/Bs require baked asymmetry, not env-var routing**
   — generalizes the anchor-self-play symmetric trap to any
   "swap one mechanism between sides" test.
2. **Log per-turn invocation counts for cost-asymmetric A/Bs** —
   generalizes the mirror confound detection to any chooser-internal
   policy comparison.
3. **Bench-then-decide for documented-heavy variants** — generalizes
   the v7_0 bench-as-experiment approach for any variant with
   ≥5× baseline per-call cost.

**PI decision:** no promotions. Friction one-liners stay in
`audit/friction.md` only. Re-promotion candidate if same patterns
fire next session.

## PI additions (from "anything to add?" step)

None.

## What this session shipped

- `lib/opp_model.py::nearest_opp_policy` — new policy in source tree
  (cheap; same cost class as lite_greedy).
- `agents/baseline/chooser.py::_select_opp_policy` — extended dispatch
  to route `BASELINE_OPP_MODEL=nearest`.
- `tests/test_opp_model_nearest.py` — 5 tests, 5/5 green.
- 3 baked-bundle variants (gitignored under `submissions/*` per
  convention): `baseline_pv_eta_nearest_opp.py`,
  `baseline_pv_eta_mirror_opp.py`, `baseline_pv_eta_nolaunch_opp.py`,
  `baseline_pv_eta_v7_0_opp.py`. Reproducible from anchor + one-function
  patch.
- `scripts/trace_pv_eta_scoring.py`, `scripts/analyze_pv_eta_trace.py`,
  `scripts/verify_mirror_bake.py`, `scripts/bench_v7_0_opp_bake.py`
  — tooling for chooser-internal instrumentation, retained for reuse.
- `audit/2026-05-29-pm-pv-eta-noop-emit-analysis.md` and
  `audit/2026-05-29-pm-nearest-opp-model-n5-directional-null.md` —
  load-bearing finding records.
- `knowledge-base/thoughts/2026-05-30-opp-axis-compute-pareto-wall.md`,
  `knowledge-base/questions/2026-05-30-seed-3493-no-belief-beats-lite-greedy.md`,
  `knowledge-base/flags/2026-05-30-opp-axis-compute-pareto-wall.md`
  — second-brain entries per Rule 36.

## Decisions worth noting (good outcome from good decision)

- Killed the first (symmetric) A/B immediately when PI flagged the
  design; did not let it run to completion to "see what happens."
- Built the verify_mirror_bake instrumentation as soon as the
  confound check question landed; did not try to reason about the
  bake correctness from the bundled-file diff.
- Skipped the v7_0 n=5 A/B based on the bench result, after
  explicitly framing the alternative ("we could burn 20-40 min on
  a mechanically-determined result, or accept the bench is the
  experiment").

## What I'd flag for next session

- **Spatial-restricted lite_greedy** is the cheapest untested
  opp-model variant that directly probes PM3's "expects opponents
  from everywhere" diagnosis. If PM3's direction is still alive,
  this is the next test.
- **Seed 3493 trace** is the cheapest disambiguator if the
  spatial-restricted A/B comes back ambiguous.
- **MLP-validated opp** (`BASELINE_OPP_MODEL=mlp`) needs a per-call
  bench before n=k A/B. The MLP forward pass might be light
  (small net) or might trip the 2× confound threshold.

## Framework version at session-end

- Commit SHA: `9f887a28b5a10f8cf1685ae19f6260985c49fe64`
- Active rules: 1..48 (CLAUDE.md `## Operating rules — concise`).
- Loaded skills this session: `postmortem` (this), `kaggle-comp`
  (implicit, via CLAUDE.md context).
