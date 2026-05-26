# 2026-05-26 PM — open questions after sa_online warm-start cycle

Branch `claude/competitive-programming-strategy-ESwSv`. Observations
only; answers belong in audits or future thoughts files.

1. **What μ does sub 53062327 (sa_v4, fate cache only) actually
   settle at?** Smoke vs random looked fine; live ladder unknown at
   session close.

2. **What μ does sub 53063161 (sa_v5, warm-start + noop opp) actually
   settle at?** Local smoke shows the agent plays but loses to peak;
   no calibration against real ladder mix.

3. **What's the cumulative μ cost of the rolling-pair-as-diagnostic
   loop today?** Four submissions consumed slots: 53059642 TIMEOUT,
   53061384 TIMEOUT, 53062327 likely ERROR-or-low, 53063161 unknown.
   Pre-session rolling pair floor was higher than current.

4. **Does the `_co_evolve` local-vs-Kaggle divergence affect any
   prior submission's parity claim?** Bundle smoke uses local
   co_evolve; Kaggle's blocked-co_evolve path may behave
   differently from what the smoke tested.

5. **Why does the closed-form `_capture_value` ranking diverge from
   SA's forward-sim score?** With opp_policy=nearest the divergence
   was severe (most positive-value warm-start emissions scored
   negative in the SA loop). With noop the divergence shrinks but
   isn't measured.

6. **Is the 1.6-CPU Kaggle box pinning at 1 core for our agent?**
   The host said "1 and 3/5 of a CPU." Our agent is single-threaded
   Python; the fractional CPU could mean cgroup CPU-time accounting
   rather than wall-time, which would affect the actTimeout
   interpretation.

7. **Does numba's JIT compile cost (~0.5-3s first call) fit in
   Kaggle's 60s overage budget on turn 1?** Numba is preloaded but
   I haven't measured.

8. **Were the strategic-head iteration session (earlier today) and
   this sa_online session on the same branch a coordination
   problem?** Two distinct workstreams sharing rolling-pair slots
   and live ladder state.

9. **What's the right granularity for warm-start re-fire across
   turns?** Currently fires only when `len(current_plan) < 3`.
   Mid-game, when carryover from prior turn's SA fills the plan,
   warm-start doesn't run — even if the carryover plan is stale
   given new observations.
