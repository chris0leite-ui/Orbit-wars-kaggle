# 2026-05-18 — Asymmetric chooser, CRN, and the recon-side rule gap

Session: claude/reverse-engineer-seat-geometry-BPJKs.

## The single biggest lesson

The chooser computes `Δ = leaf(me_action, opp_traj) -
leaf(me_idle, opp_traj)`. Both legs MUST use the SAME `opp_traj`
(common random numbers). The CRN cancels noise from the simulated
opp; what you measure is the marginal value of YOUR action, not the
absolute leaf-value of the rollout. The v11→v12→v13 progression on
this codebase paid for exactly that lesson:

- v11: idle baseline applied one-shot opp mirror at step 0 then idled
  → asymmetric (me_action + opp_step0_only vs me_idle + opp_step0_only).
- v12: pre-computed full opp trajectory once, applied identically in
  baseline + every candidate → symmetric. **Same opp behaviour in
  both legs of the Δ.**
- v13: dropped pre-computation; opp_policy called per-step in both
  legs → fully reactive symmetric.

My session-2026-05-18 change (now reverted) put Tier-1 aggressive opp
in `build_idle_baseline` and kept lite_greedy passive in `score_action`.
This reintroduced the v11 asymmetric-Δ failure with the polarity
flipped: baseline now under-counted my idle position (aggressive opp
grabs my stuff), action over-counted my action (passive opp doesn't
punish me). Every Δ pegged positive; chooser launched everything that
survived the proposer's cheap-rank ≥ -10 filter; the agent blasted
ships at bad targets and lost 32 / 32 panel games.

The audit diagnosis itself was correct (lite_greedy is more passive
than real top-10 opp; chooser is too passive). The fix was wrong
because it broke a constraint the previous fix line had specifically
established. **The methodologically correct fix is symmetric stronger
opp** — same model in both legs, just better-aligned to ladder
behaviour. That's the next-session work.

## The shape of the bug

If you find yourself proposing "different X in baseline vs
score_action," step back. The Δ semantics REQUIRE symmetry. Any
asymmetry is a bug.

Same applies to:
- Different `horizon` in baseline vs action (already handled — the
  baseline is computed for every h 0..max_horizon, action picks the
  h that matches its eta + settle).
- Different value function (`favor` vs `composite`) — the chooser
  uses `select_favor_fn()` for both legs by design.
- Different gamma — passed through identically.

## The recon-side rule gap

This session had two same-shape failures: code-change before reading
state docs.

1. **State doc miss:** `state/current.md` says our submission is
   `agents/baseline/`. I started recon at `data/main.py`. Two rounds
   wasted before PI caught.

2. **Audit history miss:** the chooser change had a documented prior
   solution in `audit/2026-05-17-state-function-principled-fix-results.md`.
   I didn't grep `audit/` for chooser-related notes before designing
   the change. Panel: 0/32.

The bootstrap-side version of this pattern (Rule 38, SessionStart
hook) is already in place. The recon-side version isn't. Promotion
candidate is drafted; awaiting next cycle.

## What's actually correct on this branch

The audits in the first half of the session stand. The behavioural
gap between top-10 and our submission is real:
- `launches_per_turn`: ours 0.5 vs top-10 1.1 (universal: d=+1.26 in
  GAP cells, d=+1.25 in EVEN cells)
- `mean_garrison_at_launch`: ours 15 vs top-10 10 (universal: d=-0.56
  in GAP, d=-0.82 in EVEN)
- `targets_neutral_fraction`: ours 0.40 vs top-10 0.27 (universal)
- `multi_launch_turn_rate`: ours 0.29 vs top-10 0.47 (mixed, gap-
  cell stronger)

The root-cause analysis (chooser models its opp as
`lite_greedy_policy` which is structurally less aggressive than real
top-10) is also correct. The fix has to be a stronger opp model
applied SYMMETRICALLY. The two candidates:

- **Vectorise `top_tier_mirror_policy`** to ~1ms per call (current
  ~10ms is from rebuilding `WorldModel` every step; cache it on the
  snap). Then wire it as the opp_policy everywhere.
- **Train Tier-2 logreg** (the placeholder at
  `lib/opp_model.py:128-140`). ≤200-float weights on the 37k labeled
  shots in `data/shot_validator/`. Fast inference, top-10-behaviour-
  matched by construction.

Both should be tried; Tier-2 is the higher-ceiling option.

## Open question for next session

Is the audit-trail dependence on
`audit/2026-05-17-state-function-principled-fix-results.md` itself a
sign that audit/ is hard to grep? The file is 7th-most-recent on the
chooser topic. A `grep -l chooser audit/2026-05-*.md` would have
surfaced it in <5 seconds. The question is whether the recon-rule
draft would have actually triggered me to run that grep, or whether
the "I've read state/current.md already" check would have felt
sufficient. The promotion candidate phrases the rule to require BOTH
state docs AND `grep audit/` — that's the right form, but a single-
phrase rule that's actually applied beats a thorough rule that's
sometimes skipped. Worth a re-read after the next promotion cycle.
