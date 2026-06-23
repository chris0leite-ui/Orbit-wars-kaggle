# 2026-06-21 — Three-lens review (mathematician · software engineer · game expert)

Read-only review of the shipped `least_resistance` agent + the just-shipped
`LR_NEUTRAL_MARGIN=0.25` lever, requested by PI before the next iteration. Three
independent reviewers; this is the consolidated, de-duplicated synthesis. Plain
English (Rule 0). Nothing here is implemented yet — it is the menu for next steps.

## The convergent finding (all three reviewers, same place)

The losses (build a 2:1 production lead, then collapse) trace to ONE root cause seen
from three angles:

- **Game expert:** the leaf assumes the opponent *launches nothing*, and the default
  threat model `allocate` *splits* the enemy's mass across all our planets — so a
  95-ship stronghold reads as ~17 ships of threat everywhere. The agent literally
  cannot feel a concentrated punch, so holding a defensive reserve never wins the
  value comparison. We keep grabbing; V2 hoards a reserve and punches one seam.
- **Mathematician:** the live objective is a *linear* expected ship-margin. Being
  2:1 ahead scores 2x, but its win-probability is ~1.0 — the value is blind to the
  *convex* win condition. A linear margin happily trades a winning lead for marginal
  expansion. That linearity *is* the lead-then-collapse bug.
- **Both → same fix:** a **win-equity / lead-aware** value — concentrated worst-case
  threat (`max`) when ahead (defend the lead), variance-seeking when behind (keep the
  comeback). This unifies the lead-collapse fix AND the comeback-aggression tradeoff
  (the builder-goes-passive-when-behind problem) into one mechanism.

This is consistent with the margin-ignored ladder rating (only win/loss counts), which
makes P(win), not ship-margin, the literal objective.

## Bugs in the lever we just shipped (math + SWE converge)

- **Stale-eta over-sizing (math §1b, SWE M2):** the neutral surplus
  `margin*(ships + prod*eta)` uses the eta of the 20-ship *probe* fleet (slow,
  eta≈16.5), but the actual massed fleet is much faster (eta≈8-12). Surplus scales
  with eta → it **systematically over-sizes**. This is the mechanical reason 0.5
  over-massed. No fixed-point between size and eta.
- **`prod*eta` on a neutral is dimensionally wrong (math §1a):** a neutral garrison
  does NOT grow over transit (that term is correct only for enemy targets, where the
  owned planet produces). On a neutral it's a value-weighting trick wearing a
  defender-count costume. If value-scaling is intended, write it as a separate value
  term.
- **Shipped ON for 4P, untested (game expert §4):** bigger expansion fleets are more
  dangerous in 4P (idle mass = a window a third party punishes), the 4P horizon is
  shorter (13 vs 18) so the payoff is less visible, and it was tuned on 2P. Recommend
  2P-only or a 4P-specific value until a 4P seat-rotated A/B exists.
- **No test (SWE T1).**

## CRITICAL software issue

- **C1 — the bake block leaks config process-wide.** `os.environ.setdefault(...)` at
  import time (main.py:62-66) mutates the global environment, so any test/harness that
  imports `main.py` then loads another agent can silently inherit our config, and the
  shipped behavior is order-dependent. Fix: move the shipped values into the gate
  *accessor defaults* and delete the bake block — behavior becomes a property of the
  code, not a global side effect (and an explicit env var still overrides).

## Other should-fix software issues

- **S4 — swallowed strict raise.** `_native_strict` (default ON) is designed to make
  native-leaf errors RAISE rather than silently fall back to the weak ship-count move.
  But the outer `try/except` around the whole pick (main.py:1963) catches that raise
  anyway → we could be silently shipping the weaker fallback with no signal. The
  `_NATIVE_LEAF_CALLS` counter exists for exactly this but is never emitted. Narrow the
  catch and/or emit a one-line stderr marker when the fallback path is taken.
- **S3 — torch threads not pinned.** No `set_num_threads(1)` in the agent (only in
  render_game). Causes (a) nondeterminism (float reductions reorder by thread count,
  undermining the "pure function of obs" parity claim) and (b) timing unpredictability
  on the ~1.6-CPU ladder host. Pin to 1 at module load.
- **S1 — `_native_value` vs `_build_native_scorer` are verbatim-duplicated (~90 lines)
  and already subtly diverging.** They must stay in sync (builder leaf == chooser
  leaf). Extract one shared `_native_reach_tensors(obs, me)` helper + a consistency
  test.
- **S2 — `_concentrate` docstring lies:** it claims the scattered producer-floor move
  is DROPPED, but `producer_me` is unconditionally first in the menu (main.py:1902).
  Either implement the drop or fix the docstring (keeping the floor is the
  anti-passivity backstop, so probably fix the docstring).

## Strip candidates (SWE M1) — dead in the shipped config

- The entire **robust ensemble** (~300 lines: `_robust*`, `_make_robust_value`,
  `_stochastic_greedy`, `_sample_futures`, `_winprob_at`, `_leader_margin`,
  `_project_outcome`, the `LR_ROBUST_CAP` block) is gated off whenever
  `LR_NATIVE_LEAF=1` (shipped) → unreachable. **Strongest strip candidate** — but NOTE
  `_project_outcome` is the one place a rank/win-condition objective already exists
  (main.py:729-732); harvest that idea before deleting.
- The **deep-rollout** subsystem (`_deep_pick`, `_iterdeepen`, `_rollout_depth`, …) is
  OFF and refuted.
- `LR_GARRISON_FLOOR` block is dead at runtime; `_native_dynamic`/`_threat_max` etc.
  default-OFF.

## Other math notes

- **Offensive-potential double-weight (math §3):** `swing = is_opp*2 + is_neutral`
  added on top of the recurrence that already prices ownership; at off_weight=0.5 it
  can fully cancel the opponent's production drag from *reachability alone, with no
  commitment* → the documented passivity mode. Replace the additive term with a
  commit-attributed competitive potential (the term's own comment already names this).
- **Offense `cap` denominator asymmetry (math §3):** defense credits OUR reinforcement
  but offense uses the enemy's bare garrison (no enemy reinforcement) → optimistic.
- **Allocation softmax conservation VERIFIED correct** (math §2). Minor: proximity
  weighting goes inert on sparse-reach sources (use per-step reach for weighting).

## Recommended prioritization (for PI sign-off, after the live-ladder replay)

1. **Free correctness wins (low risk, do regardless of strategy):** C1 (environ leak),
   S3 (pin threads), S4 (don't swallow the strict raise / emit fallback marker), S2
   (docstring), T1 (neutral-margin test). None change the intended shipped behavior;
   they make it correct, deterministic, and observable.
2. **Fix the shipped lever's math:** neutral surplus — drop `prod*eta`, size the
   surplus from the *contest/threat* to the new planet (reuse `reachable_enemy_mass`):
   cheap fleets for safe neutrals, mass only for contested. Re-render the 06-21 seeds.
3. **The big one (highest payoff):** lead-gated win-equity value = concentrated
   worst-case threat when ahead + variance when behind. Both the game expert and the
   mathematician rank this #1; it targets the dominant loss signature AND the comeback
   tradeoff with one mechanism. Needs n>=32 seat-rotated A/B vs V2 before submit.
4. **De-risk 4P:** make `LR_NEUTRAL_MARGIN` 2P-only (or 4P-specific) until a 4P panel
   A/B; the highest-value 4P lever (placement/kingmaker-avoidance) is still unaddressed.
5. **Cleanup:** strip the robust + deep-rollout dead subsystems after harvesting the
   rank objective from `_project_outcome`.

## Do NOT re-try (refuted history, reaffirmed by the game expert)
deep rollout / longer horizon (refuted + hurt this session) · leader-relative 4P
(regressed 12->6) · enemy-boost (regressed 2P) · anytime/overage-bank (null) ·
full per-turn best-response opponent (paralysis) · NEUTRAL_MARGIN=0.5 (over-masses) ·
UNCONDITIONAL THREAT_MAX or NATIVE_BUILDER (over-defend; must be lead-gated).
