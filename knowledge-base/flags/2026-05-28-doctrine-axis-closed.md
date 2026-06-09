# 2026-05-28 — Flag: doctrine axis closed, do not re-open

Status as of session end:

- Reach-frontier doctrine as a baseline replacement: **falsified**
  three ways (Rule 37 cap hit on this axis).
- Doctrine MATH stays as durable knowledge (production-time integral
  is the TrueSkill-tiebreak objective).
- Doctrine PRESCRIPTIONS (closed-form ρ chooser, hold-floor +
  gang-up, 4P delayed launch) all empirically lose to baseline.

Future-session triggers that should warrant re-opening:

- If a fundamentally different opp model (Bayesian / probabilistic
  ρ_opp, not max-over-opp) is proposed: that's a NEW axis, not a
  continuation. Re-open requires counter-experiment first per the
  thoughts entry today.
- If a top-10 self-play sample materialises (the μ ≥ 1500 gap from
  `audit/2026-05-27-between-band-stratification.md`): the share
  signal beyond μ=1400 is still untested. New data could shift
  the verdict for the descriptive doctrine, but not for the
  prescriptive operationalisations already falsified.

Don't bother iterating:
- λ_loss / λ_risk knob sweeps.
- A 3rd cushion-step value (we have 4/32 at 60 ticks vs
  baseline-current; that gap is too big for a small tweak to close).
- Reach-frontier `validate_physics` policy changes.

The `agents/reach_frontier/` and `agents/baseline_4p_cushion/`
wrappers stay in tree for "reproduce the null" reference, NOT as
submission candidates. Any submit attempt would evict the rolling
pair for a known-loser variant — don't.

## Bundler regressions to watch

`scripts/bundle_agent.py` CLI parity gate broke on the new modular
agent due to kaggle_environments sys.path mutation. Worked around
via `--skip-parity-gate` + `tests/test_bundle.py` fixture. If a
future agent adds a similar import pattern, the CLI parity gate
will re-fail — patch `_parity_gate` to invalidate import caches
before loading the source.
