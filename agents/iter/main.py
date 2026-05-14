"""iter — fast-iteration fork of v7_pv (= v7_0_drop_one + PV_GAMMA=0.99).

Day-zero behaviour: functionally equivalent to v7_pv (ladder mu=1064.4).
Edit the knobs below to A/B variants; add code under the PRE_FILTER /
POST_PROCESS hooks to fix specific bugs observed in live replays.

Eval cycle:
    python fast.py eval iter --vs-panel default --max-seeds 32       # 2P
    python -m scripts.ffa_panel --focals agents/iter/main.py --seeds 32   # 4P

Bundle + parity:
    python scripts/bundle_agent.py agents/iter
    pytest tests/test_iter_agent.py tests/test_bundle.py -q

See agents/iter/README.md for the four patch surfaces and submission gate.
"""

from __future__ import annotations

# ============================================================================
# ITER KNOBS — edit these for a knob sweep, one line per variant.
# ============================================================================
K = 10                          # lookahead horizon (8 / 10 / 12 / 15)
WALLCLOCK_MS = 700.0            # per-turn budget (ms)
ENUMERATOR_MODE = "drop_one"    # see lib.v7_search proposers
OPP_TIERS = (1,)                # opp-model tier(s); >1 entry => MAXIMIN
PV_GAMMA = 0.99                 # 1.0 = v7_0_drop_one; 0.99 = v7_pv equivalent
VALUE_FN = "composite"          # V1 — composite_capture_value (waste-aware) on stock v7_0_drop_one chooser
DEFENSIBILITY_ALPHA = 0.2       # SMALL coefficient — V2 α=1.0 over-penalised; V3 uses defens as tiebreaker only
# ============================================================================

# Dev-mode: override lib.scoring.PV_GAMMA BEFORE v7_search imports propagate
# the `from lib.scoring import PV_GAMMA` bindings into snipe/reinforce.
# Bundled form: lib.scoring is not a separate module (concatenated above),
# so this import raises ImportError and we rely on the module-scope
# PV_GAMMA rebind above — every callsite looks PV_GAMMA up by name at
# call time (verified across snipe.py + reinforce.py).
try:
    import lib.scoring as _scoring
    _scoring.PV_GAMMA = PV_GAMMA
except ImportError:
    pass

from lib.v7_search import choose


def _resolve_value_fn(name):
    if name == "default":
        return None  # lib.v7_search.score_candidate defaults to delta_us_minus_them
    if name == "composite":
        from lib.value_heads import composite_capture_value
        return composite_capture_value
    if name == "defensibility":
        from lib.value_heads import defensibility_value
        return lambda obs, mid: defensibility_value(obs, mid, weight=DEFENSIBILITY_ALPHA)
    if name == "composite_plus_defensibility":
        from lib.value_heads import composite_plus_defensibility
        return lambda obs, mid: composite_plus_defensibility(
            obs, mid, defensibility_weight=DEFENSIBILITY_ALPHA
        )
    raise ValueError(f"unknown VALUE_FN: {name!r}")


def agent(obs, configuration=None):
    # ------------------------------------------------------------------
    # PRE-FILTER HOOK — mutate `obs` or short-circuit before the chooser.
    # Examples:
    #   - if obs.step > 480 and we're locked: return []  (no late-game thrash)
    #   - mask out comet targets if a "drop comets near end" experiment.
    # Keep edits scoped to <10 lines; bigger changes go in lib/.
    # ------------------------------------------------------------------

    action = choose(
        obs, configuration,
        enumerator_mode=ENUMERATOR_MODE,
        K=K,
        wallclock_ms=WALLCLOCK_MS,
        opp_tiers=list(OPP_TIERS),
        value_fn=_resolve_value_fn(VALUE_FN),
    )

    # ------------------------------------------------------------------
    # POST-PROCESS HOOK — sanitise the action before returning.
    # Examples:
    #   - filter out launches whose trajectory leaves the play area
    #     (use lib.trajectory to project landing position)
    #   - cap fleet sizes when projected ROI < threshold
    #   - drop all launches in the final N steps if score is locked
    # ------------------------------------------------------------------

    return action
