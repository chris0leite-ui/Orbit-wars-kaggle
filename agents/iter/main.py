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
VALUE_FN = "composite"          # "default" | "composite" | "defensibility" | "composite_plus_defensibility"
                                # | "territory" | "composite_plus_territory"
DEFENSIBILITY_ALPHA = 0.2       # SMALL coefficient — V2 α=1.0 over-penalised; V3 uses defens as tiebreaker only
TERRITORY_WEIGHT = 0.01         # production×hold sums to ~5k-10k mid-game; 0.01 keeps the term ≈ ±50, comparable to delta
K_4P = 8                        # 4P-branch lookahead (choose_4p default); kept separate from K (2P)
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

from lib.v7_search import choose, choose_4p
from lib.intent import World


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
    if name == "territory":
        from lib.value_heads import territory_value
        return lambda obs, mid: territory_value(obs, mid, weight=TERRITORY_WEIGHT)
    if name == "composite_plus_territory":
        from lib.value_heads import composite_plus_territory
        return lambda obs, mid: composite_plus_territory(
            obs, mid, territory_weight=TERRITORY_WEIGHT
        )
    raise ValueError(f"unknown VALUE_FN: {name!r}")


def _detect_num_seats(world) -> int:
    """Infer seat count from owner IDs on planets + in-flight fleets.

    Inline reimplementation of `lib.v7_search._infer_num_seats` so we don't
    reach into a private helper. 2P if max non-neutral owner ID is ≤ 1;
    4P if max is ≥ 2.
    """
    max_id = -1
    for p in world.planets_by_id.values():
        if p.owner >= 0 and p.owner > max_id:
            max_id = p.owner
    raw = world.obs_raw
    fleets = raw.get("fleets", []) if isinstance(raw, dict) else getattr(raw, "fleets", [])
    for f in fleets:
        owner = int(f[1])
        if owner >= 0 and owner > max_id:
            max_id = owner
    return 4 if max_id >= 2 else 2


def agent(obs, configuration=None):
    # ------------------------------------------------------------------
    # PRE-FILTER HOOK — mutate `obs` or short-circuit before the chooser.
    # Examples:
    #   - if obs.step > 480 and we're locked: return []  (no late-game thrash)
    #   - mask out comet targets if a "drop comets near end" experiment.
    # Keep edits scoped to <10 lines; bigger changes go in lib/.
    # ------------------------------------------------------------------

    # Dispatch: 2P uses iter_v1's validated choose(enumerator_mode="drop_one",
    # opp_tiers=[1]) path. 4P uses choose_4p() instead of falling back to the
    # v3.5.1 incumbent (the pre-fix behaviour). choose_maximin is NOT used in
    # 2P because v7.1+ maximin variants regressed historically; we preserve
    # iter_v1's 2P behaviour exactly while adding 4P competence.
    world = World.from_obs(obs)
    n_seats = _detect_num_seats(world)
    value_fn = _resolve_value_fn(VALUE_FN)
    if n_seats == 4:
        action = choose_4p(
            obs, configuration,
            K=K_4P,
            wallclock_ms=WALLCLOCK_MS,
            include_recapture=True,
            value_fn=value_fn,
        )
    else:
        action = choose(
            obs, configuration,
            enumerator_mode=ENUMERATOR_MODE,
            K=K,
            wallclock_ms=WALLCLOCK_MS,
            opp_tiers=list(OPP_TIERS),
            value_fn=value_fn,
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
