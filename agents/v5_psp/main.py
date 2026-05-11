"""v5_psp — Predictive Sim<K> Player.

Synthesis of improvements A+B+D+E from the cannot-lose plan:
  A — Predictive opp modeling: when scoring candidates, opp's policy
      in the Sim<K> rollout is v3_snipe (we model opp ≈ us). This
      eliminates the 1-turn-lag problem of reactive mirror.
  B — Sim<K>-filtered actions: enumerate candidates {v3 incumbent,
      ROI sibling strategy, plus drop-one variants of v3's launches}.
      Score each via env.clone() + K turns of v3-vs-v3 rollout
      (lib/lookahead.score_action). Pick the highest expected payoff.
  D — Mixed strategies on near-ties: when top-2 candidates are within
      τ_NEAR_TIE ship-units, sample one uniformly via a per-turn RNG.
      Prevents deterministic counter-exploitation.
  E — End-scenario routing (W1/W4/D1): inherits v4_endgame's
      pre-check; if a solved subgame fires, return its action without
      paying Sim<K> cost.

Compute budget: adaptive K. Start at K=50 (AUC 0.952 vs perfect
oracle per audit/2026-05-11-lookahead-phase2-forward-sim.md). After
each candidate evaluation, check elapsed wallclock; if > 700ms, drop
remaining sims to K=30. Hard cap MAX_CANDIDATES=6 to bound worst case.

In 4P: end-scenario check still fires (W1/W4/D1 work in n-player), but
no Nash guarantee in n≥3. Fall back to v3_snipe with end-scenarios.

Plan reference: /root/.claude/plans/you-are-a-top-parallel-swan.md
"""

from __future__ import annotations

import importlib.util
import random
import time
from pathlib import Path

from lib.lookahead import enumerate_drop_one_candidates, env_from_obs, score_action


_REPO = Path(__file__).resolve().parents[2]
_V3_AGENT = None
_ROI_AGENT = None
_ENDGAME = None

# Sim<K> budget knobs
K_HIGH = 50
K_LOW = 30
WALLCLOCK_GUARD_MS = 700.0  # downshift to K_LOW after this much elapsed
MAX_CANDIDATES = 6
TAU_NEAR_TIE = 5.0  # ship-units; top-2 within τ → mixed sample


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _load(name: str, relpath: str):
    """Lazy file-path loader for sibling agents (avoids package imports)."""
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _v3():
    global _V3_AGENT
    if _V3_AGENT is None:
        _V3_AGENT = _load("_psp_v3", "agents/v3_snipe/main.py").agent
    return _V3_AGENT


def _roi():
    global _ROI_AGENT
    if _ROI_AGENT is None:
        _ROI_AGENT = _load("_psp_roi", "agents/simple/roi.py").agent
    return _ROI_AGENT


def _endgame_decision(obs, my_id: int):
    """Delegate to v4_endgame's W1/W4/D1 check. Returns
    'coast'/'freeze'/'default'."""
    global _ENDGAME
    if _ENDGAME is None:
        _ENDGAME = _load("_psp_endgame", "agents/v4_endgame/main.py")
    return _ENDGAME._end_scenario(obs, my_id)


def _detect_num_players(planets) -> int:
    return len({p[1] for p in planets if p[1] != -1})


def _candidate_actions(obs, v3_action: list, roi_action: list) -> list[list]:
    """Build the candidate set we'll score via Sim<K>.

    {v3 incumbent} ∪ {ROI sibling action} ∪ {drop-one variants of v3}.
    Deduplicated by repr; capped at MAX_CANDIDATES.
    """
    seen = set()
    out: list[list] = []
    for cand in [v3_action, roi_action] + enumerate_drop_one_candidates(v3_action):
        key = repr(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= MAX_CANDIDATES:
            break
    return out


def _score_candidates(env, my_id: int, candidates: list[list], policy):
    """Score each candidate via Sim<K> with adaptive K. Returns list of
    (score, candidate) sorted descending by score."""
    t0 = time.monotonic()
    scored: list[tuple[float, list]] = []
    for cand in candidates:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        K = K_LOW if elapsed_ms > WALLCLOCK_GUARD_MS else K_HIGH
        try:
            s = score_action(env, cand, K=K, my_id=my_id, policy=policy)
        except Exception:
            # Defensive: if a candidate breaks the rollout, treat it
            # as the worst possible score so we never pick it.
            s = float("-inf")
        scored.append((s, cand))
    scored.sort(key=lambda kv: -kv[0])
    return scored


def _pick_with_mix(scored: list[tuple[float, list]], rng: random.Random) -> list:
    """Pick the top scorer; uniform-sample from the top-tier near-ties."""
    if not scored:
        return []
    top_score = scored[0][0]
    tier = [cand for s, cand in scored if (top_score - s) <= TAU_NEAR_TIE]
    if len(tier) == 1:
        return tier[0]
    return rng.choice(tier)


def agent(obs):
    my_id = int(_obs_get(obs, "player", 0))
    planets = _obs_get(obs, "planets", []) or []

    # End-scenario gate (cheap; ~ms). If a solved subgame fires, skip
    # Sim<K> entirely.
    scenario = _endgame_decision(obs, my_id)
    if scenario in ("coast", "freeze"):
        return []

    # 4P fallback — no NE guarantee in n≥3; just play v3.
    if _detect_num_players(planets) != 2:
        return _v3()(obs)

    # Generate candidates from incumbent strategies.
    v3_action = _v3()(obs)
    try:
        roi_action = _roi()(obs)
    except Exception:
        roi_action = v3_action  # if roi breaks for some reason, dedup'd out

    candidates = _candidate_actions(obs, v3_action, roi_action)

    # If only one candidate (e.g., v3 emitted nothing and dedup collapsed all
    # variants), skip Sim<K> — there's nothing to compare.
    if len(candidates) <= 1:
        return candidates[0] if candidates else []

    # Rebuild a steppable env mirroring the current state.
    try:
        env = env_from_obs(obs)
    except Exception:
        return v3_action  # if env rebuild fails, fall back safely

    # Sim<K> scoring with v3 as the rollout policy for both sides.
    # (This is the "predictive opp modeling" — opp plays v3 in the
    # rollout, so we choose actions that beat v3 in expectation.)
    scored = _score_candidates(env, my_id, candidates, policy=_v3())

    # Per-turn RNG seeded by step + player so two PSPs in self-play
    # randomize independently but each is reproducible per game.
    step = int(_obs_get(obs, "step", 0))
    rng = random.Random((step + 1) * 1009 + (my_id + 1) * 17)
    return _pick_with_mix(scored, rng)
