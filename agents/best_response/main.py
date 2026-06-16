"""best_response — run the producer once, then simulate the best reply.

The idea (PI, 2026-06-16): the Producer is a strong but expensive policy.
Instead of calling it inside a search loop (the move that boxed every prior
search wrapper into a 3-step horizon — see the dead-end note in HANDOVER.md),
we call it ONCE per seat per turn:

  1. Run the Producer for our own seat  -> its recommended plan (a strong
     default, and the seed for our candidate moves).
  2. Run the Producer for each opponent seat -> what the Producer would do
     AS that opponent this turn (our opponent model).

Then we hand the cheap, parity-tested engine (lib.fast_sim) the work the
Producer is too slow to do in a loop: for a SPARSE set of candidate first
moves of our own, we forward-simulate the turn (us = candidate, opponents =
their predicted Producer move) and let it settle over the Producer's own
planning horizon (18 steps — the window the Producer has converged to).  We
keep the candidate whose simulated position is best.  That is a depth-1
best response to the Producer, scored by an 18-step rollout.

Why this can beat the Producer where prior wrappers tied:
  - The Producer is run O(1) times, not O(horizon), so the horizon can be 18
    instead of 3 — long enough for launched fleets to actually arrive and
    captures to resolve.
  - The candidate set is genuinely diverse (the plan, its prefixes, drop-one
    variants, and a MORE-aggressive expansion variant), not just
    {plan, idle, drop-biggest}.  The known Producer weakness is
    under-expansion; an expansion candidate that the 18-step sim prefers is
    exactly the fix the loss-mining wanted, found by simulation instead of by
    a hand-tuned flag.
  - Whatever happens, we fall back to the Producer's own move, so in the
    worst case we play the Producer.

Tunable via environment variables (all optional; defaults are sensible):
  BR_HORIZON   (int,   default 18)     simulation depth = Producer horizon
  BR_MAX_CANDS (int,   default 24)     cap on candidates evaluated per turn
  BR_TAIL      (str,   default greedy) rollout tail policy: greedy | coast
  BR_OPP       (str,   default producer) opp step-0 model: producer | greedy
  BR_W_PLANETS (float, default 4.0)    weight on planet-count delta in value
  BR_SOFT_MS   (float, default 700)    soft wallclock budget; stop searching past it

NOTE (provenance): this agent uses the vendored third-party Producer
(agents/producer, see its PROVENANCE.md) as both opponent model and candidate
source.  It is a LOCAL research/eval build.  Do not submit without resolving
the Producer's redistribution/licensing question with the PI first.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from typing import Any, Callable

# --------------------------------------------------------------------------
# Path + import bootstrap (works in-place and when loaded by file path).
# --------------------------------------------------------------------------
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # exec'd with no __file__
    _HERE = os.getcwd()
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PRODUCER_DIR = os.path.join(_REPO, "agents", "producer")
for _p in (_REPO, _PRODUCER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

# Mirror the harness convention: cap threads so local parallel A/B timing is
# realistic (~1.6 CPU on Kaggle).  Harmless on the real ladder.
try:
    torch.set_num_threads(int(os.environ.get("BR_TORCH_THREADS", "2")))
except Exception:
    pass

from lib import fast_sim  # noqa: E402
from lib.opp_model import lite_greedy_policy  # noqa: E402


def _load_producer():
    """Load agents/producer/main.py under a unique module name.

    Mirrors agents/producer/producer_agent.py: loading it as the generic
    name ``main`` would collide in ``sys.modules`` when several agents are
    loaded in one process (the fast.py / panel case).
    """
    if "producer_main" in sys.modules:
        return sys.modules["producer_main"]
    spec = importlib.util.spec_from_file_location(
        "producer_main", os.path.join(_PRODUCER_DIR, "main.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["producer_main"] = mod
    spec.loader.exec_module(mod)
    return mod


_P = _load_producer()


# --------------------------------------------------------------------------
# Config (read once from the environment).
# --------------------------------------------------------------------------
def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


HORIZON = _envi("BR_HORIZON", 18)
MAX_CANDS = _envi("BR_MAX_CANDS", 24)
TAIL = os.environ.get("BR_TAIL", "greedy").strip().lower()
OPP_MODEL = os.environ.get("BR_OPP", "producer").strip().lower()
W_PLANETS = _envf("BR_W_PLANETS", 4.0)
SOFT_MS = _envf("BR_SOFT_MS", 700.0)


# --------------------------------------------------------------------------
# Producer invocation helpers.
# --------------------------------------------------------------------------
def _producer_move(rt, obs: Any, pid: int) -> list:
    """One Producer inference for seat ``pid`` on ``obs``.

    ``single_obs_to_tensor`` takes ``player_id`` explicitly and reads
    ownership absolutely, so we get seat-``pid``'s move without mutating obs.
    """
    ot = _P.single_obs_to_tensor(obs, player_id=pid)
    with torch.no_grad():
        row = rt.tensor_action(ot)
    return _P.sparse_action_row_to_moves(row, obs, player_id=pid)


def _obs_get(obs: Any, key: str, default: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _infer_num_seats(obs: Any) -> int:
    """2 or 4, from the highest owner id present (stable from step 0:
    every player owns its home group at the start)."""
    mx = 0
    for p in _obs_get(obs, "planets", []) or []:
        if len(p) >= 2 and int(p[1]) >= 0:
            mx = max(mx, int(p[1]))
    for f in _obs_get(obs, "fleets", []) or []:
        if len(f) >= 2 and int(f[1]) >= 0:
            mx = max(mx, int(f[1]))
    return 4 if mx >= 2 else 2


# --------------------------------------------------------------------------
# Candidate generation — the sparse action set.
# --------------------------------------------------------------------------
def _dedup(cands: list[list]) -> list[list]:
    seen: set = set()
    out: list[list] = []
    for c in cands:
        key = tuple(sorted((int(w[0]), round(float(w[1]), 4), int(w[2])) for w in c))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _merge_by_source(base: list, extra: list) -> list:
    """Union of two launch lists, one wave per source planet (extra wins).

    A source can only launch what it owns, so we don't want two competing
    waves from the same planet in one candidate (the second would be clipped
    by the engine's ownership check).  Prefer the ``extra`` (more-aggressive)
    wave when both touch the same source.
    """
    by_src: dict[int, list] = {}
    for w in base:
        by_src[int(w[0])] = w
    for w in extra:
        by_src[int(w[0])] = w
    return list(by_src.values())


def _build_candidates(base_plan: list, my_obs: Any) -> list[list]:
    """Sparse, diverse candidate first-moves derived from the Producer plan.

    Includes: the plan itself, idle (coast), greedy prefixes (commit the top-k
    waves), drop-one defensive variants, and aggressive expansion variants
    (the plan unioned with a cheap ROI-greedy grab the Producer declined).
    """
    cands: list[list] = [list(base_plan), []]

    # Greedy prefixes — the Producer orders waves best-first, so a prefix is
    # "commit only the top k".  Lets the sim decide how hard to push.
    for k in range(1, len(base_plan)):
        cands.append(list(base_plan[:k]))

    # Drop-one — hold a single wave back (defensive / conserve a source).
    if len(base_plan) >= 2:
        for i in range(len(base_plan)):
            cands.append([w for j, w in enumerate(base_plan) if j != i])

    # Expansion variants: the Producer's known weakness is under-expansion.
    # Offer the sim a more-aggressive grab set; if it over-extends, the
    # 18-step rollout will punish it, so this is safe to propose.
    try:
        aggressive = lite_greedy_policy(my_obs)
    except Exception:
        aggressive = []
    if aggressive:
        cands.append(_merge_by_source(base_plan, aggressive))  # plan + extra grabs
        cands.append(list(aggressive))                          # fully aggressive
        # plan + ONE extra grab at a time (find the single best add-on).
        base_srcs = {int(w[0]) for w in base_plan}
        extras = [w for w in aggressive if int(w[0]) not in base_srcs]
        for w in extras[:6]:
            cands.append(list(base_plan) + [w])

    return _dedup(cands)


# --------------------------------------------------------------------------
# Rollout + leaf value.
# --------------------------------------------------------------------------
def _tail_policy_actions(snap, num_seats: int) -> list:
    if TAIL == "coast":
        return [[] for _ in range(num_seats)]
    return [lite_greedy_policy(snap.state[i].observation) for i in range(num_seats)]


def _rollout(snap, k: int, num_seats: int):
    s = snap
    for _ in range(k):
        if s.fake_env.done:
            break
        s = fast_sim.step(s, _tail_policy_actions(s, num_seats), in_place=True)
    return s


def _leaf_value(snap, me: int, num_seats: int) -> float:
    """Position value for seat ``me``: ship lead over the strongest rival,
    plus a planet-count lead term (favours expansion / durable holdings)."""
    obs0 = snap.state[0].observation
    ships = [0.0] * num_seats
    planets = [0.0] * num_seats
    for p in obs0.planets:
        ow = int(p[1])
        if 0 <= ow < num_seats:
            ships[ow] += float(p[5])
            planets[ow] += 1.0
    for f in obs0.fleets:
        ow = int(f[1])
        if 0 <= ow < num_seats:
            ships[ow] += float(f[6])
    others = [i for i in range(num_seats) if i != me]
    ship_lead = ships[me] - (max(ships[i] for i in others) if others else 0.0)
    planet_lead = planets[me] - (max(planets[i] for i in others) if others else 0.0)
    return ship_lead + W_PLANETS * planet_lead


# --------------------------------------------------------------------------
# Per-episode state: one persistent Producer runtime per seat.
# --------------------------------------------------------------------------
class _State:
    def __init__(self) -> None:
        self.runtimes: dict[int, Any] = {}
        self.num_seats: int | None = None
        # Diagnostics (read by probes; zero per-turn cost).
        self.turns = 0
        self.deviations = 0
        self.gain_sum = 0.0

    def runtime(self, seat: int):
        rt = self.runtimes.get(seat)
        if rt is None:
            rt = _P.ProducerLiteRuntime()
            self.runtimes[seat] = rt
        return rt

    def reset(self) -> None:
        for rt in self.runtimes.values():
            rt.reset()
        self.num_seats = None


_STATE = _State()


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------
def _decide(obs: Any, configuration: Any = None) -> list:
    t0 = time.perf_counter()
    me = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    if step == 0:
        _STATE.reset()
    if _STATE.num_seats is None:
        _STATE.num_seats = _infer_num_seats(obs)
    num_seats = _STATE.num_seats

    # 1. Run the Producer once for our seat -> base plan (+ candidate seed).
    base_plan = _producer_move(_STATE.runtime(me), obs, me)

    # 2. Run the Producer once per opponent seat -> opponent model.
    opp_actions: dict[int, list] = {}
    for j in range(num_seats):
        if j == me:
            continue
        if OPP_MODEL == "greedy":
            opp_actions[j] = lite_greedy_policy(obs)
        else:
            opp_actions[j] = _producer_move(_STATE.runtime(j), obs, j)

    # 3. Sparse best-response search over the 18-step horizon.
    candidates = _build_candidates(base_plan, obs)[:MAX_CANDS]
    snap = fast_sim.from_obs(obs, configuration,
                             episode_seed=0, num_seats=num_seats)

    def step0_actions(cand: list) -> list:
        return [cand if i == me else opp_actions.get(i, []) for i in range(num_seats)]

    best_move = base_plan
    best_val = -1e18
    base_val = None
    for idx, cand in enumerate(candidates):
        s = fast_sim.step(snap, step0_actions(cand))      # branch (clones)
        s = _rollout(s, HORIZON - 1, num_seats)
        v = _leaf_value(s, me, num_seats)
        if idx == 0:
            base_val = v
        if v > best_val:
            best_val, best_move = v, cand
        # Soft wallclock guard: never risk a timeout; keep best-so-far.
        if (time.perf_counter() - t0) * 1000.0 > SOFT_MS:
            break

    # Tie-break toward the Producer's own plan: only deviate on a real gain.
    _STATE.turns += 1
    if base_val is not None and best_val <= base_val + 1e-9:
        return base_plan
    _STATE.deviations += 1
    if base_val is not None:
        _STATE.gain_sum += best_val - base_val
    return best_move


def agent(obs, configuration=None):
    """2-arg entry point (local harness + Kaggle).

    Robust by construction: any failure falls back to the Producer's move, and
    a Producer failure falls back to idle, so the agent never crashes a game.
    """
    try:
        return _decide(obs, configuration)
    except Exception:
        try:
            me = int(_obs_get(obs, "player", 0))
            return _producer_move(_STATE.runtime(me), obs, me)
        except Exception:
            return []
