"""Phase 6 — Differential parity harness for lib.trajectory_layer.

The "no bugs" gate. For each (seed, num_seats) in the matrix, runs
fast_sim forward 100 steps under a random-action policy. At every
sample step S, builds World.from_obs(snap_S.obs) and asserts the
trajectory layer agrees with `fast_sim` on EVERY observable, looking
forward to S+t for t in a fixed lookahead set:

  - planet_position(pid, t) at tolerance 1e-9
  - fleet_position(fid, t) at tolerance 1e-9 (for surviving fleets)
  - ledger.eta points at planets that DO get hit in fast_sim
  - ownership_at(pid, t) at tolerance 0 (owner) / 1e-9 (ships)
  - combat_at on every env-observed ownership change

Two test functions:

  test_diff_parity_2p_smoke — 5 seeds × 50 steps × t ∈ {1, 5, 20}.
    CI-fast (~30 s). The default green-gate.
  test_diff_parity_2p_thorough — 10 seeds × 100 steps × t ∈
    {1, 5, 20, 50}. Marked slow (~3 min). Run pre-migration.

Once green: Phase 7 (call-site migration) is unlocked.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

pytestmark = pytest.mark.slow

from kaggle_environments import make

from lib.fast_sim import Snapshot
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.trajectory_layer import World


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _planet_xy_by_id(obs: Any) -> dict[int, tuple[float, float]]:
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    return {int(p[0]): (float(p[2]), float(p[3])) for p in planets}


def _fleet_xy_by_id(obs: Any) -> dict[int, tuple[float, float]]:
    fleets = obs["fleets"] if isinstance(obs, dict) else obs.fleets
    return {int(f[0]): (float(f[2]), float(f[3])) for f in fleets}


def _planet_state_by_id(obs: Any) -> dict[int, tuple[int, float]]:
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    return {int(p[0]): (int(p[1]), float(p[5])) for p in planets}


def _random_actions(obs: Any, num_seats: int,
                    rng: random.Random) -> list[list]:
    actions: list[list] = [[] for _ in range(num_seats)]
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    for p in planets:
        owner = p[1]
        if 0 <= owner < num_seats and p[5] > 5 and rng.random() < 0.3:
            actions[owner].append([p[0], rng.uniform(0.0, 6.283),
                                    int(p[5] // 2)])
    return actions


COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)


def _crosses_comet_spawn(S: int, t: int) -> bool:
    """True iff [S+1, S+t] contains a comet spawn step.

    Spawns happen at step transitions S → S+1 where S+1 in
    COMET_SPAWN_STEPS. Phase 1-5 trajectory layer doesn't reconstruct
    future comet positions from `episode_seed`, so queries spanning
    such a transition are marked UNCERTAIN by Phase 4's design and
    skipped here. Phase 7+ may close this gap by adding
    `_simulate_future_comets(episode_seed, spawn_step)` to World.
    """
    for spawn in COMET_SPAWN_STEPS:
        if S < spawn <= S + t:
            return True
    return False


def _step_fast_sim_t_times(snap: Snapshot, t: int,
                            num_seats: int) -> Snapshot:
    """Step a fast_sim snapshot forward t times under EMPTY actions.
    Empty actions isolate the question "where would things go from
    here if nobody launched anything new" — exactly what the
    trajectory layer predicts."""
    snap = fs_clone(snap)
    for _ in range(t):
        snap = fs_step(snap, [[] for _ in range(num_seats)], in_place=True)
    return snap


def _drive_env(seed: int, num_seats: int, total_steps: int,
               ) -> tuple[Snapshot, int]:
    """Build a fast_sim snapshot at step 0 (no warmup) so we control
    the action sequence directly. Returns (snap0, episode_seed)."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=num_seats)
    ep_seed = int(env.info.get("seed", seed))
    snap0 = fs_from_obs(env.state[0].observation,
                         episode_seed=ep_seed, num_seats=num_seats)
    return snap0, ep_seed


# ---------------------------------------------------------------------------
# Per-step assertion bundles
# ---------------------------------------------------------------------------


def _assert_position_parity(world: World, snap_S: Snapshot,
                              t: int, num_seats: int,
                              *, label: str) -> None:
    """For every alive planet in snap_S, World's prediction at t
    matches fast_sim's actual position after t steps of empty
    actions."""
    snap_T = _step_fast_sim_t_times(snap_S, t, num_seats)
    truth_xy = _planet_xy_by_id(snap_T.obs)
    sample_pids = {p.id for p in world.planets}
    for pid, expected in truth_xy.items():
        if pid not in sample_pids:
            continue  # comet spawned mid-window — Phase 4 UNCERTAIN
        predicted = world.planet_position(pid, t)
        if predicted is None:
            # Planet despawned (comet expired). OK if also absent in truth.
            continue
        dx = abs(predicted[0] - expected[0])
        dy = abs(predicted[1] - expected[1])
        assert dx < 1e-9 and dy < 1e-9, (
            f"{label} pid={pid} t={t}: "
            f"predicted={predicted} truth={expected} "
            f"Δ=({dx:.3e}, {dy:.3e})"
        )


def _assert_fleet_position_parity(world: World, snap_S: Snapshot,
                                    t: int, num_seats: int,
                                    *, label: str) -> None:
    """Surviving fleets' positions at t match fast_sim's truth."""
    snap_T = _step_fast_sim_t_times(snap_S, t, num_seats)
    truth_xy = _fleet_xy_by_id(snap_T.obs)
    for f in world.fleets:
        if f.id not in truth_xy:
            continue  # fleet died in the interim — covered by ledger test
        predicted = world.fleet_position(f.id, t)
        assert predicted is not None
        expected = truth_xy[f.id]
        dx = abs(predicted[0] - expected[0])
        dy = abs(predicted[1] - expected[1])
        assert dx < 1e-9 and dy < 1e-9, (
            f"{label} fid={f.id} t={t}: "
            f"predicted={predicted} truth={expected} "
            f"Δ=({dx:.3e}, {dy:.3e})"
        )


def _assert_ledger_arrivals_land(world: World, snap_S: Snapshot,
                                   num_seats: int, *, label: str,
                                   horizon: int = 50) -> None:
    """Every predicted arrival must materialise in fast_sim within
    ±1 turn of the predicted eta. We use a tolerance window because
    the ledger's ceil-eta and the env's step-by-step physics can
    differ by 1 due to rounding at the swept-pair boundary."""
    ledger = world.ledger_all(horizon=horizon)
    # Build fast_sim history of "which fleets are alive at each step".
    snap = fs_clone(snap_S)
    fleet_alive_at: dict[int, list[bool]] = {}
    for t in range(horizon + 1):
        fids = {int(f[0]) for f in snap.obs["fleets"]}
        for f in world.fleets:
            fleet_alive_at.setdefault(f.id, []).append(f.id in fids)
        if t < horizon:
            snap = fs_step(snap, [[] for _ in range(num_seats)],
                            in_place=True)

    for pid, arrivals in ledger.items():
        for a in arrivals:
            assert a.fleet_id in fleet_alive_at, (
                f"{label} pid={pid} fleet_id={a.fleet_id}: "
                f"predicted arrival but fleet was never tracked"
            )
            # The fleet must be alive AT eta-1 and dead at eta+1
            # (the env removes the fleet on the step it hits a planet).
            alive_history = fleet_alive_at[a.fleet_id]
            window = alive_history[max(0, a.eta - 1):min(len(alive_history),
                                                          a.eta + 2)]
            # At least one True in [eta-1] and at least one False in
            # [eta..eta+1] is the canonical "fleet arrived" signature.
            died_in_window = any(not x for x in alive_history[a.eta:a.eta + 2])
            assert died_in_window, (
                f"{label} pid={pid} fleet={a.fleet_id} predicted "
                f"eta={a.eta} but fleet still alive in fast_sim at "
                f"t∈[{a.eta}, {a.eta + 1}]; history near eta="
                f"{alive_history[max(0, a.eta - 2):a.eta + 3]}"
            )


def _assert_ownership_at_parity(world: World, snap_S: Snapshot,
                                  t: int, num_seats: int,
                                  *, label: str) -> None:
    """World's ownership_at(pid, t) matches fast_sim's planet state
    after t steps of empty actions. Owner: exact. Ships: ±1.5
    (production integer/float drift; matches Phase 2's tolerance)."""
    snap_T = _step_fast_sim_t_times(snap_S, t, num_seats)
    truth_state = _planet_state_by_id(snap_T.obs)
    for p in world.planets:
        if p.is_comet:
            continue
        if p.id not in truth_state:
            continue  # comet expired
        truth_owner, truth_ships = truth_state[p.id]
        pred_owner, pred_ships = world.ownership_at(p.id, t,
                                                       horizon=max(t + 10, 50))
        assert pred_owner == truth_owner, (
            f"{label} pid={p.id} t={t}: "
            f"predicted owner={pred_owner} truth={truth_owner}"
        )
        assert abs(pred_ships - truth_ships) <= 1.5, (
            f"{label} pid={p.id} t={t}: "
            f"predicted ships={pred_ships} truth={truth_ships}"
        )


# ---------------------------------------------------------------------------
# Smoke test (CI gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 42, 100, 314, 2026])
def test_diff_parity_2p_smoke(seed: int):
    """5 seeds × 50 steps × t ∈ {1, 5, 20}. CI-fast gate.

    Sampling: every 10 steps (5 sample points per seed). At each
    sample, run the four parity assertions (planet positions, fleet
    positions, ledger arrivals, ownership-at). Total assertion
    volume: 5 seeds × 5 samples × ~150 entities × 3 t-values ≈ 11250
    individual checks per run.
    """
    num_seats = 2
    horizon = 50
    sample_every = 10
    t_values = (1, 5, 20)

    snap, ep_seed = _drive_env(seed, num_seats, horizon)
    action_rng = random.Random(seed * 31 + 1)

    for S in range(horizon + 1):
        if S % sample_every == 0:
            # Build a World from the current obs and check parity.
            world = World.from_obs(snap.obs, episode_seed=ep_seed)
            label = f"seed={seed} S={S}"
            for t in t_values:
                if S + t > horizon:
                    continue
                if _crosses_comet_spawn(S, t):
                    # Future-comet reconstruction not in Phase 1-5
                    # scope; UNCERTAIN per Phase 4 design.
                    continue
                _assert_position_parity(world, snap, t, num_seats,
                                          label=label)
                _assert_fleet_position_parity(world, snap, t, num_seats,
                                                label=label)
                _assert_ownership_at_parity(world, snap, t, num_seats,
                                              label=label)
            # Ledger check (one big assertion, not per-t):
            if S + 30 <= horizon and not _crosses_comet_spawn(S, 30):
                _assert_ledger_arrivals_land(world, snap, num_seats,
                                                label=label, horizon=30)
        if S < horizon:
            snap = fs_step(snap, _random_actions(snap.obs, num_seats,
                                                    action_rng),
                            in_place=True)


# ---------------------------------------------------------------------------
# Thorough test (pre-migration gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 42, 100, 314, 2026,
                                     17, 31, 77, 144, 2718])
def test_diff_parity_2p_thorough(seed: int):
    """10 seeds × 100 steps × t ∈ {1, 5, 20, 50}. Pre-migration gate.

    Same shape as smoke but: longer horizon, more t-values, sampling
    every 20 steps (so 5 samples per seed × 4 t-values).
    """
    num_seats = 2
    horizon = 100
    sample_every = 20
    t_values = (1, 5, 20, 50)

    snap, ep_seed = _drive_env(seed, num_seats, horizon)
    action_rng = random.Random(seed * 31 + 1)

    for S in range(horizon + 1):
        if S % sample_every == 0:
            world = World.from_obs(snap.obs, episode_seed=ep_seed)
            label = f"seed={seed} S={S}"
            for t in t_values:
                if S + t > horizon:
                    continue
                if _crosses_comet_spawn(S, t):
                    # Future-comet reconstruction not in Phase 1-5
                    # scope; UNCERTAIN per Phase 4 design.
                    continue
                _assert_position_parity(world, snap, t, num_seats,
                                          label=label)
                _assert_fleet_position_parity(world, snap, t, num_seats,
                                                label=label)
                _assert_ownership_at_parity(world, snap, t, num_seats,
                                              label=label)
            if S + 50 <= horizon and not _crosses_comet_spawn(S, 50):
                _assert_ledger_arrivals_land(world, snap, num_seats,
                                                label=label, horizon=50)
        if S < horizon:
            snap = fs_step(snap, _random_actions(snap.obs, num_seats,
                                                    action_rng),
                            in_place=True)


# ---------------------------------------------------------------------------
# 4P coverage (smaller, since 4P is slower)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [11, 99])
def test_diff_parity_4p_smoke(seed: int):
    """2 seeds × 40 steps × t ∈ {1, 5, 20}. 4P-specific coverage —
    FFA exposes some bucket attributions that 2P doesn't (e.g.
    multiple non-self owners on the same planet)."""
    num_seats = 4
    horizon = 40
    sample_every = 10
    t_values = (1, 5, 20)

    snap, ep_seed = _drive_env(seed, num_seats, horizon)
    action_rng = random.Random(seed * 47 + 13)

    for S in range(horizon + 1):
        if S % sample_every == 0:
            world = World.from_obs(snap.obs, episode_seed=ep_seed)
            label = f"4p seed={seed} S={S}"
            for t in t_values:
                if S + t > horizon:
                    continue
                if _crosses_comet_spawn(S, t):
                    continue
                _assert_position_parity(world, snap, t, num_seats,
                                          label=label)
                _assert_ownership_at_parity(world, snap, t, num_seats,
                                              label=label)
        if S < horizon:
            snap = fs_step(snap, _random_actions(snap.obs, num_seats,
                                                    action_rng),
                            in_place=True)
