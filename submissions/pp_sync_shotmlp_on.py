from __future__ import annotations
import os as _pp_os
_pp_os.environ.setdefault('PRODUCER_PLUS_MULTI_SIZE', '1')
_pp_os.environ.setdefault('PRODUCER_PLUS_OPP_PROJECTION', '1')
_pp_os.environ.setdefault('PRODUCER_PLUS_RESPONSE_VETO', '1')
_pp_os.environ.setdefault('PRODUCER_PLUS_REACTIVE_FLOOR', '0.5')
_pp_os.environ.setdefault('PRODUCER_PLUS_REPLY_SEQ', '1')
_pp_os.environ.setdefault('PRODUCER_PLUS_FFA_SCORE', '1')
_pp_os.environ.setdefault('PRODUCER_PLUS_FFA_WEIGHTS', 'strength')
_pp_os.environ.setdefault('PRODUCER_PLUS_SYNC', '1')

# === orbit_lite.constants ===
"""Capacity and physics constants for the game (must match the engine)."""

# ---------------------------------------------------------------------------
# Capacity — tune based on GPU memory and profiling
# ---------------------------------------------------------------------------
B_DEFAULT: int = 1024   # default games per batch
P_MAX: int = 64         # planet slots per game  (real games have 24-52 planets)
F_MAX: int = 256        # fleet slots per game
A: int = 2              # players per game

# ---------------------------------------------------------------------------
# Physics (must match the game engine)
# ---------------------------------------------------------------------------
BOARD_SIZE: float = 100.0
CENTER: float = 50.0
SUN_RADIUS: float = 10.0
MAX_SHIP_SPEED: float = 6.0
ROT_RADIUS_LIMIT: float = 50.0  # planets with orbital_radius + radius < this orbit

# ---------------------------------------------------------------------------
# Observation — relative ownership encoding
# ---------------------------------------------------------------------------
OWN: int = 0      # slot belongs to the observing player
ENEMY: int = 1    # slot belongs to an opponent
NEUTRAL: int = 2  # slot is unclaimed
DEAD: int = 3     # slot is empty (alive_mask=False)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
LIBRARY_K_DEFAULT: int = 100_000  # number of starting states to pre-generate

# ---------------------------------------------------------------------------
# Comets (optional, gated by comets_enabled)
# ---------------------------------------------------------------------------
COMET_EVENTS: int = 5
COMETS_PER_EVENT: int = 4
COMET_PATH_MAX: int = 40
COMET_SPAWN_STEPS: tuple[int, ...] = (50, 150, 250, 350, 450)
COMET_RADIUS: float = 1.0
COMET_PRODUCTION: float = 1.0

# ---------------------------------------------------------------------------
# Early termination — call the game when one player dominates the leaderboard
#
# Calibrated on 535 Kaggle replays (scripts/analyze_early_termination.py).
# 2p: 100% accuracy over 242 games, saves ~48 turns / game (~30%).
# 4p: 100% accuracy on triggered (286/293) games, saves ~59 turns / game.
# ---------------------------------------------------------------------------
EARLY_TERM_MARGIN: float = 2.0          # leader_score >= MARGIN * runner_up_score
EARLY_TERM_STREAK_2P: int = 5           # consecutive turns the lead must hold
EARLY_TERM_STREAK_4P: int = 20
EARLY_TERM_PROD_WEIGHT_2P: float = 5.0  # score = 5 * production + 1 * (planet + fleet ships)
EARLY_TERM_SHIP_WEIGHT_2P: float = 1.0
EARLY_TERM_PROD_WEIGHT_4P: float = 1.0  # 4p uses production alone
EARLY_TERM_SHIP_WEIGHT_4P: float = 0.0

# ---------------------------------------------------------------------------
# Episode length (default number of game steps)
# ---------------------------------------------------------------------------
DEFAULT_EPISODE_STEPS: int = 500


# === orbit_lite.aiming ===
"""Orbit-phase helper used by the movement forecaster."""

from torch import Tensor


def orbit_phase_index_from_obs_step(obs_step: Tensor) -> Tensor:
    """Convert the observation ``step`` counter into the engine orbit phase index.

    Orbiting planets update with ``theta = orb_a0 + angvel * g_step`` *before*
    ``g_step`` is incremented for the next observation. The public observation
    carries ``step == g_step`` after that increment, so the implied phase index
    is ``max(0, step - 1)`` (and ``0`` at game start when ``step == 0``).
    """
    s = obs_step.float()
    return (s - (s > 0).to(s.dtype)).clamp(min=0.0)


# === orbit_lite.geometry ===
"""Geometry primitives. Pure tensor functions with no game-state imports."""


import torch
from torch import Tensor



# Pre-compute log(1000) once as a plain Python float for efficiency.
_LOG_1000: float = float(torch.log(torch.tensor(1000.0)).item())
_FLEET_SPEED_LUT_MAX: int = 400


def _fleet_speed_formula(ships: Tensor) -> Tensor:
    """Exact engine-matching speed formula."""
    ratio = (torch.log(ships) / _LOG_1000).clamp(max=1.0)
    return 1.0 + (MAX_SHIP_SPEED - 1.0) * ratio.pow(1.5)


def _build_fleet_speed_lut(max_ships: int) -> Tensor:
    # Index 0 is unused but keeps indexing branch-free for ships >= 1.
    idx = torch.arange(max_ships + 1, dtype=torch.float32).clamp(min=1.0)
    return _fleet_speed_formula(idx)


_FLEET_SPEED_LUT: Tensor = _build_fleet_speed_lut(_FLEET_SPEED_LUT_MAX)
# Per-(device, dtype) cache of the LUT so a CUDA stream isn't synced by an
# H→D copy on every fleet_speed call. Module-level dict, populated lazily.
_FLEET_SPEED_LUT_CACHE: dict[tuple, Tensor] = {}


def _fleet_speed_lut_on(device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (device, dtype)
    cached = _FLEET_SPEED_LUT_CACHE.get(key)
    if cached is None:
        cached = _FLEET_SPEED_LUT.to(device=device, dtype=dtype)
        _FLEET_SPEED_LUT_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# Pairwise operations  [N] × [M]  →  [N, M]
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Fleet physics
# ---------------------------------------------------------------------------

def fleet_speed(ships: Tensor) -> Tensor:
    """Travel speed for a fleet of ``ships`` ships.

    The engine ship-speed formula::

        speed = 1 + (MAX_SHIP_SPEED - 1) * (log(ships) / log(1000)) ** 1.5

    Args:
        ships: ship count, any shape; values are clamped to ≥ 1.

    Returns:
        speed in ``[1, MAX_SHIP_SPEED]``, same shape as ``ships``.
    """
    s = ships.clamp(min=1.0)
    s_lut = s.clamp(max=float(_FLEET_SPEED_LUT_MAX))
    lo = torch.floor(s_lut).long()
    hi = torch.ceil(s_lut).long()
    frac = s_lut - lo.to(dtype=s.dtype)

    lut = _fleet_speed_lut_on(s.device, s.dtype)
    speed = lut[lo] + (lut[hi] - lut[lo]) * frac

    # Over-range fleets (>``_FLEET_SPEED_LUT_MAX`` ships) use the exact
    # formula. We unconditionally compute it and select via ``torch.where``
    # rather than a ``bool(over.any())`` branch — the latter triggers a
    # host/device sync per call on CUDA which dominated the wall-clock
    # of every kernel that batches fleet_speed inside its inner loop.
    over = s > float(_FLEET_SPEED_LUT_MAX)
    speed_formula = _fleet_speed_formula(s)
    return torch.where(over, speed_formula, speed)






# ---------------------------------------------------------------------------
# Segment–circle intersection (sun / planet collision geometry)
# ---------------------------------------------------------------------------







# === orbit_lite.obs ===
"""Canonical observation parsing into a named :class:`ParsedObs` dataclass.

Converts the raw 7-field observation tensors (produced by
:func:`adapter.single_obs_to_tensor`) into named per-planet/per-fleet fields.

Field index definitions
-----------------------
``planets`` / ``initial_planets``  ``[P, 7]`` float32::

    0 – planet_id   (alive sentinel: id >= 0; padding value: -1)
    1 – owner       (absolute player index; -1 = neutral)
    2 – x           (board coordinates, 0–100)
    3 – y
    4 – radius
    5 – ships       (current count)
    6 – production  (ships added per turn when owned)

``fleets``  ``[F, 7]`` float32::

    0 – fleet_id    (alive sentinel: id >= 0)
    1 – owner
    2 – x
    3 – y
    4 – angle       (radians)
    5 – from_planet_id
    6 – ships

No field indices appear outside this module; all downstream modules consume
:class:`ParsedObs` named fields instead.
"""


from dataclasses import dataclass

import torch
from torch import Tensor




# ---------------------------------------------------------------------------
# ParsedObs
# ---------------------------------------------------------------------------

@dataclass
class ParsedObs:
    """Named per-planet fields decoded from a raw batch observation dict.

    All tensor fields have shape ``[P]`` unless stated otherwise.
    """

    # --- raw planet fields --------------------------------------------------
    alive: Tensor       # bool  – planet_id >= 0 (not a padding slot)
    x: Tensor           # float – current x position (0–100)
    y: Tensor           # float – current y position (0–100)
    r: Tensor           # float – radius
    ships: Tensor       # float – current ship count
    prod: Tensor        # float – production per turn
    owner_abs: Tensor   # float – absolute owner id (-1 = neutral)

    # --- relative ownership masks (computed from owner_abs + player_id) -----
    owned: Tensor       # bool – alive & owner_abs == player_id
    is_enemy: Tensor    # bool – alive & owner_abs >= 0 & owner_abs != player_id
    is_neutral: Tensor  # bool – alive & owner_abs < 0

    # --- orbital parameters (reconstructed from initial_planets) ------------
    orb_r: Tensor       # float – orbital radius; 0.0 for static planets
    orb_a0: Tensor      # float – initial angle from CENTER (radians)
    is_orbiting: Tensor # bool  – True for rotating planets

    # --- game scalars -------------------------------------------------------
    angvel: Tensor      # [B] float – board angular velocity (rad/turn)
    step: Tensor        # [B] float – current game step

    # --- fleet fields -------------------------------------------------------
    #  Available when parse_obs is called with include_fleets=True;
    #  shapes are [F, *] and accessed as attributes rather than being
    #  indexed per-column.
    f_alive: Tensor     # [F] bool
    f_owner: Tensor     # [F] float – absolute owner
    f_x: Tensor         # [F] float
    f_y: Tensor         # [F] float
    f_angle: Tensor     # [F] float – radians
    f_ships: Tensor     # [F] float

    # --- metadata -----------------------------------------------------------
    player_id: int
    P: int
    F: int
    device: torch.device


# ---------------------------------------------------------------------------
# parse_obs
# ---------------------------------------------------------------------------

def parse_obs(obs_tensors: dict, player_id: int | None = None) -> ParsedObs:
    """Decode a raw batch observation dict into a :class:`ParsedObs`.

    Args:
        obs_tensors: dict as produced by ``adapter.single_obs_to_tensor`` or
                     Required keys:
                     ``"planets"`` ``[P, 7]``,
                     ``"initial_planets"`` ``[P, 7]``,
                     ``"fleets"`` ``[F, 7]``,
                     ``"angular_velocity"`` scalar,
                     ``"step"`` scalar,
                     ``"player"`` scalar.
        player_id:   Which player to compute ownership masks for.  Defaults
                     to ``int(obs_tensors["player"][0])``.

    Returns:
        :class:`ParsedObs` with all tensors on the same device as ``planets``.
    """
    planets = obs_tensors["planets"]          # [P, 7]
    initial = obs_tensors["initial_planets"]  # [P, 7]
    fleets = obs_tensors["fleets"]            # [F, 7]
    angvel = obs_tensors["angular_velocity"].float()  # scalar
    step = obs_tensors["step"].float()        # scalar

    if player_id is None:
        player_id = int(obs_tensors["player"].flatten()[0].item())

    P, _ = planets.shape
    F, _ = fleets.shape
    device = planets.device

    # -- planet columns -------------------------------------------------------
    pid = planets[..., 0]        # [P]
    owner_abs = planets[..., 1]
    x = planets[..., 2]
    y = planets[..., 3]
    r = planets[..., 4]
    ships = planets[..., 5]
    prod = planets[..., 6]

    alive = pid >= 0.0

    owned = alive & (owner_abs == float(player_id))
    is_enemy = alive & (owner_abs >= 0.0) & (owner_abs != float(player_id))
    is_neutral = alive & (owner_abs < 0.0)

    # -- orbital parameters from initial_planets ------------------------------
    # A planet is "orbiting" when its distance from the board centre plus its
    # radius is below ROT_RADIUS_LIMIT (mirroring the engine's initialisation
    # logic).  We reconstruct the orbital radius and initial angle from the
    # initial position stored in the observation.
    ix = initial[..., 2]  # [P]
    iy = initial[..., 3]
    i_r = initial[..., 4]  # initial radius (same as current for orbiting)

    dx0 = ix - CENTER
    dy0 = iy - CENTER
    orb_r_raw = torch.sqrt(dx0 * dx0 + dy0 * dy0)
    orb_a0 = torch.atan2(dy0, dx0)

    # Orbiting: alive, initial orbital radius + planet radius < limit, and
    # non-trivially away from the centre (avoids treating dead/padding slots
    # with ix=iy=0 as orbiting).
    is_orbiting = alive & ((orb_r_raw + i_r) < ROT_RADIUS_LIMIT) & (orb_r_raw > 0.5)

    # Static planets carry orb_r = 0 so downstream maths stay correct.
    orb_r = torch.where(is_orbiting, orb_r_raw, torch.zeros_like(orb_r_raw))

    # -- fleet columns --------------------------------------------------------
    f_pid = fleets[..., 0]      # [F]
    f_alive = f_pid >= 0.0
    f_owner = fleets[..., 1]
    f_x = fleets[..., 2]
    f_y = fleets[..., 3]
    f_angle = fleets[..., 4]
    f_ships = fleets[..., 6]

    return ParsedObs(
        alive=alive,
        x=x, y=y, r=r,
        ships=ships, prod=prod,
        owner_abs=owner_abs,
        owned=owned,
        is_enemy=is_enemy,
        is_neutral=is_neutral,
        orb_r=orb_r,
        orb_a0=orb_a0,
        is_orbiting=is_orbiting,
        angvel=angvel,
        step=step,
        f_alive=f_alive,
        f_owner=f_owner,
        f_x=f_x, f_y=f_y,
        f_angle=f_angle,
        f_ships=f_ships,
        player_id=player_id,
        P=P, F=F,
        device=device,
    )


# === orbit_lite.movement_aiming ===
"""Aiming helpers backed by :class:`PlanetMovement`.

This module is intentionally small: it solves angle/ETA for concrete
``(source_slot, target_slot, fleet_size)`` candidates using cached future planet
positions, then masks candidates whose straight path crosses the sun or another
planet.
"""


import torch
from torch import Tensor



LAUNCH_SURFACE_OFFSET: float = 0.1
"""Fleet launch offset from source surface.

Matches Kaggle Orbit Wars engine launch placement:
``start = source + unit(angle) * (source_radius + 0.1)``.
"""

TARGET_HIT_SURFACE_OFFSET: float = 0.0
"""Extra target-surface margin for hit ETA estimation.

Kaggle/local engines register fleet-vs-planet contact when the fleet center
enters the target planet radius. ``0.0`` keeps ETA aligned with that rule.
"""

KAGGLE_SUN_RADIUS: float = SUN_RADIUS
"""Sun collision radius used by Kaggle/local engines."""











def _swept_pair_hit_mask(
    ax: Tensor,
    ay: Tensor,
    bx: Tensor,
    by: Tensor,
    p0x: Tensor,
    p0y: Tensor,
    p1x: Tensor,
    p1y: Tensor,
    r: Tensor,
) -> Tensor:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    near_static = a < 1e-12
    c_hit = c <= 0.0
    disc = b * b - 4.0 * a * c
    has_root = disc >= 0.0
    safe_a = torch.where(near_static, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    quad_hit = has_root & (t2 >= 0.0) & (t1 <= 1.0)
    return torch.where(near_static, c_hit, quad_hit)


# === orbit_lite.movement ===
"""Future planet/comet movement cache + garrison projection for one game.

``PlanetMovement`` predicts planet and comet positions from an observation, keeps
a short rolling horizon, tracks in-flight fleets, and projects per-planet owner /
ships over the horizon (the do-nothing garrison forecast agents plan against).
"""


from dataclasses import dataclass

import torch
from torch import Tensor







DEFAULT_MOVEMENT_HORIZON = 20
DEFAULT_DRIFT_EPSILON = 1e-4
DEFAULT_MAX_TRACKED_FLEETS = 64


@dataclass(frozen=True)
class MovementConfig:
    """Configuration for ``PlanetMovement`` construction and updates."""

    movement_horizon: int = DEFAULT_MOVEMENT_HORIZON
    drift_epsilon: float = DEFAULT_DRIFT_EPSILON
    track_fleets: bool = False
    player_count: int | None = None
    max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS



@dataclass(frozen=True)
class PlanetGarrisonStatus:
    """Projected planet ownership and garrison ships over cached future steps.

    ``owner`` / ``ships`` are *post-combat* values: what the planet looks like at
    the end of each future step assuming the agent does **not** act. They are the
    right oracle for "what will be there in N turns if I do nothing."

    ``pre_combat_owner`` / ``pre_combat_ships`` are the planet state *just
    before* combat resolution at each future step — after that step's production
    has been credited but before any same-step arrivals are applied. Agents
    planning their own arrival on step ``k`` should consult these (plus the
    per-step ``arrivals_by_owner``) and apply the engine combat rule themselves:
    treating their own send as an additional same-step attacker. They are
    populated only when fleet tracking is enabled.

    ``arrivals_by_owner`` mirrors ``PlanetMovement.fleet_buckets`` at the
    requested planet slots: per-step per-owner ship totals arriving on a given
    target. Shape ``[*prefix, H, A]`` where ``A`` is the number of agents. ``None``
    when fleet tracking is off.
    """

    owner: Tensor
    ships: Tensor
    pre_combat_owner: Tensor | None = None
    pre_combat_ships: Tensor | None = None
    arrivals_by_owner: Tensor | None = None




@dataclass
class PlanetMovement:
    """Rolling cache of future planet positions for a single game.

    Tensor shapes:
    - ``x``, ``y``, ``alive_by_step``: ``[H + 1, P]``
    - ``planet_ids``, ``radii``: ``[P]``
    - ``base_step``: scalar
    - optional ``fleet_buckets``: ``[P, H, A]``

    ``k == 0`` is the observation frame used to build the cache, and ``k`` is
    the number of future movement steps from that frame.
    """

    x: Tensor
    y: Tensor
    alive_by_step: Tensor
    planet_ids: Tensor
    radii: Tensor
    planet_owner: Tensor
    planet_ships: Tensor
    planet_prod: Tensor
    base_step: Tensor
    comet_planet_ids: Tensor
    comet_path_index: Tensor
    movement_horizon: int = DEFAULT_MOVEMENT_HORIZON
    drift_epsilon: float = DEFAULT_DRIFT_EPSILON
    track_fleets: bool = False
    player_count: int | None = None
    max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS
    fleet_buckets: Tensor | None = None
    fleet_last_step: Tensor | None = None
    tracked_fleet_ids: Tensor | None = None
    tracked_fleet_eta: Tensor | None = None
    tracked_fleet_target_slot: Tensor | None = None
    # Per-entry owner / ship-count of the recorded arrival. Required so
    # ``_reconcile_obs_fleets`` can subtract a phantom's contribution from
    # ``fleet_buckets`` when its fleet id vanishes from obs.
    tracked_fleet_owner: Tensor | None = None
    tracked_fleet_ships: Tensor | None = None
    garrison_owner_cache: Tensor | None = None
    garrison_ships_cache: Tensor | None = None
    garrison_pre_combat_owner_cache: Tensor | None = None
    garrison_pre_combat_ships_cache: Tensor | None = None
    garrison_dirty_from: Tensor | None = None
    # Per-batch pending launches awaiting fleet-id reconciliation against the
    # next observation. Each lane carries up to ``pending_*`` columns of
    # stashed-launch metadata; empty slots are marked by ``pending_owners ==
    # -1``. See ``stash_pending_own_launches`` and
    # ``_reconcile_pending_own_launches``. ``next_fleet_id`` and the step at
    # stash time are stored per-entry so multi-owner stash within one turn
    # works.
    pending_source_planets: Tensor | None = None   # [L] long  (-1 = empty)
    pending_ships: Tensor | None = None            # [L] long
    pending_angle: Tensor | None = None            # [L] dtype
    pending_target_slots: Tensor | None = None     # [L] long
    pending_eta: Tensor | None = None              # [L] dtype
    pending_owners: Tensor | None = None           # [L] long  (-1 = empty)
    pending_prev_nfid: Tensor | None = None        # [L] long
    pending_stash_step: Tensor | None = None       # [L] long

    @property
    def P(self) -> int:
        return int(self.planet_ids.shape[0])

    @property
    def device(self) -> torch.device:
        return self.x.device

    @property
    def dtype(self) -> torch.dtype:
        return self.x.dtype

    @property
    def config(self) -> MovementConfig:
        """Return the explicit movement config used by this cache."""
        return MovementConfig(
            movement_horizon=int(self.movement_horizon),
            drift_epsilon=float(self.drift_epsilon),
            track_fleets=bool(self.track_fleets),
            player_count=self.player_count,
            max_tracked_fleets=int(self.max_tracked_fleets),
        )

    @classmethod
    def from_obs_tensors(
        cls,
        obs_tensors: dict,
        *,
        config: MovementConfig | None = None,
        movement_horizon: int = DEFAULT_MOVEMENT_HORIZON,
        drift_epsilon: float = DEFAULT_DRIFT_EPSILON,
        track_fleets: bool = False,
        player_count: int | None = None,
        max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS,
    ) -> "PlanetMovement":
        """Build a fresh movement cache from batched observation tensors.

        The cache has movement parameters plus optional fleet tracking:
        - ``movement_horizon``: number of future steps cached.
        - ``drift_epsilon``: tolerated positional drift before rebuild.
        - ``track_fleets``: opt-in arrival buckets shaped ``[P, H, A]``.
        - ``player_count``: known player count (2 or 4), or inferred at turn 0.
        - ``max_tracked_fleets``: capacity per batch lane for in-flight fleet-id ledger rows.
        """
        cfg = config if config is not None else MovementConfig(
            movement_horizon=int(movement_horizon),
            drift_epsilon=float(drift_epsilon),
            track_fleets=bool(track_fleets),
            player_count=player_count,
            max_tracked_fleets=int(max_tracked_fleets),
        )
        built = _build_future_from_obs(obs_tensors, int(cfg.movement_horizon))
        resolved_player_count = _resolve_player_count(obs_tensors, cfg.player_count) if cfg.track_fleets else cfg.player_count
        movement = cls(
            x=built["x"],
            y=built["y"],
            alive_by_step=built["alive_by_step"],
            planet_ids=built["planet_ids"],
            radii=built["radii"],
            planet_owner=built["owner"],
            planet_ships=built["ships"],
            planet_prod=built["prod"],
            base_step=built["step"],
            comet_planet_ids=built["comet_planet_ids"],
            comet_path_index=built["comet_path_index"],
            movement_horizon=int(cfg.movement_horizon),
            drift_epsilon=float(cfg.drift_epsilon),
            track_fleets=bool(cfg.track_fleets),
            player_count=resolved_player_count,
            max_tracked_fleets=int(cfg.max_tracked_fleets),
        )
        if movement.track_fleets:
            movement._init_fleet_tracking(obs_tensors, reset_ledger=True)
            movement._ingest_obs_fleets(obs_tensors)
        return movement

    def update(self, obs_tensors: dict) -> "PlanetMovement":
        """Refresh this cache for a new observation (single game).

        If the current observation matches the cached prediction the trajectory
        is kept (same step) or rolled forward by one step. Numeric drift, step
        jumps, shape/device changes, or planet/comet identity changes trigger a
        full rebuild from the new observation.
        """
        planets = obs_tensors["planets"]
        if (
            planets.device != self.device
            or planets.shape[0] != self.P
            or int(self.x.shape[0]) != int(self.movement_horizon) + 1
        ):
            fresh = type(self).from_obs_tensors(
                obs_tensors,
                movement_horizon=self.movement_horizon,
                drift_epsilon=self.drift_epsilon,
                track_fleets=self.track_fleets,
                player_count=self.player_count,
                max_tracked_fleets=int(self.max_tracked_fleets),
            )
            self._copy_from(fresh)
            return self

        if self.track_fleets:
            current_player_count = _resolve_player_count(obs_tensors, self.player_count)
            if (
                self.fleet_buckets is None
                or self.fleet_last_step is None
                or self.tracked_fleet_ids is None
                or tuple(self.fleet_buckets.shape) != (
                    self.P,
                    int(self.movement_horizon),
                    int(current_player_count),
                )
                or self.fleet_buckets.device != self.device
                or int(self.tracked_fleet_ids.shape[0]) < int(self.max_tracked_fleets)
            ):
                self.player_count = int(current_player_count)
                self._init_fleet_tracking(obs_tensors, reset_ledger=True)

        obs_for_decision = parse_obs(obs_tensors)
        H = int(self.movement_horizon)
        planet_ids_now = planets[..., 0].long()
        radii_now = planets[..., 4].to(dtype=self.dtype)
        owner_now = planets[..., 1].to(device=self.device, dtype=torch.long)
        owner_now = torch.where(
            obs_for_decision.alive, owner_now, torch.full_like(owner_now, -1)
        )
        ships_now = planets[..., 5].to(device=self.device, dtype=self.dtype)
        prod_now = planets[..., 6].to(device=self.device, dtype=self.dtype)
        step_now = obs_for_decision.step.to(device=self.device, dtype=torch.long)
        comet_ids_now, comet_idx_now = _comet_metadata(obs_tensors, self.device)
        current_obs_x = planets[..., 2].to(device=self.device, dtype=self.dtype)
        current_obs_y = planets[..., 3].to(device=self.device, dtype=self.dtype)
        current_alive = obs_for_decision.alive

        ids_same = bool((planet_ids_now == self.planet_ids).all())
        same_step = bool(step_now == self.base_step)
        next_step = bool(step_now == (self.base_step + 1))

        comet_same = _same_2d(comet_ids_now, self.comet_planet_ids)
        comet_idx_same = _same_2d(comet_idx_now, self.comet_path_index)
        expected_next_idx = torch.where(
            self.comet_path_index >= 0,
            self.comet_path_index + 1,
            self.comet_path_index,
        )
        comet_idx_next = _same_2d(comet_idx_now, expected_next_idx)

        same_alive_ok = bool((current_alive == self.alive_by_step[0]).all())
        next_alive_ok = bool((current_alive == self.alive_by_step[1]).all())
        same_drift_ok = _position_matches(
            self.x[0], self.y[0], current_obs_x, current_obs_y,
            current_alive, float(self.drift_epsilon),
        )
        next_drift_ok = _position_matches(
            self.x[1], self.y[1], current_obs_x, current_obs_y,
            current_alive, float(self.drift_epsilon),
        )

        keep = ids_same and same_step and comet_same and comet_idx_same and same_alive_ok and same_drift_ok
        roll = ids_same and next_step and comet_same and comet_idx_next and next_alive_ok and next_drift_ok
        rebuild = not (keep or roll)

        if rebuild:
            built = _build_future_from_obs(obs_tensors, H)
        elif roll:
            # Roll-only path: build just the new last frame at offset H.
            last_offset = torch.tensor([H], dtype=torch.long, device=self.device)
            built = _build_future_from_obs(obs_tensors, H, offsets=last_offset)
        else:
            built = None

        if roll:
            assert built is not None
            self.x[:-1] = self.x[1:].clone()
            self.y[:-1] = self.y[1:].clone()
            self.alive_by_step[:-1] = self.alive_by_step[1:].clone()
            self.x[-1] = built["x"][-1]
            self.y[-1] = built["y"][-1]
            self.alive_by_step[-1] = built["alive_by_step"][-1]
            self._roll_garrison_projection()

        if rebuild:
            assert built is not None
            self.x[:] = built["x"]
            self.y[:] = built["y"]
            self.alive_by_step[:] = built["alive_by_step"]
            self._mark_garrison_dirty_all(0)

        if roll or rebuild:
            self.planet_ids[:] = planet_ids_now
            self.radii[:] = radii_now
            self.base_step = step_now
            self.comet_planet_ids = comet_ids_now
            self.comet_path_index = comet_idx_now

        self._refresh_garrison_base({
            "planet_ids": planet_ids_now,
            "radii": radii_now,
            "owner": owner_now,
            "ships": ships_now,
            "prod": prod_now,
            "step": step_now,
        })

        if self.track_fleets:
            self._roll_fleet_buckets_phase1(step_now)
            if rebuild and not ids_same:
                self._reset_fleet_tracking()
            self._reconcile_pending_own_launches(obs_tensors)
            self._ingest_obs_fleets(obs_tensors)
            self._reconcile_obs_fleets(obs_tensors)

        return self

    def all_positions(self, k: int) -> tuple[Tensor, Tensor]:
        """Return all planet positions ``k`` steps ahead as ``[P]``."""
        idx = self._k_index(k)
        return self.x[idx], self.y[idx]

    def alive_at(self, k: int) -> Tensor:
        """Return alive mask ``k`` steps ahead as ``[P]``."""
        return self.alive_by_step[self._k_index(k)]

    def position_at_slots(self, slots: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Gather future positions for slot indices of any shape."""
        slots = slots.to(device=self.device, dtype=torch.long).clamp(0, max(self.P - 1, 0))
        px, py = self.all_positions(k)
        out_x = px[slots].to(dtype=self.dtype)
        out_y = py[slots].to(dtype=self.dtype)
        return out_x, out_y


    def pairwise_distance(self, k: int) -> Tensor:
        """Return all pairwise planet distances ``k`` steps ahead, ``[P, P]``."""
        px, py = self.all_positions(k)
        dx = px.unsqueeze(1) - px.unsqueeze(0)
        dy = py.unsqueeze(1) - py.unsqueeze(0)
        return torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))







    def garrison_status(self, planet_slots: Tensor | None = None, *, max_horizon: int | None = None) -> PlanetGarrisonStatus:
        """Return projected owner and ships for selected planet slots.

        The output time axis is ``H + 1``: ``k=0`` is the current observation,
        and ``k=1..H`` are post-production/post-combat states for future turns.
        Fleet tracking must be enabled so arrivals are available.
        """
        self._require_fleet_buckets()
        slots, out_prefix = self._normalize_garrison_slots(planet_slots)
        requested_horizon = int(
            self.movement_horizon if max_horizon is None else max(0, min(int(max_horizon), int(self.movement_horizon)))
        )
        self._refresh_garrison_projection(slots, requested_horizon=requested_horizon)
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_dirty_from is not None

        owner = self.garrison_owner_cache[slots][:, : requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
        ships = self.garrison_ships_cache[slots][:, : requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
        pre_combat_owner: Tensor | None = None
        pre_combat_ships: Tensor | None = None
        if (
            self.garrison_pre_combat_owner_cache is not None
            and self.garrison_pre_combat_ships_cache is not None
        ):
            pre_combat_owner = (
                self.garrison_pre_combat_owner_cache[slots][:, : requested_horizon + 1]
                .reshape(*out_prefix, requested_horizon + 1)
            )
            pre_combat_ships = (
                self.garrison_pre_combat_ships_cache[slots][:, : requested_horizon + 1]
                .reshape(*out_prefix, requested_horizon + 1)
            )
        arrivals_by_owner: Tensor | None = None
        if self.fleet_buckets is not None and requested_horizon > 0:
            # ``fleet_buckets`` shape: [P, H, A]. Select the slots to produce
            # [*out_prefix, requested_horizon, A]; then left-pad a zero step-0
            # frame so the time axis lines up with the owner/ships caches (which
            # have an extra ``k=0`` observation slot).
            A = int(self.fleet_buckets.shape[-1])
            arrivals_full = (
                self.fleet_buckets[slots]
                .reshape(*out_prefix, int(self.movement_horizon), A)
            )
            # Trim/pad to the requested horizon: k=0 has no arrivals; k=1..H map
            # to fleet_buckets[..., 0..H-1, :].
            arrivals_trimmed = arrivals_full[..., :requested_horizon, :]
            zero_frame = torch.zeros(
                *out_prefix, 1, A, dtype=arrivals_trimmed.dtype, device=self.device
            )
            arrivals_by_owner = torch.cat([zero_frame, arrivals_trimmed], dim=-2)
        status = PlanetGarrisonStatus(
            owner=owner,
            ships=ships,
            pre_combat_owner=pre_combat_owner,
            pre_combat_ships=pre_combat_ships,
            arrivals_by_owner=arrivals_by_owner,
        )
        return status



    def _clear_pending_mask(self, mask: Tensor) -> None:
        """Reset pending-launch slots selected by ``mask`` (``[L]`` bool)."""
        if self.pending_owners is None:
            return
        self.pending_owners[mask] = -1
        assert self.pending_source_planets is not None
        self.pending_source_planets[mask] = -1
        assert self.pending_ships is not None
        self.pending_ships[mask] = 0
        assert self.pending_angle is not None
        self.pending_angle[mask] = 0.0
        assert self.pending_target_slots is not None
        self.pending_target_slots[mask] = -1
        assert self.pending_eta is not None
        self.pending_eta[mask] = 0.0
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid[mask] = 0
        assert self.pending_stash_step is not None
        self.pending_stash_step[mask] = -1

    def _ensure_pending_capacity(self, needed: int) -> None:
        """Ensure ``pending_*`` tensors have at least ``needed`` empty slots."""
        device = self.device
        if self.pending_owners is None:
            initial = max(4, int(needed))
            shape = (initial,)
            self.pending_owners = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_source_planets = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_ships = torch.zeros(shape, dtype=torch.long, device=device)
            self.pending_angle = torch.zeros(shape, dtype=self.dtype, device=device)
            self.pending_target_slots = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_eta = torch.zeros(shape, dtype=self.dtype, device=device)
            self.pending_prev_nfid = torch.zeros(shape, dtype=torch.long, device=device)
            self.pending_stash_step = torch.full(shape, -1, dtype=torch.long, device=device)
            return
        assert self.pending_owners is not None
        empty_count = int((self.pending_owners == -1).sum().item())
        shortage = int(needed) - empty_count
        if shortage <= 0:
            return
        cur_L = int(self.pending_owners.shape[0])
        # Grow generously to amortize.
        extra = max(shortage, cur_L)
        new_L = cur_L + extra
        def _grow(t: Tensor, fill: float | int) -> Tensor:
            extension = torch.full((new_L - cur_L,), fill, dtype=t.dtype, device=device)
            return torch.cat([t, extension], dim=0)
        self.pending_owners = _grow(self.pending_owners, -1)
        assert self.pending_source_planets is not None
        self.pending_source_planets = _grow(self.pending_source_planets, -1)
        assert self.pending_ships is not None
        self.pending_ships = _grow(self.pending_ships, 0)
        assert self.pending_angle is not None
        self.pending_angle = _grow(self.pending_angle, 0.0)
        assert self.pending_target_slots is not None
        self.pending_target_slots = _grow(self.pending_target_slots, -1)
        assert self.pending_eta is not None
        self.pending_eta = _grow(self.pending_eta, 0.0)
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid = _grow(self.pending_prev_nfid, 0)
        assert self.pending_stash_step is not None
        self.pending_stash_step = _grow(self.pending_stash_step, -1)

    def stash_pending_own_launches(
        self,
        *,
        owner_id: int | Tensor,
        source_slots: Tensor,
        ships: Tensor,
        angle: Tensor,
        target_slots: Tensor,
        eta: Tensor,
        valid: Tensor,
        prev_next_fleet_id: int | Tensor,
    ) -> None:
        """Stash this turn's own launches for ID reconciliation on the next obs.

        The caller has already added the bucket contribution via
        :meth:`record_fleet_arrivals` (with ``fleet_ids=None``) but must not
        seed the ``tracked_fleet_ids`` ledger yet — the engine assigns IDs in
        slot-major order across all players, so the agent cannot know its
        real IDs at action time. We stash ``(source_planet_id, ships, angle)``
        for each valid launch in emission order; the next call to
        :meth:`update` pairs them against ``obs.fleets`` entries with
        ``id >= prev_next_fleet_id`` and ``owner == owner_id`` (which are the
        engine's actual IDs for this slot's launches this turn) and writes
        the ledger with those real IDs.

        ``prev_next_fleet_id`` is ``obs.next_fleet_id`` at action time (scalar).

        Inputs are ``[L_in]`` (or broadcastable). Pending rows are appended into
        free slots, growing capacity as needed.
        """
        if not self.track_fleets:
            return
        device = self.device
        valid_mask = valid.to(device=device, dtype=torch.bool).reshape(-1)     # [L_in]
        if not bool(valid_mask.any()):
            return
        src = source_slots.to(device=device, dtype=torch.long).reshape(-1)
        ships_t = ships.to(device=device, dtype=torch.long).reshape(-1)
        angle_t = angle.to(device=device, dtype=self.dtype).reshape(-1)
        tgt_t = target_slots.to(device=device, dtype=torch.long).reshape(-1)
        eta_t = eta.to(device=device, dtype=self.dtype).reshape(-1)
        # Resolve source slot -> planet_id.
        src_safe = src.clamp(min=0, max=max(int(self.P) - 1, 0))
        source_planet_ids = self.planet_ids[src_safe]                          # [L_in]
        L_in = int(valid_mask.shape[0])
        if isinstance(prev_next_fleet_id, Tensor):
            prev_nfid_scalar = int(prev_next_fleet_id.flatten()[0].item())
        else:
            prev_nfid_scalar = int(prev_next_fleet_id)
        prev_nfid_L = torch.full((L_in,), prev_nfid_scalar, dtype=torch.long, device=device)
        owner_scalar = int(owner_id.flatten()[0].item()) if isinstance(owner_id, Tensor) else int(owner_id)
        owner_L = torch.full((L_in,), owner_scalar, dtype=torch.long, device=device)
        stash_step_scalar = int(self.base_step.item()) if isinstance(self.base_step, Tensor) else -1
        stash_step_L = torch.full((L_in,), stash_step_scalar, dtype=torch.long, device=device)

        # Clear any prior pending entries for this owner — a repeat stash for the
        # same owner within a turn replaces the previous stash.
        if self.pending_owners is not None:
            same_owner = self.pending_owners == owner_scalar                   # [L]
            if bool(same_owner.any()):
                self._clear_pending_mask(same_owner)

        per_needed = int(valid_mask.sum().item())
        self._ensure_pending_capacity(per_needed)
        assert self.pending_owners is not None

        # Place valid inputs (in ascending order) into the first empty pending
        # slots (ascending) — preserving emission order.
        empty_slots = torch.nonzero(self.pending_owners == -1, as_tuple=True)[0]
        k_in = torch.nonzero(valid_mask, as_tuple=True)[0]                     # [N]
        slot_in_pending = empty_slots[: k_in.numel()]                          # [N]
        self.pending_owners[slot_in_pending] = owner_L[k_in]
        assert self.pending_source_planets is not None
        self.pending_source_planets[slot_in_pending] = source_planet_ids[k_in]
        assert self.pending_ships is not None
        self.pending_ships[slot_in_pending] = ships_t[k_in]
        assert self.pending_angle is not None
        self.pending_angle[slot_in_pending] = angle_t[k_in]
        assert self.pending_target_slots is not None
        self.pending_target_slots[slot_in_pending] = tgt_t[k_in]
        assert self.pending_eta is not None
        self.pending_eta[slot_in_pending] = eta_t[k_in]
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid[slot_in_pending] = prev_nfid_L[k_in]
        assert self.pending_stash_step is not None
        self.pending_stash_step[slot_in_pending] = stash_step_L[k_in]

    def _reconcile_pending_own_launches(self, obs_tensors: dict) -> None:
        """Pair stashed launches against obs.fleets and seed the ledger with
        engine-assigned IDs.

        Matched stash entries (same owner / source / ships / angle, id >=
        prev_nfid) seed the ledger with their real fleet IDs. Unmatched
        entries are treated as vanished mid-flight — the engine can destroy
        a freshly-launched fleet on its first move via an obstacle the
        agent's swept-pair didn't predict (most commonly a comet between
        source and the predicted target) — and we undo the bucket-arrival
        contribution recorded at stash time so garrison projections stay
        consistent. Still hard-fails when two pending entries match the same
        obs fleet, which signals identical multi-launch from the same source
        that the engine processed in an unexpected order. Extra obs fleets
        (engine-created launches that the planner's swept-pair couldn't
        track, e.g., launches headed OOB) are left alone for
        ``_ingest_obs_fleets`` to handle.
        """
        if not self.track_fleets:
            return
        if self.pending_owners is None or self.tracked_fleet_ids is None:
            return
        active_mask = self.pending_owners != -1                                # [L]
        if not bool(active_mask.any()):
            return
        device = self.device
        step_tensor = obs_tensors.get("step")
        if step_tensor is not None:
            assert self.pending_stash_step is not None
            step_scalar = int(step_tensor.flatten()[0].item()) if isinstance(step_tensor, Tensor) else int(step_tensor)
            advanced = step_scalar > self.pending_stash_step                   # [L]
            active_mask = active_mask & advanced
        if not bool(active_mask.any()):
            return

        fleets = obs_tensors["fleets"].to(device=device)                       # [F, 7]
        fleet_ids = fleets[..., 0].to(dtype=torch.long)                        # [F]
        obs_owner = fleets[..., 1].to(dtype=torch.long)                        # [F]
        obs_angle = fleets[..., 4].to(dtype=self.dtype)                        # [F]
        obs_from = fleets[..., 5].to(dtype=torch.long)                         # [F]
        obs_ships = fleets[..., 6].to(dtype=torch.long)                        # [F]

        assert self.pending_owners is not None
        assert self.pending_source_planets is not None
        assert self.pending_ships is not None
        assert self.pending_angle is not None
        assert self.pending_target_slots is not None
        assert self.pending_eta is not None
        assert self.pending_prev_nfid is not None

        # Pairwise match every obs fleet (rows) against every active pending
        # entry (cols) -> [F, L].
        match_FL = (
            active_mask.unsqueeze(0)
            & (fleet_ids.unsqueeze(1) >= 0)
            & (obs_owner.unsqueeze(1) == self.pending_owners.unsqueeze(0))
            & (obs_from.unsqueeze(1) == self.pending_source_planets.unsqueeze(0))
            & (obs_ships.unsqueeze(1) == self.pending_ships.unsqueeze(0))
            & (obs_angle.unsqueeze(1) == self.pending_angle.unsqueeze(0))
            & (fleet_ids.unsqueeze(1) >= self.pending_prev_nfid.unsqueeze(0))
        )  # [F, L]

        # For each active pending entry, pick the smallest matching obs id.
        INF = torch.iinfo(torch.long).max
        id_for_match = torch.where(
            match_FL,
            fleet_ids.unsqueeze(1).expand_as(match_FL),
            torch.full_like(match_FL, INF, dtype=torch.long),
        )                                                                      # [F, L]
        chosen_id, _ = id_for_match.min(dim=0)                                 # [L]
        # eta_remaining = ceil(stash.eta) - 1; one turn has passed. ``eta_now
        # <= 0`` means the fleet arrived this turn (resolved + removed from obs),
        # so we don't expect an obs match. For eta_now > 0 a missing match means
        # the engine destroyed the fleet mid-flight; treat as vanished: drop the
        # pending entry, skip the ledger insert, and undo the pre-recorded bucket
        # arrival so garrison projections aren't biased by a phantom.
        eta_now = torch.ceil(self.pending_eta).to(dtype=torch.long) - 1
        expect_obs_match = active_mask & (eta_now > 0)
        no_match = expect_obs_match & (chosen_id == INF)
        matched = expect_obs_match & (chosen_id != INF)

        # Detect duplicate assignments among matched entries: two pending entries
        # pointing at the same chosen_id (identical multi-launch from one source
        # processed in an unexpected order).
        if int(active_mask.shape[0]) > 1:
            chosen_for_matched = torch.where(
                matched, chosen_id, torch.full_like(chosen_id, INF)
            )
            sorted_ids, _ = chosen_for_matched.sort()
            dup = bool(
                ((sorted_ids[1:] == sorted_ids[:-1]) & (sorted_ids[1:] != INF)).any()
            )
            if dup:
                raise AssertionError(
                    "Pending-launch reconciliation: multiple pending entries "
                    "resolved to the same engine fleet id. This usually means "
                    "multi-launch from the same source with identical "
                    "(ships, angle) tuples processed in an unexpected order."
                )

        if bool(matched.any()):
            l_idx = torch.where(matched)[0]
            real_ids = chosen_id[l_idx]
            self._ledger_bulk_insert(
                real_ids,
                eta_now[l_idx],
                self.pending_target_slots[l_idx],
                self.pending_owners[l_idx],
                self.pending_ships[l_idx].to(dtype=self.dtype),
            )
        if bool(no_match.any()):
            self._decrement_unmatched_arrivals(no_match)
        # Clear ALL pending entries we just reconciled (eta<=0 cases never make
        # it to the ledger but shouldn't linger either).
        self._clear_pending_mask(active_mask)

    def _decrement_unmatched_arrivals(self, no_match: Tensor) -> None:
        """Undo the bucket-arrival contribution recorded for a launch that
        vanished before reaching its predicted target.

        The pre-record sat at ``buckets[target_slot, ceil(eta)-1, owner]`` at
        stash time. By the time this runs, ``_roll_fleet_buckets_phase1`` has
        already shifted the bucket one step forward, so the relevant index is
        ``ceil(eta)-2 == eta_now-1``. Entries that already rolled off the
        horizon leave nothing to decrement and are skipped.
        """
        assert self.pending_eta is not None
        assert self.pending_owners is not None
        assert self.pending_ships is not None
        assert self.pending_target_slots is not None
        buckets = self._require_fleet_buckets()
        eta_now = torch.ceil(self.pending_eta).to(dtype=torch.long) - 1
        h_idx_now = eta_now - 1
        H = int(self.movement_horizon)
        Aowner = int(buckets.shape[2])
        valid = (
            no_match
            & (h_idx_now >= 0)
            & (h_idx_now < H)
            & (self.pending_target_slots >= 0)
            & (self.pending_target_slots < int(self.P))
            & (self.pending_owners >= 0)
            & (self.pending_owners < Aowner)
            & (self.pending_ships > 0)
        )
        if not bool(valid.any()):
            return
        target = self.pending_target_slots[valid]
        h_idx_sel = h_idx_now[valid]
        owner_sel = self.pending_owners[valid]
        ships_sel = self.pending_ships[valid].to(dtype=self.dtype)
        buckets.index_put_(
            (target, h_idx_sel, owner_sel),
            -ships_sel,
            accumulate=True,
        )
        self._mark_garrison_dirty(target, h_idx_sel + 1)

    def record_fleet_arrivals(
        self,
        *,
        target_slots: Tensor,
        owner_ids: Tensor | int,
        ships: Tensor,
        eta: Tensor,
        valid: Tensor | None = None,
    ) -> None:
        """Add predicted arrivals into the fleet buckets.

        ``eta`` is expressed in steps from the current observation frame; bucket
        ``eta=1`` is stored at horizon index ``0``.
        """
        buckets = self._require_fleet_buckets()
        target_slots, ships, eta = torch.broadcast_tensors(
            target_slots.to(device=self.device, dtype=torch.long),
            ships.to(device=self.device, dtype=self.dtype),
            eta.to(device=self.device, dtype=self.dtype),
        )
        if isinstance(owner_ids, int):
            owner = torch.full_like(target_slots, int(owner_ids), dtype=torch.long, device=self.device)
        else:
            owner = torch.broadcast_to(owner_ids.to(device=self.device, dtype=torch.long), target_slots.shape)
        if valid is None:
            valid_mask = torch.ones_like(target_slots, dtype=torch.bool)
        else:
            valid_mask = torch.broadcast_to(valid.to(device=self.device, dtype=torch.bool), target_slots.shape)
        h_idx = torch.ceil(eta).to(dtype=torch.long) - 1
        valid_mask = (
            valid_mask
            & (target_slots >= 0)
            & (target_slots < self.P)
            & (owner >= 0)
            & (owner < int(buckets.shape[2]))
            & (h_idx >= 0)
            & (h_idx < int(self.movement_horizon))
            & (ships > 0.0)
        )
        if not bool(valid_mask.any()):
            return
        buckets.index_put_(
            (
                target_slots[valid_mask],
                h_idx[valid_mask],
                owner[valid_mask],
            ),
            ships[valid_mask],
            accumulate=True,
        )
        self._mark_garrison_dirty(
            target_slots[valid_mask],
            h_idx[valid_mask] + 1,
        )

    def _normalize_garrison_slots(self, planet_slots: Tensor | None) -> tuple[Tensor, torch.Size]:
        if planet_slots is None:
            slots = torch.arange(self.P, dtype=torch.long, device=self.device)
            return slots, slots.shape
        raw = planet_slots.to(device=self.device, dtype=torch.long)
        out_prefix = raw.shape
        slots = raw.reshape(-1).clamp(0, max(self.P - 1, 0))
        return slots, out_prefix

    def _ensure_garrison_cache(self) -> None:
        self._ensure_garrison_cache_impl()

    def _ensure_garrison_cache_impl(self) -> None:
        expected_owner = (self.P, int(self.movement_horizon) + 1)
        expected_dirty = (self.P,)
        if (
            self.garrison_owner_cache is not None
            and self.garrison_ships_cache is not None
            and self.garrison_pre_combat_owner_cache is not None
            and self.garrison_pre_combat_ships_cache is not None
            and self.garrison_dirty_from is not None
            and tuple(self.garrison_owner_cache.shape) == expected_owner
            and tuple(self.garrison_ships_cache.shape) == expected_owner
            and tuple(self.garrison_pre_combat_owner_cache.shape) == expected_owner
            and tuple(self.garrison_pre_combat_ships_cache.shape) == expected_owner
            and tuple(self.garrison_dirty_from.shape) == expected_dirty
            and self.garrison_owner_cache.device == self.device
            and self.garrison_ships_cache.device == self.device
        ):
            return
        horizon = int(self.movement_horizon)
        self.garrison_owner_cache = torch.full(
            (self.P, horizon + 1),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.garrison_ships_cache = torch.zeros(
            self.P,
            horizon + 1,
            dtype=self.dtype,
            device=self.device,
        )
        # Pre-combat caches: planet state just before each step's combat (after
        # production has been credited). At k=0 there is no prior step, so the
        # observation IS both pre- and post-combat.
        self.garrison_pre_combat_owner_cache = self.garrison_owner_cache.clone()
        self.garrison_pre_combat_ships_cache = self.garrison_ships_cache.clone()
        self.garrison_owner_cache[:, 0] = self.planet_owner
        self.garrison_ships_cache[:, 0] = self.planet_ships
        self.garrison_pre_combat_owner_cache[:, 0] = self.planet_owner
        self.garrison_pre_combat_ships_cache[:, 0] = self.planet_ships
        self.garrison_dirty_from = torch.zeros(self.P, dtype=torch.long, device=self.device)

    def _refresh_garrison_projection(self, slots: Tensor, *, requested_horizon: int | None = None) -> None:
        self._ensure_garrison_cache()
        assert self.fleet_buckets is not None
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_dirty_from is not None

        p_idx = torch.unique(slots.reshape(-1).clamp(min=0, max=max(self.P - 1, 0)))
        if p_idx.numel() == 0:
            return

        dirty = self.garrison_dirty_from[p_idx]
        horizon = int(
            self.movement_horizon
            if requested_horizon is None
            else max(0, min(int(requested_horizon), int(self.movement_horizon)))
        )
        needs_refresh = dirty <= horizon
        if not bool(needs_refresh.any()):
            return

        p_idx = p_idx[needs_refresh]
        owner = self.planet_owner[p_idx].clone()
        ships = self.planet_ships[p_idx].clone()
        self.garrison_owner_cache[p_idx, 0] = owner
        self.garrison_ships_cache[p_idx, 0] = ships
        assert self.garrison_pre_combat_owner_cache is not None
        assert self.garrison_pre_combat_ships_cache is not None
        self.garrison_pre_combat_owner_cache[p_idx, 0] = owner
        self.garrison_pre_combat_ships_cache[p_idx, 0] = ships
        prod = self.planet_prod[p_idx]

        if horizon == 0:
            self.garrison_dirty_from[p_idx] = horizon + 1
            return

        self._fill_garrison_trajectory(
            p_idx=p_idx,
            init_owner=owner,
            init_ships=ships,
            prod=prod,
            horizon=horizon,
        )

        self.garrison_dirty_from[p_idx] = horizon + 1

    def _fill_garrison_trajectory(
        self,
        *,
        p_idx: Tensor,
        init_owner: Tensor,
        init_ships: Tensor,
        prod: Tensor,
        horizon: int,
    ) -> None:
        """Fill ``garrison_{owner,ships}_cache`` for steps ``1..horizon``.

        Decomposes the per-pair recurrence into two halves so the GPU does very
        little sequential work:

        - **Half A** (vectorized): compute the per-step combat survivor
          ``(top_owner, top1 - top2)`` over the player axis for all ``H`` steps in a
          single fused tensor op. The survivor is a pure function of that step's
          arrival vector and does not depend on the planet state, so this carries no
          inter-step dependency. Replaces ``H`` per-step ``topk`` calls with one.
        - **Half B** (sequential, branchless): walk ``k = 1..H`` advancing
          ``(state_owner, state_ships)``. Every operation is a fused ``where`` —
          there is no host sync (no ``bool(has_arrivals.any())``), no boolean
          indexing, and no per-step ``topk``. Each iteration is ~5 element-wise
          kernels over ``[N_complex]``, vs ~12 kernels + a host sync previously.

        Plus a closed-form fast path for "simple" pairs (no arrivals over the
        horizon and the planet stays alive throughout). For those pairs, owner is
        constant and ships grow linearly: ``ships[k] = ships[0] + prod * k``. We
        write the entire trajectory in one tensor assignment instead of iterating.
        Most planets in a typical match satisfy this, so the recurrent path runs
        on a small fraction of pairs.
        """
        assert self.fleet_buckets is not None
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_pre_combat_owner_cache is not None
        assert self.garrison_pre_combat_ships_cache is not None

        H = int(horizon)
        N = int(p_idx.numel())
        if N == 0 or H == 0:
            return

        # ``alive_by_step[k, p]`` is the alive mask AT END of step ``k`` (= the
        # position frame for the k-th lookahead). For step k's transition we need
        # alive at the start (``alive_step[k-1]``) and at the end (``alive_step[k]``).
        alive_step = self.alive_by_step[:, p_idx].transpose(0, 1)  # [N, H+1]
        alive_before = alive_step[:, :H]                          # [N, H]
        alive_now = alive_step[:, 1:]                             # [N, H]
        # ``fleet_buckets[p, k, a]`` = ships from owner ``a`` arriving at step ``k+1``.
        arrivals = self.fleet_buckets[p_idx, :H, :]               # [N, H, A]

        # A pair is "simple" if no fleets ever arrive at this planet over the
        # horizon AND the planet stays alive throughout. For such pairs the
        # trajectory is purely additive: owner constant, ships grow by ``prod``
        # per step (or stay zero for neutral planets). Most planets in a typical
        # match fit this profile, so this is the big algorithmic win — these
        # pairs skip the per-step recurrence entirely.
        has_any_arrival = (arrivals > 0.0).any(dim=-1).any(dim=-1)  # [N]
        alive_all_true = alive_step.all(dim=1)                       # [N]
        simple_mask = (~has_any_arrival) & alive_all_true            # [N]

        # Cache the per-pair alive trajectory before we filter to complex pairs;
        # we'll need it for the tail-continuation step below.
        alive_step_full = alive_step

        # One host sync per refresh to count simple vs complex pairs.
        n_simple = int(simple_mask.sum().item())
        n_complex = N - n_simple

        if n_simple > 0:
            simple_p = p_idx[simple_mask]
            simple_owner = init_owner[simple_mask]
            simple_ships = init_ships[simple_mask]
            simple_prod = prod[simple_mask]
            # Production accrues only for owned planets; the ``(owner >= 0)`` factor
            # collapses neutral and dead planets to zero growth.
            owner_alive_factor = (simple_owner >= 0).to(dtype=simple_ships.dtype)
            k_range = torch.arange(1, H + 1, device=self.device, dtype=simple_ships.dtype)
            ships_traj = (
                simple_ships.unsqueeze(1)
                + simple_prod.unsqueeze(1)
                * owner_alive_factor.unsqueeze(1)
                * k_range.unsqueeze(0)
            )                                                         # [N_simple, H]
            owner_traj = simple_owner.unsqueeze(1).expand(-1, H)
            # One fused write per cache, covers every step 1..H simultaneously.
            self.garrison_owner_cache[simple_p, 1 : H + 1] = owner_traj
            self.garrison_ships_cache[simple_p, 1 : H + 1] = ships_traj
            # Simple-path pairs have no arrivals across the horizon, so
            # pre-combat state at every step equals the post-combat state.
            self.garrison_pre_combat_owner_cache[simple_p, 1 : H + 1] = owner_traj
            self.garrison_pre_combat_ships_cache[simple_p, 1 : H + 1] = ships_traj

        if n_complex == 0:
            return

        complex_mask = ~simple_mask
        cp = p_idx[complex_mask]
        arrivals_c = arrivals[complex_mask]                           # [N_c, H, A]
        alive_before_c = alive_before[complex_mask]                   # [N_c, H]
        alive_now_c = alive_now[complex_mask]                         # [N_c, H]
        alive_step_c = alive_step_full[complex_mask]                  # [N_c, H+1]
        state_owner = init_owner[complex_mask].clone()                # [N_c]
        state_ships = init_ships[complex_mask].clone()                # [N_c]
        prod_c = prod[complex_mask]                                   # [N_c]

        # Half A: per-step (top1 - top2) survivor over the player axis. No
        # cross-step dependency, so it runs in one fused op rather than ``H``
        # times in the inner loop.
        A = int(arrivals_c.shape[-1])
        if A >= 2:
            top2 = arrivals_c.topk(k=2, dim=-1)
            top_ships_traj = top2.values[..., 0]
            second_ships_traj = top2.values[..., 1]
            top_owner_traj = top2.indices[..., 0].to(dtype=torch.long)
        else:
            top_ships_traj, top_owner_traj = arrivals_c.max(dim=-1)
            second_ships_traj = torch.zeros_like(top_ships_traj)
            top_owner_traj = top_owner_traj.to(dtype=torch.long)
        # Ties leave no survivor (mutual annihilation). Where both top values
        # are zero (no arrivals at this step), ``survivor_ships`` is also zero
        # and ``has_combat`` will mask the step out below.
        tied = top_ships_traj == second_ships_traj
        survivor_ships_traj = torch.where(
            tied,
            torch.zeros_like(top_ships_traj),
            (top_ships_traj - second_ships_traj).clamp(min=0.0),
        )                                                          # [N_c, H]
        survivor_owner_traj = top_owner_traj                       # [N_c, H]

        # Scalar broadcast templates for the ``where``-based death reset; using
        # scalars keeps each per-step ``where`` to a single small kernel.
        zero_ships_scalar = torch.zeros((), dtype=state_ships.dtype, device=self.device)
        neg_one_owner_scalar = torch.full((), -1, dtype=state_owner.dtype, device=self.device)
        zero_prod_scalar = torch.zeros((), dtype=prod_c.dtype, device=self.device)

        # Horizon-trim optimization: identify the latest step at which ANY complex
        # pair has a structural transition. Beyond that step every pair's
        # trajectory is determined purely by production accumulation, so we can
        # replace the rest of the H-step recurrence with one closed-form tensor
        # write (analogous to the simple-pair fast path). Two kinds of structural
        # transitions can change a pair's state:
        #   - a non-tied combat survivor lands while the planet is alive
        #     (``has_combat = (s_ships > 0) & alive_now``);
        #   - the planet's alive state flips (death or respawn) at this step.
        combat_event_per_step = (survivor_ships_traj > 0.0) & alive_now_c   # [N_c, H]
        alive_change_per_step = alive_before_c != alive_now_c                # [N_c, H]
        any_event_per_step = (combat_event_per_step | alive_change_per_step).any(dim=0)  # [H]
        # Map each step k ∈ [1, H] to itself if there's an event there, else 0.
        # The max collapses to the largest ``k`` with any event, or 0 if none.
        arange_h = torch.arange(1, H + 1, device=self.device, dtype=torch.long)
        k_last_tensor = torch.where(
            any_event_per_step,
            arange_h,
            torch.zeros_like(arange_h),
        ).max()
        # One host sync per refresh: we need ``k_last`` on the host to size the
        # Python loop. The win from shrinking the loop dwarfs the sync cost.
        k_last = int(k_last_tensor.item())

        loop_iters = max(0, k_last)
        tail_steps = H - loop_iters

        if loop_iters > 0:
            # Half B: branchless H-step recurrence. The ``(state_owner, state_ships)``
            # pair has a real cross-step dependency — an attacker capturing the planet
            # at step k flips who produces in subsequent steps — so we must walk
            # ``loop_iters`` sequentially. Each iteration is fully branchless: no host
            # sync, no boolean indexing, no ``topk``. Just element-wise ``where``s
            # over ``[N_c]``.
            for k in range(1, loop_iters + 1):
                a_before = alive_before_c[:, k - 1]
                a_now = alive_now_c[:, k - 1]
                s_owner = survivor_owner_traj[:, k - 1]
                s_ships = survivor_ships_traj[:, k - 1]

                # Production: owned planets that were alive at the start of this step.
                produces = a_before & (state_owner >= 0)
                state_ships = state_ships + torch.where(produces, prod_c, zero_prod_scalar)

                # Snapshot pre-combat state: this is what an attacker arriving
                # at step ``k`` will face from the planet itself, before any
                # same-step attacker combat is applied. Captured here so a
                # planner can synthesize "what if I also arrive this turn?"
                # using the engine's combat rule.
                pre_owner = torch.where(a_now, state_owner, neg_one_owner_scalar)
                pre_ships = torch.where(a_now, state_ships, zero_ships_scalar)
                self.garrison_pre_combat_owner_cache[cp, k] = pre_owner
                self.garrison_pre_combat_ships_cache[cp, k] = pre_ships

                # Combat against the precomputed step-k survivor. Three cases collapse
                # into two ``where`` chains masked by ``has_combat``:
                #   same owner: state_ships += s_ships  (reinforcement)
                #   ~same & state_ships <  s_ships: planet flips, ships = s_ships - state_ships
                #   ~same & state_ships >= s_ships: garrison reduced by s_ships
                has_combat = (s_ships > 0.0) & a_now
                same = state_owner == s_owner
                diff = state_ships - s_ships  # signed; |diff| is the post-combat ships count
                attacker_wins = (~same) & (diff < 0.0)
                combat_ships = torch.where(same, state_ships + s_ships, diff.abs())
                combat_owner = torch.where(attacker_wins, s_owner, state_owner)
                state_ships = torch.where(has_combat, combat_ships, state_ships)
                state_owner = torch.where(has_combat, combat_owner, state_owner)

                # End-of-step death reset: if the planet despawns this step it has
                # no owner and no garrison from now on.
                state_owner = torch.where(a_now, state_owner, neg_one_owner_scalar)
                state_ships = torch.where(a_now, state_ships, zero_ships_scalar)

                self.garrison_owner_cache[cp, k] = state_owner
                self.garrison_ships_cache[cp, k] = state_ships

        if tail_steps > 0:
            # By construction of ``k_last``, no complex pair has a structural event
            # at any step in ``(k_last, H]``: alive is constant, no combat survivors,
            # no captures. So the trajectory across the tail is closed-form:
            #   ships[k] = state_ships + prod * (k - k_last) * (alive AND owned)
            #   owner[k] = state_owner    (constant)
            # We still need to apply the "pending" death reset for pairs whose
            # ``alive_step[k_last]`` is False. When ``k_last >= 1`` the loop's last
            # iteration already did this; when ``k_last == 0`` we apply it here so
            # the closed-form formula matches the original loop's output.
            alive_at_k_last = alive_step_c[:, k_last]                  # [N_c]
            state_owner = torch.where(alive_at_k_last, state_owner, neg_one_owner_scalar)
            state_ships = torch.where(alive_at_k_last, state_ships, zero_ships_scalar)
            # Production multiplier: 1 only for pairs that are alive AND owned at
            # ``k_last`` (and therefore for the entire tail by definition).
            owner_alive_factor = (
                (state_owner >= 0).to(dtype=state_ships.dtype)
                * alive_at_k_last.to(dtype=state_ships.dtype)
            )                                                          # [N_c]
            # ``dk_range[i]`` = i + 1, the offset from ``k_last`` to step ``k_last+1+i``.
            dk_range = torch.arange(
                1, tail_steps + 1, device=self.device, dtype=state_ships.dtype
            )                                                          # [tail_steps]
            ships_traj_tail = (
                state_ships.unsqueeze(1)
                + prod_c.unsqueeze(1)
                * owner_alive_factor.unsqueeze(1)
                * dk_range.unsqueeze(0)
            )                                                          # [N_c, tail_steps]
            owner_traj_tail = state_owner.unsqueeze(1).expand(-1, tail_steps)
            self.garrison_owner_cache[cp, k_last + 1 : H + 1] = owner_traj_tail
            self.garrison_ships_cache[cp, k_last + 1 : H + 1] = ships_traj_tail
            # Tail has no structural events (no combat, no death), so the
            # pre-combat state at every tail step equals the post-combat
            # state — production only.
            self.garrison_pre_combat_owner_cache[cp, k_last + 1 : H + 1] = owner_traj_tail
            self.garrison_pre_combat_ships_cache[cp, k_last + 1 : H + 1] = ships_traj_tail

    def _roll_garrison_projection(self) -> None:
        if (
            self.garrison_owner_cache is None
            or self.garrison_ships_cache is None
            or self.garrison_pre_combat_owner_cache is None
            or self.garrison_pre_combat_ships_cache is None
            or self.garrison_dirty_from is None
        ):
            return
        horizon = int(self.movement_horizon)
        if horizon > 0:
            self.garrison_owner_cache[:, :-1] = self.garrison_owner_cache[:, 1:].clone()
            self.garrison_ships_cache[:, :-1] = self.garrison_ships_cache[:, 1:].clone()
            self.garrison_pre_combat_owner_cache[:, :-1] = (
                self.garrison_pre_combat_owner_cache[:, 1:].clone()
            )
            self.garrison_pre_combat_ships_cache[:, :-1] = (
                self.garrison_pre_combat_ships_cache[:, 1:].clone()
            )
            self.garrison_dirty_from = (self.garrison_dirty_from - 1).clamp(min=0)
            self.garrison_dirty_from = torch.minimum(
                self.garrison_dirty_from,
                torch.full_like(self.garrison_dirty_from, horizon),
            )
        else:
            self.garrison_dirty_from[:] = 0

    def _refresh_garrison_base(self, built: dict[str, Tensor]) -> None:
        owner = built["owner"].to(device=self.device, dtype=torch.long)
        ships = built["ships"].to(device=self.device, dtype=self.dtype)
        prod = built["prod"].to(device=self.device, dtype=self.dtype)
        prod_changed = tuple(self.planet_prod.shape) != tuple(prod.shape) or (self.planet_prod != prod)
        self.planet_owner = owner
        self.planet_ships = ships
        self.planet_prod = prod
        if self.garrison_owner_cache is None or self.garrison_ships_cache is None or self.garrison_dirty_from is None:
            return
        base_changed = (
            (self.garrison_owner_cache[:, 0] != owner)
            | (self.garrison_ships_cache[:, 0] != ships)
        )
        self.garrison_owner_cache[:, 0] = owner
        self.garrison_ships_cache[:, 0] = ships
        if self.garrison_pre_combat_owner_cache is not None:
            self.garrison_pre_combat_owner_cache[:, 0] = owner
        if self.garrison_pre_combat_ships_cache is not None:
            self.garrison_pre_combat_ships_cache[:, 0] = ships
        if bool(base_changed.any()):
            self.garrison_dirty_from[base_changed] = 0
        if isinstance(prod_changed, Tensor) and bool(prod_changed.any()):
            self.garrison_dirty_from[prod_changed] = torch.minimum(
                self.garrison_dirty_from[prod_changed],
                torch.ones_like(self.garrison_dirty_from[prod_changed]),
            )
        elif not isinstance(prod_changed, Tensor) and prod_changed:
            self.garrison_dirty_from[:] = torch.minimum(
                self.garrison_dirty_from,
                torch.ones_like(self.garrison_dirty_from),
            )

    def _mark_garrison_dirty(self, planet_idx: Tensor, start_step: Tensor | int) -> None:
        if self.garrison_dirty_from is None:
            return
        p = planet_idx.to(device=self.device, dtype=torch.long)
        if isinstance(start_step, int):
            start = torch.full((), int(start_step), dtype=torch.long, device=self.device)
        else:
            start = start_step.to(device=self.device, dtype=torch.long)
        p, start = torch.broadcast_tensors(p, start)
        p = p.reshape(-1)
        start = start.reshape(-1)
        if p.numel() == 0:
            return
        start = start.clamp(min=0, max=int(self.movement_horizon))
        valid = (p >= 0) & (p < self.P)
        if not bool(valid.any()):
            return
        p = p[valid]
        start = start[valid]
        flat = self.garrison_dirty_from
        unique_idx, inverse = torch.unique(p, return_inverse=True)
        if unique_idx.numel() == p.numel():
            flat[unique_idx] = torch.minimum(flat[unique_idx], start)
            return
        sentinel = int(self.movement_horizon) + 1
        candidate = torch.full((unique_idx.shape[0],), sentinel, dtype=torch.long, device=self.device)
        candidate.scatter_reduce_(0, inverse, start, reduce="amin", include_self=True)
        flat[unique_idx] = torch.minimum(flat[unique_idx], candidate)

    def _mark_garrison_dirty_all(self, start_step: int) -> None:
        if self.garrison_dirty_from is None:
            return
        self.garrison_dirty_from = torch.minimum(
            self.garrison_dirty_from,
            torch.full_like(self.garrison_dirty_from, int(start_step)),
        )

    def _init_fleet_tracking(self, obs_tensors: dict, *, reset_ledger: bool) -> None:
        _ = reset_ledger
        player_count = _resolve_player_count(obs_tensors, self.player_count)
        self.player_count = int(player_count)
        self.fleet_buckets = torch.zeros(
            self.P,
            int(self.movement_horizon),
            int(player_count),
            dtype=self.dtype,
            device=self.device,
        )
        step = obs_tensors["step"].to(device=self.device, dtype=torch.long)
        self.fleet_last_step = step.detach().clone()
        M = max(1, int(self.max_tracked_fleets))
        self.max_tracked_fleets = M
        self.tracked_fleet_ids = torch.full((M,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_eta = torch.zeros((M,), dtype=torch.long, device=self.device)
        self.tracked_fleet_target_slot = torch.full((M,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_owner = torch.zeros((M,), dtype=torch.long, device=self.device)
        self.tracked_fleet_ships = torch.zeros((M,), dtype=self.dtype, device=self.device)
        if self.garrison_dirty_from is not None:
            self.garrison_dirty_from[:] = torch.minimum(
                self.garrison_dirty_from,
                torch.full_like(self.garrison_dirty_from, 1),
            )

    def _clear_tracked_rows(self) -> None:
        if (
            self.tracked_fleet_ids is None
            or self.tracked_fleet_eta is None
            or self.tracked_fleet_target_slot is None
            or self.tracked_fleet_owner is None
            or self.tracked_fleet_ships is None
        ):
            return
        self.tracked_fleet_ids[:] = -1
        self.tracked_fleet_eta[:] = 0
        self.tracked_fleet_target_slot[:] = -1
        self.tracked_fleet_owner[:] = 0
        self.tracked_fleet_ships[:] = 0.0

    def _ledger_bulk_insert(
        self,
        fleet_ids: Tensor,
        eta_remaining: Tensor,
        target_slots: Tensor,
        owners: Tensor,
        ships: Tensor,
    ) -> None:
        if fleet_ids.numel() == 0:
            return
        assert self.tracked_fleet_ids is not None
        assert self.tracked_fleet_eta is not None
        assert self.tracked_fleet_target_slot is not None
        assert self.tracked_fleet_owner is not None
        assert self.tracked_fleet_ships is not None
        M = int(self.tracked_fleet_ids.shape[0])
        fleet_ids = fleet_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        eta_remaining = eta_remaining.to(device=self.device, dtype=torch.long).reshape(-1)
        target_slots = target_slots.to(device=self.device, dtype=torch.long).reshape(-1)
        owners = owners.to(device=self.device, dtype=torch.long).reshape(-1)
        ships = ships.to(device=self.device, dtype=self.dtype).reshape(-1)
        valid_rows = fleet_ids >= 0
        if not bool(valid_rows.any()):
            return
        fleet_ids = fleet_ids[valid_rows]
        eta_remaining = eta_remaining[valid_rows]
        target_slots = target_slots[valid_rows]
        owners = owners[valid_rows]
        ships = ships[valid_rows]
        n = int(fleet_ids.numel())
        empty_mask = self.tracked_fleet_ids == -1
        empty_count = int(empty_mask.sum().item())
        if n > empty_count:
            occupied_count = M - empty_count
            self._grow_ledger_capacity(occupied_count + n)
            assert self.tracked_fleet_ids is not None
            empty_mask = self.tracked_fleet_ids == -1

        # Place the rows into the first ``n`` empty ledger slots, ascending —
        # which preserves input order (each row keeps its emission rank).
        empty_slots = torch.nonzero(empty_mask, as_tuple=True)[0]
        slot_idx = empty_slots[:n]
        self.tracked_fleet_ids[slot_idx] = fleet_ids
        self.tracked_fleet_eta[slot_idx] = eta_remaining
        self.tracked_fleet_target_slot[slot_idx] = target_slots
        self.tracked_fleet_owner[slot_idx] = owners
        self.tracked_fleet_ships[slot_idx] = ships

    def _grow_ledger_capacity(self, required_capacity: int) -> None:
        if (
            self.tracked_fleet_ids is None
            or self.tracked_fleet_eta is None
            or self.tracked_fleet_target_slot is None
            or self.tracked_fleet_owner is None
            or self.tracked_fleet_ships is None
        ):
            return
        old_capacity = int(self.tracked_fleet_ids.shape[0])
        target_capacity = max(int(required_capacity), old_capacity)
        if target_capacity <= old_capacity:
            return
        new_capacity = max(target_capacity, old_capacity * 2)
        old_ids = self.tracked_fleet_ids
        old_eta = self.tracked_fleet_eta
        old_tgt = self.tracked_fleet_target_slot
        old_owner = self.tracked_fleet_owner
        old_ships = self.tracked_fleet_ships
        self.tracked_fleet_ids = torch.full((new_capacity,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_eta = torch.zeros((new_capacity,), dtype=torch.long, device=self.device)
        self.tracked_fleet_target_slot = torch.full((new_capacity,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_owner = torch.zeros((new_capacity,), dtype=torch.long, device=self.device)
        self.tracked_fleet_ships = torch.zeros((new_capacity,), dtype=self.dtype, device=self.device)
        self.tracked_fleet_ids[:old_capacity] = old_ids
        self.tracked_fleet_eta[:old_capacity] = old_eta
        self.tracked_fleet_target_slot[:old_capacity] = old_tgt
        self.tracked_fleet_owner[:old_capacity] = old_owner
        self.tracked_fleet_ships[:old_capacity] = old_ships

    def _ledger_decrement_and_expire(self) -> None:
        if (
            self.tracked_fleet_ids is None
            or self.tracked_fleet_eta is None
            or self.tracked_fleet_target_slot is None
            or self.tracked_fleet_owner is None
            or self.tracked_fleet_ships is None
        ):
            return
        valid = self.tracked_fleet_ids >= 0
        eta = torch.where(valid, self.tracked_fleet_eta - 1, self.tracked_fleet_eta)
        expire = valid & (eta <= 0)
        self.tracked_fleet_eta = eta
        self.tracked_fleet_ids = torch.where(expire, torch.full_like(self.tracked_fleet_ids, -1), self.tracked_fleet_ids)
        self.tracked_fleet_eta = torch.where(expire, torch.zeros_like(self.tracked_fleet_eta), self.tracked_fleet_eta)
        self.tracked_fleet_target_slot = torch.where(
            expire,
            torch.full_like(self.tracked_fleet_target_slot, -1),
            self.tracked_fleet_target_slot,
        )
        self.tracked_fleet_owner = torch.where(
            expire,
            torch.zeros_like(self.tracked_fleet_owner),
            self.tracked_fleet_owner,
        )
        self.tracked_fleet_ships = torch.where(
            expire,
            torch.zeros_like(self.tracked_fleet_ships),
            self.tracked_fleet_ships,
        )

    def _roll_fleet_buckets_phase1(self, current_step: Tensor) -> None:
        if self.fleet_buckets is None or self.fleet_last_step is None:
            return
        step = current_step.to(device=self.device, dtype=torch.long)
        delta = step - self.fleet_last_step.to(device=self.device, dtype=torch.long)
        horizon = int(self.movement_horizon)
        reset = bool((delta < 0) | (step <= 0))
        if reset:
            self.fleet_buckets[:] = 0.0
            self._clear_tracked_rows()
            self._mark_garrison_dirty_all(1)

        rolled_once = (not reset) and bool(delta == 1)
        if rolled_once and horizon > 0:
            self.fleet_buckets[:, :-1, :] = self.fleet_buckets[:, 1:, :].clone()
            self.fleet_buckets[:, -1, :] = 0.0
            self._ledger_decrement_and_expire()
            self._mark_garrison_dirty_all(1)

        delta_bad = (not reset) and bool(delta > 1)
        if delta_bad:
            self._reset_fleet_tracking()

        self.fleet_last_step = step.detach().clone()

    def _reset_fleet_tracking(self) -> None:
        if self.fleet_buckets is None:
            return
        self.fleet_buckets[:] = 0.0
        self._clear_tracked_rows()
        self._mark_garrison_dirty_all(1)

    def _ingest_obs_fleets(self, obs_tensors: dict) -> None:
        if self.fleet_buckets is None or self.tracked_fleet_ids is None or int(self.movement_horizon) <= 0:
            return
        fleets = obs_tensors["fleets"].to(device=self.device, dtype=self.dtype)
        fleet_ids = fleets[..., 0].to(dtype=torch.long)
        alive = fleet_ids >= 0
        # Pairwise compare every observed fleet id against every ledger row id;
        # shape ``[F_obs, M_ledger]`` collapsed by ``any(dim=-1)``. New (untracked)
        # alive fleets get their arrival estimated and recorded.
        tracked = (fleet_ids.unsqueeze(1) == self.tracked_fleet_ids.unsqueeze(0)).any(dim=1)
        process_mask = alive & ~tracked
        n_alive = int(alive.sum().item())
        n_tracked = int((alive & tracked).sum().item())
        n_to_process = n_alive - n_tracked
        if n_to_process == 0:
            return
        fleet_slot = torch.where(process_mask)[0]
        proc_ids = fleet_ids[fleet_slot]
        estimate = _estimate_new_fleet_arrivals(movement=self, obs_fleets=fleets, fleet_slot=fleet_slot)
        valid_owner = (estimate["owner"] >= 0) & (estimate["owner"] < int(self.fleet_buckets.shape[2]))
        valid_hit = estimate["has_hit"] & valid_owner
        if not bool(valid_hit.any()):
            return
        buckets = self._require_fleet_buckets()
        buckets.index_put_(
            (
                estimate["target_slot"][valid_hit],
                estimate["eta_index"][valid_hit],
                estimate["owner"][valid_hit],
            ),
            estimate["ships"][valid_hit],
            accumulate=True,
        )
        self._mark_garrison_dirty(
            estimate["target_slot"][valid_hit],
            estimate["eta_index"][valid_hit] + 1,
        )
        eta_remaining = estimate["eta_index"][valid_hit].to(dtype=torch.long) + 1
        self._ledger_bulk_insert(
            proc_ids[valid_hit],
            eta_remaining,
            estimate["target_slot"][valid_hit],
            estimate["owner"][valid_hit],
            estimate["ships"][valid_hit],
        )

    def _reconcile_obs_fleets(self, obs_tensors: dict) -> None:
        """Drop ledger entries whose fleet is no longer in obs.

        ``record_fleet_arrivals`` writes a fleet's predicted arrival into both
        ``fleet_buckets`` and the tracked-fleet ledger at launch time. If the
        engine destroys the fleet before it arrives (sun crossing, OOB,
        unintended planet collision), the fleet disappears from ``obs.fleets``
        but neither ``_ingest_obs_fleets`` nor ``_ledger_decrement_and_expire``
        knows to evict it — ingest only adds, decrement only fires at eta=0.

        This pass walks ``tracked_fleet_ids``, checks each non-empty entry
        against the current ``obs.fleets[..., 0]``, and for any phantom
        (in-ledger, in-flight, not-in-obs) subtracts its recorded ships from
        ``fleet_buckets`` at the entry's stored ``(target_slot, eta-1, owner)``
        and clears the row. Marks the touched garrison cells dirty so the next
        ``garrison_status`` query rebuilds them.
        """
        if (
            self.fleet_buckets is None
            or self.tracked_fleet_ids is None
            or self.tracked_fleet_eta is None
            or self.tracked_fleet_target_slot is None
            or self.tracked_fleet_owner is None
            or self.tracked_fleet_ships is None
            or int(self.movement_horizon) <= 0
        ):
            return
        obs_ids = obs_tensors["fleets"][..., 0].to(device=self.device, dtype=torch.long)  # [F]
        in_flight = (self.tracked_fleet_ids >= 0) & (self.tracked_fleet_eta > 0)
        if not bool(in_flight.any()):
            return
        # ``[M, F]`` pairwise compare; ``any(dim=-1)`` gives ledger-side in-obs.
        match = (self.tracked_fleet_ids.unsqueeze(1) == obs_ids.unsqueeze(0)).any(dim=1)
        phantom = in_flight & ~match
        if not bool(phantom.any()):
            return
        m_idx = torch.where(phantom)[0]
        h_idx = (self.tracked_fleet_eta[m_idx] - 1).clamp(min=0)
        P = int(self.fleet_buckets.shape[0])
        H = int(self.fleet_buckets.shape[1])
        A = int(self.fleet_buckets.shape[2])
        in_horizon = h_idx < H
        if not bool(in_horizon.any()):
            self.tracked_fleet_ids[m_idx] = -1
            self.tracked_fleet_eta[m_idx] = 0
            self.tracked_fleet_target_slot[m_idx] = -1
            self.tracked_fleet_owner[m_idx] = 0
            self.tracked_fleet_ships[m_idx] = 0.0
            return
        m_sel = m_idx[in_horizon]
        h_sel = h_idx[in_horizon]
        slots = self.tracked_fleet_target_slot[m_sel].clamp(min=0, max=max(P - 1, 0))
        owners = self.tracked_fleet_owner[m_sel].clamp(min=0, max=max(A - 1, 0))
        ships = self.tracked_fleet_ships[m_sel]
        self.fleet_buckets.index_put_(
            (slots, h_sel, owners),
            -ships,
            accumulate=True,
        )
        # ``h_sel`` is the bucket index; ``k = h_sel + 1`` is the corresponding
        # arrival step in garrison-projection coordinates.
        self._mark_garrison_dirty(slots, h_sel + 1)
        # Clear every phantom row (in-horizon and out-of-horizon alike).
        self.tracked_fleet_ids[m_idx] = -1
        self.tracked_fleet_eta[m_idx] = 0
        self.tracked_fleet_target_slot[m_idx] = -1
        self.tracked_fleet_owner[m_idx] = 0
        self.tracked_fleet_ships[m_idx] = 0.0

    def _require_fleet_buckets(self) -> Tensor:
        if self.fleet_buckets is None:
            raise RuntimeError("PlanetMovement fleet tracking is not enabled")
        return self.fleet_buckets

    def _k_index(self, k: int) -> int:
        if k < 0 or k > int(self.movement_horizon):
            raise IndexError(f"k must be in [0, {self.movement_horizon}], got {k}")
        return int(k)

    def _copy_from(self, other: "PlanetMovement") -> None:
        self.x = other.x
        self.y = other.y
        self.alive_by_step = other.alive_by_step
        self.planet_ids = other.planet_ids
        self.radii = other.radii
        self.planet_owner = other.planet_owner
        self.planet_ships = other.planet_ships
        self.planet_prod = other.planet_prod
        self.base_step = other.base_step
        self.comet_planet_ids = other.comet_planet_ids
        self.comet_path_index = other.comet_path_index
        self.movement_horizon = other.movement_horizon
        self.drift_epsilon = other.drift_epsilon
        self.track_fleets = other.track_fleets
        self.player_count = other.player_count
        self.max_tracked_fleets = other.max_tracked_fleets
        self.fleet_buckets = other.fleet_buckets
        self.fleet_last_step = other.fleet_last_step
        self.tracked_fleet_ids = other.tracked_fleet_ids
        self.tracked_fleet_eta = other.tracked_fleet_eta
        self.tracked_fleet_target_slot = other.tracked_fleet_target_slot
        self.tracked_fleet_owner = other.tracked_fleet_owner
        self.tracked_fleet_ships = other.tracked_fleet_ships
        self.garrison_owner_cache = other.garrison_owner_cache
        self.garrison_ships_cache = other.garrison_ships_cache
        self.garrison_dirty_from = other.garrison_dirty_from




def _resolve_player_count(obs_tensors: dict, player_count: int | None) -> int:
    if player_count is not None:
        if int(player_count) not in (2, 4):
            raise ValueError("player_count must be 2 or 4")
        return int(player_count)
    metadata_count = obs_tensors.get("player_count")
    if metadata_count is not None:
        count = int(metadata_count.flatten()[0].item()) if isinstance(metadata_count, Tensor) else int(metadata_count)
        if count not in (2, 4):
            raise ValueError("player_count metadata must be 2 or 4")
        return count
    planets = obs_tensors["planets"]
    fleets = obs_tensors["fleets"]
    planet_alive = planets[..., 0] >= 0
    fleet_alive = fleets[..., 0] >= 0
    owner_values = []
    if bool(planet_alive.any()):
        owner_values.append(planets[..., 1][planet_alive].to(dtype=torch.long))
    if bool(fleet_alive.any()):
        owner_values.append(fleets[..., 1][fleet_alive].to(dtype=torch.long))
    if not owner_values:
        return 2
    owners = torch.cat(owner_values)
    owners = owners[owners >= 0]
    if owners.numel() == 0:
        return 2
    return 4 if int(owners.max().item()) >= 2 else 2


def _estimate_new_fleet_arrivals(
    *,
    movement: PlanetMovement,
    obs_fleets: Tensor,
    fleet_slot: Tensor,
) -> dict[str, Tensor]:
    N = int(fleet_slot.numel())
    device = movement.device
    dtype = movement.dtype
    H = int(movement.movement_horizon)
    P = int(movement.P)
    if N == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        empty_float = torch.empty(0, dtype=dtype, device=device)
        return {
            "owner": empty_long,
            "target_slot": empty_long,
            "eta_index": empty_long,
            "has_hit": empty_bool,
            "ships": empty_float,
        }

    rows = obs_fleets[fleet_slot]
    owner = rows[:, 1].to(dtype=torch.long)
    x = rows[:, 2].to(dtype=dtype)
    y = rows[:, 3].to(dtype=dtype)
    angle = rows[:, 4].to(dtype=dtype)
    ships = rows[:, 6].to(dtype=dtype)

    times = torch.arange(1, H + 1, dtype=dtype, device=device).view(1, H)
    speed = fleet_speed(ships).clamp(min=1e-6)
    ux = torch.cos(angle)
    uy = torch.sin(angle)
    old_x = x.view(N, 1) + ux.view(N, 1) * speed.view(N, 1) * (times - 1.0)
    old_y = y.view(N, 1) + uy.view(N, 1) * speed.view(N, 1) * (times - 1.0)
    new_x = x.view(N, 1) + ux.view(N, 1) * speed.view(N, 1) * times
    new_y = y.view(N, 1) + uy.view(N, 1) * speed.view(N, 1) * times

    in_bounds = (new_x >= 0.0) & (new_x <= BOARD_SIZE) & (new_y >= 0.0) & (new_y <= BOARD_SIZE)
    sun_dist_sq = _point_to_segment_distance_sq(
        torch.full_like(new_x, CENTER),
        torch.full_like(new_y, CENTER),
        old_x,
        old_y,
        new_x,
        new_y,
    )
    env_kill = (~in_bounds) | (sun_dist_sq < (SUN_RADIUS * SUN_RADIUS))

    planet_x = movement.x.unsqueeze(0).expand(N, H + 1, P)
    planet_y = movement.y.unsqueeze(0).expand(N, H + 1, P)
    planet_alive = movement.alive_by_step.unsqueeze(0).expand(N, H + 1, P)
    radii = movement.radii.unsqueeze(0).expand(N, P).to(dtype=dtype)

    old_px = planet_x[:, :-1, :]
    old_py = planet_y[:, :-1, :]
    new_px = planet_x[:, 1:, :]
    new_py = planet_y[:, 1:, :]
    alive_old = planet_alive[:, :-1, :]
    check_collision = alive_old & (old_px >= 0.0) & (old_py >= 0.0)
    swept_collides = _swept_pair_hit_mask(
        old_x.unsqueeze(2),
        old_y.unsqueeze(2),
        new_x.unsqueeze(2),
        new_y.unsqueeze(2),
        old_px,
        old_py,
        new_px,
        new_py,
        radii.view(N, 1, P),
    ) & check_collision
    step_raw_has_hit = swept_collides.any(dim=2)
    hit_rank = swept_collides.to(torch.int32).cumsum(dim=2)
    first_hit = swept_collides & (hit_rank == 1)
    step_hit_slot = first_hit.to(torch.int64).argmax(dim=2)
    step_hit_slot = step_hit_slot.where(step_raw_has_hit, torch.full_like(step_hit_slot, -1))

    # Per-step ordering mirrors engine semantics: planet collision first,
    # out-of-bounds/sun checks only if no planet collision happened this step.
    # Vectorized active-mask propagation: a fleet is alive at the start of
    # turn t iff no kill event (planet hit OR env kill) has fired at any
    # turn τ < t. ``cummax`` along the time axis gives the inclusive OR;
    # shifting right by one (prepending alive=True) yields the exclusive form.
    kill_event = step_raw_has_hit | env_kill
    cum_kill_inclusive = kill_event.cummax(dim=1).values
    alive_before_t = torch.cat(
        [
            torch.ones((N, 1), dtype=torch.bool, device=device),
            ~cum_kill_inclusive[:, :-1],
        ],
        dim=1,
    )
    step_has_hit = step_raw_has_hit & alive_before_t

    has_hit = step_has_hit.any(dim=1)
    eta_index = step_has_hit.to(torch.int64).argmax(dim=1)
    target_slot = step_hit_slot.gather(1, eta_index.view(N, 1)).squeeze(1).clamp(min=0, max=max(P - 1, 0))

    return {
        "owner": owner,
        "target_slot": target_slot,
        "eta_index": eta_index,
        "has_hit": has_hit,
        "ships": ships,
    }


def _point_to_segment_distance_sq(px: Tensor, py: Tensor, x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor) -> Tensor:
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    t = ((px - x1) * dx + (py - y1) * dy) / safe_denom
    t = t.clamp(0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


def _swept_pair_hit_mask(
    ax: Tensor,
    ay: Tensor,
    bx: Tensor,
    by: Tensor,
    p0x: Tensor,
    p0y: Tensor,
    p1x: Tensor,
    p1y: Tensor,
    r: Tensor,
) -> Tensor:
    """Broadcasted swept-pair overlap check for moving fleet/planet pairs."""
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    near_static = a < 1e-12
    c_hit = c <= 0.0
    disc = b * b - 4.0 * a * c
    has_root = disc >= 0.0
    safe_a = torch.where(near_static, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    quad_hit = has_root & (t2 >= 0.0) & (t1 <= 1.0)
    return torch.where(near_static, c_hit, quad_hit)


def _build_future_from_obs(
    obs_tensors: dict,
    movement_horizon: int,
    *,
    offsets: Tensor | None = None,
) -> dict[str, Tensor]:
    """Build planet/comet positions at the requested integer step offsets.

    By default builds the full trajectory ``offsets = arange(H+1)`` (output
    ``x/y/alive_by_step`` shape ``[H+1, P]``). Callers that only need a
    subset of frames (e.g. just the new last frame ``H`` on the roll-only
    update path) can pass ``offsets`` as a 1D long tensor; the output's
    first axis matches its length.
    """
    obs = parse_obs(obs_tensors)
    H = int(movement_horizon)
    planets = obs_tensors["planets"]
    dtype = planets.dtype
    device = planets.device
    P, _ = planets.shape

    planet_ids = planets[..., 0].long()
    radii = planets[..., 4].to(dtype=dtype)
    owner = planets[..., 1].to(device=device, dtype=torch.long)
    owner = torch.where(obs.alive, owner, torch.full_like(owner, -1))
    ships = planets[..., 5].to(device=device, dtype=dtype)
    prod = planets[..., 6].to(device=device, dtype=dtype)
    step = obs.step.to(device=device, dtype=torch.long)

    if offsets is None:
        offsets_long = torch.arange(H + 1, dtype=torch.long, device=device)
    else:
        offsets_long = offsets.to(device=device, dtype=torch.long).reshape(-1)
    M = int(offsets_long.shape[0])
    offsets_d = offsets_long.to(dtype=dtype)
    future_phase = orbit_phase_index_from_obs_step(
        obs.step.to(dtype=dtype) + offsets_d
    ).to(device=device, dtype=dtype)                                          # [M]

    angle = (
        obs.orb_a0.to(dtype=dtype).view(1, P)
        + obs.angvel.to(dtype=dtype) * future_phase.view(M, 1)
    )                                                                         # [M, P]
    orb_x = CENTER + obs.orb_r.to(dtype=dtype).view(1, P) * torch.cos(angle)
    orb_y = CENTER + obs.orb_r.to(dtype=dtype).view(1, P) * torch.sin(angle)
    is_orbiting = obs.is_orbiting.view(1, P)
    x = torch.where(
        is_orbiting,
        orb_x,
        obs.x.to(dtype=dtype).view(1, P).expand(M, P),
    ).contiguous()
    y = torch.where(
        is_orbiting,
        orb_y,
        obs.y.to(dtype=dtype).view(1, P).expand(M, P),
    ).contiguous()
    alive_by_step = obs.alive.view(1, P).expand(M, P).clone()

    comet_planet_ids, comet_path_index = _comet_metadata(obs_tensors, device)
    x, y, alive_by_step = _apply_comet_paths(
        x=x,
        y=y,
        alive_by_step=alive_by_step,
        planet_ids=planet_ids,
        comet_planet_ids=comet_planet_ids,
        comet_path_index=comet_path_index,
        obs_tensors=obs_tensors,
        offsets=offsets_long,
    )
    # Override slots where offset == 0 with the obs frame (truth at "now").
    zero_idx = (offsets_long == 0).nonzero(as_tuple=True)[0]
    if int(zero_idx.numel()) > 0:
        x[zero_idx, :] = obs.x.to(dtype=dtype).view(1, P)
        y[zero_idx, :] = obs.y.to(dtype=dtype).view(1, P)
        alive_by_step[zero_idx, :] = obs.alive.view(1, P)

    return {
        "x": x,
        "y": y,
        "alive_by_step": alive_by_step,
        "planet_ids": planet_ids,
        "radii": radii,
        "owner": owner,
        "ships": ships,
        "prod": prod,
        "step": step,
        "comet_planet_ids": comet_planet_ids,
        "comet_path_index": comet_path_index,
        "_offsets": offsets_long,
    }


def _comet_metadata(obs_tensors: dict, device: torch.device) -> tuple[Tensor, Tensor]:
    comets = obs_tensors.get("comets") or {}
    comet_ids = comets.get("planet_ids")
    if comet_ids is None:
        flat_ids = obs_tensors.get("comet_planet_ids")
        if flat_ids is None:
            flat_ids = torch.full((0,), -1, dtype=torch.long, device=device)
        else:
            flat_ids = flat_ids.to(device=device, dtype=torch.long)
        path_index = torch.full((0,), -1, dtype=torch.long, device=device)
        return flat_ids, path_index
    comet_ids = comet_ids.to(device=device, dtype=torch.long)
    flat_ids = comet_ids.reshape(-1)
    path_index = comets.get("path_index")
    if path_index is None:
        path_index = torch.full((comet_ids.shape[0],), -1, dtype=torch.long, device=device)
    else:
        path_index = path_index.to(device=device, dtype=torch.long)
    return flat_ids, path_index


def _apply_comet_paths(
    *,
    x: Tensor,
    y: Tensor,
    alive_by_step: Tensor,
    planet_ids: Tensor,
    comet_planet_ids: Tensor,
    comet_path_index: Tensor,
    obs_tensors: dict,
    offsets: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply comet path overrides at the requested integer step ``offsets``.

    ``x``/``y``/``alive_by_step`` are shaped ``[M, P]`` where ``M ==
    offsets.shape[0]``. The offsets tensor is 1D long.
    """
    comets = obs_tensors.get("comets") or {}
    paths = comets.get("paths")
    ids_grid = comets.get("planet_ids")
    if paths is None or ids_grid is None or comet_planet_ids.numel() == 0:
        return x, y, alive_by_step

    M, P = x.shape
    paths = paths.to(device=x.device, dtype=x.dtype)            # [E, C, T, 2]
    ids_grid = ids_grid.to(device=x.device, dtype=torch.long)   # [E, C]
    E = int(ids_grid.shape[0])
    C = int(ids_grid.shape[1])
    T = int(paths.shape[2])
    if E == 0 or C == 0 or T == 0:
        return x, y, alive_by_step

    flat_ids = ids_grid.reshape(E * C)                          # [E*C]
    matches = (planet_ids.unsqueeze(1) == flat_ids.unsqueeze(0)) & (flat_ids.unsqueeze(0) >= 0)  # [P, E*C]
    is_comet = matches.any(dim=1)                               # [P]

    flat_slot = matches.to(torch.float32).argmax(dim=1).long()  # [P]
    flat_paths_x = paths[..., 0].reshape(E * C, T)              # [E*C, T]
    flat_paths_y = paths[..., 1].reshape(E * C, T)
    path_x_by_slot = flat_paths_x[flat_slot]                    # [P, T]
    path_y_by_slot = flat_paths_y[flat_slot]

    finite = torch.isfinite(flat_paths_x)                       # [E*C, T]
    path_len = finite.sum(dim=1).to(dtype=torch.long)           # [E*C]
    len_by_slot = path_len[flat_slot]                           # [P]
    group_idx = (flat_slot // C).clamp(min=0, max=max(E - 1, 0))  # [P]
    path_idx_by_slot = comet_path_index[group_idx]             # [P]

    offsets_v = offsets.to(device=x.device, dtype=torch.long).view(M, 1)   # [M, 1]
    future_idx = path_idx_by_slot.view(1, P) + offsets_v        # [M, P]
    valid_future = (
        is_comet.view(1, P)
        & (future_idx >= 0)
        & (future_idx < len_by_slot.view(1, P))
    )                                                          # [M, P]
    idx_clamped = future_idx.clamp(min=0, max=max(T - 1, 0))    # [M, P]
    p_index = torch.arange(P, device=x.device).view(1, P).expand(M, P)
    comet_x = path_x_by_slot[p_index, idx_clamped]             # [M, P]
    comet_y = path_y_by_slot[p_index, idx_clamped]

    x = torch.where(valid_future, comet_x, x)
    y = torch.where(valid_future, comet_y, y)
    alive_by_step = torch.where(is_comet.view(1, P), valid_future, alive_by_step)
    return x, y, alive_by_step


def _same_2d(a: Tensor, b: Tensor) -> bool:
    if a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    return bool((a == b.to(device=a.device, dtype=a.dtype)).all())


def _position_matches(
    pred_x: Tensor,
    pred_y: Tensor,
    cur_x: Tensor,
    cur_y: Tensor,
    alive: Tensor,
    epsilon: float,
) -> bool:
    diff = torch.maximum((pred_x - cur_x).abs(), (pred_y - cur_y).abs())
    diff = torch.where(alive, diff, torch.zeros_like(diff))
    return bool((diff <= float(epsilon)).all())


# === orbit_lite.distance_cache ===
"""Cross-k distance cache for the movement-backed planner.

Entry ``cross_dist[k, s, t]`` is the Euclidean distance from planet ``s`` at step
0 to planet ``t`` at step ``k`` — the *cross-time* distance a fleet must travel if
it launches now from ``s`` to intercept ``t`` at time ``k``. For static planets
this equals same-step pairwise distance; for orbiting sources the cross-time form
is the geometrically correct quantity for fleet-intercept feasibility. A
precomputed ``[K+1, P, P]`` window gives exact per-step lookups for free.
"""


from dataclasses import dataclass

import torch
from torch import Tensor




@dataclass
class DistanceCache:
    """Per-turn cross-k distance window.

    Tensor shapes:
    - ``cross_dist``: ``[K+1, P, P]`` -- ``[k, s, t] = dist(s@0, t@k)``.
    - ``alive_by_step``: ``[K+1, P]`` -- view sliced from
      ``movement.alive_by_step``.
    """

    cross_dist: Tensor
    alive_by_step: Tensor
    K: int

    @property
    def P(self) -> int:
        return int(self.cross_dist.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.cross_dist.device

    @property
    def dtype(self) -> torch.dtype:
        return self.cross_dist.dtype



def build_distance_cache(
    movement: PlanetMovement,
    *,
    max_k: int,
) -> DistanceCache:
    """Build a fresh cross-k distance cache from the rolling movement cache.

    ``max_k`` is clamped to ``movement.movement_horizon``. Caller is
    expected to clamp its own k queries the same way.
    """
    K = max(0, min(int(max_k), int(movement.movement_horizon)))
    P = int(movement.P)
    src_x0 = movement.x[0]                         # [P]
    src_y0 = movement.y[0]
    tgt_x = movement.x[: K + 1]                    # [K+1, P]
    tgt_y = movement.y[: K + 1]
    # cross[k, s, t] = dist(s@0, t@k)
    dx = src_x0.view(1, P, 1) - tgt_x.unsqueeze(1)
    dy = src_y0.view(1, P, 1) - tgt_y.unsqueeze(1)
    cross_dist = torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))
    alive_by_step = movement.alive_by_step[: K + 1]
    return DistanceCache(
        cross_dist=cross_dist,
        alive_by_step=alive_by_step,
        K=K,
    )


# ---------------------------------------------------------------------------
# Min-distance helper (replaces movement_min_distance_to_targets)
# ---------------------------------------------------------------------------


def min_distance_to_targets(
    cache: DistanceCache,
    source_mask: Tensor,
    target_mask: Tensor,
    *,
    max_k: int,
) -> Tensor:
    """Return per-target nearest-source distance using cross-k lookups.

    For each target ``t``, return the smallest
    ``dist(s@0, t@k)`` over alive valid sources ``s`` and steps
    ``k in [1, min(max_k, cache.K)]``. This is the exact analogue of
    ``movement_min_distance_to_targets`` with sampled steps replaced by the
    full integer range.
    """
    if source_mask.shape[-1] != cache.P or target_mask.shape[-1] != cache.P:
        raise ValueError("source_mask and target_mask must have shape [P]")
    K = max(0, min(int(max_k), int(cache.K)))
    if K <= 0:
        return torch.zeros(cache.P, dtype=cache.dtype, device=cache.device)
    # Clone the cross-k slice so we can ``masked_fill_`` invalid entries to +inf
    # without touching the cache's storage. The union of the three masks is
    # equivalent to ``~valid_pair = ~src_mask | ~tgt_mask | ~alive_at_k``.
    cross = cache.cross_dist[1 : K + 1].clone()    # [K, P_src, P_tgt]
    alive_steps = cache.alive_by_step[1 : K + 1]   # [K, P]
    src_mask = source_mask.to(device=cache.device, dtype=torch.bool)
    tgt_mask = target_mask.to(device=cache.device, dtype=torch.bool)
    inf_v = float("inf")
    cross.masked_fill_(~alive_steps.unsqueeze(1), inf_v)
    cross.masked_fill_(~src_mask.view(1, cache.P, 1), inf_v)
    cross.masked_fill_(~tgt_mask.view(1, 1, cache.P), inf_v)
    best_per_target = cross.amin(dim=(0, 1))       # over K and source axis
    return torch.where(torch.isfinite(best_per_target), best_per_target, torch.zeros_like(best_per_target))


# ---------------------------------------------------------------------------
# Compact candidate pairs (replaces compact_candidate_pairs for regroup)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Aiming reachability mask (precheck augmentation for movement_pairwise_grid)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------










# === orbit_lite.garrison_launch ===
"""What-if-I-launch flow projection over a :class:`PlanetGarrisonStatus`.

``PlanetGarrisonStatus`` is a per-planet ledger of projected owner / ships over a
future horizon, computed from the fleets we currently know about, assuming we do
nothing. :func:`sparse_launch_flow_delta` answers the forward-looking question an
agent faces — *"if I launch these ships, how does each player's net ship flow
(production minus combat losses) change?"* — by recomputing the production→combat
recurrence only for the planets a launch touches and diffing against the baseline.

A launch is two-sided: it debits the source planet's garrison (ships leave now,
before that turn's production) and credits the target's arrival at step ``k``.

Two leading axes are supported (single game):

- ``C`` — candidates: the different launches / launch-sets being scored.
- ``L`` — launches within a candidate: a candidate *is* a set of launches; ``L``
  is summed away during aggregation and is not an output axis.

Pass launches as ``[L]`` (no candidate axis) or ``[C, L]``.
"""


from dataclasses import dataclass

import torch
from torch import Tensor




@dataclass(frozen=True)
class LaunchSet:
    """A batched set of hypothetical launches issued on the current turn.

    All tensors share a leading prefix (empty or ``[C]``) followed by a
    trailing launch axis ``L`` (use ``L=1`` for a single launch). ``eta`` is in
    steps from the current frame (arrival lands at garrison step ``k = eta``;
    ``eta`` must be ``>= 1``). ``owner`` defaults to the acting player but is
    per-launch so opponent what-ifs are expressible.
    """

    source_slots: Tensor  # [*prefix, L] long  (planet slot to launch FROM)
    target_slots: Tensor  # [*prefix, L] long  (planet slot to launch TO)
    ships: Tensor         # [*prefix, L] float
    eta: Tensor           # [*prefix, L] float/long (steps to arrival, >= 1)
    owner: Tensor         # [*prefix, L] long
    valid: Tensor         # [*prefix, L] bool


    @property
    def has_candidate_axis(self) -> bool:
        return self.source_slots.dim() >= 2





def _per_step_survivor(arrivals: Tensor) -> tuple[Tensor, Tensor]:
    """Engine survivor over the owner axis for every step.

    ``arrivals`` is ``[..., A]``; returns ``(survivor_owner, survivor_ships)``
    over the trailing axis, applying the engine rule: survivor ships = top1 -
    top2, ties annihilate (ships 0). Owner is meaningful only where ships > 0.
    """
    A = int(arrivals.shape[-1])
    if A >= 2:
        top2 = arrivals.topk(k=2, dim=-1)
        top_ships = top2.values[..., 0]
        second_ships = top2.values[..., 1]
        top_owner = top2.indices[..., 0].to(dtype=torch.long)
    else:
        top_ships, top_owner = arrivals.max(dim=-1)
        second_ships = torch.zeros_like(top_ships)
        top_owner = top_owner.to(dtype=torch.long)
    tied = top_ships == second_ships
    survivor_ships = torch.where(
        tied, torch.zeros_like(top_ships), (top_ships - second_ships).clamp(min=0.0)
    )
    return top_owner, survivor_ships


def _run_exact_recurrence(
    *,
    init_owner: Tensor,   # [N, P] long
    init_ships: Tensor,   # [N, P] float (already source-debited)
    prod: Tensor,         # [N, P] float
    alive: Tensor,        # [N, P, H+1] bool
    arrivals: Tensor,     # [N, P, H, A] float (steps 1..H, baseline + delta)
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Walk the engine production→combat recurrence over ``k = 1..H``.

    Mirrors ``PlanetMovement._fill_garrison_trajectory`` Half B exactly, but for
    all planets without the simple/complex fast-path split (clarity over the few
    saved kernels — this is the reference path). Returns
    ``(owner, ships, pre_owner, pre_ships)`` each ``[N, P, H+1]`` with step 0 set
    to the recurrence's starting state.
    """
    N, P = init_owner.shape
    H = int(arrivals.shape[2])
    device = init_ships.device

    owner_out = torch.empty(N, P, H + 1, dtype=init_owner.dtype, device=device)
    ships_out = torch.empty(N, P, H + 1, dtype=init_ships.dtype, device=device)
    pre_owner_out = torch.empty_like(owner_out)
    pre_ships_out = torch.empty_like(ships_out)
    owner_out[..., 0] = init_owner
    ships_out[..., 0] = init_ships
    pre_owner_out[..., 0] = init_owner
    pre_ships_out[..., 0] = init_ships

    survivor_owner, survivor_ships = _per_step_survivor(arrivals)  # [N, P, H]

    state_owner = init_owner.clone()
    state_ships = init_ships.clone()
    zero_ships = torch.zeros((), dtype=state_ships.dtype, device=device)
    neg_one = torch.full((), -1, dtype=state_owner.dtype, device=device)
    zero_prod = torch.zeros((), dtype=prod.dtype, device=device)

    for k in range(1, H + 1):
        a_before = alive[..., k - 1]
        a_now = alive[..., k]
        s_owner = survivor_owner[..., k - 1]
        s_ships = survivor_ships[..., k - 1]

        # Production: owned planets alive at the start of this step.
        produces = a_before & (state_owner >= 0)
        state_ships = state_ships + torch.where(produces, prod, zero_prod)

        # Pre-combat snapshot (after production, before same-step combat).
        pre_owner_out[..., k] = torch.where(a_now, state_owner, neg_one)
        pre_ships_out[..., k] = torch.where(a_now, state_ships, zero_ships)

        # Survivor vs the prior garrison.
        has_combat = (s_ships > 0.0) & a_now
        same = state_owner == s_owner
        diff = state_ships - s_ships
        attacker_wins = (~same) & (diff < 0.0)
        combat_ships = torch.where(same, state_ships + s_ships, diff.abs())
        combat_owner = torch.where(attacker_wins, s_owner, state_owner)
        state_ships = torch.where(has_combat, combat_ships, state_ships)
        state_owner = torch.where(has_combat, combat_owner, state_owner)

        # End-of-step death reset.
        state_owner = torch.where(a_now, state_owner, neg_one)
        state_ships = torch.where(a_now, state_ships, zero_ships)

        owner_out[..., k] = state_owner
        ships_out[..., k] = state_ships

    return owner_out, ships_out, pre_owner_out, pre_ships_out


def _validate_inputs(
    status: PlanetGarrisonStatus,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
) -> tuple[int, int, int, int]:
    """Check shapes and return ``(B, P, H, A)``."""
    if status.arrivals_by_owner is None:
        raise ValueError(
            "garrison status must carry arrivals_by_owner (build it from a "
            "PlanetMovement with track_fleets=True)"
        )
    if status.pre_combat_owner is None or status.pre_combat_ships is None:
        raise ValueError("garrison status must carry pre_combat_owner/ships")
    if status.owner.dim() != 2:
        raise ValueError(
            "expected a full-board status with owner shaped [P, H+1]; got "
            f"{tuple(status.owner.shape)}"
        )
    P, H1 = status.owner.shape
    H = H1 - 1
    A = int(status.arrivals_by_owner.shape[-1])
    if int(player_count) != A:
        raise ValueError(
            f"player_count={player_count} disagrees with arrivals owner axis A={A}"
        )
    if tuple(prod.shape) != (P,):
        raise ValueError(f"prod must be [P]=({P},); got {tuple(prod.shape)}")
    if tuple(alive_by_step.shape) != (H1, P):
        raise ValueError(
            f"alive_by_step must be [H+1, P]=({H1}, {P}); got "
            f"{tuple(alive_by_step.shape)}"
        )
    return P, H, A












# ---------------------------------------------------------------------------
# Per-player flow accounting: diff two garrison statuses
# ---------------------------------------------------------------------------




@dataclass(frozen=True)
class GarrisonFlowDiff:
    """Difference in per-player flow between a current and a hypothetical status.

    Each field is ``[*prefix, A]`` (per player). ``*_delta`` is
    ``hypothetical - current``. ``net_ship_delta`` is the change in net ships
    gained (``produced - lost_to_combat``) — i.e. how much better/worse off each
    player ends up under the hypothetical, ignoring ships in transit.
    """

    player_id: int
    ships_produced_current: Tensor
    ships_produced_hypothetical: Tensor
    ships_produced_delta: Tensor
    ships_lost_combat_current: Tensor
    ships_lost_combat_hypothetical: Tensor
    ships_lost_combat_delta: Tensor
    net_ship_delta: Tensor
    # Production owned at the horizon's final step, hypothetical − current,
    # per player (``[*prefix, A]``). The flow terms above truncate a captured
    # planet's payoff at the horizon; this field lets a scorer credit the
    # post-horizon production stream (None where a path doesn't compute it).
    terminal_prod_delta: Tensor | None = None

    @property
    def player_count(self) -> int:
        return int(self.ships_produced_delta.shape[-1])





def _flow_terms_per_planet(
    *,
    owner: Tensor,        # [.., P, H+1]
    pre_owner: Tensor,    # [.., P, H+1]
    pre_ships: Tensor,    # [.., P, H+1]
    arr_full: Tensor,     # [.., P, H+1, A]
    prod: Tensor,         # [.., P] (broadcastable)
    alive_pmajor: Tensor, # [.., P, H+1] (broadcastable, planet-major)
) -> tuple[Tensor, Tensor]:
    """Per-planet production and combat losses, summed over the horizon only.

    Returns ``(produced, combat_lost)`` each ``[.., P, A]``. Combat follows
    the engine combat rule; production credits ``prod`` to the owner
    holding the planet entering each step (from ``prod``, not ship deltas, so a
    source launch debit is not mistaken for negative production).
    """
    A = int(arr_full.shape[-1])
    H = int(owner.shape[-1]) - 1
    fdtype = pre_ships.dtype
    a_idx = torch.arange(A, device=owner.device)

    # Production credited to owner at the start of each step (= owner[k-1]).
    producing_owner = owner[..., :H]                                 # [.., P, H]
    amount = prod.unsqueeze(-1) * alive_pmajor[..., :H].to(fdtype)   # [.., P, H]
    prod_owner_oh = producing_owner.unsqueeze(-1) == a_idx           # [.., P, H, A]
    produced = (amount.unsqueeze(-1) * prod_owner_oh.to(fdtype)).sum(dim=-2)  # [.., P, A]

    # Combat per step (engine top1 - top2 survivor, then survivor vs garrison).
    arr_k = arr_full[..., 1:, :]
    survivor_owner, survivor_ships = _per_step_survivor(arr_k)       # [.., P, H]
    survived = torch.where(
        a_idx == survivor_owner.unsqueeze(-1),
        survivor_ships.unsqueeze(-1),
        torch.zeros_like(survivor_ships).unsqueeze(-1),
    )
    attacker_lost = (arr_k - survived).clamp(min=0.0)                # [.., P, H, A]
    prior_owner = pre_owner[..., 1:]
    prior_ships = pre_ships[..., 1:]
    fights_garrison = (survivor_ships > 0.0) & (survivor_owner != prior_owner) & (survivor_owner >= 0)
    garrison_loss = torch.where(
        fights_garrison, torch.minimum(prior_ships, survivor_ships), torch.zeros_like(prior_ships)
    )
    is_survivor = (a_idx == survivor_owner.unsqueeze(-1)) & fights_garrison.unsqueeze(-1)
    is_prior = (
        (a_idx == prior_owner.unsqueeze(-1))
        & fights_garrison.unsqueeze(-1)
        & (prior_owner >= 0).unsqueeze(-1)
    )
    garrison_lost = garrison_loss.unsqueeze(-1) * (is_survivor.to(fdtype) + is_prior.to(fdtype))
    combat_lost = (attacker_lost + garrison_lost).sum(dim=-2)        # [.., P, A]
    return produced, combat_lost










# ---------------------------------------------------------------------------
# Sparse prototype: per-candidate flow deltas without dense [C, P, H, A]
# ---------------------------------------------------------------------------


def _normalize_launches_bcl(launches: LaunchSet) -> tuple[Tensor, ...]:
    """Return ``(src, tgt, ships, eta, owner, valid)`` shaped ``[C, L]``."""
    fields = (
        launches.source_slots, launches.target_slots, launches.ships,
        launches.eta, launches.owner, launches.valid,
    )
    if launches.has_candidate_axis:
        return fields
    return tuple(f.unsqueeze(0) for f in fields)  # [L] -> [1, L]


def sparse_launch_flow_delta(
    status: PlanetGarrisonStatus,
    *,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    launches: LaunchSet,
    player_id: int = 0,
    terminal_neutral_only: bool = False,
) -> GarrisonFlowDiff:
    """Sparse equivalent of ``diff_garrison_flow(status, apply_launches_exact(...))``.

    Returns the **same** exact per-candidate, per-player flow diff as the dense
    pipeline, but never materializes the dense ``[C, P, H, A]`` arrivals or a
    ``[C, P, H+1]`` trajectory. It exploits two facts:

    - the garrison projection is per-planet independent given the arrival
      buckets, so a launch only changes the trajectory of the planets it touches
      (its source, via the debit, and its target, via the credit);
    - untouched planets contribute zero to the flow *delta*.

    So it recomputes the recurrence only for the affected ``(candidate, planet)``
    cells (~2 per candidate for single launches, vs all ``P``) and scatter-adds
    their per-planet flow deltas into ``[C, A]``. Cost and memory scale with
    the number of affected cells, not ``B·C·P``.
    """
    P, H, A = _validate_inputs(status, prod, alive_by_step, player_count)
    device = status.owner.device
    fdtype = status.ships.dtype
    assert status.pre_combat_owner is not None and status.pre_combat_ships is not None
    assert status.arrivals_by_owner is not None

    src, tgt, ships, eta, owner, valid = _normalize_launches_bcl(launches)
    C = int(src.shape[0])
    L = int(src.shape[-1])
    src = src.to(device=device, dtype=torch.long)
    tgt = tgt.to(device=device, dtype=torch.long)
    ships = ships.to(device=device, dtype=fdtype)
    owner = owner.to(device=device, dtype=torch.long)
    valid = valid.to(device=device, dtype=torch.bool)
    h_idx = torch.ceil(eta.to(device=device, dtype=fdtype)).to(torch.long) - 1

    valid_t = valid & (ships > 0) & (tgt >= 0) & (tgt < P) & (owner >= 0) & (owner < A) & (h_idx >= 0) & (h_idx < H)
    valid_s = valid & (ships > 0) & (src >= 0) & (src < P)
    src_safe = src.clamp(0, max(P - 1, 0))
    tgt_safe = tgt.clamp(0, max(P - 1, 0))

    # Affected planets per candidate (source debit OR target credit). [C, P]
    affected = torch.zeros(C, P, dtype=fdtype, device=device)
    affected.scatter_add_(1, src_safe, valid_s.to(fdtype))
    affected.scatter_add_(1, tgt_safe, valid_t.to(fdtype))
    affected_mask = affected > 0

    # Baseline per-planet flow (shared across candidates).
    base_prod_pp, base_combat_pp = _flow_terms_per_planet(
        owner=status.owner,
        pre_owner=status.pre_combat_owner,
        pre_ships=status.pre_combat_ships,
        arr_full=status.arrivals_by_owner,
        prod=prod,
        alive_pmajor=alive_by_step.permute(1, 0),
    )                                                        # [P, A]
    base_prod = base_prod_pp.sum(dim=0)                      # [A]
    base_combat = base_combat_pp.sum(dim=0)

    produced_delta = torch.zeros(C, A, dtype=fdtype, device=device)
    combat_delta = torch.zeros(C, A, dtype=fdtype, device=device)
    terminal_prod_delta = torch.zeros(C, A, dtype=fdtype, device=device)

    if bool(affected_mask.any()):
        c_aff, p_aff = affected_mask.nonzero(as_tuple=True)         # [N]
        N = int(c_aff.numel())
        cell_id = torch.full((C, P), -1, dtype=torch.long, device=device)
        cell_id[c_aff, p_aff] = torch.arange(N, device=device)

        # Source debit per affected cell.
        debit_cp = torch.zeros(C, P, dtype=fdtype, device=device)
        debit_cp.scatter_add_(1, src_safe, torch.where(valid_s, ships, torch.zeros_like(ships)))
        debit_aff = debit_cp[c_aff, p_aff]                          # [N]

        # Target credits scattered onto the affected cells: [N, H, A].
        arr_aff = torch.zeros(N, H, A, dtype=fdtype, device=device)
        launch_cell = cell_id.gather(1, tgt_safe)                   # [C, L]
        m = valid_t
        cells, hh, oo, ss = launch_cell[m], h_idx[m], owner[m], ships[m]
        ok = cells >= 0
        arr_aff.index_put_((cells[ok], hh[ok], oo[ok]), ss[ok], accumulate=True)

        base_arr_k = status.arrivals_by_owner[..., 1:, :]           # [P, H, A]
        arrivals_cell = base_arr_k[p_aff] + arr_aff                 # [N, H, A]

        init_owner = status.owner[p_aff, 0]                         # [N]
        init_ships = (status.ships[p_aff, 0] - debit_aff).clamp(min=0.0)
        prod_aff = prod[p_aff]                                      # [N]
        alive_aff = alive_by_step[:, p_aff].transpose(0, 1)         # [N, H+1]

        # One-planet recurrence per affected cell (P=1 lane).
        o_t, _s_t, po_t, ps_t = _run_exact_recurrence(
            init_owner=init_owner.unsqueeze(1),
            init_ships=init_ships.unsqueeze(1),
            prod=prod_aff.unsqueeze(1),
            alive=alive_aff.unsqueeze(1),
            arrivals=arrivals_cell.unsqueeze(1),
        )
        zero_frame = torch.zeros(N, 1, 1, A, dtype=fdtype, device=device)
        arr_full_cell = torch.cat([zero_frame, arrivals_cell.unsqueeze(1)], dim=-2)
        hyp_prod_pp, hyp_combat_pp = _flow_terms_per_planet(
            owner=o_t, pre_owner=po_t, pre_ships=ps_t, arr_full=arr_full_cell,
            prod=prod_aff.unsqueeze(1), alive_pmajor=alive_aff.unsqueeze(1),
        )
        dprod = hyp_prod_pp.squeeze(1) - base_prod_pp[p_aff]            # [N, A]
        dcombat = hyp_combat_pp.squeeze(1) - base_combat_pp[p_aff]
        produced_delta.index_put_((c_aff,), dprod, accumulate=True)
        combat_delta.index_put_((c_aff,), dcombat, accumulate=True)

        # Owner of each affected planet at the horizon's final step,
        # hypothetical vs baseline, as per-player production deltas.
        a_oh = torch.arange(A, device=device)
        fin_hyp = o_t[:, 0, -1]                                         # [N]
        fin_base = status.owner[p_aff, -1]                              # [N]
        alive_fin = alive_aff[:, -1].to(fdtype)                         # [N]
        d_term = (prod_aff * alive_fin).unsqueeze(-1) * (
            (fin_hyp.unsqueeze(-1) == a_oh).to(fdtype)
            - (fin_base.unsqueeze(-1) == a_oh).to(fdtype)
        )                                                               # [N, A]
        if terminal_neutral_only:
            # Credit only planets that are NEUTRAL in the do-nothing world at
            # the horizon: encourages expansion without inflating enemy
            # strikes (an enemy capture counts double in the competitive
            # term, which over-drove aggression when credited).
            d_term = d_term * (fin_base < 0).to(fdtype).unsqueeze(-1)
        terminal_prod_delta.index_put_((c_aff,), d_term, accumulate=True)

    produced_current = base_prod.unsqueeze(0)                      # [1, A]
    combat_current = base_combat.unsqueeze(0)
    diff = GarrisonFlowDiff(
        player_id=int(player_id),
        ships_produced_current=produced_current,
        ships_produced_hypothetical=produced_current + produced_delta,
        ships_produced_delta=produced_delta,
        ships_lost_combat_current=combat_current,
        ships_lost_combat_hypothetical=combat_current + combat_delta,
        ships_lost_combat_delta=combat_delta,
        net_ship_delta=produced_delta - combat_delta,
        terminal_prod_delta=terminal_prod_delta,
    )
    # Squeeze the candidate axis back out for [L] launches (C == 1, no axis).
    if not launches.has_candidate_axis:
        def _sq(t: Tensor) -> Tensor:
            return t.squeeze(0)
        diff = GarrisonFlowDiff(
            player_id=diff.player_id,
            ships_produced_current=base_prod,
            ships_produced_hypothetical=_sq(diff.ships_produced_hypothetical),
            ships_produced_delta=_sq(diff.ships_produced_delta),
            ships_lost_combat_current=base_combat,
            ships_lost_combat_hypothetical=_sq(diff.ships_lost_combat_hypothetical),
            ships_lost_combat_delta=_sq(diff.ships_lost_combat_delta),
            net_ship_delta=_sq(diff.net_ship_delta),
            terminal_prod_delta=_sq(diff.terminal_prod_delta),
        )
    return diff


# === orbit_lite.intercept_aim ===
"""Fixed-fleet intercept aim — sub-turn-accurate angle for an orbiting target.

Solves the **continuous** intercept time ``t*`` (root of
``v·t = dist(target_pos(t), source) − gap`` with the target on its analytic
orbit), aims at ``target_pos(t*)``, and verifies that angle with a
fully-vectorized analytic first-contact check.

* **Root** — a continuous fixed-point iteration (no grid scan / argmax /
  bisection), free of grid-resolution artifacts.
* **Verify** — :func:`_analytic_first_contact` reproduces the engine's
  first-contact verdict exactly (swept-pair vs every planet, sun, bounds,
  lowest-slot same-step tie-break) with no engine state and no per-step loop.
  A shot is viable iff it contacts the target first.

Returns ``angle`` / ``eta`` / ``viable``.
"""

import torch
from torch import Tensor






_FP_ITERS = 6  # continuous fixed-point iterations for the intercept time
_BIG = 1_000_000.0


def intercept_angle(
    movement: PlanetMovement,
    source_slots: Tensor,
    target_slots: Tensor,
    fleet_sizes: Tensor,
    *,
    fp_iters: int = _FP_ITERS,
    active: Tensor | None = None,
) -> dict[str, Tensor]:
    """Continuous-intercept aim for a fixed fleet size (the root angle only).

    Broadcastable slot/size tensors in; ``{angle, eta, viable}`` out (same shape).
    Non-viable candidates get ``eta == inf``.

    ``active`` (optional, broadcastable to the candidate shape): a reachability
    precheck that gates the expensive body screen. The lead angle is still solved
    on the full grid, so kept candidates' angles are bit-identical; only the
    integer-exact first-contact screen is compacted to the active candidates.
    Candidates with ``active`` False resolve to non-viable. Pass a strict superset
    of viability (e.g. :func:`planner_core.reachable_mask`) for a zero-behaviour-change
    speedup — ``None`` screens everything.
    """
    dev = movement.device
    dt = movement.dtype
    H = int(movement.movement_horizon)

    src, tgt, ships = torch.broadcast_tensors(
        source_slots.to(device=dev),
        target_slots.to(device=dev),
        fleet_sizes.to(device=dev, dtype=dt),
    )
    shape = src.shape
    src = src.long().clamp(0, max(movement.P - 1, 0)).reshape(-1)
    tgt = tgt.long().clamp(0, max(movement.P - 1, 0)).reshape(-1)
    ships = ships.to(dt).clamp(min=1.0).reshape(-1)
    M = src.shape[0]

    sx, sy = movement.position_at_slots(src, 0)                       # [M]
    src_r = movement.radii[src]
    tgt_r = movement.radii[tgt]
    speed = fleet_speed(ships).clamp(min=1e-6)                        # [M]

    # Target orbit params from its integer positions: centre-relative radius +
    # phase at t=0 and the per-step angular step (auto-zero for static planets).
    t0x, t0y = movement.position_at_slots(tgt, 0)
    t1x, t1y = movement.position_at_slots(tgt, 1)
    R = torch.sqrt(((t0x - CENTER) ** 2 + (t0y - CENTER) ** 2).clamp(min=0.0))
    a0 = torch.atan2(t0y - CENTER, t0x - CENTER)
    a1 = torch.atan2(t1y - CENTER, t1x - CENTER)
    omega = torch.atan2(torch.sin(a1 - a0), torch.cos(a1 - a0))       # wrapped Δangle/step
    gap = src_r + LAUNCH_SURFACE_OFFSET + tgt_r + TARGET_HIT_SURFACE_OFFSET

    def target_pos(t: Tensor):
        ang = a0 + omega * t
        return CENTER + R * torch.cos(ang), CENTER + R * torch.sin(ang)

    # Continuous fixed point t = (dist(target_pos(t), src) - gap)/v, seeded with
    # the static-target estimate. A contraction whenever the target's radial speed
    # stays below the fleet speed (true for reachable shots); divergent guesses
    # just produce a bad angle that the verify rejects.
    d0 = torch.sqrt(((t0x - sx) ** 2 + (t0y - sy) ** 2).clamp(min=0.0))
    t_star = ((d0 - gap) / speed).clamp(min=0.0, max=float(H))
    for _ in range(int(fp_iters)):
        tx, ty = target_pos(t_star)
        d = torch.sqrt(((tx - sx) ** 2 + (ty - sy) ** 2).clamp(min=0.0))
        t_star = ((d - gap) / speed).clamp(min=0.0, max=float(H))

    tx, ty = target_pos(t_star)
    angle = torch.atan2(ty - sy, tx - sx)                             # [M]
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    launch_x = sx + cos_a * (src_r + LAUNCH_SURFACE_OFFSET)           # [M]
    launch_y = sy + sin_a * (src_r + LAUNCH_SURFACE_OFFSET)

    # Relevant flight length = distance to the intercept (+margins for the arrival
    # step and the target radius). Bounds the broad-phase cull segment to the
    # fleet's actual launch→target path. Planets beyond the target can never be the
    # first contact for a target-reaching fleet, so this preserves `viable`
    # (contact==tgt) and the viable-case `eta` exactly.
    eta_cap = (t_star + 2.0).clamp(max=float(H))
    seg_len = speed * eta_cap + tgt_r + 2.0                            # [M]

    px = movement.x[: H + 1, :]                                       # [H+1, P] (already cached)
    py = movement.y[: H + 1, :]
    radii_p = movement.radii
    alive0 = movement.alive_at(0)
    if active is None:
        contact, eta_c = _analytic_first_contact(
            launch_x=launch_x, launch_y=launch_y, cos_a=cos_a, sin_a=sin_a,
            speed=speed, px=px, py=py, p_alive0=alive0,
            radii=radii_p, H=H, seg_len=seg_len,
        )                                                             # [M]
    else:
        # Reachability gate: screen only the active candidates. The per-candidate
        # integer contact/eta are shortlist-independent, so kept candidates' verdicts
        # are bit-identical to the full screen. Compact to the active candidates,
        # screen, then scatter home; inactive cells resolve to contact = -1.
        act = active.broadcast_to(shape).reshape(M).to(torch.bool)
        n_max = max(1, int(act.sum().item()))
        order = (~act).to(torch.int8).argsort(stable=True)           # active cells first
        midx = order[:n_max]                                         # [n_max]
        keep = act[midx]
        contact_m, eta_cm = _analytic_first_contact(
            launch_x=launch_x[midx], launch_y=launch_y[midx],
            cos_a=cos_a[midx], sin_a=sin_a[midx],
            speed=speed[midx], px=px, py=py, p_alive0=alive0,
            radii=radii_p, H=H, seg_len=seg_len[midx],
        )                                                            # [n_max]
        contact = torch.full((M,), -1, dtype=contact_m.dtype, device=dev)
        eta_c = torch.full((M,), float(H), dtype=eta_cm.dtype, device=dev)
        contact[midx] = torch.where(keep, contact_m, torch.full_like(contact_m, -1))
        eta_c[midx] = torch.where(keep, eta_cm, torch.full_like(eta_cm, float(H)))

    viable = contact == tgt                                           # [M]
    eta_out = torch.where(viable, eta_c.to(dt), torch.full_like(eta_c.to(dt), float("inf")))
    return {
        "angle": angle.reshape(shape),
        "eta": eta_out.reshape(shape),
        "viable": viable.reshape(shape),
    }


def _analytic_first_contact(
    *,
    launch_x: Tensor,
    launch_y: Tensor,
    cos_a: Tensor,
    sin_a: Tensor,
    speed: Tensor,
    px: Tensor,
    py: Tensor,
    p_alive0: Tensor,
    radii: Tensor,
    H: int,
    seg_len: Tensor | None = None,
    max_bytes: int = 256 * 1024 * 1024,
):
    """First planet a fleet contacts, engine-faithful, shaped ``[M, C]``.

    Reproduces batch ``_move_fleets`` exactly: straight fleet motion at ``speed``,
    swept-pair collision vs every step-0-alive planet, OOB + point-to-segment sun
    kill (only when no planet was hit that step), and the lowest-slot same-step
    tie-break. ``launch_*``/``cos_a``/``sin_a``/``speed`` are ``[M]``; ``px``,
    ``py`` are ``[H+1, P]`` planet positions per step; ``p_alive0`` is ``[P]``
    (step-0 alive); ``radii`` is ``[P]``.
    Returns ``(contact_slot, eta)`` — ``contact_slot == -1`` and ``eta == H`` when
    the fleet contacts no planet (or dies first).

    Two-phase to keep the exact swept-pair off the common clear-shot path:

    * **Broad phase** — an AABB cull (the fleet's full-horizon segment box vs each
      planet's swept box inflated by its radius). A planet whose box can't overlap
      the segment can never be hit, so it's dropped. The per-candidate shortlist
      collapses to the few real near-path planets (~1-3 for a clear shot vs ``P``).
      Conservative → the kept set always contains every hittable planet, so the
      result is **byte-identical** to checking all ``P``.
    * **Narrow phase** — the exact swept-pair only on the shortlisted planets,
      flattened to ``N = M`` candidates and run in byte-budgeted chunks (the
      dense ``[N,K,H]`` form would OOM when the regroup grid makes ``M`` large).

    ``amin`` reductions are order-independent so chunking/culling don't perturb the
    values (byte-exact + CPU≡CUDA guarantees hold). Runs eager; the one host sync
    (max shortlist length) is cheap.
    """
    M = cos_a.shape[0]
    P = px.shape[-1]
    dev = cos_a.device
    dt = launch_x.dtype
    N = M
    big = _BIG

    lx = launch_x.reshape(N); ly = launch_y.reshape(N)
    ca = cos_a.reshape(N); sa = sin_a.reshape(N); sp = speed.reshape(N)

    # --- Broad phase: AABB cull (no time axis → cheap). The conservative segment box
    # runs launch → launch + u·seg_len, where seg_len bounds the fleet's relevant
    # flight (distance to the intercept; falls back to the full horizon v·H). The
    # planet box is its swept extent over [0,H] inflated by its radius. ---
    slen = (sp * float(H)) if seg_len is None else seg_len.reshape(N)
    end_x = lx + ca * slen; end_y = ly + sa * slen
    seg_xmin = torch.minimum(lx, end_x); seg_xmax = torch.maximum(lx, end_x)   # [N]
    seg_ymin = torch.minimum(ly, end_y); seg_ymax = torch.maximum(ly, end_y)
    bb_xmin = px.amin(0) - radii                                              # [P]
    bb_xmax = px.amax(0) + radii
    bb_ymin = py.amin(0) - radii
    bb_ymax = py.amax(0) + radii
    keep = ~(
        (seg_xmax.unsqueeze(1) < bb_xmin) | (seg_xmin.unsqueeze(1) > bb_xmax)
        | (seg_ymax.unsqueeze(1) < bb_ymin) | (seg_ymin.unsqueeze(1) > bb_ymax)
    )                                                                          # [N, P]
    K = max(1, int(keep.sum(1).amax().item()))            # one host sync (eager-cheap)
    order = (~keep).to(torch.int8).argsort(dim=1, stable=True)                 # kept first
    shortlist = order[:, :K]                                                   # [N, K]
    valid = keep.gather(1, shortlist)                                          # [N, K]

    k = torch.arange(H + 1, device=dev, dtype=dt)                              # [H+1]
    t_ax = torch.arange(H + 1, device=dev).view(1, H + 1, 1)                   # [1,H+1,1]
    step_h = torch.arange(1, H + 1, device=dev, dtype=dt).view(1, H, 1)        # [1,H,1]

    # ~16 float intermediates of [chunk, H, K] dominate; budget the largest tensor.
    bytes_per = max(1, 16 * H * K * 4)
    chunk = max(4096, max_bytes // bytes_per)
    chunk = min(chunk, max(N, 1))

    contacts: list[Tensor] = []
    etas: list[Tensor] = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sl = shortlist[s:e]                                                   # [n, K]
        fx = lx[s:e].view(-1, 1) + ca[s:e].view(-1, 1) * sp[s:e].view(-1, 1) * k   # [n, H+1]
        fy = ly[s:e].view(-1, 1) + sa[s:e].view(-1, 1) * sp[s:e].view(-1, 1) * k
        # advanced-index the K shortlisted planets directly → [n, H+1, K] (no [n,H+1,P])
        sl_e = sl.view(-1, 1, K)
        pxc = px[t_ax, sl_e]                                                  # [n, H+1, K]
        pyc = py[t_ax, sl_e]
        radc = radii[sl]                                                      # [n, K]
        alivec = p_alive0[sl] & valid[s:e]                                    # [n, K]
        real_slot = sl.to(dt)                                                 # [n, K]

        fx0 = fx[:, :-1].unsqueeze(-1); fy0 = fy[:, :-1].unsqueeze(-1)        # [n,H,1]
        fx1 = fx[:, 1:].unsqueeze(-1);  fy1 = fy[:, 1:].unsqueeze(-1)
        hit = _swept_pair_hit_mask(
            fx0, fy0, fx1, fy1,
            pxc[:, :-1, :], pyc[:, :-1, :], pxc[:, 1:, :], pyc[:, 1:, :],
            radc.unsqueeze(1),
        )                                                                     # [n,H,K]
        hit = hit & alivec.unsqueeze(1)

        planet_hit_step = torch.where(hit, step_h, torch.full_like(step_h, big)).amin(1)  # [n,K]
        first_planet_step = planet_hit_step.amin(1)                           # [n]
        is_first = planet_hit_step == first_planet_step.unsqueeze(-1)
        contact_planet = torch.where(is_first, real_slot, torch.full_like(real_slot, big)).amin(1)  # [n]

        # env death: OOB at the new position OR the segment grazes the sun (static).
        nfx = fx[:, 1:]; nfy = fy[:, 1:]; ofx = fx[:, :-1]; ofy = fy[:, :-1]   # [n,H]
        oob = (nfx < 0) | (nfx > BOARD_SIZE) | (nfy < 0) | (nfy > BOARD_SIZE)
        vx = nfx - ofx; vy = nfy - ofy
        wx = CENTER - ofx; wy = CENTER - ofy
        vv = (vx * vx + vy * vy).clamp(min=1e-12)
        t = ((wx * vx + wy * vy) / vv).clamp(0.0, 1.0)
        cxp = ofx + t * vx; cyp = ofy + t * vy
        sun = ((cxp - CENTER) ** 2 + (cyp - CENTER) ** 2) < (SUN_RADIUS * SUN_RADIUS)
        env = oob | sun                                                       # [n,H]
        death_step = torch.where(env, step_h.squeeze(-1), torch.full_like(env, big, dtype=dt)).amin(1)  # [n]

        # Planet collision resolves BEFORE env removal in the same step (<=).
        ht = (first_planet_step <= death_step) & (first_planet_step < big)
        contacts.append(torch.where(ht, contact_planet, torch.full_like(contact_planet, -1.0)).long())
        etas.append(torch.where(ht, first_planet_step, torch.full_like(first_planet_step, float(H))))

    contact = (contacts[0] if len(contacts) == 1 else torch.cat(contacts)).view(M)
    eta = (etas[0] if len(etas) == 1 else torch.cat(etas)).view(M)
    return contact, eta


# === orbit_lite.movement_step ===

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor




@dataclass(frozen=True)
class PlannedLaunches:
    source_slots: Tensor
    angle: Tensor
    ships: Tensor
    target_slots: Tensor
    eta_turns: Tensor
    valid: Tensor
    fleet_ids: Tensor



@dataclass(frozen=True)
class LaunchEntries:
    """Multi-launch table for one planning step.

    Each ``[L]`` entry encodes a single launch:

        ``source_slots[b, l]`` -> ``target_slots[b, l]`` with ``ships[b, l]``
        ships at heading ``angle[b, l]`` (rad), ETA ``eta[b, l]`` turns.

    Multiple entries may share the same ``source_slots`` value to encode
    multi-launch fan-out from a single planet. The per-source sum of
    ``ships`` over ``valid`` entries must respect that source's ship budget;
    the engine debits sources sequentially in entry order, so callers should
    plan against running residuals rather than the original budget.

    Entry order also defines the launch dispatch order — fleet IDs assigned
    via :func:`infer_planned_launches_from_entries` increase in cumulative
    order over valid entries, matching the engine's ``cumsum`` rule for
    sparse launch payloads.
    """

    source_slots: Tensor  # [L] long
    target_slots: Tensor  # [L] long
    ships: Tensor  # [L] float
    angle: Tensor  # [L] float
    eta: Tensor  # [L] float
    valid: Tensor  # [L] bool

    @property
    def width(self) -> int:
        return int(self.source_slots.shape[0])





def concat_launch_entries(entries: Sequence[LaunchEntries]) -> LaunchEntries:
    """Concatenate launch-entry tables along the L axis.

    All inputs must share the same ``B`` and per-tensor dtype/device.
    """
    if not entries:
        raise ValueError("concat_launch_entries requires at least one entry table")
    if len(entries) == 1:
        return entries[0]
    return LaunchEntries(
        source_slots=torch.cat([e.source_slots for e in entries], dim=0),
        target_slots=torch.cat([e.target_slots for e in entries], dim=0),
        ships=torch.cat([e.ships for e in entries], dim=0),
        angle=torch.cat([e.angle for e in entries], dim=0),
        eta=torch.cat([e.eta for e in entries], dim=0),
        valid=torch.cat([e.valid for e in entries], dim=0),
    )


def disambiguate_duplicate_launches(
    entries: LaunchEntries,
    *,
    epsilon: float = 1.0e-5,
) -> LaunchEntries:
    """Perturb angle on duplicate launches so they're tracker-distinguishable.

    The engine's slot-order fleet-id assignment plus the agent's
    reconciliation by ``(owner, source, ships, angle)`` cannot disambiguate
    two pending entries that share the full tuple, even though the engine
    creates two distinct fleets. ``PlanetMovement._reconcile_pending_own_launches``
    hard-fails on such collisions ("multiple pending entries resolved to the
    same engine fleet id …").

    This helper finds entries that share ``(source, angle, ships)`` with an
    earlier valid entry in the same lane and adds ``k * epsilon`` to the
    angle of the k-th duplicate. ``epsilon = 1e-5`` rad is well above
    float32's ULP at angle magnitude ~1 (≈6e-8) and well below any
    behaviorally-meaningful aim error (5e-4 unit displacement at 50-unit
    fleet range — sub-planet-radius).

    Both the engine action (``_entries_to_sparse_payload``) and the stash
    (``infer_planned_launches_from_entries``) read ``entries.angle``, so
    applying the perturbation here keeps both branches consistent — the
    engine creates fleets with the perturbed angle, the obs reports the
    perturbed angle, and the stash matches.
    """
    src = entries.source_slots                                                 # [L]
    ang = entries.angle                                                         # [L]
    ships = entries.ships                                                       # [L]
    valid = entries.valid                                                       # [L]
    L = src.shape[0]
    if L < 2 or not bool(valid.any()):
        return entries
    device = src.device
    src_i = src.unsqueeze(1)                                                    # [L, 1]
    src_j = src.unsqueeze(0)                                                    # [1, L]
    ang_i = ang.unsqueeze(1)
    ang_j = ang.unsqueeze(0)
    ships_i = ships.unsqueeze(1)
    ships_j = ships.unsqueeze(0)
    valid_i = valid.unsqueeze(1)
    valid_j = valid.unsqueeze(0)
    j_indices = torch.arange(L, device=device).view(1, L)
    i_indices = torch.arange(L, device=device).view(L, 1)
    earlier = j_indices < i_indices                                             # [L, L]
    match = (
        valid_i & valid_j
        & (src_i == src_j)
        & (ang_i == ang_j)
        & (ships_i == ships_j)
        & earlier
    )                                                                           # [L, L]
    if not bool(match.any()):
        return entries
    dup_count = match.sum(dim=1).to(ang.dtype)                                  # [L]
    new_angle = ang + dup_count * float(epsilon)
    return LaunchEntries(
        source_slots=entries.source_slots,
        target_slots=entries.target_slots,
        ships=entries.ships,
        angle=new_angle,
        eta=entries.eta,
        valid=entries.valid,
    )






def ensure_planet_movement(
    *,
    obs_tensors: dict,
    expected_cfg: MovementConfig,
    cached_movement: PlanetMovement | None,
) -> PlanetMovement:
    """Reuse the cached movement (rolled forward) if its config matches, else
    rebuild from the observation. Returns the live movement cache."""
    if cached_movement is not None and cached_movement.config == expected_cfg:
        cached_movement.update(obs_tensors)
        return cached_movement
    return PlanetMovement.from_obs_tensors(obs_tensors, config=expected_cfg)


def _resolve_player_next_fleet_id(
    obs_tensors: dict,
    *,
    device: torch.device,
) -> Tensor:
    next_fleet_id = obs_tensors.get("player_next_fleet_id", obs_tensors.get("next_fleet_id"))
    if next_fleet_id is None:
        return torch.zeros((), dtype=torch.long, device=device)
    return next_fleet_id.to(device=device, dtype=torch.long)


def infer_planned_launches_from_entries(
    *,
    obs_tensors: dict,
    movement: PlanetMovement,
    entries: LaunchEntries,
    player_id: int,
) -> PlannedLaunches:
    """Resolve fleet IDs and target/ETA arrivals for a launch table.

    Fleet IDs increase in entry order over valid launches via
    ``cumsum(valid) - valid``. This matches the engine's sparse rule and
    cleanly handles multi-launch from the same source slot (each entry receives
    a distinct fleet ID). Target/ETA are recomputed via the swept-pair physics
    in :func:`_estimate_new_fleet_arrivals`. Result is shaped ``[L]``.
    """
    source_slots = entries.source_slots
    angle = entries.angle
    ships = entries.ships
    launch_valid = entries.valid
    L = source_slots.shape[0]
    device = source_slots.device
    P = max(int(movement.P), 1)

    next_fleet_id = _resolve_player_next_fleet_id(obs_tensors, device=device)
    # ``cumsum(valid) - valid`` mirrors the engine's launch_rank formula and is
    # independent of source ordering, so it supports multi-launch per source.
    launch_long = launch_valid.to(torch.long)
    launch_rank = launch_long.cumsum(0) - launch_long
    fleet_ids = next_fleet_id + launch_rank

    src_safe = source_slots.clamp(min=0, max=P - 1)
    launch_x, launch_y = movement.position_at_slots(src_safe, 0)
    source_r = movement.radii[src_safe]
    start_x = launch_x + torch.cos(angle) * (source_r + 0.1)
    start_y = launch_y + torch.sin(angle) * (source_r + 0.1)
    source_planet_ids = movement.planet_ids[src_safe]

    rows = torch.full((L, 7), -1.0, dtype=movement.dtype, device=device)
    rows[..., 0] = fleet_ids.to(dtype=movement.dtype)
    rows[..., 1] = float(player_id)
    rows[..., 2] = start_x.to(dtype=movement.dtype)
    rows[..., 3] = start_y.to(dtype=movement.dtype)
    rows[..., 4] = angle.to(dtype=movement.dtype)
    rows[..., 5] = source_planet_ids.to(dtype=movement.dtype)
    rows[..., 6] = ships.to(dtype=movement.dtype)
    rows[..., 0] = torch.where(
        launch_valid, rows[..., 0], torch.full_like(rows[..., 0], -1.0)
    )

    target_slots = torch.zeros(L, dtype=torch.long, device=device)
    eta_turns = torch.zeros(L, dtype=torch.float32, device=device)
    intent_valid = torch.zeros(L, dtype=torch.bool, device=device)
    fleet_slot = torch.where(launch_valid)[0]
    if int(fleet_slot.numel()) > 0:
        estimate = _estimate_new_fleet_arrivals(
            movement=movement,
            obs_fleets=rows,
            fleet_slot=fleet_slot,
        )
        valid_hit = estimate["has_hit"]
        if bool(valid_hit.any()):
            src = fleet_slot[valid_hit]
            target_slots[src] = estimate["target_slot"][valid_hit]
            eta_turns[src] = estimate["eta_index"][valid_hit].to(dtype=torch.float32) + 1.0
            intent_valid[src] = True

    return PlannedLaunches(
        source_slots=source_slots,
        angle=angle,
        ships=ships,
        target_slots=target_slots,
        eta_turns=eta_turns,
        valid=intent_valid,
        fleet_ids=fleet_ids,
    )




def apply_private_planned_launches(
    *,
    movement: PlanetMovement,
    launches: PlannedLaunches,
    owner_id: int,
    obs_tensors: dict,
) -> None:
    """Record an agent's just-decided launches into its movement cache.

    Seeds the arrival buckets with the source-derived prediction but does *not*
    seed the ``tracked_fleet_ids`` ledger directly: ``launches.fleet_ids`` come
    from the global ``next_fleet_id`` plus a cumsum, which collides with other
    slots' IDs because the engine processes player actions in slot order.
    Instead the launches are stashed and paired against the next observation's
    fleets (which carry the engine's authoritative IDs) via
    ``_reconcile_pending_own_launches``.

    ``obs_tensors`` is required (we snapshot ``next_fleet_id`` for reconciliation).
    """
    if not movement.track_fleets:
        return
    movement.record_fleet_arrivals(
        target_slots=launches.target_slots,
        owner_ids=int(owner_id),
        ships=launches.ships,
        eta=launches.eta_turns,
        valid=launches.valid,
    )
    nfid = obs_tensors.get("next_fleet_id")
    if nfid is None:
        raise ValueError("obs_tensors is missing 'next_fleet_id'")
    movement.stash_pending_own_launches(
        owner_id=int(owner_id),
        source_slots=launches.source_slots,
        ships=launches.ships,
        angle=launches.angle,
        target_slots=launches.target_slots,
        eta=launches.eta_turns,
        valid=launches.valid,
        prev_next_fleet_id=nfid,
    )


# === orbit_lite.adapter ===
"""Observation/action adapter between the move-list format and tensors.

Converts an observation dict (``{"planets": [...], "fleets": [...], ...}``) into
the named tensor observation the planner consumes, and converts the planner's
sparse launch payload
(``{"from_planet_id": [L], "angle": [L], "num_ships": [L], "counts": scalar}``)
back into a move list (``[[from_planet_id, angle, ships], ...]``).
"""


from typing import Any

import torch





def _infer_player_count_from_obs(planets: list[Any], fleets: list[Any], player_id: int) -> int:
    owners: list[int] = [int(player_id)]
    for row in planets:
        if len(row) >= 2 and int(row[0]) >= 0 and int(row[1]) >= 0:
            owners.append(int(row[1]))
    for row in fleets:
        if len(row) >= 2 and int(row[0]) >= 0 and int(row[1]) >= 0:
            owners.append(int(row[1]))
    return 4 if max(owners, default=0) >= 2 else 2


def _dict_obs_to_tensor(
    obs: dict[str, Any],
    player_id: int,
    P: int = P_MAX,
    F: int = F_MAX,
    device: Any = "cpu",
) -> dict[str, Any]:
    """Convert an observation dict to a single-game tensor observation.

    Input format::

        obs["planets"] = [[planet_id, owner, x, y, radius, ships, production], ...]
        obs["fleets"]  = [[fleet_id, owner, x, y, angle, from_id, ships], ...]

    Returns a tensor observation dict::

        "planets" : [P, 7]            "fleets" : [F, 7]
        "initial_planets" : [P, 7]    "comet_planet_ids" : [G*C]
        "comets" : nested padded tensors
        "player" / "angular_velocity" / "next_fleet_id" / "step" /
        "episode_steps" / "remainingOverageTime" : scalars
    """
    dev = torch.device(device)

    planets_raw = obs.get("planets", [])
    initial_planets_raw = obs.get("initial_planets", planets_raw)
    fleets_raw = obs.get("fleets", [])
    comets_raw = obs.get("comets", [])
    comet_planet_ids_raw = obs.get("comet_planet_ids", [])
    step = int(obs.get("step", 0))
    angvel = float(obs.get("angular_velocity", 0.03))
    max_steps = int(obs.get("episode_steps", DEFAULT_EPISODE_STEPS))
    remaining_overtime = float(obs.get("remainingOverageTime", 2.0))
    next_fleet_id = int(obs.get("next_fleet_id", 0))

    planet_t = torch.zeros(P, 7, dtype=torch.float32, device=dev)
    planet_t[..., 0] = -1.0
    for i, p in enumerate(planets_raw[:P]):
        pid, owner, x, y, r, ships, prod = p[:7]
        planet_t[i, 0] = float(pid)
        planet_t[i, 1] = float(owner)
        planet_t[i, 2] = float(x)
        planet_t[i, 3] = float(y)
        planet_t[i, 4] = float(r)
        planet_t[i, 5] = float(ships)
        planet_t[i, 6] = float(prod)

    initial_planet_t = torch.zeros(P, 7, dtype=torch.float32, device=dev)
    initial_planet_t[..., 0] = -1.0
    for i, p in enumerate(initial_planets_raw[:P]):
        pid, owner, x, y, r, ships, prod = p[:7]
        initial_planet_t[i, 0] = float(pid)
        initial_planet_t[i, 1] = float(owner)
        initial_planet_t[i, 2] = float(x)
        initial_planet_t[i, 3] = float(y)
        initial_planet_t[i, 4] = float(r)
        initial_planet_t[i, 5] = float(ships)
        initial_planet_t[i, 6] = float(prod)

    fleet_t = torch.zeros(F, 7, dtype=torch.float32, device=dev)
    fleet_t[..., 0] = -1.0
    fleet_t[..., 5] = -1.0
    for i, f in enumerate(fleets_raw[:F]):
        fid, owner, x, y, angle, from_id, ships = f[:7]
        fleet_t[i, 0] = float(fid)
        fleet_t[i, 1] = float(owner)
        fleet_t[i, 2] = float(x)
        fleet_t[i, 3] = float(y)
        fleet_t[i, 4] = float(angle)
        fleet_t[i, 5] = float(from_id)
        fleet_t[i, 6] = float(ships)

    comet_ids = torch.full((COMET_EVENTS, COMETS_PER_EVENT), -1, dtype=torch.int32, device=dev)
    comet_paths = torch.full(
        (COMET_EVENTS, COMETS_PER_EVENT, COMET_PATH_MAX, 2),
        float("nan"),
        dtype=torch.float32,
        device=dev,
    )
    comet_path_index = torch.full((COMET_EVENTS,), -1, dtype=torch.int32, device=dev)
    for group_idx, group in enumerate(comets_raw[:COMET_EVENTS]):
        comet_path_index[group_idx] = int(group.get("path_index", -1))
        group_ids = group.get("planet_ids", [])
        group_paths = group.get("paths", [])
        for comet_idx, pid in enumerate(group_ids[:COMETS_PER_EVENT]):
            comet_ids[group_idx, comet_idx] = int(pid)
        for comet_idx, path in enumerate(group_paths[:COMETS_PER_EVENT]):
            for point_idx, point in enumerate(path[:COMET_PATH_MAX]):
                comet_paths[group_idx, comet_idx, point_idx, 0] = float(point[0])
                comet_paths[group_idx, comet_idx, point_idx, 1] = float(point[1])

    comet_planet_ids = torch.full(
        (COMET_EVENTS * COMETS_PER_EVENT,),
        -1,
        dtype=torch.int32,
        device=dev,
    )
    for idx, pid in enumerate(comet_planet_ids_raw[: COMET_EVENTS * COMETS_PER_EVENT]):
        comet_planet_ids[idx] = int(pid)

    return {
        "planets": planet_t,
        "fleets": fleet_t,
        "player": torch.tensor(player_id, dtype=torch.int32, device=dev),
        "player_count": torch.tensor(_infer_player_count_from_obs(planets_raw, fleets_raw, player_id), dtype=torch.int32, device=dev),
        "angular_velocity": torch.tensor(angvel, dtype=torch.float32, device=dev),
        "initial_planets": initial_planet_t,
        "next_fleet_id": torch.tensor(next_fleet_id, dtype=torch.int32, device=dev),
        "comets": {
            "planet_ids": comet_ids,
            "paths": comet_paths,
            "path_index": comet_path_index,
        },
        "comet_planet_ids": comet_planet_ids,
        "step": torch.tensor(step, dtype=torch.int32, device=dev),
        "episode_steps": torch.tensor(max_steps, dtype=torch.int32, device=dev),
        "remainingOverageTime": torch.tensor(remaining_overtime, dtype=torch.float32, device=dev),
    }


def _sparse_actions_to_list(
    action_payload: dict[str, Any],
    obs: dict[str, Any],
    player_id: int,
) -> list[list[Any]]:
    # The payload is produced by ``entries_to_sparse_payload`` and is already a
    # well-formed sparse row: ``from_planet_id``/``angle``/``num_ships`` are rank-1
    # tensors and ``counts`` is a scalar count of active launches.
    from_pid_t = action_payload["from_planet_id"]
    angle_t = action_payload["angle"]
    num_ships_t = action_payload["num_ships"]
    counts = int(action_payload["counts"].item())
    planets_by_id = {int(p[0]): p for p in obs.get("planets", []) if len(p) >= 7}
    moves: list[list[Any]] = []
    for launch_idx in range(counts):
        from_pid = int(from_pid_t[launch_idx].item())
        ships = float(num_ships_t[launch_idx].item())
        angle = float(angle_t[launch_idx].item())
        if ships < 1.0:
            continue
        source = planets_by_id.get(from_pid)
        if source is None:
            continue
        owner = int(source[1])
        available = float(source[5])
        if owner != int(player_id):
            continue
        if ships != float(round(ships)) or ships > available:
            raise ValueError(
                "Invalid launch ship count in sparse action payload at "
                f"from_planet_id={from_pid}: requested={ships}, available={available}. "
                "Counts must be finite, integer-valued, >= 0, and <= available planet ships."
            )
        moves.append([from_pid, angle, int(ships)])
    return moves


def single_obs_to_tensor(
    obs: dict[str, Any],
    *,
    player_id: int,
    P: int = P_MAX,
    F: int = F_MAX,
    device: Any = "cpu",
) -> dict[str, Any]:
    """Public wrapper: convert one observation dict to a tensor observation."""
    return _dict_obs_to_tensor(obs, player_id=player_id, P=P, F=F, device=device)


def sparse_action_row_to_moves(
    action_payload: dict[str, Any],
    obs: dict[str, Any],
    *,
    player_id: int,
) -> list[list[Any]]:
    """Decode a sparse launch payload into a move list.

    The payload may contain multiple entries from the same source planet — each
    valid entry produces a ``[from_planet_id, angle, ships]`` move in iteration
    order, mirroring how the engine processes sparse rows.
    """
    return _sparse_actions_to_list(action_payload, obs, player_id=int(player_id))




# === orbit_lite.planner_core ===
"""Flow-diff scored planner core: candidate scoring, shortlists, aim, selection.

Pure, tensor-only planning helpers for one game: the competitive net-ship-delta
scorer, target/source shortlists, capture-floor sizing, the strict-superset
reachability gate, the device-stable greedy selector, the hold-reserve cap
``safe_drain``, and the pressure-gradient regrouper.
"""


import torch
from torch import Tensor












def largest_initial_player_count(obs_tensors: dict) -> int:
    """Player count for the match: metadata if present, else distinct initial owners.

    """
    metadata_count = obs_tensors.get("player_count")
    if metadata_count is not None:
        count = (
            int(metadata_count.flatten()[0].item())
            if isinstance(metadata_count, Tensor)
            else int(metadata_count)
        )
        if count in (2, 4):
            return count
    initial = obs_tensors["initial_planets"]      # [P, 7]
    pid = initial[:, 0]
    owner = initial[:, 1]
    mask = (pid >= 0) & (owner >= 0)
    owners = owner[mask]
    n_max = 2
    if owners.numel() > 0:
        n_max = max(n_max, int(torch.unique(owners.long()).numel()))
    return n_max


# ---------------------------------------------------------------------------
# Scoring (P2): candidate launches -> competitive net-ship-delta
# ---------------------------------------------------------------------------


def make_launch_set(
    *,
    source_slots: Tensor,   # [C, L] long
    target_slots: Tensor,   # [C, L] long
    ships: Tensor,          # [C, L] float
    eta: Tensor,            # [C, L] float (steps to arrival, >= 1)
    valid: Tensor,          # [C, L] bool
    player_id: int,
) -> LaunchSet:
    """Build a candidate-axis ``LaunchSet`` owned by ``player_id``."""
    owner = torch.full_like(source_slots, int(player_id), dtype=torch.long)
    return LaunchSet(
        source_slots=source_slots.to(torch.long),
        target_slots=target_slots.to(torch.long),
        ships=ships,
        eta=eta,
        owner=owner,
        valid=valid.to(torch.bool),
    )


def competitive_score(
    diff: GarrisonFlowDiff, *, player_id: int, opp_weights: Tensor | None = None,
    terminal_prod_weight: float = 0.0,
) -> Tensor:
    """Competitive score: ``Δnet_me − Σ_opp w_opp · Δnet_opp``.

    ``diff.net_ship_delta`` is ``[*prefix, A]`` (per-player change in net ships
    gained = produced − lost-to-combat); returns ``[*prefix]``.

    Default (``opp_weights=None``): the opponent term is the equal-weight SUM
    over rivals — correct zero-sum objective for 2P, but in FFA it values
    total damage dealt, which rewards mutual-destruction trades that leave
    both fighters weaker relative to the bystanders.

    With ``opp_weights`` (``[A]`` float, 0 at ``player_id``, summing to 1
    over opponents): the opponent term becomes a weighted AVERAGE, so damage
    is valued by how much it shifts my standing against the rivals that
    actually threaten me (weights ∝ rival strength), and an unprofitable
    trade no longer scores positive just because the victim lost more.
    """
    net = diff.net_ship_delta                       # [*prefix, A]
    me = net[..., int(player_id)]
    if opp_weights is None:
        opp = net.sum(dim=-1) - me
    else:
        opp = (net * opp_weights).sum(dim=-1)
    score = me - opp
    if terminal_prod_weight != 0.0 and diff.terminal_prod_delta is not None:
        # Post-horizon production stream: the flow terms truncate a captured
        # planet's payoff at H, so a neutral whose in-horizon production only
        # repays its garrison cost scores ~0 and never clears the roi
        # threshold. Credit the production OWNED at the horizon's final step
        # for ``terminal_prod_weight`` further steps, same opponent weighting
        # as the in-horizon flow.
        term = diff.terminal_prod_delta
        t_me = term[..., int(player_id)]
        if opp_weights is None:
            t_opp = term.sum(dim=-1) - t_me
        else:
            t_opp = (term * opp_weights).sum(dim=-1)
        score = score + float(terminal_prod_weight) * (t_me - t_opp)
    return score


def score_candidates(
    status: PlanetGarrisonStatus,
    *,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    launches: LaunchSet,
    player_id: int,
    opp_weights: Tensor | None = None,
    terminal_prod_weight: float = 0.0,
    terminal_neutral_only: bool = False,
) -> Tensor:
    """Competitive score per candidate. ``[C]`` (or scalar if no candidate axis).

    Uses the sparse exact flow projector. ``opp_weights`` and
    ``terminal_prod_weight`` are forwarded to :func:`competitive_score`
    (None / 0.0 = legacy behaviour).
    """
    diff = sparse_launch_flow_delta(
        status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=int(player_count),
        launches=launches,
        player_id=int(player_id),
        terminal_neutral_only=terminal_neutral_only,
    )
    return competitive_score(
        diff, player_id=int(player_id), opp_weights=opp_weights,
        terminal_prod_weight=terminal_prod_weight,
    )


# ---------------------------------------------------------------------------
# Candidate generation + greedy selection (P3: single-source, single-k, attack)
# ---------------------------------------------------------------------------



# Selection on CPU and CUDA must agree exactly: `torch.topk` / `torch.argmax`
# break ties differently across devices, and this planner ranks by integer ship
# counts / proximity that tie constantly — so device-stable selection is what
# keeps batch-CUDA play identical to CPU. We break ties by ascending slot index
# on both devices via a stable sort / lowest-index argmax.


def _stable_topk_indices(ranked: Tensor, k: int) -> Tensor:
    """Indices of the top-``k`` along the last dim, ties broken by ascending index
    identically on CPU and CUDA (stable descending sort)."""
    order = torch.argsort(ranked, dim=-1, descending=True, stable=True)
    return order[..., :max(1, int(k))]


def _stable_argmax(scores: Tensor) -> Tensor:
    """Lowest-index argmax along the last dim, device-deterministic on ties."""
    C = int(scores.shape[-1])
    is_max = scores == scores.max(dim=-1, keepdim=True).values
    idx = torch.arange(C, device=scores.device).expand_as(scores)
    return torch.where(is_max, idx, torch.full_like(idx, C)).argmin(dim=-1)


def _candidate_indices(values: Tensor, mask: Tensor, cap: int) -> tuple[Tensor, Tensor]:
    """Top-``cap`` slot indices of ``values`` under ``mask``. ``([K] long, [K] bool)``.

    Device-stable (ascending-index tie-break) — see note above.
    """
    p_count = values.shape[0]
    k = p_count if cap <= 0 else min(int(cap), p_count)
    neg_inf = torch.full_like(values, float("-inf"))
    ranked = torch.where(mask, values, neg_inf)
    top_idx = _stable_topk_indices(ranked, max(1, k))
    top_vals = ranked[top_idx]
    return top_idx, top_vals > float("-inf")


def is_comet_planet(obs_tensors: dict, P: int, device: torch.device) -> Tensor | None:
    """Per-slot mask of active comet planets, or ``None`` if absent."""
    comet_ids = obs_tensors.get("comet_planet_ids")
    planets = obs_tensors.get("planets")
    if comet_ids is None or planets is None:
        return None
    planet_ids = planets[..., 0].long()                       # [P]
    comet_ids = comet_ids.to(device=device)
    mask = torch.zeros(P, dtype=torch.bool, device=device)
    for c in range(int(comet_ids.shape[-1])):
        cid = comet_ids[c]
        mask = mask | ((planet_ids == cid) & (cid >= 0))
    return mask


def reinforcement_timing_factor(
    eta: Tensor,
    *,
    eta_free: float,
    eta_scale: float,
) -> Tensor:
    """Reaction-likelihood ramp ``ρ(eta) ∈ [0, 1]`` for reinforcement risk.

    ``ρ = clamp((eta − eta_free) / eta_scale, 0, 1)``. Below ``eta_free`` turns of
    flight the enemy has no time to react (ρ=0); over the next ``eta_scale`` turns
    reaction likelihood ramps linearly to 1. Pure arithmetic → CPU/CUDA agree.
    """
    scale = max(float(eta_scale), 1e-6)
    return ((eta - float(eta_free)) / scale).clamp(0.0, 1.0)


def capture_floor(
    garrison_status: PlanetGarrisonStatus,
    *,
    target_idx: Tensor,        # [T] long
    k_max: int,
    capture_overhead: float,
    player_id: int,
    reinforcement: Tensor | None = None,   # [T, K'>=K] float; added before ceil
) -> Tensor:
    """Owner-aware send floor per target at arrival turn ``k``. ``[T, K]``.

    - If I **own** the target at ``k`` (reinforcement), the floor is 1 — arriving
      ships add to my garrison, there is nothing to clear.
    - Otherwise (capture / retake), the floor is ``ceil(projected_defenders_at_k +
      overhead)``.
    - ``reinforcement`` (optional, ``[T, K' ≥ K]``) is added to the defender count
      before the ceil on capture cells (not on ``mine_at_k`` reinforcement cells) —
      the ETA-aware reactive-reinforcement margin. ``None`` ⇒ today's behaviour.

    Assumes ``k_max <= H``.
    """
    ships = garrison_status.ships
    owner = garrison_status.owner
    dtype = ships.dtype if ships.is_floating_point() else torch.float32
    T = target_idx.shape[0]
    H_axis = int(ships.shape[-1])
    P = int(ships.shape[0])
    K = max(0, min(int(k_max), H_axis - 1))
    if K == 0:
        return torch.empty(T, 0, dtype=dtype, device=ships.device)
    tgt = target_idx.clamp(min=0, max=max(P - 1, 0))
    gathered = ships[tgt].to(dtype=dtype)                       # [T, H+1]
    owner_g = owner[tgt]                                        # [T, H+1]
    k_idx = torch.arange(1, K + 1, device=ships.device).view(1, K).expand(T, K)
    defenders = gathered.gather(-1, k_idx)                      # [T, K]
    mine_at_k = owner_g.gather(-1, k_idx) == int(player_id)
    if reinforcement is not None:
        # Caller passes a margin with K' >= K (built from k_max=K_eta, while this
        # function's K = min(k_max, H-1) <= K_eta); slice down to our own K.
        assert reinforcement.shape[-1] >= K, (
            f"reinforcement last dim {reinforcement.shape[-1]} < capture_floor K={K}"
        )
        extra = reinforcement[..., :K].to(dtype=dtype, device=ships.device)
    else:
        extra = 0.0
    cap = (defenders + float(capture_overhead) + extra).clamp(min=1.0).ceil()
    return torch.where(mine_at_k, torch.ones_like(cap), cap)


def attack_target_mask(obs, obs_tensors: dict) -> Tensor:
    """Enemy ∪ neutral, alive, non-comet. ``[P]`` bool."""
    mask = (obs.is_enemy | obs.is_neutral) & obs.alive
    comet = is_comet_planet(obs_tensors, obs.P, obs.device)
    if comet is not None:
        mask = mask & ~comet
    return mask


def friendly_flip_targets(
    obs, garrison_status: PlanetGarrisonStatus, *, H: int, prod: Tensor,
    background: LaunchSet | None = None,
) -> tuple[Tensor, Tensor]:
    """Own planets projected to flip within H -- with optional opp-aware
    augmentation from a background LaunchSet of opp's predicted launches.

    Without ``background``: identical to the historical behaviour, scanning
    only ``garrison_status.owner`` (fleets already in flight under the
    static-opp projection).

    With ``background``: also marks a planet as flipping at tick ``ceil(eta)``
    if a valid opp launch in the background targets it (currently mine) with
    enough ships to crack the projected defender count
    (``ships > defender + 1.0``, matching ``capture_floor``'s overhead). The
    earlier of the two flip ticks wins per planet.

    Returns ``(mask [P] bool, urgency [P] float)``. ``urgency`` ≈ projected
    ships lost if unaddressed = ``prod·(H − flip_turn) + garrison_now`` — same
    ship units as the ROI, used to fill the reserved defensive sub-quota.
    """
    P = obs.P
    device = obs.device
    pid = int(obs.player_id)
    if H <= 0:
        z = torch.zeros(P, device=device)
        return torch.zeros(P, dtype=torch.bool, device=device), z
    owner_h = garrison_status.owner[..., 1:]                     # [P, H]
    flips = obs.owned.unsqueeze(-1) & (owner_h != pid)           # currently mine, not mine at some k

    # Opp-aware augmentation: opp_proj's predicted captures of MY planets
    # show up here, so the defensive shortlist can enumerate reinforcement
    # candidates against them.
    if background is not None and int(background.source_slots.shape[-1]) > 0:
        valid = background.valid
        tgt = background.target_slots.clamp(0, max(P - 1, 0))
        eta = background.eta
        ships_bg = background.ships
        owned_tgt = obs.owned[tgt]
        # arrival absolute tick from now (1..H); clamp into [1, H] for index safety.
        arr_tick = torch.ceil(eta).long().clamp(1, H)
        # gather defender count at the arrival tick from the static-opp projection
        # via paired advanced indexing: defender[l] = garrison_status.ships[tgt[l], arr_tick[l]].
        defender = garrison_status.ships[tgt, arr_tick]
        captures = ships_bg > (defender + 1.0)
        contributes = valid & owned_tgt & captures
        if bool(contributes.any()):
            sel_tgt = tgt[contributes]
            sel_idx = arr_tick[contributes] - 1
            flips = flips.clone()
            flips[sel_tgt, sel_idx] = True

    any_flip = flips.any(dim=-1)                                 # [P]
    # earliest flip turn (lowest-index True); _stable_argmax instead of raw argmax
    # so the tie among post-flip turns resolves identically on CPU and CUDA.
    flip_turn = _stable_argmax(flips.to(torch.int64)) + 1        # 1-based; valid where any_flip
    remaining = (float(H) - flip_turn.to(prod.dtype)).clamp(min=0.0)
    urgency = prod * remaining + obs.ships
    urgency = torch.where(any_flip, urgency, torch.full_like(urgency, float("-inf")))
    return any_flip, urgency


def build_target_shortlist(
    obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask,
    background: LaunchSet | None = None,
):
    """Single unified shortlist: ``max_offensive_targets`` enemy/neutral targets by
    proximity ∪ ``max_defensive_targets`` friendly-flip targets by urgency., The
    two caps are independent (shortlist width == offensive + defensive), so each can
    be swept on its own. Returns ``(target_idx, target_exists)``.

    The optional ``background`` LaunchSet (opp_proj's predicted launches) is
    forwarded to ``friendly_flip_targets`` so the defensive lane can react to
    opp's new launches, not just to fleets already in flight.
    """
    P = obs.P
    device = obs.device
    n_attack = max(1, min(int(config.max_offensive_targets), P))
    R = max(0, min(int(config.max_defensive_targets), P))

    attack_mask = attack_target_mask(obs, obs_tensors)
    proximity = min_distance_to_targets(cache, source_mask, attack_mask, max_k=K_eta)
    attack_pref = torch.where(attack_mask, -proximity, torch.full_like(proximity, float("-inf")))
    atk_idx, atk_exists = _candidate_indices(attack_pref, attack_mask, n_attack)

    if R > 0:
        flip_mask, urgency = friendly_flip_targets(
            obs, garrison_status, H=H, prod=prod, background=background,
        )
        def_idx, def_exists = _candidate_indices(urgency, flip_mask, R)
        target_idx = torch.cat([atk_idx, def_idx], dim=0)
        target_exists = torch.cat([atk_exists, def_exists], dim=0)
    else:
        target_idx, target_exists = atk_idx, atk_exists
    return target_idx, target_exists


def reachable_mask(
    movement: PlanetMovement,
    *,
    source_idx: Tensor,      # [S] long
    target_idx: Tensor,      # [T] long
    fleet_sizes: Tensor,     # [S, T, G] float
    eta_cap: Tensor,         # [T] float (per-target reach cap)
    eps: float = 1e-4,
) -> Tensor:
    """Strict-superset reachability gate for the body screen, ``[S, T, G]`` bool.

    A cell is reachable iff some step interval ``k in [1, eta_cap[b,t]]`` admits the
    straight-line shot: ``(d_k - gap) <= fleet_speed(size) * k * (1 + eps)`` where
    ``d_k`` is the distance from the source centre @ turn 0 to the target's **swept
    segment** ``[tgt@(k-1), tgt@k]`` and ``gap = src_r + tgt_r + offsets``.

    Using the swept segment (not the point ``tgt@k``) and the surface gap makes this
    a provable *necessary condition* for ``intercept_angle`` viability: a viable shot
    contacts the target at some continuous ``t_c <= eta_cap`` with
    ``dist(src@0, tgt@t_c) - gap <= speed * t_c <= speed * ceil(t_c)``, and the
    segment distance over the interval containing ``t_c`` is ``<= dist(src@0, tgt@t_c)``.
    Hence ``viable => reachable`` (the ``eps`` absorbs fp32 boundary noise) — the gate
    never false-prunes a launch the agent would otherwise aim. ``intercept_angle``
    re-validates every survivor, so the surplus kept beyond true viability is harmless.
    """
    S, T, G = fleet_sizes.shape
    P = int(movement.P)
    dt = movement.dtype
    K = max(1, min(int(movement.movement_horizon), int(torch.ceil(eta_cap.max()).item())))
    src = source_idx.clamp(0, P - 1)
    tgt = target_idx.clamp(0, P - 1)

    # Source centre @ turn 0; target positions @ turns 0..K (segment endpoints).
    sx = movement.x[0][src].view(S, 1, 1)                                   # [S,1,1]
    sy = movement.y[0][src].view(S, 1, 1)
    tx = movement.x[: K + 1].gather(1, tgt.view(1, T).expand(K + 1, T))     # [K+1,T]
    ty = movement.y[: K + 1].gather(1, tgt.view(1, T).expand(K + 1, T))
    ax = tx[:K, :].view(1, K, T); ay = ty[:K, :].view(1, K, T)             # tgt@(k-1)
    bx = tx[1:, :].view(1, K, T); by = ty[1:, :].view(1, K, T)             # tgt@k

    # Point-to-segment distance from (sx,sy) to segment [(ax,ay),(bx,by)] → [S,K,T].
    abx = bx - ax; aby = by - ay
    apx = sx - ax; apy = sy - ay
    denom = (abx * abx + aby * aby).clamp(min=1e-12)
    u = ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0)
    cx = ax + u * abx; cy = ay + u * aby
    seg_dist = torch.sqrt(((sx - cx) ** 2 + (sy - cy) ** 2).clamp(min=0.0))  # [S,K,T]

    src_r = movement.radii[src].view(S, 1, 1)
    tgt_r = movement.radii[tgt].view(1, 1, T)
    gap = src_r + tgt_r + (LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET)
    surf = (seg_dist - gap).clamp(min=0.0)                                   # [S,K,T]

    kv = torch.arange(1, K + 1, device=movement.device, dtype=dt).view(1, K, 1)
    ratio = surf / kv
    within = kv <= eta_cap.view(1, 1, T)                                    # [1,K,T]
    ratio = torch.where(within, ratio, torch.full_like(ratio, float("inf")))
    min_ratio = ratio.amin(dim=1)                                          # [S,T]

    speed = fleet_speed(fleet_sizes.clamp(min=1.0))                          # [S,T,G]
    reachable = min_ratio.unsqueeze(-1) <= speed * (1.0 + float(eps))        # [S,T,G]
    distinct = (src.view(S, 1) != tgt.view(1, T)).unsqueeze(-1)             # [S,T,1]
    return reachable & distinct


def _greedy_select(
    *, P, W, device, dtype, score, cand_src, cand_send, cand_angle, cand_eta,
    cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, source_budget,
    target_exists, roi_threshold,
    rescore_fn=None, max_waves_per_target: int = 1,
) -> LaunchEntries:
    """Masking-only greedy over [C, L] candidates: pick the best wave each iter,
    up to ``max_waves_per_target`` waves per target, source-budget aware across
    all L contributors. Enforces the role mutex: a reinforced planet can't also
    be a source, and vice-versa.

    Force-concentration: when ``max_waves_per_target > 1`` AND ``rescore_fn`` is
    provided, the second-best candidate at an already-hit target is NOT scored
    against the same garrison as the first (which would double-count the
    capture). After each wave fires, ``rescore_fn(w_src[:w+1], w_send[:w+1],
    w_eta[:w+1], w_tgt[:w+1], w_active[:w+1])`` returns a fresh ``[C]`` score
    that the next iteration's mask uses. Default ``rescore_fn=None`` +
    ``max_waves_per_target=1`` is the legacy single-wave-per-target path.
    """
    C, L = int(cand_src.shape[0]), int(cand_src.shape[1])
    cap_per_tgt = max(1, int(max_waves_per_target))
    # Per-shortlist-target wave counter; slots where target_exists is False
    # start at the cap so they're masked out (mirrors the legacy bool init).
    waves_per_target = torch.where(
        target_exists,
        torch.zeros_like(target_exists, dtype=torch.long),
        torch.full_like(target_exists, cap_per_tgt, dtype=torch.long),
    )
    defended = torch.zeros(P, dtype=torch.bool, device=device)                   # reinforced this turn
    used_src = torch.zeros(P, dtype=torch.bool, device=device)                   # contributed this turn

    w_src = torch.zeros(W, L, dtype=torch.long, device=device)
    w_send = torch.zeros(W, L, dtype=dtype, device=device)
    w_angle = torch.zeros(W, L, dtype=dtype, device=device)
    w_eta = torch.ones(W, L, dtype=dtype, device=device)
    w_tgt = torch.zeros(W, L, dtype=torch.long, device=device)
    w_active = torch.zeros(W, L, dtype=torch.bool, device=device)

    for w in range(W):
        taken_cand = waves_per_target[cand_tgt_short] >= cap_per_tgt            # [C]
        budget_at = source_budget[cand_src]                                     # [C, L]
        can_fund = ((cand_send <= budget_at) | ~cand_active).all(dim=-1)        # [C]
        # role mutex: target not already drained as a source; no contributor is a
        # planet we're reinforcing this turn.
        tgt_used_as_src = used_src[cand_tgt_slot]                               # [C]
        contrib_defended = (defended[cand_src] & cand_active).any(dim=-1)       # [C]
        mask = torch.isfinite(score) & ~taken_cand & can_fund & ~tgt_used_as_src & ~contrib_defended
        masked = torch.where(mask, score, torch.full_like(score, float("-inf")))
        best_c = _stable_argmax(masked)                                         # scalar, device-stable
        best_score = masked[best_c]
        fired = bool(torch.isfinite(best_score) & (best_score > roi_threshold))
        if not fired:
            break

        sel_src = cand_src[best_c]                   # [L]
        sel_send = cand_send[best_c]
        sel_active = cand_active[best_c]
        w_src[w] = sel_src
        w_send[w] = torch.where(sel_active, sel_send, torch.zeros_like(sel_send))
        w_angle[w] = cand_angle[best_c]
        w_eta[w] = cand_eta[best_c]
        w_tgt[w] = cand_tgt_slot[best_c]
        w_active[w] = sel_active

        # debit all contributors' sends from their source budgets.
        debit = torch.zeros_like(source_budget)
        debit.scatter_add_(0, sel_src, torch.where(sel_active, sel_send, torch.zeros_like(sel_send)))
        source_budget = (source_budget - debit).clamp(min=0.0)
        # increment wave count for this target (cap_per_tgt=1 mirrors legacy taken-bool).
        waves_per_target[cand_tgt_short[best_c]] += 1
        # role mutex bookkeeping: mark contributors used; mark reinforced targets
        # defended. Sum active marks per planet (order-independent) and OR them in.
        src_mark = torch.zeros(P, dtype=torch.long, device=device)
        src_mark.scatter_add_(0, sel_src, sel_active.to(torch.long))
        used_src = used_src | (src_mark > 0)
        sel_tgt = cand_tgt_slot[best_c]
        sel_is_def = bool(cand_is_def[best_c])
        defended[sel_tgt] = defended[sel_tgt] | sel_is_def

        # Force-concentration: re-score candidates against the already-fired
        # waves before the next iteration's argmax. Without this, wave 2 to a
        # target picks the second-best send blindly, double-counting wave 1's
        # capture/reinforcement. Skipped on the last wave (no next iter) and
        # whenever cap_per_tgt==1 (legacy behaviour, byte-identical).
        if rescore_fn is not None and cap_per_tgt > 1 and (w + 1) < W:
            score = rescore_fn(
                w_src[: w + 1], w_send[: w + 1], w_eta[: w + 1],
                w_tgt[: w + 1], w_active[: w + 1],
            )

    # Flatten waves x contributors into a LaunchEntries table.
    WL = W * L
    entries = LaunchEntries(
        source_slots=w_src.reshape(WL),
        target_slots=w_tgt.reshape(WL),
        ships=torch.where(w_active, w_send, torch.zeros_like(w_send)).reshape(WL),
        angle=torch.where(w_active, w_angle, torch.zeros_like(w_angle)).reshape(WL),
        eta=torch.where(w_active, w_eta, torch.ones_like(w_eta)).reshape(WL),
        valid=w_active.reshape(WL),
    )
    return entries, source_budget   # source_budget = leftover ships per planet


def _plan_regroup(
    *, movement, obs, obs_tensors, garrison_status, leftover, original_ships,
    pressure, config, H,
) -> LaunchEntries:
    """Pressure-gradient marshalling of leftover ships.

    Moves ships from low-pressure planets toward nearby higher-pressure owned
    planets, capped by ``safe_drain`` (minus what attacks already drew), only when
    the destination is materially more stressed, reachable within
    ``max_regroup_time``, and **still owned at the fleet's arrival turn**.
    """
    P = obs.P
    device = obs.device
    dtype = original_ships.dtype
    pid = int(obs.player_id)
    min_send = float(config.min_ships_to_launch)

    src_mask = obs.owned & obs.alive & (leftover >= min_send)
    if not bool(src_mask.any()):
        return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_regroup_sources_per_lane), P))
    src_idx, src_exists = _candidate_indices(leftover, src_mask, S_cap)          # rank by leftover
    S = int(src_idx.shape[0])
    leftover_s = leftover[src_idx.clamp(0, P - 1)]
    orig_s = original_ships[src_idx.clamp(0, P - 1)]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain_s = safe_drain(
        garrison_status, source_idx=src_idx, source_ships=orig_s,
        H_eff=H_eff, player_id=pid,
    )
    committed_s = (orig_s - leftover_s).clamp(min=0.0)
    regroup_cap = torch.minimum(leftover_s, (drain_s - committed_s).clamp(min=0.0)).floor()
    can_send = src_exists & (regroup_cap >= min_send)
    if not bool(can_send.any()):
        return _empty_entries(device, dtype)

    # Destinations are owned, alive, non-comet planets (do-nothing projection).
    dst_mask = obs.owned & obs.alive
    comet = is_comet_planet(obs_tensors, P, device)
    if comet is not None:
        dst_mask = dst_mask & ~comet
    T_cap = max(1, min(int(config.max_regroup_targets_per_source), P))
    dst_idx, dst_exists = _candidate_indices(pressure, dst_mask, T_cap)          # rank by pressure
    T = int(dst_idx.shape[0])

    # Fixed-size regroup aim via the continuous-intercept aimer (sub-turn lead + a
    # swept first-contact body screen on an AABB-culled shortlist).
    # Strict-superset reachability precheck defers the body screen to destinations a
    # source can reach within max_regroup_time (bit-identical to the ungated path).
    regroup_active = reachable_mask(
        movement, source_idx=src_idx, target_idx=dst_idx,
        fleet_sizes=regroup_cap.view(S, 1, 1).expand(S, T, 1),
        eta_cap=torch.full((T,), float(config.max_regroup_time), device=device),
    ).squeeze(-1)                                                                # [S, T]
    aim = intercept_angle(
        movement,
        src_idx.unsqueeze(1),                                                    # [S, 1]
        dst_idx.unsqueeze(0),                                                     # [1, T]
        regroup_cap.unsqueeze(1),                                                 # [S, 1]
        active=regroup_active,
    )
    angle = aim["angle"]                                                         # [S, T]
    eta = aim["eta"]
    viable = aim["viable"]

    src_pres = pressure[src_idx.clamp(0, P - 1)].view(S, 1)
    dst_pres = pressure[dst_idx.clamp(0, P - 1)].view(1, T)
    gap = dst_pres - src_pres                                                    # [S, T]

    # arrival-turn ownership check: dst must still be mine at k = ceil(eta).
    owner = garrison_status.owner                                               # [P, H+1]
    H_axis = int(owner.shape[-1])
    dst_owner = owner[dst_idx.clamp(0, P - 1)]                                  # [T, H+1]
    k = torch.ceil(eta).clamp(min=0, max=H_axis - 1).to(torch.long)             # [S, T]
    owner_at_k = dst_owner.unsqueeze(0).expand(S, T, H_axis).gather(-1, k.unsqueeze(-1)).squeeze(-1)
    still_mine = owner_at_k == pid

    src_neq_dst = src_idx.view(S, 1) != dst_idx.view(1, T)
    valid = (
        viable & still_mine & src_neq_dst
        & (gap > float(config.regroup_pressure_delta_min))
        & (eta <= float(config.max_regroup_time))
        & can_send.view(S, 1) & dst_exists.view(1, T)
    )
    sc = torch.where(
        valid,
        gap - float(config.regroup_time_penalty_weight) * eta,
        torch.full_like(gap, float("-inf")),
    )
    best_t = _stable_argmax(sc)                                                  # [S] device-stable
    best_score = sc.gather(-1, best_t.unsqueeze(-1)).squeeze(-1)                 # [S]
    best_valid = torch.isfinite(best_score)
    s_ar = torch.arange(S, device=device)
    best_dst = dst_idx[best_t]                                                   # [S]
    best_angle = angle[s_ar, best_t]
    best_eta = eta[s_ar, best_t]

    return LaunchEntries(
        source_slots=src_idx,
        target_slots=best_dst,
        ships=torch.where(best_valid, regroup_cap, torch.zeros_like(regroup_cap)),
        angle=torch.where(best_valid, best_angle, torch.zeros_like(best_angle)),
        eta=torch.where(best_valid, best_eta, torch.ones_like(best_eta)),
        valid=best_valid,
    )


def _empty_entries(device: torch.device, dtype: torch.dtype) -> LaunchEntries:
    z = torch.zeros(0, dtype=dtype, device=device)
    zl = torch.zeros(0, dtype=torch.long, device=device)
    return LaunchEntries(
        source_slots=zl, target_slots=zl, ships=z, angle=z, eta=z,
        valid=torch.zeros(0, dtype=torch.bool, device=device),
    )


def entries_to_sparse_payload(entries: LaunchEntries, *, planet_ids: Tensor) -> dict[str, Tensor]:
    """Convert a LaunchEntries table to the sparse action-row payload."""
    L = entries.source_slots.shape[0]
    device = entries.source_slots.device
    P = int(planet_ids.shape[0])
    valid_long = entries.valid.to(torch.int64)
    counts = valid_long.sum().to(torch.int32)
    max_count = int(counts.item())
    out_from = torch.full((max_count,), -1, dtype=torch.int32, device=device)
    out_angle = torch.zeros((max_count,), dtype=torch.float32, device=device)
    out_ships = torch.zeros((max_count,), dtype=torch.float32, device=device)
    if max_count == 0:
        return {"from_planet_id": out_from, "angle": out_angle, "num_ships": out_ships, "counts": counts}
    safe_src = entries.source_slots.clamp(min=0, max=max(P - 1, 0))
    from_pid_full = planet_ids[safe_src].to(torch.int32)
    launch_rank = valid_long.cumsum(0) - valid_long
    l_idx = torch.where(entries.valid)[0]
    pos = launch_rank[l_idx]
    out_from[pos] = from_pid_full[l_idx]
    out_angle[pos] = entries.angle[l_idx].to(torch.float32)
    out_ships[pos] = entries.ships[l_idx].to(torch.float32)
    return {"from_planet_id": out_from, "angle": out_angle, "num_ships": out_ships, "counts": counts}


def empty_action_row(device: torch.device) -> dict[str, Tensor]:
    """Sparse launch payload with zero launches."""
    return {
        "from_planet_id": torch.full((0,), -1, dtype=torch.int32, device=device),
        "angle": torch.zeros((0,), dtype=torch.float32, device=device),
        "num_ships": torch.zeros((0,), dtype=torch.float32, device=device),
        "counts": torch.zeros((), dtype=torch.int32, device=device),
    }


def safe_drain(
    garrison_status: PlanetGarrisonStatus,
    *,
    source_idx: Tensor,            # [S] long — planet slots to evaluate
    source_ships: Tensor,          # [S] float — current garrison at those slots
    H_eff: Tensor,                 # scalar float — horizon to protect the source over
    player_id: int = 0,
) -> Tensor:
    """Max ships a source can shed while staying held over ``H_eff``. ``[S]``.

    Closed form, no scoring. For every source slot, over the turns ``t = 1..H``
    where the do-nothing projection still has us holding the planet (``owner == me``,
    ``ships > 0``) within ``H_eff``, the largest amount we can remove now while the
    projected garrison stays non-negative on every such turn is
    ``min_t(ships_traj[t])`` — leaving the planet at 0 ships on the worst held turn
    is allowed. Capped by ``source_ships`` (can't send more than we hold now):

        safe_drain = clamp(min(min_t held(ships_traj), source_ships), 0)

    A *doomed* source (no turn is held within ``H_eff``) has nothing to protect:
    ``min_slack`` is ``+inf`` and the cap collapses to ``source_ships`` naturally.
    """
    S = source_idx.shape[0]
    ships_cache = garrison_status.ships
    dtype = ships_cache.dtype if ships_cache.is_floating_point() else torch.float32
    device = ships_cache.device

    H_axis = int(ships_cache.shape[-1])
    H = max(H_axis - 1, 0)
    P = int(ships_cache.shape[0])
    if H == 0:
        return torch.zeros(S, dtype=dtype, device=device)

    src_idx_safe = source_idx.clamp(min=0, max=max(P - 1, 0))

    src_ships_traj = ships_cache[src_idx_safe][..., 1:].to(dtype=dtype)          # [S, H]
    src_owner_traj = garrison_status.owner[src_idx_safe][..., 1:]                 # [S, H]
    me_owned = src_owner_traj == int(player_id)

    turn_grid = torch.arange(1, H + 1, device=device, dtype=dtype).view(1, H)
    within_horizon = turn_grid <= H_eff                                          # H_eff scalar

    held = me_owned & within_horizon & (src_ships_traj > 0.0)
    inf_fill = torch.full_like(src_ships_traj, float("inf"))
    cap_traj = torch.where(held, src_ships_traj, inf_fill)
    min_slack = cap_traj.min(dim=-1).values                                       # [S]
    return torch.minimum(min_slack, source_ships.to(dtype)).clamp(min=0.0)


# === orbit_lite.opp_projection ===
"""Producer-mirror opponent projection for the producer_plus scorer.

`predict_opp_launches_via_mirror` runs Producer's own planner once per
opponent seat (with ``background=None`` to avoid recursion) and returns
the launches that planner would fire this turn as a padded `LaunchSet`,
ready to inject as background launches in our scorer.

The earlier ROI-greedy projector (ported from
``lib/joint_solver/opp_projection``) modeled the wrong agent: ROI-greedy
target selection, ``0.7 * budget`` send size, and up to 3 launches per
source over 8 ticks. The real opponent distribution is dominated by the
public Producer agent itself, whose target ranking, send sizes, and
launch counts differ materially. Using Producer's own planner as the
opponent model tracks the real opponent automatically.
"""

import torch
from torch import Tensor






# Padded L axis for the projected opp LaunchSet. Producer typically fires
# 0-3 launches per turn per seat; 24 slots is generous headroom.
MAX_L_OPP = 24


def _pack_records_to_launch_set(
    records: list[tuple[int, int, float, float, int]],
    *,
    pad_to: int,
    default_opp_id: int,
    dtype: torch.dtype,
    device: torch.device,
) -> LaunchSet:
    """Pack ``(src_slot, tgt_slot, ships, eta, opp_id)`` records into a
    padded `LaunchSet[pad_to]`. Unused slots have ``valid=False``."""
    L = max(int(pad_to), 0)
    src = torch.zeros(L, dtype=torch.long, device=device)
    tgt = torch.zeros(L, dtype=torch.long, device=device)
    ships = torch.zeros(L, dtype=dtype, device=device)
    eta = torch.ones(L, dtype=dtype, device=device)
    owner = torch.full((L,), int(default_opp_id), dtype=torch.long, device=device)
    valid = torch.zeros(L, dtype=torch.bool, device=device)
    n = min(len(records), L)
    for i in range(n):
        s, t, sh, et, op = records[i]
        src[i] = int(s)
        tgt[i] = int(t)
        ships[i] = float(sh)
        eta[i] = float(et)
        owner[i] = int(op)
        valid[i] = True
    return LaunchSet(
        source_slots=src, target_slots=tgt, ships=ships,
        eta=eta, owner=owner, valid=valid,
    )


def predict_opp_launches_via_mirror(
    *,
    plan_fn,
    obs_tensors: dict,
    movement: PlanetMovement,
    cache,
    garrison_status: PlanetGarrisonStatus,
    prod: Tensor,
    alive_by_step: Tensor,
    opp_ids: list[int],
    config,
    player_count: int,
    K_eta_override: int | None = None,
    pad_to: int = MAX_L_OPP,
    K: int = 1,
    H: int | None = None,
    base_background: LaunchSet | None = None,
) -> LaunchSet:
    """For each opponent seat, run ``plan_fn`` (Producer's planner) with
    the seat swapped to their POV.

    With ``K=1`` (default), passes ``background=base_background`` (default
    None: one-step best response, opp assumes we do nothing this turn —
    byte-identical to the original single-pass behaviour). Passing OUR OWN
    chosen launches as ``base_background`` instead yields each opponent's
    predicted REPLY to our plan (response projection, K=1 only).

    With ``K>1``, runs K successive projection rounds. Round k passes
    ``background=cumulative_bg`` (the union of all previously-projected
    opp launches, with absolute-frame etas) so opp's planner accounts for
    its own prior commitments via its internal forward-simulator. Each
    round's launches have etas shifted by ``+k`` turns before being
    appended to the cumulative records — round k is conceptually opp's
    decision at game-tick ``k``, so an eta of ``e`` from round k lands at
    absolute tick ``k + e``. Launches whose shifted eta exceeds ``H``
    (the scorer horizon) are dropped — invalid arrivals are worse than
    no arrivals.

    ``plan_fn`` is passed as a callback (rather than imported) so this
    module has no cross-package dependency -- works the same in the source
    tree and in the bundled submission.
    """
    device = obs_tensors["planets"].device
    # Ships dtype matches obs.ships used inside plan_lite_waves; obs.ships is
    # derived from obs_tensors["planets"] in parse_obs.
    sample = parse_obs(obs_tensors, player_id=int(opp_ids[0]) if opp_ids else 0)
    dtype = sample.ships.dtype

    if not opp_ids:
        return _pack_records_to_launch_set(
            [], pad_to=pad_to, default_opp_id=0,
            dtype=dtype, device=device,
        )

    K_clamped = max(1, int(K))

    if K_clamped == 1:
        # Single-pass path: bit-identical to the original implementation.
        records: list[tuple[int, int, float, float, int]] = []
        for opp_id in opp_ids:
            opp_id = int(opp_id)
            obs_opp = parse_obs(obs_tensors, player_id=opp_id)
            opp_entries = plan_fn(
                movement=movement,
                obs=obs_opp,
                obs_tensors=obs_tensors,
                cache=cache,
                garrison_status=garrison_status,
                prod=prod,
                alive_by_step=alive_by_step,
                config=config,
                player_count=int(player_count),
                K_eta_override=K_eta_override,
                background=base_background,
                # Disable force-concentration in opp simulation: the rescore
                # closure's per-wave score_candidates blows up when
                # K_opp x num_opps inner planner calls each carry their own
                # rescore. Opp is modeled with the cheap single-wave chooser.
                force_concentration=False,
            )
            # Walk the flat [L] entry table; emit one record per valid slot.
            src_cpu = opp_entries.source_slots.cpu().tolist()
            tgt_cpu = opp_entries.target_slots.cpu().tolist()
            ships_cpu = opp_entries.ships.cpu().tolist()
            eta_cpu = opp_entries.eta.cpu().tolist()
            valid_cpu = opp_entries.valid.cpu().tolist()
            for i in range(len(src_cpu)):
                if not bool(valid_cpu[i]):
                    continue
                records.append((
                    int(src_cpu[i]),
                    int(tgt_cpu[i]),
                    float(ships_cpu[i]),
                    float(eta_cpu[i]),
                    opp_id,
                ))
                if len(records) >= int(pad_to):
                    break
            if len(records) >= int(pad_to):
                break

        return _pack_records_to_launch_set(
            records, pad_to=pad_to,
            default_opp_id=int(opp_ids[0]),
            dtype=dtype, device=device,
        )

    # K > 1: multi-round projection.
    #
    # Eta cap = scorer horizon. The garrison status only has entries for
    # ticks <= H, so launches with shifted eta > H would index past the
    # status tensor and be masked-invalid by `sparse_launch_flow_delta`
    # anyway. Drop them at record time so the final LaunchSet doesn't
    # carry useless slots that inflate the L axis. Fallback order: caller
    # H, then K_eta_override, then garrison_status's own horizon (which is
    # H by construction in main.py).
    if H is not None:
        eta_cap = float(H)
    elif K_eta_override is not None:
        eta_cap = float(K_eta_override)
    else:
        eta_cap = float(max(int(garrison_status.ships.shape[-1]) - 1, 0))

    # Cache per-opp parsed observations. obs_tensors does not mutate across
    # rounds (plan_fn touches movement caches, not obs_tensors), so each
    # opp's parsed view is constant for the K rounds. Saves (K-1) *
    # len(opp_ids) parse_obs calls per turn.
    obs_per_opp = {
        int(oid): parse_obs(obs_tensors, player_id=int(oid)) for oid in opp_ids
    }

    # Intermediate cumulative_bg (the LaunchSet handed to opp's next-round
    # planner) is capped at ``pad_to`` to keep the per-round scorer's L axis
    # bounded -- inflating it would multiply wallclock for every opp
    # planner call (the broadcast inside plan_lite_waves concatenates
    # background onto each candidate). The FINAL returned LaunchSet (handed
    # to OUR main planner once per turn) grows to fit all records; that
    # single bigger scorer call costs a few percent vs. K-1 cheaper opp
    # calls.
    records: list[tuple[int, int, float, float, int]] = []
    cumulative_bg: LaunchSet | None = None
    for k in range(K_clamped):
        round_records: list[tuple[int, int, float, float, int]] = []
        for opp_id in opp_ids:
            opp_id = int(opp_id)
            opp_entries = plan_fn(
                movement=movement,
                obs=obs_per_opp[opp_id],
                obs_tensors=obs_tensors,
                cache=cache,
                garrison_status=garrison_status,
                prod=prod,
                alive_by_step=alive_by_step,
                config=config,
                player_count=int(player_count),
                K_eta_override=K_eta_override,
                background=cumulative_bg,
                # See K=1 path: opp planner runs cheap single-wave chooser
                # so multi-tick's K_opp x num_opps inner calls don't compound
                # the rescore closure cost.
                force_concentration=False,
            )
            src_cpu = opp_entries.source_slots.cpu().tolist()
            tgt_cpu = opp_entries.target_slots.cpu().tolist()
            ships_cpu = opp_entries.ships.cpu().tolist()
            eta_cpu = opp_entries.eta.cpu().tolist()
            valid_cpu = opp_entries.valid.cpu().tolist()
            for i in range(len(src_cpu)):
                if not bool(valid_cpu[i]):
                    continue
                shifted_eta = float(eta_cpu[i]) + float(k)
                if shifted_eta > eta_cap:
                    continue
                round_records.append((
                    int(src_cpu[i]),
                    int(tgt_cpu[i]),
                    float(ships_cpu[i]),
                    shifted_eta,
                    opp_id,
                ))
        records.extend(round_records)
        # Rebuild cumulative_bg for round k+1 (skip on the final round).
        # Convert absolute-frame etas to opp's-frame at game-tick k+1:
        # opp's next planner thinks "now = 0" but is actually at game-tick
        # k+1, so a launch with absolute eta E appears to opp as arriving
        # in E - (k+1) ticks. Launches with E <= k+1 have already arrived
        # from opp's POV and must not appear in opp's background.
        if k + 1 < K_clamped:
            next_round = k + 1
            opp_view_records = [
                (s, t, sh, et - float(next_round), op)
                for (s, t, sh, et, op) in records
                if et - float(next_round) > 0.0
            ]
            cumulative_bg = _pack_records_to_launch_set(
                opp_view_records, pad_to=int(pad_to),
                default_opp_id=int(opp_ids[0]),
                dtype=dtype, device=device,
            )

    # Final pad fits all accumulated records (typically <= MAX_L_OPP, but
    # 4P-K3 worst case can produce ~K * n_opps * ~3 ~ 27 records). One
    # bigger main-planner scoring call is acceptable; round-level cost
    # stays bounded by ``pad_to`` via the intermediate cumulative_bg cap
    # above.
    final_pad = max(int(pad_to), len(records))
    return _pack_records_to_launch_set(
        records, pad_to=final_pad,
        default_opp_id=int(opp_ids[0]),
        dtype=dtype, device=device,
    )


# === orbit_lite.recapture ===
"""Recapture-penalty leaf-scorer term for producer_plus.

For each candidate ``c`` that would capture target ``T`` at arrival tick
``e_c`` with ``s_c`` ships, compute a non-negative penalty in ship units
proportional to the opponent's plausible recapture-and-hold ability. The
penalty is subtracted from ``competitive_score`` to discount thin
captures the opponent can punish before the scorer horizon ends.

Composes additively with the existing scorer. The multi-tick opp
projection already debits us for opp launches in its projection window;
to avoid double-counting, the caller passes ``K_opp`` and we restrict
the recapture window to ticks past the projection (``K_recap_eff =
max(1, K_recap - K_opp)``).

Math summary, per candidate ``c`` with target short index ``t``,
absolute target slot ``T``, send size ``s_c``, arrival tick ``e_c``:

    floor_c    = capture_floor_TK[t, e_c - 1]
    captures_c = (s_c >= floor_c) & cand_valid & ~cand_is_def
                 & (we don't already own T at e_c)
    defender_c = max(0, s_c - floor_c)

For each enemy-owned alive planet ``p`` with distance ``d = cross_dist[1, p, T]``:
    reach_p_T  = ceil(d / fleet_speed(ships[p]))                  # turns to recapture
    can_reach  = reach_p_T <= K_recap_eff
    threat_p_T = (1 - safety_reserve) * ships[p] + prod[p] * reach_p_T
    threat_T   = sum_p (threat_p_T where can_reach)

    reach_recap_T = min_p reach_p_T where can_reach   (or +inf if none)
    deficit_c     = max(0, threat_T - defender_c)
    turns_lost_c  = max(0, H - e_c - reach_recap_T)
    penalty_c     = captures_c * (deficit_c > 0) * prod[T] * turns_lost_c

Default OFF behaviour at the caller (see ``_recapture_penalty_enabled``
in ``producer_plus/main.py``); this module is import-safe in the
single-pass byte-identical path.
"""

import torch
from torch import Tensor






def recapture_penalty(
    *,
    obs,
    cache: DistanceCache,
    garrison_status: PlanetGarrisonStatus,
    cand_tgt_slot: Tensor,       # [C] long — absolute planet index
    cand_tgt_short: Tensor,      # [C] long — short index into capture_floor_TK
    cand_send: Tensor,           # [C, L] float — read [:, 0]
    cand_eta: Tensor,            # [C, L] float — read [:, 0]
    cand_valid: Tensor,          # [C] bool
    cand_is_def: Tensor,         # [C] bool — own-planet reinforcement
    capture_floor_TK: Tensor,    # [T, K] from caller (planner_core.capture_floor)
    prod: Tensor,                # [P] float — per-planet production
    H: int,
    K_recap: int,
    K_opp: int,
    safety_reserve: float,
    player_id: int,
) -> Tensor:
    """Return per-candidate recapture penalty in ship units. Shape ``[C]``, all >= 0."""
    device = obs.device
    dtype = obs.ships.dtype
    C = int(cand_send.shape[0])
    P = int(obs.P)

    if C == 0 or P == 0:
        return torch.zeros(C, dtype=dtype, device=device)

    # Device contract: every input tensor must live on the same device as
    # obs. The function gathers from garrison_status.owner directly so a
    # mismatch surfaces as a cryptic CUDA error — assert up front.
    assert garrison_status.owner.device == device, (
        f"garrison_status.owner on {garrison_status.owner.device}, "
        f"expected {device}"
    )

    # Effective recap window: only count ticks past what multi-tick opp_proj
    # already modeled (the scorer already saw those via the background
    # launchset). Floored at 1 — even when multi-tick covers most of the
    # window we keep a single-tick signal so the penalty doesn't fully
    # disappear (collapsing to 0 hides recapture risk against opps whose
    # multi-tick prediction was wrong). Caller already passes
    # ``K_opp >= 0`` via the env getter, so no inner clamp.
    K_recap_eff = max(1, int(K_recap) - int(K_opp))
    K_recap_eff = min(K_recap_eff, int(cache.K))
    if K_recap_eff <= 0:
        return torch.zeros(C, dtype=dtype, device=device)

    # ----- per-(opp_planet, target) reach + threat -----
    # cross_dist at k=1 as immediate cross-time distance (exact for static
    # planets; conservative for orbitals — overestimates distance, so
    # underestimates threat → penalty leans safe).
    d_immediate = cache.cross_dist[1].to(dtype)                # [P_src, P_tgt]
    speeds = fleet_speed(obs.ships.clamp(min=1.0)).to(dtype)   # [P]
    # Reach time = ceil(distance / speed) per (src=p, tgt). Pre-ceil floor
    # at 1.0 prevents tiny distances rounding to 0; ceil(>=1.0) is >= 1 so
    # no second clamp needed.
    reach_pT_f = (d_immediate / speeds.view(P, 1).clamp(min=1e-6)).clamp(min=1.0)
    reach_pT = torch.ceil(reach_pT_f)                           # [P_src, P_tgt]

    enemy_alive = (obs.is_enemy & obs.alive).to(device)         # [P]
    self_mask = torch.eye(P, dtype=torch.bool, device=device)
    can_reach = (
        (reach_pT <= float(K_recap_eff))
        & enemy_alive.view(P, 1)
        & obs.alive.view(1, P).to(device)
        & ~self_mask
    )                                                            # [P_src, P_tgt]

    ships_f = obs.ships.to(dtype)
    prod_f = prod.to(dtype)
    safety = float(max(0.0, min(1.0, safety_reserve)))
    # threat = (1 - safety_reserve) * ships + prod * reach_time, masked.
    threat_pT_raw = (
        (1.0 - safety) * ships_f.view(P, 1)
        + prod_f.view(P, 1) * reach_pT
    )                                                            # [P_src, P_tgt]
    threat_pT = torch.where(can_reach, threat_pT_raw, torch.zeros_like(threat_pT_raw))
    threat_P = threat_pT.sum(dim=0)                              # [P] per planet

    inf_v = torch.full_like(reach_pT, float("inf"))
    reach_pT_masked = torch.where(can_reach, reach_pT, inf_v)
    reach_recap_P = reach_pT_masked.amin(dim=0)                  # [P]

    # ----- gather to per-candidate target -----
    # cand_tgt_short indexes into target_idx; threat / reach are per-planet,
    # so resolve via the absolute slot.
    tgt_abs = cand_tgt_slot.clamp(0, P - 1).to(device)           # [C]
    threat_c = threat_P[tgt_abs]                                 # [C]
    reach_recap_c = reach_recap_P[tgt_abs]                       # [C]
    prod_c = prod_f[tgt_abs]                                     # [C]

    # ----- per-candidate floor + capture detection -----
    K_floor = int(capture_floor_TK.shape[-1])
    if K_floor <= 0:
        return torch.zeros(C, dtype=dtype, device=device)
    e_send = cand_send[:, 0].to(dtype)                            # [C] ships
    e_eta = cand_eta[:, 0].to(dtype)
    e_idx = (torch.ceil(e_eta).long() - 1).clamp(0, K_floor - 1)  # [C] eta -> K index

    tshort = cand_tgt_short.clamp(0, max(int(capture_floor_TK.shape[0]) - 1, 0)).to(device)
    floor_c = capture_floor_TK[tshort, e_idx]                     # [C]
    defender_c = (e_send - floor_c).clamp(min=0.0)                # [C]

    # We capture only if send >= floor AND we don't already own at arrival tick.
    # garrison_status.owner is the do-nothing trajectory; if we already own T
    # at e_c there's no capture happening (it's reinforcement, handled by
    # cand_is_def). Be defensive and gate on it explicitly.
    owner_axis_H = int(garrison_status.owner.shape[-1])
    own_idx = (torch.ceil(e_eta).long()).clamp(0, max(owner_axis_H - 1, 0))
    own_at_arrival = (
        garrison_status.owner[tgt_abs, own_idx] == int(player_id)
    )                                                              # [C]

    captures_c = (
        (e_send >= floor_c)
        & cand_valid.to(device)
        & ~cand_is_def.to(device)
        & ~own_at_arrival
    )                                                              # [C]

    # ----- penalty -----
    deficit_c = (threat_c - defender_c).clamp(min=0.0)
    # When no enemy can reach, reach_recap_c is +inf so H - e - +inf = -inf,
    # which the clamp(min=0) maps to 0. No separate isfinite guard needed.
    e_eta_ceil = torch.ceil(e_eta)
    turns_lost_c = (float(H) - e_eta_ceil - reach_recap_c).clamp(min=0.0)

    deficit_signal = (deficit_c > 0).to(dtype)
    penalty_c = captures_c.to(dtype) * deficit_signal * prod_c * turns_lost_c
    return penalty_c.clamp(min=0.0)


# === orbit_lite.strategic_value ===
"""Long-term production value bonuses for the producer_plus scorer.

Two leaf-scorer terms that add per-candidate bonuses (in ship units)
reflecting production value past the scorer's H-tick horizon:

1. ``denial_bonus`` — opp-aware. Rewards captures of targets the
   opponent values (currently owns OR predicted to attack via
   opp_proj's background ``LaunchSet``). Encodes the intuition that
   blocking the opponent's biggest bet is itself a winning move.

2. ``opening_bonus`` — opp-agnostic. Rewards captures during the early
   game phase, when the H=18 scorer most under-values compounded
   production from a long-held planet. Linearly decays from full at
   step 0 to zero at the configured opening window.

Both terms ADD to the candidate score (recapture_penalty SUBTRACTS).
Both gated default-OFF; byte-identical when the gates are unset.

Math, per candidate ``c`` capturing target ``T`` with ``s_c`` ships at
arrival tick ``e_c``:

    future_h   = max(0, game_length_est - current_step - H)
    captures_c = (s_c >= capture_floor_TK[t, e_c-1]) & cand_valid
                 & ~cand_is_def & ~own_already_at_e_c

    # Denial:
    opp_values_T = (opp_owns_T & alive) | (sum_of_opp_proj_ships_at_T > 0)
    denial_c     = captures_c & opp_values_T
    denial_bonus = denial_c * prod[T] * future_h * weight

    # Opening:
    phase = max(0, 1 - current_step / opening_window)
    opening_bonus = captures_c * phase * prod[T] * future_h * weight

The shared ``_compute_captures()`` helper centralizes the capture-gate
logic so both bonuses (and recapture_penalty in a future refactor)
agree on what "we actually capture" means.
"""

import torch
from torch import Tensor





def _future_value_horizon(current_step: int, H: int, game_length_est: int) -> int:
    """Estimated turns of production beyond the scorer's H-tick window.

    The scorer already values production over ticks [1, H]. The bonuses
    here value the *additional* compound production from holding the
    planet for the rest of the game. Default ``game_length_est=200`` is
    a rough average orbit-wars game length; tune via env knob.
    """
    return max(0, int(game_length_est) - int(current_step) - int(H))


def _compute_captures(
    *,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_valid: Tensor,
    cand_is_def: Tensor,
    cand_tgt_slot: Tensor,
    cand_tgt_short: Tensor,
    capture_floor_TK: Tensor,
    garrison_status: PlanetGarrisonStatus,
    player_id: int,
    P: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor] | None:
    """Return ``(captures_c [C] bool, tgt_abs [C] long)`` or ``None`` if
    the shortlist is empty (no candidates can be captures).

    A candidate "captures" iff:
    - it sends >= the per-target capture_floor at the arrival tick
    - it is marked valid
    - it is not a defensive (own-planet) reinforcement
    - the post-do-nothing trajectory does not already show us owning the
      target at the arrival tick (else it's a reinforcement, not a capture)
    """
    K_floor = int(capture_floor_TK.shape[-1])
    T_floor = int(capture_floor_TK.shape[0])
    if K_floor <= 0 or T_floor <= 0:
        return None
    e_send = cand_send[:, 0].to(dtype)
    e_eta = cand_eta[:, 0].to(dtype)
    e_idx = (torch.ceil(e_eta).long() - 1).clamp(0, K_floor - 1)
    tshort = cand_tgt_short.clamp(0, T_floor - 1).to(device)
    floor_c = capture_floor_TK[tshort, e_idx]

    tgt_abs = cand_tgt_slot.clamp(0, P - 1).to(device)
    owner_axis_H = int(garrison_status.owner.shape[-1])
    own_idx = torch.ceil(e_eta).long().clamp(0, max(owner_axis_H - 1, 0))
    own_at_arrival = (
        garrison_status.owner[tgt_abs, own_idx] == int(player_id)
    )

    captures_c = (
        (e_send >= floor_c)
        & cand_valid.to(device)
        & ~cand_is_def.to(device)
        & ~own_at_arrival
    )
    return captures_c, tgt_abs


def denial_bonus(
    *,
    obs,
    background: LaunchSet | None,
    cand_tgt_slot: Tensor,
    cand_tgt_short: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_valid: Tensor,
    cand_is_def: Tensor,
    capture_floor_TK: Tensor,
    prod: Tensor,
    garrison_status: PlanetGarrisonStatus,
    H: int,
    current_step: int,
    game_length_est: int,
    weight: float,
    player_id: int,
) -> Tensor:
    """Return ``[C]`` non-negative denial bonus in ship units.

    Triggers when (a) we actually capture target ``T`` and (b) opp values
    ``T`` (owns it OR opp_proj's background LaunchSet contains a launch
    targeting it). The bonus reflects the production we deny opp by
    getting there first, summed over the post-horizon game length.
    """
    device = obs.device
    dtype = obs.ships.dtype
    C = int(cand_send.shape[0])
    P = int(obs.P)
    if C == 0 or P == 0 or weight <= 0.0:
        return torch.zeros(C, dtype=dtype, device=device)

    future_h = _future_value_horizon(current_step, H, game_length_est)
    if future_h <= 0:
        return torch.zeros(C, dtype=dtype, device=device)

    capture_info = _compute_captures(
        cand_send=cand_send, cand_eta=cand_eta,
        cand_valid=cand_valid, cand_is_def=cand_is_def,
        cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        capture_floor_TK=capture_floor_TK,
        garrison_status=garrison_status,
        player_id=player_id, P=P, device=device, dtype=dtype,
    )
    if capture_info is None:
        return torch.zeros(C, dtype=dtype, device=device)
    captures_c, tgt_abs = capture_info

    # opp_values_T[P]: opp currently owns it, OR opp_proj predicted at
    # least one launch targeting it. The "owns" arm covers attacks on
    # enemy planets (denying their production); the "intent" arm covers
    # races for neutrals opp planned to expand to.
    opp_owned_alive = (obs.is_enemy & obs.alive).to(device)
    opp_intent = torch.zeros(P, dtype=dtype, device=device)
    if background is not None and int(background.source_slots.shape[-1]) > 0:
        bg_valid = background.valid.to(device)
        bg_tgt = background.target_slots.clamp(0, P - 1).to(device)
        bg_ships = background.ships.to(device=device, dtype=dtype)
        ships_masked = torch.where(bg_valid, bg_ships, torch.zeros_like(bg_ships))
        opp_intent.scatter_add_(0, bg_tgt, ships_masked)
    opp_values_T = opp_owned_alive | (opp_intent > 0)               # [P] bool

    opp_values_c = opp_values_T[tgt_abs].to(dtype)                  # [C]
    prod_c = prod.to(device=device, dtype=dtype)[tgt_abs]           # [C]

    bonus = (
        captures_c.to(dtype)
        * opp_values_c
        * float(weight)
        * prod_c
        * float(future_h)
    )
    return bonus.clamp(min=0.0)


def opening_bonus(
    *,
    obs,
    cand_tgt_slot: Tensor,
    cand_tgt_short: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_valid: Tensor,
    cand_is_def: Tensor,
    capture_floor_TK: Tensor,
    prod: Tensor,
    garrison_status: PlanetGarrisonStatus,
    H: int,
    current_step: int,
    game_length_est: int,
    opening_window: int,
    weight: float,
    player_id: int,
) -> Tensor:
    """Return ``[C]`` non-negative opening-phase bonus in ship units.

    Linearly decays from a full ``weight × prod × future_horizon`` bonus
    at step 0 to zero at ``opening_window`` (default 30). Opp-agnostic:
    encodes the H-too-short defect at the opening, when long-held
    captures compound the most.
    """
    device = obs.device
    dtype = obs.ships.dtype
    C = int(cand_send.shape[0])
    P = int(obs.P)
    if C == 0 or P == 0 or weight <= 0.0:
        return torch.zeros(C, dtype=dtype, device=device)

    if opening_window <= 0:
        return torch.zeros(C, dtype=dtype, device=device)
    phase_factor = max(0.0, 1.0 - float(current_step) / float(opening_window))
    if phase_factor <= 0.0:
        return torch.zeros(C, dtype=dtype, device=device)

    future_h = _future_value_horizon(current_step, H, game_length_est)
    if future_h <= 0:
        return torch.zeros(C, dtype=dtype, device=device)

    capture_info = _compute_captures(
        cand_send=cand_send, cand_eta=cand_eta,
        cand_valid=cand_valid, cand_is_def=cand_is_def,
        cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        capture_floor_TK=capture_floor_TK,
        garrison_status=garrison_status,
        player_id=player_id, P=P, device=device, dtype=dtype,
    )
    if capture_info is None:
        return torch.zeros(C, dtype=dtype, device=device)
    captures_c, tgt_abs = capture_info

    prod_c = prod.to(device=device, dtype=dtype)[tgt_abs]            # [C]
    bonus = (
        captures_c.to(dtype)
        * float(weight)
        * float(phase_factor)
        * prod_c
        * float(future_h)
    )
    return bonus.clamp(min=0.0)


# === producer_plus.shot_mlp (namespaced) ===
import types as _pp_types
_pp_shot_mlp_mod = _pp_types.ModuleType('shot_mlp')
exec(compile('"""Learned shot-success filter (the May shot-validator MLP, finally trained).\n\nA tiny MLP scores every planned ATTACK wave with P(target still ours 10\nturns after arrival), trained on live-ladder episode outcomes. Waves below\na threshold are dropped before dispatch — reject-only, never proposes.\n\nTrain/serve contract\n--------------------\n`encode_shot_features` is the single feature encoder. The labeling script\n(`scripts/label_shot_outcomes.py`) calls it on raw replay arrays; the\nin-agent veto calls it on the identical positional layout rebuilt from\nParsedObs tensors (obs.py documents that the layouts match the replay\nJSON). Any feature change must bump FEATURE_VERSION and retrain.\n\nGate: PRODUCER_PLUS_SHOT_MLP=<threshold in (0,1)>  (unset/0 = OFF)\n      PRODUCER_PLUS_SHOT_MLP_2P_ONLY=1             (optional)\n\nWeights are baked between the BEGIN/END TRAINED WEIGHTS markers by\n`scripts/train_shot_mlp.py`. If the gate is on but no weights are baked,\nthe filter no-ops and warns once (a mis-built bundle must not lose games\nby raising mid-episode); `tests/test_shot_mlp.py` asserts bundles built\nwith the gate on carry weights.\n"""\n\nfrom __future__ import annotations\n\nimport base64\nimport math\nimport sys\n\nFEATURE_VERSION = 1\nN_FEATURES = 24\nLABEL_BUFFER = 10   # steps after eta to check ownership\n\n# Normalisation constants — must match data/shot_validator/schema.json.\nNORM = {\n    "max_ships": 2000.0,\n    "max_production": 5.0,\n    "max_radius": 3.0,\n    "max_fleet_speed": 6.0,\n    "max_eta": 200.0,\n    "board_diagonal": 141.42,\n    "max_planets": 40.0,\n    "episode_steps": 500.0,\n}\n\n\ndef fleet_speed(ships: float) -> float:\n    """Engine fleet-speed curve (mirrors lib/fleet.py incl. the 1-ship floor\n    and the 1000-ship cap; kept import-free)."""\n    if ships <= 1:\n        return 1.0\n    if ships >= 1000:\n        return 6.0\n    return 1.0 + (6.0 - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5\n\n\ndef encode_shot_features(\n    src_planet, target_planet, ships_sent, distance, eta, fs,\n    all_planets, all_fleets, focal_seat, step,\n):\n    """24-dim feature vector, all values in [0, 1] (ship_diff in [-1, 1]).\n\n    Planet rows are positional 7-tuples (id, owner, x, y, radius, ships,\n    production); fleet rows (id, owner, x, y, angle, from_planet, ships) —\n    the raw replay-JSON layout.\n    """\n    sps_ships = src_planet[5] / NORM["max_ships"]\n    sps_prod = src_planet[6] / NORM["max_production"]\n    sps_rad = src_planet[4] / NORM["max_radius"]\n\n    tgt_ships = target_planet[5] / NORM["max_ships"]\n    tgt_prod = target_planet[6] / NORM["max_production"]\n    tgt_rad = target_planet[4] / NORM["max_radius"]\n\n    tgt_owner = int(target_planet[1])\n    owner_mine = 1.0 if tgt_owner == focal_seat else 0.0\n    owner_neutral = 1.0 if tgt_owner == -1 else 0.0\n    owner_enemy = 1.0 if (tgt_owner != -1 and tgt_owner != focal_seat) else 0.0\n\n    src_garrison = max(1, src_planet[5])\n    shot_ships = min(1.0, ships_sent / NORM["max_ships"])\n    # Logically a launch can\'t exceed the source garrison; cap at 1.0\n    # (records sometimes have stale src.ships from pre-launch state).\n    shot_frac = min(1.0, ships_sent / src_garrison)\n    shot_dist = min(1.0, distance / NORM["board_diagonal"])\n    shot_eta = min(1.0, eta / NORM["max_eta"])\n    shot_fs = min(1.0, fs / NORM["max_fleet_speed"])\n\n    n_allied = 0\n    ship_allied = 0.0\n    n_enemy = 0\n    ship_enemy = 0.0\n    for f in all_fleets:\n        owner = int(f[1])\n        ships = float(f[6])\n        if owner == focal_seat:\n            n_allied += 1\n            ship_allied += ships\n        elif owner != -1:\n            n_enemy += 1\n            ship_enemy += ships\n    in_flight_n_allied = min(1.0, n_allied / NORM["max_planets"])\n    in_flight_n_enemy = min(1.0, n_enemy / NORM["max_planets"])\n    in_flight_ship_allied = min(1.0, ship_allied / NORM["max_ships"])\n    in_flight_ship_enemy = min(1.0, ship_enemy / NORM["max_ships"])\n\n    my_total_ships = sum(p[5] for p in all_planets if int(p[1]) == focal_seat) + ship_allied\n    enemy_total_ships = sum(p[5] for p in all_planets\n                            if int(p[1]) not in (-1, focal_seat)) + ship_enemy\n    # ship_diff is signed in [-1, 1] (clipped). Top-10 games can produce\n    # thousands of total ships; norm chosen to keep the typical\n    # distribution centred without saturating extreme blowouts.\n    ship_diff = max(-1.0, min(1.0,\n        (my_total_ships - enemy_total_ships) / NORM["max_ships"]))\n    my_total_ships_n = min(1.0, my_total_ships / NORM["max_ships"])\n    enemy_total_ships_n = min(1.0, enemy_total_ships / NORM["max_ships"])\n    meta_turn = step / NORM["episode_steps"]\n    my_planet_count = sum(1 for p in all_planets if int(p[1]) == focal_seat)\n    enemy_planet_count = sum(1 for p in all_planets if int(p[1]) not in (-1, focal_seat))\n    my_pc_n = my_planet_count / NORM["max_planets"]\n    enemy_pc_n = enemy_planet_count / NORM["max_planets"]\n\n    return [\n        sps_ships, sps_prod, sps_rad,\n        tgt_ships, tgt_prod, tgt_rad,\n        owner_mine, owner_neutral, owner_enemy,\n        shot_ships, shot_frac, shot_dist, shot_eta, shot_fs,\n        in_flight_n_allied, in_flight_ship_allied,\n        in_flight_n_enemy, in_flight_ship_enemy,\n        meta_turn, my_total_ships_n, enemy_total_ships_n,\n        ship_diff, my_pc_n, enemy_pc_n,\n    ]\n\n\n# === BEGIN TRAINED WEIGHTS (written by scripts/train_shot_mlp.py) ===\nWEIGHTS_B64 = \'vYavPcrsTz5Bw0Q+W2wwP2rinr2pNVE+lvRmPrZohD6rEKA98gTvvT7WeL3OEx0+SkGRvjLT6b2blhK8DGagPcfnEL7jIXG+in/pvg5QZ71Y0/29acMOPoAp0T7KETW9Ec2avs7QgL/Mvq49umY4v7OQ0b1cxMs+ONHAOx6Bn77IaXa9Y/7/O2nihLz7HSu+NrRZPaiBIr2jAAW9cW5DvaU0870CQIM96T34vW5XNL1i/eS6Kspqvepfaj3dhwM+5zD7PICmdj4pQs89CJmWPA9xFzx+uyq9Lj0LPqpK0T0LAJg8QaiIPfDuST6Ceys++50gPfGBJr6uICM+IgSdvW43Lj7TXBI+5yBIPM4y1rx67YA9JAoUPjXasD2Lu5c9H1xVPVhusz2Xqzy+c4jYPEs6Rz40nQo+Md/gPH1TdL1TgcS9+VTbPcfnjD7JhXG+8uilPdLS8D0Ldlk9zbXnvaFxp71PLFS9UwJavVXMp707ymk9qoWXPbbeBz6wt5u+JLLlPmjiYD5dBerAnONBPJyHQ764M8y91+PVO12dFj6b4a4/hDHVPux3Vb65ZGM/XoA+Pz+yxb8o7Qc/rHgOP44mjj6I+mO/+7QtP3E8Kj3uaQDAJJfXPtSMAD4qoNG+PguYPj5MTD6aE28+tjg7P8n6Zj3ZmWs+31E9PvWvYL8gmGc9N8D5PS8Skb0eDI0913gYvghBab0sLIs9EgcbvvU6gD5wdB+6kvXLvcSSzz3s7Ey9j/x6vTOePD6MQFA+edPQvWqqv7350YI+mVQPvpnMjjutZd689mvNu3OOmr0KxLw82bI0vYOCGj5BBIk9vkXEPVjkejsxa388bBpxPurFBL3dVgy9zUTyPnYBBzzR3VW9DuTQPBuaiD0Lbto8PcE5PRF+/b1rhvA9CzVavpqPR70ydpk+/flyPk1P6r1TPas9bDGjPnlWpL1lluE+es38PTdzgb5fo6K+hIJ7PvuMjD3kVb498wpMvg82B76z6Su+iJ7MvTa36D0araE+ZvIavZzFqz2E422+SqezPcnE5D04vDW+fFYVvjdSWL3LRgc+T0HSvDRsAb5CVyK+Ly9xPSxV+z0d86Y9eNWIvvAA1b1efze+k0JavhpM9T2BYQ0+zad5PkZT4T2NZDu+I8jaPClzHD6ZpRe+1l+VvXcL3D3AGmC9XIa7PROU175RT5I9XexWvQYwNz48K+y88T5iPcRcqb1Gp6S9BKraPAzrSr/G6HO8a3hIvrjIhT7vXSU+GE7Mvn/TT759N5s+5x9OvwJiE71fhkU9TZOgPrZJCj3lqNS/7k+TPYRBCj3IPIq7V0tbvm7nzj1JmDg+kC5bPtvUp75zRAI+W00Ev2gRQ774RlG8JmSXvp0oMT4bV7k9Dj1LPH2yMr6Clh6+0U5TPja5nj1OVKC9EtqMvjzsYL740KQ90Pssvhb1Xr7dGne/AXRWvh+Lvz2R0Mu+HSQzvacPTD64xcy9priYPI9F5jxVY5c9iBoOvaPShL6zX3A9PY6svgEaZL7FHja9ZxJYPhb3rz0m82G+QXxRv81Cjz7aELc9N5f4PcAADL88ujc+AuqJvjebFT6iWdU+AXiuvdkexL9D3Kk+a/w4v/xRUb/UhbO9abCivqx0tD6ANGy/hBZRvlNKFz3NKRI+5SAKv6bYgD1E+SQ+CoiQv/7VSD5gSYG+MEkyvn7dYT4nxkA+TquFPvhTRz21AjK9T0BmPACtob1dOKk+CVwwPWBH1LzK9EA9+LEgPBs6V70eKq09uM2SvJH9KT6Lvlg9QjGBPQ8hwz1yFhq+AN5HvuTuXD13WNI7g9qPPk4FFL7UKck8vCgkPoIqVb2NuCg+ZWfrPYvFFT17l788Q4hJPoIYyj1Mw/w+2lJyvrecjj4LkNm++7pTvlnfJb49fNI8FX2iPJZsp73XbSO84+iavoDfCD8bvps+m6g8v8RTBz6s+nq86VMBvscP4L3NEAq+7mSrPhBXMD5JTQ8/WVgivsoDhj3+V5C8tROCPrO3HL63pqM+Sh2wPrvFojwgRPs84AuGvjiY5T0E4Hi/ixLKvgeUjjuVcua9QpWYvq2g9T7/ZM0+eD+qviXUdr37iU++ujhYPvLShj3mtZu/fZrXvTMVWr/3KlW+CLUFPRwG0jtJ2Da9YC2pveLQIb15cdo+SDiTvHvliT08phO+z/CHvsVFFb7aFue9rItvPatg+z010cy9tzgWv3zhWj3nrjU+ELVsPvs2ED4cD24+a3KEvh46rb7PjGe91esOPkhURr4Vt9w9QA23vEaNrD7VNC++RkwcvqrtFr4e24Y+qweUvV4dML5Nc8i9KERvvnt3Ij0wZum9uYIePtcejT777wC/5FvnvUnv3b2ygsI8hV4iviTX9L26ohY/7F6AvaVSGD6ZVwG/SScJP4YE4D6cSQQ/7Gw6PLB63743Y2i+Brdvvt8rLz68/6E+xZ0QvikUYT5zYm8+Iu/ZPsTUFL7Efxo+o+CyvmkOz76RGQw+CoM6PoxX6r12tQc+aN8Hv6LDhT7/0gE+ggwTP8CU+z6N0r4+VjtFvkuRl74PDLg97wmZPcTaNL7YVqQ+Dy6FvQ4DGj4wl7Y9JKWtvR7AnT1SKAs/33yJPhuacr3S1am+wUrpvu/nob3dDSc+qgyIvgKTTL7S1AM+HHUJPwHKQr1Iytc+UMorvo1R2Ty50pi9nf2JvfIUR756xe+9eXqfPmeAv72NeLU9AsAaPs3Rer6KmS085g7oPaOJa73CRtK8z0uCvjxzUL0OgYg97XyUPOPleT36ImC9MwdTPcEWGj1YYzu/o0ThvYaGAz4ey+i94td7PBDq0D2V9Aw933yAvUb22r18R+s99OoyPdEt1T2HMoc++OgRv0cLJ78xMuS9c5rFPqRTb750CWm/hwyAPsr9wT7khMA98mBMPcq56T0DtPu+/bwdP4QFtT6L5oM9AmSUPmnY4b39diK+UTwxvvR5jj7eORY+jsCmO0YKqz6rCtG+Wdn5vjnUTz75XMq+iBjzPUWJsr0ri6K/GswrPsTbBL5tsc2+Uy+BPjuZKD7Tx0K/QYFwPn/4qz7pIzy+KyecPdTZzr7Ux3U+f6VyvstykLzvTJ6+o2iJPjvZGb4KX6c+BkBmuM4YWL7S8Wu+MAeIPuFUJz8p1Yc+vsNYvPd6hD4wlwa/su9NP4vw8bz0nMS9NOiovlRN/L33ps+8u3SEPTYWlD2GTak+3sUDPbhME77OdY29xaWBPieGl76aOjO/k608vtYqCb3YyAC+2MBDPLMYPDzNGUi9/MLYvR8FDD/sI0S/535av6jrmT5C9TU+khnmvjRhhT4u0T+9C5OvPjwhHz5Lj+U9MXrOPtVeRj4wRI6/mmjEvpFcYb3l+VU+oJmZPtejTT6DK9a7NQGPPoBcOD4c+BA8u0vgPSxwvj6bFHM+Q6cAvvHIcD25zZ692HDAvr33iT4ntR++MpRtvnXL27xc/JG+0/bLPTO9Mj0jcSQ+ucv9venxjz4XCKg+g0PJPL6iYr573U0+RwVuvheAfT2yHLS+MgiAPuasGT7YOlk+W2OlPuGsiL7PjrG+hi5FPus1wz5uNlE++cA1v0ejTr/c/4Q91l+CvguuST6ypyg+mFFJvlPCIj7NZNE9jwV+vlYBg75/f7c+0xb2vRDlkr4tCms9o1/TPZjhiz6Mi0u+o/SoPjkUsz5fbpE++21mvquAEr6V0Oi9ewlTvs8xvj58Pni+FgomPuE2qT6yDG69S18Dv26Rxz1NrOg7CWeyvvkLZz51RMw+8sRHPutV0r0LzRm9KABrveKzXr6/9I++T0C8vfpY7zxwFW88R510v0fX5L2/1Ws9QxybvuAa/j5HLL89ueA1vrWhYT6F7bO/awPSPRZRtr4pY2++XO8Jvuw3vL5itLA+PjZPvrUxcD5OhhK/lENPvtka1T3fhCk//bGhvtxz0768MdO+YiEHu0QbFz6Eh3y+JNLYvcgZ2TyD9x0+f3b8PWDTrz6hCcK8QExhvpzLyz4yhSa9aRQlvjlPPb5Ouak+JFfsPU/qmj7cUK494xCyPtNOZD7hGpU+wyA/v66JdL7ci5a+0ZrOPd4E/r00NIq9OtQDPlCnkz1zXZI+Dk4lPetuhj4NEMa8L2hKPtdeHL40O7E917d6vbxgEz1EwWO+3uTqPRTPkL6EW/S9adKPPDhMlDseN4+9fR/SvQCBH763vO89HOYPPgD7ET4XnmI+uCSUvJGs6z2ohZE8FqqQvVMYhj142uU9ImhJPnX7jD4MqAY+l45OPp3HuT3wQg29lcnwvZipir28ems99TuwvQYS6T06MhQ+XqIUP2JU9j2EEa29lfqyOykIy738hw0/bsdcPiUSDz6Mm2w+gsiTPgB3Kb6UQWs9gSYAvlQODz/a5XA9ZvqRvhAEZT/rFdA8WrY9vhnSvb/mKai/ozm6P9mhpL+XK7U/AVeYPlSnmr8/H8w99EGYPzFNtj1np6s/gh6jP1rAr7/3lN0/gJixvzKlxb1BR8c+g2rSPvjk3L7O5qA+tYrVvqPk+T1DIVk+3Yq9PUqMDL5R3qq6YSEGv9ckvb6B/+M+Jra1vRh24z6fHNM9HdUYv19gKL+kHSY/CPu8vlg3zT6nfUC/HzIGv+A03Dwh0gE/mKnoveWsMT8XEyw/CLf1vgvrNT+97rG+nauTPcuWtbxidMk8RFVPPesIHj6SvoE9Gu32vaNzEL3ErjG9X+j9PT+MnryG9Is9Mzl4PuQzGDzgmVo+O/v6PcRVtD04UUu8ffVPPdv6HT66OLe9Jjc4PqIqMb/mY6q948icvKwxkD5MlDW+55eJvdDbBj40VcI8aLOIPsbqE74kLaQ9UtcGvpL+672Kd8s+E4eTvG48W7t4lga+1sGaPoMFGTwsgRQ+FLgGvXiwfT4ar6E+dQlKPe1cCD9SKxU+M7Mqvi457r40x9y+sQrEPizHDr+jfh0/aeSOPgmYsb7tSMw9EHUFPxJ84b1TxKo+BxYVP1bY273EMJg+LI98vmn77r0tgGk+HKclPplqcr5iRzM+aiHuvZuxgL4ERSk+nvUnPT2QTjwGsyO+DYifvgb18L0ERWU+es+9Pel2Zj3KxEq+sy4PPfF+qT3Pclq96UMFvio63L381Q0+B57rvTcPDb7tdMc9jCsjPhwwzj01uoe9wkz7PTgHlr0w6B2+hyUgPno5Gj9TSSw/n182v1MkCz8gZOO+ofBKPeYw7T7aXQy+kmuRviEemz1KLuS+jM3Jvg6bST8V/HS+8SLnPmvgib2DmYQ+pes5Pr+9Dj5Ertc+KL7KvRNmUz6eBdc+Z84Lvjkrj75NNy49CK01vpgosL5OxSY96kBAP7Ie2j7gNjM8FA01PoF6tj44isK99EYHP5IHaL669Cu/fGcOP/2T973VhhC/EJbFPdsIgb5IZnK+mqB7PuYFBD/STs8+Md0JPeEhW71008y9E1xOPf8xpL4afbs+DJOsviVlBb/QkSA+ChEKP1ifKb4LoAQ+dnryPqmpX7sCXGE+W9Y4voTs6bzkhx4+isqpPk6xkbz2wu0+ERoCvsT7zz7tZs4+zVFvvihlsL6l28u9SpRYvQoos76cm0s+Zx9OPjLVxT5G12W+N8Z3v2eDur8ewLQ//K2Mv3W3kT8vhZC+MUxbvyjQ0ryOES8/FLe7vavOjD9b3Cg/Vfmwv86Wgz/9bW2/8vmfvGFL+j2+xhk8AqmFPqiBv70L9gY+4TaTPowLc7s3Nwk+pv7cPcElDL5Lh6A+mnZLvbiuALzQPYQ+UesXvoK6ij28+Ak++dCQPqnkI74sKPM7ltJKPVoogb5zB1Y9EMPIO2JpZD6PL2S93coSvjVaDT1urDU+AXQ7Pi3caz4wBZA9pFagvSLWcr7mio8+i1qmvp3CHz7CiSa/SOosvshWuL3PweU+tRPCPdAdhD6QSIk+15ilvfBHET/9xvm9bv7tvRs/EL5ULJC+Vns/PgFaEb+4zPo+Y5mJvofLGr8YKAm7Zws1P7gE5b2vT8s+pY8VP4+reb7M4vk88EzlviyaIz4PX9q+BXCsvhYzGz8EZZW+f1/mPpz+nj7L//m+5k8TvmKO+z5HoNY8ZCgJPxeVwj5gLPm+ufH7Pts5rr7VFBI8HoZOPKDLLD5uiDQ+PBelPhilA743dyU/JJVJPr9R5r3wEro8AeclPSTcYj48VDC+vstnPirjIT+ob4U+cEhKvskrKz702eM9NmSqvMzP6L3FkG0+FwV3vTTy7bygwjK+lJZ1PmpxAb4IWv+9mWWiPvT1cz66tLI8Mfotvj4Ytr2O8jY/to45P5af375rYCo/62wav7QLgz4ayzc/eko8vsVu0r69H4o92TNFv1OA1b5FVBk/PtPhO8ZlKz+zNbW9dhv/vpWEAr8gWTs/RP8Ovk71AT/xTiS/fq/bvnpNLz5LPsM9uqlovZb28j4uwxc+ijoFv4dMhz9jUOO+uhXJO3/Pkj4WWf283f5MvjmvgD4ojGO9d0qaPjSfhT4cJ7W96BuxvYO6UT1e4Ue+B3ynvZh+kD61/GI+OWIQPocuSL56EFK+H09fvoRX3z5dY0g+wFioPAdrLb5RzOq8CxMevi9xFr49kSM+ShjKPmMHUT12yF6+XFJ9PzlOn7yM/rO9MLaFPpmHDj4bCyA+UkeWPVu8JT4PLZA+cDK9PSFv671/4Vo9rLshPGww0j20dY6918+APtKR6T7ujl0+OX6kPSOsQb9hFeO+NfLbPqAtNb/Avyc/KPfMPkz0Lb+nMxG+VNoGP9uu+L0RtxU/YW/HPosIRb/kGw0/ScIUv2WD+L0/41Y8nrUbPlTpkD6VyTg+qFoLPtEFF77oAyG9bJkbvvCJbj7y8Be+F+lCPsJCRD5pLD0+kX6dPiX0iz6qDCG8xY/7Pa9LFT54No4+dLvKPh8JLr5PLRA/+t3sPnQYI7zrytG+3MEFPjaXAj4yXG295eQBvhq/ST/Zfbo9GmPavSJaAT6UmWk+pDZIPuCWUD33wj0+oQrpvs5XHz7pbju+59B2PSUSVL6tDaY9+i6DPj4MX7y0hJg+KXO1PIpsUL5SXOc8fTv1vuv5Pz695xS+M1buvs6/+r1cBtQ+vNSdPlsvIj64rwc9u6T1vWvsDD7Q1JW+lO24PQ8A4z5K9cM+XKNOPiiKAz9MA9k9sNndvcrU9D4lohu+84p+vO/jl70Wgww/y35dvrpW3r0lMBo+vgZxvcLtgb64Saw+YeZOPq5bVb4WQhc+fkSevfq4Aj0CygQ/HfEmvrFM3b3nPSM9RFHRP8Y8i7+WOGk+qGK2vVX8fr/kiFy9HAWFP85ooD+VjhQ/km7TvtPv+7xNQxu+GTGGvSu8aL0SjAk/zFmQPhcCIr5wBoK+4KkyPKDHUj3Ydyc+IFBuPWqkdTzzb6S9nUYivyk9yD6fwmY7Pu9TvlM6tz0SSZO+fnhAunCRtr0yL3m+2eE0PlRmuDxQBQW9p6Q2vtYuCL52AKo8Ad51PkrVBz3cjQA/IKcBvkTGhb5ofyI/8LFHvi1DXr63dsS9cyUNv2BQxz4miis+8lxhvl5w9j5WAwW9Gws5vh4+nL7CUSa+1yy2vpaDTD6sJTm+85REvgJUQ76RyIc+L8j6PkIoMj5hA9g+Y/odvmEr2D10tOE+0rgXPonp5T3yWU4+HaesPpOLrT1e/3m+eM1gvmeGf76rFxa+nirdPia93z4W6zQ+yP5OPfArS72U0HQ+KkbpPSgKZj73j549UvY2PhuoEb4GA2g+ap3OvDRSKr5Rq/o8PmHBvdVwjT5OO6I9vYeNv3INBD8dRZy+wVEwPl7+8z4p6S2+bcQRv98c6L7ogW4+\'\nWEIGHTS_META = {\'dims\': [24, 32, 16, 8, 1]}\n# === END TRAINED WEIGHTS ===\n\n_MLP_CACHE = None\n_WARNED_NO_WEIGHTS = False\n\n\ndef _decode_weights():\n    """Decode WEIGHTS_B64 into a list of (W, b) numpy pairs, or None."""\n    global _MLP_CACHE\n    if _MLP_CACHE is not None:\n        return _MLP_CACHE\n    if not WEIGHTS_B64:\n        return None\n    import numpy as np\n    raw = np.frombuffer(base64.b64decode(WEIGHTS_B64), dtype=np.float32)\n    dims = WEIGHTS_META["dims"]          # e.g. [24, 32, 16, 8, 1]\n    layers = []\n    off = 0\n    for i in range(len(dims) - 1):\n        n_in, n_out = dims[i], dims[i + 1]\n        w = raw[off:off + n_in * n_out].reshape(n_in, n_out).copy()\n        off += n_in * n_out\n        b = raw[off:off + n_out].copy()\n        off += n_out\n        layers.append((w, b))\n    assert off == raw.size, "weight blob size mismatch vs dims"\n    _MLP_CACHE = layers\n    return layers\n\n\ndef predict_success(features_rows) -> "object":\n    """P(shot succeeds) for each feature row. Returns np.ndarray [N]."""\n    import numpy as np\n    layers = _decode_weights()\n    x = np.asarray(features_rows, dtype=np.float32)\n    if x.ndim == 1:\n        x = x[None, :]\n    for i, (w, b) in enumerate(layers):\n        x = x @ w + b\n        if i < len(layers) - 1:\n            x = np.maximum(x, 0.0)        # ReLU hidden\n    return 1.0 / (1.0 + np.exp(-x[:, 0]))  # sigmoid head\n\n\ndef apply_shot_mlp_veto(entries, *, obs, threshold: float):\n    """Drop valid ATTACK waves with predicted P(success) < threshold.\n\n    ``obs`` is the ParsedObs for the CURRENT (un-debited) observation —\n    feature math mirrors the labeler: current positions, straight-line\n    distance, eta recomputed from the engine speed curve (NOT the\n    planner\'s intercept eta).\n    """\n    global _WARNED_NO_WEIGHTS\n    if _decode_weights() is None:\n        if not _WARNED_NO_WEIGHTS:\n            print("shot_mlp: gate ON but no trained weights baked — no-op",\n                  file=sys.stderr)\n            _WARNED_NO_WEIGHTS = True\n        return entries\n    import torch\n\n    valid = entries.valid\n    if int(valid.sum().item()) == 0:\n        return entries\n    P = int(obs.P)\n    tgt_safe = entries.target_slots.clamp(0, P - 1)\n    is_attack = valid & ~obs.owned[tgt_safe]\n    idx = is_attack.nonzero(as_tuple=True)[0]\n    if int(idx.shape[0]) == 0:\n        return entries\n\n    # Rebuild the replay-JSON positional rows from ParsedObs (layouts match).\n    alive_idx = obs.alive.nonzero(as_tuple=True)[0].tolist()\n    px = obs.x.tolist(); py = obs.y.tolist(); pr = obs.r.tolist()\n    pships = obs.ships.tolist(); pprod = obs.prod.tolist()\n    powner = obs.owner_abs.tolist()\n    planets_rows = [\n        (i, powner[i], px[i], py[i], pr[i], pships[i], pprod[i])\n        for i in alive_idx\n    ]\n    by_slot = {i: row for i, row in zip(alive_idx, planets_rows)}\n    f_alive = obs.f_alive.nonzero(as_tuple=True)[0].tolist()\n    fo = obs.f_owner.tolist(); fs_ = obs.f_ships.tolist()\n    fleets_rows = [(0, fo[i], 0.0, 0.0, 0.0, 0, fs_[i]) for i in f_alive]\n\n    pid = int(obs.player_id)\n    step = float(obs.step.flatten()[0].item())\n    rows = []\n    row_entry = []\n    src_l = entries.source_slots.tolist()\n    tgt_l = entries.target_slots.tolist()\n    ships_l = entries.ships.tolist()\n    for e in idx.tolist():\n        src = by_slot.get(int(src_l[e]))\n        tgt = by_slot.get(int(tgt_l[e]))\n        if src is None or tgt is None:\n            continue\n        n_ships = float(ships_l[e])\n        d = math.hypot(tgt[2] - src[2], tgt[3] - src[3])\n        v = fleet_speed(n_ships)\n        eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0\n        rows.append(encode_shot_features(\n            src, tgt, n_ships, d, eta, v,\n            planets_rows, fleets_rows, pid, step,\n        ))\n        row_entry.append(e)\n    if not rows:\n        return entries\n\n    proba = predict_success(rows)\n    drop = [e for e, p in zip(row_entry, proba) if float(p) < threshold]\n    import os\n    if os.environ.get("PRODUCER_PLUS_SHOT_MLP_DEBUG"):\n        print(f"shot_mlp[t={int(step)}] scored {len(rows)} attack waves, "\n              f"dropped {len(drop)} "\n              f"(p: {\' \'.join(f\'{float(p):.2f}\' for p in proba)})",\n              file=sys.stderr)\n    if not drop:\n        return entries\n    new_valid = entries.valid.clone()\n    new_valid[torch.tensor(drop, dtype=torch.long, device=new_valid.device)] = False\n    import dataclasses\n    return dataclasses.replace(entries, valid=new_valid)\n', 'shot_mlp.py', 'exec'), _pp_shot_mlp_mod.__dict__)
apply_shot_mlp_veto = _pp_shot_mlp_mod.apply_shot_mlp_veto

# === producer_plus.main ===


import dataclasses
import math
import os
import sys
import time
from dataclasses import dataclass


import torch
from torch import Tensor


















# Adaptive candidate-arrival horizon K_eta — ported from champion's
# capture_horizon_k (agents/baseline/launch_rules.py). Default OFF
# preserves bit-identical behaviour vs the untouched producer.
# Clamped to H so capture_floor lookups stay inside garrison_status.
def _adaptive_k_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_ADAPTIVE_K", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Multi-size enumeration per (source, target): emit three ships variants —
# (capture_floor, 2 × capture_floor, safe_drain) — instead of a single
# safe_drain candidate. Default OFF preserves bit-identical single-size
# behaviour. State/MIGRATION_PLAN.md Step 4.
def _multi_size_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_MULTI_SIZE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Multi-source coalitions: in addition to single-source candidates, emit
# L=2 pairs that combine two source planets on the same target with
# (near-)same arrival tick. Producer's planner already handles L>1
# end-to-end; this step fills the unused L axis. Default OFF preserves
# bit-identical single-source behaviour. state/MIGRATION_PLAN.md Step 5.
def _coalitions_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_COALITIONS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Opponent multi-launch projection: once per turn, project the opponents'
# next 8 ticks of launches and inject them as background LaunchSet slots
# into the per-candidate scorer. The scorer's `sparse_launch_flow_delta`
# natively handles mixed-owner LaunchSets via per-launch `owner`, so
# every candidate is now scored against "do my action AND opp does their
# projected actions" rather than "do my action while opp does nothing".
# Default OFF preserves bit-identical static-opp scoring. Migration plan
# Step 3 (redux). See orbit_lite/opp_projection.py.
def _opp_projection_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_OPP_PROJECTION", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Multi-tick opp projection: instead of projecting opp's launches at the
# current tick only, run opp's planner K successive rounds (game-ticks
# 0, 1, ..., K-1) with the cumulative previously-projected opp launches
# passed as ``background`` each round. Each round's launches are
# eta-shifted by +k turns before merging. Default 0/1 preserves the
# single-pass byte-identical behaviour. Player-count-suffixed knobs
# override the common one; 3P games fall back to the _2P suffix (3P is
# untested terrain on this comp; tune via the base var if a 3P-specific
# value is needed). NOTE: multi-tick is silently a no-op when opp_proj
# is OFF (the value is read only inside the opp_proj-gated branch in
# run_turn). Set PRODUCER_PLUS_OPP_PROJECTION=1 to activate. See
# knowledge-base/thoughts/2026-06-05-cycle-stalemate-and-horizon-
# scaling.md for the structural-defect diagnosis.
def _multi_tick_opp_k(player_count: int) -> int:
    suffix = "_4P" if int(player_count) >= 4 else "_2P"
    raw = os.environ.get(f"PRODUCER_PLUS_MULTI_TICK_OPP_K{suffix}")
    if raw is None:
        raw = os.environ.get("PRODUCER_PLUS_MULTI_TICK_OPP_K", "0")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


# Recapture penalty: per-candidate leaf-scorer discount for thin captures
# the opponent can plausibly recapture. Composes additively with
# competitive_score; the term is in ship units so the weight is a pure
# multiplier. Multi-tick opp_proj already debits for opp launches inside
# its projection window; to avoid double-counting we clip the recapture
# window via K_recap_eff = max(1, K_recap - K_opp). Default OFF preserves
# byte-identical static behaviour. See knowledge-base/thoughts/
# 2026-06-05-cycle-stalemate-and-horizon-scaling.md for motivation.
def _recapture_penalty_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_RECAPTURE_PENALTY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _recapture_penalty_weight() -> float:
    raw = os.environ.get("PRODUCER_PLUS_RECAPTURE_PENALTY_WEIGHT", "1.0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 1.0


def _recapture_k(player_count: int) -> int:
    suffix = "_4P" if int(player_count) >= 4 else "_2P"
    raw = os.environ.get(f"PRODUCER_PLUS_RECAPTURE_K{suffix}")
    if raw is None:
        raw = os.environ.get("PRODUCER_PLUS_RECAPTURE_K", "8")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 8


def _recapture_safety_reserve() -> float:
    raw = os.environ.get("PRODUCER_PLUS_RECAPTURE_SAFETY_RESERVE", "0.5")
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return 0.5


# Strategic-value bonuses: per-candidate scorer credits for the
# production we'd accrue past the scorer horizon (H=18 in 2P / 13 in 4P).
#
# Two opt-in mechanisms, each in ship units so the weight is a pure
# multiplier:
#   denial_bonus  — captures of targets the opponent values (currently
#                   owns OR predicted to attack via opp_proj's
#                   background LaunchSet). Encodes "block the opponent's
#                   biggest bet."  Opp-aware: depends on opp_proj.
#   opening_bonus — captures during the early-game phase, linearly
#                   decaying to zero at ``opening_window``. Opp-agnostic.
# Both default OFF preserves byte-identical static behaviour. Share the
# game-length estimate knob (``PRODUCER_PLUS_GAME_LENGTH_EST``).
# --- Holding-time-priced capture credit (PRODUCER_PLUS_HOLD_VALUE) -------------
# The decision-trace finding (Gregor Lied loss): the in-horizon flow scorer
# truncates capture payoffs at H, so every expansion scores ~0 against the
# fire threshold and the agent banks instead (paralysis). A FLAT terminal
# credit (TERMINAL_PROD_VALUE=12) was refuted on both referee classes — it
# rewards expansion the opponent punishes before payback. This version
# credits post-horizon production ONLY for captures the opponent cannot
# feasibly retake inside the lookahead: project the captured garrison
# (survivors + production) against the enemy's FULL routable mass at every
# later tick; any deficit ⇒ no credit. Safe rear expansions unlock;
# contested grabs stay priced by raw flow. Default 0 = byte-identical.


# --- Source-safety drain cap (PRODUCER_PLUS_SOURCE_SAFETY) ----------------------
# The economy-credit refutation chain (3 mirror routs, all decided ~step 29)
# localized the true blindspot: ``safe_drain`` caps drain by the DO-NOTHING
# projection (in-flight fleets + production), so the enemy's uncommitted
# standing reserve is invisible — the punisher simply strikes whichever home
# planet the expander just thinned. Symmetric counterpart of the reactive
# capture floor, for SOURCES: a source may shed only what keeps it able to
# survive the enemy's routable mass at every tick of the window, crediting
# its own production growth and friendly garrisons that can route help in
# time:  drain ≤ g_s + min_k( prod_s·k + help(s,k) − w·threat(s,k) ).
# Default 0 = byte-identical.


def _source_safety_weight() -> float:
    raw = os.environ.get("PRODUCER_PLUS_SOURCE_SAFETY", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _source_safety_lag() -> float:
    raw = os.environ.get("PRODUCER_PLUS_SOURCE_SAFETY_LAG", "0")
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _friendly_support_margin(
    obs, cache, source_idx: Tensor, K: int, *, lag: float = 0.0,
):
    """``[S, K]`` friendly garrison mass routable to each source by tick k.

    Mirror of ``_reactive_reinforcement_margin`` over OUR planets. Each
    helper keeps 1 ship (can't fully strip a held planet), and the source
    itself is excluded — its own garrison is already the defender.
    """
    mine = obs.owned & obs.alive
    q_idx = mine.nonzero(as_tuple=True)[0]
    Q = int(q_idx.shape[0])
    S = int(source_idx.shape[0])
    if Q == 0 or S == 0 or K <= 0:
        return None
    dtype = obs.ships.dtype
    g_q = (obs.ships[q_idx].to(dtype) - 1.0).clamp(min=0.0)          # [Q]
    speed_q = fleet_speed(g_q.clamp(min=1.0))                        # [Q]
    d = cache.cross_dist[0][q_idx][:, source_idx.clamp(0, int(obs.P) - 1)]  # [Q, S]
    eta_qs = d / speed_q.unsqueeze(-1)                               # [Q, S]
    k_grid = torch.arange(1, K + 1, device=obs.device, dtype=dtype)  # [K]
    reach = eta_qs.unsqueeze(-1) <= (k_grid.view(1, 1, K) - float(lag))
    self_mask = q_idx.view(Q, 1) == source_idx.view(1, S)
    reach = reach & ~self_mask.unsqueeze(-1)
    return (g_q.view(Q, 1, 1) * reach.to(dtype)).sum(dim=0)          # [S, K]


def _source_safety_allowance(
    obs, cache, *, source_idx: Tensor, prod: Tensor, K: int,
):
    """``[S]`` max drain that keeps each source locally defensible, or None.

    None means no constraint applies (gate off, no enemies, or empty window).
    """
    w = _source_safety_weight()
    S = int(source_idx.shape[0])
    if w <= 0.0 or S == 0 or K <= 0:
        return None
    threat = _reactive_reinforcement_margin(
        obs, cache, source_idx, K, weight=w, lag=_source_safety_lag(),
    )                                                                # [S, K] | None
    if threat is None:
        return None
    dtype = obs.ships.dtype
    src = source_idx.clamp(0, int(obs.P) - 1)
    help_sk = _friendly_support_margin(obs, cache, source_idx, K)
    if help_sk is None:
        help_sk = torch.zeros_like(threat)
    k_grid = torch.arange(1, K + 1, device=obs.device, dtype=dtype).view(1, K)
    slack = prod[src].to(dtype).unsqueeze(-1) * k_grid + help_sk - threat  # [S, K]
    allowed = obs.ships[src].to(dtype) + slack.min(dim=-1).values    # [S]
    return allowed.clamp(min=0.0)


def _hold_value() -> float:
    raw = os.environ.get("PRODUCER_PLUS_HOLD_VALUE", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _hold_value_lag() -> float:
    raw = os.environ.get("PRODUCER_PLUS_HOLD_VALUE_LAG", "2.0")
    try:
        return float(raw)
    except ValueError:
        return 2.0


def _hold_value_bonus(
    *,
    obs,
    cache,
    target_idx: Tensor,        # [T] shortlist slots
    cand_tgt_slot: Tensor,     # [C]
    cand_tgt_short: Tensor,    # [C]
    cand_send: Tensor,         # [C, L]
    cand_eta: Tensor,          # [C, L]
    cand_valid: Tensor,        # [C]
    cand_is_def: Tensor,       # [C]
    capture_floor_TK: Tensor,  # [T, K]
    prod: Tensor,              # [P]
    K: int,
) -> Tensor:
    """Per-candidate post-horizon production credit, ``[C]`` (≥ 0)."""
    device = cand_send.device
    dtype = cand_send.dtype
    C = int(cand_send.shape[0])
    lam = _hold_value()
    if lam <= 0.0 or C == 0 or K <= 0:
        return torch.zeros(C, dtype=dtype, device=device)
    P = int(obs.P)
    tgt = cand_tgt_slot.clamp(0, P - 1)
    neutral_now = obs.is_neutral[tgt] & obs.alive[tgt]
    gate = cand_valid & ~cand_is_def & neutral_now                    # [C]
    if not bool(gate.any()):
        return torch.zeros(C, dtype=dtype, device=device)

    send_tot = cand_send.sum(dim=-1)                                  # [C]
    eta_max = cand_eta.max(dim=-1).values                             # [C]
    k_arr = (eta_max.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
    floor_at_arr = (
        capture_floor_TK[cand_tgt_short.clamp(0, capture_floor_TK.shape[0] - 1)]
        .gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    )                                                                 # [C]
    # capture_floor = defenders + 1 (overhead); conquered garrison =
    # send − defenders = send − floor + 1.
    survivors = (send_tot - floor_at_arr + 1.0).clamp(min=1.0)        # [C]
    prod_t = prod[tgt].to(dtype)                                      # [C]

    margin = _reactive_reinforcement_margin(
        obs, cache, target_idx, K, weight=1.0, lag=_hold_value_lag(),
    )                                                                 # [T, K] | None
    if margin is None:
        safe = gate
    else:
        m_c = margin[cand_tgt_short.clamp(0, margin.shape[0] - 1)].to(dtype)  # [C, K]
        k_grid = torch.arange(K, device=device, dtype=dtype).view(1, K)
        dk = k_grid - k_arr.to(dtype).view(C, 1)                      # ticks after arrival
        garrison = survivors.view(C, 1) + prod_t.view(C, 1) * dk.clamp(min=0.0)
        threat = (m_c >= garrison) & (dk > 0)                         # [C, K]
        safe = gate & ~threat.any(dim=-1)
    return torch.where(
        safe, lam * prod_t, torch.zeros(C, dtype=dtype, device=device))


# --- Garrison-deficit reinforcement value (PRODUCER_PLUS_GARRISON_VALUE) -------
# Live war-ledger finding (audit 2026-06-11 night): at the 1300+ band, the
# 4P winner is whoever reinforces more (our wins: we out-garrison the top
# rival 58%/33%; our losses: 46%/61%; the Blu3s siege: 15,868 vs 942
# reinforcement ships, 42/42 vs 31/39 wave success). The flow scorer values
# reinforcement only when a known IN-FLIGHT wave makes a planet savable —
# by then the avalanche is launched and no single send parries it. This
# term prices PROACTIVE garrisoning: an own-target send earns the planet's
# holding value when the planet's local balance against the enemy's
# UNCOMMITTED reserve is negative and the send covers the deficit. Same
# balance-of-force model as the source-safety cap (push side); this is the
# pull side, chooser-internal per the three-falsifications friction note
# (thin post-pass regroup lanes land ships the chooser never uses).
# Default 0 = byte-identical.


def _living_rival_count(obs) -> int:
    """Rivals with at least one living planet (planet-only proxy)."""
    rivals = obs.owner_abs[obs.is_enemy & obs.alive]
    if rivals.numel() == 0:
        return 0
    return int(torch.unique(rivals).numel())


def _garrison_value() -> float:
    raw = os.environ.get("PRODUCER_PLUS_GARRISON_VALUE", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _garrison_value_from_step() -> int:
    """Opening gate: no proactive-garrison credit before this step. The
    land-grab phase decides production rank (wins are production-ahead@40
    in 16/17); insurance bought then costs expansion tempo (seed-6 panel
    wipe: stalled at 3 planets by t=60, dead by 170)."""
    raw = os.environ.get("PRODUCER_PLUS_GARRISON_VALUE_FROM_STEP", "0")
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 0


def _garrison_value_bonus(
    *,
    obs,
    cache,
    target_idx: Tensor,        # [T] shortlist slots
    cand_tgt_slot: Tensor,     # [C]
    cand_tgt_short: Tensor,    # [C]
    cand_send: Tensor,         # [C, L]
    cand_eta: Tensor,          # [C, L]
    cand_valid: Tensor,        # [C]
    cand_is_def: Tensor,       # [C]
    prod: Tensor,              # [P]
    K: int,
) -> Tensor:
    """Per-candidate proactive-garrison credit, ``[C]`` (>= 0).

    Deficit of own planet t over the window, judged at/after the send's
    arrival: D(t) = max_{k >= eta} [ w*threat(t,k) - (g_t + prod_t*k +
    help(t,k)) ]. A send earns lambda_g * prod_t when D > 0 (the planet is
    expected to fall to a feasible strike) and the send covers D.
    """
    device = cand_send.device
    dtype = cand_send.dtype
    C = int(cand_send.shape[0])
    lam = _garrison_value()
    w = _source_safety_weight()
    if lam <= 0.0 or C == 0 or K <= 0:
        return torch.zeros(C, dtype=dtype, device=device)
    gate = cand_valid & cand_is_def                                   # [C]
    if not bool(gate.any()):
        return torch.zeros(C, dtype=dtype, device=device)
    threat = _reactive_reinforcement_margin(
        obs, cache, target_idx, K,
        weight=(w if w > 0.0 else 1.0), lag=_source_safety_lag(),
    )                                                                 # [T, K] | None
    if threat is None:
        return torch.zeros(C, dtype=dtype, device=device)
    help_tk = _friendly_support_margin(obs, cache, target_idx, K)
    if help_tk is None:
        help_tk = torch.zeros_like(threat)
    P = int(obs.P)
    tgt_safe = target_idx.clamp(0, P - 1)
    k_grid = torch.arange(1, K + 1, device=device, dtype=dtype).view(1, K)
    base = (
        obs.ships[tgt_safe].to(dtype).unsqueeze(-1)
        + prod[tgt_safe].to(dtype).unsqueeze(-1) * k_grid
        + help_tk
    )                                                                 # [T, K]
    deficit_tk = threat - base                                        # [T, K]
    # The enemy reserve is ONE resource per rival — it cannot strike every
    # deficit simultaneously. Pricing every worst-case deficit as certain
    # turns the agent into a turtle (seed-6 panel wipe: 36 reinforce
    # launches vs 4 neutral grabs, stalled at 3 planets while rivals took
    # 12+). Credit at most R targets per turn (R = living rivals), ranked
    # by strike attractiveness to the enemy (production of the planet it
    # could feasibly take).
    n_rivals = 0
    owner_alive = obs.is_enemy & obs.alive
    if bool(owner_alive.any()):
        n_rivals = max(int(_living_rival_count(obs)), 1)
    deficit_t = deficit_tk.max(dim=-1).values                         # [T]
    prod_T = prod[tgt_safe].to(dtype)                                 # [T]
    attract = torch.where(
        deficit_t > 0.0, prod_T, torch.full_like(prod_T, float("-inf")))
    T = int(attract.shape[0])
    R = min(max(n_rivals, 1), T)
    top_idx = attract.topk(R).indices
    eligible_T = torch.zeros(T, dtype=torch.bool, device=device)
    eligible_T[top_idx] = True
    eligible_T &= deficit_t > 0.0                                     # [T]
    t_c = cand_tgt_short.clamp(0, deficit_tk.shape[0] - 1)
    d_c = deficit_tk[t_c]                                             # [C, K]
    eta_max = cand_eta.max(dim=-1).values                             # [C]
    k_arr = (eta_max.clamp(min=1.0, max=float(K)).ceil() - 1.0).view(C, 1)
    at_or_after = (
        torch.arange(K, device=device, dtype=dtype).view(1, K) >= k_arr
    )                                                                 # [C, K]
    neg_fill = torch.full_like(d_c, float("-inf"))
    D = torch.where(at_or_after, d_c, neg_fill).max(dim=-1).values    # [C]
    send_tot = cand_send.sum(dim=-1)                                  # [C]
    covers = (D > 0.0) & (send_tot >= D) & eligible_T[t_c]
    prod_t = prod[cand_tgt_slot.clamp(0, P - 1)].to(dtype)
    return torch.where(
        gate & covers, lam * prod_t,
        torch.zeros(C, dtype=dtype, device=device))


def _denial_bonus_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_DENIAL_BONUS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _denial_bonus_weight() -> float:
    raw = os.environ.get("PRODUCER_PLUS_DENIAL_WEIGHT", "0.1")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.1


def _opening_bonus_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_OPENING_BONUS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _opening_bonus_weight() -> float:
    raw = os.environ.get("PRODUCER_PLUS_OPENING_WEIGHT", "0.1")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.1


def _opening_window() -> int:
    raw = os.environ.get("PRODUCER_PLUS_OPENING_WINDOW", "30")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 30


def _game_length_est() -> int:
    raw = os.environ.get("PRODUCER_PLUS_GAME_LENGTH_EST", "200")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 200


# Force-concentration: relax the target mutex inside _greedy_select so up to
# ``max_waves_per_target`` waves can land on the same target per turn. Between
# waves we re-score candidates with the just-fired wave appended to the
# scoring LaunchSet (owner=pid) so wave 2 sees wave 1's capture/reinforcement
# and does NOT double-count. Default OFF preserves byte-identical single-wave
# behaviour (no rescore closure built, max_waves_per_target=1 passed through).
# See knowledge-base for the architectural diagnosis: scorer tuning can never
# reach candidates the chooser refuses to enumerate.
def _force_concentration_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_FORCE_CONCENTRATION", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _force_concentration_max_waves() -> int:
    raw = os.environ.get("PRODUCER_PLUS_FORCE_CONCENTRATION_MAX_WAVES", "2")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 2


# --- FFA-aware competitive score (4P objective fix) -------------------------
# Live-replay diagnosis (knowledge-base 2026-06-10): 82 of 83 4P losses end
# with us ELIMINATED, carved by 2+ opponents mid-game, while the legacy score
# (my delta minus the SUM of all opponents' deltas) scores mutual-damage
# trades positive — it optimizes total damage dealt, which in a 4-player
# free-for-all leaves both fighters weaker relative to the bystanders. The
# fix weights each opponent's delta by their strength share (weights sum to
# 1), so trades are valued by how much they shift my standing against the
# rivals that actually threaten me. 2P is byte-identical: weights are only
# built when player_count >= 3.
def _ffa_score_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_FFA_SCORE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _ffa_weight_mode() -> str:
    """``strength`` (default): weights ∝ rival planet+fleet ships.
    ``uniform``: equal weight per living rival — tests whether the
    trade-devaluation alone helps without the hit-the-leader tilt."""
    raw = os.environ.get("PRODUCER_PLUS_FFA_WEIGHTS", "strength").strip().lower()
    return raw if raw in ("strength", "uniform") else "strength"


def _ffa_opp_weights(obs_tensors: dict, *, player_id: int, player_count: int):
    """Per-opponent weights ∝ current total strength (planet + fleet ships),
    or equal-per-living-rival under ``PRODUCER_PLUS_FFA_WEIGHTS=uniform``.

    Returns a ``[player_count]`` float tensor with 0 at ``player_id``,
    summing to 1 over living opponents (all-zero if every opponent is dead).
    """
    planets = obs_tensors["planets"]            # [P, 7]: owner=col1, ships=col5
    device = planets.device
    a = int(player_count)
    strength = torch.zeros(a, dtype=planets.dtype, device=device)
    p_owner = planets[:, 1].long()
    p_mask = (planets[:, 0] >= 0) & (p_owner >= 0) & (p_owner < a)
    if bool(p_mask.any()):
        strength.scatter_add_(0, p_owner[p_mask], planets[p_mask, 5])
    fleets = obs_tensors.get("fleets")
    if fleets is not None and fleets.numel():
        f_owner = fleets[:, 1].long()           # [F, 7]: owner=col1, ships=col6
        f_mask = (fleets[:, 0] >= 0) & (f_owner >= 0) & (f_owner < a)
        if bool(f_mask.any()):
            strength.scatter_add_(0, f_owner[f_mask], fleets[f_mask, 6])
    strength[int(player_id)] = 0.0
    if _ffa_weight_mode() == "uniform":
        strength = (strength > 0).to(planets.dtype)
    total = float(strength.sum())
    if total <= 0.0:
        return torch.zeros(a, dtype=planets.dtype, device=device)
    return strength / total


# --- Commitment cost ----------------------------------------------------------
# Ported insight from the ledger branch (audit/2026-06-10-ledger-agent-from-
# first-principles.md): in-flight ships cannot change course, so committed
# capital is the army you lack when the opponent's wave lands. Our own
# evidence agrees from two sides — top teams strike at flight-time 4-5 vs
# our 7-8, and the replan/redirect family measured that ships held home
# beat every scheme for spending them. Price it: each candidate pays
# eps x ships x flight-turns (per contributing leg). Tempo tie-break toward
# near targets falls out; distant marginal attacks stop clearing the bar.


def _commit_cost_eps() -> float:
    raw = os.environ.get("PRODUCER_PLUS_COMMIT_COST", "0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _commit_flight_cost(cand_send: Tensor, cand_eta: Tensor, cand_active: Tensor) -> Tensor:
    """Σ_legs ships x eta over active legs. ``[C]`` (ship-turn units)."""
    cost = torch.where(
        cand_active, cand_send * cand_eta, torch.zeros_like(cand_send),
    )
    return cost.sum(dim=-1)


# --- Reinforcement deficit floor (defense candidate sizing fix) -------------
# capture_floor returns 1 for targets we own at the arrival tick ("arriving
# ships add to the garrison, nothing to clear"), so the multi-size enumeration
# for a defensive target is (1, 2, safe_drain) — the "exactly enough to HOLD
# the planet" size is never a candidate, and trickle sends below it are junk
# the greedy must price out one by one. For an owned target the do-nothing
# projection shows flipping at tick k_f, any reinforcement arriving at k <=
# k_f holds the planet iff it adds at least the attacker's projected margin,
# and that margin IS the projection's post-flip ship count at k_f (engine
# survivor = top1 - top2). This floor replaces 1 with (margin + overhead) on
# the pre-flip cells, giving the chooser the right-sized defense candidate
# and invalidating doomed under-sized ones. Default OFF = byte-identical.
def _reinforce_deficit_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_REINFORCE_DEFICIT", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _apply_reinforce_deficit_floor(
    floor: Tensor,                  # [T, K] from capture_floor
    *,
    garrison_status,
    target_idx: Tensor,             # [T] long
    player_id: int,
    capture_overhead: float = 1.0,
) -> Tensor:
    """Raise pre-flip reinforcement floors to the hold-the-planet deficit.

    For each target currently ours whose do-nothing projection flips it at
    tick ``k_f`` (first ``owner != me`` within the floor's K window), every
    cell ``k < k_f`` (where the planet is still ours at arrival) gets
    ``floor = max(1, ceil(ships_at_flip + overhead))`` — ``ships_at_flip``
    is the projected post-combat survivor of the new owner, i.e. exactly
    the margin our reinforcement must add to keep the planet. Cells at and
    after ``k_f`` already carry the retake floor from capture_floor's
    not-mine branch. Targets with no projected flip are untouched.
    """
    T, K = int(floor.shape[0]), int(floor.shape[-1])
    if T == 0 or K == 0:
        return floor
    owner = garrison_status.owner
    ships = garrison_status.ships
    P = int(owner.shape[0])
    pid = int(player_id)
    tgt = target_idx.clamp(0, max(P - 1, 0))
    owner_g = owner[tgt]                                    # [T, H+1]
    ships_g = ships[tgt]
    mine_now = owner_g[..., 0] == pid                       # [T]
    not_mine_k = owner_g[..., 1 : K + 1] != pid             # [T, K]
    any_flip = not_mine_k.any(dim=-1) & mine_now            # [T]
    # first flip tick (0-based index into k=1..K), device-stable on ties.
    k_f_idx = _stable_argmax(not_mine_k.to(torch.int64))    # [T]
    ships_at_flip = ships_g.gather(
        -1, (k_f_idx + 1).clamp(max=int(ships_g.shape[-1]) - 1).unsqueeze(-1)
    ).squeeze(-1)                                           # [T]
    deficit = (ships_at_flip + float(capture_overhead)).clamp(min=1.0).ceil()
    k_grid = torch.arange(K, device=floor.device).view(1, K)
    pre_flip = any_flip.view(T, 1) & (k_grid < k_f_idx.view(T, 1))   # [T, K]
    return torch.where(
        pre_flip, torch.maximum(floor, deficit.view(T, 1)), floor,
    )


# --- Overkill factor (mass-concentration attack sizing) ---------------------
# Top-ladder behavioral mining (audit/2026-06-10-top-ladder-behavior.md):
# the 1600-1750 agents launch ~half as often as we do with 2-4x the fleet
# mass (median 36-83 ships vs our 21), expand faster, and hold 2-4x our ship
# count by step 80. In our own 2P losses the opponent's median fleet is 30+
# vs 16 in our wins. The engine's multi-size lo/mid variants are sized at the
# bare capture floor — the minimal send that flips the planet — which wins
# the combat but leaves a 1-ship garrison the opponent retakes past the
# scorer horizon. OVERKILL_FACTOR scales the SIZING of the lo variant
# (floor*F, capped by safe_drain) so enumerated attacks are decisive instead
# of marginal. The floor VALIDITY gate is unchanged (a drain-sized send that
# clears the true floor stays valid), and 1.0 is byte-identical.
def _overkill_factor() -> float:
    raw = os.environ.get("PRODUCER_PLUS_OVERKILL_FACTOR", "1.0")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 1.0


# --- Mass tie-break ----------------------------------------------------------
# The exact flow scorer values a minimal capture and an overwhelming capture
# of the same target almost identically (both flip the planet within H; the
# surplus ships survive either way), and _stable_argmax then resolves the tie
# toward the LOWEST index — which is the smallest size variant. Retention
# beyond the horizon favors the larger send (the surplus garrisons the
# capture against the counter the scorer can't see). Add an epsilon-scale
# size preference (1e-4 score per ship sent) so near-ties resolve toward
# mass without distorting genuinely different scores. Default OFF.
def _mass_tiebreak_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_MASS_TIEBREAK", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# --- Regroup convoying -------------------------------------------------------
# 71% of the champion's launches are own-planet transfers at median 18 ships
# (the regroup lane fires near-continuously), while top-ladder agents move
# mass in large convoys (their overall fleet median is 36-83 vs our 21).
# Small parcels are also strictly SLOWER (fleet speed rises with ship
# count). With a positive threshold, regroup entries below it are dropped —
# ships stay garrisoned and accumulate until a convoy-sized transfer fires.
# 0 = OFF, byte-identical.
def _regroup_min_send() -> float:
    raw = os.environ.get("PRODUCER_PLUS_REGROUP_MIN_SEND", "0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


# --- Terminal production value -----------------------------------------------
# The flow scorer truncates a captured planet's payoff at the horizon
# (H=18 2P / 13 4P), so a neutral whose in-horizon production only repays its
# garrison cost scores ~0 and never clears the 1.5-ship roi threshold — the
# seed-7 expansion probe shows the planner offered dozens of valid neutral
# captures every opening turn at best-score 0..1 while the bank climbed to
# ~300 ships, expanding only on turns where the opponent projection shifted
# the do-nothing baseline negative. The weight is the number of post-horizon
# steps the production owned at the horizon's final step is credited for.


def _terminal_neutral_only() -> bool:
    return os.environ.get("PRODUCER_PLUS_TERMINAL_PROD_NEUTRAL_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _terminal_prod_value() -> float:
    raw = os.environ.get("PRODUCER_PLUS_TERMINAL_PROD_VALUE", "0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


# --- Response veto -------------------------------------------------------------
# The opp projection predicts the opponent's plan ASSUMING WE DO NOTHING, so
# reactive defense to our own waves is invisible to the scorer. Live mining
# (audit/2026-06-10-top-ladder-behavior.md): 30% of our capture-sized attacks
# fail to flip, and 65% of those failures die to defense that arrived while
# our fleet was in flight — ~321 ships/game thrown into parries a
# producer-like opponent visibly prepares. One extra mirror pass with OUR
# chosen waves as background yields each opponent's predicted REPLY; attack
# waves whose flow score under that reply is worse than doing nothing (by
# more than the margin) are dropped before dispatch.


def _response_veto_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_RESPONSE_VETO", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _response_veto_2p_only() -> bool:
    """Player-count gate so a composed bundle can run the veto in 2P while
    keeping the 4P action stream byte-identical to a measured 4P bundle."""
    return os.environ.get("PRODUCER_PLUS_RESPONSE_VETO_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _response_veto_active(player_count: int) -> bool:
    return _response_veto_enabled() and (
        (not _response_veto_2p_only()) or int(player_count) == 2
    )


def _response_veto_upsize_enabled() -> bool:
    """\"Beat the parry\": when the predicted reply kills a wave, retry the
    same target with the source's full spare budget (new aim/eta for the
    bigger, faster fleet) before dropping. The flow scorer judges whether
    over-draining the source is safe — no separate cap."""
    return os.environ.get("PRODUCER_PLUS_RESPONSE_VETO_UPSIZE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _response_veto_margin(default: float) -> float:
    """Veto margin; defaults to the planner's own roi threshold so the wave
    must clear the SAME gain bar under the predicted reply that it cleared
    without it. (A clean parry is a material-neutral trade — score exactly 0
    — so a zero margin would never veto it.)"""
    raw = os.environ.get("PRODUCER_PLUS_RESPONSE_VETO_MARGIN")
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _shot_mlp_threshold() -> float:
    """Learned shot-success filter (see shot_mlp.py). The env value is the
    rejection threshold on P(target still ours 10 turns after arrival);
    unset/0 = OFF."""
    raw = '0.30'  # hardcoded at bundle time (env-leak-proof A/B)
    if raw is None or not raw.strip():
        return 0.0
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


def _shot_mlp_2p_only() -> bool:
    return os.environ.get("PRODUCER_PLUS_SHOT_MLP_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _shot_mlp_active(player_count: int) -> bool:
    return _shot_mlp_threshold() > 0.0 and (
        (not _shot_mlp_2p_only()) or int(player_count) == 2
    )


def _replan_enabled() -> bool:
    """One-ply replan: after our waves are chosen, predict the opponent's
    reply (same mirror as the veto) and run our WHOLE planner a second time
    with that reply as background. Where the veto only drops doomed waves
    (the ships idle), the replan redirects them to the next-best action,
    plans reinforcements against the predicted counter (the reply feeds the
    defensive shortlist), and re-judges every wave with the reply's flow
    consequences in the diff."""
    return os.environ.get("PRODUCER_PLUS_REPLAN", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _replan_2p_only() -> bool:
    return os.environ.get("PRODUCER_PLUS_REPLAN_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _replan_active(player_count: int) -> bool:
    return _replan_enabled() and (
        (not _replan_2p_only()) or int(player_count) == 2
    )


# --- Background-aware floors -------------------------------------------------
# The flow SCORER sees the opponent's predicted launches (they're merged into
# every candidate's diff), but the SIZING subsystem — capture_floor, the
# defensive shortlist, safe_drain — reads the frozen do-nothing projection.
# Three measured behaviours trace to that inconsistency: attacks sized for
# garrisons that get reinforced mid-flight (the scorer then rejects the
# right-sized wave it was never offered), no toll-sniping of predicted
# captures (after THEIR fleet annihilates against a neutral, the survivor is
# cheap — invisible to static floors), and drains/regroups out of planets a
# predicted strike is about to hit. Fix: re-project the garrison trajectories
# ONCE with the background launches applied (exact engine recurrence — the
# same one the scorer trusts) and let the sizing subsystem read that.


def _bg_floors_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_BG_FLOORS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _background_adjusted_status(
    garrison_status, *, background: LaunchSet, prod: Tensor, alive_by_step: Tensor,
):
    """Garrison trajectories with the background launches applied. New status.

    Sources are debited at step 0 (a launch leaves now even if it lands past
    the horizon); arrivals land at ``ceil(eta)`` like the scorer's hypothesis
    axis; the exact production→combat recurrence is replayed. ``alive_by_step``
    is ``[H+1, P]`` (run_turn's orientation).
    """
    owner0 = garrison_status.owner[..., 0]                       # [P]
    ships0 = garrison_status.ships[..., 0]                       # [P]
    arr = garrison_status.arrivals_by_owner                      # [P, H+1, A]
    P, H1, A = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    H = H1 - 1
    fdtype = ships0.dtype if ships0.is_floating_point() else torch.float32

    sel = background.valid
    src = background.source_slots[sel].clamp(0, max(P - 1, 0))
    tgt = background.target_slots[sel].clamp(0, max(P - 1, 0))
    ships = background.ships[sel].to(fdtype)
    own = background.owner[sel].clamp(0, max(A - 1, 0))
    tick = background.eta[sel].ceil().long().clamp(min=1)

    init_ships = ships0.to(fdtype).clone()
    init_ships.index_add_(0, src, -ships)
    init_ships = init_ships.clamp(min=0.0)

    arr_delta = arr[:, 1:, :].to(fdtype).clone()                  # [P, H, A]
    in_h = tick <= H
    if bool(in_h.any()):
        arr_delta.index_put_(
            (tgt[in_h], tick[in_h] - 1, own[in_h]), ships[in_h], accumulate=True,
        )

    owner_t, ships_t, pre_o, pre_s = _run_exact_recurrence(
        init_owner=owner0.unsqueeze(0),
        init_ships=init_ships.unsqueeze(0),
        prod=prod.to(fdtype).unsqueeze(0),
        alive=alive_by_step.transpose(0, 1).unsqueeze(0),
        arrivals=arr_delta.unsqueeze(0),
    )
    return PlanetGarrisonStatus(
        owner=owner_t[0], ships=ships_t[0],
        pre_combat_owner=pre_o[0], pre_combat_ships=pre_s[0],
        arrivals_by_owner=torch.cat([arr[:, :1, :].to(fdtype), arr_delta], dim=1),
    )


def _entries_to_launch_set(entries, *, pid: int, device, dtype) -> LaunchSet:
    """Valid rows of a LaunchEntries table as a LaunchSet owned by ``pid``."""
    sel = entries.valid.nonzero(as_tuple=True)[0]
    return LaunchSet(
        source_slots=entries.source_slots[sel].to(torch.long),
        target_slots=entries.target_slots[sel].to(torch.long),
        ships=entries.ships[sel].to(dtype),
        eta=entries.eta[sel].to(dtype),
        owner=torch.full((int(sel.shape[0]),), pid, dtype=torch.long, device=device),
        valid=torch.ones(int(sel.shape[0]), dtype=torch.bool, device=device),
    )


def _reply_seq_enabled() -> bool:
    """Sequential multi-rival reply conditioning — see comment in
    _predict_reply. No-op with a single opponent (2P byte-identical)."""
    return os.environ.get("PRODUCER_PLUS_REPLY_SEQ", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _predict_reply(
    mine: LaunchSet,
    *,
    movement,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config,
    player_count: int,
    pid: int,
    K_eta_override,
    H: int,
) -> LaunchSet:
    """Each opponent's predicted reply to OUR launches, merged on the L axis.

    Mirror each opponent seat separately WITH the roi normalization their
    planner needs: with our waves as background, every opp candidate's
    flow diff inherits our attacks' damage as a large negative constant,
    so against the absolute 1.5 threshold the simulated opponent is
    paralyzed and "replies" with nothing (seed-0 instrumented game: 15
    predicted reply launches across 107 turns, 0 vetoes). Shift the
    threshold by THEIR do-nothing score, exactly as run_turn does for us.
    """
    opp_ids = [q for q in range(int(player_count)) if q != int(pid)]
    seq = _reply_seq_enabled() and len(opp_ids) > 1
    reply_parts = []
    pad = _env_int("PRODUCER_PLUS_OPP_MAX_L", MAX_L_OPP)
    base = mine
    for opp_id in opp_ids:
        dn_opp = float(_score_do_nothing(
            status=garrison_status, prod=prod, alive_by_step=alive_by_step,
            player_count=int(player_count), background=base,
            player_id=int(opp_id), opp_weights=None,
        ))
        cfg_opp = dataclasses.replace(
            config, roi_threshold=dn_opp + float(config.roi_threshold),
        )
        part = predict_opp_launches_via_mirror(
            plan_fn=plan_lite_waves,
            obs_tensors=obs_tensors, movement=movement, cache=cache,
            garrison_status=garrison_status, prod=prod, alive_by_step=alive_by_step,
            opp_ids=[int(opp_id)], config=cfg_opp, player_count=int(player_count),
            K_eta_override=K_eta_override,
            pad_to=pad,
            K=1, H=H,
            base_background=base,
        )
        reply_parts.append(part)
        if seq:
            # Sequential conditioning (PRODUCER_PLUS_REPLY_SEQ): later rivals
            # see earlier rivals' predicted launches, not just ours. The
            # independent merge prices every attack as if ALL rivals parry
            # it simultaneously with full attention (defense counted once
            # per rival) — measured to make the ungated 4P veto chronically
            # passive (eliminated by step ~200; panel 1/16). Conditioning
            # divides each rival's attention by the threats already on the
            # board, like the K-round projection does for one opponent.
            base = _cat_launch_sets([base, part])
    if len(reply_parts) == 1:
        return reply_parts[0]
    return _cat_launch_sets(reply_parts)


def _cat_launch_sets(parts: list) -> LaunchSet:
    """Concatenate LaunchSets along the L axis."""
    if len(parts) == 1:
        return parts[0]
    return LaunchSet(
        source_slots=torch.cat([r.source_slots for r in parts]),
        target_slots=torch.cat([r.target_slots for r in parts]),
        ships=torch.cat([r.ships for r in parts]),
        eta=torch.cat([r.eta for r in parts]),
        owner=torch.cat([r.owner for r in parts]),
        valid=torch.cat([r.valid for r in parts]),
    )


def _apply_response_veto(
    entries,
    *,
    movement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config,
    player_count: int,
    K_eta_override,
    H: int,
    opp_weights,
    reply_out: list | None = None,
    reply_trust: float | None = None,
):
    """Drop attack waves the opponent's predicted reply kills. See gate note.

    ``reply_out``: optional mutable list; the predicted reply LaunchSet is
    appended when the mirror runs, so a downstream pass (the redirect) can
    reuse it without a second mirror. ``reply_trust``: certainty-equivalent
    scaling of the reply's ships (None = full trust).
    """
    pid = int(obs.player_id)
    P = int(obs.P)
    valid = entries.valid
    if int(valid.sum().item()) == 0:
        return entries
    device = obs.device
    dtype = obs.ships.dtype
    tgt_safe = entries.target_slots.clamp(0, P - 1)
    is_attack = valid & ~obs.owned[tgt_safe]
    idx = is_attack.nonzero(as_tuple=True)[0]
    C = int(idx.shape[0])
    if C == 0:
        return entries

    sel = valid.nonzero(as_tuple=True)[0]
    mine = _entries_to_launch_set(entries, pid=pid, device=device, dtype=dtype)
    reply = _predict_reply(
        mine,
        movement=movement, obs_tensors=obs_tensors, cache=cache,
        garrison_status=garrison_status, prod=prod, alive_by_step=alive_by_step,
        config=config, player_count=int(player_count), pid=pid,
        K_eta_override=K_eta_override, H=H,
    )
    if reply_trust is not None:
        reply = _scale_launch_set_ships(reply, float(reply_trust))
    if reply_out is not None:
        reply_out.append(reply)

    # Score each attack wave alone under the predicted reply, against the
    # do-nothing-under-reply baseline (same normalization as roi_threshold).
    cand = make_launch_set(
        source_slots=entries.source_slots[idx].view(C, 1),
        target_slots=entries.target_slots[idx].view(C, 1),
        ships=entries.ships[idx].view(C, 1).to(dtype),
        eta=entries.eta[idx].view(C, 1).to(dtype),
        valid=torch.ones(C, 1, dtype=torch.bool, device=device),
        player_id=pid,
    )

    def _bg(t: Tensor) -> Tensor:
        return t.unsqueeze(0).expand(C, -1)

    merged = LaunchSet(
        source_slots=torch.cat([cand.source_slots, _bg(reply.source_slots)], dim=-1),
        target_slots=torch.cat([cand.target_slots, _bg(reply.target_slots)], dim=-1),
        ships=torch.cat([cand.ships, _bg(reply.ships)], dim=-1),
        eta=torch.cat([cand.eta, _bg(reply.eta)], dim=-1),
        owner=torch.cat([cand.owner, _bg(reply.owner)], dim=-1),
        valid=torch.cat([cand.valid, _bg(reply.valid)], dim=-1),
    )
    scores = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=merged, player_id=pid,
        opp_weights=opp_weights, terminal_prod_weight=_terminal_prod_value(),
        terminal_neutral_only=_terminal_neutral_only(),
    )
    if _hold_value() > 0.0:
        # Price the holding-time capture credit consistently in the veto:
        # without this the veto re-scores hold-value-justified captures at
        # their raw ~0 flow and drops every launch the credit enabled
        # (verified on the Gregor Lied trace: 4 waves pre-veto, 0 post).
        tgt_e = entries.target_slots[idx].clamp(0, P - 1)
        K_v = max(1, min(
            int(K_eta_override) if K_eta_override is not None else int(config.horizon),
            H,
        ))
        _rf_w_v = _reactive_floor_for(int(player_count))
        _rf_m_v = (
            _reactive_reinforcement_margin(obs, cache, tgt_e, K_v, weight=_rf_w_v)
            if _rf_w_v > 0.0 else None
        )
        floor_e = capture_floor(
            garrison_status, target_idx=tgt_e, k_max=K_v,
            capture_overhead=1.0, player_id=pid, reinforcement=_rf_m_v,
        )                                                            # [C, K_b]
        K_b = int(floor_e.shape[-1])
        if K_b > 0:
            scores = scores + _hold_value_bonus(
                obs=obs, cache=cache, target_idx=tgt_e,
                cand_tgt_slot=tgt_e,
                cand_tgt_short=torch.arange(C, device=device),
                cand_send=entries.ships[idx].view(C, 1).to(dtype),
                cand_eta=entries.eta[idx].view(C, 1).to(dtype),
                cand_valid=torch.ones(C, dtype=torch.bool, device=device),
                cand_is_def=obs.owned[tgt_e],
                capture_floor_TK=floor_e, prod=prod, K=K_b,
            )
    dn = _score_do_nothing(
        status=garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), background=reply, player_id=pid,
        opp_weights=opp_weights,
    )
    margin = _response_veto_margin(float(config.roi_threshold))
    keep = (scores - dn) >= margin
    if bool(keep.all()):
        return entries

    new_valid = entries.valid.clone()
    new_ships = entries.ships.clone()
    new_angle = entries.angle.clone()
    new_eta = entries.eta.clone()
    drop = idx[~keep]
    new_valid[drop] = False

    if _response_veto_upsize_enabled() and movement is not None and int(drop.shape[0]) > 0:
        # "Beat the parry": retry each killed wave at the source's full
        # spare budget (everything not already committed by the plan),
        # with aim/eta recomputed for the bigger — and therefore FASTER —
        # fleet. The flow scorer judges whether stripping the source is
        # safe (the debit is part of the diff); only drop when even the
        # full send fails the margin under the reply.
        committed = torch.zeros(P, dtype=dtype, device=device)
        committed.scatter_add_(
            0, entries.source_slots[sel].clamp(0, P - 1), entries.ships[sel].to(dtype),
        )
        D = int(drop.shape[0])
        d_src = entries.source_slots[drop].clamp(0, P - 1)
        d_tgt = entries.target_slots[drop].clamp(0, P - 1)
        spare = (obs.ships.to(dtype)[d_src] - committed[d_src]).clamp(min=0.0).floor()
        up_size = entries.ships[drop].to(dtype) + spare
        aim = intercept_angle(movement, d_src, d_tgt, up_size)
        up_viable = (spare >= 1.0) & aim["viable"] & (aim["eta"] <= float(H))
        if bool(up_viable.any()):
            cand_up = make_launch_set(
                source_slots=d_src.view(D, 1),
                target_slots=d_tgt.view(D, 1),
                ships=up_size.view(D, 1),
                eta=aim["eta"].to(dtype).view(D, 1),
                valid=up_viable.view(D, 1),
                player_id=pid,
            )

            def _bgD(t: Tensor) -> Tensor:
                return t.unsqueeze(0).expand(D, -1)

            merged_up = LaunchSet(
                source_slots=torch.cat([cand_up.source_slots, _bgD(reply.source_slots)], dim=-1),
                target_slots=torch.cat([cand_up.target_slots, _bgD(reply.target_slots)], dim=-1),
                ships=torch.cat([cand_up.ships, _bgD(reply.ships)], dim=-1),
                eta=torch.cat([cand_up.eta, _bgD(reply.eta)], dim=-1),
                owner=torch.cat([cand_up.owner, _bgD(reply.owner)], dim=-1),
                valid=torch.cat([cand_up.valid, _bgD(reply.valid)], dim=-1),
            )
            scores_up = score_candidates(
                garrison_status, prod=prod, alive_by_step=alive_by_step,
                player_count=int(player_count), launches=merged_up, player_id=pid,
                opp_weights=opp_weights, terminal_prod_weight=_terminal_prod_value(),
        terminal_neutral_only=_terminal_neutral_only(),
            )
            keep_up = up_viable & ((scores_up - dn) >= margin)
            if bool(keep_up.any()):
                ui = drop[keep_up]
                new_valid[ui] = True
                new_ships[ui] = up_size[keep_up].to(new_ships.dtype)
                new_angle[ui] = aim["angle"][keep_up].to(new_angle.dtype)
                new_eta[ui] = aim["eta"][keep_up].to(new_eta.dtype)

    return LaunchEntries(
        source_slots=entries.source_slots, target_slots=entries.target_slots,
        ships=new_ships, angle=new_angle, eta=new_eta,
        valid=new_valid,
    )


def _apply_replan(
    entries,
    *,
    movement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config,
    player_count: int,
    K_eta_override,
    H: int,
    opp_weights,
):
    """One-ply replan: re-run the planner with the predicted reply as background.

    Pass 1 (the caller's ``entries``) planned against the opponent's
    do-nothing-conditioned launches. This pass predicts each opponent's
    best response to OUR pass-1 waves and plans from scratch against it:
    waves the reply kills are not just dropped but their ships redirected,
    reinforcements appear against the predicted counter (the reply feeds
    ``friendly_flip_targets``), and every candidate's flow diff carries the
    reply's consequences. The roi threshold is re-normalized by our
    do-nothing-under-reply score, mirroring run_turn's opp-projection shift.

    Skips (returns pass 1 unchanged) when pass 1 fired nothing — the reply
    to an empty plan is what pass 1 already planned against — or when the
    predicted reply is empty (nothing to adapt to).
    """
    pid = int(obs.player_id)
    if int(entries.valid.sum().item()) == 0:
        return entries
    device = obs.device
    dtype = obs.ships.dtype
    mine = _entries_to_launch_set(entries, pid=pid, device=device, dtype=dtype)
    reply = _predict_reply(
        mine,
        movement=movement, obs_tensors=obs_tensors, cache=cache,
        garrison_status=garrison_status, prod=prod, alive_by_step=alive_by_step,
        config=config, player_count=int(player_count), pid=pid,
        K_eta_override=K_eta_override, H=H,
    )
    if int(reply.valid.sum().item()) == 0:
        return entries
    dn = float(_score_do_nothing(
        status=garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), background=reply, player_id=pid,
        opp_weights=opp_weights,
    ))
    cfg2 = dataclasses.replace(
        config, roi_threshold=dn + float(config.roi_threshold),
    )
    return plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=garrison_status, prod=prod,
        alive_by_step=alive_by_step, config=cfg2,
        player_count=int(player_count), K_eta_override=K_eta_override,
        background=reply, opp_weights=opp_weights,
    )


# --- Redirect ---------------------------------------------------------------
# The veto is a filter: when the predicted reply kills a wave, the freed
# ships idle. The full one-ply replan fixed that but measured 2-2 on paired
# seeds with a clear failure mode (decision_diff seed 0: 16 capture-sized
# launches vs the live stack's 24) — pass 2 treats predicted PARRIES as
# fixed background even for attacks it then doesn't make, so the whole plan
# goes conservative. The redirect keeps pass 1 + veto untouched and re-plans
# ONLY the freed budget: surviving waves are committed (sources debited,
# their effects + the reply in the scorer background), and one extra planner
# pass spends what the veto freed on next-best actions. No reopened
# commitments -> no phantom-parry suppression of the plan.


def _redirect_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_REDIRECT", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _redirect_2p_only() -> bool:
    return os.environ.get("PRODUCER_PLUS_REDIRECT_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _redirect_active(player_count: int) -> bool:
    return _redirect_enabled() and (
        (not _redirect_2p_only()) or int(player_count) == 2
    )


def _apply_redirect(
    entries,
    *,
    reply: LaunchSet,
    movement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config,
    player_count: int,
    K_eta_override,
    H: int,
    opp_weights,
):
    """Spend the veto-freed budget on next-best actions. Appends new waves.

    ``entries`` is the post-veto table (some rows invalidated). The surviving
    waves are committed: their sends are debited from the planner's view of
    our garrisons and their effects ride in the scorer background alongside
    the predicted ``reply``, so a second wave at an already-attacked target
    scores ~0 marginal and is naturally suppressed. The roi threshold is
    re-normalized by the do-nothing score under that combined background.
    """
    pid = int(obs.player_id)
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    committed = _entries_to_launch_set(entries, pid=pid, device=device, dtype=dtype)
    if int(committed.source_slots.shape[-1]) > 0:
        debit = torch.zeros_like(obs.ships)
        debit.scatter_add_(
            0, committed.source_slots.clamp(0, P - 1), committed.ships.to(obs.ships.dtype),
        )
        obs2 = dataclasses.replace(obs, ships=(obs.ships - debit).clamp(min=0.0))
        bg2 = _cat_launch_sets([reply, committed])
    else:
        obs2 = obs
        bg2 = reply
    dn = float(_score_do_nothing(
        status=garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), background=bg2, player_id=pid,
        opp_weights=opp_weights,
    ))
    cfg2 = dataclasses.replace(
        config, roi_threshold=dn + float(config.roi_threshold),
    )
    extra = plan_lite_waves(
        movement=movement, obs=obs2, obs_tensors=obs_tensors, cache=cache,
        garrison_status=garrison_status, prod=prod,
        alive_by_step=alive_by_step, config=cfg2,
        player_count=int(player_count), K_eta_override=K_eta_override,
        background=bg2, opp_weights=opp_weights,
    )
    if int(extra.valid.sum().item()) == 0:
        return entries
    return LaunchEntries(
        source_slots=torch.cat([entries.source_slots, extra.source_slots]),
        target_slots=torch.cat([entries.target_slots, extra.target_slots]),
        ships=torch.cat([entries.ships, extra.ships]),
        angle=torch.cat([entries.angle, extra.angle]),
        eta=torch.cat([entries.eta, extra.eta]),
        valid=torch.cat([entries.valid, extra.valid]),
    )


# --- Reply trust --------------------------------------------------------------
# Everything reply-conditioned (the veto, the projection background) assumes
# the rivals run our planner. Against producer-derived opponents that mirror
# is near-exact; against originals it is confidently wrong, and a wrong
# parry prediction vetoes good attacks. Honest fix: VERIFY the model online.
# Each turn, check whether last turn's predicted launches materialized as
# real fleets (matched by source planet + owner, ships within 2x), keep an
# exponential moving accuracy, and price replies at trust-scaled strength
# (certainty-equivalent: a reply believed with p=0.4 carries 0.4x ships).
# Producer-likes: trust stays high, behavior unchanged. Originals: the veto
# degrades gracefully toward the unconditioned stack instead of parrying
# ghosts.


def _reply_trust_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_REPLY_TRUST", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


_REPLY_TRUST_FLOOR = 0.25
_REPLY_TRUST_ALPHA = 0.2


def _record_reply_prediction(memory, background: LaunchSet | None, obs_tensors: dict) -> None:
    """Stash this turn's predicted opp launches + current fleet ids for
    next turn's verification."""
    fleets = obs_tensors["fleets"]
    ids = fleets[..., 0].long()
    memory.trust_fleet_ids = set(int(i) for i in ids.tolist() if int(i) >= 0)
    preds = []
    if background is not None and int(background.source_slots.shape[-1]) > 0:
        planets = obs_tensors["planets"]
        pid_of_slot = planets[..., 0].long()
        sel = background.valid.nonzero(as_tuple=True)[0]
        for i in sel.tolist():
            src_slot = int(background.source_slots[i].item())
            preds.append((
                int(pid_of_slot[src_slot].item()),          # source planet id
                int(background.owner[i].item()),
                float(background.ships[i].item()),
            ))
    memory.trust_predictions = preds


def _update_reply_trust(memory, obs_tensors: dict, *, pid: int) -> float:
    """EMA prediction recall; returns current trust in [floor, 1]."""
    trust = getattr(memory, "trust_ema", None)
    if trust is None:
        trust = 1.0                       # start trusting (the live behavior)
    preds = getattr(memory, "trust_predictions", None)
    known_ids = getattr(memory, "trust_fleet_ids", None)
    if preds:
        fleets = obs_tensors["fleets"]
        new_enemy = []
        for row in fleets.tolist():
            fleet_id, owner, _x, _y, _ang, from_id, ships = row[:7]
            if int(fleet_id) < 0 or int(owner) == int(pid):
                continue
            if known_ids is not None and int(fleet_id) in known_ids:
                continue
            new_enemy.append((int(from_id), int(owner), float(ships)))
        matched = 0
        pool = list(new_enemy)
        for p_src, p_owner, p_ships in preds:
            hit = None
            for j, (f_src, f_owner, f_ships) in enumerate(pool):
                if f_src == p_src and f_owner == p_owner and (
                    0.5 * p_ships <= f_ships <= 2.0 * p_ships
                ):
                    hit = j
                    break
            if hit is not None:
                matched += 1
                pool.pop(hit)
        recall = matched / len(preds)
        trust = (1.0 - _REPLY_TRUST_ALPHA) * trust + _REPLY_TRUST_ALPHA * recall
    memory.trust_ema = trust
    return max(_REPLY_TRUST_FLOOR, min(1.0, trust))


def _scale_launch_set_ships(launches: LaunchSet, factor: float) -> LaunchSet:
    if factor >= 1.0:
        return launches
    return LaunchSet(
        source_slots=launches.source_slots, target_slots=launches.target_slots,
        ships=launches.ships * float(factor), eta=launches.eta,
        owner=launches.owner, valid=launches.valid,
    )


# --- Opening search ----------------------------------------------------------
# The pre-contact opening is a deterministic single-player scheduling problem
# (PI thesis; neutral garrisons are static, planet motion is rigid rotation,
# production compounds, and the total-ship lead stops changing hands by step
# 30-54 — so opening production IS the game). The greedy planner expands "by
# accident" (horizon-truncated capture payoffs). This ports the beam search
# from scripts/opening_optimum.py in-agent: each turn while step < window,
# search capture schedules maximizing total production by the opening
# horizon, and emit the launches due NOW. The rest of the pipeline (defense
# lane, veto, regroup) runs as usual on the remaining budget. Pure Python,
# time-boxed; turn budget headroom is ~800 ms.


def _opening_search_window() -> int:
    return max(0, _env_int("PRODUCER_PLUS_OPENING_SEARCH", 0))


def _opening_search_horizon() -> int:
    return max(10, _env_int("PRODUCER_PLUS_OPENING_HORIZON", 40))


def _opening_search_beam() -> int:
    return max(8, _env_int("PRODUCER_PLUS_OPENING_BEAM", 64))


_OPENING_TIMEBOX_S = 0.25
_LOG1000 = math.log(1000.0)
_BOARD_CENTER = 50.0


def _fleet_speed_py(s: float) -> float:
    if s <= 1:
        return 1.0
    return 1.0 + 5.0 * (math.log(min(s, 1000.0)) / _LOG1000) ** 1.5


class _OpeningBoard:
    """Deterministic kinematics + static garrisons from the CURRENT obs.

    t=0 is *now*: angles are taken from current positions, so re-planning
    every turn stays consistent as planets rotate.
    """

    def __init__(self, obs_tensors: dict, pid: int):
        planets = obs_tensors["planets"].detach().cpu()
        self.angvel = float(obs_tensors["angular_velocity"].flatten()[0].item())
        self.planets: dict[int, dict] = {}
        self.mine: list[int] = []
        self.enemy: list[int] = []
        self.neutrals: list[int] = []
        for row in planets.tolist():
            planet_id, owner, x, y, r, ships, prod = row[:7]
            planet_id = int(planet_id)
            if planet_id < 0:
                continue
            ox, oy = x - _BOARD_CENTER, y - _BOARD_CENTER
            orb_r = math.hypot(ox, oy)
            self.planets[planet_id] = dict(
                owner=int(owner), r=float(r), ships=float(ships),
                prod=float(prod), orb_r=orb_r, a0=math.atan2(oy, ox),
                orbiting=(orb_r + float(r)) < _BOARD_CENTER,
            )
            if int(owner) == int(pid):
                self.mine.append(planet_id)
            elif int(owner) >= 0:
                self.enemy.append(planet_id)
            else:
                self.neutrals.append(planet_id)

    def pos(self, planet_id: int, t: float):
        p = self.planets[planet_id]
        a = p["a0"] + (self.angvel * t if p["orbiting"] else 0.0)
        return (_BOARD_CENTER + p["orb_r"] * math.cos(a),
                _BOARD_CENTER + p["orb_r"] * math.sin(a))

    def eta(self, src: int, tgt: int, size: float, t: float) -> int:
        sp = _fleet_speed_py(size)
        sx, sy = self.pos(src, t)
        e = 1.0
        for _ in range(4):
            tx, ty = self.pos(tgt, t + e)
            d = math.hypot(tx - sx, ty - sy) - self.planets[tgt]["r"]
            e = max(1.0, d / sp)
        return max(1, math.ceil(e))


def _opening_search_plan(
    obs_tensors: dict, *, pid: int, claimed: set[int],
    horizon: int, beam_width: int, timebox_s: float = _OPENING_TIMEBOX_S,
) -> list[tuple[int, int, float]]:
    """Beam-search the capture schedule; return launches due NOW.

    Returns ``[(src_planet_id, tgt_planet_id, size)]`` for schedule entries
    with launch time 0. ``claimed`` excludes neutrals already targeted by
    our in-flight opening waves (they're treated as spoken for).
    Safe-only filter: only neutrals at least as reachable by us as by any
    enemy planet (race-losing targets are the midgame planner's problem).
    """
    board = _OpeningBoard(obs_tensors, pid)
    if not board.mine:
        return []
    neutrals = []
    safe_margin = float(_env_int("PRODUCER_PLUS_OPENING_SAFE_MARGIN", 3))
    for n in board.neutrals:
        if n in claimed:
            continue
        g = board.planets[n]["ships"] + 1.0
        ours = min(board.eta(s, n, g, 0) for s in board.mine)
        if board.enemy:
            # Contested-race guard (canon: safe/contested/unsafe neutrals;
            # measured: lane losses on collapse seeds are race losses,
            # 182 wasted ships vs the referee's converging fleets). The
            # bare ours<=theirs test uses the enemy's CURRENT planets,
            # but their reach grows as they expand — demand a margin.
            theirs = min(board.eta(s, n, g, 0) for s in board.enemy)
            if ours + safe_margin > theirs:
                continue
        neutrals.append(n)
    if not neutrals:
        return []

    t_start = time.perf_counter()
    # State: (t, owned dict items, captured frozenset, produced, flights, plan)
    start = (0.0, tuple((p, board.planets[p]["ships"]) for p in board.mine),
             frozenset(), 0.0, (), ())

    def advance(state, until):
        t, owned_t, captured, produced, flights, plan = state
        owned = dict(owned_t)
        fl = sorted(flights)
        while t < until:
            step_to = until
            if fl and fl[0][0] < step_to:
                step_to = fl[0][0]
            dt = step_to - t
            for p in owned:
                owned[p] += board.planets[p]["prod"] * dt
            produced += sum(board.planets[p]["prod"] for p in owned) * dt
            t = step_to
            while fl and fl[0][0] <= t:
                _at, tgt, size = fl.pop(0)
                g = board.planets[tgt]["ships"]
                owned[tgt] = max(1.0, size - g)
                captured = captured | {tgt}
        return (t, tuple(sorted(owned.items())), captured, produced,
                tuple(fl), plan)

    def held_value(state):
        fin = advance(state, float(horizon))
        return fin[3]

    best_value = held_value(start)
    best_plan: tuple = ()
    frontier = [start]
    for _depth in range(10):
        if time.perf_counter() - t_start > timebox_s:
            break
        nxt = []
        for state in frontier:
            t, owned_t, captured, produced, flights, plan = state
            owned = dict(owned_t)
            for n in neutrals:
                if n in captured or n in owned:
                    continue
                g = board.planets[n]["ships"] + 1.0
                for src in owned:
                    have = owned[src]
                    prod_src = board.planets[src]["prod"]
                    need = g + 1.0          # keep 1 ship home
                    wait = 0.0 if have >= need else (
                        math.inf if prod_src <= 0
                        else math.ceil((need - have) / prod_src))
                    t_launch = t + wait
                    if t_launch >= horizon:
                        continue
                    e = board.eta(src, n, g, t_launch)
                    if t_launch + e >= horizon + 10:
                        continue
                    s2 = advance(state, t_launch)
                    t2, owned2_t, cap2, prod2, fl2, plan2 = s2
                    owned2 = dict(owned2_t)
                    if owned2.get(src, 0.0) < need:
                        continue
                    owned2[src] -= g
                    fl3 = tuple(sorted(fl2 + ((t_launch + e, n, g),)))
                    plan3 = plan2 + ((t_launch, src, n, g),)
                    nxt.append((t2, tuple(sorted(owned2.items())), cap2,
                                prod2, fl3, plan3))
        if not nxt:
            break

        def h(s):
            t, owned_t, _cap, produced, fl, _plan = s
            rate = sum(board.planets[p]["prod"] for p, _ in owned_t)
            opt = produced + rate * (horizon - t)
            for at, tgt, _sz in fl:
                if at < horizon:
                    opt += board.planets[tgt]["prod"] * (horizon - at)
            return opt

        nxt.sort(key=h, reverse=True)
        frontier = nxt[:beam_width]
        for state in frontier:
            v = held_value(state)
            if v > best_value:
                best_value = v
                best_plan = state[5]

    return [(src, tgt, size) for (t_launch, src, tgt, size) in best_plan
            if t_launch <= 0.5]


def _opening_reserve_k() -> int:
    """Worst-case reserve window in turns (0 = off). Planet Wars canon
    (Melis's full-attack future): ships may leave only if the source
    survives a POSSIBLE strike, not just the fleets already in flight —
    the do-nothing projection is blind pre-contact, which is exactly when
    the searcher launches. The reserve = enemy garrison mass that could
    reach the source within this window (full-garrison fleet speed)."""
    return max(0, _env_int("PRODUCER_PLUS_OPENING_RESERVE_K", 8))


def _opening_reserve_filter(
    rows: list[tuple[int, int, float]],
    ships_by_slot: dict[int, float],
    reserve_by_slot: dict[int, float],
) -> list[tuple[int, int, float]]:
    """Drop launches whose source would dip below its worst-case reserve."""
    return [
        (s, t, size) for (s, t, size) in rows
        if ships_by_slot.get(s, 0.0) - size >= reserve_by_slot.get(s, 0.0)
    ]


def _opening_hold_filter(
    rows: list[tuple[int, int, float]], drain_by_slot: dict[int, float],
) -> list[tuple[int, int, float]]:
    """Drop scheduled launches the source can't afford under hold discipline.

    The searcher's keep-1-home rule is a single-player safety model — its
    first measured composition stripped sources bare and was punished
    (attribution leg: -27% @120, one map dead by step 115). A capture wave
    is all-or-nothing: clamping below the garrison floor just annihilates,
    so unaffordable launches are SKIPPED (the per-turn re-plan retries when
    the garrison has grown).
    """
    return [
        (s, t, size) for (s, t, size) in rows
        if size <= drain_by_slot.get(s, 0.0)
    ]


def _emit_opening_entries(
    due: list[tuple[int, int, float]], *, movement, obs, obs_tensors: dict,
    garrison_status, H: int, cache=None,
):
    """Aim the due launches with the REAL intercept solver. LaunchEntries."""
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    planet_ids = obs_tensors["planets"][..., 0].long()
    P = int(obs.P)
    slot_of = {int(planet_ids[i].item()): i for i in range(P)}
    rows = []
    for src_pid, tgt_pid, size in due:
        s, t = slot_of.get(src_pid), slot_of.get(tgt_pid)
        if s is None or t is None:
            continue
        size = float(min(size, max(float(obs.ships[s].item()) - 1.0, 0.0)))
        if size < 1.0:
            continue
        rows.append((s, t, size))
    if rows:
        src_slots = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
        drains = safe_drain(
            garrison_status, source_idx=src_slots,
            source_ships=obs.ships[src_slots].to(dtype),
            H_eff=torch.full((), float(H), dtype=dtype, device=device),
            player_id=pid,
        )
        drain_by_slot = {
            int(src_slots[i].item()): float(drains[i].item())
            for i in range(int(src_slots.shape[0]))
        }
        rows = _opening_hold_filter(rows, drain_by_slot)
    _rk = _opening_reserve_k()
    if rows and _rk > 0 and cache is not None:
        src_slots = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
        margin = _reactive_reinforcement_margin(
            obs, cache, src_slots, _rk, weight=1.0, lag=0.0,
        )
        if margin is not None:
            reserve_by_slot = {
                int(src_slots[i].item()): float(margin[i, _rk - 1].item())
                for i in range(int(src_slots.shape[0]))
            }
            ships_by_slot = {
                int(src_slots[i].item()): float(obs.ships[src_slots[i]].item())
                for i in range(int(src_slots.shape[0]))
            }
            rows = _opening_reserve_filter(rows, ships_by_slot, reserve_by_slot)
    if not rows:
        return None
    src = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
    tgt = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    ships = torch.tensor([r[2] for r in rows], dtype=dtype, device=device)
    aim = intercept_angle(movement, src, tgt, ships)
    ok = aim["viable"]
    if not bool(ok.any()):
        return None
    return LaunchEntries(
        source_slots=src, target_slots=tgt, ships=ships,
        angle=aim["angle"].to(dtype), eta=aim["eta"].to(dtype),
        valid=ok,
    )


# --- Neutral shortlist quota -----------------------------------------------------
# The offensive shortlist is the N nearest enemy-or-neutral planets with NO
# class quota: once a frontline forms, the nearest non-owned planets are
# mostly ENEMY planets and neutral expansion targets are crowded out of the
# candidate set entirely (seed-7 probe: neutral candidate counts collapse
# from 20-45 to 0-8 at first contact; SiestaGuru loss: zero neutral captures
# for 40 steps). A visibility defect independent of valuation — the scorer
# never sees the option. The quota appends the Q nearest neutral targets not
# already shortlisted.


def _neutral_shortlist_quota() -> int:
    return max(0, _env_int("PRODUCER_PLUS_NEUTRAL_SHORTLIST", 0))


def _append_neutral_quota(
    target_idx: Tensor, target_exists: Tensor, *, obs, cache, source_mask,
    K_eta: int, quota: int,
):
    """Append the nearest `quota` neutral targets missing from the shortlist."""
    neutral_mask = obs.is_neutral & obs.alive
    if not bool(neutral_mask.any()):
        return target_idx, target_exists
    proximity = min_distance_to_targets(cache, source_mask, neutral_mask, max_k=int(K_eta))
    pref = torch.where(
        neutral_mask, -proximity, torch.full_like(proximity, float("-inf")))
    n_idx, n_exists = _candidate_indices(pref, neutral_mask, int(quota))
    dup = (n_idx.view(-1, 1) == target_idx.view(1, -1)).any(dim=-1)
    n_exists = n_exists & ~dup
    return (
        torch.cat([target_idx, n_idx], dim=0),
        torch.cat([target_exists, n_exists], dim=0),
    )


# --- Reactive floor -------------------------------------------------------------
# capture_floor's `reinforcement` margin hook has been dormant (always None):
# enemy floors assume the defender's garrison sits still while our fleet
# flies. SiestaGuru loss anatomy (episode 79438024): 9 capture-sized strikes
# failed — 700 ships — to defense routed in during our eta-7..8 flights,
# which the 1-ply response veto cannot see either. This margin adds, per
# target t and arrival turn k, the garrison the defender can ROUTE to t by
# k: enemy planets q whose travel q->t fits within (k − reaction lag), at
# the speed of their full garrison (big fleets fly faster). Weight scales
# the margin (1.0 = assume full rerouting).


def _reactive_floor_weight() -> float:
    raw = os.environ.get("PRODUCER_PLUS_REACTIVE_FLOOR", "0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _reactive_floor_for(player_count: int) -> float:
    """Player-count gate (same pattern as the veto's): a composed bundle can
    run the reactive floor in 2P while keeping 4P byte-identical."""
    only2p = os.environ.get(
        "PRODUCER_PLUS_REACTIVE_FLOOR_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on")
    if only2p and int(player_count) != 2:
        return 0.0
    return _reactive_floor_weight()


def _reactive_floor_lag() -> float:
    """Reaction lag in turns before rerouted defense starts counting
    (PRODUCER_PLUS_REACTIVE_FLOOR_LAG, default 2.0 — the value the floor
    shipped with; exposed for joint knob tuning)."""
    raw = os.environ.get("PRODUCER_PLUS_REACTIVE_FLOOR_LAG", "2.0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 2.0


def _reactive_reinforcement_margin(
    obs, cache, target_idx: Tensor, K: int, *, weight: float, lag: float | None = None,
):
    """``[T, K]`` reroutable enemy support per target/arrival-turn, or None."""
    if lag is None:
        lag = _reactive_floor_lag()
    enemy = obs.is_enemy & obs.alive
    q_idx = enemy.nonzero(as_tuple=True)[0]
    Q = int(q_idx.shape[0])
    T = int(target_idx.shape[0])
    if Q == 0 or T == 0 or K <= 0:
        return None
    dtype = obs.ships.dtype
    g_q = obs.ships[q_idx].to(dtype).clamp(min=1.0)                  # [Q]
    speed_q = fleet_speed(g_q)                                       # [Q]
    d = cache.cross_dist[0][q_idx][:, target_idx.clamp(0, int(obs.P) - 1)]  # [Q, T]
    eta_qt = d / speed_q.unsqueeze(-1)                               # [Q, T]
    k_grid = torch.arange(1, K + 1, device=obs.device, dtype=dtype)  # [K]
    reach = eta_qt.unsqueeze(-1) <= (k_grid.view(1, 1, K) - float(lag))
    # The target's own garrison is already the defender — exclude q == t.
    self_mask = q_idx.view(Q, 1) == target_idx.view(1, T)
    reach = reach & ~self_mask.unsqueeze(-1)
    support = (g_q.view(Q, 1, 1) * reach.to(dtype)).sum(dim=0)       # [T, K]
    return float(weight) * support


# --- Forward redistribution ----------------------------------------------------


def _regroup_forward_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_REGROUP_FORWARD", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _regroup_forward_time(default: float) -> float:
    raw = os.environ.get("PRODUCER_PLUS_REGROUP_FORWARD_TIME")
    if raw is None or not raw.strip():
        return max(float(default), 12.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return max(float(default), 12.0)


# --- 2P-only gate for the mass mechanisms ------------------------------------
# Local evidence splits by player count: mass beats the champion head-to-head
# in 2P (35/64) and holds vs producer (22/32), but costs first-place rate in
# the 4P pool. With this gate set, MASS_TIEBREAK / REGROUP_MIN_SEND /
# OVERKILL_FACTOR apply only when player_count == 2; 3+ player games keep
# champion behavior (and compose with the 4P-only FFA objective fix).
def _mass_2p_only() -> bool:
    return os.environ.get("PRODUCER_PLUS_MASS_2P_ONLY", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _mass_active(player_count: int) -> bool:
    return (not _mass_2p_only()) or int(player_count) == 2


def _overkill_factor_for(player_count: int) -> float:
    return _overkill_factor() if _mass_active(player_count) else 1.0


# --- Class-split overkill -----------------------------------------------------
# Replay mining of the top-3 teams (mine_decision_rules.py, appended to
# audit/2026-06-10-top-ladder-behavior.md): attack sizing is CLASS-dependent —
# ~1.3x the garrison on neutral targets (cheap, front-loaded expansion) but
# 2.6-4.6x at median (7.5-10x at p75) on enemy planets, with 60-89-ship
# median fleets. A single overkill factor over-sizes neutral grabs and
# under-sizes enemy strikes. Unset -> the single-knob path, bit-identical.


def _overkill_factor_enemy() -> float | None:
    raw = os.environ.get("PRODUCER_PLUS_OVERKILL_FACTOR_ENEMY")
    if raw is None or not raw.strip():
        return None
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return None


def _overkill_for_targets(obs, target_idx: Tensor, player_count: int, dtype):
    """Scalar (legacy) or per-target ``[T]`` overkill multiplier for sizes_lo."""
    base = _overkill_factor_for(player_count)
    enemy = _overkill_factor_enemy() if _mass_active(player_count) else None
    if enemy is None:
        return base
    is_enemy_t = obs.is_enemy[target_idx.clamp(0, int(obs.P) - 1)]
    return torch.where(
        is_enemy_t,
        torch.full_like(is_enemy_t, enemy, dtype=dtype),
        torch.full_like(is_enemy_t, base, dtype=dtype),
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def compute_k_eta_for_step(step: int, *, H: int) -> int:
    H_int = max(1, int(H))
    if not _adaptive_k_enabled():
        return H_int
    floor = _env_int("PRODUCER_PLUS_ADAPTIVE_K_FLOOR", 10)
    k_open = _env_int("PRODUCER_PLUS_ADAPTIVE_K_OPEN", 20)
    t_settle = _env_int("PRODUCER_PLUS_ADAPTIVE_K_TSETTLE", 30)
    floor = max(1, floor)
    if t_settle <= 0 or int(step) >= t_settle or k_open <= floor:
        decayed = floor
    else:
        raw = k_open - (k_open - floor) * int(step) / float(t_settle)
        decayed = max(floor, int(round(raw)))
    return max(1, min(H_int, decayed))


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs.  """

    
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12         # enemy/neutral proximity targets
    max_defensive_targets: int = 4          
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5              # fire if score > this
    min_ships_to_launch: float = 4.0
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``.

    Consumed only as the **regroup gradient** (rank owned planets by how stressed
    they are, move ships up the gradient). For each planet ``t``, sums a
    distance-decayed share of every enemy source's **current** garrison that could
    straight-line reach ``t`` within ``horizon`` turns, using the step-0 centre
    distance ``cross_dist[0]``. The decay ``(1 - d/(speed·H))₊`` weights nearer
    enemies more, giving a graded frontline signal in ship-mass units.

    Approximations: ignores target orbital drift over the horizon, production
    accrued in flight, the per-owner split, and in-flight enemy fleets. Pure
    arithmetic on cached tensors
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)    # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye              # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)                       # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)                                            # [P] summed over sources


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    K_eta_override: int | None = None,
    background: LaunchSet | None = None,
    force_concentration: bool | None = None,
    opp_weights: Tensor | None = None,
    sync_sink: list | None = None,
):
    """Single-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta_raw = int(K_eta_override) if K_eta_override is not None else int(config.horizon)
    K_eta = max(1, min(K_eta_raw, H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    # Background-aware floors: the sizing subsystem (shortlist flips, drain,
    # capture floors) reads trajectories with the predicted opponent launches
    # applied; the SCORER below keeps the static baseline because it merges
    # the background into every candidate's diff itself. ``bg_flip=None``
    # because the adjusted trajectories already contain the predicted flips.
    status_sizing = garrison_status
    bg_flip = background
    if (
        _bg_floors_enabled() and background is not None
        and int(background.source_slots.shape[-1]) > 0
        and bool(background.valid.any())
    ):
        status_sizing = _background_adjusted_status(
            garrison_status, background=background, prod=prod,
            alive_by_step=alive_by_step,
        )
        bg_flip = None
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, status_sizing, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
        background=bg_flip,
    )
    _nq = _neutral_shortlist_quota()
    if _nq > 0:
        target_idx, target_exists = _append_neutral_quota(
            target_idx, target_exists, obs=obs, cache=cache,
            source_mask=source_mask, K_eta=K_eta, quota=_nq,
        )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        status_sizing, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )                                                                            # [S]
    _ss_allow = _source_safety_allowance(
        obs, cache, source_idx=source_idx, prod=prod, K=int(K_eta),
    )
    if _ss_allow is not None:
        # Second cap: keep every source locally defensible against the
        # enemy's UNCOMMITTED reserve (safe_drain only sees in-flight).
        drain = torch.minimum(drain, _ss_allow)

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]

    _rf_w = _reactive_floor_for(int(player_count))
    _rf_margin = (
        _reactive_reinforcement_margin(
            obs, cache, target_idx, int(K_eta), weight=_rf_w,
        ) if _rf_w > 0.0 else None
    )
    floor = capture_floor(
        status_sizing, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
        reinforcement=_rf_margin,
    )                                                                            # [T, K]
    if _reinforce_deficit_enabled():
        floor = _apply_reinforce_deficit_floor(
            floor, garrison_status=status_sizing, target_idx=target_idx,
            player_id=pid, capture_overhead=1.0,
        )
    K = int(floor.shape[-1])

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)

    if _multi_size_enabled() and _coalitions_enabled():
        # --- Step 4 + Step 5 composed: multi-size single-source + L=2 coalitions
        # Single-source: 3 size variants per (s, t), padded to L=2 with slot 1
        # inactive. Coalitions: safe_drain per contributor (no size variation
        # on the coalition side — that's the controlled compose), L=2.
        # C_total = S*T*N + T*C(K_src, 2). Target mutex blocks losing
        # variants/coalitions once any candidate wins a target.
        L = 2
        N = 3

        # ===== Stage A: Step 4 multi-size variants =====
        sizes_hi = drain.view(S, 1).expand(S, T).floor()                          # [S, T]
        active_hi = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_hi.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)
        aim_hi = intercept_angle(
            movement,
            source_idx.unsqueeze(1), target_idx.unsqueeze(0),
            sizes_hi, active=active_hi,
        )
        eta_hi = aim_hi["eta"]                                                    # [S, T]
        if K > 0:
            k_arr_hi = (eta_hi.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr_hi = (
                floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr_hi.unsqueeze(-1)).squeeze(-1)
            )
        else:
            floor_at_arr_hi = torch.ones(S, T, dtype=dtype, device=device)

        sizes_lo = torch.minimum((floor_at_arr_hi * _overkill_for_targets(obs, target_idx, player_count, dtype)).ceil().clamp(min=1.0), sizes_hi)
        sizes_mid = torch.minimum(2.0 * sizes_lo, sizes_hi)
        sizes_3 = torch.stack([sizes_lo, sizes_mid, sizes_hi], dim=-1)            # [S, T, N]

        active_3 = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_3, eta_cap=eta_cap,
        )
        aim_3 = intercept_angle(
            movement,
            source_idx.view(S, 1, 1), target_idx.view(1, T, 1),
            sizes_3, active=active_3,
        )
        angle_3 = aim_3["angle"]                                                  # [S, T, N]
        eta_3 = aim_3["eta"]                                                      # [S, T, N]
        viable_3 = aim_3["viable"] & (eta_3 <= eta_cap.view(1, T, 1))

        if K > 0:
            k_arr_3 = (eta_3.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr_3 = (
                floor.view(1, T, 1, K).expand(S, T, N, K)
                .gather(-1, k_arr_3.unsqueeze(-1)).squeeze(-1)
            )
        else:
            k_arr_3 = torch.zeros(S, T, N, dtype=torch.long, device=device)
            floor_at_arr_3 = torch.ones(S, T, N, dtype=dtype, device=device)
        clears_floor_3 = sizes_3 >= floor_at_arr_3
        ships_ok_3 = sizes_3 <= source_ships.view(S, 1, 1)
        # Leg-level viability WITHOUT the floor gate (for Fix-A coalitions).
        viable_only_3 = (
            viable_3 & (sizes_3 >= 1.0) & ships_ok_3
            & src_neq_tgt.unsqueeze(-1)
            & source_exists.view(S, 1, 1) & target_exists.view(1, T, 1)
        )                                                                         # [S, T, N]
        valid_3 = viable_only_3 & clears_floor_3                                  # [S, T, N]

        # Pack multi-size single-source into [C_ms, L=2] padded.
        C_ms = S * T * N
        ms_src_planet = source_idx.view(S, 1, 1).expand(S, T, N).reshape(C_ms)
        ms_src_padded = torch.stack([ms_src_planet, ms_src_planet], dim=-1)       # [C_ms, 2]
        ms_send_flat = torch.where(valid_3, sizes_3, torch.zeros_like(sizes_3)).reshape(C_ms)
        ms_send_padded = torch.stack(
            [ms_send_flat, torch.zeros_like(ms_send_flat)], dim=-1)
        ms_angle_padded = torch.stack(
            [angle_3.reshape(C_ms), torch.zeros_like(ms_send_flat)], dim=-1)
        ms_eta_flat = torch.where(valid_3, eta_3, torch.ones_like(eta_3)).reshape(C_ms)
        ms_eta_padded = torch.stack([ms_eta_flat, torch.ones_like(ms_eta_flat)], dim=-1)
        ms_valid_flat = valid_3.reshape(C_ms)                                     # [C_ms]
        ms_active = torch.stack(
            [ms_valid_flat, torch.zeros_like(ms_valid_flat)], dim=-1)
        ms_tgt_slot = target_idx.view(1, T, 1).expand(S, T, N).reshape(C_ms)
        ms_tgt_short = torch.arange(T, device=device).view(1, T, 1).expand(S, T, N).reshape(C_ms)

        # ===== Stage B: Step 5 coalitions (safe_drain per contributor) =====
        # Source ranking + per-pair aim use the safe_drain variant (sizes_hi).
        # Use viable_only_base (not valid_base) for ranking so sources that
        # CAN'T clear floor alone still enter the coalition pool — those are
        # the ones a Fix-A coalition actually helps.
        viable_only_base = viable_only_3[..., -1]                                 # [S, T]
        clears_floor_base = clears_floor_3[..., -1]                               # [S, T]
        eta_base = eta_3[..., -1]                                                 # [S, T]
        angle_base = angle_3[..., -1]                                             # [S, T]
        k_arr_base = k_arr_3[..., -1]                                             # [S, T]
        floor_at_arr_base = floor_at_arr_3[..., -1]                               # [S, T]

        K_src = min(_env_int("PRODUCER_PLUS_COALITION_K", 6), int(S))
        K_src = max(0, K_src)

        if S >= 2 and K_src >= 2:
            ranked_per_tgt = torch.where(
                viable_only_base, -eta_base, torch.full_like(eta_base, float("-inf"))
            ).transpose(0, 1)
            top_src_per_tgt = _stable_topk_indices(ranked_per_tgt, K_src)
            pair_idx = torch.triu_indices(K_src, K_src, offset=1, device=device)
            pair_a = pair_idx[0]
            pair_b = pair_idx[1]
            P_pairs = int(pair_a.numel())
        else:
            P_pairs = 0

        if P_pairs > 0:
            T_idx_pair = torch.arange(T, device=device).view(T, 1).expand(T, P_pairs)
            sA = top_src_per_tgt[:, pair_a]
            sB = top_src_per_tgt[:, pair_b]
            sizesA = sizes_hi[sA, T_idx_pair]
            sizesB = sizes_hi[sB, T_idx_pair]
            etaA = eta_base[sA, T_idx_pair]
            etaB = eta_base[sB, T_idx_pair]
            angleA = angle_base[sA, T_idx_pair]
            angleB = angle_base[sB, T_idx_pair]
            viableA = viable_only_base[sA, T_idx_pair]
            viableB = viable_only_base[sB, T_idx_pair]
            clearsA = clears_floor_base[sA, T_idx_pair]
            clearsB = clears_floor_base[sB, T_idx_pair]
            k_arr_A = k_arr_base[sA, T_idx_pair]
            k_arr_B = k_arr_base[sB, T_idx_pair]
            floor_joint = floor_at_arr_base[sA, T_idx_pair]   # == [sB] when k_arr_A==k_arr_B

            # Fix-A gate: coalitions only fire on targets where NEITHER source
            # clears floor alone, but their joint same-tick arrival does.
            eta_strict = k_arr_A == k_arr_B
            neither_alone = ~clearsA & ~clearsB
            joint_clears = (sizesA + sizesB) >= floor_joint
            distinct_src = sA != sB
            valid_pair = (
                viableA & viableB & neither_alone & joint_clears
                & eta_strict & distinct_src
            )                                                                    # [T, P_pairs]

            C_coal = T * P_pairs
            coal_src = torch.stack(
                [source_idx[sA].reshape(C_coal), source_idx[sB].reshape(C_coal)],
                dim=-1,
            )
            sendA = torch.where(valid_pair, sizesA, torch.zeros_like(sizesA))
            sendB = torch.where(valid_pair, sizesB, torch.zeros_like(sizesB))
            coal_send = torch.stack(
                [sendA.reshape(C_coal), sendB.reshape(C_coal)], dim=-1)
            coal_angle = torch.stack(
                [angleA.reshape(C_coal), angleB.reshape(C_coal)], dim=-1)
            etaA_safe = torch.where(valid_pair, etaA, torch.ones_like(etaA))
            etaB_safe = torch.where(valid_pair, etaB, torch.ones_like(etaB))
            coal_eta = torch.stack(
                [etaA_safe.reshape(C_coal), etaB_safe.reshape(C_coal)], dim=-1)
            coal_active = torch.stack(
                [valid_pair.reshape(C_coal), valid_pair.reshape(C_coal)], dim=-1)
            coal_tgt_short = torch.arange(T, device=device).view(T, 1).expand(T, P_pairs).reshape(C_coal)
            coal_tgt_slot = target_idx[coal_tgt_short]
            coal_valid = valid_pair.reshape(C_coal)

            cand_src = torch.cat([ms_src_padded, coal_src], dim=0)
            cand_send = torch.cat([ms_send_padded, coal_send], dim=0)
            cand_angle = torch.cat([ms_angle_padded, coal_angle], dim=0)
            cand_eta = torch.cat([ms_eta_padded, coal_eta], dim=0)
            cand_active = torch.cat([ms_active, coal_active], dim=0)
            cand_tgt_slot = torch.cat([ms_tgt_slot, coal_tgt_slot], dim=0)
            cand_tgt_short = torch.cat([ms_tgt_short, coal_tgt_short], dim=0)
            cand_valid = torch.cat([ms_valid_flat, coal_valid], dim=0)
        else:
            cand_src = ms_src_padded
            cand_send = ms_send_padded
            cand_angle = ms_angle_padded
            cand_eta = ms_eta_padded
            cand_active = ms_active
            cand_tgt_slot = ms_tgt_slot
            cand_tgt_short = ms_tgt_short
            cand_valid = ms_valid_flat

        cand_is_def = target_is_mine[cand_tgt_short]
        C = int(cand_src.shape[0])
    elif _coalitions_enabled():
        # --- Step 5: single-size base + L=2 multi-source coalitions ------------
        # Stage 1 — per-(s, t) single-size base (mirrors the else: branch).
        sizes = drain.view(S, 1).expand(S, T).floor()                            # [S, T]
        active = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)                                                            # [S, T]
        aim = intercept_angle(
            movement,
            source_idx.unsqueeze(1),
            target_idx.unsqueeze(0),
            sizes,
            active=active,
        )
        angle = aim["angle"]                                                     # [S, T]
        eta = aim["eta"]                                                         # [S, T]
        viable = aim["viable"] & (eta <= eta_cap.view(1, T))
        if K > 0:
            k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr = (
                floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
            )
        else:
            k_arr = torch.zeros(S, T, dtype=torch.long, device=device)
            floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
        clears_floor = sizes >= floor_at_arr
        # Leg-level viability WITHOUT the floor gate. Used by Fix-A coalitions:
        # a coalition needs both legs to be physically viable (aim works, src
        # exists, src != tgt, ships >= 1) but is NOT required to clear floor
        # alone — that's the whole point of overlapping fleets.
        viable_only = (
            viable & (sizes >= 1.0) & src_neq_tgt
            & source_exists.view(S, 1) & target_exists.view(1, T)
        )                                                                        # [S, T]
        valid = viable_only & clears_floor                                       # [S, T]

        L = 2
        C_base = S * T

        # Stage 2 — top-K_src viable sources per target, ranked by -eta (fast
        # arrivers first). Use `viable_only` (not `valid`) so sources that
        # CAN'T clear floor alone still enter the coalition pool — those are
        # the ones a coalition actually helps.
        K_src = min(_env_int("PRODUCER_PLUS_COALITION_K", 6), int(S))
        K_src = max(0, K_src)
        if S >= 2 and K_src >= 2:
            ranked_per_tgt = torch.where(
                viable_only, -eta, torch.full_like(eta, float("-inf"))
            ).transpose(0, 1)                                                    # [T, S]
            top_src_per_tgt = _stable_topk_indices(ranked_per_tgt, K_src)        # [T, K_src]

            # Stage 3 — enumerate (a < b) pairs across the K_src pool.
            pair_idx = torch.triu_indices(K_src, K_src, offset=1, device=device) # [2, P_pairs]
            pair_a = pair_idx[0]
            pair_b = pair_idx[1]
            P_pairs = int(pair_a.numel())
        else:
            P_pairs = 0

        if P_pairs > 0:
            T_idx = torch.arange(T, device=device).view(T, 1).expand(T, P_pairs)
            sA = top_src_per_tgt[:, pair_a]                                      # [T, P_pairs]
            sB = top_src_per_tgt[:, pair_b]
            sizesA = sizes[sA, T_idx]                                            # [T, P_pairs]
            sizesB = sizes[sB, T_idx]
            etaA = eta[sA, T_idx]
            etaB = eta[sB, T_idx]
            angleA = angle[sA, T_idx]
            angleB = angle[sB, T_idx]
            viableA = viable_only[sA, T_idx]
            viableB = viable_only[sB, T_idx]
            clearsA = clears_floor[sA, T_idx]
            clearsB = clears_floor[sB, T_idx]
            k_arr_A = k_arr[sA, T_idx]
            k_arr_B = k_arr[sB, T_idx]
            floor_joint = floor_at_arr[sA, T_idx]   # == floor_at_arr[sB, T_idx] when k_arr_A == k_arr_B

            # Stage 4 — Fix-A gate: coalitions only cover targets where NEITHER
            # single source can clear floor alone, but the combined same-tick
            # arrival can. This makes coalitions a true superset extension
            # rather than overlap with single-source captures.
            eta_strict = k_arr_A == k_arr_B
            neither_alone = ~clearsA & ~clearsB
            joint_clears = (sizesA + sizesB) >= floor_joint
            distinct_src = sA != sB
            valid_pair = (
                viableA & viableB & neither_alone & joint_clears
                & eta_strict & distinct_src
            )                                                                    # [T, P_pairs]

        # Stage 5 — pack [C_total, L=2]. Single-source candidates pad slot 1
        # with active=False; greedy's `~cand_active` short-circuit + the
        # send=0 mask make padded slots no-op.
        base_src_planet = source_idx.view(S, 1).expand(S, T).reshape(C_base)
        base_src_padded = torch.stack(
            [base_src_planet, base_src_planet], dim=-1)                          # [C_base, 2]
        base_send_flat = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C_base)
        base_send_padded = torch.stack(
            [base_send_flat, torch.zeros_like(base_send_flat)], dim=-1)
        base_angle_padded = torch.stack(
            [angle.reshape(C_base), torch.zeros_like(base_send_flat)], dim=-1)
        base_eta_flat = torch.where(valid, eta, torch.ones_like(eta)).reshape(C_base)
        base_eta_padded = torch.stack(
            [base_eta_flat, torch.ones_like(base_eta_flat)], dim=-1)
        base_valid_flat = valid.reshape(C_base)                                  # [C_base]
        base_active = torch.stack(
            [base_valid_flat, torch.zeros_like(base_valid_flat)], dim=-1)
        base_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C_base)
        base_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C_base)

        if P_pairs > 0:
            C_coal = T * P_pairs
            # source_idx maps S-axis index → planet slot.
            coal_src = torch.stack(
                [source_idx[sA].reshape(C_coal), source_idx[sB].reshape(C_coal)],
                dim=-1,
            )                                                                    # [C_coal, 2]
            sendA = torch.where(valid_pair, sizesA, torch.zeros_like(sizesA))
            sendB = torch.where(valid_pair, sizesB, torch.zeros_like(sizesB))
            coal_send = torch.stack(
                [sendA.reshape(C_coal), sendB.reshape(C_coal)], dim=-1)
            coal_angle = torch.stack(
                [angleA.reshape(C_coal), angleB.reshape(C_coal)], dim=-1)
            etaA_safe = torch.where(valid_pair, etaA, torch.ones_like(etaA))
            etaB_safe = torch.where(valid_pair, etaB, torch.ones_like(etaB))
            coal_eta = torch.stack(
                [etaA_safe.reshape(C_coal), etaB_safe.reshape(C_coal)], dim=-1)
            coal_active = torch.stack(
                [valid_pair.reshape(C_coal), valid_pair.reshape(C_coal)], dim=-1)
            coal_tgt_short = torch.arange(T, device=device).view(T, 1).expand(T, P_pairs).reshape(C_coal)
            coal_tgt_slot = target_idx[coal_tgt_short]
            coal_valid = valid_pair.reshape(C_coal)

            cand_src = torch.cat([base_src_padded, coal_src], dim=0)
            cand_send = torch.cat([base_send_padded, coal_send], dim=0)
            cand_angle = torch.cat([base_angle_padded, coal_angle], dim=0)
            cand_eta = torch.cat([base_eta_padded, coal_eta], dim=0)
            cand_active = torch.cat([base_active, coal_active], dim=0)
            cand_tgt_slot = torch.cat([base_tgt_slot, coal_tgt_slot], dim=0)
            cand_tgt_short = torch.cat([base_tgt_short, coal_tgt_short], dim=0)
            cand_valid = torch.cat([base_valid_flat, coal_valid], dim=0)
        else:
            cand_src = base_src_padded
            cand_send = base_send_padded
            cand_angle = base_angle_padded
            cand_eta = base_eta_padded
            cand_active = base_active
            cand_tgt_slot = base_tgt_slot
            cand_tgt_short = base_tgt_short
            cand_valid = base_valid_flat

        cand_is_def = target_is_mine[cand_tgt_short]
        C = int(cand_src.shape[0])
    elif _multi_size_enabled():
        # --- Three fleet sizes per (source, target): capture_floor, 2× floor,
        # safe_drain. Each variant gets its own aim (fleet speed depends on
        # ships, so eta differs). Packed as [C=S*T*N, L=1] so greedy's target
        # mutex blocks losing variants once one wins the wave.
        N = 3
        # Step 1 — compute the largest variant (safe_drain) first and use its
        # eta to read the capture floor; size_lo = floor at that eta.
        sizes_hi = drain.view(S, 1).expand(S, T).floor()                          # [S, T]
        active_hi = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_hi.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)                                                             # [S, T]
        aim_hi = intercept_angle(
            movement,
            source_idx.unsqueeze(1), target_idx.unsqueeze(0),
            sizes_hi, active=active_hi,
        )
        eta_hi = aim_hi["eta"]                                                    # [S, T]
        if K > 0:
            k_arr_hi = (eta_hi.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr_hi = (
                floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr_hi.unsqueeze(-1)).squeeze(-1)
            )                                                                     # [S, T]
        else:
            floor_at_arr_hi = torch.ones(S, T, dtype=dtype, device=device)

        # Floor at hi's eta gives the minimum to capture; cap by drain so a
        # single launch can never over-drain the source.
        sizes_lo = torch.minimum((floor_at_arr_hi * _overkill_for_targets(obs, target_idx, player_count, dtype)).ceil().clamp(min=1.0), sizes_hi) # [S, T]
        sizes_mid = torch.minimum(2.0 * sizes_lo, sizes_hi)                       # [S, T]
        sizes_3 = torch.stack([sizes_lo, sizes_mid, sizes_hi], dim=-1)            # [S, T, N]

        # Step 2 — recompute reachability + aim per variant (each variant's eta
        # depends on its own fleet speed via fleet_speed(ships)).
        active_3 = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_3, eta_cap=eta_cap,
        )                                                                         # [S, T, N]
        aim_3 = intercept_angle(
            movement,
            source_idx.view(S, 1, 1), target_idx.view(1, T, 1),
            sizes_3, active=active_3,
        )
        angle_3 = aim_3["angle"]                                                  # [S, T, N]
        eta_3 = aim_3["eta"]                                                      # [S, T, N]
        viable_3 = aim_3["viable"] & (eta_3 <= eta_cap.view(1, T, 1))

        # Step 3 — capture-floor gate at each variant's own arrival turn.
        if K > 0:
            k_arr_3 = (eta_3.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr_3 = (
                floor.view(1, T, 1, K).expand(S, T, N, K)
                .gather(-1, k_arr_3.unsqueeze(-1)).squeeze(-1)
            )                                                                     # [S, T, N]
        else:
            floor_at_arr_3 = torch.ones(S, T, N, dtype=dtype, device=device)
        clears_floor_3 = sizes_3 >= floor_at_arr_3                                # [S, T, N]
        ships_ok_3 = sizes_3 <= source_ships.view(S, 1, 1)                        # [S, T, N]

        valid_3 = (
            viable_3 & clears_floor_3 & (sizes_3 >= 1.0) & ships_ok_3
            & src_neq_tgt.unsqueeze(-1)
            & source_exists.view(S, 1, 1) & target_exists.view(1, T, 1)
        )                                                                         # [S, T, N]

        L = 1
        C = S * T * N
        cand_src = source_idx.view(S, 1, 1).expand(S, T, N).reshape(C, L)
        cand_tgt_slot = target_idx.view(1, T, 1).expand(S, T, N).reshape(C)
        cand_tgt_short = (
            torch.arange(T, device=device).view(1, T, 1).expand(S, T, N).reshape(C)
        )
        cand_send = torch.where(valid_3, sizes_3, torch.zeros_like(sizes_3)).reshape(C, L)
        cand_angle = angle_3.reshape(C, L)
        cand_eta = torch.where(valid_3, eta_3, torch.ones_like(eta_3)).reshape(C, L)
        cand_active = valid_3.reshape(C, L)
        cand_valid = valid_3.reshape(C)

        # --- Sync-pair stage (PRODUCER_PLUS_SYNC; only on the REAL planning
        # pass, sync_sink=None on mirror/replan passes keeps them unchanged).
        # Two-source candidates on targets NEITHER source cracks alone but a
        # joint same-tick arrival does: leg sizes = safe_drain, joint floor
        # read at the LATER leg's arrival tick, the nearer leg scored at the
        # far leg's eta (its launch is deferred via a memory hold; d=0 pairs
        # are plain same-tick coalitions and launch immediately).
        _sy_ksrc = min(_sync_k_src(), S)
        if sync_sink is not None and K > 0 and _sy_ksrc >= 2:
            sizes_hi_v = sizes_3[..., -1]                                         # [S, T]
            eta_hi_v = eta_3[..., -1]
            angle_hi_v = angle_3[..., -1]
            k_arr_hi = k_arr_3[..., -1]
            clears_hi = clears_floor_3[..., -1]
            viable_only_hi = (
                viable_3[..., -1] & (sizes_hi_v >= 1.0) & ships_ok_3[..., -1]
                & src_neq_tgt & source_exists.view(S, 1) & target_exists.view(1, T)
            )                                                                     # [S, T]
            ranked = torch.where(
                viable_only_hi, -eta_hi_v, torch.full_like(eta_hi_v, float("-inf"))
            ).transpose(0, 1)                                                     # [T, S]
            top_src = _stable_topk_indices(ranked, _sy_ksrc)                       # [T, Ksrc]
            pair_idx = torch.triu_indices(_sy_ksrc, _sy_ksrc, offset=1, device=device)
            pa, pb = pair_idx[0], pair_idx[1]
            Pp = int(pa.numel())
            if Pp > 0:
                Tidx = torch.arange(T, device=device).view(T, 1).expand(T, Pp)
                sA = top_src[:, pa]                                               # [T, Pp]
                sB = top_src[:, pb]
                kA = k_arr_hi[sA, Tidx]
                kB = k_arr_hi[sB, Tidx]
                a_is_far = kA >= kB
                k_sync = torch.maximum(kA, kB)
                d_gap = (kA - kB).abs()
                floor_sync = floor.gather(-1, k_sync.clamp(0, K - 1))             # [T, Pp]
                szA = sizes_hi_v[sA, Tidx]
                szB = sizes_hi_v[sB, Tidx]
                valid_pair = (
                    viable_only_hi[sA, Tidx] & viable_only_hi[sB, Tidx]
                    & ~clears_hi[sA, Tidx] & ~clears_hi[sB, Tidx]
                    & ((szA + szB) >= floor_sync)
                    & (source_idx[sA] != source_idx[sB])
                    & (d_gap <= _sync_dmax())
                )                                                                 # [T, Pp]
                # --- floor-proportional pair sizing. Full safe_drain per leg
                # doubles committed capital on one target (the week's known
                # disease — confirmed by the first mirror leg: 1/12, in-flight
                # share 67% vs 45%). Right-size the pair to the joint floor ×
                # overkill, split proportionally to each leg's drain; smaller
                # fleets fly SLOWER, so re-aim and re-check the floor at the
                # later arrival. Pairs where the lo sizing fails the re-check
                # fall back to full drain (the gate proved drain clears).
                ov_t = _overkill_for_targets(obs, target_idx, player_count, dtype)
                if not torch.is_tensor(ov_t):   # scalar (legacy) form
                    ov_t = torch.full((T,), float(ov_t), dtype=dtype, device=device)
                need = (floor_sync * ov_t.view(T, 1)).ceil().clamp(min=2.0)        # [T, Pp]
                pair_sum = (szA + szB).clamp(min=1.0)
                nA = (need * szA / pair_sum).ceil().clamp(min=1.0)
                nB = (need * szB / pair_sum).ceil().clamp(min=1.0)
                src_a_slots = source_idx[sA]                                       # [T, Pp]
                src_b_slots = source_idx[sB]
                tgt_pp = target_idx.view(T, 1).expand(T, Pp)
                aim_loA = intercept_angle(movement, src_a_slots, tgt_pp, nA, active=valid_pair)
                aim_loB = intercept_angle(movement, src_b_slots, tgt_pp, nB, active=valid_pair)
                etaLA = aim_loA["eta"]
                etaLB = aim_loB["eta"]
                kLA = (etaLA.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
                kLB = (etaLB.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
                k_sync_lo = torch.maximum(kLA, kLB)
                floor_lo = floor.gather(-1, k_sync_lo)
                cap_pp = eta_cap.view(T, 1)
                lo_ok = (
                    valid_pair & aim_loA["viable"] & aim_loB["viable"]
                    & (etaLA <= cap_pp) & (etaLB <= cap_pp)
                    & ((nA + nB) >= floor_lo)
                )
                szA_f = torch.where(lo_ok, nA, szA)
                szB_f = torch.where(lo_ok, nB, szB)
                etaA_f = torch.where(lo_ok, etaLA, eta_hi_v[sA, Tidx])
                etaB_f = torch.where(lo_ok, etaLB, eta_hi_v[sB, Tidx])
                angA_f = torch.where(lo_ok, aim_loA["angle"], angle_hi_v[sA, Tidx])
                angB_f = torch.where(lo_ok, aim_loB["angle"], angle_hi_v[sB, Tidx])
                kA_f = torch.where(lo_ok, kLA, kA)
                kB_f = torch.where(lo_ok, kLB, kB)
                a_is_far = kA_f >= kB_f
                k_sync = torch.maximum(kA_f, kB_f)
                d_gap = (kA_f - kB_f).abs()
                valid_pair = valid_pair & (d_gap <= _sync_dmax())
                if bool((valid_pair & (d_gap > 0)).any()):
                    # Delayed pairs telegraph: the far leg is visible for the
                    # whole hold window, so the joint size must clear the
                    # FULL-reaction floor (reinforcement weight 1.0, no
                    # reaction lag) at the synced arrival — not the live
                    # stack's discounted 0.5/lag-2 floor. This is what the
                    # -46% mirror rout of unconditioned holds was made of.
                    margin_full = _reactive_reinforcement_margin(
                        obs, cache, target_idx, int(K_eta), weight=1.0, lag=0.0,
                    )
                    if margin_full is not None:
                        floor_full = capture_floor(
                            status_sizing, target_idx=target_idx, k_max=K_eta,
                            capture_overhead=1.0, player_id=pid,
                            reinforcement=margin_full,
                        )
                        ff_sync = floor_full.gather(-1, k_sync.clamp(0, K - 1))
                        valid_pair = valid_pair & (
                            (d_gap == 0) | ((szA_f + szB_f) >= ff_sync))
                eta_far = torch.where(a_is_far, etaA_f, etaB_f)
                delayed_a = (~a_is_far) & (d_gap > 0)
                delayed_b = a_is_far & (d_gap > 0)
                etaA_eff = torch.where(delayed_a, eta_far, etaA_f)
                etaB_eff = torch.where(delayed_b, eta_far, etaB_f)

                C_sy = T * Pp
                m = valid_pair
                zero = torch.zeros_like(szA)
                one = torch.ones_like(etaA_f)
                sy_src = torch.stack(
                    [src_a_slots.reshape(C_sy), src_b_slots.reshape(C_sy)], dim=-1)
                sy_send = torch.stack(
                    [torch.where(m, szA_f, zero).reshape(C_sy),
                     torch.where(m, szB_f, zero).reshape(C_sy)], dim=-1)
                sy_angle = torch.stack(
                    [angA_f.reshape(C_sy), angB_f.reshape(C_sy)], dim=-1)
                sy_eta = torch.stack(
                    [torch.where(m, etaA_eff, one).reshape(C_sy),
                     torch.where(m, etaB_eff, one).reshape(C_sy)], dim=-1)
                sy_active = torch.stack([m.reshape(C_sy), m.reshape(C_sy)], dim=-1)
                sy_tgt_short = (
                    torch.arange(T, device=device).view(T, 1).expand(T, Pp).reshape(C_sy)
                )
                sy_tgt_slot = target_idx[sy_tgt_short]
                sy_valid = m.reshape(C_sy)

                for t_i, p_i in (m & (d_gap > 0)).nonzero(as_tuple=False).tolist():
                    far_a = bool(a_is_far[t_i, p_i].item())
                    sz_far, sz_near = (szA_f, szB_f) if far_a else (szB_f, szA_f)
                    sl_far, sl_near = (src_a_slots, src_b_slots) if far_a else (src_b_slots, src_a_slots)
                    sync_sink.append({
                        "near_src": int(sl_near[t_i, p_i].item()),
                        "far_src": int(sl_far[t_i, p_i].item()),
                        "tgt": int(target_idx[t_i].item()),
                        "eta": float(eta_far[t_i, p_i].item()),
                        "near_ships": float(sz_near[t_i, p_i].item()),
                        "far_ships": float(sz_far[t_i, p_i].item()),
                        "arrival_dt": int(k_sync[t_i, p_i].item()) + 1,
                    })

                base0 = cand_src[:, 0]
                cand_src = torch.cat([torch.stack([base0, base0], dim=-1), sy_src], dim=0)
                cand_send = torch.cat(
                    [torch.cat([cand_send, torch.zeros(C, 1, dtype=dtype, device=device)], dim=-1),
                     sy_send], dim=0)
                cand_angle = torch.cat(
                    [torch.cat([cand_angle, torch.zeros(C, 1, dtype=dtype, device=device)], dim=-1),
                     sy_angle], dim=0)
                cand_eta = torch.cat(
                    [torch.cat([cand_eta, torch.ones(C, 1, dtype=dtype, device=device)], dim=-1),
                     sy_eta], dim=0)
                cand_active = torch.cat(
                    [torch.cat([cand_active, torch.zeros(C, 1, dtype=torch.bool, device=device)], dim=-1),
                     sy_active], dim=0)
                cand_tgt_slot = torch.cat([cand_tgt_slot, sy_tgt_slot], dim=0)
                cand_tgt_short = torch.cat([cand_tgt_short, sy_tgt_short], dim=0)
                cand_valid = torch.cat([cand_valid, sy_valid], dim=0)
                L = 2
                C = int(cand_src.shape[0])

        cand_is_def = target_is_mine[cand_tgt_short]                              # [C]
    else:
        # --- single fleet size = the max garrison launch (safe_drain) -----------
        # Engine needs integer ship counts; floor (never exceed what's available).
        sizes = drain.view(S, 1).expand(S, T).floor()                            # [S, T]

        # Strict-superset reachability precheck (always on): defers the body screen to
        # candidates that can physically reach the target in time.
        active = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)                                                            # [S, T]
        aim = intercept_angle(
            movement,
            source_idx.unsqueeze(1),                                             # [S, 1]
            target_idx.unsqueeze(0),                                             # [1, T]
            sizes,                                                               # [S, T]
            active=active,
        )
        angle = aim["angle"]                                                     # [S, T]
        eta = aim["eta"]
        viable = aim["viable"] & (eta <= eta_cap.view(1, T))

        # Capture-floor gate at each fleet's arrival turn (defenders grow with k). The
        # single size must clear the defender it lands on (size >= floor_at_arr). Owned
        # targets have floor 1 (reinforcement), so any positive send clears.
        if K > 0:
            k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)  # [S,T]
            floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
        else:
            floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
        clears_floor = sizes >= floor_at_arr                                     # [S, T]

        valid = (
            viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
            & source_exists.view(S, 1) & target_exists.view(1, T)
        )                                                                        # [S, T]

        # --- pack one candidate per (source, target); contributor axis L = 1 ----
        L = 1
        C = S * T
        cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
        cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
        cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
        cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
        cand_angle = angle.reshape(C, L)
        cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
        cand_active = valid.reshape(C, L)
        cand_valid = valid.reshape(C)
        cand_is_def = target_is_mine[cand_tgt_short]                             # [C]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    # If opp_projection is on, broadcast the projected opp launches to the
    # candidate axis and concat them onto launches' L axis BEFORE scoring.
    # The scorer's per-launch `owner` + `_per_step_survivor`'s owner-axis
    # topk handle the mixed-owner combat correctly. Greedy below still
    # operates on the ORIGINAL [C, L_my] tensors — opp slots never enter
    # greedy's budget / role-mutex view.
    scoring_launches = launches
    if background is not None:
        L_opp = int(background.source_slots.shape[-1])
        if L_opp > 0:
            def _bg(t):
                return t.unsqueeze(0).expand(C, -1)
            scoring_launches = LaunchSet(
                source_slots=torch.cat([launches.source_slots, _bg(background.source_slots)], dim=-1),
                target_slots=torch.cat([launches.target_slots, _bg(background.target_slots)], dim=-1),
                ships=torch.cat([launches.ships, _bg(background.ships)], dim=-1),
                eta=torch.cat([launches.eta, _bg(background.eta)], dim=-1),
                owner=torch.cat([launches.owner, _bg(background.owner)], dim=-1),
                valid=torch.cat([launches.valid, _bg(background.valid)], dim=-1),
            )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=scoring_launches, player_id=pid,
        opp_weights=opp_weights,
        terminal_prod_weight=_terminal_prod_value(),
        terminal_neutral_only=_terminal_neutral_only(),
    )                                                                            # [C]
    _cc = _commit_cost_eps()
    if _cc > 0.0:
        score = score - _cc * _commit_flight_cost(cand_send, cand_eta, cand_active)
    # Capture the base competitive score before additive terms so the
    # force-concentration rescore can re-derive the addon contribution per
    # iteration without recomputing recapture/denial/opening. Only allocated
    # when force-concentration is ON — OFF path is byte-identical. The kwarg
    # override lets the opp-projection inner calls disable FC explicitly so
    # the rescore closure doesn't run inside K-round opponent simulation
    # (where the K_opp x num_opps multiplier would blow up wallclock).
    if force_concentration is None:
        _fc_enabled = _force_concentration_enabled()
    else:
        _fc_enabled = bool(force_concentration)
    _fc_base_score = score.clone() if _fc_enabled else None
    if _recapture_penalty_enabled():
        # Subtract a non-negative recapture discount per candidate. The
        # penalty is in ship units (prod[T] * turns_lost), additive with
        # competitive_score. K_opp is read from the multi-tick env knob
        # only when opp projection is on; otherwise pass 0 so we don't
        # subtract a window the scorer never modeled.
        pen = recapture_penalty(
            obs=obs, cache=cache, garrison_status=garrison_status,
            cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_send=cand_send, cand_eta=cand_eta,
            cand_valid=cand_valid, cand_is_def=cand_is_def,
            capture_floor_TK=floor,
            prod=prod, H=H,
            K_recap=_recapture_k(int(player_count)),
            K_opp=(
                _multi_tick_opp_k(int(player_count)) if _opp_projection_enabled() else 0
            ),
            safety_reserve=_recapture_safety_reserve(),
            player_id=pid,
        )
        score = score - pen * float(_recapture_penalty_weight())
    if _denial_bonus_enabled() or _opening_bonus_enabled():
        # Resolve current_step once (cheap) for both bonuses.
        _cur_step = int(obs_tensors["step"].max().item())
        if _denial_bonus_enabled():
            # Rewards captures of targets opp values (currently owns OR
            # opp_proj's background launches target it). Encodes
            # "blocking the opponent's biggest bet."
            d_bonus = denial_bonus(
                obs=obs, background=background,
                cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
                cand_send=cand_send, cand_eta=cand_eta,
                cand_valid=cand_valid, cand_is_def=cand_is_def,
                capture_floor_TK=floor, prod=prod,
                garrison_status=garrison_status,
                H=H, current_step=_cur_step,
                game_length_est=_game_length_est(),
                weight=_denial_bonus_weight(),
                player_id=pid,
            )
            score = score + d_bonus
        if _opening_bonus_enabled():
            # Opp-agnostic early-game boost: linearly decays from full at
            # step 0 to zero at ``opening_window`` (default 30). Encodes
            # the horizon-too-short defect during the opening expansion.
            o_bonus = opening_bonus(
                obs=obs,
                cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
                cand_send=cand_send, cand_eta=cand_eta,
                cand_valid=cand_valid, cand_is_def=cand_is_def,
                capture_floor_TK=floor, prod=prod,
                garrison_status=garrison_status,
                H=H, current_step=_cur_step,
                game_length_est=_game_length_est(),
                opening_window=_opening_window(),
                weight=_opening_bonus_weight(),
                player_id=pid,
            )
            score = score + o_bonus
    if _hold_value() > 0.0:
        # Holding-time-priced capture credit: post-horizon production for
        # captures the opponent cannot feasibly retake within the window.
        score = score + _hold_value_bonus(
            obs=obs, cache=cache, target_idx=target_idx,
            cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_send=cand_send, cand_eta=cand_eta,
            cand_valid=cand_valid, cand_is_def=cand_is_def,
            capture_floor_TK=floor, prod=prod, K=K,
        )
    if _garrison_value() > 0.0 and (
        int(obs_tensors["step"].max().item()) >= _garrison_value_from_step()
    ):
        # Proactive-garrison credit: reinforcing an own planet whose local
        # balance vs the enemy's uncommitted reserve is negative.
        score = score + _garrison_value_bonus(
            obs=obs, cache=cache, target_idx=target_idx,
            cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_send=cand_send, cand_eta=cand_eta,
            cand_valid=cand_valid, cand_is_def=cand_is_def,
            prod=prod, K=K,
        )
    # Force-concentration rescore closure: between greedy waves, re-score
    # every candidate against the just-fired waves so wave 2 at a target sees
    # wave 1's commitment (no double-counting). Uses the same `scoring_launches`
    # the initial score saw, plus the committed waves appended to the L axis
    # owner=pid. Add-on terms (recapture/denial/opening) depend only on
    # per-candidate state, so we precompute their offset once and add it back.
    _fc_rescore_fn = None
    _fc_max_waves = 1
    if _fc_enabled:
        _fc_addon_offset = score - _fc_base_score
        _fc_max_waves = _force_concentration_max_waves()
        _fc_C = int(cand_src.shape[0])
        _fc_scoring_launches = scoring_launches

        def _fc_rescore(c_src, c_send, c_eta, c_tgt, c_active):
            flat_src = c_src.reshape(-1)
            flat_send = c_send.reshape(-1)
            flat_eta = c_eta.reshape(-1)
            flat_tgt = c_tgt.reshape(-1)
            flat_active = c_active.reshape(-1)
            L_done = int(flat_src.shape[0])
            if L_done == 0:
                _new = _fc_base_score + _fc_addon_offset
                return torch.where(cand_valid, _new, torch.full_like(_new, float("-inf")))
            flat_owner = torch.full(
                (L_done,), int(pid), dtype=torch.long, device=device,
            )

            def _bc(t):
                return t.unsqueeze(0).expand(_fc_C, -1)

            merged = LaunchSet(
                source_slots=torch.cat(
                    [_fc_scoring_launches.source_slots, _bc(flat_src)], dim=-1,
                ),
                target_slots=torch.cat(
                    [_fc_scoring_launches.target_slots, _bc(flat_tgt)], dim=-1,
                ),
                ships=torch.cat([_fc_scoring_launches.ships, _bc(flat_send)], dim=-1),
                eta=torch.cat([_fc_scoring_launches.eta, _bc(flat_eta)], dim=-1),
                owner=torch.cat([_fc_scoring_launches.owner, _bc(flat_owner)], dim=-1),
                valid=torch.cat([_fc_scoring_launches.valid, _bc(flat_active)], dim=-1),
            )
            new_base = score_candidates(
                garrison_status, prod=prod, alive_by_step=alive_by_step,
                player_count=int(player_count), launches=merged, player_id=pid,
                opp_weights=opp_weights,
                terminal_prod_weight=_terminal_prod_value(),
        terminal_neutral_only=_terminal_neutral_only(),
            )
            _new = new_base + _fc_addon_offset
            return torch.where(cand_valid, _new, torch.full_like(_new, float("-inf")))

        _fc_rescore_fn = _fc_rescore
    if _mass_tiebreak_enabled() and _mass_active(player_count):
        total_send = (cand_send * cand_active.to(cand_send.dtype)).sum(dim=-1)  # [C]
        score = score + 1e-4 * total_send
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    # Cross-wave over-drain guard for multi-size: single-size's accidental
    # invariant — at-most-one wave per source, because cand_send==drain — does
    # NOT hold for multi-size, where size_lo<drain lets multiple small launches
    # from the same source fire across waves and total more than safe_drain.
    # Cap the source budget at drain so the cumulative sent per source stays
    # safe. OFF path unchanged to preserve bit-identical OFF parity.
    source_budget = obs.ships.to(dtype).clone()
    if _multi_size_enabled() or _coalitions_enabled():
        src_planet = source_idx.clamp(0, P - 1)
        source_budget[src_planet] = torch.minimum(source_budget[src_planet], drain)

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=source_budget,
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
        rescore_fn=_fc_rescore_fn, max_waves_per_target=_fc_max_waves,
    )

    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
    # Forward redistribution (Planet Wars canon, confirmed by the del Toro
    # loss: 121 idle rear garrison vs their 39 at step 40): the default
    # "materially more stressed" gate (delta 0.25) strands leftover ships on
    # rear planets with no local gradient. The forward gate lowers the delta
    # to "any strictly forward flow" and extends the flight cap, so rear
    # garrisons stream toward the frontier turn after turn. Strictly-positive
    # gap keeps the flow one-directional (no backwash loops).
    cfg_regroup = config
    if _regroup_forward_enabled():
        cfg_regroup = dataclasses.replace(
            config,
            regroup_pressure_delta_min=0.0,
            max_regroup_time=_regroup_forward_time(float(config.max_regroup_time)),
        )
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=cfg_regroup, H=H,
    )
    _convoy_min = _regroup_min_send() if _mass_active(player_count) else 0.0
    if _convoy_min > 0.0 and int(regroup_entries.ships.shape[0]) > 0:
        keep = regroup_entries.ships >= _convoy_min
        regroup_entries = LaunchEntries(
            source_slots=regroup_entries.source_slots,
            target_slots=regroup_entries.target_slots,
            ships=regroup_entries.ships,
            angle=regroup_entries.angle,
            eta=regroup_entries.eta,
            valid=regroup_entries.valid & keep,
        )
    if _snipe_hold_enabled():
        reserved = _snipe_hold_reserved_sources(
            obs=obs, garrison_status=garrison_status, background=background,
            wave_entries=wave_entries, H=H, movement=movement,
        )
        if reserved is not None and int(regroup_entries.ships.shape[0]) > 0:
            keep_r = ~reserved[regroup_entries.source_slots.clamp(0, P - 1)]
            regroup_entries = LaunchEntries(
                source_slots=regroup_entries.source_slots,
                target_slots=regroup_entries.target_slots,
                ships=regroup_entries.ships,
                angle=regroup_entries.angle,
                eta=regroup_entries.eta,
                valid=regroup_entries.valid & keep_r,
            )
    return concat_launch_entries([wave_entries, regroup_entries])


# --- Snipe-hold (toll-sniping reservation) -------------------------------------
# The planner is now-or-never: when the projection shows an opponent flipping
# a planet at tick k_f, arriving at k_f+1 costs survivor+1 ships (a fraction
# of the pre-flip garrison — "let them pay the toll"), but if our fleet would
# arrive too early there is no way to WAIT, and the regroup lane drains the
# idle ships away before the window opens. v1: detect flip events, find owned
# sources that can afford the snipe from ships remaining after this turn's
# waves and reach the target around k_f+1, and RESERVE them — their regroup
# entries are filtered this turn. Re-planning fires the actual snipe when the
# timing lines up (capture_floor at that arrival already reflects the thin
# survivor). Design: kb/thoughts/2026-06-10-snipe-hold-design.md.


def _snipe_hold_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_SNIPE_HOLD", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _snipe_hold_reserved_sources(
    *, obs, garrison_status, background, wave_entries, H: int, movement,
):
    """Mask [P] of sources to hold home for a dated snipe, or None."""
    P = int(obs.P)
    pid = int(obs.player_id)
    device = obs.device
    dtype = obs.ships.dtype
    traj = garrison_status.owner                       # [P, H+1]
    init = traj[:, 0]
    is_opp = (traj != pid) & (traj >= 0)
    became_opp = is_opp & (init.unsqueeze(-1) != traj)
    flips = became_opp & ~obs.owned.unsqueeze(-1)
    flip_any = flips.any(dim=-1)
    if not bool(flip_any.any()):
        return None
    # First flip tick per planet and the survivor garrison at that tick.
    k_idx = torch.arange(traj.shape[-1], device=device).expand_as(flips)
    k_f = torch.where(flips, k_idx, torch.full_like(k_idx, traj.shape[-1])).min(dim=-1).values
    surv = garrison_status.ships.gather(
        -1, k_f.clamp(0, traj.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    # Ships left at each source after this turn's committed waves.
    committed = torch.zeros(P, dtype=dtype, device=device)
    if int(wave_entries.ships.shape[0]) > 0:
        v = wave_entries.valid
        committed.scatter_add_(
            0, wave_entries.source_slots[v].clamp(0, P - 1), wave_entries.ships[v].to(dtype))
    remaining = (obs.ships.to(dtype) - committed).clamp(min=0.0)
    reserved = torch.zeros(P, dtype=torch.bool, device=device)
    flip_planets = flip_any.nonzero(as_tuple=True)[0]
    src_planets = (obs.owned & obs.alive).nonzero(as_tuple=True)[0]
    if int(src_planets.shape[0]) == 0:
        return None
    for fp in flip_planets.tolist():
        kf = int(k_f[fp].item())
        if kf <= 0 or kf >= H:
            continue
        cost = float(surv[fp].item()) + 2.0
        for sp in src_planets.tolist():
            if reserved[sp] or float(remaining[sp].item()) < cost:
                continue
            aim = intercept_angle(
                movement,
                torch.tensor([sp], device=device),
                torch.tensor([fp], device=device),
                torch.tensor([cost], dtype=dtype, device=device),
            )
            if not bool(aim["viable"][0]):
                continue
            eta_now = float(aim["eta"][0].item())
            # Reserve only when we would arrive EARLY if we launched now —
            # i.e. waiting is exactly what unlocks the cheap capture.
            if eta_now < (kf + 1):
                reserved[sp] = True
                break
    return reserved if bool(reserved.any()) else None


# --- Synchronized multi-source arrivals (delayed launches) ---------------------
# Planet Wars canon: staggered waves die piecemeal to the 1:1 garrison trade;
# multi-source SAME-TICK arrivals are the capture mechanism for targets no
# single source can crack. The planner is now-or-never, so the second half of
# the mechanism is a HOLD: the nearer source's leg is scored at the far leg's
# arrival tick (exact under the flow scorer — arrival credit lands at
# ceil(eta); the tick-0 source debit makes the score conservative, since the
# held ships actually keep defending home), then diverted post-veto into a
# memory-held schedule and launched on the last turn that still makes the
# arrival date (re-aimed fresh, so orbit drift cannot desynchronize it).
# Default OFF preserves byte-identical behaviour.


def _sync_enabled() -> bool:
    return os.environ.get("PRODUCER_PLUS_SYNC", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _sync_dmax() -> int:
    """Max hold length in ticks (gap between the pair's natural arrivals).

    Default 0 = same-tick coalitions only, NO holds. Delayed legs (d > 0)
    measured 3/12 (-46% @250) vs the live stack on 2026-06-11: the far leg
    telegraphs the attack for the whole hold window and a reply-aware
    defender reinforces past the pair's joint size, while the d=0 ablation
    sat at exact mirror parity (7/12, -0.7%). Holds stay opt-in for
    redemption experiments.
    """
    return max(0, _env_int("PRODUCER_PLUS_SYNC_DMAX", 0))


def _sync_k_src() -> int:
    """Per-target source pool for pair enumeration (nearest-first)."""
    return max(2, _env_int("PRODUCER_PLUS_SYNC_K", 6))


def _sync_max_holds() -> int:
    """Cap on concurrently pending holds (commitment-exposure guard)."""
    return max(1, _env_int("PRODUCER_PLUS_SYNC_MAX_HOLDS", 4))


def _sync_entry_key(src_slot: int, tgt_slot: int, eta: float, ships: float):
    return (int(src_slot), int(tgt_slot), round(float(eta), 4), int(round(float(ships))))


def _process_sync_holds(memory, *, obs, obs_tensors: dict, movement, current_step: int):
    """Advance pending holds one turn. Returns ``(exec_entries, debit)``.

    Per hold: cancel if the source is lost/drained or the target died or
    flipped to us; launch NOW (fresh aim) if waiting one more turn would miss
    the arrival date; release if the date became unreachable (orbit drift
    beyond 1 tick of slack); otherwise keep holding. ``debit`` reserves both
    kept and just-launched ships against the planner's budget view.
    """
    holds = getattr(memory, "sync_holds", None)
    if current_step == 0 or holds is None:
        holds = []
    if not holds:
        memory.sync_holds = []
        return None, None
    device = obs.device
    dtype = obs.ships.dtype
    P = int(obs.P)
    planet_ids = obs_tensors["planets"][..., 0].long()
    slot_of = {int(planet_ids[i].item()): i for i in range(P)}
    kept: list = []
    exec_rows: list = []
    debit = torch.zeros_like(obs.ships)
    for h in holds:
        s = slot_of.get(int(h["src_id"]))
        t = slot_of.get(int(h["tgt_id"]))
        if s is None or t is None:
            continue
        if not bool(obs.owned[s]) or not bool(obs.alive[s]) or not bool(obs.alive[t]):
            continue
        if bool(obs.owned[t]):
            continue  # captured by other means — release the reserve
        ships = float(h["ships"])
        if float(obs.ships[s].item()) < ships:
            continue  # combat ate the reserve — the sized pair is broken
        remaining = int(h["arrival_step"]) - current_step
        aim = intercept_angle(
            movement,
            torch.tensor([s], dtype=torch.long, device=device),
            torch.tensor([t], dtype=torch.long, device=device),
            torch.tensor([ships], dtype=dtype, device=device),
        )
        if not bool(aim["viable"][0]):
            continue
        eta_now = float(aim["eta"][0].item())
        if math.ceil(eta_now) >= remaining:
            # Last turn that can make the date (1 tick of late slack).
            if math.ceil(eta_now) <= remaining + 1:
                exec_rows.append(
                    (s, t, ships, float(aim["angle"][0].item()), eta_now))
                debit[s] += ships
            continue
        kept.append(h)
        debit[s] += ships
    memory.sync_holds = kept
    entries = None
    if exec_rows:
        entries = LaunchEntries(
            source_slots=torch.tensor([r[0] for r in exec_rows], dtype=torch.long, device=device),
            target_slots=torch.tensor([r[1] for r in exec_rows], dtype=torch.long, device=device),
            ships=torch.tensor([r[2] for r in exec_rows], dtype=dtype, device=device),
            angle=torch.tensor([r[3] for r in exec_rows], dtype=dtype, device=device),
            eta=torch.tensor([r[4] for r in exec_rows], dtype=dtype, device=device),
            valid=torch.ones(len(exec_rows), dtype=torch.bool, device=device),
        )
    if not bool((debit > 0).any()):
        debit = None
    return entries, debit


def _divert_sync_entries(entries, *, sink: list, obs_tensors: dict, current_step: int, memory):
    """Post-veto: convert chosen delayed near-legs into memory holds.

    A near-leg entry is identified by its (source, target, scored eta, ships)
    signature from the generation sink. It is NEVER launched this turn (its
    angle/eta describe the future synced flight, not a launch-now one); it
    becomes a hold only if its far partner survived selection + veto —
    otherwise it is dropped outright, because its size only makes sense
    jointly. NOTE: assumes the veto did not resize entries (upsize default
    OFF); a resized leg simply fails the signature match and launches as-is.
    """
    if not sink or int(entries.valid.shape[0]) == 0:
        return entries
    by_key: dict = {}
    for r in sink:
        by_key.setdefault(
            _sync_entry_key(r["near_src"], r["tgt"], r["eta"], r["near_ships"]), []
        ).append(r)
    E = int(entries.valid.shape[0])
    sig = set()
    for j in range(E):
        if bool(entries.valid[j]):
            sig.add(_sync_entry_key(
                entries.source_slots[j], entries.target_slots[j],
                entries.eta[j], entries.ships[j]))
    valid = entries.valid.clone()
    planet_ids = obs_tensors["planets"][..., 0].long()
    holds = list(getattr(memory, "sync_holds", []) or [])
    max_h = _sync_max_holds()
    changed = False
    for j in range(E):
        if not bool(valid[j]):
            continue
        key = _sync_entry_key(
            entries.source_slots[j], entries.target_slots[j],
            entries.eta[j], entries.ships[j])
        rs = by_key.get(key)
        if not rs:
            continue
        valid[j] = False
        changed = True
        partner = next(
            (r for r in rs if _sync_entry_key(
                r["far_src"], r["tgt"], r["eta"], r["far_ships"]) in sig),
            None,
        )
        if partner is not None and len(holds) < max_h:
            holds.append({
                "src_id": int(planet_ids[int(entries.source_slots[j].item())].item()),
                "tgt_id": int(planet_ids[int(entries.target_slots[j].item())].item()),
                "ships": float(entries.ships[j].item()),
                "arrival_step": current_step + int(partner["arrival_dt"]),
            })
    memory.sync_holds = holds
    if not changed:
        return entries
    return LaunchEntries(
        source_slots=entries.source_slots, target_slots=entries.target_slots,
        ships=entries.ships, angle=entries.angle, eta=entries.eta, valid=valid,
    )


def _score_do_nothing(
    *,
    status,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    background: LaunchSet,
    player_id: int,
    opp_weights: Tensor | None = None,
) -> Tensor:
    """Score of NOT launching anything while opp executes their projected
    plan. Returns a scalar tensor. Used to renormalize roi_threshold under
    opp-aware scoring.
    """
    L_opp = int(background.source_slots.shape[-1])
    if L_opp == 0:
        return torch.tensor(0.0, device=background.source_slots.device)
    # 1 candidate with the background's L slots, all owner=opp_id, our
    # contribution = nothing.
    bg = LaunchSet(
        source_slots=background.source_slots.unsqueeze(0),
        target_slots=background.target_slots.unsqueeze(0),
        ships=background.ships.unsqueeze(0),
        eta=background.eta.unsqueeze(0),
        owner=background.owner.unsqueeze(0),
        valid=background.valid.unsqueeze(0),
    )
    score = score_candidates(
        status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=bg, player_id=int(player_id),
        opp_weights=opp_weights,
        terminal_prod_weight=_terminal_prod_value(),
        terminal_neutral_only=_terminal_neutral_only(),
    )
    return score.flatten()[0]


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    current_step = int(obs_tensors["step"].max().item())
    K_eta_override = compute_k_eta_for_step(current_step, H=H)

    # Opponent model: run Producer's own planner from each opp seat with
    # background=None (one-step best response, opp assumes we do nothing
    # this turn). Returns the opp's predicted launches for this turn as a
    # padded LaunchSet that we pass as `background` to our own planner.
    # Default OFF preserves bit-identical static-opp scoring.
    #
    # The roi_threshold needs a per-turn shift because competitive_score is
    # measured against a do-nothing-by-everyone baseline (garrison_status).
    # In the static-opp world, do-nothing-by-me also means do-nothing-by-
    # opp, so do_nothing_score = 0 and a 1.5 absolute threshold == "1.5
    # ships of differential gain over not firing." In the opp-aware world,
    # do-nothing-by-me still leaves opp's projected launches running, so
    # do_nothing_score = -opp_gain (a per-turn constant, usually < 0).
    # To preserve the threshold's semantic meaning -- "fire only if you
    # gain >= roi_threshold ships over doing nothing" -- shift the
    # absolute threshold by do_nothing_score.
    background = None
    reply_trust = None
    cfg = config
    # FFA objective weights: only built for 3+ player games so the 2P path
    # stays byte-identical (None -> legacy equal-weight opponent sum).
    opp_w = None
    if _ffa_score_enabled() and int(player_count) >= 3:
        opp_w = _ffa_opp_weights(
            obs_tensors, player_id=int(obs.player_id), player_count=int(player_count),
        )
    if _opp_projection_enabled():
        opp_ids = [
            pid for pid in range(int(player_count)) if pid != int(obs.player_id)
        ]
        K_opp = max(1, _multi_tick_opp_k(int(player_count)))
        background = predict_opp_launches_via_mirror(
            plan_fn=plan_lite_waves,
            obs_tensors=obs_tensors, movement=movement, cache=cache,
            garrison_status=status, prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            opp_ids=opp_ids, config=config, player_count=int(player_count),
            K_eta_override=K_eta_override,
            pad_to=_env_int("PRODUCER_PLUS_OPP_MAX_L", MAX_L_OPP),
            K=K_opp,
            H=H,
        )
        if _reply_trust_enabled():
            # Verify last turn's prediction first, then stash this turn's
            # RAW prediction for next turn, then price at trusted strength.
            reply_trust = _update_reply_trust(
                memory, obs_tensors, pid=int(obs.player_id))
            _record_reply_prediction(memory, background, obs_tensors)
            background = _scale_launch_set_ships(background, reply_trust)
        do_nothing_score = float(_score_do_nothing(
            status=status, prod=movement.planet_prod,
            alive_by_step=alive_by_step, player_count=int(player_count),
            background=background, player_id=int(obs.player_id),
            opp_weights=opp_w,
        ))
        cfg = dataclasses.replace(
            config, roi_threshold=do_nothing_score + float(config.roi_threshold),
        )

    opening_entries = None
    obs_for_plan = obs
    _osw = _opening_search_window()
    if _osw > 0 and current_step < _osw:
        claimed = getattr(memory, "opening_claimed", None)
        if claimed is None or current_step == 0:
            claimed = set()
        due = _opening_search_plan(
            obs_tensors, pid=int(obs.player_id), claimed=claimed,
            horizon=_opening_search_horizon(), beam_width=_opening_search_beam(),
        )
        if due:
            opening_entries = _emit_opening_entries(
                due, movement=movement, obs=obs, obs_tensors=obs_tensors,
                garrison_status=status, H=H, cache=cache,
            )
        if opening_entries is not None:
            # Claim targets across turns; debit the planner's budget view so
            # the greedy pass can't double-spend the opening sends.
            planet_ids_now = obs_tensors["planets"][..., 0].long()
            sel = opening_entries.valid.nonzero(as_tuple=True)[0]
            for i in sel.tolist():
                claimed.add(int(planet_ids_now[int(opening_entries.target_slots[i].item())].item()))
            debit = torch.zeros_like(obs.ships)
            debit.scatter_add_(
                0, opening_entries.source_slots[sel].clamp(0, int(obs.P) - 1),
                opening_entries.ships[sel].to(obs.ships.dtype),
            )
            obs_for_plan = dataclasses.replace(
                obs, ships=(obs.ships - debit).clamp(min=0.0))
        memory.opening_claimed = claimed

    # Sync holds: advance pending delayed launches (execute / keep / release)
    # and reserve their ships against the planner's budget view. The sink
    # collects this turn's delayed-leg signatures for post-veto diversion.
    sync_sink = None
    sync_exec_entries = None
    if _sync_enabled():
        sync_sink = []
        sync_exec_entries, sync_debit = _process_sync_holds(
            memory, obs=obs, obs_tensors=obs_tensors, movement=movement,
            current_step=current_step,
        )
        if sync_debit is not None:
            obs_for_plan = dataclasses.replace(
                obs_for_plan, ships=(obs_for_plan.ships - sync_debit).clamp(min=0.0))

    entries = plan_lite_waves(
        movement=movement, obs=obs_for_plan, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=cfg, player_count=int(player_count),
        K_eta_override=K_eta_override,
        background=background,
        opp_weights=opp_w,
        sync_sink=sync_sink,
    )
    if sync_exec_entries is not None:
        entries = concat_launch_entries([sync_exec_entries, entries])
    if opening_entries is not None:
        entries = LaunchEntries(
            source_slots=torch.cat([opening_entries.source_slots, entries.source_slots]),
            target_slots=torch.cat([opening_entries.target_slots, entries.target_slots]),
            ships=torch.cat([opening_entries.ships, entries.ships]),
            angle=torch.cat([opening_entries.angle, entries.angle]),
            eta=torch.cat([opening_entries.eta, entries.eta]),
            valid=torch.cat([opening_entries.valid, entries.valid]),
        )
    if _replan_active(int(player_count)) and _opp_projection_enabled():
        # Raw config (not the roi-shifted cfg): the replan re-normalizes the
        # roi threshold itself against its own reply background.
        entries = _apply_replan(
            entries,
            movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
            garrison_status=status, prod=movement.planet_prod,
            alive_by_step=alive_by_step, config=config,
            player_count=int(player_count), K_eta_override=K_eta_override,
            H=H, opp_weights=opp_w,
        )
    if _response_veto_active(int(player_count)) and _opp_projection_enabled():
        # Raw config (not the roi-shifted cfg): the reply mirror plans the
        # opponent exactly like the original projection pass did, and the
        # veto margin's do-nothing normalization is computed fresh here.
        _reply_box: list = []
        _valid_before = int(entries.valid.sum().item())
        entries = _apply_response_veto(
            entries,
            movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
            garrison_status=status, prod=movement.planet_prod,
            alive_by_step=alive_by_step, config=config,
            player_count=int(player_count), K_eta_override=K_eta_override,
            H=H, opp_weights=opp_w,
            reply_out=_reply_box,
            reply_trust=reply_trust,
        )
        # Redirect: only when the veto actually freed budget (waves dropped)
        # — otherwise pass 1 already spent everything it wanted to.
        if (
            _redirect_active(int(player_count)) and _reply_box
            and int(entries.valid.sum().item()) < _valid_before
        ):
            entries = _apply_redirect(
                entries,
                reply=_reply_box[0],
                movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
                garrison_status=status, prod=movement.planet_prod,
                alive_by_step=alive_by_step, config=config,
                player_count=int(player_count), K_eta_override=K_eta_override,
                H=H, opp_weights=opp_w,
            )
    if _shot_mlp_active(int(player_count)):
        # Learned filter LAST so it judges the final wave set (including
        # redirect spends). Reject-only; sync partner-orphan handling below
        # already covers veto-dropped legs.
        entries = apply_shot_mlp_veto(
            entries, obs=obs, threshold=_shot_mlp_threshold(),
        )
    if sync_sink:
        # After all entry filters: chosen delayed legs become memory holds
        # (or are dropped if their far partner did not survive the veto) —
        # they must never reach the payload or the private-fleet cache.
        entries = _divert_sync_entries(
            entries, sink=sync_sink, obs_tensors=obs_tensors,
            current_step=current_step, memory=memory,
        )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    cfg = CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()
    # Optional override of the scorer's lookahead horizon. Bumping H lets
    # the scorer see longer-term outcomes (e.g. the recapture leg of an
    # exchange cycle) and properly value stockpiling vs cyclical attacks.
    # Cost scales linearly in H; defaults unchanged when env unset.
    env_h = os.environ.get(
        "PRODUCER_PLUS_HORIZON_4P" if int(player_count) >= 4 else "PRODUCER_PLUS_HORIZON_2P"
    )
    if env_h:
        try:
            cfg = dataclasses.replace(cfg, horizon=int(env_h))
        except ValueError:
            pass
    return cfg


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        self.opening_claimed: set | None = None
        self.trust_ema: float | None = None
        self.trust_predictions: list | None = None
        self.trust_fleet_ids: set | None = None
        self.sync_holds: list | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.opening_claimed = None
        self.trust_ema = None
        self.trust_predictions = None
        self.trust_fleet_ids = None
        self.sync_holds = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
            mem.opening_claimed = None
            mem.trust_ema = None
            mem.trust_predictions = None
            mem.trust_fleet_ids = None
            mem.sync_holds = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)




# === bundle entry point (Kaggle expects 2-arg agent) ===
_pp_inner_agent = agent
def agent(obs, configuration=None):  # noqa: F811  shadow for harness
    return _pp_inner_agent(obs)
