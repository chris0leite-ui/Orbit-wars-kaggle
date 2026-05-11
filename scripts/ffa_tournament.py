"""4P FFA tournament primitive (`scripts/tournament.py`'s sibling for N≥3).

Why a sibling rather than a retrofit: the 2P harness's `PairStat` and
`matrix[a][b]` data model is pair-symmetric. 4P FFA needs first-place-rate
on N-tuples, which doesn't compress to a square winrate matrix. Keeping
the 2P module untouched preserves its callers (strategy_panel, ablation,
eval_v1, manifold_check) and avoids retro-fitting placement semantics into
binary win/loss code paths.

Public API:

    run_ffa_tournament(
        focal: AgentSpec,
        background: Sequence[AgentSpec],
        seeds: Sequence[int],
        *,
        players: int = 4,
        rotate_seats: bool = True,
        out_dir: Path | None = None,
        workers: int = mp.cpu_count() or 1,
        progress: bool = False,
    ) -> FFAResult

Each (seed, focal_seat) gives one game. With `rotate_seats=True` the focal
plays every seat 0..players-1 per seed (so 32 seeds × 4 seats = 128 games
per focal). Background opponents rotate around the focal — seat 0
position is filled by the focal when its turn, otherwise by background[0],
and so on cyclically.

CLI smoke:
    python -m scripts.ffa_tournament smoke
        v2 vs {weakest, enemy_first, baseline}, 2 seeds × 4 seats = 8 games.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from kaggle_environments import make

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO / "audit" / "tournaments"
BUILTIN_AGENTS = {"random", "starter"}

AgentSpec = str | Callable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FFAGameRecord:
    seed: int
    focal_seat: int                  # 0..players-1
    seat_names: list[str]            # name per seat
    rewards: list[float | None]
    statuses: list[str]
    n_steps: int
    focal_turn_ms: list[float] = field(default_factory=list)

    def focal_first_place(self) -> bool:
        """True iff focal's reward == max(rewards).

        Live env semantics: +1 for any seat that tied the max ship-count,
        -1 otherwise. We treat any seat-of-focal that hit max as a win.
        """
        if self.rewards[self.focal_seat] is None:
            return False
        return self.rewards[self.focal_seat] == max(
            r for r in self.rewards if r is not None
        )


@dataclass
class FFAResult:
    focal: str
    background: list[str]
    players: int
    seeds: list[int]
    games: list[FFAGameRecord]

    @property
    def n_games(self) -> int:
        return len(self.games)

    @property
    def first_place_count(self) -> int:
        return sum(1 for g in self.games if g.focal_first_place())

    @property
    def first_place_rate(self) -> float:
        return self.first_place_count / self.n_games if self.n_games else 0.0

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        return _wilson_ci(self.first_place_count, self.n_games, z=z)

    def p95_focal_turn_ms(self) -> float:
        all_ms = [t for g in self.games for t in g.focal_turn_ms]
        return _p95(all_ms)

    def to_json(self) -> dict:
        return {
            "focal": self.focal,
            "background": self.background,
            "players": self.players,
            "seeds": list(self.seeds),
            "n_games": self.n_games,
            "first_place_count": self.first_place_count,
            "first_place_rate": self.first_place_rate,
            "wilson_lo_95": self.wilson_ci()[0],
            "wilson_hi_95": self.wilson_ci()[1],
            "p95_focal_turn_ms": self.p95_focal_turn_ms(),
            "games": [asdict(g) for g in self.games],
        }


# ---------------------------------------------------------------------------
# Stats helpers (parity with scripts/tournament.py)
# ---------------------------------------------------------------------------


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    import math
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(statistics.quantiles(xs, n=100)[94]) if len(xs) >= 20 else float(max(xs))


# ---------------------------------------------------------------------------
# Agent loading (mirrors tournament._load_agent — kept in-module to avoid
# cross-script imports that complicate multiprocessing pickle).
# ---------------------------------------------------------------------------


def _load_agent(spec: AgentSpec) -> AgentSpec:
    if callable(spec):
        return spec
    if isinstance(spec, str):
        if spec in BUILTIN_AGENTS:
            return spec
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"agent file not found: {spec}")
        mod_spec = importlib.util.spec_from_file_location(
            f"_ffa_agent_{path.stem}_{id(path)}", path
        )
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"could not import agent at {spec}")
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[mod_spec.name] = module
        mod_spec.loader.exec_module(module)
        if not hasattr(module, "agent"):
            raise AttributeError(f"{spec} has no `agent` callable")
        return module.agent
    raise TypeError(f"unsupported agent spec: {type(spec)!r}")


def _timed(callable_agent: Callable, sink: list[float]) -> Callable:
    """Same wrapper as tournament._timed — record per-turn wallclock in ms."""
    inner_argcount = (
        callable_agent.__code__.co_argcount
        if hasattr(callable_agent, "__code__") else 2
    )

    def wrapped(observation, configuration):
        t0 = time.perf_counter()
        try:
            args = (observation, configuration)[:inner_argcount]
            return callable_agent(*args)
        finally:
            sink.append((time.perf_counter() - t0) * 1000.0)

    return wrapped


# ---------------------------------------------------------------------------
# Per-game runner
# ---------------------------------------------------------------------------


def _run_one(
    seat_specs: list[AgentSpec],
    seat_names: list[str],
    seed: int,
    focal_seat: int,
) -> FFAGameRecord:
    seat_agents: list[AgentSpec] = []
    focal_times: list[float] = []
    for i, spec in enumerate(seat_specs):
        a = _load_agent(spec)
        if i == focal_seat and callable(a):
            a = _timed(a, focal_times)
        seat_agents.append(a)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(seat_agents)
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    statuses = [s.status for s in final]
    return FFAGameRecord(
        seed=seed,
        focal_seat=focal_seat,
        seat_names=list(seat_names),
        rewards=rewards,
        statuses=statuses,
        n_steps=len(env.steps),
        focal_turn_ms=focal_times,
    )


def _seat_assignment(
    focal_name: str, background_names: list[str], focal_seat: int,
    focal_spec: AgentSpec, background_specs: list[AgentSpec],
) -> tuple[list[AgentSpec], list[str]]:
    """Rotate `background` around `focal_seat` so it occupies the chosen seat."""
    n = len(background_specs) + 1
    seats_specs: list[AgentSpec] = [None] * n  # type: ignore[list-item]
    seats_names: list[str] = [""] * n
    seats_specs[focal_seat] = focal_spec
    seats_names[focal_seat] = focal_name
    j = 0
    for i in range(n):
        if i == focal_seat:
            continue
        seats_specs[i] = background_specs[j]
        seats_names[i] = background_names[j]
        j += 1
    return seats_specs, seats_names


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------


def _run_one_task(task: tuple) -> FFAGameRecord:
    seat_specs, seat_names, seed, focal_seat = task
    return _run_one(seat_specs, seat_names, seed, focal_seat)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_ffa_tournament(
    focal: AgentSpec,
    background: Sequence[AgentSpec],
    seeds: Sequence[int],
    *,
    focal_name: str = "focal",
    background_names: Sequence[str] | None = None,
    players: int = 4,
    rotate_seats: bool = True,
    out_dir: Path | None = None,
    workers: int = 1,
    progress: bool = False,
) -> FFAResult:
    """Run focal vs a fixed background panel over the seed bag.

    `players`: total seats in the game (1 focal + (players-1) background).
    `rotate_seats=True`: each seed produces `players` games (focal in each
        seat 0..players-1). Removes the documented A.6 seat-asymmetry from
        the first-place-rate estimate.
    `rotate_seats=False`: each seed produces 1 game with focal in seat 0.
    """
    if len(background) != players - 1:
        raise ValueError(
            f"background must have exactly players-1 entries "
            f"(got {len(background)} for players={players})"
        )

    bg_names = list(background_names) if background_names is not None else [
        spec if isinstance(spec, str) else f"bg{i}"
        for i, spec in enumerate(background)
    ]

    # Worker mode requires string-spec agents.
    if workers > 1 and (
        not isinstance(focal, str) or any(not isinstance(b, str) for b in background)
    ):
        raise ValueError(
            "workers>1 requires string agent specs (paths); pickle-friendly only"
        )

    seat_iter = range(players) if rotate_seats else (0,)
    tasks: list[tuple] = []
    for seed in seeds:
        for seat in seat_iter:
            seats_specs, seats_names = _seat_assignment(
                focal_name, bg_names, seat,
                focal, list(background),
            )
            tasks.append((seats_specs, seats_names, seed, seat))

    games: list[FFAGameRecord] = []
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(_run_one_task, tasks)):
                games.append(rec)
                if progress:
                    print(
                        f"[{i+1}/{len(tasks)}] seed={rec.seed} "
                        f"focal_seat={rec.focal_seat} "
                        f"focal_reward={rec.rewards[rec.focal_seat]} "
                        f"steps={rec.n_steps}",
                        flush=True,
                    )
    else:
        for i, task in enumerate(tasks):
            rec = _run_one_task(task)
            games.append(rec)
            if progress:
                print(
                    f"[{i+1}/{len(tasks)}] seed={rec.seed} "
                    f"focal_seat={rec.focal_seat} "
                    f"focal_reward={rec.rewards[rec.focal_seat]} "
                    f"steps={rec.n_steps}",
                    flush=True,
                )

    result = FFAResult(
        focal=focal_name,
        background=bg_names,
        players=players,
        seeds=list(seeds),
        games=games,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"ffa-{stamp}.json"
        out_path.write_text(json.dumps(result.to_json(), indent=2) + "\n")
    return result


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def _cli_smoke() -> int:
    from scripts._agent_paths import resolve_agent_path
    focal = resolve_agent_path("agents/v2/main.py")
    bg = [
        resolve_agent_path("weakest"),
        resolve_agent_path("enemy_first"),
        resolve_agent_path("baseline"),
    ]
    res = run_ffa_tournament(
        focal=focal,
        background=bg,
        focal_name="v2",
        background_names=["weakest", "enemy_first", "baseline"],
        seeds=[42, 7],
        rotate_seats=True,
        workers=mp.cpu_count() or 1,
        out_dir=DEFAULT_OUT_DIR,
        progress=True,
    )
    lo, hi = res.wilson_ci()
    print(
        f"\nv2 first-place: {res.first_place_count}/{res.n_games} "
        f"({res.first_place_rate:.1%})  Wilson95=[{lo:.1%},{hi:.1%}]  "
        f"p95_turn_ms={res.p95_focal_turn_ms():.1f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="smoke",
                        choices=["smoke"], help="Currently only `smoke`.")
    parser.parse_args(argv)
    return _cli_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
