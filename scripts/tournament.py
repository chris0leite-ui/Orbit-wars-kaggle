"""Local tournament fixture for Orbit Wars (ISSUES.md D.1).

Wraps `kaggle_environments.make("orbit_wars").run(...)` to produce a
winrate matrix over an arbitrary panel of agents and a seed bag, with
per-side P0/P1 splits (so the documented self-play asymmetry, A.6,
is visible rather than averaged out) and per-turn wallclock for the
1-second budget gate.

Public API
----------
    run_tournament(agents, seeds, *, include_self_play=True, out_dir=None)
        agents : Mapping[str, str | Callable]
            name -> file path or already-imported callable. The string
            "random" / "starter" maps to `kaggle_environments` builtins.
        seeds  : Sequence[int]
        include_self_play : if False, skip pairs where row_agent == col_agent.
        out_dir : if given, persist a JSON snapshot to
                  `<out_dir>/<utc>.json`. Default: None (in-memory only).

Returns a `TournamentResult` whose `.matrix[row][col]` contains a
`PairStat` recording outcomes for games where `row` plays as P0 and
`col` plays as P1.

CLI form (smoke):
    python -m scripts.tournament smoke
        random vs shipped baseline, 4 seeds, both sides.

Reward signal is the source of truth for winrate (see
`audit/friction.md::env-not-fully-seed-deterministic`); ship deltas and
turn counts are noisy estimators.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from kaggle_environments import make

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO / "audit" / "tournaments"
BUILTIN_AGENTS = {"random", "starter"}

AgentSpec = str | Callable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    seed: int
    rewards: list[float | None]
    statuses: list[str]
    n_steps: int
    final_ship_delta_p0_minus_p1: float
    p0_turn_ms: list[float]
    p1_turn_ms: list[float]


@dataclass
class PairStat:
    p0_name: str
    p1_name: str
    n: int = 0
    p0_wins: int = 0
    p1_wins: int = 0
    draws: int = 0
    mean_ship_delta_p0_minus_p1: float = 0.0
    p0_p95_turn_ms: float = 0.0
    p1_p95_turn_ms: float = 0.0
    wilson_lo: float = 0.0
    wilson_hi: float = 0.0
    games: list[GameRecord] = field(default_factory=list)

    @property
    def p0_winrate(self) -> float:
        return self.p0_wins / self.n if self.n else 0.0


@dataclass
class TournamentResult:
    timestamp_utc: str
    agents: dict[str, str]
    seeds: list[int]
    include_self_play: bool
    matrix: dict[str, dict[str, PairStat]]

    def to_json_dict(self) -> dict:
        """JSON snapshot with the per-turn timing arrays dropped.

        `p95_turn_ms` is preserved on each PairStat, so the load-bearing
        budget gate is still readable. The per-turn arrays would inflate
        the artifact by ~30x with no analytical benefit.
        """
        def _stat(stat: PairStat) -> dict:
            d = asdict(stat)
            for game in d.get("games", []):
                game.pop("p0_turn_ms", None)
                game.pop("p1_turn_ms", None)
            return d

        return {
            "timestamp_utc": self.timestamp_utc,
            "agents": self.agents,
            "seeds": self.seeds,
            "include_self_play": self.include_self_play,
            "matrix": {
                a: {b: _stat(stat) for b, stat in row.items()}
                for a, row in self.matrix.items()
            },
        }


# ---------------------------------------------------------------------------
# Agent loading + timing wrapper
# ---------------------------------------------------------------------------


def _load_agent(spec: AgentSpec) -> AgentSpec:
    """Resolve an AgentSpec to either a builtin name or a callable."""
    if callable(spec):
        return spec
    if isinstance(spec, str):
        if spec in BUILTIN_AGENTS:
            return spec
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"agent file not found: {spec}")
        mod_spec = importlib.util.spec_from_file_location(
            f"_agent_{path.stem}_{id(path)}", path
        )
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"could not import agent at {spec}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        if not hasattr(module, "agent"):
            raise AttributeError(f"{spec} has no `agent` callable")
        return module.agent
    raise TypeError(f"unsupported agent spec: {type(spec)!r}")


def _timed(callable_agent: Callable, sink: list[float]) -> Callable:
    """Wrap an agent callable to record per-turn wallclock in milliseconds.

    `kaggle_environments.agent.Agent.act` trims its `(obs, config)` arg pair
    to `agent.__code__.co_argcount` before calling. The wrapper therefore
    declares an explicit 2-positional-arg signature so the env never strips
    `obs` away, and trims internally to whatever the inner agent expects.
    """
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


def _final_ships_by_owner(state) -> dict[int, int]:
    """Sum ships on owned planets + in fleets per player at the final step."""
    obs0 = state[0].observation
    by_owner: dict[int, int] = {}
    for p in obs0.get("planets", []):
        owner, ships = p[1], p[5]
        if owner >= 0:
            by_owner[owner] = by_owner.get(owner, 0) + ships
    for f in obs0.get("fleets", []):
        owner, ships = f[1], f[6]
        by_owner[owner] = by_owner.get(owner, 0) + ships
    return by_owner


def _run_one(
    p0_spec: AgentSpec,
    p1_spec: AgentSpec,
    seed: int,
) -> GameRecord:
    p0 = _load_agent(p0_spec)
    p1 = _load_agent(p1_spec)
    p0_times: list[float] = []
    p1_times: list[float] = []
    if callable(p0):
        p0 = _timed(p0, p0_times)
    if callable(p1):
        p1 = _timed(p1, p1_times)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([p0, p1])
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    statuses = [s.status for s in final]
    ships = _final_ships_by_owner(final)
    return GameRecord(
        seed=seed,
        rewards=rewards,
        statuses=statuses,
        n_steps=len(env.steps),
        final_ship_delta_p0_minus_p1=float(ships.get(0, 0) - ships.get(1, 0)),
        p0_turn_ms=p0_times,
        p1_turn_ms=p1_times,
    )


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(statistics.quantiles(xs, n=100)[94]) if len(xs) >= 20 else float(max(xs))


def _classify(rewards: list[float | None]) -> str:
    """Map a final-step reward pair to one of {p0_win, p1_win, draw}.

    kaggle_environments.evaluate returns +1 / -1 / 0 for win / loss / draw,
    but a simultaneous-tie at the step limit has been observed as
    rewards=[1, 1] (both rewarded). Any non-strict-loss for P0 with P0 > P1
    counts as a P0 win; same mirror for P1; otherwise draw.
    """
    r0, r1 = rewards[0] or 0, rewards[1] or 0
    if r0 > r1:
        return "p0_win"
    if r1 > r0:
        return "p1_win"
    return "draw"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_tournament(
    agents: Mapping[str, AgentSpec],
    seeds: Sequence[int],
    *,
    include_self_play: bool = True,
    out_dir: Path | None = None,
    progress: bool = False,
) -> TournamentResult:
    if not agents:
        raise ValueError("agents map is empty")
    if not seeds:
        raise ValueError("seeds is empty")

    names = list(agents.keys())
    matrix: dict[str, dict[str, PairStat]] = {a: {} for a in names}
    total_games = sum(
        len(seeds)
        for a in names
        for b in names
        if include_self_play or a != b
    )
    done = 0

    for a in names:
        for b in names:
            if not include_self_play and a == b:
                continue
            stat = PairStat(p0_name=a, p1_name=b)
            for seed in seeds:
                rec = _run_one(agents[a], agents[b], seed)
                stat.games.append(rec)
                stat.n += 1
                kind = _classify(rec.rewards)
                if kind == "p0_win":
                    stat.p0_wins += 1
                elif kind == "p1_win":
                    stat.p1_wins += 1
                else:
                    stat.draws += 1
                done += 1
                if progress:
                    print(
                        f"[{done}/{total_games}] {a} (P0) vs {b} (P1) seed={seed}: "
                        f"{kind} steps={rec.n_steps} dShips={rec.final_ship_delta_p0_minus_p1:+.0f}",
                        flush=True,
                    )
            stat.mean_ship_delta_p0_minus_p1 = (
                statistics.fmean(g.final_ship_delta_p0_minus_p1 for g in stat.games)
                if stat.games
                else 0.0
            )
            stat.p0_p95_turn_ms = _p95(
                [t for g in stat.games for t in g.p0_turn_ms]
            )
            stat.p1_p95_turn_ms = _p95(
                [t for g in stat.games for t in g.p1_turn_ms]
            )
            stat.wilson_lo, stat.wilson_hi = _wilson_ci(stat.p0_wins, stat.n)
            matrix[a][b] = stat

    result = TournamentResult(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        agents={k: (v if isinstance(v, str) else f"<callable:{v.__name__}>") for k, v in agents.items()},
        seeds=list(seeds),
        include_self_play=include_self_play,
        matrix=matrix,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{result.timestamp_utc.replace(':', '').replace('-', '')}.json"
        out_path.write_text(json.dumps(result.to_json_dict(), indent=2))
        if progress:
            print(f"wrote {out_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _smoke() -> int:
    baseline = str(REPO / "data" / "main.py")
    result = run_tournament(
        agents={"random": "random", "baseline": baseline},
        seeds=[42, 1, 7, 13],
        include_self_play=False,
        out_dir=DEFAULT_OUT_DIR,
        progress=True,
    )
    print()
    print("=== smoke summary ===")
    for row in result.matrix:
        for col, stat in result.matrix[row].items():
            print(
                f"{row} (P0) vs {col} (P1): {stat.p0_wins}/{stat.n} "
                f"(Wilson 95% {stat.wilson_lo:.2f}..{stat.wilson_hi:.2f}); "
                f"p95 turn ms P0={stat.p0_p95_turn_ms:.1f} P1={stat.p1_p95_turn_ms:.1f}; "
                f"mean dShips P0-P1={stat.mean_ship_delta_p0_minus_p1:+.0f}"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Orbit Wars local tournament")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke", help="random vs baseline, 4 seeds, both sides")
    args = parser.parse_args(argv)
    if args.cmd == "smoke":
        return _smoke()
    return 2


if __name__ == "__main__":
    sys.exit(main())
