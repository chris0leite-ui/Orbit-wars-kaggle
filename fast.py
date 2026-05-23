"""fast.py — one-file iteration entry point for Orbit Wars.

Purpose
-------
Triage and evaluate a new agent idea against known baselines in <2 min.
Replaces the diffuse set of harnesses under `scripts/` (run_ablations,
run_v7_wide_deep_smoke, tournament, ffa_tournament, ab_variants,
strategy_panel, ...) for the common "is this idea better?" question.

Three subcommands:

    python fast.py smoke <agent>        # cheap triage vs random + nearest
    python fast.py eval  <agent>        # adaptive Wilson-gated A/B vs v7_0
    python fast.py play  <agent>        # one game, verbose, for inspection
    python fast.py bench <agent>        # per-turn ms vs the 1000ms budget

Agent specs
-----------
A "<agent>" can be any of:
    - a baseline short name: v7_0, v4_planner, v7_minimax, v3.5.1, random,
      starter, nearest
    - a path to a .py file with `def agent(obs, configuration=None)`
    - a directory containing main.py (the agents/<name>/ convention)

The submissions bundle path is preferred when both source and bundle
exist (parity-safer for evaluation).

Design — one file, on purpose
-----------------------------
This script intentionally stays as a single file. The seams aren't
proven yet; premature modularisation is the very disease this is curing.
If `fast.py` grows past ~600 lines or sprouts a second non-trivial
caller (e.g. notebook), split then.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------

# Short-name → spec (resolved by `resolve_agent_spec`).
# Order of preference: bundled submission > source agent > kaggle builtin.
_BASELINES: dict[str, str] = {
    # Bundled, parity-tested, primary baseline.
    "v7_0":       str(REPO / "submissions" / "v7_0_drop_one.py"),
    "v7_1":       str(REPO / "submissions" / "v7_1_open_drop_comets.py"),
    "v4_planner": str(REPO / "submissions" / "v4_planner.py"),
    "v7_minimax": str(REPO / "submissions" / "v7_minimax.py"),
    "v3.5.1":     str(REPO / "submissions" / "v3.5.1.py"),
    # Cheap source-tree baselines for smoke triage.
    "nearest":    str(REPO / "agents" / "simple" / "nearest.py"),
    "roi":        str(REPO / "agents" / "simple" / "roi.py"),
    # kaggle_environments builtins (passed as strings to env.run).
    "random":     "random",
    "starter":    "starter",
}

_KAGGLE_BUILTINS = {"random", "starter"}

# Smoke panel: two cheap floors for "dead-on-arrival" triage.
SMOKE_OPPONENTS = ["random", "nearest"]
DEFAULT_BASELINE = "v7_0"
# Pre-submit calibration panel: 3 architecturally distinct opponents
# covering the live ladder's distribution (drop-one chooser / receding-
# horizon planner / aggressive snipe). Closes the
# `local-overpredict-2x` friction: v3.5.1 (5/12) and geo v3.1 (5/14)
# both passed single-opponent A/Bs vs v7_0 but regressed on the ladder
# because the panel was a monoculture. Use via `fast.py eval --vs-panel`.
DEFAULT_PANEL = ["v7_0", "v4_planner", "v3.5.1"]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def p_quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    if len(xs) < 2:
        return float(xs[0])
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[k])


# ---------------------------------------------------------------------------
# Agent resolution
# ---------------------------------------------------------------------------


def resolve_agent_spec(spec: str) -> tuple[str, str]:
    """Return (display_name, path-or-builtin) for an agent spec.

    Acceptable inputs:
        "v7_0"                          -> baseline registry
        "random"                        -> kaggle builtin (returned as-is)
        "submissions/v7_0_drop_one.py"  -> literal path
        "agents/v7_minimax"             -> directory; reads main.py inside
        "agents/v7_minimax/main.py"     -> direct file path
    """
    if spec in _BASELINES:
        target = _BASELINES[spec]
        if target in _KAGGLE_BUILTINS:
            return (spec, target)
        if not Path(target).is_file():
            raise FileNotFoundError(
                f"baseline {spec!r} points at {target!r} but file is missing"
            )
        return (spec, target)
    if spec in _KAGGLE_BUILTINS:
        return (spec, spec)
    p = Path(spec)
    if p.is_dir():
        main_py = p / "main.py"
        if not main_py.is_file():
            raise FileNotFoundError(f"{p} is a directory but has no main.py")
        return (p.name, str(main_py.resolve()))
    if p.is_file():
        # Prefer the agents/<dir>/main.py convention's directory name
        # over the literal "main" stem for display.
        if p.name == "main.py" and p.parent != p.parent.parent:
            return (p.parent.name, str(p.resolve()))
        return (p.stem, str(p.resolve()))
    # Convention fallback: `geo` -> `agents/geo/main.py`, `roi` -> `agents/simple/roi.py`.
    # Matches scripts/_agent_paths.py resolution order.
    dir_main = REPO / "agents" / spec / "main.py"
    if dir_main.is_file():
        return (spec, str(dir_main.resolve()))
    simple_py = REPO / "agents" / "simple" / f"{spec}.py"
    if simple_py.is_file():
        return (spec, str(simple_py.resolve()))
    raise FileNotFoundError(f"unknown agent spec: {spec!r}")


# ---------------------------------------------------------------------------
# Agent loading + timed wrapper
# ---------------------------------------------------------------------------


def _load_callable(path_or_builtin: str) -> Callable | str:
    """Load an agent from a file path, or return the builtin name."""
    if path_or_builtin in _KAGGLE_BUILTINS:
        return path_or_builtin
    path = Path(path_or_builtin)
    mod_spec = importlib.util.spec_from_file_location(
        f"_fastagent_{path.stem}_{id(path)}", path
    )
    if mod_spec is None or mod_spec.loader is None:
        raise ImportError(f"could not load module at {path}")
    module = importlib.util.module_from_spec(mod_spec)
    # Register before exec so @dataclass / single-file bundles resolve names.
    sys.modules[mod_spec.name] = module
    mod_spec.loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise AttributeError(f"{path} has no top-level `agent` callable")
    return module.agent


def _timed(callable_agent: Callable, sink: list[float]) -> Callable:
    """Wrap an agent to record per-turn wallclock in milliseconds.

    Matches the trick in scripts/tournament.py::_timed: the wrapper
    declares 2 positional args so kaggle_environments doesn't strip
    `obs` away, then we trim to the inner agent's actual arg count.
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
# Single-game runner
# ---------------------------------------------------------------------------


@dataclass
class GameResult:
    seed: int
    outcome: str             # "p0_win" | "p1_win" | "draw" | "error"
    rewards: tuple[float | None, float | None]
    n_steps: int
    p0_turn_ms: list[float]
    p1_turn_ms: list[float]
    # Termination + final-state fields (Phase B — elimination instrumentation).
    # "elimination" = at most 1 player still has planets or in-flight fleets;
    # "turn_cap" = game hit episodeSteps-2 with both players still alive;
    # "unknown" = could not inspect the final observation (error path).
    terminated_by: str = "unknown"
    final_planet_counts: tuple[int, int] = (0, 0)
    final_fleet_counts: tuple[int, int] = (0, 0)


def _classify(rewards: tuple) -> str:
    r0 = rewards[0] if rewards[0] is not None else 0
    r1 = rewards[1] if rewards[1] is not None else 0
    if r0 > r1:
        return "p0_win"
    if r1 > r0:
        return "p1_win"
    return "draw"


def _classify_termination(final_obs) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Inspect the env's final observation. Returns:
       (termination_kind, (p0_planets, p1_planets), (p0_fleets, p1_fleets))

    Matches the interpreter's alive-player rule (lib/game/interpreter.py:856-864):
    a player is alive iff they own >=1 planet OR have >=1 in-flight fleet.
    Termination is "elimination" if the alive-set has <=1 player at the
    final tick (the only other way to terminate is the turn cap)."""
    planets = list(getattr(final_obs, "planets", []) or [])
    fleets = list(getattr(final_obs, "fleets", []) or [])
    p0_planets = sum(1 for p in planets if int(p[1]) == 0)
    p1_planets = sum(1 for p in planets if int(p[1]) == 1)
    p0_fleets = sum(1 for f in fleets if int(f[1]) == 0)
    p1_fleets = sum(1 for f in fleets if int(f[1]) == 1)
    alive = set()
    for p in planets:
        if int(p[1]) != -1:
            alive.add(int(p[1]))
    for f in fleets:
        alive.add(int(f[1]))
    kind = "elimination" if len(alive) <= 1 else "turn_cap"
    return kind, (p0_planets, p1_planets), (p0_fleets, p1_fleets)


def play_one(seed: int, p0_path: str, p1_path: str, *,
             record_timing: bool = True) -> GameResult:
    """Play a single 2P game and return outcome + per-turn timing."""
    # Late import: kaggle_environments is slow to import; defer to worker.
    from kaggle_environments import make

    p0 = _load_callable(p0_path)
    p1 = _load_callable(p1_path)
    p0_times: list[float] = []
    p1_times: list[float] = []
    if record_timing and callable(p0):
        p0 = _timed(p0, p0_times)
    if record_timing and callable(p1):
        p1 = _timed(p1, p1_times)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run([p0, p1])
    except Exception:
        return GameResult(
            seed=seed,
            outcome="error",
            rewards=(None, None),
            n_steps=len(env.steps) if hasattr(env, "steps") else 0,
            p0_turn_ms=p0_times,
            p1_turn_ms=p1_times,
        )
    final = env.steps[-1]
    rewards = (final[0].reward, final[1].reward)
    term_kind, planet_counts, fleet_counts = _classify_termination(
        final[0].observation
    )
    return GameResult(
        seed=seed,
        outcome=_classify(rewards),
        rewards=rewards,
        n_steps=len(env.steps),
        p0_turn_ms=p0_times,
        p1_turn_ms=p1_times,
        terminated_by=term_kind,
        final_planet_counts=planet_counts,
        final_fleet_counts=fleet_counts,
    )


# Picklable worker for ProcessPoolExecutor. Tuple args (not kwargs) to keep
# the multiprocessing call site simple.
def _play_one_task(args: tuple[int, str, str]) -> GameResult:
    seed, p0_path, p1_path = args
    return play_one(seed, p0_path, p1_path)


# ---------------------------------------------------------------------------
# Balanced A/B (focal at both seats)
# ---------------------------------------------------------------------------


@dataclass
class PanelStat:
    focal: str
    opp: str
    focal_wins: int = 0
    opp_wins: int = 0
    draws: int = 0
    errors: int = 0
    n: int = 0
    focal_p0_wins: int = 0
    focal_p1_wins: int = 0
    focal_turn_ms: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0
    # Per-game outcomes for the --by-archetype reporter. Each entry is
    # (seed, focal_won). One seed appears twice (P0 + P1 sides).
    per_game: list[tuple[int, bool]] = field(default_factory=list)

    @property
    def winrate(self) -> float:
        return self.focal_wins / self.n if self.n else 0.0

    @property
    def wilson(self) -> tuple[float, float]:
        return wilson_ci(self.focal_wins, self.n)


def _absorb(stat: PanelStat, result: GameResult, focal_is_p0: bool) -> None:
    stat.n += 1
    if result.outcome == "error":
        stat.errors += 1
        return
    focal_won = (focal_is_p0 and result.outcome == "p0_win") or \
                (not focal_is_p0 and result.outcome == "p1_win")
    opp_won   = (focal_is_p0 and result.outcome == "p1_win") or \
                (not focal_is_p0 and result.outcome == "p0_win")
    if focal_won:
        stat.focal_wins += 1
        if focal_is_p0:
            stat.focal_p0_wins += 1
        else:
            stat.focal_p1_wins += 1
    elif opp_won:
        stat.opp_wins += 1
    else:
        stat.draws += 1
    times = result.p0_turn_ms if focal_is_p0 else result.p1_turn_ms
    stat.focal_turn_ms.extend(times)
    stat.per_game.append((result.seed, focal_won))


def _balanced_pairs(seeds: Sequence[int], focal_path: str, opp_path: str
                    ) -> list[tuple[int, str, str, bool]]:
    """Build a balanced seed × seat schedule. Returns (seed, p0, p1, focal_is_p0)."""
    pairs: list[tuple[int, str, str, bool]] = []
    for s in seeds:
        pairs.append((s, focal_path, opp_path, True))   # focal as P0
        pairs.append((s, opp_path, focal_path, False))  # focal as P1
    return pairs


def play_panel(focal_path: str, focal_name: str,
               opp_path: str, opp_name: str,
               seeds: Sequence[int], workers: int) -> PanelStat:
    """Run focal vs opp on all seeds, both seats, in parallel."""
    stat = PanelStat(focal=focal_name, opp=opp_name)
    pairs = _balanced_pairs(seeds, focal_path, opp_path)
    t0 = time.perf_counter()
    if workers <= 1:
        for seed, p0, p1, focal_is_p0 in pairs:
            r = _play_one_task((seed, p0, p1))
            _absorb(stat, r, focal_is_p0)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_play_one_task, (seed, p0, p1)): focal_is_p0
                for seed, p0, p1, focal_is_p0 in pairs
            }
            for fut in as_completed(futs):
                _absorb(stat, fut.result(), focal_is_p0=futs[fut])
    stat.elapsed_s = time.perf_counter() - t0
    return stat


# ---------------------------------------------------------------------------
# smoke — cheap triage
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    focal_name, focal_path = resolve_agent_spec(args.agent)
    seeds = list(range(args.seeds))
    print(f"== smoke {focal_name} ({focal_path}) ==")
    print(f"   {args.seeds} seeds × 2 seats × {len(SMOKE_OPPONENTS)} opp "
          f"= {2*args.seeds*len(SMOKE_OPPONENTS)} games, {args.workers} workers\n")
    print(f"   {'opponent':14s} {'wins':>8s} {'%':>6s} {'Wlo':>6s} {'Whi':>6s} "
          f"{'p95ms':>7s} {'seconds':>8s}")
    rows = []
    for opp_short in SMOKE_OPPONENTS:
        opp_name, opp_path = resolve_agent_spec(opp_short)
        stat = play_panel(focal_path, focal_name, opp_path, opp_name,
                          seeds, args.workers)
        lo, hi = stat.wilson
        p95 = p_quantile(stat.focal_turn_ms, 0.95)
        rows.append((opp_name, stat))
        print(f"   {opp_name:14s} {stat.focal_wins:>4d}/{stat.n:<3d}  "
              f"{100*stat.winrate:>5.1f}  {lo:>5.2f}  {hi:>5.2f}  "
              f"{p95:>6.0f}  {stat.elapsed_s:>7.1f}")
    # Triage verdict: must beat both floors decisively.
    bad = [name for name, s in rows if s.wilson[0] < 0.55]
    print()
    if not bad:
        print("   verdict: PASS  (clears both smoke floors with Wlo≥0.55)")
        return 0
    print(f"   verdict: WEAK  (does not clear smoke floor vs: {', '.join(bad)})")
    return 1


# ---------------------------------------------------------------------------
# eval — adaptive Wilson-gated A/B
# ---------------------------------------------------------------------------


def _eval_vs_one(focal_path: str, focal_name: str,
                 opp_path: str, opp_name: str,
                 max_seeds: int, gate: float, workers: int,
                 seed_pool: Sequence[int] | None = None,
                 full_panel: bool = False,
                 ) -> tuple[str, float, float, int, int, list[float], float, list[tuple[int, bool]]]:
    """Run the adaptive Wilson-gated A/B vs a single opponent.

    `seed_pool` overrides the default ``range(max_seeds)`` seed source —
    used by ``--geometry-panel`` to draw from
    ``lib.seed_panel.SEED_PANEL_128_INTERLEAVED`` instead. The pool is
    sliced by tier index (not by value) so adaptive tiering still works.

    `full_panel=True` disables adaptive early-stop entirely; every seed
    in ``seed_pool`` (or ``range(max_seeds)``) is played.

    Returns: (verdict, wlo, whi, wins, n, focal_turn_ms, total_elapsed_s,
              per_game_outcomes).
    """
    if seed_pool is None:
        seed_pool = list(range(max_seeds))
    else:
        seed_pool = list(seed_pool[:max_seeds])
    max_seeds = len(seed_pool)

    tiers: list[int] = []
    n = 16
    while n <= max_seeds:
        tiers.append(n)
        if n >= max_seeds:
            break
        n = min(max_seeds, n * 2)
    if tiers and tiers[-1] < max_seeds:
        tiers.append(max_seeds)
    if not tiers:
        tiers = [max_seeds]

    cumulative_wins = 0
    cumulative_n = 0
    cumulative_times: list[float] = []
    total_elapsed = 0.0
    last_seed_idx = 0
    verdict = "FAIL"
    per_game: list[tuple[int, bool]] = []

    for tier_n in tiers:
        new_seeds = seed_pool[last_seed_idx:tier_n]
        if not new_seeds:
            continue
        stat = play_panel(focal_path, focal_name, opp_path, opp_name,
                          new_seeds, workers)
        per_game.extend(stat.per_game)
        cumulative_wins += stat.focal_wins
        cumulative_n += stat.n
        cumulative_times.extend(stat.focal_turn_ms)
        total_elapsed += stat.elapsed_s
        last_seed_idx = tier_n
        lo, hi = wilson_ci(cumulative_wins, cumulative_n)
        wr = cumulative_wins / cumulative_n if cumulative_n else 0.0
        print(f"   n={cumulative_n:>3d}  "
              f"wins={cumulative_wins:>3d}/{cumulative_n:<3d} "
              f"({100*wr:>5.1f}%)  Wlo={lo:.3f}  Whi={hi:.3f}  "
              f"elapsed={stat.elapsed_s:.1f}s", end="  ")
        if not full_panel and lo >= gate:
            verdict = "PASS"
            print(f"-> STOP  verdict=PASS  (Wlo≥{gate})")
            break
        if not full_panel and hi < gate:
            verdict = "FAIL"
            print(f"-> STOP  verdict=FAIL  (Whi<{gate})")
            break
        if tier_n >= max_seeds:
            verdict = "PASS" if lo >= gate else ("FAIL" if hi < gate else "INCONCLUSIVE")
            print(f"-> STOP  verdict={verdict}  (max seeds reached)")
            break
        print("-> CONTINUE  (CI brackets gate)")

    lo, hi = wilson_ci(cumulative_wins, cumulative_n)
    return (verdict, lo, hi, cumulative_wins, cumulative_n,
            cumulative_times, total_elapsed, per_game)


def _parse_panel_arg(s: str | None) -> list[str]:
    """Parse a --vs-panel argument string into a list of opponent specs.

    Accepts 'default' (use DEFAULT_PANEL) or comma-separated names like
    'v7_0,v4_planner,v3.5.1'. Whitespace around commas is tolerated.
    """
    if not s or s == "default":
        return list(DEFAULT_PANEL)
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _report_by_archetype(focal_name: str,
                         per_opp_games: list[tuple[str, list[tuple[int, bool]]]]
                         ) -> None:
    """Print a per-archetype winrate breakdown for the focal agent.

    Each opponent's per-game outcomes are grouped by archetype using
    ``lib.seed_panel.ARCHETYPE_OF_SEED``. Seeds outside the panel are
    bucketed as ``<not-in-panel>``. Each seed contributes 2 games
    (P0 + P1 sides) because ``_balanced_pairs`` plays both seats.
    """
    try:
        from lib.seed_panel import ARCHETYPE_OF_SEED
    except Exception as e:
        print(f"\n   [by-archetype skipped: {e}]")
        return

    print(f"\n   per-archetype winrate ({focal_name}):")
    for opp_name, games in per_opp_games:
        if not games:
            continue
        bucket: dict[str, list[int]] = {}
        for seed, focal_won in games:
            arch = ARCHETYPE_OF_SEED.get(seed, "<not-in-panel>")
            bucket.setdefault(arch, []).append(1 if focal_won else 0)
        rows = sorted(bucket.items())
        if len(per_opp_games) > 1:
            print(f"     vs {opp_name}:")
        extremes = 0
        for arch, wins in rows:
            wr = sum(wins) / len(wins)
            flag = ""
            if wr <= 0.25:
                flag = " <-- LOSING"
                extremes += 1
            elif wr >= 0.75:
                flag = " <-- winning"
                extremes += 1
            print(f"       {arch:<55s}  {sum(wins):>2d}/{len(wins):<2d}  "
                  f"{wr:>5.0%}{flag}")
        n_archs = len(rows)
        print(f"     [{n_archs} archetypes, {extremes} extreme "
              f"(<=25% or >=75%)]")


def cmd_eval(args: argparse.Namespace) -> int:
    focal_name, focal_path = resolve_agent_spec(args.agent)
    gate = float(args.gate)

    seed_pool: Sequence[int] | None = None
    if getattr(args, "geometry_panel", False):
        # Interleaved order ensures the first N (for any N) covers
        # archetypes round-robin instead of clustering by archetype.
        from lib.seed_panel import SEED_PANEL_128_INTERLEAVED
        seed_pool = SEED_PANEL_128_INTERLEAVED
        if args.max_seeds == 64:  # argparse default — bump to full panel
            args.max_seeds = 128
        print(f"== using geometry panel: {len(seed_pool)} seeds, "
              f"32 archetypes, interleaved ==")

    panel: list[str]
    if args.vs_panel is not None:
        # Hard gate: panel mode requires --require-h2h <champion>. Closes the
        # panel-passes-h2h-vs-current-fails friction (4 recurrences: v13/v14/
        # v17/v18). Panel measures opponents one-at-a-time; the ladder is a
        # mixture + same-family agents play differently. Champion h2h is the
        # only signal that catches non-transitive A>B>C>A regressions.
        if args.require_h2h is None:
            print("REFUSE: --vs-panel requires --require-h2h <current-champion>.")
            print("        Reason: panel pass without h2h-vs-champion has been")
            print("        a false-positive in 4 of the last 8 submissions.")
            print("        Pass the current rolling champion's source path, e.g.:")
            print("          --require-h2h agents/baseline")
            print("        To bypass (not recommended), set FAST_PY_SKIP_H2H_GATE=1.")
            import os as _os
            if _os.environ.get("FAST_PY_SKIP_H2H_GATE") != "1":
                return 2
            print("WARNING: FAST_PY_SKIP_H2H_GATE=1 — gate bypassed.")
        panel = _parse_panel_arg(args.vs_panel)
        if args.vs != DEFAULT_BASELINE:
            print(f"WARNING: --vs {args.vs!r} ignored when --vs-panel is set")
        # Prepend the required h2h so the champion is always opponent #1
        # and a champion-only failure short-circuits the rest of the panel.
        if args.require_h2h is not None and args.require_h2h not in panel:
            panel = [args.require_h2h] + panel
    else:
        panel = [args.vs]

    if len(panel) > 1:
        print(f"== panel-eval {focal_name} vs [{', '.join(panel)}]  "
              f"gate Wlo≥{gate:.2f} per opponent ==")
        print("   (3-opponent calibration panel; closes local-overpredict-2x friction)")

    per_opponent_results: list[tuple[str, str, float, float, int, int]] = []
    overall_times: list[float] = []
    overall_elapsed = 0.0
    per_opp_games: list[tuple[str, list[tuple[int, bool]]]] = []

    for opp_spec in panel:
        opp_name, opp_path = resolve_agent_spec(opp_spec)
        if opp_name == focal_name:
            print(f"-- skipping {opp_name}: same agent as focal --")
            continue
        if len(panel) > 1:
            print(f"\n-- vs {opp_name} --")
        else:
            print(f"== eval {focal_name} vs {opp_name}  gate Wlo≥{gate:.2f} ==")
        verdict, lo, hi, wins, n, times, elapsed, per_game = _eval_vs_one(
            focal_path, focal_name, opp_path, opp_name,
            args.max_seeds, gate, args.workers,
            seed_pool=seed_pool,
            full_panel=getattr(args, "full_panel", False),
        )
        per_opponent_results.append((opp_name, verdict, lo, hi, wins, n))
        overall_times.extend(times)
        overall_elapsed += elapsed
        per_opp_games.append((opp_name, per_game))

    p50 = p_quantile(overall_times, 0.50)
    p95 = p_quantile(overall_times, 0.95)
    pmax = max(overall_times) if overall_times else 0.0
    print(f"\n   focal turn-ms  p50={p50:.0f}  p95={p95:.0f}  max={pmax:.0f}"
          f"   total elapsed {overall_elapsed:.1f}s")

    if getattr(args, "by_archetype", False):
        _report_by_archetype(focal_name, per_opp_games)

    if len(per_opponent_results) > 1:
        print("\n   per-opponent summary:")
        for opp_name, verdict, lo, hi, wins, n in per_opponent_results:
            wr = (100 * wins / n) if n else 0.0
            print(f"     {opp_name:>14s}  {wins:>3d}/{n:<3d} ({wr:>5.1f}%)  "
                  f"Wlo={lo:.3f}  Whi={hi:.3f}  -> {verdict}")
        # Panel verdict: PASS iff every opponent cleared Wlo≥gate.
        # Catches non-transitive A>B>C>A loops (H22 rationale).
        all_pass = all(v == "PASS" for _, v, _, _, _, _ in per_opponent_results)
        any_fail = any(v == "FAIL" for _, v, _, _, _, _ in per_opponent_results)
        worst_lo = min((lo for _, _, lo, _, _, _ in per_opponent_results),
                       default=0.0)
        panel_verdict = "PASS" if all_pass else ("FAIL" if any_fail else "INCONCLUSIVE")
        print(f"   panel verdict: {panel_verdict}  (worst Wlo={worst_lo:.3f})")
        return 0 if all_pass else 1

    if not per_opponent_results:
        print("WARNING: no opponents to evaluate")
        return 1
    return 0 if per_opponent_results[0][1] == "PASS" else 1


# ---------------------------------------------------------------------------
# play — single game, verbose
# ---------------------------------------------------------------------------


def cmd_play(args: argparse.Namespace) -> int:
    focal_name, focal_path = resolve_agent_spec(args.agent)
    opp_name,   opp_path   = resolve_agent_spec(args.vs)
    print(f"== play  {focal_name} (P{0 if not args.swap else 1}) "
          f"vs {opp_name} (P{1 if not args.swap else 0})  seed={args.seed} ==")
    if args.swap:
        p0_path, p0_name = opp_path, opp_name
        p1_path, p1_name = focal_path, focal_name
    else:
        p0_path, p0_name = focal_path, focal_name
        p1_path, p1_name = opp_path, opp_name
    t0 = time.perf_counter()
    result = play_one(args.seed, p0_path, p1_path)
    elapsed = time.perf_counter() - t0
    winner = {"p0_win": p0_name, "p1_win": p1_name,
              "draw": "(draw)", "error": "(error)"}[result.outcome]
    print(f"   outcome:  {result.outcome}   winner: {winner}")
    print(f"   rewards:  P0={result.rewards[0]}  P1={result.rewards[1]}")
    print(f"   n_steps:  {result.n_steps}")
    print(f"   wallclock {elapsed:.1f}s")
    if result.p0_turn_ms:
        print(f"   {p0_name} turn-ms  p50={p_quantile(result.p0_turn_ms, 0.50):.0f}"
              f"  p95={p_quantile(result.p0_turn_ms, 0.95):.0f}"
              f"  max={max(result.p0_turn_ms):.0f}")
    if result.p1_turn_ms:
        print(f"   {p1_name} turn-ms  p50={p_quantile(result.p1_turn_ms, 0.50):.0f}"
              f"  p95={p_quantile(result.p1_turn_ms, 0.95):.0f}"
              f"  max={max(result.p1_turn_ms):.0f}")
    return 0 if result.outcome != "error" else 1


# ---------------------------------------------------------------------------
# bench — per-turn budget check
# ---------------------------------------------------------------------------


def cmd_bench(args: argparse.Namespace) -> int:
    focal_name, focal_path = resolve_agent_spec(args.agent)
    opp_name,   opp_path   = resolve_agent_spec(args.vs)
    print(f"== bench {focal_name} vs {opp_name}  budget 1000ms ==")
    all_times: list[float] = []
    elapsed = 0.0
    for seed in range(args.games):
        t0 = time.perf_counter()
        result = play_one(seed, focal_path, opp_path)
        elapsed += time.perf_counter() - t0
        all_times.extend(result.p0_turn_ms)
        print(f"   seed={seed:<3d} n_steps={result.n_steps:<4d} "
              f"focal p95={p_quantile(result.p0_turn_ms, 0.95):>4.0f}ms  "
              f"max={max(result.p0_turn_ms) if result.p0_turn_ms else 0:>4.0f}ms  "
              f"outcome={result.outcome}")
    if not all_times:
        print("   no timing samples")
        return 1
    p50 = p_quantile(all_times, 0.50)
    p95 = p_quantile(all_times, 0.95)
    p99 = p_quantile(all_times, 0.99)
    pmax = max(all_times)
    over = sum(1 for t in all_times if t >= 1000.0)
    print(f"\n   focal {focal_name}: n={len(all_times)}  "
          f"p50={p50:.0f}  p95={p95:.0f}  p99={p99:.0f}  max={pmax:.0f}ms  "
          f"over_1000ms={over}")
    print(f"   total wallclock {elapsed:.1f}s")
    verdict = "PASS" if p95 < 800 and over == 0 else "WATCH"
    print(f"   verdict: {verdict}  (gate: p95<800ms AND zero >=1000ms)")
    return 0 if verdict == "PASS" else 1


# ---------------------------------------------------------------------------
# baselines — list registry
# ---------------------------------------------------------------------------


def cmd_baselines(args: argparse.Namespace) -> int:
    print("baseline registry:")
    for name, path in _BASELINES.items():
        exists = "builtin" if path in _KAGGLE_BUILTINS else (
            "ok" if Path(path).is_file() else "MISSING"
        )
        print(f"   {name:14s} {exists:8s} {path}")
    return 0


# ---------------------------------------------------------------------------
# elim-sweep — winrate + elimination-rate vs an opponent
# ---------------------------------------------------------------------------


def cmd_elim_sweep(args: argparse.Namespace) -> int:
    """Run focal vs opp on N seeds × seat-swap, report winrate AND
    elimination-rate. Elimination = focal won AND env terminated with at most
    one player alive (opp ended with zero planets AND zero in-flight fleets).
    Score-wins (focal led at turn cap with opp still alive) count as wins but
    NOT as elimination wins."""
    focal_name, focal_path = resolve_agent_spec(args.agent)
    opp_name,   opp_path   = resolve_agent_spec(args.vs)
    seeds = list(range(args.seeds))
    pairs = _balanced_pairs(seeds, focal_path, opp_path)
    print(f"== elim-sweep {focal_name} vs {opp_name}  "
          f"seeds={args.seeds} (×2 seats) = {len(pairs)} trials ==")

    wins = elim_wins = score_wins = errors = draws = 0
    n = 0
    rows: list[tuple] = []
    all_focal_turn_ms: list[float] = []
    t0 = time.perf_counter()

    if args.workers <= 1:
        results = ((p, _play_one_task((p[0], p[1], p[2]))) for p in pairs)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(_play_one_task, (seed, p0, p1)):
                    (seed, p0, p1, focal_is_p0)
                for seed, p0, p1, focal_is_p0 in pairs
            }
            results = ((futs[fut], fut.result()) for fut in as_completed(futs))

    for (seed, _p0, _p1, focal_is_p0), r in results:
        n += 1
        focal_ms = r.p0_turn_ms if focal_is_p0 else r.p1_turn_ms
        p95 = p_quantile(focal_ms, 0.95) if focal_ms else 0.0
        all_focal_turn_ms.extend(focal_ms)
        if r.outcome == "error":
            errors += 1
            rows.append((seed, focal_is_p0, "error", "—",
                         r.final_planet_counts, r.final_fleet_counts, p95))
            continue
        focal_won = (focal_is_p0 and r.outcome == "p0_win") or \
                    (not focal_is_p0 and r.outcome == "p1_win")
        if focal_won:
            wins += 1
            if r.terminated_by == "elimination":
                elim_wins += 1
            else:
                score_wins += 1
        elif r.outcome == "draw":
            draws += 1
        rows.append((seed, focal_is_p0, r.outcome, r.terminated_by,
                     r.final_planet_counts, r.final_fleet_counts, p95))

    elapsed = time.perf_counter() - t0
    win_lo, win_hi = wilson_ci(wins, n)
    elim_lo, elim_hi = wilson_ci(elim_wins, n)
    pool_p50 = p_quantile(all_focal_turn_ms, 0.50) if all_focal_turn_ms else 0.0
    pool_p95 = p_quantile(all_focal_turn_ms, 0.95) if all_focal_turn_ms else 0.0
    pool_p99 = p_quantile(all_focal_turn_ms, 0.99) if all_focal_turn_ms else 0.0
    pool_max = max(all_focal_turn_ms) if all_focal_turn_ms else 0.0
    over_1000 = sum(1 for t in all_focal_turn_ms if t >= 1000.0)

    rows.sort(key=lambda r: (r[0], 0 if r[1] else 1))
    print(f"\n{'seed':>4} {'seat':>4} {'outcome':>8} {'end':>12} "
          f"{'planets(p0,p1)':>16} {'fleets(p0,p1)':>16} {'p95ms':>7}")
    for seed, focal_is_p0, outcome, end, planets, fleets, p95 in rows:
        seat = "p0" if focal_is_p0 else "p1"
        print(f"{seed:>4} {seat:>4} {outcome:>8} {end:>12} "
              f"{str(planets):>16} {str(fleets):>16} {p95:>7.0f}")

    print(f"\n   wins:        {wins}/{n}   winrate     "
          f"{wins/n:.1%}  Wilson [{win_lo:.3f}, {win_hi:.3f}]")
    print(f"   elim wins:   {elim_wins}/{n}   elim-rate   "
          f"{elim_wins/n:.1%}  Wilson [{elim_lo:.3f}, {elim_hi:.3f}]")
    print(f"   score-only:  {score_wins}/{n}   ({score_wins/n:.1%})  "
          f"(focal won but opp still alive at turn cap)")
    print(f"   draws:       {draws}/{n}   errors: {errors}")
    print(f"   wallclock {elapsed:.1f}s  "
          f"per-turn focal p50={pool_p50:.0f} p95={pool_p95:.0f} "
          f"p99={pool_p99:.0f} max={pool_max:.0f}ms  over1000ms={over_1000}")
    elim_pass = elim_lo >= 0.86
    wallclock_pass = pool_p95 < 1000.0 and over_1000 == 0
    print(f"   verdict: elim={'PASS' if elim_pass else 'FAIL'}  "
          f"wallclock={'PASS' if wallclock_pass else 'FAIL'}  "
          f"(gates: elim Wilson-lo >= 0.86; focal p95 < 1000ms AND no turn >=1000ms)")
    return 0 if (elim_pass and wallclock_pass) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fast.py",
        description="One-file iteration entry point for Orbit Wars.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("smoke", help="cheap triage vs random + nearest")
    sp.add_argument("agent")
    sp.add_argument("--seeds", type=int, default=16)
    sp.add_argument("--workers", type=int, default=8)
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser("eval", help="adaptive Wilson-gated A/B")
    sp.add_argument("agent")
    sp.add_argument("--vs", default=DEFAULT_BASELINE,
                    help=f"single opponent (default: {DEFAULT_BASELINE}); "
                         f"ignored if --vs-panel is set")
    sp.add_argument("--vs-panel", default=None,
                    help=f"multi-opponent calibration panel — required before any "
                         f"submission. 'default' = {','.join(DEFAULT_PANEL)}, "
                         f"or pass a comma-separated list of agent names")
    sp.add_argument("--require-h2h", default=None,
                    help="MANDATORY when --vs-panel is set: source path or "
                         "registry name of the current rolling champion. Champion "
                         "h2h is prepended to the panel; PASS requires ALL panel "
                         "opponents (including champion) to clear Wlo>=gate. "
                         "Closes panel-passes-h2h-vs-current-fails (4 recurrences).")
    sp.add_argument("--max-seeds", type=int, default=64)
    sp.add_argument("--gate", type=float, default=0.55,
                    help="Wilson 95%% lower-bound gate (default: 0.55)")
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--geometry-panel", action="store_true",
                    help="Draw seeds from lib.seed_panel.SEED_PANEL_128_INTERLEAVED "
                         "(32 archetypes round-robin) instead of range(0, max-seeds). "
                         "Auto-bumps --max-seeds to 128 if it's still the default. "
                         "Combine with --by-archetype for per-cell winrate breakdown. "
                         "Built by scripts/build_seed_panel.py; see "
                         "audit/2026-05-18-seed-panel.md.")
    sp.add_argument("--by-archetype", action="store_true",
                    help="After eval, print per-archetype focal winrate. Useful "
                         "with --geometry-panel; also works with range() seeds "
                         "(only intersecting seeds are reported).")
    sp.add_argument("--full-panel", action="store_true",
                    help="Disable adaptive early-stop (both PASS-stop and "
                         "FAIL-stop) and run every seed in --max-seeds / the "
                         "geometry panel. Use when you care about full "
                         "per-archetype coverage, not the gate verdict.")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("play", help="single game, verbose")
    sp.add_argument("agent")
    sp.add_argument("--vs", default=DEFAULT_BASELINE)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--swap", action="store_true",
                    help="play focal as P1 instead of P0")
    sp.set_defaults(func=cmd_play)

    sp = sub.add_parser("bench", help="per-turn ms vs 1000ms budget")
    sp.add_argument("agent")
    sp.add_argument("--vs", default=DEFAULT_BASELINE)
    sp.add_argument("--games", type=int, default=3,
                    help="number of games to sample timing over (default: 3)")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("baselines", help="list baseline registry")
    sp.set_defaults(func=cmd_baselines)

    sp = sub.add_parser("elim-sweep",
                        help="winrate + elimination-rate vs an opponent")
    sp.add_argument("agent")
    sp.add_argument("--vs", default="random",
                    help="opponent (default: random)")
    sp.add_argument("--seeds", type=int, default=16,
                    help="number of seeds; each played twice (seat-swap)")
    sp.add_argument("--workers", type=int, default=4)
    sp.set_defaults(func=cmd_elim_sweep)

    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
