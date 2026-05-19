"""Phase A — Analytics Verification Suite (Tests 1-4 orchestrator).

Runs the four wallclock/replay-heavy checks and writes the audit doc
`audit/2026-05-19-analytics-verification.md`. Test 5 (capture-math units)
runs separately as pytest cases in `tests/test_analytics.py`.

  1. project() vs. direct fast_sim loop. Bit-exact gate.
  2. project() determinism. Same inputs → same float (twice).
  3. v3.1 self-play balance at n=8 (~50/50 ± 2).
  4. v3.1 vs random baseline at n=8 (≥7/8 = 87.5%).

Usage:
  python scripts/verify_analytics.py
  python scripts/verify_analytics.py --skip-3 --skip-4   # for fast iteration

Exits 0 even on FAILures so the audit doc always gets written. Inspect
the doc + the exit summary for PASS/FAIL.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure repo root on sys.path so `lib.*` / `agents.*` import when run
# as a script.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib import fast_sim                                      # noqa: E402
from lib.opp_model import lite_greedy_policy                  # noqa: E402

from agents.trajectory_roi.main import (                      # noqa: E402
    project,
    _obs_from_snap,
    _terminal_value,
    K_HORIZON,
)
from agents.trajectory_roi import main as trajectory_roi_main # noqa: E402


REPLAY_DIR = _REPO / "audit/live-episodes/52784853"
AUDIT_DOC = _REPO / "audit/2026-05-19-analytics-verification.md"
SAMPLE_TURNS = (25, 60, 100, 140, 180)
TEST_K = 30   # K-step horizon for projection-vs-reality
N_AB_GAMES = 8


# ---- data class for results ---------------------------------------------


@dataclass
class CaseResult:
    name: str
    passed: bool
    note: str = ""


@dataclass
class TestResult:
    title: str
    passed: bool
    cases: list[CaseResult] = field(default_factory=list)
    summary: str = ""


# ---- replay loading ------------------------------------------------------


def _load_2p_replays(n: int = 5) -> list[tuple[str, dict]]:
    """Find up to `n` 2P replay files in REPLAY_DIR."""
    out: list[tuple[str, dict]] = []
    for path in sorted(REPLAY_DIR.glob("episode-*-replay.json")):
        with open(path) as f:
            try:
                r = json.load(f)
            except Exception:
                continue
        if "steps" not in r:
            continue
        if not r["steps"] or len(r["steps"][0]) != 2:
            continue
        out.append((path.name, r))
        if len(out) >= n:
            break
    return out


def _obs_from_replay(replay: dict, turn: int, seat: int) -> dict | None:
    """Extract a seat-perspective obs dict from a replay at the given turn."""
    if turn >= len(replay["steps"]):
        return None
    seat_state = replay["steps"][turn][seat]
    obs = seat_state.get("observation", {})
    if not obs.get("planets"):
        return None
    # The replay's "observation" already contains all the fields we need.
    return {
        "player": int(obs.get("player", seat)),
        "step": int(obs.get("step", turn)),
        "planets": [list(p) for p in obs.get("planets", [])],
        "fleets": [list(f) for f in obs.get("fleets", []) or []],
        "comets": list(obs.get("comets", []) or []),
        "comet_planet_ids": list(obs.get("comet_planet_ids", []) or []),
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
        "initial_planets": [list(p) for p in
                            obs.get("initial_planets",
                                    obs.get("planets", []))],
        "next_fleet_id": int(obs.get("next_fleet_id", 0)),
    }


# ---- Test 1: projection vs reality ---------------------------------------


def _direct_project(obs: dict, my_id: int, opp_id: int,
                    my_turn0_emits: list, K: int) -> float:
    """Independent re-implementation of `project()`'s loop. Same primitives
    (fast_sim.from_obs/step + lite_greedy_policy + _terminal_value) wired
    explicitly so a divergence pinpoints a bug in project's wiring."""
    snap = fast_sim.from_obs(obs, configuration=None)
    actions: list[Any] = [None, None]
    actions[my_id] = my_turn0_emits
    actions[opp_id] = lite_greedy_policy(_obs_from_snap(snap, opp_id))
    snap = fast_sim.step(snap, actions)
    for _ in range(K - 1):
        if snap.fake_env.done:
            break
        actions[my_id] = lite_greedy_policy(_obs_from_snap(snap, my_id))
        actions[opp_id] = lite_greedy_policy(_obs_from_snap(snap, opp_id))
        snap = fast_sim.step(snap, actions)
    return _terminal_value(snap, my_id)


def test_1_projection_vs_reality() -> TestResult:
    title = "Test 1 — Projection vs Reality"
    res = TestResult(title=title, passed=True)
    replays = _load_2p_replays(n=5)
    if not replays:
        res.passed = False
        res.summary = f"FAIL: no 2P replays found under {REPLAY_DIR}"
        return res

    for fname, replay in replays:
        # Pick the first sample turn that exists in this replay.
        turn = next((t for t in SAMPLE_TURNS if t < len(replay["steps"])), None)
        if turn is None:
            continue
        for seat in (0, 1):
            obs = _obs_from_replay(replay, turn, seat)
            if obs is None:
                continue
            my_id = seat
            opp_id = 1 - seat
            # Two variants: passive (my_emits=[]) and agent-emit.
            for variant, emits in (
                ("passive", []),
                ("agent",
                 trajectory_roi_main.agent(obs, configuration=None)),
            ):
                predicted = project(obs, my_id, opp_id, emits, K=TEST_K)
                actual = _direct_project(obs, my_id, opp_id, emits, K=TEST_K)
                delta = predicted - actual
                exact = abs(delta) < 1e-9
                rel = abs(delta) / max(abs(actual), 1.0)
                ok = exact
                note = (f"{fname[:35]}@t{turn} seat{seat} {variant}: "
                        f"pred={predicted:.4f} actual={actual:.4f} "
                        f"delta={delta:+.4e} (rel={rel:.2%})")
                res.cases.append(CaseResult(name=note, passed=ok, note=""))
                if not ok:
                    res.passed = False
    res.summary = (f"{sum(c.passed for c in res.cases)}/{len(res.cases)} "
                   f"cases bit-exact")
    return res


# ---- Test 2: projection determinism --------------------------------------


def test_2_projection_determinism() -> TestResult:
    title = "Test 2 — Projection Determinism"
    res = TestResult(title=title, passed=True)
    replays = _load_2p_replays(n=5)
    if not replays:
        res.passed = False
        res.summary = f"FAIL: no 2P replays found under {REPLAY_DIR}"
        return res

    for fname, replay in replays:
        turn = next((t for t in SAMPLE_TURNS if t < len(replay["steps"])), None)
        if turn is None:
            continue
        for seat in (0, 1):
            obs = _obs_from_replay(replay, turn, seat)
            if obs is None:
                continue
            my_id = seat
            opp_id = 1 - seat
            v1 = project(obs, my_id, opp_id, [], K=TEST_K)
            v2 = project(obs, my_id, opp_id, [], K=TEST_K)
            ok = (v1 == v2)
            note = (f"{fname[:35]}@t{turn} seat{seat}: "
                    f"v1={v1:.6f} v2={v2:.6f} identical={ok}")
            res.cases.append(CaseResult(name=note, passed=ok))
            if not ok:
                res.passed = False
    res.summary = (f"{sum(c.passed for c in res.cases)}/{len(res.cases)} "
                   f"cases identical")
    return res


# ---- Tests 3, 4: A/B sweeps via kaggle_environments ----------------------


def _evaluate_2p(agent_left, agent_right, n_episodes: int, seeds=None):
    """Run n head-to-heads. Returns (left_wins, right_wins, draws,
    elapsed_s, per_game)."""
    from kaggle_environments import make
    per_game = []
    left_wins = right_wins = draws = 0
    t0 = time.perf_counter()
    for i in range(n_episodes):
        seed = seeds[i] if seeds else (i + 1)
        env = make("orbit_wars", debug=False,
                   configuration={"seed": seed})
        env.reset(num_agents=2)
        steps = env.run([agent_left, agent_right])
        # final rewards
        last = steps[-1]
        r_left = last[0]["reward"]
        r_right = last[1]["reward"]
        if r_left > r_right:
            left_wins += 1
            outcome = "L"
        elif r_right > r_left:
            right_wins += 1
            outcome = "R"
        else:
            draws += 1
            outcome = "D"
        per_game.append({
            "seed": seed,
            "turns": len(steps),
            "outcome": outcome,
            "reward_left": r_left,
            "reward_right": r_right,
        })
    return left_wins, right_wins, draws, time.perf_counter() - t0, per_game


def test_3_self_play_balance(n: int) -> TestResult:
    title = f"Test 3 — Self-play Balance (n={n})"
    res = TestResult(title=title, passed=True)
    troi = trajectory_roi_main.agent
    seat0_wins, seat1_wins, draws, elapsed, games = _evaluate_2p(
        troi, troi, n)
    rate = seat0_wins / n if n else 0.0
    # Spec gate: 0.4 ≤ rate ≤ 0.6 → PASS (balanced).
    # WARN zone: 0.2 ≤ rate < 0.4 or 0.6 < rate ≤ 0.8 → small-n noise OR
    #   mild asymmetry; needs n≥16 to disambiguate.
    # FAIL: rate < 0.2 or rate > 0.8.
    if 0.4 <= rate <= 0.6:
        gate = "PASS"
    elif 0.2 <= rate <= 0.8:
        gate = "WARN"
    else:
        gate = "FAIL"
    res.summary = (
        f"seat0_wins={seat0_wins} seat1_wins={seat1_wins} draws={draws} "
        f"rate={rate:.0%} elapsed={elapsed:.1f}s gate={gate}")
    res.passed = (gate == "PASS")
    for g in games:
        res.cases.append(CaseResult(
            name=f"seed={g['seed']} turns={g['turns']} outcome={g['outcome']}",
            passed=True,
        ))
    return res


def test_4_vs_random(n: int) -> TestResult:
    title = f"Test 4 — vs Random (n={n} per seat)"
    res = TestResult(title=title, passed=True)
    troi = trajectory_roi_main.agent
    # Seat 0 = trajectory_roi, seat 1 = random
    a_wins, b_wins, draws_a, elapsed_a, games_a = _evaluate_2p(
        troi, "random", n)
    # Seat 1 = trajectory_roi, seat 0 = random
    b2_wins, a2_wins, draws_b, elapsed_b, games_b = _evaluate_2p(
        "random", troi, n)
    ours_as_left = a_wins
    ours_as_right = a2_wins
    total_ours = ours_as_left + ours_as_right
    total_games = 2 * n
    res.summary = (
        f"as_seat0={ours_as_left}/{n} as_seat1={ours_as_right}/{n} "
        f"total={total_ours}/{total_games} "
        f"elapsed={elapsed_a + elapsed_b:.1f}s")
    # PASS gate: ≥ 87.5% wins (≥7/8 per side at n=8). Below 80% blocks.
    pass_per_side = math.ceil(n * 0.875)
    ok = (ours_as_left >= pass_per_side and ours_as_right >= pass_per_side)
    res.passed = ok
    for g in games_a:
        res.cases.append(CaseResult(
            name=f"seat0 seed={g['seed']} turns={g['turns']} "
                 f"outcome={g['outcome']} (L=ours)",
            passed=True,
        ))
    for g in games_b:
        res.cases.append(CaseResult(
            name=f"seat1 seed={g['seed']} turns={g['turns']} "
                 f"outcome={g['outcome']} (R=ours)",
            passed=True,
        ))
    if not ok:
        res.summary += f"  (FAIL: need ≥{pass_per_side}/{n} per side)"
    return res


# ---- Audit doc emission --------------------------------------------------


def _render_audit(results: list[TestResult], test5_summary: str) -> str:
    lines: list[str] = []
    lines.append("# 2026-05-19 — Analytics Verification (trajectory_roi v3.1)")
    lines.append("")
    lines.append("> Phase A blocker for the goal-directed `trajectory_portfolio`")
    lines.append("> planner. See `/root/.claude/plans/optimized-questing-shell.md`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Test | Result | Notes |")
    lines.append("|---|---|---|")
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        lines.append(f"| {r.title} | {flag} | {r.summary} |")
    lines.append(f"| Test 5 — Capture Math Units (pytest) | {test5_summary} |"
                 f"  | 3 deterministic units; run with "
                 f"`pytest tests/test_analytics.py` |")
    lines.append("")
    for r in results:
        lines.append(f"## {r.title}")
        lines.append("")
        lines.append(f"**Result:** {'PASS' if r.passed else 'FAIL'} — {r.summary}")
        lines.append("")
        if r.cases:
            lines.append("Per-case detail:")
            lines.append("")
            for c in r.cases:
                marker = "✓" if c.passed else "✗"
                lines.append(f"- {marker} {c.name}")
            lines.append("")
    lines.append("## Phase B Gating")
    lines.append("")
    all_pass = all(r.passed for r in results)
    if all_pass:
        lines.append("- [x] Tests 1-4 all PASS → unblocked pending Test 5 pytest.")
    else:
        lines.append("- [ ] One or more checks FAILED — diagnose and fix before "
                     "v4 build.")
        for r in results:
            if not r.passed:
                lines.append(f"  - {r.title}: {r.summary}")
    lines.append("")
    lines.append("## Bugs Found")
    lines.append("")
    bugs = [r for r in results if not r.passed]
    if not bugs:
        lines.append("None — all checks PASS.")
    else:
        for r in bugs:
            lines.append(f"- **{r.title}** — {r.summary}")
    lines.append("")
    return "\n".join(lines)


# ---- driver --------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-3", action="store_true",
                        help="Skip the self-play balance check.")
    parser.add_argument("--skip-4", action="store_true",
                        help="Skip the vs-random check.")
    parser.add_argument("--skip-1", action="store_true",
                        help="Skip projection-vs-reality check.")
    parser.add_argument("--skip-2", action="store_true",
                        help="Skip projection-determinism check.")
    parser.add_argument("--ab-n", type=int, default=N_AB_GAMES,
                        help="Episode count per side for Tests 3, 4.")
    parser.add_argument("--no-write", action="store_true",
                        help="Run tests but don't overwrite the audit doc.")
    args = parser.parse_args()

    results: list[TestResult] = []
    t_start = time.perf_counter()

    if args.skip_1:
        print("[1/4] Projection vs reality — SKIPPED", flush=True)
    else:
        print("[1/4] Projection vs reality ...", flush=True)
        r1 = test_1_projection_vs_reality()
        print(f"      {r1.summary}", flush=True)
        results.append(r1)

    if args.skip_2:
        print("[2/4] Projection determinism — SKIPPED", flush=True)
    else:
        print("[2/4] Projection determinism ...", flush=True)
        r2 = test_2_projection_determinism()
        print(f"      {r2.summary}", flush=True)
        results.append(r2)

    if args.skip_3:
        print("[3/4] Self-play balance — SKIPPED", flush=True)
    else:
        print(f"[3/4] Self-play balance (n={args.ab_n}) ...", flush=True)
        r3 = test_3_self_play_balance(args.ab_n)
        print(f"      {r3.summary}", flush=True)
        results.append(r3)

    if args.skip_4:
        print("[4/4] vs Random — SKIPPED", flush=True)
    else:
        print(f"[4/4] vs Random (n={args.ab_n} per seat) ...", flush=True)
        r4 = test_4_vs_random(args.ab_n)
        print(f"      {r4.summary}", flush=True)
        results.append(r4)

    # Probe whether Test 5 passes (pytest invocation done separately).
    test5_summary = "see pytest output"

    if args.no_write:
        elapsed = time.perf_counter() - t_start
        print(f"\nSkipping audit-doc write (--no-write)  ({elapsed:.1f}s total)")
    else:
        doc = _render_audit(results, test5_summary)
        AUDIT_DOC.parent.mkdir(parents=True, exist_ok=True)
        AUDIT_DOC.write_text(doc)
        elapsed = time.perf_counter() - t_start
        print(f"\nWrote {AUDIT_DOC}  ({elapsed:.1f}s total)")
    n_pass = sum(r.passed for r in results)
    print(f"PASS: {n_pass}/{len(results)} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
