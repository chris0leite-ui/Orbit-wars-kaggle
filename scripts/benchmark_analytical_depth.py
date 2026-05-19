"""Benchmark analytical-depth costs for trajectory_roi v3.

Per `/root/.claude/plans/read-the-handover-do-abundant-quokka.md`,
v3 asks: before we build depth-3 forward-projection, measure how
much it actually costs. Answer five questions:

Q1. ms-per-step for fast_sim.step with empty actions / cheap opp /
    mirror-opp opp.
Q2. ms to forward-project ONE plan for K=50.
Q3. ms to forward-project ONE plan for K=100.
Q4. Plans/turn budget at K=50 and K=100 within 100/500/1000 ms.
Q5. Does v2's ~60-candidate enumeration fit each budget?

The benchmark captures a representative mid-game obs by running
self-play under baseline for N turns, then times fast_sim.step on
that state with three opp policies:

  - empty:       opp emits nothing each step (pure step cost).
  - lite_greedy: opp runs lib.opp_model.lite_greedy_policy.
  - mirror-v2:   opp runs the v2 agent (mirror-opp).

Reads ONLY. Writes ONE audit file under audit/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402

from lib import fast_sim  # noqa: E402
from lib.opp_model import lite_greedy_policy  # noqa: E402


def _capture_obs_after_turns(turns: int, agent_path: str) -> dict:
    """Run a self-play game and return the obs at `turns` turn."""
    env = make("orbit_wars", debug=False)
    agent_path_obj = REPO / agent_path
    if not agent_path_obj.exists():
        raise FileNotFoundError(agent_path_obj)
    # Run the env up to `turns` turns.
    env.run([str(agent_path_obj), str(agent_path_obj)])
    # Grab the snapshot at step `turns`.
    if turns >= len(env.steps):
        turns = len(env.steps) - 1
    state_pair = env.steps[turns]
    obs = state_pair[0]["observation"]
    return dict(obs) if not isinstance(obs, dict) else obs


def _mk_snap(obs):
    return fast_sim.from_obs(obs, configuration=None)


def _obs_us(snap):
    """Materialise a dict-form obs for seat 0."""
    s_obs = snap.state[0].observation
    return {
        "player": int(getattr(s_obs, "player", 0)),
        "step": int(getattr(s_obs, "step", 0)),
        "planets": [list(p) for p in s_obs.planets],
        "fleets": [list(f) for f in (s_obs.fleets or [])],
        "comets": list(getattr(s_obs, "comets", [])),
        "comet_planet_ids": list(getattr(s_obs, "comet_planet_ids", [])),
        "angular_velocity": float(getattr(s_obs, "angular_velocity", 0.0)),
        "initial_planets": [list(p) for p in getattr(s_obs, "initial_planets", s_obs.planets)],
    }


def _obs_opp(snap):
    s_obs = snap.state[1].observation
    return {
        "player": 1,
        "step": int(getattr(s_obs, "step", 0)),
        "planets": [list(p) for p in s_obs.planets],
        "fleets": [list(f) for f in (s_obs.fleets or [])],
        "comets": list(getattr(s_obs, "comets", [])),
        "comet_planet_ids": list(getattr(s_obs, "comet_planet_ids", [])),
        "angular_velocity": float(getattr(s_obs, "angular_velocity", 0.0)),
        "initial_planets": [list(p) for p in getattr(s_obs, "initial_planets", s_obs.planets)],
    }


def _project(obs_initial, K: int, opp_policy_name: str):
    """Project K turns from obs_initial under (empty, lite_greedy,
    mirror-v2) opp policy. Returns elapsed wall-time in ms."""
    snap = _mk_snap(obs_initial)
    t0 = time.perf_counter()
    for _ in range(K):
        if snap.fake_env.done:
            break
        if opp_policy_name == "empty":
            opp_emits = []
        elif opp_policy_name == "lite_greedy":
            opp_emits = lite_greedy_policy(_obs_opp(snap))
        elif opp_policy_name == "mirror-v2":
            from agents.trajectory_roi.main import agent as v2_agent
            opp_emits = v2_agent(_obs_opp(snap))
        else:
            raise ValueError(opp_policy_name)
        # Our emits: empty (we're just measuring step + opp cost).
        snap = fast_sim.step(snap, [[], opp_emits])
    return (time.perf_counter() - t0) * 1000.0


def _bench_step_only(obs, n_calls: int = 1000) -> float:
    """Time pure fast_sim.step (no policy calls). Returns ms per step."""
    snap = _mk_snap(obs)
    # Pre-warm so JIT-likes settle.
    for _ in range(10):
        snap = fast_sim.step(snap, [[], []])
    t0 = time.perf_counter()
    for _ in range(n_calls):
        if snap.fake_env.done:
            snap = _mk_snap(obs)
        snap = fast_sim.step(snap, [[], []])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms / n_calls


def _bench_project(obs, K: int, opp_name: str, n_trials: int = 20) -> tuple[float, float]:
    """Returns (median_ms, mean_ms) per K-turn projection."""
    times = []
    for _ in range(n_trials):
        times.append(_project(obs, K, opp_name))
    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return median, mean


def run_benchmark(obs_label: str, obs: dict, out_lines: list[str]) -> None:
    out_lines.append(f"\n## {obs_label} obs")
    p_count = len(obs.get("planets", []))
    f_count = len(obs.get("fleets", []) or [])
    step = int(obs.get("step", 0))
    out_lines.append(f"\n- planets: {p_count}, fleets: {f_count}, step: {step}\n")

    # Q1 — single-step cost.
    ms_per_step = _bench_step_only(obs, n_calls=500)
    out_lines.append(f"### Q1. Pure fast_sim.step\n\n- **{ms_per_step:.3f} ms / step**\n")

    # Q2/Q3 — K-step projection under 3 opp policies.
    out_lines.append("### Q2/Q3. K-turn projection (median over 20 trials)\n")
    out_lines.append("| opp policy | K=10 | K=30 | K=50 | K=100 |")
    out_lines.append("|---|---:|---:|---:|---:|")
    medians = {}
    for opp_name in ("empty", "lite_greedy", "mirror-v2"):
        row = [opp_name]
        for K in (10, 30, 50, 100):
            med, _ = _bench_project(obs, K, opp_name, n_trials=15 if opp_name != "mirror-v2" else 8)
            medians[(opp_name, K)] = med
            row.append(f"{med:.1f}")
        out_lines.append("| " + " | ".join(row) + " |")
    out_lines.append("")

    # Q4 — plans/turn budgets.
    out_lines.append("### Q4. Plans per turn at given budget (lite_greedy projection)\n")
    out_lines.append("| K | budget=100ms | budget=500ms | budget=1000ms |")
    out_lines.append("|---|---:|---:|---:|")
    for K in (30, 50, 100):
        med = medians[("lite_greedy", K)]
        if med <= 0:
            continue
        out_lines.append(
            f"| {K} | {int(100 / med)} | {int(500 / med)} | {int(1000 / med)} |"
        )
    out_lines.append("")

    out_lines.append("### Q4-bis. Plans per turn — full mirror-v2 opp at each step\n")
    out_lines.append("| K | budget=100ms | budget=500ms | budget=1000ms |")
    out_lines.append("|---|---:|---:|---:|")
    for K in (30, 50, 100):
        med = medians[("mirror-v2", K)]
        if med <= 0:
            continue
        out_lines.append(
            f"| {K} | {int(100 / med)} | {int(500 / med)} | {int(1000 / med)} |"
        )
    out_lines.append("")

    # Q5 — does v2's ~60 candidates fit?
    out_lines.append("### Q5. Does v2's ~60 candidates fit at K=50?\n")
    for opp_name in ("lite_greedy", "mirror-v2"):
        med = medians[(opp_name, 50)]
        budget_60 = 60 * med
        verdict = "OK" if budget_60 < 1000 else "OVER BUDGET"
        out_lines.append(
            f"- {opp_name}: 60 plans × {med:.1f} ms = **{budget_60:.0f} ms** — {verdict}"
        )
    out_lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit/2026-05-19-analytical-depth-benchmark.md")
    ap.add_argument("--turns", default="20,80,180", help="comma-sep turn counts to capture")
    ap.add_argument("--agent", default="agents/baseline/main.py")
    args = ap.parse_args()

    out_lines: list[str] = [
        "# Analytical-depth benchmark — 2026-05-19",
        "",
        "Per the v3 plan in `/root/.claude/plans/read-the-handover-",
        "do-abundant-quokka.md`. Measures forward-projection cost to",
        "decide v3 depth-3 design parameters.",
        "",
        f"Self-play agent for obs capture: `{args.agent}`",
    ]

    turn_list = [int(t) for t in args.turns.split(",")]
    labels = ["early", "mid", "late"]
    for turns, label in zip(turn_list, labels):
        print(f"--- capturing {label} obs (turn={turns}) ---", flush=True)
        try:
            obs = _capture_obs_after_turns(turns, args.agent)
        except Exception as e:  # noqa: BLE001
            out_lines.append(f"\n## {label} obs (turn~{turns}) — CAPTURE FAILED: {e}")
            continue
        print(f"--- benching {label} obs (planets={len(obs.get('planets', []))}, fleets={len(obs.get('fleets', []) or [])}) ---", flush=True)
        run_benchmark(label, obs, out_lines)

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"--- wrote {out_path}")


if __name__ == "__main__":
    main()
