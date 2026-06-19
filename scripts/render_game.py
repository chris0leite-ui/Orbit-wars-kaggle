"""Render one orbit_wars game (2P or 4P) to a self-contained HTML replay you can
open in a browser. Sets agent env-vars BEFORE loading the agents so gated knobs
apply (e.g. LR_WIN_LEAF=1).

Usage:
    # 2P
    python scripts/render_game.py --seed 6001 \
        --agents agents/least_resistance/main.py,agents/v2/main.py --out /tmp/g.html
    # 4P (focal vs three V2)
    LR_WIN_LEAF=1 python scripts/render_game.py --seed 7 \
        --agents agents/least_resistance/main.py,agents/v2/main.py,agents/v2/main.py,agents/v2/main.py \
        --out /tmp/g4.html
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agents" / "producer"))

from fast import resolve_agent_spec, _load_callable  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--agents", required=True,
                    help="comma-separated agent specs/paths; 2 -> 2P, 4 -> 4P")
    ap.add_argument("--out", default="/tmp/game.html")
    args = ap.parse_args()

    from kaggle_environments import make

    specs = [s.strip() for s in args.agents.split(",") if s.strip()]
    names, callables = [], []
    for s in specs:
        try:
            name, path = resolve_agent_spec(s)
        except FileNotFoundError:
            name, path = Path(s).stem, s
        names.append(name)
        callables.append(_load_callable(path))

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=True)
    env.run(callables)
    final = env.steps[-1]
    rewards = [final[i].reward for i in range(len(callables))]
    best = max((r if r is not None else -1e9) for r in rewards)
    winners = [f"P{i}({names[i]})" for i, r in enumerate(rewards) if r == best]
    print(f"seed={args.seed}  {len(callables)}P  agents={names}  "
          f"steps={len(env.steps)}  rewards={rewards}  winner={winners}")

    html = env.render(mode="html", width=900, height=700)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out}  ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
