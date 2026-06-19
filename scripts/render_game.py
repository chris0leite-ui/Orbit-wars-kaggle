"""Render one orbit_wars game to a self-contained HTML replay you can open in a
browser. Sets agent env-vars BEFORE loading the agents so gated knobs apply.

Usage:
    python scripts/render_game.py --seed 6000 --p0 lr --p1 v2 --out /tmp/game.html
    LR_WIN_LEAF=1 python scripts/render_game.py ...
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agents" / "producer"))

from fast import resolve_agent_spec, _load_callable  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--p0", default="lr")
    ap.add_argument("--p1", default="v2")
    ap.add_argument("--out", default="/tmp/game.html")
    args = ap.parse_args()

    from kaggle_environments import make

    _, p0_path = resolve_agent_spec(args.p0)
    _, p1_path = resolve_agent_spec(args.p1)
    p0 = _load_callable(p0_path)
    p1 = _load_callable(p1_path)

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=True)
    env.run([p0, p1])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    winner = "P0" if (r0 or 0) > (r1 or 0) else ("P1" if (r1 or 0) > (r0 or 0) else "draw")
    print(f"seed={args.seed}  {args.p0}(P0) vs {args.p1}(P1)  "
          f"steps={len(env.steps)}  rewards=({r0},{r1})  winner={winner}")

    html = env.render(mode="html", width=900, height=700)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out}  ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
