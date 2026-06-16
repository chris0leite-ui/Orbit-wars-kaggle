"""iso_game.py — env-ISOLATED producer_plus games (fixes the env-leak bug).

The producer_plus bundles configure themselves via os.environ.setdefault at
import. os.environ is process-global, so loading two differently-configured
bundles in one process leaks the first's flags into the second — a "static"
opponent silently inherits the focal's PRODUCER_PLUS_OPP_PROJECTION etc.

Fix: load ONE engine file as N separate module instances (separate _RUNTIME
each), and wrap every seat so that, immediately before its turn, it (1) clears
all PRODUCER_PLUS_* vars and (2) applies its OWN config. The per-turn gate
reads are then correct because the env.run loop calls seats sequentially.

    python scripts/iso_game.py --mode 2p --seed 7 --steps 500 \
        --focal anti_strongest --opp static
    python scripts/iso_game.py --mode 4p --seed 0 --steps 400 \
        --focal anti_strongest --opp static     # focal vs 3 opp copies
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The full producer_plus engine, all mechanisms present, gated by env vars.
ENGINE = str(REPO / "submissions" / "_iso_engine.py")

CONFIGS: dict[str, dict[str, str]] = {
    "static": {},
    "inverse": {"PRODUCER_PLUS_OPP_PROJECTION": "1"},
    "anti_strongest": {
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "strongest",
    },
    # Our best submission: the 1280 champion `vetorf4p_seq_strength`.
    "champion": {
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REPLY_SEQ": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "strength",
    },
    # The champion with the new relative-strongest objective swapped in
    # (FFA_WEIGHTS strength -> strongest). Tests the PI's idea ON the best agent.
    "champion_strongest": {
        "PRODUCER_PLUS_MULTI_SIZE": "1",
        "PRODUCER_PLUS_OPP_PROJECTION": "1",
        "PRODUCER_PLUS_RESPONSE_VETO": "1",
        "PRODUCER_PLUS_REACTIVE_FLOOR": "0.5",
        "PRODUCER_PLUS_REPLY_SEQ": "1",
        "PRODUCER_PLUS_FFA_SCORE": "1",
        "PRODUCER_PLUS_FFA_WEIGHTS": "strongest",
    },
}


def _ensure_engine():
    """Build the bare full-engine bundle if absent (regenerable, not committed).
    A bundle is self-contained, so loading it under N module names yields N
    independent _RUNTIME instances — exactly what per-seat isolation needs."""
    if not Path(ENGINE).exists():
        import subprocess
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "bundle_producer_plus.py"),
             "--variant", "bare", "--out", ENGINE], check=True)


def _load_engine(tag: str):
    _ensure_engine()
    name = f"_iso_{tag}"
    spec = importlib.util.spec_from_file_location(name, ENGINE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m.agent


def _iso(agent, cfg: dict[str, str]):
    items = dict(cfg)

    def wrapped(obs, conf=None):
        for k in [k for k in os.environ if k.startswith("PRODUCER_PLUS_")]:
            del os.environ[k]
        os.environ.update(items)
        return agent(obs)
    return wrapped


def _score(obs, pid: int) -> float:
    pl = obs["planets"] if isinstance(obs, dict) else obs.planets
    fl = obs["fleets"] if isinstance(obs, dict) else obs.fleets
    return float(sum(p[5] for p in pl if p[1] == pid)
                 + sum(f[6] for f in fl if f[1] == pid))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["2p", "4p"], default="2p")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--focal", default="anti_strongest", choices=list(CONFIGS))
    ap.add_argument("--opp", default="static", choices=list(CONFIGS))
    ap.add_argument("--swap", action="store_true",
                    help="place focal at the LAST seat (2P: P1) to control seat bias")
    a = ap.parse_args(argv)

    from kaggle_environments import make
    nplayers = 4 if a.mode == "4p" else 2
    focal_pid = (nplayers - 1) if a.swap else 0
    seats = [None] * nplayers
    seats[focal_pid] = _iso(_load_engine("focal"), CONFIGS[a.focal])
    oi = 0
    for p in range(nplayers):
        if p == focal_pid:
            continue
        seats[p] = _iso(_load_engine(f"opp{oi}"), CONFIGS[a.opp])
        oi += 1

    env = make("orbit_wars", configuration={"seed": a.seed, "episodeSteps": a.steps}, debug=False)
    env.run(seats)
    n = len(env.steps)

    def obs_at(t):
        st = env.steps[t][0]
        return st["observation"] if isinstance(st, dict) else st.observation

    opp_pids = [p for p in range(nplayers) if p != focal_pid]
    of = obs_at(n - 1)
    scores = [_score(of, p) for p in range(nplayers)]
    focal_s = scores[focal_pid]
    best_opp = max(scores[p] for p in opp_pids)
    rank = 1 + sum(1 for p in opp_pids if scores[p] > focal_s)  # focal's rank (1 = best)
    # peak focal lead over the strongest opponent across the game (2p insight)
    peak = -1e9
    for t in range(n):
        o = obs_at(t)
        m = _score(o, focal_pid) - max(_score(o, p) for p in opp_pids)
        if m > peak:
            peak = m
    margin = focal_s - best_opp
    won = "Y" if margin > 0 else "."
    early = "*" if n < a.steps else " "
    print(f"{a.mode} seed {a.seed:>2d} P{focal_pid}: focal({a.focal})={focal_s:>6.0f} "
          f"best_opp({a.opp})={best_opp:>6.0f}  margin={margin:>+7.0f}  "
          f"won={won}  rank={rank}/{nplayers}  peak_lead=+{peak:>5.0f}  "
          f"steps={n}{early}  all={[round(s) for s in scores]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
