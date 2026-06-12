"""Guards for the learned shot-success filter (agents/producer_plus/shot_mlp.py).

1. Train/serve encoder parity: the labeler's inline encoder and the agent's
   encoder must agree bit-for-bit (the labeler stays import-free per its
   docstring, so the two copies are pinned here instead of shared).
2. Fleet-speed parity with the engine curve in lib/fleet.py.
3. Bundles built with the PRODUCER_PLUS_SHOT_MLP gate baked ON must carry
   trained weights (the in-agent fallback is a warn+no-op; a weightless
   gated bundle is a build error).
"""

import importlib.util
import random
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


labeler = _load("labeler", REPO / "scripts" / "label_shot_outcomes.py")
shot_mlp = _load("shot_mlp", REPO / "agents" / "producer_plus" / "shot_mlp.py")


def _random_planet(rng, pid, owner):
    return (
        pid, owner,
        rng.uniform(0, 100), rng.uniform(0, 100),   # x, y
        rng.uniform(0.5, 3.0),                       # radius
        rng.uniform(0, 1500),                        # ships
        rng.choice([0, 1, 2, 3, 4, 5]),              # production
    )


def _random_fleet(rng, fid, owner):
    return (fid, owner, rng.uniform(0, 100), rng.uniform(0, 100),
            rng.uniform(-3.14, 3.14), rng.randint(0, 30),
            rng.uniform(1, 400))


def test_encoder_parity_labeler_vs_agent():
    rng = random.Random(7)
    for trial in range(200):
        n_pl = rng.randint(4, 30)
        owners = [-1, 0, 1, 2, 3]
        planets = [_random_planet(rng, i, rng.choice(owners))
                   for i in range(n_pl)]
        fleets = [_random_fleet(rng, i, rng.choice(owners))
                  for i in range(rng.randint(0, 12))]
        focal = rng.choice([0, 1])
        src = rng.choice(planets)
        tgt = rng.choice(planets)
        ships = rng.uniform(1, 600)
        import math
        d = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
        v = shot_mlp.fleet_speed(ships)
        eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
        step = rng.randint(0, 499)

        a = labeler._encode_features(
            src, tgt, ships, d, eta, v, planets, fleets, focal, step)
        b = shot_mlp.encode_shot_features(
            src, tgt, ships, d, eta, v, planets, fleets, focal, step)
        assert a == b, f"trial {trial}: encoder drift\n{a}\n{b}"


def test_fleet_speed_parity_with_engine():
    sys.path.insert(0, str(REPO))
    from lib.fleet import speed as engine_speed
    for ships in (1, 2, 5, 17, 100, 350, 999, 1000, 1800):
        assert abs(shot_mlp.fleet_speed(ships) - engine_speed(ships)) < 1e-9


def test_gated_bundle_carries_weights(tmp_path):
    out = tmp_path / "gated.py"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "bundle_producer_plus.py"),
         "--variant", "vetorf4p_sync_shotmlp", "--out", str(out)],
        check=True, capture_output=True,
    )
    text = out.read_text()
    # The module is inlined as a repr-escaped exec string (namespaced), so
    # match substrings rather than lines. A real blob is thousands of
    # base64 chars on the WEIGHTS_B64 line.
    assert "WEIGHTS_B64 = None" not in text, (
        "bundle bakes the shot-MLP gate ON but carries no trained "
        "weights — run scripts/train_shot_mlp.py before bundling"
    )
    assert re.search(r"WEIGHTS_B64 = .{1000,}", text), (
        "WEIGHTS_B64 blob missing or implausibly small in bundle"
    )
    # Gate must be hardcoded (env-leak-proof), not in the env header.
    assert "hardcoded at bundle time" in text
    assert "'PRODUCER_PLUS_SHOT_MLP'" not in text.split("# === ")[0], (
        "threshold must not ride the env header (process-global leak)"
    )
