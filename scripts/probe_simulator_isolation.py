"""Does the simulator stay bit-exact mirror-symmetric under no-op agents?

Setup: both agents always return []. Then the only state evolution comes
from the simulator (orbital rotation + production + comet spawns). If
the simulator is genuinely seat-symmetric, every planet at position
(x, y) should always have a mirror partner at (100-x, 100-y), at the
BIT level. This isolates the simulator from any agent-side tie-break.
"""
from kaggle_environments import make

BOARD = 100.0
SEED = 1


def noop(obs):
    return []


env = make("orbit_wars", configuration={"seed": SEED}, debug=False)
env.run([noop, noop])

first_break = None
break_detail = None
for t, step_state in enumerate(env.steps):
    planets = step_state[0].observation.planets
    # Bit-exact mirror check: for each planet at (x, y), require a sibling
    # at exactly (BOARD - x, BOARD - y) using == on the floats.
    by_pos = {(p[2], p[3]): p for p in planets}
    for p in planets:
        mx, my = BOARD - p[2], BOARD - p[3]
        sibling = by_pos.get((mx, my))
        if sibling is None:
            first_break = t
            break_detail = (
                f"pid={p[0]} at ({p[2]!r}, {p[3]!r}) has no bit-exact "
                f"mirror sibling at ({mx!r}, {my!r})"
            )
            # Search for closest planet to mirror position to characterise drift
            nearest = min(
                planets,
                key=lambda q: (q[2] - mx) ** 2 + (q[3] - my) ** 2
            )
            break_detail += (
                f"\n  closest planet to expected mirror: pid={nearest[0]} "
                f"at ({nearest[2]!r}, {nearest[3]!r}), "
                f"x-diff={nearest[2] - mx!r}, y-diff={nearest[3] - my!r}"
            )
            break
    if first_break is not None:
        break
else:
    print(f"All {len(env.steps)} turns: planet positions stay bit-exact mirror.")

if first_break is not None:
    print(f"Simulator-side mirror violation first appears at turn {first_break}:")
    print("  " + break_detail)
