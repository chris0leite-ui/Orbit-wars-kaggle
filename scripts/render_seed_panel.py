"""Render a 32-archetype preview grid for the seed panel.

For each of the 32 archetype cells, picks ONE representative seed and draws
its initial planet layout in a mini-plot. Static planets in dark grey,
rotating in blue. Sun at the centre, ROTATION_RADIUS_LIMIT circle dashed.

Output: audit/seed-panel/preview.png
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from lib.geometry import BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS
from lib.geometry_features import extract_geometry
from lib.orbit import is_orbiting
from lib.seed_panel import SEED_PANEL_BY_ARCHETYPE


def render_one(ax, seed: int, title: str) -> None:
    env_init = extract_geometry(seed)  # forces a fresh init via the env
    # Re-init to grab the full planet list (extract_geometry doesn't keep it).
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset()
    planets = list(env.state[0].observation.planets)

    # Sun
    ax.add_patch(Circle((CENTER, CENTER), SUN_RADIUS, color="gold", zorder=2))
    # Rotation limit circle (dashed)
    ax.add_patch(Circle(
        (CENTER, CENTER), ROTATION_RADIUS_LIMIT,
        fill=False, linestyle="--", linewidth=0.4, edgecolor="grey", zorder=1,
    ))

    for p in planets:
        pid, owner, x, y, r, ships, prod = p
        if is_orbiting(p):
            color = "#5b9bd5"
        else:
            color = "#444"
        ax.add_patch(Circle((x, y), r, color=color, ec="none", lw=0.4, zorder=3))

    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(0, BOARD_SIZE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{title}\nseed={seed} n={env_init['n_planets']} prod={env_init['total_production']:.0f} "
        f"rot={env_init['rotating_share']:.0%} split={env_init['size_split']:+.2f}",
        fontsize=6,
    )


def main() -> int:
    archetypes = sorted(SEED_PANEL_BY_ARCHETYPE.keys())
    n = len(archetypes)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, name in enumerate(archetypes):
        seeds = SEED_PANEL_BY_ARCHETYPE[name]
        if not seeds:
            continue
        # Use the first seed (panel order is stable thanks to farthest-point seeding)
        render_one(axes_flat[i], seeds[0], name)

    for j in range(len(archetypes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"Orbit Wars seed panel — 32 archetypes ({len(archetypes)} cells, 1 representative seed each)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = REPO / "audit" / "seed-panel" / "preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
