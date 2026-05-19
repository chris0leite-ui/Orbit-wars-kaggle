<!--
DRAFT — Kaggle discussion post.
Not posted. PI reviews & pastes manually into the Kaggle web UI.
When posting:
  1. Title goes in the title field (not the body).
  2. Replace `PASTE_KAGGLE_UPLOAD_URL_HERE` after uploading
     audit/seed-panel/preview.png via the Kaggle editor.
  3. Strip this HTML comment.
-->

## Title

128 seeds, 32 game-shape archetypes — a small eval panel

---

## Body

Hi all — sharing a small panel of seeds we've been using for local A/B
testing, in case it's useful. Comes up partly in response to
[this question on fixed-seed testing](https://www.kaggle.com/competitions/orbit-wars/discussion/695447).

It's **128 seeds, stratified across 32 game-shape "archetypes"**, so
when you compare two versions of your agent you can see *which kinds
of boards* a change helps or hurts — not just an overall winrate.

The 32 archetypes come from three axes:

- **Production level** (4 bins): `low_prod` / `med_low_prod` /
  `med_high_prod` / `high_prod` — proxy for game pace.
- **Rotating share** (4 bins): `mostly_static` / `mixed_static` /
  `mixed_rotating` / `mostly_rotating` — how much of the board is
  on orbital paths.
- **Size split** (2 bins): `big_static` vs `big_rotating` — whether
  the larger planets are stationary or rotating.

4 × 4 × 2 = 32 cells. 4 seeds per cell, picked to spread inside each
cell as well. Seeds were drawn from `range(0, 10_000)`.

### Preview

One representative board per archetype (rows = production level,
columns walk rotation × size-split):

![Seed panel preview — 32 archetypes](PASTE_KAGGLE_UPLOAD_URL_HERE)

(Sun is gold; static planets grey; rotating planets blue. Dashed
circle = max rotating radius.)

### The seeds

The list is **interleaved by archetype**: the first 32 entries cover
all 32 cells once, the first 64 cover each cell twice, etc. So if
you only want a quick A/B you can take `SEEDS[:32]` and still get
balanced geometry coverage.

```python
SEEDS = [
    5199, 2083, 3493, 1649, 3233,  405, 3335, 1030, 1467,   78,   32, 1900,  647,  417,    1, 2560,
     272,  585, 1265,  741,  489, 2537,  422,  787,  455,  324,  119,  828, 1049,  906, 1117, 1990,
    5274, 2661, 3774, 2794, 3578, 7045, 4333, 1153, 2412, 1750, 2078, 2957, 1843,  451, 1725, 4676,
     662, 1217, 4461, 4785, 5675, 3403, 4814, 1336, 2996, 2509, 3959, 2867, 2572, 2476, 1282, 2393,
    7542, 6328, 5923, 4252, 4027, 9408, 4693, 2726, 3154, 7166, 6858, 4393, 2177, 2663, 1948, 6475,
    4353, 6462, 5981, 8516, 7770, 6593, 4963, 3473, 3520, 7337, 8763, 9017, 6202, 5047, 1571, 6294,
    8909, 9327, 8514, 9240, 6548, 9658, 9812, 8686, 8717, 8866, 7079, 9306, 7314, 8782, 2419, 8526,
    9118, 9984, 8855, 9227, 8624, 8968, 7186, 9013, 4633, 8603, 9102, 9547, 9453, 7947, 3462, 9427,
]

BY_ARCHETYPE = {
    "low_prod__mostly_static__big_static":         [2560, 4676, 6475, 8526],
    "low_prod__mostly_static__big_rotating":       [   1, 1725, 1948, 2419],
    "low_prod__mixed_static__big_static":          [1900, 2957, 4393, 9306],
    "low_prod__mixed_static__big_rotating":        [  32, 2078, 6858, 7079],
    "low_prod__mixed_rotating__big_static":        [  78, 1750, 7166, 8866],
    "low_prod__mixed_rotating__big_rotating":      [1467, 2412, 3154, 8717],
    "low_prod__mostly_rotating__big_static":       [ 417,  451, 2663, 8782],
    "low_prod__mostly_rotating__big_rotating":     [ 647, 1843, 2177, 7314],
    "med_low_prod__mostly_static__big_static":     [1990, 2393, 6294, 9427],
    "med_low_prod__mostly_static__big_rotating":   [1117, 1282, 1571, 3462],
    "med_low_prod__mixed_static__big_static":      [ 828, 2867, 9017, 9547],
    "med_low_prod__mixed_static__big_rotating":    [ 119, 3959, 8763, 9102],
    "med_low_prod__mixed_rotating__big_static":    [ 324, 2509, 7337, 8603],
    "med_low_prod__mixed_rotating__big_rotating":  [ 455, 2996, 3520, 4633],
    "med_low_prod__mostly_rotating__big_static":   [ 906, 2476, 5047, 7947],
    "med_low_prod__mostly_rotating__big_rotating": [1049, 2572, 6202, 9453],
    "med_high_prod__mostly_static__big_static":    [ 787, 1336, 3473, 9013],
    "med_high_prod__mostly_static__big_rotating":  [ 422, 4814, 4963, 7186],
    "med_high_prod__mixed_static__big_static":     [ 741, 4785, 8516, 9227],
    "med_high_prod__mixed_static__big_rotating":   [1265, 4461, 5981, 8855],
    "med_high_prod__mixed_rotating__big_static":   [ 585, 1217, 6462, 9984],
    "med_high_prod__mixed_rotating__big_rotating": [ 272,  662, 4353, 9118],
    "med_high_prod__mostly_rotating__big_static":  [2537, 3403, 6593, 8968],
    "med_high_prod__mostly_rotating__big_rotating":[ 489, 5675, 7770, 8624],
    "high_prod__mostly_static__big_static":        [1030, 1153, 2726, 8686],
    "high_prod__mostly_static__big_rotating":      [3335, 4333, 4693, 9812],
    "high_prod__mixed_static__big_static":         [1649, 2794, 4252, 9240],
    "high_prod__mixed_static__big_rotating":       [3493, 3774, 5923, 8514],
    "high_prod__mixed_rotating__big_static":       [2083, 2661, 6328, 9327],
    "high_prod__mixed_rotating__big_rotating":     [5199, 5274, 7542, 8909],
    "high_prod__mostly_rotating__big_static":      [ 405, 7045, 9408, 9658],
    "high_prod__mostly_rotating__big_rotating":    [3233, 3578, 4027, 6548],
}
```

### How we use it

```python
from kaggle_environments import make

wins = {arch: [0, 0] for arch in BY_ARCHETYPE}  # [my_wins, total]

for archetype, seeds in BY_ARCHETYPE.items():
    for seed in seeds:
        for my_seat in (0, 1):                  # play both sides
            env = make("orbit_wars", configuration={"seed": seed})
            agents = ["my_agent.py", "opponent.py"]
            if my_seat == 1:
                agents = agents[::-1]
            env.run(agents)
            # ...score the game, update wins[archetype]...
```

Two small habits that made this useful for us:

- **Play both seats** for every seed. Otherwise a change that helps
  on one home-planet can look like it helps overall.
- **Aggregate per archetype**, not just overall. A new version that
  wins well in aggregate can still lose every game on a specific
  archetype slice — that's exactly the kind of regression a
  stratified panel catches.

### Notes

- Built against the simulator in `data/main.py`. Seeds in
  `range(0, 10_000)` are reproducible there.
- **2-player only.** 4-player games sample the same initial-planet
  RNG, so the geometry features carry over, but 4P-specific dynamics
  (3 opponents, scoring) aren't covered.
- The archetype labels are just human-readable bin names — nothing
  magical, you can re-bin however you like.

If you run it against your agents, let us know what you find.

---

*This post was drafted by Claude Code (Anthropic's coding assistant)
on behalf of our team. Replies in this thread are not automated; we
may pop in if something interesting comes up. Use, modify, or repost
the seed list freely — no attribution needed.*
