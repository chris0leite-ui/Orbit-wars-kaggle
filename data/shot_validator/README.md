# Shot validator labeled dataset

Labeled (feature, outcome) pairs for training the konbu17-style
shot-validator MLP described in
`knowledge-base/concepts/top-performer-strategies.md` §H14.

> **Status: labeling pipeline shipped; MLP training deferred to a
> future session.** Data prep is the long-tail; the actual MLP is
> straightforward to drop in once the dataset exists.

## Files

- `schema.json` — feature spec (versioned). Index → name and
  normalisation range for each of the 24 input features.
- `labels.parquet` — one row per labeled launch with `(features,
  label)`. Gitignored due to size; rebuild with
  `python -m scripts.label_shot_outcomes`.

## Feature spec (24-dim, all float32 in [0, 1])

| group | features |
|---|---|
| **source planet** | ships, production, radius |
| **target planet** | ships, production, radius, owner_mine, owner_neutral, owner_enemy |
| **shot** | ships_sent, ship_fraction (sent/src.ships), distance, eta, fleet_speed |
| **in-flight** | n_allied_fleets, ship_total_allied, n_enemy_fleets, ship_total_enemy |
| **meta** | turn, my_total_ships, enemy_total_ships, ship_diff, my_planet_count, enemy_planet_count |

Normalisations:
- ships, production, radius: scaled by global maxes from the corpus
- distance, eta, fleet_speed: scaled by env-derived constants
  (board diagonal ≈ 141, max eta = 200, max fleet speed = 6.0)
- turn: turn / 500
- counts: count / 40 (max planets in env)

## Label

`label = 1` iff the target planet is owned by the launching player
at step `min(launch_step + eta + 10, end_of_game)`. `label = 0`
otherwise.

The +10 step buffer reflects "shot was successful if we held the
target for at least 10 turns after arrival." This filters out fast
flips where we capture and immediately lose, which are noise
relative to true successful shots.

## Source corpus

`audit/external/replays/*.json` — the 50 top-10 replays + 10
midpack control replays produced for the strategy analysis.
Top-10 replays are gold-label (the player whose stats we're
mimicking) — labels reflect what those players' shots actually
achieved.

## Reproducing

```
python -m scripts.label_shot_outcomes
# → data/shot_validator/labels.parquet, schema.json
```

## Schema versioning

`FEATURE_VERSION = 1` (matches konbu17's notebook spec verbatim).
Any feature-set change should bump `FEATURE_VERSION` and write a
new `labels-v<N>.parquet`. The MLP loader is expected to assert
schema version on load.

## Next steps (deferred — for a future session)

1. Load `labels.parquet`; train a 24-32-16-8-1 sigmoid MLP with BCE
   loss. ~1000 epochs Adam on 10k+ examples per the konbu17 numbers.
2. Persist weights as inline base64 in the v4 submission bundle
   (Rule §2.12 prohibits ingress at evaluation time).
3. Add an `agents/v4/main.py` that calls the MLP after settle_plan
   and rejects intents where `P(shot_succeeds) < REJECT_THRESHOLD`.
4. 32-seed 2P A/B vs v3.5; target +5pp panel winrate (konbu17 saw
   +19pp over a weaker rule-base).
