# 2026-06-18 — The snowball aim + the 4P midgame-collapse diagnosis

> Long session. Many missteps (documented below as lessons), ending in a genuine
> breakthrough about *what we're aiming for* and *where our agent actually fails*.
> Live agent unchanged: shipped `least_resistance` (all experiments were default-OFF
> and have been reverted). PI = Chris.

## The aim (PI-confirmed): the SNOWBALL

After studying a top-agent game, the PI confirmed the target playstyle:
**aggressive, compounding expansion to many planets + a large ship reserve, acting
MORE and more productively — not less.** More planets → more production → bigger
bank → more captures → more planets.

## Elite study — Piotr Gabrys (#8, ~1597), a recent 4P ladder game (episode 80432392)

Piotr won; we (calibration sub) were eliminated. Per-seat over 135 steps:

| agent | launches | active turns | max ship bank | max planets |
|---|---|---|---|---|
| **Piotr #8 (WINNER)** | **55** | **42** | **880** | **19** |
| konbu17 | 18 | 15 | 215 | 4 |
| us | 14 | 14 | 162 | 3 |

The elite acts ~3-4x more than us, banks ~5x more, expands ~6x more. It does NOT
under-act or hoard idly — it acts a lot AND every action compounds.

## The real failure: 4P MIDGAME COLLAPSE (not under-expansion)

Diagnosed on the shipped agent, 4P vs the strong field (V2+Roman+konbu), env.run,
3 seeds:

```
seed 1234: peak 13 planets @step70  -> final 0  (loss)
seed 77:   peak  7 planets @step74  -> final 0  (loss)
seed 2026: peak 15 planets @step72  -> final 0  (loss)
```

We expand FINE early (7-15 planets by ~step 72 — comparable to the elite's pace),
then **collapse to 0**. Free neutrals are gone by ~step 40 (we grabbed our share).
The bank at peak is thin (114-475 ships across many planets). Mechanism:
**`LR_DEFEND` is hard-OFF in 4P** (`os.environ.get("LR_DEFEND", "1" if num_seats<=2
else "0")`), so our thin planets are never reinforced and the 3-opponent field
overruns them. The elite banks 880 and holds.

So the long-standing "4P under-expansion" framing was WRONG — the problem is holding
the midgame so the snowball can keep compounding.

## Methodology lessons (these caused most of the wasted effort this session)

1. **Use `env.run` for ALL eval — never hand-roll manual `env.step` loops.** My
   throwaway /tmp scripts manually stepped the env and fed opponents their
   observations by hand; that corrupted the opponent (V2 launched fleets out of
   bounds, lost games it should win). The project tools (`fast.py eval`,
   `verify_confirm.py`) use env.run and were never broken.
2. **Single games are pure noise here.** Same seed, same shipped agent, two env.run
   runs gave 27-0 AND 0-28. Never conclude from one game; average over n>=32.
3. **Restricting the agent's actions loses.** Dead ends this session, all reverted:
   one-capture cap, concentrate-into-one-strike, accumulate-and-strike, and
   `LR_CONFIDENT` (drop low-confidence captures + reflexive reinforcements + far-off
   launches). `LR_CONFIDENT` made us idle ~78% of turns (vs shipped ~33%) and lose
   3/3 — the exact opposite of the elite. The producer's frequent action is
   load-bearing.
4. **The "fleets into the sun" alarm was false** — a crude infinite-ray detector;
   the precise path-to-target check shows the producer never aims across the sun.

## Next steps (in progress)

- **Testing `LR_DEFEND=1` in 4P** (existing lever, no new code) via a paired env.run
  A/B vs V2+Roman+konbu, n=32, seat-rotated (`/tmp/ab_defend_4p.py`). Question: does
  reinforcing held planets stop the midgame collapse? RESULT: _pending_.
- If defense alone turtles (the old finding) or isn't enough, build a hold/reserve
  mechanism that keeps a defensive garrison per planet WITHOUT stopping expansion —
  the elite's expand-AND-hold-AND-bank balance.
- Validate everything on the real harness, averaged. The bar is beating the field in
  4P (60% of the ladder), where we currently collapse.
