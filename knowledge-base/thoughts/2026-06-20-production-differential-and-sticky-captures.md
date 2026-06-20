# 2026-06-20 — PI thought: production differential + sticky captures

> Append-only (Rule 35). Transcribed from PI voice-dump during a replay-watching
> session. Plain-English (Rule 0).

## The thought (PI's words, lightly cleaned)

When we launch attacks, we often attack **multiple** opponent planets at once, and
**many fail to capture** — the opponent defends or recaptures. The only way to win
is to **capture planets that stick**, so that **our production keeps increasing and
the opponent's production keeps decreasing**. That idea has to get *into the head*,
into the *thinking*, of our agent.

## Supporting replays the PI watched

- **seed 106499442, 2P vs kusui26 — LOSS.** Step 63: we (blue) are spread thin
  across many small planets with fleets scattered to several targets; opponent
  (orange) holds concentrated. Final: ChrisLeiteScha 1107 (−6) LOSS, kusui26 +9.
  → over-attacking / scatter, captures that don't convert. **2P, not just 4P.**
- **seed 1912745358, 4P (~step 50)** — earlier: drain a planet into an attack that
  bounces, then lose it.
- **seed 1991632357, 4P (~step 147)** — earlier: far / small launches, low impact.

All three are facets of the same thing: we spend ships on attacks that **do not
result in a held production gain**.

## How this maps to the code (diagnosis)

The agent's objective today is **ship-count differential at a fixed horizon**, with
the forward model assuming the **opponent launches nothing**:

- 2P chooser leaf `_project_value` (`agents/least_resistance/main.py:436`):
  `my ships − opponents' ships` at horizon H (18 turns 2P / 13 turns 4P). No
  production framing; opponent passive → a capture the opponent *would* recapture is
  scored as if it sticks → **over-credits non-sticking captures** and scatter.
- 4P robust leaf `_project_outcome` (`main.py:467`, default-OFF except the 4P probe):
  already closer — it credits **production** (`pcred * prod`) and **defensibility**
  (`vuln` = enemy mass that can reach each of my planets − its ships,
  `main.py:511-527`). But it's 4P-only and, on seed 1912745358, the robust config
  actually *lost* where the plain 2-ply won → mis-tuned / too cautious as shipped.

So the principle the PI wants is **not yet the agent's objective**. The fix
direction: make the value the agent optimizes a **production differential that only
counts captures that hold** — own production up, opponent production down, captures
discounted by recapture risk (defensibility). This subsumes the earlier "exposure
fix" (don't drain/expose) and the "far/small launch" waste.

## Implications / open tensions

- **2P-regression history (HANDOVER):** the capability/production leaf lost 2P 0/4
  vs the proven ship-count 2-ply in an A/B. BUT that was win-rate A/B; the PI now
  judges by watching, and the PI's complaint is we attack **too much / scatter** —
  the *opposite* of the old "too passive" worry. A hold-aware production objective
  that is more selective about captures may reconcile both. Watch for over-caution.
- The greedy plan **construction** still uses the producer's ship-based
  `score_units` (`main.py:1288`); even with a production-differential *chooser*, the
  candidates offered may already be scatter. May need the construction to be
  hold/production-aware too, not just the final pick.
- Metric to watch in replays: **production share over time** (mine vs each
  opponent). "Are we growing ours and shrinking theirs?" is the PI's win condition,
  so the trace harness should report it, not ship counts.

## PI follow-up observation (same session) — the corner planet

> seed 167497264, 4P, step 35, we are GREEN (2nd); YELLOW (S4K50M) is 1st.
> PI: "why do we not capture the planet in the corner like yellow does?"

This is the **mirror image** of the far/small-launch complaint and confirms the
diagnosis. Two causes, both pointing at the same fix:

1. **Short sight.** 4P planning horizon is `PROJECT_HORIZON_4P = 13` turns. A far
   corner capture's fleet arrives *after* the window, so the leaf never sees the
   production payoff — only ships leaving home → scored ~worthless. Need a LONGER
   sight for the production objective so far defensible captures are visible.
2. **Ship-count objective.** A far-but-DEFENSIBLE corner (safe, few neighbours,
   adds production forever) looks identical to empty space under ship-count. Under a
   production+hold objective it scores high (production credit, low vuln).

Reconciliation of the two complaints: skip far ENEMY captures that **bounce / won't
hold**; TAKE far NEUTRAL captures that are **defensible and add production**. The
discriminator is "production you can hold," exactly the PI principle. The fix needs:
production-credit + defensibility (hold) + adequate horizon (sight).

## Next action (this session)

Embed a production-differential + hold-aware objective behind a default-OFF knob,
render before/after replays (incl. the 2P loss seed 106499442) with a
production-share trace, and iterate on PI feedback. No A/B (PI directive).
