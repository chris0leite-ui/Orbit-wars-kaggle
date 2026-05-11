# v3_snipe (52544634) — critical review

> Compiled 2026-05-11 from live-ladder replays (n=34 v3_snipe, n=57 v2 prior)
> + replay-driven instrumented re-execution (`scripts/episode_postmortem.py`)
> + parallel-branch architecture review (precision-physics-engine-ymJkA).

## TL;DR

v3_snipe **lifted live μ by +90.2 (965.3 → 1055.5)** despite **absolute
winrate dropping** from 50.9% (v2) to 41.2% (v3) over the first 34 games.
TrueSkill matched it against ~150μ stronger opponents — the gain comes from
opponent quality, not games-won-per-attempt. The headline error mode is
**combat sizing, not physics**: v3 bounces 14.7% of its fleets off enemy
planets (v2: 7.6%), having traded the physics-loss bucket (vanish/sun/oob
went 18%→9%) for an under-sized-fleet bucket. 4P FFA is the worst seat
(35.3%, sharply below 2P's 47.1%), reversing a v2 anecdotal high-water mark.

**Three concrete defects, ordered by EV/cost:**
1. `arrival_size` is mis-calibrated for v3's more-ambitious target selection
   — fleets arrive too small and bounce. (`lib/mechanism.py::arrival_size`)
2. The bundler does not pin lib/ source state; our local re-execution
   matches the live agent on only 53% of turns, blocking us from doing
   counterfactual analysis after a submit. (`scripts/bundle_agent.py`)
3. The `reinforce` mission class produces 0.2 candidates/turn (vs 116
   snipe/turn); its `DEFAULT_HORIZON=110` scan is doing real work but
   firing too rarely to defend home planets in time. (`lib/world_model.py:34`,
   `lib/missions/reinforce.py`)

**Parallel-architecture A/B settled:** the precision-physics agent
(parallel branch) lost 6/16 to v3_snipe AND its p95 turn time (1444ms)
exceeds the 1s `actTimeout` — not submittable as-is. Keep v3_snipe in
the rolling-last-2 slot. Details in §6.5.

## 1. Live data (34 episodes, first ~26 h after submit)

```
metric              v2 (52532938)   v3_snipe (52544634)   Δ
publicScore (μ)     965.3           1055.5                +90.2
total winrate       50.9% (29/57)   41.2% (14/34)         -9.7pp
2P winrate          53.8% (21/39)   47.1% (8/17)          -6.7pp
4P winrate          44.4% (8/18)    35.3% (6/17)          -9.1pp
median ep steps     189             178                   -11
unique losses-to    27 opponents    20 opponents          -
```

v3 plays SHORTER games on average (median 178 vs 189). Games-where-we-lose
average **88 fleets launched in 2P / 58 in 4P** (median 85 / 54), versus
**159 / 138** in wins. The dominant signature in a loss: **we run out of
runway before the game settles**. There is no single opponent who
counters us — losses are spread across 20 distinct ladder players, which
rules out a "specific cheese strategy" hypothesis.

## 2. Fleet outcome breakdown (3,532 v3 fleets vs 8,739 v2 fleets)

The 2026-05-11 capture-success probe counts "reached origin planet" and
says v3 is at 97.2%. **The metric that matters is "reached AND was
strategically useful," which is much lower:**

```
outcome              v2          v3_snipe       Δ
captured             31.1%       39.8%          +8.7pp   ✅ better target picking
reinforced_self      31.5%       32.9%          +1.4pp   ≈ same
bounced_enemy        7.6%        14.7%          +7.1pp   ❌ DOUBLED — under-sized fleets
bounced_neutral      0.5%        0.4%           ≈ same   negligible
arrived_but_lost     0.1%        0.3%           +0.2pp   negligible
vanished_in_space    17.0%       8.7%           -8.3pp   ✅ trajectory-fix worked
sun                  1.2%        0.0%           -1.2pp   ✅ trajectory-fix worked
alive_at_end         10.9%       3.2%           -7.7pp   v3 ends games faster
```

**The trade is explicit:** v3's full-trajectory ray-cast eliminated the
physics-loss bucket (vanish + sun + alive-at-end went from 29.1% to 11.9%),
but the freed-up emission budget went into more-ambitious attacks that
**fail combat at the target**. The 7.1pp jump in `bounced_enemy` is the
single largest regression. At 3,532 fleets, that's ~250 wasted launches
in 34 games (~7 per game) — the same order of magnitude as the wins-lost
delta.

Mechanism behind this: v3 picks targets by `value / (ships + distance + 1.0)`
which favours **high-production planets** regardless of garrison. `arrival_size`
bumps ships by predicted production growth, but doesn't model the
**defender's incoming fleets** — when an enemy stacks two same-step
arrivals at a contested planet, our `+1` cushion is wiped out.
`lib/missions/reinforce.py:84` uses post-flip survivor count to size
defence (under-counts pressure on multi-attacker arrivals); a parallel
sizing bug on the offense side, in `lib/mechanism.py::arrival_size`,
under-counts opposition.

## 3. Win-vs-loss decomposition

```
                        2P-win    2P-loss    4P-win    4P-loss
n games                 8         9          6         11
avg fleets/game         159       88         138       58
median steps            184       163        226       184
captured %              35.9      37.9       43.3      45.2
bounced_enemy %         9.7       23.6       11.1      18.1
vanished_in_space %     11.0      8.0        5.0       10.1
idle-source rate (med)  17.7%     0%         25.0%     0%
```

**Counterintuitive but consistent:** idle-source rate is HIGH in wins and
ZERO in losses. Idle source = "I have surplus ships and no useful target
this turn." When we're winning we have plenty of ships and only some need
launching; when we're losing every source is contributing every turn but
the contribution isn't enough. **Idle source is a symptom of strength,
not weakness in this data — reversing the original hypothesis.**

The bounce-rate gap (2P: 23.6% in losses vs 9.7% in wins; 4P: 18.1% vs 11.1%)
is the dominant differential. **In losses we send fleets that arrive and
lose the combat at twice the rate of wins.**

## 4. Code-level critique (concrete defects)

### 4.1 `arrival_size` undercounts adversary stacking

`lib/mechanism.py::arrival_size` adjusts ship count by predicted **own**
production growth during flight, but does not account for the defender's
incoming fleets at the same arrival step. In contested 2P endgame and
4P FFA crossfire, two adversary fleets can same-step-arrive at our
target. Per `lib/combat.py:22-76`, the largest attacker minus second-largest
survives; our `attacker_strength + 1` cushion only beats one defender at
a time. Live data: 2P losses bounce at **23.6%**, 4P losses at **18.1%**
— both vastly above the ~10% baseline in wins.

### 4.2 Reinforce class produces 0.2 candidates per turn (vs 116 snipe)

`lib/missions/reinforce.py` scans the `WorldModel` timeline for planets
predicted to flip enemy and proposes defensive missions. Over 11,674 turns
across 91 episodes, **reinforce produced ~3,500 candidates total — about
30 per game** (0.2/turn) vs **134,000 snipe candidates** (116/turn). The
`DEFAULT_HORIZON=110` (lib/world_model.py:34) caps the timeline scan
before many flip-events become visible; raise to ~250–500 (matches
`time_to_hold`'s `EPISODE_STEPS=500` framing in `lib/missions/snipe.py:29`)
to give reinforce more triggers.

The cost: WorldModel timeline build scales linearly with horizon × planets,
roughly +20ms per turn at horizon=250 vs 110. At current p95=14ms / p99=27ms
that doubles our turn budget; still 70× under the 1s actTimeout. Worth it.

### 4.3 Bundle-vs-source drift (project-level, surfaced in this review)

The replay-driven re-execution matched the live agent's actions on only
**53% of v3_snipe's turns** and **31% of v2's turns**. The agent is
deterministic; this is bundle drift. Either:
- `scripts/bundle_agent.py` reorders or inlines lib/ source in a way that
  affects iteration order or set ordering;
- the live env runs a different version of `kaggle_environments` than the
  local install (`Planet` is imported from
  `kaggle_environments.envs.orbit_wars.orbit_wars`, `lib/intent.py:18`).

This blocks us from doing **counterfactual analysis on live replays**:
"would our current code have done X here?" Promote `tag: bundler-action-match-drift`
to the friction log; fix the bundler to either (a) emit a content-addressed
bundle hash so we can pin source state, or (b) run a deterministic
self-play parity check post-bundle.

### 4.4 No idle-source fallback (originally hypothesised but mostly inverted)

`lib/planner.py:72-96` will produce zero intents for a source if all its
candidates are filtered by the same-turn arrival ledger. The fix (try the
next-best mission for that source) is still correct **architecturally**,
but the empirical signal flipped: idle-source rate is HIGHER in wins than
losses. Implementing the fallback may help marginally; do not lead with it.

### 4.5 Other code-level points carried from the design review

These were flagged in the plan but the live data doesn't suggest they
dominate; tag for follow-up if the top items are addressed.

- Magic numbers in score function: `+1.0` (snipe.py:72, reinforce.py:95),
  `SUN_SAFETY=0.5` (trajectory.py:49). No ablation data.
- `comet_aim` excluded; fast comets still use straight-line lead_aim_v2 and
  miss. Worth re-enabling after 4.1 is fixed.
- O(N²) snipe enumeration safe at current planet count; revisit if board
  grows.

## 5. Game-design framing (binds the action items)

| Mechanic | Implication |
| --- | --- |
| TrueSkill updates on win/loss/draw only, **not** ship margin | Our `value = production × time_to_hold` is the wrong objective in principle; +1 ship at game-end is worth as much as +1000. The +90μ lift came from **fewer-but-against-harder-opponents** wins, not from margin-stacking. |
| Win = highest ship count at step 500 OR last-team-standing | Median v3 game length is 178 steps — most games end **before** step 500 by elimination. The `time_to_hold = EPISODE_STEPS - step` factor is over-weighting late-game production that never materialises. |
| Fleet speed = `1 + 5·(log ships / log 1000)^1.5` | Bigger fleets arrive faster. v3's linear-distance cost gives no credit to the speed bonus → systematic under-sizing on long-range targets. Compounds with 4.1. |
| 4P FFA = ~50% of v3's ladder games (17/34) | v3 has no 4P-specific logic. v2's earlier 85.7% 4P over 7 games was small-sample noise; over 18 games v2 settled at 44.4%, and v3 regressed to 35.3% — likely from `reinforce`'s timeline scan reserving ships v2 would have launched. |
| Validation episode pre-ladder; crashes = 0 games | Silent drops produce empty turns (safe). v3's full-trajectory guards are net-positive on this dimension (zero sun-kills, vs v2's 107). |

## 6. Project-level critique

### 6.1 Parity-grade-work pattern, confirmed

Three v3 builds (v3.0 bit-exact, v3.1 drop-one 50/50, v3_snipe local
57.8%) produced one live μ lift. That's not "parity" — it's "wins
slowly." But the **+90μ in 24 h vs +473μ to cliff** projection says we
need 5 more equivalently-good builds in 43 days to close the gap. Five
days per cycle if everything goes well — tight but feasible.

### 6.2 The 8-seed-noise habit is now load-bearing for slot decisions

The 8-seed 68.8% lift from v3.1 collapsed to 50/50 at 32 seeds. v3_snipe's
local 32-seed 57.8% had Wilson-lo 45.6%, consistent with parity. Live
absolute winrate dropped 9.7pp. **Local seeded benchmarks are
non-predictive of live μ; both 8-seed and 32-seed point estimates are
plausibly noise relative to the live opponent distribution.** The
real-time gate must be live data after ≥24h, not the local panel.

### 6.3 Rolling-last-2 eviction discipline failed once today

v3_snipe push evicted v1.2/roi (μ=1006.9) without a pre-recorded
decision (`friction.md` tag `rolling-last-2-tradeoff-needs-explicit-decision-record`).
**Outcome was good** (v3 at 1055.5 > v1.2 at 1006.9 by 48μ) but the
process is brittle.

### 6.4 Bundler fragility — TWICE in one session, plus action-match drift

- `tag: bundler-missing-block-e-modules` — first bundle attempt crashed
  10/10 self-play with `NameError`.
- **New:** action-match drift (this review, §4.3). The bundler is not
  reproducibility-grade. Promote to Rule 12 sub-clause: every submit
  must include a bundle hash that survives lib/ edits.

### 6.5 Parallel-architecture work (precision-physics-engine-ymJkA)

A sibling branch shipped a deterministic intercept solver + global greedy
planner (`agents/precision/`, 5 commits, last 12:41 UTC). It has 11
tests including 4P + packaged-submission, an explicit 0.85s per-turn
deadline, and won the first smoke seed (seed 42, 200 steps) against
v3_snipe. The full **8-seed 2P A/B (workers=4, 500 steps, both seats)
result:**

```
v3_snipe P0 vs precision P1: v3=5, prec=3, draws=0   (v3 P0 WR 62.5%, Wilson [30.6%, 86.3%])
precision P0 vs v3_snipe P1: prec=3, v3=5, draws=0   (prec P0 WR 37.5%, Wilson [13.7%, 69.4%])
OVERALL: precision wins 6/16 = 37.5% vs v3_snipe
v3_snipe p95 turn time: 32.9 ms     precision p95 turn time: 1444.6 ms
```

**Verdict: precision is NOT submittable as-is.** Two blockers:
1. **Loses head-to-head locally (37.5%).** seed-42 was noise.
2. **p95 turn time exceeds the 1s `actTimeout` by 44%.** Live env will
   timeout some turns (env-enforced 1s) and likely fail the validation
   episode. Even though the agent has a 0.85s internal deadline, the
   measured p95 is 1.4s — suggests the deadline check is happening
   AFTER the per-turn planner has already done its expensive work, or
   the local hardware is slower than the agent assumes.

**Slot recommendation:** keep v3_snipe in the rolling-last-2; spend the
next slot on §7-P1 (`arrival_size` fix). Do NOT submit precision until
both blockers are addressed. The precision branch may have value as a
SCORING ENGINE we adopt (its enemy-projection logic could inform §4.1),
but as a full-agent submission it's a regression.

Tournament artifact: `audit/tournaments/20260511T143953Z.json`.

## 7. Improvement directions ranked by EV/cost

| ID | Action | Site | EV (live μ) | Cost | Risk |
| --- | --- | --- | --- | --- | --- |
| **P1** | Fix `arrival_size` to account for adversary same-step stacking (`+1` cushion → `+ max(predicted_enemy_arrivals)`) | `lib/mechanism.py::arrival_size` | High (+30–60μ) — addresses doubled bounce rate | 2-4h | Low; localised; gate via 32-seed A/B + live |
| **P2** | Raise `DEFAULT_HORIZON` 110 → 250, re-test reinforce candidate rate | `lib/world_model.py:34` | Medium (+10–25μ) — defence rarely triggers today | 1h + retiming | Low; turn budget still 50× under timeout |
| **P3** | Bundler reproducibility hash + post-bundle parity check vs source | `scripts/bundle_agent.py` | High (unlocks reliable replay analysis) | 3-4h | Low; tooling |
| ~~P4~~ | ~~A/B precision vs v3_snipe~~ | resolved | precision loses 6/16 + timeout violation | done | n/a |
| **P5** | 4P-specific logic: when ranked 3rd-or-worse, withhold attacks against the strongest player ("spoiler" mode) | new `lib/missions/spoiler.py` | Medium (+10–20μ on 50% of games) | 4-6h | Medium; needs panel infra for 4P |
| **P6** | Add `+1` cushion → +k buffer ablation on `lib/missions/reinforce.py:84`, switch survivor→pre-flip sizing | `lib/missions/reinforce.py` | Medium (+5–15μ) | 1h | Low |
| **P7** | Comet aim re-enable (currently excluded); fast comets miss with lead_aim_v2 | `lib/mechanism.py:503` | Low-Medium (+5–10μ) | 2h | Needs ablation pair |

**Do NOT:**
- Spend another slot on a v3_snipe variant before P1 lands. The next push
  must move the bounce-rate needle or the architecture (e.g. precision).
- Rerun the 8-seed local smoke as a lift-claim. The data shows local
  benchmarks are non-predictive past ±20pp; trust 32-seed Wilson-lo only
  and confirm live within 24h.

## 8. What we measured but cannot yet act on

- **Per-opponent skill gradient**: 20 unique opponents beat us once each.
  No clustering. To dig deeper would require pulling each opponent's
  public agent (if available) and running them through the postmortem
  too — out of scope for this review.
- **Which fleet-outcome subset is "wasted ships"**: we count
  bounced_enemy as a loss but they DO deplete enemy garrison by some
  amount. A true cost accounting would track `(ships_we_sent −
  enemy_garrison_reduction)` per bounce. Quick to add to the postmortem
  script if needed.

## 9. Reproducing this review

```bash
KAGGLE_API_TOKEN="$KAGGLE_KEY" python -m scripts.live_episode_summary 52544634 --pull
KAGGLE_USERNAME=ChrisLeiteScha python -m scripts.episode_postmortem 52544634
KAGGLE_USERNAME=ChrisLeiteScha python -m scripts.episode_postmortem 52532938
```

Postmortem outputs land in `audit/live-episodes/<sid>/postmortem/`. The
roll-up.json is the data source for §2 and §3 tables above.

## 10. References

- `audit/live-episodes/52544634/summary.json` — live winrate aggregator.
- `audit/live-episodes/52544634/postmortem/roll-up.json` — per-fleet outcome
  classification + per-turn telemetry.
- `audit/live-episodes/52532938/postmortem/roll-up.json` — same for v2.
- `audit/2026-05-11-capture-success-probe.json` — the 97.2% "reached"
  number that this review reframes.
- `state/hypothesis-board.md` — open ideas (H4 gang_up, H8 bipartite, H6
  spoiler tactics) — P5 corresponds to H6.
- `scripts/episode_postmortem.py` — the diagnostic tool (new this session).
- `origin/claude/precision-physics-engine-ymJkA:agents/precision/` —
  parallel-architecture agent for the §6.5 A/B.
