# v3_snipe (52544634) — games analysis + improvement backlog

> Companion to `audit/2026-05-11-v3-snipe-critical-review.md`.
> Drills into PER-GAME patterns in the 34 live replays + validates
> one fix candidate via 32-seed A/B.

## Headline findings

1. **Losses end in elimination, not attrition.** 95% of losses
   (19/20) have us eliminated (zero planets, zero fleets) by step
   158 median. 0% of wins ever lose all material. The dominant
   failure mode isn't "fall behind on points" — it's "get cleaned
   off the board completely."
2. **We do lose home in most wins too — recovery is the
   differentiator.** 79% of wins lose all initial-home planets at some
   point; 100% of losses do. **In wins we recover to a median of 28
   planets after home loss; in losses we peak at 6.** The strategic
   weakness is comeback, not defence.
3. **In-flight volume gap.** During the "tied" phase of a tracked 2P
   loss (steps 95–112), the opponent carried 600–900 ships in flight
   while we carried 300–500 — roughly 2x. We launch similar fleet
   COUNTS as opponents (109 vs 112 in the case study) but smaller fleets
   on average (44.6 vs 51.4 ships). The opp out-ships us in the
   offensive flow.
4. **Leader-snowball in 4P FFA is unmodelled.** Median % of our 4P
   captures-from-leader in WINS = 58%; in LOSSES = 45%. Wins
   correlate with attacking the leader more, but v3 has NO explicit
   leader detection or spoiler logic — the variance is just whoever
   happens to have nearby planets.
5. **"One ship too little" bounces are real but the obvious fix
   regresses overall.** 38 of 518 enemy bounces (7%) hit margin
   ∈ {+0, +1}: we sent exactly the garrison count or one ship over,
   then lost combat to one extra production tick. Fixing the formula
   to `target.ships + production*(eta+1) + 1` converts those bounces
   to captures BUT over-sizes the much more common static-target
   captures (where flight ETA is over-estimated by `(r_src + r_target)
   / v`), wasting ships elsewhere. **A/B v3.3 vs v3_snipe_frozen lost
   27/64 = 42.2% Wilson [30.9%, 54.4%]** — net regression. The fix
   needs to be targeted (orbiting + comet only) rather than blanket.

## Per-game deep dives

### Case 1: 4P snowball (episode-76299722, loss)

```
step  P0_TensorFlower   P1_fgwiebfaoish   P2_us_ChrisLeite   P3_Viltrum
   0      1/  10            1/  10              1/  10            1/  10
  25      2/  85            3/ 102           *  3/ 102            4/  84
  50      5/ 203            4/ 198           *  4/ 284            7/ 269
  75      4/ 314            4/ 214           *  3/ 369            9/ 473
 100      2/ 175            5/ 201           *  3/ 225           10/ 477
 150      0/   0            4/ 157           *  1/ 141           15/ 979
 193      0/   0            0/   0           *  0/   0           20/2355
```

P3 (Viltrum) ran away with this. Between step 25 and 75, Viltrum went from
4 planets to 9 while everyone else stagnated. **Our agent never targeted
Viltrum specifically** — we just attacked whoever was closest, which
included P0 and P1, helping Viltrum become un-catchable.

### Case 2: 2P swing (episode-76301845, loss vs ohkawa3)

At step 100 we owned 17 planets / 1058 ships; opp owned 15/1429. We
were *ahead* on planet count. Between step 100 and 130 we lost 10
planets while opp gained 10. Underlying cause: during the "even"
phase (steps 95-112), opp continuously held ~2x our in-flight volume.
Their fleets landed; ours hadn't yet been launched.

## Game-design level diagnoses

| Pattern | What v3_snipe does today | What it should do |
| --- | --- | --- |
| Elimination snowball | Score targets by ROI / cost; no concept of "we're being wiped" | When losing material rapidly, prioritise re-establishing a productive base; concentrate fleets rather than spreading |
| Recovery after home loss | Greedy snipe to whichever neutral planets are reachable; no coordination | gang_up mission class: simultaneous-arrival counter-attacks on the strongest enemy base |
| 4P leader snowball | All opponents treated equally in target selection | Spoiler mode: when ranked 3rd or 4th, withhold attacks on the WEAKEST and target the LEADER |
| Sub-1-ship bounces | `arrival_size` uses `production*eta`; off by one at static-target arrival turn | TARGETED fix (orbiting/comet only); for statics keep current formula since eta is over-estimated by `(r_src + r_target)/v` |
| In-flight volume gap | Same-turn arrival ledger blocks "double commit" | Allow controlled gang-up; ledger should permit second source when first is fragile |

## Improvement backlog (ranked by expected EV)

### A. (NEW, HIGH EV) Targeted off-by-one fix for orbiting/comet targets
The blanket `(eta + 1)` fix regressed (above). But on **orbiting** and
**comet** targets the v3.2 formula genuinely under-sizes by one
production tick. Implement as a conditional in `lib/mechanism.py::arrival_size`:

```python
is_dynamic = (
    target.id in world.comet_ids
    or is_orbiting(target_tuple)
)
prod_ticks = eta + (1 if is_dynamic else 0)
static_needed = target.ships + target.production * prod_ticks + 1
```

Add a `tests/test_mech_arrival_size_dynamic_targets.py` regression that
covers comet AND orbiting cases. Re-A/B at 32 seeds. **Expected lift:
small but positive — converts ~38 bounces without over-sizing statics.**

### B. (NEW, HIGH EV) 4P spoiler mission class
Detect leader at each turn (max ships among non-us). When we're ranked
3rd or 4th, apply a +50% score multiplier to missions whose target is
owned by the leader. Implement in `lib/missions/snipe.py::propose_snipe_missions`:

```python
ranks = sorted(((pid, total_ships(pid)) for pid in players), key=lambda x: -x[1])
our_rank = next(i for i, (pid, _) in enumerate(ranks) if pid == world.my_id)
leader_pid = ranks[0][0]
leader_bonus = 1.5 if our_rank >= 2 else 1.0
# ...inside the (src, target) loop:
if target.owner == leader_pid:
    score *= leader_bonus
```

**Expected lift: 4P-FFA winrate +10-20pp.** New test
`tests/test_mission_snipe_spoiler.py` covering rank detection +
score boost.

### C. (KNOWN, MEDIUM EV) gang_up mission class (H4 from hypothesis-board)
Multi-source simultaneous-arrival timer. When a target's predicted
garrison exceeds any single source's full garrison, propose a
gang_up mission combining N sources. Requires planner-level
coordination (not just per-source greedy). ~6-8h implementation.

### D. (KNOWN, MEDIUM EV) Recapture mission class
Sibling to reinforce — when we LOST a planet recently, score "retake
this before enemy fortifies." Roman has this; we don't. ~3-4h. Most
relevant to the comeback gap from §1.

### E. (NEW, LOW-MEDIUM EV) Score-function rebalance for big-fleet ROI
Current denominator: `(ships + distance + 1.0)`. Linear ship-cost
penalty makes us prefer tiny fleets. But TrueSkill rewards win/loss
binary, not margin — small inefficiencies don't help. Try
`(0.5 * ships + distance + 1.0)` so the ROI shape prefers
overwhelming-force commitments. Validates via local A/B + live diff.

### F. (DEFER) Comet aim mechanism re-enable
The 22.5% ablation regression from 3.5.C was likely caused by the
endpoint-only path-check bug we already fixed. Re-test with the
current trajectory-ray-cast guards.

## What we tried in this session

| Item | Result |
| --- | --- |
| Local↔live parity gate | 100% match (was 53% due to instrumentation bugs); permanent test added |
| P1 arrival_size adversary stacking (v3.2) | 32-seed A/B Wilson [45.6%, 69.1%]; 4P 93.8% vs frozen 90.6% — landed |
| P2 DEFAULT_HORIZON 110 → 250 (v3.2) | Landed; reinforce class still rarely fires (0.2/turn) → 4P spoiler is the higher-EV next step |
| P3 reinforce sizing (claimed wrong in critical review) | Retracted; formula is correct |
| Off-by-one one-ship-too-little fix (v3.3, blanket) | 32-seed A/B 27/64 = 42.2% Wilson [30.9%, 54.4%] — REVERTED |

## Reproducing this analysis

```bash
KAGGLE_API_TOKEN="$KAGGLE_KEY" python -m scripts.live_episode_summary 52544634
python -m scripts.episode_postmortem 52544634   # 100% parity, fleet outcome breakdown
# Specific analyses are inline-scripts run against postmortem-episode-*.json.
```

Live data files:
- `audit/live-episodes/52544634/summary.json`
- `audit/live-episodes/52544634/postmortem/roll-up.json`
- `audit/live-episodes/52544634/episode-*-replay.json`
- `audit/tournaments/20260511T19*.json` (v3.3 A/B; regressed)

## Cross-references

- Critical review: `audit/2026-05-11-v3-snipe-critical-review.md`
- Hypothesis board: `state/hypothesis-board.md` (H4 gang_up, H6 spoiler)
- Strategy ledger: `state/mechanism-ledger.md`
