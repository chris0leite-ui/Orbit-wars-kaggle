# 2026-06-01 — Live-replay diagnosis: ship-hoarding / under-expansion

Source: 48 live ladder games of sub **53248277** (size-balance fix, baked
ON), pulled via `replay_mine.py --pull`, confound-controlled against 120
champion games (sub 53182323). Triggered by PI ladder observations:
"not expanding quickly enough (vs istinetz)" + "failing to utilize ships,
losing midlands (vs xdddd)".

## Headline
The agent **hoards ships on a few planets instead of converting them into
territory.** This is the dominant live-loss mode, and it is a **baseline
trait** (not caused by the size-balance fix).

## Fleet-outcome distribution (replay-mine)
| outcome | champion 53182323 (n=120) | size-balance 53248277 (n=48) |
|---|---|---|
| defense (reinforce own planet) | 55.8% | **59.1%** |
| win (capture) | 22.1% | 25.2% |
| waste_attack (bounced) | **11.4%** | **4.6%** |
| waste_trajectory (sun/oob/vanish) | 10.1% | 10.0% |
| inflight | 0.7% | 1.0% |

- ~56–59% of all fleets just shuffle ships between planets we already own;
  only ~25% capture anything. Over-defensive posture is **baseline**.
- Size-balance fix **worked on its target**: bounced attacks 11.4%→4.6%
  (mode-D "send enough / skip unwinnable"). It nudged defense +3.3pp
  (mode-A clamp) — i.e. it made the hoarding slightly WORSE, which is the
  likely reason it came out neutral in the n=64 geometry A/B (40.6%).
- **waste_trajectory ~10% in BOTH** — untouched; Rule 47 budget is <2%.
  ~8% of all ship production thrown away flying into sun/off-map. Secondary
  but concrete lever.

## Per-game timelines (planets / ships; we lost all three)
**ep 78399717 — 2P vs istinetz (clean reproduction):**
step 49: us 3pl/**195sh** vs istinetz 9pl/329sh — we sat on 195 ships across
3 planets (~65/planet) while istinetz spread into 9. → 0 by step 99.

**ep 78398509 — 2P vs xdddd:**
step 114: us 15pl/**1123sh** vs xdddd 18pl/1052sh — MORE ships, FEWER planets,
kept losing planets while sitting on the pile. → 0 by step 228 (xdddd 35pl).

**ep 78399284 — 4P vs istinetz:**
comparable ships early, fewer planets; istinetz ran to 36 planets.

Signature in all three: **comparable-or-more ships than the winner, but
fewer planets, and slow neutral capture.** Out-shipped, under-expanded.

## Live performance context (do NOT over-read — confounded by time/opponent pool)
- 53248277: 32/48 = 66.7% (2P 72.4%, 4P 57.9%). μ=1180.3 at ~3h (settling).
- 4P weaker than 2P — consistent with team-up/suppression concern.

## Implication
- Lever is **expansion / ship-utilization, NOT defense.** Convert hoarded
  reserves into territory, especially early.
- The size-balance fix's mode-A clamp opposes this — candidate for revert.
- Two mechanism hypotheses to disambiguate via turn-by-turn trace of
  ep 78399717: (a) idle-planet coordination gate (solo launch 21% vs joint
  89% — idle planets don't fire because a lone launch looks unprofitable),
  (b) value function under-crediting territory vs holding ships. Rule 40:
  fix the valuation, not a hoarding cap.
