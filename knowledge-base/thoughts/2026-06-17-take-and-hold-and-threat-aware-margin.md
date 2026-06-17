# 2026-06-17 — take-and-hold beats Producer V2; next = threat-aware dynamic margin

*Session record. The live agent is `least_resistance` (NOT producer_plus — that
line is superseded). This session diagnosed why we lose to the strongest peer
(Slawek's Producer V2), built the fix, confirmed it at proper n=32, and shipped.*

## TL;DR

- **Shipped `least_resistance` "take-and-hold"** (sub **53768768**, 2026-06-17,
  tarball sha `dc3a4f17`). Two gated levers now **default-ON**:
  - `LR_HOLD_MARGIN=0.5` — size **enemy** captures to take AND HOLD (surplus to
    survive the retake); larger sizes force source-combining ⇒ fewer/bigger
    **concentrated** fleets emerge naturally.
  - `LR_DEFEND=1` — **reinforce** our own planets an enemy fleet is about to flip
    (a regroup/defense move the agent was *structurally blind to*: candidate
    `targets` excluded own planets — `main.py` ~line 559).
- **Confirmed at n=32 INDEPENDENT seeds** (one game/seed, NO seat-reuse — see the
  methodology rule below): **2P vs Producer V2 OFF 14/32 → ON 21/32 (+7)**;
  **4P vs {V2, Roman-1224, konbu17} OFF 18/32 → ON 19/32 (parity, no regression)**.
  Both modes ≥ baseline; max turn < 570 ms.
- **Status at session end:** sub 53768768 was **PENDING ~2 h** with no score. I
  reproduced Kaggle's validation locally with the EXACT bundle (real loader, repo
  off `sys.path`, self-play 2P + 4P-to-250): clean, max 469 ms, 0 turns > 1 s, no
  error, entry = `agent`. **The bundle is sound; the delay is almost certainly the
  Kaggle eval queue, not us.** Backstop in the rolling pair: `lr-fixed` sub
  53741746 @ **μ 1115** (settled down from 1142). **First fresh-session task:
  re-check 53768768 → COMPLETE (passed) or Error (diagnose).**

## How we got here (the arc that worked)

PI watched a replay of us **losing to V2** (seed 768065184): even until ~turn 46,
then our planet count bled 8→0 while V2 snowballed to 16. PI's read, confirmed in
code: **V2 combines ships into fewer/bigger fleets and regroups; we spray minimum-
force grabs that don't hold, and we can't defend.** Code confirms:
- `targets = [p for p in planets if p.owner != me]` → we can **only** attack, never
  reinforce/regroup.
- `size = ceil(defenders)+1` → **minimum** force; captures don't survive the retake.

Fix = least-resistance pointed at **"take AND hold the opponent's production"**
instead of "grab any cheap production": hold-sizing (concentration emerges) +
reinforce (defense). Replays after the fix: 2P we now snowball and win the same
seed; 4P we eliminated all three opponents (one seed — not representative, see 4P
note). Confirmed at n=32.

## Conceptual insights (durable)

1. **Depth ≠ breadth.** Deeper search was null because the agent only ever
   *generates* one move-class (minimum-force grabs). "You can't out-calculate a
   move you never put on the table." The lever is **breadth + robustness in
   candidate generation**, not look-ahead depth.
2. **Concentration/regroup aren't add-ons** — they're what least-resistance
   becomes when the objective is "break the opponent" (take+hold) rather than
   "grab a planet". The PI's framing.
3. **Objective, valuation, and moves must change together.** Every prior lever
   changed only one and failed (see refuted list).

## Refuted this session — DO NOT re-walk (all null/regressing vs strong opponents)

Tested at proper sample against the **producer / Producer V2** (not weak v7_0):
- **leader-relative 4P objective** (gap-to-strongest via `opp_weights`): regressed
  4P (12/16 → 6/16 vs producer). The 2026-06-14 "wrong-objective" theory did **not**
  hold empirically.
- **value-commit** (value-ordered greedy): washed out / regressed.
- **enemy-boost** (rank ×1.5 on enemy targets): regressed 2P (it over-fires where
  the scorer already values denial); we gated it 4P-only, then dropped it.
- **anytime** (wider 2-ply + bank budget): null; and it never actually spent the
  60 s overage bank (agent finishes ~600 ms because it runs out of PLANS, not time).
- **deep rollout search** (`LR_ROLLOUT_DEPTH`, root-branch + K-turn producer
  rollout): parity vs producer at n=16 (fixed 2 / broke 2). All still gated
  default-OFF in `main.py` — **strip them in a cleanup** (they're dead weight).

## The benchmark map (real public agents, local)

Pulled real Kaggle kernels into `audit/external/` (gitignored). **Reproduce:**
`kaggle kernels pull <ref> -p audit/external/kernels-pulled/<name>`, then extract
the `%%writefile *.py` cell that contains `def agent` → a runnable `main.py`.
**Gotchas:**
- **Arity:** some agents are `def agent(obs)` (1-arg), others `agent(obs, cfg)`.
  Call via `env.run([...])` (the env handles arity) or an arity-aware wrapper —
  a 2-arg call to a 1-arg agent raises and looks like "idle" (this bit me).
- **Producer V2 / Roman "I'm Stronger"** import `orbit_lite`. **V2 runs fine on
  OUR `agents/producer/orbit_lite`** (verified) — no conflict with our agent.
  (Slawek's exact orbit_lite is also pulled to `audit/external/producer-utils/`.)

Confirmed-playing public agents (launch-count checked): **konbu17** (~85%-panel
ML-hybrid), **Roman-1224**, **ykhnkf-1100**, **vickimar-1110**, **Producer V2**.

Where we stand vs them:
- We **crush** the published field — 2P **6/6** vs konbu17 / Roman-1224 / ykhnkf;
  4P **8/8** vs {konbu17, Roman-1224, ykhnkf}.
- **Producer V2 is our one real peer** (we're built on the Producer's `orbit_lite`
  scorer). 2P vs V2: was a coin-flip on cherry-sized samples; at n=32 independent,
  **OFF 14/32, ON(take-and-hold) 21/32**.

## Methodology rule learned (now in `.claude/skills/kaggle-comp/improvements.md`)

**A/B independence: one game per fresh seed; rotate the SEAT across DIFFERENT
seeds, never within a seed.** Replaying one map from multiple seats = correlated
games; counting N×S as the sample size overstates confidence (it manufactured a
false 2P "lift" earlier, AND masked the real one under correlation). OFF/ON share
each seed+seat (valid *paired* diff); independence holds across seeds.
`scripts/verify_confirm.py` does this correctly (64 distinct seeds, 32/block).

## NEXT TASK — threat-aware dynamic margin (the middle option)

The flat `LR_HOLD_MARGIN=0.5` is a **placeholder constant** (Rule 40 band-aid). The
right margin emerges from the situation: **size each enemy capture to survive the
opponent's actual retake threat**, not a fixed fraction.

**Spec:**
- In `agents/least_resistance/main.py`, the candidate loop's `size` computation
  (currently `size = ceil(defenders)+1 (+ ceil(hold_margin*defenders))`), for
  **enemy** targets compute a per-target `retake_threat(T)`:
  - over enemy planets `P_e` (owner ≠ me, ≠ −1): their garrison counts as threat
    **iff** `P_e` can reach `T` within ~(our_eta + small window):
    `dist(P_e,T)/fleet_speed(P_e.ships) ≲ our_eta + W`;
  - plus enemy fleets already heading at `T`; plus opp production over `W`.
  - `retake_margin = MAX over reachable enemy planets` (survive the strongest
    single counter) — start with MAX (simpler, less over-commit) before trying a
    discounted SUM.
  - `size = ceil(defenders) + 1 + retake_margin`. Cap it (don't over-commit beyond
    the target's value). Isolated planet → ~0 margin (cheap); contested → big
    margin or **emergent skip** (can't afford ⇒ not funded).
- Gate `LR_DYN_MARGIN` (default OFF first), keep flat `LR_HOLD_MARGIN` as fallback.
- **Reuse before reinventing:** check `lib/world_model` (arrival ledger),
  `lib/geo/sense.py`, and `agents/producer/orbit_lite` for an existing
  "nearby-enemy-force / reach" helper.
- **A/B:** vs the **shipped flat-0.5 baseline**, `scripts/verify_confirm.py`
  pattern — **n≥32 INDEPENDENT seeds**, 2P + 4P, vs Producer V2 + the field.
  Bar: ON ≥ flat-0.5 in BOTH modes (ideally beats it). Same Rule 46 smoke + Rule
  42 gate + PI sign-off before any submit.

## Other queued levers (after the dynamic margin)

- **4P is only at parity** with the strong field (~60% of the ladder). Take-and-hold
  helped 2P most; 4P needs its own lever (FFA targeting / who-to-eliminate-first /
  kingmaker). Bigger prize than the margin.
- **Adopt V2's newer `orbit_lite` scorer** under our wrapper (we have its code) —
  could lift everywhere; our scorer is an older Producer fork.
- **Harden the slow turn:** earlier a 1441 ms 4P-vs-field midgame turn appeared
  (self-play is clean ≤570 ms). Bound the total turn so we never eat overage on the
  ladder.
- **Strip the refuted dead-weight flags** (leader-relative / value-commit /
  enemy-boost / anytime / rollout) from `main.py`.
