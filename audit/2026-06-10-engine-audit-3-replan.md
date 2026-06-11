# Engine audit #3 (2026-06-10 night) — three lenses, three mechanisms queued

PI directive: re-examine the architecture strategically (game expert /
mathematician / software engineer), accumulate evidence, implement the
strongest lift thoroughly. No submissions left today (5/5 used), so tonight
is build + measure; tomorrow has 5 fresh slots.

## Lens findings

**Game expert — defense sizing is structurally absent from the size menu.**
When a wave's target is our own planet, `capture_floor` returns 1 (correct:
arrivals just add to the garrison). But the multi-size grid derives its three
candidate sizes from that floor: {ceil(floor·overkill) ≈ 1–2, 2× that, full
safe_drain}. The one size that matters for a defense — the projected deficit
at flip time — is not in the menu unless safe_drain happens to coincide.
This is the modeling shape of the measured hold-rate gap (ours 0.59 vs top
teams 0.74–0.85). `_apply_reinforce_deficit_floor` fixes exactly this and
already exists, but its only verdict came from the 4P vs-producer panel —
the yardstick that failed three live checks. Re-judging under the corrected
referees (champion + attribution vs the live stack).

**Mathematician — the veto is a filter where the model wants a fixed point.**
Pass 1 plans against the opponent's do-nothing-conditioned launches; the
veto predicts the reply to pass 1 and can only DROP waves. Three losses of
value: (a) vetoed ships idle instead of taking the next-best action; (b) the
defensive lane never sees the predicted counter (friendly_flip_targets
accepts a background — the reply is never fed to it on our side); (c) kept
waves are never re-sized/re-timed under the reply. The one-ply replan runs
the whole planner a second time with the reply as background and the roi
threshold re-normalized by do-nothing-under-reply (the same paralysis fix
the veto's mirror needed). Infrastructure was already in place:
plan_lite_waves takes background; the mirror takes base_background=mine.

**Software engineer — the turn budget makes the second pass nearly free.**
Live p50 71 ms / max 141 ms against a 1000 ms gate. Replan adds ~1 planner
pass + 1 mirror per opponent; with the veto verifying pass 2 the full stack
is ~5 planner-scale passes — still well inside budget (to be confirmed by
the Rule 46 smoke before any submit).

## Implemented tonight

- `PRODUCER_PLUS_REPLAN` (+ `_2P_ONLY` gate), `_apply_replan` in
  producer_plus/main.py; reply prediction extracted to `_predict_reply`
  (shared with the veto — veto tests 12/12 green, refactor behavior-neutral).
  Skips when pass 1 fired nothing or the reply is empty. 10 new tests
  (tests/test_replan.py) — gating, skip conditions, roi re-normalization.
- Bundle variants: `vetorf_replan` (replan + veto verify on the live stack),
  `replan_rf` (replan replaces veto — subsumption test), `vetorf_deficit`
  (deficit re-judge), `veto_rf_nq` (survivor stacking: does the quota's
  +31% @80 early lead convert when the reactive floor guards the frontier?).

## Measurement queue (margin harness, n=8, truncated at step 150, one job at a time)

1. veto_rf_nq vs champion + attribution vs veto_rf  (RUNNING)
2. Rule 46 timing smoke for vetorf_replan
3. vetorf_replan vs champion
4. vetorf_replan vs veto_rf (attribution — the submit-relevant gate)
5. replan_rf vs veto_rf (does replan subsume the veto?)
6. vetorf_deficit vs veto_rf (attribution)

Two-gate protocol: champion win AND positive attribution vs the live stack.

## Result: survivor stacking (veto+rf+nq), measured 2026-06-10 ~22:30 UTC

- vs champion: **6/8 wins, lead@80 +41.1% (ahead 8/8 games), @120 +54.7%**
  — strongest champion leg of the night (rf alone: 6/8, +28.3% @80).
- attribution vs veto_rf: **paired mean 0.0% at every checkpoint, 1 win /
  2 losses / 2 exact-mirror draws** in 5 clean games (3 games hit in-game
  timeouts — concurrent bundling/pytest during the leg, lesson re-learned:
  NOTHING runs alongside a measurement leg, however small it looks).
- Verdict: **fails gate 2 — not promoted.** The quota's early expansion
  adds nothing the rf stack wasn't already converting; its champion-leg
  dominance is another instance of why champion wins alone don't gate
  (upsize precedent). Revisit only with a late-game conversion fix.

## Results v2 (2026-06-11 morning, namespaced referee, container restart in between)

First relaunch was VOID — same-process env contamination (both agents ran the
focal stack; exact mirror draws on seeds 0-1, seat-split results elsewhere).
Fixed with a PPNSX-namespaced referee copy (_ns_veto_rf.py); seeds 0-1 then
diverged properly (Rule 38 reproduce-verify). In-game "None reward" errors
were 4-core box contention, NOT agent defects: solo reruns show replan max
turn 274 ms (budget 1000 ms).

Attribution vs the live stack (veto + reactive floor), seeds 0-3 paired:

- **replan (one-ply, full)**: truncated leg 4/6 with +19% paired @120 on the
  clean seeds — but solo FULL games on the two broken seeds were 0/4. Honest
  read: seeds split 2-2, big margins both directions. decision_diff on a
  losing seed: UNDER-AGGRESSION — 16 capture-sized launches vs the live
  stack's 24; pass 2 treats predicted parries as fixed even for attacks it
  then doesn't make (phantom-parry conservatism). NOT promotable as-is.
- **background-aware floors**: 3/7, paired -15% @120 / -22% @250 —
  ELIMINATED (defensive over-caution: predicted strikes feeding safe_drain
  hold garrisons home / evacuate too eagerly on imperfect predictions).
- **deficit-sized defense**: 4 wins / 2 draws / 1 loss, paired -0.6% @120 /
  +8.6% @250; one seed bit-identical (mechanism never fired). Mild,
  not decisive — candidate passenger for a future composite.

## Next mechanism (from the replan diagnosis): REDIRECT

Keep plan→veto unchanged; when the veto kills waves, run ONE extra planning
pass with surviving waves committed (sources debited, committed effects +
predicted reply in the scorer background) and spend only the freed ships on
next-best actions. Recovers the idle-ship value the veto leaves on the table
without reopening pass-1 commitments (no oscillation, no phantom-parry
suppression of the whole plan).

## Result: redirect (2026-06-11 ~08:15 UTC) — ELIMINATED

2/7 vs the live stack, paired -25.1% @120 / -38.0% @250. Won only seed 0
(where the full replan lost), collapsed on seeds 1-2 (where the replan won).
Family conclusion across veto/replan/redirect: the veto's conservatism IS
the value — holding vetoed ships home beats every measured scheme for
spending them this turn. One-ply reply-exploitation axis closed at the
current modeling depth.

## Pivot: 4P ungating measurement (RUNNING)

4P = 60% of ladder volume; every shipped mechanism is 2P-gated there. Panel:
vetorf4p_ffa (veto + rf active in 4P) vs ffa_uniform control, background
3 × namespaced ffa_uniform, seeds 4 × 4 seats = 16 games per focal.

## 4P panel resolution (2026-06-11 ~11:00 UTC)

Panel raw: vetorf4p 1/16; ffa_uniform "control" 16/16. The control reading
is a SCORING ARTIFACT, not contamination (earlier in-chat contamination
theory retracted): probe C (new-code plain vs 3 namespaced old-code plain,
clean subprocess) played a PERFECT 4-way mirror — 500 steps, 783.0 ships
each, rewards [1,1,1,1]. In a material tie the engine gives every seat
reward 1, and ffa_panel counts reward==1 as first place → 16 mirror
stalemates scored as 16/16. Tools note: ffa_panel first-place metric is
inflated by draws for self-similar agents; and never run two env-gated
bundles as focals in one panel invocation (env leak risk remains real even
though it wasn't the cause here).

Probe C is also the off-path parity proof for today's refactors in 4P:
one divergent decision would have broken the bit-perfect mirror.

REAL finding — veto ungated in 4P is genuinely bad: vetorf4p eliminated
step 198 (probe A), new-code veto_only eliminated step 162 (probe B2),
panel 1/16. Modeling cause: _predict_reply mirrors each of the 3 opponents
independently and MERGES all replies into one background — every attack is
priced as if all three rivals parry it simultaneously (triple-counted
defense) → chronic passivity → first to be carved in FFA. Correct in 2P
(one opponent). The 2P gates were protecting us from exactly this.

Fix direction (next mechanism): 4P reply model — select/weight the merged
reply (nearest/strongest rival only, or 1/(n-1) weighting) before pricing.

Probe B1 (old-code veto_only, 4P): inner python dies silently twice
(empty stdout, no traceback — native-crash signature). Moot for decisions;
not investigating a dead-end bundle.

## Build-day verdicts (2026-06-11 afternoon, fast harnesses + dead-seat guards)

- **4P reply fix (sequential conditioning): CONFIRMED on 16 maps.** Fresh
  12: final share 44.2% (even=25%), mean rank 1.83, rank1 6/12, eliminated
  1/12. The broken merge was eliminated by ~step 200 on essentially every
  map. 4P submission case now live.
- **Coalition rescue (coalitions + deficit floors, zero new code): 7/8 vs
  the live stack, paired +21.9% @250, ahead 4/4 seeds @250.** Late-growing
  margin = defenses holding. Strongest 2P attribution since the reactive
  floor. Confirmation on fresh seeds queued.
- **Opening searcher: ELIMINATED as-built** (3/8, -27% @120; seed-1
  collapse dec=14). Root cause: the scheduler's keep-1-home rule is a
  single-player safety model — it strips sources; a real opponent punishes.
  Backlog: cap scheduler sends by the planner's own hold discipline
  (safe_drain), then re-judge.
- Ledger-branch ports landed: commitment cost (gated), leader-objective 4P
  variant, dead-seat guards in both harnesses, _ref_ledger_v1_2 referee.

## Verdicts (2026-06-11 mid-day): tuner refuted, searcher defect isolated

- **Tuned knobs (rf 1.2 / margin 3.0 / H 16): 0/16 on fresh seeds 0-7** —
  the +0.567 tune objective was 3-seed overfit; the confirmation gate
  worked. Mechanism of failure: hyper-conservatism -> out-expanded ->
  eliminated (the passivity cliff again). Tuner needs more seeds/eval +
  smaller steps. SHIPPED CONSTANTS STAND.
- **Opening v2 hold filter never binds**: games step-identical to v1 (the
  do-nothing projection is threat-blind pre-contact, so safe_drain is
  fully permissive exactly when the searcher launches). True defect =
  missing worst-case reserve (Planet Wars canon: Melis's full-attack
  future). v3 adds PRODUCER_PLUS_OPENING_RESERVE_K (default 8): launch
  only if the source survives the enemy garrison mass reachable within K
  turns (reactive-floor geometry pointed at our own sources).
- **Expansion lane (window 150) raw power confirmed**: 4/8 with +57/+96%
  @250 dominance where it survives; collapses trace to the same searcher
  defect. Re-judge both windows with the reserve.
- Forward redistribution (canon port) leg in flight.

## Verdict: forward redistribution — ELIMINATED (1/8, paired -49% @120)

Healthy @40, collapsing after: ships perpetually in flight between own
planets (per-turn pressure recompute -> chasing, strictly-positive gap
prevents direct backwash but not cycles). The canon's redistribution lives
inside Melis's transfer scoring + worst-case surplus; the naive ungate
buys churn. Converging lesson (with commitment-cost data): in-flight time
is the underpriced cost. Possible future shape: transfers priced through
the flow scorer like attacks, not a separate lane.

## Closing verdict: expansion lane SHELVED (2026-06-11)

Wide panel (fresh seeds 4-9): 6/12, paired +1.8% @250 / -5.5% @120.
Combined 20 games: 10-10. A coin-flip amplifier — converts even games to
routs both directions (collapse seeds trace to chaotic 1-ship step-1
divergences, not a fixable filter; reserve + race margin improved the
WINNING games' margins but never touched the losing mode). Revisit only
with an opponent-aware schedule (the searcher plans single-player; the
greedy's flow-diff at least prices the opponent's static future).
Iterations spent: 4 (hold filter -> non-binding; worst-case reserve ->
binding but not the loss mode; race margin -> ditto; wide panel -> parity).

## Verdict: commitment tax — fails confirmation, shelved (2026-06-11 ~13:00)

Fresh seeds 4-9: 5/12, paired +17.2% @250 driven by three +100% blowouts
vs steady moderate losses; 3/6 seeds ahead. Combined 11-9 over 20 games.
Same variance-amplifier profile as the expansion lane. eps goes to tuner
v2 as a knob. Nothing earns a slot today; both held.

## REFUTATION with mechanism: horizon 22 — 2/12, paired -53% @250

The expansion-undervaluation hypothesis predicted longer horizon helps;
the opposite happened, and the conflation explains it: in this engine the
horizon sets scoring depth AND the launch-reach cap (K_eta = H). H=22
licenses longer fleet commitments — the week's convergent disease
(in-flight capital underpriced). Clean controlled support for the
reach-discipline thread: tuner hinted H=16; top teams fly eta 4-5 vs our
7-8. Next (zero-code): isolated H=16, and the shelved ADAPTIVE_K
(reach 20 -> 10 by step 30; parity on the old stack, untested on
veto+rf). Opening-gap re-measurement on the new sub's 47 live replays
queued alongside.

## Day close-out finding (2026-06-11 ~14:00): the local optimum may be a MIRROR ARTIFACT

Recorded facts: h22 2/12 (-53%), h16 2/12 (-66%), adaptive reach 0/12
(-87%) — every horizon/reach perturbation loses massively to the incumbent
head-to-head. Live opening gap median -6% (we now BEAT the single-source
benchmark; one 59% tail game). Seven falsifications since the morning
submission, all measured VS OUR OWN STACK.

Suspicion for the next frame: self-play attribution has a measurement
moat. Near-identical agents diverge chaotically (observed: exact mirror
draws, seat-split outcomes, 1-ship step-1 butterflies deciding games), so
subtle improvements read as coin flips (lane 10-10, cc 11-9) and any
calibration shift reads as a rout (the whole mechanism stack is co-tuned
at H=18). Only mirror-DOMINATING mechanisms (veto, rf) ever cleared the
gate. Historical support: mass won locally vs champion and regressed
live; ffa_uniform won modestly locally and lifted live. The referee IS
the bottleneck.

## Panel re-judgment (2026-06-11 ~15:00): the panel AGREES with the mirror

Fresh perspective test: re-judge the live stack (control) and both shelved
candidates (expansion lane, commitment tax) against three NON-mirror
referees, seeds 0-3 both seats, 150-step truncation.

| candidate        | vs v7_0          | vs ledger v1.2    | vs old champion   |
|------------------|------------------|-------------------|-------------------|
| live stack (ctl) | 8/8 +88.8%@120   | 8/8 +99.9%@120    | 6/8 +34.4%@120    |
| expansion lane   | 8/8 +92.5%@120   | 8/8 +99.9%@120    | 5/8 +20.0%@120    |
| commitment tax   | 8/8 +80.6%@120   | 8/8 +85.6%@120    | 6/8 +30.0%@120    |

Candidate-minus-control deltas: flat on the two saturated referees,
mildly NEGATIVE on the only discriminating one (lane -14pp@120, cc
-4pp@120 vs control against the champion). Conclusions:

1. The mirror verdicts were honest — neither shelved mechanism holds
   hidden panel value. The week's falsifications stand.
2. Two of three referees SATURATE (everything wins ~100%). The local
   referee pool tops out below our level; panel-delta measurement cannot
   find subtle edges any more than the mirror can. The 1400-1700 ladder
   styles are not reproducible locally.
3. Therefore the escalation: (a) live submissions are the only honest
   instrument against the real field (~60 slots to deadline) — local
   measurement's role shrinks to safety gating + mirror-domination
   detection; (b) next build is the qualitatively new capability with the
   most convergent external evidence: DELAYED-LAUNCH / SYNCHRONIZED
   ARRIVALS (Planet Wars canon "key winning strategy"; unlocks
   outwaiting, banking, timed snipes; everything this week said holding
   ships is undervalued).

## Delayed-launch / synchronized-arrivals build — verdict chain (2026-06-11 evening)

Built PRODUCER_PLUS_SYNC (default OFF): two-source pair candidates on
targets neither source cracks alone, joint floor at the later leg's
arrival tick, nearer leg held in memory and fired on the last turn that
makes the shared arrival date (fresh re-aim). 13 unit tests; in-process
probe confirms holds create/execute/release in real games.

Mirror legs (vs live stack, 6 seeds x 2 seats, 150-step truncation):
1. Holds, full-drain legs:        1/12, -37.8%@120 / -44.4%@250
2. Holds, floor-proportional:     3/12, -34.0%@120 / -45.8%@250
3. Holds + FULL-reaction floor gate (weight 1.0, lag 0 at sync tick):
                                  3/12, -25.2%@120 / -37.1%@250
4. ABLATION SYNC_DMAX=0 (same-tick coalitions only, no holds):
                                  7/12, -0.6%@120 / -0.7%@250  (parity)
5. Coalitions vs old champion:    4/8, +25.9%@120 (control 6/8 +34.4% — flat/slightly down)

Mechanism (seed-4 autopsy): the far leg telegraphs the attack for the
whole hold window; the reply-aware defender reinforces past the pair's
joint size and counterattacks the frozen near source; a canceled hold
leaves the far fleet to die alone (hold frozen at step 60 -> collapse
60->70; our in-flight share 67% vs their 45%). Even the full-reaction
floor gate doesn't save it -> the refutation is conceptual: freezing
capital to synchronize against a reactive equal loses on tempo. The 2010
canon's synchronized attacks worked against non-reply-aware bots; ours
punishes them.

Also caught: _overkill_for_targets returns a plain float in the legacy
path; the first lo-sizing draft crashed EVERY turn from step 0 and the
kaggle env swallowed the exception (silent forfeit-by-passivity, games
"played" to step 75-200). sync_probe now surfaces agent crashes. Lesson
for all future in-process measurement: a flipped game outcome with zero
mechanism activity = suspect a swallowed exception first.

Disposition: vetorf_sync variant = same-tick coalitions only (DMAX
default 0), mirror-parity, champion-flat — NOT submit-worthy on current
evidence. Holds opt-in behind SYNC_DMAX>0 for future redemption (e.g.
against non-reply-aware ladder opponents — but that is exactly what we
cannot measure locally).

## Terminal production value — confirmed diagnosis, refuted flat fix (2026-06-11 late)

Decision trace on the Gregor Lied live loss (scripts/decision_trace.py,
steps 16/22/27): all six planets drainable (176 ships), 13 targets
shortlisted, EVERY capture candidate scores +0.0 against threshold +1.5 —
three consecutive turns of zero launches. Root cause confirmed: in-horizon
flow truncates production payoffs at H=18 (the +5 neutral at eta 18 scores
literally 0). Live-loss mining agrees: wins are production-ahead @40 in
16/17, losses behind in 9/17 (median -8 @70); in-flight share is NOT
discriminative (59% in both).

Flat-lambda fix verdicts (TERMINAL_PROD_VALUE=12 on live stack):
- vs mirror:        2/12, -38%@120 (deficit opens by step 40)
- neutral-only:     0/12, -99.8%@250 (pure expansion credit = maximal rout)
- vs old champion:  4/8, +13%@120/+20%@250 — BELOW control (6/8/+34/+51)
Both instrument classes agree: lambda=12 overshoots into unsafe expansion;
the banker punishes the invested capital faster than it pays back. NOT a
mirror artifact.

Next iteration (not built): holding-time-priced credit — terminal value
per target = production x expected HOLDING time given the opponent's
feasible retake (the ledger branch's capture pricing), instead of a flat
constant. The paralysis defect is real; the fix must price counter-safety.
