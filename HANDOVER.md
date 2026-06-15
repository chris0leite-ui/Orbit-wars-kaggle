# HANDOVER.md — next-session brief

> Refreshed **2026-06-15** (end of the loss-mining session). Supersedes the
> 2026-06-14 version. ~8 days to the 06-23 deadline. Full session record:
> `knowledge-base/thoughts/2026-06-15-loss-mining-grounded-fixes.md`.
>
> **Update 2026-06-15 (positional-game session, branch
> `claude/practical-hamilton-oth767`):** built two framework terms on
> producer_plus, both gated default-OFF, byte-identical when off, unit-tested,
> NOT submitted: **(1) frontier/gateway value** (reach factor) — reproduction
> showed it's mismatched to corner-neglect (a candidate-*generation* truncation;
> frontier is *scoring*), only useful composed with the `expand`/wideshortlist
> generation fix (accelerates capture 499→95 on seed 641308308); **(2)
> tenure/durability** (`seq_strength_tenure`) — discounts captures we can't hold,
> targets the confirmed collapse driver. **Key meta-finding:** both drivers
> (corner-neglect, collapse) manifest only vs top opponents (~1600+) we can't run
> locally → local verification is exhausted, the ladder is the only judge. See
> `audit/2026-06-15-frontier-gateway-value-spec.md`,
> `audit/2026-06-15-tenure-durability-spec.md`, and the
> `knowledge-base/thoughts/2026-06-15-*` entries. **Architecture question
> settled:** built a clean-room positional `Φ` agent (`agents/phi/`) as an
> instrument; weight sweep says economy+tempo win, options/reach + caution hurt
> (so frontier+tenure are the wrong levers). Our **old baseline IS a native `Φ`
> agent but loses 0–4 to producer** (clean balanced-seat test) — don't revive
> it. **Producer is the substrate; the one on-thesis untested lever is
> `hold_value`** (gated post-horizon economy). Refuted/parked: frontier, tenure,
> baseline revival, clean-`Φ` rewrite.

## State of play

- **Our line is `producer_plus`** (heuristic; ~70 `PRODUCER_PLUS_*` env-flag
  behaviours on the `orbit_lite` engine). The "1280" agent = the
  **`vetorf4p_seq_strength`** flag set. Field context: top of the leaderboard
  is ~1600–1780; we're ~1220–1280 (drifted down as the field strengthened —
  not a regression of our agent). Prize zone is ~300–500 μ up: a stretch.
- **This session brought the producer_plus lineage ONTO this branch**
  (`claude/festive-knuth-roggck`, commit `1e2e747`): `agents/producer_plus/`,
  the matching `agents/producer/orbit_lite/` (16 modules), and
  `scripts/bundle_producer_plus.py`. The real agent **runs here now** — no
  worktree needed. (Native home is still `claude/awesome-clarke-ixy57v`; see
  the consolidation flag below.)
- ⚠️ **`agents/producer/main.py` on this branch is the BARE engine (0 flags) —
  NOT the 1280 agent.** Any A/B against it is the wrong base. Use
  `agents/producer_plus/` with the seq_strength flags.

## The method that worked (use it — it beat everything else)

**Replay-mine → reproduce → diagnose the specific bug → find the fix → ladder A/B.**
The PI's eyeball + the 46 real loss replays produced every win this session;
the systematic/speculative routes (search wrapper, distillation, blind
flag-guessing) all failed. Loop:
1. Mine `audit/live-episodes/53564198/episode-*-replay.json` (45 real losses;
   `info.TeamNames` = us, `info.seed` = reproduction seed, `rewards` = loser).
2. Reproduce on the seed in the **producer_plus mirror** (two
   `ProducerLiteRuntime` instances); find the specific bad behaviour.
3. Sweep flags to find what fixes it; verify on-seed.
4. Add a `seq_strength_<name>` variant to `bundle_producer_plus.py`, bundle,
   Rule-46 smoke, fire to ladder.

## Grounded findings — the loss landscape (46 real losses)

1. **Under-expansion = #1 driver (~76% of losses).** We trail on planet count
   by **median step 30**; hold **5–6 planets vs winners' 8 by step 60**. Far
   high-value planets fall outside the nearest-K neutral shortlist, so they're
   never even candidates (the PI's corner-neglect observation generalised).
   **Verified fix:** `NEUTRAL_SHORTLIST=20` **+ a deeper horizon** lifts
   planets@60 from 6 → 8 (= winner rate). Shortlist surfaces far targets,
   horizon makes us value them — **either alone is only partial.**
2. **Collapse = long-game cluster (~12/15 long losses).** We build a lead
   (peak 6–10 planets) then lose planets after the peak; 6 losses were
   even/ahead at step 60 then lost everything. Mechanism murkier: 2P looks
   like being out-played by stronger opponents; 4P like lose-a-lead gang-up.
   **Not cleanly flag-fixable.** Reserve: `GARRISON_VALUE_FROM_STEP`
   (late-game-only defence, avoids the global over-defence penalty that sank
   `garval`). Tension: this cautions against *over*-expanding — watch the
   expansion fix's long-game results.

## On the ladder NOW (read ~2026-06-16)

A/Bs are **SERIAL** — only the rolling-2 (newest two) actively ladder; evicted
subs **freeze** (confirmed: garval froze the instant it was evicted). ~7 rounds
left. Run as **king-of-the-hill**: best holds one slot, one challenger takes the
other.

Current rolling pair (both grounded fixes):
| sub | variant | flags added to seq_strength |
|---|---|---|
| `53714433` | `seq_strength_expand` | `NEUTRAL_SHORTLIST=20` + `HORIZON_2P=30` + `HORIZON_4P=18` |
| `53711823` | `seq_strength_wideshortlist` | `NEUTRAL_SHORTLIST=20` |

**Read:** `kaggle competitions submissions orbit-wars`. Warm-up at ~4 h was
noisy (both ~1130, below the ~1220 field) — **don't conclude before ~24 h**
(Rule 12). Decision tree:
- `expand` clearly > field → expansion direction works → push **h45** (the
  upside, 8 planets) or sweep more expansion; next challenger takes the other slot.
- both ≈ field → diagnoses right, fixes don't convert → back to mining (do the
  **4P dissection** — worst format at 46%).
- `expand` < `wide-shortlist` → the horizon is hurting (over-extension/collapse
  tension) → drop the horizon, keep shortlist.

## Queued (built, smoke-passed, in `bundle_producer_plus.py`)
- `seq_strength_fc` (force-concentration), `seq_strength_denial` — speculative,
  deprioritised behind the grounded fixes.
- Next grounded candidate: dissect the 4P losses; the collapse defence flag.

## Dead-end map — DON'T re-walk (all failed, locally AND on the ladder)
- Search wrapper over producer (fast_sim rollouts): tied **28% / 25%** (weak &
  strong opponent models). Search adds nothing.
- Deeper planning on the bare engine: hurt. Distillation / hand-condensed fast
  policy: failed — a strong policy's strength **is** its expensive forward sim;
  no cheap copy preserves it.
- Learning agents on the real ladder: `oracle_rw` (IL) **1018**, `rl_v7` (RL)
  **938** — far below producer 1280. Models/compute don't beat the heuristic here.
- Defensive direction (`garval`): ladder **1181 < 1280**.
- The **~12× `lib/world_model` ledger speedup** (ring-crossing reframe, commit
  `262a4e3`) is real + exact but helps **lib tools / the v3.x lineage, NOT
  producer_plus** (separate `orbit_lite`). Banked, not a champion lever.

## Submissions / kept pair
- Budget **5/day**. Rolling-2 = final-eval pair. Per PI: **ladder is for A/B
  now; curate the best-2 as the kept pair in the last ~2 days.** Don't hoard.
- Bundle+fire: add `seq_strength_<name>` to `bundle_producer_plus.py` VARIANTS →
  `python scripts/bundle_producer_plus.py --variant <name> --out /tmp/x.py` →
  Rule-46 smoke (full-game max < 1000 ms) → `kaggle competitions submit -c
  orbit-wars -f /tmp/x.py -m "..."` → log the Rule-42 claim in
  `state/MULTI_BRANCH.md`.

## Environment
- producer_plus needs **`torch`** (CPU). `cffi` in `requirements.txt` (simulator
  import). Confirm both in the bootstrap.

## Open questions / flags
- **Q (ladder, ~24 h):** do `expand` / `wide-shortlist` actually beat the field?
- **FLAG (latency):** HORIZON in 2P spikes (~880 ms one-off warmup fluke;
  midgame ~464 ms). Chose `h30` over `h45` for safety. Watch for ladder timeouts
  (= losses); drop horizon if any appear.
- **FLAG (branch):** producer_plus now lives on both `festive-knuth` (here) and
  `awesome-clarke` (native). Consolidate to avoid drift.
- **FLAG (ceiling):** 1280 → 1600+ is a large gap; grounded fixes may close some
  but top-5 in 8 days is a stretch — set expectations.

## Pointers
- `knowledge-base/thoughts/2026-06-15-loss-mining-grounded-fixes.md` — this session.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
- `state/STRATEGY.md` — updated this session to the producer_plus + loss-driven line.
- `CLAUDE.md` — process rules.
