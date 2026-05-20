# Large→small launch leak — confound-controlled re-audit (LEAK REJECTED)

**Branch:** `claude/audit-workflow-performance-btjeK`
**Plan:** `/root/.claude/plans/let-s-figure-out-how-purrfect-mist.md`
**ISSUES leaf:** A.8 (now `null` — see verdict)
**Inputs:** 42 4P live games of submission 52827111, 3,688 launches by
us, target reconstructed via `lib.trajectory.predict_fleet_fate`.

## Context

The v1 audit (`scripts/large_to_small_audit.py`, commit `2c24d5e`)
reported large(prod≥4)→small(prod≤2) launches as **net −0.139 production
per launch** vs **+0.191** for the reverse direction, suggesting the
agent systematically exposes high-production planets to capture low-value
targets. PI flagged two confounds:

1. **Selection bias** — high-production planets produce more ships, so
   they launch more (both in count and in ships-per-launch). Per-launch
   NET inflates a per-source bias into a per-launch leak signal.
2. **End-state attribution bias** — "src lost by end-of-game" in a lost
   episode is tautological (all our planets flip), and the v1 last-launch
   debit attributes the entire loss to whichever launch happened to be
   most recent.

## Confound controls applied (v2)

`scripts/large_to_small_audit_v2.py`:

1. **Per-ship NET** (`NET_short_per_ship = (prod_gain_landing −
   prod_loss_short) / sum(ships_deployed)`) replaces per-launch NET as
   the primary metric. Selection bias resolved.
2. **Short-window src-loss attribution** (`src_lost_within_20_pre_relaunch`):
   src owner ≠ our_seat anywhere in `(landing_step, min(landing_step+20,
   next_launch_step_from_src, n_steps−1)]`. Each launch attributed
   independently — no last-launch tautology.
3. **Landing-time outcome** (`tgt_owned_at_landing`): owner at
   `landing_step+1`. Separates tactical success from strategic outcome.
4. **Episode-window stratification**: early (t≤150) / mid (151..350) /
   late (>350). Verdict hinges on early+mid only — late is end-state-
   biased by construction.
5. Tier convention canonicalised: `small = prod ∈ {1, 2}`, `mid = 3`,
   `large ≥ 4`. (Fixes a v1 boolean bug that bucketed prod=2 as `large`.)

## Result

### NET_short_per_ship (PRIMARY METRIC) by window

| Direction | Early | Mid | All 4P |
|---|---:|---:|---:|
| **large→small** | **+0.0085** | **+0.0033** | **+0.007** |
| **small→large** | **+0.0389** | **+0.0176** | **+0.029** |
| delta | +0.0305 | +0.0142 | +0.022 |

Both directions are **positive per ship in both early and mid windows**.
The leak hypothesis predicted large→small < 0 in early+mid — it isn't.

### What the v1 signal really was

The v1 per-launch numbers reproduce exactly (`NETlnch` column matches),
so the bug isn't in the data extraction. The per-launch numerator was
end-of-game attribution + last-launch debit, both of which are uniformly
zero in won episodes and uniformly bad in lost episodes — i.e. it
measured "are we winning the episode" via a launch-tier proxy, not the
quality of any individual launch.

Concrete check from v1: in won episodes, the large→small NET was **+0.7
production** total across all 256 launches (because src_lost_end was 0
in every won bucket). In lost episodes, the same launch class was **−91
production** across 121 launches (because everyone of our sources eventually
flipped). The v1 metric was reading the end state, not the launch's
contribution.

### Residual signal that IS real

Launches up the production gradient (small/mid → large) still pay 3–5×
more per ship than launches down (large → small/mid):
- small→large: +0.029 production / ship
- large→small: +0.007 production / ship

This is plausible without being a leak: large planets are central and
high-value, so attacks INTO them pay more; large planets are also the
natural attackers because they have surplus ships, so attacks FROM them
go wherever ships are needed (often periphery, which is small). The
asymmetry doesn't imply the agent is making bad src-selection decisions
— it could equally be that large→small launches are accepting a smaller
absolute return because they're the only launches *available* given
geometry.

### Short-window src-loss rate by src tier

| src tier | src_loss_short_rate (all 4P) |
|---|---:|
| small(1-2) | 18–22% |
| mid(3) | 10–14% |
| large(4+) | 8–11% |

Large sources have the **lowest** short-window loss rate, not the
highest. The "we expose large planets" framing isn't supported — large
sources have more residual defense ships after launches AND benefit from
the existing `_source_survives_launch` and `_target_holdable_after_capture`
filters that protect them.

## Verdict and disposition

- **A.8 leaf → `null`** (LEAK REJECTED). v1 signal was selection + end-state
  bias.
- **Phase B (opp model cheap-capture bonus in `lib/opp_model.py`) is NOT
  implemented.** The plan's strict gate held.
- **Process learning:** the v1 audit was published with per-launch NET
  as the headline. Per-launch denominators are unsafe when the actor
  whose actions are being counted has variable launch frequency by the
  trait being analysed. Candidate Rule 41 (per-attempt vs per-resource
  normalisation): when comparing decision-classes that differ in their
  natural frequency, normalise by the underlying resource (ships, time,
  ops budget) not the count of decisions.

## Artifacts

- `scripts/large_to_small_audit_v2.py` — v2 audit tool
- `audit/2026-05-21-large-to-small-v2.jsonl` — 3,688 per-launch rows
- this file
- v1 reference (unchanged): `scripts/large_to_small_audit.py`,
  `audit/2026-05-21-large-to-small-launches.jsonl`
