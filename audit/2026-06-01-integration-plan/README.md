# Integration plan — slot-reservation + champion-strategy-rules-00JzI

**Goal:** combine three orthogonal env-var-gated fixes into a single
submission to beat the current best evidence (00JzI's joint_sync at
μ=1156, which we evicted).

## Where we are (2026-06-01 07:35 UTC)

### Live ladder

| Sub ID | Agent | μ | Branch |
|---|---|---:|---|
| 53243763 | baseline_pv_eta_vh_dist_slotres (PENDING→815.8) | **815.8** | hqNVM (ours) — position 1 |
| 53239342 | baseline_pv_eta_vh_dist_composite | 537.7 | hqNVM (ours) — position 2 |
| 53227546 | baseline_pv_eta_vh_dist | 801.1 (evicted) | hqNVM (ours) |
| 53223160 | baseline_joint_sync_submit | ~1147 (evicted) | 00JzI |
| 53212044 | baseline_pv_eta_vh_b3smoke | ~1142 (evicted) | hqNVM (ours) |
| 53182323 | baseline_launch_rules_universal | 1183.7 (evicted) | live champion |

**Slotres at μ=815.8 is +14 over evicted dist** — the slot-reservation
mechanism works directionally but the lift is small. The proven peak on
either branch is joint_sync at μ~1147-1156. Integration target is to
get back above that floor.

### Branch surface

| Branch | Most recent feature | Status | Lines vs main |
|---|---|---|---:|
| **hqNVM** (ours) | `BASELINE_SLOT_RESERVATION=3/2/2` in chooser_trajectory.py | live μ=815.8 (sub 53243763) | chooser_trajectory.py +70 |
| **00JzI** | `BASELINE_JOINT_SYNC=1` + `BASELINE_SIZE_BALANCE=1` | joint_sync live μ=1147 (evicted); size_balance n=16 75% local (not submitted) | chooser_trajectory.py +426; proposer.py +287 |

## The three fixes are orthogonal

| Bottleneck addressed | Fix | Branch | Surface |
|---|---|---|---|
| Wrong ship counts per candidate | `BASELINE_SIZE_BALANCE=1` (SUPPRESS+UPSIZE+CLAMP) | 00JzI | proposer.py |
| Missing multi-source synchronized attacks | `BASELINE_JOINT_SYNC=1` (+ MAX_PAIRS, SRC_K, HOLD) | 00JzI | chooser_trajectory.py post-loop |
| Wallclock monopolised by defenses | `BASELINE_SLOT_RESERVATION=3/2/2` | hqNVM | chooser_trajectory.py pre-loop |

Each can be enabled independently. The fixes operate at different stages
of the pipeline — proposer (size_balance) → prerank assembly →
chooser pre-loop (slot reservation) → main loop scoring → chooser
post-loop (joint sync). No semantic overlap.

## Integration approach

**Use 00JzI as base** (more live evidence, more code volume). Cherry-pick
our slot-reservation commit on top. Conflict surface is narrow — our
patch sits in front of the main scoring loop; their joint-sync generator
sits after it.

### Step-by-step

1. **Start a fresh integration branch from 00JzI HEAD:**
   ```bash
   git fetch origin
   git checkout -b claude/integration-slotres-jointsync-XYZ \
     origin/claude/champion-strategy-rules-00JzI
   ```

2. **Cherry-pick the slot-reservation commit from hqNVM:**
   ```bash
   git cherry-pick 097b474
   ```
   Expected conflicts in `agents/baseline/chooser_trajectory.py` only.
   The conflict is in the env-var read block (around line 257 — both
   branches added env-var reads in the same neighborhood) and around
   line 894 (the main scoring loop — they likely added code nearby).
   Resolve by keeping BOTH blocks; the patches don't conflict semantically.

3. **Cherry-pick our probe script too:**
   ```bash
   git cherry-pick 6549eeb
   ```
   `scripts/probe_candidate_distribution.py` is new in our branch — no
   conflicts.

4. **Verify default-OFF parity** (Rule 46):
   ```bash
   pytest tests/test_bundle.py -q
   ```
   Both branches' patches default to OFF; with no env vars set, behavior
   must be byte-identical to 00JzI's base. **If parity breaks, that's
   the first thing to debug.**

5. **Re-run our probe with slot reservation alone**, on the integration
   branch's code:
   ```bash
   BASELINE_SLOT_RESERVATION=3/2/2 \
   python scripts/probe_candidate_distribution.py \
     audit/live-episodes/53239342/episode-78367540-replay.json \
     --step 100 --seat 0 --vh 1.0
   ```
   Confirm slot reservation still surfaces attack/expansion candidates
   on the merged codebase.

6. **Create the combined wrapper** at
   `agents/baseline_pv_eta_vh_dist_jsr/main.py` ("jsr" = joint-sync +
   size-balance + slot-reservation). Env vars set:
   ```python
   _os.environ.setdefault("BASELINE_PV_ETA", "1")
   _os.environ.setdefault("BASELINE_OPP_TIER", "2")
   _os.environ.setdefault("BASELINE_OPP_FILTER_THRESHOLD", "0.15")
   _os.environ.setdefault("BASELINE_VH_LAMBDA", "1.0")
   _os.environ.setdefault("BASELINE_SLOT_RESERVATION", "3/2/2")
   _os.environ.setdefault("BASELINE_JOINT_SYNC", "1")
   _os.environ.setdefault("BASELINE_JOINT_SYNC_MAX_PAIRS", "30")
   _os.environ.setdefault("BASELINE_JOINT_SYNC_SRC_K", "3")
   _os.environ.setdefault("BASELINE_JOINT_SYNC_HOLD", "1")
   _os.environ.setdefault("BASELINE_SIZE_BALANCE", "1")
   _os.environ["KINEMATIC_TABLE_ENABLED"] = "0"
   # Standard peak-orbitfix preamble (joint aggr, neutral bonus, etc).
   ```
   Pattern source: `agents/baseline_pv_eta_vh_dist_slotres/main.py`.

7. **Create a bundler** at `scripts/bundle_pv_eta_vh_dist_jsr.py` —
   clone of `scripts/bundle_pv_eta_vh_dist_slotres.py` with the wrapper
   path swapped. Both blobs (`_OPP_BOOSTER_B64` from
   `data/opp_distill/distill_booster.txt` + `_VH_MODEL_B64` from
   `data/value_head/value_head_model.txt`) still apply.

8. **Build + Rule-46 gates:**
   ```bash
   python scripts/bundle_pv_eta_vh_dist_jsr.py
   pytest tests/test_bundle.py -q
   python fast.py play submissions/baseline_pv_eta_vh_dist_jsr.py
   ```
   Wallclock must stay ≤ 1000 ms. Slotres smoke ran max=860 ms; adding
   joint_sync expands candidate count slightly so monitor closely. If
   max > 1000 ms, drop slot reservation from 3/2/2 to 2/2/2 (5 slots).

9. **Action distribution check** (1-game classification harness used
   in this session):
   - Attack% should stay ≥ 20% (slot res working)
   - Joint-sync emits should appear (multiple sources hitting same
     target same turn — visible as identical eta + close angles)
   - Defense% should drop further toward ~30-40%

10. **Submit decision (Rule 42):**
    - Eviction target: whatever is in position 2 at the time. If
      composite μ=537.7 is still there, safe eviction. If slotres
      μ=815.8 is in position 2, we'd evict our own lift.
    - PI sign-off required.
    - Push-claim board entry in `state/MULTI_BRANCH.md` before submit.

## Risks & open questions

| Risk | Mitigation / open |
|---|---|
| 00JzI lands a new commit on their branch tomorrow, our integration becomes stale | Pin to the SHA we branch from. If they ship something promising afterward, re-rebase. |
| Cherry-pick conflicts in chooser_trajectory.py are bigger than expected (their 426 LOC includes many `os.environ.get` reads near where we added ours) | Resolve by accepting both blocks; the patches are semantically independent. If structurally tangled, manually re-apply our 70 LOC block on top. |
| Their `SIZE_BALANCE` changes proposer's `cheap_marginal_value` outputs in a way that violates our slot-reservation's assumption about per-class cheap_delta distributions | Re-run the probe step 5 to verify slot res still surfaces attack candidates. If size_balance makes attacks have HIGHER cheap_delta (which is the whole point of size_balance), our slot reservation becomes redundant for attacks — could disable it for attacks (`3/2/2` → `0/2/2`) and rely on size_balance. |
| `BASELINE_JOINT_SYNC` candidates target enemy planets (good for slot res's "attack" class) BUT joint-sync entries are added to prerank by a different code path that bypasses our slot reservation | Verify by reading their joint-sync generator (chooser_trajectory.py around `BASELINE_JOINT_SYNC` block). If joint-sync candidates are added to `scored` directly rather than going through `prerank`, our slot reservation never sees them — which is fine, they're already added. |
| Live μ < 815.8 (worse than slot-res alone) | Falsifies the compounding hypothesis. Pivot: pick the single strongest individual feature (likely joint_sync alone) and submit that. |
| Live μ > 1000 but < 1147 | Partial lift; reasonable. Frees us to iterate further. |
| Live μ > 1147 | Strongest agent on either branch. Becomes the new floor. |

## Verification list (must pass before submit)

- [ ] `pytest tests/test_bundle.py -q` — 10/10 default-off parity
- [ ] `BASELINE_SLOT_RESERVATION=3/2/2 python scripts/probe_candidate_distribution.py ...` — attack candidate scored with positive leaf_delta on the merged code (≥1 attack with leaf_delta>0 at the test state)
- [ ] `python fast.py play submissions/baseline_pv_eta_vh_dist_jsr.py` — full game, max turn ≤ 1000 ms
- [ ] Action distribution on the smoke replay: attack% ≥ 20%, defense% ≤ 50%
- [ ] `state/MULTI_BRANCH.md` push-claim row filled in
- [ ] PI explicit "submit"

## What we DON'T do this session

- Modify any `lib/` files. Their branch already removed `kinematic_table.py` (which we hard-disabled in `lib/trajectory.py`); confirm `KINEMATIC_TABLE_ENABLED=0` is still wired correctly post-integration.
- Modify `agents/baseline/main.py`, `value.py`, `proposer.py` beyond what cherry-pick brings in. Treat 00JzI's versions as canonical.
- Retrain any model. Both blobs (opp booster + B.3 head) are reused as-is.
- Run a multi-game local A/B. The previous heavy-vs-heavy timing (60 min n=32 timeout) is still a constraint. Direction-of-effect comes from the probe and the action-distribution single-game check.

## Pointers

- `agents/baseline/chooser_trajectory.py:_parse_slot_reservation` — our slot-reservation env-var parser (lives at line ~260)
- `agents/baseline/chooser_trajectory.py:_classify_target` — our target classifier helper
- `agents/baseline_pv_eta_vh_dist_slotres/main.py` — wrapper-agent pattern to clone
- `scripts/bundle_pv_eta_vh_dist_slotres.py` — bundler pattern to clone
- `scripts/probe_candidate_distribution.py` — single-state probe used to verify slot-reservation behavior
- `audit/live-episodes/53239342/episode-78367540-replay.json` step 100 — the canonical test state where bare composite returned 0 attacks, slot res returned 2 attacks
