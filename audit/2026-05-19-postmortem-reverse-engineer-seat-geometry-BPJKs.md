# Postmortem — 2026-05-19 reverse-engineer-seat-geometry-BPJKs

## What went wrong

- **No bad decisions to retake given priors at decision-time.**
  The priors+ROI implementation was clean: 29 unit tests, byte-
  identity invariant at λ=0 / ROI off, baseline smoke green at
  active defaults (3, 2, ROI on). The A/B falsification was
  decisive (5/32 vs latest submission, Wilson [0.069, 0.318]) and
  surfaced real signal (priors win 2/2 on low-prod-static boards
  across every sweep variant).
- **One mild decision-cost flag:** jumped into priors+ROI work on
  the BPJKs branch base without first reconciling HANDOVER.md
  against the actual current Kaggle state. Rule 32 (`git fetch`)
  ran but didn't expose how stale HANDOVER was — bootstrap reported
  "ahead 2 / behind 1" while origin/main had ~20 commits of
  architectural progress (trajectory chooser, value heads,
  bug fixes #3/#4/#12/#14) merged into a single "behind 1" merge
  commit. Cost: ~30 min planning h2h vs v15 (the wrong opponent)
  before PI intervened.
- **No rule-bypass failures.** Bundle parity gate was skipped
  intentionally and justified (would compare against the priors+ROI
  tree, not the bundled commit's tree).

## Frictions logged this session

Cross-links to `audit/friction.md::2026-05-19`:

- `handover-staleness-vs-kaggle-state` — bootstrap didn't reconcile
  HANDOVER against `kaggle competitions submissions`; planned
  h2h against wrong champion for 30 min before PI override.
- `bundle-filename-collides-with-focal-name` — `fast.py eval --vs
  /tmp/.../baseline.py` skipped 0 games because `Path.stem` matched
  the focal registry name "baseline." ~9 min wasted.
- `same-config-runs-diverge-9pp` — denom_floor=1.0 returned 15.6%
  in the original A/B and 6.2% in the sweep. Caps single-32-seed
  confidence.
- `max-seeds-overridden-by-geometry-panel` — sweep ran 32 seeds
  per variant when `--max-seeds 16` was passed; ~24 min extra
  compute. Documentation gap, not a blocker.

## Promotion candidates (PI ratified: YES — all 4 promoted 2026-05-19)

### [ ] [CROSS-CUTTING] Session-start `kaggle competitions submissions` reconciliation when HANDOVER claims a champion

`tag: handover-staleness-vs-kaggle-state` (2026-05-19,
claude/reverse-engineer-seat-geometry-BPJKs).

Third recurrence of the "framework didn't catch that the baseline
truth wasn't being verified" pattern. Prior recurrences:
- `wrong-file-recon-skipped-state-md` (2026-05-18)
- `agent-introspection-skipped-bootstrap` (2026-05-13)

Each cost ≥30 min planning waste + one PI override.

**Where to insert:** `bootstrap.sh` or session-start hook output;
also reference in CLAUDE.md Rule 32.

**What to add:** When `HANDOVER.md` is >24h old OR the current
branch is "behind N" origin/main with N≥3, run `kaggle competitions
submissions <comp> | head -10` and require the agent to reconcile
the latest submission's commit/description against HANDOVER's
"current champion" framing BEFORE forming any h2h plan. Display
the most recent 2-3 submissions in the bootstrap summary so the
gap is impossible to miss.

**Why:** Rule 32 catches code-tree freshness; this gap catches
narrative-freshness against the comp's ground truth. HANDOVER is
the load-bearing strategic document; if it's wrong about the
champion, every downstream plan is wrong.

### [ ] [CODE-COMP-DISCOVERED] Bundle-vs-focal name collision in `fast.py eval --vs <path>`

`tag: bundle-filename-collides-with-focal-name` (2026-05-19).

`scripts/bundle_agent.py` defaults to `submissions/<agent_dir>.py`
which collides with `fast.py:152::resolve_agent_spec` returning
`Path.stem` for arbitrary file paths. When the focal agent is
"baseline" and we A/B against a bundle of `agents/baseline/` at
a different commit, `fast.py` silently skips all games via the
`same agent as focal` short-circuit.

**Where to insert:** `fast.py:152` `resolve_agent_spec` — when
the path's stem equals the focal name, suffix it (e.g.,
`<stem>@<short-hash-of-path>`). Alternatively, `scripts/
bundle_agent.py` accepts a `--name` override and the A/B
convention uses it.

**Why:** Single-occurrence this session (~9 min cost), but the
failure mode is silent (no error; just 0 games and a one-line
SKIP). Anyone repeating the A/B-vs-bundle pattern will hit it.

### [ ] [CODE-COMP-DISCOVERED] A=A control variance characterisation in `fast.py eval`

`tag: same-config-runs-diverge-9pp` (2026-05-19).

Same config at `BASELINE_ROI_DENOM_FLOOR=1.0` produced 15.6% in
one 32-seed run and 6.2% in a back-to-back 32-seed run vs the
same opponent on the same geometry panel.

**Where to insert:** new test or `fast.py eval --self-control`
flag that runs A=A on a known-stable pair and reports the empirical
variance band.

**Why:** Low-priority single occurrence; doesn't block. But the
9-point swing on identical config means any single 32-seed A/B
result has ≥±5pp implicit noise. Wilson intervals understate
this. Important to characterise so we don't over-interpret
borderline gates (Wlo near 0.55).

### [ ] [DOCS] Clarify `--max-seeds` vs `--geometry-panel` interaction

`tag: max-seeds-overridden-by-geometry-panel` (2026-05-19).

`fast.py eval --geometry-panel --max-seeds 16` ran 32 seeds per
variant (per the panel's 32-archetype cell count?), not 16. Help
text says "Auto-bumps --max-seeds to 128 if it's still the
default" but doesn't describe the floor when the user explicitly
passes a smaller value.

**Where to insert:** `fast.py` argparse help text for `--max-seeds`
or `--geometry-panel`.

**Why:** Documentation fix; single occurrence; the actual
behaviour (more seeds = tighter Wilson) wasn't harmful, just
budget-misaligned.

## PI additions (from step 4)

- "promote as suggested" — PI ratified all 4 promotion candidates
  for inclusion in `.claude/skills/kaggle-comp/improvements.md`.
  Done in same commit.

## Framework version at session-end

- Commit SHA: `4b60e65` (`roi: v3_snipe-style additive cost
  denominator on cheap_marginal_value`)
- Branch: `claude/reverse-engineer-seat-geometry-BPJKs`
- Active CLAUDE.md rules: 1..40 (per `## Operating rules — concise`
  in CLAUDE.md, 2026-05-16 last addition: Rule 40 modeling-
  correctness over restriction-tuning).
- Loaded skills this session: `postmortem`, `kaggle-comp` (via
  reference). No `loop` / `review` / `security-review` invocations.
- Compute spent this session: ~12 min A/B + ~48 min denom_floor
  sweep + ~3 min full smoke test + ~20s unit test sweeps. ~63 min
  CPU total.
- Submissions used this session: 0 (Rule 1 PI sign-off required;
  not pursued — A/B failed the gate).

---

# Session B — community-engagement (Kaggle discussion + dataset)

## What went wrong

**No bad decisions given priors at decision-time.** The Kaggle post
+ dataset shipped clean. Self-audits in response to PI prompts
caught two leaks before publication.

**Two process-level near-misses — same pattern:**

1. **Draft of the post embedded the v7_0 internal regression cluster
   as an "example."** The "Aggregate per archetype" tip read
   *"a new version that wins 80% in aggregate can still lose every
   game on, say, `med_low_prod__mixed_*` boards."* Both the 80% and
   the cell name were pulled verbatim from `audit/2026-05-18-seed-
   panel.md` lines 141 / 152. Caught + abstracted in commit `7c40e68`
   after PI asked "does anything leak from our strategy?"

2. **Initial Kaggle Dataset description included our internal hosting
   motivation.** First draft: *"Hosted here so it can be embedded in
   a Kaggle discussion post without exposing a private GitHub."* PI
   caught: "describe what it is, not what it means for us." Rewrote
   as a standalone artifact description.

Same shape both times: **public-facing draft inherited internal
framing/findings from source material; only caught when PI prompted
a leak-pass.** PI shouldn't be the leak-discovery pass.

## Frictions logged this session

Cross-links to `audit/friction.md::2026-05-19`:

- `public-artifact-internal-framing-leaks-through` — initial Kaggle
  post + dataset description both contained internal-source framing;
  PI prompted the leak-pass instead of finding a pre-vetted draft.
- `kaggle-cli-discussions-read-only` — verified empirically: no
  `create_topic` / `post_topic` in CLI or Python API. Posting is
  web-UI-only. Settled-once fact.
- `pre-draft-duplicate-check-missed` — went straight to drafting
  the post without first scanning existing forum topics; PI
  prompted the check, which then ran cleanly (no duplicates found
  across all 158 topics).
- `senduserfile-account-session-invalid-error` — `SendUserFile`
  delivery failed on PI's side with `account_session_invalid`
  twice. Env-side auth issue (claude.ai session, not the harness),
  but I should have proactively offered the GitHub-URL fallback
  the first time it failed rather than re-trying SendUserFile.

## Promotion candidates (PI ratified: NO — recorded only)

Presented 4 candidates; PI replied "no" to "anything to add /
promote." Candidates stay here as data points; not pushed to
`improvements.md`.

1. **[WORKFLOW] Public-artifact leak-pass before handing to PI.**
   Tag: `public-artifact-internal-framing-leaks-through`. 2× this
   session; pattern is "draft inherits internal framing from
   source material." Concrete check: grep draft against
   `v[0-9]+_*`, strategy terms (`snipe|ROI|priority.prior|
   marginal`), internal paths (`lib/|scripts/|fast\.py`),
   audit-hot-list archetype names — abstract or remove any hit.
   Cost-evidence: 3-turn back-and-forth × 2 artifacts; would have
   been a real leak if PI hadn't been on the wheel.

2. **[SETTLED-FACT] Kaggle CLI has no discussion-create endpoint.**
   Tag: `kaggle-cli-discussions-read-only`. Would belong in
   `comp-context.md` per Rule 8. Saves future agents from
   re-investigating the same dead end.

3. **[WORKFLOW] Kaggle Dataset is the canonical no-leak host for
   forum images.** Tag: `forum-image-via-kaggle-dataset`. Recipe:
   stage image + `dataset-metadata.json` (CC0 license, factual
   description with no internal framing) in a temp folder; `kaggle
   datasets create --public`; wait for processing (`kaggle datasets
   status`); open in browser; right-click image → Copy image
   address → paste into `![](url)` in forum editor.

4. **[WORKFLOW] Pre-draft scan for prior community posts.** Tag:
   `pre-draft-duplicate-check-missed`. Use
   `api.competition_list_topics('<comp>', sort_by='new', page=N)`
   across all pages; scan titles for thematic duplicates before
   committing to a draft. Cheap (~30 s) prerequisite.

## PI additions (from step 4)

- "no" — PI declined to add frictions, promote candidates, or
  ratify further actions.

## Framework version at session-end (Session B)

- Commit SHA at session-B start: `d763aa5` (postmortem promotion
  commit from Session A).
- Commit SHA at session-B end (pre-wrap-commit): `fe269cd`
  (preview re-render dropping P0/P1 home colors).
- Branch: `claude/reverse-engineer-seat-geometry-BPJKs`.
- Active CLAUDE.md rules: 1..40 (unchanged from Session A).
- Loaded skills this session: `postmortem`.
- Compute spent this session: negligible (drafting + Matplotlib
  render + Kaggle Dataset upload). No agent eval / no panel.
- Submissions used this session: 0 to the competition. 1 public
  Kaggle artifact published (discussion post + dataset).

## Outputs published to Kaggle (Session B)

- Discussion post: drafted in
  `audit/2026-05-19-discussion-draft-seed-panel.md`, posted
  manually by PI via web UI (URL not captured in this transcript).
- Dataset: `chrisleitescha/orbit-wars-seed-panel-preview` (CC0,
  released to host the preview image for the post). 460 KB PNG.
