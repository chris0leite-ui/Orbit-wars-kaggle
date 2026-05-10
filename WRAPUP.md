# WRAPUP.md — procedure for "wrap up" and "prepare handover"

When PI says **"wrap up"** → run section A.
When PI says **"prepare handover"** → run section A then section B.
Both must end with a push to the current branch.

Cap ≤150 lines. This is a checklist agents follow verbatim.

---

## A. Wrap up (every session end, every branch)

1. **Surface state.** `git fetch origin && git status` and report a
   one-line summary (branch, ahead/behind, modified count) to PI.

2. **Stage tracked work.** `git add` only these paths if changed:
   - `audit/`
   - `scripts/`
   - `kernels/`
   - `submissions/`
   - `requirements.txt`
   - `CLAUDE.md`
   - `HANDOVER.md`
   - `comp-context.md`
   - `WRAPUP.md`
   - `ISSUES.md`

   Never `git add -A` / `git add .` — avoid sensitive files (`.env`,
   `kaggle.json`) and large binaries.

3. **Update CLAUDE.md `## Current state` YAML.** Edit only fields
   that changed today, with today's audits as the source of truth:
   - `date` (today, ISO format) and `days_to_deadline` (recompute
     against `comp-context.md::deadline`)
   - `session_log` (prepend a new entry; cap ≤2 entries — older
     narratives roll into per-day audit notes, not the YAML)
   - `submissions_used_today`, `submissions_used_total`
   - `mechanism_families_explored` (append new entries; never reorder)
   - `tournament_rank_today`, `our_best_rank`
   - `gate_status`, `headroom_to_top5pct`
   - `plateau_days`, `saturation_count`

3b. **Update ISSUES.md leaf status (Rule 18).** For each leaf this
    branch owned today, update its `[owner: ... | status: ...]` to
    reflect outcome: `wip` → `done` (PASS), `null` (falsified), or
    `parked` (blocked / deprioritised). When fully resolved with a
    crisp result, move the one-liner to the "Falsified or dead"
    section. If a re-decomposition trigger has fired (see bottom
    of ISSUES.md), flag it in the commit message for the next
    strategy-critic-loop owner.

4. **Append friction one-liners.** To `audit/friction.md` under a
   `## YYYY-MM-DD` heading (create if absent today). Format already
   established:
   ```
   - `tag: <kebab-slug>` — <session/day context>: <what happened>.
     <Root cause>. **Fix:** <concrete action>.
   ```
   One entry per distinct friction event today. Reuse existing tags
   when possible.

4b. **Run the postmortem skill.** Follow
    `.claude/skills/postmortem/SKILL.md` steps 1–6: identify what
    went wrong, draft promotion candidates from today's friction
    entries, ask PI for additions, ask PI to ratify promotions,
    write `audit/YYYY-MM-DD-postmortem*.md`, stage outputs.
    Blocks on PI replies; do not bypass.

5. **File-size guard.** If `CLAUDE.md` > 50k tokens or `HANDOVER.md`
   > 150 lines, archive the oldest sections to
   `audit/archive-YYYY-MM-DD-<topic>.md` BEFORE step 6 and update
   pointers. Never silently truncate.

5b. **Calibration snapshot (Rule 26).** If a per-comp calibration
    script exists (predicted-rank vs actual-rank for code-comps),
    run it and append the table to today's postmortem (step 4b).
    Used by postmortem step 2 to count PI overrides. If 0/M overrides
    for 2 consecutive postmortems, postmortem flags stamp risk in
    HANDOVER.md `## Where we are` (friction tag `pi-stamp-risk`).

6. **Commit.** Use HEREDOC, structured message:
   ```
   git commit -m "$(cat <<'EOF'
   Day-N wrap: <one-line summary>

   - <probe-tag>: <result>
   - <probe-tag>: <…>

   <Claude Code footer>
   EOF
   )"
   ```

7. **Push.** `git push -u origin <current-branch>`. On network failure,
   retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s).
   Never `--force` to a shared branch without explicit PI approval.

---

## B. Prepare handover (end of day OR on PI request)

1. **Run section A first** if it has not already run this turn.

2. **Read inputs:**
   - Today's `audit/YYYY-MM-DD-*.md` (per-probe records).
   - Today's `audit/friction.md` appends.
   - Last 3 commits on current branch.
   - `CLAUDE.md ## Current state` YAML.
   - Current `HANDOVER.md`.

3. **Rewrite HANDOVER.md** preserving the established 4-section
   format. Length budget ≤150 lines.
   - **Where we are** — current submitted agent, gap to top-5%,
     budget used, days remaining.
   - **Today's progress** — load-bearing findings only; cite
     audit paths.
   - **Falsified-or-dead** — one-liner per dead lever.
   - **Next-session first-action** — ranked moves with EV / cost.
   Keep tone dense and synthesised, not journal-style.

4. **Update `## Pointers`.** One line per new audit added today:
   `- audit/YYYY-MM-DD-<slug>.md — <one-sentence purpose>`.

5. **Repeat section A steps 6–7** (commit + push) so the new
   HANDOVER ships in the same flow.

---

## Parallel-branch convention (Rule 15 addendum)

While a non-`main` branch is alive, write today's notes to a
section heading inside HANDOVER.md of the form:

```
## Day-N PM <branch-slug>
```

…where `<branch-slug>` is the part of the branch name after
`claude/` (e.g. `orbit-war-setup-KbeKq`).

Rules:
- Each branch only edits **its own** `Day-N PM <slug>` section.
- Never edit another branch's section. Never edit the morning
  synthesis sections.
- The scribe (whichever agent runs `prepare handover` on the
  merge-target branch) consolidates all `Day-N PM *` sections
  into the next morning's synthesis and removes the per-branch
  sections at that time.

This keeps merge conflicts trivially resolvable: disjoint H2
sections never collide, even when 3+ branches push the same hour.
