# Orbit Wars — agent competition

Repo for the Kaggle **Orbit Wars** competition
(https://www.kaggle.com/competitions/orbit-wars), a code/agent
tournament with a $50k prize pool. Submissions are Python agents
wrapped in Kaggle Notebooks, evaluated by tournament play (TrueSkill /
Elo).

This repo is seeded from the s6e5 (F1 pit-stops) tabular reference
comp. Process discipline carries over verbatim; tabular-only tools
(probe.py, OOF anchors, Spearman pre-submit diff) are absent. Files
and rules tagged `[TABULAR-ONLY]` are kept for cross-comp comparison
and skipped on this comp.

## Reading order on day 1

1. `CLAUDE.md` — 36 operating rules. Five rules are tagged
   `[TABULAR-ONLY]`; skim and skip on Orbit Wars.
2. `SETUP.md` — onboarding sequence (Kaggle auth, bootstrap, simulator
   probe, reference notebooks).
3. `.claude/skills/kaggle-comp/SKILL.md` — the kaggle-comp skill's
   load order; read its umbrella note about tabular vs code/agent
   sections.
4. `.claude/skills/kaggle-comp/improvements.md` — cross-comp lessons.
   Read entries tagged `[CROSS-CUTTING]` and `[ADAPT-FOR-CODE-COMP]`;
   skip `[TABULAR-ONLY]`.
5. `comp-context.md` — settled-once facts (mostly TBD on Day 1; the
   day-1 agent fills these from `kaggle competitions view`).

## Bootstrap

```bash
cp .comp.env.template .comp.env  # already pre-filled with COMP="orbit-wars"
bash bootstrap.sh
```

Kaggle credentials must be in place at `~/.kaggle/kaggle.json` (mode
600) before running. See SETUP.md.

## Layout

| Path | Role |
|---|---|
| `CLAUDE.md` | Operating rules + pointers (read every session) |
| `SETUP.md` | Day-1 onboarding (read once) |
| `WRAPUP.md` | "Wrap up" / "prepare handover" procedure |
| `HANDOVER.md` | Next-session brief (rewritten every session) |
| `ISSUES.md` | Live problem-tree / claim board |
| `comp-context.md` | Settled-once facts (filled day 1) |
| `state/` | Mutable session state (current agent, calibration ladder, hypotheses, mechanism ledger) |
| `audit/` | Dated probe + friction notes |
| `knowledge-base/` | PI second-brain (permanent — Rules 35-36) |
| `data/` | Comp data downloaded by `bootstrap.sh` |
| `external/kernels/` | Reference notebooks pulled via `kaggle kernels pull` |
| `submissions/` | Kernel build artifacts |
| `.claude/skills/` | `kaggle-comp` (process loops) and `postmortem` |
