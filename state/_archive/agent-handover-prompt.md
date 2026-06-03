# Agent handover prompt — paste this into a fresh Claude Code session

This is the canonical day-1 agent prompt for a fresh Claude Code
session whose cwd is the cloned Orbit Wars repo. PI must have set up
Kaggle credentials beforehand — either `~/.kaggle/kaggle.json`
(mode 600) OR `KAGGLE_API_TOKEN=KGAT_…` in the environment.

---

```
You are the day-1 agent for the Kaggle Orbit Wars competition
(https://www.kaggle.com/competitions/orbit-wars). The PI has cloned
this repo and invoked you with cwd set to the repo root. Your Kaggle
credentials should already be in place — either ~/.kaggle/kaggle.json
(mode 600) or the KAGGLE_API_TOKEN env var (KGAT_… form). Verify
before doing anything else; if `kaggle competitions list -s orbit`
fails with 401/403, stop and ask the PI to fix the token.

Orbit Wars is a real-time strategy code/agent competition: 100x100
continuous 2D space, 2 or 4 players, 500-turn games, planets rotating
around a sun, $50k prize pool, deadline 2026-06-23 23:59 UTC. 2382
teams already entered. Submissions are Python agents (main.py with
agent(obs) function), evaluated by TrueSkill-style ladder. The repo
seed reflects this: tabular-only tooling (probe.py, 4-gate leakage
filter, OOF anchors, Spearman pre-submit diff) is absent. What
carries over is process discipline.

Read these in order before doing anything else:
1. CLAUDE.md — 36 operating rules. Five rules tagged [TABULAR-ONLY]
   skip on Orbit Wars; the rest apply. Note Rule 12: cap is 5/day.
2. SETUP.md — onboarding sequence (Kaggle auth, bootstrap, simulator
   probe, CLI quick-reference).
3. comp-context.md — already filled with verified facts (game spec,
   agent IO, scoring, eval, turn order). Read it; do not re-derive.
4. .claude/skills/kaggle-comp/SKILL.md — invocation rules and load
   order; read its umbrella note about tabular vs code/agent sections.
5. .claude/skills/kaggle-comp/improvements.md — entries tagged
   [CROSS-CUTTING] and [ADAPT-FOR-CODE-COMP]. Skip [TABULAR-ONLY].

Your day-1 task is read-only-discovery + first-baseline prep. No
submissions; no agent training run. Specifically:

(a) Run `kaggle competitions list -s orbit` to confirm slug + your
    team is registered; cross-check fields in comp-context.md against
    `kaggle competitions pages orbit-wars --content` (rules,
    evaluation, data licence, external-data policy). Update any TBDs.

(b) Run `bash bootstrap.sh`. It will:
    - Install requirements (kaggle, kaggle-environments>=1.28.0).
    - Download the comp into data/ (3 files: README.md, agents.md,
      main.py — the comp ships its own working baseline).
    - Run a random-vs-random smoke episode.
    Capture exit status. Inventory data/ into
    audit/<today>-day-1-data-inventory.md.

(c) Read data/README.md end-to-end (full game spec — board geometry,
    planet types, fleet speed formula, combat rules, observation
    fields, action format, turn order). Read data/agents.md (CLI
    workflow) and data/main.py (Nearest Planet Sniper baseline).

(d) Run the shipped baseline locally against `random` for ≥5
    different seeds; record winrate. Then run baseline-vs-baseline
    self-play for ≥5 seeds; verify it passes (this is the validation
    gate Kaggle will run on every submit). Log per-seed reward + ship
    counts to audit/<today>-day-1-data-inventory.md.

(e) Sanity-check the orbit-prediction math: compute the trajectory of
    one orbiting planet over 100 turns using initial_planets +
    angular_velocity, and compare against the env's actual planet
    positions at step 100. This is load-bearing for any agent that
    pre-targets where a planet WILL be.

(f) Draft ISSUES.md problem-tree with concrete children: env-dynamics
    sub-leaves (orbit prediction; combat math; comet-spawn timing),
    agent-class choice (heuristic / search / IL / RL), opponent-
    modelling (TrueSkill matchmaking implications — beating the
    shipped baseline ≠ winning at μ=900), training-eval infra
    (local-tournament fixture; opponent panel = [random, baseline,
    our v0..vN]), submission packaging (single main.py vs tar.gz).
    Mark each [owner: unclaimed | status: open].

(g) Write the first HANDOVER.md (4 sections; see WRAPUP.md for
    format). Include in "Next-session first-action": the simplest
    1-step improvement on the shipped baseline (e.g. send 110% of
    the target's garrison instead of +1; or pick the highest-
    production target instead of the nearest).

Do NOT submit anything. Do NOT start an RL training run. Do NOT
invoke `kaggle competitions submit` for any reason today.

When (a)–(g) are done, report findings to PI in 6-8 sentences and
STOP. Wait for PI sign-off before any compute beyond local
simulator runs.
```

---

## Why this prompt looks the way it does

- Slug, deadline, env name (`orbit_wars` underscore), submission cap
  (5/day), file inventory, eval method (TrueSkill), and rolling-last-2
  rule are all **verified facts** (from `kaggle competitions pages` +
  comp-shipped README/agents.md). The day-1 agent does not waste a
  session rediscovering these.
- The "pull external notebooks" step is **dropped**. The comp ships
  its own working baseline; that's a stronger starting point than
  guessing user/slug pairs for community notebooks.
- Step (e) (orbit-prediction sanity check) is added because that math
  is load-bearing for ANY non-myopic agent and easy to get wrong.
- Step (f) explicitly flags the **TrueSkill matchmaking trap**:
  beating the shipped baseline at μ=600 does not mean beating
  similar-skill opponents at μ=900. Optimising for the local panel is
  a known plateau pattern.
- The prompt holds a hard line on **no submission** day-1: validation
  episodes, ladder games, and TrueSkill σ all take time to settle —
  one informed submit on day 2 is worth three speculative ones on
  day 1.
