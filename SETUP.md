# SETUP — starting Orbit Wars

This is the onboarding checklist for a fresh container, fresh laptop,
or this brand-new repo. **Read this once, top to bottom, on day 1.**

For day-to-day rules and process, see `CLAUDE.md`. For session-start
read order on an *existing* competition, see `HANDOVER.md`.

> **Orbit Wars note (code/agent comp).** This SETUP was originally
> written for tabular comps. The artifact-Kaggle-Dataset section that
> used to live here is dropped — code/agent comps don't produce OOF
> arrays in the tabular sense. The reference-notebook and
> simulator-probe steps replace it.

## What you need before starting

1. **A Kaggle account** with API access enabled
   (https://www.kaggle.com/settings → "Create New Token" downloads
   `kaggle.json`).
2. **`~/.kaggle/kaggle.json` in place, mode 600.** That is the
   PI-recommended path for this repo. Two env-var fallbacks
   (`KAGGLE_USERNAME` + `KAGGLE_KEY`, or `KAGGLE_API_TOKEN`) also work
   via `bootstrap.sh`, but the `~/.kaggle/kaggle.json` path is simpler
   and more portable.
3. **Python 3.10+** with `pip`.
4. **The competition slug**: `orbit-wars`.

## Day 1 — first 15 minutes

### Step 1. Clone and bootstrap

```bash
git clone <repo-url>
cd <repo>
cp .comp.env.template .comp.env  # already pre-filled with COMP="orbit-wars"
bash bootstrap.sh
```

What `bootstrap.sh` does, in order:

1. Resolves Kaggle credentials. If `KAGGLE_USERNAME` + `KAGGLE_KEY` are
   set, it materialises `~/.kaggle/kaggle.json` so the standard
   `kaggle …` commands work everywhere.
2. Installs Python requirements from `requirements.txt`.
3. Downloads the competition data into `data/` if not already present.
4. (Code-comp addendum.) Pulls the top reference notebooks into
   `external/kernels/` for cross-reference. The exact slugs are
   filled in by the day-1 agent — see Step 3 below.

### Step 2. Confirm the data downloaded

```bash
ls -lh data/
```

Orbit Wars ships **3 files** (~17 KB total, no train/test CSVs):

- `README.md` — full game spec (board, planets, fleets, comets, combat,
  observation/action format). The load-bearing read on Day 1.
- `agents.md` — getting-started guide; CLI workflow; submit examples.
- `main.py` — working baseline agent ("Nearest Planet Sniper" heuristic).
  Use this as the local opponent for your first variants.

If the download fails with an authentication error, `~/.kaggle/kaggle.json`
is not in place or your token expired. Re-create it from
https://www.kaggle.com/settings.

### Step 3. Read the comp's own docs (replaces external-notebook pull)

The comp ships a working baseline. Skip the external-notebook pull on
Day 1 — read `data/README.md` and `data/agents.md` first, run the
shipped `data/main.py` against `random` locally, and only pull
external notebooks if you hit a plateau later.

```bash
# Read the comp description / rules / evaluation pages:
kaggle competitions pages orbit-wars --content                     # full
kaggle competitions pages orbit-wars --content --page-name evaluation
kaggle competitions pages orbit-wars --content --page-name rules
```

> Note: the older `kaggle competitions view` subcommand was removed
> from the CLI; use `competitions pages --content` instead.

### Step 4. Local simulator probe

```bash
pip install "kaggle-environments>=1.28.0"

python -c "
from kaggle_environments import make
env = make('orbit_wars', configuration={'seed': 42}, debug=True)
env.run(['random', 'random'])
final = env.steps[-1]
print([(i, s.reward, s.status) for i, s in enumerate(final)])
"
```

The env name is `orbit_wars` (underscore), not `orbit-wars`. If the
import or `make()` fails, install the pinned version above — Orbit
Wars requires `kaggle-environments >= 1.28.0`.

Then run the shipped baseline against `random`:

```bash
python -c "
from kaggle_environments import make
env = make('orbit_wars', configuration={'seed': 42}, debug=True)
env.run(['data/main.py', 'random'])
print([(i, s.reward, s.status) for i, s in enumerate(env.steps[-1])])
"
```

### Step 5. Code-comp CLI quick-reference (new in this comp)

These subcommands replace tabular-comp habits:

```bash
# Status + submissions you've made:
kaggle competitions submissions orbit-wars

# Per-submission ladder games:
kaggle competitions episodes <SUBMISSION_ID>

# Replay JSON for a specific game (visualisation/analysis):
kaggle competitions replay <EPISODE_ID> -p replays/

# Per-agent log output (debug crashes / behavior):
kaggle competitions logs <EPISODE_ID> 0    # agent index 0
kaggle competitions logs <EPISODE_ID> 1    # agent index 1

# Live leaderboard:
kaggle competitions leaderboard orbit-wars -s
```

## Where the PI's voice goes

The PI uses voice-to-text and dumps thoughts freely. Those go in
`knowledge-base/thoughts/`, dated, one file per session or per topic:

```
knowledge-base/thoughts/2026-05-09-kickoff-priorities.md
```

The agent transcribes lightly, never overwrites, and links related
entries. **This folder is permanent** — do not archive or compress
it during cleanup. See `knowledge-base/README.md` for the full
PI second-brain layout.

## Sanity check before declaring "ready to work"

Day-1 agent verifies:

- [ ] `kaggle competitions list -s orbit` shows orbit-wars with
      `userHasEntered: True`.
- [ ] `data/` has the comp bundle unzipped.
- [ ] `external/kernels/` has at least 1 reference notebook.
- [ ] `comp-context.md` is filled in (slug, deadline, daily submission
      cap, agent IO spec, episode length, opponent pool, replay format).
- [ ] `ISSUES.md` has the first problem-tree (5 children).
- [ ] `HANDOVER.md` has the 4-section day-1 brief.

## Next: read these in order

1. `CLAUDE.md` — rules + pointers.
2. `comp-context.md` — settled-once facts for Orbit Wars.
3. `state/current.md` — current submitted agent and what axes are open.
4. `audit/friction.md` — recent recurring snags.
