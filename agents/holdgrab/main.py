"""holdgrab — opponent-agnostic "grab the production you can hold".

Entry point for both local ``fast.py`` runs and Kaggle submission. The agent
never models an opponent's policy: it values planets by the production-time
integral and treats the opponent only as worst-case physical reach (a bounded
full-attack-future force test). The strategy emits abstract capture intents;
the shared mechanism layer (``DEFAULT_MECHANISMS``) realizes them into
physically-valid actions (lead-aim, sun-avoid, path-detour, OOB-guard,
production-aware sizing). See ``config.py`` for the doctrine pointer + tunables.
"""

from __future__ import annotations

from lib.intent import realize
from lib.mechanism import DEFAULT_MECHANISMS

from agents.holdgrab.chooser import select
from agents.holdgrab.config import DEFAULT
from agents.holdgrab.rollout import choose as rollout_choose
from agents.holdgrab.world_view import build_turn_view


def agent(obs, configuration=None):
    cfg = DEFAULT
    view = build_turn_view(obs, cfg)
    if not view.my_sources or not view.targets:
        return []
    if cfg.use_rollout:
        return rollout_choose(view, cfg, view.world.obs_raw, configuration)
    intents = select(view, cfg)
    return realize(intents, view.world.obs_raw, mechanisms=DEFAULT_MECHANISMS,
                   model=view.model)
