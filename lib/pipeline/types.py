"""Type signatures for the analytical-agent pipeline stages.

Each pipeline stage is a callable with a typed input and output. Stage
implementations live in sibling modules and are composed via
`lib.pipeline.compose`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from lib.intent import World
from lib.joint_solver.columns import Column
from lib.world_model import WorldModel


# ---------------------------------------------------------------------------
# Stage outputs (dataclasses)
# ---------------------------------------------------------------------------


@dataclass
class TurnContext:
    """Output of Stage 1 (Perception). Everything downstream stages need.

    Parsed once per turn from the raw observation; passed by reference into
    every later stage so they can read but not re-derive.
    """
    obs_d: dict
    configuration: Optional[Any]
    me: int
    num_seats: int
    step_now: int
    omega: float
    planets: list  # list[Planet]
    fleets: list   # list[Fleet]
    my_planets: list
    other_planets: list
    world: World
    model: WorldModel
    # Empty/degenerate signals: if True, downstream stages should short-circuit.
    is_empty_obs: bool = False
    is_no_targets: bool = False


@dataclass
class CandidateSet:
    """Output of Stage 2 (Candidate generation).

    `prerank` is a list of tuples of the form
    `(cheap_delta, src, tgt, ships, angle, eta, horizon_hint, wait_N)`.
    The shape is dictated by `agents.baseline.proposer.propose` and is
    consumed unchanged by Stage 3.
    """
    prerank: list


@dataclass
class PrerankedColumns:
    """Output of Stage 3 (Pre-rank / pre-filter).

    `columns` is the candidate set the Decision rule will operate on.
    `n_before_filter` and `n_after_filter` are diagnostics; alternative
    Stage-3 implementations may apply different filtering policies.
    """
    columns: list  # list[Column]
    n_before_filter: int
    n_after_filter: int
    # Set by endgame-portfolio focus when active (2P only). Empty = no focus.
    portfolio: list = field(default_factory=list)
    portfolio_filtered: bool = False
    is_winning_state: bool = False


@dataclass
class OppModelResult:
    """Output of Stage 4 (Opp model).

    `opp_arrivals` is a list of `(target_pid, eta_absolute, opp_owner, ships)`
    tuples — the shape `predict_opp_multi_launch` returns.
    `augmented_model` is `model` with `opp_arrivals` merged into its ledger
    so the Decision rule's leaf evaluation sees the opp-augmented timelines.
    """
    opp_arrivals: list
    augmented_model: WorldModel


@dataclass
class DecisionResult:
    """Output of Stage 5 (Decision rule).

    `moves` is `[[src_id, angle, ships], ...]` ready to emit (post Stage-6
    leaf evaluation, post Stage-5 decision logic). For the reference
    implementation this is the wait_N==0 subset of `fired_columns`;
    alternative implementations may include carry-forward columns or other
    transformations.
    `fired_columns` is every Column the decision rule selected (any wait_N).
    """
    moves: list
    fired_columns: list  # list[Column]
    objective: float
    status: str
    n_x_vars: int = 0
    n_y_vars: int = 0
    n_constraints: int = 0
    per_planet_chosen: dict = field(default_factory=dict)
    per_planet_value: dict = field(default_factory=dict)


@dataclass
class CommittedMoves:
    """Output of Stage 7 (Commit).

    `moves` is the final list emitted this turn. For the stateless
    reference implementation this equals `DecisionResult.moves`;
    alternative implementations may decant from a persistent schedule
    and/or commit new wait_N>0 fires for future decant.
    """
    moves: list
    # Persisted state: alternative implementations may write into a
    # module-level store keyed by (my_id, game_id). For the reference
    # stateless commit, this is always None.
    persisted_state: Optional[Any] = None


@dataclass
class OpeningResult:
    """Output of the optional OpeningOverride stage.

    If `committed` is non-None, the pipeline short-circuits: stages 2-7 are
    skipped this turn and `committed.moves` is emitted. If None, the
    standard pipeline runs.
    """
    committed: Optional[CommittedMoves]
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage type aliases (Callable signatures)
# ---------------------------------------------------------------------------


Perception = Callable[[dict, Optional[Any]], TurnContext]
CandidateGen = Callable[[TurnContext], CandidateSet]
PrerankStage = Callable[[CandidateSet, TurnContext], PrerankedColumns]
OppModelStage = Callable[[TurnContext], OppModelResult]
DecisionStage = Callable[
    [PrerankedColumns, OppModelResult, TurnContext], DecisionResult
]
CommitStage = Callable[[DecisionResult, TurnContext], CommittedMoves]
OpeningStage = Callable[[TurnContext], OpeningResult]
