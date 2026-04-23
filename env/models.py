"""
models.py — Typed data models for CICDRepairEnv.

Uses Pydantic v2 for full type validation and serialization.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class ActionID(IntEnum):
    """Discrete action identifiers (0–7)."""
    restart_step       = 0
    install_dependency = 1
    change_version     = 2
    set_env_variable   = 3
    clear_cache        = 4
    rollback           = 5
    use_memory_fix     = 6
    ignore_continue    = 7


ACTION_NAMES: dict[int, str] = {
    0: "restart_step",
    1: "install_dependency",
    2: "change_version",
    3: "set_env_variable",
    4: "clear_cache",
    5: "rollback",
    6: "use_memory_fix",
    7: "ignore_continue",
}

# Actions that incur a penalty if used incorrectly.
DESTRUCTIVE_ACTION_IDS: frozenset[int] = frozenset({5, 7})


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------

class StochasticConfig(BaseModel):
    """
    Controls the stochastic transition function.

    When sigma > 0, the environment introduces real-world CI/CD noise:
    intermittent failures, action corruption, and log noise injection.
    All event probabilities are scaled by sigma, so sigma=0.0 preserves
    full determinism.
    """
    sigma: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Master noise scale.  0.0 = deterministic, 1.0 = max stochasticity.",
    )
    seed: int | None = Field(
        default=None,
        description="RNG seed for reproducible stochastic episodes.  None = random.",
    )
    intermittent_failure_prob: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description=(
            "Base probability that a correct action triggers a transient failure "
            "(e.g. pip network timeout).  Actual prob = sigma * this value."
        ),
    )
    action_corruption_prob: float = Field(
        default=0.02, ge=0.0, le=1.0,
        description=(
            "Base probability that the executed action differs from the requested "
            "one (simulates flaky CI runner).  Actual prob = sigma * this value."
        ),
    )
    log_noise_prob: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description=(
            "Base probability of injecting irrelevant warning lines into logs "
            "each step.  Actual prob = sigma * this value."
        ),
    )

    model_config = {"frozen": True}


class RewardConfig(BaseModel):
    """
    Formalized, configurable reward weights.

    R_total = clamp(sum(root + progress + efficiency + success - destructive), 0, 1)
    """
    root_cause_bonus:   float = Field(default=0.20, description="Bonus for first correct action.")
    progress_total:     float = Field(default=0.30, description="Total progress reward distributed across correct steps.")
    efficiency_bonus:   float = Field(default=0.10, description="Bonus for completing in optimal steps with zero errors.")
    success_reward:     float = Field(default=0.40, description="Reward for completing the full repair sequence.")
    destructive_penalty: float = Field(default=0.10, description="Penalty for destructive actions used incorrectly.")

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """A single agent action."""
    action_id: ActionID

    model_config = {"frozen": True}


class Observation(BaseModel):
    """
    The observation returned to the agent at each timestep.
    This is the ONLY information the agent is guaranteed to see.
    """
    pipeline_stage:    str             = Field(..., description="Current CI/CD stage name.")
    failure_log:       str             = Field(..., description="Raw log text from the failed stage.")
    error_type:        str             = Field(..., description="High-level error category.")
    available_actions: list[int]       = Field(..., description="List of valid action IDs.")
    memory_hints:      list[str]       = Field(default_factory=list, description="Hints from the memory bank.")
    step_count:        int             = Field(..., ge=0, description="Number of steps taken so far.")
    pipeline_healthy:  bool            = Field(..., description="True only when pipeline is fully repaired.")
    progress_pct:      float           = Field(..., ge=0.0, le=1.0, description="Repair progress (0.0–1.0).")
    task_score:        Optional[float] = Field(default=None, description="Final score when done, in range [0.01, 0.99]")

    model_config = {"frozen": True}


class EnvironmentState(BaseModel):
    """
    Full internal environment state.
    Agents MAY receive this but should not rely on it being available in all setups.
    """
    task_id:               str
    difficulty:            str
    pipeline_stage:        str
    failure_type:          str
    failure_log:           str
    required_sequence:     list[int]
    repair_sequence_taken: list[int]   = Field(default_factory=list)
    sequence_position:     int         = 0
    step_count:            int         = 0
    max_steps:             int         = 10
    pipeline_healthy:      bool        = False
    cumulative_reward:     float       = 0.0
    memory_bank:           dict        = Field(default_factory=dict)
    memory_hints:          list[str]   = Field(default_factory=list)
    root_cause_identified: bool        = False
    efficiency_eligible:   bool        = True
    done:                  bool        = False
    stochastic_seed:       int | None  = Field(default=None, description="Seed used for this episode's RNG.")

    model_config = {"arbitrary_types_allowed": True}
