"""
grader.py — Evaluation harness for CICDRepairEnv.

Usage:
    from grader import grade_agent, grade_all

    def my_agent(obs, state):
        return Action(action_id=1)

    results = grade_all(my_agent)
    print(results)
"""

from __future__ import annotations

from typing import Callable, Any, List, Dict

from env import CICDRepairEnv, Action, compute_episode_score
from env.models import Observation, EnvironmentState, StochasticConfig, RewardConfig
from env.tasks import TIER_IDS


# Type alias for agent callables
AgentFn = Callable[[Observation, EnvironmentState], Action]

# Canonical tier IDs (backward-compat aliases resolved inside CICDRepairEnv.reset)
_TIERS = TIER_IDS

_DIFFICULTIES = ("easy", "medium", "hard")

def _normalize_rows(rows: List[Dict]) -> List[tuple]:
    """
    Normalise a list of dictionaries for invariant comparison.
    Rounds floats to 2 decimal places and sorts by keys/rows.
    """
    def normalize_val(v: Any) -> str:
        if isinstance(v, float):
            return str(round(v, 2))
        return str(v)

    normalized = [
        tuple(sorted((k, normalize_val(v)) for k, v in row.items()))
        for row in rows
    ]
    return sorted(normalized)



def grade_agent(
    agent_fn: AgentFn,
    difficulty: str,
    *,
    stochastic: StochasticConfig | None = None,
    reward_config: RewardConfig | None = None,
    procedural: bool = False,
) -> float:
    """
    Run a single episode of the given difficulty and return the final score.

    The agent is called as: agent_fn(obs, env.state()) -> Action

    Args:
        agent_fn:      Callable accepting (Observation, EnvironmentState) → Action.
        difficulty:    Tier ID ("tier_1"/"tier_2"/"tier_3") or alias ("easy"/"medium"/"hard").
        stochastic:    Optional stochastic config (sigma > 0 for noisy episodes).
        reward_config: Optional reward weight overrides.
        procedural:    If True, use procedurally-generated logs.

    Returns:
        Final normalised score in [0.15, 0.85].
    """
    if difficulty not in _DIFFICULTIES and difficulty not in _TIERS:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. Valid options: {list(_DIFFICULTIES)} or {list(_TIERS)}"
        )

    env = CICDRepairEnv(stochastic=stochastic, reward_config=reward_config)
    obs = env.reset(difficulty, procedural=procedural)

    try:
        while True:
            action = agent_fn(obs, env.state())
            obs, _reward, done, _info = env.step(action)
            if done:
                break
    except Exception as e:
        # If the agent crashes, terminate the episode
        env._state.done = True

    final_score = compute_episode_score(env.state())
    return float(final_score)


def grade_all(
    agent_fn: AgentFn,
    *,
    stochastic: StochasticConfig | None = None,
    reward_config: RewardConfig | None = None,
    procedural: bool = False,
) -> dict[str, float]:
    """
    Run the agent on all three difficulties and return a summary.

    Returns:
        dict with keys "easy", "medium", "hard", and "average".
    """
    results: dict[str, float] = {}
    for difficulty in _DIFFICULTIES:
        results[difficulty] = grade_agent(
            agent_fn, difficulty,
            stochastic=stochastic,
            reward_config=reward_config,
            procedural=procedural,
        )

    results["average"] = round(
        sum(results[d] for d in _DIFFICULTIES) / len(_DIFFICULTIES), 4
    )
    return results
