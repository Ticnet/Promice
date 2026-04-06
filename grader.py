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

from typing import Callable

from env import CICDRepairEnv, Action
from env.models import Observation, EnvironmentState


# Type alias for agent callables
AgentFn = Callable[[Observation, EnvironmentState], Action]

_DIFFICULTIES = ("easy", "medium", "hard")


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def grade_agent(agent_fn: AgentFn, difficulty: str) -> float:
    """
    Run a single episode of the given difficulty and return the final score.

    The agent is called as: agent_fn(obs, env.state()) -> Action

    Args:
        agent_fn:   Callable accepting (Observation, EnvironmentState) → Action.
        difficulty: One of "easy", "medium", "hard".

    Returns:
        Final normalised score in [0.0, 1.0].
    """
    if difficulty not in _DIFFICULTIES:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. Valid options: {list(_DIFFICULTIES)}"
        )

    env = CICDRepairEnv()
    obs = env.reset(difficulty)

    while True:
        action = agent_fn(obs, env.state())
        obs, _reward, done, _info = env.step(action)
        if done:
            break

    final_score = _clamp(env.state().cumulative_reward)
    return round(final_score, 4)


def grade_all(agent_fn: AgentFn) -> dict[str, float]:
    """
    Run the agent on all three difficulties and return a summary.

    Returns:
        dict with keys "easy", "medium", "hard", and "average".
    """
    results: dict[str, float] = {}
    for difficulty in _DIFFICULTIES:
        results[difficulty] = grade_agent(agent_fn, difficulty)

    results["average"] = round(
        sum(results[d] for d in _DIFFICULTIES) / len(_DIFFICULTIES), 4
    )
    return results
