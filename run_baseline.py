"""
run_baseline.py — Rule-based baseline agent for CICDRepairEnv.

The baseline uses simple log-pattern matching and observation fields
to deterministically solve all three tiers with a perfect score.

Expected output (deterministic mode):
    Tier 1 : 0.99
    Tier 2 : 0.99
    Tier 3 : 0.99
    Average: 0.99
"""

from __future__ import annotations

from env import Action
from env.models import Observation, EnvironmentState
from grader import grade_all


def baseline_agent(obs: Observation, state: EnvironmentState) -> Action:
    """
    Rule-based agent that uses semantic log analysis (not exact strings).

    Decision priority (highest to lowest):
      1. ABI / linker error     → use_memory_fix (6)
      2. Cache + version conflict → clear_cache (4) then change_version (2)
      3. ModuleNotFoundError     → install_dependency (1)
      4. Missing env variable    → set_env_variable (3)
      5. Default fallback        → restart_step (0)
    """
    log = obs.failure_log.lower()
    error_type = obs.error_type.lower()

    # Priority 1: ABI mismatch — memory fix is the only correct action.
    if "abi" in log or "abilink" in error_type or "abi_mismatch" in error_type:
        return Action(action_id=6)  # use_memory_fix

    # Also catch tasks where memory_hints reference ABI / memory fix
    if obs.memory_hints and ("abi" in " ".join(obs.memory_hints).lower()):
        return Action(action_id=6)  # use_memory_fix

    # Priority 2: cache + version conflict (tier_2, 2-step sequence)
    if "cache" in log and ("version" in log or "conflict" in log):
        if state.sequence_position == 0:
            return Action(action_id=4)  # clear_cache (first)
        else:
            return Action(action_id=2)  # change_version (second)

    # Priority 3: missing Python dependency
    if "importerror" in log or "modulenotfounderror" in log or "no module named" in log:
        return Action(action_id=1)  # install_dependency

    # Priority 4: missing env variable
    if "environment variable" in log or ("env" in log and "missing" in log):
        return Action(action_id=3)  # set_env_variable

    # Default fallback
    return Action(action_id=0)  # restart_step


if __name__ == "__main__":
    results = grade_all(baseline_agent)

    print("\n=== CICDRepairEnv — Baseline Agent Results ===")
    print(f"  Easy    : {results['easy']}")
    print(f"  Medium  : {results['medium']}")
    print(f"  Hard    : {results['hard']}")
    print(f"  ─────────────────────────────────")
    print(f"  Average : {results['average']}")
    print("================================================\n")
