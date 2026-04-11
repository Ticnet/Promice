"""
cicd_env.py — Core environment implementation for CICDRepairEnv.

Implements the OpenEnv-style interface:
    env.reset(task_id) -> Observation
    env.step(action)   -> (Observation, float, bool, dict)
    env.state()        -> EnvironmentState

Supports two operational modes controlled at construction time:

  1. **Deterministic** (default, sigma = 0):
     Same task, same actions → same rewards, same outcome every time.

  2. **Stochastic** (sigma > 0):
     Introduces real-world CI/CD noise — intermittent failures during
     correct actions, action corruption from flaky runners, and
     irrelevant log noise injection.

Procedural log generation can be enabled per-episode via the ``procedural``
flag in ``reset()``.
"""

from __future__ import annotations

import copy
import random

from env.models import (
    Action,
    DESTRUCTIVE_ACTION_IDS,
    EnvironmentState,
    Observation,
    RewardConfig,
    StochasticConfig,
)
from env.tasks import TASKS, resolve_task_id
from env.procedural import generate_log, generate_noise_line, generate_memory_hints_tier3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_score(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Normalise a raw reward/score into the strict OpenEnv (0, 1) exclusive range.
    Formula: 0.15 + (clamped_raw * 0.7)  →  range [0.15, 0.85]
    
    This range is strictly within (0, 1), satisfying Phase 2 validation.
    """
    clamped_raw = max(lo, min(hi, value))
    return round(0.15 + (clamped_raw * 0.7), 4)


def normalize_step_reward(raw: float) -> float:
    """
    Map raw step rewards (approx [-0.2, 1.2]) to a safe (0.1, 0.9) range.
    Ensures no negative rewards are logged, satisfying Phase 2 validation.
    """
    clamped = max(-0.2, min(1.2, raw))
    return round(0.1 + ((clamped + 0.2) / 1.4) * 0.8, 4)


def compute_episode_score(state: EnvironmentState) -> float:
    """
    Multi-component weighted episode score, strictly mapped to [0.15, 0.85].

    Components (weights sum to 1.0):
        success    (0.40): 1.0 if pipeline fully repaired, else 0.0
        progress   (0.25): fraction of repair sequence completed
        efficiency (0.15): optimal_steps / actual_steps (capped at 1.0)
        safety     (0.10): 1 - (destructive_misuses / max_steps)
        speed      (0.10): 1 - (steps_taken / max_steps)
    """
    from env.tasks import TASKS

    task = TASKS.get(state.task_id, {})
    optimal = task.get("optimal_steps", 1)
    seq_len = len(state.required_sequence) if state.required_sequence else 1

    # Component 1: Success — binary completion signal
    success = 1.0 if state.pipeline_healthy else 0.0

    # Component 2: Progress — fraction of sequence completed
    progress = state.sequence_position / max(seq_len, 1)

    # Component 3: Efficiency — how close to optimal step count
    steps = max(state.step_count, 1)
    efficiency = min(1.0, optimal / steps)

    # Component 4: Safety — penalise destructive actions used incorrectly
    destructive_misuses = sum(
        1 for a in state.repair_sequence_taken
        if a in DESTRUCTIVE_ACTION_IDS and a not in state.required_sequence
    )
    safety = max(0.0, 1.0 - destructive_misuses / max(state.max_steps, 1))

    # Component 5: Speed — reward finishing quickly
    speed = max(0.0, 1.0 - state.step_count / max(state.max_steps, 1))

    raw_sum = (
        success * 0.40
        + progress * 0.25
        + efficiency * 0.15
        + safety * 0.10
        + speed * 0.10
    )
    return normalize_score(raw_sum)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Internal clamp to [0, 1] without normalization."""
    return max(lo, min(hi, value))


def _build_observation(state: EnvironmentState) -> Observation:
    """Construct the agent-visible observation from current internal state."""
    progress_pct = (
        state.sequence_position / len(state.required_sequence)
        if state.required_sequence
        else 1.0
    )
    available_actions = list(range(8))  # all actions always available
    task_score = compute_episode_score(state) if state.done else None
    
    return Observation(
        pipeline_stage=state.pipeline_stage,
        failure_log=state.failure_log,
        error_type=state.failure_type,
        available_actions=available_actions,
        memory_hints=list(state.memory_hints),
        step_count=state.step_count,
        pipeline_healthy=state.pipeline_healthy,
        progress_pct=round(progress_pct, 4),
        task_score=task_score,
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CICDRepairEnv:
    """
    CI/CD pipeline repair RL environment with optional stochasticity.

    Usage (deterministic - backward-compatible):
        env = CICDRepairEnv()
        obs = env.reset("tier_1")          # or "easy"
        while True:
            action = agent(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break

    Usage (stochastic):
        from env.models import StochasticConfig
        cfg = StochasticConfig(sigma=0.3, seed=42)
        env = CICDRepairEnv(stochastic=cfg)
        obs = env.reset("tier_2", procedural=True)
    """

    def __init__(
        self,
        stochastic: StochasticConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._stochastic = stochastic or StochasticConfig()
        self._reward_cfg = reward_config or RewardConfig()
        self._state: EnvironmentState | None = None
        self._rng: random.Random = random.Random()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stochastic_config(self) -> StochasticConfig:
        return self._stochastic

    @property
    def reward_config(self) -> RewardConfig:
        return self._reward_cfg

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = "tier_1",
        *,
        procedural: bool = False,
    ) -> Observation:
        """
        Reset the environment for the given task.

        Args:
            task_id:     One of "tier_1", "tier_2", "tier_3" (or legacy "easy"/"medium"/"hard").
            procedural:  If True, generate a fresh log using the procedural engine.

        Returns:
            Initial observation.
        """
        # Resolve legacy aliases
        canonical_id = resolve_task_id(task_id)

        if canonical_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid options: {list(TASKS.keys())}"
            )

        task = TASKS[canonical_id]

        # Initialise per-episode RNG
        seed = self._stochastic.seed
        if seed is None and self._stochastic.sigma > 0:
            seed = random.randint(0, 2**31 - 1)
        self._rng = random.Random(seed)

        # Determine failure log
        if procedural and task.get("procedural", False):
            failure_log = generate_log(
                canonical_id,
                self._rng,
                inject_noise=(self._stochastic.sigma > 0),
            )
        else:
            failure_log = task["failure_log"]

        # Determine memory hints / bank (procedural for tier_3)
        if procedural and canonical_id == "tier_3":
            memory_hints, memory_bank = generate_memory_hints_tier3(self._rng)
        else:
            memory_hints = list(task["memory_hints"])
            memory_bank = dict(task["memory_bank"])

        self._state = EnvironmentState(
            task_id=canonical_id,
            difficulty=canonical_id,
            pipeline_stage=task["pipeline_stage"],
            failure_type=task["failure_type"],
            failure_log=failure_log,
            required_sequence=list(task["required_sequence"]),
            repair_sequence_taken=[],
            sequence_position=0,
            step_count=0,
            max_steps=task["max_steps"],
            pipeline_healthy=False,
            cumulative_reward=0.0,
            memory_bank=memory_bank,
            memory_hints=memory_hints,
            root_cause_identified=False,
            efficiency_eligible=True,
            done=False,
            stochastic_seed=seed,
        )

        return _build_observation(self._state)

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Apply an action to the environment.

        When sigma > 0, the following stochastic events may occur:
          - **Intermittent failure**: A correct action may transiently fail
            (no progress, no penalty - simulates pip timeout, docker blip).
          - **Action corruption**: The executed action may differ from the
            requested one (simulates flaky CI runner).
          - **Log noise**: Irrelevant warning lines may be injected into the
            failure log.

        Args:
            action: Action model with a valid action_id (0–7).

        Returns:
            (observation, reward, done, info)

            Note: The 'reward' returned is the raw step increment.
            The 'info["cumulative_reward"]' is the normalized task score in [0.15, 0.85].
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        state = self._state
        sigma = self._stochastic.sigma
        rc = self._reward_cfg
        action_id = int(action.action_id)
        step_reward: float = 0.0

        # 1. Increment step count
        state.step_count += 1
        state.repair_sequence_taken.append(action_id)

        # ── Stochastic: action corruption ──────────────────────────
        if sigma > 0:
            corruption_prob = sigma * self._stochastic.action_corruption_prob
            if self._rng.random() < corruption_prob:
                # Replace requested action with a random different action
                alternatives = [a for a in range(8) if a != action_id]
                action_id = self._rng.choice(alternatives)

        # 2. Determine expected action
        seq_len = len(state.required_sequence)
        expected = (
            state.required_sequence[state.sequence_position]
            if state.sequence_position < seq_len
            else None
        )

        if expected is not None and action_id == expected:
            # ── Stochastic: intermittent failure ───────────────────
            if sigma > 0:
                ifail_prob = sigma * self._stochastic.intermittent_failure_prob
                if self._rng.random() < ifail_prob:
                    # Transient failure — no progress, no penalty
                    # Just skip the reward but don't penalise
                    state.efficiency_eligible = False
                    # Fall through to timeout check
                    obs = _build_observation(state)
                    info: dict = {
                        "step_count":        state.step_count,
                        "sequence_position": state.sequence_position,
                        "action_taken":      action_id,
                        "action_correct":    False,
                        "intermittent_failure": True,
                        "cumulative_reward": compute_episode_score(state),
                        "done":              state.done,
                        "pipeline_healthy":  state.pipeline_healthy,
                    }
                    # Check timeout
                    if not state.done and state.step_count >= state.max_steps:
                        state.done = True
                        info["done"] = True
                    return obs, 0.0, state.done, info

            # ---- CORRECT action ----------------------------------------
            # Progress reward: distributed evenly across sequence
            progress_reward = rc.progress_total / seq_len
            step_reward += progress_reward

            # Root-cause bonus: first correct action only
            if not state.root_cause_identified:
                step_reward += rc.root_cause_bonus
                state.root_cause_identified = True

            # Advance sequence
            state.sequence_position += 1

            # Check if sequence is now complete
            if state.sequence_position == seq_len:
                step_reward += rc.success_reward  # success reward

                # Efficiency bonus: completed in optimal (minimum) steps
                task_cfg = TASKS[state.task_id]
                optimal = task_cfg["optimal_steps"]
                if state.step_count == optimal and state.efficiency_eligible:
                    step_reward += rc.efficiency_bonus

                state.pipeline_healthy = True
                state.done = True

        else:
            # ---- WRONG action ------------------------------------------
            state.efficiency_eligible = False

            # Destructive penalty
            if action_id in DESTRUCTIVE_ACTION_IDS:
                step_reward -= rc.destructive_penalty

        # ── Stochastic: log noise injection ────────────────────────
        if sigma > 0:
            noise_prob = sigma * self._stochastic.log_noise_prob
            if self._rng.random() < noise_prob:
                noise_line = generate_noise_line(self._rng)
                state.failure_log = state.failure_log.rstrip("\n") + "\n" + noise_line + "\n"

        # 3. Timeout check
        if not state.done and state.step_count >= state.max_steps:
            state.done = True

        # 4. Accumulate reward (clamped at episode end, not per-step)
        state.cumulative_reward += step_reward

        # Build observation
        obs = _build_observation(state)

        # Build info dict
        normalized_step_reward = normalize_step_reward(step_reward)
        current_cumulative_score = compute_episode_score(state)
        
        info = {
            "step_count":        state.step_count,
            "sequence_position": state.sequence_position,
            "action_taken":      action_id,
            "action_correct":    (expected is not None and action_id == expected),
            "cumulative_reward": current_cumulative_score,
            "done":              state.done,
            "pipeline_healthy":  state.pipeline_healthy,
        }

        return obs, normalized_step_reward, state.done, info

    def state(self) -> EnvironmentState:
        """Return a copy of the current internal state (read-only snapshot)."""
        if self._state is None:
            raise RuntimeError("Call reset() before accessing state().")
        return self._state.model_copy(deep=True)
