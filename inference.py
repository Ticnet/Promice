"""
inference.py — LLM agent inference script for CICDRepairEnv.

Reads credentials from environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key (used as OpenAI API key).

Emits structured stdout logs in [START] / [STEP] / [END] format.

Usage:
    API_BASE_URL=https://api.openai.com/v1 \\
    MODEL_NAME=gpt-4o-mini \\
    HF_TOKEN=your_key \\
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import re
import time
from typing import Optional

from openai import OpenAI

# Bring the local env package onto the path when run from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import CICDRepairEnv, Action, compute_episode_score
from env.models import Observation, EnvironmentState, ACTION_NAMES

# ---------------------------------------------------------------------------
# Config — read from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Use HF_TOKEN as the primary API key
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY") or ""

BENCHMARK   = "CICDRepairEnv"
MAX_STEPS   = 10            # hard cap per episode
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.8

TASKS = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# Structured log helpers  — MUST follow [START] / [STEP] / [END] format exactly
# ---------------------------------------------------------------------------

def log_start(*, task_id: str) -> None:
    payload = json.dumps({"task": task_id, "env": BENCHMARK, "model": MODEL_NAME}, ensure_ascii=False)
    print(f"[START] {payload}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    payload = json.dumps(
        {"step": step, "action": action, "reward": round(reward, 4), "done": done, "error": error},
        ensure_ascii=False,
    )
    print(f"[STEP] {payload}", flush=True)


def log_end(*, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    payload = json.dumps(
        {
            "success": success,
            "steps": steps,
            "score": round(score, 4),
            "rewards": [round(r, 4) for r in rewards],
        },
        ensure_ascii=False,
    )
    print(f"[END] {payload}", flush=True)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    action_table = "\n".join(
        f"  {aid}: {name}" for aid, name in ACTION_NAMES.items()
    )
    return f"""You are an AI agent repairing a broken CI/CD pipeline.
At each step you receive an observation and must choose exactly ONE action.

Available actions (by ID):
{action_table}

Destructive actions (penalised if wrong): 5=rollback, 7=ignore_continue

Rules:
- Read the failure_log and error_type carefully.
- If memory_hints are present about ABI/compiler issues, prefer use_memory_fix (6).
- If the log shows a version/cache conflict, clear_cache (4) first then change_version (2).
- If the log shows a missing Python module, use install_dependency (1).
- Avoid destructive actions unless certain.

Respond ONLY with a JSON object: {{"action_id": <integer 0-7>}}
No explanation. No markdown. Just the JSON.
"""


def build_user_message(obs: Observation, step: int, history: list[str]) -> str:
    history_block = "\n".join(history[-5:]) if history else "None"
    hints_block   = "\n".join(obs.memory_hints) if obs.memory_hints else "None"
    return f"""--- Step {step} ---
pipeline_stage : {obs.pipeline_stage}
error_type     : {obs.error_type}
progress_pct   : {obs.progress_pct:.0%}
memory_hints   :
{hints_block}

failure_log (last 30 lines):
{obs.failure_log.strip()}

recent_history :
{history_block}

Choose action_id (0-7). Respond with JSON only: {{"action_id": <n>}}"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, system: str, user: str) -> int:
    """Call the LLM and parse an action_id from the response. Returns -1 on failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.0,       # deterministic
            max_tokens=32,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()

        # Try direct JSON parse
        try:
            data = json.loads(raw)
            return int(data["action_id"])
        except Exception:
            pass

        # Fallback: regex extraction
        match = re.search(r'"?action_id"?\s*:\s*(\d)', raw)
        if match:
            return int(match.group(1))

        print(f"[DEBUG] Could not parse action from LLM response: {raw!r}", flush=True)
        return 0   # default: restart_step

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 0


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_id: str) -> float:
    """Run one episode and return score in [0, 1]."""
    log_start(task_id=task_id)

    env = CICDRepairEnv()
    obs = env.reset(task_id)

    system_prompt = build_system_prompt()
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.1
    success = False

    try:
        for step in range(1, MAX_STEPS + 1):
            if env.state().done:
                break

            user_msg = build_user_message(obs, step, history)
            action_id = call_llm(client, system_prompt, user_msg)

            # Clamp to valid range
            action_id = max(0, min(7, action_id))
            action = Action(action_id=action_id)

            obs, reward, done, info = env.step(action)

            action_name = ACTION_NAMES[action_id]
            error: Optional[str] = None

            # With the new server logic, 'reward' is already the incremental normalized delta
            incremental_reward = reward
            episode_score = compute_episode_score(env.state())
            
            rewards.append(incremental_reward)
            steps_taken = step
            history.append(f"Step {step}: {action_name} -> reward {incremental_reward:+.4f}")

            log_step(step=step, action=action_name, reward=incremental_reward, done=done, error=error)

            if done:
                break

        score = compute_episode_score(env.state())
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or OPENAI_API_KEY not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"\n{'='*55}", flush=True)
    print(f"  CICDRepairEnv — LLM Inference  |  model: {MODEL_NAME}", flush=True)
    print(f"{'='*55}", flush=True)

    results: dict[str, float] = {}
    for task_id in TASKS:
        print(f"\n--- Running task: {task_id} ---", flush=True)
        score = run_episode(client, task_id)
        results[task_id] = score
        print(f"  Score ({task_id}): {score:.4f}", flush=True)
        time.sleep(0.5)   # brief pause between tasks

    avg = sum(results.values()) / len(results)
    results["average"] = round(avg, 4)

    print(f"\n{'='*55}", flush=True)
    print(f"  Final Results", flush=True)
    print(f"{'='*55}", flush=True)
    for task_id in TASKS:
        print(f"  {task_id.capitalize():8s}: {results[task_id]:.4f}", flush=True)
    print(f"  {'─'*20}", flush=True)
    print(f"  Average : {results['average']:.4f}", flush=True)
    print(f"{'='*55}\n", flush=True)


if __name__ == "__main__":
    main()
