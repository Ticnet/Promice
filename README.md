# CICDRepairEnv: A Deterministic RL Environment for Automated Pipeline Fault Resolution

CICDRepairEnv is a rigorously structured, deterministic Reinforcement Learning (RL) environment designed to evaluate the sequential decision-making capabilities of autonomous agents—both LLM-based and rule-based—in diagnosing and repairing Continuous Integration/Continuous Deployment (CI/CD) pipeline failures.

Unlike stochastic environments, CICDRepairEnv is entirely deterministic: identical state-action sequences yield identical transitions and rewards. This ensures zero-variance benchmarking for agent reasoning, log comprehension, and memory exploitation.

---

## 1. Theoretical Formulation

The environment models pipeline repair as a discrete-time Markov Decision Process (MDP).

- **State Space ($S$):** A structured observation containing the current pipeline stage, raw failure logs, categorical error types, memory hints, and progress metrics.
- **Action Space ($A$):** A discrete set of 8 distinct remediation actions.
- **Reward Function ($R$):** A dense, step-wise reward function clamped to $R \in [0.0, 1.0]$, prioritizing task completion, path efficiency, and penalizing destructive mutations.

---

## 2. Evaluation Capabilities

Agents evaluated in this environment must demonstrate proficiency across several axes of intelligence:

| Competency Axis | Evaluation Metric |
| :--- | :--- |
| **Diagnostic Comprehension** | Parsing unstructured, multi-line stack traces to identify root causes. |
| **Sequential Logic** | Executing interdependent actions in a strict prerequisite order. |
| **Memory Exploitation** | Retrieving and applying external patches from an isolated memory bank. |
| **Risk Aversion** | Avoiding destructive state mutations (e.g., premature rollbacks). |
| **Path Optimization** | Achieving pipeline health in the mathematical minimum number of steps. |

---

## 3. Architecture & State Flow

The system architecture enforces strict isolation between the agent policy and the environment state machine.

```text
┌─────────────────────────────────────────────────────┐
│                   CICDRepairEnv                     │
│                                                     │
│  ┌─────────┐  reset(task_id)   ┌──────────────┐     │
│  │  Agent  │ ────────────────▶│  Environment │     │
│  │ Policy  │ ◀────────────────│  State       │     │
│  │         │   Observation     │  Machine     │     │
│  │         │                   │              │     │
│  │         │ ──── Action ────▶│  step()      │     │
│  │         │ ◀── obs, reward, │  Reward Calc │     │
│  │         │     done, info ── │              │     │
│  └─────────┘                   └──────────────┘     │
└─────────────────────────────────────────────────────┘
```

---

## 4. Environment Dynamics

### 4.1 Action Space ($A$)

The agent interacts via a discrete integer action space $a \in \{0, 1, ..., 7\}$.

| `a` | Action Concept | Mutates State? | Destructive Penalty |
| :--- | :--- | :--- | :--- |
| 0 | `restart_step` | Yes | No |
| 1 | `install_dependency` | Yes | No |
| 2 | `change_version` | Yes | No |
| 3 | `set_env_variable` | Yes | No |
| 4 | `clear_cache` | Yes | No |
| 5 | `rollback` | Yes | Yes (−0.10) |
| 6 | `use_memory_fix` | Yes | No |
| 7 | `ignore_continue` | Yes | Yes (−0.10) |

### 4.2 Observation Space ($S$)

At each timestep $t$, the environment emits an observation state $s_t$:

- `pipeline_stage` *(str)*: Active execution phase (e.g., `dependency_resolution`).
- `failure_log` *(str)*: Raw stdout/stderr traces.
- `error_type` *(str)*: Abstracted error classification.
- `available_actions` *(list[int])*: Valid action mask.
- `memory_hints` *(list[str])*: Contextual vectors for required patches.
- `step_count` *(int)*: Current timestep $t$.
- `pipeline_healthy` *(bool)*: Terminal success flag.
- `progress_pct` *(float)*: Normalized completion scalar $[0.0, 1.0]$.

### 4.3 Reward Formulation

The environment provides a dense reward signal designed to guide policy optimization and accurately score LLM-based agents. Total cumulative reward is strictly bounded.

$$R_{total} = \min\left(1.0,\ \max\left(0.0,\ \sum \left(R_{root} + R_{prog} + R_{eff} + R_{term} - P_{dest}\right)\right)\right)$$

- **Root Cause Identification** ($R_{root}$ = +0.20): Awarded upon the first state-appropriate action.
- **State Progression** ($R_{prog}$ = +0.30): Distributed fractionally across valid sequential steps.
- **Path Efficiency** ($R_{eff}$ = +0.10): Retained only if $\Delta t = t_{optimal}$ with zero invalid actions.
- **Terminal Success** ($R_{term}$ = +0.40): Awarded upon achieving `pipeline_healthy = True`.
- **Destructive Penalty** ($P_{dest}$ = 0.10): Subtracted for reckless actions (`rollback`, `ignore_continue`).

---

## 5. Benchmark Suite

The environment provides pre-configured failure topologies of increasing structural complexity.

### Tier 1: Single-Step Dependency Resolution

- **Failure Vector:** `ModuleNotFoundError: No module named 'sklearn'`
- **Target Stage:** `install_packages`
- **Optimal Policy:** `[install_dependency]`
- **Complexity:** 1 step. Tests basic log-to-action mapping.

### Tier 2: Multi-Step State Manipulation

- **Failure Vector:** Cache conflict between `torch==2.0.0` and required `torch==2.1.0`.
- **Target Stage:** `dependency_resolution`
- **Optimal Policy:** `[clear_cache]` → `[change_version]`
- **Complexity:** 2 steps. Tests strict sequential ordering; reversing the order yields zero progression.

### Tier 3: Memory-Augmented Patching

- **Failure Vector:** Linker failure due to GCC 9 vs GCC 11 ABI mismatch.
- **Target Stage:** `native_extension_build`
- **Optimal Policy:** `[use_memory_fix]`
- **Complexity:** Tests the agent's ability to recognize unsolvable local states and exploit external memory contexts. Standard actions deterministically fail.

---

## 6. Installation & Execution

### 6.1 Standard Installation

It is recommended to use an isolated virtual environment.

```bash
git clone https://github.com/your-org/cicd_repair_env.git
cd cicd_repair_env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Containerized Execution (Recommended)

To ensure absolute host isolation, especially during LLM inference evaluations:

```bash
docker build -t cicd-repair-env .
```

Run the Gradio visualizer interface locally:

```bash
docker run --rm -p 7860:7860 cicd-repair-env
```

Execute an LLM-based agent evaluation:

```bash
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o" \
  -e HF_TOKEN="your_api_key_here" \
  cicd-repair-env python inference.py
```

---

## 7. Agent Integration API

Integrating custom agents requires implementing a policy function that maps the Observation space to the Action space.

```python
from env import CICDRepairEnv, Action
from env.models import Observation, EnvironmentState

def optimized_policy(obs: Observation, state: EnvironmentState) -> Action:
    """
    Evaluates the current state observation and returns the optimal corrective action.
    """
    log_context = obs.failure_log.lower()

    # Example heuristic rule
    if "modulenotfounderror" in log_context:
        return Action(action_id=1)

    return Action(action_id=0)  # Fallback

# Evaluation execution
env = CICDRepairEnv()
obs = env.reset("tier_1")
done = False

while not done:
    action = optimized_policy(obs, env.state)
    obs, reward, done, info = env.step(action)

print(f"Final Execution Score: {info['cumulative_reward']}")
```


