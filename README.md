# CICDRepairEnv: An RL Environment for Automated Pipeline Fault Resolution

CICDRepairEnv is a rigorously structured Reinforcement Learning (RL) environment designed to evaluate the sequential decision-making capabilities of autonomous agents - both LLM-based and rule-based - in diagnosing and repairing Continuous Integration/Continuous Deployment (CI/CD) pipeline failures.

The environment supports **two operational modes**:

| Mode | Description |
|---|---|
| **Deterministic** (sigma = 0) | Identical state-action sequences yield identical transitions and rewards. Zero-variance benchmarking. |
| **Stochastic** (sigma > 0) | Introduces real-world CI/CD noise: intermittent failures, action corruption, and log noise injection. Scaled by the master noise parameter sigma in [0, 1]. |

**Procedural log generation** prevents agent overfitting to static strings by randomising package names, versions, timestamps, and file paths while preserving semantic error markers.

---

## 1. Theoretical Formulation

The environment models pipeline repair as a discrete-time Markov Decision Process (MDP).

- **State Space ($S$):** A structured observation containing the current pipeline stage, raw failure logs, categorical error types, memory hints, and progress metrics.
- **Action Space ($A$):** A discrete set of 8 distinct remediation actions.
- **Transition Function (T):** Parameterised by a stochastic noise scale sigma in [0,1]. When sigma = 0, T is fully deterministic. When sigma > 0, events are sampled as:
  - Intermittent failure: P_fail = sigma * p_intermittent
  - Action corruption: P_corrupt = sigma * p_corruption
  - Log noise injection: P_noise = sigma * p_log_noise
- **Reward Function ($R$):** A dense, configurable reward function (via `RewardConfig`) clamped to $R \in [0.0, 1.0]$.

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
| **Noise Robustness** | Maintaining correct decisions despite stochastic action corruption and log noise. |

---

## 3. Architecture & State Flow

```text
┌─────────────────────────────────────────────────────────────┐
│                      CICDRepairEnv                          │
│                                                             │
│  ┌─────────┐  reset(tier_id)    ┌──────────────────┐        │
│  │  Agent  │ ────────────────── ▶│  Environment     │        │
│  │ Policy  │ ◀────────────────── │  State Machine   │        │
│  │         │   Observation      │                  │        │
│  │         │                    │  StochasticConfig│        │
│  │         │ ──── Action ─────▶ │  sigma-gated events  │        │
│  │         │ ◀── obs, reward,   │  RewardConfig    │        │
│  │         │     done, info ──  │  Procedural Gen  │        │
│  └─────────┘                    └──────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Environment Dynamics

### 4.1 Action Space ($A$)

| `a` | Action Concept | Mutates State? | Destructive Penalty |
| :--- | :--- | :--- | :--- |
| 0 | `restart_step` | Yes | No |
| 1 | `install_dependency` | Yes | No |
| 2 | `change_version` | Yes | No |
| 3 | `set_env_variable` | Yes | No |
| 4 | `clear_cache` | Yes | No |
| 5 | `rollback` | Yes | Yes (-0.10) |
| 6 | `use_memory_fix` | Yes | No |
| 7 | `ignore_continue` | Yes | Yes (-0.10) |

### 4.2 Observation Space ($S$)

At each timestep $t$, the environment emits an observation state $s_t$:

- `pipeline_stage` *(str)*: Active execution phase (e.g., `dependency_resolution`).
- `failure_log` *(str)*: Raw stdout/stderr traces (static or procedurally generated).
- `error_type` *(str)*: Abstracted error classification.
- `available_actions` *(list[int])*: Valid action mask.
- `memory_hints` *(list[str])*: Contextual vectors for required patches.
- `step_count` *(int)*: Current timestep $t$.
- `pipeline_healthy` *(bool)*: Terminal success flag.
- `progress_pct` *(float)*: Normalized completion scalar $[0.0, 1.0]$.

### 4.3 Reward Formulation

The reward function is configurable via `RewardConfig`. Default weights:

$$R_{total} = \min\left(1.0,\ \max\left(0.0,\ \sum \left(R_{root} + R_{prog} + R_{eff} + R_{term} - P_{dest}\right)\right)\right)$$

| Component | Symbol | Default | Condition |
|---|---|---|---|
| Root Cause Identification | $R_{root}$ | +0.20 | First state-appropriate action |
| State Progression | $R_{prog}$ | +0.30 | Distributed across valid sequential steps |
| Path Efficiency | $R_{eff}$ | +0.10 | $\Delta t = t_{optimal}$ with zero invalid actions |
| Terminal Success | $R_{term}$ | +0.40 | `pipeline_healthy = True` |
| Destructive Penalty | $P_{dest}$ | 0.10 | `rollback` or `ignore_continue` used incorrectly |

### 4.4 Stochastic Transition Events (sigma > 0)

| Event | Default Base Prob | Effective Prob | Effect |
|---|---|---|---|
| Intermittent Failure | 0.05 | sigma * 0.05 | Correct action transiently fails (no progress, no penalty) |
| Action Corruption | 0.02 | sigma * 0.02 | Executed action silently differs from requested |
| Log Noise Injection | 0.10 | sigma * 0.10 | Irrelevant warning lines appended to failure log |

---

## 5. Benchmark Suite

Tasks are organised into Tiers of increasing structural complexity. This design scales naturally - adding new failure classes requires only defining a new tier.

### Tier 1: Single-Step Dependency Resolution

- **Failure Vector:** `ModuleNotFoundError: No module named '<package>'`
- **Target Stage:** `install_packages`
- **Optimal Policy:** `[install_dependency]`
- **Complexity:** 1 step. Tests basic log-to-action mapping.

### Tier 2: Multi-Step State Manipulation

- **Failure Vector:** Cache conflict between cached and required package versions.
- **Target Stage:** `dependency_resolution`
- **Optimal Policy:** `[clear_cache]` -> `[change_version]`
- **Complexity:** 2 steps. Tests strict sequential ordering.

### Tier 3: Memory-Augmented Patching

- **Failure Vector:** Linker failure due to GCC ABI mismatch.
- **Target Stage:** `native_extension_build`
- **Optimal Policy:** `[use_memory_fix]`
- **Complexity:** Tests ability to exploit external memory contexts.

### Tier 4: Distributed State Reconciliation *(Planned)*

- **Failure Vector:** Kubernetes pod deployment failures, container registry auth errors, service mesh configuration drift.
- **Target Stage:** `container_orchestration`
- **Complexity:** Multi-agent coordination across distributed state.

---

## 6. Installation & Execution

### 6.1 Standard Installation

```bash
git clone https://github.com/your-org/cicd_repair_env.git
cd cicd_repair_env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Containerized Execution (Recommended)

```bash
docker build -t cicd-repair-env .
docker run --rm -p 7860:7860 cicd-repair-env
```

### 6.3 Stochastic Training Mode

```python
from env import CICDRepairEnv, StochasticConfig

cfg = StochasticConfig(sigma=0.3, seed=42)
env = CICDRepairEnv(stochastic=cfg)
obs = env.reset("tier_2", procedural=True)

while True:
    action = agent(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

---

## 7. Agent Integration API

```python
from env import CICDRepairEnv, Action, StochasticConfig, RewardConfig
from env.models import Observation, EnvironmentState

# Deterministic mode (backward-compatible)
env = CICDRepairEnv()
obs = env.reset("tier_1")    # or "easy" (legacy alias)

# Stochastic mode with procedural logs
env = CICDRepairEnv(
    stochastic=StochasticConfig(sigma=0.5, seed=42),
    reward_config=RewardConfig(root_cause_bonus=0.25),
)
obs = env.reset("tier_2", procedural=True)

done = False
while not done:
    action = optimized_policy(obs, env.state())
    obs, reward, done, info = env.step(action)

print(f"Final Execution Score: {info['cumulative_reward']}")
```

---

## 8. Backward Compatibility

Legacy task IDs are fully supported as aliases:

| Legacy ID | Canonical ID |
|---|---|
| `"easy"` | `"tier_1"` |
| `"medium"` | `"tier_2"` |
| `"hard"` | `"tier_3"` |

`CICDRepairEnv()` with no arguments produces **exactly the same** deterministic behaviour as the original implementation.
