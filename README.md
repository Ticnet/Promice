---
title: CICDRepairEnv
emoji: 🔧
colorFrom: indigo
colorTo: gray
sdk: docker
app_file: server/app.py
pinned: false
tags:
  - openenv
---

# CICDRepairEnv: An RL Environment for Automated Pipeline Fault Resolution

CICDRepairEnv is a rigorously structured Reinforcement Learning (RL) environment designed to evaluate the sequential decision-making capabilities of autonomous agents - both LLM-based and rule-based - in diagnosing and repairing Continuous Integration/Continuous Deployment (CI/CD) pipeline failures.

### Motivation
In modern software engineering, CI/CD pipelines are the backbone of delivery. However, pipeline failures are frequent, often transient, and require significant human intervention to diagnose from unstructured logs. **CICDRepairEnv** fills the gap for a standardized, real-world benchmark that evaluates whether agents can understand logs, exploit external memory patches, and perform sequential repairs without human assistance.

The environment supports **two operational modes**:

| Mode | Description |
|---|---|
| **Deterministic** (sigma = 0) | Identical state-action sequences yield identical transitions and rewards. Zero-variance benchmarking. |
| **Stochastic** (sigma > 0) | Introduces real-world CI/CD noise: intermittent failures, action corruption, and log noise injection. Scaled by the master noise parameter sigma in [0, 1]. |

Procedural log generation randomises package names, versions, timestamps, and file paths at each episode, preventing agents from overfitting to static strings while preserving semantic error markers.

---

## Contents

1. [Theoretical Formulation](#1-theoretical-formulation)
2. [Evaluation Capabilities](#2-evaluation-capabilities)
3. [Architecture & State Flow](#3-architecture--state-flow)
4. [Environment Dynamics](#4-environment-dynamics)
5. [Benchmark Suite](#5-benchmark-suite)
6. [Installation & Execution](#6-installation--execution)
7. [Agent Integration API](#7-agent-integration-api)

---

## 1. Theoretical Formulation

CICDRepairEnv implements a discrete-time **Markov Decision Process (MDP)**:

| Component | Definition |
|---|---|
| **State Space** $S$ | Structured observation: pipeline stage, raw failure logs, error type, memory hints, progress metrics. |
| **Action Space** $A$ | 8 discrete remediation actions. |
| **Transition Function** $T$ | Parameterised by `sigma`. Deterministic when `sigma = 0`; stochastic events are sampled at `sigma`-scaled probabilities when `sigma > 0`. |
| **Reward Function** $R$ | Dense, configurable via `RewardConfig`. Normalised to $R \in [0.15,\ 0.85]$ for stable evaluation. |

---

## 2. Evaluation Capabilities

Agents are assessed across six axes of competency:

| Axis | Metric |
|---|---|
| **Diagnostic Comprehension** | Parsing unstructured, multi-line stack traces to identify root causes. |
| **Sequential Logic** | Executing interdependent actions in strict prerequisite order. |
| **Memory Exploitation** | Retrieving and applying patches from an isolated memory bank. |
| **Risk Aversion** | Avoiding destructive state mutations such as premature rollbacks. |
| **Path Optimisation** | Resolving pipeline health in the minimum number of steps. |
| **Noise Robustness** | Maintaining correct decisions despite action corruption and log noise. |

---

## 3. Architecture & State Flow

```
┌─────────────────────────────────────────────────────────────┐
│                       CICDRepairEnv                         │
│                                                             │
│  ┌──────────┐  reset(tier_id)   ┌──────────────────────┐    │
│  │  Agent   │ ────────────────▶│  State Machine       │    │
│  │  Policy  │ ◀────────────────│                      │    │
│  │          │   Observation     │  StochasticConfig    │    │
│  │          │                   │  sigma-gated events  │    │
│  │          │ ──── Action ────▶│  RewardConfig        │    │
│  │          │ ◀── obs, reward, │  Procedural Gen      │    │
│  │          │     done, info    │                      │    │
│  └──────────┘                   └──────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Environment Dynamics

### 4.1 Action Space

| ID | Action | Mutates State | Destructive Penalty |
|---|---|---|---|
| `0` | `restart_step` | ✓ | — |
| `1` | `install_dependency` | ✓ | — |
| `2` | `change_version` | ✓ | — |
| `3` | `set_env_variable` | ✓ | — |
| `4` | `clear_cache` | ✓ | — |
| `5` | `rollback` | ✓ | −0.10 |
| `6` | `use_memory_fix` | ✓ | — |
| `7` | `ignore_continue` | ✓ | −0.10 |

### 4.2 Observation Space

At each timestep $t$, the environment emits $s_t$ containing:

| Field | Type | Description |
|---|---|---|
| `pipeline_stage` | `str` | Active execution phase (e.g. `dependency_resolution`). |
| `failure_log` | `str` | Raw stdout/stderr traces, static or procedurally generated. |
| `error_type` | `str` | Abstracted error classification. |
| `available_actions` | `list[int]` | Valid action mask for the current state. |
| `memory_hints` | `list[str]` | Contextual vectors for required patches. |
| `step_count` | `int` | Current timestep $t$. |
| `pipeline_healthy` | `bool` | Terminal success flag. |
| `progress_pct` | `float` | Normalised completion scalar $[0.0,\ 1.0]$. |

### 4.3 Reward Function
 
 $$R_{\text{raw}} = \min\!\left(1.0,\ \max\!\left(0.0,\ R_{\text{root}} + R_{\text{prog}} + R_{\text{eff}} + R_{\text{term}} - P_{\text{dest}}\right)\right)$$
 
 $$R_{\text{final}} = 0.15 + (R_{\text{raw}} \times 0.7)$$
 
 The environment uses a **normalized scoring range** $[0.15, 0.85]$ to ensure reactivity and prevent absolute zero/unity artifacts in agent evaluation.

| Component | Symbol | Default | Condition |
|---|---|---|---|
| Root Cause Identification | $R_{\text{root}}$ | +0.20 | First state-appropriate action taken. |
| State Progression | $R_{\text{prog}}$ | +0.30 | Distributed across valid sequential steps. |
| Path Efficiency | $R_{\text{eff}}$ | +0.10 | $\Delta t = t_{\text{optimal}}$ with zero invalid actions. |
| Terminal Success | $R_{\text{term}}$ | +0.40 | `pipeline_healthy = True`. |
| Destructive Penalty | $P_{\text{dest}}$ | −0.10 | `rollback` or `ignore_continue` used incorrectly. |

All weights are overridable via `RewardConfig`.

### 4.4 Stochastic Transition Events

Active when `sigma > 0`. Effective probability scales linearly with `sigma`.

| Event | Base Probability | Effective Probability | Effect |
|---|---|---|---|
| Intermittent Failure | 0.05 | `sigma × 0.05` | Correct action transiently fails — no progress, no penalty. |
| Action Corruption | 0.02 | `sigma × 0.02` | Executed action silently differs from the requested action. |
| Log Noise Injection | 0.10 | `sigma × 0.10` | Irrelevant warning lines appended to `failure_log`. |

---

## 5. Benchmark Suite

Tasks are organised into tiers of increasing structural complexity. New failure classes can be introduced by defining a new tier.

### Tier 1: Single-Step Dependency Resolution (Difficulty: Easy)

| Property | Value |
|---|---|
| **Failure Vector** | `ModuleNotFoundError: No module named '<package>'` |
| **Target Stage** | `install_packages` |
| **Optimal Policy** | `[install_dependency]` |
| **Complexity** | 1 step — tests basic log-to-action mapping. |

### Tier 2: Multi-Step State Manipulation (Difficulty: Medium)

| Property | Value |
|---|---|
| **Failure Vector** | Cache conflict between cached and required package versions. |
| **Target Stage** | `dependency_resolution` |
| **Optimal Policy** | `[clear_cache]` → `[change_version]` |
| **Complexity** | 2 steps — tests strict sequential ordering. |

### Tier 3: Memory-Augmented Multi-Step Patching (Difficulty: Hard)

| Property | Value |
|---|---|
| **Failure Vector** | Linker failure due to GCC ABI mismatch. |
| **Target Stage** | `native_extension_build` |
| **Optimal Policy** | `[set_env_variable]` → `[clear_cache]` → `[use_memory_fix]` |
| **Complexity** | 3 steps — tests strict sequential ordering, log interpretation, and memory exploitation. |

### Tier 4 — Distributed State Reconciliation *(Planned)*

| Property | Value |
|---|---|
| **Failure Vector** | Kubernetes pod deployment failures, container registry auth errors, service mesh configuration drift. |
| **Target Stage** | `container_orchestration` |
| **Complexity** | Multi-agent coordination across distributed state. |

---

## 6. Installation & Execution

### Standard Installation

```bash
git clone https://github.com/your-org/cicd_repair_env.git
cd cicd_repair_env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Containerised Execution *(Recommended)*

```bash
docker build -t cicd-repair-env .
docker run --rm -p 7860:7860 cicd-repair-env
```

### Stochastic Training Mode

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
from env import CICDRepairEnv, Action, StochasticConfig, RewardConfig, normalize_reward
from env.models import Observation, EnvironmentState

# Deterministic mode (backward-compatible)
env = CICDRepairEnv()
obs = env.reset("tier_1")

# Stochastic mode with procedural logs and custom reward weights
env = CICDRepairEnv(
    stochastic=StochasticConfig(sigma=0.5, seed=42),
    reward_config=RewardConfig(root_cause_bonus=0.25),
)
obs = env.reset("tier_2", procedural=True)

done = False
while not done:
    action = optimized_policy(obs, env.state())
    obs, reward, done, info = env.step(action)

print(f"Final Execution Score: {normalize_reward(info['cumulative_reward']):.4f}")
```

---

## 8. Backward Compatibility

Legacy task IDs are fully supported as aliases during environment reset:

| Legacy ID | Canonical ID |
|---|---|
| `"easy"` | `"tier_1"` |
| `"medium"` | `"tier_2"` |
| `"hard"` | `"tier_3"` |

`CICDRepairEnv()` with no arguments produces **exactly the same** deterministic behaviour as the original implementation.
