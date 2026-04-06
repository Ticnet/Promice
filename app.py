"""
app.py — Gradio web UI for CICDRepairEnv (HuggingFace Spaces deployment).

Allows judges and users to interact with the environment directly in a browser:
- Select task tier
- Toggle stochastic mode and procedural log generation
- Read CI/CD failure log and observation
- Choose actions
- See live reward and progress
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from env import CICDRepairEnv, Action, StochasticConfig
from env.models import ACTION_NAMES, DESTRUCTIVE_ACTION_IDS
from grader import grade_all
from run_baseline import baseline_agent

# ---------------------------------------------------------------------------
# Shared state (Gradio State per session)
# ---------------------------------------------------------------------------

def make_fresh_state():
    return {"env": CICDRepairEnv(), "history": [], "total_reward": 0.0, "done": False}

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def reset_episode(task_id: str, sigma: float, procedural: bool, session: dict) -> tuple:
    stochastic = StochasticConfig(sigma=sigma) if sigma > 0 else None
    env = CICDRepairEnv(stochastic=stochastic)
    obs = env.reset(task_id, procedural=procedural)
    session["env"] = env
    session["history"] = []
    session["total_reward"] = 0.0
    session["done"] = False
    return _render(obs, session, last_reward=None, last_action=None)


def take_action(action_id_str: str, session: dict) -> tuple:
    env: CICDRepairEnv = session["env"]

    if session["done"]:
        obs = env.state()
        return _render_state(obs, session, None, None)

    action_id = int(action_id_str.split(":")[0].strip())
    action = Action(action_id=action_id)

    obs, reward, done, info = env.step(action)
    session["total_reward"] += reward
    session["done"] = done

    action_name = ACTION_NAMES[action_id]
    destructive = " DESTRUCTIVE" if action_id in DESTRUCTIVE_ACTION_IDS else ""
    intermittent = " INTERMITTENT FAILURE" if info.get("intermittent_failure") else ""
    hist_line = (
        f"Step {obs.step_count}: **{action_name}** {destructive}{intermittent} -> "
        f"reward={reward:+.4f} | cumulative={session['total_reward']:.4f}"
    )
    session["history"].append(hist_line)

    return _render(obs, session, last_reward=reward, last_action=action_name)


def run_baseline_all(session: dict) -> str:
    results = grade_all(baseline_agent)
    lines = ["### Baseline Agent Results\n"]
    for k, v in results.items():
        bar = "X" * int(v * 20)
        label = k.replace("tier_", "Tier ").capitalize()
        lines.append(f"**{label:10s}**: `{v:.4f}`  `{bar}`")
    return "\n".join(lines)


def _render(obs, session: dict, last_reward, last_action: str | None) -> tuple:
    env: CICDRepairEnv = session["env"]
    state = env.state()
    sigma = env.stochastic_config.sigma

    status_icon = "Pipeline Healthy!" if obs.pipeline_healthy else (
        "Episode Failed (timeout)" if session["done"] and not obs.pipeline_healthy
        else "Repairing..."
    )

    # Progress bar
    pct = int(obs.progress_pct * 20)
    progress_bar = "#" * pct + "-" * (20 - pct)

    mode_label = f"sigma={sigma:.2f}" if sigma > 0 else "Deterministic"

    info_md = f"""### Episode Status
| Field | Value |
|---|---|
| **Stage** | `{obs.pipeline_stage}` |
| **Error Type** | `{obs.error_type}` |
| **Step** | {obs.step_count} / {state.max_steps} |
| **Progress** | `{progress_bar}` {obs.progress_pct:.0%} |
| **Cumulative Reward** | `{session['total_reward']:.4f}` |
| **Mode** | {mode_label} |
| **Status** | {status_icon} |
"""

    hints_md = ""
    if obs.memory_hints:
        hints_md = "### Memory Hints\n" + "\n".join(f"- {h}" for h in obs.memory_hints)
    else:
        hints_md = "### Memory Hints\n*None for this task.*"

    last_action_md = ""
    if last_action is not None:
        reward_color = "[+]" if last_reward and last_reward > 0 else ("[-]" if last_reward and last_reward < 0 else "[ ]")
        last_action_md = f"**Last action:** `{last_action}` -> {reward_color} reward = `{last_reward:+.4f}`"

    history_md = "### History\n"
    history_md += "\n".join(session["history"]) if session["history"] else "*No actions taken yet.*"

    return (
        obs.failure_log,
        info_md,
        hints_md,
        last_action_md,
        history_md,
        session,
    )


def _render_state(state, session, last_reward, last_action):
    """Fallback when episode is already done."""
    return (
        state.failure_log,
        "### Episode already finished. Press **Reset** to start a new episode.",
        "",
        "",
        "\n".join(session["history"]),
        session,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

ACTION_CHOICES = [f"{aid}: {name}" for aid, name in ACTION_NAMES.items()]

TIER_DESCRIPTIONS = {
    "tier_1": "Tier 1 - Single-Step Dependency Resolution (ModuleNotFoundError)",
    "tier_2": "Tier 2 - Multi-Step State Manipulation (cache/version conflict)",
    "tier_3": "Tier 3 - Memory-Augmented Patching (ABI mismatch, memory fix required)",
}

with gr.Blocks(
    title="CICDRepairEnv",
) as demo:

    session_state = gr.State(make_fresh_state)

    gr.Markdown("""
# CICDRepairEnv
**A CI/CD pipeline repair RL environment with stochastic and procedural modes.**
Select a tier, configure noise (sigma), and choose repair actions to fix the pipeline.
Earn rewards for correct actions; destructive actions lose points.
""")

    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(
                choices=list(TIER_DESCRIPTIONS.keys()),
                value="tier_1",
                label="Task Tier",
                info="\n".join(TIER_DESCRIPTIONS.values()),
            )

            with gr.Row():
                sigma_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                    label="Stochastic σ (0 = deterministic)",
                )
                procedural_cb = gr.Checkbox(
                    label="Procedural Logs",
                    value=False,
                    info="Generate fresh randomised logs per episode",
                )

            reset_btn = gr.Button("Reset Episode", variant="primary")

            gr.Markdown("### Choose Action")
            action_dropdown = gr.Dropdown(
                choices=ACTION_CHOICES,
                value=ACTION_CHOICES[0],
                label="Action",
            )
            step_btn = gr.Button("▶️ Take Action", variant="secondary")

            gr.Markdown("---")
            baseline_btn = gr.Button("Run Baseline Agent (all tiers)", variant="stop")
            baseline_output = gr.Markdown("*Click above to run the rule-based baseline.*")

        with gr.Column(scale=2):
            log_display = gr.Textbox(
                label="CI/CD Failure Log",
                lines=18,
                interactive=False,
            )
            info_display    = gr.Markdown()
            hints_display   = gr.Markdown()
            action_feedback = gr.Markdown()
            history_display = gr.Markdown()

    # ---- Wire events ----

    reset_btn.click(
        fn=reset_episode,
        inputs=[task_dropdown, sigma_slider, procedural_cb, session_state],
        outputs=[log_display, info_display, hints_display, action_feedback, history_display, session_state],
    )

    step_btn.click(
        fn=take_action,
        inputs=[action_dropdown, session_state],
        outputs=[log_display, info_display, hints_display, action_feedback, history_display, session_state],
    )

    baseline_btn.click(
        fn=run_baseline_all,
        inputs=[session_state],
        outputs=[baseline_output],
    )

    # Auto-reset on load
    demo.load(
        fn=reset_episode,
        inputs=[task_dropdown, sigma_slider, procedural_cb, session_state],
        outputs=[log_display, info_display, hints_display, action_feedback, history_display, session_state],
    )

    gr.Markdown("""
---
### Quick Reference - Action Space
| ID | Action | Destructive? |
|---|---|---|
| 0 | restart_step | No |
| 1 | install_dependency | No |
| 2 | change_version | No |
| 3 | set_env_variable | No |
| 4 | clear_cache | No |
| 5 | rollback | Yes (-0.10) |
| 6 | use_memory_fix | No |
| 7 | ignore_continue | Yes (-0.10) |

### Stochastic Events (sigma > 0)
| Event | Effect |
|---|---|
| Intermittent Failure | Correct action may transiently fail (no progress, no penalty) |
| Action Corruption | Executed action may differ from requested (flaky CI runner) |
| Log Noise | Irrelevant warning lines injected into failure log |

**Max reward per episode: 1.0** | Built for OpenEnv
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    )
