"""
app.py — Unified FastAPI + Gradio server for CICDRepairEnv.

Serves:
  1.  Gradio UI (Interactive browser access) at /
  2.  Headless RL API (Automated validation) at /reset, /step, /state
"""

from __future__ import annotations

import os
import sys
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict

from fastapi import FastAPI, Request, Body, HTTPException, Query
from fastapi.responses import JSONResponse
import gradio as gr
import uvicorn

# Ensure root components are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import CICDRepairEnv, Action, StochasticConfig, compute_episode_score
from env.models import ACTION_NAMES, DESTRUCTIVE_ACTION_IDS
from grader import grade_all
from run_baseline import baseline_agent

# ---------------------------------------------------------------------------
# Session Registry (replaces the old _SINGLETON_ENV)
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """An isolated environment instance bound to a session token."""
    env: CICDRepairEnv
    last_score: float = 0.01
    created_at: float = field(default_factory=time.time)

_SESSIONS: Dict[str, Session] = {}
_MAX_SESSIONS = 100  # hard cap to prevent memory leaks


def _create_session(env: CICDRepairEnv) -> str:
    """Register a new session.  Evicts the oldest if at capacity."""
    if len(_SESSIONS) >= _MAX_SESSIONS:
        oldest_id = min(_SESSIONS, key=lambda k: _SESSIONS[k].created_at)
        del _SESSIONS[oldest_id]
    session_id = uuid.uuid4().hex[:12]
    _SESSIONS[session_id] = Session(env=env)
    return session_id


def _get_session(session_id: str) -> Session:
    """Look up a session or raise 404."""
    if session_id not in _SESSIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    return _SESSIONS[session_id]


# ---------------------------------------------------------------------------
# FastAPI Endpoints (OpenEnv Headless API)
# ---------------------------------------------------------------------------

app = FastAPI(title="CICDRepairEnv API")

@app.post("/reset")
async def reset_api(payload: dict = Body({})):
    """Reset the environment via API.  Returns a new session_id."""
    task_id = payload.get("task_id", "tier_1")
    sigma = payload.get("sigma", 0.0)

    stochastic = StochasticConfig(sigma=sigma) if sigma > 0 else None
    env = CICDRepairEnv(stochastic=stochastic)
    obs = env.reset(task_id)

    session_id = _create_session(env)
    return {"session_id": session_id, **obs.model_dump()}

@app.post("/step")
async def step_api(payload: dict = Body(...)):
    """Apply an action via API.  Requires session_id."""
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing 'session_id' in payload.")

    session = _get_session(session_id)
    try:
        action_id = int(payload["action_id"])
        action = Action(action_id=action_id)
        obs, _raw_reward, done, info = session.env.step(action)

        current_score = compute_episode_score(session.env.state())
        incremental_reward = round(current_score - session.last_score, 4)
        session.last_score = current_score

        return {
            "session_id": session_id,
            "observation": obs.model_dump(),
            "reward": incremental_reward,
            "done": done,
            "info": {
                **info,
                "cumulative_reward": current_score,
                "incremental_reward": incremental_reward
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/state")
async def state_api(session_id: str = Query(...)):
    """Get internal state via API.  Requires session_id query param."""
    session = _get_session(session_id)
    try:
        st = session.env.state()
        state_data = st.model_dump()
        state_data["cumulative_reward"] = compute_episode_score(st)
        return {"session_id": session_id, **state_data}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/health")
async def health():
    return {"status": "ok", "benchmark": "CICDRepairEnv"}

# ---------------------------------------------------------------------------
# Gradio UI (Browser Interface)
# ---------------------------------------------------------------------------

def make_fresh_state():
    return {"env": CICDRepairEnv(), "history": [], "total_reward": 0.0, "done": False}

def reset_episode(task_id: str, sigma: float, session: dict) -> tuple:
    stochastic = StochasticConfig(sigma=sigma) if sigma > 0 else None
    env = CICDRepairEnv(stochastic=stochastic)
    obs = env.reset(task_id)
    session["env"] = env
    session["history"] = []
    session["total_reward"] = 0.0
    session["done"] = False
    return _render(obs, session, last_reward=None, last_action=None)

def take_action(action_id_str: str, session: dict) -> tuple:
    env: CICDRepairEnv = session["env"]
    if session["done"]:
        return _render_state(env.state(), session, None, None)

    action_id = int(action_id_str.split(":")[0].strip())
    action = Action(action_id=action_id)
    obs, reward, done, info = env.step(action)
    session["total_reward"] += reward
    session["done"] = done

    action_name = ACTION_NAMES[action_id]
    destructive = " DESTRUCTIVE" if action_id in DESTRUCTIVE_ACTION_IDS else ""
    intermittent = " INTERMITTENT FAILURE" if info.get("intermittent_failure") else ""
    hist_line = f"Step {obs.step_count}: **{action_name}** {destructive}{intermittent} -> reward={reward:+.4f}"
    session["history"].append(hist_line)
    return _render(obs, session, last_reward=reward, last_action=action_name)

def run_baseline_all(session: dict) -> str:
    results = grade_all(baseline_agent)
    lines = ["### Baseline Agent Results\n"]
    for k, v in results.items():
        label = k.replace("tier_", "Tier ").replace("_", " ").capitalize()
        lines.append(f"**{label:10s}**: `{v:.4f}`")
    return "\n".join(lines)

def _render(obs, session: dict, last_reward, last_action: str | None) -> tuple:
    env: CICDRepairEnv = session["env"]
    state = env.state()
    pct = int(obs.progress_pct * 20)
    progress_bar = "#" * pct + "-" * (20 - pct)
    status_icon = "Pipeline Healthy!" if obs.pipeline_healthy else ("Failed" if session["done"] else "Repairing...")

    info_md = f"""### Episode Status
| Field | Value |
|---|---|
| **Stage** | `{obs.pipeline_stage}` |
| **Error Type** | `{obs.error_type}` |
| **Progress** | `{progress_bar}` {obs.progress_pct:.0%} |
| **Cumulative Reward** | `{compute_episode_score(env.state()):.4f}` |
| **Status** | {status_icon} |
"""
    hints_md = "### Memory Hints\n" + ("\n".join(f"- {h}" for h in obs.memory_hints) if obs.memory_hints else "*None*")
    last_action_md = f"**Last action:** `{last_action}` -> reward = `{last_reward:+.4f}`" if last_action else ""
    history_md = "### History\n" + ("\n".join(session["history"]) if session["history"] else "*None*")

    return (obs.failure_log, info_md, hints_md, last_action_md, history_md, session)

def _render_state(state, session, last_reward, last_action):
    return (state.failure_log, "### Finished. Press Reset.", "", "", "\n".join(session["history"]), session)

ACTION_CHOICES = [f"{aid}: {name}" for aid, name in ACTION_NAMES.items()]
TIER_DESCRIPTIONS = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}

with gr.Blocks(title="CICDRepairEnv", theme=gr.themes.Soft()) as demo:
    session_state = gr.State(make_fresh_state)
    gr.Markdown("# CICDRepairEnv\nPipeline repair RL environment.")
    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(choices=list(TIER_DESCRIPTIONS.keys()), value="easy", label="Tier")
            sigma_slider = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Stochastic σ")
            reset_btn = gr.Button("Reset Episode", variant="primary")
            action_dropdown = gr.Dropdown(choices=ACTION_CHOICES, value=ACTION_CHOICES[0], label="Action")
            step_btn = gr.Button("▶️ Take Action", variant="secondary")
            baseline_btn = gr.Button("Run Baseline Agent", variant="stop")
            baseline_output = gr.Markdown()
        with gr.Column(scale=2):
            log_display = gr.Textbox(label="CI/CD Log", lines=18, interactive=False)
            info_display = gr.Markdown()
            hints_display = gr.Markdown()
            action_feedback = gr.Markdown()
            history_display = gr.Markdown()

    reset_btn.click(reset_episode, [task_dropdown, sigma_slider, session_state], [log_display, info_display, hints_display, action_feedback, history_display, session_state])
    step_btn.click(take_action, [action_dropdown, session_state], [log_display, info_display, hints_display, action_feedback, history_display, session_state])
    baseline_btn.click(run_baseline_all, [session_state], [baseline_output])
    demo.load(reset_episode, [task_dropdown, sigma_slider, session_state], [log_display, info_display, hints_display, action_feedback, history_display, session_state])

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    """Entry point for the server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
