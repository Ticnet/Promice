# syntax=docker/dockerfile:1
# HuggingFace Spaces compatible Dockerfile
# Serves the Gradio UI on port 7860 (HF Spaces default)
FROM python:3.11-slim

LABEL maintainer="CICDRepairEnv"
LABEL description="Deterministic RL environment for CI/CD pipeline repair - OpenEnv submission"

# HuggingFace Spaces requires uid 1000
RUN groupadd -g 1000 cicd && useradd -u 1000 -g cicd -ms /bin/bash cicd

WORKDIR /app

# Install deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Verify core imports at build time
RUN python -c "from env import CICDRepairEnv, Action, Observation, EnvironmentState; env = CICDRepairEnv(); obs = env.reset('tier_1'); print('env imports OK'); print('reset stage:', obs.pipeline_stage)"

# Verify inference script is importable
RUN python -c "import inference; print('inference.py importable')"

# Expose Gradio port
EXPOSE 7860

# Switch to non-root
USER cicd

# Default: run Gradio UI (HF Spaces)
# For CLI baseline: docker run ... python run_baseline.py
# For inference:    docker run -e HF_TOKEN=xxx -e MODEL_NAME=gpt-4o-mini -e API_BASE_URL=https://api.openai.com/v1 ... python inference.py
CMD ["python", "server/app.py"]
