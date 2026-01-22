# 1. Use the same Debian-based image as the API
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Hydra often needs to know where it is running
    HYDRA_FULL_ERROR=1

# 2. Install System Dependencies (Same as API)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Python Dependencies
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN uv sync --frozen --no-install-project

# 4. Copy Code & Configs
COPY src src/
COPY configs configs/
# Note: We do NOT copy 'models/' because the trainer CREATES the model
# Note: We do NOT copy 'data/' because we will mount it dynamically

# 5. Metadata
COPY README.md README.md
COPY LICENSE LICENSE

# 6. Install Project
RUN uv sync --frozen

# 7. Default Command
# This runs the training script when the container starts
ENTRYPOINT ["uv", "run", "src/project/train.py"]