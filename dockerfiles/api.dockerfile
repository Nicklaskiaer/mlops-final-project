FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/hf_cache

# 'libgomp1' fixes your specific error (OpenMP for PyTorch)
# 'libsndfile1' is required by Torchaudio/Librosa to read .wav files
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN uv sync --frozen --no-install-project

COPY src src/
COPY models/checkpoints models/checkpoints/

COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.project.api:app", "--host", "0.0.0.0", "--port", "8080"]