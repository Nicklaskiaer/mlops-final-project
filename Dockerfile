FROM ghcr.io/astral-sh/uv:python3.11-alpine AS base

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

RUN uv sync --frozen --no-install-project

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "src.project.api:app", "--host", "0.0.0.0", "--port", "8000"]