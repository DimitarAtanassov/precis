# syntax=docker/dockerfile:1.9
# Multi-stage build: dependencies are resolved with uv in a builder stage, then
# only the virtualenv + source are copied into a slim, non-root runtime image.

FROM python:3.13-slim-bookworm AS builder

# Pinned uv binary for reproducible, fast installs.
COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Copy only what is needed to build the package and resolve the locked deps.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install runtime dependencies (no dev/lint/test groups) from the frozen lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.13-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PRECIS_LOG_JSON=true

WORKDIR /app

# Run as an unprivileged user.
RUN groupadd --system app && useradd --system --gid app --home-dir /app app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY pyproject.toml README.md ./

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["uvicorn", "precis.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
