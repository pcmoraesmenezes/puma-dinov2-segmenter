# Multi-stage build for the PUMA segmentation API — CPU-only target (self-hosted
# server has no GPU). Uses `uv pip install` (uv as the installer, not `uv sync`)
# with an explicit package list, self-contained on purpose: this repo is a
# member of a larger personal uv workspace (~/pyproject.toml), and a production
# image should never depend on the developer's workspace layout.

# ---- Builder stage ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    fastapi "uvicorn[standard]" python-multipart pillow

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

COPY config.py ./config.py
COPY utils/ ./utils/
COPY pipeline/04_model.py ./pipeline/04_model.py
COPY serving/ ./serving/

ENV PUMA_CHECKPOINT_DIR=/models
VOLUME ["/models"]

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8080"]
