# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
# This includes gcc for compiling packages like psutil
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy requirements first for better caching
COPY pyproject.toml uv.lock ./

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# Copy the entire application
COPY . .

# Create directories for models and MLflow artifacts
RUN mkdir -p /app/models /app/mlruns

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Expose the API port
EXPOSE 8000

# Run the FastAPI application using uv run
CMD ["uv", "run", "uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]

