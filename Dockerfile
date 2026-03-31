# MLX-Flash-Compress Docker (for non-Mac development/testing)
# NOTE: MLX requires Apple Silicon. This Dockerfile is for:
#   1. Running tests and benchmarks on Linux (synthetic only)
#   2. CI/CD pipelines
#   3. Building documentation
# For actual inference, run natively on macOS with Apple Silicon.

FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml .
COPY mlx_flash_compress/ mlx_flash_compress/
COPY tests/ tests/
COPY csrc/ csrc/
COPY docs/ docs/
COPY README.md .

# Install Python deps (skip mlx/mlx-lm — not available on Linux)
RUN pip install --no-cache-dir \
    lz4 zstandard numpy psutil tabulate pytest

# Run tests (compression + cache tests work without MLX)
RUN python -m pytest tests/test_compression.py tests/test_cache.py -v

# Default: run synthetic benchmark
CMD ["python", "-m", "mlx_flash_compress.bench", "--synthetic"]
