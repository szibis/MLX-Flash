# MLX-Flash Docker
# NOTE: MLX requires Apple Silicon Metal GPU for inference.
# This container is for: testing, CI/CD, and packaging.
# For actual inference, run natively on macOS with Apple Silicon.

FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/szibis/MLX-Flash"
LABEL org.opencontainers.image.description="MLX-Flash: MoE expert caching for Apple Silicon"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml PYPI_README.md ./
COPY mlx_flash_compress/ ./mlx_flash_compress/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY assets/ ./assets/
COPY README.md ./

# Install Python deps (skip mlx/mlx-lm — not available on Linux)
RUN pip install --no-cache-dir \
    numpy psutil tabulate pytest lz4 zstandard safetensors huggingface-hub && \
    pip install --no-cache-dir -e .

EXPOSE 8080

# Default: run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-k", "not test_auto_detect"]
