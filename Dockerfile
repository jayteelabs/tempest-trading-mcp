# Multi-stage Dockerfile for tempest-tradingview-mcp
# D9: uv package manager | D7: stdio transport only | D4: ta-lib from source

# ── Stage 1: Builder — install ta-lib C library + Python dependencies ──
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Install system dependencies for ta-lib C extension (D4, Q3)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Build and install ta-lib C library from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /app

# Copy project metadata for dependency resolution
COPY pyproject.toml ./

# Create venv and install production dependencies only (V3 fix: no dev extras)
RUN uv venv && \
    uv pip install --no-deps . && \
    uv pip install \
        "mcp>=1.0.0" \
        "yfinance>=0.2.0" \
        "ccxt>=4.0.0" \
        "pandas>=2.0.0" \
        "numpy>=1.24.0" \
        "ta-lib>=0.4.0" \
        "structlog>=23.0.0" \
        "httpx>=0.25.0" \
        "python-dotenv>=1.0.0"

# Copy source code
COPY src/ ./src/

# ── Stage 2: Production — minimal runtime ──
FROM python:3.12-slim AS production

# Install ta-lib C library runtime (COPY compiled artifacts from builder — no recompile)
COPY --from=builder /usr/lib/libta_lib.a /usr/lib/libta_lib.a
COPY --from=builder /usr/lib/libta_lib.so* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Install only runtime OS deps (no build tools needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 tempest && \
    useradd --uid 1000 --gid tempest --shell /bin/bash --create-home tempest

WORKDIR /app

# Copy virtual environment from builder (production deps only — V3 fix)
COPY --from=builder /app/.venv /home/tempest/.venv

# Copy source code from builder
COPY --from=builder /app/src /app/src

# Set environment variables
ENV PATH="/home/tempest/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TA_LIBRARY_PATH=/usr/lib \
    TA_INCLUDE_PATH=/usr/include

# Switch to non-root user
USER tempest

# MCP server entry point (stdio transport — D7)
CMD ["python", "-m", "tempest_mcp.server"]
