# Multi-stage Dockerfile for tempest-tradingview-mcp
# D9: uv package manager | D7: HTTP/SSE transport on :9001 | D4: ta-lib from source

# ── Stage 1: Builder — install ta-lib C library + Python dependencies ──
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ARG TA_LIB_VERSION=0.4.0
ARG TA_LIB_SHA256=9ff41efcb1c011a4b4b6dfc91610b06e39b1d7973ed5d4dee55029a0ac4dc651

# Install system dependencies for ta-lib C extension (D4, Q3)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Build and install ta-lib C library from source
RUN wget -O ta-lib-${TA_LIB_VERSION}-src.tar.gz "https://sourceforge.net/projects/ta-lib/files/ta-lib/${TA_LIB_VERSION}/ta-lib-${TA_LIB_VERSION}-src.tar.gz/download" && \
    echo "${TA_LIB_SHA256}  ta-lib-${TA_LIB_VERSION}-src.tar.gz" | sha256sum -c - && \
    tar -xzf ta-lib-${TA_LIB_VERSION}-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-${TA_LIB_VERSION}-src.tar.gz

WORKDIR /app

# Copy project metadata for dependency resolution
COPY pyproject.toml ./
COPY README.md ./

# Copy source code BEFORE installing package (required for hatchling)
COPY src/ ./src/

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
    TA_INCLUDE_PATH=/usr/include \
    PYTHONPATH=/app/src

# Healthcheck: probe SSE surface via HEAD request to /sse.
# SSE is a persistent stream — HEAD returns headers immediately without waiting for body.
# Exit 0 (healthy) if endpoint exists (2xx-4xx), exit 1 (unhealthy) if 5xx or unreachable.
# Connection failures exit cleanly without printing a Python traceback.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["python", "-c", "exec(\"import http.client, sys\\nstatus = 500\\nconn = None\\ntry:\\n    conn = http.client.HTTPConnection('localhost', 9001, timeout=5)\\n    conn.request('HEAD', '/sse')\\n    status = conn.getresponse().status\\nexcept OSError:\\n    sys.exit(1)\\nfinally:\\n    if conn is not None:\\n        conn.close()\\nsys.exit(0 if status < 500 else 1)\\n\")"]

# Switch to non-root user
USER tempest

# MCP server entry point (HTTP/SSE transport — D7)
EXPOSE 9001
CMD ["python", "-m", "tempest_mcp.server"]
