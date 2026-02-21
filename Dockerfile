# =============================================================================
# LSTM Stock Prediction Model - Upgraded Dockerfile
# =============================================================================
# This Dockerfile builds a containerized environment for running the
# LSTM stock prediction model with production-ready best practices.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Image
# -----------------------------------------------------------------------------
# Using specific Python version for reproducibility
FROM python:3.10-slim-bookworm AS base

# Labels for maintainability
LABEL maintainer="ML Team" \
      version="2.0.0" \
      description="LSTM Stock Prediction Model with Streamlit UI"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -----------------------------------------------------------------------------
# Stage 2: Dependencies
# -----------------------------------------------------------------------------
FROM base AS deps

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3: Application
# -----------------------------------------------------------------------------
FROM base AS app

# Create non-root user for security (best practice for production)
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Create directories
RUN mkdir -p /app/src/artifacts/models /app/.streamlit

# Set working directory
WORKDIR /app

# Copy virtual environment from deps stage
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements for dependency tracking
COPY --from=deps /app/requirements.txt .

# Copy project files - using src/ folder structure
COPY src/ ./src/
COPY app.py .
COPY streamlit_app.py .

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
# Improved health check that verifies Streamlit is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# -----------------------------------------------------------------------------
# Default Command
# -----------------------------------------------------------------------------
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
