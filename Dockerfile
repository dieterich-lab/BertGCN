# Production-ready BertGCN Docker Image
FROM python:3.8-slim as base

LABEL maintainer="Clinical AI Team <team@clinicalai.org>"
LABEL description="BertGCN Clinical Text Classification Framework"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.5.1
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Development stage
FROM base as development
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --with dev,monitoring && rm -rf $POETRY_CACHE_DIR
COPY . .
CMD ["poetry", "run", "python", "-m", "bertgcn.cli"]

# Production stage
FROM base as production
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR
COPY src ./src
COPY models ./models
COPY configs ./configs

# Create non-root user
RUN useradd --create-home --shell /bin/bash bertgcn
USER bertgcn

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["poetry", "run", "bertgcn", "serve"]