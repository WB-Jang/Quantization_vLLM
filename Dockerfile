FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.5 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/.poetry_cache

# System dependencies + Python 3.11 (deadsnakes PPA — Ubuntu 22.04 ships 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl git build-essential ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# Dependency layer — rebuilt only when pyproject.toml or poetry.lock changes
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root \
    && rm -rf "$POETRY_CACHE_DIR"

# Source code
COPY . .

CMD ["poetry", "run", "python", "main.py", "--help"]
