FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

COPY main.py pyproject.toml .
COPY astromorph/ astromorph/

RUN uv venv --python 3.12 && \
    uv sync && \
    uv cache clean

RUN useradd -m -u 1000 astromorph \
    && chown -R astromorph:astromorph /app

USER astromorph

ENTRYPOINT ["uv", "run", "python", "-m", "astromorph"]
