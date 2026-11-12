# OpenStrategy Studio in a container (phase 4 spike: works, and suits a home server or a
# machine you do not want to install Python on).
#
#   docker build -f packaging/docker/Dockerfile -t openstrategy-studio .
#   docker run --rm -p 127.0.0.1:8501:8501 -v studio-data:/data openstrategy-studio
#
# ⚠️ vectorbt carries the Commons Clause: run this for yourself or inside your
# organisation, never as a service you sell.

# Match the uv version that wrote uv.lock (`uv --version`).
ARG UV_VERSION=0.9
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim-bookworm
# git: vectorbtX is installed from GitHub until it is on PyPI.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    OPENSTRATEGY_STUDIO_HOME=/data \
    NUMBA_CACHE_DIR=/data/numba_cache

WORKDIR /app
# Dependencies first, so editing the Studio's code does not reinstall them.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

RUN useradd --create-home studio && mkdir -p /data && chown studio /data
USER studio
VOLUME /data
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["/app/.venv/bin/openstrategy-studio", "run", "--headless", "--address", "0.0.0.0", "--port", "8501"]
