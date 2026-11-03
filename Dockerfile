# StructFlow gateway image.
#
#   docker build -t structflow .                          # small: fakes + vLLM over HTTP
#   docker build -t structflow --build-arg EXTRAS=ml .    # adds torch + transformers
#                                                         # for the real small models
ARG PYTHON_VERSION=3.12

# --- Stage 1: build a virtual environment with StructFlow in it -----------------
FROM python:${PYTHON_VERSION}-slim AS build
ARG EXTRAS=""
WORKDIR /src
COPY pyproject.toml ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}"

# --- Stage 2: the runtime image: the venv only, no source, no build tools -------
FROM python:${PYTHON_VERSION}-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache
COPY --from=build /opt/venv /opt/venv
USER 10001
WORKDIR /home/app
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]
CMD ["uvicorn", "--factory", "structflow.gateway.app:create_app", \
    "--host", "0.0.0.0", "--port", "8080"]
