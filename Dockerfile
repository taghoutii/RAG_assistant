# One image for both the API and the UI; docker-compose.yml picks the command of each.
# The corpus, indexes and LLM cache are NOT in the image: they are mounted at run time.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# CPU-only PyTorch first (same pin as CI): the default wheel pulls several GB of CUDA libraries.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# Dependencies before the code, so editing the code does not reinstall them. Editable install:
# the package finds configs/, data/, cache/ and logs/ relative to the repo root (/app).
COPY pyproject.toml README.md ./
RUN mkdir -p src/ragpapers && touch src/ragpapers/__init__.py && pip install -e .

# The embedding model is downloaded at build time, so the container never needs the Hub.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1

COPY configs ./configs
COPY .streamlit ./.streamlit
# The manifest (committed) gives the UI its corpus facts; the API container mounts data/ over it.
COPY data/corpus_manifest.yaml ./data/corpus_manifest.yaml
COPY src ./src

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/cache /app/logs \
    && chown -R app /app
USER app

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# One worker: the local Qdrant index can be opened by a single process.
CMD ["uvicorn", "ragpapers.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
