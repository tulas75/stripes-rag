FROM python:3.12-slim

# Minimal runtime deps for Docling (OpenCV/PDF rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer cache)
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy application code
COPY src/ src/
COPY app.py ./

# Reinstall so the entry point picks up local source
RUN pip install --no-cache-dir --no-deps .

# Pre-download the embedding model into the image so startup is instant
ARG EMBEDDING_MODEL=BAAI/bge-m3
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

EXPOSE 8501

# Default: run the Streamlit app. Override with "stripes" for CLI usage.
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
