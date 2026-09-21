FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    CHROMA_DIR=/home/user/app/chroma_db

# Non-root user (uid 1000) -- also what Hugging Face Spaces expects.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only torch keeps the image far smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY --chown=user:user . .
RUN chown -R user:user /home/user
USER user

# Bake the embedding model and the Chroma index into the image, so the container
# starts offline (no model download, no ingestion at runtime).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
 && python ingest.py

# MOCK_LLM is intentionally NOT set -> default = deterministic offline mock mode.
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
