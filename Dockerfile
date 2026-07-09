# FastAPI prediction backend, packaged for Hugging Face Spaces (Docker SDK).
# Python 3.13 matches the environment where the model artifact was verified to
# load, so joblib unpickling stays consistent.
FROM python:3.13-slim

# LightGBM (the market-price model) needs the OpenMP runtime at load/predict time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as a non-root user with UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Install runtime deps first so this layer is cached across code changes.
COPY --chown=user requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs to serve predictions. The relative paths in
# src/infer/predict.py (artifacts/, data/deploy/, reports/figures/) resolve
# against this WORKDIR.
COPY --chown=user src ./src
COPY --chown=user artifacts ./artifacts
COPY --chown=user data/deploy ./data/deploy
COPY --chown=user reports/figures ./reports/figures

# HF Spaces expects the app on port 7860.
EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
