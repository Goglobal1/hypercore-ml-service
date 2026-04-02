FROM python:3.12-slim

WORKDIR /app

# Install build dependencies (gcc/g++ needed for shap, ruptures)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Cache bust: 2026-04-02-v5
RUN echo "Build timestamp: $(date)"

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL application code
COPY . .

# Verify files exist
RUN ls -la app/core/endpoints/ && ls -la app/core/pathways/

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
