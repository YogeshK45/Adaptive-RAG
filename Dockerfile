# Use official lightweight Python base image
FROM python:3.11-slim

# Set environment variables
# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is sent straight to terminal (unbuffered)
ENV PYTHONUNBUFFERED=1
# Set application port
ENV PORT=8000
# Cache directory for Hugging Face models
ENV HF_HOME=/root/.cache/huggingface

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for building wheels, SSL certs, and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the sentence-transformers embedding model for instantaneous startup
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# Copy all application source code and configuration files
COPY . .

# Expose the application port
EXPOSE 8000

# Set default startup command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
