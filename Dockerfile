# Use Python 3.9 slim image as base - lightweight but includes necessary tools
FROM python:3.9-slim

# Set metadata for the image
LABEL maintainer="your-email@example.com"
LABEL description="YOLOv8 Car Damage Detection API"
LABEL version="1.0"

# Set environment variables for better Python behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    MODEL_PATH=Model/best.pt

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libgtk-3-0 \
    libgl1-mesa-dri \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app/ ./app/
COPY Model/ ./Model/

# Create necessary directories and set proper permissions
RUN mkdir -p /app/temp /app/outputs /app/uploads /app/static /app/annotated_images && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port that FastAPI runs on
EXPOSE 8000

# Health check to verify the container is working
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
