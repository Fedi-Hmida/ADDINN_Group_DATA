# YOLOv8 Car Damage Detection API - Deployment Guide

## Overview
This package contains a FastAPI application for car damage detection using YOLOv8 model.

## Package Contents
```
├── app/                    # Application source code
│   ├── main.py            # FastAPI application entry point
│   ├── config.py          # Configuration settings
│   ├── models.py          # Pydantic models for request/response
│   ├── utils.py           # Utility functions for image processing
│   └── __init__.py        # Package initialization
├── Model/                 # ML Model files
│   └── best.pt           # YOLOv8 trained model weights
├── Dockerfile            # Docker container configuration
├── docker-compose.yml    # Docker orchestration
├── requirements.txt      # Python dependencies
├── .dockerignore        # Docker ignore patterns
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore patterns
├── TEAMMATE_ACCESS_GUIDE.md  # Network access documentation
└── DEPLOYMENT_GUIDE.md  # This file
```

## Deployment Options

### Option 1: Docker Deployment (Recommended)
1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. Or build manually:
   ```bash
   docker build -t car-damage-api .
   docker run -p 8000:8000 car-damage-api
   ```

### Option 2: Direct Python Deployment
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints
- **Health Check**: `GET /health`
- **Damage Detection**: `POST /detect-damage`
- **API Documentation**: `GET /docs` (Swagger UI)
- **OpenAPI Schema**: `GET /openapi.json`

## Environment Configuration
1. Copy `.env.example` to `.env`
2. Modify environment variables as needed:
   - `MODEL_PATH`: Path to the YOLOv8 model file
   - `MAX_IMAGE_SIZE`: Maximum allowed image size
   - `CONFIDENCE_THRESHOLD`: Detection confidence threshold
   - `LOG_LEVEL`: Application logging level

## Requirements
- Python 3.9+
- Docker (for containerized deployment)
- 2GB+ RAM (for model loading)
- 1GB+ disk space

## Network Access
The API will be available on:
- Local: http://localhost:8000
- Network: http://[server-ip]:8000

For team access configuration, see `TEAMMATE_ACCESS_GUIDE.md`

## Health Check
Verify deployment with:
```bash
curl http://localhost:8000/health
```

## Support
Contact the development team for any deployment issues or questions.
