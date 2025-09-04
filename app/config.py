"""
Configuration settings for YOLOv8 FastAPI application
"""

import os
from pathlib import Path

class Settings:
    """Application settings"""
    
    # API Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "Model/best.pt")
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    # Directory Configuration
    UPLOAD_DIR: str = "uploads"
    STATIC_DIR: str = "static"
    ANNOTATED_DIR: str = "annotated_images"
    
    # Detection Configuration
    DEFAULT_CONFIDENCE: float = 0.5
    MAX_CONFIDENCE: float = 1.0
    MIN_CONFIDENCE: float = 0.0
    
    # Performance Configuration
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 1))
    
    def __init__(self):
        # Ensure directories exist
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.STATIC_DIR, exist_ok=True)
        os.makedirs(self.ANNOTATED_DIR, exist_ok=True)

# Create settings instance
settings = Settings()
