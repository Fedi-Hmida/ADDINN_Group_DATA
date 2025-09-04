"""
FastAPI YOLOv8 Object Detection API

A production-ready REST API for YOLOv8 object detection with image upload,
inference, and annotated image retrieval capabilities.
"""

import os
import uuid
import asyncio
from typing import List, Optional
from datetime import datetime
import logging

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from .config import settings
from .models import DetectionResponse, DetectionResult, ErrorResponse
from .utils import validate_image, save_uploaded_file, create_annotated_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="A production-ready API for YOLOv8 object detection with image upload and annotation capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving annotated images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global model variable
model = None

class ModelManager:
    """Singleton class to manage YOLO model loading and inference"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str):
        """Load YOLOv8 model with GPU support if available"""
        try:
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading model on device: {device}")
            
            self._model = YOLO(model_path)
            self._model.to(device)
            
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model classes: {list(self._model.names.values())}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    def predict(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[dict]:
        """Run inference on image"""
        if self._model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Run inference
            results = self._model(image, conf=conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Extract box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Convert to xywh format
                        x = float(x1)
                        y = float(y1)
                        width = float(x2 - x1)
                        height = float(y2 - y1)
                        
                        # Extract confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self._model.names[class_id]
                        
                        detections.append({
                            "detection_id": str(uuid.uuid4()),
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": {
                                "x": x,
                                "y": y,
                                "width": width,
                                "height": height
                            }
                        })
            
            return detections
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    def get_class_names(self) -> dict:
        """Get model class names"""
        if self._model is None:
            return {}
        return self._model.names

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    logger.info("Starting up YOLOv8 API...")
    
    # Load the model
    model_path = settings.MODEL_PATH
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model_manager.load_model(model_path)
    logger.info("YOLOv8 API startup complete!")

@app.get("/", response_model=dict)
async def root():
    """Health check endpoint"""
    return {
        "message": "YOLOv8 Object Detection API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available(),
        "model_classes": model_manager.get_class_names()
    }

@app.get("/health", response_model=dict)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model_manager._model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold for detections"),
    save_annotated: bool = Query(True, description="Whether to save annotated image"),
    return_annotated_url: bool = Query(True, description="Whether to return URL for annotated image")
):
    """
    Detect objects in uploaded image
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **confidence_threshold**: Minimum confidence for detections (0.0-1.0)
    - **save_annotated**: Whether to save annotated image
    - **return_annotated_url**: Whether to return URL for annotated image
    """
    
    # Validate file
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Validate image file
    await validate_image(file)
    
    # Save uploaded file
    upload_id = str(uuid.uuid4())
    uploaded_file_path = await save_uploaded_file(file, upload_id)
    
    try:
        # Read and process image
        image = cv2.imread(uploaded_file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image file")
        
        # Run inference
        detections = model_manager.predict(image, confidence_threshold)
        
        # Prepare response
        response_data = {
            "request_id": upload_id,
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "filename": file.filename,
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2]
            },
            "detection_count": len(detections),
            "detections": detections,
            "inference_params": {
                "confidence_threshold": confidence_threshold
            }
        }
        
        # Save annotated image if requested
        if save_annotated and detections:
            annotated_filename = f"annotated_{upload_id}.jpg"
            annotated_path = os.path.join("static", annotated_filename)
            
            # Create annotated image in background
            background_tasks.add_task(
                create_annotated_image, 
                image, 
                detections, 
                annotated_path
            )
            
            if return_annotated_url:
                response_data["annotated_image_url"] = f"/static/{annotated_filename}"
        
        return DetectionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    finally:
        # Clean up uploaded file
        if os.path.exists(uploaded_file_path):
            os.remove(uploaded_file_path)

@app.get("/annotated/{filename}")
async def get_annotated_image(filename: str):
    """Retrieve annotated image by filename"""
    file_path = os.path.join("static", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Annotated image not found")
    
    return FileResponse(
        file_path,
        media_type="image/jpeg",
        filename=filename
    )

@app.get("/classes", response_model=dict)
async def get_classes():
    """Get available object classes"""
    return {
        "classes": model_manager.get_class_names(),
        "class_count": len(model_manager.get_class_names())
    }

@app.delete("/cleanup")
async def cleanup_files():
    """Clean up old uploaded and annotated files"""
    try:
        # Clean up static files older than 1 hour
        static_dir = "static"
        current_time = datetime.now().timestamp()
        cleaned_files = 0
        
        for filename in os.listdir(static_dir):
            file_path = os.path.join(static_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 3600:  # 1 hour
                    os.remove(file_path)
                    cleaned_files += 1
        
        return {
            "message": f"Cleaned up {cleaned_files} old files",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=True,
            message=exc.detail,
            status_code=exc.status_code,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=True,
            message="Internal server error",
            status_code=500,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1  # Use single worker for model consistency
    )
