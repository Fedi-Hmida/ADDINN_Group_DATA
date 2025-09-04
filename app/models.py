"""
Pydantic models for request/response validation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")

class DetectionResult(BaseModel):
    """Single detection result"""
    detection_id: str = Field(..., description="Unique detection identifier")
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")

class ImageInfo(BaseModel):
    """Image information"""
    filename: str = Field(..., description="Original filename")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    channels: int = Field(..., description="Number of color channels")

class InferenceParams(BaseModel):
    """Inference parameters used"""
    confidence_threshold: float = Field(..., description="Confidence threshold used")

class DetectionResponse(BaseModel):
    """Complete detection response"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="Processing timestamp")
    image_info: ImageInfo = Field(..., description="Information about the processed image")
    detection_count: int = Field(..., description="Number of detections found")
    detections: List[DetectionResult] = Field(default_factory=list, description="List of detections")
    inference_params: InferenceParams = Field(..., description="Parameters used for inference")
    annotated_image_url: Optional[str] = Field(None, description="URL to annotated image if available")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: bool = Field(True, description="Error flag")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(..., description="Error timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_count: int = Field(0, description="Number of available GPUs")
    timestamp: str = Field(..., description="Check timestamp")

class ClassesResponse(BaseModel):
    """Available classes response"""
    classes: Dict[int, str] = Field(..., description="Available object classes")
    class_count: int = Field(..., description="Number of classes")
