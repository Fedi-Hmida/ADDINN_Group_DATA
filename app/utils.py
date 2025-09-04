"""
Utility functions for the YOLOv8 FastAPI application
"""

import os
import uuid
import asyncio
from typing import List
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import UploadFile, HTTPException

from .config import settings

logger = logging.getLogger(__name__)

async def validate_image(file: UploadFile) -> None:
    """
    Validate uploaded image file
    
    Args:
        file: FastAPI UploadFile object
        
    Raises:
        HTTPException: If validation fails
    """
    
    # Check file extension
    if file.filename:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
    
    # Check file size
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Reset file pointer
    await file.seek(0)
    
    # Validate image content
    try:
        image = Image.open(file.file)
        image.verify()
        await file.seek(0)  # Reset after verification
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )

async def save_uploaded_file(file: UploadFile, upload_id: str) -> str:
    """
    Save uploaded file to disk
    
    Args:
        file: FastAPI UploadFile object
        upload_id: Unique identifier for the upload
        
    Returns:
        str: Path to saved file
    """
    
    # Create filename with upload ID
    file_ext = Path(file.filename).suffix if file.filename else ".jpg"
    filename = f"{upload_id}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    try:
        # Save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

def create_annotated_image(image: np.ndarray, detections: List[dict], output_path: str) -> None:
    """
    Create annotated image with bounding boxes
    
    Args:
        image: OpenCV image array
        detections: List of detection dictionaries
        output_path: Path to save annotated image
    """
    
    try:
        # Create a copy of the image
        annotated_image = image.copy()
        
        # Define colors for different classes (BGR format)
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 0),    # Dark Green
            (128, 128, 128) # Gray
        ]
        
        for detection in detections:
            # Extract detection data
            bbox = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            class_id = detection["class_id"]
            
            # Get coordinates
            x = int(bbox["x"])
            y = int(bbox["y"])
            width = int(bbox["width"])
            height = int(bbox["height"])
            
            # Calculate rectangle corners
            x1, y1 = x, y
            x2, y2 = x + width, y + height
            
            # Choose color based on class ID
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(
                annotated_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),  # White text
                thickness
            )
        
        # Save annotated image
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"Saved annotated image: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create annotated image: {str(e)}")
        # Don't raise exception as this is non-critical

def cleanup_old_files(directory: str, max_age_hours: int = 1) -> int:
    """
    Clean up old files from directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep (in hours)
        
    Returns:
        int: Number of files cleaned up
    """
    
    import time
    
    if not os.path.exists(directory):
        return 0
    
    current_time = time.time()
    cleaned_count = 0
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                max_age_seconds = max_age_hours * 3600
                
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    cleaned_count += 1
                    logger.info(f"Cleaned up old file: {file_path}")
        
        return cleaned_count
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return cleaned_count

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
