@echo off
echo ========================================
echo YOLOv8 API Test Script
echo ========================================
echo.

REM Test health endpoint
echo Testing API health...
curl -X GET "http://localhost:8000/health"
echo.
echo.

REM Test with sample image (you need to provide an image path)
echo To test car damage detection, use:
echo curl -X POST "http://localhost:8000/detect" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path\to\your\car-image.jpg"
echo.

REM Get API info
echo Getting API information...
curl -X GET "http://localhost:8000/"
echo.
echo.

echo ========================================
echo API Endpoints Available:
echo ========================================
echo POST /detect              - Upload image for damage detection
echo GET  /health              - Check API health
echo GET  /annotated/{image_id} - Get annotated image
echo DELETE /cleanup           - Clean temporary files
echo GET  /docs                - Swagger UI documentation
echo GET  /redoc               - Alternative documentation
echo ========================================

pause
