"""
FlashDepth FastAPI Inference Service

A production-ready FastAPI service for real-time video depth estimation
using FlashDepth (Full) model.
"""
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference_service import FlashDepthInferenceService
from schemas import (
    HealthResponse, 
    HealthStatus,
    InferenceRequest, 
    InferenceResponse,
    PerformanceTestRequest,
    PerformanceTestResponse,
    ErrorResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="FlashDepth Inference API",
    description="Real-time streaming video depth estimation at 2K resolution using FlashDepth (Full)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference service instance
inference_service: Optional[FlashDepthInferenceService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the inference service on startup"""
    global inference_service
    try:
        config_path = os.getenv("FLASHDEPTH_CONFIG_PATH", "configs/flashdepth")
        model_checkpoint = os.getenv("FLASHDEPTH_MODEL_CHECKPOINT", None)
        
        print(f"Initializing FlashDepth service with config: {config_path}")
        inference_service = FlashDepthInferenceService(
            config_path=config_path,
            model_checkpoint=model_checkpoint
        )
        
        if inference_service.is_model_loaded():
            print("✅ FlashDepth model loaded successfully")
        else:
            print("❌ Failed to load FlashDepth model")
            
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        print(traceback.format_exc())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global inference_service
    if inference_service:
        # Cleanup if needed
        print("🔄 Shutting down FlashDepth service")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service and model status
    
    Returns:
        HealthResponse: Service health status including model availability
    """
    global inference_service
    
    model_loaded = inference_service is not None and inference_service.is_model_loaded()
    cuda_available = torch.cuda.is_available()
    
    if model_loaded:
        status = HealthStatus.HEALTHY
        message = "FlashDepth inference service is ready"
    else:
        status = HealthStatus.UNHEALTHY
        message = "Model not loaded or service unavailable"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        cuda_available=cuda_available,
        message=message
    )


@app.post("/infer", response_model=InferenceResponse)
async def infer_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(..., description="Video file for depth estimation"),
    save_depth_npy: bool = True,
    save_video: bool = True,
    max_frames: Optional[int] = None,
    output_name: Optional[str] = None
):
    """
    Perform depth estimation inference on uploaded video
    
    Args:
        video_file: Uploaded video file (MP4, AVI, MOV supported)
        save_depth_npy: Whether to save depth maps as numpy files
        save_video: Whether to save output video with depth visualization
        max_frames: Maximum frames to process (None for all frames)
        output_name: Custom output name (auto-generated if None)
    
    Returns:
        InferenceResponse: Inference results including timing and output paths
    """
    global inference_service
    
    if not inference_service or not inference_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or inference service unavailable"
        )
    
    # Validate file
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    file_extension = Path(video_file.filename).suffix.lower()
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {', '.join(supported_extensions)}"
        )
    
    # Create temporary file for upload
    temp_video_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_video_path = temp_file.name
            # Write uploaded content to temp file
            content = await video_file.read()
            temp_file.write(content)
        
        # Generate output directory name
        timestamp = int(time.time())
        if output_name:
            output_dir = f"{output_name}_{timestamp}"
        else:
            base_name = Path(video_file.filename).stem
            output_dir = f"{base_name}_depth_{timestamp}"
        
        # Run inference
        try:
            results = inference_service.infer_video(
                video_path=temp_video_path,
                output_dir=output_dir,
                save_depth_npy=save_depth_npy,
                save_video=save_video,
                max_frames=max_frames
            )
            
            # Schedule cleanup of temp file
            background_tasks.add_task(cleanup_temp_file, temp_video_path)
            
            return InferenceResponse(
                success=True,
                message="Inference completed successfully",
                results=results
            )
            
        except Exception as e:
            # Cleanup temp file on error
            cleanup_temp_file(temp_video_path)
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            )
    
    except Exception as e:
        if temp_video_path:
            cleanup_temp_file(temp_video_path)
        raise HTTPException(
            status_code=500,
            detail=f"Request processing failed: {str(e)}"
        )


@app.post("/performance-test", response_model=PerformanceTestResponse)
async def performance_test(request: PerformanceTestRequest):
    """
    Run performance benchmark with synthetic data
    
    Args:
        request: Performance test configuration
    
    Returns:
        PerformanceTestResponse: Benchmark results including FPS metrics
    """
    global inference_service
    
    if not inference_service or not inference_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or inference service unavailable"
        )
    
    try:
        shape = (
            request.batch_size, 
            request.sequence_length, 
            3, 
            request.height, 
            request.width
        )
        
        # Create modified inference service method for custom num_runs
        results = await run_custom_performance_test(shape, request.num_runs)
        
        return PerformanceTestResponse(
            success=True,
            message="Performance test completed",
            results=results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Performance test failed: {str(e)}"
        )


async def run_custom_performance_test(shape, num_runs):
    """Custom performance test with configurable runs"""
    global inference_service
    
    dummy_input = torch.randn(shape, device=inference_service.device, dtype=torch.float32)
    batch = [dummy_input, None, "dummy"]
    
    # Warmup runs
    for _ in range(10):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = inference_service.model(batch, dummy_timing=True)
    
    # Timed runs
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = inference_service.model(batch, dummy_timing=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = (num_runs * shape[1]) / total_time  # Total frames / total time
    
    return {
        'shape': list(shape),
        'num_runs': num_runs,
        'total_time': total_time,
        'avg_time_per_batch': total_time / num_runs,
        'fps': fps
    }


def cleanup_temp_file(file_path: str):
    """Cleanup temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Warning: Failed to cleanup temp file {file_path}: {e}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_type=exc.__class__.__name__
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error occurred",
            error_type=exc.__class__.__name__
        ).model_dump()
    )


def main():
    """Main entry point for the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlashDepth Inference API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        access_log=True
    )


if __name__ == "__main__":
    main()