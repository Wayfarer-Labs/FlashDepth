"""
Pydantic v2 schemas for FlashDepth inference API
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class HealthStatus(str, Enum):
    """Health check status enum"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Health check response schema"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "cuda_available": True,
                "message": "FlashDepth inference service is ready"
            }
        }
    )
    
    status: HealthStatus
    model_loaded: bool
    cuda_available: bool
    message: str


class InferenceRequest(BaseModel):
    """Video inference request schema"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "save_depth_npy": True,
                "save_video": True,
                "max_frames": 100,
                "output_name": "my_video_depth"
            }
        }
    )
    
    save_depth_npy: bool = Field(
        default=True, 
        description="Whether to save depth maps as numpy files"
    )
    save_video: bool = Field(
        default=True, 
        description="Whether to save output video with depth visualization"
    )
    max_frames: Optional[int] = Field(
        default=None, 
        description="Maximum number of frames to process (None for all frames)",
        ge=1
    )
    output_name: Optional[str] = Field(
        default=None,
        description="Custom name for output files (auto-generated if not provided)",
        min_length=1,
        max_length=100
    )


class InferenceResponse(BaseModel):
    """Video inference response schema"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Inference completed successfully",
                "results": {
                    "input_video": "uploaded_video.mp4",
                    "output_dir": "output_1703123456",
                    "num_frames": 120,
                    "inference_time": 5.67,
                    "fps": 21.16,
                    "video_shape": [1, 120, 3, 720, 1280],
                    "depth_shape": [1, 120, 1, 720, 1280]
                }
            }
        }
    )
    
    success: bool
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PerformanceTestRequest(BaseModel):
    """Performance test request schema"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "batch_size": 1,
                "sequence_length": 105,
                "height": 1148,
                "width": 2044,
                "num_runs": 100
            }
        }
    )
    
    batch_size: int = Field(default=1, description="Batch size", ge=1, le=4)
    sequence_length: int = Field(default=105, description="Number of frames", ge=1, le=300)
    height: int = Field(default=1148, description="Frame height", ge=256, le=2048)
    width: int = Field(default=2044, description="Frame width", ge=256, le=2048)
    num_runs: int = Field(default=100, description="Number of benchmark runs", ge=1, le=1000)


class PerformanceTestResponse(BaseModel):
    """Performance test response schema"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Performance test completed",
                "results": {
                    "shape": [1, 105, 3, 1148, 2044],
                    "num_runs": 100,
                    "total_time": 4.15,
                    "avg_time_per_batch": 0.0415,
                    "fps": 24.12
                }
            }
        }
    )
    
    success: bool
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response schema"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Model not loaded or initialization failed",
                "error_type": "ModelNotLoadedError"
            }
        }
    )
    
    detail: str
    error_type: Optional[str] = None