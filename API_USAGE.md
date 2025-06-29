# FlashDepth API Usage Guide

This guide covers how to use the FlashDepth inference API for real-time video depth estimation.

## Quick Start

### 1. Download Model Checkpoint

First, download the FlashDepth (Full) model checkpoint:

```bash
# Download from HuggingFace
wget https://huggingface.co/Eyeline-Research/FlashDepth/resolve/main/flashdepth/iter_43002.pth -O configs/flashdepth/iter_43002.pth
```

### 2. Build and Run with Docker

```bash
# Build the Docker image
docker build -t flashdepth .

# Run the container
docker run --gpus all -p 8000:8000 -v $(pwd)/configs:/app/configs -v $(pwd)/outputs:/app/outputs flashdepth
```

### 3. Using Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f flashdepth-api

# Stop the service
docker-compose down
```

## API Endpoints

### Health Check

Check if the service and model are ready:

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "message": "FlashDepth inference service is ready"
}
```

### Video Inference

Upload a video for depth estimation:

```bash
curl -X POST "http://localhost:8000/infer" \
  -F "video_file=@your_video.mp4" \
  -F "save_depth_npy=true" \
  -F "save_video=true" \
  -F "max_frames=100"
```

**Parameters:**
- `video_file`: Video file (MP4, AVI, MOV, MKV, WebM)
- `save_depth_npy`: Save depth maps as numpy files (default: true)
- `save_video`: Save output video with depth visualization (default: true)
- `max_frames`: Maximum frames to process (optional)
- `output_name`: Custom output name (optional)

**Response:**
```json
{
  "success": true,
  "message": "Inference completed successfully",
  "results": {
    "input_video": "your_video.mp4",
    "output_dir": "your_video_depth_1703123456",
    "num_frames": 120,
    "inference_time": 5.67,
    "fps": 21.16,
    "video_shape": [1, 120, 3, 720, 1280],
    "depth_shape": [1, 120, 1, 720, 1280]
  }
}
```

### Performance Testing

Run performance benchmarks:

```bash
curl -X POST "http://localhost:8000/performance-test" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_size": 1,
    "sequence_length": 105,
    "height": 1148,
    "width": 2044,
    "num_runs": 100
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Performance test completed",
  "results": {
    "shape": [1, 105, 3, 1148, 2044],
    "num_runs": 100,
    "total_time": 4.15,
    "avg_time_per_batch": 0.0415,
    "fps": 24.12
  }
}
```

## Python Client Example

```python
import requests
import json

# Health check
response = requests.get("http://localhost:8000/health")
print(f"Health: {response.json()}")

# Video inference
with open("your_video.mp4", "rb") as f:
    files = {"video_file": f}
    data = {
        "save_depth_npy": True,
        "save_video": True,
        "max_frames": 100
    }
    response = requests.post("http://localhost:8000/infer", files=files, data=data)
    result = response.json()
    print(f"Inference result: {json.dumps(result, indent=2)}")

# Performance test
perf_data = {
    "batch_size": 1,
    "sequence_length": 105,
    "height": 1148,
    "width": 2044,
    "num_runs": 50
}
response = requests.post("http://localhost:8000/performance-test", json=perf_data)
perf_result = response.json()
print(f"Performance: {perf_result['results']['fps']:.2f} FPS")
```

## Interactive API Documentation

Once the service is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Environment Variables

Configure the service using environment variables:

- `FLASHDEPTH_CONFIG_PATH`: Path to model configuration (default: "configs/flashdepth")
- `FLASHDEPTH_MODEL_CHECKPOINT`: Path to model checkpoint file
- `CUDA_VISIBLE_DEVICES`: GPU device ID to use (default: "0")

## Output Files

The API saves results to the `outputs/` directory:

```
outputs/
├── your_video_depth_1703123456/
│   ├── depth_000000.npy          # Depth maps (if save_depth_npy=true)
│   ├── depth_000001.npy
│   ├── ...
│   └── depth_output.mp4          # Depth visualization video (if save_video=true)
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid file format, missing parameters)
- `503`: Service unavailable (model not loaded)
- `500`: Internal server error

**Error Response Format:**
```json
{
  "detail": "Model not loaded or initialization failed",
  "error_type": "ModelNotLoadedError"
}
```

## Performance Notes

- **GPU Memory**: The service requires ~8GB GPU memory for FlashDepth (Full)
- **Input Resolution**: Videos with long side > 2044px are automatically resized
- **Batch Processing**: Currently supports single video processing per request
- **Supported Formats**: MP4, AVI, MOV, MKV, WebM

## TODOs & Future Enhancements

### Data Egress to Tigris
**TODO**: Implement automatic upload of processed results to Tigris storage after inference completion.

This feature should include:
- Upload depth maps (.npy files) to Tigris bucket
- Upload output videos to Tigris bucket  
- Return Tigris URLs in API response
- Optional cleanup of local files after successful upload
- Configuration for Tigris credentials and bucket settings

## Troubleshooting

### Model Not Loading
1. Ensure the checkpoint file exists at the configured path
2. Check GPU availability with `nvidia-smi`
3. Verify CUDA compatibility

### Out of Memory
1. Reduce `max_frames` parameter
2. Use smaller input resolution
3. Monitor GPU memory usage

### Slow Performance
1. Enable model compilation (set `eval.compile=true` in config.yaml)
2. Use appropriate GPU (A100 recommended for best performance)
3. Optimize Docker container resources