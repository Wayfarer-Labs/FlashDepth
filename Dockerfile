# FlashDepth Inference Service Dockerfile
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build and install local mamba package
RUN cd mamba && \
    export MAMBA_FORCE_BUILD=TRUE && \
    export MAX_JOBS=4 && \
    pip install -e . --no-build-isolation && \
    cd ..

# Create directories for models and outputs
RUN mkdir -p configs/flashdepth && \
    mkdir -p outputs && \
    chmod 755 outputs

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV FLASHDEPTH_CONFIG_PATH=configs/flashdepth
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]