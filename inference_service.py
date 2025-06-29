"""
FlashDepth Inference Service - Standalone inference module for video depth estimation
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Union, List, Tuple
import logging
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import tempfile
import time

from flashdepth.model import FlashDepth
from dataloaders.random_dataset import RandomDataset
from utils.helpers import *


class FlashDepthInferenceService:
    """Standalone inference service for FlashDepth models"""
    
    def __init__(self, config_path: str = "configs/flashdepth", model_checkpoint: Optional[str] = None):
        """
        Initialize FlashDepth inference service
        
        Args:
            config_path: Path to model configuration directory
            model_checkpoint: Path to model checkpoint file (.pth)
        """
        self.config_path = config_path
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.cfg = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_logging()
        self.load_model()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self) -> bool:
        """
        Load FlashDepth model and configuration
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load configuration
            config_file = os.path.join(self.config_path, "config.yaml")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            self.cfg = OmegaConf.load(config_file)
            self.cfg.config_dir = self.config_path
            self.cfg.inference = True
            
            # Initialize model
            model_kwargs = dict(
                batch_size=1,  # Single batch for inference
                hybrid_configs=self.cfg.hybrid_configs if hasattr(self.cfg, 'hybrid_configs') else None,
                training=False,  # Inference mode
                **self.cfg.model,
            )
            
            self.model = FlashDepth(**model_kwargs)
            self.model = self.model.to(self.device)
            
            # Load checkpoint
            checkpoint_path = self.model_checkpoint or os.path.join(self.config_path, "iter_43002.pth")
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (DDP wrapper)
                cleaned_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('module.', '') if key.startswith('module.') else key
                    cleaned_state_dict[new_key] = value
                
                self.model.load_state_dict(cleaned_state_dict, strict=False)
            else:
                self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            self.model.eval()
            
            # Compile model for better performance if supported
            if hasattr(self.cfg.eval, 'compile') and self.cfg.eval.compile:
                try:
                    self.model = torch.compile(self.model)
                    self.logger.info("Model compiled for optimized inference")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")
            
            self.logger.info("FlashDepth model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return self.model is not None
    
    def preprocess_video(self, video_path: str, max_frames: Optional[int] = None) -> torch.Tensor:
        """
        Preprocess video for inference
        
        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process
            
        Returns:
            torch.Tensor: Preprocessed video tensor of shape (1, T, 3, H, W)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed (keep aspect ratio, long side <= 2044)
                h, w = frame.shape[:2]
                if max(h, w) > 2044:
                    scale = 2044 / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Convert to tensor and normalize
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)  # HWC -> CHW
                frames.append(frame)
                frame_count += 1
        
        finally:
            cap.release()
        
        if not frames:
            raise ValueError("No frames found in video")
        
        # Stack frames and add batch dimension
        video_tensor = torch.stack(frames, dim=0).unsqueeze(0)  # (1, T, C, H, W)
        return video_tensor
    
    @torch.no_grad()
    def infer_video(
        self, 
        video_path: str, 
        output_dir: Optional[str] = None,
        save_depth_npy: bool = True,
        save_video: bool = True,
        max_frames: Optional[int] = None
    ) -> dict:
        """
        Run inference on video
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            save_depth_npy: Whether to save depth maps as numpy files
            save_video: Whether to save output video
            max_frames: Maximum frames to process
            
        Returns:
            dict: Inference results with timing and output paths
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Setup output directory
        if output_dir is None:
            output_dir = f"output_{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess video
        self.logger.info(f"Processing video: {video_path}")
        video_tensor = self.preprocess_video(video_path, max_frames)
        video_tensor = video_tensor.to(self.device)
        
        B, T, C, H, W = video_tensor.shape
        self.logger.info(f"Video shape: {video_tensor.shape}")
        
        # Run inference
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            batch = [video_tensor, None, "random"]  # Format expected by model
            
            eval_args = {
                'save_depth_npy': save_depth_npy,
                'save_vis_map': False,
                'out_video': save_video,
                'out_mp4': True,
                'use_mamba': self.cfg.model.use_mamba,
                'resolution': getattr(self.cfg.eval, 'save_res', 518),
                'print_time': True,
                'loss_type': self.cfg.training.loss_type,
                'use_all_frames': True,
                'use_metrics': False,
                'dummy_timing': False
            }
            
            output_path = os.path.join(output_dir, f"depth_output")
            depth_output, img_grid = self.model(batch, gif_path=output_path, **eval_args)
        
        end_time = time.time()
        inference_time = end_time - start_time
        fps = T / inference_time if inference_time > 0 else 0
        
        results = {
            'input_video': video_path,
            'output_dir': output_dir,
            'num_frames': T,
            'inference_time': inference_time,
            'fps': fps,
            'video_shape': list(video_tensor.shape),
            'depth_shape': list(depth_output.shape) if depth_output is not None else None
        }
        
        self.logger.info(f"Inference completed in {inference_time:.2f}s ({fps:.2f} FPS)")
        return results
    
    @torch.no_grad()
    def infer_dummy(self, shape: Tuple[int, int, int, int, int] = (1, 105, 3, 1148, 2044)) -> dict:
        """
        Run dummy inference for performance testing
        
        Args:
            shape: Input tensor shape (B, T, C, H, W)
            
        Returns:
            dict: Performance metrics
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.logger.info(f"Running dummy inference with shape: {shape}")
        
        # Create dummy input tensor
        dummy_input = torch.randn(shape, device=self.device, dtype=torch.float32)
        batch = [dummy_input, None, "dummy"]
        
        # Warmup runs
        for _ in range(10):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _ = self.model(batch, dummy_timing=True)
        
        # Timed runs
        start_time = time.time()
        num_runs = 100
        
        for _ in range(num_runs):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _ = self.model(batch, dummy_timing=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        fps = (num_runs * shape[1]) / total_time  # Total frames / total time
        
        results = {
            'shape': shape,
            'num_runs': num_runs,
            'total_time': total_time,
            'avg_time_per_batch': total_time / num_runs,
            'fps': fps
        }
        
        self.logger.info(f"Dummy inference: {total_time:.2f}s total, {fps:.2f} FPS")
        return results