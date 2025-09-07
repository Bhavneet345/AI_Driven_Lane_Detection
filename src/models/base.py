"""
Base class for lane detection models.
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

class ModelType(Enum):
    """Supported model types."""
    OPENCV = "opencv"
    YOLOP = "yolop"
    SCNN = "scnn"

class LaneDetectionResult:
    """Container for lane detection results."""
    
    def __init__(
        self,
        lanes: List[np.ndarray],
        confidence: Optional[float] = None,
        processing_time: Optional[float] = None,
        model_type: Optional[ModelType] = None
    ):
        self.lanes = lanes  # List of lane line points
        self.confidence = confidence
        self.processing_time = processing_time
        self.model_type = model_type
    
    def __len__(self) -> int:
        return len(self.lanes)
    
    def is_empty(self) -> bool:
        return len(self.lanes) == 0

class BaseLaneDetector(ABC):
    """Abstract base class for lane detection models."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model from file."""
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for model inference."""
        pass
    
    @abstractmethod
    def postprocess(self, outputs: Any, original_shape: Tuple[int, int]) -> LaneDetectionResult:
        """Postprocess model outputs to extract lane lines."""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> LaneDetectionResult:
        """Run inference on input image."""
        pass
    
    def __call__(self, image: np.ndarray) -> LaneDetectionResult:
        """Make the detector callable."""
        return self.predict(image)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.__class__.__name__,
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self.is_loaded
        }
