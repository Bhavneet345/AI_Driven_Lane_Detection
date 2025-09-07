"""
Model implementations for lane detection.
Supports YOLOP, SCNN, and other deep learning models.
"""

from .base import BaseLaneDetector
from .yolop import YOLOPDetector
from .scnn import SCNNDetector, SCNNModelWrapper

__all__ = ["BaseLaneDetector", "YOLOPDetector", "SCNNDetector", "SCNNModelWrapper"]
