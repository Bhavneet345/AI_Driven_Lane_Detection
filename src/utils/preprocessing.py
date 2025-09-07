"""
Image preprocessing utilities for enhanced lane detection in tunnel conditions.
Includes CLAHE, gamma correction, and other enhancement techniques.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional, Union
from enum import Enum

class PreprocessingMethod(Enum):
    """Available preprocessing methods."""
    NONE = "none"
    CLAHE = "clahe"
    GAMMA = "gamma"
    CLAHE_GAMMA = "clahe_gamma"
    HISTOGRAM_EQUALIZATION = "hist_eq"

def apply_clahe(
    image: np.ndarray, 
    clip_limit: float = 2.0, 
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input image (BGR or grayscale)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def apply_gamma_correction(
    image: np.ndarray, 
    gamma: float = 1.0
) -> np.ndarray:
    """
    Apply gamma correction to enhance image brightness.
    
    Args:
        image: Input image
        gamma: Gamma value (< 1 for brighter, > 1 for darker)
        
    Returns:
        Gamma-corrected image
    """
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    
    # Apply gamma correction
    return cv2.LUT(image, table)

def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization for better contrast.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Histogram equalized image
    """
    if len(image.shape) == 3:
        # Convert to YUV and equalize Y channel
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(image)

def preprocess_image(
    image: np.ndarray,
    method: Union[str, PreprocessingMethod] = PreprocessingMethod.CLAHE,
    gamma: float = 0.8,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply specified preprocessing method to enhance image for lane detection.
    
    Args:
        image: Input image
        method: Preprocessing method to apply
        gamma: Gamma value for gamma correction
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_size: CLAHE tile grid size
        
    Returns:
        Preprocessed image
    """
    if isinstance(method, str):
        method = PreprocessingMethod(method)
    
    if method == PreprocessingMethod.NONE:
        return image
    elif method == PreprocessingMethod.CLAHE:
        return apply_clahe(image, clahe_clip_limit, clahe_tile_size)
    elif method == PreprocessingMethod.GAMMA:
        return apply_gamma_correction(image, gamma)
    elif method == PreprocessingMethod.CLAHE_GAMMA:
        # Apply CLAHE first, then gamma correction
        enhanced = apply_clahe(image, clahe_clip_limit, clahe_tile_size)
        return apply_gamma_correction(enhanced, gamma)
    elif method == PreprocessingMethod.HISTOGRAM_EQUALIZATION:
        return apply_histogram_equalization(image)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

def enhance_for_tunnel_conditions(
    image: np.ndarray,
    brightness_boost: float = 1.2,
    contrast_boost: float = 1.1,
    gamma: float = 0.7
) -> np.ndarray:
    """
    Specialized enhancement for tunnel conditions with low lighting.
    
    Args:
        image: Input image
        brightness_boost: Brightness multiplier
        contrast_boost: Contrast multiplier
        gamma: Gamma correction value
        
    Returns:
        Enhanced image optimized for tunnel conditions
    """
    # Convert to float for processing
    enhanced = image.astype(np.float32) / 255.0
    
    # Apply brightness and contrast
    enhanced = enhanced * brightness_boost
    enhanced = np.clip(enhanced, 0, 1)
    
    # Apply contrast
    enhanced = (enhanced - 0.5) * contrast_boost + 0.5
    enhanced = np.clip(enhanced, 0, 1)
    
    # Apply gamma correction
    enhanced = np.power(enhanced, gamma)
    
    # Convert back to uint8
    enhanced = (enhanced * 255).astype(np.uint8)
    
    # Apply CLAHE for local contrast enhancement
    enhanced = apply_clahe(enhanced, clip_limit=3.0, tile_grid_size=(8, 8))
    
    return enhanced
