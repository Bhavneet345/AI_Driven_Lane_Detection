#!/usr/bin/env python3
"""
Test script for SCNN model wrapper functionality.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.scnn import SCNNModelWrapper, SCNNDetector

def test_scnn_wrapper():
    """Test the SCNN model wrapper functionality."""
    print("Testing SCNN Model Wrapper")
    print("=" * 40)
    
    # Create a test image
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    print(f"Test image shape: {test_image.shape}")
    
    # Test 1: SCNNModelWrapper with config loading
    print("\n1. Testing SCNNModelWrapper with config loading")
    try:
        wrapper = SCNNModelWrapper(device="cpu")
        wrapper.load_from_config("configs/default.yaml")
        
        # Test forward pass
        lane_mask = wrapper.forward(test_image)
        print(f" Lane mask generated successfully")
        print(f"  - Shape: {lane_mask.shape}")
        print(f"  - Dtype: {lane_mask.dtype}")
        print(f"  - Unique values: {np.unique(lane_mask)}")
        print(f"  - Min/Max: {lane_mask.min()}/{lane_mask.max()}")
        
    except Exception as e:
        print(f" Error: {e}")
    
    # Test 2: SCNNDetector with config loading
    print("\n2. Testing SCNNDetector with config loading")
    try:
        detector = SCNNDetector(device="cpu")
        detector.load_from_config("configs/default.yaml")
        
        # Test prediction
        result = detector.predict(test_image)
        print(f" Prediction completed successfully")
        print(f"  - Number of lanes: {len(result.lanes)}")
        print(f"  - Processing time: {result.processing_time:.4f}s")
        print(f"  - Model type: {result.model_type}")
        
        # Test direct forward pass
        lane_mask = detector.forward(test_image)
        print(f" Direct forward pass successful")
        print(f"  - Lane mask shape: {lane_mask.shape}")
        
    except Exception as e:
        print(f" Error: {e}")
    
    # Test 3: SCNNDetector with model path
    print("\n3. Testing SCNNDetector with model path")
    try:
        detector = SCNNDetector(
            model_path="models/scnn.pth",  # This won't exist, should use dummy
            device="cpu"
        )
        detector.load_model()
        
        # Test prediction
        result = detector.predict(test_image)
        print(f" Prediction with model path successful")
        print(f"  - Number of lanes: {len(result.lanes)}")
        print(f"  - Processing time: {result.processing_time:.4f}s")
        
    except Exception as e:
        print(f" Error: {e}")
    
    # Test 4: Lane mask visualization
    print("\n4. Testing lane mask visualization")
    try:
        wrapper = SCNNModelWrapper(device="cpu")
        wrapper.load_from_config("configs/default.yaml")
        lane_mask = wrapper.forward(test_image)
        
        # Create visualization
        vis_image = test_image.copy()
        colored_mask = cv2.applyColorMap(lane_mask, cv2.COLORMAP_JET)
        
        # Blend with original image
        blended = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
        
        # Save visualization
        output_dir = Path("runs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "scnn_lane_mask_test.jpg"
        cv2.imwrite(str(output_path), blended)
        
        print(f" Lane mask visualization saved to: {output_path}")
        
    except Exception as e:
        print(f" Error: {e}")
    
    print("\n" + "=" * 40)
    print("SCNN Wrapper Test Complete!")

if __name__ == "__main__":
    test_scnn_wrapper()
