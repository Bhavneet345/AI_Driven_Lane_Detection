#!/usr/bin/env python3
"""
Compare Hough line detection vs RANSAC + polynomial curve fitting.
"""

import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.runtime.pipeline import LaneDetectionPipeline
from src.utils.ransac_lane_detection import compare_detection_methods

def create_test_image():
    """Create a test image with curved lanes."""
    # Create a test image
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Add some background noise
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Draw curved lanes
    h, w = img.shape[:2]
    
    # Left lane (curved)
    left_points = []
    for y in range(h//2, h, 5):
        x = int(w//4 + 100 * np.sin((y - h//2) * 0.01) + 50 * np.cos((y - h//2) * 0.005))
        left_points.append((x, y))
    
    # Right lane (curved)
    right_points = []
    for y in range(h//2, h, 5):
        x = int(3*w//4 + 80 * np.sin((y - h//2) * 0.008) + 40 * np.cos((y - h//2) * 0.003))
        right_points.append((x, y))
    
    # Draw lanes
    for i in range(len(left_points) - 1):
        cv2.line(img, left_points[i], left_points[i+1], (255, 255, 255), 8)
    
    for i in range(len(right_points) - 1):
        cv2.line(img, right_points[i], right_points[i+1], (255, 255, 255), 8)
    
    # Add some noise and artifacts
    # Random lines
    for _ in range(20):
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (np.random.randint(0, w), np.random.randint(0, h))
        cv2.line(img, pt1, pt2, (100, 100, 100), 2)
    
    # Random dots
    for _ in range(100):
        pt = (np.random.randint(0, w), np.random.randint(0, h))
        cv2.circle(img, pt, 2, (150, 150, 150), -1)
    
    return img

def test_detection_methods():
    """Test and compare different detection methods."""
    print("Comparing Lane Detection Methods")
    print("=" * 40)
    
    # Create test image
    test_image = create_test_image()
    
    # Save test image
    os.makedirs("runs", exist_ok=True)
    cv2.imwrite("runs/test_image.jpg", test_image)
    print("Test image saved: runs/test_image.jpg")
    
    # Test configurations
    configs = {
        "hough": {
            "detection_method": "opencv",
            "preprocessing": {"method": "none"},
            "opencv": {
                "canny": {"low": 50, "high": 150},
                "hough": {"rho": 1, "theta": 0.0174533, "threshold": 50, "min_line_len": 50, "max_line_gap": 150},
                "roi": {"top_ratio": 0.55}
            }
        },
        "ransac": {
            "detection_method": "ransac",
            "preprocessing": {"method": "none"},
            "opencv": {
                "canny": {"low": 50, "high": 150},
                "roi": {"top_ratio": 0.55}
            },
            "ransac": {
                "min_samples": 10,
                "residual_threshold": 2.0,
                "polynomial_degree": 2
            }
        },
        "ransac_enhanced": {
            "detection_method": "ransac",
            "preprocessing": {"method": "clahe_gamma", "gamma": 0.8, "clahe_clip_limit": 2.0},
            "opencv": {
                "canny": {"low": 30, "high": 100},
                "roi": {"top_ratio": 0.55}
            },
            "ransac": {
                "min_samples": 8,
                "residual_threshold": 1.5,
                "polynomial_degree": 3
            }
        }
    }
    
    results = {}
    
    for method_name, config in configs.items():
        print(f"\nTesting {method_name.upper()} method...")
        
        try:
            # Create pipeline
            pipeline = LaneDetectionPipeline(config)
            
            # Run detection multiple times for timing
            times = []
            for _ in range(10):
                start_time = time.time()
                overlay, result = pipeline.detect_lanes(test_image)
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # Count detected lanes
            num_lanes = len(result.lanes)
            
            # Calculate lane quality metrics
            lane_lengths = []
            for lane in result.lanes:
                if len(lane) > 1:
                    # Calculate approximate length
                    length = np.sum(np.sqrt(np.sum(np.diff(lane, axis=0)**2, axis=1)))
                    lane_lengths.append(length)
            
            avg_lane_length = np.mean(lane_lengths) if lane_lengths else 0
            
            results[method_name] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "num_lanes": num_lanes,
                "avg_lane_length": avg_lane_length,
                "fps": 1.0 / avg_time,
                "success": True
            }
            
            print(f"   Success")
            print(f"    - Average time: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
            print(f"    - FPS: {1.0/avg_time:.1f}")
            print(f"    - Lanes detected: {num_lanes}")
            print(f"    - Average lane length: {avg_lane_length:.1f}px")
            
            # Save result image
            output_path = f"runs/{method_name}_result.jpg"
            cv2.imwrite(output_path, overlay)
            print(f"    - Result saved: {output_path}")
            
        except Exception as e:
            print(f"   Error: {e}")
            results[method_name] = {"success": False, "error": str(e)}
    
    # Print comparison summary
    print(f"\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    
    print(f"{'Method':<15} {'Time (ms)':<12} {'FPS':<8} {'Lanes':<8} {'Length':<10} {'Status'}")
    print("-" * 70)
    
    for method_name, result in results.items():
        if result.get("success", False):
            print(f"{method_name:<15} {result['avg_time']*1000:<12.1f} {result['fps']:<8.1f} {result['num_lanes']:<8} {result['avg_lane_length']:<10.1f} ")
        else:
            print(f"{method_name:<15} {'N/A':<12} {'N/A':<8} {'N/A':<8} {'N/A':<10}  {result.get('error', 'Unknown error')}")
    
    # Performance comparison
    if all(r.get("success", False) for r in results.values()):
        print(f"\nPerformance Analysis:")
        hough_fps = results["hough"]["fps"]
        ransac_fps = results["ransac"]["fps"]
        ransac_enhanced_fps = results["ransac_enhanced"]["fps"]
        
        print(f"  - RANSAC vs Hough speedup: {ransac_fps/hough_fps:.2f}x")
        print(f"  - Enhanced RANSAC vs Hough speedup: {ransac_enhanced_fps/hough_fps:.2f}x")
        print(f"  - Enhanced RANSAC vs RANSAC speedup: {ransac_enhanced_fps/ransac_fps:.2f}x")
        
        # Quality comparison
        hough_lanes = results["hough"]["num_lanes"]
        ransac_lanes = results["ransac"]["num_lanes"]
        ransac_enhanced_lanes = results["ransac_enhanced"]["num_lanes"]
        
        print(f"\nQuality Analysis:")
        print(f"  - Hough lanes detected: {hough_lanes}")
        print(f"  - RANSAC lanes detected: {ransac_lanes}")
        print(f"  - Enhanced RANSAC lanes detected: {ransac_enhanced_lanes}")
    
    print(f"\nComparison complete! Check 'runs/' directory for result images.")

def test_with_real_image():
    """Test with a real image if available."""
    print(f"\nTesting with real image...")
    
    # Try to find a test image
    test_paths = [
        "examples/sample_video.mp4",
        "examples/test_image.jpg",
        "examples/test_image.png"
    ]
    
    test_image = None
    for path in test_paths:
        if os.path.exists(path):
            if path.endswith('.mp4'):
                # Extract first frame from video
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    test_image = frame
                    break
            else:
                test_image = cv2.imread(path)
                if test_image is not None:
                    break
    
    if test_image is None:
        print("No real test image found, using synthetic image.")
        return
    
    print(f"Using real image: {test_image.shape}")
    
    # Test with real image
    config = {
        "detection_method": "ransac",
        "preprocessing": {"method": "clahe_gamma"},
        "opencv": {"canny": {"low": 50, "high": 150}, "roi": {"top_ratio": 0.55}},
        "ransac": {"min_samples": 10, "residual_threshold": 2.0, "polynomial_degree": 2}
    }
    
    pipeline = LaneDetectionPipeline(config)
    overlay, result = pipeline.detect_lanes(test_image)
    
    print(f"Real image test:")
    print(f"  - Lanes detected: {len(result.lanes)}")
    print(f"  - Processing time: {result.processing_time*1000:.1f}ms")
    
    cv2.imwrite("runs/real_image_result.jpg", overlay)
    print(f"  - Result saved: runs/real_image_result.jpg")

if __name__ == "__main__":
    test_detection_methods()
    test_with_real_image()
