#!/usr/bin/env python3
"""
Script to evaluate and compare different lane detection models.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.runtime.pipeline import LaneDetectionPipeline
from src.utils.metrics import compare_models, LaneDetectionMetrics

def create_test_configs():
    """Create test configurations for different methods."""
    configs = {
        "opencv_baseline": {
            "detection_method": "opencv",
            "preprocessing": {"method": "none"},
            "opencv": {
                "canny": {"low": 50, "high": 150},
                "hough": {"rho": 1, "theta": 0.0174533, "threshold": 50, "min_line_len": 50, "max_line_gap": 150},
                "roi": {"top_ratio": 0.55}
            }
        },
        "opencv_clahe": {
            "detection_method": "opencv",
            "preprocessing": {"method": "clahe", "clahe_clip_limit": 2.0, "clahe_tile_size": [8, 8]},
            "opencv": {
                "canny": {"low": 50, "high": 150},
                "hough": {"rho": 1, "theta": 0.0174533, "threshold": 50, "min_line_len": 50, "max_line_gap": 150},
                "roi": {"top_ratio": 0.55}
            }
        },
        "opencv_gamma": {
            "detection_method": "opencv",
            "preprocessing": {"method": "gamma", "gamma": 0.8},
            "opencv": {
                "canny": {"low": 50, "high": 150},
                "hough": {"rho": 1, "theta": 0.0174533, "threshold": 50, "min_line_len": 50, "max_line_gap": 150},
                "roi": {"top_ratio": 0.55}
            }
        },
        "yolop": {
            "detection_method": "yolop",
            "preprocessing": {"method": "clahe_gamma", "gamma": 0.8, "clahe_clip_limit": 2.0},
            "models": {
                "yolop": {
                    "model_path": "models/yolop.pth",
                    "input_size": [640, 640],
                    "confidence_threshold": 0.5,
                    "device": "cpu"
                }
            }
        },
        "scnn": {
            "detection_method": "scnn",
            "preprocessing": {"method": "clahe_gamma", "gamma": 0.8, "clahe_clip_limit": 2.0},
            "models": {
                "scnn": {
                    "model_path": "models/scnn.pth",
                    "input_size": [512, 256],
                    "confidence_threshold": 0.5,
                    "device": "cpu"
                }
            }
        }
    }
    return configs

def run_benchmark_comparison(configs, output_dir="benchmark_results"):
    """Run benchmark comparison on synthetic data."""
    print("Running benchmark comparison...")
    
    import numpy as np
    import time
    
    # Create synthetic test data
    h, w = 720, 1280
    test_frames = []
    for i in range(10):
        frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        # Add synthetic lane features
        cv2.line(frame, (w//4, h), (w//3, h//2), (255, 255, 255), 5)
        cv2.line(frame, (3*w//4, h), (2*w//3, h//2), (255, 255, 255), 5)
        test_frames.append(frame)
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nTesting {config_name}...")
        
        try:
            pipeline = LaneDetectionPipeline(config)
            metrics = LaneDetectionMetrics()
            
            # Warmup
            for _ in range(5):
                _, _ = pipeline.detect_lanes(test_frames[0])
            
            # Benchmark
            start_time = time.time()
            for frame in test_frames:
                _, result = pipeline.detect_lanes(frame)
                if result.processing_time:
                    metrics.processing_times.append(result.processing_time)
            
            total_time = time.time() - start_time
            
            # Store results
            results[config_name] = {
                "avg_fps": len(test_frames) / total_time,
                "avg_processing_time": np.mean(metrics.processing_times) if metrics.processing_times else 0,
                "min_processing_time": np.min(metrics.processing_times) if metrics.processing_times else 0,
                "max_processing_time": np.max(metrics.processing_times) if metrics.processing_times else 0,
                "total_frames": len(test_frames)
            }
            
            print(f"  Average FPS: {results[config_name]['avg_fps']:.1f}")
            print(f"  Average processing time: {results[config_name]['avg_processing_time']*1000:.1f}ms")
            
        except Exception as e:
            print(f"  Error testing {config_name}: {e}")
            results[config_name] = {"error": str(e)}
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "benchmark_results.txt")
    
    with open(results_file, 'w') as f:
        f.write("Benchmark Comparison Results\n")
        f.write("=" * 40 + "\n\n")
        
        for config_name, result in results.items():
            f.write(f"{config_name}:\n")
            if "error" in result:
                f.write(f"  Error: {result['error']}\n")
            else:
                f.write(f"  Average FPS: {result['avg_fps']:.1f}\n")
                f.write(f"  Average processing time: {result['avg_processing_time']*1000:.1f}ms\n")
                f.write(f"  Min processing time: {result['min_processing_time']*1000:.1f}ms\n")
                f.write(f"  Max processing time: {result['max_processing_time']*1000:.1f}ms\n")
            f.write("\n")
    
    print(f"\nBenchmark results saved to: {results_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare lane detection models")
    parser.add_argument("--test_data", type=str, help="Path to test images")
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth annotations")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                       help="Output directory for results")
    parser.add_argument("--benchmark_only", action="store_true", 
                       help="Run only benchmark comparison (no ground truth needed)")
    parser.add_argument("--configs", type=str, nargs="+", 
                       help="Specific configs to test (default: all)")
    
    args = parser.parse_args()
    
    # Get test configurations
    all_configs = create_test_configs()
    
    if args.configs:
        configs = {name: all_configs[name] for name in args.configs if name in all_configs}
    else:
        configs = all_configs
    
    print(f"Testing {len(configs)} configurations...")
    
    if args.benchmark_only or not args.ground_truth:
        # Run benchmark comparison
        run_benchmark_comparison(configs, args.output_dir)
    else:
        # Run full evaluation with ground truth
        if not args.test_data:
            print("Error: --test_data required for full evaluation")
            return
        
        # Create pipelines
        pipelines = {}
        for config_name, config in configs.items():
            try:
                pipelines[config_name] = LaneDetectionPipeline(config)
            except Exception as e:
                print(f"Failed to create pipeline for {config_name}: {e}")
        
        # Run comparison
        results = compare_models(pipelines, args.test_data, args.ground_truth, args.output_dir)
        
        print(f"\nEvaluation complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
