#!/usr/bin/env python3
"""
Model Export Script: Convert PyTorch models to ONNX and validate output equivalence.
Uses YOLOP as placeholder model for demonstration.
"""

import argparse
import os
import sys
import time
import torch
import torch.onnx
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.yolop import YOLOPDetector
from src.utils.onnx_export import export_pytorch_to_onnx, verify_onnx_model

class ModelExporter:
    """Comprehensive model export and validation utility."""
    
    def __init__(self, output_dir: str = "exported_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "onnx").mkdir(exist_ok=True)
        (self.output_dir / "pytorch").mkdir(exist_ok=True)
        (self.output_dir / "validation").mkdir(exist_ok=True)
        
        self.results = {}
    
    def create_placeholder_yolop_model(self, input_size: Tuple[int, int] = (640, 640)) -> torch.nn.Module:
        """Create a placeholder YOLOP model for demonstration."""
        
        class PlaceholderYOLOP(torch.nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.input_size = input_size
                
                # Simplified YOLOP-like architecture
                self.backbone = torch.nn.Sequential(
                    # Initial conv layers
                    torch.nn.Conv2d(3, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True),
                    
                    # ResNet-like blocks
                    self._make_layer(64, 128, 2),
                    self._make_layer(128, 256, 2),
                    self._make_layer(256, 512, 2),
                    self._make_layer(512, 1024, 2),
                )
                
                # Detection head (simplified)
                self.detection_head = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten(),
                    torch.nn.Linear(1024, 512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(512, 1000),  # Placeholder output size
                )
                
                # Lane detection head
                self.lane_head = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(128, 1, 1),  # Single channel lane mask
                )
            
            def _make_layer(self, in_channels: int, out_channels: int, blocks: int):
                layers = []
                layers.append(torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
                layers.append(torch.nn.BatchNorm2d(out_channels))
                layers.append(torch.nn.ReLU(inplace=True))
                
                for _ in range(blocks - 1):
                    layers.append(torch.nn.Conv2d(out_channels, out_channels, 3, padding=1))
                    layers.append(torch.nn.BatchNorm2d(out_channels))
                    layers.append(torch.nn.ReLU(inplace=True))
                
                return torch.nn.Sequential(*layers)
            
            def forward(self, x):
                # Backbone features
                features = self.backbone(x)
                
                # Detection output
                detection = self.detection_head(features)
                
                # Lane segmentation
                lane_mask = self.lane_head(features)
                lane_mask = torch.nn.functional.interpolate(
                    lane_mask, 
                    size=(self.input_size[0], self.input_size[1]), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                return detection, lane_mask
        
        return PlaceholderYOLOP(input_size)
    
    def save_pytorch_model(self, model: torch.nn.Module, model_name: str) -> str:
        """Save PyTorch model to file."""
        model_path = self.output_dir / "pytorch" / f"{model_name}.pth"
        
        # Save model state dict
        torch.save(model.state_dict(), model_path)
        
        # Also save full model for easier loading
        full_model_path = self.output_dir / "pytorch" / f"{model_name}_full.pth"
        torch.save(model, full_model_path)
        
        print(f"PyTorch model saved: {model_path}")
        print(f"Full model saved: {full_model_path}")
        
        return str(model_path)
    
    def export_to_onnx(self, 
                      model: torch.nn.Module, 
                      model_name: str,
                      input_size: Tuple[int, int] = (640, 640),
                      batch_size: int = 1) -> str:
        """Export PyTorch model to ONNX format."""
        
        model.eval()
        onnx_path = self.output_dir / "onnx" / f"{model_name}.onnx"
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])
        
        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['detection_output', 'lane_mask'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'detection_output': {0: 'batch_size'},
                    'lane_mask': {0: 'batch_size'}
                },
                verbose=False
            )
            
            print(f"ONNX model exported: {onnx_path}")
            return str(onnx_path)
            
        except Exception as e:
            print(f"ONNX export failed: {e}")
            return None
    
    def validate_onnx_model(self, onnx_path: str, pytorch_model: torch.nn.Module, 
                           input_size: Tuple[int, int] = (640, 640)) -> Dict[str, Any]:
        """Validate ONNX model against PyTorch model."""
        
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not available. Install with: pip install onnxruntime")
            return {"error": "onnxruntime not available"}
        
        # Create test input
        test_input = torch.randn(1, 3, input_size[0], input_size[1])
        
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_detection, pytorch_lane = pytorch_model(test_input)
            pytorch_detection = pytorch_detection.cpu().numpy()
            pytorch_lane = pytorch_lane.cpu().numpy()
        
        # ONNX inference
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        
        onnx_outputs = session.run(None, {input_name: test_input.numpy()})
        onnx_detection, onnx_lane = onnx_outputs
        
        # Calculate differences
        detection_diff = np.abs(pytorch_detection - onnx_detection).max()
        lane_diff = np.abs(pytorch_lane - onnx_lane).max()
        
        # Calculate relative differences
        detection_rel_diff = detection_diff / (np.abs(pytorch_detection).max() + 1e-8)
        lane_rel_diff = lane_diff / (np.abs(pytorch_lane).max() + 1e-8)
        
        validation_results = {
            "pytorch_detection_shape": pytorch_detection.shape,
            "onnx_detection_shape": onnx_detection.shape,
            "pytorch_lane_shape": pytorch_lane.shape,
            "onnx_lane_shape": onnx_lane.shape,
            "detection_max_diff": float(detection_diff),
            "lane_max_diff": float(lane_diff),
            "detection_rel_diff": float(detection_rel_diff),
            "lane_rel_diff": float(lane_rel_diff),
            "validation_passed": detection_rel_diff < 1e-3 and lane_rel_diff < 1e-3
        }
        
        print(f"Validation Results:")
        print(f"  Detection max diff: {detection_diff:.6f}")
        print(f"  Lane max diff: {lane_diff:.6f}")
        print(f"  Detection rel diff: {detection_rel_diff:.6f}")
        print(f"  Lane rel diff: {lane_rel_diff:.6f}")
        print(f"  Validation passed: {validation_results['validation_passed']}")
        
        return validation_results
    
    def benchmark_models(self, pytorch_model: torch.nn.Module, onnx_path: str, 
                        input_size: Tuple[int, int] = (640, 640), num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark PyTorch vs ONNX performance."""
        
        try:
            import onnxruntime as ort
        except ImportError:
            return {"error": "onnxruntime not available"}
        
        # Create test input
        test_input = torch.randn(1, 3, input_size[0], input_size[1])
        
        # PyTorch benchmark
        pytorch_model.eval()
        pytorch_times = []
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = pytorch_model(test_input)
        
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = pytorch_model(test_input)
            pytorch_times.append(time.time() - start_time)
        
        # ONNX benchmark
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        onnx_times = []
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: test_input.numpy()})
        
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run(None, {input_name: test_input.numpy()})
            onnx_times.append(time.time() - start_time)
        
        # Calculate statistics
        pytorch_avg = np.mean(pytorch_times)
        onnx_avg = np.mean(onnx_times)
        speedup = pytorch_avg / onnx_avg
        
        benchmark_results = {
            "pytorch_avg_time": float(pytorch_avg),
            "onnx_avg_time": float(onnx_avg),
            "speedup": float(speedup),
            "pytorch_fps": float(1.0 / pytorch_avg),
            "onnx_fps": float(1.0 / onnx_avg),
            "num_runs": num_runs
        }
        
        print(f"Benchmark Results ({num_runs} runs):")
        print(f"  PyTorch avg time: {pytorch_avg*1000:.2f}ms")
        print(f"  ONNX avg time: {onnx_avg*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  PyTorch FPS: {1.0/pytorch_avg:.1f}")
        print(f"  ONNX FPS: {1.0/onnx_avg:.1f}")
        
        return benchmark_results
    
    def export_yolop_model(self, input_size: Tuple[int, int] = (640, 640)) -> Dict[str, Any]:
        """Export YOLOP model with full validation."""
        
        print(f"Exporting YOLOP model (input size: {input_size})")
        print("=" * 50)
        
        # Create placeholder model
        model = self.create_placeholder_yolop_model(input_size)
        model_name = f"yolop_{input_size[0]}x{input_size[1]}"
        
        # Save PyTorch model
        pytorch_path = self.save_pytorch_model(model, model_name)
        
        # Export to ONNX
        onnx_path = self.export_to_onnx(model, model_name, input_size)
        
        if onnx_path is None:
            return {"error": "ONNX export failed"}
        
        # Validate outputs
        print(f"\nValidating ONNX model...")
        validation_results = self.validate_onnx_model(onnx_path, model, input_size)
        
        # Benchmark performance
        print(f"\nBenchmarking models...")
        benchmark_results = self.benchmark_models(model, onnx_path, input_size)
        
        # Save results
        results = {
            "model_name": model_name,
            "input_size": input_size,
            "pytorch_path": pytorch_path,
            "onnx_path": onnx_path,
            "validation": validation_results,
            "benchmark": benchmark_results,
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save results to file
        results_path = self.output_dir / "validation" / f"{model_name}_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        self.results[model_name] = results
        
        print(f"\nExport complete! Results saved to: {results_path}")
        return results
    
    def export_multiple_sizes(self, sizes: List[Tuple[int, int]] = [(640, 640), (512, 512), (416, 416)]) -> Dict[str, Any]:
        """Export models for multiple input sizes."""
        
        print(f"Exporting models for multiple sizes: {sizes}")
        print("=" * 60)
        
        all_results = {}
        
        for size in sizes:
            print(f"\nExporting for size {size[0]}x{size[1]}...")
            try:
                results = self.export_yolop_model(size)
                all_results[f"{size[0]}x{size[1]}"] = results
            except Exception as e:
                print(f"Failed to export for size {size}: {e}")
                all_results[f"{size[0]}x{size[1]}"] = {"error": str(e)}
        
        # Save combined results
        combined_path = self.output_dir / "validation" / "all_results.yaml"
        with open(combined_path, 'w') as f:
            yaml.dump(all_results, f, default_flow_style=False)
        
        print(f"\nAll exports complete! Combined results saved to: {combined_path}")
        return all_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive export report."""
        
        report_path = self.output_dir / "export_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Model Export Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Total models exported: {len(self.results)}\n")
            f.write(f"- Output directory: {self.output_dir}\n\n")
            
            f.write("## Model Details\n\n")
            for model_name, results in self.results.items():
                f.write(f"### {model_name}\n\n")
                f.write(f"- Input size: {results.get('input_size', 'N/A')}\n")
                f.write(f"- PyTorch path: {results.get('pytorch_path', 'N/A')}\n")
                f.write(f"- ONNX path: {results.get('onnx_path', 'N/A')}\n")
                
                validation = results.get('validation', {})
                if 'validation_passed' in validation:
                    f.write(f"- Validation: {' PASSED' if validation['validation_passed'] else ' FAILED'}\n")
                    f.write(f"  - Detection max diff: {validation.get('detection_max_diff', 'N/A'):.6f}\n")
                    f.write(f"  - Lane max diff: {validation.get('lane_max_diff', 'N/A'):.6f}\n")
                
                benchmark = results.get('benchmark', {})
                if 'speedup' in benchmark:
                    f.write(f"- Performance:\n")
                    f.write(f"  - PyTorch FPS: {benchmark.get('pytorch_fps', 'N/A'):.1f}\n")
                    f.write(f"  - ONNX FPS: {benchmark.get('onnx_fps', 'N/A'):.1f}\n")
                    f.write(f"  - Speedup: {benchmark.get('speedup', 'N/A'):.2f}x\n")
                
                f.write("\n")
        
        print(f"Export report generated: {report_path}")
        return str(report_path)

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX with validation")
    parser.add_argument("--output_dir", type=str, default="exported_models", 
                       help="Output directory for exported models")
    parser.add_argument("--input_size", type=int, nargs=2, default=[640, 640], 
                       help="Input size (height width)")
    parser.add_argument("--multiple_sizes", action="store_true", 
                       help="Export for multiple input sizes")
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 512, 416], 
                       help="Input sizes for multiple export")
    parser.add_argument("--benchmark_runs", type=int, default=100, 
                       help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = ModelExporter(args.output_dir)
    
    if args.multiple_sizes:
        # Export for multiple sizes
        sizes = [(size, size) for size in args.sizes]
        results = exporter.export_multiple_sizes(sizes)
    else:
        # Export for single size
        input_size = tuple(args.input_size)
        results = exporter.export_yolop_model(input_size)
    
    # Generate report
    exporter.generate_report()
    
    print(f"\nExport process completed!")
    print(f"Check {exporter.output_dir} for exported models and results.")

if __name__ == "__main__":
    main()
