#!/usr/bin/env python3
"""
Script to export PyTorch models to ONNX and TensorRT formats.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.onnx_export import (
    export_yolop_to_onnx, 
    export_scnn_to_onnx,
    create_model_export_script,
    verify_onnx_model
)
from src.utils.tensorrt_utils import (
    build_engine_from_onnx,
    create_tensorrt_engine_script
)

def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX and TensorRT")
    parser.add_argument("--model_type", choices=["yolop", "scnn", "both"], default="both",
                       help="Type of model to export")
    parser.add_argument("--pytorch_path", type=str, help="Path to PyTorch model")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")
    parser.add_argument("--input_size", type=int, nargs=2, default=[640, 640], 
                       help="Input size (height width)")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"], 
                       default="fp16", help="TensorRT precision")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX models")
    
    args = parser.parse_args()
    
    # Create output directories
    onnx_dir = os.path.join(args.output_dir, "onnx")
    tensorrt_dir = os.path.join(args.output_dir, "tensorrt")
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(tensorrt_dir, exist_ok=True)
    
    models_to_export = []
    if args.model_type in ["yolop", "both"]:
        models_to_export.append(("yolop", [640, 640]))
    if args.model_type in ["scnn", "both"]:
        models_to_export.append(("scnn", [512, 256]))
    
    for model_name, input_size in models_to_export:
        print(f"\nExporting {model_name.upper()} model...")
        
        # Set model path
        if args.pytorch_path:
            pytorch_path = args.pytorch_path
        else:
            pytorch_path = f"models/{model_name}.pth"
        
        if not os.path.exists(pytorch_path):
            print(f"Warning: PyTorch model not found at {pytorch_path}")
            print(f"Creating placeholder export script...")
            
            # Create export script for later use
            script_path = create_model_export_script(
                model_name, pytorch_path, onnx_dir, input_size
            )
            print(f"Export script created: {script_path}")
            continue
        
        # Export to ONNX
        onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
        try:
            if model_name == "yolop":
                export_yolop_to_onnx(pytorch_path, onnx_path, input_size)
            else:
                export_scnn_to_onnx(pytorch_path, onnx_path, input_size)
            
            print(f"ONNX model exported: {onnx_path}")
            
            # Verify ONNX model
            if args.verify:
                input_shape = (1, 3, input_size[0], input_size[1])
                if verify_onnx_model(onnx_path, input_shape):
                    print(f"ONNX model verification successful")
                else:
                    print(f"ONNX model verification failed")
            
            # Build TensorRT engine
            try:
                engine_path = os.path.join(tensorrt_dir, f"{model_name}.trt")
                build_engine_from_onnx(
                    onnx_path, 
                    engine_path, 
                    input_shape=(1, 3, input_size[0], input_size[1]),
                    precision=args.precision
                )
                print(f"TensorRT engine built: {engine_path}")
            except Exception as e:
                print(f"TensorRT build failed: {e}")
                print("Creating TensorRT build script...")
                create_tensorrt_engine_script(onnx_path, tensorrt_dir, input_size, args.precision)
        
        except Exception as e:
            print(f"Export failed for {model_name}: {e}")
    
    print(f"\nExport complete! Check {args.output_dir} for exported models.")

if __name__ == "__main__":
    main()
