"""
ONNX export utilities for model conversion and deployment.
"""

from __future__ import annotations
import torch
import torch.onnx
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os
from pathlib import Path

def export_pytorch_to_onnx(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    output_path: str,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 11
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        input_shape: Input tensor shape (batch, channels, height, width)
        output_path: Path to save ONNX model
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axes configuration
        opset_version: ONNX opset version
        
    Returns:
        Path to exported ONNX model
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Default names
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
    
    # Default dynamic axes for batch dimension
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"Model exported to ONNX: {output_path}")
    return output_path

def export_yolop_to_onnx(
    model_path: str,
    output_path: str,
    input_size: Tuple[int, int] = (640, 640),
    batch_size: int = 1
) -> str:
    """
    Export YOLOP model to ONNX format.
    
    Args:
        model_path: Path to PyTorch YOLOP model
        output_path: Path to save ONNX model
        input_size: Input image size (height, width)
        batch_size: Batch size for export
        
    Returns:
        Path to exported ONNX model
    """
    # Load PyTorch model
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Input shape: (batch, channels, height, width)
    input_shape = (batch_size, 3, input_size[0], input_size[1])
    
    # YOLOP specific configuration
    input_names = ["images"]
    output_names = ["output0"]  # YOLOP typically has one output
    
    dynamic_axes = {
        "images": {0: "batch_size"},
        "output0": {0: "batch_size"}
    }
    
    return export_pytorch_to_onnx(
        model=model,
        input_shape=input_shape,
        output_path=output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

def export_scnn_to_onnx(
    model_path: str,
    output_path: str,
    input_size: Tuple[int, int] = (512, 256),
    batch_size: int = 1
) -> str:
    """
    Export SCNN model to ONNX format.
    
    Args:
        model_path: Path to PyTorch SCNN model
        output_path: Path to save ONNX model
        input_size: Input image size (height, width)
        batch_size: Batch size for export
        
    Returns:
        Path to exported ONNX model
    """
    # Load PyTorch model
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Input shape: (batch, channels, height, width)
    input_shape = (batch_size, 3, input_size[0], input_size[1])
    
    # SCNN specific configuration
    input_names = ["input"]
    output_names = ["segmentation_output"]
    
    dynamic_axes = {
        "input": {0: "batch_size"},
        "segmentation_output": {0: "batch_size"}
    }
    
    return export_pytorch_to_onnx(
        model=model,
        input_shape=input_shape,
        output_path=output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

def verify_onnx_model(onnx_path: str, input_shape: Tuple[int, int, int, int]) -> bool:
    """
    Verify ONNX model by running inference.
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape
        
    Returns:
        True if model is valid, False otherwise
    """
    try:
        import onnxruntime as ort
        
        # Create session
        session = ort.InferenceSession(onnx_path)
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"ONNX model verification successful: {onnx_path}")
        print(f"Output shape: {[output.shape for output in outputs]}")
        
        return True
        
    except Exception as e:
        print(f"ONNX model verification failed: {e}")
        return False

def optimize_onnx_model(
    input_path: str,
    output_path: str,
    optimization_level: str = "basic"
) -> str:
    """
    Optimize ONNX model for better performance.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model
        optimization_level: Optimization level ("basic", "extended", "all")
        
    Returns:
        Path to optimized ONNX model
    """
    try:
        import onnx
        from onnx import optimizer
        
        # Load model
        model = onnx.load(input_path)
        
        # Define optimization passes
        if optimization_level == "basic":
            passes = ['eliminate_identity', 'eliminate_nop_transpose', 'fuse_consecutive_transposes']
        elif optimization_level == "extended":
            passes = ['eliminate_identity', 'eliminate_nop_transpose', 'fuse_consecutive_transposes',
                     'fuse_transpose_into_gemm', 'eliminate_unused_initializer']
        else:  # all
            passes = optimizer.get_available_passes()
        
        # Optimize model
        optimized_model = optimizer.optimize(model, passes)
        
        # Save optimized model
        onnx.save(optimized_model, output_path)
        
        print(f"ONNX model optimized and saved: {output_path}")
        return output_path
        
    except ImportError:
        print("ONNX optimizer not available. Install onnx with: pip install onnx")
        return input_path
    except Exception as e:
        print(f"ONNX optimization failed: {e}")
        return input_path

def create_model_export_script(
    model_type: str,
    model_path: str,
    output_dir: str = "models/onnx",
    input_size: Tuple[int, int] = (640, 640)
) -> str:
    """
    Create a script to export models to ONNX format.
    
    Args:
        model_type: Type of model ("yolop" or "scnn")
        model_path: Path to PyTorch model
        output_dir: Directory to save ONNX models
        input_size: Input image size
        
    Returns:
        Path to created export script
    """
    os.makedirs(output_dir, exist_ok=True)
    
    script_content = f'''#!/usr/bin/env python3
"""
Auto-generated script to export {model_type.upper()} model to ONNX.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.onnx_export import export_{model_type}_to_onnx, verify_onnx_model, optimize_onnx_model

def main():
    model_path = "{model_path}"
    output_dir = "{output_dir}"
    input_size = {input_size}
    
    # Create output filename
    model_name = os.path.basename(model_path).replace('.pth', '').replace('.pt', '')
    onnx_path = os.path.join(output_dir, f"{{model_name}}_onnx.onnx")
    optimized_path = os.path.join(output_dir, f"{{model_name}}_optimized.onnx")
    
    print(f"Exporting {{model_path}} to ONNX...")
    
    # Export to ONNX
    exported_path = export_{model_type}_to_onnx(
        model_path=model_path,
        output_path=onnx_path,
        input_size=input_size
    )
    
    # Verify model
    input_shape = (1, 3, input_size[0], input_size[1])
    if verify_onnx_model(exported_path, input_shape):
        print("Model verification successful!")
        
        # Optimize model
        optimized_path = optimize_onnx_model(exported_path, optimized_path)
        print(f"Export complete! Files saved to {{output_dir}}")
    else:
        print("Model verification failed!")

if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(output_dir, f"export_{model_type}_onnx.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Export script created: {script_path}")
    return script_path
