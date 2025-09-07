"""
TensorRT utilities for optimized inference acceleration.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import os
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available. Install with: pip install tensorrt")

class TensorRTLogger(trt.ILogger):
    """Custom TensorRT logger."""
    
    def __init__(self, verbose: bool = False):
        trt.ILogger.__init__(self)
        self.verbose = verbose
    
    def log(self, severity, msg):
        if self.verbose or severity <= trt.Logger.Severity.ERROR:
            print(f"TensorRT: {msg}")

class TensorRTEngine:
    """TensorRT engine wrapper for optimized inference."""
    
    def __init__(
        self,
        engine_path: str,
        input_shape: Tuple[int, int, int, int],
        output_shapes: List[Tuple[int, ...]],
        device: int = 0
    ):
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is not available")
        
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.output_shapes = output_shapes
        self.device = device
        
        self.logger = TensorRTLogger()
        self.engine = None
        self.context = None
        self.stream = None
        
        self._load_engine()
        self._allocate_buffers()
    
    def _load_engine(self):
        """Load TensorRT engine from file."""
        if os.path.exists(self.engine_path):
            # Load existing engine
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
    
    def _allocate_buffers(self):
        """Allocate GPU memory buffers."""
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_bindings):
            binding = self.engine.get_binding_name(i)
            size = trt.volume(self.engine.get_binding_shape(i)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.allocations.append((host_mem, device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs.append((binding, host_mem, device_mem))
            else:
                self.outputs.append((binding, host_mem, device_mem))
    
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run inference on input data."""
        # Copy input data to host buffer
        np.copyto(self.inputs[0][1], input_data.ravel())
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(self.inputs[0][2], self.inputs[0][1], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=[inp[2] for inp in self.inputs] + 
                                     [out[2] for out in self.outputs], 
                                     stream_handle=self.stream.handle)
        
        # Transfer outputs back to CPU
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output[1], output[2], self.stream)
            outputs.append(output[1])
        
        # Synchronize
        self.stream.synchronize()
        
        # Reshape outputs
        reshaped_outputs = []
        for i, output in enumerate(outputs):
            shape = self.output_shapes[i]
            reshaped_outputs.append(output.reshape(shape))
        
        return reshaped_outputs
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'allocations'):
            for host_mem, device_mem in self.allocations:
                device_mem.free()

def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    input_shape: Tuple[int, int, int, int],
    max_batch_size: int = 1,
    precision: str = "fp16",
    workspace_size: int = 1 << 30  # 1GB
) -> str:
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        input_shape: Input tensor shape (batch, channels, height, width)
        max_batch_size: Maximum batch size
        precision: Precision mode ("fp32", "fp16", "int8")
        workspace_size: Workspace size in bytes
        
    Returns:
        Path to built engine
    """
    if not TENSORRT_AVAILABLE:
        raise ImportError("TensorRT is not available")
    
    logger = TensorRTLogger(verbose=True)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size
    
    # Set precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: INT8 requires calibration dataset
    
    # Set optimization profile
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1, *input_shape[1:]), (max_batch_size, *input_shape[1:]), (max_batch_size, *input_shape[1:]))
    config.add_optimization_profile(profile)
    
    # Build engine
    print("Building TensorRT engine...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("Failed to build TensorRT engine")
        return None
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved: {engine_path}")
    return engine_path

def create_tensorrt_engine_script(
    onnx_path: str,
    output_dir: str = "models/tensorrt",
    input_size: Tuple[int, int] = (640, 640),
    precision: str = "fp16"
) -> str:
    """
    Create a script to build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Directory to save TensorRT engine
        input_size: Input image size
        precision: Precision mode
        
    Returns:
        Path to created script
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = Path(onnx_path).stem
    engine_path = os.path.join(output_dir, f"{model_name}.trt")
    
    script_content = f'''#!/usr/bin/env python3
"""
Auto-generated script to build TensorRT engine from ONNX model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.tensorrt_utils import build_engine_from_onnx

def main():
    onnx_path = "{onnx_path}"
    engine_path = "{engine_path}"
    input_size = {input_size}
    precision = "{precision}"
    
    print(f"Building TensorRT engine from {{onnx_path}}...")
    print(f"Input size: {{input_size}}")
    print(f"Precision: {{precision}}")
    
    # Build engine
    result_path = build_engine_from_onnx(
        onnx_path=onnx_path,
        engine_path=engine_path,
        input_shape=(1, 3, input_size[0], input_size[1]),
        precision=precision
    )
    
    if result_path:
        print(f"TensorRT engine built successfully: {{result_path}}")
    else:
        print("Failed to build TensorRT engine")

if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(output_dir, f"build_{model_name}_tensorrt.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"TensorRT build script created: {script_path}")
    return script_path

class TensorRTDetector:
    """TensorRT-based detector wrapper."""
    
    def __init__(
        self,
        engine_path: str,
        input_shape: Tuple[int, int, int, int],
        output_shapes: List[Tuple[int, ...]]
    ):
        self.engine = TensorRTEngine(engine_path, input_shape, output_shapes)
        self.input_shape = input_shape
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for TensorRT inference."""
        # Resize image
        h, w = image.shape[:2]
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        resized = cv2.resize(image, (target_w, target_h))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to NCHW format
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor
    
    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """Run TensorRT inference."""
        input_tensor = self.preprocess(image)
        outputs = self.engine.infer(input_tensor)
        return outputs
