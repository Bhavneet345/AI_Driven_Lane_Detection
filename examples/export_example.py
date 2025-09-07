#!/usr/bin/env python3
"""
Example script demonstrating model export functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from export import ModelExporter

def main():
    """Demonstrate model export functionality."""
    print("Model Export Example")
    print("=" * 30)
    
    # Create exporter
    exporter = ModelExporter(output_dir="example_exported_models")
    
    # Example 1: Export single model
    print("\n1. Exporting single YOLOP model (640x640)")
    results = exporter.export_yolop_model(input_size=(640, 640))
    
    if 'error' not in results:
        print(" Export successful!")
        print(f"  - PyTorch model: {results['pytorch_path']}")
        print(f"  - ONNX model: {results['onnx_path']}")
        print(f"  - Validation: {'PASSED' if results['validation']['validation_passed'] else 'FAILED'}")
        print(f"  - Speedup: {results['benchmark']['speedup']:.2f}x")
    else:
        print(f" Export failed: {results['error']}")
    
    # Example 2: Export multiple sizes
    print("\n2. Exporting models for multiple sizes")
    sizes = [(640, 640), (512, 512), (416, 416)]
    all_results = exporter.export_multiple_sizes(sizes)
    
    print(f"Exported {len(all_results)} models:")
    for size, result in all_results.items():
        if 'error' not in result:
            print(f"  - {size}:  Success (speedup: {result['benchmark']['speedup']:.2f}x)")
        else:
            print(f"  - {size}:  {result['error']}")
    
    # Example 3: Generate report
    print("\n3. Generating export report")
    report_path = exporter.generate_report()
    print(f"Report generated: {report_path}")
    
    print("\nExport examples completed!")

if __name__ == "__main__":
    main()
