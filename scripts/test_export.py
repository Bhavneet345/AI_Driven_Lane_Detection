#!/usr/bin/env python3
"""
Test script for model export functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from export import ModelExporter

def test_export_functionality():
    """Test the model export functionality."""
    print("Testing Model Export Functionality")
    print("=" * 40)
    
    # Create exporter
    exporter = ModelExporter(output_dir="test_exported_models")
    
    # Test 1: Single model export
    print("\n1. Testing single model export (640x640)")
    try:
        results = exporter.export_yolop_model(input_size=(640, 640))
        print(" Single model export successful")
        print(f"  - Model name: {results.get('model_name', 'N/A')}")
        print(f"  - ONNX path: {results.get('onnx_path', 'N/A')}")
        print(f"  - Validation passed: {results.get('validation', {}).get('validation_passed', 'N/A')}")
    except Exception as e:
        print(f" Single model export failed: {e}")
    
    # Test 2: Multiple sizes export
    print("\n2. Testing multiple sizes export")
    try:
        sizes = [(640, 640), (512, 512)]
        results = exporter.export_multiple_sizes(sizes)
        print(" Multiple sizes export successful")
        print(f"  - Exported {len(results)} models")
        for size, result in results.items():
            if 'error' not in result:
                print(f"    - {size}:  Success")
            else:
                print(f"    - {size}:  {result['error']}")
    except Exception as e:
        print(f" Multiple sizes export failed: {e}")
    
    # Test 3: Generate report
    print("\n3. Testing report generation")
    try:
        report_path = exporter.generate_report()
        print(f" Report generated: {report_path}")
    except Exception as e:
        print(f" Report generation failed: {e}")
    
    print("\n" + "=" * 40)
    print("Export Test Complete!")

if __name__ == "__main__":
    test_export_functionality()
