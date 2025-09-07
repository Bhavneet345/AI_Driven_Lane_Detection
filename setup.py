#!/usr/bin/env python3
"""
Setup script for AI-Driven Lane Detection system.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f" {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "models/onnx",
        "models/tensorrt", 
        "data/ground_truth",
        "data/test_images",
        "evaluation_results",
        "runs",
        "examples"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f" Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies."""
    # Install basic dependencies first
    basic_deps = [
        "opencv-python",
        "numpy", 
        "PyYAML",
        "rich",
        "scikit-learn",
        "matplotlib",
        "Pillow",
        "pytest"
    ]
    
    for dep in basic_deps:
        if not run_command(f"pip3 install {dep}", f"Installing {dep}"):
            print(f"  Warning: Failed to install {dep}")
    
    # Try to install PyTorch (optional)
    if not run_command("pip3 install torch", "Installing PyTorch"):
        print("  Warning: PyTorch installation failed - deep learning features may not work")
    
    # Try to install ONNX (optional)
    if not run_command("pip3 install onnx onnxruntime", "Installing ONNX"):
        print("  Warning: ONNX installation failed - model export features may not work")

def create_sample_data():
    """Create sample data for testing."""
    if run_command("python3 scripts/create_sample_video.py", "Creating sample video"):
        print(" Sample video created successfully")
    else:
        print("  Warning: Failed to create sample video")

def make_executable():
    """Make scripts executable."""
    scripts = [
        "main.py",
        "run_tests.py", 
        "run_export.py",
        "scripts/create_sample_video.py",
        "scripts/run_tests.py",
        "scripts/export_models.py",
        "scripts/evaluate_models.py",
        "tests/test_postprocess.py",
        "tests/test_fps_tracker.py",
        "tests/test_basic.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
            print(f" Made executable: {script}")

def verify_installation():
    """Verify the installation."""
    print("\n Verifying installation...")
    
    # Test basic imports
    try:
        import cv2
        print(" OpenCV imported successfully")
    except ImportError:
        print(" OpenCV import failed")
    
    try:
        import numpy as np
        print(" NumPy imported successfully")
    except ImportError:
        print(" NumPy import failed")
    
    try:
        import sklearn
        print(" scikit-learn imported successfully")
    except ImportError:
        print(" scikit-learn import failed")
    
    # Test main entry point
    if run_command("python3 main.py --help", "Testing main entry point"):
        print(" Main entry point working")
    else:
        print(" Main entry point failed")
    
    # Test basic functionality
    if run_command("python3 tests/test_basic.py", "Running basic tests"):
        print(" Basic tests passed")
    else:
        print(" Basic tests failed")

def main():
    """Main setup function."""
    print(" Setting up AI-Driven Lane Detection System")
    print("=" * 50)
    
    # Create directories
    print("\n Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n Installing dependencies...")
    install_dependencies()
    
    # Create sample data
    print("\n Creating sample data...")
    create_sample_data()
    
    # Make scripts executable
    print("\n Making scripts executable...")
    make_executable()
    
    # Verify installation
    print("\n Verifying installation...")
    verify_installation()
    
    print("\n" + "=" * 50)
    print(" Setup completed!")
    print("\n Next steps:")
    print("1. Run tests: python3 run_tests.py")
    print("2. Run inference: python3 main.py --input examples/sample_video.mp4")
    print("3. Export models: python3 run_export.py")
    print("4. Check README.md for more examples")

if __name__ == "__main__":
    main()
