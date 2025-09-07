#!/usr/bin/env python3
"""
Main entry point for the AI-Driven Lane Detection system.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli.infer import main

if __name__ == "__main__":
    main()
