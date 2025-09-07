#!/usr/bin/env python3
"""
Test runner entry point.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scripts.run_tests import main

if __name__ == "__main__":
    main()
