#!/usr/bin/env python3
"""
Test runner script for the lane detection system.
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def run_all_tests():
    """Run all tests in the tests directory."""
    # Discover and run tests
    test_dir = Path(__file__).parent.parent / "tests"
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern="test_*.py")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n" + "=" * 50)
    print(f"Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

def run_specific_test(test_name):
    """Run a specific test."""
    test_dir = Path(__file__).parent.parent / "tests"
    test_file = test_dir / f"test_{test_name}.py"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return False
    
    # Import and run specific test
    sys.path.append(str(test_dir))
    
    try:
        module_name = f"test_{test_name}"
        module = __import__(module_name)
        
        # Run the test
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except Exception as e:
        print(f"Error running test {test_name}: {e}")
        return False

def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for lane detection system")
    parser.add_argument("--test", type=str, help="Run specific test (e.g., 'postprocess')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.test:
        print(f"Running specific test: {args.test}")
        success = run_specific_test(args.test)
    else:
        print("Running all tests...")
        success = run_all_tests()
    
    if success:
        print("\n All tests passed!")
        sys.exit(0)
    else:
        print("\n Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
