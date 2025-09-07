#!/usr/bin/env python3
"""
Basic tests that don't require external dependencies.
"""

import sys
import unittest
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests."""
    
    def test_imports(self):
        """Test that basic imports work."""
        try:
            from src.utils.viz import FPSTracker, draw_fps
            self.assertTrue(True, "FPS tracker imports successfully")
        except ImportError as e:
            self.fail(f"Failed to import FPS tracker: {e}")
    
    def test_fps_tracker_basic(self):
        """Test basic FPS tracker functionality without OpenCV."""
        try:
            from src.utils.viz import FPSTracker
            
            # Test initialization
            tracker = FPSTracker()
            self.assertEqual(tracker.ema, 0.9)
            self.assertEqual(tracker.fps, 0.0)
            
            # Test first tick
            fps = tracker.tick()
            self.assertGreater(fps, 0.0)
            self.assertLess(fps, 1000000.0)  # Allow for very high FPS in testing
            
            # Test multiple ticks
            for _ in range(5):
                time.sleep(0.01)
                fps = tracker.tick()
                self.assertGreater(fps, 0.0)
            
        except ImportError as e:
            self.skipTest(f"Skipping FPS tracker test due to import error: {e}")
    
    def test_numpy_operations(self):
        """Test basic numpy operations for polyline extraction simulation."""
        # Create dummy binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        # Add some white pixels (lane-like structure)
        mask[50:60, 20:80] = 255  # Horizontal line
        mask[30:70, 40:45] = 255  # Vertical line
        
        # Test edge point extraction simulation
        y_coords, x_coords = np.where(mask > 0)
        edge_points = np.column_stack((x_coords, y_coords))
        
        self.assertGreater(len(edge_points), 0)
        self.assertEqual(edge_points.shape[1], 2)
        
        # Test sorting
        sorted_points = edge_points[edge_points[:, 1].argsort()]
        self.assertTrue(np.all(np.diff(sorted_points[:, 1]) >= 0))
    
    def test_polynomial_fitting_simulation(self):
        """Test polynomial fitting simulation without sklearn."""
        # Create dummy lane points
        x = np.array([100, 105, 110, 115, 120, 125])
        y = np.array([200, 210, 220, 230, 240, 250])
        
        # Simulate polynomial fitting (simplified)
        if len(x) >= 3:
            # Simple linear fit simulation
            coeffs = np.polyfit(x, y, 1)
            y_fit = np.polyval(coeffs, x)
            
            # Check that fit is reasonable
            mse = np.mean((y - y_fit) ** 2)
            self.assertLess(mse, 100.0)  # Should be small error
    
    def test_lane_validation_simulation(self):
        """Test lane validation logic simulation."""
        # Valid lane (enough points)
        valid_lane = np.array([[100, 200], [105, 210], [110, 220], [115, 230]], dtype=np.int32)
        
        # Invalid lane (too few points)
        invalid_lane = np.array([[100, 200]], dtype=np.int32)
        
        # Simulate validation
        def validate_lane(lane, min_points=2):
            return len(lane) >= min_points
        
        self.assertTrue(validate_lane(valid_lane))
        self.assertFalse(validate_lane(invalid_lane))
    
    def test_performance_measurement(self):
        """Test performance measurement functionality."""
        # Test timing
        start_time = time.time()
        
        # Simulate some work
        dummy_data = np.random.rand(1000, 1000)
        result = np.sum(dummy_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertGreater(processing_time, 0.0)
        self.assertLess(processing_time, 1.0)  # Should be fast
        self.assertIsInstance(result, float)
    
    def test_fps_calculation_simulation(self):
        """Test FPS calculation simulation."""
        # Simulate frame processing times
        frame_times = [0.016, 0.017, 0.015, 0.018, 0.016]  # ~60 FPS
        
        # Calculate FPS
        avg_frame_time = np.mean(frame_times)
        fps = 1.0 / avg_frame_time
        
        self.assertGreater(fps, 50.0)
        self.assertLess(fps, 70.0)
        self.assertAlmostEqual(fps, 60.0, delta=10.0)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty array
        empty_array = np.array([])
        self.assertEqual(len(empty_array), 0)
        
        # Single point
        single_point = np.array([[100, 200]])
        self.assertEqual(single_point.shape, (1, 2))
        
        # Zero array
        zero_array = np.zeros((10, 10))
        self.assertEqual(zero_array.shape, (10, 10))
        self.assertTrue(np.all(zero_array == 0))

def run_basic_tests():
    """Run basic tests without external dependencies."""
    print("Running Basic Tests (No External Dependencies)")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBasicFunctionality)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n" + "=" * 50)
    print(f"Basic Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
    print(f"  Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.skipped:
        print(f"\nSkipped:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_basic_tests()
    
    if success:
        print("\n All basic tests passed!")
        print("\nNote: For full testing with OpenCV and other dependencies,")
        print("install requirements: pip install -r requirements.txt")
        print("Then run: python3 tests/test_postprocess.py")
    else:
        print("\n Some basic tests failed!")
    
    sys.exit(0 if success else 1)
