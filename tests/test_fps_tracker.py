#!/usr/bin/env python3
"""
Test FPS tracker functionality specifically.
"""

import sys
import unittest
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.viz import FPSTracker, draw_fps

class TestFPSTracker(unittest.TestCase):
    """Test FPS tracker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fps_tracker = FPSTracker()
    
    def test_initialization(self):
        """Test FPS tracker initialization."""
        tracker = FPSTracker()
        self.assertEqual(tracker.ema, 0.9)
        self.assertEqual(tracker.fps, 0.0)
        self.assertIsNotNone(tracker.prev)
    
    def test_initialization_with_custom_ema(self):
        """Test FPS tracker initialization with custom EMA."""
        tracker = FPSTracker(ema=0.5)
        self.assertEqual(tracker.ema, 0.5)
        self.assertEqual(tracker.fps, 0.0)
    
    def test_first_tick(self):
        """Test first tick behavior."""
        fps = self.fps_tracker.tick()
        self.assertGreater(fps, 0.0)
        self.assertLess(fps, 100000.0)  # Allow for very high FPS in testing
    
    def test_multiple_ticks(self):
        """Test multiple ticks."""
        fps_values = []
        
        for _ in range(10):
            time.sleep(0.01)  # 10ms delay
            fps = self.fps_tracker.tick()
            fps_values.append(fps)
        
        # All FPS values should be positive
        for fps in fps_values:
            self.assertGreater(fps, 0.0)
            self.assertLess(fps, 2000000.0)  # Allow for very high FPS in testing
        
        # FPS should be relatively stable (not jumping wildly)
        fps_std = np.std(fps_values)
        self.assertLess(fps_std, 100.0)  # Should not vary too much
    
    def test_ema_smoothing(self):
        """Test that EMA smoothing works correctly."""
        tracker = FPSTracker(ema=0.1)  # Low EMA for fast response
        
        # Simulate varying frame times
        fps_values = []
        for i in range(20):
            if i % 2 == 0:
                time.sleep(0.01)  # 10ms
            else:
                time.sleep(0.02)  # 20ms
            fps = tracker.tick()
            fps_values.append(fps)
        
        # FPS should be smoothed
        self.assertGreater(len(fps_values), 0)
        
        # Check that FPS values are reasonable
        for fps in fps_values:
            self.assertGreater(fps, 0.0)
            self.assertLess(fps, 2000000.0)  # Allow for very high FPS in testing
    
    def test_high_frequency_ticks(self):
        """Test high frequency ticks."""
        start_time = time.time()
        
        for _ in range(1000):
            fps = self.fps_tracker.tick()
        
        total_time = time.time() - start_time
        
        # Should be very fast
        self.assertLess(total_time, 0.1)  # 1000 ticks in < 100ms
        
        # Final FPS should be reasonable
        self.assertGreater(fps, 0.0)
        self.assertLess(fps, 100000.0)  # Allow for very high FPS in testing
    
    def test_draw_fps_function(self):
        """Test draw_fps function."""
        # Create test frame
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        original_frame = frame.copy()
        
        # Draw FPS
        draw_fps(frame, 30.5)
        
        # Frame should be modified
        self.assertFalse(np.array_equal(frame, original_frame))
        
        # Frame should still be valid
        self.assertEqual(frame.shape, original_frame.shape)
        self.assertEqual(frame.dtype, original_frame.dtype)
    
    def test_draw_fps_with_different_values(self):
        """Test draw_fps with different FPS values."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        
        # Test various FPS values
        test_fps_values = [0.0, 1.0, 30.0, 60.0, 120.0, 999.9]
        
        for fps in test_fps_values:
            test_frame = frame.copy()
            draw_fps(test_frame, fps)
            
            # Should not crash and frame should be modified
            self.assertFalse(np.array_equal(test_frame, frame))
    
    def test_fps_calculation_accuracy(self):
        """Test FPS calculation accuracy."""
        tracker = FPSTracker(ema=0.1)  # Low EMA for fast response
        
        # Simulate known frame rate
        target_fps = 10.0  # 10 FPS
        frame_time = 1.0 / target_fps  # 100ms per frame
        
        fps_values = []
        for _ in range(20):
            time.sleep(frame_time)
            fps = tracker.tick()
            fps_values.append(fps)
        
        # Average FPS should be close to target
        avg_fps = np.mean(fps_values[-10:])  # Use last 10 values
        fps_error = abs(avg_fps - target_fps) / target_fps
        
        # Allow 20% error due to timing precision
        self.assertLess(fps_error, 0.2)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with very high FPS
        tracker = FPSTracker()
        
        for _ in range(100):
            fps = tracker.tick()
            self.assertGreater(fps, 0.0)
            self.assertLess(fps, 2000000.0)  # Allow for very high FPS in testing
        
        # Test with very low FPS
        time.sleep(1.0)  # 1 second delay
        fps = tracker.tick()
        self.assertGreater(fps, 0.0)
        self.assertLess(fps, 10.0)  # Should be low but not zero

if __name__ == "__main__":
    unittest.main(verbosity=2)
