#!/usr/bin/env python3
"""
Create a sample video with lane-like features for testing.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_sample_video(output_path: str = "examples/sample_video.mp4", duration: int = 10, fps: int = 30):
    """Create a sample video with synthetic lane lines."""
    
    # Create examples directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Video properties
    width, height = 1280, 720
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some background noise
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Add road surface
        road_color = (40, 40, 40)
        cv2.rectangle(frame, (0, height//2), (width, height), road_color, -1)
        
        # Add lane lines with slight movement
        time_offset = frame_num * 0.1
        
        # Left lane line (curved)
        left_points = []
        for y in range(height//2, height, 5):
            x = int(width//4 + 50 * np.sin((y - height//2) * 0.01 + time_offset))
            left_points.append((x, y))
        
        # Right lane line (curved)
        right_points = []
        for y in range(height//2, height, 5):
            x = int(3*width//4 + 40 * np.sin((y - height//2) * 0.008 + time_offset))
            right_points.append((x, y))
        
        # Draw lane lines
        for i in range(len(left_points) - 1):
            cv2.line(frame, left_points[i], left_points[i+1], (255, 255, 255), 5)
        
        for i in range(len(right_points) - 1):
            cv2.line(frame, right_points[i], right_points[i+1], (255, 255, 255), 5)
        
        # Add center line (dashed)
        for y in range(height//2, height, 20):
            x = width//2 + int(10 * np.sin((y - height//2) * 0.02 + time_offset))
            cv2.line(frame, (x, y), (x, y+10), (255, 255, 0), 3)
        
        # Add some moving objects (cars)
        car_x = int(width//2 + 100 * np.sin(time_offset * 0.5))
        car_y = height//2 + 100
        cv2.rectangle(frame, (car_x-30, car_y-20), (car_x+30, car_y+20), (100, 100, 255), -1)
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Sample video created: {output_path}")
    print(f"Duration: {duration}s, FPS: {fps}, Resolution: {width}x{height}")

if __name__ == "__main__":
    create_sample_video()
