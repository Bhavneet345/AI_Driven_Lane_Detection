"""
RANSAC-based lane detection with polynomial curve fitting.
Replaces traditional Hough line detection for better curved lane handling.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class RANSACLaneDetector:
    """RANSAC-based lane detection with polynomial curve fitting."""
    
    def __init__(
        self,
        min_samples: int = 10,
        residual_threshold: float = 2.0,
        max_trials: int = 1000,
        polynomial_degree: int = 2,
        min_lane_length: float = 50.0,
        max_lane_gap: float = 100.0,
        lane_width_range: Tuple[float, float] = (200.0, 600.0)
    ):
        """
        Initialize RANSAC lane detector.
        
        Args:
            min_samples: Minimum number of samples for RANSAC
            residual_threshold: Maximum residual for inliers
            max_trials: Maximum number of RANSAC trials
            polynomial_degree: Degree of polynomial for curve fitting
            min_lane_length: Minimum lane length in pixels
            max_lane_gap: Maximum gap between lane segments
            lane_width_range: Expected lane width range (min, max)
        """
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.polynomial_degree = polynomial_degree
        self.min_lane_length = min_lane_length
        self.max_lane_gap = max_lane_gap
        self.lane_width_range = lane_width_range
        
        # Initialize RANSAC regressor
        self.ransac = RANSACRegressor(
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            random_state=42
        )
        
        # Initialize polynomial features
        self.poly_features = PolynomialFeatures(degree=polynomial_degree)
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('poly', self.poly_features),
            ('ransac', self.ransac)
        ])
    
    def detect_lanes(
        self, 
        edge_image: np.ndarray, 
        roi_top_ratio: float = 0.55
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Detect lanes using RANSAC and polynomial curve fitting.
        
        Args:
            edge_image: Binary edge image
            roi_top_ratio: Top ratio for region of interest
            
        Returns:
            Tuple of (detected_lanes, overlay_image)
        """
        # Apply region of interest
        roi_edges = self._apply_roi(edge_image, roi_top_ratio)
        
        # Find edge points
        edge_points = self._extract_edge_points(roi_edges)
        
        if len(edge_points) < self.min_samples:
            return [], self._create_overlay(edge_image, [])
        
        # Separate left and right lane candidates
        left_candidates, right_candidates = self._separate_lane_candidates(edge_points, edge_image.shape)
        
        # Detect lanes using RANSAC
        detected_lanes = []
        
        # Detect left lane
        left_lane = self._detect_single_lane(left_candidates, 'left')
        if left_lane is not None:
            detected_lanes.append(left_lane)
        
        # Detect right lane
        right_lane = self._detect_single_lane(right_candidates, 'right')
        if right_lane is not None:
            detected_lanes.append(right_lane)
        
        # Create overlay
        overlay = self._create_overlay(edge_image, detected_lanes)
        
        return detected_lanes, overlay
    
    def _apply_roi(self, edge_image: np.ndarray, top_ratio: float) -> np.ndarray:
        """Apply region of interest mask."""
        h, w = edge_image.shape[:2]
        mask = np.zeros_like(edge_image)
        
        # Create trapezoidal ROI for lane detection
        roi_points = np.array([
            [0, h],  # Bottom left
            [w * 0.1, int(h * top_ratio)],  # Top left
            [w * 0.9, int(h * top_ratio)],  # Top right
            [w, h]   # Bottom right
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [roi_points], 255)
        return cv2.bitwise_and(edge_image, mask)
    
    def _extract_edge_points(self, edge_image: np.ndarray) -> np.ndarray:
        """Extract edge points from binary image."""
        # Find all non-zero points
        y_coords, x_coords = np.where(edge_image > 0)
        
        if len(x_coords) == 0:
            return np.array([])
        
        # Combine coordinates
        points = np.column_stack((x_coords, y_coords))
        
        # Sort by y-coordinate (top to bottom)
        points = points[points[:, 1].argsort()]
        
        return points
    
    def _separate_lane_candidates(
        self, 
        edge_points: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Separate edge points into left and right lane candidates."""
        if len(edge_points) == 0:
            return np.array([]), np.array([])
        
        h, w = image_shape[:2]
        center_x = w // 2
        
        # Separate points based on x-coordinate
        left_mask = edge_points[:, 0] < center_x
        right_mask = edge_points[:, 0] >= center_x
        
        left_candidates = edge_points[left_mask]
        right_candidates = edge_points[right_mask]
        
        return left_candidates, right_candidates
    
    def _detect_single_lane(
        self, 
        candidates: np.ndarray, 
        lane_side: str
    ) -> Optional[np.ndarray]:
        """Detect a single lane using RANSAC and polynomial fitting."""
        if len(candidates) < self.min_samples:
            return None
        
        # Prepare data for polynomial fitting
        x = candidates[:, 0].reshape(-1, 1)
        y = candidates[:, 1].reshape(-1, 1)
        
        try:
            # Fit polynomial using RANSAC
            self.pipeline.fit(x, y)
            
            # Get inlier mask
            inlier_mask = self.ransac.inlier_mask_
            inlier_x = x[inlier_mask].flatten()
            inlier_y = y[inlier_mask].flatten()
            
            if len(inlier_x) < self.min_samples:
                return None
            
            # Generate lane points using fitted polynomial
            lane_points = self._generate_lane_points(
                inlier_x, inlier_y, lane_side
            )
            
            # Validate lane
            if self._validate_lane(lane_points):
                return lane_points
            else:
                return None
                
        except Exception as e:
            print(f"Error fitting {lane_side} lane: {e}")
            return None
    
    def _generate_lane_points(
        self, 
        x_points: np.ndarray, 
        y_points: np.ndarray, 
        lane_side: str
    ) -> np.ndarray:
        """Generate smooth lane points using fitted polynomial."""
        if len(x_points) == 0:
            return np.array([])
        
        # Sort points by y-coordinate
        sort_indices = np.argsort(y_points)
        x_sorted = x_points[sort_indices]
        y_sorted = y_points[sort_indices]
        
        # Create y-coordinates for interpolation
        y_min, y_max = int(y_sorted.min()), int(y_sorted.max())
        y_interp = np.arange(y_min, y_max + 1, 5)  # 5-pixel steps
        
        # Interpolate x-coordinates
        x_interp = np.interp(y_interp, y_sorted, x_sorted)
        
        # Combine coordinates
        lane_points = np.column_stack((x_interp, y_interp)).astype(np.int32)
        
        return lane_points
    
    def _validate_lane(self, lane_points: np.ndarray) -> bool:
        """Validate detected lane based on various criteria."""
        if len(lane_points) < 2:
            return False
        
        # Check minimum length
        if len(lane_points) < self.min_lane_length / 5:  # Approximate length check
            return False
        
        # Check for reasonable curvature (not too sharp turns)
        if len(lane_points) > 3:
            # Calculate curvature
            x = lane_points[:, 0]
            y = lane_points[:, 1]
            
            # Simple curvature check using second derivative
            if len(x) > 4:
                dx = np.gradient(x)
                dy = np.gradient(y)
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                
                curvature = np.abs(ddx * dy - ddy * dx) / (dx**2 + dy**2)**1.5
                max_curvature = np.max(curvature)
                
                # Reject lanes with too sharp curvature
                if max_curvature > 0.1:  # Threshold for maximum curvature
                    return False
        
        return True
    
    def _create_overlay(
        self, 
        original_image: np.ndarray, 
        lanes: List[np.ndarray]
    ) -> np.ndarray:
        """Create overlay image with detected lanes."""
        overlay = original_image.copy()
        
        # Convert to color if grayscale
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        
        # Draw lanes
        for i, lane in enumerate(lanes):
            if len(lane) < 2:
                continue
            
            # Choose color based on lane index
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            
            # Draw lane as connected line segments
            for j in range(len(lane) - 1):
                pt1 = tuple(lane[j].astype(int))
                pt2 = tuple(lane[j + 1].astype(int))
                cv2.line(overlay, pt1, pt2, color, 3)
            
            # Draw lane points
            for point in lane:
                cv2.circle(overlay, tuple(point.astype(int)), 2, color, -1)
        
        return overlay
    
    def detect_lanes_with_confidence(
        self, 
        edge_image: np.ndarray, 
        roi_top_ratio: float = 0.55
    ) -> Tuple[List[np.ndarray], List[float], np.ndarray]:
        """
        Detect lanes with confidence scores.
        
        Returns:
            Tuple of (detected_lanes, confidence_scores, overlay_image)
        """
        lanes, overlay = self.detect_lanes(edge_image, roi_top_ratio)
        confidences = []
        
        for lane in lanes:
            # Calculate confidence based on number of inliers and lane length
            confidence = min(1.0, len(lane) / 100.0)  # Simple confidence metric
            confidences.append(confidence)
        
        return lanes, confidences, overlay

def detect_lanes_ransac(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    roi_top_ratio: float = 0.55,
    min_samples: int = 10,
    residual_threshold: float = 2.0,
    polynomial_degree: int = 2
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Detect lanes using RANSAC and polynomial curve fitting.
    
    Args:
        frame: Input image frame
        canny_low: Canny edge detection low threshold
        canny_high: Canny edge detection high threshold
        roi_top_ratio: Top ratio for region of interest
        min_samples: Minimum samples for RANSAC
        residual_threshold: RANSAC residual threshold
        polynomial_degree: Polynomial degree for curve fitting
        
    Returns:
        Tuple of (overlay_image, detected_lanes)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blur, canny_low, canny_high)
    
    # Initialize RANSAC detector
    detector = RANSACLaneDetector(
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        polynomial_degree=polynomial_degree
    )
    
    # Detect lanes
    lanes, overlay = detector.detect_lanes(edges, roi_top_ratio)
    
    return overlay, lanes

def compare_detection_methods(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    roi_top_ratio: float = 0.55
) -> Dict[str, Any]:
    """
    Compare Hough line detection vs RANSAC detection.
    
    Args:
        frame: Input image frame
        canny_low: Canny edge detection low threshold
        canny_high: Canny edge detection high threshold
        roi_top_ratio: Top ratio for region of interest
        
    Returns:
        Dictionary with comparison results
    """
    import time
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    
    # Apply ROI
    h, w = edges.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, h), (0, int(h*roi_top_ratio)), (w, int(h*roi_top_ratio)), (w, h)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    edges_roi = cv2.bitwise_and(edges, mask)
    
    results = {}
    
    # Hough line detection
    start_time = time.time()
    hough_lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )
    hough_time = time.time() - start_time
    
    # RANSAC detection
    start_time = time.time()
    detector = RANSACLaneDetector()
    ransac_lanes, ransac_overlay = detector.detect_lanes(edges_roi, roi_top_ratio)
    ransac_time = time.time() - start_time
    
    # Count detected lanes
    hough_count = len(hough_lines) if hough_lines is not None else 0
    ransac_count = len(ransac_lanes)
    
    results = {
        'hough': {
            'lines': hough_count,
            'time': hough_time,
            'method': 'HoughLinesP'
        },
        'ransac': {
            'lanes': ransac_count,
            'time': ransac_time,
            'method': 'RANSAC + Polynomial'
        },
        'comparison': {
            'speedup': hough_time / ransac_time if ransac_time > 0 else 0,
            'detection_ratio': ransac_count / hough_count if hough_count > 0 else 0
        }
    }
    
    return results
