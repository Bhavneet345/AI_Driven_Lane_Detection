from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import time
from ..utils.preprocessing import preprocess_image, PreprocessingMethod
from ..utils.ransac_lane_detection import detect_lanes_ransac, RANSACLaneDetector
from ..models import YOLOPDetector, SCNNDetector, BaseLaneDetector
from ..models.base import LaneDetectionResult
from ..models.base import ModelType

def region_of_interest(img: np.ndarray, top_ratio: float = 0.55) -> np.ndarray:
    """Apply region of interest mask to focus on lane area."""
    h, w = img.shape[:2]
    mask = np.zeros_like(img[:, :, 0])
    polygon = np.array([[(0, h), (0, int(h*top_ratio)), (w, int(h*top_ratio)), (w, h)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, img, mask=mask)

def detect_lanes_opencv(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough: Dict[str, float] = None,
    roi_top_ratio: float = 0.55,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """OpenCV-based lane detection using Canny edge detection and Hough transforms."""
    if hough is None:
        hough = dict(rho=1, theta=np.pi/180, threshold=50, min_line_len=50, max_line_gap=150)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    edges_roi = region_of_interest(edges, roi_top_ratio)
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=hough.get("rho",1),
        theta=hough.get("theta",np.pi/180),
        threshold=int(hough.get("threshold",50)),
        minLineLength=int(hough.get("min_line_len",50)),
        maxLineGap=int(hough.get("max_line_gap",150)),
    )

    overlay = frame.copy()
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            cv2.line(overlay, (x1,y1), (x2,y2), (0,255,0), 3)

    return overlay, lines

def detect_lanes_ransac_opencv(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    roi_top_ratio: float = 0.55,
    min_samples: int = 10,
    residual_threshold: float = 2.0,
    polynomial_degree: int = 2
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """RANSAC-based lane detection with polynomial curve fitting."""
    overlay, lanes = detect_lanes_ransac(
        frame=frame,
        canny_low=canny_low,
        canny_high=canny_high,
        roi_top_ratio=roi_top_ratio,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        polynomial_degree=polynomial_degree
    )
    
    return overlay, lanes

def detect_lanes_basic(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough: Dict[str, float] = None,
    roi_top_ratio: float = 0.55,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Legacy function for backward compatibility."""
    return detect_lanes_opencv(frame, canny_low, canny_high, hough, roi_top_ratio)

def create_pipeline_from_config(config: Dict[str, Any]) -> LaneDetectionPipeline:
    """Create a pipeline from configuration dictionary."""
    return LaneDetectionPipeline(config)

class LaneDetectionPipeline:
    """Enhanced lane detection pipeline supporting multiple methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = None
        self.preprocessing_method = config.get("preprocessing", {}).get("method", "none")
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the appropriate detector based on configuration."""
        detection_method = self.config.get("detection_method", "opencv")
        
        if detection_method == "opencv":
            self.detector = None  # Use OpenCV directly
        elif detection_method == "ransac":
            self.detector = None  # Use RANSAC directly
        elif detection_method == "yolop":
            model_config = self.config.get("models", {}).get("yolop", {})
            self.detector = YOLOPDetector(
                model_path=model_config.get("model_path"),
                device=model_config.get("device", "cpu"),
                input_size=tuple(model_config.get("input_size", [640, 640])),
                confidence_threshold=model_config.get("confidence_threshold", 0.5),
                nms_threshold=model_config.get("nms_threshold", 0.4)
            )
        elif detection_method == "scnn":
            model_config = self.config.get("models", {}).get("scnn", {})
            self.detector = SCNNDetector(
                model_path=model_config.get("model_path"),
                device=model_config.get("device", "cpu"),
                input_size=tuple(model_config.get("input_size", [512, 256])),
                confidence_threshold=model_config.get("confidence_threshold", 0.5)
            )
        else:
            raise ValueError(f"Unknown detection method: {detection_method}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image preprocessing based on configuration."""
        if self.preprocessing_method == "none":
            return image
        
        preprocessing_config = self.config.get("preprocessing", {})
        
        return preprocess_image(
            image,
            method=self.preprocessing_method,
            gamma=preprocessing_config.get("gamma", 0.8),
            clahe_clip_limit=preprocessing_config.get("clahe_clip_limit", 2.0),
            clahe_tile_size=tuple(preprocessing_config.get("clahe_tile_size", [8, 8]))
        )
    
    def detect_lanes(self, frame: np.ndarray) -> Tuple[np.ndarray, LaneDetectionResult]:
        """
        Detect lanes in the input frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (overlay_image, detection_result)
        """
        start_time = time.time()
        
        # Apply preprocessing
        processed_frame = self.preprocess_image(frame)
        
        # Run detection
        if self.detector is None:
            detection_method = self.config.get("detection_method", "opencv")
            
            if detection_method == "ransac":
                # Use RANSAC method
                ransac_config = self.config.get("ransac", {})
                canny_config = self.config.get("opencv", {}).get("canny", {})
                roi_config = self.config.get("opencv", {}).get("roi", {})
                
                overlay, lanes = detect_lanes_ransac_opencv(
                    processed_frame,
                    canny_low=canny_config.get("low", 50),
                    canny_high=canny_config.get("high", 150),
                    roi_top_ratio=roi_config.get("top_ratio", 0.55),
                    min_samples=ransac_config.get("min_samples", 10),
                    residual_threshold=ransac_config.get("residual_threshold", 2.0),
                    polynomial_degree=ransac_config.get("polynomial_degree", 2)
                )
                
                result = LaneDetectionResult(
                    lanes=lanes,
                    confidence=0.85,  # RANSAC typically more reliable
                    processing_time=time.time() - start_time,
                    model_type=ModelType.OPENCV  # Still using OpenCV for preprocessing
                )
            else:
                # Use traditional OpenCV method
                opencv_config = self.config.get("opencv", {})
                canny_config = opencv_config.get("canny", {})
                hough_config = opencv_config.get("hough", {})
                roi_config = opencv_config.get("roi", {})
                
                overlay, lines = detect_lanes_opencv(
                    processed_frame,
                    canny_low=canny_config.get("low", 50),
                    canny_high=canny_config.get("high", 150),
                    hough=hough_config,
                    roi_top_ratio=roi_config.get("top_ratio", 0.55)
                )
                
                # Convert lines to LaneDetectionResult format
                lanes = []
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        lane_points = np.array([[x1, y1], [x2, y2]], dtype=np.int32)
                        lanes.append(lane_points)
                
                result = LaneDetectionResult(
                    lanes=lanes,
                    confidence=0.8,  # Placeholder confidence for OpenCV
                    processing_time=time.time() - start_time,
                    model_type=ModelType.OPENCV
                )
        else:
            # Use deep learning model
            result = self.detector.predict(processed_frame)
            result.processing_time = time.time() - start_time
            
            # Create overlay
            overlay = frame.copy()
            for lane in result.lanes:
                if len(lane) >= 2:
                    # Draw lane line
                    for i in range(len(lane) - 1):
                        pt1 = tuple(lane[i].astype(int))
                        pt2 = tuple(lane[i + 1].astype(int))
                        cv2.line(overlay, pt1, pt2, (0, 255, 0), 3)
        
        return overlay, result
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the current detector."""
        if self.detector is None:
            return {
                "method": "opencv",
                "preprocessing": self.preprocessing_method
            }
        else:
            info = self.detector.get_model_info()
            info["preprocessing"] = self.preprocessing_method
            return info

def create_pipeline_from_config(config_path: str) -> LaneDetectionPipeline:
    """Create a pipeline from configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return LaneDetectionPipeline(config)
