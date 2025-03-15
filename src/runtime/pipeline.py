from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional, Dict

def region_of_interest(img: np.ndarray, top_ratio: float = 0.55) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros_like(img[:, :, 0])
    polygon = np.array([[(0, h), (0, int(h*top_ratio)), (w, int(h*top_ratio)), (w, h)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, img, mask=mask)

def detect_lanes_basic(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough: Dict[str, float] = None,
    roi_top_ratio: float = 0.55,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
