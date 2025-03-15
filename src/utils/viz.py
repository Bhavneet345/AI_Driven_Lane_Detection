from __future__ import annotations
import cv2
import time

class FPSTracker:
    def __init__(self, ema: float = 0.9):
        self.ema = ema
        self.prev = time.time()
        self.fps = 0.0

    def tick(self) -> float:
        now = time.time()
        dt = max(1e-6, now - self.prev)
        inst = 1.0 / dt
        self.fps = self.ema * self.fps + (1 - self.ema) * inst if self.fps > 0 else inst
        self.prev = now
        return self.fps

def draw_fps(frame, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
