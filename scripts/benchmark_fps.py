import time, numpy as np, cv2
from src.runtime.pipeline import detect_lanes_basic
from src.utils.viz import FPSTracker

def main():
    h, w = 720, 1280
    frame = (np.random.rand(h, w, 3) * 255).astype("uint8")
    fps = FPSTracker()
    # warmup
    for _ in range(30):
        _ = detect_lanes_basic(frame)
    # measure
    N = 200
    start = time.time()
    for _ in range(N):
        _ = detect_lanes_basic(frame)
        fps.tick()
    dur = time.time() - start
    print(f"Frames: {N}, total: {dur:.2f}s, avgFPS: {N/dur:.1f}")

if __name__ == "__main__":
    main()
