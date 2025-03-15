from __future__ import annotations
import argparse, yaml, time, os
import cv2
from pathlib import Path
from ..runtime.pipeline import detect_lanes_basic
from ..utils.viz import FPSTracker, draw_fps

def load_cfg(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    p = argparse.ArgumentParser(description="Lane Detection Baseline (OpenCV)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("input", nargs="?", default=None, help="video path or 'webcam'")
    p.add_argument("--save_video", type=str, default=None, help="override save (true/false)")
    p.add_argument("--show", type=str, default=None, help="override show (true/false)")
    return p.parse_args()

def str2bool(s):
    return str(s).lower() in {"1","true","yes","y","on"}

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    inp = args.input or cfg.get("input", "webcam")
    save_video = str2bool(args.save_video) if args.save_video is not None else bool(cfg.get("save_video", True))
    show = str2bool(args.show) if args.show is not None else bool(cfg.get("show", False))

    cap = cv2.VideoCapture(0) if inp == "webcam" else cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise SystemExit(f"Could not open input: {inp}")

    os.makedirs("runs", exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    out_path = Path("runs") / f"lane_demo_{int(time.time())}.mp4"
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (w,h))

    fps_t = FPSTracker()
    canny = cfg.get("canny", {})
    hough = cfg.get("hough", {})
    roi_top = cfg.get("roi", {}).get("top_ratio", 0.55)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        overlay, _ = detect_lanes_basic(
            frame,
            canny_low=int(canny.get("low",50)),
            canny_high=int(canny.get("high",150)),
            hough=hough,
            roi_top_ratio=float(roi_top),
        )
        fps = fps_t.tick()
        draw_fps(overlay, fps)

        if writer is not None:
            writer.write(overlay)
        if show:
            cv2.imshow("lanes", overlay)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    if save_video:
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
