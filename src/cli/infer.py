from __future__ import annotations
import argparse, yaml, time, os
import cv2
import numpy as np
from pathlib import Path
from ..runtime.pipeline import LaneDetectionPipeline, create_pipeline_from_config
from ..utils.viz import FPSTracker, draw_fps
from ..utils.metrics import LaneDetectionMetrics

def load_cfg(path: str) -> dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Enhanced Lane Detection System")
    p.add_argument("--config", default="configs/default.yaml", help="Configuration file path")
    p.add_argument("input", nargs="?", default=None, help="Video path or 'webcam'")
    p.add_argument("--save_video", type=str, default=None, help="Override save (true/false)")
    p.add_argument("--show", type=str, default=None, help="Override show (true/false)")
    p.add_argument("--method", type=str, default=None, help="Detection method (opencv/ransac/yolop/scnn)")
    p.add_argument("--preprocessing", type=str, default=None, help="Preprocessing method")
    p.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    p.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    p.add_argument("--ground_truth", type=str, help="Ground truth path for evaluation")
    return p.parse_args()

def str2bool(s):
    """Convert string to boolean."""
    return str(s).lower() in {"1","true","yes","y","on"}

def run_inference(pipeline: LaneDetectionPipeline, cap, writer, show: bool, benchmark: bool = False):
    """Run lane detection inference loop."""
    fps_t = FPSTracker()
    metrics = LaneDetectionMetrics() if benchmark else None
    
    frame_count = 0
    total_frames = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # Run detection
        overlay, result = pipeline.detect_lanes(frame)
        
        # Update FPS
        fps = fps_t.tick()
        draw_fps(overlay, fps)
        
        # Add detection info to overlay
        info = pipeline.get_detector_info()
        cv2.putText(overlay, f"Method: {info.get('method', 'unknown')}", 
                   (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Lanes: {len(result.lanes)}", 
                   (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if result.processing_time:
            cv2.putText(overlay, f"Time: {result.processing_time*1000:.1f}ms", 
                       (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Update metrics
        if metrics is not None:
            # For benchmark mode, we don't have ground truth, so just track processing time
            metrics.processing_times.append(result.processing_time or 0)
        
        # Save video
        if writer is not None:
            writer.write(overlay)
        
        # Show video
        if show:
            cv2.imshow("Lane Detection", overlay)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        frame_count += 1
        total_frames += 1
    
    return metrics, total_frames

def run_benchmark(pipeline: LaneDetectionPipeline, num_frames: int = 200):
    """Run benchmark on synthetic data."""
    print(f"Running benchmark with {num_frames} synthetic frames...")
    
    # Create synthetic frame
    h, w = 720, 1280
    synthetic_frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    
    # Add some synthetic lane-like features
    cv2.line(synthetic_frame, (w//4, h), (w//3, h//2), (255, 255, 255), 5)
    cv2.line(synthetic_frame, (3*w//4, h), (2*w//3, h//2), (255, 255, 255), 5)
    
    metrics = LaneDetectionMetrics()
    
    # Warmup
    for _ in range(30):
        _, _ = pipeline.detect_lanes(synthetic_frame)
    
    # Benchmark
    start_time = time.time()
    for i in range(num_frames):
        _, result = pipeline.detect_lanes(synthetic_frame)
        if result.processing_time:
            metrics.processing_times.append(result.processing_time)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{num_frames} frames...")
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\nBenchmark Results:")
    print(f"Total frames: {num_frames}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {num_frames/total_time:.1f}")
    print(f"Average processing time: {np.mean(metrics.processing_times)*1000:.1f}ms")
    print(f"Min processing time: {np.min(metrics.processing_times)*1000:.1f}ms")
    print(f"Max processing time: {np.max(metrics.processing_times)*1000:.1f}ms")
    
    return metrics

def main():
    """Main function."""
    args = parse_args()
    cfg = load_cfg(args.config)
    
    # Override config with command line arguments
    if args.method:
        cfg["detection_method"] = args.method
    if args.preprocessing:
        cfg["preprocessing"]["method"] = args.preprocessing
    
    # Create pipeline
    pipeline = LaneDetectionPipeline(cfg)
    
    # Print pipeline info
    info = pipeline.get_detector_info()
    print(f"Initialized pipeline:")
    print(f"  Method: {info.get('method', 'unknown')}")
    print(f"  Preprocessing: {info.get('preprocessing', 'none')}")
    if 'model_path' in info:
        print(f"  Model: {info['model_path']}")
    if 'device' in info:
        print(f"  Device: {info['device']}")
    
    # Handle benchmark mode
    if args.benchmark:
        num_frames = cfg.get("performance", {}).get("test_frames", 200)
        run_benchmark(pipeline, num_frames)
        return
    
    # Handle evaluation mode
    if args.evaluate:
        if not args.ground_truth:
            print("Error: --ground_truth required for evaluation mode")
            return
        
        from ..utils.metrics import evaluate_model_on_dataset
        metrics = evaluate_model_on_dataset(
            pipeline, 
            args.input or cfg.get("input", "data/test_images"),
            args.ground_truth,
            cfg.get("evaluation", {}).get("output_dir", "evaluation_results")
        )
        return
    
    # Regular inference mode
    inp = args.input or cfg.get("input", "webcam")
    save_video = str2bool(args.save_video) if args.save_video is not None else bool(cfg.get("save_video", True))
    show = str2bool(args.show) if args.show is not None else bool(cfg.get("show", False))
    
    # Open video source
    cap = cv2.VideoCapture(0) if inp == "webcam" else cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise SystemExit(f"Could not open input: {inp}")
    
    # Setup output
    os.makedirs("runs", exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    
    out_path = Path("runs") / f"lane_demo_{int(time.time())}.mp4"
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (w, h))
    
    try:
        # Run inference
        metrics, total_frames = run_inference(pipeline, cap, writer, show, 
                                            cfg.get("performance", {}).get("benchmark", False))
        
        # Print final statistics
        if metrics and metrics.processing_times:
            print(f"\nFinal Statistics:")
            print(f"Total frames processed: {total_frames}")
            print(f"Average FPS: {1.0/np.mean(metrics.processing_times):.1f}")
            print(f"Average processing time: {np.mean(metrics.processing_times)*1000:.1f}ms")
    
    finally:
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        if save_video:
            print(f"Video saved: {out_path}")

if __name__ == "__main__":
    main()
