# AI‑Driven Lane Detection in Tunnels (Real‑Time Baseline)

> **Status:** Public baseline scaffold. The classic deep models (YOLOP / SCNN) are *plug‑in ready*, and this repo ships a minimal OpenCV baseline so you can run a demo instantly and extend to DL later.

## Why this exists
I built a fast, reproducible baseline to support the project mentioned on my resume (Mar 2025). This repo:
- Runs a **real‑time demo** on video/webcam using OpenCV (Canny + Hough) to draw lane lines
- Has a clean **project structure** ready for **PyTorch** models (YOLOP, SCNN)
- Provides **configs**, **CLI**, **scripts**, and a spot for **weights**

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run inference on a video
python -m src.cli.infer input=examples/sample_video.mp4 save_video=true show=false

# Benchmark FPS (synthetic frames if no video provided)
python scripts/benchmark_fps.py
```

Outputs are written to `runs/` with timestamps.

## Project Structure
```
lane-tunnel-rt/
  configs/
    default.yaml
  src/
    cli/infer.py
    runtime/pipeline.py
    utils/viz.py
  scripts/
    benchmark_fps.py
  runs/                 # outputs
  examples/             # put sample videos here
  requirements.txt
  README.md
```

## Extend to Deep Learning
This baseline is intentionally simple. To upgrade:
- Implement model wrappers in `src/models/` (YOLOP / SCNN)
- Replace `runtime/pipeline.py` processing with model inference
- Add training/eval CLIs (Hydra or Argparse)

## Notes
- This is an **initial public baseline**; code is productionizable and well‑documented.
- Demo uses OpenCV only (no GPU dependency) so anyone can run quickly.
- Deep model weights are not included here; integrate as needed.

---

**Author:** Bhavneet Singh  
**License:** MIT
