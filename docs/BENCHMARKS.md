# Benchmarks

All numbers were measured on the same machine, same webcam and the same code
path that ships in this repository. They are **indicative**, not guarantees:
a busy machine (browser playing video, IDE indexing) swings the numbers by
±20 %, so treat differences under ~15 % as noise.

## Test bench

| | |
|---|---|
| CPU | Intel Core Ultra 7 155H (6P + 8E + 2LP, 22 threads) |
| GPU | Intel Arc integrated (OpenVINO `GPU.0`) |
| RAM | 32 GB |
| Camera | generic UVC webcam, 640×480 @ 30 fps |
| OS | Windows 11 |
| Python | 3.13, venv |
| Models | YOLOv8m (det) + YOLOv8n-pose, both OpenVINO **640**, MediaPipe, MiDaS_small |

## Methodology

* **Frame time**: median over 40–80 frames (`--headless N` prints it), not the
  mean — a single OpenVINO kernel-compile frame would otherwise poison the
  average.
* **FPS** = `1000 / median_frame_ms`.
* Detection and pose run **in parallel** in two threads, so the pipeline cost is
  closer to `max(det, pose)` than `det + pose`.
* The loop is camera-bound at 30 fps, so sub-33 ms frame times are not
  observable in a live window; the `--benchmark` numbers measure pure inference.
* Every configuration was run at least twice; the table reports a typical run.

## 1. Backend comparison (`--benchmark`, det+pose, 640)

| Backend | Frame time | FPS | Note |
|---|---|---|---|
| torch CPU | 178 ms | 5.6 | 6 threads per model |
| OpenVINO CPU | 188 ms | 5.3 | reuses the exported model |
| **OpenVINO intel:gpu** | **24 ms** | **40.9** | 7.4× faster than torch |

The single biggest win of the project. Two findings worth knowing:

* `intel:auto` **resolves to the CPU plugin** here (3.1 FPS at 640). The GPU has
  to be requested explicitly.
* Export resolution is baked in: an OpenVINO model exported at 384 runs at
  83.5 FPS but sees smaller objects worse. 640 is the balance used above.

## 2. Full pipeline

| Profile | Median frame | FPS |
|---|---|---|
| base (detection + pose + actions + distance) | 56 ms | 18.0 |
| `--track --speed --trails --async-grab --export-csv` | 66–79 ms | 12.6–15.1 |
| `--all` (eyes + gestures + drawing + distance) | 66–71 ms | 14–15 |

### Capture: sync vs async

Same flags, 50 frames, two runs each:

| Mode | Run 1 | Run 2 | Median |
|---|---|---|---|
| sync `cap.read()` | 73 ms / 13.6 FPS | 75 ms / 13.2 FPS | 74 ms |
| `--async-grab` | 65 ms / 15.1 FPS | 66 ms / 15.5 FPS | 65.5 ms |

Gain ≈ **+12–15 %**, plus the jitter disappears: the loop never waits for the
sensor, it takes whatever frame is in the one-slot queue (dropped frames are
counted in the HUD as `drop N`).

> Why not more? The pipeline was already inference-bound at ~65 ms while the
> camera delivers a frame every 33 ms. Async capture removes the *waiting*, not
> the inference.

## 3. Individual features

| Feature | Cost | Comment |
|---|---|---|
| MediaPipe face mesh (`--eyes`, `--emotions`) | ~15–20 ms / pass, every 2nd frame | one pass feeds gaze **and** blendshapes |
| MediaPipe hand mesh (`--gestures`, `--draw`) | ~15 ms / pass, every 2nd frame | |
| OpenCV shapes (`--shapes`) | ~20 ms full-res, every 2nd frame | cached between refreshes |
| Segmentation (`--bg-blur`) | **48 ms** warm / **6 500 ms** first | every 3rd frame, worker thread |
| MiDaS depth (`--depth`, `--depth-map`) | 200–300 ms / frame | every 3rd frame |
| EasyOCR (`--ocr`) | **888 ms** / pass | every 20th frame, worker thread |
| Gaze heatmap (`--heatmap`) | < 1 ms | 3×3 gaussian stamp + ROI blend |

Heatmap cost was verified directly (`--eyes` vs `--eyes --heatmap`):
81/66 ms vs 86/88 ms — inside the noise floor.

## 4. MiDaS startup: torch.hub vs ONNX

| Path | Load | Inference | Total pipeline |
|---|---|---|---|
| `torch.hub` (cold process) | 5.4 s + 0.8 s transforms | 0.21 s | ~25 s end-to-end |
| **ONNX (cached)** | **1.0 s** | **0.31 s** | **~2 s** end-to-end |

Used by `save_3d_scene.py`. `torch.jit.trace` of MiDaS fails
(`Graphs differed across invocations` — dynamic ops in geffnet), so the ONNX
dynamo exporter is the workable path; it writes `midas_small.onnx` plus a 63 MB
external weights file.

## 5. What did *not* help

* `torch.set_num_threads(4/6/8)` — within noise (±1 FPS); 6 is the default.
* CPU affinity pinning to P-cores — **slower** (135 ms vs 71 ms), because the
  hybrid scheduler handles the E-cores better than a hard mask.
* Half precision on CPU — no gain in PyTorch CPU path.

## Reproducing

```bash
python src/realtime_vision.py --benchmark                 # backend table
python src/realtime_vision.py --headless 60 --track --speed
python src/realtime_vision.py --headless 60 --track --speed --async-grab
```
