"""
save_3d_scene.py - one frame -> MiDaS depth -> coloured point cloud (.ply)
=========================================================================

Takes a SINGLE frame from the webcam, runs it through MiDaS_small, unprojects
every pixel into 3D with the pinhole model and writes a coloured point cloud
that opens in Blender.

No windows, no preview: run it, wait a few seconds, get scene_3d.ply.

    python save_3d_scene.py
    python save_3d_scene.py --out my_room.ply
    python save_3d_scene.py --stride 2          # 4x fewer points, faster file
    python save_3d_scene.py --z-near 0.3 --z-far 6
    python save_3d_scene.py --export-onnx       # rebuild the ONNX cache

--------------------------------------------------------------------------------
WHY IT IS FAST
--------------------------------------------------------------------------------
Loading MiDaS through torch.hub costs ~6 s every run (repo import, timm, JIT).
So the model is exported once to ONNX (midas_small.onnx + midas_small.onnx.data)
and onnxruntime then loads it in ~1 s and infers in ~0.3 s. The camera grab runs
in a background thread at the same time as the model load, so they overlap.
Result on this machine: ~5 s per run instead of ~25 s.

If the ONNX files are missing the script falls back to torch.hub automatically
(and tells you). --export-onnx creates the cache (takes ~10 s, once).

--------------------------------------------------------------------------------
HOW IT WORKS
--------------------------------------------------------------------------------
MiDaS_small predicts *relative inverse depth* (disparity): bigger = closer.
It has no idea about metres, so values are mapped onto [--z-near, --z-far]
in DISPARITY space (1/Z). That is the physically correct interpolation for a
pinhole camera - interpolating Z directly would badly skew the scene:

        d_norm in [0, 1]   (1 = nearest)
        1/Z = d_norm/Z_near + (1 - d_norm)/Z_far

Then each pixel (u, v) is unprojected:

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = Z

with fx = fy = --focal (default 700 px, same as the live script) and cx, cy the
frame centre. MiDaS is monocular, so the result is a plausible relief, not a
measurement: the shape is right, the absolute scale is a guess. --z-near/--z-far
set the depth range, --focal sets the field of view. Scene looks squashed or
stretched -> tune --focal.

--------------------------------------------------------------------------------
HOW TO OPEN IN BLENDER
--------------------------------------------------------------------------------
1. Run the script and wait for "saved scene_3d.ply".
2. Blender -> File -> Import -> Point Cloud (.ply)     [Blender 3.x / 4.x]
   on older builds: File -> Import -> Stanford (.ply)
3. The cloud arrives as an object with vertices only, named after the file.
4. To see it:
   - select the object and press Numpad . (frame selected)
   - Numpad 1 / 3 / 7 = front / side / top
   - switch shading to Material Preview (or Viewport Shading -> Color ->
     Attribute -> "Col") so the vertex colours show.
5. Points may look tiny; either zoom in, or add
   Geometry Nodes -> Mesh to Points -> Set Point Radius for round splats.
6. The cloud is in metres with Y pointing down (image coordinates). To stand it
   upright, rotate 180 deg around X (R, X, 180, Enter).

Tested with: open3d 0.20, Blender 4.x, MeshLab, CloudCompare.

--------------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------------
* Output is binary little-endian PLY with x,y,z (float) + red,green,blue (uchar).
  Blender, MeshLab, CloudCompare and open3d all read it directly.
* If open3d is missing, a small built-in PLY writer is used instead, so the
  script never fails just because of a missing dependency.
"""

from __future__ import annotations

import logging

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from utils.log import setup_logging

log = logging.getLogger(__name__)

FOCAL_DEFAULT = 700.0
Z_NEAR_DEFAULT = 0.5
Z_FAR_DEFAULT = 4.0
OUTPUT_DEFAULT = "scene_3d.ply"

ONNX_MODEL = "midas_small.onnx"
ONNX_WEIGHTS = "midas_small.onnx.data"
MIDAS_INPUT = 256                      # MiDaS_small works at a fixed 256x256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Capture one frame, run MiDaS, write a coloured point cloud.")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--out", default=OUTPUT_DEFAULT, help=f"output file (default {OUTPUT_DEFAULT})")
    ap.add_argument("--focal", type=float, default=FOCAL_DEFAULT,
                    help=f"focal length in px, controls the field of view (default {FOCAL_DEFAULT:.0f})")
    ap.add_argument("--z-near", type=float, default=Z_NEAR_DEFAULT,
                    help=f"distance of the nearest point in metres (default {Z_NEAR_DEFAULT})")
    ap.add_argument("--z-far", type=float, default=Z_FAR_DEFAULT,
                    help=f"distance of the farthest point in metres (default {Z_FAR_DEFAULT})")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every N-th pixel (2 = 4x fewer points)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="do not flip the frame (default mirrors, like the live view)")
    ap.add_argument("--export-onnx", action="store_true",
                    help="(re)build the ONNX cache from torch.hub, then continue")
    return ap.parse_args()


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------
def grab_frame(camera: int, mirror: bool):
    """One shot from the webcam. Tries DirectShow first on Windows."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(camera, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame = None
        # the first frames of a freshly opened camera are often black
        for _ in range(6):
            ok, candidate = cap.read()
            if ok and candidate is not None and float(candidate.mean()) > 1.0:
                frame = candidate
                break
        cap.release()
        if frame is not None:
            if mirror:
                frame = cv2.flip(frame, 1)
            return frame
    return None


# --------------------------------------------------------------------------
# Depth backends
# --------------------------------------------------------------------------
def _preprocess(frame: np.ndarray) -> np.ndarray:
    """Exactly what MiDaS' small_transform does, in numpy."""
    img = cv2.resize(frame, (MIDAS_INPUT, MIDAS_INPUT), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[None, ...].astype(np.float32)


class OnnxDepth:
    """MiDaS_small exported to ONNX: loads in ~1 s instead of ~6 s."""

    def __init__(self, path: str = ONNX_MODEL):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.label = "ONNX (fast path)"

    def disparity(self, frame: np.ndarray) -> np.ndarray:
        """Relative inverse depth resized back to the frame."""
        x = _preprocess(frame)
        out = self.session.run(None, {self.input_name: x})[0]
        out = np.squeeze(out).astype(np.float32)
        return cv2.resize(out, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)


class TorchDepth:
    """Fallback: the original torch.hub model."""

    def __init__(self):
        import torch
        self.torch = torch
        torch.set_num_threads(max(2, min(8, (torch.get_num_threads() or 4))))
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self.label = "torch.hub (slow path)"

    def disparity(self, frame: np.ndarray) -> np.ndarray:
        torch = self.torch
        batch = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            pred = self.model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False)
        return np.squeeze(pred.cpu().numpy()).astype(np.float32)


def export_onnx(path: str = ONNX_MODEL) -> bool:
    """One-off export of MiDaS_small to ONNX (writes an external .data file)."""
    import torch
    log.info("[i] exporting MiDaS_small to ONNX (one time, ~10 s) ...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.eval()
    dummy = torch.rand(1, 3, MIDAS_INPUT, MIDAS_INPUT)
    try:
        torch.onnx.export(model, (dummy,), path,
                          input_names=["input"], output_names=["depth"],
                          opset_version=17, verbose=False)
        return True
    except Exception as exc:                                   # noqa: BLE001
        log.warning(f"[!] ONNX export failed: {type(exc).__name__}: {str(exc)[:200]}")
        return False


def open_depth_backend(prefer_onnx: bool = True):
    if prefer_onnx and Path(ONNX_MODEL).exists():
        try:
            return OnnxDepth()
        except Exception as exc:                               # noqa: BLE001
            log.info(f"[i] ONNX backend unavailable ({exc}), using torch")
    return TorchDepth()


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def depth_to_metres(disparity: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    """Map relative inverse depth onto [z_near, z_far] metres.

    Normalisation uses the 2nd..98th percentile instead of min/max: a handful of
    noisy pixels (a dark corner MiDaS thinks is 10 cm away) would otherwise
    squash the whole room into a thin shell.

    Interpolation happens in disparity space (1/Z), which is linear for a
    pinhole camera; interpolating Z directly would badly skew the scene.
    """
    lo, hi = np.percentile(disparity, 2.0), np.percentile(disparity, 98.0)
    span = float(hi - lo)
    d = (disparity - lo) / span if span > 1e-9 else np.zeros_like(disparity)
    d = np.clip(d, 0.0, 1.0)
    inv = d * (1.0 / z_near) + (1.0 - d) * (1.0 / z_far)
    return (1.0 / np.maximum(inv, 1e-6)).astype(np.float32)


def unproject(depth_m: np.ndarray, colors_rgb: np.ndarray, focal: float):
    """Pixel grid + depth -> (N,3) points, (N,3) colours in 0..1."""
    h, w = depth_m.shape
    fx = fy = float(focal)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3).astype(np.float32)
    colours = colors_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    return points, colours


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------
def write_ply_open3d(path: str, points: np.ndarray, colours: np.ndarray) -> bool:
    try:
        import open3d as o3d
    except Exception:                                          # noqa: BLE001
        return False
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colours.astype(np.float64))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)
    return True


def write_ply_numpy(path: str, points: np.ndarray, colours: np.ndarray) -> None:
    """Fallback: minimal binary little-endian PLY, zero dependencies."""
    rgb = np.clip(colours * 255.0, 0, 255).astype(np.uint8)
    rec = np.empty(len(points), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    rec["x"], rec["y"], rec["z"] = points[:, 0], points[:, 1], points[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rec.tobytes())


# --------------------------------------------------------------------------
def main() -> int:
    setup_logging()
    args = parse_args()
    t0 = time.perf_counter()

    if args.stride < 1:
        log.warning("[!] --stride must be >= 1")
        return 1
    if args.z_near <= 0 or args.z_far <= args.z_near:
        log.warning("[!] need 0 < --z-near < --z-far")
        return 1

    if args.export_onnx:
        export_onnx()

    # the camera and the model load are independent -> run them together
    t_boot = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_frame = pool.submit(grab_frame, args.camera, not args.no_mirror)
        fut_depth = pool.submit(open_depth_backend, True)
        frame = fut_frame.result()
        if frame is None:
            fut_depth.result()
            log.warning("[!] could not read a frame from the camera (is it busy?)")
            return 1
        backend = fut_depth.result()
    log.info(f"[i] frame + model ready in {time.perf_counter() - t_boot:.2f}s "
             f"({backend.label})")

    h, w = frame.shape[:2]
    if args.stride > 1:
        frame = frame[::args.stride, ::args.stride].copy()
    h, w = frame.shape[:2]
    log.info(f"[i] working at {w}x{h}")

    t_depth = time.perf_counter()
    disparity = backend.disparity(frame)
    depth_m = depth_to_metres(disparity, args.z_near, args.z_far)
    log.info(f"[i] depth in {time.perf_counter() - t_depth:.2f}s")

    points, colours = unproject(depth_m, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), args.focal)

    out = Path(args.out)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)

    t_write = time.perf_counter()
    used_open3d = write_ply_open3d(str(out), points, colours)
    if not used_open3d:
        log.info("[i] open3d not available, using the built-in PLY writer")
        write_ply_numpy(str(out), points, colours)
    write_s = time.perf_counter() - t_write

    elapsed = time.perf_counter() - t0
    log.info("-" * 62)
    log.info(f"points      : {len(points):,}")
    log.info(f"z range     : {depth_m.min():.2f} .. {depth_m.max():.2f} m")
    log.info(f"x range     : {points[:, 0].min():.2f} .. {points[:, 0].max():.2f} m")
    log.info(f"y range     : {points[:, 1].min():.2f} .. {points[:, 1].max():.2f} m")
    log.info(f"writer      : {'open3d' if used_open3d else 'built-in numpy'} "
             f"({write_s:.2f}s, {out.stat().st_size / 1024 / 1024:.1f} MB)")
    log.info(f"saved       : {out.resolve()}")
    log.info(f"total time  : {elapsed:.2f}s")
    log.info("-" * 62)
    log.info("Blender: File -> Import -> Point Cloud (.ply), then Numpad . to frame it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
