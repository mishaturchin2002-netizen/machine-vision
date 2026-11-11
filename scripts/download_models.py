"""
download_models.py - fetch the model weights this project needs.

Downloads (into ``models/``):

    yolov8m.pt          COCO detector            ~50 MB
    yolov8n-pose.pt     human pose (17 kpts)      ~7 MB
    yolov8n-seg.pt      person segmentation       ~7 MB
    face_landmarker.task  MediaPipe face mesh    ~3.6 MB
    hand_landmarker.task  MediaPipe hand mesh    ~7.5 MB
    midas_small.onnx      MiDaS depth (ONNX)     ~64 MB (+ external .data)

and exports the three YOLO models to OpenVINO (``*_openvino_model/``), which is
what makes the app fast on Intel Arc / CPU.

Usage
-----
    python scripts/download_models.py                 # everything
    python scripts/download_models.py --skip-openvino # only the .pt weights
    python scripts/download_models.py --force         # re-download / re-export
    python scripts/download_models.py --models-dir ./models

Exit code is 0 only when every requested artefact is present and readable.
"""

from __future__ import annotations

import argparse
import logging
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("download_models")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"

YOLO_MODELS = {
    "yolov8m.pt": ("detector (COCO, 80 classes)", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt"),
    "yolov8n-pose.pt": ("pose, 17 keypoints", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt"),
    "yolov8n-seg.pt": ("instance segmentation", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-seg.pt"),
}

MEDIAPIPE_MODELS = {
    "face_landmarker.task": (
        "face mesh + blendshapes (gaze, emotions)",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
    ),
    "hand_landmarker.task": (
        "hand mesh (gestures, air drawing)",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
    ),
}

ONNX_MODEL = "midas_small.onnx"
ONNX_WEIGHTS = "midas_small.onnx.data"
MIDAS_INPUT = 256


def human(n: float) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def download(url: str, target: Path, force: bool = False) -> bool:
    """Download with progress and a size sanity check."""
    if target.exists() and target.stat().st_size > 0 and not force:
        log.info("  [skip] %s already present (%s)", target.name, human(target.stat().st_size))
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(blocks * block_size, total)
        pct = done * 100 // total
        if blocks % 200 == 0 or done >= total:
            print(f"\r    {target.name}: {pct:3d}%  {human(done)} / {human(total)}", end="", flush=True)

    try:
        log.info("  downloading %s", target.name)
        urllib.request.urlretrieve(url, tmp, reporthook=hook)
        print()
        if tmp.stat().st_size < 1024:
            raise OSError("downloaded file is suspiciously small")
        tmp.replace(target)
        log.info("  [ok]   %s (%s)", target.name, human(target.stat().st_size))
        return True
    except Exception as exc:                                  # noqa: BLE001
        print()
        log.error("  [fail] %s: %s", target.name, exc)
        tmp.unlink(missing_ok=True)
        return False


def export_openvino(pt_path: Path) -> bool:
    """Export a YOLO .pt to OpenVINO next to it."""
    from ultralytics import YOLO

    out_dir = pt_path.with_name(f"{pt_path.stem}_openvino_model")
    if out_dir.exists() and any(out_dir.iterdir()):
        log.info("  [skip] %s already exported", out_dir.name)
        return True
    try:
        log.info("  exporting %s -> OpenVINO (CPU, ~10 s) ...", pt_path.name)
        YOLO(str(pt_path)).export(format="openvino", imgsz=640, verbose=False)
        log.info("  [ok]   %s", out_dir.name)
        return True
    except Exception as exc:                                  # noqa: BLE001
        log.error("  [fail] OpenVINO export of %s: %s", pt_path.name, exc)
        return False


def export_midas(models_dir: Path, force: bool = False) -> bool:
    """Export MiDaS_small to ONNX - 5x faster startup than torch.hub."""
    onnx_path = models_dir / ONNX_MODEL
    data_path = models_dir / ONNX_WEIGHTS
    if onnx_path.exists() and data_path.exists() and not force:
        log.info("  [skip] %s already present", ONNX_MODEL)
        return True
    try:
        import torch
    except ImportError:
        log.warning("  [skip] torch is not installed, cannot export MiDaS")
        return False
    try:
        log.info("  downloading MiDaS_small via torch.hub (~80 MB, once) ...")
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        model.eval()
        dummy = torch.rand(1, 3, MIDAS_INPUT, MIDAS_INPUT)
        log.info("  exporting to ONNX ...")
        torch.onnx.export(model, (dummy,), str(onnx_path),
                          input_names=["input"], output_names=["depth"],
                          opset_version=17, verbose=False)
        log.info("  [ok]   %s (+ external weights)", ONNX_MODEL)
        return True
    except Exception as exc:                                  # noqa: BLE001
        log.error("  [fail] MiDaS export: %s", exc)
        log.error("         the depth features will fall back to torch.hub (slower)")
        return False


def verify(models_dir: Path) -> list[str]:
    """Return a list of missing/broken artefacts."""
    problems: list[str] = []
    for name in list(YOLO_MODELS) + list(MEDIAPIPE_MODELS):
        path = models_dir / name
        if not path.exists() or path.stat().st_size < 1024:
            problems.append(name)
    for stem in ("yolov8m", "yolov8n-pose", "yolov8n-seg"):
        ov = models_dir / f"{stem}_openvino_model"
        if not ov.exists() or not any(ov.iterdir()):
            problems.append(f"{stem}_openvino_model/")
    if not (models_dir / ONNX_MODEL).exists():
        problems.append(ONNX_MODEL)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Download and export all model weights.")
    ap.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    ap.add_argument("--force", action="store_true", help="re-download/re-export everything")
    ap.add_argument("--skip-openvino", action="store_true", help="keep .pt only")
    ap.add_argument("--skip-midas", action="store_true", help="skip the depth model")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    models_dir = Path(args.models_dir).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)
    log.info("models directory: %s", models_dir)

    t0 = time.perf_counter()
    ok = True

    log.info("[1/3] YOLO weights")
    for name, (_desc, url) in YOLO_MODELS.items():
        ok &= download(url, models_dir / name, force=args.force)

    log.info("[2/3] MediaPipe task files")
    for name, (_desc, url) in MEDIAPIPE_MODELS.items():
        ok &= download(url, models_dir / name, force=args.force)

    if args.skip_openvino:
        log.info("[3/3] OpenVINO export skipped (--skip-openvino)")
    else:
        log.info("[3/3] OpenVINO export")
        for name in YOLO_MODELS:
            pt = models_dir / name
            if pt.exists():
                ok &= export_openvino(pt)

    if not args.skip_midas:
        log.info("[+]  MiDaS depth -> ONNX")
        export_midas(models_dir, force=args.force)

    log.info("-" * 60)
    if args.skip_openvino:
        log.info("done in %.1fs (OpenVINO export skipped)", time.perf_counter() - t0)
        return 0 if ok else 1

    problems = verify(models_dir)
    if problems:
        log.error("missing/incomplete artefacts:")
        for p in problems:
            log.error("  - %s", p)
        return 1

    size = sum(f.stat().st_size for f in models_dir.rglob("*") if f.is_file())
    log.info("done in %.1fs - all artefacts present (%s)", time.perf_counter() - t0, human(size))
    log.info("run the app with:  python src/realtime_vision.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
