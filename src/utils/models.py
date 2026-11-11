"""Model discovery/loading and video-source opening.

Model paths are resolved against the repository's models/ directory first,
then the current directory, so the project works both after
download_models.py and with weights dropped next to the scripts."""

from __future__ import annotations

import logging

import sys
from pathlib import Path
import cv2
from utils.constants import *

log = logging.getLogger(__name__)
def ov_dir_for(pt_path: str) -> Path:
    p = Path(pt_path)
    return p.with_name(f"{p.stem}_openvino_model")


def load_model(pt_path: str, backend: str):
    """Returns (model, label, is_openvino)."""
    from ultralytics import YOLO

    ov_dir = ov_dir_for(pt_path)
    if backend in ("auto", "openvino") and ov_dir.exists():
        return YOLO(str(ov_dir)), f"OpenVINO({ov_dir.name})", True
    if backend == "openvino":
        raise SystemExit(
            f"[!] OpenVINO model folder not found: {ov_dir}\n"
            f"    create it with:  python -c \"from ultralytics import YOLO; "
            f"YOLO('{pt_path}').export(format='openvino', imgsz=640)\"")
    return YOLO(pt_path), f"torch({Path(pt_path).name})", False


def open_source(src, mirror: bool):
    try:
        src = int(src)
    except (TypeError, ValueError):
        pass
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(src, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        if ok:
            log.info(f"[i] source opened (backend={backend}, source={src!r}, "
                  f"mirror={mirror})")
            return cap
        cap.release()
    log.warning(f"[!] cannot open source {src!r}")
    return None
