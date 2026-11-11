"""Person segmentation for --bg-blur / --bg-remove.

yolov8n-seg runs in a worker thread every N frames; the cached mask is
composited under the live frame."""

from __future__ import annotations

import logging

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from utils.constants import *
from utils.models import ov_dir_for

log = logging.getLogger(__name__)
class Segmenter:
    """Person mask from yolov8n-seg, refreshed in a background thread.

    Segmentation is noticeably heavier than detection, so it runs every
    --seg-every frames (default 3) off the main thread and the mask is cached.
    A mask that is one or two frames old is invisible to the eye and costs the
    frame rate nothing.
    """

    def __init__(self, pt_path: str, backend: str, ov_device: str,
                 every: int = 3, conf: float = 0.40):
        from ultralytics import YOLO

        ov_dir = ov_dir_for(pt_path)
        if backend in ("auto", "openvino") and ov_dir.exists():
            self.model = YOLO(str(ov_dir))
            self.kwargs = dict(device=ov_device, imgsz=640)
            self.label = f"OpenVINO({ov_dir.name})"
        else:
            if backend == "openvino":
                raise FileNotFoundError(f"{ov_dir} missing - export it first")
            self.model = YOLO(pt_path)
            self.kwargs = dict(imgsz=640)
            self.label = f"torch({Path(pt_path).name})"
        self.kwargs.update(conf=conf, verbose=False)

        self.every = max(1, int(every))
        self.mask = None            # uint8 0/255, frame size
        self.ms = 0.0
        self.runs = 0
        self._busy = False
        self._pool = ThreadPoolExecutor(max_workers=1)

    def maybe_submit(self, frame, frame_idx: int) -> None:
        if self._busy or frame_idx % self.every != 0:
            return
        self._busy = True
        self._pool.submit(self._run, frame.copy())

    def _run(self, frame) -> None:
        t0 = time.perf_counter()
        try:
            result = self.model.predict(frame, **self.kwargs)[0]
            mask = np.zeros(frame.shape[:2], np.uint8)
            if result.masks is not None and result.boxes is not None and len(result.boxes):
                data = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for i, cls_id in enumerate(classes):
                    if cls_id != 0:                     # 0 = person only
                        continue
                    person = data[i]
                    # masks come back at the model input size (640x640), not at
                    # the frame size - resize before combining
                    if person.shape[:2] != mask.shape[:2]:
                        person = cv2.resize(person, (mask.shape[1], mask.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                    mask = np.maximum(mask, (person > 0.5).astype(np.uint8))
            self.mask = mask * 255
            self.runs += 1
        except Exception as exc:                              # noqa: BLE001
            log.warning(f"[!] segmentation failed: {type(exc).__name__}: {exc}")
        finally:
            self.ms = (time.perf_counter() - t0) * 1000.0
            self._busy = False

    def apply(self, frame, mode: str):
        """mode = 'blur' or 'remove'. Returns the composited frame."""
        if self.mask is None:
            return frame
        soft = cv2.GaussianBlur(self.mask, (9, 9), 0).astype(np.float32) * (1.0 / 255.0)
        soft = soft[..., None]
        if mode == "remove":
            background = np.empty_like(frame)
            background[:] = (0, 255, 0)                     # chroma green
        else:
            background = cv2.GaussianBlur(frame, (0, 0), 14)
        out = frame.astype(np.float32) * soft + background.astype(np.float32) * (1.0 - soft)
        return out.astype(np.uint8)

    def close(self) -> None:
        self._pool.shutdown(wait=False)
