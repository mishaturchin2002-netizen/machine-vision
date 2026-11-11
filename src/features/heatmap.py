"""Gaze heatmap accumulation, live overlay and PNG export."""

from __future__ import annotations

import time
from pathlib import Path
import cv2
import numpy as np
from utils.constants import *
class GazeHeatmap:
    """Accumulates where the eyes were, blends a live overlay, saves a PNG.

    Accumulation is a 3x3 gaussian stamp per frame (9 adds in a slice) instead of
    a drawn circle - drawing per frame would cost far more, and the blur at the
    end does the smoothing anyway.

    The CSV gets one row per accumulated point: timestamp,x,y,z
      z = metres when a person distance estimate is available,
          otherwise relative MiDaS depth 0..1 (--depth), otherwise empty.
          MiDaS_small returns relative disparity, not metres - it cannot be
          converted to metres without camera intrinsics.
    """

    def __init__(self, width: int, height: int, csv_path: str = HEATMAP_CSV,
                 alpha: float = HEATMAP_ALPHA, viz_every: int = HEATMAP_VIZ_EVERY):
        self.w, self.h = int(width), int(height)
        self.accum = np.zeros((self.h, self.w), np.float32)
        self.count = 0
        self.alpha = alpha
        self.viz_every = max(1, viz_every)
        self.overlay = None

        k = cv2.getGaussianKernel(HEATMAP_SPOT, -1)
        spot = np.outer(k, k).astype(np.float32)
        self.spot = spot / spot.max()

        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv = None
        self._open_csv()

    def _open_csv(self) -> None:
        fresh = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._csv = open(self.csv_path, "a", encoding="utf-8", buffering=1)
        if fresh:
            self._csv.write("timestamp,x,y,z\n")

    def add(self, point, z=None) -> None:
        if point is None:
            return
        x = int(np.clip(round(float(point[0])), 0, self.w - 1))
        y = int(np.clip(round(float(point[1])), 0, self.h - 1))
        r = HEATMAP_SPOT // 2
        x0, x1 = max(0, x - r), min(self.w, x + r + 1)
        y0, y1 = max(0, y - r), min(self.h, y + r + 1)
        ky, kx = r - (y - y0), r - (x - x0)
        self.accum[y0:y1, x0:x1] += self.spot[ky:ky + (y1 - y0), kx:kx + (x1 - x0)]
        self.count += 1
        if self._csv is not None:
            z_str = f"{z:.3f}" if z is not None else ""
            self._csv.write(f"{time.time():.3f},{x},{y},{z_str}\n")

    def clear(self) -> None:
        self.accum[:] = 0.0
        self.count = 0
        self.overlay = None

    def _colorize(self, blur: int):
        if self.count == 0:
            return None
        canvas = cv2.GaussianBlur(self.accum, (blur, blur), 0)
        if float(canvas.max()) <= 0:
            return None
        norm = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def refresh_overlay(self) -> None:
        """Rebuild the coloured map; called every HEATMAP_VIZ_EVERY frames.

        Only the region that actually has heat gets a soft alpha ramp, so the
        video stays untouched everywhere else (a full-frame alpha would tint the
        whole picture blue, because JET maps 0 to dark blue).
        """
        if self.count == 0:
            self.overlay = None
            self._roi = None
            return
        blur = cv2.GaussianBlur(self.accum, (HEATMAP_LIVE_BLUR, HEATMAP_LIVE_BLUR), 0)
        peak = float(blur.max())
        if peak <= 0:
            self.overlay = None
            self._roi = None
            return

        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
        self.overlay = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

        alpha = np.clip(norm / 90.0, 0.0, 1.0)          # soft ramp, not a hard mask
        ys, xs = np.nonzero(alpha > 0.02)
        if len(ys) == 0:
            self._roi = None
            return
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        self._roi = (x0, y0, x1, y1)
        self._alpha_roi = (alpha[y0:y1, x0:x1, None] * self.alpha).astype(np.float32)

    def blend(self, frame):
        if self.overlay is None or getattr(self, "_roi", None) is None:
            return frame
        ov = self.overlay
        if ov.shape[:2] != frame.shape[:2]:
            ov = cv2.resize(ov, (frame.shape[1], frame.shape[0]))
        x0, y0, x1, y1 = self._roi
        x1 = min(x1, frame.shape[1])
        y1 = min(y1, frame.shape[0])
        if x1 <= x0 or y1 <= y0:
            return frame
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        a = self._alpha_roi[: y1 - y0, : x1 - x0]
        frame[y0:y1, x0:x1] = (roi * (1.0 - a) + ov[y0:y1, x0:x1].astype(np.float32) * a).astype(np.uint8)
        return frame

    def save(self, path: str = HEATMAP_OUTPUT):
        colored = self._colorize(HEATMAP_FINAL_BLUR)
        if colored is None:
            return None
        cv2.imwrite(str(path), colored)
        return str(path)

    def close(self) -> None:
        if self._csv is not None:
            try:
                self._csv.flush()
                self._csv.close()
            except Exception:                                 # noqa: BLE001
                pass
            self._csv = None
