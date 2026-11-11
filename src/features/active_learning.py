"""Active learning: saves frames the detector was unsure about."""

from __future__ import annotations

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
from utils.constants import *
class LowConfidenceSaver:
    def __init__(self, folder: str, threshold: float, cooldown: float):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.cooldown = cooldown
        self.last = 0.0
        self.saved = 0
        self._pool = ThreadPoolExecutor(max_workers=1)

    def maybe_save(self, frame, objects) -> bool:
        weak = [(n, c) for n, c, *_ in objects if c < self.threshold]
        if not weak:
            return False
        now = time.time()
        if now - self.last < self.cooldown:
            return False
        self.last = now
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tags = "_".join(f"{n}{c:.2f}".replace(".", "") for n, c in weak[:3])
        self._pool.submit(cv2.imwrite, str(self.folder / f"{stamp}_{self.saved:05d}_{tags}.jpg"),
                          frame.copy())
        self.saved += 1
        return True

    def close(self):
        self._pool.shutdown(wait=False)
