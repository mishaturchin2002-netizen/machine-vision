"""EasyOCR text reading in a background thread."""

from __future__ import annotations

import logging

import time
from concurrent.futures import ThreadPoolExecutor
import cv2
from utils.constants import *

log = logging.getLogger(__name__)
class OcrEngine:
    """Reads text from the centre of the frame, ~once every `every` frames.

    EasyOCR on CPU takes hundreds of milliseconds: running it inline would drop
    the whole pipeline to 1-3 FPS. So detection is fire-and-forget - the last
    recognised string is simply drawn until a new one replaces it.
    """

    def __init__(self, langs=("en", "ru"), every: int = 20, margin: int = 100):
        import easyocr
        log.info(f"[i] ocr: initialising EasyOCR {list(langs)} (first run downloads models) ...")
        self.reader = easyocr.Reader(list(langs), gpu=False, verbose=False)
        self.every = max(1, every)
        self.margin = margin
        self.text = ""
        self.ms = 0.0
        self.runs = 0
        self._busy = False
        self._pool = ThreadPoolExecutor(max_workers=1)

    def maybe_submit(self, frame, frame_idx: int) -> None:
        if self._busy or frame_idx % self.every != 0:
            return
        h, w = frame.shape[:2]
        m = self.margin
        if h <= 2 * m + 8 or w <= 2 * m + 8:      # tiny frame: use all of it
            crop = frame
        else:
            crop = frame[m:h - m, m:w - m]
        self._busy = True
        self._pool.submit(self._run, crop.copy())

    def _run(self, crop) -> None:
        t0 = time.perf_counter()
        try:
            parts = self.reader.readtext(crop, detail=0, paragraph=True)
            self.text = " ".join(p.strip() for p in parts if p.strip())[:30]
        except Exception as exc:                              # noqa: BLE001
            log.warning(f"[!] ocr failed: {exc}")
            self.text = ""
        finally:
            self.ms = (time.perf_counter() - t0) * 1000.0
            self.runs += 1
            self._busy = False

    def draw(self, frame, top_offset: int = 0) -> None:
        if not self.text:
            return
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        (tw, th), base = cv2.getTextSize(self.text, font, scale, thick)
        x = max(frame.shape[1] - tw - 14, 4)
        y = th + 14 + int(top_offset)     # pushed below the status panel
        cv2.rectangle(frame, (x - 6, y - th - 8), (x + tw + 6, y + base + 4), (40, 40, 40), -1)
        cv2.putText(frame, self.text, (x, y), font, scale, (120, 230, 255), thick, cv2.LINE_AA)

    def close(self):
        self._pool.shutdown(wait=False)
