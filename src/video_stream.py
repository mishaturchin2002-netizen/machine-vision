"""Background-thread frame grabber.

cv2.VideoCapture.read() blocks for a whole frame period, which caps the
pipeline at the camera rate. VideoStream decouples capture from
processing with a maxsize=1 queue and hands out the freshest frame."""

from __future__ import annotations

import queue
import threading
class VideoStream:
    """Grabs frames in a background thread; read() just takes the newest one.

    `cv2.VideoCapture.read()` blocks until the sensor delivers the next frame
    (~33 ms at 30 fps), so the main loop spends a third of its time waiting and
    the pipeline can never go faster than the camera. Here a daemon thread does
    the waiting and the loop pulls the latest frame from a maxsize=1 queue: if
    inference is slower than the camera the old frame is dropped, if it is
    faster the loop simply runs on the frame that is already there.

    Wraps an already-opened cv2.VideoCapture so `open_source()` keeps working.
    Deliberately does NOT mirror: the single flip point stays in the main loop.
    """

    def __init__(self, cap, name: str = "stream"):
        self.cap = cap
        self.name = name
        self.frames = 0                      # frames grabbed by the thread
        self.dropped = 0                     # frames overwritten before being read
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._failed = False
        self._thread = threading.Thread(target=self._run, name=f"grab-{name}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._failed = True
                break
            self.frames += 1
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                # keep only the freshest frame; the consumer is behind
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    self.dropped += 1

    def read(self, timeout: float = 1.0):
        try:
            return True, self._queue.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def get(self, prop: int):
        return self.cap.get(prop)

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()
