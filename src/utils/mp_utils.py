"""MediaPipe Tasks import shim shared by the face and hand trackers."""

from __future__ import annotations

from utils.constants import *
def mp_modules():
    """Import mediapipe Tasks API pieces, tolerating the old import layouts."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    try:
        from mediapipe.tasks.python import BaseOptions
    except ImportError:                                       # pragma: no cover
        from mediapipe.tasks.python.core.base_options import BaseOptions
    return mp, vision, BaseOptions
