"""Monocular depth with MiDaS_small (torch.hub, cached locally)."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from utils.constants import *

log = logging.getLogger(__name__)
class DepthEstimator:
    def __init__(self):
        import torch
        self.torch = torch
        log.info("[i] depth: fetching MiDaS_small via torch.hub (first run downloads ~80 MB) ...")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self.last_norm = None

    def infer(self, frame) -> np.ndarray:
        """Raw depth as a normalised uint8 map at frame size (0..255)."""
        torch = self.torch
        batch = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            pred = self.model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False)
        depth = pred.squeeze().cpu().numpy()
        norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.last_norm = norm          # sampled for the heatmap CSV (relative depth)
        return norm

    @staticmethod
    def colorize(norm: np.ndarray, colormap: int) -> np.ndarray:
        return cv2.applyColorMap(norm, colormap)

    def blended(self, frame, colormap: int = cv2.COLORMAP_INFERNO, alpha: float = 0.45):
        """Depth painted over the frame (used by --depth)."""
        norm = self.infer(frame)
        heat = self.colorize(norm, colormap)
        return cv2.addWeighted(frame, 1.0 - alpha, heat, alpha, 0)
