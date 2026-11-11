"""Configuration loading for the vision lab.

Priority (highest wins):

    1. command-line flags        --conf 0.5
    2. config.yaml               conf: 0.4
    3. argparse defaults         (built into the CLI)

The YAML is intentionally flat-ish: it mirrors argparse destination names
(`det_model`, `pose_imgsz`, ...) so that `argparse.set_defaults()` can consume
it directly. A few section keys are re-mapped in `_ALIASES`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

# nested YAML key -> argparse dest
_ALIASES = {
    "models.det": "det_model",
    "models.pose": "pose_model",
    "models.seg": "seg_model",
    "inference.backend": "backend",
    "inference.device": "ov_device",
    "inference.imgsz": "imgsz",
    "inference.pose_imgsz": "pose_imgsz",
    "inference.conf": "conf",
    "inference.pose_conf": "pose_conf",
    "inference.threads": "threads",
    "inference.mirror": "mirror",
    "distance.person_height": "person_height",
    "distance.calibration_focal": "calibration_focal",
    "distance.focal_config": "focal_config",
    "features.seg_every": "seg_every",
    "features.depth_every": "depth_every",
    "features.eyes_every": "eyes_every",
    "features.hands_every": "hands_every",
    "features.shape_every": "shape_every",
    "features.lowconf": "lowconf",
    "features.lowconf_cooldown": "lowconf_cooldown",
    "paths.lowconf_dir": "lowconf_dir",
    "paths.csv": "csv_path",
    "paths.heatmap": "heatmap_output",
    "paths.gaze_csv": "gaze_csv",
}


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml and return a flat {argparse_dest: value} mapping.

    Returns an empty dict (never raises) when the file is absent or broken, so
    the CLI keeps working out of the box. Broken YAML is reported as a warning.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        return {}

    try:
        import yaml
    except ImportError:                                       # pragma: no cover
        log.warning("pyyaml is not installed, ignoring %s", cfg_path)
        return {}

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("cannot read %s: %s", cfg_path, exc)
        return {}

    if not isinstance(raw, dict):
        log.warning("%s must contain a mapping at the top level", cfg_path)
        return {}

    flat = _flatten(raw)
    resolved: dict[str, Any] = {}
    for key, value in flat.items():
        if value is None:
            continue
        dest = _ALIASES.get(key, key.replace(".", "_"))
        resolved[dest] = value

    log.info("config loaded from %s (%d keys)", cfg_path, len(resolved))
    return resolved


def apply_runtime(cfg: dict[str, Any]) -> None:
    """Push values that live in module constants (not argparse) into place.

    PERSON_HEIGHT and CALIBRATION_FOCAL are read inside utils.calibration at
    call time, so assigning them here is enough - no need to thread them
    through every call signature.
    """
    try:
        from utils import calibration
    except ImportError:                                       # pragma: no cover
        return

    height = cfg.get("person_height")
    if isinstance(height, (int, float)) and height > 0:
        calibration.PERSON_HEIGHT = float(height)

    focal = cfg.get("calibration_focal")
    if isinstance(focal, (int, float)) and focal > 0:
        calibration.CALIBRATION_FOCAL = float(focal)

    focal_path = cfg.get("focal_config")
    if isinstance(focal_path, str) and focal_path:
        calibration.FOCAL_CONFIG = focal_path
