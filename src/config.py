"""Convenience re-export: ``import config; config.load()``.

The real implementation lives in ``utils/config.py``; this shim exists so the
project root of the src package has a single obvious entry point for tooling
and for the quick check:  python -c "import config; print(config.load())"
"""

from __future__ import annotations

from utils.config import DEFAULT_CONFIG, REPO_ROOT, apply_runtime, load  # noqa: F401

__all__ = ["load", "apply_runtime", "DEFAULT_CONFIG", "REPO_ROOT"]
