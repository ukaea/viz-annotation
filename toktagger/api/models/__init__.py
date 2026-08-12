"""Contains code for implemented ML models."""

import importlib.util
import os
from fastapi import HTTPException


def models_dependencies_installed() -> bool:
    return importlib.util.find_spec("ray") is not None


# When the driver is launched via `uv run`, Ray by default re-launches every worker
# through a fresh `uv run` subprocess to mirror the driver's environment. That
# re-resolves the project from the worker's cwd on every spawn and is prone to
# VIRTUAL_ENV mismatches that crash workers with ModuleNotFoundError: ray. Disable
# it so workers just reuse the driver's already-working interpreter instead. This
# is a no-op unless the driver is actually run via `uv run`, so it's always safe.
# Must be set before `ray` is imported anywhere, since Ray reads it at import time.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def check_models_enabled():
    if not models_dependencies_installed():
        raise HTTPException(
            status_code=503,
            detail="ML model features are disabled (optional dependencies missing)",
        )


if models_dependencies_installed():
    from toktagger.api.models.dtw_motif import DTWMotifModel as DTWMotifModel
    from toktagger.api.models.stumpy_motif import StumpyMotifModel as StumpyMotifModel
    from toktagger.api.models.minirocket import MiniRocketModel as MiniRocketModel
    from toktagger.api.models.shapelet import (
        ShapeletTransformModel as ShapeletTransformModel,
    )
    from toktagger.api.models.disruption import DisruptionCNN as DisruptionCNN
    from toktagger.api.models.temp import VideoCNN as VideoCNN
