"""Contains code for implemented ML models."""

import importlib.util
from fastapi import HTTPException


def models_dependencies_installed() -> bool:
    return importlib.util.find_spec("ray") is not None


def check_models_enabled():
    if not models_dependencies_installed():
        raise HTTPException(
            status_code=503,
            detail="ML model features are disabled (optional dependencies missing)",
        )


if models_dependencies_installed():
    from toktagger.api.models.disruption import DisruptionCNN as DisruptionCNN
