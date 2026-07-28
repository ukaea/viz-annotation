import logging
from pathlib import Path
from urllib.request import urlretrieve

import torch

# import toktagger.api.config as config
import os

logger = logging.getLogger(__name__)

# Pretrained checkpoints available from the Ultralytics v8.4.0 assets release.
_ULTRALYTICS_ASSET_BASE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0"
)

MODEL_URLS = {
    "yolov8n.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/yolov8n.pt",
    "yolo11n.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/yolo11n.pt",
    "yolo26n.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/yolo26n.pt",
    "yolo26x.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/yolo26x.pt",
    "rtdetr-x.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/rtdetr-x.pt",
    "rtdetr-l.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/rtdetr-l.pt",
}

MODEL_FAMILIES = {
    "yolov8n.pt": "yolo",
    "yolo11n.pt": "yolo",
    "yolo26n.pt": "yolo",
    "yolo26x.pt": "yolo",
    "rtdetr-x.pt": "rtdetr",
    "rtdetr-l.pt": "rtdetr",
}


def check_pretrained_model_availability(
    model_name: str,
    force_download: bool = False,
) -> Path:
    """Return the local path to a pretrained model.

    The model is downloaded into TokTagger's cache when it is not already
    available. Setting ``force_download`` replaces an existing cached model.
    """
    if model_name not in MODEL_URLS:
        available_models = ", ".join(MODEL_URLS)
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models}"
        )

    model_family = MODEL_FAMILIES[model_name]
    model_dir = get_toktagger_cache_dir() / "pretrained" / "ultralytics" / model_family
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / model_name

    if model_path.exists() and not force_download:
        logger.debug(
            "Model already exists at %s; skipping download.",
            model_path,
        )
        return model_path

    logger.info(
        "Downloading pretrained model %s to %s.",
        model_name,
        model_path,
    )

    # Download the model
    urlretrieve(MODEL_URLS[model_name], model_path)

    return model_path


def get_torch_device() -> torch.device:
    """Use Apple Metal when available, otherwise use the CPU.

    Ray controls NVIDIA GPU visibility for its workers, but it does not manage
    Apple Metal as a GPU resource. MPS therefore needs explicit selection.
    """
    # check if the torch version is compatible with mps
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_toktagger_cache_dir() -> Path:
    """Return the model-storage directory supplied to the Ray worker."""
    cache_dir = Path(os.environ["MODEL_STORAGE"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
