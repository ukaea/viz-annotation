import logging
from pathlib import Path
from urllib.request import urlretrieve
import os

import torch
from ultralytics import settings

logger = logging.getLogger(__name__)

# Pretrained checkpoints available from the Ultralytics v8.4.0 assets release.
_ULTRALYTICS_ASSET_BASE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0"
)

MODEL_URLS = {
    "yolov8n.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/yolov8n.pt",
    "yolo11n.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/yolo11n.pt",
    "yolo26n.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/yolo26n.pt",
    "yolo26m.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/yolo26m.pt",
    "yolo26l.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/yolo26l.pt",
    "yolo26x.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/yolo26x.pt",
    "rtdetr-x.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/rtdetr-x.pt",
    "rtdetr-l.pt": f"{_ULTRALYTICS_ASSET_BASE_URL}/rtdetr-l.pt",
}

MODEL_FAMILIES = {
    "yolov8n.pt": "yolo",
    "yolo11n.pt": "yolo",
    "yolo26n.pt": "yolo",
    "yolo26m.pt": "yolo",
    "yolo26l.pt": "yolo",
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

    # Download to a temporary file so an interrupted transfer cannot leave a
    # partial checkpoint at the final cache path.
    temporary_path = model_path.with_name(f"{model_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)

    try:
        urlretrieve(MODEL_URLS[model_name], temporary_path)
        temporary_path.replace(model_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

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


def resolve_weights_path(
    results_dir: Path,
    weights_filename: str | None = None,
) -> Path:
    """
    Resolve an Ultralytics checkpoint within a model results directory.
    An explicitly supplied filename takes precedence. Otherwise, prefer the
    best training checkpoint and fall back to the final checkpoint.
    """
    # If a specific weights file was supplied for loading
    if weights_filename is not None:
        weights_path = results_dir.joinpath(weights_filename)

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Could not find model weights at {weights_path}"
            )

        return weights_path

    # Use the default Ultralytics checkpoint
    # ultralytics weights are saved in weights directory
    weights_dir = results_dir.joinpath("weights")
    for filename in ("best.pt", "last.pt"):
        weights_path = weights_dir.joinpath(filename)

        if weights_path.is_file():
            return weights_path

    raise FileNotFoundError(
        f"Could not find best.pt or last.pt in {results_dir.joinpath('weights')}"
    )

def prepare_ultralytics_amp_weights() -> None:
    """Prepare the checkpoint used by Ultralytics' CUDA AMP check.

    Ultralytics performs the check using a hard-coded `yolo26n.pt`.
    Cache that checkpoint under MODEL_STORAGE and configure Ultralytics
    to reuse it instead of downloading another copy into the working directory.

    REMOVE BELOW COMMENT AFTER SOMEONE HAS TESTED IT ON A NVIDIA GPU.
    https://github.com/ukaea/toktagger/pull/326#discussion_r3754183103
    It is downloaded into TokTagger's cache on the first GPU training run,
    even when the user selected a different model size.
    """
    amp_checkpoint = check_pretrained_model_availability("yolo26n.pt")
    settings.update(weights_dir=str(amp_checkpoint.parent))