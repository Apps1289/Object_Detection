"""
models/loader.py
----------------
Handles YOLOv8 model downloading and loading.
Supports multiple model sizes: nano, small, medium, large, xlarge.
Uses /tmp for weights cache so it works on Render (read-only filesystem).
"""

import os
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_OPTIONS = {
    "YOLOv8n (Nano — fastest)":   "yolov8n.pt",
    "YOLOv8s (Small — balanced)":  "yolov8s.pt",
    "YOLOv8m (Medium — accurate)": "yolov8m.pt",
    "YOLOv8l (Large — very accurate)": "yolov8l.pt",
    "YOLOv8x (XLarge — most accurate)": "yolov8x.pt",
}

# Render has a read-only filesystem except /tmp — use that for weights cache
WEIGHTS_DIR = Path(os.environ.get("YOLO_CONFIG_DIR", "/tmp/ultralytics")) / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def get_model_path(weights_filename: str) -> Path:
    return WEIGHTS_DIR / weights_filename


def load_model(weights_filename: str = "yolov8n.pt") -> YOLO:
    """
    Load a YOLOv8 model from local /tmp cache or download automatically.
    On Render, the first request triggers a download; subsequent requests use cache.
    """
    model_path = get_model_path(weights_filename)
    try:
        if model_path.exists():
            logger.info(f"Loading cached model: {model_path}")
            model = YOLO(str(model_path))
        else:
            logger.info(f"Downloading model: {weights_filename}")
            model = YOLO(weights_filename)
            downloaded = Path(weights_filename)
            if downloaded.exists():
                downloaded.rename(model_path)
                logger.info(f"Cached at: {model_path}")
        logger.info(f"Model ready — {weights_filename}")
        return model
    except Exception as exc:
        logger.error(f"Model load failed '{weights_filename}': {exc}")
        raise RuntimeError(f"Model loading failed: {exc}") from exc


def get_class_names(model: YOLO) -> dict:
    return model.names
