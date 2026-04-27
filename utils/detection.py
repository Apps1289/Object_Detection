"""
utils/detection.py
------------------
Core detection helpers: run inference, draw bounding boxes,
annotate frames with confidence scores and class labels.
"""

import time
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ── Colour palette (BGR) — one colour per class, cycling if > 80 classes ──────
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def _get_color(class_id: int) -> tuple:
    """Deterministic colour for a given class index."""
    return _PALETTE[class_id % len(_PALETTE)]


# ── Main detection function ────────────────────────────────────────────────────

def run_detection(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float = 0.40,
    iou_threshold: float = 0.45,
) -> tuple[np.ndarray, list[dict], float]:
    """
    Run YOLOv8 inference on a single BGR frame.

    Args:
        model:          Loaded YOLO model.
        frame:          BGR numpy array (from OpenCV).
        conf_threshold: Minimum confidence to display a detection.
        iou_threshold:  NMS IoU threshold.

    Returns:
        annotated_frame: Frame with bounding boxes drawn.
        detections:      List of dicts with keys:
                         class_id, class_name, confidence, bbox (x1,y1,x2,y2).
        inference_ms:    Inference time in milliseconds.
    """
    t0 = time.perf_counter()
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    detections: list[dict] = []
    annotated = frame.copy()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            color = _get_color(cls_id)

            detections.append({
                "class_id":   cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox":       (x1, y1, x2, y2),
            })

            _draw_box(annotated, x1, y1, x2, y2, cls_name, conf, color)

    return annotated, detections, inference_ms


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    confidence: float,
    color: tuple,
) -> None:
    """Draw a single bounding box with label and confidence on frame (in-place)."""
    thickness = max(1, int(min(frame.shape[:2]) / 300))
    font_scale = max(0.4, min(frame.shape[:2]) / 1000)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    text = f"{label} {confidence:.0%}"
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_y = max(y1, th + 6)
    cv2.rectangle(
        frame,
        (x1, label_y - th - baseline - 4),
        (x1 + tw + 4, label_y + baseline - 2),
        color,
        -1,  # filled
    )

    # Label text (white)
    cv2.putText(
        frame, text,
        (x1 + 2, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def overlay_stats(
    frame: np.ndarray,
    fps: Optional[float],
    inference_ms: float,
    count: int,
) -> np.ndarray:
    """
    Overlay FPS, inference time, and detection count in the top-left corner.

    Args:
        frame:        BGR frame to annotate.
        fps:          Frames per second (None if not applicable).
        inference_ms: Inference time in ms.
        count:        Number of objects detected.

    Returns:
        Annotated frame (copy).
    """
    out = frame.copy()
    lines = [
        f"Objects: {count}",
        f"Infer: {inference_ms:.1f} ms",
    ]
    if fps is not None:
        lines.insert(0, f"FPS: {fps:.1f}")

    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, w / 1500)
    thick = max(1, w // 500)
    pad = 8

    for i, line in enumerate(lines):
        (tw, th), bl = cv2.getTextSize(line, font, scale, thick)
        y = pad + (th + pad) * (i + 1)
        # Shadow
        cv2.putText(out, line, (pad + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        # Text
        cv2.putText(out, line, (pad, y), font, scale, (0, 255, 80), thick, cv2.LINE_AA)

    return out


# ── Image utilities ────────────────────────────────────────────────────────────

def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB/RGBA image to a BGR numpy array."""
    img = image.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR numpy array to RGB (for Streamlit display)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
