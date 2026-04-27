"""
utils/video.py
--------------
Video processing helpers: frame iteration, FPS tracking,
and saving annotated video output.
"""

import time
import logging
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── FPS Tracker ────────────────────────────────────────────────────────────────

class FPSTracker:
    """
    Exponential moving-average FPS tracker.
    Call .tick() once per frame; read .fps for current estimate.
    """

    def __init__(self, alpha: float = 0.05):
        self._alpha = alpha
        self._fps: float = 0.0
        self._last: float = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            instant = 1.0 / dt
            self._fps = (
                instant if self._fps == 0.0
                else self._alpha * instant + (1 - self._alpha) * self._fps
            )
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ── Frame iterator ─────────────────────────────────────────────────────────────

def iter_video_frames(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR frames from a video file one by one.

    Args:
        video_path: Path to the video file.

    Yields:
        numpy BGR frames.

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def get_video_metadata(video_path: str) -> dict:
    """Return basic metadata (fps, width, height, frame_count) for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    meta = {
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


# ── Video writer ───────────────────────────────────────────────────────────────

def save_annotated_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float = 25.0,
) -> str:
    """
    Write a list of annotated BGR frames to an MP4 file.

    Args:
        frames:      List of annotated BGR numpy frames.
        output_path: Destination path for the output video.
        fps:         Output frame rate.

    Returns:
        The output path on success.

    Raises:
        RuntimeError: If the writer fails to initialise.
    """
    if not frames:
        raise ValueError("No frames to write.")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_path}")

    for frame in frames:
        writer.write(frame)

    writer.release()
    logger.info(f"Annotated video saved → {output_path}")
    return output_path


def make_temp_video_path(suffix: str = ".mp4") -> str:
    """Create a named temporary file path for video output."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    return tmp.name
