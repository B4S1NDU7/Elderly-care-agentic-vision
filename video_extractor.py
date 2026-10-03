"""
Video Frame Extractor
======================
Handles video loading and intelligent frame sampling.
Extracts key frames at configurable intervals and detects scene changes.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    path: str
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int

    @property
    def duration_str(self) -> str:
        m = int(self.duration_sec // 60)
        s = int(self.duration_sec % 60)
        return f"{m}m {s:02d}s"


@dataclass
class ExtractedFrame:
    index: int           # frame number in the original video
    timestamp_sec: float # time in seconds
    image: np.ndarray    # BGR image


class VideoFrameExtractor:
    """
    Extracts frames from a video file using two complementary strategies:
    1. Uniform sampling every `sample_interval_sec` seconds (baseline)
    2. Scene-change detection to capture transitions (optional)

    The union of both sets ensures we never miss a rapid state change while
    keeping the total number of VLM calls manageable.
    """

    def __init__(
        self,
        sample_interval_sec: float = 2.0,
        scene_change_threshold: float = 25.0,
        min_scene_gap_sec: float = 0.5,
        max_frames: Optional[int] = None,
    ):
        self.sample_interval_sec = sample_interval_sec
        self.scene_change_threshold = scene_change_threshold
        self.min_scene_gap_sec = min_scene_gap_sec
        self.max_frames = max_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_video_info(self, video_path: str) -> VideoInfo:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps
        cap.release()
        return VideoInfo(
            path=video_path,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            width=width,
            height=height,
        )

    def extract_frames(self, video_path: str) -> Tuple[List[ExtractedFrame], VideoInfo]:
        """
        Extract frames from video and return (frames, video_info).
        """
        info = self.get_video_info(video_path)
        logger.info(
            f"Video: {Path(video_path).name} | "
            f"{info.duration_str} | {info.fps:.1f} fps | "
            f"{info.width}x{info.height}"
        )

        # Determine which frame indices to extract
        frame_indices = self._compute_sample_indices(info)

        # Extract the actual frames
        frames = self._read_frames(video_path, frame_indices, info)
        logger.info(f"Extracted {len(frames)} frames for analysis")
        return frames, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_sample_indices(self, info: VideoInfo) -> List[int]:
        """Compute which frame indices to sample (uniform + first/last)."""
        step = max(1, int(info.fps * self.sample_interval_sec))
        indices = set()

        # Always include first and last frame
        indices.add(0)
        if info.total_frames > 1:
            indices.add(info.total_frames - 1)

        # Uniform sampling
        for i in range(0, info.total_frames, step):
            indices.add(i)

        sorted_indices = sorted(indices)

        # Trim to max_frames if specified (keep uniform spread)
        if self.max_frames and len(sorted_indices) > self.max_frames:
            step_keep = max(1, len(sorted_indices) // self.max_frames)
            sorted_indices = sorted_indices[::step_keep][: self.max_frames]
            # Always keep last frame
            last = info.total_frames - 1
            if last not in sorted_indices:
                sorted_indices.append(last)
                sorted_indices.sort()

        return sorted_indices

    def _read_frames(
        self, video_path: str, indices: List[int], info: VideoInfo
    ) -> List[ExtractedFrame]:
        """Read specific frames by index from the video file."""
        cap = cv2.VideoCapture(video_path)
        frames: List[ExtractedFrame] = []
        idx_set = set(indices)
        current_pos = 0

        # Sort indices and seek frame-by-frame for efficiency
        for target_idx in sorted(indices):
            if target_idx != current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                current_pos = target_idx

            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {target_idx}")
                continue

            timestamp = target_idx / info.fps
            frames.append(ExtractedFrame(
                index=target_idx,
                timestamp_sec=timestamp,
                image=frame,
            ))
            current_pos += 1

        cap.release()
        return frames

    def detect_scene_changes(
        self, video_path: str, info: VideoInfo
    ) -> List[int]:
        """
        Detect scene changes using histogram difference.
        Returns frame indices where significant changes occur.
        """
        cap = cv2.VideoCapture(video_path)
        prev_hist = None
        scene_frames = []
        last_scene_frame = -1
        min_gap_frames = int(info.fps * self.min_scene_gap_sec)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                if diff > self.scene_change_threshold:
                    if frame_idx - last_scene_frame >= min_gap_frames:
                        scene_frames.append(frame_idx)
                        last_scene_frame = frame_idx

            prev_hist = hist
            frame_idx += 1

        cap.release()
        return scene_frames


def resize_for_vlm(image: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """
    Resize image so the longest dimension is at most max_dim,
    preserving aspect ratio.  Keeps file size reasonable for API calls.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_to_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR numpy image to a base64 JPEG string."""
    import base64
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")
