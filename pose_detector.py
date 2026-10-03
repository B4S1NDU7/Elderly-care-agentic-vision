"""
Pose & Detection Analyzer
==========================
Uses YOLOv8 for person detection and MediaPipe Pose for skeleton extraction.
Provides structured pose features that help the VLM agent make better decisions.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PoseFeatures:
    """Structured features extracted from pose estimation."""
    person_detected: bool
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]]  # x1,y1,x2,y2

    # Pose landmarks (normalized 0-1)
    landmarks: Optional[Dict[str, Dict[str, float]]]  # name -> {x, y, z, visibility}

    # High-level derived features
    body_is_horizontal: bool = False       # lying down
    body_is_vertical: bool = False         # standing / walking
    body_is_diagonal: bool = False         # sitting or transitioning
    head_above_hips: bool = True
    hips_above_knees: bool = True
    knees_above_ankles: bool = True
    is_moving: bool = False                # motion between consecutive frames
    relative_height_ratio: float = 0.0    # bbox height / frame height
    centroid_x_normalized: float = 0.5    # horizontal position 0=left, 1=right
    centroid_y_normalized: float = 0.5    # vertical position 0=top, 1=bottom

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_detected": self.person_detected,
            "confidence": round(self.confidence, 3),
            "body_is_horizontal": self.body_is_horizontal,
            "body_is_vertical": self.body_is_vertical,
            "body_is_diagonal": self.body_is_diagonal,
            "head_above_hips": self.head_above_hips,
            "hips_above_knees": self.hips_above_knees,
            "knees_above_ankles": self.knees_above_ankles,
            "is_moving": self.is_moving,
            "relative_height_ratio": round(self.relative_height_ratio, 3),
            "centroid_x": round(self.centroid_x_normalized, 3),
            "centroid_y": round(self.centroid_y_normalized, 3),
        }

    def posture_description(self) -> str:
        """Return a human-readable posture hint."""
        if not self.person_detected:
            return "No person detected in frame."
        parts = []
        if self.body_is_horizontal:
            parts.append("body appears HORIZONTAL (likely lying)")
        elif self.body_is_vertical:
            parts.append("body appears VERTICAL (standing/walking)")
        else:
            parts.append("body appears DIAGONAL/SEATED")

        parts.append(f"relative height ratio={self.relative_height_ratio:.2f}")
        if not self.head_above_hips:
            parts.append("head NOT above hips (unusual posture)")
        return "; ".join(parts)


@dataclass
class BedRegion:
    """Detected bed region in the frame."""
    detected: bool
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1,y1,x2,y2
    confidence: float = 0.0

    def person_on_bed(self, person_bbox: Optional[Tuple[int, int, int, int]]) -> bool:
        """Check if the person's lower body overlaps with the bed region."""
        if not self.detected or person_bbox is None or self.bbox is None:
            return False
        # Use bottom-half of person bbox for overlap test
        px1, py1, px2, py2 = person_bbox
        person_bottom = (px1, py1 + (py2 - py1) // 2, px2, py2)
        bx1, by1, bx2, by2 = self.bbox
        # Check overlap
        ix1 = max(person_bottom[0], bx1)
        iy1 = max(person_bottom[1], by1)
        ix2 = min(person_bottom[2], bx2)
        iy2 = min(person_bottom[3], by2)
        return ix1 < ix2 and iy1 < iy2


class PoseDetector:
    """
    Combines YOLOv8 person detection with MediaPipe Pose landmark extraction.

    Pipeline:
    1. YOLOv8 detects all persons → keep highest-confidence detection
    2. Crop the person region
    3. MediaPipe Pose extracts 33-landmark skeleton
    4. Compute high-level features (horizontal/vertical body, relative height, etc.)
    """

    # COCO 17 landmark names (matching YOLOv8-pose output)
    _COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(self, yolo_model: str = "yolov8n-pose.pt", device: str = "cpu"):
        self._yolo = None
        self._mp_pose = None
        self._yolo_model_name = yolo_model
        self._device = device
        self._prev_centroid: Optional[Tuple[float, float]] = None
        self._motion_threshold = 0.04  # normalized centroid shift

    def _ensure_loaded(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(self._yolo_model_name)
                logger.info(f"YOLO loaded: {self._yolo_model_name}")
            except Exception as e:
                logger.warning(f"YOLO unavailable: {e}")

        # MediaPipe optional fallback
        if self._mp_pose is None and hasattr(cv2, 'face'):
            try:
                import mediapipe as mp
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                    self._mp_pose = mp.solutions.pose.Pose(
                        static_image_mode=True,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.4,
                    )
            except Exception:
                pass

    def analyze(self, image: np.ndarray) -> PoseFeatures:
        """Run full person detection + pose estimation on an image (BGR)."""
        self._ensure_loaded()
        h, w = image.shape[:2]

        # Step 1: Person detection + keypoint extraction with YOLOv8-pose
        person_bbox, det_conf, landmarks = self._detect_person(image)

        if person_bbox is None:
            self._prev_centroid = None
            return PoseFeatures(
                person_detected=False,
                confidence=0.0,
                bbox=None,
                landmarks=None,
            )

        # Step 2: Fallback to MediaPipe if YOLO keypoints not available
        if landmarks is None and self._mp_pose is not None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = self._extract_landmarks(rgb, h, w)

        # Step 3: Derive features
        features = self._compute_features(
            person_bbox, landmarks, det_conf, h, w
        )
        return features

    def detect_bed_region(self, image: np.ndarray) -> BedRegion:
        """
        Simple heuristic bed detection using YOLOv8.
        YOLO's 'bed' class (COCO class 59) is used if available.
        Falls back to the lower-center region if not detected.
        """
        self._ensure_loaded()
        if self._yolo is None:
            # Fallback: assume bed is in the lower-center portion of the frame
            h, w = image.shape[:2]
            return BedRegion(
                detected=True,
                bbox=(w // 4, h // 2, 3 * w // 4, h),
                confidence=0.3,
            )

        try:
            results = self._yolo(image, verbose=False, classes=[59])  # 59 = bed in COCO
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    box = r.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    return BedRegion(detected=True, bbox=(x1, y1, x2, y2), confidence=conf)
        except Exception as e:
            logger.debug(f"Bed detection error: {e}")

        # Fallback heuristic
        h, w = image.shape[:2]
        return BedRegion(
            detected=True,
            bbox=(w // 6, h // 3, 5 * w // 6, h),
            confidence=0.2,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_person(
        self, image: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float, Optional[Dict[str, Dict[str, float]]]]:
        """Return bounding box, confidence, and keypoint landmarks of the most prominent person."""
        if self._yolo is None:
            return None, 0.0, None
        try:
            results = self._yolo(image, verbose=False)
            best_box, best_conf, best_landmarks = None, 0.0, None
            h, w = image.shape[:2]
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for i, box in enumerate(r.boxes):
                    cls_id = int(box.cls[0]) if hasattr(box, "cls") else 0
                    if cls_id != 0:
                        continue
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = tuple(map(int, box.xyxy[0].tolist()))
                        if hasattr(r, "keypoints") and r.keypoints is not None and len(r.keypoints) > i:
                            kp_tensor = r.keypoints.data[i].cpu().numpy()
                            best_landmarks = {}
                            for idx, name in enumerate(self._COCO_KEYPOINTS):
                                if idx < len(kp_tensor):
                                    kx, ky, kconf = kp_tensor[idx]
                                    best_landmarks[name] = {
                                        "x": float(kx / max(w, 1)),
                                        "y": float(ky / max(h, 1)),
                                        "z": 0.0,
                                        "visibility": float(kconf),
                                    }
            return best_box, best_conf, best_landmarks
        except Exception as e:
            logger.debug(f"Person detection error: {e}")
            return None, 0.0, None

    def _extract_landmarks(
        self, rgb: np.ndarray, h: int, w: int
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Run MediaPipe Pose and return named landmarks."""
        if self._mp_pose is None:
            return None
        try:
            result = self._mp_pose.process(rgb)
            if not result.pose_landmarks:
                return None
            out = {}
            lm_list = result.pose_landmarks.landmark
            for name, idx in self._MP_LANDMARKS.items():
                lm = lm_list[idx]
                out[name] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                }
            return out
        except Exception as e:
            logger.debug(f"Pose extraction error: {e}")
            return None

    def _compute_features(
        self,
        bbox: Tuple[int, int, int, int],
        landmarks: Optional[Dict[str, Dict[str, float]]],
        confidence: float,
        frame_h: int,
        frame_w: int,
    ) -> PoseFeatures:
        x1, y1, x2, y2 = bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        rel_height = bbox_h / frame_h
        cx = ((x1 + x2) / 2) / frame_w
        cy = ((y1 + y2) / 2) / frame_h

        # Motion detection vs previous frame
        is_moving = False
        if self._prev_centroid is not None:
            dx = cx - self._prev_centroid[0]
            dy = cy - self._prev_centroid[1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            is_moving = dist > self._motion_threshold
        self._prev_centroid = (cx, cy)

        # Posture from bbox aspect ratio (fast heuristic)
        aspect = bbox_h / max(bbox_w, 1)
        body_horizontal = aspect < 0.9 and rel_height < 0.4
        body_vertical = aspect > 1.8 and rel_height > 0.35

        # Landmark-based posture (more reliable when available)
        head_above_hips = True
        hips_above_knees = True
        knees_above_ankles = True

        if landmarks:
            nose_y = landmarks.get("nose", {}).get("y", 0.0)
            l_hip_y = landmarks.get("left_hip", {}).get("y", 1.0)
            r_hip_y = landmarks.get("right_hip", {}).get("y", 1.0)
            hip_y = (l_hip_y + r_hip_y) / 2

            l_knee_y = landmarks.get("left_knee", {}).get("y", 1.0)
            r_knee_y = landmarks.get("right_knee", {}).get("y", 1.0)
            knee_y = (l_knee_y + r_knee_y) / 2

            l_ankle_y = landmarks.get("left_ankle", {}).get("y", 1.0)
            r_ankle_y = landmarks.get("right_ankle", {}).get("y", 1.0)
            ankle_y = (l_ankle_y + r_ankle_y) / 2

            # In image coords, smaller y = higher up
            head_above_hips = nose_y < hip_y
            hips_above_knees = hip_y < knee_y
            knees_above_ankles = knee_y < ankle_y

            # Recalculate posture from landmarks
            vert_span = abs(nose_y - ankle_y)
            l_shoulder_x = landmarks.get("left_shoulder", {}).get("x", 0.5)
            r_shoulder_x = landmarks.get("right_shoulder", {}).get("x", 0.5)
            horiz_span = abs(l_shoulder_x - r_shoulder_x)

            body_horizontal = (vert_span < 0.35) or (not head_above_hips and vert_span < 0.5)
            body_vertical = vert_span > 0.5 and head_above_hips and hips_above_knees

        body_diagonal = not body_horizontal and not body_vertical

        return PoseFeatures(
            person_detected=True,
            confidence=confidence,
            bbox=bbox,
            landmarks=landmarks,
            body_is_horizontal=body_horizontal,
            body_is_vertical=body_vertical,
            body_is_diagonal=body_diagonal,
            head_above_hips=head_above_hips,
            hips_above_knees=hips_above_knees,
            knees_above_ankles=knees_above_ankles,
            is_moving=is_moving,
            relative_height_ratio=rel_height,
            centroid_x_normalized=cx,
            centroid_y_normalized=cy,
        )

    def draw_pose(self, image: np.ndarray, features: PoseFeatures) -> np.ndarray:
        """Draw pose features on image for debugging."""
        vis = image.copy()
        if features.bbox:
            x1, y1, x2, y2 = features.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if features.landmarks:
            h, w = image.shape[:2]
            for name, lm in features.landmarks.items():
                if lm.get("visibility", 0) > 0.3:
                    px = int(lm["x"] * w)
                    py = int(lm["y"] * h)
                    cv2.circle(vis, (px, py), 4, (255, 0, 0), -1)

        # Label
        label = features.posture_description()[:60]
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return vis
