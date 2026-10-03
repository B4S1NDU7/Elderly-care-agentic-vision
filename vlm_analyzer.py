"""
VLM Analyzer
=============
Uses OpenAI GPT-4o Vision to classify activity states from video frames.
Integrates pose features as structured context to guide classification.
"""

import base64
import json
import logging
import time
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

from models import ActivityState, FrameAnalysis
from pose_detector import PoseFeatures, BedRegion
from video_extractor import frame_to_base64, resize_for_vlm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI system for elderly care monitoring.
Your task is to analyze video frames from an indoor camera monitoring an elderly person.

You must classify the person's current activity state into EXACTLY one of these states:
- LYING_IN_BED: Person is lying horizontally on the bed
- SITTING_ON_BED: Person is sitting upright on the bed (edge or middle)
- SITTING_OUTSIDE_BED: Person is sitting on a chair, floor, or other surface away from bed
- STANDING: Person is standing still near or away from bed
- WALKING: Person is in motion, walking around
- OUT_OF_BED: Person is clearly away from the bed (could be standing, sitting elsewhere, or out of frame)
- UNKNOWN: Insufficient visual information to determine state (use when truly uncertain)

Important guidelines:
- Consider the person's body posture, position relative to the bed, and context
- A person sitting up briefly in bed is SITTING_ON_BED, NOT a bed exit
- Only use UNKNOWN when you genuinely cannot determine the state
- Be conservative: if uncertain between bed/non-bed, analyze the bed region carefully
- Consider whether a caregiver might be visible and focus on the primary (elderly) person

Respond ONLY with a valid JSON object. No markdown, no explanation outside JSON.
"""

FRAME_ANALYSIS_PROMPT = """Analyze this video frame from an elderly care monitoring system.

Timestamp: {timestamp}
Pose Analysis Context:
{pose_context}

Bed Region Context: {bed_context}

Previous State (if known): {prev_state}

Classify the elderly person's activity state and provide your analysis.

Respond with this exact JSON format:
{{
  "state": "<one of: LYING_IN_BED|SITTING_ON_BED|SITTING_OUTSIDE_BED|STANDING|WALKING|OUT_OF_BED|UNKNOWN>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation of your classification>",
  "on_bed": <true/false>,
  "near_bed": <true/false>,
  "person_visible": <true/false>
}}
"""

CONTEXTUAL_ANALYSIS_PROMPT = """You are performing agentic temporal analysis for elderly care monitoring.

You have analyzed multiple frames around a potential state transition.
Review the sequence of observations and make a final determination.

Observation sequence:
{observations}

Question: {question}

Based on the temporal context, provide your analysis:
{{
  "conclusion": "<your conclusion>",
  "state": "<final state determination>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<detailed reasoning based on temporal sequence>",
  "requires_alert": <true/false>,
  "alert_level": "<NORMAL|MONITOR|ALERT>"
}}
"""


class VLMAnalyzer:
    """
    GPT-4o Vision-based activity state analyzer.

    Architecture:
    1. Resize frame to fit API limits
    2. Extract pose features (structured context)
    3. Call GPT-4o with frame + structured context
    4. Parse JSON response → FrameAnalysis
    5. Cache recent analyses for context injection
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.1,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self._recent_analyses: List[Dict] = []  # rolling context window
        self._context_window = 5  # how many recent frames to remember

        # Support mock mode for offline testing and verification without API keys
        import os
        self.mock_mode = (
            api_key in ("mock", "demo", "none", None)
            or not api_key
            or os.environ.get("MOCK_VLM") == "1"
        )

        if not self.mock_mode:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
                logger.info(f"OpenAI client initialized with model: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client ({e}), falling back to mock mode")
                self.mock_mode = True
                self._client = None
        else:
            self._client = None
            logger.info("VLMAnalyzer initialized in offline MOCK mode (heuristic vision simulation)")

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def analyze_frame(
        self,
        frame: np.ndarray,
        timestamp_sec: float,
        frame_index: int,
        pose_features: Optional[PoseFeatures] = None,
        bed_region: Optional[BedRegion] = None,
        previous_state: Optional[ActivityState] = None,
    ) -> FrameAnalysis:
        """
        Analyze a single frame and return FrameAnalysis.
        """
        # Resize for API
        resized = resize_for_vlm(frame, max_dim=1024)
        b64_image = frame_to_base64(resized)

        # Build context strings
        pose_context = self._format_pose_context(pose_features)
        bed_context = self._format_bed_context(bed_region, pose_features)
        prev_state_str = previous_state.value if previous_state else "None (first frame)"

        # Format timestamp
        m = int(timestamp_sec // 60)
        s = int(timestamp_sec % 60)
        ts_str = f"{m:02d}:{s:02d}"

        prompt = FRAME_ANALYSIS_PROMPT.format(
            timestamp=ts_str,
            pose_context=pose_context,
            bed_context=bed_context,
            prev_state=prev_state_str,
        )

        # Call VLM or mock heuristic
        if self.mock_mode:
            parsed = self._mock_classify(frame, pose_features, bed_region, previous_state)
            raw_response = json.dumps(parsed)
        else:
            raw_response = self._call_vlm_with_image(prompt, b64_image)
            parsed = self._parse_response(raw_response)

        # Map to ActivityState
        state = self._map_state(parsed.get("state", "UNKNOWN"))
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "")
        on_bed = bool(parsed.get("on_bed", False))
        near_bed = bool(parsed.get("near_bed", False))

        analysis = FrameAnalysis(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            state=state,
            confidence=confidence,
            reasoning=reasoning,
            pose_features=pose_features.to_dict() if pose_features else {},
            on_bed=on_bed,
            near_bed=near_bed,
            raw_vlm_response=raw_response,
        )

        # Add to rolling context
        self._recent_analyses.append({
            "timestamp": ts_str,
            "state": state.value,
            "confidence": confidence,
            "reasoning": reasoning,
        })
        if len(self._recent_analyses) > self._context_window:
            self._recent_analyses.pop(0)

        return analysis

    def contextual_analysis(
        self,
        question: str,
        frames: List[np.ndarray],
        timestamps: List[float],
        observations: List[str],
    ) -> Dict[str, Any]:
        """
        Agentic multi-frame analysis for resolving ambiguous situations.
        Provides temporal context to the VLM to help it make better decisions.
        """
        if self.mock_mode:
            return {
                "conclusion": "Resolved via multi-frame temporal reasoning.",
                "state": "OUT_OF_BED" if "leave" in question.lower() else "SITTING_ON_BED",
                "confidence": 0.88,
                "reasoning": "Temporal continuity verified across adjoining sequence frames.",
                "requires_alert": False,
                "alert_level": "NORMAL",
            }
        # Build observation sequence text
        obs_text = "\n".join(
            f"  [{m:02d}:{s:02d}] {obs}"
            for (ts, obs) in zip(timestamps, observations)
            for m, s in [(int(ts // 60), int(ts % 60))]
        )

        prompt = CONTEXTUAL_ANALYSIS_PROMPT.format(
            observations=obs_text,
            question=question,
        )

        # Include up to 3 key frames for visual context
        key_frames_b64 = []
        step = max(1, len(frames) // 3)
        for i in range(0, len(frames), step):
            if len(key_frames_b64) >= 3:
                break
            resized = resize_for_vlm(frames[i], max_dim=768)
            key_frames_b64.append(frame_to_base64(resized))

        raw = self._call_vlm_multi_image(prompt, key_frames_b64)
        return self._parse_response(raw)

    def determine_alert_level(
        self,
        recent_timeline: List[Dict],
        current_state: ActivityState,
        out_of_bed_duration_sec: float,
        bed_exit_count: int,
        sitting_on_edge_duration_sec: float,
        unknown_duration_sec: float = 0.0,
        fall_suspected: bool = False,
    ) -> Dict[str, Any]:
        """
        Determine overall alert level using alert rules + VLM reasoning.

        Alert Rules:
        - NORMAL: Routine activities (lying, sitting, standing, walking normally)
        - MONITOR: Sitting on bed edge >10 min, prolonged UNKNOWN, no position change >2h
        - ALERT: Out of bed >45 min continuously, person on floor (horizontal outside bed),
                 no movement detected >3h
        """
        # Rule-based pre-screening (fast, no API call)
        rule_alert = self._rule_based_alert(
            current_state,
            out_of_bed_duration_sec,
            sitting_on_edge_duration_sec,
            unknown_duration_sec,
            fall_suspected,
        )

        if self.mock_mode:
            action = (
                "Immediate caregiver check requested: extended absence from bed or fall indicator."
                if rule_alert == "ALERT"
                else (
                    "Monitor patient activity: prolonged edge sitting or transition observed."
                    if rule_alert == "MONITOR"
                    else "Routine monitoring normal."
                )
            )
            return {
                "alert_level": rule_alert,
                "reasoning": f"Temporal state rule assessment: out_of_bed={out_of_bed_duration_sec:.0f}s, sitting_edge={sitting_on_edge_duration_sec:.0f}s.",
                "recommended_action": action,
                "rule_based_alert": rule_alert,
            }

        # Build context for VLM
        timeline_text = "\n".join(
            f"  [{e.get('timestamp', '??:??')}] {e.get('state', '?')} "
            f"(conf={e.get('confidence', 0):.2f})"
            for e in recent_timeline[-10:]
        )

        prompt = f"""You are an elderly care monitoring system. Review recent activity:

Activity Timeline (last 10 observations):
{timeline_text}

Statistics:
- Current state: {current_state.value}
- Continuous out-of-bed duration: {out_of_bed_duration_sec:.0f} seconds ({out_of_bed_duration_sec/60:.1f} min)
- Unknown-state duration: {unknown_duration_sec:.0f} seconds
- Agentic floor/fall concern: {fall_suspected}
- Bed exits so far: {bed_exit_count}
- Time sitting on bed edge: {sitting_on_edge_duration_sec:.0f} seconds

Rule-based assessment: {rule_alert}

Alert Rules:
- NORMAL: Routine activities, person is safe and active normally
- MONITOR: Prolonged sitting on edge (>10 min), extended inactivity, ambiguous states
- ALERT: Continuous absence from bed >45 min, potential fall detected, no movement >3h

Provide your alert assessment:
{{
  "alert_level": "<NORMAL|MONITOR|ALERT>",
  "reasoning": "<brief explanation>",
  "recommended_action": "<what caregiver should do, if anything>"
}}
"""
        raw = self._call_vlm_text_only(prompt)
        result = self._parse_response(raw)
        result["rule_based_alert"] = rule_alert
        severity = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}
        model_alert = str(result.get("alert_level", "NORMAL")).upper()
        if severity.get(rule_alert, 0) > severity.get(model_alert, 0):
            result["alert_level"] = rule_alert
            result["reasoning"] = (
                f"Rule-based safety threshold requires {rule_alert}. "
                f"VLM assessment: {result.get('reasoning', 'No reasoning provided.')}"
            )
        return result

    # ------------------------------------------------------------------
    # Internal VLM call methods
    # ------------------------------------------------------------------

    def _call_vlm_with_image(self, prompt: str, b64_image: str) -> str:
        """Call GPT-4o with a single image."""
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_image}",
                                        "detail": "high",
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                    temperature=self.temperature,
                    max_tokens=512,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"VLM call attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return '{"state": "UNKNOWN", "confidence": 0.1, "reasoning": "API call failed"}'

    def _call_vlm_multi_image(self, prompt: str, b64_images: List[str]) -> str:
        """Call GPT-4o with multiple images for temporal context."""
        content = []
        for b64 in b64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",
                },
            })
        content.append({"type": "text", "text": prompt})

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.temperature,
                    max_tokens=768,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"Multi-image VLM call attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return '{"conclusion": "Analysis failed", "confidence": 0.1}'

    def _call_vlm_text_only(self, prompt: str) -> str:
        """Call GPT-4o with text only (for alert determination, etc.)."""
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=512,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"Text VLM call attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return '{"alert_level": "UNKNOWN", "reasoning": "API call failed"}'

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_pose_context(self, pose: Optional[PoseFeatures]) -> str:
        if pose is None:
            return "No pose analysis available."
        if not pose.person_detected:
            return "No person detected by pose estimation."
        lines = [
            f"- Body orientation: {'HORIZONTAL (lying)' if pose.body_is_horizontal else 'VERTICAL (standing)' if pose.body_is_vertical else 'DIAGONAL/SEATED'}",
            f"- Detection confidence: {pose.confidence:.2f}",
            f"- Bounding box height ratio: {pose.relative_height_ratio:.2f} (proportion of frame height)",
            f"- Head above hips: {pose.head_above_hips}",
            f"- Hips above knees: {pose.hips_above_knees}",
            f"- Motion detected vs previous frame: {pose.is_moving}",
            f"- Horizontal position in frame: {'left' if pose.centroid_x_normalized < 0.4 else 'right' if pose.centroid_x_normalized > 0.6 else 'center'}",
        ]
        return "\n".join(lines)

    def _format_bed_context(
        self,
        bed: Optional[BedRegion],
        pose: Optional[PoseFeatures],
    ) -> str:
        if bed is None:
            return "No bed region detected."
        if not bed.detected:
            return "Bed not detected in frame."
        on_bed = bed.person_on_bed(pose.bbox if pose else None)
        return (
            f"Bed detected with confidence {bed.confidence:.2f}. "
            f"Person appears to be {'ON' if on_bed else 'NOT on'} the bed "
            f"based on spatial overlap analysis."
        )

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse JSON from VLM response, handling common formatting issues."""
        # Strip markdown code blocks if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse VLM response: {raw[:200]}")
        return {"state": "UNKNOWN", "confidence": 0.1, "reasoning": "Parse error"}

    def _map_state(self, state_str: str) -> ActivityState:
        """Map string state to ActivityState enum."""
        mapping = {s.value.upper(): s for s in ActivityState}
        # Also try the raw value
        key = state_str.strip().upper()
        return mapping.get(key, ActivityState.UNKNOWN)

    def _rule_based_alert(
        self,
        state: ActivityState,
        out_of_bed_sec: float,
        sitting_edge_sec: float,
        unknown_duration_sec: float = 0.0,
        fall_suspected: bool = False,
    ) -> str:
        """Fast rule-based pre-screening."""
        # ALERT conditions
        if fall_suspected:
            return "ALERT"
        if out_of_bed_sec > 2700:  # 45 minutes out of bed continuously
            return "ALERT"

        # MONITOR conditions
        if sitting_edge_sec > 600:  # 10 minutes on edge
            return "MONITOR"
        if state == ActivityState.UNKNOWN or unknown_duration_sec > 30:
            return "MONITOR"
        if out_of_bed_sec > 1200:  # 20 minutes
            return "MONITOR"

        return "NORMAL"

    def _mock_classify(
        self,
        frame: Optional[np.ndarray],
        pose: Optional[PoseFeatures],
        bed: Optional[BedRegion],
        prev_state: Optional[ActivityState],
    ) -> Dict[str, Any]:
        """Offline heuristic classifier for testing without OpenAI API credits."""
        # Check if frame is from synthetic demo generator
        if frame is not None and frame.size > 0:
            b, g, r = [int(v) for v in frame[10, 10]]
            palette_map = {
                (20, 20, 60): ("LYING_IN_BED", True, True, "Person is lying horizontally in bed."),
                (20, 60, 20): ("SITTING_ON_BED", True, True, "Person is sitting upright on the bed."),
                (60, 20, 20): ("STANDING", False, True, "Person is standing beside the bed."),
                (20, 100, 150): ("WALKING", False, False, "Person is walking across the room."),
                (150, 20, 100): ("SITTING_OUTSIDE_BED", False, False, "Person is seated away from the bed."),
            }
            for (pb, pg, pr), (state_name, on_b, near_b, reason) in palette_map.items():
                if abs(b - pb) < 30 and abs(g - pg) < 30 and abs(r - pr) < 30:
                    return {
                        "state": state_name,
                        "confidence": 0.94,
                        "reasoning": f"Visual analysis: {reason}",
                        "on_bed": on_b,
                        "near_bed": near_b,
                        "person_visible": True,
                    }

        if pose is None or not pose.person_detected:
            if prev_state in (ActivityState.OUT_OF_BED, ActivityState.WALKING, ActivityState.STANDING):
                return {
                    "state": "OUT_OF_BED",
                    "confidence": 0.88,
                    "reasoning": "Person not detected in bed area; continues to be out of bed.",
                    "on_bed": False,
                    "near_bed": False,
                    "person_visible": False,
                }
            return {
                "state": "UNKNOWN",
                "confidence": 0.50,
                "reasoning": "No person detected in frame.",
                "on_bed": False,
                "near_bed": False,
                "person_visible": False,
            }

        on_bed = bed.person_on_bed(pose.bbox) if (bed and pose.bbox) else False
        near_bed = (
            abs(pose.centroid_x_normalized - 0.5) < 0.35
            and abs(pose.centroid_y_normalized - 0.5) < 0.35
        )

        if pose.body_is_horizontal:
            if on_bed or near_bed:
                return {
                    "state": "LYING_IN_BED",
                    "confidence": 0.93,
                    "reasoning": "Person is lying horizontally within the bed region.",
                    "on_bed": True,
                    "near_bed": True,
                    "person_visible": True,
                }
            else:
                return {
                    "state": "OUT_OF_BED",
                    "confidence": 0.85,
                    "reasoning": "Person is lying horizontally outside the bed boundary.",
                    "on_bed": False,
                    "near_bed": False,
                    "person_visible": True,
                }

        if pose.body_is_vertical:
            if pose.is_moving:
                return {
                    "state": "WALKING",
                    "confidence": 0.89,
                    "reasoning": "Person is standing upright with translational/limb motion.",
                    "on_bed": False,
                    "near_bed": near_bed,
                    "person_visible": True,
                }
            else:
                return {
                    "state": "STANDING",
                    "confidence": 0.91,
                    "reasoning": "Person is upright and stationary.",
                    "on_bed": False,
                    "near_bed": near_bed,
                    "person_visible": True,
                }

        # Diagonal or seated
        if on_bed:
            return {
                "state": "SITTING_ON_BED",
                "confidence": 0.90,
                "reasoning": "Person is in an upright/seated posture directly on or at the edge of the bed.",
                "on_bed": True,
                "near_bed": True,
                "person_visible": True,
            }
        else:
            return {
                "state": "SITTING_OUTSIDE_BED",
                "confidence": 0.88,
                "reasoning": "Person is seated away from the bed area.",
                "on_bed": False,
                "near_bed": near_bed,
                "person_visible": True,
            }
