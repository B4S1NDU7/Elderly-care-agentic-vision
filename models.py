"""
Elderly Care Agentic Vision System
====================================
Core data models and state definitions.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# State Definitions
# ---------------------------------------------------------------------------

class ActivityState(str, Enum):
    LYING_IN_BED = "lying_in_bed"
    SITTING_ON_BED = "sitting_on_bed"
    SITTING_OUTSIDE_BED = "sitting_outside_bed"
    STANDING = "standing"
    WALKING = "walking"
    OUT_OF_BED = "out_of_bed"
    UNKNOWN = "unknown"


class AlertLevel(str, Enum):
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    ALERT = "ALERT"


class BedEventType(str, Enum):
    BED_EXIT = "bed_exit"
    BED_RETURN = "bed_return"


# ---------------------------------------------------------------------------
# Frame-level analysis result
# ---------------------------------------------------------------------------

@dataclass
class FrameAnalysis:
    """Result of analyzing a single video frame."""
    frame_index: int
    timestamp_sec: float
    state: ActivityState
    confidence: float
    reasoning: str
    pose_features: Dict[str, Any] = field(default_factory=dict)
    detection_boxes: List[Dict] = field(default_factory=list)
    near_bed: bool = False
    on_bed: bool = False
    raw_vlm_response: str = ""

    @property
    def timestamp_str(self) -> str:
        m = int(self.timestamp_sec // 60)
        s = int(self.timestamp_sec % 60)
        return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# State segment (merged consecutive same-state frames)
# ---------------------------------------------------------------------------

@dataclass
class StateSegment:
    """A contiguous period of the same activity state."""
    state: ActivityState
    start_sec: float
    end_sec: float
    confidence: float
    frame_indices: List[int] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def start_str(self) -> str:
        return _sec_to_mmss(self.start_sec)

    @property
    def end_str(self) -> str:
        return _sec_to_mmss(self.end_sec)

    @property
    def duration_str(self) -> str:
        return _sec_to_mmss(self.duration_sec)


# ---------------------------------------------------------------------------
# Bed events
# ---------------------------------------------------------------------------

@dataclass
class BedEvent:
    """A confirmed bed-exit or bed-return event."""
    event_type: BedEventType
    start_time_sec: float
    confirmed_time_sec: float
    previous_state: ActivityState
    current_state: ActivityState
    confidence: float
    decision: AlertLevel
    context_reasoning: str = ""

    def to_dict(self) -> Dict:
        return {
            "event": self.event_type.value,
            "start_time": _sec_to_hhmmss(self.start_time_sec),
            "confirmed_time": _sec_to_hhmmss(self.confirmed_time_sec),
            "previous_state": self.previous_state.value,
            "current_state": self.current_state.value,
            "confidence": round(self.confidence, 2),
            "decision": self.decision.value,
            "context_reasoning": self.context_reasoning,
        }


# ---------------------------------------------------------------------------
# Full analysis report
# ---------------------------------------------------------------------------

@dataclass
class AnalysisReport:
    """Complete analysis report for a video."""
    video_path: str
    observation_duration_sec: float
    activity_duration_sec: Dict[str, float]
    timeline: List[StateSegment]
    bed_events: List[BedEvent]
    final_state: ActivityState
    final_alert: AlertLevel
    alert_reasoning: str
    processing_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def bed_exit_count(self) -> int:
        return sum(1 for e in self.bed_events if e.event_type == BedEventType.BED_EXIT)

    @property
    def bed_return_count(self) -> int:
        return sum(1 for e in self.bed_events if e.event_type == BedEventType.BED_RETURN)

    @property
    def total_in_bed_sec(self) -> float:
        in_bed_states = {ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED}
        return sum(
            self.activity_duration_sec.get(s.value, 0)
            for s in in_bed_states
        )

    @property
    def total_out_of_bed_sec(self) -> float:
        out_states = {
            ActivityState.SITTING_OUTSIDE_BED,
            ActivityState.STANDING,
            ActivityState.WALKING,
            ActivityState.OUT_OF_BED,
        }
        return sum(
            self.activity_duration_sec.get(s.value, 0)
            for s in out_states
        )

    @property
    def longest_out_of_bed_period_sec(self) -> float:
        """Find the longest confirmed absence, including brief unknown gaps."""
        max_period = 0.0
        active_exit_start = None
        for event in sorted(self.bed_events, key=lambda item: item.confirmed_time_sec):
            if event.event_type == BedEventType.BED_EXIT:
                if active_exit_start is None:
                    active_exit_start = event.start_time_sec
            elif (
                event.event_type == BedEventType.BED_RETURN
                and active_exit_start is not None
            ):
                max_period = max(
                    max_period,
                    event.confirmed_time_sec - active_exit_start,
                )
                active_exit_start = None

        if active_exit_start is not None:
            max_period = max(
                max_period,
                self.observation_duration_sec - active_exit_start,
            )
        return max_period

    def to_summary_dict(self) -> Dict:
        return {
            "observation_duration_sec": round(self.observation_duration_sec, 1),
            "activity_duration_sec": {
                k: round(v, 1) for k, v in self.activity_duration_sec.items()
            },
            "bed_exit_count": self.bed_exit_count,
            "bed_return_count": self.bed_return_count,
            "total_in_bed_sec": round(self.total_in_bed_sec, 1),
            "total_out_of_bed_sec": round(self.total_out_of_bed_sec, 1),
            "longest_out_of_bed_period_sec": round(self.longest_out_of_bed_period_sec, 1),
            "final_state": self.final_state.value,
            "final_alert": self.final_alert.value,
            "alert_reasoning": self.alert_reasoning,
        }

    def get_activity_summary_human(self) -> Dict[str, str]:
        return {k: _sec_to_human(v) for k, v in self.activity_duration_sec.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sec_to_mmss(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


def _sec_to_hhmmss(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _sec_to_human(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    if m == 0:
        return f"{s}s"
    return f"{m}m {s:02d}s"
