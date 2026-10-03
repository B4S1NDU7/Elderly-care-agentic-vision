"""
State Machine & Temporal Tracker
==================================
Manages state transitions, temporal smoothing, and segment building.

Key responsibilities:
- Temporal smoothing (prevent noisy single-frame state changes)
- Transition validation (physically plausible sequences only)
- Segment merging (consecutive same-state frames → segments)
- Duration accumulation
"""

import logging
from typing import List, Optional, Dict, Tuple, Deque
from collections import deque, defaultdict

from models import (
    ActivityState,
    FrameAnalysis,
    StateSegment,
    BedEvent,
    BedEventType,
    AlertLevel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid transition graph (what state transitions are physically plausible)
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: Dict[ActivityState, List[ActivityState]] = {
    ActivityState.LYING_IN_BED: [
        ActivityState.SITTING_ON_BED,
        ActivityState.UNKNOWN,
        ActivityState.LYING_IN_BED,
    ],
    ActivityState.SITTING_ON_BED: [
        ActivityState.LYING_IN_BED,
        ActivityState.STANDING,
        ActivityState.SITTING_ON_BED,
        ActivityState.SITTING_OUTSIDE_BED,  # edge case
        ActivityState.UNKNOWN,
    ],
    ActivityState.STANDING: [
        ActivityState.SITTING_ON_BED,
        ActivityState.WALKING,
        ActivityState.SITTING_OUTSIDE_BED,
        ActivityState.OUT_OF_BED,
        ActivityState.LYING_IN_BED,  # falling asleep
        ActivityState.STANDING,
        ActivityState.UNKNOWN,
    ],
    ActivityState.WALKING: [
        ActivityState.STANDING,
        ActivityState.OUT_OF_BED,
        ActivityState.SITTING_OUTSIDE_BED,
        ActivityState.WALKING,
        ActivityState.UNKNOWN,
        ActivityState.SITTING_ON_BED,  # returning
    ],
    ActivityState.OUT_OF_BED: [
        ActivityState.WALKING,
        ActivityState.STANDING,
        ActivityState.SITTING_OUTSIDE_BED,
        ActivityState.OUT_OF_BED,
        ActivityState.UNKNOWN,
        ActivityState.SITTING_ON_BED,  # returning
    ],
    ActivityState.SITTING_OUTSIDE_BED: [
        ActivityState.STANDING,
        ActivityState.WALKING,
        ActivityState.SITTING_OUTSIDE_BED,
        ActivityState.OUT_OF_BED,
        ActivityState.UNKNOWN,
    ],
    ActivityState.UNKNOWN: [s for s in ActivityState],  # UNKNOWN can go anywhere
}

# Minimum frames a state must persist before it's accepted (smoothing)
MIN_STABLE_FRAMES: Dict[ActivityState, int] = {
    ActivityState.LYING_IN_BED: 2,
    ActivityState.SITTING_ON_BED: 2,
    ActivityState.SITTING_OUTSIDE_BED: 2,
    ActivityState.STANDING: 1,
    ActivityState.WALKING: 1,
    ActivityState.OUT_OF_BED: 2,
    ActivityState.UNKNOWN: 1,
}

# Minimum duration (seconds) before a segment is committed
MIN_SEGMENT_DURATION_SEC = 3.0

# Thresholds for bed event detection
BED_EXIT_TRANSITION_WINDOW_SEC = 30.0   # look back this far to confirm exit
BED_RETURN_TRANSITION_WINDOW_SEC = 30.0


class TemporalStateTracker:
    """
    Manages state history and produces a smooth, transition-aware timeline.

    Algorithm:
    1. Each new frame analysis is added to a pending buffer
    2. When enough frames confirm a new state, commit the transition
    3. Merge adjacent same-state segments
    4. Detect bed events from confirmed transition sequences
    """

    def __init__(
        self,
        smoothing_window: int = 3,
        min_segment_sec: float = MIN_SEGMENT_DURATION_SEC,
    ):
        self.smoothing_window = smoothing_window
        self.min_segment_sec = min_segment_sec

        # Raw frame analyses
        self._frame_analyses: List[FrameAnalysis] = []

        # Current confirmed state
        self._current_state: Optional[ActivityState] = None
        self._current_state_start_sec: float = 0.0

        # Pending state buffer (for smoothing)
        self._pending_buffer: Deque[FrameAnalysis] = deque(maxlen=smoothing_window + 2)

        # Committed segments
        self._segments: List[StateSegment] = []

        # Duration accumulator
        self._duration_acc: Dict[str, float] = defaultdict(float)

        # Bed event tracking
        self._bed_events: List[BedEvent] = []
        self._last_in_bed_state: Optional[ActivityState] = None
        self._last_in_bed_time: float = 0.0
        self._consecutive_out_of_bed_start: Optional[float] = None
        self._sitting_on_edge_start: Optional[float] = None
        self._sitting_on_edge_duration: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame(self, analysis: FrameAnalysis):
        """Add a new frame analysis to the tracker."""
        self._frame_analyses.append(analysis)
        self._pending_buffer.append(analysis)

        # Try to commit a state transition
        self._process_pending()

    def finalize(self, video_duration_sec: float):
        """
        Called after all frames are processed.
        Closes the last open segment and fills any gaps.
        """
        if self._current_state is not None:
            # Close the last segment
            self._close_segment(video_duration_sec)

        # Fill any uncovered time with UNKNOWN
        self._fill_gaps(video_duration_sec)

        # Recompute durations from finalized segments
        self._recompute_durations()

        logger.info(
            f"Finalized: {len(self._segments)} segments, "
            f"{len(self._bed_events)} bed events"
        )

    @property
    def segments(self) -> List[StateSegment]:
        return self._segments

    @property
    def bed_events(self) -> List[BedEvent]:
        return self._bed_events

    @property
    def duration_map(self) -> Dict[str, float]:
        return dict(self._duration_acc)

    @property
    def current_state(self) -> Optional[ActivityState]:
        return self._current_state

    @property
    def sitting_on_edge_duration_sec(self) -> float:
        return self._sitting_on_edge_duration

    # ------------------------------------------------------------------
    # State transition processing
    # ------------------------------------------------------------------

    def _process_pending(self):
        """
        Apply temporal smoothing and decide whether to commit a state change.

        Strategy:
        - Count votes for each state in the buffer
        - The winning state must have majority and meet minimum frame count
        - If winning state != current state, validate transition then commit
        """
        if len(self._pending_buffer) < 2:
            return

        # Vote count for each state in buffer
        votes: Dict[ActivityState, float] = defaultdict(float)
        for fa in self._pending_buffer:
            votes[fa.state] += fa.confidence

        # Winning state (by weighted confidence)
        winning_state = max(votes, key=votes.__getitem__)
        winning_count = sum(
            1 for fa in self._pending_buffer if fa.state == winning_state
        )
        min_required = MIN_STABLE_FRAMES.get(winning_state, 2)

        if winning_count < min_required:
            return  # not enough frames to commit

        if winning_state == self._current_state:
            return  # no change

        # Validate transition
        if self._current_state is not None:
            valid = VALID_TRANSITIONS.get(self._current_state, [])
            if winning_state not in valid:
                logger.debug(
                    f"Ignoring implausible transition: "
                    f"{self._current_state} → {winning_state}"
                )
                # Allow it if we're very confident (>0.85) - handles edge cases
                avg_conf = votes[winning_state] / winning_count
                if avg_conf < 0.85:
                    return

        # Commit transition
        latest = self._pending_buffer[-1]
        self._commit_transition(winning_state, latest.timestamp_sec, latest.frame_index)

    def _commit_transition(
        self,
        new_state: ActivityState,
        timestamp: float,
        frame_index: int,
    ):
        """Commit a state transition and start a new segment."""
        if self._current_state is not None:
            self._close_segment(timestamp)

        logger.debug(
            f"State transition: {self._current_state} → {new_state} "
            f"@ {timestamp:.1f}s"
        )

        # Update bed-related tracking
        self._update_bed_tracking(new_state, timestamp)

        self._current_state = new_state
        self._current_state_start_sec = timestamp

    def _close_segment(self, end_time: float):
        """Close the current segment."""
        if self._current_state is None:
            return

        duration = end_time - self._current_state_start_sec
        if duration < 0.5:
            return  # skip tiny segments

        frame_idxs = [
            fa.frame_index
            for fa in self._frame_analyses
            if self._current_state_start_sec <= fa.timestamp_sec <= end_time
        ]
        avg_conf = (
            sum(fa.confidence for fa in self._frame_analyses
                if self._current_state_start_sec <= fa.timestamp_sec <= end_time)
            / max(len(frame_idxs), 1)
        )

        seg = StateSegment(
            state=self._current_state,
            start_sec=self._current_state_start_sec,
            end_sec=end_time,
            confidence=avg_conf,
            frame_indices=frame_idxs,
        )
        self._segments.append(seg)

    def _fill_gaps(self, video_duration_sec: float):
        """
        Ensure the timeline covers 0 to video_duration_sec completely.
        Fill any gaps with UNKNOWN segments.
        """
        if not self._segments:
            if video_duration_sec > 0:
                self._segments.append(StateSegment(
                    state=ActivityState.UNKNOWN,
                    start_sec=0.0,
                    end_sec=video_duration_sec,
                    confidence=0.0,
                ))
            return

        # Sort segments by start time
        self._segments.sort(key=lambda s: s.start_sec)

        # Fill gap at beginning
        if self._segments[0].start_sec > 1.0:
            self._segments.insert(0, StateSegment(
                state=ActivityState.UNKNOWN,
                start_sec=0.0,
                end_sec=self._segments[0].start_sec,
                confidence=0.0,
            ))

        # Fill gaps between segments
        filled = [self._segments[0]]
        for seg in self._segments[1:]:
            prev = filled[-1]
            gap = seg.start_sec - prev.end_sec
            if gap > 1.0:
                filled.append(StateSegment(
                    state=ActivityState.UNKNOWN,
                    start_sec=prev.end_sec,
                    end_sec=seg.start_sec,
                    confidence=0.0,
                ))
            filled.append(seg)

        # Fill gap at end
        last = filled[-1]
        if last.end_sec < video_duration_sec - 1.0:
            filled.append(StateSegment(
                state=ActivityState.UNKNOWN,
                start_sec=last.end_sec,
                end_sec=video_duration_sec,
                confidence=0.0,
            ))

        self._segments = filled

    def _recompute_durations(self):
        """Recompute duration map from finalized segments."""
        self._duration_acc = defaultdict(float)
        for seg in self._segments:
            self._duration_acc[seg.state.value] += seg.duration_sec

        # Ensure all states present (even if 0)
        for state in ActivityState:
            if state.value not in self._duration_acc:
                self._duration_acc[state.value] = 0.0

    # ------------------------------------------------------------------
    # Bed event detection
    # ------------------------------------------------------------------

    IN_BED_STATES = {ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED}
    OUT_STATES = {
        ActivityState.STANDING, ActivityState.WALKING,
        ActivityState.OUT_OF_BED, ActivityState.SITTING_OUTSIDE_BED,
    }

    def _update_bed_tracking(self, new_state: ActivityState, timestamp: float):
        """Update bed event tracking on each state transition."""
        old_state = self._current_state

        # Track in-bed history
        if new_state in self.IN_BED_STATES:
            self._last_in_bed_state = new_state
            self._last_in_bed_time = timestamp

        # Sitting on bed edge tracking (MONITOR trigger)
        if new_state == ActivityState.SITTING_ON_BED:
            if self._sitting_on_edge_start is None:
                self._sitting_on_edge_start = timestamp
        else:
            if self._sitting_on_edge_start is not None:
                self._sitting_on_edge_duration += timestamp - self._sitting_on_edge_start
                self._sitting_on_edge_start = None

        # Bed EXIT detection
        # Trigger: was in bed → now out of bed (via STANDING or WALKING)
        if (
            old_state in self.IN_BED_STATES
            and new_state in {ActivityState.STANDING, ActivityState.WALKING}
        ):
            self._check_and_emit_bed_exit(old_state, new_state, timestamp)

        # Direct exit (e.g., SITTING_ON_BED → OUT_OF_BED in one jump)
        elif (
            old_state in self.IN_BED_STATES
            and new_state in {ActivityState.OUT_OF_BED, ActivityState.SITTING_OUTSIDE_BED}
        ):
            self._check_and_emit_bed_exit(old_state, new_state, timestamp)

        # Also emit when transition completes (STANDING → WALKING → OUT_OF_BED)
        elif (
            old_state == ActivityState.STANDING
            and new_state in {ActivityState.WALKING, ActivityState.OUT_OF_BED}
            and self._consecutive_out_of_bed_start is None
        ):
            # Look back to see if this follows a bed-related state
            recent_in_bed = self._was_recently_in_bed(timestamp, window_sec=60)
            if recent_in_bed:
                self._emit_bed_exit(
                    start_time=self._last_in_bed_time,
                    confirmed_time=timestamp,
                    prev_state=self._last_in_bed_state or old_state,
                    current_state=new_state,
                )

        # Bed RETURN detection
        # Trigger: was out of bed → now sitting/lying in bed
        if (
            old_state in self.OUT_STATES
            and new_state in self.IN_BED_STATES
            and self._consecutive_out_of_bed_start is not None
        ):
            self._emit_bed_return(
                start_time=self._consecutive_out_of_bed_start,
                confirmed_time=timestamp,
                prev_state=old_state,
                current_state=new_state,
            )

        # Track continuous out-of-bed periods
        if new_state in self.OUT_STATES:
            if self._consecutive_out_of_bed_start is None:
                self._consecutive_out_of_bed_start = timestamp
        elif new_state in self.IN_BED_STATES:
            self._consecutive_out_of_bed_start = None

    def _check_and_emit_bed_exit(
        self,
        prev_state: ActivityState,
        current_state: ActivityState,
        timestamp: float,
    ):
        """Emit a bed exit event (with confidence check)."""
        # Do not emit if this is just a brief position adjustment
        # Heuristic: the person must have been in bed for > 30 seconds
        time_in_bed = timestamp - self._last_in_bed_time
        if time_in_bed < 5:  # ignore if in bed for less than 5 seconds
            return

        self._emit_bed_exit(
            start_time=self._last_in_bed_time,
            confirmed_time=timestamp,
            prev_state=prev_state,
            current_state=current_state,
        )

    def _emit_bed_exit(
        self,
        start_time: float,
        confirmed_time: float,
        prev_state: ActivityState,
        current_state: ActivityState,
    ):
        """Create and record a BED_EXIT event."""
        # Set consecutive out-of-bed start
        if self._consecutive_out_of_bed_start is None:
            self._consecutive_out_of_bed_start = confirmed_time

        # Confidence based on how clear the transition is
        confidence = 0.85
        if current_state == ActivityState.WALKING:
            confidence = 0.92
        elif current_state == ActivityState.OUT_OF_BED:
            confidence = 0.90

        event = BedEvent(
            event_type=BedEventType.BED_EXIT,
            start_time_sec=start_time,
            confirmed_time_sec=confirmed_time,
            previous_state=prev_state,
            current_state=current_state,
            confidence=confidence,
            decision=AlertLevel.MONITOR,  # Bed exits are worth monitoring
            context_reasoning=(
                f"Person transitioned from {prev_state.value} → {current_state.value} "
                f"at {confirmed_time:.0f}s. This confirms a bed exit."
            ),
        )
        self._bed_events.append(event)
        logger.info(
            f"BED_EXIT detected @ {confirmed_time:.1f}s "
            f"({prev_state.value} → {current_state.value})"
        )

    def _emit_bed_return(
        self,
        start_time: float,
        confirmed_time: float,
        prev_state: ActivityState,
        current_state: ActivityState,
    ):
        """Create and record a BED_RETURN event."""
        out_duration = confirmed_time - start_time

        # Determine alert level based on how long they were out
        if out_duration > 2700:  # > 45 min
            alert = AlertLevel.ALERT
        elif out_duration > 1200:  # > 20 min
            alert = AlertLevel.MONITOR
        else:
            alert = AlertLevel.NORMAL

        event = BedEvent(
            event_type=BedEventType.BED_RETURN,
            start_time_sec=start_time,
            confirmed_time_sec=confirmed_time,
            previous_state=prev_state,
            current_state=current_state,
            confidence=0.88,
            decision=alert,
            context_reasoning=(
                f"Person returned to bed after {out_duration:.0f}s out of bed. "
                f"Transition: {prev_state.value} → {current_state.value}."
            ),
        )
        self._bed_events.append(event)
        logger.info(
            f"BED_RETURN detected @ {confirmed_time:.1f}s "
            f"(was out for {out_duration:.0f}s)"
        )

    def _was_recently_in_bed(self, timestamp: float, window_sec: float) -> bool:
        """Check if the person was in bed within the last window_sec seconds."""
        if self._last_in_bed_time is None:
            return False
        return (timestamp - self._last_in_bed_time) <= window_sec

    def get_summary_stats(self) -> Dict:
        """Return summary statistics for reporting."""
        total_in_bed = sum(
            self._duration_acc.get(s.value, 0)
            for s in self.IN_BED_STATES
        )
        total_out = sum(
            self._duration_acc.get(s.value, 0)
            for s in self.OUT_STATES
        )
        return {
            "total_in_bed_sec": total_in_bed,
            "total_out_of_bed_sec": total_out,
            "bed_exit_count": sum(
                1 for e in self._bed_events if e.event_type == BedEventType.BED_EXIT
            ),
            "bed_return_count": sum(
                1 for e in self._bed_events if e.event_type == BedEventType.BED_RETURN
            ),
            "sitting_on_edge_duration_sec": self._sitting_on_edge_duration,
        }
