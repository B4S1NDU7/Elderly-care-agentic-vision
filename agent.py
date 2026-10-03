"""
Agentic Decision Engine
========================
Implements the agentic loop that decides when additional temporal context
is needed, calls the VLM for clarification, and makes final state decisions.

This is the "brain" of the system — it goes beyond simple per-frame classification
and reasons about sequences of observations.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from models import ActivityState, FrameAnalysis, AlertLevel, BedEventType
from vlm_analyzer import VLMAnalyzer
from state_tracker import TemporalStateTracker
from video_extractor import ExtractedFrame

# Mirror of TemporalStateTracker class constants (defined here to avoid circular refs)
IN_BED_STATES = {ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED}
OUT_STATES = {
    ActivityState.STANDING, ActivityState.WALKING,
    ActivityState.OUT_OF_BED, ActivityState.SITTING_OUTSIDE_BED,
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agentic triggers — situations where we need more context
# ---------------------------------------------------------------------------

AMBIGUOUS_TRANSITIONS = {
    # Sitting up could be adjusting position OR preparing to leave
    (ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED),
    # Standing briefly — is this a bed exit or just adjusting?
    (ActivityState.SITTING_ON_BED, ActivityState.STANDING),
    # Walking — where is the person going?
    (ActivityState.STANDING, ActivityState.WALKING),
    # Could be returning or just nearby
    (ActivityState.WALKING, ActivityState.SITTING_ON_BED),
    # Unclear
    (ActivityState.UNKNOWN, ActivityState.STANDING),
    (ActivityState.UNKNOWN, ActivityState.WALKING),
}


class AgentDecision:
    """Represents a single agentic reasoning step."""
    def __init__(
        self,
        trigger: str,
        action_taken: str,
        frames_analyzed: List[float],
        finding: str,
        conclusion: str,
        confidence: float,
    ):
        self.trigger = trigger
        self.action_taken = action_taken
        self.frames_analyzed = frames_analyzed
        self.finding = finding
        self.conclusion = conclusion
        self.confidence = confidence

    def __str__(self):
        return (
            f"Trigger: {self.trigger}\n"
            f"Action: {self.action_taken}\n"
            f"Finding: {self.finding}\n"
            f"Conclusion: {self.conclusion} (conf={self.confidence:.2f})"
        )


class AgenticEngine:
    """
    Agentic loop that:
    1. Monitors ongoing VLM frame analyses
    2. Detects ambiguous situations requiring temporal context
    3. Requests additional frame analysis from the VLM
    4. Makes final state decisions based on multi-frame evidence
    5. Determines alert levels

    Example flow:
    Observation: Person appears beside bed.
    → Trigger: Potential bed exit. Need temporal context.
    → Action: Analyze previous 3 frames + next 3 frames
    → Finding: Person was lying in bed 8s ago; now walking away
    → Conclusion: BED_EXIT confirmed (conf=0.92)
    """

    def __init__(
        self,
        vlm: VLMAnalyzer,
        tracker: TemporalStateTracker,
        all_frames: List[ExtractedFrame],
        lookback_sec: float = 15.0,
        lookahead_sec: float = 15.0,
    ):
        self.vlm = vlm
        self.tracker = tracker
        self.all_frames = all_frames
        self.lookback_sec = lookback_sec
        self.lookahead_sec = lookahead_sec
        self._decisions: List[AgentDecision] = []

    @property
    def decisions(self) -> List[AgentDecision]:
        return self._decisions

    # ------------------------------------------------------------------
    # Main agentic interface
    # ------------------------------------------------------------------

    def evaluate_transition(
        self,
        prev_analysis: Optional[FrameAnalysis],
        curr_analysis: FrameAnalysis,
        all_analyses_so_far: List[FrameAnalysis],
    ) -> FrameAnalysis:
        """
        Evaluate a potential state transition and decide if more context is needed.
        Returns (possibly updated) current analysis.
        """
        if prev_analysis is None:
            return curr_analysis

        prev_state = prev_analysis.state
        curr_state = curr_analysis.state
        transition = (prev_state, curr_state)

        # Check if this is an ambiguous transition
        if transition in AMBIGUOUS_TRANSITIONS or curr_state == ActivityState.UNKNOWN:
            return self._resolve_ambiguity(
                curr_analysis, all_analyses_so_far, transition
            )

        # Check confidence — low confidence warrants investigation
        if curr_analysis.confidence < 0.55:
            return self._resolve_low_confidence(curr_analysis, all_analyses_so_far)

        return curr_analysis

    def evaluate_bed_exit(
        self,
        potential_exit_analysis: FrameAnalysis,
        all_analyses: List[FrameAnalysis],
    ) -> Tuple[Optional[bool], float, str]:
        """
        Agentic bed exit verification.
        Returns (is_confirmed, confidence, reasoning).
        """
        ts = potential_exit_analysis.timestamp_sec

        # Gather context frames
        context_frames, context_analyses = self._gather_context(
            ts, lookback=self.lookback_sec, lookahead=self.lookahead_sec,
            all_analyses=all_analyses,
        )

        if not context_frames:
            return None, 0.0, "Insufficient context frames to verify a bed exit."

        # Build observation list
        observations = [
            f"{a.state.value} (conf={a.confidence:.2f}): {a.reasoning[:80]}"
            for a in context_analyses
        ]

        question = (
            "Based on the temporal sequence, is this a genuine bed exit event "
            "(person actually left the bed) or just a position adjustment "
            "(sitting up, turning over, etc.)? Provide your determination."
        )

        result = self.vlm.contextual_analysis(
            question=question,
            frames=[f.image for f in context_frames],
            timestamps=[f.timestamp_sec for f in context_frames],
            observations=observations,
        )

        conclusion = result.get("conclusion", "")
        confidence = float(result.get("confidence", 0.7))
        reasoning = result.get("reasoning", "")

        explicit_decision = result.get("is_bed_exit")
        if isinstance(explicit_decision, bool):
            is_exit: Optional[bool] = explicit_decision
        else:
            text_lower = (conclusion + " " + reasoning).lower()
            negative_evidence = (
                "not genuine",
                "not a genuine",
                "not confirmed",
                "not an exit",
                "not a bed exit",
                "no bed exit",
                "no genuine exit",
                "did not leave",
                "did not exit",
                "remains in bed",
                "position adjustment",
                "adjusting position",
                "remained on the bed",
                "returned to bed",
            )
            positive_evidence = ("genuine", "confirmed", "bed exit")
            if any(phrase in text_lower for phrase in negative_evidence):
                is_exit = False
            elif any(phrase in text_lower for phrase in positive_evidence):
                is_exit = True
            else:
                is_exit = None

        decision = AgentDecision(
            trigger="Potential bed exit detected",
            action_taken="Analyzed surrounding temporal context with VLM",
            frames_analyzed=[f.timestamp_sec for f in context_frames],
            finding=reasoning,
            conclusion=f"Bed exit {'CONFIRMED' if is_exit else 'REJECTED'}: {conclusion}",
            confidence=confidence,
        )
        self._decisions.append(decision)
        logger.info(f"Agentic bed exit eval: {'CONFIRMED' if is_exit else 'REJECTED'} (conf={confidence:.2f})")

        return is_exit, confidence, reasoning

    def evaluate_lying_on_floor(
        self,
        analysis: FrameAnalysis,
        all_analyses: List[FrameAnalysis],
    ) -> Tuple[bool, str]:
        """
        Check if a horizontal body is on the bed or on the floor (fall risk).
        Returns (is_on_bed, reasoning).
        """
        ts = analysis.timestamp_sec
        context_frames, _ = self._gather_context(
            ts, lookback=5.0, lookahead=5.0, all_analyses=all_analyses
        )

        if not context_frames:
            return True, "Insufficient context to determine floor vs. bed."

        question = (
            "The person appears to be lying horizontally. "
            "Carefully examine the frame: is the person lying on the BED "
            "(normal sleeping position) or on the FLOOR (potential fall)? "
            "Look at the bed position, floor texture, and body height."
        )
        observations = [f"Frame at {ts:.0f}s: {analysis.reasoning}"]

        result = self.vlm.contextual_analysis(
            question=question,
            frames=[f.image for f in context_frames[:2]],
            timestamps=[f.timestamp_sec for f in context_frames[:2]],
            observations=observations,
        )

        reasoning = result.get("reasoning", result.get("conclusion", ""))
        text = reasoning.lower()
        on_bed = "floor" not in text or "bed" in text

        decision = AgentDecision(
            trigger="Person lying horizontally — floor vs bed check",
            action_taken="Contextual frame analysis for floor/bed determination",
            frames_analyzed=[f.timestamp_sec for f in context_frames],
            finding=reasoning,
            conclusion=f"Person determined to be on {'BED' if on_bed else 'FLOOR'}",
            confidence=float(result.get("confidence", 0.7)),
        )
        self._decisions.append(decision)
        return on_bed, reasoning

    def determine_final_alert(
        self,
        final_state: ActivityState,
        out_of_bed_sec: float,
        bed_exit_count: int,
        sitting_on_edge_sec: float,
        recent_timeline: List[Dict],
        unknown_duration_sec: float = 0.0,
        fall_suspected: bool = False,
    ) -> Tuple[AlertLevel, str]:
        """
        Determine final alert level using VLM + rules.
        """
        result = self.vlm.determine_alert_level(
            recent_timeline=recent_timeline,
            current_state=final_state,
            out_of_bed_duration_sec=out_of_bed_sec,
            bed_exit_count=bed_exit_count,
            sitting_on_edge_duration_sec=sitting_on_edge_sec,
            unknown_duration_sec=unknown_duration_sec,
            fall_suspected=fall_suspected,
        )

        alert_str = result.get("alert_level", "NORMAL")
        reasoning = result.get("reasoning", "Routine activity, no concerns.")

        try:
            alert = AlertLevel(alert_str)
        except ValueError:
            alert = AlertLevel.NORMAL

        return alert, reasoning

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_ambiguity(
        self,
        curr_analysis: FrameAnalysis,
        all_analyses: List[FrameAnalysis],
        transition: Tuple,
    ) -> FrameAnalysis:
        """Use temporal context to resolve an ambiguous state transition."""
        ts = curr_analysis.timestamp_sec
        context_frames, context_analyses = self._gather_context(
            ts, lookback=self.lookback_sec, lookahead=0.0,
            all_analyses=all_analyses,
        )

        if not context_frames:
            return curr_analysis

        observations = [
            f"{a.state.value} (conf={a.confidence:.2f}): {a.reasoning[:80]}"
            for a in context_analyses
        ]
        question = (
            f"The system detected a potential transition: "
            f"{transition[0].value} → {transition[1].value}. "
            f"Based on the temporal sequence of observations, what is the "
            f"most likely current activity state? Consider whether this is "
            f"a genuine transition or a temporary fluctuation."
        )

        result = self.vlm.contextual_analysis(
            question=question,
            frames=[f.image for f in context_frames],
            timestamps=[f.timestamp_sec for f in context_frames],
            observations=observations,
        )

        # Update analysis if we got a better state
        new_state_str = result.get("state", curr_analysis.state.value)
        try:
            new_state = ActivityState(new_state_str.lower())
        except ValueError:
            new_state = curr_analysis.state

        new_conf = float(result.get("confidence", curr_analysis.confidence))
        new_reasoning = result.get("reasoning", curr_analysis.reasoning)

        if new_conf > curr_analysis.confidence:
            curr_analysis.state = new_state
            curr_analysis.confidence = new_conf
            curr_analysis.reasoning = f"[Agentic] {new_reasoning}"

            decision = AgentDecision(
                trigger=f"Ambiguous transition: {transition[0].value} → {transition[1].value}",
                action_taken=f"Analyzed {len(context_frames)} context frames",
                frames_analyzed=[f.timestamp_sec for f in context_frames],
                finding=new_reasoning,
                conclusion=f"State determined: {new_state.value} (conf={new_conf:.2f})",
                confidence=new_conf,
            )
            self._decisions.append(decision)

        return curr_analysis

    def _resolve_low_confidence(
        self,
        curr_analysis: FrameAnalysis,
        all_analyses: List[FrameAnalysis],
    ) -> FrameAnalysis:
        """Try to resolve a low-confidence classification using surrounding frames."""
        ts = curr_analysis.timestamp_sec
        context_frames, context_analyses = self._gather_context(
            ts, lookback=10.0, lookahead=0.0, all_analyses=all_analyses
        )

        if len(context_analyses) < 2:
            return curr_analysis

        # Simple majority vote from recent high-confidence frames
        recent_high_conf = [
            a for a in context_analyses[-4:]
            if a.confidence >= 0.65 and a.state != ActivityState.UNKNOWN
        ]
        if recent_high_conf:
            # Use the most common state among recent high-confidence frames
            from collections import Counter
            vote = Counter(a.state for a in recent_high_conf)
            best_state, count = vote.most_common(1)[0]
            if count >= 2:
                curr_analysis.state = best_state
                curr_analysis.confidence = 0.7
                curr_analysis.reasoning = (
                    f"[Agentic] Low confidence resolved by majority vote "
                    f"from {count} recent frames: {best_state.value}"
                )

        return curr_analysis

    def _gather_context(
        self,
        center_ts: float,
        lookback: float,
        lookahead: float,
        all_analyses: List[FrameAnalysis],
    ) -> Tuple[List[ExtractedFrame], List[FrameAnalysis]]:
        """Gather frames and analyses within a time window."""
        start = center_ts - lookback
        end = center_ts + lookahead

        # Filter frames
        context_frames = [
            f for f in self.all_frames
            if start <= f.timestamp_sec <= end
        ]
        context_analyses = [
            a for a in all_analyses
            if start <= a.timestamp_sec <= end
        ]

        # Limit to 6 frames max for API cost control
        if len(context_frames) > 6:
            step = len(context_frames) // 6
            context_frames = context_frames[::step][:6]

        return context_frames, context_analyses

    def format_decisions_report(self) -> str:
        """Format all agent decisions as a readable report."""
        if not self._decisions:
            return "No agentic interventions were needed."
        lines = ["AGENTIC REASONING LOG", "=" * 50]
        for i, d in enumerate(self._decisions, 1):
            lines.append(f"\n[{i}] {d}")
            lines.append("-" * 40)
        return "\n".join(lines)
