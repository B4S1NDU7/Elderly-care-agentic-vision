import unittest
import numpy as np

from agent import AgenticEngine
from models import ActivityState, BedEventType, FrameAnalysis
from state_tracker import TemporalStateTracker
from vlm_analyzer import VLMAnalyzer
from video_extractor import ExtractedFrame


def frame(index, state):
    return FrameAnalysis(
        frame_index=index,
        timestamp_sec=float(index * 2),
        state=state,
        confidence=1.0,
        reasoning="test observation",
    )


class TemporalStateTrackerTests(unittest.TestCase):
    def track(self, states, duration=40):
        tracker = TemporalStateTracker(smoothing_window=1)
        for index, state in enumerate(states):
            tracker.add_frame(frame(index, state))
        tracker.finalize(duration)
        return tracker

    def test_brief_standing_then_return_does_not_emit_bed_exit(self):
        states = (
            [ActivityState.LYING_IN_BED] * 3
            + [ActivityState.STANDING] * 3
            + [ActivityState.LYING_IN_BED] * 3
        )

        tracker = self.track(states)

        self.assertEqual(tracker.bed_events, [])

    def test_exit_requires_movement_and_return_is_counted_once(self):
        states = (
            [ActivityState.LYING_IN_BED] * 3
            + [ActivityState.STANDING] * 3
            + [ActivityState.LYING_IN_BED] * 3
            + [ActivityState.STANDING] * 3
            + [ActivityState.WALKING] * 3
            + [ActivityState.SITTING_ON_BED] * 3
        )

        tracker = self.track(states)
        event_types = [event.event_type for event in tracker.bed_events]

        self.assertEqual(
            event_types,
            [BedEventType.BED_EXIT, BedEventType.BED_RETURN],
        )
        self.assertEqual(tracker.bed_events[0].confirmed_time_sec, 24.0)
        self.assertEqual(tracker.bed_events[1].confirmed_time_sec, 30.0)
        self.assertEqual(tracker.get_summary_stats()["longest_out_of_bed_period_sec"], 12.0)

    def test_exit_candidate_can_be_confirmed_after_a_long_stand(self):
        states = (
            [ActivityState.LYING_IN_BED] * 3
            + [ActivityState.STANDING] * 25
            + [ActivityState.WALKING] * 3
        )

        tracker = self.track(states, duration=70)

        self.assertEqual(
            sum(event.event_type == BedEventType.BED_EXIT for event in tracker.bed_events),
            1,
        )

    def test_current_absence_is_included_in_longest_duration(self):
        states = (
            [ActivityState.LYING_IN_BED] * 3
            + [ActivityState.STANDING] * 3
            + [ActivityState.WALKING] * 3
        )

        tracker = self.track(states, duration=40)

        self.assertEqual(
            tracker.get_summary_stats()["longest_out_of_bed_period_sec"],
            34.0,
        )

    def test_finalized_timeline_covers_video_duration(self):
        tracker = self.track([ActivityState.UNKNOWN] * 5, duration=40)

        self.assertAlmostEqual(
            sum(segment.duration_sec for segment in tracker.segments),
            40.0,
        )
        self.assertEqual(tracker.duration_map["unknown"], 40.0)

    def test_alert_thresholds_remain_active_without_agent(self):
        analyzer = VLMAnalyzer(api_key="mock")

        result = analyzer.determine_alert_level(
            recent_timeline=[],
            current_state=ActivityState.OUT_OF_BED,
            out_of_bed_duration_sec=2701,
            bed_exit_count=1,
            sitting_on_edge_duration_sec=0,
        )

        self.assertEqual(result["alert_level"], "ALERT")

    def test_vlm_cannot_downgrade_rule_based_alert(self):
        analyzer = VLMAnalyzer(api_key="mock")
        analyzer.mock_mode = False
        analyzer._call_vlm_text_only = lambda prompt: (
            '{"alert_level":"NORMAL","reasoning":"Routine activity."}'
        )

        result = analyzer.determine_alert_level(
            recent_timeline=[],
            current_state=ActivityState.OUT_OF_BED,
            out_of_bed_duration_sec=2701,
            bed_exit_count=1,
            sitting_on_edge_duration_sec=0,
        )

        self.assertEqual(result["alert_level"], "ALERT")
        self.assertIn("Rule-based safety threshold", result["reasoning"])

    def test_long_unknown_and_agentic_floor_risk_raise_alerts(self):
        analyzer = VLMAnalyzer(api_key="mock")
        unknown_result = analyzer.determine_alert_level(
            recent_timeline=[],
            current_state=ActivityState.WALKING,
            out_of_bed_duration_sec=0,
            bed_exit_count=0,
            sitting_on_edge_duration_sec=0,
            unknown_duration_sec=31,
        )
        fall_result = analyzer.determine_alert_level(
            recent_timeline=[],
            current_state=ActivityState.UNKNOWN,
            out_of_bed_duration_sec=0,
            bed_exit_count=0,
            sitting_on_edge_duration_sec=0,
            fall_suspected=True,
        )

        self.assertEqual(unknown_result["alert_level"], "MONITOR")
        self.assertEqual(fall_result["alert_level"], "ALERT")

    def test_agent_uses_future_context_to_confirm_real_exit(self):
        analyses = [
            frame(0, ActivityState.LYING_IN_BED),
            frame(1, ActivityState.SITTING_ON_BED),
            frame(2, ActivityState.STANDING),
            frame(3, ActivityState.WALKING),
            frame(4, ActivityState.SITTING_OUTSIDE_BED),
        ]
        frames = [
            ExtractedFrame(index=item.frame_index, timestamp_sec=item.timestamp_sec, image=np.zeros((2, 2, 3), dtype=np.uint8))
            for item in analyses
        ]
        agent = AgenticEngine(
            vlm=VLMAnalyzer(api_key="mock"),
            tracker=TemporalStateTracker(),
            all_frames=frames,
        )

        confirmed, confidence, _ = agent.evaluate_bed_exit(analyses[2], analyses)

        self.assertTrue(confirmed)
        self.assertGreaterEqual(confidence, 0.9)

    def test_agent_rejects_stand_then_return_without_move_away(self):
        analyses = [
            frame(0, ActivityState.LYING_IN_BED),
            frame(1, ActivityState.SITTING_ON_BED),
            frame(2, ActivityState.STANDING),
            frame(3, ActivityState.LYING_IN_BED),
        ]
        frames = [
            ExtractedFrame(index=item.frame_index, timestamp_sec=item.timestamp_sec, image=np.zeros((2, 2, 3), dtype=np.uint8))
            for item in analyses
        ]
        agent = AgenticEngine(
            vlm=VLMAnalyzer(api_key="mock"),
            tracker=TemporalStateTracker(),
            all_frames=frames,
        )

        confirmed, _, _ = agent.evaluate_bed_exit(analyses[2], analyses)

        self.assertFalse(confirmed)


if __name__ == "__main__":
    unittest.main()
