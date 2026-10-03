import json
import unittest
from pathlib import Path

from evaluation import EvaluationMetrics
from models import (
    ActivityState,
    AlertLevel,
    AnalysisReport,
    BedEvent,
    BedEventType,
    StateSegment,
)


class EvaluationTests(unittest.TestCase):
    def test_ground_truth_template_durations_match_half_open_timeline(self):
        template_path = Path(__file__).resolve().parents[1] / "ground_truth_template.json"
        ground_truth = json.loads(template_path.read_text(encoding="utf-8"))
        duration_by_state = {}
        for segment in ground_truth["timeline"]:
            state = segment["state"]
            duration_by_state[state] = duration_by_state.get(state, 0) + (
                segment["end_sec"] - segment["start_sec"]
            )

        self.assertEqual(
            duration_by_state,
            {
                key: value
                for key, value in ground_truth["activity_duration_sec"].items()
                if value
            },
        )
        self.assertEqual(sum(duration_by_state.values()), 1200)
        self.assertEqual(ground_truth["bed_exit_times_sec"], [320, 912])
        self.assertEqual(ground_truth["bed_return_times_sec"], [582, 1155])

    def test_metrics_use_timestamp_matches_and_include_confusion_matrix(self):
        report = AnalysisReport(
            video_path="synthetic.mp4",
            observation_duration_sec=10,
            activity_duration_sec={
                "lying_in_bed": 5,
                "walking": 5,
            },
            timeline=[
                StateSegment(ActivityState.LYING_IN_BED, 0, 5, 1.0),
                StateSegment(ActivityState.WALKING, 5, 10, 1.0),
            ],
            bed_events=[
                BedEvent(
                    event_type=BedEventType.BED_EXIT,
                    start_time_sec=4,
                    confirmed_time_sec=5,
                    previous_state=ActivityState.LYING_IN_BED,
                    current_state=ActivityState.WALKING,
                    confidence=0.9,
                    decision=AlertLevel.MONITOR,
                )
            ],
            final_state=ActivityState.WALKING,
            final_alert=AlertLevel.NORMAL,
            alert_reasoning="test",
        )
        ground_truth = {
            "activity_duration_sec": {"lying_in_bed": 5, "walking": 5},
            "bed_exit_count": 1,
            "bed_return_count": 0,
            "bed_exit_times_sec": [50],
            "bed_return_times_sec": [],
            "timeline": [
                {"start_sec": 0, "end_sec": 5, "state": "lying_in_bed"},
                {"start_sec": 5, "end_sec": 10, "state": "walking"},
            ],
        }

        metrics = EvaluationMetrics(report, ground_truth).compute_all()

        self.assertEqual(metrics["timeline_metrics"]["overall_accuracy"], 1.0)
        self.assertEqual(
            metrics["timeline_metrics"]["confusion_matrix"],
            {"lying_in_bed": {"lying_in_bed": 5}, "walking": {"walking": 5}},
        )
        exits = metrics["bed_event_metrics"]["bed_exits"]
        self.assertEqual(exits["true_positives"], 0)
        self.assertEqual(exits["false_positives"], 1)
        self.assertEqual(exits["false_negatives"], 1)
        self.assertEqual(exits["precision"], 0.0)
        self.assertEqual(exits["recall"], 0.0)

    def test_longest_absence_uses_bed_events_across_unknown_gaps(self):
        exit_event = BedEvent(
            event_type=BedEventType.BED_EXIT,
            start_time_sec=5,
            confirmed_time_sec=8,
            previous_state=ActivityState.SITTING_ON_BED,
            current_state=ActivityState.WALKING,
            confidence=0.9,
            decision=AlertLevel.MONITOR,
        )
        return_event = BedEvent(
            event_type=BedEventType.BED_RETURN,
            start_time_sec=5,
            confirmed_time_sec=20,
            previous_state=ActivityState.UNKNOWN,
            current_state=ActivityState.LYING_IN_BED,
            confidence=0.9,
            decision=AlertLevel.NORMAL,
        )
        report = AnalysisReport(
            video_path="synthetic.mp4",
            observation_duration_sec=30,
            activity_duration_sec={},
            timeline=[
                StateSegment(ActivityState.WALKING, 8, 12, 0.9),
                StateSegment(ActivityState.UNKNOWN, 12, 16, 0.0),
                StateSegment(ActivityState.LYING_IN_BED, 16, 30, 0.9),
            ],
            bed_events=[exit_event, return_event],
            final_state=ActivityState.LYING_IN_BED,
            final_alert=AlertLevel.NORMAL,
            alert_reasoning="test",
        )

        self.assertEqual(report.longest_out_of_bed_period_sec, 15)


if __name__ == "__main__":
    unittest.main()
