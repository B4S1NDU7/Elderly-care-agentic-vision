"""Run deterministic temporal stress scenarios without video or API access."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from evaluation import EvaluationMetrics
from models import (
    ActivityState,
    AlertLevel,
    AnalysisReport,
    FrameAnalysis,
)
from state_tracker import TemporalStateTracker


def _scenario(
    name: str,
    description: str,
    ground_truth: List[Dict],
    prediction: Optional[List[Dict]] = None,
    bed_exit_times_sec: Optional[List[float]] = None,
    bed_return_times_sec: Optional[List[float]] = None,
) -> Dict:
    return {
        "name": name,
        "description": description,
        "ground_truth": ground_truth,
        "prediction": prediction or ground_truth,
        "bed_exit_times_sec": bed_exit_times_sec or [],
        "bed_return_times_sec": bed_return_times_sec or [],
    }


SCENARIOS = [
    _scenario(
        "routine_states_and_bed_events",
        "Turning while lying, sitting up, standing, walking, leaving, chair sitting, and returning to bed.",
        [
            {"start_sec": 0, "end_sec": 8, "state": "lying_in_bed"},
            {"start_sec": 8, "end_sec": 14, "state": "sitting_on_bed"},
            {"start_sec": 14, "end_sec": 16, "state": "standing"},
            {"start_sec": 16, "end_sec": 25, "state": "walking"},
            {"start_sec": 25, "end_sec": 31, "state": "out_of_bed"},
            {"start_sec": 31, "end_sec": 39, "state": "sitting_outside_bed"},
            {"start_sec": 39, "end_sec": 43, "state": "walking"},
            {"start_sec": 43, "end_sec": 47, "state": "sitting_on_bed"},
            {"start_sec": 47, "end_sec": 60, "state": "lying_in_bed"},
        ],
        bed_exit_times_sec=[16],
        bed_return_times_sec=[43],
    ),
    _scenario(
        "brief_stand_and_return",
        "The person sits up, briefly stands, and returns to bed without moving away.",
        [
            {"start_sec": 0, "end_sec": 6, "state": "lying_in_bed"},
            {"start_sec": 6, "end_sec": 10, "state": "sitting_on_bed"},
            {"start_sec": 10, "end_sec": 12, "state": "standing"},
            {"start_sec": 12, "end_sec": 16, "state": "sitting_on_bed"},
            {"start_sec": 16, "end_sec": 24, "state": "lying_in_bed"},
        ],
    ),
    _scenario(
        "blanket_occlusion_unknown",
        "Partial blanket occlusion is labeled UNKNOWN until the person is visible again.",
        [
            {"start_sec": 0, "end_sec": 5, "state": "lying_in_bed"},
            {"start_sec": 5, "end_sec": 10, "state": "unknown"},
            {"start_sec": 10, "end_sec": 20, "state": "lying_in_bed"},
        ],
    ),
    _scenario(
        "caregiver_identity_switch",
        "Injected classifier failure: during resident occlusion, a caregiver is mistaken for the resident.",
        [
            {"start_sec": 0, "end_sec": 5, "state": "lying_in_bed"},
            {"start_sec": 5, "end_sec": 10, "state": "unknown"},
            {"start_sec": 10, "end_sec": 20, "state": "lying_in_bed"},
        ],
        [
            {"start_sec": 0, "end_sec": 5, "state": "lying_in_bed"},
            {"start_sec": 5, "end_sec": 10, "state": "sitting_outside_bed"},
            {"start_sec": 10, "end_sec": 20, "state": "lying_in_bed"},
        ],
    ),
    _scenario(
        "poor_lighting_forced_posture",
        "Injected classifier failure: uncertain dark frames are forced to LYING_IN_BED instead of UNKNOWN.",
        [
            {"start_sec": 0, "end_sec": 5, "state": "lying_in_bed"},
            {"start_sec": 5, "end_sec": 10, "state": "unknown"},
            {"start_sec": 10, "end_sec": 20, "state": "lying_in_bed"},
        ],
        [
            {"start_sec": 0, "end_sec": 20, "state": "lying_in_bed"},
        ],
    ),
    _scenario(
        "camera_view_loss_false_absence",
        "Injected classifier failure: temporary loss of view is misread as OUT_OF_BED.",
        [
            {"start_sec": 0, "end_sec": 5, "state": "lying_in_bed"},
            {"start_sec": 5, "end_sec": 10, "state": "unknown"},
            {"start_sec": 10, "end_sec": 20, "state": "lying_in_bed"},
        ],
        [
            {"start_sec": 0, "end_sec": 5, "state": "lying_in_bed"},
            {"start_sec": 5, "end_sec": 10, "state": "out_of_bed"},
            {"start_sec": 10, "end_sec": 20, "state": "lying_in_bed"},
        ],
    ),
]


def _state_at(timeline: List[Dict], timestamp: float) -> ActivityState:
    for segment in timeline:
        if segment["start_sec"] <= timestamp < segment["end_sec"]:
            return ActivityState(segment["state"])
    raise ValueError(f"Timeline does not cover timestamp {timestamp}.")


def _durations(timeline: List[Dict]) -> Dict[str, float]:
    result = {}
    for segment in timeline:
        state = segment["state"]
        result[state] = result.get(state, 0.0) + (
            segment["end_sec"] - segment["start_sec"]
        )
    return result


def evaluate_scenario(scenario: Dict) -> Dict:
    """Run scripted observations through the tracker and evaluate against labels."""
    truth = scenario["ground_truth"]
    prediction = scenario["prediction"]
    duration = max(int(segment["end_sec"]) for segment in truth)
    if max(int(segment["end_sec"]) for segment in prediction) != duration:
        raise ValueError(f"{scenario['name']}: prediction and label durations differ.")

    tracker = TemporalStateTracker(smoothing_window=1)
    for timestamp in range(duration):
        state = _state_at(prediction, timestamp)
        tracker.add_frame(
            FrameAnalysis(
                frame_index=timestamp,
                timestamp_sec=float(timestamp),
                state=state,
                confidence=1.0,
                reasoning="Scripted classifier observation; no image model is run.",
            )
        )
    tracker.finalize(float(duration))

    gt = {
        "observation_duration_sec": duration,
        "activity_duration_sec": _durations(truth),
        "bed_exit_times_sec": scenario["bed_exit_times_sec"],
        "bed_return_times_sec": scenario["bed_return_times_sec"],
        "bed_exit_count": len(scenario["bed_exit_times_sec"]),
        "bed_return_count": len(scenario["bed_return_times_sec"]),
        "event_tolerance_sec": 1.0,
        "timeline": truth,
    }
    report = AnalysisReport(
        video_path=f"scripted://{scenario['name']}",
        observation_duration_sec=float(duration),
        activity_duration_sec=tracker.duration_map,
        timeline=tracker.segments,
        bed_events=tracker.bed_events,
        final_state=tracker.current_state or ActivityState.UNKNOWN,
        final_alert=AlertLevel.NORMAL,
        alert_reasoning="Alert policy is not evaluated by this scripted state-only suite.",
    )
    metrics = EvaluationMetrics(report, gt).compute_all()
    event_metrics = metrics["bed_event_metrics"]
    expected_by_second = {
        int(t): _state_at(truth, t).value for t in range(duration)
    }
    predicted_by_second = {
        int(t): _state_at(prediction, t).value for t in range(duration)
    }
    observed_mismatches = []
    for timestamp in range(duration):
        expected = expected_by_second[timestamp]
        actual = predicted_by_second[timestamp]
        if expected != actual:
            observed_mismatches.append(
                {"time_sec": timestamp, "expected": expected, "observed": actual}
            )

    case = {
        "name": scenario["name"],
        "description": scenario["description"],
        "observation_duration_sec": duration,
        "activity_accuracy": metrics["timeline_metrics"]["overall_accuracy"],
        "duration_mae_sec": metrics["duration_metrics"]["mean_absolute_error_sec"],
        "duration_errors_sec": {
            state: values["abs_error_sec"]
            for state, values in metrics["duration_metrics"]["per_state"].items()
        },
        "bed_exit_metrics": event_metrics["bed_exits"],
        "bed_return_metrics": event_metrics["bed_returns"],
        "predicted_timeline": [
            {
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "state": segment.state.value,
            }
            for segment in tracker.segments
        ],
        "observed_label_mismatches": observed_mismatches,
        "scripted_labels_not_image_model_output": True,
    }
    return case


def evaluate_suite() -> Dict:
    cases = [evaluate_scenario(scenario) for scenario in SCENARIOS]
    total_seconds = sum(case["observation_duration_sec"] for case in cases)
    correct_seconds = sum(
        case["activity_accuracy"] * case["observation_duration_sec"]
        for case in cases
    )
    aggregate = {
        "scenario_count": len(cases),
        "evaluated_seconds": total_seconds,
        "weighted_activity_accuracy": round(correct_seconds / total_seconds, 3),
        "bed_exit_true_positives": sum(
            case["bed_exit_metrics"]["true_positives"] for case in cases
        ),
        "bed_exit_false_positives": sum(
            case["bed_exit_metrics"]["false_positives"] for case in cases
        ),
        "bed_exit_false_negatives": sum(
            case["bed_exit_metrics"]["false_negatives"] for case in cases
        ),
        "bed_return_true_positives": sum(
            case["bed_return_metrics"]["true_positives"] for case in cases
        ),
        "bed_return_false_positives": sum(
            case["bed_return_metrics"]["false_positives"] for case in cases
        ),
        "bed_return_false_negatives": sum(
            case["bed_return_metrics"]["false_negatives"] for case in cases
        ),
    }
    return {
        "evaluation_type": "scripted_temporal_stress_test",
        "important_limit": (
            "Inputs are scripted state labels. This suite tests temporal tracking "
            "and event logic, not image recognition, pose estimation, or VLM accuracy."
        ),
        "aggregate": aggregate,
        "scenarios": cases,
    }


def format_report(result: Dict) -> str:
    aggregate = result["aggregate"]
    lines = [
        "SCRIPTED TEMPORAL STRESS EVALUATION",
        "=" * 72,
        "LIMIT: Scripted state labels are fed directly to the temporal tracker.",
        "This is not a visual-model evaluation and is not real-video evidence.",
        "",
        f"Scenarios: {aggregate['scenario_count']}",
        f"Total scripted observation: {aggregate['evaluated_seconds']}s",
        f"Weighted state accuracy: {aggregate['weighted_activity_accuracy']:.1%}",
        (
            "Bed exits: "
            f"TP={aggregate['bed_exit_true_positives']} "
            f"FP={aggregate['bed_exit_false_positives']} "
            f"FN={aggregate['bed_exit_false_negatives']}"
        ),
        (
            "Bed returns: "
            f"TP={aggregate['bed_return_true_positives']} "
            f"FP={aggregate['bed_return_false_positives']} "
            f"FN={aggregate['bed_return_false_negatives']}"
        ),
        "",
        f"{'Scenario':<34} {'Accuracy':>9} {'Duration MAE':>13} {'Exit FP':>8} {'Return FP':>10}",
        "-" * 80,
    ]
    for case in result["scenarios"]:
        lines.append(
            f"{case['name']:<34} "
            f"{case['activity_accuracy']:>8.1%} "
            f"{case['duration_mae_sec']:>11.1f}s "
            f"{case['bed_exit_metrics']['false_positives']:>8} "
            f"{case['bed_return_metrics']['false_positives']:>10}"
        )
    lines.extend(["", "SCRIPTED FAILURE EXAMPLES", "-" * 80])
    for case in result["scenarios"]:
        if case["observed_label_mismatches"] or case["bed_exit_metrics"]["false_positives"]:
            lines.append(f"- {case['name']}: {case['description']}")
            lines.append(
                f"  Accuracy={case['activity_accuracy']:.1%}, "
                f"duration MAE={case['duration_mae_sec']:.1f}s, "
                f"exit FP={case['bed_exit_metrics']['false_positives']}, "
                f"return FP={case['bed_return_metrics']['false_positives']}."
            )
            if case["observed_label_mismatches"]:
                example = case["observed_label_mismatches"][0]
                lines.append(
                    f"  Example at {example['time_sec']}s: expected "
                    f"{example['expected']}, observed {example['observed']}."
                )
    return "\n".join(lines) + "\n"


def write_reports(output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_suite()
    json_path = output_dir / "challenging_case_evaluation.json"
    text_path = output_dir / "failure_case_examples.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(format_report(result), encoding="utf-8")
    return {"json": json_path, "text": text_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scripted temporal stress tests; no video or API key required."
    )
    parser.add_argument(
        "--output",
        default="sample_output",
        help="Directory for JSON and Markdown evaluation reports.",
    )
    args = parser.parse_args()
    reports = write_reports(Path(args.output))
    print(format_report(evaluate_suite()), end="")
    print(f"Saved machine-readable results: {reports['json']}")
    print(f"Saved failure examples: {reports['text']}")


if __name__ == "__main__":
    main()
