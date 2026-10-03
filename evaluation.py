"""
Evaluation Module
==================
Computes quantitative metrics for the elderly care monitoring system.
Supports comparison against ground truth annotations.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from models import (
    ActivityState, AnalysisReport, StateSegment, BedEvent, BedEventType,
    _sec_to_human, _sec_to_mmss,
)

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Computes evaluation metrics for the monitoring system.

    Metrics computed:
    1. Activity Recognition: State classification accuracy per state
    2. Bed Events: Precision, recall, F1 for bed exit/return
    3. Duration Estimation: MAE and MAPE vs ground truth
    4. Confusion analysis: Which states are confused most often
    """

    def __init__(self, report: AnalysisReport, ground_truth: Dict):
        self.report = report
        self.gt = ground_truth

    def compute_all(self) -> Dict:
        """Compute all metrics and return as dictionary."""
        metrics = {}

        # Duration metrics
        metrics["duration_metrics"] = self._duration_metrics()

        # Bed event metrics
        metrics["bed_event_metrics"] = self._bed_event_metrics()

        # Timeline overlap metrics
        metrics["timeline_metrics"] = self._timeline_overlap_metrics()

        # Coverage
        total_pred = sum(self.report.activity_duration_sec.values())
        metrics["coverage"] = {
            "video_duration_sec": self.report.observation_duration_sec,
            "analyzed_duration_sec": total_pred,
            "coverage_pct": (total_pred / max(self.report.observation_duration_sec, 1)) * 100,
        }

        return metrics

    def _duration_metrics(self) -> Dict:
        """MAE and MAPE for activity duration estimation."""
        gt_durations = self.gt.get("activity_duration_sec", {})
        pred_durations = self.report.activity_duration_sec

        results = {}
        total_abs_error = 0.0
        n = 0

        for state in ActivityState:
            key = state.value
            gt_val = gt_durations.get(key, 0.0)
            pred_val = pred_durations.get(key, 0.0)

            if gt_val > 0 or pred_val > 0:
                abs_error = abs(pred_val - gt_val)
                rel_error = abs_error / max(gt_val, 1) * 100
                results[key] = {
                    "ground_truth_sec": round(gt_val, 1),
                    "ground_truth_human": _sec_to_human(gt_val),
                    "predicted_sec": round(pred_val, 1),
                    "predicted_human": _sec_to_human(pred_val),
                    "abs_error_sec": round(abs_error, 1),
                    "abs_error_human": _sec_to_human(abs_error),
                    "relative_error_pct": round(rel_error, 1),
                }
                total_abs_error += abs_error
                n += 1

        mean_abs_error = total_abs_error / max(n, 1)
        return {
            "per_state": results,
            "mean_absolute_error_sec": round(mean_abs_error, 1),
            "mean_absolute_error_human": _sec_to_human(mean_abs_error),
        }

    def _bed_event_metrics(self) -> Dict:
        """Precision, recall, F1 for bed exit and return events."""
        gt_exits = self.gt.get("bed_exit_count", 0)
        gt_returns = self.gt.get("bed_return_count", 0)

        pred_exits = self.report.bed_exit_count
        pred_returns = self.report.bed_return_count

        # Tolerance-based matching: event is correct if within ±30s
        gt_exit_times = self.gt.get("bed_exit_times_sec", [])
        gt_return_times = self.gt.get("bed_return_times_sec", [])
        pred_exit_times = [
            e.confirmed_time_sec for e in self.report.bed_events
            if e.event_type == BedEventType.BED_EXIT
        ]
        pred_return_times = [
            e.confirmed_time_sec for e in self.report.bed_events
            if e.event_type == BedEventType.BED_RETURN
        ]

        exit_metrics = self._event_metrics(
            gt_exit_times, pred_exit_times, gt_exits, pred_exits
        )
        return_metrics = self._event_metrics(
            gt_return_times, pred_return_times, gt_returns, pred_returns
        )

        return {
            "bed_exits": {
                "ground_truth_count": gt_exits,
                "predicted_count": pred_exits,
                **exit_metrics,
            },
            "bed_returns": {
                "ground_truth_count": gt_returns,
                "predicted_count": pred_returns,
                **return_metrics,
            },
        }

    def _event_metrics(
        self,
        gt_times: List[float],
        pred_times: List[float],
        gt_count: int,
        pred_count: int,
        tolerance_sec: float = 30.0,
    ) -> Dict:
        """Compute precision/recall/F1 with temporal tolerance."""
        if not gt_times or not pred_times:
            # Fall back to count-based
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
        else:
            matched_gt = set()
            tp = 0
            for pt in pred_times:
                for i, gt in enumerate(gt_times):
                    if i not in matched_gt and abs(pt - gt) <= tolerance_sec:
                        tp += 1
                        matched_gt.add(i)
                        break
            fp = pred_count - tp
            fn = gt_count - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    def _timeline_overlap_metrics(self) -> Dict:
        """
        Compute per-second timeline overlap between predicted and ground truth.
        This is the most robust activity recognition metric.
        """
        gt_timeline = self.gt.get("timeline", [])
        if not gt_timeline:
            return {"note": "No ground truth timeline provided"}

        video_dur = int(self.report.observation_duration_sec)

        # Build second-by-second ground truth array
        gt_array = [None] * (video_dur + 1)
        for seg in gt_timeline:
            start = int(seg.get("start_sec", 0))
            end = int(seg.get("end_sec", video_dur))
            state = seg.get("state", "unknown")
            for t in range(start, min(end + 1, len(gt_array))):
                gt_array[t] = state

        # Build second-by-second prediction array
        pred_array = [None] * (video_dur + 1)
        for seg in self.report.timeline:
            start = int(seg.start_sec)
            end = int(seg.end_sec)
            for t in range(start, min(end + 1, len(pred_array))):
                pred_array[t] = seg.state.value

        # Compute per-state accuracy
        state_stats: Dict[str, Dict] = {}
        correct = 0
        total = 0

        for t in range(video_dur + 1):
            gt = gt_array[t]
            pred = pred_array[t]
            if gt is None:
                continue
            total += 1
            state_stats.setdefault(gt, {"correct": 0, "total": 0})
            state_stats[gt]["total"] += 1
            if gt == pred:
                correct += 1
                state_stats[gt]["correct"] += 1

        overall_accuracy = correct / max(total, 1)
        per_state_accuracy = {
            state: round(v["correct"] / max(v["total"], 1), 3)
            for state, v in state_stats.items()
        }

        return {
            "overall_accuracy": round(overall_accuracy, 3),
            "per_state_accuracy": per_state_accuracy,
            "evaluated_seconds": total,
        }

    def format_report(self) -> str:
        """Format metrics as a readable text report."""
        metrics = self.compute_all()
        lines = [
            "QUANTITATIVE EVALUATION",
            "=" * 60,
            "",
            "1. DURATION ESTIMATION METRICS",
            "-" * 40,
        ]

        dm = metrics.get("duration_metrics", {})
        lines.append(
            f"  {'State':<25} {'GT':>10} {'Pred':>10} {'Error':>10}"
        )
        lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        for state, vals in dm.get("per_state", {}).items():
            lines.append(
                f"  {state:<25} {vals['ground_truth_human']:>10} "
                f"{vals['predicted_human']:>10} "
                f"{vals['abs_error_human']:>10}"
            )
        lines.append(
            f"\n  Mean Absolute Error: {dm.get('mean_absolute_error_human', 'N/A')}"
        )

        lines.extend([
            "",
            "2. BED EVENT METRICS",
            "-" * 40,
        ])
        bem = metrics.get("bed_event_metrics", {})
        for event_type, vals in bem.items():
            lines.append(f"  {event_type.replace('_', ' ').title()}:")
            lines.append(f"    GT={vals.get('ground_truth_count', '?')}  "
                        f"Pred={vals.get('predicted_count', '?')}  "
                        f"Precision={vals.get('precision', '?')}  "
                        f"Recall={vals.get('recall', '?')}  "
                        f"F1={vals.get('f1', '?')}")

        lines.extend([
            "",
            "3. ACTIVITY RECOGNITION ACCURACY",
            "-" * 40,
        ])
        tm = metrics.get("timeline_metrics", {})
        if "overall_accuracy" in tm:
            lines.append(f"  Overall Accuracy: {tm['overall_accuracy']:.1%}")
            lines.append("  Per-State Accuracy:")
            for state, acc in tm.get("per_state_accuracy", {}).items():
                lines.append(f"    {state:<25} {acc:.1%}")

        return "\n".join(lines)


def load_ground_truth_template() -> Dict:
    """Return a ground truth template with documentation."""
    return {
        "_comment": "Ground truth annotation format. Fill in for evaluation.",
        "video_name": "video_filename.mp4",
        "observation_duration_sec": 1200,
        "activity_duration_sec": {
            "lying_in_bed": 702,
            "sitting_on_bed": 128,
            "sitting_outside_bed": 95,
            "standing": 63,
            "walking": 167,
            "out_of_bed": 0,
            "unknown": 45,
        },
        "bed_exit_count": 2,
        "bed_return_count": 2,
        "bed_exit_times_sec": [308, 780],
        "bed_return_times_sec": [550, 1050],
        "timeline": [
            {"start_sec": 0, "end_sec": 272, "state": "lying_in_bed"},
            {"start_sec": 272, "end_sec": 308, "state": "sitting_on_bed"},
            {"start_sec": 308, "end_sec": 320, "state": "standing"},
            {"start_sec": 320, "end_sec": 461, "state": "walking"},
            {"start_sec": 461, "end_sec": 555, "state": "sitting_outside_bed"},
            {"start_sec": 555, "end_sec": 582, "state": "walking"},
            {"start_sec": 582, "end_sec": 601, "state": "sitting_on_bed"},
            {"start_sec": 601, "end_sec": 900, "state": "lying_in_bed"},
        ],
    }
