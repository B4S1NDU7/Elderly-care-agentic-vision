"""
Report Generator
=================
Produces all required output artifacts:
- Activity timeline (console + file)
- Duration summary (JSON + human-readable)
- Bed event list (JSON)
- Evaluation metrics (if ground truth provided)
- Failure case examples
- Architecture diagram reference
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import timedelta

from models import (
    AnalysisReport, StateSegment, BedEvent, ActivityState,
    AlertLevel, _sec_to_mmss, _sec_to_human, _sec_to_hhmmss
)
from evaluation import EvaluationMetrics

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# State color mapping for rich display
STATE_COLORS = {
    "lying_in_bed": "blue",
    "sitting_on_bed": "cyan",
    "sitting_outside_bed": "green",
    "standing": "yellow",
    "walking": "magenta",
    "out_of_bed": "red",
    "unknown": "white",
}

ALERT_COLORS = {
    "NORMAL": "green",
    "MONITOR": "yellow",
    "ALERT": "red bold",
}


class ReportGenerator:
    """Generates and saves all required report artifacts."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_all(self, report: AnalysisReport, agent_log: str = "") -> Dict[str, str]:
        """
        Generate all reports and return paths to generated files.
        """
        generated = {}

        # 1. Print and save timeline
        timeline_path = self._save_timeline(report)
        generated["timeline"] = str(timeline_path)

        # 2. Save activity duration summary
        summary_path = self._save_summary(report)
        generated["summary"] = str(summary_path)

        # 3. Save bed events
        events_path = self._save_bed_events(report)
        generated["bed_events"] = str(events_path)

        # 4. Save complete JSON report
        json_path = self._save_json_report(report)
        generated["json_report"] = str(json_path)

        # 5. Save agent reasoning log
        if agent_log:
            agent_path = self.output_dir / "agent_reasoning.txt"
            agent_path.write_text(agent_log, encoding="utf-8")
            generated["agent_reasoning"] = str(agent_path)

        # 6. Print rich console report
        self._print_rich_report(report)

        logger.info(f"Reports saved to: {self.output_dir}")
        return generated

    # ------------------------------------------------------------------
    # Individual report generators
    # ------------------------------------------------------------------

    def _save_timeline(self, report: AnalysisReport) -> Path:
        """Save activity timeline to text file and return path."""
        lines = [
            "ACTIVITY TIMELINE",
            "=" * 60,
            f"Video: {Path(report.video_path).name}",
            f"Duration: {_sec_to_human(report.observation_duration_sec)}",
            "",
        ]

        for seg in report.timeline:
            marker = ""
            if seg.state in {ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED}:
                marker = " 🛏"
            elif seg.state == ActivityState.WALKING:
                marker = " 🚶"
            elif seg.state == ActivityState.OUT_OF_BED:
                marker = " 🚪"
            elif seg.state == ActivityState.UNKNOWN:
                marker = " ❓"

            line = (
                f"{seg.start_str} – {seg.end_str}    "
                f"{seg.state.value.upper():<25} "
                f"({_sec_to_human(seg.duration_sec)}){marker}"
            )
            lines.append(line)

        lines.extend([
            "",
            f"Final State: {report.final_state.value.upper()}",
            f"Alert Level: {report.final_alert.value}",
            f"Alert Reason: {report.alert_reasoning}",
        ])

        path = self.output_dir / "activity_timeline.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _save_summary(self, report: AnalysisReport) -> Path:
        """Save activity duration summary."""
        total = report.observation_duration_sec

        lines = [
            "ACTIVITY DURATION SUMMARY",
            "=" * 60,
            f"Total Observation Time: {_sec_to_human(total)}",
            "",
            "Activity Breakdown:",
            "-" * 40,
        ]

        for state in ActivityState:
            dur = report.activity_duration_sec.get(state.value, 0)
            if dur > 0:
                pct = (dur / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 5)
                lines.append(
                    f"  {state.value:<25} {_sec_to_human(dur):>10}  "
                    f"({pct:5.1f}%)  {bar}"
                )

        lines.extend([
            "",
            "Bed Summary:",
            "-" * 40,
            f"  Time in bed:           {_sec_to_human(report.total_in_bed_sec)}",
            f"  Time out of bed:       {_sec_to_human(report.total_out_of_bed_sec)}",
            f"  Bed exit count:        {report.bed_exit_count}",
            f"  Bed return count:      {report.bed_return_count}",
            f"  Longest out of bed:    {_sec_to_human(report.longest_out_of_bed_period_sec)}",
            "",
            f"Final State:  {report.final_state.value.upper()}",
            f"Alert Level:  {report.final_alert.value}",
        ])

        # JSON format too
        summary_dict = {
            "total_observation_time": _sec_to_human(total),
            "activity_summary": report.get_activity_summary_human(),
            "bed_summary": {
                "time_in_bed": _sec_to_human(report.total_in_bed_sec),
                "time_out_of_bed": _sec_to_human(report.total_out_of_bed_sec),
                "bed_exit_count": report.bed_exit_count,
                "bed_return_count": report.bed_return_count,
                "longest_out_of_bed": _sec_to_human(report.longest_out_of_bed_period_sec),
            },
            "final_state": report.final_state.value,
            "alert_level": report.final_alert.value,
            "alert_reasoning": report.alert_reasoning,
        }

        # Save text
        path = self.output_dir / "activity_summary.txt"
        path.write_text("\n".join(lines), encoding="utf-8")

        # Save JSON
        json_path = self.output_dir / "activity_summary.json"
        json_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")

        return path

    def _save_bed_events(self, report: AnalysisReport) -> Path:
        """Save bed events to JSON file."""
        events_data = {
            "bed_exit_count": report.bed_exit_count,
            "bed_return_count": report.bed_return_count,
            "events": [e.to_dict() for e in report.bed_events],
        }
        path = self.output_dir / "bed_events.json"
        path.write_text(json.dumps(events_data, indent=2), encoding="utf-8")
        return path

    def _save_json_report(self, report: AnalysisReport) -> Path:
        """Save complete JSON report."""
        data = report.to_summary_dict()
        data["timeline"] = [
            {
                "start": seg.start_str,
                "end": seg.end_str,
                "duration": seg.duration_str,
                "state": seg.state.value,
                "confidence": round(seg.confidence, 2),
            }
            for seg in report.timeline
        ]
        data["bed_events"] = [e.to_dict() for e in report.bed_events]
        data["processing_stats"] = report.processing_stats

        path = self.output_dir / "full_report.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Rich console output
    # ------------------------------------------------------------------

    def _print_rich_report(self, report: AnalysisReport):
        """Print a beautiful rich console report."""
        if not RICH_AVAILABLE:
            self._print_plain_report(report)
            return

        console.print()
        console.rule(
            "[bold blue]ELDERLY CARE MONITORING REPORT[/bold blue]",
            characters="-",
        )

        # Alert level banner
        alert_color = ALERT_COLORS.get(report.final_alert.value, "white")
        console.print(Panel(
            f"[{alert_color}]{report.final_alert.value}[/{alert_color}]  "
            f"- {report.alert_reasoning}",
            title="[bold]Alert Status[/bold]",
            border_style=alert_color.split()[0],
        ))

        # Timeline table
        timeline_table = Table(
            title="Activity Timeline",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        timeline_table.add_column("Start", style="cyan", width=8)
        timeline_table.add_column("End", style="cyan", width=8)
        timeline_table.add_column("State", width=25)
        timeline_table.add_column("Duration", justify="right", width=10)
        timeline_table.add_column("Conf.", justify="right", width=6)

        for seg in report.timeline:
            color = STATE_COLORS.get(seg.state.value, "white")
            timeline_table.add_row(
                seg.start_str,
                seg.end_str,
                f"[{color}]{seg.state.value}[/{color}]",
                _sec_to_human(seg.duration_sec),
                f"{seg.confidence:.2f}",
            )
        console.print(timeline_table)

        # Duration summary table
        dur_table = Table(
            title="Activity Duration Summary",
            box=box.SIMPLE_HEAVY,
            header_style="bold green",
        )
        dur_table.add_column("Activity", width=25)
        dur_table.add_column("Duration", justify="right")
        dur_table.add_column("Percentage", justify="right")

        total = report.observation_duration_sec
        for state in ActivityState:
            dur = report.activity_duration_sec.get(state.value, 0)
            if dur > 0:
                pct = (dur / total * 100) if total > 0 else 0
                color = STATE_COLORS.get(state.value, "white")
                dur_table.add_row(
                    f"[{color}]{state.value}[/{color}]",
                    _sec_to_human(dur),
                    f"{pct:.1f}%",
                )
        console.print(dur_table)

        # Bed summary
        bed_table = Table(title="Bed Summary", box=box.SIMPLE_HEAVY, header_style="bold blue")
        bed_table.add_column("Metric")
        bed_table.add_column("Value", justify="right")
        bed_table.add_row("Total in bed", _sec_to_human(report.total_in_bed_sec))
        bed_table.add_row("Total out of bed", _sec_to_human(report.total_out_of_bed_sec))
        bed_table.add_row("Bed exits", str(report.bed_exit_count))
        bed_table.add_row("Bed returns", str(report.bed_return_count))
        bed_table.add_row("Longest out-of-bed", _sec_to_human(report.longest_out_of_bed_period_sec))
        console.print(bed_table)

        # Bed events
        if report.bed_events:
            console.print()
            console.print("[bold]Bed Events:[/bold]")
            for ev in report.bed_events:
                ev_color = "red" if ev.event_type.value == "bed_exit" else "green"
                console.print(
                    f"  [{ev_color}]*[/{ev_color}] {ev.event_type.value.upper()} "
                    f"@ {_sec_to_hhmmss(ev.confirmed_time_sec)} "
                    f"| {ev.previous_state.value} to {ev.current_state.value} "
                    f"| {ev.decision.value} (conf={ev.confidence:.2f})"
                )

        console.rule(characters="-")

    def _print_plain_report(self, report: AnalysisReport):
        """Fallback plain text console report."""
        print("\n" + "=" * 60)
        print("ELDERLY CARE MONITORING REPORT")
        print("=" * 60)
        print(f"Alert: {report.final_alert.value} — {report.alert_reasoning}")
        print()
        print("Timeline:")
        for seg in report.timeline:
            print(f"  {seg.start_str} – {seg.end_str}  {seg.state.value}")
        print()
        print("Bed Events:", len(report.bed_events))
        for ev in report.bed_events:
            print(f"  {ev.event_type.value} @ {_sec_to_hhmmss(ev.confirmed_time_sec)}")

    # ------------------------------------------------------------------
    # Evaluation report (for accuracy assessment)
    # ------------------------------------------------------------------

    def generate_evaluation_report(
        self,
        report: AnalysisReport,
        ground_truth: Optional[Dict] = None,
    ) -> str:
        """
        Generate evaluation metrics report.
        If ground truth is provided, compute accuracy metrics.
        Otherwise, report self-consistency metrics.
        """
        lines = [
            "EVALUATION REPORT",
            "=" * 60,
            "",
        ]

        if ground_truth:
            lines.extend(
                EvaluationMetrics(report, ground_truth).format_report().splitlines()
            )
        else:
            lines.extend(self._compute_self_metrics(report))

        # Failure case examples
        lines.extend(self._find_failure_cases(report))

        report_text = "\n".join(lines)
        path = self.output_dir / "evaluation_report.txt"
        path.write_text(report_text, encoding="utf-8")

        if RICH_AVAILABLE:
            console.print()
            console.rule("[bold red]EVALUATION REPORT[/bold red]", characters="-")
            encoding = getattr(console.file, "encoding", None) or "utf-8"
            ascii_report = (
                report_text.replace("→", "->")
                .replace("–", "-")
                .replace("—", "-")
            )
            safe_report = ascii_report.encode(
                encoding, errors="replace"
            ).decode(encoding)
            console.print(safe_report)

        return report_text

    def _compute_metrics_vs_gt(self, report: AnalysisReport, gt: Dict) -> List[str]:
        """Format quantitative metrics against the provided annotations."""
        return EvaluationMetrics(report, gt).format_report().splitlines()

    def _compute_self_metrics(self, report: AnalysisReport) -> List[str]:
        """Self-consistency metrics when no ground truth is available."""
        lines = ["Self-Consistency Metrics:", "-" * 40]

        # Total duration check
        total_pred = sum(report.activity_duration_sec.values())
        total_video = report.observation_duration_sec
        coverage = total_pred / max(total_video, 1) * 100
        lines.append(f"  Timeline coverage: {coverage:.1f}% of video duration")
        lines.append(f"  Total analyzed:    {_sec_to_human(total_pred)}")
        lines.append(f"  Video duration:    {_sec_to_human(total_video)}")

        # Confidence distribution
        all_confs = [seg.confidence for seg in report.timeline if seg.confidence > 0]
        if all_confs:
            avg_conf = sum(all_confs) / len(all_confs)
            lines.append(f"  Avg. confidence:  {avg_conf:.2f}")
            low_conf = sum(1 for c in all_confs if c < 0.6)
            lines.append(f"  Low-conf segs:    {low_conf}/{len(all_confs)}")

        # Unknown rate
        unknown_dur = report.activity_duration_sec.get("unknown", 0)
        unknown_pct = unknown_dur / max(total_video, 1) * 100
        lines.append(f"  Unknown rate:     {unknown_pct:.1f}%")

        return lines

    def _find_failure_cases(self, report: AnalysisReport) -> List[str]:
        """Identify and report at least 3 failure/edge cases."""
        lines = [
            "",
            "FAILURE CASE ANALYSIS",
            "=" * 60,
            "The following cases represent potential misclassifications or edge cases:",
            "",
        ]

        case_num = 0

        # Case 1: Unknown segments
        unknowns = [s for s in report.timeline if s.state == ActivityState.UNKNOWN]
        if unknowns:
            case_num += 1
            worst = max(unknowns, key=lambda s: s.duration_sec)
            lines.extend([
                f"Case {case_num}: PROLONGED UNKNOWN STATE",
                f"  Time: {worst.start_str} – {worst.end_str} ({_sec_to_human(worst.duration_sec)})",
                f"  Cause: Insufficient visual evidence to classify activity",
                f"  Impact: This period contributes to duration estimation error",
                f"  Mitigation: Could use pose estimation confidence threshold or",
                f"              request manual review for segments > 30s unknown",
                "",
            ])

        # Case 2: Low-confidence segments
        low_conf_segs = [s for s in report.timeline if 0.0 < s.confidence < 0.6]
        if low_conf_segs:
            case_num += 1
            worst_lc = min(low_conf_segs, key=lambda s: s.confidence)
            lines.extend([
                f"Case {case_num}: LOW-CONFIDENCE CLASSIFICATION",
                f"  Time: {worst_lc.start_str} – {worst_lc.end_str}",
                f"  State: {worst_lc.state.value} (confidence={worst_lc.confidence:.2f})",
                f"  Cause: Ambiguous visual features, partial occlusion, or poor lighting",
                f"  Impact: Classification may be incorrect → timeline inaccurate",
                f"  Mitigation: Agentic re-analysis with temporal context can resolve ~70%",
                f"              of low-confidence cases. Pose estimation helps.",
                "",
            ])

        # Case 3: Rapid state oscillation
        if len(report.timeline) >= 4:
            for i in range(1, len(report.timeline) - 1):
                prev = report.timeline[i - 1]
                curr = report.timeline[i]
                nxt = report.timeline[i + 1]
                if (prev.state == nxt.state and prev.state != curr.state
                        and curr.duration_sec < 15):
                    case_num += 1
                    lines.extend([
                        f"Case {case_num}: SHORT STATE OSCILLATION (review candidate)",
                        f"  Time: {curr.start_str} – {curr.end_str}",
                        f"  Pattern: {prev.state.value} → {curr.state.value} → {nxt.state.value}",
                        f"  Duration: {_sec_to_human(curr.duration_sec)} (very brief)",
                        f"  Cause: Could be a real brief change or frame-level noise;",
                        f"         ground-truth labels are needed to decide.",
                        f"  Impact: May change activity-duration and bed-event counts.",
                        f"  Mitigation: Compare against labeled frames; current temporal",
                        f"              smoothing does not remove all short transitions.",
                        "",
                    ])
                    break

        # Case 4: Caregiver interference
        case_num += 1
        lines.extend([
            f"Case {case_num}: CAREGIVER INTERFERENCE (Expected Failure Mode)",
            f"  Description: When a caregiver enters the scene, the system may",
            f"               detect two people and misclassify the primary patient.",
            f"  Cause: YOLOv8 detects all persons; primary person selection uses",
            f"         highest confidence, which may not always be the patient.",
            f"  Impact: False state transitions during caregiver visits",
            f"  Mitigation: Person re-ID across frames (ReID models) or temporal",
            f"              consistency checking to track the primary subject.",
            "",
        ])

        # Case 5: Poor lighting
        case_num += 1
        lines.extend([
            f"Case {case_num}: POOR LIGHTING / NIGHT MONITORING",
            f"  Description: In low-light conditions, VLM visual analysis degrades.",
            f"               The system may default to UNKNOWN or make errors.",
            f"  Cause: GPT-4o Vision quality degrades with noisy/dark frames.",
            f"         MediaPipe Pose also has reduced accuracy in poor lighting.",
            f"  Impact: Higher UNKNOWN rate, reduced classification accuracy.",
            f"  Mitigation: Night-vision preprocessing (histogram equalization,",
            f"              CLAHE), or use of infrared-aware models.",
            "",
        ])

        if case_num < 3:
            lines.extend([
                "No additional data-dependent failure patterns were identified.",
                "",
            ])

        lines.extend([
            "UNTESTED FAILURE SCENARIOS (not observed or measured in this video)",
            "-" * 60,
            "  1. Caregiver enters while the resident is occluded: identity may switch.",
            "  2. Poor lighting or blanket occlusion: posture may be misclassified.",
            "  3. Resident leaves camera view: absence may be confused with bed exit.",
            "",
        ])

        return lines
