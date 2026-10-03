"""
Main Pipeline Orchestrator
============================
Coordinates all system components:
  VideoFrameExtractor → PoseDetector → VLMAnalyzer → AgenticEngine
  → TemporalStateTracker → ReportGenerator

Usage:
  pipeline = ElderlyCarePipeline(config)
  report = pipeline.analyze("video.mp4")
"""

import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from models import (
    ActivityState, FrameAnalysis, AnalysisReport, AlertLevel,
    StateSegment, _sec_to_human,
)
from video_extractor import VideoFrameExtractor, ExtractedFrame, VideoInfo
from pose_detector import PoseDetector, BedRegion
from vlm_analyzer import VLMAnalyzer
from state_tracker import TemporalStateTracker
from agent import AgenticEngine
from report_generator import ReportGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class PipelineConfig:
    """All configurable parameters for the pipeline."""

    def __init__(
        self,
        openai_api_key: str,
        vlm_model: str = "gpt-4o",
        sample_interval_sec: float = 2.0,
        yolo_model: str = "yolov8n-pose.pt",
        smoothing_window: int = 3,
        min_segment_sec: float = 3.0,
        output_dir: str = "output",
        max_frames: Optional[int] = None,
        enable_agentic_resolution: bool = True,
        enable_pose_estimation: bool = True,
        temperature: float = 0.1,
        ground_truth: Optional[Dict] = None,
        verbose: bool = True,
    ):
        self.openai_api_key = openai_api_key
        self.vlm_model = vlm_model
        self.sample_interval_sec = sample_interval_sec
        self.yolo_model = yolo_model
        self.smoothing_window = smoothing_window
        self.min_segment_sec = min_segment_sec
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.enable_agentic_resolution = enable_agentic_resolution
        self.enable_pose_estimation = enable_pose_estimation
        self.temperature = temperature
        self.ground_truth = ground_truth
        self.verbose = verbose


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class ElderlyCarePipeline:
    """
    End-to-end elderly care activity monitoring pipeline.

    Architecture:
    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Video     │───▶│   Frame      │───▶│    Pose     │
    │   File      │    │  Extractor   │    │  Detector   │
    └─────────────┘    └──────────────┘    └─────────────┘
                                                  │
                                                  ▼
                       ┌──────────────┐    ┌─────────────┐
                       │   Temporal   │◀───│    VLM      │
                       │   Tracker   │    │  Analyzer   │
                       └──────────────┘    └─────────────┘
                              │                   │
                              ▼                   ▼
                       ┌──────────────┐    ┌─────────────┐
                       │   Agentic    │───▶│   Report    │
                       │   Engine     │    │ Generator   │
                       └──────────────┘    └─────────────┘
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._setup_logging()

        # Initialize components
        logger.info("Initializing pipeline components...")

        self.extractor = VideoFrameExtractor(
            sample_interval_sec=config.sample_interval_sec,
            max_frames=config.max_frames,
        )

        self.pose_detector = PoseDetector(
            yolo_model=config.yolo_model,
            device="cpu",
        ) if config.enable_pose_estimation else None

        self.vlm = VLMAnalyzer(
            api_key=config.openai_api_key,
            model=config.vlm_model,
            temperature=config.temperature,
        )

        self.reporter = ReportGenerator(output_dir=config.output_dir)

        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # Main analysis method
    # ------------------------------------------------------------------

    def analyze(self, video_path: str) -> AnalysisReport:
        """
        Run full analysis on a video file.
        Returns AnalysisReport with timeline, durations, events, and alert.
        """
        start_time = time.time()
        logger.info(f"Starting analysis: {video_path}")

        # Step 1: Extract frames
        logger.info("[1/5] Extracting frames...")
        frames, video_info = self.extractor.extract_frames(video_path)

        if not frames:
            raise ValueError(f"No frames extracted from: {video_path}")

        # Step 2: Detect bed region (once from a representative frame)
        logger.info("[2/5] Detecting bed region...")
        bed_region = self._detect_bed_region(frames)

        # The tracker is used for state definitions while the agent reviews
        # observations, then rebuilt from the agent-corrected analyses below.
        tracker = TemporalStateTracker(
            smoothing_window=self.config.smoothing_window,
            min_segment_sec=self.config.min_segment_sec,
        )

        # Step 4: Analyze each frame
        logger.info(f"[3/5] Analyzing {len(frames)} frames with VLM...")
        all_analyses = self._analyze_frames(frames, video_info, bed_region)

        # Step 5: Agentic resolution of ambiguous cases
        if self.config.enable_agentic_resolution:
            logger.info("[4/5] Running agentic resolution...")
            agent = AgenticEngine(
                vlm=self.vlm,
                tracker=tracker,
                all_frames=frames,
            )
            self._run_agentic_resolution(agent, all_analyses, frames, tracker)
        else:
            agent = None

        tracker = TemporalStateTracker(
            smoothing_window=self.config.smoothing_window,
            min_segment_sec=self.config.min_segment_sec,
        )
        for analysis in all_analyses:
            tracker.add_frame(analysis)

        # Finalize tracker
        tracker.finalize(video_info.duration_sec)

        # Step 6: Determine final alert level
        logger.info("[5/5] Generating report...")
        stats = tracker.get_summary_stats()
        final_state = tracker.current_state or ActivityState.UNKNOWN

        alert_result = self.vlm.determine_alert_level(
            current_state=final_state,
            out_of_bed_duration_sec=stats["longest_out_of_bed_period_sec"],
            bed_exit_count=stats["bed_exit_count"],
            sitting_on_edge_duration_sec=stats["sitting_on_edge_duration_sec"],
            unknown_duration_sec=tracker.duration_map.get(
                ActivityState.UNKNOWN.value, 0.0
            ),
            fall_suspected=any(
                analysis.reasoning.startswith("[Agentic: floor risk]")
                for analysis in all_analyses
            ),
            recent_timeline=[
                {
                    "timestamp": seg.start_str,
                    "state": seg.state.value,
                    "confidence": seg.confidence,
                }
                for seg in tracker.segments[-10:]
            ],
        )
        try:
            final_alert = AlertLevel(alert_result.get("alert_level", "NORMAL"))
        except ValueError:
            logger.warning(
                "Ignoring invalid alert level returned by VLM: %r",
                alert_result.get("alert_level"),
            )
            final_alert = AlertLevel.NORMAL
        alert_reason = alert_result.get("reasoning", "No alert reasoning provided.")

        elapsed = time.time() - start_time

        # Build report
        report = AnalysisReport(
            video_path=video_path,
            observation_duration_sec=video_info.duration_sec,
            activity_duration_sec=tracker.duration_map,
            timeline=tracker.segments,
            bed_events=tracker.bed_events,
            final_state=final_state,
            final_alert=final_alert,
            alert_reasoning=alert_reason,
            processing_stats={
                "frames_analyzed": len(all_analyses),
                "total_frames": video_info.total_frames,
                "processing_time_sec": round(elapsed, 1),
                "sample_interval_sec": self.config.sample_interval_sec,
                "vlm_model": self.config.vlm_model,
                "agentic_decisions": len(agent.decisions) if agent else 0,
            },
        )

        # Generate all output files
        agent_log = agent.format_decisions_report() if agent else ""
        self.reporter.generate_all(report, agent_log=agent_log)
        self.reporter.generate_evaluation_report(
            report, ground_truth=self.config.ground_truth
        )

        logger.info(
            f"Analysis complete in {_sec_to_human(elapsed)}. "
            f"Alert: {final_alert.value}"
        )
        return report

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _detect_bed_region(self, frames: List[ExtractedFrame]) -> Optional[BedRegion]:
        """Detect the bed region from the first few frames."""
        if self.pose_detector is None:
            return None
        # Try first 5 frames and use the best detection
        best_bed = None
        for frame in frames[:5]:
            bed = self.pose_detector.detect_bed_region(frame.image)
            if best_bed is None or bed.confidence > best_bed.confidence:
                best_bed = bed
        if best_bed and best_bed.detected:
            logger.info(
                f"Bed region detected with confidence {best_bed.confidence:.2f}"
            )
        else:
            logger.warning("Bed region not detected; using heuristic fallback")
        return best_bed

    def _analyze_frames(
        self,
        frames: List[ExtractedFrame],
        video_info: VideoInfo,
        bed_region: Optional[BedRegion],
    ) -> List[FrameAnalysis]:
        """Analyze all frames sequentially with pose + VLM."""
        all_analyses: List[FrameAnalysis] = []
        prev_state: Optional[ActivityState] = None

        try:
            from tqdm import tqdm
            frame_iter = tqdm(frames, desc="Analyzing frames", unit="frame")
        except ImportError:
            frame_iter = frames

        for i, frame in enumerate(frame_iter):
            # Pose estimation
            pose_features = None
            if self.pose_detector is not None:
                try:
                    pose_features = self.pose_detector.analyze(frame.image)
                except Exception as e:
                    logger.debug(f"Pose error on frame {frame.index}: {e}")

            # VLM analysis
            try:
                analysis = self.vlm.analyze_frame(
                    frame=frame.image,
                    timestamp_sec=frame.timestamp_sec,
                    frame_index=frame.index,
                    pose_features=pose_features,
                    bed_region=bed_region,
                    previous_state=prev_state,
                )
            except Exception as e:
                logger.warning(f"VLM error on frame {frame.index}: {e}")
                analysis = FrameAnalysis(
                    frame_index=frame.index,
                    timestamp_sec=frame.timestamp_sec,
                    state=ActivityState.UNKNOWN,
                    confidence=0.0,
                    reasoning=f"Analysis failed: {e}",
                )

            all_analyses.append(analysis)
            prev_state = analysis.state

            if self.config.verbose and i % 10 == 0:
                logger.info(
                    f"  Frame {i+1}/{len(frames)} @ {frame.timestamp_sec:.1f}s: "
                    f"{analysis.state.value} (conf={analysis.confidence:.2f})"
                )

        return all_analyses

    def _run_agentic_resolution(
        self,
        agent: AgenticEngine,
        all_analyses: List[FrameAnalysis],
        frames: List[ExtractedFrame],
        tracker: TemporalStateTracker,
    ):
        """
        Run the agentic engine to resolve ambiguous transitions.
        Re-evaluates analyses where confidence is low or transitions are ambiguous.
        """
        for i in range(1, len(all_analyses)):
            prev = all_analyses[i - 1]
            curr = all_analyses[i]

            # Check for bed exit scenarios
            if (
                prev.state in tracker.IN_BED_STATES
                and curr.state in {ActivityState.STANDING, ActivityState.WALKING}
                and curr.confidence > 0.6
            ):
                is_exit, conf, reasoning = agent.evaluate_bed_exit(
                    curr, all_analyses[:i]
                )
                if is_exit is False:
                    # Revert the exit classification
                    logger.info(
                        f"Agentic: Rejected potential bed exit @ {curr.timestamp_sec:.1f}s"
                    )
                    all_analyses[i].state = ActivityState.SITTING_ON_BED
                    all_analyses[i].confidence = conf
                    all_analyses[i].reasoning = f"[Agentic rejected exit] {reasoning}"
                elif is_exit is None:
                    all_analyses[i].state = ActivityState.UNKNOWN
                    all_analyses[i].confidence = 0.0
                    all_analyses[i].reasoning = (
                        f"[Agentic unresolved exit] {reasoning}"
                    )

            # Check horizontal body (floor vs bed)
            if (
                curr.state == ActivityState.LYING_IN_BED
                and curr.confidence < 0.7
                and curr.pose_features.get("body_is_horizontal", False)
                and not curr.on_bed
            ):
                is_on_bed, reasoning = agent.evaluate_lying_on_floor(
                    curr, all_analyses[:i]
                )
                if not is_on_bed:
                    logger.warning(
                        f"Agentic: Person may be on FLOOR @ {curr.timestamp_sec:.1f}s"
                    )
                    all_analyses[i].state = ActivityState.UNKNOWN
                    all_analyses[i].confidence = 0.5
                    all_analyses[i].reasoning = f"[Agentic: floor risk] {reasoning}"

    def _setup_logging(self):
        level = logging.DEBUG if self.config.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
