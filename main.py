"""
Elderly Care Agentic Vision System - CLI Entry Point
======================================================
Usage:
  python main.py --video VIDEO_PATH [OPTIONS]

Examples:
  python main.py --video input/room_footage.mp4
  python main.py --video input/room.mp4 --interval 3 --output results/
  python main.py --video input/room.mp4 --gt ground_truth.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Elderly Care Agentic Vision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for reports (default: output/)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=2.0,
        help="Frame sampling interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--gt", "--ground-truth",
        default=None,
        help="Path to ground truth JSON file for evaluation",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose estimation (faster but less accurate)",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Disable agentic resolution (faster, uses simple per-frame analysis)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to analyze (for quick testing)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="VLM temperature (default: 0.1, more deterministic)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in offline mock mode without calling OpenAI API (uses vision heuristics)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if args.mock:
            api_key = "mock"
        else:
            print(
                "ERROR: OpenAI API key not found.\n"
                "Set it via --api-key or OPENAI_API_KEY environment variable.\n"
                "Or use --mock to run in offline heuristic mode without API credits.\n"
                "Or create a .env file with: OPENAI_API_KEY=sk-...",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.mock:
        api_key = "mock"

    # Validate video file
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Load ground truth if provided
    ground_truth = None
    if args.gt:
        gt_path = Path(args.gt)
        if not gt_path.exists():
            print(f"WARNING: Ground truth file not found: {gt_path}", file=sys.stderr)
        else:
            with open(gt_path) as f:
                ground_truth = json.load(f)
            print(f"Loaded ground truth: {gt_path}")

    # Build pipeline config
    from pipeline import PipelineConfig, ElderlyCarePipeline

    config = PipelineConfig(
        openai_api_key=api_key,
        vlm_model=args.model,
        sample_interval_sec=args.interval,
        output_dir=args.output,
        max_frames=args.max_frames,
        enable_agentic_resolution=not args.no_agent,
        enable_pose_estimation=not args.no_pose,
        temperature=args.temperature,
        ground_truth=ground_truth,
        verbose=not args.quiet,
    )

    print(f"\nElderly Care Agentic Vision System")
    print(f"{'=' * 50}")
    print(f"Video:    {video_path}")
    print(f"Model:    {config.vlm_model}")
    print(f"Interval: {config.sample_interval_sec}s")
    print(f"Output:   {config.output_dir}/")
    print(f"Pose:     {'enabled' if config.enable_pose_estimation else 'disabled'}")
    print(f"Agent:    {'enabled' if config.enable_agentic_resolution else 'disabled'}")
    print()

    # Run pipeline
    try:
        pipeline = ElderlyCarePipeline(config)
        report = pipeline.analyze(str(video_path))

        print(f"\nDone! Reports saved to: {config.output_dir}/")
        print("  - activity_timeline.txt")
        print("  - activity_summary.txt / activity_summary.json")
        print("  - bed_events.json")
        print("  - full_report.json")
        print("  - evaluation_report.txt")
        if config.enable_agentic_resolution:
            print("  - agent_reasoning.txt")

        print(f"\nFinal Alert: {report.final_alert.value}")
        print(f"Final State: {report.final_state.value}")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
