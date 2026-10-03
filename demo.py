"""
Demo Script — Generates a synthetic test video and runs full analysis.
========================================================================
This script creates a realistic synthetic elderly care video for demonstration
without needing a real camera recording.

Usage:
  python demo.py --api-key YOUR_KEY
  python demo.py --api-key YOUR_KEY --duration 120  # 2-minute demo
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def create_synthetic_video(output_path: str, duration_sec: int = 120, fps: int = 10):
    """
    Create a synthetic indoor scene video simulating an elderly person
    going through various states: lying, sitting up, getting up, walking,
    returning to bed.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    # Define scenario timeline (state, duration_sec)
    scenario = [
        ("lying_in_bed",         30),
        ("sitting_on_bed",       10),
        ("standing",              5),
        ("walking",              15),
        ("sitting_outside_bed",  20),
        ("walking",              10),
        ("sitting_on_bed",        5),
        ("lying_in_bed",         25),
    ]

    # Color palettes for each state
    state_colors = {
        "lying_in_bed":         (20, 20, 60),
        "sitting_on_bed":       (20, 60, 20),
        "standing":             (60, 20, 20),
        "walking":              (20, 100, 150),
        "sitting_outside_bed":  (150, 20, 100),
    }

    # Body representation colors
    body_color = (200, 160, 120)    # skin tone
    bed_color = (180, 180, 200)     # light bed
    floor_color = (140, 120, 100)

    # Proportional scaling so all states and bed events occur regardless of duration
    base_total = sum(d for _, d in scenario)
    scale = duration_sec / max(base_total, 1)
    scaled_scenario = [(s, max(2, int(round(d * scale)))) for s, d in scenario]

    frame_idx = 0

    for state, dur in scaled_scenario:
        bg_color = state_colors.get(state, (80, 80, 80))
        frames_in_state = dur * fps

        for f in range(frames_in_state):
            if frame_idx >= total_frames:
                break

            # Add noise for realism
            noise = np.random.randint(-8, 8, (height, width, 3), dtype=np.int16)
            bg = np.clip(
                np.full((height, width, 3), bg_color, dtype=np.int16) + noise,
                0, 255
            ).astype(np.uint8)

            # Draw room elements
            # Floor
            cv2.rectangle(bg, (0, height // 2), (width, height), floor_color, -1)

            # Bed
            bed_x1, bed_y1 = 50, height // 3
            bed_x2, bed_y2 = width - 50, height - 50
            cv2.rectangle(bg, (bed_x1, bed_y1), (bed_x2, bed_y2), bed_color, -1)
            # Pillow
            cv2.rectangle(bg, (bed_x1 + 10, bed_y1 + 10),
                           (bed_x1 + 100, bed_y1 + 60), (220, 220, 240), -1)

            # Draw person based on state
            t = f / max(frames_in_state, 1)  # 0→1 progress within state
            _draw_person(bg, state, t, width, height, body_color, bed_x1, bed_y1, bed_x2, bed_y2)

            # Timestamp overlay
            sec = frame_idx // fps
            m, s = divmod(sec, 60)
            ts_text = f"{m:02d}:{s:02d}  [{state.upper()}]"
            cv2.putText(bg, ts_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            out.write(bg)
            frame_idx += 1

        if frame_idx >= total_frames:
            break

    out.release()
    print(f"Synthetic video created: {output_path} ({duration_sec}s, {fps}fps)")


def _draw_person(bg, state, t, w, h, color, bx1, by1, bx2, by2):
    """Draw a simple stick-figure person representing the given state."""
    cx = w // 2
    bed_cx = (bx1 + bx2) // 2
    bed_cy = (by1 + by2) // 2

    if state == "lying_in_bed":
        # Horizontal on bed
        py = by1 + (by2 - by1) // 3
        # Body (horizontal rectangle)
        cv2.rectangle(bg, (bx1 + 80, py - 15), (bx2 - 40, py + 15), color, -1)
        # Head
        cv2.circle(bg, (bx1 + 60, py), 20, color, -1)
        # Blanket overlay
        cv2.rectangle(bg, (bx1 + 80, py - 5), (bx2 - 40, py + 30),
                      (180, 100, 100), 2)

    elif state == "sitting_on_bed":
        # Sitting upright on bed edge
        sx = bx1 + 80
        sy = by2 - 60
        # Torso (vertical)
        cv2.rectangle(bg, (sx - 15, sy - 60), (sx + 15, sy), color, -1)
        # Head
        cv2.circle(bg, (sx, sy - 75), 20, color, -1)
        # Legs (horizontal, dangling)
        cv2.rectangle(bg, (sx - 10, sy), (sx + 50, sy + 15), color, -1)

    elif state == "standing":
        # Standing beside bed
        sx = bx1 - 20 + int(t * 30)
        sy = by2 - 30
        # Torso
        cv2.rectangle(bg, (sx - 12, sy - 80), (sx + 12, sy), color, -1)
        # Head
        cv2.circle(bg, (sx, sy - 100), 20, color, -1)
        # Legs
        cv2.rectangle(bg, (sx - 12, sy), (sx - 4, sy + 60), color, -1)
        cv2.rectangle(bg, (sx + 4, sy), (sx + 12, sy + 60), color, -1)

    elif state == "walking":
        # Moving across frame
        progress = int(t * (w - 100))
        sx = 50 + progress
        sy = h - 100
        # Walking pose (alternating legs)
        leg_swing = int(20 * np.sin(t * 20))
        cv2.rectangle(bg, (sx - 12, sy - 80), (sx + 12, sy), color, -1)
        cv2.circle(bg, (sx, sy - 100), 18, color, -1)
        cv2.line(bg, (sx, sy), (sx - 15 + leg_swing, sy + 60), color, 10)
        cv2.line(bg, (sx, sy), (sx + 15 - leg_swing, sy + 60), color, 10)

    elif state == "sitting_outside_bed":
        # Sitting on a chair in corner
        chair_x = bx2 + 30 if bx2 + 100 < w else 30
        sy = h - 80
        # Chair
        cv2.rectangle(bg, (chair_x - 25, sy - 5), (chair_x + 25, sy + 40), (100, 80, 60), 2)
        # Person sitting
        cv2.rectangle(bg, (chair_x - 15, sy - 80), (chair_x + 15, sy), color, -1)
        cv2.circle(bg, (chair_x, sy - 100), 20, color, -1)
        cv2.rectangle(bg, (chair_x - 15, sy), (chair_x + 40, sy + 15), color, -1)


def main():
    parser = argparse.ArgumentParser(description="Elderly Care Demo — generates synthetic video and runs analysis")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--duration", type=int, default=120, help="Demo video duration in seconds")
    parser.add_argument("--output-dir", default="demo_output")
    parser.add_argument("--no-video-gen", action="store_true", help="Skip video generation (use existing demo_video.mp4)")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--mock", action="store_true", help="Run in offline mock mode without OpenAI API credits")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Notice: No OpenAI API key provided. Running in offline MOCK mode (heuristic vision simulation).")
        print("(To use live GPT-4o Vision, provide --api-key or set OPENAI_API_KEY in your .env)\n")
        api_key = "mock"
    elif args.mock:
        api_key = "mock"

    # Generate synthetic video
    video_path = "demo_video.mp4"
    if not args.no_video_gen:
        print("Generating synthetic elderly care video...")
        create_synthetic_video(video_path, duration_sec=args.duration)

    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Run full analysis
    from pipeline import PipelineConfig, ElderlyCarePipeline

    config = PipelineConfig(
        openai_api_key=api_key,
        vlm_model=args.model,
        sample_interval_sec=args.interval,
        output_dir=args.output_dir,
        enable_agentic_resolution=True,
        enable_pose_estimation=True,
        verbose=True,
    )

    pipeline = ElderlyCarePipeline(config)
    report = pipeline.analyze(video_path)

    print(f"\nDemo complete! Check {args.output_dir}/ for reports.")
    return report


if __name__ == "__main__":
    main()
