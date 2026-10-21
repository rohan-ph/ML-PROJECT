"""
VisionGuard SNN - Test Fixture Generator
=========================================
Generates controlled, timestamped synthetic video fixtures for validating:
  1. Exact [max(0, t_start - 10), min(duration, t_end + 10)] evidence clipping.
  2. Frame-difference SNN activity generation.
  3. Edge case: Early onset where t_start < 10.0s (clamped to 0.0s).
  4. SHA-256 hash and metadata generation.
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path

def create_synthetic_stream(
    output_path: Path,
    duration_sec: float = 60.0,
    fps: int = 16,
    width: int = 320,
    height: int = 240,
    incident_start_sec: float = 30.0,
    incident_end_sec: float = 35.0,
    stream_name: str = "StreamFixture"
):
    """
    Creates a video file where each frame explicitly renders:
      - Current timestamp and frame index
      - Visual motion patterns (moving blocks) to trigger non-zero frame differences
      - High-frequency chaotic motion during incident window to mimic violent dynamics
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = int(duration_sec * fps)
    start_frame = int(incident_start_sec * fps)
    end_frame = int(incident_end_sec * fps)
    
    rng = np.random.RandomState(42)
    
    # State for normal motion
    box_x = 50
    box_dx = 2
    
    for f in range(total_frames):
        t_sec = f / fps
        is_incident = (start_frame <= f <= end_frame)
        
        # Base background
        if is_incident:
            # Flashing alert tone with chaotic motion
            frame = np.full((height, width, 3), (30, 30, 70), dtype=np.uint8)
            # Add violent visual disruption / rapid random blocks
            for _ in range(12):
                rx = rng.randint(20, width - 60)
                ry = rng.randint(40, height - 60)
                rw = rng.randint(20, 50)
                rh = rng.randint(20, 50)
                color = (rng.randint(0, 50), rng.randint(0, 50), rng.randint(180, 255))
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, -1)
            status_text = "SIMULATED VIOLENCE (TEST FIXTURE)"
            status_color = (0, 0, 255)
        else:
            # Calm scene: subtle dark background with slow moving object
            frame = np.full((height, width, 3), (25, 30, 25), dtype=np.uint8)
            box_x = (box_x + box_dx) % (width - 40)
            cv2.rectangle(frame, (box_x, 100), (box_x + 30, 140), (0, 180, 0), -1)
            status_text = "NORMAL CALM ACTIVITY (TEST FIXTURE)"
            status_color = (0, 255, 0)
            
        # Draw HUD overlays
        cv2.putText(frame, f"{stream_name} | {t_sec:05.2f}s / {duration_sec:05.2f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {f:04d}/{total_frames:04d} | FPS: {fps}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)
        
        # Draw boundary indicator line at top
        progress = int((f / total_frames) * width)
        cv2.line(frame, (0, 3), (progress, 3), status_color, 3)
        
        writer.write(frame)
        
    writer.release()
    print(f"Generated fixture video: {output_path} ({duration_sec}s, {total_frames} frames)")


def generate_all_fixtures(data_dir: str = "data"):
    data_path = Path(data_dir)
    videos_dir = data_path / "incident_timing" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Standard 60s video (incident 30s - 35s) -> Expected clip [20s, 45s] = 25s
    v1_path = videos_dir / "stream_fixture_60s.mp4"
    create_synthetic_stream(
        output_path=v1_path,
        duration_sec=60.0,
        fps=16,
        incident_start_sec=30.0,
        incident_end_sec=35.0,
        stream_name="Standard 60s Stream"
    )
    
    # 2. Early onset 20s video (incident 3s - 8s) -> Expected clip [0s, 18s] = 18s
    v2_path = videos_dir / "stream_fixture_20s_early.mp4"
    create_synthetic_stream(
        output_path=v2_path,
        duration_sec=20.0,
        fps=16,
        incident_start_sec=3.0,
        incident_end_sec=8.0,
        stream_name="Early Onset 20s Stream"
    )

    # 3. Write annotations.csv
    anno_path = data_path / "incident_timing" / "annotations.csv"
    with open(anno_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "video_path", "incident_start_seconds", "incident_end_seconds", "label", "source"])
        writer.writerow([
            "stream_fixture_60s",
            "data/incident_timing/videos/stream_fixture_60s.mp4",
            "30.0",
            "35.0",
            "violence",
            "controlled_synthetic_benchmark"
        ])
        writer.writerow([
            "stream_fixture_20s_early",
            "data/incident_timing/videos/stream_fixture_20s_early.mp4",
            "3.0",
            "8.0",
            "violence",
            "controlled_early_onset_benchmark"
        ])
    print(f"Generated ground truth annotations: {anno_path}")


if __name__ == "__main__":
    generate_all_fixtures()
