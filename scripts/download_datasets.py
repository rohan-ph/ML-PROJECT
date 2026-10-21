"""
VisionGuard SNN - Dataset Acquisition & Verification Script
============================================================
Handles legitimate, verified dataset acquisition, verification, and manifest generation
for RWF-2000, Hockey Fight, RLVS, and long surveillance incident timing sets.
"""

import os
import sys
import argparse
import logging
import hashlib
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

DATASETS_METADATA = {
    "rwf2000": {
        "title": "RWF-2000: Violent Video Database",
        "description": "2,000 surveillance video clips (5s each @ 30fps) captured by surveillance cameras.",
        "kaggle_slug": "vulamnguyen/rwf2000",
        "alt_slug": "muksito/rwf2000",
        "official_repo": "https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection",
        "license": "Research & Educational Use Only",
        "expected_files": {
            "train_fight": 800,
            "train_nonfight": 800,
            "val_fight": 200,
            "val_nonfight": 200
        },
        "target_subpath": "raw/rwf2000"
    },
    "hockey": {
        "title": "Hockey Fight Dataset",
        "description": "1,000 action clips (500 fight, 500 non-fight) from National Hockey League games.",
        "kaggle_slug": "yassershrief/hockey-fight-vidoes",
        "license": "Academic / Non-Commercial Research",
        "expected_files": {
            "fight": 500,
            "nonfight": 500
        },
        "target_subpath": "raw/hockey_fight"
    },
    "rlvs": {
        "title": "Real Life Violence Situations (RLVS)",
        "description": "2,000 videos extracted from YouTube depicting real-world violence and non-violence.",
        "kaggle_slug": "mohamedmustafa/real-life-violence-situations-dataset",
        "license": "Research / Non-Commercial",
        "expected_files": {
            "violence": 1000,
            "nonviolence": 1000
        },
        "target_subpath": "raw/rlvs"
    }
}


def check_kaggle_auth() -> bool:
    """Verifies whether Kaggle credentials exist either in ~/.kaggle/kaggle.json or environment."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        logging.info("Found Kaggle credentials in environment variables.")
        return True
    
    if kaggle_json.exists():
        logging.info(f"Found Kaggle credentials at {kaggle_json}")
        return True
    
    logging.warning("No Kaggle credentials detected.")
    logging.info("To configure Kaggle API for automatic download:")
    logging.info("  1. Create a Kaggle account at https://www.kaggle.com")
    logging.info("  2. Go to 'Account' -> 'Create New API Token' (downloads kaggle.json)")
    logging.info(f"  3. Place kaggle.json at: {kaggle_json}")
    logging.info("  OR set environment variables KAGGLE_USERNAME and KAGGLE_KEY.")
    return False


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def inspect_video(video_path: Path) -> Optional[Dict]:
    """Inspects video integrity, FPS, duration, and resolution."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps and fps > 0 else 0
        cap.release()
        return {
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": round(duration, 2)
        }
    except Exception as e:
        logging.warning(f"Failed to inspect video {video_path}: {e}")
        return None


def generate_manifest(data_dir: Path) -> Path:
    """Scans all datasets in data_dir and generates dataset_manifest.csv."""
    manifest_path = data_dir / "dataset_manifest.csv"
    records = []

    # 1. Check RWF-2000
    rwf_dir = data_dir / "raw" / "rwf2000"
    for split in ["train", "val"]:
        for cls_name in ["Fight", "NonFight"]:
            target = rwf_dir / split / cls_name
            if target.exists():
                for vid in list(target.glob("*.avi")) + list(target.glob("*.mp4")):
                    info = inspect_video(vid)
                    records.append({
                        "dataset": "rwf2000",
                        "split": split,
                        "class": cls_name,
                        "filename": vid.name,
                        "rel_path": str(vid.relative_to(data_dir)).replace("\\", "/"),
                        "size_bytes": vid.stat().st_size,
                        "fps": info["fps"] if info else "corrupt",
                        "duration_sec": info["duration"] if info else "corrupt",
                        "resolution": f"{info['width']}x{info['height']}" if info else "corrupt"
                    })

    # 2. Check Hockey Fight
    hockey_dir = data_dir / "raw" / "hockey_fight"
    for cls_name in ["Fight", "NonFight"]:
        target = hockey_dir / cls_name
        if target.exists():
            for vid in list(target.glob("*.avi")) + list(target.glob("*.mp4")):
                info = inspect_video(vid)
                records.append({
                    "dataset": "hockey_fight",
                    "split": "all",
                    "class": cls_name,
                    "filename": vid.name,
                    "rel_path": str(vid.relative_to(data_dir)).replace("\\", "/"),
                    "size_bytes": vid.stat().st_size,
                    "fps": info["fps"] if info else "corrupt",
                    "duration_sec": info["duration"] if info else "corrupt",
                    "resolution": f"{info['width']}x{info['height']}" if info else "corrupt"
                })

    # 3. Check RLVS
    rlvs_dir = data_dir / "raw" / "rlvs"
    for cls_name in ["Violence", "NonViolence"]:
        target = rlvs_dir / cls_name
        if target.exists():
            for vid in list(target.glob("*.avi")) + list(target.glob("*.mp4")):
                info = inspect_video(vid)
                records.append({
                    "dataset": "rlvs",
                    "split": "all",
                    "class": cls_name,
                    "filename": vid.name,
                    "rel_path": str(vid.relative_to(data_dir)).replace("\\", "/"),
                    "size_bytes": vid.stat().st_size,
                    "fps": info["fps"] if info else "corrupt",
                    "duration_sec": info["duration"] if info else "corrupt",
                    "resolution": f"{info['width']}x{info['height']}" if info else "corrupt"
                })

    # 4. Check Sample / Incident Timing
    incident_dir = data_dir / "incident_timing" / "videos"
    if incident_dir.exists():
        for vid in list(incident_dir.glob("*.mp4")) + list(incident_dir.glob("*.avi")):
            info = inspect_video(vid)
            records.append({
                "dataset": "incident_timing",
                "split": "test",
                "class": "surveillance_stream",
                "filename": vid.name,
                "rel_path": str(vid.relative_to(data_dir)).replace("\\", "/"),
                "size_bytes": vid.stat().st_size,
                "fps": info["fps"] if info else "corrupt",
                "duration_sec": info["duration"] if info else "corrupt",
                "resolution": f"{info['width']}x{info['height']}" if info else "corrupt"
            })

    # Write manifest CSV
    fieldnames = ["dataset", "split", "class", "filename", "rel_path", "size_bytes", "fps", "duration_sec", "resolution"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logging.info(f"Manifest written to {manifest_path} with {len(records)} entries.")
    return manifest_path


def download_dataset_kaggle(slug: str, target_dir: Path):
    """Downloads dataset using Kaggle CLI or kaggle python module."""
    target_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(target_dir), "--unzip"]
    logging.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def create_incident_timing_template(data_dir: Path):
    """Creates the standard incident timing annotation CSV if not already present."""
    timing_file = data_dir / "incident_timing" / "annotations.csv"
    if not timing_file.exists():
        with open(timing_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "video_path", "incident_start_seconds", "incident_end_seconds", "label", "source"])
            # Example documented fixture
            writer.writerow([
                "stream_fixture_001",
                "data/incident_timing/videos/stream_fixture_001.mp4",
                "30.0",
                "35.0",
                "fight",
                "benchmark_synthetic_stream"
            ])
            writer.writerow([
                "stream_fixture_002_early",
                "data/incident_timing/videos/stream_fixture_002_early.mp4",
                "3.0",
                "8.0",
                "fight",
                "benchmark_early_boundary_fixture"
            ])
        logging.info(f"Created incident annotations at {timing_file}")


def main():
    parser = argparse.ArgumentParser(description="Acquire and organize violence detection datasets.")
    parser.add_argument("--dataset", choices=["rwf2000", "hockey", "rlvs", "all"], default="all",
                        help="Dataset to acquire")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Project data root directory")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Only scan and generate dataset_manifest.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    logging.info(f"Data root: {data_dir}")

    create_incident_timing_template(data_dir)

    if args.manifest_only:
        generate_manifest(data_dir)
        return

    has_auth = check_kaggle_auth()
    targets = ["rwf2000", "hockey", "rlvs"] if args.dataset == "all" else [args.dataset]

    for ds in targets:
        meta = DATASETS_METADATA[ds]
        target_path = data_dir / meta["target_subpath"]
        target_path.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"\n--- Checking Dataset: {meta['title']} ---")
        logging.info(f"Description: {meta['description']}")
        logging.info(f"License: {meta['license']}")
        logging.info(f"Target location: {target_path}")

        existing = list(target_path.rglob("*.mp4")) + list(target_path.rglob("*.avi"))
        if existing:
            logging.info(f"Found {len(existing)} existing video files for {ds}. Skipping download.")
            continue

        if has_auth:
            logging.info(f"Attempting download via Kaggle: {meta['kaggle_slug']}...")
            try:
                download_dataset_kaggle(meta["kaggle_slug"], target_path)
            except Exception as e:
                logging.error(f"Download failed for {meta['kaggle_slug']}: {e}")
                if "alt_slug" in meta:
                    logging.info(f"Trying alternative slug: {meta['alt_slug']}...")
                    try:
                        download_dataset_kaggle(meta["alt_slug"], target_path)
                    except Exception as e2:
                        logging.error(f"Alternative download failed: {e2}")
        else:
            logging.warning(f"Kaggle authentication not configured. Skipping automated download of {ds}.")
            logging.info(f"Manual download instructions:")
            logging.info(f"  1. Download ZIP from: https://www.kaggle.com/datasets/{meta['kaggle_slug']}")
            logging.info(f"  2. Extract files into: {target_path}")

    # Always generate/update manifest
    generate_manifest(data_dir)


if __name__ == "__main__":
    main()
