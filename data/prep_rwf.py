"""
Preprocessing pipeline for video datasets.
Reads raw videos, resamples to target FPS, resizes, crops, computes frame differences,
and saves the output as .npy arrays for fast loading during training.
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_video(video_path: Path, target_fps: int = 16, size: int = 112) -> tuple:
    """
    Process a single video: read, resample, resize, crop, and compute differences.
    Returns:
        rgb_frames: np.ndarray shape [T, 3, H, W]
        diff_frames: np.ndarray shape [T, 3, H, W]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return None, None

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 30.0  # Fallback assumption

    # Determine frame sampling ratio
    frame_interval = orig_fps / target_fps
    
    frames = []
    frame_idx = 0
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count >= frame_idx * frame_interval:
            # Resize short side to `size`
            h, w = frame.shape[:2]
            if h < w:
                new_h = size
                new_w = int(w * (size / h))
            else:
                new_w = size
                new_h = int(h * (size / w))
                
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            # Center crop to size x size
            start_y = (new_h - size) // 2
            start_x = (new_w - size) // 2
            frame_cropped = frame_resized[start_y:start_y+size, start_x:start_x+size]
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_idx += 1
            
        count += 1
        
    cap.release()
    
    if not frames:
        return None, None
        
    # Convert to float32 and normalize [0, 1]
    rgb_frames = np.array(frames, dtype=np.float32) / 255.0
    
    # Compute frame differences
    diff_frames = np.zeros_like(rgb_frames)
    diff_frames[1:] = np.abs(rgb_frames[1:] - rgb_frames[:-1])
    
    # Transpose to [T, C, H, W]
    rgb_frames = np.transpose(rgb_frames, (0, 3, 1, 2))
    diff_frames = np.transpose(diff_frames, (0, 3, 1, 2))
    
    return rgb_frames, diff_frames

def process_dataset(data_dir: str, output_dir: str, dataset: str, fps: int, res: int):
    raw_dir = Path(data_dir) / "raw" / dataset
    out_dir_ds = Path(output_dir) / dataset
    
    if not raw_dir.exists():
        logging.error(f"Raw directory not found: {raw_dir}")
        return
        
    video_files = list(raw_dir.rglob("*.avi")) + list(raw_dir.rglob("*.mp4"))
    if not video_files:
        logging.warning(f"No video files found in {raw_dir}")
        return
        
    logging.info(f"Processing {len(video_files)} videos from {dataset}...")
    
    for vid_path in tqdm(video_files):
        rgb, diff = process_video(vid_path, target_fps=fps, size=res)
        if rgb is None:
            continue
            
        # Determine relative path to recreate directory structure
        rel_path = vid_path.relative_to(raw_dir)
        vid_out_dir = out_dir_ds / rel_path.parent
        vid_out_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = vid_path.stem
        rgb_path = vid_out_dir / f"{base_name}_rgb.npy"
        diff_path = vid_out_dir / f"{base_name}_diff.npy"
        
        np.save(str(rgb_path), rgb)
        np.save(str(diff_path), diff)

def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for SNN violence detection.")
    parser.add_argument("--data-dir", type=str, default="d:/projectzz/ML PROJ/snn-evidence/data",
                        help="Base directory for raw data")
    parser.add_argument("--output-dir", type=str, default="d:/projectzz/ML PROJ/snn-evidence/data/processed",
                        help="Directory to save preprocessed .npy files")
    parser.add_argument("--fps", type=int, default=16, help="Target FPS to resample to")
    parser.add_argument("--resolution", type=int, default=112, help="Target resolution (H and W)")
    parser.add_argument("--dataset", choices=["rwf2000", "hockey", "rlvs", "all"], default="rwf2000")
    
    args = parser.parse_args()
    
    datasets = ["rwf2000", "hockey", "rlvs"] if args.dataset == "all" else [args.dataset]
    
    for ds in datasets:
        process_dataset(args.data_dir, args.output_dir, ds, args.fps, args.resolution)
        
if __name__ == "__main__":
    main()
